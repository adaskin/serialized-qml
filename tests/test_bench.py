import json
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, "src")
import bench
import bench_classical
import datasets
import pennylane as qml
from qcircuit import SerializedQC


@pytest.mark.parametrize("T,k,head", [(4, 26, "linear"), (6, 3, "linear"), (14, 7, "mlp"), (49, 10, "mlp")])
def test_parameter_formula_matches_the_circuit(T, k, head):
    model = SerializedQC(n_steps=T, n_out=k, layers=4)
    if head == "mlp":                                   # the head built in bench.build_model
        nf = model.linear.in_features
        model.linear = torch.nn.Sequential(torch.nn.Linear(nf, bench.HEAD_HIDDEN), torch.nn.ReLU(), torch.nn.Linear(bench.HEAD_HIDDEN, k))
    n = sum(p.numel() for p in model.parameters())
    assert n == bench_classical.quantum_params(T, 4, k, linear=head == "linear", hidden=bench.HEAD_HIDDEN)


def test_build_model_uses_the_head_of_the_run():
    lin, mlp = bench.build_model(bench.BRun("letter", 42)), bench.build_model(bench.BRun("letter", 42, head="mlp"))
    assert isinstance(lin.linear, torch.nn.Linear) and isinstance(mlp.linear, torch.nn.Sequential)
    assert sum(p.numel() for p in lin.parameters()) == bench_classical.quantum_params(4, 4, 26)


def test_enc_scale_rule_reproduces_the_mnist_grid():
    assert bench.BRun("fashion", 42).enc_scale == 0.33               # 49 patches, as 14x14 MNIST
    assert bench.BRun("letter", 42).enc_scale == 1.0                 # 4 patches: capped at the 8x8 value
    assert all(r.layers == 4 and r.memory == "coherent" and r.epochs == 100 for r in bench.run_set("main"))
    assert len(bench.run_set("main")) == 18 and len(bench.run_set("measured")) == 12 and len(bench.run_set("head")) == 3 and len(bench.run_set("pilot")) == 6


def test_names_are_unique():
    runs = bench.run_set("main") + bench.run_set("head") + bench.run_set("pilot")
    assert len({r.name for r in runs}) == len(runs)
    assert len({r.name for r in bench_classical.grid()}) == len(bench_classical.grid())


def _have(name):
    try:
        datasets.load(name)
        return True
    except (FileNotFoundError, OSError):
        return False


@pytest.mark.parametrize("name", list(datasets.SPECS))
def test_split_is_disjoint_stratified_and_shared_by_both_pipelines(name):
    if not _have(name):
        pytest.skip(f"raw data for {name} not present")
    spec = datasets.SPECS[name]
    xq, yq, xvq, yvq, xtq, ytq = datasets.make_split(name, 42)
    xc, yc, xvc, yvc, xtc, ytc = datasets.make_split(name, 42, patches=False)
    assert xq.shape[1:] == (spec.steps, 4) and 0 <= xq.min() and xq.max() <= 1
    assert torch.equal(yq, yc) and torch.equal(ytq, ytc)
    assert torch.allclose(xq.sum((1, 2)), xc.reshape(len(xc), -1).sum(1))      # patches are a reordering (zero padding adds nothing)
    assert len(yq) <= datasets.TRAIN_CAP and set(yq.tolist()) <= set(range(spec.n_classes))
    # the test examples do not depend on the seed (their scaling uses the seed's training split), train and val draws do
    raw, raw2 = datasets.split_arrays(name, 42), datasets.split_arrays(name, 43)
    assert np.array_equal(raw[4], raw2[4]) and np.array_equal(raw[5], raw2[5])
    assert not np.array_equal(raw[1], raw2[1])
    if spec.kind == "tabular" and name == "waveform":                           # continuous features: any shared row is a leak between splits
        rows = lambda a: {r.tobytes() for r in a}
        assert not (rows(raw[0]) & rows(raw[2])) and not (rows(raw[0]) & rows(raw[4])) and not (rows(raw[2]) & rows(raw[4]))


def test_tiny_run_trains_and_resumes(tmp_path, monkeypatch):
    if not _have("letter"):
        pytest.skip("raw data for letter not present")
    monkeypatch.setattr(bench, "RESULTS", tmp_path / "res")
    monkeypatch.setattr(bench, "CKPT", tmp_path / "ckpt")
    run = bench.BRun("letter", 42, layers=1, epochs=2, train_cap=64, eval_every=1, batch=32)
    res = json.load(open(bench.train(run)))
    assert res["final"]["epoch"] == 2 and "test_bacc" in res["final"] and res["params"] > 0
    assert not (tmp_path / "ckpt" / f"{run.name}.pt").exists()


def test_amplitude_runs_use_seven_qubits_and_normalisable_vectors():
    import bench_amplitude as ba
    runs = ba.grid()
    assert len(runs) == 72 and {r.qubits for r in runs} == {7} and {r.layers for r in runs} == {4, 8}
    assert len({r.name for r in runs}) == 72
    x = torch.zeros(2, 4, 4)
    v = ba.prepare(x, ba.ARun("fashion", "7", 4, 42))                 # an all-zero image still gives a unit-norm vector after embedding
    assert v.shape == (2, 128) and (v.abs().sum(1) > 0).all()
    t = torch.rand(3, 21)
    assert ba.prepare(t, ba.ARun("waveform", "7", 4, 42)).shape == (3, 128)
    assert ba.ARun("waveform", "min", 4, 42).qubits == 5 and ba.ARun("fashion", "min", 4, 42).qubits == 8


def test_batches_cover_every_spec_once_and_run_sets_follow_the_batch():
    flat = [d for b in datasets.BATCHES.values() for d in b]
    assert sorted(flat) == sorted(datasets.SPECS) and len(flat) == len(set(flat)) == 12
    for batch, ds in datasets.BATCHES.items():
        assert {r.dataset for r in bench.run_set("main", batch)} == set(ds)
        tab = {d for d in ds if datasets.SPECS[d].kind == "tabular"}
        assert {r.dataset for r in bench.run_set("measured", batch)} == tab and len(bench.run_set("measured", batch)) == 3 * len(tab)
    assert datasets.SPECS["spambase"].steps == 15 and datasets.SPECS["creditdefault"].steps == 6 and datasets.SPECS["pneumonia"].steps == 49


def test_pauli_readout_extends_zz_without_changing_it():
    x = torch.rand(4, 3, 4, dtype=torch.float64)

    def features(readout):
        torch.manual_seed(1)
        return SerializedQC(n_steps=3, n_out=4, layers=2, readout=readout).features(x)
    base, full = features("z+zz"), features(bench.ALL_PAULI)
    assert base.shape == (4, 14) and full.shape == (4, 42)                       # 7 qubits: 7 + 7 terms per Pauli type
    assert torch.allclose(base[:, :7], full[:, :7]) and torch.allclose(base[:, 7:], full[:, 21:28])    # Z and ZZ unchanged
    assert ((full[:, 0:7] ** 2 + full[:, 7:14] ** 2 + full[:, 14:21] ** 2) <= 1 + 1e-9).all()     # <X>^2 + <Y>^2 + <Z>^2 <= 1 per qubit
    r = bench.BRun("letter", 42, readout=bench.ALL_PAULI)
    assert r.name == "letter_coherent_L4_linear_seed42_rozzzxxxyyy" and bench.BRun("letter", 42).name == "letter_coherent_L4_linear_seed42"
    assert len(bench.run_set("readout")) == 9


def test_multibasis_runs_and_resized_baselines():
    import classical
    import train
    runs = train.multibasis()
    assert len(runs) == 3 and all(r.folder.name == "multibasis" and r.name.endswith("_rozzzxxxyyy") for r in runs)
    assert all(r.folder == train.RESULTS and "_ro" not in r.name for r in train.grid())           # the grid is unchanged
    model = SerializedQC(n_steps=16, n_out=10, layers=4, readout=bench.ALL_PAULI)
    assert sum(p.numel() for p in model.parameters()) == classical.quantum_params(8, 4, 10, feats=42)
    for m in ("mlp", "cnn"):
        c = classical.CRun(m, 4, "10class", 8, 0.01, 42, feats=42)
        p = sum(q.numel() for q in classical.make_model(c).parameters())
        assert abs(p - c.target) / c.target < 0.15 and c.folder.name == "multibasis"
    assert {r.dataset for r in bench.run_set("multibasis")} == {"covertype", "landsat", "drybean"}
    r = bench_classical.BCRun("letter", "mlp_mb", 0.01, 42)
    assert r.target == bench_classical.quantum_params(4, 4, 26, feats=42) == 1817


# ----------------------------------------------------------------------------- review controls (Tier 1)
import bench_struct


def _fake_split(n_tr=64, n_te=40, T=4, k=26, seed=0):
    g = torch.Generator().manual_seed(seed)
    mk = lambda n: (torch.rand(n, T, 4, generator=g), torch.randint(0, k, (n,), generator=g))
    return (*mk(n_tr), *mk(n_te), *mk(n_te))


def test_product_gates_keep_the_parameter_count_and_give_a_product_state():
    a, b = SerializedQC(n_steps=3, n_out=4, layers=2), SerializedQC(n_steps=3, n_out=4, layers=2, entangle=False)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    with torch.no_grad():
        for p in (b.rot, b.chain, b.anc_rot, b.anc_chain, b.coupling):
            p.uniform_(-3, 3)
    f = b.features(torch.rand(5, 3, 4, dtype=torch.float64))          # z (7 values), then zz on the ring (7)
    z, zz = f[:, :7], f[:, 7:]
    assert torch.allclose(zz, torch.stack([z[:, i] * z[:, (i + 1) % 7] for i in range(7)], 1), atol=1e-10)
    tape = qml.workflow.construct_tape(b.qnode)(torch.rand(1, 3, 4), *b.weights())
    assert not any(op.name == "CRot" for op in tape.operations)


def test_tied_runs_share_one_block_and_the_matched_ones_restore_the_count():
    untied, tied = bench.build_model(bench.BRun("letter", 42)), bench.build_model(bench.BRun("letter", 42, tie=True))
    deep = bench.build_model(bench.BRun("letter", 42, tie=True, layers=16))
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert tied.rot.shape[0] == 1 and n(untied) - n(tied) == 3 * 4 * 39 and n(deep) == n(untied)


def test_frozen_run_trains_only_the_head_and_stores_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "REVIEW", tmp_path / "res")
    monkeypatch.setattr(bench, "CKPT", tmp_path / "ckpt")
    monkeypatch.setattr(datasets, "make_split", lambda *a, **k: _fake_split())
    run = bench.BRun("letter", 42, layers=1, epochs=2, batch=32, eval_every=1, frozen=True)
    model0 = bench.build_model(run)
    res = json.load(open(bench.train(run)))
    assert res["trainable"] == 14 * 26 + 26 and res["params"] > res["trainable"]
    assert len(res["test_pred"]) == res["n_test"] == len(res["test_y"])
    acc = np.mean(np.array(res["test_pred"]) == np.array(res["test_y"]))
    assert abs(acc - res["final"]["test_acc"]) < 1e-9
    assert model0.rot.abs().max() > 1                                   # uniform random angles, not the near-identity start


def test_feature_permutation_reorders_columns_before_patching(monkeypatch):
    rng = np.random.default_rng(0)
    x, y = rng.random((300, 21)).astype(np.float32), rng.integers(0, 3, 300)
    monkeypatch.setattr(datasets, "split_arrays", lambda *a, **k: (x[:200], y[:200], x[200:250], y[200:250], x[250:], y[250:]))
    plain, perm = datasets.make_split("waveform", 42, patches=False), datasets.make_split("waveform", 42, patches=False, perm=1)
    order = np.random.default_rng(1).permutation(21)
    assert torch.equal(perm[0], plain[0][:, order]) and torch.equal(perm[1], plain[1])
    assert datasets.make_split("waveform", 42, perm=1)[0].shape == (200, 6, 4)


def test_review_run_sets_are_kept_apart_and_unique():
    runs = bench.run_set("review")
    assert len({r.name for r in runs}) == len(runs) and all(r.folder == bench.REVIEW for r in runs)
    assert all(r.folder == bench.RESULTS for r in bench.run_set("main"))
    assert len(bench.run_set("tied")) == len(bench.run_set("product")) == len(bench.run_set("frozen")) == 24
    assert len(bench.run_set("perm")) == 18 and len(bench.run_set("pred")) == 8 and len(bench.run_set("tiedmatched")) == 12


def test_mnist_review_split_is_the_grid_split():
    import data
    import train as grid_train
    try:
        a = datasets.make_split("mnist8_10class", 43)
    except Exception as e:                              # no MNIST file and no download here
        pytest.skip(f"MNIST not available: {e!r}"[:120])
    b = data.make_split(grid_train.Run("10class", 8, 4, "coherent", 43))
    assert all(torch.equal(u, v) for u, v in zip(a, b))
    assert bench.BRun("mnist8_10class", 43).enc_scale == grid_train.Run("10class", 8, 4, "coherent", 43).enc_scale


@pytest.mark.parametrize("model", ["seqrnn", "mps", "hmm"])
def test_structured_classical_models_follow_the_patches(model):
    for d, T in (("letter", 4), ("fashion", 49)):
        run = bench_struct.SRun(d, model, 0.01, 42)
        m = bench_struct.build(run)
        out = m(torch.rand(3, T, 4))
        assert out.shape == (3, run.spec.n_classes) and torch.isfinite(out).all()
        if model != "hmm":
            assert abs(bench_struct.count(m) - run.target) / run.target < 0.25


def test_structured_grid_lists_every_run_once():
    runs = bench_struct.grid()
    assert len(runs) == 402 and len({r.name for r in runs}) == len(runs)
