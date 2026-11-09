import dataclasses
import json
import sys

import pytest
import torch

sys.path.insert(0, "src")
import classical
import train
from data import TASKS, make_split
from qcircuit import SerializedQC

TINY = dict(train_pc=8, val_pc=4, test_pc=4, epochs=3, batch=8, eval_every=1)


def test_grid_is_complete_and_names_are_unique():
    runs = train.grid()
    assert len(runs) == 96 and len({r.name for r in runs}) == 96                    # 8x8 and 14x14 only
    assert {r.image_size for r in runs} == {8, 14}
    assert {r.epochs for r in runs} == {100}                                         # one fixed number of epochs everywhere
    assert [train.ENC_SCALE[s] for s in (8, 14, 28)] == [1.0, 0.33, 0.08]


def test_classical_grid_covers_every_image_size_and_task():
    runs = classical.grid()
    assert len({r.name for r in runs}) == len(runs) == 3 * 4 * 5 * 3 * 3 and {r.epochs for r in runs} == {100}


@pytest.mark.parametrize("size,layers,k", [(8, 2, 2), (14, 4, 10), (28, 2, 4)])
def test_parameter_formula_matches_the_circuit(size, layers, k):
    model = SerializedQC(n_steps=(size // 2) ** 2, layers=layers, n_out=k)
    assert sum(p.numel() for p in model.parameters()) == classical.quantum_params(size, layers, k)


def test_classical_models_are_sized_like_the_circuit():
    for task in TASKS:
        for size in (8, 14, 28):
            for model, layers in classical.MODELS[1:]:
                run = classical.CRun(model, layers, task, size, 0.01, 42)
                p = sum(q.numel() for q in classical.make_model(run).parameters())
                assert abs(p - run.target) / run.target < 0.15, (task, size, model, layers, p, run.target)


def test_quantum_and_classical_models_see_the_same_split():
    q = train.Run("4class", 8, 2, "coherent", 43, **{k: v for k, v in TINY.items() if k.endswith("_pc")})
    c = classical.CRun("mlp", 2, "4class", 8, 0.01, 43, **{k: v for k, v in TINY.items() if k.endswith("_pc")})
    xq, yq, *_ = make_split(q)
    xc, yc, *_ = make_split(c, patches=False)
    assert xq.shape == (32, 16, 4) and xc.shape == (32, 8, 8) and torch.equal(yq, yc)
    assert torch.allclose(xq.sum((1, 2)), xc.sum((1, 2)))            # same images, the patches are just a reordering


@pytest.mark.parametrize("memory", ["coherent", "measured"])
def test_run_writes_result_and_resumes_exactly(tmp_path, monkeypatch, memory):
    monkeypatch.setattr(train, "CKPT", tmp_path / "ckpt")
    a = train.Run("3v5", 8, 1, memory, 42, **TINY)
    monkeypatch.setattr(train, "RESULTS", tmp_path / "a")
    full = json.load(open(train.train(a)))
    assert full["final"]["epoch"] == 3 and (tmp_path / "a" / f"{a.name}.json").exists() and not (tmp_path / "ckpt" / f"{a.name}.pt").exists()

    monkeypatch.setattr(train, "RESULTS", tmp_path / "b")
    real, calls = torch.save, []

    def flaky(*args, **kw):                                          # the process "dies" while saving the checkpoint of epoch 2
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(*args, **kw)

    monkeypatch.setattr(torch, "save", flaky)
    with pytest.raises(KeyboardInterrupt):
        train.train(a)
    monkeypatch.setattr(torch, "save", real)
    resumed = json.load(open(train.train(a)))
    assert [r["train_loss"] for r in resumed["history"]] == [r["train_loss"] for r in full["history"]]
    assert resumed["final"]["test_acc"] == full["final"]["test_acc"]


def test_classical_run_writes_result(tmp_path, monkeypatch):
    monkeypatch.setattr(classical, "RESULTS", tmp_path)
    for model, match in classical.MODELS:
        res = json.load(open(classical.train(classical.CRun(model, match, "3v5", 8, 0.01, 42, **TINY))))
        assert res["final"]["epoch"] == 3 and 0 <= res["final"]["test_acc"] <= 1


def test_checkpointed_segments_give_the_same_outputs_and_gradients():
    torch.manual_seed(0)
    kw = dict(n_steps=7, n_mem=2, layers=2, n_out=3)
    whole, chunks = SerializedQC(**kw, segment=100), SerializedQC(**kw, segment=3)       # 7 patches: chunks of 3, 3 and 1
    chunks.load_state_dict(whole.state_dict())
    x = torch.rand(4, 7, 4)
    out_whole, out_chunks = whole(x), chunks(x)
    assert torch.allclose(out_whole, out_chunks, atol=1e-10)
    out_whole.square().sum().backward()
    out_chunks.square().sum().backward()
    for (name, a), (_, b) in zip(whole.named_parameters(), chunks.named_parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-9), name


def test_amplitude_model_at_zero_angles_is_the_expectation_of_the_encoded_state():
    from amplitude import AmplitudeQC
    model = AmplitudeQC(3, 2, 2)
    with torch.no_grad():
        model.rot.zero_(); model.chain.zero_()                       # every gate is the identity
    x = torch.rand(5, 8, dtype=torch.float64)
    p = x ** 2 / (x ** 2).sum(1, keepdim=True)                       # outcome probabilities of the encoded state
    bits = torch.tensor([[(i >> (2 - q)) & 1 for q in range(3)] for i in range(8)], dtype=torch.float64)   # qubit 0 = most significant bit
    z = 1 - 2 * bits
    expected = torch.cat([p @ z, p @ (z * z.roll(-1, 1))], 1)        # <Z_q> and the ring <Z_q Z_q+1>
    assert torch.allclose(model.features(x), expected, atol=1e-10)
    assert sum(q.numel() for q in model.parameters()) == 2 * (3 * 3 + 3 * 2) + (6 * 2 + 2)


def test_amplitude_grid_and_input_shapes():
    import amplitude
    runs = amplitude.grid()
    assert len(runs) == 3 * 4 * 2 * 3 * 3 and len({r.name for r in runs}) == len(runs)
    x = torch.rand(6, 28, 28)
    assert amplitude.prepare(x, amplitude.AmpRun(4, 2, "3v5", 28, 42)).shape == (6, 16)
    assert amplitude.prepare(x, amplitude.AmpRun(7, 2, "3v5", 28, 42)).shape == (6, 128)
    assert amplitude.AmpRun(7, 2, "3v5", 8, 42).side == 8 and amplitude.AmpRun(7, 2, "3v5", 14, 42).side == 11


def test_amplitude_run_writes_result(tmp_path, monkeypatch):
    import amplitude
    monkeypatch.setattr(amplitude, "RESULTS", tmp_path)
    res = json.load(open(amplitude.train(amplitude.AmpRun(4, 2, "3v5", 8, 42, **TINY))))
    assert res["final"]["epoch"] == 3 and 0 <= res["final"]["test_acc"] <= 1 and res["params"] == 2 * 21 + 8 * 2 + 2


def test_diagnostics_are_distinct_from_the_grid_and_kept_apart():
    names = {r.name for r in train.grid()}
    d = train.diagnostics()
    assert len(d) == 3 and len({r.name for r in d}) == 3 and not names & {r.name for r in d}
    assert all(r.folder != train.RESULTS for r in d) and all(r.folder == train.RESULTS for r in train.grid())


def test_init_seed_changes_only_the_initial_angles():
    a = train.Run("10class", 8, 2, "measured", 44)
    same = train.Run("10class", 8, 2, "measured", 44, init_seed=44)
    other = train.Run("10class", 8, 2, "measured", 44, init_seed=142)
    swapped = train.Run("10class", 8, 2, "measured", 42, init_seed=44)
    flat = lambda r: torch.cat([p.detach().flatten() for p in train.build_model(r).parameters()])
    assert torch.equal(flat(a), flat(same)) and torch.equal(flat(a), flat(swapped))     # the seed-44 angles on the images of seed 42
    assert not torch.equal(flat(a), flat(other))
    xa, xo, xs = (train.make_split(r)[0] for r in (a, other, swapped))
    assert torch.equal(xa, xo) and not torch.equal(xa, xs)                                  # images follow `seed`, not `init_seed`
