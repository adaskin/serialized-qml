"""Classical baselines for the extra datasets, on exactly the splits of src/bench.py (same arrays, same caps, same seeds).

    python src/bench_classical.py --workers 4        everything (finished runs are skipped)
    python src/bench_classical.py --list

Models (torch, same recipe as src/classical.py: 100 epochs, Adam + cosine to 5%, batch 256, clip 5, last epoch; the learning rate
0.001 / 0.003 / 0.01 is picked per model by mean validation accuracy in src/bench_report.py, never by test accuracy):
  logistic   one linear layer on the features / pixels
  mlp        one hidden layer sized to the parameter count of the circuit with L = 4 on the same dataset
  mlp_mb     Letter, Covertype, Landsat, Dry Bean: the MLP sized to the circuit with the multi-basis readout (42 features)
  mlp_head   Letter only: the same MLP sized to the circuit with the MLP head (control of the head experiment)
  cnn        images only, the small CNN of src/classical.py sized the same way
Strong baseline (scikit-learn, not tuned): HistGradientBoostingClassifier (300 iterations, lr 0.1, l2 1.0; without the l2 term seeds 42 and 44 of Covertype diverged to ~60%) on the raw features / pixels.
Also stored: the majority-class accuracy of the test set (chance level for the imbalanced sets). Balanced accuracy = mean per-class recall.
"""
import os

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import datasets
from classical import CNN

RESULTS = Path("data/results/bench/classical")
LRS = (0.001, 0.003, 0.01)
SEEDS = (42, 43, 44)
MULTIBASIS_SETS = ("letter", "covertype", "landsat", "drybean")   # the multi-basis group (more than 4 classes, cheap)
LAYERS = 4                                                   # the circuit the MLP / CNN are sized against


def quantum_params(T, layers, k, n_data=4, n_mem=3, linear=True, hidden=64, feats=None):
    """Trainable parameters of SerializedQC (z+zz readout on all qubits; linear head, or a one-hidden-layer MLP head). Checked in tests."""
    n = n_data + n_mem
    nf = 2 * n if feats is None else feats
    circuit = T * layers * (3 * n + 3 * (n - 1)) + 3 * n_mem + 3 * (n_mem - 1) + 3 * n_mem * n_data + 2 * 3 * n_data
    return circuit + (nf * k + k if linear else nf * hidden + hidden + hidden * k + k)


@dataclass(frozen=True)
class BCRun:
    dataset: str
    model: str                        # logistic | mlp | cnn | hgb
    lr: float                         # 0 for hgb
    seed: int
    epochs: int = 100
    batch: int = 256
    clip: float = 5.0
    train_cap: int = 10000
    eval_every: int = 10

    @property
    def spec(self):
        return datasets.SPECS[self.dataset]

    @property
    def target(self):
        feats = 42 if self.model.endswith("_mb") else None              # *_mb: sized to the circuit with the multi-basis readout
        return quantum_params(self.spec.steps, LAYERS, self.spec.n_classes, linear=self.model != "mlp_head", feats=feats)

    @property
    def name(self):
        return f"{self.dataset}_{self.model}_lr{self.lr:g}_seed{self.seed}"


def make_model(run):
    spec, k = run.spec, run.spec.n_classes
    d = spec.n_features if spec.kind == "tabular" else spec.n_features ** 2
    if run.model == "logistic":
        return nn.Sequential(nn.Flatten(), nn.Linear(d, k))
    if run.model in ("mlp", "mlp_head", "mlp_mb"):                           # mlp_head: sized to the circuit with the MLP head (Letter control)
        h = max(1, round((run.target - k) / (d + 1 + k)))
        return nn.Sequential(nn.Flatten(), nn.Linear(d, h), nn.ReLU(), nn.Linear(h, k))
    count = lambda c: 18 * c * c + 12 * c + 2 * c * k + k
    return CNN(min(range(1, 129), key=lambda c: abs(count(c) - run.target)), k)


@torch.no_grad()
def evaluate(model, x, y, k, bs=1000):
    loss, pred = 0.0, []
    for i in range(0, len(x), bs):
        out = model(x[i: i + bs])
        loss += F.cross_entropy(out, y[i: i + bs], reduction="sum").item()
        pred.append(out.argmax(1))
    pred = torch.cat(pred)
    recall = [(pred[y == c] == c).float().mean().item() for c in range(k) if (y == c).any()]
    return loss / len(x), (pred == y).float().mean().item(), sum(recall) / len(recall)


def balanced(pred, y, k):
    recall = [(pred[y == c] == c).mean() for c in range(k) if (y == c).any()]
    return float(np.mean(recall))


def train_hgb(run):
    from sklearn.ensemble import HistGradientBoostingClassifier
    xtr, ytr, xva, yva, xte, yte = datasets.make_split(run.dataset, run.seed, patches=False, train_cap=run.train_cap)
    flat = lambda a: a.reshape(len(a), -1).numpy()
    t0 = time.time()
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, l2_regularization=1.0, early_stopping=False, random_state=run.seed)
    clf.fit(flat(xtr), ytr.numpy())
    k = run.spec.n_classes
    out = {}
    for key, x, y in (("train", xtr, ytr), ("val", xva, yva), ("test", xte, yte)):
        p = clf.predict(flat(x))
        out[f"{key}_acc"], out[f"{key}_bacc"] = float((p == y.numpy()).mean()), balanced(p, y.numpy(), k)
    return dict(final=out, params=None, seconds=time.time() - t0, history=[])


def train_torch(run):
    torch.set_num_threads(1)
    torch.manual_seed(run.seed)
    xtr, ytr, xva, yva, xte, yte = datasets.make_split(run.dataset, run.seed, patches=False, train_cap=run.train_cap)
    k = run.spec.n_classes
    model = make_model(run)
    opt = torch.optim.Adam(model.parameters(), lr=run.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, run.epochs, eta_min=0.05 * run.lr)
    gen = torch.Generator().manual_seed(run.seed + 1)
    hist, t0 = [], time.time()
    for epoch in range(1, run.epochs + 1):
        perm = torch.randperm(len(xtr), generator=gen)
        tl = tc = 0.0
        for i in range(0, len(xtr), run.batch):
            idx = perm[i: i + run.batch]
            out = model(xtr[idx])
            loss = F.cross_entropy(out, ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), run.clip)
            opt.step()
            tl += loss.item() * len(idx)
            tc += (out.argmax(1) == ytr[idx]).sum().item()
        row = dict(epoch=epoch, train_loss=tl / len(xtr), train_acc=tc / len(xtr))
        if epoch % run.eval_every == 0 or epoch == run.epochs:
            row["val_loss"], row["val_acc"], row["val_bacc"] = evaluate(model, xva, yva, k)
            row["test_loss"], row["test_acc"], row["test_bacc"] = evaluate(model, xte, yte, k)
        hist.append(row)
        sched.step()
    return dict(final=hist[-1], params=sum(p.numel() for p in model.parameters()), seconds=time.time() - t0, history=hist)


def train(run):
    path = RESULTS / f"{run.name}.json"
    if path.exists():
        return str(path)
    res = train_hgb(run) if run.model == "hgb" else train_torch(run)
    ytest = datasets.make_split(run.dataset, run.seed, patches=False, train_cap=run.train_cap)[5]
    res.update(run=asdict(run), name=run.name, target_params=run.target, majority_test=float(np.bincount(ytest.numpy()).max() / len(ytest)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res))
    f = res["final"]
    print(f"{run.name}: params={res['params']} val={f['val_acc']:.3f} test={f['test_acc']:.3f} bacc={f['test_bacc']:.3f} "
          f"({res['seconds']:.0f}s)", flush=True)
    return str(path)


def grid():
    runs = []
    for d, spec in datasets.SPECS.items():
        for seed in SEEDS:
            runs.append(BCRun(d, "hgb", 0.0, seed))
            if d in MULTIBASIS_SETS:                             # re-sized to the multi-basis circuit (42 readout features)
                runs += [BCRun(d, m, lr, seed) for m in ("mlp_mb",) + (("cnn_mb",) if spec.kind == "image" else ()) for lr in LRS]
            if d == "letter":                                    # control of the head experiment: same parameter count as circuit + MLP head
                runs += [BCRun(d, "mlp_head", lr, seed) for lr in LRS]
            for m in ("logistic", "mlp") + (("cnn",) if spec.kind == "image" else ()):
                runs += [BCRun(d, m, lr, seed) for lr in LRS]
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    runs = [r for r in grid() if not a.datasets or r.dataset in a.datasets]
    if a.list:
        print("\n".join(r.name for r in runs), f"\n{len(runs)} runs")
        return
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for _ in pool.imap_unordered(train, runs):
            pass


if __name__ == "__main__":
    main()
