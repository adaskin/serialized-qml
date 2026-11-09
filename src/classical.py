"""Classical baselines on the same images, splits, epochs and optimizer as the quantum grid.

    python src/classical.py --workers 4        run every baseline (finished runs are skipped)
    python src/classical.py --list

Models: logistic regression, a one-hidden-layer MLP and a small CNN (two 3x3 conv + ReLU + max-pool blocks, global average pool,
linear layer). The MLP and the CNN are sized to have about as many parameters as the quantum circuit with 2 and with 4 layers on
the same image size and task. Training: 100 epochs, Adam with cosine decay to 5%, batch 256, gradient clip 5, last-epoch numbers.
The learning rate is the only tuned setting: 0.001, 0.003 and 0.01 are all run, and src/report.py keeps, per model, the one with
the best mean validation accuracy (never the test accuracy).
"""
import os

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):     # one core per run: set before numpy / torch load
    os.environ.setdefault(var, "1")

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import download_data
from data import TASKS, make_split

RESULTS = Path("data/results/classical")
LRS = (0.001, 0.003, 0.01)
SIZES, SEEDS = (8, 14, 28), (42, 43, 44)
MODELS = (("logistic", None), ("mlp", 2), ("mlp", 4), ("cnn", 2), ("cnn", 4))   # (model, quantum layers it is sized against)


def quantum_params(image_size, layers, n_classes, patch=(2, 2), n_mem=3, feats=None):
    """Trainable parameters of SerializedQC (default readout z+zz on all qubits: 2n features; feats=6n for the multi-basis readout);
    checked against the class in the tests."""
    nd, T = patch[0] * patch[1], (image_size // patch[0]) * (image_size // patch[1])
    n = nd + n_mem
    w_t = layers * (3 * n + 3 * (n - 1))
    w_c = 3 * n_mem + 3 * (n_mem - 1) + 3 * n_mem * nd
    feats = 2 * n if feats is None else feats
    return T * w_t + w_c + 2 * 3 * nd + (feats * n_classes + n_classes)


@dataclass(frozen=True)
class CRun:
    model: str                        # logistic | mlp | cnn
    match: int                        # quantum layers whose parameter count the model is sized against (None for logistic)
    task: str
    image_size: int
    lr: float
    seed: int
    dataset: str = "mnist"
    patch: tuple = (2, 2)
    batch: int = 256
    epochs: int = 100
    clip: float = 5.0
    train_pc: int = 1000
    val_pc: int = 200
    test_pc: int = None
    eval_every: int = 10
    feats: int = 14                   # readout features of the circuit it is sized against (42: multi-basis readout)

    @property
    def classes(self):
        return TASKS[self.task]

    @property
    def target(self):
        return None if self.match is None else quantum_params(self.image_size, self.match, len(self.classes), feats=self.feats)

    @property
    def folder(self):                 # the multi-basis-sized baselines stay out of the grid's tables
        return RESULTS if self.feats == 14 else RESULTS.parent / "multibasis"

    @property
    def name(self):
        m = self.model + ("" if self.match is None else f"-L{self.match}") + ("" if self.feats == 14 else f"-f{self.feats}")
        return f"{self.dataset}_{self.task}_s{self.image_size}_{m}_lr{self.lr:g}_seed{self.seed}"


class CNN(nn.Module):
    def __init__(self, c, k):
        super().__init__()
        self.conv1, self.conv2, self.out = nn.Conv2d(1, c, 3, padding=1), nn.Conv2d(c, 2 * c, 3, padding=1), nn.Linear(2 * c, k)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x[:, None])), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.out(x.mean((2, 3)))


def make_model(run):
    side, k, target = run.image_size, len(run.classes), run.target
    if run.model == "logistic":
        return nn.Sequential(nn.Flatten(), nn.Linear(side * side, k))
    if run.model == "mlp":                                       # side^2 -> h -> k has h (side^2 + 1 + k) + k parameters
        h = max(1, round((target - k) / (side * side + 1 + k)))
        return nn.Sequential(nn.Flatten(), nn.Linear(side * side, h), nn.ReLU(), nn.Linear(h, k))
    count = lambda c: 18 * c * c + 12 * c + 2 * c * k + k        # parameters of CNN(c, k)
    return CNN(min(range(1, 129), key=lambda c: abs(count(c) - target)), k)


@torch.no_grad()
def evaluate(model, x, y, bs=1000):
    loss = corr = 0.0
    for i in range(0, len(x), bs):
        out = model(x[i: i + bs])
        loss += F.cross_entropy(out, y[i: i + bs], reduction="sum").item()
        corr += (out.argmax(1) == y[i: i + bs]).sum().item()
    return loss / len(x), corr / len(x)


def train(run):
    path = run.folder / f"{run.name}.json"
    if path.exists():
        return str(path)
    torch.set_num_threads(1)
    torch.manual_seed(run.seed)
    xtr, ytr, xva, yva, xte, yte = make_split(run, patches=False)
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
        row = dict(epoch=epoch, train_loss=tl / len(xtr), train_acc=tc / len(xtr), seconds=time.time() - t0)
        if epoch % run.eval_every == 0 or epoch == run.epochs:
            row["val_loss"], row["val_acc"] = evaluate(model, xva, yva)
            row["test_loss"], row["test_acc"] = evaluate(model, xte, yte)
        hist.append(row)
        sched.step()
    res = dict(run=asdict(run), name=run.name, params=sum(p.numel() for p in model.parameters()), final=hist[-1], history=hist,
               seconds=time.time() - t0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res))
    f = hist[-1]
    print(f"{run.name}: params={res['params']} train={f['train_acc']:.3f} val={f['val_acc']:.3f} test={f['test_acc']:.3f} "
          f"({res['seconds']:.0f}s)", flush=True)
    return str(path)


def multibasis():
    """MLP and CNN re-sized to the circuit with the multi-basis readout (42 features), 10 classes 8x8, L = 4."""
    return [CRun(m, 4, "10class", 8, lr, seed, feats=42) for m in ("mlp", "cnn") for lr in LRS for seed in SEEDS]


def grid():
    return [CRun(m, match, task, size, lr, seed) for size in SIZES for task in TASKS for m, match in MODELS for lr in LRS for seed in SEEDS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--set", default="grid", choices=("grid", "multibasis"), help="multibasis: MLP / CNN sized to the 42-feature circuit")
    a = ap.parse_args()
    runs = sorted({"grid": grid, "multibasis": multibasis}[a.set](), key=lambda r: -r.image_size ** 2 * len(r.classes) * (r.target or 1))      # slowest first
    if a.list:
        print("\n".join(r.name for r in runs), f"\n{len(runs)} runs")
        return
    download_data.ensure("mnist")
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for _ in pool.imap_unordered(train, runs):
            pass


if __name__ == "__main__":
    main()
