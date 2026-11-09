"""Amplitude-encoding baseline: the whole image is the amplitude vector of an n-qubit state, followed by L variational layers.

    python src/amplitude.py --workers 4        run every baseline (finished runs are skipped)
    python src/amplitude.py --list

n qubits hold 2^n pixel values, so a larger image is first shrunk (area average) to the largest square that fits: 4 qubits take 4x4
(16 values), 7 qubits take 8x8 as it is and 11x11 (121 of 128 values) for 14x14 and 28x28; the vector is zero-padded to 2^n and
normalised by the embedding. The layers are those of the serialized circuit (Rot on every qubit, then a CRot chain), the readout is
<Z_q> and <Z_q Z_q+1> (ring) followed by a linear layer. Training is the same as in the grid: 100 epochs, Adam lr 0.01 with cosine
decay to 5%, batch 256, gradient clip 5, last epoch reported, same images and splits. The number of parameters depends on the qubits
and layers only, not on the image size.
"""
import os

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):     # one core per run: set before numpy / torch load
    os.environ.setdefault(var, "1")

import argparse
import json
import math
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F

import download_data
from classical import evaluate
from data import TASKS, make_split

RESULTS = Path("data/results/amplitude")
SIZES, SEEDS, QUBITS, LAYERS = (8, 14, 28), (42, 43, 44), (4, 7), (2, 4, 8)


class AmplitudeQC(nn.Module):
    """Input (batch, 2^n_qubits) non-negative values; output (batch, n_out) logits."""

    def __init__(self, n_qubits=7, layers=2, n_out=10, init_scale=0.08):
        super().__init__()
        n = self.n = n_qubits
        self.rot = nn.Parameter(torch.randn(layers, n, 3, dtype=torch.float64) * init_scale)
        self.chain = nn.Parameter(torch.randn(layers, n - 1, 3, dtype=torch.float64) * init_scale)
        self.pairs = [(i, (i + 1) % n) for i in range(n if n > 2 else n - 1)]
        self.linear = nn.Linear(n + len(self.pairs), n_out, dtype=torch.float64)
        nn.init.zeros_(self.linear.bias)
        self.qnode = qml.QNode(self.circuit, qml.device("default.qubit", wires=n), interface="torch", diff_method="backprop")

    def circuit(self, x, rot, chain):
        qml.AmplitudeEmbedding(x, wires=range(self.n), normalize=True)
        for l in range(len(rot)):
            for q in range(self.n):
                qml.Rot(*rot[l, q], wires=q)
            for q in range(self.n - 1):
                qml.CRot(*chain[l, q], wires=[q, q + 1])
        return ([qml.expval(qml.PauliZ(q)) for q in range(self.n)] +
                [qml.expval(qml.PauliZ(a) @ qml.PauliZ(b)) for a, b in self.pairs])

    def features(self, x):
        return torch.stack(list(self.qnode(x, self.rot, self.chain)), dim=-1)

    def forward(self, x):
        return self.linear(self.features(x))


@dataclass(frozen=True)
class AmpRun:
    qubits: int
    layers: int
    task: str
    image_size: int
    seed: int
    dataset: str = "mnist"
    patch: tuple = (2, 2)             # unused (the split code reads it)
    lr: float = 0.01
    batch: int = 256
    epochs: int = 100
    clip: float = 5.0
    train_pc: int = 1000
    val_pc: int = 200
    test_pc: int = None
    eval_every: int = 10

    @property
    def classes(self):
        return TASKS[self.task]

    @property
    def side(self):
        return min(self.image_size, math.isqrt(2 ** self.qubits))        # side of the image that is encoded

    @property
    def name(self):
        return f"{self.dataset}_{self.task}_s{self.image_size}_amp{self.qubits}-L{self.layers}_seed{self.seed}"


def prepare(x, run):
    """(B, H, W) images -> (B, 2^qubits) amplitude vectors."""
    if run.side < x.shape[-1]:
        x = F.interpolate(x[:, None], size=(run.side, run.side), mode="area")[:, 0]
    x = x.flatten(1)
    return F.pad(x, (0, 2 ** run.qubits - x.shape[1])).double()


def train(run):
    path = RESULTS / f"{run.name}.json"
    if path.exists():
        return str(path)
    torch.set_num_threads(1)
    torch.manual_seed(run.seed)
    xtr, ytr, xva, yva, xte, yte = [prepare(t, run) if i % 2 == 0 else t for i, t in enumerate(make_split(run, patches=False))]
    model = AmplitudeQC(run.qubits, run.layers, len(run.classes))
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


def grid():
    return [AmpRun(q, l, task, size, seed) for size in SIZES for task in TASKS for q in QUBITS for l in LAYERS for seed in SEEDS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    runs = sorted(grid(), key=lambda r: -len(r.classes) * r.layers * r.qubits)                  # slowest first
    if a.list:
        print("\n".join(r.name for r in runs), f"\n{len(runs)} runs")
        return
    download_data.ensure("mnist")
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for _ in pool.imap_unordered(train, runs):
            pass


if __name__ == "__main__":
    main()
