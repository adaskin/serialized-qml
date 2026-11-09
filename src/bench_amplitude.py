"""Amplitude-encoding baselines for the extra datasets (the model of src/amplitude.py on the splits of src/bench.py).

    python src/bench_amplitude.py --workers 4        run everything (finished runs are skipped)
    python src/bench_amplitude.py --list

The whole example (all features / pixels) is the amplitude vector of an n-qubit state, followed by L layers of the serialized
circuit's block (Rot on every qubit, CRot chain), Z + ZZ readout and a linear head; no serialization, so the parameter count depends
on n and L only. Two register sizes per dataset:
  min   the fewest qubits that hold every value without shrinking: ceil(log2(features)): Letter 4, Waveform 5, Wall-Following 5,
        Covertype 6, the 14x14 images 8 (196 of 256 values)
  7     the 7 qubits of the serialized circuit (4 system + 3 ancilla); tabular vectors are zero-padded to 128, the 14x14 images are
        shrunk by area averaging to 11x11 (121 values), as in the MNIST grid (src/amplitude.py)
Run: the 7-qubit register with 4 layers (the frozen serialized setting) and with 8 layers ("min" is supported by the code but not run). Same data splits, training recipe (100 epochs, Adam 0.01 cosine to 5%, batch 256, clip 5,
last epoch) and seeds 42-44 as everything else; balanced accuracy is stored as well. Results: data/results/bench/amplitude.
"""
import os

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import json
import math
import multiprocessing as mp
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

import datasets
from amplitude import AmplitudeQC
from bench_classical import evaluate

RESULTS = Path("data/results/bench/amplitude")
SEEDS, LAYERS, SIZES = (42, 43, 44), (4, 8), ("7",)      # 7 qubits (the serialized circuit's qubit count), 4 layers (its frozen setting) and 8 layers; "min" is supported but not run


@dataclass(frozen=True)
class ARun:
    dataset: str
    size: str                         # "min" | "7": register size, see above
    layers: int
    seed: int
    lr: float = 0.01
    batch: int = 256
    epochs: int = 100
    clip: float = 5.0
    train_cap: int = 10000
    eval_every: int = 10

    @property
    def spec(self):
        return datasets.SPECS[self.dataset]

    @property
    def features(self):
        return self.spec.n_features if self.spec.kind == "tabular" else self.spec.n_features ** 2

    @property
    def qubits(self):
        return 7 if self.size == "7" else math.ceil(math.log2(self.features))

    @property
    def side(self):                   # images: side of the encoded image (shrunk if the register is too small)
        return min(self.spec.n_features, math.isqrt(2 ** self.qubits))

    @property
    def name(self):
        return f"{self.dataset}_amp{self.qubits}-L{self.layers}_seed{self.seed}"


def prepare(x, run):
    """tabular (B, F) or image (B, 14, 14) in [0, 1] -> (B, 2^qubits) amplitude vectors (an all-zero vector would not normalise: set to |0>)."""
    if x.ndim == 3 and run.side < x.shape[-1]:
        x = F.interpolate(x[:, None], size=(run.side, run.side), mode="area")[:, 0]
    x = F.pad(x.flatten(1), (0, 2 ** run.qubits - x[0].numel())).double()
    zero = x.abs().sum(1) == 0
    x[zero, 0] = 1.0
    return x


def train(run):
    path = RESULTS / f"{run.name}.json"
    if path.exists():
        return str(path)
    torch.set_num_threads(1)
    torch.manual_seed(run.seed)
    split = datasets.make_split(run.dataset, run.seed, patches=False, train_cap=run.train_cap)
    xtr, ytr, xva, yva, xte, yte = [prepare(t, run) if i % 2 == 0 else t for i, t in enumerate(split)]
    k = run.spec.n_classes
    model = AmplitudeQC(run.qubits, run.layers, k)
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
            row["val_loss"], row["val_acc"], row["val_bacc"] = evaluate(model, xva, yva, k)
            row["test_loss"], row["test_acc"], row["test_bacc"] = evaluate(model, xte, yte, k)
        hist.append(row)
        sched.step()
    res = dict(run=asdict(run), name=run.name, qubits=run.qubits, params=sum(p.numel() for p in model.parameters()), final=hist[-1],
               history=hist, seconds=time.time() - t0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res))
    f = hist[-1]
    print(f"{run.name}: params={res['params']} train={f['train_acc']:.3f} val={f['val_acc']:.3f} test={f['test_acc']:.3f} "
          f"bacc={f['test_bacc']:.3f} ({res['seconds']:.0f}s)", flush=True)
    return str(path)


def grid():
    return [ARun(d, s, l, seed) for d in datasets.SPECS for s in SIZES for l in LAYERS for seed in SEEDS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    runs = [r for r in grid() if not a.datasets or r.dataset in a.datasets]
    runs.sort(key=lambda r: -(r.spec.n_classes * r.layers * r.qubits * min(r.train_cap, 10000)))
    if a.list:
        print("\n".join(f"{r.name} ({r.qubits} qubits, side {r.side})" for r in runs), f"\n{len(runs)} runs")
        return
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for _ in pool.imap_unordered(train, runs):
            pass


if __name__ == "__main__":
    main()
