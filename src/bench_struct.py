"""Classical models with the structure of the serialized circuit (review control), on exactly the splits and patches of src/bench.py.

    python src/bench_struct.py --workers 8          everything (finished runs are skipped); --list prints the runs

All read the input as the circuit does, patch by patch (B, T, 4), and have position-specific parameters, so their size grows with T:
  seqrnn  site-dependent recurrent network: h_0 = 0, h_t = tanh(A_t h_{t-1} + B_t x_t + c_t), logits = W h_T + b; the hidden size H is
          chosen so that the parameter count is closest to that of the circuit with L = 4 on the same dataset
  mps     matrix-product-state classifier: every feature is a site with the local map (cos(pi x / 2), sin(pi x / 2)) and its own tensor
          (2, D, D); the boundary vector is carried from site to site (normalized at every site) and a linear layer maps it to the
          logits; the bond dimension D is chosen as for H
  hmm     input-driven 8-state stochastic finite-state machine, the classical analogue of the measured circuit (3 bits between patches):
          p_0 = state 0, p_t = p_{t-1} P_t(x_t) with P_t(x)[s, :] = softmax(U_t[s] x + c_t[s]); logits = W [p_T, x_T] + b. Its size is
          fixed by the 8 states (8 * 8 * 5 parameters per patch), not matched
  mlp     the one-hidden-layer MLP of src/bench_classical.py again, seed 42 on the tabular sets, now storing test predictions
Recipe of src/bench_classical.py: 100 epochs, Adam + cosine to 5%, batch 256, clip 5, last epoch; learning rate 0.001 / 0.003 / 0.01
chosen afterwards by mean validation accuracy. Test predictions of the last epoch are stored for paired tests.
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
import torch.nn as nn
import torch.nn.functional as F

import datasets
from bench_classical import BCRun, LRS, SEEDS, make_model, quantum_params

RESULTS = Path("data/results/review/classical")
N_STATES = 8


class SeqRNN(nn.Module):
    def __init__(self, T, k, H, d=4):
        super().__init__()
        self.A = nn.Parameter(torch.randn(T, H, H) / math.sqrt(H))
        self.B = nn.Parameter(torch.randn(T, H, d) / math.sqrt(d))
        self.c = nn.Parameter(torch.zeros(T, H))
        self.out = nn.Linear(H, k)

    def forward(self, x):                                   # x: (B, T, 4)
        h = x.new_zeros(len(x), self.A.shape[1])
        for t in range(x.shape[1]):
            h = torch.tanh(h @ self.A[t].T + x[:, t] @ self.B[t].T + self.c[t])
        return self.out(h)


class MPS(nn.Module):
    def __init__(self, T, k, D, d=4):
        super().__init__()
        n = T * d
        eye = torch.eye(D).expand(n, 2, D, D)
        self.A = nn.Parameter(eye + 0.1 * torch.randn(n, 2, D, D))   # near the identity, so the product neither vanishes nor explodes
        self.out = nn.Linear(D, k)

    def forward(self, x):
        x = x.reshape(len(x), -1)                           # sites in serialisation order
        phi = torch.stack([torch.cos(math.pi * x / 2), torch.sin(math.pi * x / 2)], -1)   # (B, n, 2)
        v = x.new_ones(len(x), self.A.shape[-1]) / math.sqrt(self.A.shape[-1])
        for j in range(x.shape[1]):
            v = torch.einsum("bd,bde->be", v, torch.einsum("bp,pde->bde", phi[:, j], self.A[j]))
            v = v / (v.norm(dim=1, keepdim=True) + 1e-12)
        return self.out(v)


class HMM(nn.Module):
    def __init__(self, T, k, S=N_STATES, d=4):
        super().__init__()
        self.U = nn.Parameter(0.1 * torch.randn(T, S, S, d))
        self.c = nn.Parameter(torch.zeros(T, S, S))
        self.out = nn.Linear(S + d, k)

    def forward(self, x):
        S = self.c.shape[1]
        p = x.new_zeros(len(x), S)
        p[:, 0] = 1
        for t in range(x.shape[1]):
            P = torch.softmax(torch.einsum("sud,bd->bsu", self.U[t], x[:, t]) + self.c[t], dim=-1)   # (B, S, S), rows sum to 1
            p = torch.einsum("bs,bsu->bu", p, P)
        return self.out(torch.cat([p, x[:, -1]], 1))


def count(m):
    return sum(p.numel() for p in m.parameters())


@dataclass(frozen=True)
class SRun:
    dataset: str
    model: str                        # seqrnn | mps | hmm | mlp
    lr: float
    seed: int
    epochs: int = 100
    batch: int = 256
    clip: float = 5.0
    train_cap: int = 10000
    eval_every: int = 10

    @property
    def spec(self):
        return datasets.ALL_SPECS[self.dataset]

    @property
    def target(self):
        return quantum_params(self.spec.steps, 4, self.spec.n_classes)

    @property
    def name(self):
        return f"{self.dataset}_{self.model}_lr{self.lr:g}_seed{self.seed}"


def build(run):
    T, k = run.spec.steps, run.spec.n_classes
    if run.model == "seqrnn":
        return SeqRNN(T, k, min(range(1, 129), key=lambda H: abs(count(SeqRNN(T, k, H)) - run.target)))
    if run.model == "mps":
        return MPS(T, k, min(range(1, 33), key=lambda D: abs(count(MPS(T, k, D)) - run.target)))
    if run.model == "hmm":
        return HMM(T, k)
    return make_model(BCRun(run.dataset, "mlp", run.lr, run.seed))


@torch.no_grad()
def evaluate(model, x, y, k, flat, bs=1000, preds=None):
    loss, pred = 0.0, []
    for i in range(0, len(x), bs):
        out = model(flat(x[i: i + bs]))
        loss += F.cross_entropy(out, y[i: i + bs], reduction="sum").item()
        pred.append(out.argmax(1))
    pred = torch.cat(pred)
    if preds is not None:
        preds.extend(pred.tolist())
    recall = [(pred[y == c] == c).float().mean().item() for c in range(k) if (y == c).any()]
    return loss / len(x), (pred == y).float().mean().item(), sum(recall) / len(recall)


def train(run):
    path = RESULTS / f"{run.name}.json"
    if path.exists():
        return str(path)
    torch.set_num_threads(1)
    torch.manual_seed(run.seed)
    mlp = run.model == "mlp"
    xtr, ytr, xva, yva, xte, yte = datasets.make_split(run.dataset, run.seed, patches=not mlp, train_cap=run.train_cap)
    flat = lambda a: a
    k = run.spec.n_classes
    model = build(run)
    opt = torch.optim.Adam(model.parameters(), lr=run.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, run.epochs, eta_min=0.05 * run.lr)
    gen = torch.Generator().manual_seed(run.seed + 1)
    hist, t0, preds = [], time.time(), None
    for epoch in range(1, run.epochs + 1):
        perm = torch.randperm(len(xtr), generator=gen)
        tl = tc = 0.0
        for i in range(0, len(xtr), run.batch):
            idx = perm[i: i + run.batch]
            out = model(flat(xtr[idx]))
            loss = F.cross_entropy(out, ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), run.clip)
            opt.step()
            tl += loss.item() * len(idx)
            tc += (out.argmax(1) == ytr[idx]).sum().item()
        row = dict(epoch=epoch, train_loss=tl / len(xtr), train_acc=tc / len(xtr))
        if epoch % run.eval_every == 0 or epoch == run.epochs:
            row["val_loss"], row["val_acc"], row["val_bacc"] = evaluate(model, xva, yva, k, flat)
            preds = [] if epoch == run.epochs else None
            row["test_loss"], row["test_acc"], row["test_bacc"] = evaluate(model, xte, yte, k, flat, preds=preds)
        hist.append(row)
        sched.step()
    res = dict(run=asdict(run), name=run.name, params=count(model), target_params=run.target, final=hist[-1], history=hist,
               seconds=time.time() - t0, test_pred=preds, test_y=yte.tolist())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res))
    f = res["final"]
    print(f"{run.name}: params={res['params']} (circuit {run.target}) val={f['val_acc']:.3f} test={f['test_acc']:.3f} ({res['seconds']:.0f}s)",
          flush=True)
    return str(path)


DATASETS = datasets.BATCHES["A1"] + datasets.BATCHES["A2"] + ("mnist8_3v5", "mnist8_10class")


def grid():
    runs = [SRun(d, m, lr, s) for d in DATASETS for m in ("seqrnn", "mps", "hmm") for s in SEEDS for lr in LRS]
    runs += [SRun(d, "mlp", lr, 42) for d in DATASETS if datasets.ALL_SPECS[d].kind == "tabular" for lr in LRS]
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    runs = [r for r in grid() if (not a.datasets or r.dataset in a.datasets) and (not a.models or r.model in a.models)]
    if a.list:
        print("\n".join(f"{r.name} target {r.target}" for r in runs), f"\n{len(runs)} runs")
        return
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for _ in pool.imap_unordered(train, runs):
            pass


if __name__ == "__main__":
    main()
