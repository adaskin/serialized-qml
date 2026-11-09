"""Serialized circuit on the extra datasets (Tier A of the dataset plan). The frozen configuration of the MNIST grid is reused unchanged.

    python src/bench.py --list                              the runs of every set, with the estimated cost
    python src/bench.py --time [--batch A2]                 seconds per training step / estimated seconds per epoch (3 steps per dataset)
    python src/bench.py --set pilot --workers 4             1 seed, 10 epochs, coherent L=4, linear head
    python src/bench.py --set main --workers 10             3 seeds (42-44), 100 epochs, coherent L=4, linear head
    python src/bench.py --set measured --workers 6          measured memory (ancilla measured after every patch), Letter / Waveform / Wall / Covertype, 3 seeds
    python src/bench.py --set readout --workers 4           readout test: Z, ZZ + X, XX, Y, YY (42 features instead of 14) for Letter / Wall / Waveform, 3 seeds
    python src/bench.py --set head --workers 4              one head experiment: MLP head instead of the linear head, Letter only (3 seeds)
    python src/bench.py --set review --workers 10           Tier 1 controls asked for by the (mock) review, results in data/results/review:
                                                            tied blocks, product-gate blocks, frozen random circuit, feature order, reruns
                                                            that store test predictions (see run_set); --list prints them

Frozen (identical to src/train.py): 4 system qubits (one value each) + 3 ancilla qubits, xyz encoding, Z + ZZ readout, init 0.08,
coherent memory, Adam lr 0.01 cosine to 5%, batch 256, gradient clip 5, last epoch reported, no early stopping or model selection.
Only dataset-dependent: the number of patches T (tabular: ceil(features / 4), images: 49 at 14x14) and the encoding scale
min(1, 16 / T) (= the 1.0 / 0.33 of the 8x8 / 14x14 MNIST grid, where it is about 16 / number of patches).
Besides accuracy the results store the balanced accuracy (mean per-class recall) of val and test.
One JSON per run in data/results/bench/quantum, resumable from data/ckpt/bench.
"""
import os

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import json
import math
import multiprocessing as mp
import resource
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import datasets
from qcircuit import SerializedQC

RESULTS = Path("data/results/bench/quantum")
REVIEW = Path("data/results/review/quantum")         # Tier 1 review controls (kept out of the tables of the main benchmark)
CKPT = Path("data/ckpt/bench")
SEEDS = (42, 43, 44)
HEAD_HIDDEN = 64


@dataclass(frozen=True)
class BRun:
    dataset: str
    seed: int
    layers: int = 4
    head: str = "linear"              # linear | mlp (one hidden layer of HEAD_HIDDEN units on the Z + ZZ features)
    readout: str = "z+zz"             # "z+zz" is the frozen setting; see SerializedQC for the other terms (x, xx, y, yy)
    epochs: int = 100
    train_cap: int = 10000
    n_mem: int = 3
    memory: str = "coherent"
    lr: float = 0.01
    batch: int = 256
    clip: float = 5.0
    init_scale: float = 0.08
    eval_every: int = 10
    # review controls (defaults = the main benchmark)
    tie: bool = False                 # one W block reused at every patch (a recurrent cell) instead of position-specific blocks
    entangle: bool = True             # False: every CRot replaced by a Rot on its target (product state, same parameter count)
    frozen: bool = False              # circuit angles drawn uniformly in [-pi, pi) and frozen, encoding frozen; only the linear head trains
    perm: int = 0                     # > 0: tabular features in a random order (numpy seed perm) before grouping into patches
    tag: str = ""                     # "pred": rerun of a main-benchmark run (same settings) that stores the test predictions

    @property
    def spec(self):
        return datasets.ALL_SPECS[self.dataset]

    @property
    def review(self):
        return self.tie or not self.entangle or self.frozen or self.perm or self.tag or self.spec.kind == "mnist"

    @property
    def folder(self):
        return REVIEW if self.review else RESULTS

    @property
    def n_steps(self):
        return self.spec.steps

    @property
    def enc_scale(self):
        return min(1.0, round(16 / self.n_steps, 2))

    @property
    def name(self):
        tail = ("" if self.readout == "z+zz" else f"_ro{self.readout.replace('+', '')}") + ("" if self.train_cap == 10000 else f"_cap{self.train_cap}") + ("" if self.epochs == 100 else f"_e{self.epochs}")
        tail += ("_tied" if self.tie else "") + ("" if self.entangle else "_product") + ("_frozen" if self.frozen else "")
        tail += (f"_perm{self.perm}" if self.perm else "") + (f"_{self.tag}" if self.tag else "")
        return f"{self.dataset}_{self.memory}_L{self.layers}_{self.head}_seed{self.seed}{tail}"

    @property
    def cost(self):
        return (self.epochs * math.ceil(self.train_cap / self.batch) * self.n_steps * (13 * self.layers + 29) * (4.5 if self.memory == "measured" else 1)
                * (0.4 if self.frozen else 1))

    @property
    def memory_gb(self):
        """Estimated peak RAM (same formulas as src/train.py)."""
        if self.memory == "measured":
            return 1.5 + 1.3 * 0.0042 * self.batch * (1 + 0.125 * (self.layers - 2))
        return 1.0 + 1.3 * self.batch * min(self.n_steps, 8) * (250 + 70 * (self.layers - 2)) * 1e-6


def build_model(run):
    torch.manual_seed(run.seed)
    model = SerializedQC(n_steps=run.n_steps, n_data=4, n_mem=run.n_mem, n_out=run.spec.n_classes, layers=run.layers,
                         memory=run.memory, init_scale=run.init_scale, enc_scale=run.enc_scale, readout=run.readout,
                         tie_weights=run.tie, entangle=run.entangle)
    if run.frozen:                                                     # random features: a random circuit, only the linear head is trained
        with torch.no_grad():
            for p in (model.rot, model.chain, model.anc_rot, model.anc_chain, model.coupling):
                p.uniform_(-math.pi, math.pi)
        for name, p in model.named_parameters():
            p.requires_grad_(name.startswith("linear"))
    if run.head == "mlp":                                              # same circuit, a small classical head instead of the linear layer
        nf = model.linear.in_features
        model.linear = nn.Sequential(nn.Linear(nf, HEAD_HIDDEN), nn.ReLU(), nn.Linear(HEAD_HIDDEN, run.spec.n_classes)).double()
    return model


@torch.no_grad()
def evaluate(model, x, y, k, bs=500, preds=None):
    """(loss, accuracy, balanced accuracy); with a list `preds`, the predicted classes are appended to it."""
    loss, pred = 0.0, []
    for i in range(0, len(x), bs):
        out = model(x[i: i + bs])
        loss += F.cross_entropy(out, y[i: i + bs], reduction="sum").item()
        pred.append(out.argmax(1))
    pred = torch.cat(pred)
    if preds is not None:
        preds.extend(pred.tolist())
    recall = [(pred[y == c] == c).float().mean().item() for c in range(k) if (y == c).any()]
    return loss / len(x), (pred == y).float().mean().item(), sum(recall) / len(recall)


def train(run):
    path = run.folder / f"{run.name}.json"
    if path.exists():
        return str(path)
    torch.set_num_threads(1)
    xtr, ytr, xva, yva, xte, yte = datasets.make_split(run.dataset, run.seed, train_cap=run.train_cap, perm=run.perm)
    k = run.spec.n_classes
    model = build_model(run)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=run.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, run.epochs, eta_min=0.05 * run.lr)
    gen = torch.Generator().manual_seed(run.seed + 1)
    hist, t0, first = [], time.time(), 1
    ckpt = CKPT / f"{run.name}.pt"
    if ckpt.exists():
        c = torch.load(ckpt, weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"]); gen.set_state(c["gen"])
        hist, first, t0 = c["hist"], len(c["hist"]) + 1, time.time() - c["hist"][-1]["seconds"]
    preds = None
    for epoch in range(first, run.epochs + 1):
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
        row = dict(epoch=epoch, train_loss=tl / len(xtr), train_acc=tc / len(xtr), seconds=time.time() - t0,
                   peak_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
        if epoch % run.eval_every == 0 or epoch == run.epochs:
            row["val_loss"], row["val_acc"], row["val_bacc"] = evaluate(model, xva, yva, k)
            preds = [] if epoch == run.epochs else None                # per-example test predictions of the reported (last) epoch
            row["test_loss"], row["test_acc"], row["test_bacc"] = evaluate(model, xte, yte, k, preds=preds)
        hist.append(row)
        sched.step()
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), gen=gen.get_state(), hist=hist),
                   f"{ckpt}.tmp")
        os.replace(f"{ckpt}.tmp", ckpt)
        print(f"{run.name} epoch {epoch}: train {row['train_acc']:.3f}" + (f" val {row['val_acc']:.3f}" if "val_acc" in row else "")
              + f" ({row['seconds']:.0f}s)", flush=True)
    if preds is None:                                                  # resumed after the last epoch's evaluation was lost: evaluate again
        preds = []
        evaluate(model, xte, yte, k, preds=preds)
    res = dict(run=asdict(run), name=run.name, params=sum(p.numel() for p in model.parameters()),
               trainable=sum(p.numel() for p in model.parameters() if p.requires_grad), n_train=len(ytr), n_val=len(yva),
               n_test=len(yte), final=hist[-1], history=hist, seconds=time.time() - t0, test_pred=preds, test_y=yte.tolist())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res))
    ckpt.unlink(missing_ok=True)
    f = hist[-1]
    print(f"DONE {run.name}: params={res['params']} train={f['train_acc']:.3f} val={f['val_acc']:.3f} test={f['test_acc']:.3f} "
          f"bacc={f['test_bacc']:.3f} ({res['seconds'] / 3600:.2f} h)", flush=True)
    return str(path)


def time_steps(run, n=3):
    """Seconds per training step (batch 256) of this run on one core, and the resulting estimate for an epoch / the whole run."""
    torch.set_num_threads(1)
    xtr, ytr, *_ = datasets.make_split(run.dataset, run.seed, train_cap=run.train_cap, perm=run.perm)
    model = build_model(run)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=run.lr)
    ts = []
    for i in range(n + 1):
        t = time.time()
        loss = F.cross_entropy(model(xtr[: run.batch]), ytr[: run.batch])
        opt.zero_grad(); loss.backward(); opt.step()
        ts.append(time.time() - t)
    step = sum(ts[1:]) / n                                             # the first step includes start-up
    epoch = step * math.ceil(len(ytr) / run.batch)
    return dict(dataset=run.dataset, n_train=len(ytr), T=run.n_steps, sec_step=step, sec_epoch=epoch, hours_100_epochs=epoch * 100 / 3600,
                peak_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)


# ----------------------------------------------------------------------------- run sets
READOUT_SETS = ("letter", "wallrobot", "waveform")
ALL_PAULI = "z+zz+x+xx+y+yy"
HEAD_SETS = ("letter",)                  # one experiment only: the 26-class set (the main runs keep the linear head of the grid)
# Tier 1 review controls: the cheap core set (coherent, L = 4, 3 seeds), compared with the main benchmark / MNIST grid runs of the same seed
REVIEW_TABULAR = ("waveform", "wallrobot", "letter", "drybean", "landsat", "spambase")
REVIEW_CORE = REVIEW_TABULAR + ("mnist8_3v5", "mnist8_10class")
TIED_MATCHED = ("waveform", "wallrobot", "letter", "drybean")   # 4-6 patches: the tied block made L * T layers deep (same parameter count)


def run_set(name, batch="A1"):
    ds = datasets.BATCHES[batch]
    if name == "pilot":
        return [BRun(d, 42, epochs=10) for d in ds]
    if name == "main":
        return [BRun(d, s) for d in ds for s in SEEDS]
    if name == "measured":                  # hardware-style memory (ancilla measured after every patch); only the tabular sets (4-15 patches):
        return [BRun(d, s, memory="measured") for d in ds if datasets.SPECS[d].kind == "tabular" for s in SEEDS]   # images: days per run
    if name == "readout":                   # simple test: Z, ZZ plus X, XX, Y, YY expectation values, all given to the linear head
        return [BRun(d, s, readout=ALL_PAULI) for d in READOUT_SETS for s in SEEDS]
    if name == "multibasis":                # multi-basis readout for the cheap tasks with more than 4 classes (Letter: already in "readout")
        return [BRun(d, s, readout=ALL_PAULI) for d in ("covertype", "landsat", "drybean") for s in SEEDS]
    if name == "control":                   # capacity control of the readout test: Letter, Z + ZZ readout but deeper (L = 9: 1869 parameters, about the 1817 of the 42-feature readout at L = 4)
        return [BRun("letter", s, layers=9) for s in SEEDS]
    if name == "head":
        return [BRun(d, s, head="mlp") for d in HEAD_SETS for s in SEEDS]
    if name == "review":                    # every Tier 1 circuit run, cheapest first within a seed (see the subsets below)
        return [r for sub in ("pred", "tied", "tiedmatched", "product", "frozen", "perm") for r in run_set(sub)]
    if name == "pred":                      # main-benchmark settings again, seed 42, now storing test predictions (paired tests)
        return [BRun(d, 42, tag="pred") for d in datasets.BATCHES["A1"] + datasets.BATCHES["A2"] if datasets.SPECS[d].kind == "tabular"]
    if name == "tied":
        return [BRun(d, s, tie=True) for d in REVIEW_CORE for s in SEEDS]
    if name == "tiedmatched":
        return [BRun(d, s, tie=True, layers=4 * datasets.SPECS[d].steps) for d in TIED_MATCHED for s in SEEDS]
    if name == "product":
        return [BRun(d, s, entangle=False) for d in REVIEW_CORE for s in SEEDS]
    if name == "frozen":
        return [BRun(d, s, frozen=True) for d in REVIEW_CORE for s in SEEDS]
    if name == "perm":                      # three random feature orders, seed 42 (the file order is perm 0 = the main runs)
        return [BRun(d, 42, perm=p) for d in REVIEW_TABULAR for p in (1, 2, 3)]
    if name == "mnistref":                  # the MNIST grid cells the controls are compared with, rerun here with test predictions
        return [BRun(d, s) for d in ("mnist8_3v5", "mnist8_10class") for s in SEEDS]
    raise ValueError(name)


def available_gb():
    with open("/proc/meminfo") as f:
        return next(int(line.split()[1]) for line in f if line.startswith("MemAvailable")) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="main", choices=("pilot", "main", "head", "measured", "readout", "control", "multibasis", "review", "pred",
                                                      "tied", "tiedmatched", "product", "frozen", "perm", "mnistref"))
    ap.add_argument("--datasets", nargs="*", help="only these datasets")
    ap.add_argument("--batch", default="A1", choices=tuple(datasets.BATCHES), help="dataset batch: A1 (first six) or A2 (the next six)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ram-gb", type=float, default=12)
    ap.add_argument("--min-free-gb", type=float, default=2)
    ap.add_argument("--nice", type=int, default=0)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--time", action="store_true", help="time 3 training steps per dataset instead of training")
    ap.add_argument("--memory", default="coherent", choices=("coherent", "measured"), help="with --time: which memory mode to time")
    a = ap.parse_args()
    if a.nice:
        os.nice(a.nice)
    if a.time:
        if a.set in ("pilot", "main"):
            for d in a.datasets or datasets.BATCHES[a.batch]:
                print(json.dumps(time_steps(BRun(d, 42, memory=a.memory))), flush=True)
        else:                               # one timing per distinct (dataset, variant) of the set, seed 42
            seen = set()
            for r in run_set(a.set, a.batch):
                key = (r.dataset, r.tie, r.layers, r.entangle, r.frozen)
                if r.seed == 42 and key not in seen and (not a.datasets or r.dataset in a.datasets):
                    seen.add(key)
                    print(json.dumps(dict(name=r.name, **time_steps(r))), flush=True)
        return
    runs = [r for r in run_set(a.set, a.batch) if not a.datasets or r.dataset in a.datasets]
    runs = sorted(runs, key=lambda r: (r.seed, -r.cost))
    pending = [r for r in runs if not (r.folder / f"{r.name}.json").exists()]
    if a.list:
        for r in runs:
            print(r.name, f"T={r.n_steps}", f"cost {r.cost / 1e6:.0f}", "" if r in pending else "done")
        print(f"{len(runs)} runs, {len(pending)} not finished")
        return
    running, used = {}, 0.0
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        while pending or running:
            for r in list(pending):
                if len(running) >= a.workers:
                    break
                if not running or (used + r.memory_gb <= a.ram_gb and available_gb() >= a.min_free_gb):
                    running[pool.submit(train, r)] = r
                    used += r.memory_gb
                    pending.remove(r)
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for f in done:
                r = running.pop(f)
                used -= r.memory_gb
                if f.exception():
                    print(f"FAILED {r.name}: {f.exception()!r}", flush=True)


if __name__ == "__main__":
    main()
