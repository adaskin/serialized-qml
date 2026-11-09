"""Train SerializedQC on patches of MNIST: the fixed experimental grid. One run = one JSON file in data/results/quantum/.

    python src/train.py --workers 10                 run the whole grid (finished runs are skipped, unfinished ones resume)
    python src/train.py --list                       print the runs
    python src/train.py --set diagnostics --nice 10  re-runs of the stalled 10-class 8x8 measured L2 seed-44 run with other initial angles

Every run uses the same settings: 100 epochs, no early stopping or model selection (the reported numbers are those of the last
epoch), Adam lr 0.01 with cosine decay to 5%, batch 256, 1000 training images per class, 200 validation images per class
(monitoring only), the full MNIST test set of the chosen digits. Varied: image size, task, memory, layers, seed.
"""
import os

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):     # one core per run: set before numpy / torch load
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
import torch.nn.functional as F

import download_data
from data import TASKS, make_split
from qcircuit import SerializedQC

RESULTS = Path("data/results/quantum")
CKPT = Path("data/ckpt")          # per-epoch state of unfinished runs (git-ignored); a restarted run resumes from here
ENC_SCALE = {8: 1.0, 14: 0.33, 28: 0.08}   # about 16 / (number of patches); 14x14 was chosen in a pilot; 28x28 (screens only) is not trained


@dataclass(frozen=True)
class Run:
    task: str                         # key of data.TASKS
    image_size: int                   # 8, 14 or 28 (MNIST is area-resized); patches are 2x2, so 16, 49 or 196 patches
    layers: int                       # L: the W_t block is repeated L times
    memory: str                       # coherent | measured (ancilla measured after every patch, see qcircuit.py)
    seed: int
    dataset: str = "mnist"
    patch: tuple = (2, 2)             # pixels per patch = system qubits
    n_mem: int = 3                    # ancilla qubits
    readout: str = "z+zz"
    encoding: str = "xyz"
    lr: float = 0.01                  # Adam, cosine decay to 5% of lr over the epochs
    batch: int = 256
    epochs: int = 100
    clip: float = 5.0                 # gradient-norm clip
    init_scale: float = 0.08
    init_seed: int = None             # seed of the initial angles (None: seed). The images and the batch order always follow `seed`
    train_pc: int = 1000              # images per class
    val_pc: int = 200
    test_pc: int = None               # None: all MNIST test images of the class
    eval_every: int = 10

    @property
    def classes(self):
        return TASKS[self.task]

    @property
    def n_steps(self):
        return (self.image_size // self.patch[0]) * (self.image_size // self.patch[1])

    @property
    def enc_scale(self):
        return ENC_SCALE[self.image_size]

    @property
    def name(self):
        extra = ("" if self.init_scale == 0.08 else f"_init{self.init_scale:g}") + ("" if self.init_seed is None else f"_initseed{self.init_seed}")
        extra += "" if self.readout == "z+zz" else f"_ro{self.readout.replace('+', '')}"
        return f"{self.dataset}_{self.task}_s{self.image_size}_{self.memory}_L{self.layers}_seed{self.seed}{extra}"

    @property
    def folder(self):
        """Runs with their own initial-angle seed are diagnostics, runs with another readout are the multi-basis experiment; both stay
        out of the grid's tables."""
        if self.readout != "z+zz":
            return RESULTS.parent / "multibasis"
        return RESULTS if self.init_seed is None else RESULTS.parent / "diagnostics"

    @property
    def cost(self):
        """Relative running time: training steps x gates per forward pass (measured memory simulates 8 ancilla start states)."""
        steps = self.epochs * math.ceil(len(self.classes) * self.train_pc / self.batch)
        return steps * self.n_steps * (13 * self.layers + 29) * (4.5 if self.memory == "measured" else 1)

    @property
    def memory_gb(self):
        """Estimated peak RAM of a run, from one training step measured on the cloud machine (python 3.13, torch 2.14 CPU, pennylane
        0.45) with 30% margin: process base 1 GB; coherent: about 0.25-0.4 MB per patch and image of the 8 patches kept for the
        backward pass (SerializedQC.segment); measured memory: about 4 MB per image. Another torch build used 5-10x more."""
        if self.memory == "measured":
            return 1.5 + 1.3 * 0.0042 * self.batch * (1 + 0.125 * (self.layers - 2))
        return 1.0 + 1.3 * self.batch * min(self.n_steps, 8) * (250 + 70 * (self.layers - 2)) * 1e-6


# ----------------------------------------------------------------------------- the grid
SIZES, SEEDS, LAYERS = (8, 14), (42, 43, 44), (2, 4)     # not 28x28: 196 patches, 74% of them empty, did not train (reports/caution_28x28.md)


def grid():
    """8x8 and 14x14: every task x memory x layers x seed (96 runs)."""
    return [Run(task, size, layers, memory, seed) for size in SIZES for task in TASKS for memory in ("coherent", "measured")
            for layers in LAYERS for seed in SEEDS]


# ----------------------------------------------------------------------------- diagnostics (not part of the 96-run grid)
# Seed 44 of 10-class 8x8 measured L2 stalled at 47% training accuracy (seeds 42, 43: 70%, 72%). The same images with two other initial
# angles, and the seed-44 angles on the images of seed 42, tell a bad initialisation from a bad split.
def diagnostics():
    return [Run("10class", 8, 2, "measured", 44, init_seed=142), Run("10class", 8, 2, "measured", 44, init_seed=242),
            Run("10class", 8, 2, "measured", 42, init_seed=44)]


# ----------------------------------------------------------------------------- multi-basis readout (not part of the 96-run grid)
# Z, ZZ plus X, XX, Y, YY of the same final state (42 features instead of 14), the rule for tasks with more than 4 classes; only the
# cheap 8x8 cell: 10 classes, coherent, L = 4 (results in data/results/multibasis).
MULTIBASIS = "z+zz+x+xx+y+yy"


def multibasis():
    return [Run("10class", 8, 4, "coherent", seed, readout=MULTIBASIS) for seed in SEEDS]


# ----------------------------------------------------------------------------- training
def build_model(run):
    torch.manual_seed(run.seed if run.init_seed is None else run.init_seed)
    return SerializedQC(n_steps=run.n_steps, n_data=run.patch[0] * run.patch[1], n_mem=run.n_mem, n_out=len(run.classes),
                        layers=run.layers, encoding=run.encoding, readout=run.readout, memory=run.memory,
                        init_scale=run.init_scale, enc_scale=run.enc_scale)


@torch.no_grad()
def evaluate(model, x, y, bs=400):
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
    xtr, ytr, xva, yva, xte, yte = make_split(run)
    model = build_model(run)
    opt = torch.optim.Adam(model.parameters(), lr=run.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, run.epochs, eta_min=0.05 * run.lr)
    gen = torch.Generator().manual_seed(run.seed + 1)
    hist, t0, patch_grad, first = [], time.time(), {}, 1
    ckpt = CKPT / f"{run.name}.pt"
    if ckpt.exists():                                            # resume an interrupted run
        c = torch.load(ckpt, weights_only=False)
        model.load_state_dict(c["model"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"]); gen.set_state(c["gen"])
        hist, patch_grad, first, t0 = c["hist"], c["patch_grad"], len(c["hist"]) + 1, time.time() - c["hist"][-1]["seconds"]
    for epoch in range(first, run.epochs + 1):
        perm = torch.randperm(len(xtr), generator=gen)
        tl = tc = gn = 0.0
        for i in range(0, len(xtr), run.batch):
            idx = perm[i: i + run.batch]
            out = model(xtr[idx])
            loss = F.cross_entropy(out, ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn += torch.nn.utils.clip_grad_norm_(model.parameters(), run.clip).item() * len(idx)   # norm before clipping
            if epoch in (1, run.epochs) and i == 0:     # gradient norm of each patch's W_t (rot + chain parameters)
                patch_grad[epoch] = torch.cat([model.rot.grad.flatten(1), model.chain.grad.flatten(1)], 1).norm(dim=1).tolist()
            opt.step()
            tl += loss.item() * len(idx)
            tc += (out.argmax(1) == ytr[idx]).sum().item()
        row = dict(epoch=epoch, train_loss=tl / len(xtr), train_acc=tc / len(xtr), grad_norm=gn / len(xtr), seconds=time.time() - t0,
                   peak_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6)
        if epoch % run.eval_every == 0 or epoch == run.epochs:
            row["val_loss"], row["val_acc"] = evaluate(model, xva, yva)
            row["test_loss"], row["test_acc"] = evaluate(model, xte, yte)
        hist.append(row)
        sched.step()
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), gen=gen.get_state(),
                        hist=hist, patch_grad=patch_grad), f"{ckpt}.tmp")
        os.replace(f"{ckpt}.tmp", ckpt)
        print(f"{run.name} epoch {epoch}: train {row['train_acc']:.3f}" + (f" val {row['val_acc']:.3f}" if "val_acc" in row else "")
              + f" ({row['seconds']:.0f}s, peak RAM {row['peak_gb']:.1f} GB)", flush=True)
    res = dict(run=asdict(run), name=run.name, params=sum(p.numel() for p in model.parameters()), final=hist[-1], history=hist,
               patch_grad_norm=patch_grad, seconds=time.time() - t0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res))
    ckpt.unlink(missing_ok=True)
    f = hist[-1]
    print(f"DONE {run.name}: params={res['params']} train={f['train_acc']:.3f} val={f['val_acc']:.3f} test={f['test_acc']:.3f} "
          f"({res['seconds'] / 3600:.1f} h)", flush=True)
    return str(path)


def available_gb():
    with open("/proc/meminfo") as f:
        return next(int(line.split()[1]) for line in f if line.startswith("MemAvailable")) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10, help="runs in parallel (one core each)")
    ap.add_argument("--ram-gb", type=float, default=60, help="runs are only started while their estimated RAM fits in this budget")
    ap.add_argument("--min-free-gb", type=float, default=12, help="and while the machine has at least this much available RAM")
    ap.add_argument("--set", default="grid", choices=("grid", "diagnostics", "multibasis"), help="grid: the 96 runs; diagnostics, multibasis: see above")
    ap.add_argument("--nice", type=int, default=0, help="lower the priority of this job (and its workers) so that other jobs keep their cores")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.nice:
        os.nice(a.nice)
    # one complete grid per seed (42, then 43, then 44), slowest runs first within a seed: whenever the job is stopped, the finished
    # seeds form complete grids
    runs = sorted({"grid": grid, "diagnostics": diagnostics, "multibasis": multibasis}[a.set](), key=lambda r: (r.seed, -r.cost))
    pending = [r for r in runs if not (r.folder / f"{r.name}.json").exists()]
    if a.list:
        for r in runs:
            print(r.name, f"cost {r.cost / 1e6:.0f}", f"{r.memory_gb:.1f} GB", "" if r in pending else "done")
        print(f"{len(runs)} runs, {len(pending)} not finished")
        return
    download_data.ensure("mnist")
    running, used = {}, 0.0
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context("spawn")) as pool:
        while pending or running:
            for r in list(pending):                      # next run in priority order that still fits the RAM budget
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
