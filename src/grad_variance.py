"""Barren-plateau test asked for by the (mock) review: the variance of the gradient over random initializations as the register grows.

The serialized circuit is fixed at 4 system qubits; the register grows through the memory: n_mem = 1..7 ancillas, n = 5..11 qubits.
For every n, number of patches T and initialization, the circuit (coherent memory, L = 4, Z + ZZ readout, no training) is drawn
`--inits` times; each draw sees the same batch of random inputs (uniform in [0, 1]). Two costs, both without the linear head:
  local   C = mean over the batch of <Z_s0>                           (one qubit)
  global  C = mean over the batch of <Z_s0 Z_s1 ... Z_a(n_mem-1)>   (all qubits; the observable of the classic barren-plateau result)
and the gradient of C with respect to two angles: the first RY-angle of the first Rot on s0 in the first patch's block W_1 ("first")
and in the last patch's block W_T ("last"). Inits:
  identity  all angles N(0, 0.08^2) as in every training run (encoding scales 1 + 3% noise, offsets 0)
  uniform   block angles uniform in [-pi, pi) (the random-circuit setting of McClean et al.; encoding as above)
Var over the inits is printed and written to data/results/review/grad_variance.json. An exponential decay of Var with n is the
barren-plateau signature; a flat curve at a fixed n says nothing about larger n.

    python src/grad_variance.py --inits 200            (cloud CPU: 0.3-0.8 s per draw at T = 4, 3.7 s at n = 11, T = 16; ~1.5 core-h)
"""
import os

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import json
import math
import time
from pathlib import Path

import pennylane as qml
import torch

from qcircuit import SerializedQC

OUT = Path("data/results/review/grad_variance.json")


def costs(model, x):
    """(local, global) cost of the circuit's final state, mean over the batch."""
    w = model.w
    dev = qml.device("default.qubit", wires=w)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(x, *weights):
        for t in range(model.T):
            model.patch(t, x[:, t], *weights)
        prod = qml.PauliZ(w[0])
        for q in w[1:]:
            prod = prod @ qml.PauliZ(q)
        return qml.expval(qml.PauliZ(w[0])), qml.expval(prod)

    loc, glob = circuit(x, *model.weights())
    return loc.mean(), glob.mean()


def grads(n_mem, T, init, seed, x, layers=4):
    torch.manual_seed(seed)
    model = SerializedQC(n_steps=T, n_data=4, n_mem=n_mem, n_out=2, layers=layers, init_scale=0.08)
    if init == "uniform":
        with torch.no_grad():
            for p in (model.rot, model.chain, model.anc_rot, model.anc_chain, model.coupling):
                p.uniform_(-math.pi, math.pi)
    out = {}
    for name, c in zip(("local", "global"), costs(model, x)):
        g = torch.autograd.grad(c, model.rot, retain_graph=True)[0]      # rot: (T, layers, n, 3)
        out[name] = dict(first=g[0, 0, 0, 1].item(), last=g[-1, 0, 0, 1].item())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n-mem", type=int, nargs="*", default=list(range(1, 8)))
    ap.add_argument("--patches", type=int, nargs="*", default=[4, 16])
    ap.add_argument("--out", default=str(OUT), help="one file per process when several run side by side")
    a = ap.parse_args()
    out = Path(a.out)
    rows = json.loads(out.read_text()) if out.exists() else []
    done = {(r["n_mem"], r["T"], r["init"]) for r in rows}
    for T in a.patches:
        x = torch.rand(a.batch, T, 4, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
        for n_mem in a.n_mem:
            for init in ("identity", "uniform"):
                if (n_mem, T, init) in done:
                    continue
                t0 = time.time()
                gs = [grads(n_mem, T, init, 1000 + i, x) for i in range(a.inits)]
                row = dict(n_mem=n_mem, n=4 + n_mem, T=T, init=init, inits=a.inits, batch=a.batch, seconds=time.time() - t0)
                for cost in ("local", "global"):
                    for where in ("first", "last"):
                        v = torch.tensor([g[cost][where] for g in gs], dtype=torch.float64)
                        row[f"var_{cost}_{where}"] = v.var().item()
                        row[f"meanabs_{cost}_{where}"] = v.abs().mean().item()
                rows.append(row)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(rows, indent=1))
                print(f"n={row['n']:2d} T={T:2d} {init:8s} var local first {row['var_local_first']:.2e} last {row['var_local_last']:.2e}  "
                      f"global first {row['var_global_first']:.2e} last {row['var_global_last']:.2e}  ({row['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
