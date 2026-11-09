"""Expressivity probe: rank of the Jacobian of the circuit features (Z and ZZ expectations) with respect to the
quantum parameters at a random point. A rank close to min(#samples*#features, #parameters) means the parameters
are independent directions (no redundant parameters).   Usage: python scripts/expressivity.py <layers> [n_steps]"""
import sys

import numpy as np
import torch

sys.path.insert(0, "src")
from qcircuit import SerializedQC

layers, T = int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 4
torch.manual_seed(0)
torch.set_num_threads(1)
m = SerializedQC(n_steps=T, n_out=10, layers=layers, init_scale=0.5)
params = [p for name, p in m.named_parameters() if not name.startswith("linear")]
n_par = sum(p.numel() for p in params)
feats = m.features(torch.rand(64, T, 4, dtype=torch.float64)).reshape(-1)       # 64 samples x 14 features
J = np.stack([torch.cat([g.reshape(-1) for g in torch.autograd.grad(f, params, retain_graph=True)]).numpy() for f in feats])
s = np.linalg.svd(J, compute_uv=False)
print(f"patches={T} layers={layers}: params={n_par}  jacobian rows={J.shape[0]}  rank={int((s > 1e-8 * s[0]).sum())}", flush=True)
