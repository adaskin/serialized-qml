"""Patch-serialized quantum classifier: one PennyLane circuit class, native parameterized gates only.

Register: n_data system qubits (one pixel each) + n_mem ancilla qubits. The image is fed patch by patch; patch t has
its own trainable parameters (nothing is shared between patches except the ancilla block W_c)::

    for t in 1..T:                                   # T = number of patches
        encoding   per pixel x on its system qubit:  RY(pi(s1 x + o1))  RZ(pi(s2 sqrt(1-x^2) + o2))  RX(pi(s3 x^3 + o3))
        W_t        `layers` times:  Rot on every qubit, then a CRot chain (0->1, 1->2, ..., n-2->n-1)
        W_c        Rot on every ancilla qubit, CRot chain on the ancilla, then CRot from every ancilla qubit
                   to every system qubit (shared by all patches)
    readout:  <Z_q> and/or <Z_q Z_q+1> (ring) on all qubits (or on the system qubits only)  ->  linear layer  ->  class logits

entangle=False (review control, product-gate blocks): every CRot(c -> t) is replaced by a Rot on its target t with the same three
angles, so the parameter count is unchanged but the state stays a product state (the ancilla never couples to the system).

Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi) has 3 angles; CRot is the controlled Rot (3 angles), so every
two-qubit gate is parameterized and there is no CNOT or CZ. Trainable parameters per patch: layers * (3n + 3(n-1)).

What is carried from one patch to the next (`memory`):
    "coherent"  all qubits keep their quantum state through the whole image (needs coherence over all patches)
    "measured"  hardware-style: every patch is a fresh short circuit. The system qubits start in |0>. The ancilla qubits
                start from the state measured at the end of the previous patch (|0> for the first patch of every image)
                and are measured in Z at the end of each patch (all qubits are measured after the last patch). Only
                the n_mem measured bits connect patches. The outcome distribution is propagated exactly: for each of the
                2^n_mem possible previous outcomes the patch circuit is simulated and the outcome probabilities are chained.
"""

import math

import pennylane as qml
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

ENCODINGS = {"xyz": [(qml.RY, lambda x: x), (qml.RZ, lambda x: torch.sqrt(torch.clamp(1 - x * x, min=0.0))),
                     (qml.RX, lambda x: x ** 3)]}


class SerializedQC(nn.Module):
    """Input (batch, n_steps, n_data) pixels in [0, 1]; output (batch, n_out) logits.

    device="default.qubit" (torch backprop) vectorises over the batch; "lightning.qubit" (adjoint, coherent memory
    only) runs one circuit per sample, which is much slower for batched training (see reports/circuit.md)."""

    def __init__(self, n_steps=16, n_data=4, n_mem=3, n_out=10, layers=4, encoding="xyz", tie_weights=False,
                 shared_ancilla=True, readout="z+zz", measure="all", memory="coherent", init_scale=0.08,
                 enc_scale=1.0, device="default.qubit", diff_method="backprop", segment=8, entangle=True):
        """readout: "+"-joined terms: "z", "zz" or "z+zz" (default of every run so far), also "x", "y", "xx", "yy" (e.g. "z+zz+x+xx+y+yy": all are expectation values of the same final state, concatenated into the linear layer; on hardware X and Y need their own basis-rotated settings);  measure: "all" qubits or the "system" qubits only;  n_mem=0 removes the ancilla.
        segment: coherent memory with more than `segment` patches is run in chunks of `segment` patches whose forward pass is
        recomputed in the backward pass (gradient checkpointing; same numbers, about 1.3x the time, RAM proportional to `segment`
        instead of to the number of patches)."""
        super().__init__()
        assert memory in ("coherent", "measured") and (memory == "coherent" or n_mem > 0)
        shared_ancilla = shared_ancilla and n_mem > 0
        self.T, self.nd, self.nm, self.L, self.n = n_steps, n_data, n_mem, layers, n_data + n_mem
        self.w = [f"s{i}" for i in range(n_data)] + [f"a{j}" for j in range(n_mem)]     # system s*, ancilla a*
        self.tie, self.shared_ancilla, self.encoding, self.memory, self.segment = tie_weights, shared_ancilla, ENCODINGS[encoding], memory, segment
        self.entangle = entangle
        m = self.n if measure == "all" else n_data                                   # number of measured qubits
        self.kinds = readout.split("+")                                             # "z", "x", "y": one per qubit; "zz", "xx", "yy": ring pairs
        self.singles = [k for k in self.kinds if len(k) == 1]
        self.pair_kinds = [k[0] for k in self.kinds if len(k) == 2]
        assert set(self.singles) <= set("xyz") and all(k[0] == k[1] and k[0] in "xyz" for k in self.kinds if len(k) == 2)
        self.measured = self.w[:m]
        self.zz_pairs = [(i, (i + 1) % m) for i in range(m if m > 2 else m - 1)] if self.pair_kinds else []   # ring of neighbouring qubits
        n_features = m * len(self.singles) + len(self.zz_pairs) * len(self.pair_kinds)
        if device != "default.qubit":
            torch.set_default_dtype(torch.float64)   # other devices return results in the torch default dtype
        n, g = self.n, lambda *s: nn.Parameter(torch.randn(*s, dtype=torch.float64) * init_scale)
        R = 1 if tie_weights else n_steps
        self.rot, self.chain = g(R, layers, n, 3), g(R, layers, n - 1, 3)                      # W_t
        if shared_ancilla:                                                                       # W_c
            self.anc_rot, self.anc_chain, self.coupling = g(n_mem, 3), g(max(n_mem - 1, 0), 3), g(n_mem, n_data, 3)
        ng = len(self.encoding)
        self.enc_scale = nn.Parameter(enc_scale * (torch.ones(ng, n_data, dtype=torch.float64) + 0.03 * torch.randn(ng, n_data, dtype=torch.float64)))
        self.enc_offset = nn.Parameter(torch.zeros(ng, n_data, dtype=torch.float64))
        self.linear = nn.Linear(n_features, n_out, dtype=torch.float64)
        nn.init.zeros_(self.linear.bias)
        dev = qml.device(device, wires=self.w)
        self.qnode = qml.QNode(self.circuit, dev, interface="torch", diff_method=diff_method)
        self.patch_qnode = qml.QNode(self.patch_circuit, dev, interface="torch", diff_method=diff_method)
        self.segment_qnode = qml.QNode(self.segment_circuit, dev, interface="torch", diff_method=diff_method)

    # ------------------------------------------------------------------ gates
    def crot(self, angles, control, target):
        """CRot(control -> target), or with entangle=False a Rot on the target with the same angles (product-gate control)."""
        if self.entangle:
            qml.CRot(*angles, wires=[control, target])
        else:
            qml.Rot(*angles, wires=target)

    def patch(self, t, x, rot, chain, enc_scale, enc_offset, anc_rot, anc_chain, coupling):
        """Gates of patch t; x has shape (batch, n_data)."""
        nd, nm, n, w = self.nd, self.nm, self.n, self.w
        for q in range(nd):                                          # encode the patch's pixels
            for g, (gate, f) in enumerate(self.encoding):
                gate(math.pi * (enc_scale[g, q] * f(x[:, q]) + enc_offset[g, q]), wires=w[q])
        i = 0 if self.tie else t
        for l in range(self.L):                                      # W_t
            for q in range(n):
                qml.Rot(*rot[i, l, q], wires=w[q])
            for q in range(n - 1):
                self.crot(chain[i, l, q], w[q], w[q + 1])
        if self.shared_ancilla:                                      # W_c
            for j in range(nm):
                qml.Rot(*anc_rot[j], wires=w[nd + j])
            for j in range(nm - 1):
                self.crot(anc_chain[j], w[nd + j], w[nd + j + 1])
            for j in range(nm):
                for q in range(nd):
                    self.crot(coupling[j, q], w[nd + j], w[q])

    def observables(self):
        P = {"x": qml.PauliX, "y": qml.PauliY, "z": qml.PauliZ}
        obs = [qml.expval(P[k](q)) for k in self.singles for q in self.measured]
        return obs + [qml.expval(P[k](self.measured[a]) @ P[k](self.measured[b])) for k in self.pair_kinds for a, b in self.zz_pairs]

    def circuit(self, x, *weights):                                  # memory = "coherent": the whole image in one circuit
        for t in range(self.T):
            self.patch(t, x[:, t], *weights)
        return self.observables()

    def segment_circuit(self, t0, t1, x, start, last, *weights):     # memory = "coherent": patches t0..t1-1 from a given state
        qml.StatePrep(start, wires=self.w)
        for t in range(t0, t1):
            self.patch(t, x[:, t], *weights)
        return self.observables() if last else qml.state()

    def patch_circuit(self, t, x, start, last, *weights):            # memory = "measured": one patch from given start states
        qml.StatePrep(start, wires=self.w)
        self.patch(t, x, *weights)
        return self.observables() if last else qml.probs(wires=self.w[self.nd:])

    # ------------------------------------------------------------------ model
    def weights(self):
        wc = (self.anc_rot, self.anc_chain, self.coupling) if self.shared_ancilla else (None, None, None)
        return (self.rot, self.chain, self.enc_scale, self.enc_offset, *wc)

    def features(self, x):
        if self.memory == "coherent" and self.T <= self.segment:
            return torch.stack(list(self.qnode(x, *self.weights())), dim=-1)
        if self.memory == "coherent":                                # long image: checkpointed chunks, the state is passed on
            state = torch.zeros(len(x), 2 ** self.n, dtype=torch.complex128)
            state[:, 0] = 1
            for t0 in range(0, self.T, self.segment):
                t1 = min(t0 + self.segment, self.T)
                run = lambda s, x, *ws, t0=t0, t1=t1: self.segment_qnode(t0, t1, x, s, t1 == self.T, *ws)
                state = checkpoint(run, state, x, *self.weights(), use_reentrant=False)
            return torch.stack(list(state), dim=-1)
        b, k = len(x), 2 ** self.nm                                  # k possible measured ancilla states
        start = torch.zeros(b * k, 2 ** self.n, dtype=torch.complex128)   # system |0>, ancilla |m>: basis index m
        start[torch.arange(b * k), torch.arange(k).repeat(b)] = 1
        p = torch.zeros(b, k, dtype=torch.float64)                   # distribution of the ancilla bits before patch t
        p[:, 0] = 1                                                  # every image starts from |0>
        for t in range(self.T):
            xt = x[:, t].repeat_interleave(k, 0)                     # row b*k+m = (image b, previous outcome m)
            if t < self.T - 1:
                chain = lambda p, xt, *ws, t=t: torch.einsum("bm,bmk->bk", p, self.patch_qnode(t, xt, start, False, *ws).reshape(b, k, k))
                p = checkpoint(chain, p, xt, *self.weights(), use_reentrant=False)   # chain the outcome probabilities
                                                                                     # (recomputed in backward: saves memory)
            else:
                f = torch.stack(list(self.patch_qnode(t, xt, start, True, *self.weights())), dim=-1).reshape(b, k, -1)
                return torch.einsum("bm,bmf->bf", p, f)

    def forward(self, x):
        return self.linear(self.features(x.to(torch.float64)))

    def draw(self, **kwargs):
        """matplotlib figure of the whole circuit (use a small n_steps / layers)."""
        return qml.draw_mpl(self.qnode, decimals=None, **kwargs)(torch.zeros(1, self.T, self.nd), *self.weights())
