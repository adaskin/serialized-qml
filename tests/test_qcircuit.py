import sys
import unittest

import numpy as np
import pennylane as qml
import torch

sys.path.insert(0, "src")
from qcircuit import SerializedQC

SMALL = dict(n_steps=3, n_data=2, n_mem=2, n_out=3, layers=2)


def data(n=5, T=3, nd=2):
    torch.manual_seed(1)
    return torch.rand(n, T, nd, dtype=torch.float64), torch.randint(0, 3, (n,))


def tape(m, x):
    return qml.tape.make_qscript(m.circuit)(x, *m.weights())


class TestSerializedQC(unittest.TestCase):
    def test_parameter_count(self):
        m = SerializedQC(n_steps=16, n_out=10, layers=4)          # 4 system + 3 ancilla qubits
        per_patch = 4 * (3 * 7 + 3 * 6)
        wc = 3 * 3 + 3 * 2 + 3 * 3 * 4
        self.assertEqual(sum(p.numel() for p in m.parameters()), 16 * per_patch + wc + 2 * 3 * 4 + 14 * 10 + 10)

    def test_only_parameterized_native_gates(self):
        x, _ = data()
        ops = tape(SerializedQC(**SMALL), x).operations
        self.assertEqual({type(o).__name__ for o in ops}, {"RY", "RZ", "RX", "Rot", "CRot"})
        self.assertTrue(all(o.num_params > 0 for o in ops))        # no fixed CNOT / CZ

    def test_every_ancilla_is_coupled_to_every_system_qubit(self):
        x, _ = data()
        m = SerializedQC(**SMALL)
        pairs = {tuple(o.wires) for o in tape(m, x).operations if type(o).__name__ == "CRot"}
        self.assertTrue({(a, s) for a in ("a0", "a1") for s in ("s0", "s1")} <= pairs)

    def test_patches_have_independent_parameters(self):
        x, y = data()
        m = SerializedQC(**SMALL)
        torch.nn.functional.cross_entropy(m(x), y).backward()
        self.assertEqual(m.rot.shape[0], SMALL["n_steps"])
        for t in range(SMALL["n_steps"]):
            self.assertGreater(m.rot.grad[t].abs().sum().item(), 0)
        self.assertEqual(SerializedQC(**SMALL, tie_weights=True).rot.shape[0], 1)

    def test_memory_carries_early_patches_to_the_output(self):
        x, _ = data()
        m = SerializedQC(**SMALL)
        x2 = x.clone()
        x2[:, 0] += 0.3
        self.assertGreater((m.features(x) - m.features(x2)).abs().max().item(), 1e-4)

    def test_features_are_expectation_values(self):
        x, _ = data()
        f = SerializedQC(**SMALL).features(x)
        self.assertEqual(f.shape, (5, 8))
        self.assertTrue(((f >= -1 - 1e-12) & (f <= 1 + 1e-12)).all())

    def test_readout_and_ancilla_options(self):
        x, y = data()
        for kw, n_features in [(dict(readout="z"), 4), (dict(readout="zz"), 4), (dict(measure="system"), 3),
                               (dict(readout="z", measure="system"), 2), (dict(n_mem=0), 3), (dict(n_mem=0, measure="system"), 3)]:
            m = SerializedQC(**{**SMALL, **kw})
            self.assertEqual(m.features(x).shape, (5, n_features), kw)
            torch.nn.functional.cross_entropy(m(x), y).backward()
        self.assertFalse(hasattr(SerializedQC(**{**SMALL, "n_mem": 0}), "coupling"))

    def test_measured_memory_matches_density_matrix_simulation(self):
        """Ancilla measured after every patch, system reloaded from |0>: exact chain of outcome probabilities
        against a density-matrix simulation with a Z measurement (PhaseDamping(1)) and a reset (AmplitudeDamping(1))."""
        x, y = data()
        for layers in (1, 2):
            m = SerializedQC(**{**SMALL, "layers": layers}, memory="measured")

            @qml.qnode(qml.device("default.mixed", wires=m.w), interface="torch", diff_method="backprop")
            def reference(x, *weights):
                for t in range(m.T):
                    m.patch(t, x[:, t], *weights)
                    if t < m.T - 1:
                        for a in m.w[m.nd:]:
                            qml.PhaseDamping(1.0, wires=a)
                        for q in m.w[:m.nd]:
                            qml.AmplitudeDamping(1.0, wires=q)
                return m.observables()

            ref = torch.stack(list(reference(x, *m.weights())), dim=-1)
            self.assertTrue(torch.allclose(m.features(x), ref, atol=1e-5), layers)
            coherent = SerializedQC(**{**SMALL, "layers": layers})
            coherent.load_state_dict(m.state_dict())
            self.assertGreater((m.features(x) - coherent.features(x)).abs().max().item(), 1e-3)   # the measurement matters

    def test_measured_memory_with_one_patch_is_the_coherent_circuit(self):
        x, y = data(T=1)
        ref = SerializedQC(**{**SMALL, "n_steps": 1})
        m = SerializedQC(**{**SMALL, "n_steps": 1}, memory="measured")
        m.load_state_dict(ref.state_dict())
        self.assertTrue(torch.allclose(m.features(x), ref.features(x), atol=1e-12))

    def test_measured_memory_trains_every_patch(self):
        x, y = data()
        m = SerializedQC(**SMALL, memory="measured")
        torch.nn.functional.cross_entropy(m(x), y).backward()
        for t in range(SMALL["n_steps"]):
            self.assertGreater(m.rot.grad[t].abs().sum().item(), 0)

    def test_gradient_matches_finite_differences(self):
        x, y = data()
        m = SerializedQC(**SMALL)
        loss = lambda: torch.nn.functional.cross_entropy(m(x), y)
        g = torch.autograd.grad(loss(), m.chain)[0]
        eps = 1e-6
        with torch.no_grad():
            for idx in [(0, 0, 0, 0), (1, 1, 2, 1), (2, 0, 1, 2)]:
                old = m.chain[idx].item()
                m.chain[idx] = old + eps
                up = loss().item()
                m.chain[idx] = old - eps
                dn = loss().item()
                m.chain[idx] = old
                self.assertAlmostEqual((up - dn) / (2 * eps), g[idx].item(), places=6)

    def test_lightning_matches_default_qubit(self):
        x, y = data()
        a = SerializedQC(**SMALL)
        b = SerializedQC(**SMALL, device="lightning.qubit", diff_method="adjoint")
        b.load_state_dict(a.state_dict())
        la, lb = a(x), b(x)
        self.assertTrue(torch.allclose(la, lb, atol=1e-12))
        ga = torch.autograd.grad(torch.nn.functional.cross_entropy(la, y), list(a.parameters()))
        gb = torch.autograd.grad(torch.nn.functional.cross_entropy(lb, y), list(b.parameters()))
        for p, q in zip(ga, gb):
            self.assertTrue(torch.allclose(p, q, atol=1e-10))


if __name__ == "__main__":
    unittest.main()
