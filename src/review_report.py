"""Report of the review experiments (data/results/review): reports/review.md and reports/figures/fig_gradvar.png.

    python src/review_report.py

Numbers come from the result files through the helpers of scripts/paper_figures.py (the same ones that write the paper's tables).
"""
import importlib.util
import json
import statistics as st
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("paper_figures", ROOT / "scripts/paper_figures.py")
pf = importlib.util.module_from_spec(spec)
sys.modules["paper_figures"] = pf
spec.loader.exec_module(pf)

OUT = ROOT / "reports/review.md"
NAMES = {**dict(pf.BENCH), "mnist8_3v5": "MNIST 8x8 3 vs 5", "mnist8_10class": "MNIST 8x8 10 classes"}


def ms(rs, key="test_acc"):
    m, s, _ = pf.stat(rs, key)
    return f"{m:.1f} ± {s:.1f}"


def paired(r, d, k):
    ref = {x["run"]["seed"]: x["final"]["test_acc"] for x in r[(d, "ref")]}
    v = [100 * (x["final"]["test_acc"] - ref[x["run"]["seed"]]) for x in r[(d, k)] if x["run"]["seed"] in ref]
    return st.mean(v), min(v), max(v)


def controls(r):
    cols = (("tied", "tied blocks"), ("tiedmatched", "tied, same params"), ("product", "product gates"), ("frozen", "frozen random"))
    rows = ["| dataset | k | N | circuit | " + " | ".join(c for _, c in cols) + " |", "|---|---:|---:|---:|" + "---:|" * len(cols)]
    diffs = ["| dataset | " + " | ".join(c for _, c in cols) + " |", "|---|" + "---:|" * len(cols)]
    for d, _ in pf.REVIEW_SETS:
        rows.append(f"| {NAMES[d]} | {pf.REVIEW_K[d]} | {pf.REVIEW_T[d]} | {ms(r[(d, 'ref')])} | "
                    + " | ".join(ms(r[(d, k)]) if r.get((d, k)) else "–" for k, _ in cols) + " |")
        diffs.append(f"| {NAMES[d]} | " + " | ".join("{:+.1f} ({:+.1f} to {:+.1f})".format(*paired(r, d, k)) if r.get((d, k)) else "–"
                                                    for k, _ in cols) + " |")
    return rows, diffs


def feature_order(r):
    rows = ["| dataset | file order (seed 42) | order 1 | order 2 | order 3 |", "|---|---:|---:|---:|---:|"]
    for d, _ in pf.REVIEW_SETS:
        perm = sorted(r.get((d, "perm"), []), key=lambda x: x["run"]["perm"])
        if not perm:
            continue
        ref = next(x for x in r[(d, "ref")] if x["run"]["seed"] == 42)["final"]["test_acc"]
        rows.append(f"| {NAMES[d]} | {100 * ref:.1f} | " + " | ".join(f"{100 * (x['final']['test_acc'] - ref):+.1f}" for x in perm) + " |")
    return rows


def structured(r, b):
    q, best = pf.collect()
    rows = ["| dataset | circuit | measured | MLP | seq. RNN | MPS | 8-state HMM | params: circuit / RNN / MPS / HMM |", "|---|" + "---:|" * 7]
    for d in [d for d, _ in pf.BENCH] + ["mnist8_3v5", "mnist8_10class"]:
        mn = d.startswith("mnist8")
        rs = {"ref": r.get((d, "ref")) or b.get((d, "circuit")),
              "measured": q[(d.split("_")[1], 8, "measured", 4)] if mn else (r.get((d, "measured")) or b.get((d, "measured"))),
              "mlp": best[(d.split("_")[1], 8, "mlp", 4)] if mn else b.get((d, "mlp")),
              "seqrnn": r.get((d, "seqrnn")), "mps": r.get((d, "mps")), "hmm": r.get((d, "hmm"))}
        means = {c: pf.stat(v)[0] for c, v in rs.items() if v}
        top = max(means.values())
        cell = lambda c: "–" if c not in means else (f"**{means[c]:.1f}**" if pf.bold_if(means[c], top) else f"{means[c]:.1f}")
        p = [rs["ref"][0]["params"]] + [rs[m][0]["params"] for m in ("seqrnn", "mps", "hmm")]
        rows.append(f"| {NAMES[d]} | " + " | ".join(cell(c) for c in ("ref", "measured", "mlp", "seqrnn", "mps", "hmm"))
                    + " | " + " / ".join(map(str, p)) + " |")
    return rows


def mcnemar(r):
    rows = ["| dataset | circuit − MLP (points) | only circuit right | only MLP right | p (exact) |", "|---|---:|---:|---:|---:|"]
    for d, name in pf.BENCH:
        qq = r.get((d, "pred"))
        if not qq or d not in pf.BENCH_MLP:
            continue
        lr = pf.BENCH_MLP[d][0]["run"]["lr"]
        mm = [x for x in pf.load("review/classical") if x["run"]["dataset"] == d and x["run"]["model"] == "mlp" and x["run"]["lr"] == lr]
        a = [p == t for p, t in zip(qq[0]["test_pred"], qq[0]["test_y"])]
        c = [p == t for p, t in zip(mm[0]["test_pred"], mm[0]["test_y"])]
        n01, n10 = sum(u and not v for u, v in zip(a, c)), sum(v and not u for u, v in zip(a, c))
        n, k = n01 + n10, min(n01, n10)
        p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
        rows.append(f"| {name} | {100 * (sum(a) - sum(c)) / len(a):+.1f} | {n01} | {n10} | {p:.2g} |")
    return rows


def gradvar():
    rows_ = json.loads((ROOT / "data/results/review/grad_variance.json").read_text())
    out = []
    for T in (4, 16):
        out += [f"\n**N = {T} patches** (variance over {rows_[0]['inits']} initializations; first block W_1 / last block W_N)\n",
                "| n | identity, local | identity, global | uniform, local | uniform, global |", "|---:|---:|---:|---:|---:|"]
        for n in range(5, 12):
            g = {x["init"]: x for x in rows_ if x["T"] == T and x["n"] == n}
            if len(g) < 2:
                continue
            cell = lambda i, c: f"{g[i][f'var_{c}_first']:.1e} / {g[i][f'var_{c}_last']:.1e}"
            out.append(f"| {n} | {cell('identity', 'local')} | {cell('identity', 'global')} | {cell('uniform', 'local')} | {cell('uniform', 'global')} |")
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    q, _ = pf.collect()
    b = pf.bench_collect()
    pf.BENCH_MLP.update({d: b[(d, "mlp")] for d, _ in pf.BENCH})
    r = pf.review_collect(q, b)
    n_q, n_c = len(pf.load("review/quantum")), len(pf.load("review/classical"))
    pf.FIG = ROOT / "reports/figures"
    pf.fig_gradvar()
    import matplotlib.pyplot as plt
    plt.gcf().savefig(ROOT / "reports/figures/fig_gradvar.png", dpi=150, bbox_inches="tight")
    (ROOT / "reports/figures/fig_gradvar.pdf").unlink(missing_ok=True)
    ctrl, diffs = controls(r)
    text = f"""# Review experiments (Tier 1)

Experiments added after the mock npj Quantum Information review of 8 November 2026, on the splits and with the training recipe of the main runs
(100 epochs, Adam 0.01 with cosine decay, batch 256, last epoch, seeds 42-44). Result files: `data/results/review/` ({n_q} circuit runs,
{n_c} classical runs, `grad_variance.json`); see `data/results/README.md`. Generated by `python src/review_report.py`; the paper tables come
from the same files (`scripts/paper_figures.py`). Test accuracy in %, mean ± sample standard deviation over three seeds.

## Summary

- **Every part of the circuit that was removed costs accuracy, most on the tasks with many classes.**
  - Tying the blocks loses little on Waveform and Dry Bean, a few points on Landsat, Spambase, Wall-Following and MNIST 3 vs 5, and 9 to 12 points on Letter and MNIST 10 classes.
  - A tied block made deep enough to have the original parameter count recovers the loss on Waveform and Dry Bean and part of it on Letter, but not on Wall-Following.
- **Product gates lower the accuracy on all eight tasks**, by up to 23 points on Letter.
  - They remove every interaction between qubits, including the link through the memory, so this shows that interactions matter, not that entanglement as such is a resource.
- **The frozen random circuit with a trained head is far behind** (17 to 68 points), so the accuracy comes from training the circuit.
- **Feature order:** three random orders of the tabular features leave Waveform and Dry Bean unchanged and raise the accuracy on the other four sets, on Letter by 3 to 6 points. The file order used in the paper was not favourable to the circuit.
- **Structured classical models** read patch by patch with position-specific parameters, as the circuit does.
  - On the tabular sets the RNN and MPS match the MLP and are mostly ahead of the circuit.
  - On the 49-patch image sets the circuit is ahead of both (clearly on Kuzushiji-MNIST).
  - The 8-state HMM is the weakest model everywhere, and the measured circuit (the same eight states) beats it on 7 of 8 tabular sets and on both MNIST tasks.
- **Paired McNemar tests** (seed 42): the circuit's small deficits to the MLP on Waveform, Wall-Following, Covertype and Spambase are significant, while Credit Default, Landsat and Dry Bean show no significant difference.
- **Gradient variance:**
  - From uniform random angles the variance halves with every qubit (5 to 11 qubits), for local and global cost and for 4 and 16 patches: a barren plateau.
  - From the near-identity start of the training runs, the local-cost variance stays at about 1e-2, while the global-cost variance still falls with n.
  - The number of patches matters much less than the number of qubits.

## 1. Circuit controls

Coherent memory, L = 4. k: classes, N: patches.
- **Tied blocks:** one block W at every patch.
- **Tied, same params:** the tied block with N x L layers, run on the 4- to 6-patch sets.
- **Product gates:** every CRot replaced by a Rot on its target with the same angles.
- **Frozen random:** block angles uniform in [-pi, pi), encoding at its initial values, only the linear head trained.

{chr(10).join(ctrl)}

Control minus circuit, paired by seed (mean, and min to max over the three seeds), in points:

{chr(10).join(diffs)}

## 2. Feature order

Seed 42. Test accuracy with the file order, and the change with three random column orders (numpy seeds 1-3, the same for every training seed).

{chr(10).join(feature_order(r))}

## 3. Classical models with the structure of the circuit

Test accuracy; the best mean of each row in bold.
- **Seq. RNN:** h_t = tanh(A_t h_(t-1) + B_t x_t + c_t), with its own weights at every patch.
- **MPS:** one (2, D, D) tensor per feature with the map (cos(pi x/2), sin(pi x/2)) and a normalized boundary vector.
- **8-state HMM:** p_t = p_(t-1) P_t(x_t) with softmax rows; the readout sees [p_N, x_N].

RNN and MPS are sized as closely as H and D allow to the circuit; the HMM size is fixed by its eight states. The learning rate (0.001, 0.003, 0.01) is chosen by mean validation accuracy. For MNIST the MLP is the parameter-matched MLP of the grid. Measured memory was not run on the 49-patch image sets.

{chr(10).join(structured(r, b))}

## 4. Paired test against the MLP

Seed-42 reruns of the circuit and of the MLP (at its selected learning rate) with stored test predictions; the reruns reproduce the
original accuracies (circuit exactly, MLP to within 0.1 points). Exact two-sided McNemar (binomial) test on the discordant examples.

{chr(10).join(mcnemar(r))}

## 5. Gradient variance against the number of qubits

The untrained coherent circuit, L = 4, with 1 to 7 memory qubits (n = 5 to 11), is evaluated on the same 16 random inputs for every draw.
- **Local cost:** batch mean of <Z_s0>.
- **Global cost:** batch mean of <Z x ... x Z> over all n qubits.
- **Derivative:** with respect to the R_Y angle of the first Rot on s0 in the first layer of W_1 (first) and W_N (last).
- **Identity:** N(0, 0.08^2) as in training.
- **Uniform:** block angles uniform in [-pi, pi).

![gradient variance](figures/fig_gradvar.png)
{chr(10).join(gradvar())}

## Caveats

- The product-gate control also cuts the memory off from the input, so it tests interactions between qubits together with entanglement.
- The frozen circuit sits in the regime where gradients and readout features concentrate; it is a weak random-feature model, not the best fixed quantum feature map.
- The feature-order control has one training seed; the depth-matched tied control ran only on the four sets with four to six patches.
- The structured classical baselines are not exactly parameter-matched (RNN within 8%, MPS within 21% and about 20% smaller on the image sets, HMM 1.5 to 2 times larger).
- The gradient study covers 5 to 11 qubits and one angle per block; it shows the trend over this range, not an asymptotic law.
"""
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
