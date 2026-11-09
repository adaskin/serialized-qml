"""Tables for reports/bench_results.md from data/results/bench/{quantum,classical,amplitude}: mean +- std over seeds of the test accuracy
(and balanced accuracy), with two readings of every training run:

  last      the last epoch (the protocol of the MNIST grid: no model selection)
  best val  the checkpoint with the highest validation accuracy. Val and test are evaluated every 10 epochs (10, 20, ..., 100), so this
            is the best of ten checkpoints (ties go to the later one); nothing was retrained. Gradient boosting has no epochs: one value.

For the torch baselines the learning rate is the one with the best mean validation accuracy under the same reading.

    python src/bench_report.py [--out reports/bench_results.md] [--quantum-dir data/results/bench/quantum]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import datasets

ROOT = Path("data/results/bench")


def load(folder):
    return [json.loads(p.read_text()) for p in sorted(Path(folder).glob("*.json"))]


def pick(r, select):
    """The row of a run's history that is reported: its last epoch, or its best-validation checkpoint (later one on ties)."""
    if select == "last" or not r.get("history") or "val_acc" not in r["history"][-1]:
        return r["final"]
    rows = [h for h in r["history"] if "val_acc" in h]
    return max(rows, key=lambda h: (h["val_acc"], h["epoch"]))


def ms(vals):
    v = np.array(vals) * 100
    return f"{v.mean():.1f} ± {v.std():.1f}" if len(v) > 1 else f"{v.mean():.1f}"


def best_lr(runs, select):
    by_lr = defaultdict(list)
    for r in runs:
        by_lr[r["run"]["lr"]].append(r)
    lr = max(by_lr, key=lambda l: np.mean([pick(r, select)["val_acc"] for r in by_lr[l]]))
    return by_lr[lr]


def cell(runs, key, select):
    return ms([pick(r, select)[key] for r in runs]) if runs else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/bench_results.md")
    ap.add_argument("--quantum-dir", default=str(ROOT / "quantum"))
    a = ap.parse_args()
    quantum, classical, amplitude = load(a.quantum_dir), load(ROOT / "classical"), load(ROOT / "amplitude")
    qsel = lambda d, head, mem, ep, ro="z+zz": [r for r in quantum if r["run"]["dataset"] == d and r["run"]["head"] == head and r["run"]["layers"] == 4
                                                 and r["run"]["memory"] == mem and r["run"]["epochs"] == ep and r["run"].get("readout", "z+zz") == ro]
    asel = lambda d, layers: [r for r in amplitude if r["run"]["dataset"] == d and r["run"]["layers"] == layers]
    cols = ["logistic", "MLP (L4-sized)", "CNN (L4-sized)", "grad. boosting", "circuit L4 coherent, linear", "circuit L4 measured, linear",
            "amplitude 7q L4", "amplitude 7q L8", "circuit L4 coherent, MLP head", "MLP sized to the MLP-head circuit",
            "circuit L4 coherent, Z+ZZ+X+XX+Y+YY"]
    out = []
    for select, sname in (("last", "last epoch"), ("bestval", "best-validation checkpoint")):
        for key, title in (("test_acc", "Test accuracy (%)"), ("test_bacc", "Test balanced accuracy (%)")):
            out += [f"### {title}, {sname}, mean ± std over seeds", "",
                    "| dataset | classes | T | majority class | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 4)]
            for d, spec in datasets.SPECS.items():
                cl = [r for r in classical if r["run"]["dataset"] == d]
                by = lambda m: [r for r in cl if r["run"]["model"] == m]
                # predicting the majority class always: its accuracy is the majority share, its balanced accuracy is 1 / classes
                maj = (ms([r["majority_test"] for r in by("hgb")]) if by("hgb") else "-") if key == "test_acc" else f"{100 / spec.n_classes:.1f}"
                cells = [cell(best_lr(by("logistic"), select) if by("logistic") else [], key, select),
                         cell(best_lr(by("mlp"), select) if by("mlp") else [], key, select),
                         cell(best_lr(by("cnn"), select) if by("cnn") else [], key, select),
                         cell(by("hgb"), key, select),
                         cell(qsel(d, "linear", "coherent", 100), key, select), cell(qsel(d, "linear", "measured", 100), key, select),
                         cell(asel(d, 4), key, select), cell(asel(d, 8), key, select),
                         cell(qsel(d, "mlp", "coherent", 100), key, select),
                         cell(best_lr(by("mlp_head"), select) if by("mlp_head") else [], key, select),
                         cell(qsel(d, "linear", "coherent", 100, "z+zz+x+xx+y+yy"), key, select)]
                out.append(f"| {d} | {spec.n_classes} | {spec.steps} | {maj} | " + " | ".join(cells) + " |")
            out.append("")
    out += ["### Circuit runs (100 epochs unless named _e10)", "",
            "| run | params | train (last) | val (last) | test (last) | test (best val) | best-val epoch | balanced test (last) | hours |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(quantum, key=lambda r: r["name"]):
        f, b = r["final"], pick(r, "bestval")
        out.append(f"| {r['name']} | {r['params']} | {100 * f['train_acc']:.1f} | {100 * f['val_acc']:.1f} | {100 * f['test_acc']:.1f} | "
                   f"{100 * b['test_acc']:.1f} | {b['epoch']} | {100 * f['test_bacc']:.1f} | {r['seconds'] / 3600:.2f} |")
    out += ["", "### Sizes and parameter counts", "",
            "| dataset | n_train | n_val | n_test | circuit L4 linear | amplitude 7q L4 | amplitude 7q L8 | circuit L4 MLP head | MLP (L4-sized) |",
            "|---|---|---|---|---|---|---|---|---|"]
    first = lambda rs, k="params": rs[0][k] if rs else "-"
    for d in datasets.SPECS:
        ql = qsel(d, "linear", "coherent", 100)
        cm = [r for r in classical if r["run"]["dataset"] == d and r["run"]["model"] == "mlp"]
        out.append(f"| {d} | {first(ql, 'n_train')} | {first(ql, 'n_val')} | {first(ql, 'n_test')} | {first(ql)} | {first(asel(d, 4))} | "
                   f"{first(asel(d, 8))} | {first(qsel(d, 'mlp', 'coherent', 100))} | {first(cm)} |")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
