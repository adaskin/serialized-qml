"""Tables and figures of the whole grid: python src/report.py  ->  reports/results.md and reports/figures/fig_*.png

Numbers are the last epoch (100), mean +- standard deviation over seeds (n in brackets). Classical baselines: per model, the
learning rate with the best mean validation accuracy is used (see classical.py). Test accuracy is never used for selection.
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import TASKS

OUT, FIG = Path("reports"), Path("reports/figures")
SIZES = (8, 14, 28)
TASK_TITLE = {"3v5": "3 vs 5", "4v9": "4 vs 9", "4class": "4 classes (0-3)", "10class": "10 classes"}
BLUE, ORANGE, AQUA, VIOLET, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#6b6a66"


NOTE_28 = ("Note: the serialized circuit was not run at 28x28 (196 patches). The simulation was too slow: one epoch took about 47 minutes for "
           "3 vs 5 and 4 vs 9 and about 2 hours for 10 classes with coherent memory (about 1 hour for 3 vs 5 with measured memory), on a machine "
           "shared with other jobs, so 100 epochs would need about 3 days (2 classes) to 8 days (10 classes) per run. With the grid recipe the circuit "
           "also did not train (reports/caution_28x28.md). The classical and amplitude-encoding baselines are shown at 28x28.")


STALL = (("10class", 8, "measured", 2), " ¹")      # the cell with the stalled seed 44
FOOTNOTE_STALL = ("¹ Seed 44 of this cell stalled at about 47% training accuracy (seeds 42 and 43 reached about 70%); the mean and the standard "
                  "deviation include it. Two re-runs with other initial angles on the same seed-44 data also stalled (test accuracy 51% and 55%), "
                  "while the seed-44 angles on the seed-42 data reached 71%, so the stall comes with the data of that seed, not with the initial "
                  "angles (`data/results/diagnostics`).")


def load(folder):
    return [json.loads(p.read_text()) for p in sorted(Path("data/results", folder).glob("*.json"))]


def collect():
    quantum, classical = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for r in load("quantum"):
        q = r["run"]
        quantum[(q["task"], q["image_size"], q["memory"], q["layers"])].append(r)
    for r in load("amplitude"):                                # amplitude-encoding baselines share the dict, memory = "amp<qubits>"
        a = r["run"]
        quantum[(a["task"], a["image_size"], f"amp{a['qubits']}", a["layers"])].append(r)
    for r in load("classical"):
        c = r["run"]
        classical[(c["task"], c["image_size"], c["model"], c["match"])][c["lr"]].append(r)
    best = {}                                                  # classical: the learning rate with the best mean validation accuracy
    for key, by_lr in classical.items():
        n = max(len(v) for v in by_lr.values())
        full = {lr: v for lr, v in by_lr.items() if len(v) == n}
        best[key] = max(full.values(), key=lambda v: st.mean(r["final"]["val_acc"] for r in v))
    return quantum, best


def stat(rs, field="test_acc"):
    v = [100 * r["final"][field] for r in rs]
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0, len(v)) if v else None


def cell(rs, field="test_acc"):
    s = stat(rs, field) if rs else None
    return "-" if s is None else f"{s[0]:.1f} ± {s[1]:.1f} ({s[2]})"


def models():
    """(label, kind, key into the quantum / classical dict, colour, marker, linestyle)"""
    return ([(f"quantum {m}, L={L}", "q", (m, L), c, mk, ls) for m, c, mk in (("coherent", BLUE, "o"), ("measured", ORANGE, "s"))
             for L, ls in ((4, "-"), (2, "--"))] +
            [("logistic regression", "c", ("logistic", None), GREY, "x", ":")] +
            [(f"{name} (sized as L={L})", "c", (key, L), c, mk, ls) for key, name, c, mk in (("mlp", "MLP", AQUA, "^"), ("cnn", "CNN", VIOLET, "D"))
             for L, ls in ((4, "-"), (2, "--"))] +
            [(f"amplitude encoding, {n} qubits, L={L}", "a", (f"amp{n}", L), c, mk, ls) for n, c in ((7, AQUA), (4, VIOLET))
             for L, mk, ls in ((8, "v", "-"), (4, "^", "--"), (2, "<", ":"))])


def runs_of(quantum, best, task, size, kind, key):
    return quantum.get((task, size) + key, []) if kind in "qa" else best.get((task, size) + key, [])


def tables(quantum, best):
    md = [NOTE_28 + "\n"]
    for field, title in (("test_acc", "Test accuracy (%)"), ("train_acc", "Training accuracy (%) at the last epoch")):
        md.append(f"## {title}, mean ± std over seeds (n)\n")
        for task in TASKS:
            md += [f"### {TASK_TITLE[task]}\n", "| model | " + " | ".join(f"{s}x{s}" for s in SIZES) + " |", "|---|" + "---|" * len(SIZES)]
            for label, kind, key, *_ in models():
                md.append(f"| {label} | " + " | ".join(cell(runs_of(quantum, best, task, s, kind, key), field)
                                                      + (STALL[1] if (task, s) + key == STALL[0] else "") for s in SIZES) + " |")
            md.append("")
            if task == "10class":
                md += [FOOTNOTE_STALL, ""]
    md += ["## Trainable parameters (patches: 8x8 = 16, 14x14 = 49, 28x28 = 196; always 7 qubits)\n",
           "| model | task | " + " | ".join(f"{s}x{s}" for s in SIZES) + " |", "|---|---|" + "---|" * len(SIZES)]
    for label, kind, key, *_ in models():
        for task in TASKS:
            ps = [(runs_of(quantum, best, task, s, kind, key) or [{}])[0].get("params", "-") for s in SIZES]
            md.append(f"| {label} | {TASK_TITLE[task]} | " + " | ".join(map(str, ps)) + " |")
    chosen = defaultdict(list)
    for (task, s, model, match), rs in best.items():
        chosen[(model, match)].append(f"{task}/{s}: {rs[0]['run']['lr']:g}")
    md += ["", "## Learning rates selected for the classical baselines (best mean validation accuracy)\n"]
    md += [f"- {m}{'' if L is None else f' (sized as L={L})'}: " + ", ".join(v) for (m, L), v in sorted(chosen.items(), key=str)]
    return md


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def lines(ax, quantum, best, task, selection):
    for label, kind, key, color, marker, ls in selection:
        pts = [(i, stat(runs_of(quantum, best, task, s, kind, key))) for i, s in enumerate(SIZES)]
        pts = [(i, s) for i, s in pts if s]
        if pts:
            ax.errorbar([i for i, _ in pts], [s[0] for _, s in pts], [s[1] for _, s in pts], color=color, marker=marker, ls=ls, lw=1.6,
                        ms=6, mec="white", mew=0.8, capsize=3, elinewidth=1, label=label)
    ax.set_xticks(range(len(SIZES)), [f"{s}x{s}\n{(s // 2) ** 2} patches" for s in SIZES])
    ax.set_xlim(-0.3, len(SIZES) - 0.7)
    style(ax)


def figures(quantum, best):
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 150})
    allm = models()
    # 1. the circuit only: coherent vs measured memory, 2 vs 4 layers, growing image size
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8), sharex=True)
    for ax, task in zip(axs, TASKS):
        lines(ax, quantum, best, task, allm[:4])
        ax.set_title(TASK_TITLE[task])
    axs[0].set_ylabel("test accuracy (%)")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Serialized circuit (7 qubits): test accuracy vs image size, 100 epochs, mean ± std over seeds", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig_quantum.png", bbox_inches="tight")
    # 2. circuit vs classical models of the same size (row 1: sized as L=4, row 2: sized as L=2)
    fig, axs = plt.subplots(2, 4, figsize=(15, 7), sharex=True)
    for row, L in enumerate((4, 2)):
        sel = [(m[0].split(",")[0].split(" (")[0],) + m[1:] for m in allm if (m[1] == "q" and m[2][1] == L) or m[2] == ("logistic", None)
               or (m[1] == "c" and m[2][1] == L)]
        for ax, task in zip(axs[row], TASKS):
            lines(ax, quantum, best, task, sel)
            ax.set_title(TASK_TITLE[task] if row == 0 else "")
        axs[row, 0].set_ylabel(f"circuit L={L} vs classical models of equal size\ntest accuracy (%)")
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout()
    fig.savefig(FIG / "fig_vs_classical.png", bbox_inches="tight")
    # 2b. the circuit vs amplitude encoding of the whole image on 7 and on 4 qubits (colour = qubits, line style = layers)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.8), sharex=True)
    sel = allm[:2] + [m for m in allm if m[1] == "a"]
    for ax, task in zip(axs, TASKS):
        lines(ax, quantum, best, task, sel)
        ax.set_title(TASK_TITLE[task])
    axs[0].set_ylabel("test accuracy (%)")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    fig.savefig(FIG / "fig_amplitude.png", bbox_inches="tight")
    # 3. parameters grow with the image, the number of qubits does not
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for L, color, marker in ((4, BLUE, "o"), (2, ORANGE, "s")):
        ps = [(runs_of(quantum, best, "10class", s, "q", ("coherent", L)) or [{}])[0].get("params") for s in SIZES]
        ax.plot(range(len(SIZES)), [p or float("nan") for p in ps], color=color, marker=marker, lw=1.6, ms=6, mec="white", label=f"circuit, L={L} (10 classes)")
    ps = [(runs_of(quantum, best, "10class", s, "c", ("logistic", None)) or [{}])[0].get("params") for s in SIZES]
    ax.plot(range(len(SIZES)), [p or float("nan") for p in ps], color=GREY, marker="x", ls=":", lw=1.6, label="logistic regression")
    ax.set_yscale("log"); ax.set_xticks(range(len(SIZES)), [f"{s}x{s}" for s in SIZES]); ax.set_ylabel("trainable parameters")
    ax.set_title("Parameters grow with the patches; qubits stay at 7"); ax.legend(frameon=False, fontsize=8.5); style(ax)
    fig.tight_layout()
    fig.savefig(FIG / "fig_params.png", bbox_inches="tight")
    # 4. training curves of the circuit (L=4): colour = image size, solid = coherent, dashed = measured; dots = test accuracy
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, task in zip(axs, TASKS):
        for s, color in zip(SIZES, (BLUE, ORANGE, AQUA)):
            for memory, ls in (("coherent", "-"), ("measured", "--")):
                rs = quantum.get((task, s, memory, 4), [])
                if not rs:
                    continue
                epochs = [h["epoch"] for h in rs[0]["history"]]
                ax.plot(epochs, [100 * st.mean(r["history"][i]["train_acc"] for r in rs) for i in range(len(epochs))], color=color, ls=ls, lw=1.4,
                        label=f"{s}x{s} {memory}")
                ev = [i for i, h in enumerate(rs[0]["history"]) if "test_acc" in h]
                ax.plot([epochs[i] for i in ev], [100 * st.mean(r["history"][i]["test_acc"] for r in rs) for i in ev], color=color, ls="none",
                        marker="o", ms=4, mec="white")
        ax.set_title(TASK_TITLE[task]); ax.set_xlabel("epoch"); style(ax)
    axs[0].set_ylabel("accuracy (%): line = train, dots = test")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(FIG / "fig_curves.png", bbox_inches="tight")


def main():
    quantum, best = collect()
    n_q, n_c = sum(len(v) for v in quantum.values()), sum(len(v) for v in best.values())
    md = ["# Results of the fixed grid\n",
          f"Finished runs: {n_q} quantum (serialized circuit and amplitude-encoding baselines), {n_c} classical (best learning rate only). Fixed settings: MNIST, 2x2 patches (4 pixel qubits + 3 ancilla "
          "qubits), 1000 training images per class, 200 validation images per class (monitoring only), the full MNIST test set of the chosen "
          "digits, 100 epochs, Adam lr 0.01 with cosine decay to 5%, batch 256, no early stopping or model selection (last epoch is reported).\n",
          "![circuit](figures/fig_quantum.png)\n", "![vs classical](figures/fig_vs_classical.png)\n", "![vs amplitude encoding](figures/fig_amplitude.png)\n", "![parameters](figures/fig_params.png)\n",
          "![curves](figures/fig_curves.png)\n"] + tables(quantum, best)
    OUT.mkdir(exist_ok=True)
    (OUT / "results.md").write_text("\n".join(md) + "\n")
    figures(quantum, best)
    print(f"reports/results.md written ({n_q} circuit runs, {n_c} classical runs)")


if __name__ == "__main__":
    main()
