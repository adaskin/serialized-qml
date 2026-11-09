"""Multi-basis readout (Z, ZZ, X, XX, Y, YY: 42 features instead of 14) against the frozen Z + ZZ circuit and classical models re-sized to it.

    python src/multibasis_report.py   ->  reports/multibasis.md, reports/figures/fig_multibasis.png

Group (cheap runs with more than 4 classes, decided 2026-11-06): MNIST 10 classes 8x8, Letter, Covertype, Landsat, Dry Bean; Wall-Following
(4 classes) and Waveform (3 classes) are small-class checks. Everything is coherent memory, L = 4, linear head, seeds 42-44, 100 epochs, last
epoch (the grid protocol). The MLP / CNN baselines have one hidden layer / the small CNN, sized to the parameter count of the circuit they
are compared with; their learning rate is picked by mean validation accuracy.
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import datasets

MB = "z+zz+x+xx+y+yy"
FIG, OUT = Path("reports/figures"), Path("reports/multibasis.md")
GROUP = ("mnist", "letter", "covertype", "landsat", "drybean", "wallrobot", "waveform")
TITLE = {"mnist": "MNIST 8x8", "letter": "Letter", "covertype": "Covertype", "landsat": "Landsat", "drybean": "Dry Bean",
         "wallrobot": "Wall-Following", "waveform": "Waveform"}
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6b6a66"


def load(pattern, folder):
    return [json.loads(p.read_text()) for p in sorted(Path(folder).glob(pattern))]


def best_lr(runs):
    by = defaultdict(list)
    for r in runs:
        by[r["run"]["lr"]].append(r)
    return by[max(by, key=lambda l: np.mean([r["final"]["val_acc"] for r in by[l]]))] if by else []


def collect():
    """dataset -> {model: runs} with models zz, mb (circuits), mlp, cnn (sized to Z+ZZ), mlp_mb, cnn_mb (sized to the 42-feature circuit)."""
    rows = {}
    q, c = "data/results/bench/quantum", "data/results/bench/classical"
    for d in GROUP:
        if d == "mnist":
            m = dict(zz=load("mnist_10class_s8_coherent_L4_seed4?.json", "data/results/quantum"),
                     mb=load("mnist_10class_s8_coherent_L4_seed4?_rozzzxxxyyy.json", "data/results/multibasis"),
                     mlp=best_lr(load("mnist_10class_s8_mlp-L4_lr*.json", "data/results/classical")),
                     cnn=best_lr(load("mnist_10class_s8_cnn-L4_lr*.json", "data/results/classical")),
                     mlp_mb=best_lr(load("mnist_10class_s8_mlp-L4-f42_lr*.json", "data/results/multibasis")),
                     cnn_mb=best_lr(load("mnist_10class_s8_cnn-L4-f42_lr*.json", "data/results/multibasis")))
            m["classes"], m["T"] = 10, 16
        else:
            m = dict(zz=load(f"{d}_coherent_L4_linear_seed4?.json", q), mb=load(f"{d}_coherent_L4_linear_seed4?_rozzzxxxyyy.json", q),
                     mlp=best_lr(load(f"{d}_mlp_lr*.json", c)), cnn=[], mlp_mb=best_lr(load(f"{d}_mlp_mb_lr*.json", c)), cnn_mb=[])
            m["classes"], m["T"] = datasets.SPECS[d].n_classes, datasets.SPECS[d].steps
        if d == "letter":
            m["zz_L9"] = load("letter_coherent_L9_linear_seed4?.json", q)
        rows[d] = m
    return rows


def acc(runs):
    v = np.array([100 * r["final"]["test_acc"] for r in runs])
    return (v.mean(), v.std()) if len(v) else (np.nan, np.nan)


def fmt(runs, params=True):
    if not runs:
        return "-"
    m, s = acc(runs)
    return f"{m:.1f} ± {s:.1f}" + (f" ({runs[0]['params']})" if params else "")


def table(rows):
    out = ["| dataset | classes | T | circuit Z+ZZ | circuit Z+ZZ+X+XX+Y+YY | gain | MLP sized to Z+ZZ circuit | MLP sized to 42-feature circuit "
           "| CNN sized to Z+ZZ circuit | CNN sized to 42-feature circuit |", "|---|---|---|---|---|---|---|---|---|---|"]
    for d, m in rows.items():
        gain = acc(m["mb"])[0] - acc(m["zz"])[0]
        out.append(f"| {TITLE[d]} | {m['classes']} | {m['T']} | {fmt(m['zz'])} | {fmt(m['mb'])} | {gain:+.1f} | {fmt(m['mlp'])} | "
                   f"{fmt(m['mlp_mb'])} | {fmt(m['cnn'])} | {fmt(m['cnn_mb'])} |")
    return out


def figure(rows):
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 150})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [2.3, 1]})
    series = (("zz", "circuit, Z+ZZ (14 features)", GREY), ("mb", "circuit, Z+ZZ+X+XX+Y+YY (42 features)", BLUE),
              ("mlp_mb", "MLP sized to the 42-feature circuit", ORANGE), ("cnn_mb", "CNN sized to the 42-feature circuit", AQUA))
    ds = [d for d in GROUP if rows[d]["mlp_mb"]]                     # the group with re-sized baselines
    w = 0.2
    for i, (key, label, color) in enumerate(series):
        for j, d in enumerate(ds):
            if not rows[d][key]:
                continue
            m, s = acc(rows[d][key])
            ax.bar(j + (i - 1.5) * w, m, w * 0.9, yerr=s, color=color, label=label if j == 0 or key == "cnn_mb" else None,
                   error_kw=dict(lw=0.8, capsize=2, ecolor="#333"))
    ax.set_xticks(range(len(ds)), [f"{TITLE[d]}\n({rows[d]['classes']} classes)" for d in ds])
    ax.set_ylim(50, 100)
    ax.set_xlim(-0.6, len(ds) - 0.4)
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Coherent circuit, L = 4, linear head: readout in one vs three Pauli bases")
    fig.legend(frameon=False, fontsize=8.5, loc="lower center", ncol=4, bbox_to_anchor=(0.36, -0.06))
    # gain of the extra bases (points), sorted; the Letter capacity control (Z+ZZ, L = 9, same parameter count as 42 features) below it
    gains = sorted(((acc(m["mb"])[0] - acc(m["zz"])[0], f"{TITLE[d]} ({m['classes']})") for d, m in rows.items()))
    gains = [(acc(rows["letter"]["zz_L9"])[0] - acc(rows["letter"]["zz"])[0], "Letter, Z+ZZ L = 9 (control)")] + gains
    for i, (g, label) in enumerate(gains):
        bx.barh(i, g, 0.7, color=GREY if "control" in label else BLUE)
        bx.text(g + (0.15 if g >= 0 else -0.15), i, f"{g:+.1f}", va="center", ha="left" if g >= 0 else "right", fontsize=8.5, color="#333")
    bx.set_yticks(range(len(gains)), [l for _, l in gains])
    bx.axvline(0, color="#999", lw=0.8)
    bx.set_xlim(-2.2, 10.5)
    bx.set_xlabel("test accuracy gain over Z+ZZ, L = 4 (points)")
    bx.set_title("Gain of the extra bases (classes)")
    for a in (ax, bx):
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y" if a is ax else "x", color="#e5e5e5", lw=0.6)
        a.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIG / "fig_multibasis.png", bbox_inches="tight")


def main():
    rows = collect()
    t = table(rows)
    lc = rows["letter"]["zz_L9"]
    OUT.write_text("\n".join([
        "# Multi-basis readout: generated table", "",
        "Generated by `python src/multibasis_report.py` (setup and discussion in `reports/bench.md`). Test accuracy %, last epoch, mean ± std over "
        "seeds 42-44, trainable parameters in brackets (circuit + linear head; the classical models are sized to these counts). T = patches.", ""]
        + t + ["", f"Capacity control: Letter, Z+ZZ with L = 9: {fmt(lc)}.", "", "![multi-basis](figures/fig_multibasis.png)", ""]))
    figure(rows)
    print("\n".join(t))


if __name__ == "__main__":
    main()
