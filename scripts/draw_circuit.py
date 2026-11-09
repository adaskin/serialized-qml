"""Figures of the circuit: (1) PennyLane's own drawing of one patch, (2) the serialized overview with coherent memory,
(3) the overview with the ancilla measured after every patch (memory="measured").
Usage: python scripts/draw_circuit.py   (writes reports/figures/circuit_one_patch.png, circuit_overview.png, circuit_measured.png)"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, "src")
from qcircuit import SerializedQC

out = Path("reports/figures")
out.mkdir(parents=True, exist_ok=True)

# 1. one patch, 2 layers (the production circuit repeats the W_t layer 4 times), drawn by PennyLane from the model itself
fig, _ = SerializedQC(n_steps=1, layers=2).draw(style="pennylane", fontsize=11)
fig.savefig(out / "circuit_one_patch.png", dpi=90, bbox_inches="tight")

# 2. overview of the serialization
L, n, nd, nm = 4, 7, 4, 3
wt = L * (3 * n + 3 * (n - 1))
wc = 3 * nm + 3 * (nm - 1) + 3 * nm * nd
BLUE, ORANGE, GREY = "#2a6fb0", "#d9822b", "#555555"
ys = {f"s{i}": 6.0 - i for i in range(nd)} | {f"a{j}": 1.4 - j for j in range(nm)}

fig, ax = plt.subplots(figsize=(17, 6.6))
ax.set_xlim(-2.3, 17.4); ax.set_ylim(-2.9, 8.2); ax.axis("off")
for name, y in ys.items():
    col = BLUE if name[0] == "s" else ORANGE
    ax.plot([-0.4, 16.2], [y, y], color=col, lw=1.4, zorder=1)
    ax.text(-0.55, y, name, ha="right", va="center", color=col, fontsize=12, fontweight="bold")
ax.text(-2.2, 4.5, "system (one pixel per qubit)", rotation=90, ha="center", va="center", color=BLUE, fontsize=10)
ax.text(-2.2, 0.4, "ancilla memory", rotation=90, ha="center", va="center", color=ORANGE, fontsize=10)


def box(x0, x1, y0, y1, text, face, edge, fs=10, lw=1.5, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=face, ec=edge, lw=lw, zorder=3, alpha=alpha))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, text, ha="center", va="center", fontsize=fs, zorder=4)


def patch(x0, title, pixels, wt_text):
    ax.text(x0 + 1.95, 7.6, title, ha="center", fontsize=12, fontweight="bold")
    ax.plot([x0 - 0.15, x0 + 4.15], [7.2, 7.2], color=GREY, lw=1)
    for i in range(nd):                                    # pixel -> encoding gates
        y = ys[f"s{i}"]
        ax.text(x0 - 0.1, y + 0.32, pixels[i], ha="left", va="bottom", fontsize=9, color=GREY, zorder=4)
        box(x0 + 0.05, x0 + 1.15, y - 0.28, y + 0.28, "RY·RZ·RX", "#e4eefa", BLUE, fs=8.5)
    box(x0 + 1.4, x0 + 2.75, ys["a2"] - 0.45, ys["s0"] + 0.45, wt_text, "#eef6e8", "#4f8f3a", fs=10)
    box(x0 + 3.0, x0 + 4.05, ys["a2"] - 0.45, ys["s0"] + 0.45, "W_c\nshared by\nall patches\n\nRot, CRot\nchain, CRot\nancilla →\nevery system\nqubit", "#fbeede", ORANGE, fs=9)
    ax.text(x0 + 2.07, ys["a2"] - 0.85, f"{wt} parameters\n(own to this patch)", ha="center", va="top", fontsize=10, color="#4f8f3a")


patch(0.0, "patch 1", ["x[1,0]", "x[1,1]", "x[1,2]", "x[1,3]"], f"W_1\n\n{L} ×\n[ Rot on all\n7 qubits,\nCRot chain\ns0→…→a2 ]")
patch(4.9, "patch 2", ["x[2,0]", "x[2,1]", "x[2,2]", "x[2,3]"], f"W_2\n\n{L} ×\n[ Rot on all\n7 qubits,\nCRot chain\ns0→…→a2 ]")
ax.text(9.95, 4.5, "· · ·", ha="center", va="center", fontsize=26, color=GREY)
patch(10.9, "patch T", ["x[T,0]", "x[T,1]", "x[T,2]", "x[T,3]"], f"W_T\n\n{L} ×\n[ Rot on all\n7 qubits,\nCRot chain\ns0→…→a2 ]")
box(15.2, 16.4, ys["a2"] - 0.45, ys["s0"] + 0.45, "⟨Z_q⟩\n⟨Z_q Z_q+1⟩\n(14 values)\n\n↓\nlinear\n↓\nlogits", "#f1e9f7", "#7a4aa0", fs=9)
ax.text(8.2, -2.75, f"memory = 'coherent': the same {n} qubits are reused for every patch with no reset and no mid-circuit measurement; the next patch's pixels are applied as new rotations.\n"
        f"Parameters = T·{wt}  +  W_c {wc}  +  encoding scales/offsets {2 * 3 * nd}  +  linear readout  (grows linearly with the number of patches; qubits and depth per patch stay fixed)",
        ha="center", va="center", fontsize=10.5, color=GREY)
fig.savefig(out / "circuit_overview.png", dpi=100, bbox_inches="tight")
print("figures written to", out)


# 3. memory="measured": every patch is a fresh short circuit; only the measured ancilla bits connect the patches
fig, ax = plt.subplots(figsize=(19.5, 6.6))
ax.set_xlim(-2.3, 22.2); ax.set_ylim(-2.9, 8.2); ax.axis("off")
GREEN = "#4f8f3a"
xs = [0.0, 6.6, 13.2]
for k, x0 in enumerate(xs):
    last = k == len(xs) - 1
    for name, y in ys.items():
        col = BLUE if name[0] == "s" else ORANGE
        ax.plot([x0 - 0.9, x0 + 4.7 if (name[0] == 'a' or last) else x0 + 4.1], [y, y], color=col, lw=1.4, zorder=1)
        start = "|0>" if (name[0] == "s" or k == 0) else "|m>"
        ax.text(x0 - 0.95, y, start, ha="right", va="center", color=col, fontsize=10)
        if name[0] == "s":
            ax.text(x0 - 0.58, y, "", ha="right")
    ax.text(x0 + 1.95, 7.6, ["patch 1", "patch t", "patch T"][k], ha="center", fontsize=12, fontweight="bold")
    ax.plot([x0 - 0.15, x0 + 4.15], [7.2, 7.2], color=GREY, lw=1)
    for i in range(nd):
        y = ys[f"s{i}"]
        box(x0 + 0.05, x0 + 1.15, y - 0.28, y + 0.28, "RY·RZ·RX", "#e4eefa", BLUE, fs=8.5)
        ax.text(x0 - 0.1, y + 0.32, f"x[{['1', 't', 'T'][k]},{i}]", ha="left", va="bottom", fontsize=9, color=GREY, zorder=4)
    box(x0 + 1.4, x0 + 2.75, ys["a2"] - 0.45, ys["s0"] + 0.45, f"W_{['1', 't', 'T'][k]}\n\n{L} ×\n[ Rot on all\n7 qubits,\nCRot chain\ns0→…→a2 ]", "#eef6e8", GREEN, fs=10)
    box(x0 + 3.0, x0 + 4.05, ys["a2"] - 0.45, ys["s0"] + 0.45, "W_c\nshared", "#fbeede", ORANGE, fs=9)
    for name in (ys if last else [n_ for n_ in ys if n_[0] == "a"]):          # meters: ancilla after every patch, all qubits after the last
        y = ys[name]
        box(x0 + 4.1, x0 + 4.7, y - 0.25, y + 0.25, "Z", "#f1e9f7", "#7a4aa0", fs=9)
    if not last:
        for j in range(nm):                                                   # classical bit -> start state of the next patch
            y = ys[f"a{j}"]
            ax.annotate("", xy=(xs[k + 1] - 1.45, y), xytext=(x0 + 4.7, y), arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=1.4, ls="--"))
        ax.text((x0 + 4.7 + xs[k + 1] - 1.45) / 2, ys["a0"] + 0.5, "3 bits", ha="center", fontsize=9, color=ORANGE)
        if k == 0:
            ax.text(x0 + 4.2, ys["s3"] - 0.55, "system qubits discarded", ha="left", va="center", fontsize=8.5, color=BLUE)
ax.text(11.75, 4.5, "· · ·", ha="center", va="center", fontsize=26, color=GREY)
ax.text(18.7, 3.5, "⟨Z_q⟩, ⟨Z_q Z_q+1⟩\n→ linear\n→ logits", ha="left", va="center", fontsize=10, color="#7a4aa0")
ax.text(9.5, -2.75, "memory = 'measured': every patch is a short fresh circuit. The system qubits start in |0>; the ancilla starts from the state measured after the previous patch\n"
        "(|0> at the start of each image) and is measured again at the end. Only the 3 measured bits connect the patches; the outcome probabilities are chained exactly.",
        ha="center", va="center", fontsize=10.5, color=GREY)
fig.savefig(out / "circuit_measured.png", dpi=100, bbox_inches="tight")
print("measured-memory figure written")
