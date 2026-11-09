"""Figures and tables of the paper's method section, rebuilt from data/results (no torch needed):

    python scripts/paper_figures.py      ->  paper/figures/fig_capacity.pdf, fig_classical.pdf, fig_gradients.pdf, fig_bench.pdf,
                                             paper/tables/mnist_results.tex, bench_results.tex

Re-run it when the grid has finished; the .tex files only \\input / \\includegraphics these outputs, so no number is typed by hand.
Numbers: test accuracy at the last epoch (100), mean +- std over seeds. Classical models: learning rate with the best mean validation accuracy.
"""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG, TAB = ROOT / "paper/figures", ROOT / "paper/tables"
TASKS = ("3v5", "4v9", "4class", "10class")
TITLE = {"3v5": "3 vs 5", "4v9": "4 vs 9", "4class": "4 classes (0-3)", "10class": "10 classes"}
BLUE, ORANGE, AQUA, VIOLET, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#6b6a66"
CELLS = ((8, 2), (8, 4), (14, 2), (14, 4))       # the four sizes of the circuit: (image size, layers L)


def load(folder):
    return [json.loads(p.read_text()) for p in sorted((ROOT / "data/results" / folder).glob("*.json"))]


def collect():
    """q[(task, size, kind, L)] -> runs, kind in coherent / measured / amp7 / amp4; best[(task, size, model, L)] -> runs of the best lr"""
    q, c = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for r in load("quantum"):
        x = r["run"]
        q[(x["task"], x["image_size"], x["memory"], x["layers"])].append(r)
    for r in load("amplitude"):
        x = r["run"]
        q[(x["task"], x["image_size"], f"amp{x['qubits']}", x["layers"])].append(r)
    for r in load("classical"):
        x = r["run"]
        c[(x["task"], x["image_size"], x["model"], x["match"])][x["lr"]].append(r)
    best = {}
    for key, by_lr in c.items():
        n = max(len(v) for v in by_lr.values())
        best[key] = max((v for v in by_lr.values() if len(v) == n), key=lambda v: st.mean(r["final"]["val_acc"] for r in v))
    return q, best


def stat(rs, key="test_acc"):
    v = [100 * r["final"][key] for r in rs]
    return (st.mean(v), st.stdev(v) if len(v) > 1 else 0.0, len(v)) if v else None


def params(rs):
    return rs[0]["params"] if rs else None


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def errplot(ax, x, rs, **kw):
    s = stat(rs)
    if s:
        kw.setdefault("mec", "white" if kw.get("marker") != "x" else kw.get("color"))
        ax.errorbar(x, s[0], s[1], capsize=3, elinewidth=1, ms=6.5, mew=1.2 if kw.get("marker") == "x" else 0.8, **kw)


def fig_capacity(q, best):
    """accuracy vs number of trainable parameters: the serialized circuit (parameters grow with patches and L, qubits fixed at 7)
    and the amplitude-encoding circuits on 4 and 7 qubits (14x14 inputs, 2/4/8 layers) with the same block and readout."""
    fig, axs = plt.subplots(2, 2, figsize=(6.4, 5.0))
    axs = axs.ravel()
    for ax, task in zip(axs, TASKS):
        for mem, color, marker, label in (("coherent", BLUE, "o", "serialized, coherent memory (7 qubits)"),
                                          ("measured", ORANGE, "s", "serialized, measured memory (7 qubits)")):
            pts = [(params(q[(task, s, mem, L)]), stat(q[(task, s, mem, L)])) for s, L in CELLS if q.get((task, s, mem, L))]
            ax.errorbar([p for p, _ in pts], [s[0] for _, s in pts], [s[1] for _, s in pts], color=color, marker=marker, lw=1.8, ms=6.5,
                        mec="white", capsize=3, elinewidth=1, label=label, zorder=3)
        for n, color, marker in ((7, AQUA, "^"), (4, VIOLET, "v")):
            pts = [(params(q[(task, 14, f"amp{n}", L)]), stat(q[(task, 14, f"amp{n}", L)])) for L in (2, 4, 8) if q.get((task, 14, f"amp{n}", L))]
            ax.errorbar([p for p, _ in pts], [s[0] for _, s in pts], [s[1] for _, s in pts], color=color, marker=marker, lw=1.6, ls="--", ms=6.5,
                        mec="white", capsize=3, elinewidth=1, label=f"amplitude encoding, {n} qubits")
        ax.set_xscale("log")
        ax.set_xticks([100, 1000, 10000], ["100", "1000", "10000"])
        ax.minorticks_off()
        ax.set_title(TITLE[task])
        style(ax)
    for ax in axs[2:]:
        ax.set_xlabel("trainable parameters")
    for ax in axs[::2]:
        ax.set_ylabel("test accuracy (%)")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=1, frameon=False, bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.35)
    fig.savefig(FIG / "fig_capacity.pdf", bbox_inches="tight")


def fig_classical(q, best):
    """the serialized circuit (coherent and measured memory) against an MLP and a CNN with the same number of parameters, for the four sizes of the circuit"""
    fig, axs = plt.subplots(2, 2, figsize=(6.4, 5.4))
    axs = axs.ravel()
    series = (("serialized, coherent memory", BLUE, "o", lambda t, s, L: q.get((t, s, "coherent", L))),
              ("serialized, measured memory", ORANGE, "s", lambda t, s, L: q.get((t, s, "measured", L))),
              ("MLP, same parameters", AQUA, "^", lambda t, s, L: best.get((t, s, "mlp", L))),
              ("CNN, same parameters", VIOLET, "D", lambda t, s, L: best.get((t, s, "cnn", L))),
              ("logistic regression", GREY, "x", lambda t, s, L: best.get((t, s, "logistic", None))))
    for ax, task in zip(axs, TASKS):
        for j, (label, color, marker, get) in enumerate(series):
            for i, (s, L) in enumerate(CELLS):
                errplot(ax, i + (j - 2) * 0.14, get(task, s, L) or [], color=color, marker=marker, ls="none", label=label if i == 0 else None)
        ticks = [f"{s}x{s}\nL={L}\n{params(q.get((task, s, 'coherent', L)) or []) or ''}".rstrip() for s, L in CELLS]
        ax.set_xticks(range(len(CELLS)), ticks, fontsize=7.5)
        ax.set_title(TITLE[task])
        style(ax)
    for ax in axs[::2]:
        ax.set_ylabel("test accuracy (%)")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    fig.savefig(FIG / "fig_classical.pdf", bbox_inches="tight")


def mean(rs):
    s = stat(rs) if rs else None
    return None if s is None else s[0]


def bold_if(v, top):
    """True when the mean v is the best of its group, ties at the printed precision included"""
    return v is not None and top is not None and round(v, 1) == round(top, 1)


def table(q, best):
    """supplementary table: test accuracy for every task, circuit variant and the classical models of equal size; best of each column in bold"""
    def cell(rs, bold, mark=""):
        s = stat(rs) if rs else None
        if s is None:
            return "--"
        v = f"\\mathbf{{{s[0]:.1f}}}" if bold else f"{s[0]:.1f}"
        return f"${v}\\pm{s[1]:.1f}${mark}" + (f"\\,({s[2]})" if s[2] != 3 else "")
    rows = [("serialized, coherent", lambda t, s, L: q.get((t, s, "coherent", L))), ("serialized, measured", lambda t, s, L: q.get((t, s, "measured", L))),
            ("MLP, same parameters", lambda t, s, L: best.get((t, s, "mlp", L))), ("CNN, same parameters", lambda t, s, L: best.get((t, s, "cnn", L))),
            ("logistic regression", lambda t, s, L: best.get((t, s, "logistic", None)))]
    out = ["% generated by scripts/paper_figures.py -- do not edit", "\\begin{tabular}{@{}l" + "cc" * 4 + "@{}}", "\\toprule",
           " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{TITLE[t]}}}" for t in TASKS) + " \\\\",
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
           " & " + " & ".join("$8{\\times}8$ & $14{\\times}14$" for _ in TASKS) + " \\\\"]
    for L in (2, 4):
        out += ["\\midrule", f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{$L={L}$ (classical models sized to the circuit with $L={L}$)}}}} \\\\"]
        block = [(name, get) for name, get in rows if not (name.startswith("logistic") and L == 4)]
        top = {(t, s): max((m for m in (mean(get(t, s, L)) for _, get in block) if m is not None), default=None) for t in TASKS for s in (8, 14)}
        for name, get in block:
            cells = []
            for t in TASKS:
                for s in (8, 14):
                    rs = get(t, s, L) or []
                    stalled = (t, s, name, L) == ("10class", 8, "serialized, measured", 2)
                    cells.append(cell(rs, bold_if(mean(rs), top[(t, s)]), "$^\\dagger$" if stalled else ""))
            out.append(name + " & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    TAB.mkdir(exist_ok=True)
    (TAB / "mnist_results.tex").write_text("\n".join(out) + "\n")


def amp_table(q):
    """supplementary table: amplitude-encoding circuits on MNIST (4 and 7 qubits, L = 2, 4, 8); best of each column in bold"""
    rows = [(n, L) for n in (4, 7) for L in (2, 4, 8)]
    out = ["% generated by scripts/paper_figures.py -- do not edit", "\\begin{tabular}{@{}ll" + "cc" * 4 + "@{}}", "\\toprule",
           " & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{TITLE[t]}}}" for t in TASKS) + " \\\\",
           "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}",
           "qubits & $L$ & " + " & ".join("$8{\\times}8$ & $14{\\times}14$" for _ in TASKS) + " \\\\", "\\midrule"]
    top = {(t, s): max(mean(q.get((t, s, f"amp{n}", L), [])) or -1 for n, L in rows) for t in TASKS for s in (8, 14)}
    for i, (n, L) in enumerate(rows):
        if i == 3:
            out.append("\\midrule")
        cells = []
        for t in TASKS:
            for s in (8, 14):
                rs = q.get((t, s, f"amp{n}", L), [])
                st_ = stat(rs) if rs else None
                if st_ is None:
                    cells.append("--")
                    continue
                v = f"\\mathbf{{{st_[0]:.1f}}}" if bold_if(st_[0], top[(t, s)]) else f"{st_[0]:.1f}"
                cells.append(f"${v}\\pm{st_[1]:.1f}$")
        out.append(f"{n} & {L} & " + " & ".join(cells) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "mnist_amplitude.tex").write_text("\n".join(out) + "\n")
    p = sorted(params(q[k]) for k in q if k[2].startswith("amp") and k[1] in (8, 14))
    print(f"amplitude MNIST params {p[0]}-{p[-1]}")


# ---------------------------------------------------------------- twelve further datasets (data/results/bench, reports/bench.md)
BENCH = (("waveform", "Waveform"), ("wallrobot", "Wall-Following"), ("covertype", "Covertype"), ("letter", "Letter"),
         ("spambase", "Spambase"), ("creditdefault", "Credit Default"), ("landsat", "Landsat"), ("drybean", "Dry Bean"),
         ("fashion", "Fashion-MNIST"), ("bloodmnist", "BloodMNIST"), ("pneumonia", "PneumoniaMNIST"), ("kmnist", "Kuzushiji-MNIST"))
BENCH_K = {"waveform": 3, "wallrobot": 4, "covertype": 7, "letter": 26, "spambase": 2, "creditdefault": 2, "landsat": 6, "drybean": 7,
           "fashion": 10, "bloodmnist": 8, "pneumonia": 2, "kmnist": 10}


def bench_collect():
    """b[(dataset, model)] -> runs (last epoch), model in circuit / measured / readout42 / control_L9 / circuit_mlphead / amp4 / amp8 /
    logistic / mlp / cnn / hgb / mlp_mb / mlp_head (torch baselines: the learning rate with the best mean validation accuracy)"""
    b = defaultdict(list)
    for r in load("bench/quantum"):
        x = r["run"]
        if x["epochs"] != 100:
            continue
        ro = x.get("readout") or "z+zz"
        kind = ("circuit_mlphead" if x["head"] == "mlp" else "control_L9" if x["layers"] == 9 else
                "readout42" if ro != "z+zz" else "measured" if x["memory"] == "measured" else "circuit")
        b[(x["dataset"], kind)].append(r)
    for r in load("bench/amplitude"):
        b[(r["run"]["dataset"], f"amp{r['run']['layers']}")].append(r)
    by_lr = defaultdict(lambda: defaultdict(list))
    for r in load("bench/classical"):
        by_lr[(r["run"]["dataset"], r["run"]["model"])][r["run"]["lr"]].append(r)
    for key, d in by_lr.items():
        b[key] = max(d.values(), key=lambda v: st.mean(r["final"]["val_acc"] for r in v))
    return b


def amp_by_val(b, d):
    """the amplitude-encoding circuit (L = 4 or 8) with the higher mean validation accuracy, never chosen on test accuracy"""
    return max((b[(d, "amp4")], b[(d, "amp8")]), key=lambda rs: st.mean(r["final"]["val_acc"] for r in rs))


def bench_table(b, key="test_acc", name_out="bench_results"):
    """table of the twelve datasets: test accuracy (key test_acc) or balanced accuracy (key test_bacc) at the last epoch, mean +- std over three seeds"""
    cols = ("logistic", "mlp", "cnn", "hgb", "amp4", "amp8", "circuit", "measured")
    def cell(rs, bold):
        s = stat(rs, key)
        if s is None:
            return "--"
        v = f"{s[0]:.1f}"
        return (f"$\\mathbf{{{v}}}$" if bold else f"${v}$") + f"{{\\scriptsize$\\pm{s[1]:.1f}$}}"
    out = ["% generated by scripts/paper_figures.py -- do not edit", "\\begin{tabular}{@{}lrrr" + "r" * len(cols) + "@{}}", "\\toprule",
           " & & & & \\multicolumn{4}{c}{classical} & \\multicolumn{2}{c}{amplitude, 7 qubits} & \\multicolumn{2}{c@{}}{serialized, 7 qubits} \\\\",
           "\\cmidrule(lr){5-8}\\cmidrule(lr){9-10}\\cmidrule(l){11-12}",
           "dataset & $k$ & $N$ & params & logistic & MLP & CNN & grad.\\ boost. & $L{=}4$ & $L{=}8$ & "
           "\\multicolumn{1}{c}{coherent} & \\multicolumn{1}{c@{}}{measured} \\\\",
           "\\midrule"]
    for i, (d, name) in enumerate(BENCH):
        if i == 8:
            out.append("\\midrule")
        means = {c: stat(b.get((d, c), []), key)[0] for c in cols if b.get((d, c))}
        top = max(means.values())
        rs = b[(d, "circuit")]
        n_patch = {"waveform": 6, "wallrobot": 6, "covertype": 14, "letter": 4, "spambase": 15, "creditdefault": 6, "landsat": 9,
                   "drybean": 4}.get(d, 49)
        out.append(f"{name} & {BENCH_K[d]} & {n_patch} & {params(rs)} & " +
                   " & ".join(cell(b.get((d, c), []), bold_if(means.get(c), top)) for c in cols) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / f"{name_out}.tex").write_text("\n".join(out) + "\n")


def fig_bench(b):
    """difference in test accuracy to the classical MLP of the same size, per dataset; a: single models, b: the boosted-tree ensemble"""
    rows = sorted(BENCH, key=lambda dn: stat(b[(dn[0], "circuit")])[0] - stat(b[(dn[0], "mlp")])[0])
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.0, 5.0), sharey=True, gridspec_kw={"width_ratios": (2.6, 1), "wspace": 0.08})
    single = (("serialized, coherent memory", BLUE, "o", 60, lambda d: b.get((d, "circuit"))),
              ("serialized, measured memory", ORANGE, "s", 52, lambda d: b.get((d, "measured"))),
              ("serialized, coherent, three-basis readout", VIOLET, "D", 44, lambda d: b.get((d, "readout42"))),
              ("amplitude encoding, 7 qubits ($L=4$ or 8, by validation)", AQUA, "^", 60, lambda d: amp_by_val(b, d)))
    ensemble = (("gradient-boosted trees (ensemble of 300 trees)", GREY, "X", 60, lambda d: b.get((d, "hgb"))),)
    for a, series, title in ((ax, single, "a  single models"), (bx, ensemble, "b  ensemble, for reference")):
        a.axvspan(-2.5, 2.5, color="#eeede9", zorder=0)
        a.axvline(0, color=GREY, lw=0.8, zorder=1)
        for j, (label, color, marker, size, get) in enumerate(series):
            xs, ys = [], []
            for i, (d, _) in enumerate(rows):
                rs = get(d)
                if rs:
                    xs.append(stat(rs)[0] - stat(b[(d, "mlp")])[0])
                    ys.append(i + (j - (len(series) - 1) / 2) * 0.19)
            a.scatter(xs, ys, color=color, marker=marker, s=size, label=label, zorder=3, edgecolors="white", linewidths=0.7)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        a.grid(axis="x", color="#e4e3df", lw=0.8)
        a.set_axisbelow(True)
        a.tick_params(length=0)
        a.set_title(title, loc="left", fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(rows)), [n for _, n in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    fig.supxlabel("test accuracy minus that of the MLP with the same number of parameters (points)", fontsize=9, y=0.02)
    handles = [h for a in (ax, bx) for h in a.get_legend_handles_labels()[0]]
    labels = [l for a in (ax, bx) for l in a.get_legend_handles_labels()[1]]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=8)
    fig.savefig(FIG / "fig_bench.pdf", bbox_inches="tight")


def bench_summary(b):
    """the numbers quoted in the text"""
    diff = lambda d, a, c: stat(b[(d, a)])[0] - stat(b[(d, c)])[0]
    for d, name in BENCH:
        amp = stat(amp_by_val(b, d))[0]
        line = (f"{name:16s} circuit-MLP {diff(d, 'circuit', 'mlp'):+5.1f}  circuit-amp {stat(b[(d, 'circuit')])[0] - amp:+5.1f}  "
                f"circuit-hgb {diff(d, 'circuit', 'hgb'):+5.1f}")
        if b.get((d, "measured")):
            line += f"  measured-coherent {diff(d, 'measured', 'circuit'):+5.1f}"
        if b.get((d, "readout42")):
            line += f"  readout42-circuit {diff(d, 'readout42', 'circuit'):+5.1f}"
        print(line)


def readout_table(q, b):
    """three-basis readout: circuit with Z, ZZ (14 features) vs Z, ZZ, X, XX, Y, YY (42), and the MLP re-sized to the 42-feature circuit"""
    mb = load("multibasis")
    mnist42 = [r for r in mb if r["run"].get("readout") == "z+zz+x+xx+y+yy"]
    by_lr = defaultdict(list)
    for r in mb:
        if r["run"].get("model") == "mlp":
            by_lr[r["run"]["lr"]].append(r)
    mnist_mlp = max(by_lr.values(), key=lambda v: st.mean(r["final"]["val_acc"] for r in v))
    rows = [("MNIST $8\\times8$", 10, q[("10class", 8, "coherent", 4)], mnist42, mnist_mlp)]
    for d, name in (("letter", "Letter"), ("covertype", "Covertype"), ("landsat", "Landsat"), ("drybean", "Dry Bean"),
                    ("wallrobot", "Wall-Following"), ("waveform", "Waveform")):
        rows.append((name, BENCH_K[d], b[(d, "circuit")], b[(d, "readout42")], b.get((d, "mlp_mb"), [])))
    def f(rs, top):
        v = f"{stat(rs)[0]:.1f}"
        return ("$\\mathbf{" + v + "}" if bold_if(stat(rs)[0], top) else "$" + v) + f"\\pm{stat(rs)[1]:.1f}$"
    out = ["% generated by scripts/paper_figures.py -- do not edit", "\\begin{tabular}{@{}lrlllr@{}}", "\\toprule",
           "dataset & $k$ & $Z$, $ZZ$ (14 features) & $+X$, $XX$, $Y$, $YY$ (42) & gain & MLP, size of 42-feature circuit \\\\", "\\midrule"]
    for name, k, a, c, m in rows:
        n = lambda rs: f"{params(rs):,}".replace(",", "\\,")
        top = max(stat(rs)[0] for rs in (a, c, m) if rs)
        out.append(f"{name} & {k} & {f(a, top)} ({n(a)}) & {f(c, top)} ({n(c)}) & ${stat(c)[0] - stat(a)[0]:+.1f}$ & "
                   + (f(m, top) if m else "--") + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "readout_results.tex").write_text("\n".join(out) + "\n")
    ctrl = b[("letter", "control_L9")]
    print(f"Letter control L=9: {stat(ctrl)[0]:.1f} +- {stat(ctrl)[1]:.1f} ({params(ctrl)}), gain {stat(ctrl)[0] - stat(b[('letter', 'circuit')])[0]:+.1f}")
    head, mh = b[("letter", "circuit_mlphead")], b[("letter", "mlp_head")]
    print(f"Letter MLP head: {stat(head)[0]:.1f} ({params(head)}); MLP of that size: {stat(mh)[0]:.1f} ({params(mh)})")


def letter_table(b):
    """supplementary table: Letter, circuit readout/head/depth variants against a one-hidden-layer MLP of the same size"""
    rows = [("$Z$, $ZZ$ (14 features), linear head, $L=4$", b[("letter", "circuit")], b[("letter", "mlp")]),
            ("$Z$, $ZZ$ (14 features), linear head, $L=9$", b[("letter", "control_L9")], []),
            ("$+X$, $XX$, $Y$, $YY$ (42 features), linear head, $L=4$", b[("letter", "readout42")], b[("letter", "mlp_mb")]),
            ("$Z$, $ZZ$ (14 features), MLP head (64 units), $L=4$", b[("letter", "circuit_mlphead")], b[("letter", "mlp_head")])]
    n = lambda rs: f"{params(rs):,}".replace(",", "\\,")
    def f(rs, bold):
        v = f"{stat(rs)[0]:.1f}"
        return ("$\\mathbf{" + v + "}" if bold else "$" + v) + f"\\pm{stat(rs)[1]:.1f}$"
    out = ["% generated by scripts/paper_figures.py -- do not edit", "\\begin{tabular}{@{}lrlrl@{}}", "\\toprule",
           " & \\multicolumn{2}{c}{circuit} & \\multicolumn{2}{c@{}}{MLP, same size} \\\\",
           "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}",
           "readout and head & params & accuracy & params & accuracy \\\\", "\\midrule"]
    for name, c, m in rows:
        top = max(stat(rs)[0] for rs in (c, m) if rs)
        out.append(f"{name} & {n(c)} & {f(c, bool(m) and bold_if(stat(c)[0], top))} & " +
                   (f"{n(m)} & {f(m, bold_if(stat(m)[0], top))}" if m else "-- & --") + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "letter_variants.tex").write_text("\n".join(out) + "\n")


def fig_gradients(q):
    """trainability diagnostics of the MNIST grid, 10 classes, L = 4: (a) whole-model gradient norm (before clipping, mean over the
    batches of an epoch) against the epoch; (b, c) gradient norm of each patch's W_i (first batch of epoch 1 and of epoch 100) divided by
    its mean over the patches, against the relative patch position. The per-patch values are read after gradient clipping, which rescales
    all patches of a batch alike, so the ratios plotted are not affected by it. Lines: geometric mean over the three seeds; bands: min-max."""
    import math
    cfg = (("coherent, $8\\times8$ (16 patches)", BLUE, "-", ("10class", 8, "coherent", 4)),
           ("coherent, $14\\times14$ (49 patches)", AQUA, "-", ("10class", 14, "coherent", 4)),
           ("measured, $8\\times8$ (16 patches)", ORANGE, "-", ("10class", 8, "measured", 4)),
           ("measured, $14\\times14$ (49 patches)", VIOLET, "-", ("10class", 14, "measured", 4)))
    gmean = lambda v: math.exp(st.mean(math.log(max(x, 1e-12)) for x in v))
    fig, axs = plt.subplots(1, 3, figsize=(9.6, 3.0))
    for label, color, ls, key in cfg:
        rs = q[key]
        ep = [h["epoch"] for h in rs[0]["history"]]
        per_epoch = list(zip(*[[h["grad_norm"] for h in r["history"]] for r in rs]))
        axs[0].plot(ep, [gmean(v) for v in per_epoch], color=color, ls=ls, lw=1.5, label=label)
        axs[0].fill_between(ep, [min(v) for v in per_epoch], [max(v) for v in per_epoch], color=color, alpha=0.1, lw=0)
        for ax, e in ((axs[1], "1"), (axs[2], "100")):
            rel = [[g / st.mean(r["patch_grad_norm"][e]) for g in r["patch_grad_norm"][e]] for r in rs]
            n = len(rel[0])
            x = [(t + 0.5) / n for t in range(n)]
            cols = list(zip(*rel))
            ax.plot(x, [gmean(v) for v in cols], color=color, ls=ls, lw=1.5)
            ax.fill_between(x, [min(v) for v in cols], [max(v) for v in cols], color=color, alpha=0.1, lw=0)
    axs[0].set(yscale="log", xlabel="epoch", ylabel="gradient norm, whole model", title="a  training")
    for ax, t in ((axs[1], "b  per patch, epoch 1"), (axs[2], "c  per patch, epoch 100")):
        ax.set(yscale="log", xlabel="patch position $i/N$ (first $\\to$ last)", title=t, xlim=(0, 1))
        ax.axhline(1, color=GREY, lw=0.7)
    axs[1].set_ylabel("$\\|\\nabla_{W_i}\\mathcal{L}\\|$ / mean over patches")
    axs[2].set_ylim(1e-4, 10)
    axs[1].set_ylim(1e-4, 10)
    for ax in axs:
        style(ax)
        ax.grid(axis="y", which="major", color="#e4e3df", lw=0.8)
        ax.title.set_ha("left")
        ax.title.set_x(0.0)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_gradients.pdf", bbox_inches="tight")


# ---------------------------------------------------------------- Tier 1 review controls (data/results/review)
REVIEW_SETS = (("waveform", "Waveform"), ("wallrobot", "Wall-Following"), ("letter", "Letter"), ("drybean", "Dry Bean"),
               ("landsat", "Landsat"), ("spambase", "Spambase"), ("mnist8_3v5", "MNIST $8\\times8$, 3 vs.\\ 5"),
               ("mnist8_10class", "MNIST $8\\times8$, 10 classes"))
REVIEW_K = {**BENCH_K, "mnist8_3v5": 2, "mnist8_10class": 10}
REVIEW_T = {"waveform": 6, "wallrobot": 6, "letter": 4, "drybean": 4, "landsat": 9, "spambase": 15, "mnist8_3v5": 16, "mnist8_10class": 16}


def review_collect(q, b):
    """r[(dataset, kind)] -> runs; kind: ref (the main circuit runs these controls are compared with), tied, tiedmatched, product, frozen,
    perm (seed 42, three feature orders), pred (seed-42 rerun with stored predictions); and the structured classical models (seqrnn,
    mps, hmm, mlp_pred) with the learning rate of the best mean validation accuracy."""
    r = defaultdict(list)
    for x in load("review/quantum"):
        u = x["run"]
        kind = ("tiedmatched" if u["tie"] and u["layers"] > 4 else "tied" if u["tie"] else "product" if not u["entangle"] else
                "frozen" if u["frozen"] else "perm" if u["perm"] else u["tag"] or "ref")
        r[(u["dataset"], kind)].append(x)
    for d, _ in REVIEW_SETS:
        r[(d, "ref")] = q[(d.split("_")[1], 8, "coherent", 4)] if d.startswith("mnist8") else b[(d, "circuit")]
        if not d.startswith("mnist8"):
            r[(d, "measured")] = b[(d, "measured")]
    by_lr = defaultdict(lambda: defaultdict(list))
    for x in load("review/classical"):
        u = x["run"]
        by_lr[(u["dataset"], u["model"])][u["lr"]].append(x)
    for key, d in by_lr.items():
        r[(key[0], key[1] if key[1] != "mlp" else "mlp_pred")] = max(d.values(), key=lambda v: st.mean(x["final"]["val_acc"] for x in v))
    return r


def review_table(r):
    """Table of the circuit controls: test accuracy (mean +- std over the seeds present) of the circuit and of each control."""
    cols = (("ref", "circuit"), ("tied", "tied blocks"), ("tiedmatched", "tied, same params"), ("product", "product gates"),
            ("frozen", "frozen random"))
    cell = lambda rs: "--" if not rs else f"${stat(rs)[0]:.1f}${{\\scriptsize$\\pm{stat(rs)[1]:.1f}$}}" + ("" if len(rs) == 3 else f"$^{{{len(rs)}}}$")
    out = ["% generated by scripts/paper_figures.py -- do not edit", "\\begin{tabular}{@{}lrr" + "r" * len(cols) + "r@{}}", "\\toprule",
           "dataset & $k$ & $N$ & " + " & ".join(c for _, c in cols) + " & feature order \\\\", "\\midrule"]
    for d, name in REVIEW_SETS:
        perm = r.get((d, "perm"), [])
        ref42 = [x for x in r[(d, "ref")] if x["run"]["seed"] == 42]
        if perm and ref42:
            diffs = [100 * (x["final"]["test_acc"] - ref42[0]["final"]["test_acc"]) for x in perm]
            po = f"${min(diffs):+.1f}$ to ${max(diffs):+.1f}$"
        else:
            po = "--"
        out.append(f"{name} & {REVIEW_K[d]} & {REVIEW_T[d]} & " + " & ".join(cell(r.get((d, k), [])) for k, _ in cols) + f" & {po} \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "review_controls.tex").write_text("\n".join(out) + "\n")


def struct_table(r, b):
    """Classical models with the structure of the circuit (patch by patch, position-specific parameters) on every dataset."""
    rows = [(d, n) for d, n in BENCH] + [("mnist8_3v5", "MNIST $8\\times8$, 3 vs.\\ 5"), ("mnist8_10class", "MNIST $8\\times8$, 10 classes")]
    cols = ("ref", "measured", "mlp", "seqrnn", "mps", "hmm")
    out = ["% generated by scripts/paper_figures.py -- do not edit", "\\begin{tabular}{@{}l" + "r" * len(cols) + "@{}}", "\\toprule",
           " & \\multicolumn{2}{c}{serialized circuit} & \\multicolumn{4}{c@{}}{classical} \\\\",
           "\\cmidrule(lr){2-3}\\cmidrule(l){4-7}",
           "dataset & coherent & measured & MLP & seq.\\ RNN & MPS & 8-state HMM \\\\", "\\midrule"]
    q, best = collect()
    for d, name in rows:
        rs = {"ref": r.get((d, "ref")) or b.get((d, "circuit")),
              "measured": (r.get((d, "measured")) or b.get((d, "measured"))) if not d.startswith("mnist8") else q[(d.split("_")[1], 8, "measured", 4)],
              "mlp": b.get((d, "mlp")) if not d.startswith("mnist8") else best[(d.split("_")[1], 8, "mlp", 4)],
              "seqrnn": r.get((d, "seqrnn")), "mps": r.get((d, "mps")), "hmm": r.get((d, "hmm"))}
        means = {c: stat(v)[0] for c, v in rs.items() if v}
        top = max(means.values())
        out.append(f"{name} & " + " & ".join("--" if c not in means else
                   (f"$\\mathbf{{{means[c]:.1f}}}$" if bold_if(means[c], top) else f"${means[c]:.1f}$") for c in cols) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    (TAB / "struct_baselines.tex").write_text("\n".join(out) + "\n")
    ps = {m: sorted(x[0]["params"] for (d, k), x in r.items() if k == m) for m in ("seqrnn", "mps", "hmm")}
    print("structured baseline params", {m: (v[0], v[-1]) for m, v in ps.items() if v})


def fig_gradvar():
    """Variance of the gradient over 200 random initializations against the number of qubits (memory 1-7 ancillas)."""
    path = ROOT / "data/results/review/grad_variance.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text())
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    for ax, cost, title in ((axs[0], "local", "a  local cost $\\langle Z_{s_0}\\rangle$"), (axs[1], "global", "b  global cost $\\langle Z^{\\otimes n}\\rangle$")):
        for init, color, label in (("identity", BLUE, "near-identity start (as trained)"), ("uniform", ORANGE, "uniform random angles")):
            for T, ls, mk in ((16, "-", "o"), (4, "--", "s")):
                pts = sorted((x["n"], x[f"var_{cost}_first"]) for x in rows if x["init"] == init and x["T"] == T)
                if pts:
                    ax.plot(*zip(*pts), color=color, ls=ls, marker=mk, ms=4, lw=1.4, label=f"{label}, $N={T}$")
        ns = list(range(5, 12))
        ax.plot(ns, [0.9e-3 * 2.0 ** -(n - 5) for n in ns], color=GREY, lw=0.8, ls=":", label="$\\propto 2^{-n}$")
        ax.set(yscale="log", xlabel="qubits $n$ (4 system + 1 to 7 memory)", title=title)
        style(ax)
        ax.title.set_ha("left")
        ax.title.set_x(0.0)
    axs[0].set_ylabel("Var of $\\partial C/\\partial\\theta$ (first patch)")
    fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.2), fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_gradvar.pdf", bbox_inches="tight")


def review_summary(r):
    """numbers quoted in the text: control minus circuit (mean over the seeds present), and paired McNemar tests on seed 42"""
    from math import comb
    def paired(d, k):                       # control minus circuit, averaged over the seeds both have
        ref = {x["run"]["seed"]: x["final"]["test_acc"] for x in r[(d, "ref")]}
        v = [100 * (x["final"]["test_acc"] - ref[x["run"]["seed"]]) for x in r[(d, k)] if x["run"]["seed"] in ref]
        return st.mean(v), min(v), max(v), len(v)
    for d, name in REVIEW_SETS:
        print(f"{d:15s} " + "  ".join("{} {:+5.1f} ({:+.1f}..{:+.1f}) [{}]".format(k, *paired(d, k)) for k in ("tied", "tiedmatched", "product", "frozen")
                                       if r.get((d, k))))
    for d, name in BENCH:
        q, m = r.get((d, "pred")), r.get((d, "mlp_pred"))
        if not q or not m:
            continue
        lr = next(x["run"]["lr"] for x in BENCH_MLP[d]) if d in BENCH_MLP else None
        mm = [x for x in load("review/classical") if x["run"]["dataset"] == d and x["run"]["model"] == "mlp" and x["run"]["lr"] == lr]
        a = [p == t for p, t in zip(q[0]["test_pred"], q[0]["test_y"])]
        c = [p == t for p, t in zip(mm[0]["test_pred"], mm[0]["test_y"])]
        n01, n10 = sum(u and not v for u, v in zip(a, c)), sum(v and not u for u, v in zip(a, c))
        k, n = min(n01, n10), n01 + n10
        pval = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
        print(f"McNemar {name:15s} circuit-MLP {100 * (sum(a) - sum(c)) / len(a):+5.1f}  ({n01} vs {n10})  p = {pval:.2g}")


BENCH_MLP = {}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 150, "pdf.fonttype": 42})
    q, best = collect()
    fig_capacity(q, best)
    fig_classical(q, best)
    fig_gradients(q)
    table(q, best)
    amp_table(q)
    b = bench_collect()
    bench_table(b)
    bench_table(b, "test_bacc", "bench_balanced")
    fig_bench(b)
    bench_summary(b)
    readout_table(q, b)
    letter_table(b)
    BENCH_MLP.update({d: b[(d, "mlp")] for d, _ in BENCH})
    rv = review_collect(q, b)
    review_table(rv)
    struct_table(rv, b)
    fig_gradvar()
    review_summary(rv)
    done = sum(len(v) for k, v in q.items() if k[2] in ("coherent", "measured"))
    print(f"paper figures and table written ({done} of 96 circuit runs found)")


if __name__ == "__main__":
    main()
