"""
fig2 cell maps built from the five 3x3 runs in ../run{1..5}_3x3 (one trial per
cell per run -> 5 trials per cell pooled).  Same visual language as
fig2_cell_maps_5_run.png: hit-rate colour + "hit%\nmean error" per cell, black
dots for every scored prediction, drawn to the 402 x 778 pt screen.
Output: fig2_cell_maps_5_run_real.png
"""
import os, csv, json
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RUNS = [f"run{i}_3x3" for i in range(1, 6)]
SW, SH = 402.0, 778.0
ROWS = COLS = 3

INK, INK3, LINE = "#12161b", "#78838f", "#d6dbe0"
GREENRED = matplotlib.colors.LinearSegmentedColormap.from_list(
    "gr", ["#c94f3d", "#e8b04b", "#f2efe6", "#7fc9a4", "#1baf7a"])

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 9,
    "axes.edgecolor": LINE, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def trial_rows(path):
    """First block of the multi-block trials.csv export."""
    out, header = [], None
    for line in open(path):
        line = line.rstrip("\n")
        if not line.strip() or line.startswith("#"):
            if header is not None:
                break
            continue
        rec = next(csv.reader([line]))
        if header is None:
            header = rec
        else:
            out.append(dict(zip(header, rec)))
    return out


# ---- load ------------------------------------------------------------------
runs = []
for d in RUNS:
    p = os.path.join(ROOT, d)
    m = json.load(open(f"{p}/meta.json"))
    s = m["summary"]
    runs.append(dict(name=d, meta=m, acc=s["accuracy_pct"],
                     ppd=s["mean_error_pt"] / s["mean_error_deg"],
                     rows=trial_rows(f"{p}/trials.csv")))

cw, ch = SW / COLS, SH / ROWS
hits = np.zeros((ROWS, COLS))          # hits per cell, out of 5
errs = [[[] for _ in range(COLS)] for _ in range(ROWS)]
dots = []                              # every scored prediction
per_run_hit = {}                       # (run, r, c) -> 0/1

for k, run in enumerate(runs):
    for t in run["rows"]:
        r, c = int(t["row"]), int(t["col"])
        hit = int(t["hit"])
        hits[r, c] += hit
        errs[r][c].append(float(t["err_pt"]) / run["ppd"])
        dots.append((float(t["pred_x"]), float(t["pred_y"])))
        per_run_hit[(k, r, c)] = hit
dots = np.array(dots)
pooled_acc = 100 * hits.sum() / (ROWS * COLS * len(runs))


def frame(ax):
    ax.set_xlim(0, SW); ax.set_ylim(SH, 0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_color(LINE)


fig = plt.figure(figsize=(12.6, 3.9))
gs = fig.add_gridspec(1, 7, width_ratios=[1.42, .04, 1, 1, 1, 1, 1], wspace=.20)

# ---- pooled panel ----------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
for r in range(ROWS):
    for c in range(COLS):
        h = hits[r, c] / len(runs)
        ed = float(np.mean(errs[r][c]))
        ax.add_patch(Rectangle((c * cw, r * ch), cw, ch, facecolor=GREENRED(h),
                               edgecolor="white", lw=1.6, zorder=1))
        ax.text(c * cw + cw / 2, r * ch + ch / 2, f"{100*h:.0f}%\n{ed:.1f}°",
                ha="center", va="center", fontsize=8.6,
                color=(INK if .25 < h < .85 else "white"), fontweight="bold", zorder=3)
ax.scatter(dots[:, 0], dots[:, 1], s=11, color=INK, alpha=.6, edgecolors="none", zorder=4)
frame(ax)
ax.set_title(f"3x3 pooled, 5 runs — {pooled_acc:.1f} % accurate", loc="left")

# ---- one small panel per run ----------------------------------------------
for k, run in enumerate(runs):
    a = fig.add_subplot(gs[0, k + 2])
    for r in range(ROWS):
        for c in range(COLS):
            a.add_patch(Rectangle((c * cw, r * ch), cw, ch,
                                  facecolor=GREENRED(float(per_run_hit[(k, r, c)])),
                                  edgecolor="white", lw=1.4, zorder=1))
    pts = np.array([(float(t["pred_x"]), float(t["pred_y"])) for t in run["rows"]])
    a.scatter(pts[:, 0], pts[:, 1], s=13, color=INK, alpha=.75, edgecolors="none", zorder=4)
    frame(a)
    a.set_title(f"run {k+1} — {run['acc']:.1f} %", loc="left", fontsize=9.5)

fig.text(0.0, -0.045,
         f"Figure 2 — Per-cell hit rate for the 3x3 grid. Left: the five runs pooled, so 5 trials per "
         f"cell; colour and top figure are the hit rate, below it the mean error in degrees of visual "
         f"angle. Right: the same map for each run on its own, where a cell is simply hit (green) or "
         f"missed (red). Everything is drawn to the 402 x 778 pt screen and black dots are the "
         f"{len(dots)} scored predictions. Pooled accuracy is {pooled_acc:.1f} % "
         f"({int(hits.sum())} of {ROWS*COLS*len(runs)}); per run it is "
         + ", ".join(f"{r['acc']:.1f} %" for r in runs) + ".",
         ha="left", va="top", fontsize=7.8, color=INK3, wrap=True, transform=fig.transFigure)

path = f"{HERE}/fig2_cell_maps_5_run_real.png"
fig.savefig(path); plt.close(fig)

for r in range(ROWS):
    print("  ".join(f"{hits[r,c]:.0f}/5 {np.mean(errs[r][c]):.2f}deg" for c in range(COLS)))
print(f"pooled {pooled_acc:.1f}%  n={len(dots)}")
print("wrote", path)
