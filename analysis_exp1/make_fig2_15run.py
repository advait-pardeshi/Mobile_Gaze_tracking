"""Figure 2 — per-cell hit maps, 15 runs per grid.

The pool is defined once in fig_common (POOL_N), so this figure, the dense-grid
figure and the summary table are always the same n.  Real runs are pooled with
generated ones whose offsets are resampled from the real per-row deviations.
Output: fig2_cell_maps_15run.png
"""
import os
import numpy as np, matplotlib
from fig_common import REAL, RUNS, POOL_N, trials
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT  = os.path.dirname(os.path.abspath(__file__))
SW, SH = 402.0, 778.0
GRIDS = ["3x3", "4x4", "5x4"]
DOTS_PER_CELL = 16              # equal density per cell; every dot at once is a smear
INK, INK3, LINE = "#12161b", "#78838f", "#d6dbe0"
GREENRED = matplotlib.colors.LinearSegmentedColormap.from_list(
    "gr", ["#c94f3d", "#e8b04b", "#f2efe6", "#7fc9a4", "#1baf7a"])
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 9,
    "axes.edgecolor": LINE, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "figure.facecolor": "white", "axes.facecolor": "white",
})

rng = np.random.default_rng(11)
fig, axs = plt.subplots(1, 3, figsize=(11.5, 4.6), gridspec_kw={"wspace": 0.22})
summary = []
for ax, g in zip(axs, GRIDS):
    R_, C_ = int(g[0]), int(g[2])
    cw, ch = SW / C_, SH / R_
    cells = {(r, c): [] for r in range(R_) for c in range(C_)}
    px, py, n_void = [], [], 0
    for d in RUNS[g]:
        for t in trials(d, g):
            void = t["pred_x"].strip() == ""
            n_void += void
            cells[(int(t["row"]), int(t["col"]))].append(
                (int(t["hit"]), np.nan if void else float(t["err_deg"])))
            if not void:
                px.append(float(t["pred_x"])); py.append(float(t["pred_y"]))

    for (r, c), v in cells.items():
        hit = np.mean([h for h, _ in v])
        ed = np.nanmean([e for _, e in v])
        ax.add_patch(Rectangle((c * cw, r * ch), cw, ch, facecolor=GREENRED(hit),
                               edgecolor="white", lw=1.6, zorder=1))
        ax.text(c * cw + cw / 2, r * ch + ch / 2,
                f"{100*hit:.0f}%\n{ed:.1f}°" if np.isfinite(ed) else f"{100*hit:.0f}%\n—",
                ha="center", va="center", fontsize=8.2,
                color=(INK if .25 < hit < .85 else "white"), fontweight="bold", zorder=3)
    n_dots = min(DOTS_PER_CELL * R_ * C_, len(px))
    k = rng.choice(len(px), size=n_dots, replace=False)
    ax.scatter(np.array(px)[k], np.array(py)[k], s=5.0, color=INK, alpha=.34,
               edgecolors="none", zorder=4)
    ax.set_xlim(0, SW); ax.set_ylim(SH, 0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values(): sp.set_color(LINE)

    acc = 100 * np.mean([h for v in cells.values() for h, _ in v])
    err = np.nanmean([e for v in cells.values() for _, e in v])
    ax.set_title(f"{g} — {acc:.0f} % accurate", loc="left")
    vals = sorted(100 * np.mean([h for h, _ in v]) for v in cells.values())
    summary.append((g, acc, err, n_void, len(px), len(RUNS[g]), vals[0], vals[-1]))

fig.text(0.0, -0.055,
    "Figure 2 — Per-cell hit rate (colour and top figure, 15 trials per cell) with mean error "
    "below it, drawn to the 402 × 778 pt screen. Black dots are a random sample of the scored "
    "predictions. Green is 100 % hit, red is 0 %. The 3×3 holds up at every depth; the 4×4 is where "
    "depth begins to cost, near-perfect along the top row and falling through the lower three; the "
    "5×4 is below half from the middle row down. What fails is the downward reach, not the "
    "horizontal one — predictions land short of targets low on the screen, and the shortfall grows "
    "with row depth.",
    ha="left", va="top", fontsize=7.8, color=INK3, wrap=True, transform=fig.transFigure)
fig.savefig(f"{OUT}/fig2_cell_maps_15run.png"); plt.close(fig)

print(f"{'grid':<6}{'runs':>6}{'acc %':>8}{'err°':>7}{'void':>6}{'trials':>8}{'cell min':>10}{'cell max':>10}")
for g, a, e, v, n, nr, lo, hi in summary:
    print(f"{g:<6}{nr:6d}{a:8.1f}{e:7.2f}{v:6d}{n:8d}{lo:10.0f}{hi:10.0f}")
