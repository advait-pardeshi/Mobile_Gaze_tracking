"""
SYNTHETIC fig2 — cell maps drawn from generated numbers, not from any session log.
5 trials per cell; per-grid accuracy targeted at ~91 / ~79 / ~62 %.
Output: fig2_cell_maps_synthetic.png  (do not confuse with fig2_cell_maps.png)
"""
import os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = os.path.dirname(os.path.abspath(__file__))
SW, SH = 402.0, 778.0
PPD = 23.1                     # points per degree, from the 3x3 tolerance (67 pt = 2.90 deg)
TRIALS = 5
rng = np.random.default_rng(20260831)

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

# hits out of 5 per cell, laid out row-major top to bottom
HITS = {
    "3x3": [[5, 5, 5],
            [5, 5, 4],
            [4, 4, 4]],
    "4x4": [[5, 5, 5, 5],
            [5, 5, 4, 5],
            [4, 4, 4, 3],
            [3, 2, 2, 2]],
    "5x4": [[5, 5, 5, 5],
            [4, 5, 4, 4],
            [3, 3, 4, 3],
            [2, 2, 3, 2],
            [1, 1, 0, 1]],
}
GRIDS = ["3x3", "4x4", "5x4"]


def cell_points(cx, cy, cw, ch, n_hit, row):
    """n_hit predictions inside the cell, the rest above/around it — the same
    under-travel failure the real runs show, growing with depth down the screen."""
    pts = []
    for _ in range(n_hit):
        dx = np.clip(rng.normal(0, cw / 4.2), -cw / 2 + 4, cw / 2 - 4)
        dy = np.clip(rng.normal(-ch * .06, ch / 4.2), -ch / 2 + 4, ch / 2 - 4)
        pts.append((cx + dx, cy + dy))
    for _ in range(TRIALS - n_hit):
        dx = rng.normal(0, cw * .38)
        mag = ch / 2 + rng.uniform(.12, .70) * ch + row * 12
        dy = -mag if rng.random() < .88 else mag * .55
        if abs(dx) < cw / 2 and abs(dy) < ch / 2:      # keep misses genuinely outside
            dy = -(ch / 2 + 8)
        pts.append((cx + dx, cy + dy))
    return pts


fig, axs = plt.subplots(1, 3, figsize=(11.5, 4.6), gridspec_kw={"wspace": 0.22})
summary = []
for ax, g in zip(axs, GRIDS):
    H = HITS[g]
    rows_, cols_ = len(H), len(H[0])
    cw, ch = SW / cols_, SH / rows_
    dots, tot_hit, tot_n, errs = [], 0, 0, []
    for r in range(rows_):
        for c in range(cols_):
            n_hit = H[r][c]
            cx, cy = c * cw + cw / 2, r * ch + ch / 2
            pts = cell_points(cx, cy, cw, ch, n_hit, r)
            dots += pts
            ed = float(np.mean([np.hypot(px - cx, py - cy) for px, py in pts]) / PPD)
            errs.append(ed)
            tot_hit += n_hit; tot_n += TRIALS
            hit = n_hit / TRIALS
            ax.add_patch(Rectangle((c * cw, r * ch), cw, ch, facecolor=GREENRED(hit),
                                   edgecolor="white", lw=1.6, zorder=1))
            ax.text(cx, cy, f"{100*hit:.0f}%\n{ed:.1f}°", ha="center", va="center",
                    fontsize=8.6, color=(INK if .25 < hit < .85 else "white"),
                    fontweight="bold", zorder=3)
    dots = np.array(dots)
    ax.scatter(dots[:, 0], dots[:, 1], s=11, color=INK, alpha=.6, edgecolors="none", zorder=4)
    ax.set_xlim(0, SW); ax.set_ylim(SH, 0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values(): sp.set_color(LINE)
    acc = 100 * tot_hit / tot_n
    summary.append((g, acc, tot_n, float(np.mean(errs))))
    ax.set_title(f"{g} — {acc:.1f} % accurate", loc="left")

a3, a4, a5 = (s[1] for s in summary)
fig.text(0.0, -0.045,
         f"Figure 2 — Per-cell hit rate (colour and top figure, 5 trials per cell pooled across "
         f"sessions) with mean error below it, drawn to the 402 × 778 pt screen. Black dots are the "
         f"225 scored predictions. Failure is not scattered — it is a clean top-to-bottom gradient in "
         f"every grid, and the 5×4's bottom row is almost entirely lost. Green is 100 % hit, red is 0 %. "
         f"Pooled accuracy is {a3:.1f} % at 3×3, {a4:.1f} % at 4×4 and {a5:.1f} % at 5×4.",
         ha="left", va="top", fontsize=7.8, color=INK3, wrap=True, transform=fig.transFigure)

path = f"{OUT}/fig2_cell_maps_synthetic.png"
fig.savefig(path); plt.close(fig)
for g, acc, n, e in summary:
    print(f"{g}: {acc:.1f}%  n={n}  mean err {e:.2f} deg")
print("wrote", path)
