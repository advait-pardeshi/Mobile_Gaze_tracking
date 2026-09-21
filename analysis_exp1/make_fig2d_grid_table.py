"""Figure 2d — the plain resolution table: grid, cell size, trials, accuracy.
No tolerance, no mean error. 15 runs per grid; the caption discloses the real/resampled split.
Output: fig2d_grid_table.png
"""
import numpy as np
import matplotlib.pyplot as plt
from fig_common import DIMS, REAL, GEN, SW, SH, trials, GREENRED, INK, INK2, INK3, LINE, OUT

GRIDS = ["3x3", "4x4", "5x4", "6x4", "9x9"]
RUNS_PER_GRID = 15              # equal n per grid, so the accuracy column is comparable

rows = []
for g in GRIDS:
    R_, C_ = DIMS[g]
    runs = (REAL[g] + GEN[g])[:RUNS_PER_GRID]
    assert len(runs) == RUNS_PER_GRID, f"{g}: only {len(runs)} runs available"
    hits = [int(t["hit"]) for d in runs for t in trials(d, g)]
    rows.append(dict(grid=g.replace("x", "×"), cells=R_ * C_,
                     cellsz=f"{SW/C_:.1f} × {SH/R_:.1f}", runs=len(runs),
                     n=len(hits), acc=100 * np.mean(hits),
                     nreal=min(len(REAL[g]), RUNS_PER_GRID),
                     racc=100 * np.mean([int(t["hit"]) for d in REAL[g]
                                         for t in trials(d, g)])))

COLS = [("Grid", "grid"), ("Cells", "cells"), ("Cell (pt)", "cellsz"),
        ("Runs", "runs"), ("Trials", "n"), ("Accuracy", "acc")]
xw = [0.13, 0.12, 0.24, 0.13, 0.16, 0.22]
x0 = np.concatenate([[0.0], np.cumsum(xw)])

fig, ax = plt.subplots(figsize=(7.2, 2.9))
ax.set_axis_off()
yh = 1.0 / (len(rows) + 1)
for j, (hdr, _) in enumerate(COLS):
    ax.text(x0[j] + xw[j] / 2, 1 - yh * 0.45, hdr, ha="center", va="center",
            fontsize=9.2, color=INK2, fontweight="bold")
ax.plot([0, 1], [1 - yh * 0.92] * 2, color=INK, lw=1.1)

for i, r in enumerate(rows):
    y = 1 - yh * (i + 1.45)
    for j, (_, key) in enumerate(COLS):
        v = r[key]
        txt = f"{v:.1f} %" if key == "acc" else (f"{v:d}" if isinstance(v, int) else str(v))
        bold = key in ("grid", "acc")
        ax.text(x0[j] + xw[j] / 2, y, txt, ha="center", va="center", fontsize=9.4,
                color=INK if bold else INK2, fontweight="bold" if bold else "normal")
    j = [c[1] for c in COLS].index("acc")
    ax.add_patch(plt.Rectangle((x0[j] + 0.012, y - yh * 0.36), xw[j] - 0.024, yh * 0.72,
                               facecolor=GREENRED(r["acc"] / 100), alpha=.5,
                               zorder=0, edgecolor="none"))
    if i < len(rows) - 1:
        ax.plot([0, 1], [y - yh * 0.5] * 2, color=LINE, lw=.7)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

fig.text(0.0, -0.06,
    "Figure 2d — Grid resolution against accuracy, 15 runs per grid on the 402 × 778 pt screen. "
    "One run visits every cell once, so trials = runs × cells. Accuracy is the share of trials "
    "whose prediction fell inside the target cell; trials that produced no prediction are counted "
    "as misses. Accuracy falls monotonically with resolution and reaches zero at the 9×9. Of the "
    "15 runs per grid, " + ", ".join(f"{r['nreal']} ({r['grid']})" for r in rows) + " are recorded "
    "sessions; the remainder are resampled from the recorded per-row deviations of the same grid. "
    "Over the recorded runs alone the accuracies are "
    + ", ".join(f"{r['racc']:.1f} %" for r in rows) + " respectively.",
    ha="left", va="top", fontsize=7.8, color=INK3, wrap=True, transform=fig.transFigure)
fig.savefig(f"{OUT}/fig2d_grid_table.png"); plt.close(fig)

for r in rows:
    print(f"{r['grid']:<6}{r['cells']:4d} cells{r['runs']:5d} runs ({r['nreal']} real)"
          f"{r['n']:7d} trials{r['acc']:8.1f}%  real-only {r['racc']:5.1f}%")
