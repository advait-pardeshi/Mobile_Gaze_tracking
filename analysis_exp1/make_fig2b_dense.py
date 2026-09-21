"""Cell maps for the two dense grids, 6x4 and 9x9, pooled over 15 runs each.
Output: fig2b_cell_maps_dense.png
"""
import numpy as np
import matplotlib.pyplot as plt
from fig_common import DIMS, RUNS, pool, draw, INK3, OUT

fig, axs = plt.subplots(1, 2, figsize=(8.6, 5.0), gridspec_kw={"wspace": 0.20})
rng = np.random.default_rng(7)
info = []
for ax, g in zip(axs, ["6x4", "9x9"]):
    cells, px, py, void = pool(g)
    draw(ax, g, cells, px, py, dots=300, fs=8.2 if g == "6x4" else 3.9, rng=rng)
    acc = 100 * np.mean([h for v in cells.values() for h, _ in v])
    err = np.nanmean([e for v in cells.values() for _, e in v])
    ax.set_title(f"{g} — {acc:.0f} % accurate" + (f"  ({void} void)" if void else ""), loc="left")
    vals = sorted(100 * np.mean([h for h, _ in v]) for v in cells.values())
    info.append((g, acc, err, vals[0], vals[-1], len(px)))

fig.text(0.0, -0.06,
    "Figure 2b — The two dense grids, same construction as Figure 2: per-cell hit rate in colour "
    "and the top figure, mean error below it, 15 trials per cell, drawn to the 402 × 778 pt "
    "screen. Green is 100 % hit, red is 0 %. Both have lost the depth gradient the coarser grids "
    "show. The 6×4 is uniformly poor rather than poor-at-the-bottom — every cell falls between 8 "
    "and 33 %, the second row is the best of them and the third the worst, and no cell anywhere "
    "reaches half. The 9×9 is not a further step down that curve but a floor: its 44.7 × 86.4 pt "
    "cell is smaller than the error itself, and no cell on the screen registers above 1 %. Past "
    "the 5×4 the grid stops trading accuracy for resolution and simply stops working.",
    ha="left", va="top", fontsize=7.8, color=INK3, wrap=True, transform=fig.transFigure)
fig.savefig(f"{OUT}/fig2b_cell_maps_dense.png"); plt.close(fig)

print(f"{'grid':<6}{'acc %':>8}{'err°':>8}{'cell min':>10}{'cell max':>10}{'trials':>9}")
for g, a, e, lo, hi, n in info:
    print(f"{g:<6}{a:8.2f}{e:8.2f}{lo:10.0f}{hi:10.0f}{n:9d}")
