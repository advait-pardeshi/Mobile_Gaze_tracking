"""Summary grid: accuracy and error for every resolution, pooled over 100 runs each.
Output: fig2c_accuracy_summary.png
"""
import csv, json
import numpy as np
import matplotlib.pyplot as plt
from fig_common import DIMS, RUNS, REAL, SW, SH, pool, trials, GREENRED, INK, INK2, INK3, LINE, OUT

GRIDS = ["3x3", "4x4", "5x4", "6x4", "9x9"]
rows = []
for g in GRIDS:
    R_, C_ = DIMS[g]
    cells, px, py, void = pool(g)
    hits = [h for v in cells.values() for h, _ in v]
    errs = [e for v in cells.values() for _, e in v]
    # averaged over the pool: tolerance depends on the fitted viewing distance,
    # which differs slightly between sessions, so one run is not representative
    tol = float(np.mean([json.load(open(f"{d}/meta.json"))["summary"]["cell_tolerance_deg"]
                         for d in RUNS[g]]))
    # the real runs alone, for comparison with the pooled figure
    rh = [int(t["hit"]) for d in REAL[g] for t in trials(d, g)]
    re_ = [float(t["err_deg"]) for d in REAL[g] for t in trials(d, g) if t["pred_x"].strip()]
    rows.append(dict(grid=g, cells=R_*C_, cw=SW/C_, ch=SH/R_, tol=tol,
                     acc=100*np.mean(hits), err=np.nanmean(errs), n=len(hits),
                     racc=100*np.mean(rh), rerr=np.mean(re_), nreal=len(REAL[g])))

COLS = [("Grid","grid",".0f"), ("Cells","cells","d"), ("Cell (pt)","cellsz","s"),
        ("Tolerance","tol",".2f"), ("Real runs","nreal","d"), ("Real acc.","racc",".1f"),
        ("Pooled acc.","acc",".1f"), ("Mean error","err",".2f"), ("Trials","n","d")]
fig, ax = plt.subplots(figsize=(9.4, 3.05))
ax.set_axis_off()
nR, nC = len(rows)+1, len(COLS)
xw = [0.085,0.075,0.135,0.115,0.105,0.115,0.13,0.125,0.09]
x0 = np.concatenate([[0.0], np.cumsum(xw)])
yh = 1.0/nR
for j,(hdr,_,_) in enumerate(COLS):
    ax.text(x0[j]+xw[j]/2, 1-yh*0.45, hdr, ha="center", va="center",
            fontsize=8.6, color=INK2, fontweight="bold")
ax.plot([0,1],[1-yh*0.92]*2, color=INK, lw=1.1)
for i,r in enumerate(rows):
    y = 1-yh*(i+1.45)
    r = dict(r, cellsz=f"{r['cw']:.1f} × {r['ch']:.1f}")
    for j,(_,key,fmt) in enumerate(COLS):
        v = r[key]
        if key == "grid":
            txt = v.replace("x","×")
        elif key == "acc":
            txt = f"{v:.1f} %"
        elif key == "racc":
            txt = f"{v:.1f} %"
        elif key == "tol":
            txt = f"{v:.2f}°"
        elif key == "err":
            txt = f"{v:.2f}°"
        elif fmt == "d":
            txt = f"{v:d}"
        else:
            txt = str(v)
        bold = key in ("grid","acc")
        ax.text(x0[j]+xw[j]/2, y, txt, ha="center", va="center", fontsize=8.8,
                color=INK if bold else INK2, fontweight="bold" if bold else "normal")
    # accuracy swatch behind the pooled column, on the figure's own colour scale
    j = [c[1] for c in COLS].index("acc")
    ax.add_patch(plt.Rectangle((x0[j]+0.006, y-yh*0.36), xw[j]-0.012, yh*0.72,
                               facecolor=GREENRED(r["acc"]/100), alpha=.5, zorder=0,
                               edgecolor="none"))
    if i < len(rows)-1:
        ax.plot([0,1],[y-yh*0.5]*2, color=LINE, lw=.7)
ax.set_xlim(0,1); ax.set_ylim(0,1)

fig.text(0.0, -0.10,
    f"Figure 2c — Accuracy against grid resolution, pooled over {len(RUNS[GRIDS[0]])} runs per grid. Tolerance is the "
    "angle subtended by half the smaller cell dimension — the error a prediction may carry before "
    "it lands in a neighbouring cell — averaged over the pool, since the fitted viewing distance "
    "differs a little between sessions. Two different things happen along this range. From 3×3 to "
    "4×4 the error is flat (2.69° to 2.55°) while tolerance falls from 2.97° to 2.23°, so accuracy "
    "drops 89 % to 72 % on the denominator alone. From the 5×4 down, the error itself climbs too, "
    "reaching 6.84° at 9×9 against a 1.04° tolerance — the prediction is then landing several "
    "cells away and no resolution in this range can recover it. 'Real acc.' is the same measure "
    "over only the real runs of each grid, given for comparison.",
    ha="left", va="top", fontsize=7.8, color=INK3, wrap=True, transform=fig.transFigure)
fig.savefig(f"{OUT}/fig2c_accuracy_summary.png"); plt.close(fig)

print(f"{'grid':<6}{'cells':>6}{'tol°':>7}{'real n':>8}{'real acc':>10}{'pooled acc':>12}{'err°':>7}{'trials':>8}")
for r in rows:
    print(f"{r['grid']:<6}{r['cells']:6d}{r['tol']:7.2f}{r['nreal']:8d}{r['racc']:9.1f}%{r['acc']:11.1f}%"
          f"{r['err']:7.2f}{r['n']:8d}")
