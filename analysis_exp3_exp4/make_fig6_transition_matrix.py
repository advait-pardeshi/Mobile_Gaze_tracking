"""Figure 6 — Experiment 4 cell-to-cell transition-time matrix.

M_cell x M_cell: cell (i, j) is the mean time (ms) to produce a selection in
grid cell i given that the previous selection in the same trial was cell j.
Built from the per-selection `since_prev_s` column of every exp4 run.
"""
import csv, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/advait/Downloads/EXP 3"
OUT  = "/Users/advait/Desktop/GazeTracking/Mobile_Gaze_tracking-main/analysis_exp3_exp4"
os.makedirs(OUT, exist_ok=True)
S = sorted(glob.glob(os.path.join(ROOT, "session_*")))

# ── house style (matches make_figures.py) ──────────────────────
INK, INK2, INK3 = "#12161b", "#4c5763", "#78838f"
LINE = "#d6dbe0"
S1, S2 = "#2a78d6", "#eb6834"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("b", RAMP)

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": LINE, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9.5,
    "axes.grid": True, "grid.color": "#e6eaed", "grid.linewidth": 0.7,
    "xtick.color": INK3, "ytick.color": INK3,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.frameon": False, "legend.fontsize": 8.5,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
    "figure.facecolor": "white", "axes.facecolor": "white",
})

def tidy(ax, xgrid=False, ygrid=True):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(LINE); ax.spines["bottom"].set_color(LINE)
    ax.xaxis.grid(xgrid); ax.yaxis.grid(ygrid)
    ax.set_axisbelow(True)

def cap(fig, text):
    fig.text(0.0, -0.045, text, ha="left", va="top", fontsize=7.8,
             color=INK3, wrap=True, transform=fig.transFigure)

def sect(p):
    blocks, cur = [], None
    for line in open(p):
        line = line.rstrip("\n")
        if not line.strip() or line.startswith("#"):
            cur = None; continue
        if cur is None:
            cur = {"h": line.split(","), "r": []}; blocks.append(cur)
        else:
            cur["r"].append(next(csv.reader([line])))
    return [[dict(zip(b["h"], r)) for r in b["r"]] for b in blocks]

# ── collect transitions ────────────────────────────────────────
N = 9
times = [[[] for _ in range(N)] for _ in range(N)]   # times[i][j] = list of ms
dest_times = [[] for _ in range(N)]                  # marginal by destination
KIND = {}
for s in S:
    blocks = sect(os.path.join(s, "exp4/run1_predictive_cued/trials.csv"))
    sel, geom = blocks[0], blocks[2]
    for g in geom:
        KIND[int(g["cell_idx"])] = g["kind_at_start"]
    prev_cell, prev_trial = None, None
    for r in sel:
        i, tr = int(r["cell_idx"]), r["trial"]
        ms = float(r["since_prev_s"]) * 1000.0
        if prev_cell is not None and prev_trial == tr:
            times[i][prev_cell].append(ms)
            dest_times[i].append(ms)
        prev_cell, prev_trial = i, tr

M = np.full((N, N), np.nan)
C = np.zeros((N, N), dtype=int)
for i in range(N):
    for j in range(N):
        if times[i][j]:
            M[i, j] = float(np.mean(times[i][j])); C[i, j] = len(times[i][j])

n_trans = C.sum()
vmin, vmax = np.nanmin(M), np.nanmax(M)
LAB = [f"{k+1}\n({k//3},{k%3})" for k in range(N)]
SHORT = {"word": "word", "back": "BACK", "done": "DONE"}

# ── figure ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(12.6, 5.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 0.62, 0.72], wspace=0.42)

# A — the matrix
ax = fig.add_subplot(gs[0])
im = ax.imshow(M, cmap=CMAP, vmin=vmin, vmax=vmax)
ax.set_xticks(range(N)); ax.set_yticks(range(N))
ax.set_xticklabels([f"{k+1}" for k in range(N)])
ax.set_yticklabels([f"{k+1}" for k in range(N)])
ax.set_xlabel("Previous cell  $j$", labelpad=16)
ax.set_ylabel("Selected cell  $i$", labelpad=8)
mid = vmin + 0.55 * (vmax - vmin)
for i in range(N):
    for j in range(N):
        if np.isnan(M[i, j]):
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#f4f6f8",
                                       edgecolor="white", lw=1.2, hatch="///",
                                       zorder=2))
            ax.text(j, i, "–", ha="center", va="center", fontsize=9,
                    color="#c3cad1", zorder=3)
        else:
            fg = "white" if M[i, j] > mid else INK
            ax.text(j, i - 0.13, f"{M[i, j]:.0f}", ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=fg, zorder=3)
            ax.text(j, i + 0.24, f"n={C[i, j]}", ha="center", va="center",
                    fontsize=6.6, color=fg, alpha=0.75, zorder=3)
for k in range(N + 1):
    ax.axhline(k - .5, color="white", lw=1.2, zorder=4)
    ax.axvline(k - .5, color="white", lw=1.2, zorder=4)
# mark the fixed-function cells
import matplotlib.transforms as mtransforms
xtr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
for k, kd in KIND.items():
    if kd != "word":
        ax.text(-0.92, k, SHORT[kd], ha="right", va="center", fontsize=6.8,
                color=S2, fontweight="bold")
        ax.text(k, -0.055, SHORT[kd], ha="center", va="top", fontsize=6.8,
                color=S2, fontweight="bold", transform=xtr, clip_on=False)
ax.grid(False)
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_title("A · Mean time to selection (ms), by previous cell", loc="left", pad=10)
cb = fig.colorbar(im, ax=ax, fraction=0.042, pad=0.025)
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7.5, color=INK3)
cb.set_label("ms", fontsize=8, color=INK2)

# B — destination marginal on the physical 3x3 grid
ax = fig.add_subplot(gs[1])
gridv = np.full((3, 3), np.nan)
gridn = np.zeros((3, 3), dtype=int)
for k in range(N):
    if dest_times[k]:
        gridv[k // 3, k % 3] = float(np.mean(dest_times[k]))
        gridn[k // 3, k % 3] = len(dest_times[k])
ax.imshow(gridv, cmap=CMAP, vmin=vmin, vmax=vmax)
gmid = vmin + 0.55 * (vmax - vmin)
for r in range(3):
    for c in range(3):
        k = r * 3 + c
        v = gridv[r, c]
        if np.isnan(v):
            ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, facecolor="#f4f6f8",
                                       edgecolor="white", lw=1.4, hatch="///", zorder=2))
            ax.text(c, r, "never\nselected", ha="center", va="center", fontsize=6.8,
                    color=INK3, zorder=3)
        else:
            fg = "white" if v > gmid else INK
            ax.text(c, r - 0.16, f"{v:.0f}", ha="center", va="center",
                    fontsize=11.5, fontweight="bold", color=fg, zorder=3)
            ax.text(c, r + 0.2, f"cell {k+1} · n={gridn[r, c]}", ha="center", va="center",
                    fontsize=6.6, color=fg, alpha=0.8, zorder=3)
for k in range(4):
    ax.axhline(k - .5, color="white", lw=1.4, zorder=4)
    ax.axvline(k - .5, color="white", lw=1.4, zorder=4)
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_title("B · Row marginal, on the\n     on-screen grid (ms)", loc="left", pad=10, y=1.0)

# C — origin marginal: how long the *next* pick takes after leaving cell j
ax = fig.add_subplot(gs[2]); tidy(ax, xgrid=True, ygrid=False)
om, on_, oy = [], [], []
for j in range(N):
    col = [t for i in range(N) for t in times[i][j]]
    if col:
        om.append(float(np.mean(col))); on_.append(len(col)); oy.append(j)
order = np.argsort(om)
om = [om[o] for o in order]; on_ = [on_[o] for o in order]; oy = [oy[o] for o in order]
ypos = range(len(om))
cols = [S2 if KIND.get(j) != "word" else S1 for j in oy]
ax.barh(list(ypos), om, color=cols, height=0.6, zorder=3)
for y, v, nn in zip(ypos, om, on_):
    ax.text(v + 60, y, f"{v:.0f}  (n={nn})", va="center", fontsize=7.6, color=INK)
ax.set_yticks(list(ypos))
ax.set_yticklabels([f"from cell {j+1}" + ("" if KIND.get(j) == "word" else f" · {SHORT[KIND[j]]}")
                    for j in oy], fontsize=8)
ax.set_xlabel("Mean time of the following selection (ms)")
ax.set_xlim(0, max(om) * 1.34)
ax.set_title("C · Column marginal", loc="left")

cap(fig, f"Figure 6 — Experiment 4 cell-to-cell transition times. Cell (i, j) is the mean wall-clock time from the previous "
          f"selection to a selection landing in grid cell i, given the previous selection was cell j; {n_trans} within-trial "
          f"transitions over 5 sessions x 3 cued trials populate {(C>0).sum()} of the 81 pairs, so hatched cells are unobserved, "
          f"not zero, and single-observation cells (n=1) carry no error bar. Cell 7 (BACK) was never selected and cell 9 (DONE) "
          f"ends a trial, so their row/column are structurally sparse. Word dwell was 1.0 s and control dwell 1.5 s [Table T-params], "
          f"so every value here is dwell plus search, saccade and refresh-lockout time; the 0.4 s post-refresh lockout is included.")
fig.savefig(f"{OUT}/fig6_exp4_transition_matrix.png"); plt.close(fig)

# ── console summary ────────────────────────────────────────────
print(f"transitions            : {n_trans}")
print(f"pairs populated        : {(C>0).sum()}/81")
print(f"range (ms)             : {vmin:.0f} – {vmax:.0f}")
allt = [t for i in range(N) for j in range(N) for t in times[i][j]]
print(f"grand mean / median ms : {np.mean(allt):.0f} / {np.median(allt):.0f}")
for k in range(N):
    if dest_times[k]:
        print(f"  -> cell {k+1} ({KIND.get(k,'?'):4s}) n={len(dest_times[k]):2d}  mean {np.mean(dest_times[k]):6.0f} ms")
print(f"wrote {OUT}/fig6_exp4_transition_matrix.png")
