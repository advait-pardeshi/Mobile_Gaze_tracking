"""Figure 7 — Experiment 4, combined results.

Left  : 3x3 position map — mean time to make a selection landing in each grid
        cell, pooled over every selection in every run.
Right : every trial — the question asked, the answer the participant composed,
        and how long it took.
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
    "xtick.color": INK3, "ytick.color": INK3,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.frameon": False, "legend.fontsize": 8.5,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
    "figure.facecolor": "white", "axes.facecolor": "white",
})

def cap(fig, text):
    fig.text(0.0, -0.02, text, ha="left", va="top", fontsize=7.8,
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

# ── collect ────────────────────────────────────────────────────
N = 9
cell_times = [[] for _ in range(N)]     # every selection landing in cell k, ms
KIND, ROWS = {}, []
for s in S:
    blocks = sect(os.path.join(s, "exp4/run1_predictive_cued/trials.csv"))
    sel, geom, trials = blocks[0], blocks[2], blocks[3]
    for g in geom:
        KIND[int(g["cell_idx"])] = g["kind_at_start"]
    for r in sel:
        cell_times[int(r["cell_idx"])].append(float(r["since_prev_s"]) * 1000.0)
    for t in trials:
        ROWS.append({
            "q": t["question_text"],
            "a": t["composed"],
            "n": int(t["word_selections"]),
            "t": float(t["response_s"]),
        })
for r in ROWS:
    r["per"] = r["t"] / r["n"] if r["n"] else float("nan")

grid = np.full((3, 3), np.nan); gn = np.zeros((3, 3), dtype=int)
for k in range(N):
    if cell_times[k]:
        grid[k // 3, k % 3] = float(np.mean(cell_times[k])); gn[k // 3, k % 3] = len(cell_times[k])
vmin, vmax = np.nanmin(grid), np.nanmax(grid)
FN = {"word": "", "back": "BACK", "done": "DONE"}

# ── figure ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(14.2, 5.7))
gs = fig.add_gridspec(1, 2, width_ratios=[0.78, 1.42], wspace=0.12)

# A — position map
ax = fig.add_subplot(gs[0])
im = ax.imshow(grid, cmap=CMAP, vmin=vmin, vmax=vmax)
mid = vmin + 0.55 * (vmax - vmin)
for r in range(3):
    for c in range(3):
        k = r * 3 + c; v = grid[r, c]
        pos = f"row {r+1}, col {c+1}"
        if np.isnan(v):
            ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, facecolor="#f4f6f8",
                                       edgecolor="white", lw=2, hatch="///", zorder=2))
            ax.text(c, r - 0.12, "never selected", ha="center", va="center",
                    fontsize=8.5, color=INK3, zorder=3)
            ax.text(c, r + 0.08, pos, ha="center", va="center", fontsize=7, color=INK3, zorder=3)
            ax.text(c, r + 0.26, FN.get(KIND.get(k, ""), ""), ha="center", va="center",
                    fontsize=7.4, color=S2, fontweight="bold", zorder=3)
        else:
            fg = "white" if v > mid else INK
            ax.text(c, r - 0.17, f"{v/1000:.2f} s", ha="center", va="center",
                    fontsize=17, fontweight="bold", color=fg, zorder=3)
            ax.text(c, r + 0.06, pos, ha="center", va="center", fontsize=7.6,
                    color=fg, alpha=0.85, zorder=3)
            ax.text(c, r + 0.22, f"{gn[r, c]} selections", ha="center", va="center",
                    fontsize=7, color=fg, alpha=0.7, zorder=3)
            f = FN.get(KIND.get(k, ""), "")
            if f:
                ax.text(c, r + 0.38, f, ha="center", va="center", fontsize=7.6,
                        color=fg, fontweight="bold", zorder=3)
for k in range(4):
    ax.axhline(k - .5, color="white", lw=2, zorder=4)
    ax.axvline(k - .5, color="white", lw=2, zorder=4)
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_title("A · Mean time to select, by position on the phone", loc="left", pad=12)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7.5, color=INK3)
cb.set_label("ms", fontsize=8, color=INK2)

# B — trial table
ax = fig.add_subplot(gs[1]); ax.axis("off")
X = [0.005, 0.300, 0.735, 0.815, 0.925]
HD = ["Question", "Answer the participant composed", "words", "time", "s / word"]
AL = ["left", "left", "right", "right", "right"]
top, dy = 0.965, 0.0545
for x, h, a in zip(X, HD, AL):
    ax.text(x if a == "left" else x + 0.075, top, h, fontsize=8.4, color=INK2,
            fontweight="bold", ha=a, va="center", transform=ax.transAxes)
ax.plot([0, 1], [top - 0.026] * 2, color=INK2, lw=1.1, transform=ax.transAxes, clip_on=False)

for i, r in enumerate(ROWS):
    y = top - 0.032 - dy * (i + 0.62)
    if i % 2 == 0:
        ax.add_patch(plt.Rectangle((-0.008, y - dy / 2 + 0.004), 1.016, dy - 0.006,
                                   facecolor="#f7f9fa", edgecolor="none",
                                   transform=ax.transAxes, zorder=0, clip_on=False))
    ax.text(X[0], y, r["q"], fontsize=8.1, color=INK2,
            va="center", transform=ax.transAxes, zorder=2)
    ax.text(X[1], y, r["a"], fontsize=8.6, color=INK, va="center",
            fontweight="medium", transform=ax.transAxes, zorder=2)
    ax.text(X[2] + 0.075, y, f"{r['n']}", fontsize=8.3, color=INK2, ha="right",
            va="center", transform=ax.transAxes, zorder=2)
    ax.text(X[3] + 0.075, y, f"{r['t']:.1f} s", fontsize=8.3, color=INK, ha="right",
            va="center", fontweight="bold", transform=ax.transAxes, zorder=2)
    ax.text(X[4] + 0.075, y, f"{r['per']:.2f}", fontsize=8.3, color=INK2, ha="right",
            va="center", transform=ax.transAxes, zorder=2)

yb = top - 0.032 - dy * (len(ROWS) + 0.62) + dy / 2
ax.plot([0, 1], [yb] * 2, color=LINE, lw=1, transform=ax.transAxes, clip_on=False)
tot_n = sum(r["n"] for r in ROWS); tot_t = sum(r["t"] for r in ROWS)
ax.text(X[0], yb - 0.035, f"{len(ROWS)} answers over {len(S)} sessions",
        fontsize=8.4, color=INK, fontweight="bold", va="center", transform=ax.transAxes)
ax.text(X[2] + 0.075, yb - 0.035, f"{tot_n}", fontsize=8.4, color=INK,
        fontweight="bold", ha="right", va="center", transform=ax.transAxes)
ax.text(X[3] + 0.075, yb - 0.035, f"{tot_t/len(ROWS):.1f} s", fontsize=8.4, color=INK,
        fontweight="bold", ha="right", va="center", transform=ax.transAxes)
ax.text(X[4] + 0.075, yb - 0.035, f"{tot_t/tot_n:.2f}", fontsize=8.4, color=INK,
        fontweight="bold", ha="right", va="center", transform=ax.transAxes)
ax.set_title("B · Every answer given: question, what was composed, and how long it took",
             loc="left", pad=12)

cap(fig, f"Figure 7 — Experiment 4, all runs pooled. A: each of the nine grid cells shows the mean time from the previous "
          f"selection (or from the grid going live, for the first pick of an answer) to a selection landing in that cell, over "
          f"{sum(gn.flat)} selections; counts differ per cell because the participant chose where to look. The bottom-left cell "
          f"was never used. B: one row per answer — mean {tot_t/len(ROWS):.1f} s and {tot_n/len(ROWS):.1f} words per answer, "
          f"{tot_t/tot_n:.2f} s per word, zero corrections in any trial. Word dwell 1.0 s, control dwell 1.5 s, plus a 0.4 s "
          f"lockout after each grid refresh; all three are inside every time shown.")
fig.savefig(f"{OUT}/fig7_exp4_combined.png"); plt.close(fig)

print(f"selections {sum(gn.flat)}  answers {len(ROWS)}  words {tot_n}")
for k in range(N):
    if cell_times[k]:
        print(f"  row {k//3+1} col {k%3+1} ({KIND.get(k,'?'):4s}) n={len(cell_times[k]):2d}  {np.mean(cell_times[k]):6.0f} ms")
print(f"wrote {OUT}/fig7_exp4_combined.png")
