"""Figure 8 — Experiment 3, combined results.

Left  : 4x3 position map — mean time to make a selection landing in each grid
        cell, pooled over every selection in every run. Word-to-cell assignment
        is reshuffled every session, so this is a position effect, not a word
        effect.
Right : every run — the sentence asked for, the sentence composed, the time;
        and a timeline of every individual selection with the word chosen.
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

def tidy(ax, xgrid=True, ygrid=False):
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.xaxis.grid(xgrid, color="#e6eaed", lw=0.7); ax.yaxis.grid(ygrid)
    ax.set_axisbelow(True)

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
R, C = 4, 3
N = R * C
cell_times = [[] for _ in range(N)]
RUNS = []
for run_no, s in enumerate(S, 1):
    lbl = f"Run {run_no}"
    sel, geom, ov = sect(os.path.join(s, "exp3/run1_communication/trials.csv"))
    ov = ov[0]
    picks = []
    for r in sel:
        k = int(r["cell_idx"])
        t = float(r["since_prev_s"])
        cell_times[k].append(t * 1000.0)
        picks.append({"word": r["word"], "ok": r["correct"] == "1",
                      "t": t, "at": float(r["at_s"])})
    RUNS.append({
        "lbl": lbl,
        "target": ov["target_sentence"],
        "composed": ov["composed_sentence"],
        "n": int(ov["n_selections"]),
        "ok": int(ov["correct"]),
        "acc": float(ov["selection_accuracy_pct"]),
        "dur": float(ov["duration_s"]),
        "per": float(ov["mean_s_per_selection"]),
        "wpm": float(ov["words_per_min"]),
        "done": ov["completed"] == "1",
        "picks": picks,
    })

grid = np.full((R, C), np.nan); gn = np.zeros((R, C), dtype=int)
for k in range(N):
    if cell_times[k]:
        grid[k // C, k % C] = float(np.mean(cell_times[k])); gn[k // C, k % C] = len(cell_times[k])
vmin, vmax = np.nanmin(grid), np.nanmax(grid)
n_sel = sum(len(t) for t in cell_times)

# ── figure ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(14.6, 6.9))
gs = fig.add_gridspec(2, 2, width_ratios=[0.70, 1.50], height_ratios=[1.0, 1.05],
                      wspace=0.13, hspace=0.42)

# A — position map
ax = fig.add_subplot(gs[:, 0])
im = ax.imshow(grid, cmap=CMAP, vmin=vmin, vmax=vmax)
mid = vmin + 0.55 * (vmax - vmin)
for r in range(R):
    for c in range(C):
        v = grid[r, c]
        if np.isnan(v):
            ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, facecolor="#f4f6f8",
                                       edgecolor="white", lw=2, hatch="///", zorder=2))
            ax.text(c, r - 0.06, "never selected", ha="center", va="center",
                    fontsize=8.2, color=INK3, zorder=3)
            ax.text(c, r + 0.14, f"row {r+1}, col {c+1}", ha="center", va="center",
                    fontsize=7, color=INK3, zorder=3)
        else:
            fg = "white" if v > mid else INK
            ax.text(c, r - 0.15, f"{v/1000:.2f} s", ha="center", va="center",
                    fontsize=15.5, fontweight="bold", color=fg, zorder=3)
            ax.text(c, r + 0.10, f"row {r+1}, col {c+1}", ha="center", va="center",
                    fontsize=7.4, color=fg, alpha=0.85, zorder=3)
            ax.text(c, r + 0.28, f"{gn[r, c]} selections", ha="center", va="center",
                    fontsize=6.9, color=fg, alpha=0.7, zorder=3)
for k in range(R + 1): ax.axhline(k - .5, color="white", lw=2, zorder=4)
for k in range(C + 1): ax.axvline(k - .5, color="white", lw=2, zorder=4)
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_title("A · Mean time to select, by position on the phone", loc="left", pad=12)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7.5, color=INK3)
cb.set_label("ms", fontsize=8, color=INK2)

# B — run table
ax = fig.add_subplot(gs[0, 1]); ax.axis("off")
X = [0.005, 0.115, 0.470, 0.585, 0.700, 0.840]
HD = ["Run", "Sentence the participant composed", "picks", "correct", "time", "s / pick"]
AL = ["left", "left", "right", "right", "right", "right"]
top, dy = 0.90, 0.132
for x, h, a in zip(X, HD, AL):
    ax.text(x if a == "left" else x + 0.10, top, h, fontsize=8.4, color=INK2,
            fontweight="bold", ha=a, va="center", transform=ax.transAxes)
ax.plot([0, 1], [top - 0.062] * 2, color=INK2, lw=1.1, transform=ax.transAxes, clip_on=False)
for i, r in enumerate(RUNS):
    y = top - 0.075 - dy * (i + 0.62)
    if i % 2 == 0:
        ax.add_patch(plt.Rectangle((-0.008, y - dy / 2 + 0.01), 1.016, dy - 0.014,
                                   facecolor="#f7f9fa", edgecolor="none",
                                   transform=ax.transAxes, zorder=0, clip_on=False))
    ax.text(X[0], y, r["lbl"], fontsize=8.1, color=INK2, va="center",
            transform=ax.transAxes, zorder=2)
    ax.text(X[1], y, r["composed"], fontsize=8.6, color=INK if r["done"] else S2,
            va="center", transform=ax.transAxes, zorder=2)
    for x, v in zip(X[2:], [f"{r['n']}", f"{r['ok']}", f"{r['dur']:.1f} s", f"{r['per']:.2f}"]):
        ax.text(x + 0.10, y, v, fontsize=8.3, ha="right", va="center",
                color=INK if "s" in v else INK2,
                fontweight="bold" if "s" in v and "." in v else "normal",
                transform=ax.transAxes, zorder=2)
yb = top - 0.075 - dy * (len(RUNS) + 0.62) + dy / 2
ax.plot([0, 1], [yb] * 2, color=LINE, lw=1, transform=ax.transAxes, clip_on=False)
tn = sum(r["n"] for r in RUNS); tok = sum(r["ok"] for r in RUNS)
td = sum(r["dur"] for r in RUNS)
ax.text(X[0], yb - 0.085, f"5 runs, target sentence “{RUNS[0]['target']}” every time",
        fontsize=8.4, color=INK, fontweight="bold", va="center", transform=ax.transAxes)
for x, v in zip(X[2:], [f"{tn}", f"{tok}", f"{td/len(RUNS):.1f} s", f"{td/tn:.2f}"]):
    ax.text(x + 0.10, yb - 0.085, v, fontsize=8.4, color=INK, fontweight="bold",
            ha="right", va="center", transform=ax.transAxes)
ax.set_title("B · Every sentence composed, and how long it took", loc="left", pad=10)

# C — every selection on a timeline
ax = fig.add_subplot(gs[1, 1]); tidy(ax)
for i, r in enumerate(RUNS):
    y = len(RUNS) - 1 - i
    ax.plot([0, r["dur"]], [y, y], color="#e3e8ec", lw=2.4, zorder=1,
            solid_capstyle="round")
    for p in r["picks"]:
        col = S1 if p["ok"] else S2
        ax.scatter([p["at"]], [y], s=54, color=col, zorder=3, edgecolor="white", lw=1.1)
        ax.text(p["at"], y + 0.20, p["word"], ha="center", va="bottom", fontsize=7.2,
                color=col, fontweight="bold", zorder=3)
        ax.text(p["at"], y - 0.20, f"{p['t']:.1f}", ha="center", va="top", fontsize=6.6,
                color=INK3, zorder=3)
ax.set_yticks(range(len(RUNS)))
ax.set_yticklabels([r["lbl"] for r in RUNS][::-1], fontsize=8)
ax.set_ylim(-0.95, len(RUNS) - 0.38)
ax.set_xlim(-0.6, max(r["dur"] for r in RUNS) + 1.2)
ax.set_xlabel("Seconds from the grid going live  ·  word above each mark, seconds it took below")
ax.scatter([], [], s=54, color=S1, label="advanced the sentence")
ax.scatter([], [], s=54, color=S2, label="wrong cell / repeat")
ax.legend(loc="lower right", ncol=1, borderpad=0.6, labelspacing=0.5,
          handletextpad=0.5)
ax.set_title("C · Every selection: what was chosen, when, how long", loc="left", pad=8)

cap(fig, f"Figure 8 — Experiment 3, all runs pooled. A: each of the twelve grid cells shows the mean time from the previous "
          f"selection (or from the grid going live, for the first pick) to a selection landing in that cell, over {n_sel} "
          f"selections; the word-to-cell assignment is reshuffled every session, so this is position, not vocabulary. Counts "
          f"differ per cell because the participant chose where to look. B: one row per run against the same target sentence, "
          f"orange where the sentence was not completed. C: every selection in time order; a mark is orange when it did not "
          f"advance the sentence. Dwell was 1.0 s throughout, and is inside every time shown.")
fig.savefig(f"{OUT}/fig8_exp3_combined.png"); plt.close(fig)

print(f"selections {n_sel}  runs {len(RUNS)}")
for k in range(N):
    if cell_times[k]:
        print(f"  row {k//C+1} col {k%C+1}  n={len(cell_times[k]):2d}  {np.mean(cell_times[k]):6.0f} ms")
    else:
        print(f"  row {k//C+1} col {k%C+1}  n= 0  never selected")
print(f"wrote {OUT}/fig8_exp3_combined.png")
