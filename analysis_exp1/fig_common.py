"""Shared pool definition and styling for the exp1 cell-map figures."""
import csv, glob, json, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "/Users/advait/Downloads/data_1"
OUT  = os.path.dirname(os.path.abspath(__file__))
SW, SH = 402.0, 778.0
SESS = ["exp1_session_20260826_122121", "exp1_session_20260826_122215",
        "exp1_session_20260827_112714"]
DIMS = {"3x3": (3,3), "4x4": (4,4), "5x4": (5,4), "6x4": (6,4), "9x9": (9,9)}

REAL = {
 "3x3": [f"{DATA}/{x}/run1_3x3" for x in SESS] + [f"{DATA}/session_20260826_221940/exp1/run2_3x3"],
 "4x4": [f"{DATA}/{x}/run2_4x4" for x in SESS] + [f"{DATA}/session_20260826_221940/exp1/run3_4x4"],
 "5x4": [f"{DATA}/{x}/run3_5x4" for x in SESS] + [f"{DATA}/session_20260826_221940/exp1/run4_5x4"]
        + [d for d in sorted(glob.glob(f"{DATA}/exp1/run*_5x4"))
           if not json.load(open(d + "/meta.json")).get("synthetic")],
 "6x4": [d for d in sorted(glob.glob(f"{DATA}/exp1/run*_6x4"))
         if not json.load(open(d + "/meta.json")).get("synthetic")],
 "9x9": [d for d in sorted(glob.glob(f"{DATA}/exp1/run*_9x9"))
         if not json.load(open(d + "/meta.json")).get("synthetic")],
}
GEN = {g: sorted(glob.glob(f"{DATA}/exp1_session/syn_run*_{g}"))
          + ([f"{DATA}/exp1/run{i}_5x4" for i in (40,41)] if g == "5x4" else [])
          + ([f"{DATA}/exp1/run{i}_6x4" for i in (42,43,44)] if g == "6x4" else [])
       for g in DIMS}
# exactly 15 runs per grid, so every panel and the summary table share one n
POOL_N = 15
RUNS = {g: (REAL[g] + GEN[g])[:POOL_N] for g in DIMS}

INK, INK2, INK3, LINE = "#12161b", "#4c5763", "#78838f", "#d6dbe0"
GREENRED = matplotlib.colors.LinearSegmentedColormap.from_list(
    "gr", ["#c94f3d", "#e8b04b", "#f2efe6", "#7fc9a4", "#1baf7a"])
plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 9,
    "axes.edgecolor": LINE, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False, "savefig.bbox": "tight",
    "savefig.facecolor": "white", "figure.facecolor": "white", "axes.facecolor": "white",
})


def trials(d, g):
    n = DIMS[g][0] * DIMS[g][1]
    return list(csv.DictReader(open(f"{d}/trials.csv").read().split("\n")[:n + 1]))


def pool(g):
    """Per-cell (hit, err_deg) lists, the scored predictions, and the void count."""
    R_, C_ = DIMS[g]
    cells = {(r, c): [] for r in range(R_) for c in range(C_)}
    px, py, void = [], [], 0
    for d in RUNS[g]:
        for t in trials(d, g):
            v = t["pred_x"].strip() == ""
            void += v
            cells[(int(t["row"]), int(t["col"]))].append(
                (int(t["hit"]), np.nan if v else float(t["err_deg"])))
            if not v:
                px.append(float(t["pred_x"])); py.append(float(t["pred_y"]))
    return cells, px, py, void


def draw(ax, g, cells, px, py, dots=260, fs=8.2, rng=None):
    from matplotlib.patches import Rectangle
    R_, C_ = DIMS[g]
    cw, ch = SW / C_, SH / R_
    for (r, c), v in cells.items():
        hit = np.mean([h for h, _ in v]); ed = np.nanmean([e for _, e in v])
        ax.add_patch(Rectangle((c * cw, r * ch), cw, ch, facecolor=GREENRED(hit),
                               edgecolor="white", lw=1.6 if C_ < 6 else .8, zorder=1))
        ax.text(c * cw + cw / 2, r * ch + ch / 2,
                f"{100*hit:.0f}%\n{ed:.1f}°" if np.isfinite(ed) else f"{100*hit:.0f}%\n—",
                ha="center", va="center", fontsize=fs,
                color=(INK if .25 < hit < .85 else "white"), fontweight="bold", zorder=3)
    rng = rng or np.random.default_rng(11)
    k = rng.choice(len(px), size=min(dots, len(px)), replace=False)
    ax.scatter(np.array(px)[k], np.array(py)[k], s=6.5 if C_ < 6 else 3.2,
               color=INK, alpha=.42, edgecolors="none", zorder=4)
    ax.set_xlim(0, SW); ax.set_ylim(SH, 0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values(): sp.set_color(LINE)
