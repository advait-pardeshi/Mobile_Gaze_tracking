"""
fig2 cell maps for the 5x4, 6x4 and 9x9 grids, built from the runs in
/Users/advait/Desktop/data_1/exp1 (10 runs per grid -> 10 trials per cell).
Same visual language as fig2_cell_maps_5_run.png: hit-rate colour + "hit%\nmean
error" per cell, black dots for every scored prediction, drawn to the
402 x 778 pt screen.
Output: fig2_cell_maps_5x4_6x4_9x9.png
"""
import os, csv, json, glob, re
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "/Users/advait/Desktop/data_1/exp1"
GRIDS = ["5x4", "6x4", "9x9"]
SW, SH = 402.0, 778.0

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


def run_dirs(grid):
    ds = glob.glob(f"{SRC}/run*_{grid}")
    return sorted(ds, key=lambda p: int(re.search(r"run(\d+)_", p).group(1)))


fig, axs = plt.subplots(1, 3, figsize=(11.5, 4.6), gridspec_kw={"wspace": 0.22})
summary = []

for ax, grid in zip(axs, GRIDS):
    dirs = run_dirs(grid)
    m0 = json.load(open(f"{dirs[0]}/meta.json"))
    rows_, cols_ = m0["grid"]["rows"], m0["grid"]["cols"]
    cw, ch = SW / cols_, SH / rows_

    hits = np.zeros((rows_, cols_))
    att = np.zeros((rows_, cols_))
    errs = [[[] for _ in range(cols_)] for _ in range(rows_)]
    dots, voids = [], 0

    for d in dirs:
        m = json.load(open(f"{d}/meta.json"))
        s = m["summary"]
        ppd = s["mean_error_pt"] / s["mean_error_deg"]      # points per degree
        for t in trial_rows(f"{d}/trials.csv"):
            r, c = int(t["row"]), int(t["col"])
            att[r, c] += 1
            hits[r, c] += int(t["hit"])
            if t["pred_x"].strip() == "":                   # logged trial with no frames
                voids += 1
                continue
            errs[r][c].append(float(t["err_pt"]) / ppd)
            dots.append((float(t["pred_x"]), float(t["pred_y"])))

    dots = np.array(dots)
    fs = float(np.clip(cw / SW * 26.0, 4.2, 8.6))           # label size follows cell width
    for r in range(rows_):
        for c in range(cols_):
            h = hits[r, c] / att[r, c]
            ed = float(np.mean(errs[r][c])) if errs[r][c] else np.nan
            ax.add_patch(Rectangle((c * cw, r * ch), cw, ch, facecolor=GREENRED(h),
                                   edgecolor="white", lw=1.6 if cols_ < 9 else 1.0, zorder=1))
            ax.text(c * cw + cw / 2, r * ch + ch / 2,
                    f"{100*h:.0f}%\n{ed:.1f}°" if np.isfinite(ed) else f"{100*h:.0f}%",
                    ha="center", va="center", fontsize=fs,
                    color=(INK if .25 < h < .85 else "white"), fontweight="bold", zorder=3)

    ax.scatter(dots[:, 0], dots[:, 1], s=11 if cols_ < 9 else 4.5, color=INK,
               alpha=.6 if cols_ < 9 else .45, edgecolors="none", zorder=4)
    ax.set_xlim(0, SW); ax.set_ylim(SH, 0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_color(LINE)

    acc = 100 * hits.sum() / att.sum()
    mean_err = float(np.mean([e for row in errs for cell in row for e in cell]))
    summary.append((grid, acc, int(att.sum()), len(dirs), mean_err, voids, len(dots)))
    ax.set_title(f"{grid} — {acc:.1f} % accurate", loc="left")

a5, a6, a9 = (s[1] for s in summary)
n_tot = sum(s[2] for s in summary)
n_dots = sum(s[6] for s in summary)
n_void = sum(s[5] for s in summary)
e5, e6, e9 = (s[4] for s in summary)

fig.text(0.0, -0.045,
         f"Figure 2 — Per-cell hit rate (colour and top figure, 10 trials per cell pooled across the "
         f"10 runs recorded for each grid) with mean error below it, drawn to the 402 × 778 pt screen. "
         f"Black dots are the {n_dots} scored predictions ({n_void} of the {n_tot} trials logged no "
         f"frames and are counted as misses but cannot be plotted). Accuracy collapses as the cells shrink: "
         f"{a5:.1f} % at 5×4, {a6:.1f} % at 6×4 and {a9:.1f} % at 9×9, where the 1.0° cell tolerance is "
         f"far below the {e9:.1f}° mean error, so not one of the 810 trials lands in its target cell. "
         f"Mean error itself barely moves ({e5:.1f}° / {e6:.1f}° / {e9:.1f}°) — the grids are not getting "
         f"harder to look at, the target is getting smaller than the tracker's error. "
         f"Green is 100 % hit, red is 0 %.",
         ha="left", va="top", fontsize=7.8, color=INK3, wrap=True, transform=fig.transFigure)

path = f"{HERE}/fig2_cell_maps_5x4_6x4_9x9.png"
fig.savefig(path); plt.close(fig)
for g, acc, n, nr, e, v, nd in summary:
    print(f"{g}: {acc:.1f}%  n={n} ({nr} runs)  dots={nd}  mean err {e:.2f} deg  voids={v}")
print("wrote", path)
