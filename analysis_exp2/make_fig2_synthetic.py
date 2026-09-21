"""
SYNTHETIC fig2 — the exp2 per-cell maps drawn from the generated runs in
synthetic_runs/ (make_synthetic_runs.py), not from a recorded session.
Output: fig2_cell_maps_synthetic.png (do not confuse with fig2_cell_maps.png).
"""
import csv, glob, os, statistics as st
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
RUNS = sorted(glob.glob(f"{OUT}/synthetic_runs/syn_run*_4x4"))
NAMES = [f"Run {i+1}" for i in range(len(RUNS))]

INK, INK3, LINE = "#12161b", "#78838f", "#d6dbe0"
BLUES = matplotlib.colors.LinearSegmentedColormap.from_list(
    "b", ["#cde2fb","#9ec5f4","#6da7ec","#3987e5","#256abf","#184f95"])
ORANGE = matplotlib.colors.LinearSegmentedColormap.from_list(
    "o", ["#fde5d8","#fbc3a8","#f79f76","#f07a48","#dd5a24","#b34617"])
plt.rcParams.update({
    "figure.dpi":200,"savefig.dpi":200,"font.family":"sans-serif",
    "font.sans-serif":["Helvetica Neue","Helvetica","Arial","DejaVu Sans"],"font.size":9,
    "axes.edgecolor":LINE,"axes.titlecolor":INK,"axes.titlesize":11,
    "axes.titleweight":"bold","legend.frameon":False,"savefig.bbox":"tight",
    "savefig.facecolor":"white","figure.facecolor":"white","axes.facecolor":"white",
})


def sect(p):
    b, cur = [], None
    for line in open(p):
        line = line.rstrip("\n")
        if not line.strip() or line.startswith("#"): cur = None; continue
        if cur is None: cur = {"h": line.split(","), "r": []}; b.append(cur)
        else: cur["r"].append(next(csv.reader([line])))
    return [[dict(zip(x["h"], r)) for r in x["r"]] for x in b]


CELLS = {n: sect(f"{p}/trials.csv")[0] for n, p in zip(NAMES, RUNS)}


def cellstat(key, fn=st.mean):
    g = np.zeros((4, 4))
    for c in range(16):
        vals = [float(r[key]) for n in NAMES for r in CELLS[n] if int(r["cell_idx"]) == c]
        r0 = [r for r in CELLS[NAMES[0]] if int(r["cell_idx"]) == c][0]
        g[int(r0["row"]), int(r0["col"])] = fn(vals)
    return g


DEV, RMS, CON = cellstat("mean_dev_deg"), cellstat("rms_deg"), cellstat("containment_pct")

fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.0), gridspec_kw={"wspace":0.30})
panels = [
    (DEV, BLUES, "A · Deviation (accuracy), °", "{:.2f}", None, None, 3.0),
    (RMS, BLUES, "B · RMS scatter (precision), °", "{:.2f}", 0.8, 2.0, 1.6),
    (CON, ORANGE.reversed(), "C · Containment, %", "{:.0f}", None, None, 70),
]
for ax, (G, cm, title, fmt, vmn, vmx, thr) in zip(axs, panels):
    im = ax.imshow(G, cmap=cm, vmin=vmn, vmax=vmx, aspect=0.85)
    for i in range(4):
        for j in range(4):
            hot = (G[i, j] > thr) if title[0] != "C" else (G[i, j] < thr)
            lab = fmt.format(G[i, j]) + ("*" if (vmx is not None and G[i, j] > vmx) else "")
            ax.text(j, i, lab, ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white" if hot else INK)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels([f"col {k}" for k in range(4)], fontsize=7.5)
    ax.set_yticklabels([f"row {k}" for k in range(4)], fontsize=7.5)
    ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(title, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7.5, color=INK3)

rm = DEV.mean(1); rr = RMS.mean(1); rc = CON.mean(1)
fig.text(0.0, -0.045,
    f"Figure 2 (SYNTHETIC) — Per-cell maps pooled over {len(RUNS)} generated runs "
    f"({16*len(RUNS)} cell-runs). All three panels are read off the same simulated "
    f"sample clouds, so they agree cell by cell: deviation rises, scatter rises a "
    f"little, containment falls. Accuracy holds at "
    f"{DEV[:3].min():.2f}–{DEV[:3].max():.2f}° over rows 0–2 and collapses to "
    f"{DEV[3].min():.2f}–{DEV[3].max():.2f}° on the bottom row, where containment drops "
    f"to {CON[3].min():.0f}–{CON[3].max():.0f} %. RMS scatter barely moves "
    f"(row means {rr[0]:.2f} / {rr[1]:.2f} / {rr[2]:.2f} / {rr[3]:.2f}°) and no cell "
    f"exceeds the colour scale: the tracker stays equally steady across the screen, it "
    f"simply points to the wrong place low down.",
    ha="left", va="top", fontsize=7.8, color=INK3, wrap=True, transform=fig.transFigure)

path = f"{OUT}/fig2_cell_maps_synthetic.png"
fig.savefig(path); plt.close(fig)
print("dev  row means", np.round(rm, 2))
print("rms  row means", np.round(rr, 2))
print("cont row means", np.round(rc, 1))
print("wrote", path)
