import csv, os, json, math, statistics as st
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

SP  = "/private/tmp/claude-501/-Users-advait-Desktop-GazeTracking-Mobile-Gaze-tracking-main/babf1d27-2b25-429a-b4a4-f6616018cd2d/scratchpad/exp2"
OUT = "/Users/advait/Desktop/GazeTracking/Mobile_Gaze_tracking-main/analysis_exp2"
os.makedirs(OUT, exist_ok=True)
RUNS = [
    ("Run 1", "/Users/advait/Downloads/EXp 2/session_20260826_200514/exp2/run1_4x4"),
    ("Run 4", "/Users/advait/Downloads/EXp 2/exp2_fixation_stability_4x4_run4_20260826_123603"),
    ("Run 5", f"{SP}/exp2_fixation_stability_4x4_run5_20260826_125428/exp2_fixation_stability_4x4_run5_20260826_125428"),
    ("Run 6", f"{SP}/exp2_fixation_stability_4x4_run6_20260826_125605/exp2_fixation_stability_4x4_run6_20260826_125605"),
    ("Run 7", f"{SP}/exp2_fixation_stability_4x4_run7_20260826_125739/exp2_fixation_stability_4x4_run7_20260826_125739"),
]
NAMES = [n for n, _ in RUNS]

INK, INK2, INK3 = "#12161b", "#4c5763", "#78838f"
LINE = "#d6dbe0"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
RAMP  = ["#cde2fb","#9ec5f4","#6da7ec","#3987e5","#256abf","#184f95"]
RAMPO = ["#fde5d8","#fbc3a8","#f79f76","#f07a48","#dd5a24","#b34617"]
BLUES  = matplotlib.colors.LinearSegmentedColormap.from_list("b", RAMP)
ORANGE = matplotlib.colors.LinearSegmentedColormap.from_list("o", RAMPO)

plt.rcParams.update({
    "figure.dpi":200,"savefig.dpi":200,"font.family":"sans-serif",
    "font.sans-serif":["Helvetica Neue","Helvetica","Arial","DejaVu Sans"],"font.size":9,
    "axes.edgecolor":LINE,"axes.labelcolor":INK2,"axes.titlecolor":INK,
    "axes.titlesize":11,"axes.titleweight":"bold","axes.labelsize":9.5,
    "axes.grid":True,"grid.color":"#e6eaed","grid.linewidth":.7,
    "xtick.color":INK3,"ytick.color":INK3,"xtick.labelsize":8.5,"ytick.labelsize":8.5,
    "legend.frameon":False,"legend.fontsize":8.5,
    "savefig.bbox":"tight","savefig.facecolor":"white",
    "figure.facecolor":"white","axes.facecolor":"white",
})
def tidy(ax, xgrid=False, ygrid=True):
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(LINE); ax.spines["bottom"].set_color(LINE)
    ax.xaxis.grid(xgrid); ax.yaxis.grid(ygrid); ax.set_axisbelow(True)
def cap(fig, t):
    fig.text(0.0, -0.045, t, ha="left", va="top", fontsize=7.8, color=INK3,
             wrap=True, transform=fig.transFigure)

def sect(p):
    b, cur = [], None
    for line in open(p):
        line = line.rstrip("\n")
        if not line.strip() or line.startswith("#"): cur = None; continue
        if cur is None: cur = {"h": line.split(","), "r": []}; b.append(cur)
        else: cur["r"].append(next(csv.reader([line])))
    return [[dict(zip(x["h"], r)) for r in x["r"]] for x in b]
F = float
CELLS, SAMPS, METAS = {}, {}, {}
for n, p in RUNS:
    bl = sect(os.path.join(p, "trials.csv"))
    CELLS[n], SAMPS[n] = bl[0], bl[1]
    METAS[n] = json.load(open(os.path.join(p, "meta.json")))
SUM = {n: METAS[n]["summary"] for n in NAMES}
TZ  = {n: SUM[n]["mean_dev_pt"]/SUM[n]["mean_dev_deg"] for n in NAMES}

# ══ FIG 1 — per-run outcome and the accuracy/precision split ══
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.4), gridspec_kw={"wspace":0.34})
x = np.arange(5)

ax = axs[0]; tidy(ax)
dev = [SUM[n]["mean_dev_deg"] for n in NAMES]
ax.bar(x, dev, 0.6, color=S1, zorder=3)
for i, v in enumerate(dev):
    ax.text(i, v+0.07, f"{v:.2f}°", ha="center", fontsize=8.8, color=INK, fontweight="bold")
    ax.text(i, 0.16, f"{SUM[NAMES[i]]['containment_pct']:.0f}%", ha="center",
            va="bottom", fontsize=8, color="white", fontweight="bold")
ax.axhline(np.mean(dev), color=INK3, ls="--", lw=1.1, zorder=1)
ax.text(-0.45, np.mean(dev)+0.09, f"mean {np.mean(dev):.2f}°", ha="left", fontsize=7.8, color=INK3)
ax.set_xticks(x); ax.set_xticklabels(NAMES)
ax.set_ylabel("Mean deviation from centre (°)"); ax.set_ylim(0, 3.9)
ax.set_title("A · Fixation deviation by run", loc="left")

ax = axs[1]; tidy(ax)
w = 0.38
bias = [SUM[n]["bias_pt"] for n in NAMES]; sd = [SUM[n]["sd_pt"] for n in NAMES]
ax.bar(x-w/2, bias, w, color=S2, label="Bias — systematic offset", zorder=3)
ax.bar(x+w/2, sd,  w, color=S1, label="SD — random scatter", zorder=3)
for i in range(5):
    ax.text(i-w/2, bias[i]+1.5, f"{bias[i]:.0f}", ha="center", fontsize=7.8, color=INK)
    ax.text(i+w/2, sd[i]+1.5,   f"{sd[i]:.0f}",   ha="center", fontsize=7.8, color=INK)
ax.set_xticks(x); ax.set_xticklabels(NAMES)
ax.set_ylabel("Points"); ax.set_ylim(0, 92)
ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), fontsize=8)
ax.set_title("B · Bias dwarfs scatter in every run", loc="left")

ax = axs[2]; tidy(ax, ygrid=False)
share = [100*b*b/(b*b+s*s) for b, s in zip(bias, sd)]
ax.barh(x, share, 0.55, color=S2, label="Bias²", zorder=3)
ax.barh(x, [100-s for s in share], 0.55, left=share, color=S1, label="Scatter²", zorder=3)
for i, v in enumerate(share):
    ax.text(v-2, i, f"{v:.0f} %", ha="right", va="center", color="white",
            fontsize=8.5, fontweight="bold")
ax.set_yticks(x); ax.set_yticklabels(NAMES); ax.invert_yaxis()
ax.set_xlim(0, 100); ax.set_xlabel("Share of the squared error budget (%)")
ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.10), fontsize=8)
ax.set_title("C · 92 % of the error is correctable bias", loc="left", pad=18)

cap(fig, "Figure 1 — Experiment 2, 4×4 fixation-stability grid: one target per cell, 1 s unscored settle + 4 s scored capture, 16 cells in shuffled order (~80 s per run), 5 runs. The figure inside each bar in panel A is containment — the share of scored samples that fell inside the target cell. Deviation decomposes as dev² ≈ bias² + scatter², where bias is the offset of the fixation cloud's centroid from the target and scatter is the spread of samples about that centroid. Pooled bias 49.9 pt vs scatter 14.4 pt: the estimator is precise but systematically off. More smoothing cannot fix this; a richer calibration model can.")
fig.savefig(f"{OUT}/fig1_accuracy_vs_precision.png"); plt.close(fig)

# ══ FIG 2 — the 4x4 maps ══
def cellstat(key, fn=st.mean):
    g = np.zeros((4,4))
    for c in range(16):
        vals = [F(r[key]) for n in NAMES for r in CELLS[n] if int(r["cell_idx"]) == c]
        r0 = [r for r in CELLS["Run 1"] if int(r["cell_idx"]) == c][0]
        g[int(r0["row"]), int(r0["col"])] = fn(vals)
    return g
DEV = cellstat("mean_dev_deg")
RMS = cellstat("rms_deg")
CON = cellstat("containment_pct")

fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.0), gridspec_kw={"wspace":0.30})
panels = [
    (DEV, BLUES, "A · Deviation (accuracy), °", "{:.2f}", None, None, 1.55),
    (RMS, BLUES, "B · RMS scatter (precision), °", "{:.2f}", 0.8, 2.0, 1.6),
    (CON, ORANGE.reversed(), "C · Containment, %", "{:.0f}", None, None, 70),
]
for ax, (G, cm, title, fmt, vmn, vmx, thr) in zip(axs, panels):
    im = ax.imshow(G, cmap=cm, vmin=vmn, vmax=vmx, aspect=0.85)
    for i in range(4):
        for j in range(4):
            hot = (G[i,j] > thr) if title[0] != "C" else (G[i,j] < thr)
            lab = fmt.format(G[i,j]) + ("*" if (vmx is not None and G[i,j] > vmx) else "")
            ax.text(j, i, lab, ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white" if hot else INK)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(["col 0","col 1","col 2","col 3"], fontsize=7.5)
    ax.set_yticklabels(["row 0","row 1","row 2","row 3"], fontsize=7.5)
    ax.grid(False)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(title, loc="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7.5, color=INK3)

cap(fig, "Figure 2 — Per-cell maps pooled over 5 runs (80 cell-runs). Accuracy collapses on the bottom row (4.6–6.2° vs 1.2–2.5° elsewhere) and containment falls to 43–61 %, yet RMS scatter barely moves once the single catastrophic cell is excluded (row means 1.38 / 1.10 / 1.16 / 1.44°). The tracker stays equally steady across the screen; it simply points to the wrong place low down. *In panel B, cell (row 3, col 0) reads 3.82° and sits above the colour scale — it is inflated by a single catastrophic cell-run in Run 1 (14.28°).")
fig.savefig(f"{OUT}/fig2_cell_maps.png"); plt.close(fig)

# ══ FIG 3 — bias vector field ══
BX = np.zeros((4,4)); BY = np.zeros((4,4))
for c in range(16):
    bx = [st.mean(F(r["dev_x_pt"]) for r in SAMPS[n] if int(r["cell_idx"])==c) for n in NAMES]
    by = [st.mean(F(r["dev_y_pt"]) for r in SAMPS[n] if int(r["cell_idx"])==c) for n in NAMES]
    r0 = [r for r in CELLS["Run 1"] if int(r["cell_idx"])==c][0]
    BX[int(r0["row"]), int(r0["col"])] = st.mean(bx)
    BY[int(r0["row"]), int(r0["col"])] = st.mean(by)

fig = plt.figure(figsize=(11.5, 4.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.0], wspace=0.34)

ax = fig.add_subplot(gs[0])
for i in range(4):
    for j in range(4):
        ax.add_patch(plt.Rectangle((j-.5, i-.5), 1, 1, fill=False, ec=LINE, lw=1))
SC = 1/150
for i in range(4):
    for j in range(4):
        dx, dy = BX[i,j]*SC, BY[i,j]*SC
        big = abs(BY[i,j]) > 60
        col = S2 if big else S1
        mag = math.hypot(BX[i,j], BY[i,j])
        k = min(1.0, mag/70)
        ax.annotate("", xy=(j+dx, i+dy), xytext=(j, i),
                    arrowprops=dict(
                        arrowstyle=f"-|>,head_width={0.05+0.11*k:.3f},head_length={0.11+0.23*k:.3f}",
                        color=col, lw=1.2+0.6*k, shrinkA=2.5, shrinkB=0))
        ax.plot([j], [i], "o", ms=4.5, color=INK3, zorder=5)
        if big:
            ax.text(j+0.14, i+dy+0.10, f"{BY[i,j]:.0f} pt", ha="left", fontsize=7.4,
                    color=S2, fontweight="bold")
ax.set_xlim(-.6, 3.6); ax.set_ylim(3.6, -0.85)
ax.set_xticks(range(4)); ax.set_yticks(range(4))
ax.set_xticklabels([f"col {i}" for i in range(4)], fontsize=7.5)
ax.set_yticklabels([f"row {i}" for i in range(4)], fontsize=7.5)
ax.grid(False); ax.set_aspect(0.85)
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_title("A · Where the gaze cloud actually sat", loc="left", pad=10)
ax.set_xlabel("dot = target, arrow tip = fixation centroid\nevery arrow points UP: the estimator under-shoots downward gaze",
              fontsize=7.6, color=INK2, labelpad=8)

ax = fig.add_subplot(gs[1]); tidy(ax)
rows = np.arange(4)
by_row = BY.mean(axis=1); bx_row = BX.mean(axis=1)
ax.bar(rows-0.19, bx_row, 0.36, color=S1, label="Horizontal bias", zorder=3)
ax.bar(rows+0.19, by_row, 0.36, color=S2, label="Vertical bias", zorder=3)
for i in rows:
    ax.text(i+0.19, by_row[i]-6, f"{by_row[i]:.0f}", ha="center", va="top",
            fontsize=8, color=INK, fontweight="bold")
ax.axhline(0, color=INK, lw=1)
ax.set_xticks(rows); ax.set_xticklabels([f"row {i}" for i in rows])
ax.set_ylabel("Mean bias (pt)   ·   negative = above target")
ax.set_ylim(-115, 32); ax.legend(loc="lower left")
ax.set_title("B · Vertical bias grows down the screen", loc="left")

ax = fig.add_subplot(gs[2]); tidy(ax)
labels = ["as scored", "− global\noffset", "− per-row\ncorrection", "− per-cell\n(floor)"]
vals = [2.66, 2.28, 1.77, 1.27]
colr = [S2, "#f0a07a", "#8fb8e8", S1]
ax.bar(range(4), vals, 0.6, color=colr, zorder=3)
for i, v in enumerate(vals):
    ax.text(i, v+0.06, f"{v:.2f}°", ha="center", fontsize=9, color=INK, fontweight="bold")
    if i:
        ax.text(i, 0.13, f"−{100*(1-v/vals[0]):.0f} %", ha="center", va="bottom",
                fontsize=7.8, color="white", fontweight="bold")
ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Mean deviation (°)"); ax.set_ylim(0, 3.15)
ax.set_title("C · What a bias correction would buy", loc="left")

cap(fig, "Figure 3 — Panel A: mean bias vector per cell, pooled over 5 runs, drawn to scale (arrow length: one cell width = 150 pt of bias; labels give the vertical component in points). Panel C re-scores the same samples after removing a systematic offset a richer calibration model could absorb: a single global offset recovers 14 %, a per-row vertical correction 33 %, and the per-cell floor of 1.27° is the true precision limit of this pipeline.")
fig.savefig(f"{OUT}/fig3_bias_field.png"); plt.close(fig)

# ══ FIG 4 — settling within the scored window ══
fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.5),
                        gridspec_kw={"width_ratios":[1.4, 1.0], "wspace":0.27})
ax = axs[0]; tidy(ax)
edges = np.arange(0, 4.25, 0.5); ctr = (edges[:-1]+edges[1:])/2
pool = []
for n, col in zip(NAMES, [S2, "#9ec5f4", "#6da7ec", "#3987e5", "#256abf"]):
    byc = {}
    for r in SAMPS[n]: byc.setdefault(r["cell_idx"], []).append(r)
    T, D = [], []
    for c, rs in byc.items():
        t0 = min(F(y["t_s"]) for y in rs)
        for x in rs: T.append(F(x["t_s"])-t0); D.append(F(x["dev_pt"]))
    T, D = np.array(T), np.array(D)
    ys = [D[(T>=a)&(T<b)].mean() if ((T>=a)&(T<b)).any() else np.nan
          for a, b in zip(edges[:-1], edges[1:])]
    pool.append(ys)
    ax.plot(ctr, ys, "-o", color=col, lw=1.8, ms=4.5, label=n, zorder=3,
            markeredgecolor="white", markeredgewidth=.8)
ax.plot(ctr, np.nanmean(pool, axis=0), "-", color=INK, lw=2.8, zorder=4, label="Pooled")
ax.set_xlabel("Time since scoring began for that cell (s)")
ax.set_ylabel("Mean deviation (pt)")
ax.set_xlim(0, 4.1); ax.xaxis.set_major_locator(MultipleLocator(1))
ax.legend(ncol=3, loc="upper right")
ax.axvspan(0, 1, color=S2, alpha=.08, zorder=1)
ax.text(.5, ax.get_ylim()[0]+4, "first second\nstill converging", ha="center",
        fontsize=7.6, color=S2, fontweight="bold")
ax.set_title("A · The 1 s settle is not enough", loc="left")

ax = axs[1]; tidy(ax)
asis = [53.8, 63.1, 45.0, 50.1, 65.5]; trim = [44.1, 58.9, 41.9, 48.6, 61.1]
x = np.arange(5); w = 0.38
ax.bar(x-w/2, asis, w, color=S2, label="As scored (4 s)", zorder=3)
ax.bar(x+w/2, trim, w, color=S1, label="First second discarded", zorder=3)
for i in range(5):
    ax.text(i, max(asis[i], trim[i])+1.8, f"−{100*(1-trim[i]/asis[i]):.0f} %",
            ha="center", fontsize=7.8, color=INK, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(NAMES)
ax.set_ylabel("Mean deviation (pt)"); ax.set_ylim(0, 78); ax.legend(loc="upper left", ncol=1)
ax.set_title("B · Re-scoring on the last 3 s", loc="left")

cap(fig, "Figure 4 — Deviation falls 30 % across the scored window (68.2 pt in the first second, 47.9 pt in the last), so the fixation is still converging when scoring starts and the reported mean is inflated. Discarding the first scored second recovers 3–18 % per run. Extend the settle to ~2 s, or score only the last 3 s, before quoting a stability figure.")
fig.savefig(f"{OUT}/fig4_settling.png"); plt.close(fig)

# ══ FIG 5 — outliers, stability over the run, grid-design implication ══
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.4), gridspec_kw={"wspace":0.33})

ax = axs[0]; tidy(ax)
allrms = sorted(F(r["rms_deg"]) for n in NAMES for r in CELLS[n])
good = [v for v in allrms if v <= 3]; bad = [v for v in allrms if v > 3]
bins = np.arange(0.4, 3.4, 0.2)
ax.hist(good, bins=bins, color=S1, edgecolor="white", zorder=3)
ax.bar([3.3], [len(bad)], width=0.2, color=S2, edgecolor="white", zorder=3, align="edge")
ax.set_xticks([0.5,1.0,1.5,2.0,2.5,3.0,3.4])
ax.set_xticklabels(["0.5","1.0","1.5","2.0","2.5","3.0",">3"])
ax.annotate(f"{len(bad)} cell-runs above 3°  (5 %)\nworst 14.28°, 5.61°, 4.17°, 4.11°",
            xy=(3.38, len(bad)), xytext=(1.62, 12.6), fontsize=7.6, color=S2,
            arrowprops=dict(arrowstyle="->", color=S2, lw=.9))
ax.text(0.46, 15.4, f"the other 76 cell-runs:\nmean {np.mean(good):.2f}°, median {np.median(good):.2f}°",
        fontsize=8, color=INK2)
ax.set_xlabel("Per-cell RMS scatter (°)"); ax.set_ylabel("Cell-runs (n = 80)")
ax.set_ylim(0, 19)
ax.set_title("A · The noise floor is tight", loc="left")

ax = axs[1]; tidy(ax)
qs = {1:[],2:[],3:[],4:[]}
for n in NAMES:
    for r in CELLS[n]:
        v = F(r["rms_deg"])
        if v < 10: qs[(int(r["order"])-1)//4 + 1].append(v)
mu = [np.mean(qs[k]) for k in sorted(qs)]
se = [np.std(qs[k])/math.sqrt(len(qs[k])) for k in sorted(qs)]
ax.errorbar(range(1,5), mu, yerr=se, fmt="-o", color=S1, lw=2, ms=8,
            capsize=4, markeredgecolor="white", markeredgewidth=1, zorder=3)
ax.set_xticks(range(1,5))
ax.set_xticklabels(["cells 1–4","cells 5–8","cells 9–12","cells 13–16"], fontsize=8)
ax.set_ylabel("Mean RMS scatter (°)"); ax.set_ylim(0.6, 2.3)
ax.text(1.05, 2.12, "r(order, RMS) = +0.05 — no fatigue\nacross the 80 s run",
        fontsize=8, color=INK2)
ax.set_title("B · Precision holds over the run", loc="left")

ax = axs[2]; tidy(ax, ygrid=False)
allpt = sorted(F(r["rms_pt"]) for n in NAMES for r in CELLS[n])
p50 = float(np.median(allpt)); p90 = float(np.percentile(allpt, 90))
grids = [("9×9\n(Exp 1)", 402/9, 778/9), ("6×4", 402/4, 778/6),
         ("4×3\n(Exp 3)", 134, 149), ("3×3\n(Exp 4)", 134, 199)]
y = np.arange(len(grids))
half = [min(w, h)/2 for _, w, h in grids]
ax.barh(y, half, 0.55, color=[S2 if h < p90 else S1 for h in half], zorder=3)
ax.axvline(p50, color=INK3, ls=":", lw=1.4)
ax.axvline(p90, color=S2, ls="--", lw=1.4)
ax.text(p50, -0.72, f"median {p50:.0f} pt", fontsize=7.4, color=INK3, ha="center")
ax.text(p90, -0.55, f"90th pct {p90:.0f} pt", fontsize=7.4, color=S2, ha="center")
for i, h in enumerate(half):
    ax.text(h+1.5, i, f"{h:.0f} pt", va="center", fontsize=8.5, color=INK, fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels([g[0] for g in grids], fontsize=8)
ax.set_ylim(3.6, -0.95)
ax.set_xlim(0, 95); ax.set_xlabel("Cell half-width (pt)")
ax.set_title("C · Which grids this floor can resolve", loc="left")

cap(fig, "Figure 5 — Panel C: a cell is resolvable when its half-width clears the per-sample error at that position. The 4×3 and 3×3 grids used in Experiments 3 and 4 (67 pt half-width) clear even the 90th-percentile per-cell RMS of 44 pt; a 9×9 grid (22 pt half-width) sits at the median of 21 pt and is not resolvable by this pipeline. Bars are coloured orange where the half-width falls below the 90th-percentile error.")
fig.savefig(f"{OUT}/fig5_noise_floor.png"); plt.close(fig)

print("written to", OUT)
for f in sorted(os.listdir(OUT)): print("  ", f)
