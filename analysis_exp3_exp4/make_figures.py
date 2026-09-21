import csv, os, glob, math, statistics as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator

ROOT = "/Users/advait/Downloads/EXP 3"
OUT  = "/Users/advait/Desktop/GazeTracking/Mobile_Gaze_tracking-main/analysis_exp3_exp4"
os.makedirs(OUT, exist_ok=True)
S = sorted(glob.glob(os.path.join(ROOT, "session_*")))
LBL = [os.path.basename(s)[-6:] for s in S]
LBL = [l[:2]+":"+l[2:4]+":"+l[4:] for l in LBL]

# ── house style ────────────────────────────────────────────────
INK, INK2, INK3 = "#12161b", "#4c5763", "#78838f"
LINE = "#d6dbe0"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]

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

# ── data loaders ───────────────────────────────────────────────
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

VAL  = [sect(os.path.join(s, "validation/run1_9dot/trials.csv"))[0] for s in S]
E3   = [sect(os.path.join(s, "exp3/run1_communication/trials.csv"))   for s in S]
E4   = [sect(os.path.join(s, "exp4/run1_predictive_cued/trials.csv")) for s in S]

# ═══ FIG 1 — calibration: per-session error + per-dot map + row means ═══
dot_err = np.array([[float(r["err_deg"]) for r in v] for v in VAL])      # 5 x 9
dot_mean = dot_err.mean(axis=0)
sess_mean = dot_err.mean(axis=1)

fig = plt.figure(figsize=(11.5, 3.5))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.85, 1.0], wspace=0.55)

ax = fig.add_subplot(gs[0]); tidy(ax)
cols = [S2 if m >= 2.0 else S1 for m in sess_mean]
ax.bar(range(5), sess_mean, color=cols, width=0.62, zorder=3)
for i, m in enumerate(sess_mean):
    ax.text(i, m + 0.06, f"{m:.2f}", ha="center", fontsize=8.5, color=INK, fontweight="bold")
ax.axhline(2.0, color=INK3, ls="--", lw=1)
ax.text(4.45, 2.06, "marginal threshold", ha="right", fontsize=7.6, color=INK3)
ax.set_xticks(range(5)); ax.set_xticklabels(LBL, rotation=30, ha="right")
ax.set_ylabel("Mean angular error (°)"); ax.set_ylim(0, 2.85)
ax.set_title("A · Validation error by session", loc="left")

ax = fig.add_subplot(gs[1])
grid = dot_mean.reshape(3, 3)
im = ax.imshow(grid, cmap=matplotlib.colors.LinearSegmentedColormap.from_list("b", RAMP),
               vmin=1.10, vmax=2.00, aspect=0.8)
for i in range(3):
    for j in range(3):
        v = grid[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11,
                fontweight="bold", color="white" if v > 1.55 else INK)
        ax.text(j, i + 0.28, f"dot {i*3+j+1}", ha="center", va="center",
                fontsize=6.8, color="white" if v > 1.55 else INK3)
ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_title("B · Mean error by screen position (°)", loc="left", pad=12)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7.5, color=INK3)

ax = fig.add_subplot(gs[2]); tidy(ax, xgrid=True, ygrid=False)
rows = grid.mean(axis=1)
ax.barh([2, 1, 0], rows, color=[S1, S1, S2], height=0.55, zorder=3)
for y, v in zip([2, 1, 0], rows):
    ax.text(v + 0.03, y, f"{v:.2f}°", va="center", fontsize=9, color=INK, fontweight="bold")
ax.set_yticks([2, 1, 0]); ax.set_yticklabels(["Top row", "Middle row", "Bottom row"])
ax.set_xlabel("Mean angular error (°)"); ax.set_xlim(0, 2.3)
ax.set_title("C · Error by screen row", loc="left")

cap(fig, "Figure 1 — Nine-dot calibration validation, 45 dots over 5 sessions. The bottom row carries 47 % more error than the top (1.90° vs 1.29°), while precision (RMS scatter) stays flat at 1.1–1.5° everywhere — a systematic pose-fit bias at downward gaze, not extra noise. Angular error uses the calibration's fitted virtual eye-to-screen distance (mean |tz| = 1 186 pt).")
fig.savefig(f"{OUT}/fig1_calibration.png"); plt.close(fig)

# ═══ FIG 2 — Exp 3 accuracy, rate, margin ═══
e3_ov = [b[2][0] for b in E3]
acc  = [float(o["selection_accuracy_pct"]) for o in e3_ov]
wpm  = [float(o["words_per_min"]) for o in e3_ov]
sps  = [float(o["mean_s_per_selection"]) for o in e3_ov]

# per-pick normalised offset
norm, offs = [], []
for sel, layout, _ in E3:
    L = {r["cell_idx"]: r for r in layout}
    for r in sel:
        if r["correct"] != "1": continue
        c = L[r["cell_idx"]]
        cx = (float(c["x_min"]) + float(c["x_max"])) / 2
        cy = (float(c["y_min"]) + float(c["y_max"])) / 2
        hw = (float(c["x_max"]) - float(c["x_min"])) / 2
        hh = (float(c["y_max"]) - float(c["y_min"])) / 2
        dx, dy = float(r["pred_x"]) - cx, float(r["pred_y"]) - cy
        offs.append(math.hypot(dx, dy))
        norm.append(max(abs(dx) / hw, abs(dy) / hh))

fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.4))
fig.subplots_adjust(wspace=0.32)

ax = axs[0]; tidy(ax)
x = np.arange(5)
cols = [S1 if a == 100 else (S2 if a < 70 else "#6da7ec") for a in acc]
ax.bar(x, acc, 0.6, color=cols, zorder=3)
for i in range(5):
    ax.text(i, acc[i] + 2.5, f"{acc[i]:.0f} %", ha="center", fontsize=9,
            color=INK, fontweight="bold")
    ax.text(i, 4, f"{wpm[i]:.1f}\nwpm", ha="center", va="bottom", fontsize=7.6,
            color="white", fontweight="bold")
ax.axhline(75.0, color=INK3, ls="--", lw=1.1, xmin=0.16, zorder=1)
ax.text(4.45, 77, "pooled 75.0 %", ha="right", fontsize=7.8, color=INK3)
ax.set_xticks(x); ax.set_xticklabels(LBL, rotation=30, ha="right")
ax.set_ylim(0, 118); ax.set_ylabel("Selection accuracy (%)")
ax.set_title("A · Exp 3 per-run outcome", loc="left")

ax = axs[1]; tidy(ax)
ax.hist(norm, bins=np.arange(0.25, 1.05, 0.075), color=S1, edgecolor="white", zorder=3)
ax.axvline(1.0, color=INK, lw=1.6)
ax.text(0.985, ax.get_ylim()[1]*0.92, "tile border", ha="right", fontsize=8, color=INK, fontweight="bold")
ax.axvline(np.mean(norm), color=S2, lw=1.6, ls="--")
ax.text(np.mean(norm), 4.28, f"mean {np.mean(norm):.2f}",
        ha="center", fontsize=8, color=S2, fontweight="bold")
ax.set_ylim(0, 4.75)
ax.axvspan(0.8, 1.0, color=S2, alpha=0.09, zorder=1)
ax.text(0.9, ax.get_ylim()[1]*0.5, "outer\nfifth\n5 %", ha="center", fontsize=7.5, color=S2)
ax.set_xlabel("Offset from tile centre  (1.0 = tile border)")
ax.set_ylabel("Correct selections (n = 21)")
ax.set_title("B · Confirmed picks sit well inside the tile", loc="left")

ax = axs[2]; tidy(ax, xgrid=True, ygrid=False)
kinds = ["Correct", "Spatial\nsubstitution", "Unintended\nre-selection"]
vals  = [21, 5, 2]
colr  = [S1, S2, S3]
ax.barh(range(3), vals, color=colr, height=0.55, zorder=3)
for i, v in enumerate(vals):
    ax.text(v + 0.4, i, f"{v}   ({v/28*100:.1f} %)", va="center", fontsize=8.5, color=INK, fontweight="bold")
ax.set_yticks(range(3)); ax.set_yticklabels(kinds); ax.invert_yaxis()
ax.set_xlim(0, 27); ax.set_xlabel("Selections (n = 28)")
ax.set_title("C · Exp 3 outcome taxonomy", loc="left")

cap(fig, "Figure 2 — Experiment 3, static 4×3 word grid, 1.0 s dwell. 28 selections over 5 runs; tiles 134 × 149 pt, reshuffled per run. Words-per-minute is printed inside each bar in panel A (mean 16.5, SD 3.9) — the rate is far more stable than the accuracy, because an error costs a re-pick but little time.")
fig.savefig(f"{OUT}/fig2_exp3.png"); plt.close(fig)

# ═══ FIG 3 — Exp 4 attribution ═══
SLOT = [0,1,2,3,4,5,7]
on = hit = off = 0
cheb = {0:0, 1:0, 2:0}
per_sess = []
for sel, grids, geo, trials, ov in E4:
    o = h = 0
    for r in sel:
        if r["kind"] != "word": continue
        ir = int(r["intended_rank"])
        if ir < 0:
            off += 1; continue
        on += 1; o += 1
        want, got = SLOT[ir], int(r["cell_idx"])
        d = max(abs(want//3 - got//3), abs(want%3 - got%3))
        cheb[d] += 1
        if int(r["chosen_rank"]) == ir: hit += 1; h += 1
    per_sess.append((o, h))

fig = plt.figure(figsize=(11.5, 3.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.0], wspace=0.33)

ax = fig.add_subplot(gs[0]); tidy(ax, xgrid=False, ygrid=False)
segs = [(hit, S1, "Correct\ncell", f"{hit}"),
        (on-hit, S2, "Wrong\ncell", f"{on-hit}"),
        (off, S3, "Needed word no longer on the grid", f"{off}")]
left = 0
for v, c, lab, n in segs:
    ax.barh(0, v, left=left, color=c, height=0.44, zorder=3)
    ax.text(left + v/2, 0.055, n, ha="center", va="center", fontsize=13,
            color="white", fontweight="bold")
    ax.text(left + v/2, -0.10, lab, ha="center", va="center", fontsize=7.6,
            color="white")
    left += v
ax.plot([0, 0, on, on], [-0.30, -0.37, -0.37, -0.30], color=INK3, lw=1)
ax.text(0, -0.45, f"{on} scorable — needed word was on the grid\nconditional accuracy {hit/on*100:.1f} %",
        ha="left", va="top", fontsize=8.5, color=INK)
ax.plot([on, on, 65, 65], [-0.30, -0.37, -0.37, -0.30], color=INK3, lw=1)
ax.text(0, -0.72, f"{off} unscorable — downstream of a divergence, {off/65*100:.0f} % of all picks",
        ha="left", va="top", fontsize=8.5, color=INK2)
ax.set_xlim(0, 65); ax.set_ylim(-0.95, 0.42)
ax.set_yticks([]); ax.set_xlabel("Word selections (n = 65)")
for sp in ("left","top","right"): ax.spines[sp].set_visible(False)
ax.set_title("A · Why cued answers failed", loc="left")

ax = fig.add_subplot(gs[1]); tidy(ax)
ks = [0, 1, 2]; vs = [cheb[k] for k in ks]
ax.bar(ks, vs, color=[S1, S2, S3], width=0.6, zorder=3)
for k, v in zip(ks, vs):
    ax.text(k, v + 0.25, f"{v}\n{v/on*100:.0f} %", ha="center", fontsize=8.5,
            color=INK, fontweight="bold")
ax.set_xticks(ks)
ax.set_xticklabels(["same cell\n(correct)", "adjacent\n(tracker error)", "two cells\n(NOT tracker)"])
ax.set_ylim(0, 14); ax.set_ylabel(f"Selections (n = {on})")
ax.set_title("B · Distance to the needed cell", loc="left")

ax = fig.add_subplot(gs[2]); tidy(ax)
val_x = sess_mean
cond  = [h/o*100 if o else np.nan for o, h in per_sess]
ax.scatter(val_x, acc, s=95, color=S1, zorder=4, label="Exp 3 accuracy")
ax.scatter(val_x, cond, s=95, color=S2, marker="s", zorder=4, label="Exp 4 conditional acc.")
m, b = np.polyfit(val_x, acc, 1)
xs = np.linspace(0.95, 2.55, 10)
ax.plot(xs, m*xs + b, color=S1, lw=1.4, ls="--", zorder=3)
r = np.corrcoef(val_x, acc)[0, 1]
ax.text(2.45, 96, f"r = {r:+.2f}", ha="right", fontsize=9, color=S1, fontweight="bold")
r4 = np.corrcoef(val_x, cond)[0, 1]
ax.text(2.45, 86, f"r = {r4:+.2f}", ha="right", fontsize=9, color=S2, fontweight="bold")
ax.set_xlabel("Validation error (°)"); ax.set_ylabel("Selection accuracy (%)")
ax.set_ylim(0, 112); ax.legend(loc="lower left", fontsize=8)
ax.set_title("C · Calibration vs task accuracy", loc="left")

cap(fig, "Figure 3 — Experiment 4, predictive 3×3 grid, cued condition. Panel B: a 1.5° estimator on 134 × 199 pt cells cannot skip a cell, so the 7 two-cell picks are deliberate branch choices, not tracker error. Panel C plots Exp 4 CONDITIONAL accuracy (correct picks among the 26 scorable ones); n = 5, so r is a direction, not an effect size.")
fig.savefig(f"{OUT}/fig3_exp4_attribution.png"); plt.close(fig)

# ═══ FIG 4 — timing ═══
g3, g4w, g4c = [], [], []
for sel, _, _ in E3:
    g3 += [float(r["since_prev_s"]) for r in sel]
for sel, _, _, _, _ in E4:
    g4w += [float(r["since_prev_s"]) for r in sel if r["kind"] == "word"]
    g4c += [float(r["since_prev_s"]) for r in sel if r["kind"] != "word"]

fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.5), gridspec_kw={"width_ratios":[1.5, 1.0], "wspace":0.28})

ax = axs[0]; tidy(ax, xgrid=True, ygrid=False)
data = [g3, g4w, g4c]
names = [f"Exp 3 words\n1.0 s dwell (n={len(g3)})",
         f"Exp 4 words\n1.0 s dwell (n={len(g4w)})",
         f"Exp 4 controls\n1.5 s dwell (n={len(g4c)})"]
cols = [S1, S2, S3]
bp = ax.boxplot(data, vert=False, widths=0.5, patch_artist=True, showfliers=False,
                medianprops=dict(color=INK, lw=2), whiskerprops=dict(color=LINE),
                capprops=dict(color=LINE), boxprops=dict(edgecolor=LINE))
for patch, c in zip(bp["boxes"], cols):
    patch.set_facecolor(c); patch.set_alpha(0.18)
rng = np.random.default_rng(7)
for i, (d, c) in enumerate(zip(data, cols), start=1):
    ax.scatter(d, i + rng.uniform(-0.15, 0.15, len(d)), s=22, color=c,
               alpha=0.65, edgecolor="white", linewidth=0.6, zorder=4)
ax.axvline(1.0, color=INK, ls="--", lw=1.4)
ax.text(1.18, 0.42, "1.0 s dwell charged\nby the system", fontsize=8, color=INK)
ax.set_ylim(0.25, 3.75)
for i, d in enumerate(data, start=1):
    ax.text(12.2, i, f"med {np.median(d):.2f} s", va="center", fontsize=8.2, color=INK2)
ax.set_yticks([1, 2, 3]); ax.set_yticklabels(names)
ax.set_xlabel("Interval between confirmed selections (s)"); ax.set_xlim(0, 14)
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.set_title("A · Dwell-to-dwell interval", loc="left")

ax = axs[1]; tidy(ax)
lab = ["Exp 3", "Exp 4"]
dwell = [1.0, 1.0]
over  = [np.mean(g3) - 1.0, np.mean(g4w) - 1.0]
ax.bar(lab, dwell, color=S1, width=0.5, label="Dwell charged", zorder=3)
ax.bar(lab, over, bottom=dwell, color=S2, width=0.5, label="Search + saccade", zorder=3)
for i in range(2):
    ax.text(i, 0.5, "1.00 s", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.text(i, 1 + over[i]/2, f"{over[i]:.2f} s", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.text(i, 1 + over[i] + 0.12, f"{1+over[i]:.2f} s total", ha="center", fontsize=8.5, color=INK, fontweight="bold")
ax.set_ylabel("Mean seconds per selection"); ax.set_ylim(0, 4.8)
ax.legend(loc="upper right")
ax.set_title("B · Where the time actually goes", loc="left")

cap(fig, "Figure 4 — Search dominates: ~73 % of every selection is the participant finding the next word, not the tracker confirming it. Halving the dwell threshold would improve the rate by at most ~13 %.")
fig.savefig(f"{OUT}/fig4_timing.png"); plt.close(fig)

# ═══ FIG 5 — signal quality ═══
def load(p): return list(csv.DictReader(open(p)))
qrows = []
for s, lb in zip(S, LBL):
    for exp, run in (("exp3","run1_communication"), ("exp4","run1_predictive_cued")):
        R = load(os.path.join(s, exp, run, "samples.csv"))
        t = [float(r["t_s"]) for r in R]
        fps = 1/np.mean(np.diff(t))
        blink = 100*sum(1 for r in R if r.get("blink_held") not in (None,"","0"))/len(R)
        inc = [r["in_cell"] for r in R if r.get("in_cell") not in (None,"")]
        incp = 100*sum(1 for v in inc if v=="1")/len(inc)
        P = [r for r in R if r["pred_x"] not in ("","None")]
        jf = np.median([math.hypot(float(b["pred_x"])-float(a["pred_x"]), float(b["pred_y"])-float(a["pred_y"])) for a,b in zip(P,P[1:])])
        Q = [r for r in R if r["raw_pred_x"] not in ("","None")]
        jr = np.median([math.hypot(float(b["raw_pred_x"])-float(a["raw_pred_x"]), float(b["raw_pred_y"])-float(a["raw_pred_y"])) for a,b in zip(Q,Q[1:])])
        yaw = np.std([float(r["head_yaw_deg"]) for r in R])
        pit = np.std([float(r["head_pitch_deg"]) for r in R])
        qrows.append(dict(session=lb, exp=exp, n=len(R), fps=fps, blink=blink,
                          in_cell=incp, jit_filt=jf, jit_raw=jr, yaw=yaw, pitch=pit))

fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.3), gridspec_kw={"wspace":0.33})
labels = [f"{r['session']}\n{r['exp'].upper()}" for r in qrows]

ax = axs[0]; tidy(ax)
x = np.arange(len(qrows)); w = 0.4
ax.bar(x - w/2, [r["jit_raw"] for r in qrows], w, color="#9ec5f4", label="Raw", zorder=3)
ax.bar(x + w/2, [r["jit_filt"] for r in qrows], w, color="#184f95", label="One-Euro filtered", zorder=3)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
ax.set_ylabel("Median frame-to-frame jitter (pt)"); ax.set_ylim(0, 52)
ax.legend(loc="upper right", ncol=2, fontsize=7.6)
ax.text(0.02, 0.82, "33 % reduction — 22.4 pt vs 33.3 pt\nfiltered residual = 17 % of a 134 pt cell",
        transform=ax.transAxes, fontsize=7.6, color=INK2)
ax.set_title("A · Smoothing effect", loc="left")

ax = axs[1]; tidy(ax)
ecol = [S1 if r["exp"] == "exp3" else S2 for r in qrows]
ax.bar(x, [r["in_cell"] for r in qrows], color=ecol, width=0.62, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
ax.set_ylabel("Frames inside the active cell (%)"); ax.set_ylim(0, 112)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=S1, label="Exp 3"), Patch(color=S2, label="Exp 4")],
          loc="upper right", ncol=2, fontsize=7.6)
ax.set_title("B · On-target frame share", loc="left")

ax = axs[2]; tidy(ax)
ax.bar(x, [r["blink"] for r in qrows], color=ecol, width=0.62, zorder=3)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
ax.set_ylabel("Blink-gated frames (%)"); ax.set_ylim(0, 19)
ax.annotate("outlier run — also the slowest\n(10.1 wpm) and the only run\nwith re-selection errors",
            xy=(0.25, 15.6), xytext=(2.0, 15.0), fontsize=7.4, color=INK2,
            arrowprops=dict(arrowstyle="->", color=INK3, lw=0.9))
ax.axhline(2.5, color=INK3, ls=":", lw=1)
ax.text(9.4, 3.0, "mean 2.5 %", ha="right", fontsize=7.4, color=INK3)
ax.set_title("C · Blink gating", loc="left")

cap(fig, "Figure 5 — Pipeline telemetry from samples.csv. Effective logging rate 10.0 Hz (SD 0.3): a 1.0 s dwell is decided on roughly ten samples. Head pose near-static throughout (yaw SD 1.26°, pitch SD 0.97°).")
fig.savefig(f"{OUT}/fig5_signal_quality.png"); plt.close(fig)

print("figures written to", OUT)
for f in sorted(os.listdir(OUT)): print("  ", f)
