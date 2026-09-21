import os, math
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from common import load, load_validation, in_cell, affine_fit, OUT, GRIDS, SW, SH

os.makedirs(OUT, exist_ok=True)
INK, INK2, INK3 = "#12161b", "#4c5763", "#78838f"
LINE = "#d6dbe0"
S1C, S2C, S3C = "#2a78d6", "#eb6834", "#1baf7a"
RAMP  = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
RAMPO = ["#fde5d8", "#fbc3a8", "#f79f76", "#f07a48", "#dd5a24", "#b34617"]
BLUES  = matplotlib.colors.LinearSegmentedColormap.from_list("b", RAMP)
ORANGE = matplotlib.colors.LinearSegmentedColormap.from_list("o", RAMPO)
GREENRED = matplotlib.colors.LinearSegmentedColormap.from_list(
    "gr", ["#c94f3d", "#e8b04b", "#f2efe6", "#7fc9a4", "#1baf7a"])

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200, "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 9,
    "axes.edgecolor": LINE, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9.5,
    "axes.grid": True, "grid.color": "#e6eaed", "grid.linewidth": .7,
    "xtick.color": INK3, "ytick.color": INK3, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.frameon": False, "legend.fontsize": 8.5,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
    "figure.facecolor": "white", "axes.facecolor": "white",
})
def tidy(ax, xgrid=False, ygrid=True):
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(LINE); ax.spines["bottom"].set_color(LINE)
    ax.xaxis.grid(xgrid); ax.yaxis.grid(ygrid); ax.set_axisbelow(True)
def cap(fig, t):
    fig.text(0.0, -0.045, t, ha="left", va="top", fontsize=7.8, color=INK3,
             wrap=True, transform=fig.transFigure)

R, T, S = load()
V = load_validation()
D = T.dropna(subset=["Pred_x"]).copy()
C = S[S.phase == "capture"].copy()
SESS = ["S1", "S2", "S3"]
SC = {"S1": S1C, "S2": S2C, "S3": S3C}
ACC = {g: 100 * T[T.Grid == g].Hit.mean() for g in GRIDS}
ERR = {g: D[D.Grid == g].Err_deg.mean() for g in GRIDS}
TOL = {g: T[T.Grid == g].Cell_tolerance_deg.mean() for g in GRIDS}

# ══ FIG 1 — the accuracy-versus-resolution curve and its crossing point ══
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.4), gridspec_kw={"wspace": 0.34})
x = np.arange(3)

ax = axs[0]; tidy(ax)
for s in SESS:
    v = [100 * T[(T.Session == s) & (T.Grid == g)].Hit.mean() for g in GRIDS]
    ax.plot(x, v, "-o", color=SC[s], lw=1.6, ms=5, label=s, zorder=3)
pooled = [ACC[g] for g in GRIDS]
ax.plot(x, pooled, "-o", color=INK, lw=2.6, ms=7, label="pooled", zorder=4)
for i, v in enumerate(pooled):
    ax.text(i, v + 4.5, f"{v:.0f} %", ha="center", fontsize=9, color=INK, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(GRIDS)
ax.set_ylim(0, 108); ax.set_ylabel("Cell accuracy (%)")
ax.legend(loc="lower left", ncol=2)
ax.set_title("A · Accuracy vs grid density", loc="left")

ax = axs[1]; tidy(ax)
w = 0.38
e = [ERR[g] for g in GRIDS]; t = [TOL[g] for g in GRIDS]
ax.bar(x - w/2, e, w, color=S2C, label="Mean error", zorder=3)
ax.bar(x + w/2, t, w, color=S1C, label="Cell tolerance", zorder=3)
for i in range(3):
    ax.text(i - w/2, e[i] + .09, f"{e[i]:.2f}°", ha="center", fontsize=8.2, color=INK)
    ax.text(i + w/2, t[i] + .09, f"{t[i]:.2f}°", ha="center", fontsize=8.2, color=INK)
    ax.text(i, 5.05, f"×{e[i]/t[i]:.2f}", ha="center", fontsize=9,
            color=(S2C if e[i] > t[i] else S3C), fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(GRIDS)
ax.set_ylim(0, 5.6); ax.set_ylabel("Degrees of visual angle")
ax.legend(loc="upper left", ncol=1, bbox_to_anchor=(0, 0.93))
ax.set_title("B · Error against budget", loc="left")

ax = axs[2]; tidy(ax, xgrid=True)
half = np.array([SW / int(g[2]) / 2 for g in GRIDS])
tolpt = np.array([TOL[g] * D.Points_per_degree.mean() for g in GRIDS])
for g, c in zip(GRIDS, [S1C, S2C, S3C]):
    sub = D[D.Grid == g]
    ax.scatter(sub.Err_deg, np.random.default_rng(0).normal(GRIDS.index(g), .10, len(sub)),
               s=16, color=c, alpha=.55, edgecolors="none", zorder=3)
    ax.plot([TOL[g], TOL[g]], [GRIDS.index(g) - .34, GRIDS.index(g) + .34],
            color=INK, lw=2, zorder=4)
    frac = 100 * (sub.Err_deg < sub.Cell_tolerance_deg).mean()
    ax.text(8.1, GRIDS.index(g), f"{frac:.0f} % under", ha="right", va="center",
            fontsize=8.4, color=INK, fontweight="bold")
ax.set_yticks(x); ax.set_yticklabels(GRIDS); ax.invert_yaxis()
ax.set_xlim(0, 8.3); ax.set_xlabel("Per-trial error (°) — black bar is that grid's cell tolerance")
ax.set_title("C · Every trial, scored", loc="left")

cap(fig, "Figure 1 — Experiment 1 re-run on the validated build: three sessions × three grids (3×3, 4×4, 5×4), one visit per cell in shuffled order, 1 s unscored dwell + 2 s scored capture, 135 trials. A trial is a hit when the capture-window mean prediction falls inside the highlighted cell; the cell tolerance is the angle subtended by half the smaller cell dimension — the error budget a prediction has before it lands in a neighbour. Accuracy falls 93 % - 73 % - 33 %, and mean error crosses the tolerance between 3×3 and 4×4. Panel B is the key comparison: mean error is 2.55° at 3×3 and 2.57° at 4×4 — unchanged — while the budget shrinks from 2.90° to 2.17°. The curve is being driven by the denominator.")
fig.savefig(f"{OUT}/fig1_resolution_curve.png"); plt.close(fig)

# ══ FIG 2 — per-cell maps, drawn to screen geometry ══
fig, axs = plt.subplots(1, 3, figsize=(11.5, 4.6), gridspec_kw={"wspace": 0.22})
for ax, g in zip(axs, GRIDS):
    sub = T[T.Grid == g]
    rows_, cols_ = sub.Row.max() + 1, sub.Col.max() + 1
    cw, ch = SW / cols_, SH / rows_
    for r in range(rows_):
        for c in range(cols_):
            cell = sub[(sub.Row == r) & (sub.Col == c)]
            hit = cell.Hit.mean()
            ax.add_patch(Rectangle((c * cw, r * ch), cw, ch, facecolor=GREENRED(hit),
                                   edgecolor="white", lw=1.6, zorder=1))
            ed = cell.Err_deg.mean()
            lbl = f"{100*hit:.0f}%\n{ed:.1f}°" if np.isfinite(ed) else f"{100*hit:.0f}%\n—"
            ax.text(c * cw + cw / 2, r * ch + ch / 2, lbl, ha="center", va="center",
                    fontsize=8.6, color=(INK if .25 < hit < .85 else "white"),
                    fontweight="bold", zorder=3)
    # where the predictions actually landed
    sc = D[D.Grid == g]
    ax.scatter(sc.Pred_x, sc.Pred_y, s=11, color=INK, alpha=.6, edgecolors="none", zorder=4)
    ax.set_xlim(0, SW); ax.set_ylim(SH, 0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values(): sp.set_color(LINE)
    n_void = int((sub.Void_no_frames == "yes").sum())
    ax.set_title(f"{g} — {ACC[g]:.0f} % accurate" + (f"  ({n_void} void trials)" if n_void else ""),
                 loc="left")
cap(fig, "Figure 2 — Per-cell hit rate (colour and top figure, 3 trials per cell pooled across sessions) with mean error below it, drawn to the 402 × 778 pt screen. Black dots are the 132 scored predictions. Failure is not scattered — it is a clean top-to-bottom gradient in every grid, and the 5×4's bottom row is almost entirely lost. Green is 100 % hit, red is 0 %. The three void trials in the 5×4 bottom row produced no prediction at all and are counted as misses.")
fig.savefig(f"{OUT}/fig2_cell_maps.png"); plt.close(fig)

# ══ FIG 3 — the vertical axis is the whole story ══
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.5), gridspec_kw={"wspace": 0.32})

ax = axs[0]; tidy(ax)
for s in SESS:
    sub = D[D.Session == s]
    ax.scatter(sub.Cell_cy, sub.Pred_y, s=14, color=SC[s], alpha=.55,
               edgecolors="none", zorder=3, label=s)
    m, b = np.polyfit(sub.Cell_cy, sub.Pred_y, 1)
    xs = np.array([0, SH])
    ax.plot(xs, m * xs + b, color=SC[s], lw=1.6, zorder=4)
    lx = {"S1": 770, "S2": 690, "S3": 610}[s]
    ax.text(lx, m * lx + b - 26, f"gain {m:.2f}", fontsize=8.2, color=SC[s],
            va="top", ha="right", fontweight="bold")
ax.plot([0, SH], [0, SH], color=INK3, ls="--", lw=1.1, zorder=2)
ax.text(120, 150, "perfect", fontsize=8, color=INK3, rotation=38)
ax.set_xlabel("Target y (pt)"); ax.set_ylabel("Predicted y (pt)")
ax.set_xlim(0, SH + 30); ax.set_ylim(-40, SH); ax.legend(loc="upper left")
ax.set_title("A · The estimate under-travels downward", loc="left")

ax = axs[1]; tidy(ax)
bands = [0, 150, 300, 450, 600, 780]
mid = [(a + b) / 2 for a, b in zip(bands[:-1], bands[1:])]
by = [D[(D.Cell_cy > a) & (D.Cell_cy <= b)].Dev_y_pt.mean() for a, b in zip(bands[:-1], bands[1:])]
bx = [D[(D.Cell_cy > a) & (D.Cell_cy <= b)].Dev_x_pt.mean() for a, b in zip(bands[:-1], bands[1:])]
ax.axhline(0, color=INK3, lw=1)
ax.plot(mid, by, "-o", color=S2C, lw=2, ms=6, label="vertical bias", zorder=4)
ax.plot(mid, bx, "-o", color=S1C, lw=1.6, ms=5, label="horizontal bias", zorder=3)
for m_, v in zip(mid, by):
    ax.text(m_, v - 9, f"{v:+.0f}", ha="center", fontsize=8, color=S2C)
ax.set_xlabel("Target y on screen (pt)"); ax.set_ylabel("Bias (pt) — negative is above target")
ax.set_ylim(-100, 40); ax.legend(loc="lower left")
ax.set_title("B · Bias grows the lower you look", loc="left")

ax = axs[2]; tidy(ax)
acc = [100 * T[(T.Cell_cy > a) & (T.Cell_cy <= b)].Hit.mean() for a, b in zip(bands[:-1], bands[1:])]
err = [D[(D.Cell_cy > a) & (D.Cell_cy <= b)].Err_deg.mean() for a, b in zip(bands[:-1], bands[1:])]
ax.bar(mid, acc, 120, color=S1C, zorder=3)
for m_, v in zip(mid, acc):
    ax.text(m_, v + 2, f"{v:.0f} %", ha="center", fontsize=8.4, color=INK, fontweight="bold")
ax.set_ylim(0, 104); ax.set_ylabel("Accuracy (%)", color=S1C)
ax.set_xlabel("Target y on screen (pt)")
ax2 = ax.twinx(); ax2.grid(False)
ax2.plot(mid, err, "-o", color=S2C, lw=2, ms=6, zorder=4)
ax2.set_ylabel("Mean error (°)", color=S2C); ax2.set_ylim(0, 5.4)
for sp in ("top",): ax2.spines[sp].set_visible(False)
ax.set_title("C · Half the screen carries the failure", loc="left")

cap(fig, "Figure 3 — Vertical structure, pooled across all three grids so the comparison is by absolute screen position rather than by grid. The predicted y tracks the target with r ≥ 0.96 but travels short of it, and the shortfall grows with depth down the screen: bias -27 pt in the top band, -79 pt in the bottom. Horizontal bias stays inside ±27 pt. The consequence is a hit rate that drops from 91 % in the top 150 pt to 45 % below 600 pt. Session S3's vertical gain of 0.77 is the extreme case — its estimate covers only three quarters of the screen's height, and its 5×4 run scored 15 %.")
fig.savefig(f"{OUT}/fig3_vertical_bias.png"); plt.close(fig)

# ══ FIG 4 — what a correction recovers ══
fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.8), gridspec_kw={"wspace": 0.34})

def hitrate(px, py, t): return 100 * float(in_cell(px, py, t).mean())
raw, off, aff, cv = [], [], [], []
budget = []
for g in GRIDS:
    sub = D[D.Grid == g].reset_index(drop=True)
    raw.append(100 * T[T.Grid == g].Hit.mean())
    ox = sub.Pred_x - sub.groupby("Session").Dev_x_pt.transform("mean")
    oy = sub.Pred_y - sub.groupby("Session").Dev_y_pt.transform("mean")
    off.append(hitrate(ox, oy, sub))
    ax_ = np.zeros(len(sub)); ay = np.zeros(len(sub))
    cx_ = np.zeros(len(sub)); cy_ = np.zeros(len(sub))
    dax = np.zeros(len(sub)); day = np.zeros(len(sub))
    for s, idx in sub.groupby("Session").groups.items():
        q = sub.loc[idx]
        px, py, *_ = affine_fit(q, q); ax_[idx], ay[idx] = px, py
        dax[idx], day[idx] = px - q.Cell_cx, py - q.Cell_cy
        tr = D[(D.Session == s) & (D.Grid != g)]
        px, py, *_ = affine_fit(tr, q); cx_[idx], cy_[idx] = px, py
    aff.append(hitrate(ax_, ay, sub)); cv.append(hitrate(cx_, cy_, sub))
    dox = sub.Dev_x_pt - sub.groupby("Session").Dev_x_pt.transform("mean")
    doy = sub.Dev_y_pt - sub.groupby("Session").Dev_y_pt.transform("mean")
    budget.append((float((np.hypot(sub.Dev_x_pt, sub.Dev_y_pt) / sub.Points_per_degree).mean()),
                   float((np.hypot(dox, doy) / sub.Points_per_degree).mean()),
                   float((np.hypot(dax, day) / sub.Points_per_degree).mean())))

ax = axs[0]; tidy(ax)
w = 0.2
for i, (v, c, l) in enumerate([(raw, INK3, "as run"), (off, "#9ec5f4", "− session offset"),
                               (cv, S1C, "− affine (held out)"), (aff, "#184f95", "− affine (in-sample)")]):
    ax.bar(x + (i - 1.5) * w, v, w, color=c, label=l, zorder=3)
for i, g in enumerate(GRIDS):
    ax.text(i - 1.5 * w, raw[i] + 2, f"{raw[i]:.0f}", ha="center", fontsize=7.6, color=INK)
    ax.text(i + 0.5 * w, cv[i] + 2, f"{cv[i]:.0f}", ha="center", fontsize=7.6,
            color=S1C, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(GRIDS); ax.set_ylim(0, 108)
ax.set_ylabel("Cell accuracy (%)")
ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.005), fontsize=7.6)
ax.set_title("A · 5×4 recovers from 33 % to 61 %", loc="left", pad=48)

ax = axs[1]; tidy(ax)
b0 = [b[0] for b in budget]; b1 = [b[1] for b in budget]; b2 = [b[2] for b in budget]
ax.bar(x, b0, 0.55, color="#f2d3c4", label="as scored", zorder=2)
ax.bar(x, b1, 0.55, color=S2C, label="− session offset", zorder=3)
ax.bar(x, b2, 0.55, color=INK, label="− session affine", zorder=4)
for i in range(3):
    ax.text(i, b0[i] + .09, f"{b0[i]:.2f}°", ha="center", fontsize=8.2, color=INK)
    ax.text(i, b2[i] / 2, f"{b2[i]:.2f}°", ha="center", va="center", fontsize=8.2,
            color="white", fontweight="bold")
ax.axhline(1.43, color=S3C, ls="--", lw=1.4, zorder=5, label="Exp 2 noise floor, 1.43° RMS")
ax.set_xticks(x); ax.set_xticklabels(GRIDS); ax.set_ylim(0, 5.2)
ax.set_ylabel("Mean error (°)")
ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(0.5, 1.005), fontsize=7.6)
ax.set_title("B · Corrected, all three grids agree", loc="left", pad=48)

ax = axs[2]; tidy(ax, xgrid=True, ygrid=False)
lbl, vals, cols = [], [], []
for s in SESS:
    for g in GRIDS:
        sub = D[(D.Session == s) & (D.Grid == g)]
        tr = D[(D.Session == s) & (D.Grid != g)]
        px, py, *_ = affine_fit(tr, sub)
        lbl.append(f"{s} {g}")
        vals.append((100 * sub.Hit.mean(), hitrate(px, py, sub)))
        cols.append(SC[s])
y = np.arange(len(lbl))
for i, (a, b) in enumerate(vals):
    ax.plot([a, b], [i, i], color=(S3C if b >= a else "#c94f3d"), lw=2.4, zorder=2,
            solid_capstyle="round")
    ax.scatter([a], [i], s=26, color=INK3, zorder=3)
    ax.scatter([b], [i], s=40, color=cols[i], zorder=4)
ax.set_yticks(y); ax.set_yticklabels(lbl, fontsize=8); ax.invert_yaxis()
ax.set_xlim(0, 108); ax.set_xlabel("Accuracy (%) — grey dot as run, coloured dot held-out corrected")
ax.set_title("C · Held out, run by run", loc="left", pad=48)

cap(fig, "Figure 4 — Counterfactual re-scoring. 'Session offset' subtracts one (x, y) constant per session; 'affine' fits a per-axis scale and offset per session. In-sample is a ceiling — it is fitted and evaluated on the same trials. Held-out is the honest number: the correction is fitted on that session's other two grids and applied unseen, lifting 5×4 from 33 % to 61 % and 4×4 from 73 % to 85 %. Panel B is the point of the whole experiment: once a per-session affine is removed, 3×3, 4×4 and 5×4 land at 1.42°, 1.34° and 1.38° — indistinguishable, and sitting on the 1.43° RMS noise floor Experiment 2 measured independently. Panel C shows the two runs where the held-out fit made things worse (S2 3×3, S3 4×4), the cost of extrapolating one grid's geometry onto another.")
fig.savefig(f"{OUT}/fig4_counterfactual.png"); plt.close(fig)

# ══ FIG 5 — protocol, signal quality, and the validation gap ══
fig, axs = plt.subplots(1, 4, figsize=(13.0, 3.1), gridspec_kw={"wspace": 0.40})

ax = axs[0]; tidy(ax)
edges = np.arange(0, 3.25, 0.25)
mids, errs, dsh = [], [], []
for a, b in zip(edges[:-1], edges[1:]):
    sub = S[(S.t_rel >= a) & (S.t_rel < b)]
    if len(sub) < 10: continue
    mids.append((a + b) / 2); errs.append(sub.err_pt.mean())
    dsh.append((sub.phase == "dwell").mean())
ax.axvspan(0, 1, color="#fdf0e9", zorder=0)
ax.plot(mids, errs, "-o", color=S2C, lw=2, ms=4.5, zorder=3)
ax.text(0.5, 270, "dwell\n(excluded)", ha="center", fontsize=8, color="#b34617")
ax.text(2.05, 270, "scored capture", ha="center", fontsize=8, color=INK2)
ax.axhline(np.mean([e for m, e in zip(mids, errs) if m > 1]), color=INK3, ls="--", lw=1.1)
ax.set_xlabel("Time since target appeared (s)"); ax.set_ylabel("Frame error (pt)")
ax.set_ylim(0, 330); ax.set_xlim(0, 3)
ax.set_title("A · 1 s of dwell is enough", loc="left")

ax = axs[1]; tidy(ax)
w = 0.38
tr = [D[D.Grid == g].Err_pt.mean() for g in GRIDS]
sc = []
for g in GRIDS:
    cs = C[C.Grid == g]
    mu = cs.groupby(["Session", "trial"])[["pred_x", "pred_y"]].transform("mean")
    sc.append(math.hypot((cs.pred_x - mu.pred_x).std(), (cs.pred_y - mu.pred_y).std()))
ax.bar(x - w/2, tr, w, color=S2C, label="trial error (accuracy)", zorder=3)
ax.bar(x + w/2, sc, w, color=S1C, label="within-trial scatter", zorder=3)
for i in range(3):
    ax.text(i - w/2, tr[i] + 2.5, f"{tr[i]:.0f}", ha="center", fontsize=8, color=INK)
    ax.text(i + w/2, sc[i] + 2.5, f"{sc[i]:.0f}", ha="center", fontsize=8, color=INK)
ax.set_xticks(x); ax.set_xticklabels(GRIDS); ax.set_ylim(0, 128)
ax.set_ylabel("Points"); ax.legend(loc="upper left", fontsize=7.8)
ax.set_title("B · The fixations are steady", loc="left")

ax = axs[2]; tidy(ax)
def unwrap(a): return np.degrees(np.unwrap(np.radians(np.asarray(a, float))))
yaws = [unwrap(S[S.Session == s].head_yaw_deg).std() for s in SESS]
pits = [unwrap(S[S.Session == s].head_pitch_deg).std() for s in SESS]
rols = [unwrap(S[S.Session == s].head_roll_deg).std() for s in SESS]
xs = np.arange(3); w = 0.26
ax.bar(xs - w, yaws, w, color=S1C, label="yaw", zorder=3)
ax.bar(xs,     pits, w, color=S2C, label="pitch", zorder=3)
ax.bar(xs + w, rols, w, color=S3C, label="roll", zorder=3)
ax.set_xticks(xs); ax.set_xticklabels(SESS); ax.set_ylabel("Head-pose SD (°)")
ax.set_ylim(0, 2.4); ax.legend(loc="upper left", ncol=3, fontsize=7.8)
ax.set_title("C · The head barely moves", loc="left")

ax = axs[3]; tidy(ax)
for run, c in zip(sorted(V.Run.unique()), [S1C, S2C]):
    sub = V[V.Run == run]
    ax.scatter(sub.Dot, sub.Err_deg, s=34, color=c, zorder=4, label=f"{run} ({sub.Err_deg.mean():.2f}° mean)")
    ax.plot(sub.Dot, sub.Err_deg, color=c, lw=1.2, alpha=.6, zorder=3)
ax.axhline(2.175, color="#c94f3d", ls="--", lw=1.5, zorder=2)
ax.text(9.4, 2.28, "5×4 cell tolerance", fontsize=7.6, color="#c94f3d", ha="right")
ax.set_xticks(range(1, 10)); ax.set_xlabel("Validation dot (top-left to bottom-right)")
ax.set_ylabel("Error (°)"); ax.set_ylim(0, 3.4); ax.legend(loc="upper left", fontsize=7.6)
ax.set_title("D · 'Good' validation, failing grid", loc="left")

cap(fig, "Figure 5 — Protocol and signal quality. A: frame error collapses the moment the dwell window ends and is flat at 82–84 pt for the whole 2 s capture — the 1 s dwell is sufficient and the exclusion of dwell frames is doing real work (188 pt vs 83 pt). B: within-trial scatter is 23–31 pt against a 60–105 pt trial error, so the fixations are steady and the error is offset, not jitter. C: head-pose SD is under 1.6° on every axis after angle unwrapping, and |r| with error never exceeds 0.26. D: the two 9-dot validations preceding session S2 both returned verdict 'good' at 1.80° and 1.64° mean error — yet 5 of their 18 dots exceed the 5×4 grid's own 2.17° cell tolerance, the worst at 2.97°, and that session's 5×4 run then scored 30 %. The gate should compare the worst dot against the tolerance of the grid about to be run.")
fig.savefig(f"{OUT}/fig5_protocol_quality.png"); plt.close(fig)

print("figures written to", OUT)
for f in sorted(os.listdir(OUT)):
    if f.endswith(".png"): print("   ", f)
