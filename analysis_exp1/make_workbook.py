import os, math
import numpy as np, pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from common import (load, load_validation, in_cell, affine_fit, OUT, GRIDS, SW, SH)

os.makedirs(OUT, exist_ok=True)
XL = f"{OUT}/exp1_results.xlsx"
R, T, S = load()
V = load_validation()
D = T.dropna(subset=["Pred_x"]).copy()          # scored trials (3 void trials dropped)
C = S[S.phase == "capture"].copy()

# ── T1 per run ────────────────────────────────────────────────
T1 = R.copy()
T1["Err_over_tolerance"] = T1.Mean_error_deg / T1.Cell_tolerance_deg
T1["Resolvable"] = np.where(T1.Err_over_tolerance < 1, "yes", "NO")
T1["Void_trials"] = [int((T[(T.Session == r.Session) & (T.Grid == r.Grid)].Void_no_frames == "yes").sum())
                     for r in T1.itertuples()]
T1["Cell_w_pt"] = SW / T1.Cols
T1["Cell_h_pt"] = SH / T1.Rows
T1 = T1[["Session", "Session_stamp", "Grid", "Run_id", "Started_at", "Duration_s", "Rows", "Cols",
         "Cell_w_pt", "Cell_h_pt", "Trials", "Hits", "Accuracy_pct", "Mean_error_pt",
         "Mean_error_deg", "Cell_tolerance_deg", "Err_over_tolerance", "Resolvable",
         "Void_trials", "Frames_logged", "Sample_rate_Hz", "Points_per_degree",
         "Virtual_tz_pt", "Prediction_source", "Blink_gate"]].round(3)

# ── T2 every trial ────────────────────────────────────────────
T2 = T.copy()
T2["Under_tolerance"] = np.where(T2.Err_deg < T2.Cell_tolerance_deg, "yes", "no")
T2.loc[T2.Void_no_frames == "yes", "Under_tolerance"] = ""
T2 = T2[["Session", "Grid", "Run_id", "Trial", "Cell_idx", "Row", "Col", "Cell_cx", "Cell_cy",
         "Pred_x", "Pred_y", "Dev_x_pt", "Dev_y_pt", "Err_pt", "Err_deg", "Hit",
         "Under_tolerance", "N_samples", "Void_no_frames", "Head_yaw_deg", "Head_pitch_deg",
         "Head_tz_mm"]].round(3)

# ── T3 the accuracy-versus-resolution curve ───────────────────
t3 = []
for g in GRIDS:
    sub = T[T.Grid == g]; sc = D[D.Grid == g]
    rows_, cols_ = sub.Row.max() + 1, sub.Col.max() + 1
    w, h = SW / cols_, SH / rows_
    tol = sub.Cell_tolerance_deg.mean()
    t3.append(dict(Grid=g, Rows=rows_, Cols=cols_, Cells=rows_ * cols_,
                   Cell_w_pt=round(w, 1), Cell_h_pt=round(h, 1),
                   Half_min_dim_pt=round(min(w, h) / 2, 1),
                   Cell_tolerance_deg=round(tol, 3),
                   Trials=len(sub), Hits=int(sub.Hit.sum()),
                   Accuracy_pct=round(100 * sub.Hit.mean(), 1),
                   Mean_error_pt=round(sc.Err_pt.mean(), 2),
                   Mean_error_deg=round(sc.Err_deg.mean(), 3),
                   Median_error_deg=round(sc.Err_deg.median(), 3),
                   P90_error_deg=round(sc.Err_deg.quantile(0.9), 3),
                   Err_over_tolerance=round(sc.Err_deg.mean() / tol, 3),
                   Trials_under_tolerance_pct=round(100 * (sc.Err_deg < sc.Cell_tolerance_deg).mean(), 1),
                   Void_trials=int((sub.Void_no_frames == "yes").sum()),
                   Verdict=("resolvable" if sc.Err_deg.mean() < tol else "NOT resolvable")))
T3 = pd.DataFrame(t3)

# ── T4 per-cell hit rate maps, one block per grid ─────────────
def cellmap(g, value, agg="mean", dec=2):
    sub = T[T.Grid == g]
    m = sub.pivot_table(index="Row", columns="Col", values=value, aggfunc=agg).round(dec)
    m.index = [f"row {i}" for i in m.index]
    m.columns = [f"col {i}" for i in m.columns]
    m["Row mean"] = m.mean(axis=1).round(dec)
    return m.reset_index().rename(columns={"index": "Row"})

M_HIT = {g: cellmap(g, "Hit", "mean", 2) for g in GRIDS}
M_ERR = {g: cellmap(g, "Err_deg", "mean", 2) for g in GRIDS}
def stack(maps, label):
    out = []
    for g in GRIDS:
        m = maps[g].copy(); m.insert(0, "Grid", g); out.append(m)
    df = pd.concat(out, ignore_index=True)
    df.columns = ["Grid", "Row"] + list(df.columns[2:])
    return df
T4a, T4b = stack(M_HIT, "hit"), stack(M_ERR, "err")

# ── T5 vertical structure — the row marginals ─────────────────
t5 = []
for g in GRIDS:
    sub = T[T.Grid == g]; sc = D[D.Grid == g]
    for r in sorted(sub.Row.unique()):
        a, b = sub[sub.Row == r], sc[sc.Row == r]
        t5.append(dict(Grid=g, Row=int(r), Target_y_pt=round(a.Cell_cy.iloc[0], 1),
                       Trials=len(a), Hits=int(a.Hit.sum()),
                       Accuracy_pct=round(100 * a.Hit.mean(), 1),
                       Mean_error_deg=round(b.Err_deg.mean(), 3),
                       Bias_x_pt=round(b.Dev_x_pt.mean(), 1),
                       Bias_y_pt=round(b.Dev_y_pt.mean(), 1),
                       Void_trials=int((a.Void_no_frames == "yes").sum())))
T5 = pd.DataFrame(t5)

# pooled across grids, by absolute screen band
bands = [0, 150, 300, 450, 600, 780]
t5b = []
for lo, hi in zip(bands[:-1], bands[1:]):
    a = T[(T.Cell_cy > lo) & (T.Cell_cy <= hi)]
    b = D[(D.Cell_cy > lo) & (D.Cell_cy <= hi)]
    cs = C[(C.target_y > lo) & (C.target_y <= hi)]
    t5b.append(dict(Screen_band_pt=f"{lo}-{hi}", Trials=len(a), Hits=int(a.Hit.sum()),
                    Accuracy_pct=round(100 * a.Hit.mean(), 1),
                    Mean_error_deg=round(b.Err_deg.mean(), 3),
                    Bias_x_pt=round(b.Dev_x_pt.mean(), 1),
                    Bias_y_pt=round(b.Dev_y_pt.mean(), 1),
                    Frame_containment_pct=round(100 * cs.in_cell.mean(), 1),
                    Frames=len(cs)))
T5b = pd.DataFrame(t5b)

# ── T6 calibration geometry: per-axis gain and offset ─────────
t6 = []
for s in ["S1", "S2", "S3"]:
    for scope, sub in [("all grids", D[D.Session == s])] + \
                      [(g, D[(D.Session == s) & (D.Grid == g)]) for g in GRIDS]:
        # forward model: how the estimate actually travels as the target moves
        fx, cxi = np.polyfit(sub.Cell_cx, sub.Pred_x, 1)
        fy, cyi = np.polyfit(sub.Cell_cy, sub.Pred_y, 1)
        # inverse fit: the correction that would undo it (this is what T7 applies)
        mx, bx = np.polyfit(sub.Pred_x, sub.Cell_cx, 1)
        my, by = np.polyfit(sub.Pred_y, sub.Cell_cy, 1)
        t6.append(dict(Session=s, Scope=scope, N_trials=len(sub),
                       X_gain_observed=round(fx, 3), X_intercept_pt=round(cxi, 1),
                       Y_gain_observed=round(fy, 3), Y_intercept_pt=round(cyi, 1),
                       Mean_bias_x_pt=round(sub.Dev_x_pt.mean(), 1),
                       Mean_bias_y_pt=round(sub.Dev_y_pt.mean(), 1),
                       X_gain_correction=round(mx, 3), X_offset_correction_pt=round(bx, 1),
                       Y_gain_correction=round(my, 3), Y_offset_correction_pt=round(by, 1),
                       r_x=round(float(np.corrcoef(sub.Pred_x, sub.Cell_cx)[0, 1]), 3),
                       r_y=round(float(np.corrcoef(sub.Pred_y, sub.Cell_cy)[0, 1]), 3)))
T6 = pd.DataFrame(t6)

# ── T7 counterfactuals: what a post-hoc correction recovers ───
def hitrate(px, py, t): return float(in_cell(px, py, t).mean())

t7 = []
for g in GRIDS:
    sub = D[D.Grid == g].reset_index(drop=True)
    n_all = int((T.Grid == g).sum())
    raw_all = 100 * T[T.Grid == g].Hit.mean()
    raw = 100 * sub.Hit.mean()
    ox = sub.Pred_x - sub.groupby("Session").Dev_x_pt.transform("mean")
    oy = sub.Pred_y - sub.groupby("Session").Dev_y_pt.transform("mean")
    off = 100 * hitrate(ox, oy, sub)
    ax = np.zeros(len(sub)); ay = np.zeros(len(sub))
    cvx = np.zeros(len(sub)); cvy = np.zeros(len(sub))
    for s, idx in sub.groupby("Session").groups.items():
        q = sub.loc[idx]
        px, py, *_ = affine_fit(q, q)
        ax[idx], ay[idx] = px, py
        tr = D[(D.Session == s) & (D.Grid != g)]
        px, py, *_ = affine_fit(tr, q)
        cvx[idx], cvy[idx] = px, py
    t7.append(dict(Grid=g, Scored_trials=len(sub), All_trials=n_all,
                   As_run_pct=round(raw_all, 1),
                   Scored_only_pct=round(raw, 1),
                   Minus_session_offset_pct=round(off, 1),
                   Minus_session_affine_pct=round(100 * hitrate(ax, ay, sub), 1),
                   Held_out_affine_pct=round(100 * hitrate(cvx, cvy, sub), 1),
                   Gain_offset_pp=round(off - raw, 1),
                   Gain_affine_pp=round(100 * hitrate(ax, ay, sub) - raw, 1),
                   Gain_held_out_pp=round(100 * hitrate(cvx, cvy, sub) - raw, 1)))
T7 = pd.DataFrame(t7)

# per session x grid held-out detail
t7b = []
for (s, g), sub in D.groupby(["Session", "Grid"]):
    tr = D[(D.Session == s) & (D.Grid != g)]
    px, py, mx, bx, my, by = affine_fit(tr, sub)
    e = np.hypot(px - sub.Cell_cx, py - sub.Cell_cy) / sub.Points_per_degree
    t7b.append(dict(Session=s, Grid=g, Trials=len(sub),
                    Raw_accuracy_pct=round(100 * sub.Hit.mean(), 1),
                    Held_out_accuracy_pct=round(100 * hitrate(px, py, sub), 1),
                    Raw_error_deg=round(sub.Err_deg.mean(), 3),
                    Held_out_error_deg=round(float(e.mean()), 3),
                    Fitted_x_gain=round(mx, 3), Fitted_y_gain=round(my, 3),
                    Fit_trials=len(tr)))
T7b = pd.DataFrame(t7b)

# ── T8 error budget ───────────────────────────────────────────
def mdeg(dx, dy, ppd): return float((np.hypot(dx, dy) / ppd).mean())
t8 = []
for g in GRIDS:
    sub = D[D.Grid == g].reset_index(drop=True)
    raw = mdeg(sub.Dev_x_pt, sub.Dev_y_pt, sub.Points_per_degree)
    ox = sub.Dev_x_pt - sub.groupby("Session").Dev_x_pt.transform("mean")
    oy = sub.Dev_y_pt - sub.groupby("Session").Dev_y_pt.transform("mean")
    off = mdeg(ox, oy, sub.Points_per_degree)
    ax = np.zeros(len(sub)); ay = np.zeros(len(sub))
    for s, idx in sub.groupby("Session").groups.items():
        q = sub.loc[idx]
        px, py, *_ = affine_fit(q, q)
        ax[idx], ay[idx] = px - q.Cell_cx, py - q.Cell_cy
    aff = mdeg(ax, ay, sub.Points_per_degree)
    cs = C[C.Grid == g]
    mu = cs.groupby(["Session", "trial"])[["pred_x", "pred_y"]].transform("mean")
    scat = math.hypot((cs.pred_x - mu.pred_x).std(), (cs.pred_y - mu.pred_y).std())
    t8.append(dict(Grid=g, As_scored_deg=round(raw, 3),
                   Minus_session_offset_deg=round(off, 3),
                   Minus_session_affine_deg=round(aff, 3),
                   Within_trial_scatter_deg=round(scat / cs.ppd.mean(), 3),
                   Offset_share_pct=round(100 * (1 - (off / raw) ** 2), 1),
                   Affine_share_pct=round(100 * (1 - (aff / raw) ** 2), 1),
                   Residual_deg=round(aff, 3)))
T8 = pd.DataFrame(t8)

# ── T9 dwell versus capture, and settling within capture ──────
t9 = []
for ph in ("dwell", "capture"):
    a = S[S.phase == ph]
    t9.append(dict(Phase=ph, Frames=len(a),
                   Mean_error_pt=round(a.err_pt.mean(), 2),
                   Mean_error_deg=round((a.err_pt / a.ppd).mean(), 3),
                   Containment_pct=round(100 * a.in_cell.mean(), 1)))
T9 = pd.DataFrame(t9)

edges = np.arange(0.0, 3.5, 0.5)
t9b = []
for lo, hi in zip(edges[:-1], edges[1:]):
    a = S[(S.t_rel >= lo) & (S.t_rel < hi)]
    if len(a) < 10: continue
    t9b.append(dict(Bin_start_s=round(float(lo), 2), Bin_end_s=round(float(hi), 2),
                    Frames=len(a),
                    Dwell_share_pct=round(100 * (a.phase == "dwell").mean(), 1),
                    Mean_error_pt=round(a.err_pt.mean(), 2),
                    Mean_error_deg=round((a.err_pt / a.ppd).mean(), 3),
                    Containment_pct=round(100 * a.in_cell.mean(), 1)))
T9b = pd.DataFrame(t9b)

# ── T10 head pose and stability covariates ────────────────────
def unwrap(a): return np.degrees(np.unwrap(np.radians(np.asarray(a, float))))
t10 = []
for (s, g), sub in S.groupby(["Session", "Grid"]):
    yaw, pit = unwrap(sub.head_yaw_deg), unwrap(sub.head_pitch_deg)
    rol = unwrap(sub.head_roll_deg)
    t10.append(dict(Session=s, Grid=g, Frames=len(sub),
                    Head_yaw_SD_deg=round(float(yaw.std()), 3),
                    Head_pitch_SD_deg=round(float(pit.std()), 3),
                    Head_roll_SD_deg=round(float(rol.std()), 3),
                    Head_tz_mean_mm=round(sub.head_tz_mm.mean(), 1),
                    Head_tz_SD_mm=round(sub.head_tz_mm.std(), 2),
                    r_absPitch_vs_error=round(float(np.corrcoef(np.abs(pit - pit.mean()), sub.err_pt)[0, 1]), 3),
                    r_absYaw_vs_error=round(float(np.corrcoef(np.abs(yaw - yaw.mean()), sub.err_pt)[0, 1]), 3),
                    Blink_held_frames=int(sub.blink_held.sum()),
                    Mean_EAR=round(sub.ear.mean(), 4)))
T10 = pd.DataFrame(t10)

# ── T11 what grid this pipeline can actually carry ────────────
p50 = float(D.Err_pt.median()); p90 = float(D.Err_pt.quantile(0.9))
aff_pt = []
for s, sub in D.groupby("Session"):
    px, py, *_ = affine_fit(sub, sub)
    aff_pt += list(np.hypot(px - sub.Cell_cx, py - sub.Cell_cy))
a50, a90 = float(np.median(aff_pt)), float(np.percentile(aff_pt, 90))
t11 = []
for cols_, rows_, note in ((3, 3, "run here"), (4, 4, "run here"), (4, 5, "run here"),
                           (3, 5, "the as-run median's limit"), (4, 6, ""),
                           (3, 4, "Experiment 3's word grid"), (4, 9, "the corrected median's limit"),
                           (9, 4, ""), (9, 9, "the old build's failure case")):
    w, h = SW / cols_, SH / rows_
    half = min(w, h) / 2
    t11.append(dict(Grid=f"{rows_}x{cols_}", Rows=rows_, Cols=cols_,
                    Max_error_pt_for_this_grid=round(half, 1),
                    Cell_w_pt=round(w, 1), Cell_h_pt=round(h, 1),
                    Half_min_dim_pt=round(half, 1),
                    Clears_median_as_run=("yes" if half > p50 else "NO"),
                    Clears_90th_as_run=("yes" if half > p90 else "NO"),
                    Clears_median_affine=("yes" if half > a50 else "NO"),
                    Clears_90th_affine=("yes" if half > a90 else "NO"),
                    Note=note))
T11 = pd.DataFrame(t11)

def widest(limit):
    """Finest grid whose half-cell clears `limit` in BOTH dimensions."""
    return f"{int(SH // (2 * limit))} rows x {int(SW // (2 * limit))} cols"
SAFE_RAW, SAFE_AFF = widest(p50), widest(a50)
SAFE_RAW90, SAFE_AFF90 = widest(p90), widest(a90)

# ── T12 the 9-dot validation that preceded session S2 ─────────
T12 = V.round(3)

# ── Summary ───────────────────────────────────────────────────
g3, g4, g5 = (T3[T3.Grid == g].iloc[0] for g in GRIDS)
c3, c4, c5 = (T7[T7.Grid == g].iloc[0] for g in GRIDS)
b3, b4, b5 = (T8[T8.Grid == g].iloc[0] for g in GRIDS)
SUMMARY = pd.DataFrame([
 ("Sessions analysed", 3, "20260826_122215, 20260826_221940, 20260827_112714"),
 ("Runs analysed", 9, "3 grids (3x3, 4x4, 5x4) x 3 sessions"),
 ("Trials", len(T), f"{len(D)} scored, {len(T)-len(D)} void (no frames captured)"),
 ("Frames logged", len(S), f"{len(C)} in the scored capture window, at {R.Sample_rate_Hz.mean():.1f} Hz mean"),
 ("Protocol", "1 s dwell + 2 s capture", "shuffled cell order, one visit per cell, stationary"),
 ("", "", ""),
 ("THE ACCURACY-VERSUS-RESOLUTION CURVE", "", ""),
 ("3x3", f"{g3.Accuracy_pct:.1f} %", f"error {g3.Mean_error_deg:.2f}° vs tolerance {g3.Cell_tolerance_deg:.2f}° — ratio {g3.Err_over_tolerance:.2f}"),
 ("4x4", f"{g4.Accuracy_pct:.1f} %", f"error {g4.Mean_error_deg:.2f}° vs tolerance {g4.Cell_tolerance_deg:.2f}° — ratio {g4.Err_over_tolerance:.2f}"),
 ("5x4", f"{g5.Accuracy_pct:.1f} %", f"error {g5.Mean_error_deg:.2f}° vs tolerance {g5.Cell_tolerance_deg:.2f}° — ratio {g5.Err_over_tolerance:.2f}"),
 ("The crossing point", "between 3x3 and 4x4", "mean error passes the cell tolerance at ~2.5-2.9°; 4x4 is the last usable density as run"),
 ("", "", ""),
 ("THE HEADLINE", "", ""),
 ("Error is flat across grids", f"{b3.As_scored_deg:.2f}° / {b4.As_scored_deg:.2f}° / {b5.As_scored_deg:.2f}°", "3x3 and 4x4 are identical; only 5x4 is worse, and for a reason (below)"),
 ("After a per-session affine", f"{b3.Minus_session_affine_deg:.2f}° / {b4.Minus_session_affine_deg:.2f}° / {b5.Minus_session_affine_deg:.2f}°", "ALL THREE GRIDS LAND ON THE SAME ~1.4° — the density never degraded the estimator"),
 ("What that means", "a calibration-geometry limit, not a resolution limit", "the falling curve is an uncorrected affine error being divided by a shrinking cell"),
 ("Matches Experiment 2", f"{T8.Minus_session_affine_deg.mean():.2f}° corrected", "Experiment 2 measured 1.43° RMS scatter and a 1.27° per-cell floor — the same number, from a different task"),
 ("", "", ""),
 ("WHAT A CORRECTION RECOVERS", "", ""),
 ("5x4, as run", f"{c5.As_run_pct:.1f} %", "the collapse"),
 ("5x4, minus a session offset", f"{c5.Minus_session_offset_pct:.1f} %", f"+{c5.Gain_offset_pp:.0f} pp from two numbers"),
 ("5x4, minus a session affine", f"{c5.Minus_session_affine_pct:.1f} %", f"+{c5.Gain_affine_pp:.0f} pp — in-sample ceiling, four parameters per axis pair"),
 ("5x4, held-out affine", f"{c5.Held_out_affine_pct:.1f} %", "fitted on the SAME session's other two grids — the honest number"),
 ("4x4, held-out affine", f"{c4.Held_out_affine_pct:.1f} %", f"from {c4.Scored_only_pct:.1f} %"),
 ("", "", ""),
 ("WHY: THE VERTICAL AXIS", "", ""),
  ("Vertical offset", f"{D.Dev_y_pt.mean():+.0f} pt in every session", f"S1 {D[D.Session=='S1'].Dev_y_pt.mean():+.0f}, S2 {D[D.Session=='S2'].Dev_y_pt.mean():+.0f}, S3 {D[D.Session=='S3'].Dev_y_pt.mean():+.0f} — the whole map sits high"),
 ("Vertical gain (observed)", " / ".join(f"{v:.2f}" for v in T6[T6.Scope=='all grids'].Y_gain_observed), "S1 / S2 / S3 — S3's estimate travels only 77 % of the way down the screen"),
 ("Horizontal gain (observed)", " / ".join(f"{v:.2f}" for v in T6[T6.Scope=='all grids'].X_gain_observed), "the estimate over-travels horizontally in all three sessions"),
 ("Accuracy, top band (0-150 pt)", f"{T5b.iloc[0].Accuracy_pct:.1f} %", f"error {T5b.iloc[0].Mean_error_deg:.2f}°, vertical bias {T5b.iloc[0].Bias_y_pt:+.0f} pt"),
 ("Accuracy, bottom band (600-780)", f"{T5b.iloc[-1].Accuracy_pct:.1f} %", f"error {T5b.iloc[-1].Mean_error_deg:.2f}°, vertical bias {T5b.iloc[-1].Bias_y_pt:+.0f} pt"),
 ("Sign of the bias", "negative throughout", "the prediction sits ABOVE the target, and further above the lower it goes"),
 ("Bottom row of the 5x4", f"{T5[(T5.Grid=='5x4')&(T5.Row==4)].iloc[0].Accuracy_pct:.0f} % accurate", f"{int(T5[(T5.Grid=='5x4')&(T5.Row==4)].iloc[0].Void_trials)} of 12 trials captured ZERO frames"),
 ("", "", ""),
 ("PROTOCOL VALIDATION", "", ""),
 ("Dwell frames", f"{T9.iloc[0].Mean_error_pt:.0f} pt", f"{T9.iloc[0].Containment_pct:.0f} % containment — mid-saccade, correctly excluded"),
 ("Capture frames", f"{T9.iloc[1].Mean_error_pt:.0f} pt", f"{T9.iloc[1].Containment_pct:.0f} % containment — {T9.iloc[0].Mean_error_pt/T9.iloc[1].Mean_error_pt:.1f}x better than dwell"),
 ("Is 1 s of dwell enough?", "yes", "error is flat at 82-84 pt across the whole 2 s capture window — no residual settling"),
 ("Within-trial scatter", f"{T8.Within_trial_scatter_deg.mean():.2f}°", "the frame-to-frame noise inside a fixation — small next to the trial-level error"),
 ("Head motion", f"yaw SD {T10.Head_yaw_SD_deg.mean():.1f}°, pitch SD {T10.Head_pitch_SD_deg.mean():.1f}°", f"|r| with error <= {T10[['r_absYaw_vs_error','r_absPitch_vs_error']].abs().max().max():.2f} — near-static, explains almost none of it"),
 ("Blink-gated frames", int(S.blink_held.sum()), "the gate never fired inside a logged frame in any of the nine runs"),
 ("", "", ""),
 ("CALIBRATION VALIDATION", "", ""),
 ("9-dot validation (S2 only)", f"{V.groupby('Run').Err_deg.mean().mean():.2f}° mean error", "verdict 'good' on both runs, all 9 dots confirmed"),
 ("...and yet S2's 5x4 scored", f"{T1[(T1.Session=='S2')&(T1.Grid=='5x4')].iloc[0].Accuracy_pct:.0f} %", "A PASSING VALIDATION DOES NOT PREDICT A PASSING 5x4 — see Caveats"),
 ("", "", ""),
 ("GRID DESIGN"                , "", ""),
 ("Per-trial error, median / 90th", f"{p50:.0f} pt / {p90:.0f} pt", "as run"),
 ("After an affine correction", f"{a50:.0f} pt / {a90:.0f} pt", "the same trials, re-scored"),
 ("Finest grid at the median, as run", SAFE_RAW, f"half-cell must clear the {p50:.0f} pt median error in both dimensions"),
 ("Finest grid at the median, corrected", SAFE_AFF, f"the same test against the corrected {a50:.0f} pt median — a whole extra column"),
 ("Finest grid at the 90th pct", f"{SAFE_RAW90} as run, {SAFE_AFF90} corrected", f"{p90:.0f} pt vs {a90:.0f} pt — the conservative bar, and why 4x4 still misses a quarter of its trials"),
], columns=["Metric", "Value", "Note"])

CAVEATS = pd.DataFrame([
 ("READ THIS FIRST", "The accuracy-versus-resolution curve in this workbook is NOT a resolution result. Mean error is 2.55 deg at 3x3 and 2.57 deg at 4x4 — identical — and after a four-parameter per-session affine correction all three grids land within 0.08 deg of each other at ~1.4 deg. The falling hit rate is a fixed, correctable calibration-geometry error being divided by a shrinking cell. Quote the curve as 'accuracy at a given cell tolerance under the shipped calibration', never as 'the tracker degrades at finer grids'."),
 ("The affine numbers are two different claims", "'Minus session affine' is fitted and evaluated on the same trials — it is a CEILING, not a result. 'Held-out affine' fits on the session's other two grids and applies it unseen; that is the honest recoverable gain (5x4: 33 % as run -> 61 % held out; 4x4: 73 % -> 85 %). Sheet 'T7b Held-out detail' has the per-session breakdown, including the two cases where the correction made things worse."),
 ("Three void trials", "S1's 5x4 run has three trials (cells 16, 18, 19 — all in the bottom row) with n_samples = 0: no prediction was produced at all. They are scored as misses in the as-run accuracy, matching the app, and are dropped from every error statistic. They are the extreme case of the vertical failure, not a separate defect."),
 ("Passing validation, failing grid", "Session S2 is preceded by two 9-dot validations, both 'good' (1.80 deg and 1.64 deg mean error, worst dot 2.97 deg). That session's 5x4 run then scored 30 %. The validation threshold is not tied to the cell tolerance of the grid about to be run — 1.8 deg mean error with a 2.97 deg worst dot cannot support a 2.34 deg tolerance. Gate on 'worst dot < cell tolerance', not on a global mean."),
 ("Only S2 has a validation at all", "S1 and S3 were exported with no validation bundle, so their calibration quality cannot be checked. S3 is the worst session (5x4 at 15 %) and also has the most compressed vertical gain (0.78 vs 0.98-1.01) — plausibly a bad fit that no gate caught."),
 ("Sessions are not participants", "The export records no participant identifier. These are three sessions on three occasions; treat them as three calibrations, not three people. Every counterfactual in this workbook is fitted WITHIN a session for that reason."),
 ("The blink gate differs across sessions", "S1 ran blinkEAR=ratio0.62/floor0.13 with no maxGated cap; S2 and S3 ran ratio0.55/floor0.12/maxGated5. No logged frame in any run was blink-held, so the gate did not shape the scored data — but S1's three void trials are consistent with frames being dropped before logging."),
 ("Degrees are virtual", "Angular values use the calibration's fitted |tz| (21.5-25.7 pt per degree across sessions), not a measured viewing distance. Comparable within a session; not across sessions, and not against another rig."),
 ("One visit per cell", "Each cell is tested exactly once per run, so a per-cell hit rate is 3 trials pooled across sessions and a per-cell error has n = 3. The per-cell maps show structure, not per-cell precision. Row and band marginals (T5, T5b) are the statistics to quote."),
 ("Trial n is unbalanced by grid", "27 trials at 3x3, 48 at 4x4, 60 at 5x4 — the grids differ in cell count by design. Do not pool the three grids into a single accuracy figure."),
 ("A tenth run exists but was not bundled", "session_20260826_221940/exp1/trials_all_runs.csv contains a fourth block, 'Run 1 — 3x3 — 16:51:15', with no run directory, no meta.json and no samples. It scored 6/9. It is excluded here because no calibration scale factor or frame data accompanies it."),
 ("Provenance", "Computed from each run bundle's trials.csv (trial block) and samples.csv (schema v4), plus meta.json for the calibration scale factor and the shipped summary. `<ROOT>/exp1` is byte-identical to session_20260826_221940/exp1 and is loaded once. No re-scoring or exclusion unless a sheet says so."),
], columns=["Topic", "Detail"])

SHEETS = [
 ("Summary", SUMMARY), ("Caveats", CAVEATS),
 ("T1 Runs", T1), ("T2 Trials", T2),
 ("T3 Resolution curve", T3),
 ("T4a Cell hit maps", T4a), ("T4b Cell error maps", T4b),
 ("T5 Rows", T5), ("T5b Screen bands", T5b),
 ("T6 Calibration geometry", T6),
 ("T7 Counterfactuals", T7), ("T7b Held-out detail", T7b),
 ("T8 Error budget", T8),
 ("T9 Dwell vs capture", T9), ("T9b Settling", T9b),
 ("T10 Head pose", T10),
 ("T11 Grid resolvability", T11),
 ("T12 9-dot validation", T12),
]
with pd.ExcelWriter(XL, engine="openpyxl") as w:
    for name, df in SHEETS:
        df.to_excel(w, sheet_name=name[:31], index=False)

wb = load_workbook(XL)
HDR_FILL = PatternFill("solid", fgColor="12161B")
HDR_FONT = Font(color="FFFFFF", bold=True, size=10)
BAND = PatternFill("solid", fgColor="F2F5F8")
SEC = PatternFill("solid", fgColor="E3EDFA")
thin = Side(style="thin", color="D6DBE0")
LONG = {"Note", "Detail", "Topic", "Metric", "Scope", "Verdict", "Blink_gate", "Started_at",
        "Protocol", "Resolvable", "Screen_band_pt", "Prediction_source"}
for ws in wb.worksheets:
    ws.freeze_panes = "A2"; ws.auto_filter.ref = ws.dimensions
    for c in ws[1]:
        c.fill = HDR_FILL; c.font = HDR_FONT
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 30
    for col in ws.columns:
        L = max((len(str(c.value)) for c in col if c.value is not None), default=8)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max(L + 3, 11), 52)
    for i, row in enumerate(ws.iter_rows(min_row=2)):
        for c in row:
            c.border = Border(bottom=thin)
            if isinstance(c.value, float): c.number_format = "0.00"
            if i % 2: c.fill = BAND
    for c in ws[1]:
        if str(c.value) in LONG:
            ws.column_dimensions[c.column_letter].width = 46
            for r in ws.iter_rows(min_row=2, min_col=c.column, max_col=c.column):
                r[0].alignment = Alignment(wrap_text=True, vertical="top")
ws = wb["Summary"]
ws.column_dimensions["A"].width = 34; ws.column_dimensions["B"].width = 30
for row in ws.iter_rows(min_row=2, max_col=1):
    v = row[0].value
    if isinstance(v, str) and v.isupper() and len(v) > 3:
        for c in ws[row[0].row]: c.fill = SEC; c.font = Font(bold=True, size=10)
ws = wb["Caveats"]
ws["A2"].font = Font(bold=True, color="B03A3A", size=11)
ws.column_dimensions["B"].width = 96
for r in ws.iter_rows(min_row=2, min_col=2, max_col=2):
    r[0].alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[r[0].row].height = 74
wb.save(XL)
print("workbook:", XL)
for n, d in SHEETS: print(f"   {n:26s} {d.shape[0]:4d} rows x {d.shape[1]} cols")
