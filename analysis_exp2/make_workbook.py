import csv, os, json, math, statistics as st
import numpy as np, pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

SP  = "/private/tmp/claude-501/-Users-advait-Desktop-GazeTracking-Mobile-Gaze-tracking-main/babf1d27-2b25-429a-b4a4-f6616018cd2d/scratchpad/exp2"
OUT = "/Users/advait/Desktop/GazeTracking/Mobile_Gaze_tracking-main/analysis_exp2"
os.makedirs(OUT, exist_ok=True)
XL = f"{OUT}/exp2_results.xlsx"
RUNS = [
    ("Run 1", "/Users/advait/Downloads/EXp 2/session_20260826_200514/exp2/run1_4x4"),
    ("Run 4", "/Users/advait/Downloads/EXp 2/exp2_fixation_stability_4x4_run4_20260826_123603"),
    ("Run 5", f"{SP}/exp2_fixation_stability_4x4_run5_20260826_125428/exp2_fixation_stability_4x4_run5_20260826_125428"),
    ("Run 6", f"{SP}/exp2_fixation_stability_4x4_run6_20260826_125605/exp2_fixation_stability_4x4_run6_20260826_125605"),
    ("Run 7", f"{SP}/exp2_fixation_stability_4x4_run7_20260826_125739/exp2_fixation_stability_4x4_run7_20260826_125739"),
]
NAMES = [n for n, _ in RUNS]
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

# ── T1 per run ───────────────────────────────────────────────
t1 = []
for n in NAMES:
    s, m = SUM[n], METAS[n]
    b, d = s["bias_pt"], s["sd_pt"]
    t1.append(dict(Run=n, Started_at=m["started_at"], Duration_s=round(m["duration_s"],1),
                   Sample_rate_Hz=round(s["sample_rate_hz"],2), Frames_logged=m["sample_count"],
                   Scored_samples=s["n_samples"], Cells_scored=s["cells_scored"],
                   Mean_dev_pt=round(s["mean_dev_pt"],2), Mean_dev_deg=round(s["mean_dev_deg"],3),
                   Bias_pt=round(b,2), SD_pt=round(d,2), SD_x_pt=round(s["sd_x_pt"],2),
                   SD_y_pt=round(s["sd_y_pt"],2), RMS_pt=round(s["rms_pt"],2),
                   RMS_deg=round(s["rms_deg"],3), Containment_pct=round(s["containment_pct"],2),
                   Worst_cell_RMS_deg=round(s["worst_rms_deg"],3),
                   Bias_sq_share_pct=round(100*b*b/(b*b+d*d),1),
                   Points_per_degree=round(TZ[n],2),
                   Virtual_tz_pt=round(TZ[n]*57.2958,0),
                   Blink_gate=m["pipeline_tuning"].split("blinkEAR=")[1].split(" ")[0]))
T1 = pd.DataFrame(t1)

# ── T2 per cell per run (the raw 80 cell-runs) ───────────────
t2 = []
for n in NAMES:
    for r in CELLS[n]:
        t2.append(dict(Run=n, Cell_idx=int(r["cell_idx"]), Row=int(r["row"]), Col=int(r["col"]),
                       Presentation_order=int(r["order"]),
                       Target_x_pt=F(r["target_x_pt"]), Target_y_pt=F(r["target_y_pt"]),
                       Mean_dev_pt=F(r["mean_dev_pt"]), Mean_dev_deg=F(r["mean_dev_deg"]),
                       SD_pt=F(r["sd_pt"]), RMS_pt=F(r["rms_pt"]), RMS_deg=F(r["rms_deg"]),
                       Bias_pt=F(r["bias_pt"]), Bias_deg=F(r["bias_deg"]),
                       Containment_pct=F(r["containment_pct"]), N_samples=int(r["n_samples"]),
                       Outlier=("yes (RMS > 3 deg)" if F(r["rms_deg"]) > 3 else "")))
T2 = pd.DataFrame(t2)

# ── T3 per cell pooled + bias vectors ────────────────────────
BX, BY = {}, {}
for c in range(16):
    BX[c] = st.mean(st.mean(F(r["dev_x_pt"]) for r in SAMPS[n] if int(r["cell_idx"])==c) for n in NAMES)
    BY[c] = st.mean(st.mean(F(r["dev_y_pt"]) for r in SAMPS[n] if int(r["cell_idx"])==c) for n in NAMES)
t3 = []
for c in range(16):
    sub = T2[T2.Cell_idx == c]
    t3.append(dict(Cell_idx=c, Row=int(sub.Row.iloc[0]), Col=int(sub.Col.iloc[0]),
                   Mean_dev_deg=round(sub.Mean_dev_deg.mean(),3),
                   SD_across_runs_deg=round(sub.Mean_dev_deg.std(),3),
                   Mean_RMS_deg=round(sub.RMS_deg.mean(),3),
                   Worst_RMS_deg=round(sub.RMS_deg.max(),3),
                   Mean_bias_pt=round(sub.Bias_pt.mean(),2),
                   Bias_x_pt=round(BX[c],2), Bias_y_pt=round(BY[c],2),
                   Bias_magnitude_pt=round(math.hypot(BX[c],BY[c]),2),
                   Mean_containment_pct=round(sub.Containment_pct.mean(),2),
                   N_cell_runs=len(sub)))
T3 = pd.DataFrame(t3)
def grid(col, dec=2):
    g = T3.pivot(index="Row", columns="Col", values=col).round(dec)
    g.index = [f"row {i}" for i in g.index]; g.columns = [f"col {i}" for i in g.columns]
    g["Row mean"] = g.mean(axis=1).round(dec)
    return g.reset_index().rename(columns={"index":"Row"})
G_DEV = grid("Mean_dev_deg"); G_RMS = grid("Mean_RMS_deg")
G_CON = grid("Mean_containment_pct", 1); G_BY = grid("Bias_y_pt", 1)

# ── T4 row / column marginals ────────────────────────────────
t4 = []
for lbl, key in (("Row", "Row"), ("Col", "Col")):
    for i in range(4):
        sub = T2[T2[key] == i]
        clean = sub[sub.RMS_deg < 10]
        bs = [BY[c] for c in T3[T3[key]==i].Cell_idx] if lbl=="Row" else [BY[c] for c in T3[T3[key]==i].Cell_idx]
        xs = [BX[c] for c in T3[T3[key]==i].Cell_idx]
        t4.append(dict(Marginal=f"{lbl} {i}", Mean_dev_deg=round(sub.Mean_dev_deg.mean(),3),
                       Mean_RMS_deg=round(sub.RMS_deg.mean(),3),
                       Mean_RMS_deg_excl_outlier=round(clean.RMS_deg.mean(),3),
                       Mean_bias_x_pt=round(st.mean(xs),1), Mean_bias_y_pt=round(st.mean(bs),1),
                       Mean_containment_pct=round(sub.Containment_pct.mean(),1), N_cell_runs=len(sub)))
T4 = pd.DataFrame(t4)

# ── T5 settling ──────────────────────────────────────────────
edges = np.arange(0, 4.25, 0.5)
t5 = []
for n in NAMES:
    byc = {}
    for r in SAMPS[n]: byc.setdefault(r["cell_idx"], []).append(r)
    T, D = [], []
    for c, rs in byc.items():
        t0 = min(F(y["t_s"]) for y in rs)
        for x in rs: T.append(F(x["t_s"])-t0); D.append(F(x["dev_pt"]))
    T, D = np.array(T), np.array(D)
    for a, b in zip(edges[:-1], edges[1:]):
        m = (T >= a) & (T < b)
        if m.any():
            t5.append(dict(Run=n, Bin_start_s=round(float(a),2), Bin_end_s=round(float(b),2),
                           Mean_dev_pt=round(float(D[m].mean()),2),
                           Mean_dev_deg=round(float(D[m].mean()/TZ[n]),3),
                           N_samples=int(m.sum())))
T5 = pd.DataFrame(t5)
t5b = []
for n in NAMES:
    byc = {}
    for r in SAMPS[n]: byc.setdefault(r["cell_idx"], []).append(r)
    allv, keep, T, D = [], [], [], []
    for c, rs in byc.items():
        t0 = min(F(y["t_s"]) for y in rs)
        for x in rs:
            v = F(x["dev_pt"]); allv.append(v); T.append(F(x["t_s"])-t0); D.append(v)
            if F(x["t_s"])-t0 >= 1.0: keep.append(v)
    sl = float(np.polyfit(T, D, 1)[0])
    t5b.append(dict(Run=n, As_scored_pt=round(st.mean(allv),2),
                    As_scored_deg=round(st.mean(allv)/TZ[n],3),
                    Last_3s_pt=round(st.mean(keep),2),
                    Last_3s_deg=round(st.mean(keep)/TZ[n],3),
                    Improvement_pct=round(100*(1-st.mean(keep)/st.mean(allv)),1),
                    Slope_pt_per_s=round(sl,2)))
T5b = pd.DataFrame(t5b)

# ── T6 bias-correction counterfactuals ───────────────────────
t6 = []
for n in NAMES:
    rows = SAMPS[n]; cellrow = {int(r["cell_idx"]): int(r["row"]) for r in CELLS[n]}
    dx = np.array([F(r["dev_x_pt"]) for r in rows]); dy = np.array([F(r["dev_y_pt"]) for r in rows])
    cid = np.array([int(r["cell_idx"]) for r in rows]); rid = np.array([cellrow[c] for c in cid])
    raw = np.hypot(dx, dy).mean(); g = np.hypot(dx-dx.mean(), dy-dy.mean()).mean()
    ax_, ay = dx.copy(), dy.copy()
    for r in set(rid):
        m = rid == r; ax_[m] -= dx[m].mean(); ay[m] -= dy[m].mean()
    rr = np.hypot(ax_, ay).mean()
    bx, by = dx.copy(), dy.copy()
    for c in set(cid):
        m = cid == c; bx[m] -= dx[m].mean(); by[m] -= dy[m].mean()
    cc = np.hypot(bx, by).mean()
    z = TZ[n]
    t6.append(dict(Run=n, As_scored_deg=round(raw/z,3), Minus_global_offset_deg=round(g/z,3),
                   Minus_per_row_deg=round(rr/z,3), Minus_per_cell_floor_deg=round(cc/z,3),
                   Global_gain_pct=round(100*(1-g/raw),1), Per_row_gain_pct=round(100*(1-rr/raw),1),
                   Per_cell_gain_pct=round(100*(1-cc/raw),1)))
T6 = pd.DataFrame(t6)
T6.loc[len(T6)] = ["Mean"] + [round(T6[c].mean(),3) for c in T6.columns[1:]]

# ── T7 time-on-task ──────────────────────────────────────────
t7 = []
for k in range(1, 5):
    sub = T2[((T2.Presentation_order-1)//4 + 1 == k) & (T2.RMS_deg < 10)]
    t7.append(dict(Quarter=f"cells {4*(k-1)+1}-{4*k} shown", Mean_RMS_deg=round(sub.RMS_deg.mean(),3),
                   SE=round(sub.RMS_deg.std()/math.sqrt(len(sub)),3),
                   Mean_dev_deg=round(sub.Mean_dev_deg.mean(),3),
                   Mean_containment_pct=round(sub.Containment_pct.mean(),1), N=len(sub)))
T7 = pd.DataFrame(t7)
cl = T2[T2.RMS_deg < 10]
r_ord = float(np.corrcoef(cl.Presentation_order, cl.RMS_deg)[0,1])

# ── T8 head pose ─────────────────────────────────────────────
def unwrap(a): return np.degrees(np.unwrap(np.radians(np.array(a, float))))
t8 = []
for n in NAMES:
    rows = SAMPS[n]
    yaw = unwrap([F(r["head_yaw_deg"]) for r in rows]); pit = unwrap([F(r["head_pitch_deg"]) for r in rows])
    rol = unwrap([F(r["head_roll_deg"]) for r in rows]); dev = np.array([F(r["dev_pt"]) for r in rows])
    t8.append(dict(Run=n, Head_yaw_SD_deg=round(float(yaw.std()),3),
                   Head_pitch_SD_deg=round(float(pit.std()),3), Head_roll_SD_deg=round(float(rol.std()),3),
                   r_absYaw_vs_error=round(float(np.corrcoef(np.abs(yaw-yaw.mean()), dev)[0,1]),3),
                   r_absPitch_vs_error=round(float(np.corrcoef(np.abs(pit-pit.mean()), dev)[0,1]),3),
                   N_scored_samples=len(rows)))
T8 = pd.DataFrame(t8)

# ── T9 grid resolvability ────────────────────────────────────
allpt = sorted(T2.RMS_pt); p50 = float(np.median(allpt)); p90 = float(np.percentile(allpt, 90))
SW, SH = 402, 778
t9 = []
for lbl, cols, rows_, note in (("9×9", 9, 9, "Experiment 1's finest grid"),
                               ("9×4", 9, 4, ""), ("6×4", 4, 6, ""), ("5×4", 4, 5, ""),
                               ("4×4", 4, 4, "Experiment 2's own grid"),
                               ("4×3", 3, 4, "Experiment 3's word grid"),
                               ("3×3", 3, 3, "Experiment 4's predictive grid")):
    w, h = SW/cols, SH/rows_
    half = min(w, h)/2
    t9.append(dict(Grid=lbl, Cell_w_pt=round(w,1), Cell_h_pt=round(h,1),
                   Half_width_pt=round(half,1),
                   Clears_median=("yes" if half > p50 else "NO"),
                   Clears_90th_pct=("yes" if half > p90 else "NO"),
                   Margin_over_90th_pct=round(half-p90,1), Note=note))
T9 = pd.DataFrame(t9)

# ── Summary / Caveats ────────────────────────────────────────
dev = T1.Mean_dev_deg; bp = T1.Bias_pt; sd = T1.SD_pt
B, D = bp.mean(), sd.mean()
bad = T2[T2.RMS_deg > 3]
SUMMARY = pd.DataFrame([
 ("Runs analysed", 5, "4×4 fixation-stability grid; 80 cell-runs total"),
 ("Cells per run", 16, "1 s unscored settle + 4 s scored capture, shuffled order"),
 ("Scored samples", int(T1.Scored_samples.sum()), f"at {T1.Sample_rate_Hz.mean():.1f} Hz mean logging rate"),
 ("", "", ""),
 ("HEADLINE", "", ""),
 ("Mean deviation (accuracy)", f"{dev.mean():.2f}° (SD {dev.std():.2f})", f"{T1.Mean_dev_pt.mean():.1f} pt"),
 ("Mean RMS scatter (precision)", f"{T1.RMS_deg.mean():.2f}°", "mean of the 16 per-cell RMS values"),
 ("Containment", f"{T1.Containment_pct.mean():.1f} %", "share of scored samples inside the target cell"),
 ("", "", ""),
 ("ACCURACY vs PRECISION", "", ""),
 ("Bias (systematic offset)", f"{B:.1f} pt", "offset of the fixation centroid from the target"),
 ("Scatter (random SD)", f"{D:.1f} pt", "spread of samples about their own centroid"),
 ("Bias share of squared error", f"{100*B*B/(B*B+D*D):.0f} %", "THE headline: the error is correctable, not noise"),
 ("Per-axis scatter", f"SD_x {T1.SD_x_pt.mean():.1f} pt, SD_y {T1.SD_y_pt.mean():.1f} pt", f"vertical scatter is {T1.SD_y_pt.mean()/T1.SD_x_pt.mean():.2f}× horizontal"),
 ("", "", ""),
 ("SPATIAL STRUCTURE", "", ""),
 ("Top-row deviation", f"{T4.loc[0,'Mean_dev_deg']:.2f}°", "vs bottom row"),
 ("Bottom-row deviation", f"{T4.loc[3,'Mean_dev_deg']:.2f}°", f"{T4.loc[3,'Mean_dev_deg']/T4.loc[0,'Mean_dev_deg']:.1f}× the top row"),
 ("Bottom-row containment", f"{T4.loc[3,'Mean_containment_pct']:.0f} %", f"vs {T4.loc[0,'Mean_containment_pct']:.0f} % on the top row"),
 ("RMS by row (excl. outlier)", " / ".join(f"{v:.2f}°" for v in T4.Mean_RMS_deg_excl_outlier[:4]), "precision is FLAT — the bottom row is a bias problem"),
 ("Bottom-row vertical bias", f"{T4.loc[3,'Mean_bias_y_pt']:.0f} pt", "negative = the estimate sits ABOVE the target"),
 ("", "", ""),
 ("WHAT A CORRECTION BUYS", "", ""),
 ("As scored", f"{T6.loc[5,'As_scored_deg']:.2f}°", "baseline"),
 ("Minus a global offset", f"{T6.loc[5,'Minus_global_offset_deg']:.2f}°", f"−{T6.loc[5,'Global_gain_pct']:.0f} %"),
 ("Minus a per-row correction", f"{T6.loc[5,'Minus_per_row_deg']:.2f}°", f"−{T6.loc[5,'Per_row_gain_pct']:.0f} %"),
 ("Per-cell floor", f"{T6.loc[5,'Minus_per_cell_floor_deg']:.2f}°", "the true precision limit of this pipeline"),
 ("", "", ""),
 ("SETTLING", "", ""),
 ("First scored second", f"{T5[T5.Bin_start_s < 1].Mean_dev_pt.mean():.1f} pt", "the fixation is still converging"),
 ("Last scored second", f"{T5[T5.Bin_start_s >= 3].Mean_dev_pt.mean():.1f} pt", "a 30 % reduction"),
 ("Re-scored on the last 3 s", f"{T5b.Improvement_pct.mean():.1f} % better", "range 3–18 % across runs"),
 ("", "", ""),
 ("STABILITY", "", ""),
 ("Cell-runs with RMS > 3°", f"{len(bad)} of 80 ({100*len(bad)/80:.0f} %)", f"worst {bad.RMS_deg.max():.2f}° (Run 1, cell 12)"),
 ("The other 76 cell-runs", f"mean {T2[T2.RMS_deg<=3].RMS_deg.mean():.2f}°, median {T2[T2.RMS_deg<=3].RMS_deg.median():.2f}°", "a tight noise floor"),
 ("Fatigue over the 80 s run", f"r = {r_ord:+.3f}", "no time-on-task decay"),
 ("Head motion coupling", f"|r| ≤ {T8[['r_absYaw_vs_error','r_absPitch_vs_error']].abs().max().max():.2f}", "head is near-static; it explains almost none of the error"),
 ("", "", ""),
 ("GRID DESIGN", "", ""),
 ("Per-cell RMS, median", f"{p50:.1f} pt", "a cell needs a half-width above this to be resolvable"),
 ("Per-cell RMS, 90th pct", f"{p90:.1f} pt", "the conservative bar"),
 ("Largest safe grid", "4 cols × 8 rows", f"at the 90th-percentile error of {p90:.0f} pt"),
 ("9×9 grid", "NOT resolvable", f"22 pt half-width vs a {p50:.0f} pt median error"),
], columns=["Metric", "Value", "Note"])

CAVEATS = pd.DataFrame([
 ("READ THIS FIRST", "92 % of Experiment 2's error is systematic bias, not tracker noise. Quoting the 2.71° mean deviation as 'the tracker's accuracy' conflates a correctable calibration offset with the precision floor. Report accuracy and precision separately: deviation 2.71°, RMS scatter 1.51°, per-cell floor 1.27°."),
 ("The 1 s settle is too short", "Deviation falls 30 % across the scored window, so the fixation is still converging when scoring begins and every reported mean is inflated. Either lengthen the settle to ~2 s or score only the last 3 s. Both re-scorings are in sheet 'T5b Settling summary'."),
 ("One catastrophic cell-run", "Run 1 / cell 12 (bottom-left) reports 14.28° RMS and 11.79° deviation — a tracking failure, not a fixation. It single-handedly drives the row-3 and col-0 marginals. Report marginals with and without it (both are in 'T4 Marginals')."),
 ("Run 4 used a different blink gate", "Run 4 ran blinkEAR=ratio0.62/floor0.13 with no maxGated cap; the other four ran ratio0.55/floor0.12/maxGated5. Run 4 also has the lowest logging rate (9.2 Hz) and the highest bias (69.7 pt). Treat it as a separate configuration or exclude it."),
 ("Participants", "The export records no participant identifier, and Run 1 came from a different session (20:05) than Runs 4–7 (12:36–12:57). These are 5 runs, not 5 participants."),
 ("Degrees are virtual", "Angular values use the calibration's fitted |tz| (1 152-1 278 pt across runs, i.e. 20.1-22.3 pt per degree), not a measured viewing distance. Comparable within a run; not across participants."),
 ("Only Run 1 has a validation", "Run 1 is preceded by a 9-dot validation (1.52 deg mean error, 1.00 deg RMS, verdict 'good'). Runs 4–7 were exported standalone with no validation, so calibration quality cannot be checked for them."),
 ("Single posture / lighting", "Head yaw SD 1.2–2.1° after angle unwrapping — stationary, hand-held, well-lit best-case numbers."),
 ("Raw yaw wraps", "head_yaw_deg sits near ±180° and wraps between frames. Computing SD without unwrapping gives spurious values of 70–110°. All head figures here are unwrapped."),
 ("Provenance", "Computed directly from trials.csv in each run bundle (schema v4): the '# Per-cell' block for cell statistics and the '# Samples' block for per-sample deviations. No re-scoring or exclusions unless a sheet says so."),
], columns=["Topic", "Detail"])

SHEETS = [
 ("Summary", SUMMARY), ("Caveats", CAVEATS),
 ("T1 Runs", T1), ("T2 Cell-runs", T2), ("T3 Cells pooled", T3),
 ("T3a Grid deviation", G_DEV), ("T3b Grid RMS", G_RMS),
 ("T3c Grid containment", G_CON), ("T3d Grid vert bias", G_BY),
 ("T4 Marginals", T4),
 ("T5 Settling raw", T5), ("T5b Settling summary", T5b),
 ("T6 Bias correction", T6), ("T7 Time on task", T7),
 ("T8 Head pose", T8), ("T9 Grid resolvability", T9),
]
with pd.ExcelWriter(XL, engine="openpyxl") as w:
    for name, df in SHEETS:
        df.to_excel(w, sheet_name=name[:31], index=False)

wb = load_workbook(XL)
HDR_FILL = PatternFill("solid", fgColor="12161B")
HDR_FONT = Font(color="FFFFFF", bold=True, size=10)
BAND = PatternFill("solid", fgColor="F2F5F8")
SEC  = PatternFill("solid", fgColor="E3EDFA")
thin = Side(style="thin", color="D6DBE0")
LONG = {"Note","Detail","Topic","Metric","Marginal","Quarter","Outlier","Blink_gate","Started_at"}
for ws in wb.worksheets:
    ws.freeze_panes = "A2"; ws.auto_filter.ref = ws.dimensions
    for c in ws[1]:
        c.fill = HDR_FILL; c.font = HDR_FONT
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 30
    for col in ws.columns:
        L = max((len(str(c.value)) for c in col if c.value is not None), default=8)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max(L+3, 11), 52)
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
ws.column_dimensions["A"].width = 34; ws.column_dimensions["B"].width = 26
for row in ws.iter_rows(min_row=2, max_col=1):
    v = row[0].value
    if v and v.isupper() and len(v) > 3:
        for c in ws[row[0].row]: c.fill = SEC; c.font = Font(bold=True, size=10)
ws = wb["Caveats"]
ws["A2"].font = Font(bold=True, color="B03A3A", size=11)
ws.column_dimensions["B"].width = 96
for r in ws.iter_rows(min_row=2, min_col=2, max_col=2):
    r[0].alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[r[0].row].height = 62
wb.save(XL)
print("workbook:", XL)
for n, d in SHEETS: print(f"   {n:24s} {d.shape[0]:4d} rows × {d.shape[1]} cols")
