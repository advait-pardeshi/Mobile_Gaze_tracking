"""Shared loader for the Experiment 1 (grid accuracy vs resolution) re-run bundles."""
import csv, os, json, math
import numpy as np, pandas as pd

ROOT = "/Users/advait/Downloads/New Folder With Items 3"
OUT  = "/Users/advait/Desktop/GazeTracking/Mobile_Gaze_tracking-main/analysis_exp1"

# Three unique sessions. `<ROOT>/exp1` is a byte-identical duplicate of
# session_20260826_221940/exp1 and is deliberately not loaded twice.
SESSIONS = [
    ("S1", "20260826_122215", f"{ROOT}/exp1_session_20260826_122215",
     [("3x3", "run1_3x3"), ("4x4", "run2_4x4"), ("5x4", "run3_5x4")]),
    ("S2", "20260826_221940", f"{ROOT}/session_20260826_221940/exp1",
     [("3x3", "run2_3x3"), ("4x4", "run3_4x4"), ("5x4", "run4_5x4")]),
    ("S3", "20260827_112714", f"{ROOT}/exp1_session_20260827_112714",
     [("3x3", "run1_3x3"), ("4x4", "run2_4x4"), ("5x4", "run3_5x4")]),
]
VALIDATION = f"{ROOT}/session_20260826_221940/validation"
GRIDS = ["3x3", "4x4", "5x4"]
GRID_LABEL = {"3x3": "3x3", "4x4": "4x4", "5x4": "5x4"}
SW, SH = 402.0, 778.0
F = float


def sect(path):
    """Split one of the multi-block CSV exports into a list of dict-rows per block."""
    blocks, cur = [], None
    for line in open(path):
        line = line.rstrip("\n")
        if not line.strip() or line.startswith("#"):
            cur = None
            continue
        if cur is None:
            cur = {"h": next(csv.reader([line])), "r": []}
            blocks.append(cur)
        else:
            cur["r"].append(next(csv.reader([line])))
    return [[dict(zip(b["h"], r)) for r in b["r"]] for b in blocks]


def load():
    """Return (runs, trials, samples) — one row per run, per trial, per logged frame."""
    runs, trials, samples = [], [], []
    for sid, stamp, base, rl in SESSIONS:
        for grid, d in rl:
            p = os.path.join(base, d)
            m = json.load(open(f"{p}/meta.json"))
            s = m["summary"]
            ppd = s["mean_error_pt"] / s["mean_error_deg"]     # points per degree of visual angle
            tz = ppd * 57.2958                                 # the calibration's fitted |tz|, in points
            rows, cells, _ = sect(f"{p}/trials.csv")
            runs.append(dict(
                Session=sid, Session_stamp=stamp, Grid=GRID_LABEL[grid], Run_dir=d,
                Run_id=m["run_id"], Started_at=m["started_at"],
                Duration_s=m["duration_s"], Rows=m["grid"]["rows"], Cols=m["grid"]["cols"],
                Trials=s["trial_count"], Hits=s["hits"], Accuracy_pct=s["accuracy_pct"],
                Mean_error_pt=s["mean_error_pt"], Mean_error_deg=s["mean_error_deg"],
                Cell_tolerance_deg=s["cell_tolerance_deg"],
                Frames_logged=m["sample_count"],
                Sample_rate_Hz=m["sample_count"] / m["duration_s"],
                Prediction_source=m["prediction_source"],
                Blink_gate=m["pipeline_tuning"].split("blinkEAR=")[1].split(" ")[0],
                Points_per_degree=ppd, Virtual_tz_pt=tz))
            for r in rows:
                void = r["pred_x"].strip() == ""
                trials.append(dict(
                    Session=sid, Grid=GRID_LABEL[grid], Run_id=m["run_id"],
                    Trial=int(r["trial"]), Cell_idx=int(r["cell_idx"]),
                    Row=int(r["row"]), Col=int(r["col"]),
                    Cell_cx=F(r["cell_cx"]), Cell_cy=F(r["cell_cy"]),
                    Cell_x_min=F(r["cell_x_min"]), Cell_x_max=F(r["cell_x_max"]),
                    Cell_y_min=F(r["cell_y_min"]), Cell_y_max=F(r["cell_y_max"]),
                    Pred_x=np.nan if void else F(r["pred_x"]),
                    Pred_y=np.nan if void else F(r["pred_y"]),
                    Err_pt=np.nan if void else F(r["err_pt"]),
                    Err_deg=np.nan if void else F(r["err_deg"]),
                    Hit=int(r["hit"]), N_samples=int(r["n_samples"]),
                    Void_no_frames="yes" if void else "",
                    Head_yaw_deg=np.nan if void else F(r["head_yaw_deg"]),
                    Head_pitch_deg=np.nan if void else F(r["head_pitch_deg"]),
                    Head_tz_mm=np.nan if void else F(r["head_tz_mm"]),
                    Cell_tolerance_deg=s["cell_tolerance_deg"], Points_per_degree=ppd))
            sf = pd.read_csv(f"{p}/samples.csv")
            sf.insert(0, "Session", sid); sf.insert(1, "Grid", GRID_LABEL[grid])
            sf["ppd"] = ppd
            samples.append(sf)

    R = pd.DataFrame(runs)
    T = pd.DataFrame(trials)
    T["Dev_x_pt"] = T.Pred_x - T.Cell_cx
    T["Dev_y_pt"] = T.Pred_y - T.Cell_cy
    S = pd.concat(samples, ignore_index=True)
    S["t_rel"] = S.t_s - S.groupby(["Session", "Grid", "trial"]).t_s.transform("min")
    return R, T, S


def load_validation():
    """The 9-dot calibration validation that precedes session S2."""
    out = []
    for d in ("run1_9dot", "run2_9dot"):
        p = f"{VALIDATION}/{d}"
        m = json.load(open(f"{p}/meta.json"))
        dots = sect(f"{p}/trials.csv")[0]
        for r in dots:
            out.append(dict(Run=d.replace("run", "Validation ").replace("_9dot", ""),
                            Started_at=m["started_at"], Dot=int(r["dot"]),
                            Target_x_pt=F(r["target_x_pt"]), Target_y_pt=F(r["target_y_pt"]),
                            Confirm_order=int(r["confirm_order"]),
                            Confirmed_at_s=F(r["confirmed_at_s"]),
                            Err_x_pt=F(r["err_x_pt"]), Err_y_pt=F(r["err_y_pt"]),
                            Err_pt=F(r["err_pt"]), Err_deg=F(r["err_deg"]),
                            SD_pt=F(r["sd_pt"]), RMS_pt=F(r["rms_pt"]), RMS_deg=F(r["rms_deg"]),
                            N_samples=int(r["n_samples"]), Verdict=m["summary"]["verdict"]))
    return pd.DataFrame(out)


def in_cell(px, py, t):
    return ((px >= t.Cell_x_min) & (px <= t.Cell_x_max) &
            (py >= t.Cell_y_min) & (py <= t.Cell_y_max))


def affine_fit(train, test):
    """Fit a per-axis scale+offset on `train` predictions -> targets, apply to `test`."""
    mx, bx = np.polyfit(train.Pred_x, train.Cell_cx, 1)
    my, by = np.polyfit(train.Pred_y, train.Cell_cy, 1)
    return mx * test.Pred_x + bx, my * test.Pred_y + by, mx, bx, my, by
