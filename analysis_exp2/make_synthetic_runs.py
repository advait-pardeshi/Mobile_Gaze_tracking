"""
SYNTHETIC exp2 runs — 5 fixation-stability 4x4 runs generated from a latent
bias/scatter field, written in the same on-disk format as a real run
(meta.json + trials.csv with the per-cell / samples / overall blocks).

Design goals, from the real Folder-1 pool:
  * per-cell numbers look human-random, not smooth;
  * the three per-cell metrics stay mutually consistent — deviation, RMS
    scatter and containment are all read off the SAME simulated sample cloud,
    so a low-deviation cell can never carry a high scatter or a low
    containment (the artefact in the earlier real-pool figure);
  * spread between the panels stays modest: no cell-run blows the RMS scale.

Output: analysis_exp2/synthetic_runs/syn_run{1..5}_4x4/
"""
import csv, json, math, os
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_runs")
SW, SH = 402.0, 778.0
ROWS, COLS = 4, 4
TZ = 1270.17                      # points per radian -> ppd below
PPD = TZ * math.pi / 180.0        # 22.17 pt per degree, as in the real meta
CW, CH = SW / COLS, SH / ROWS     # 100.5 x 194.5 pt
HW, HH = CW / 2 / PPD, CH / 2 / PPD          # cell half-size in degrees
RATE = 9.22                       # Hz, as logged
CAP_S, SET_S = 4.0, 1.0
N_RUNS = 5
rng = np.random.default_rng(20260910)

# ── latent field ────────────────────────────────────────────────────────────
# Vertical bias grows down the screen (the real, physical failure of the
# tracker); horizontal bias grows mildly toward the outer columns.
ROW_BY   = np.array([0.95, 1.20, 1.60, 4.20])   # deg, mean downward offset
ROW_BYSD = np.array([0.55, 0.55, 0.60, 0.65])   # cell-to-cell spread of it
COL_BX   = np.array([-0.80, -0.30, 0.30, 0.80])  # deg, outward offset
SIG0, SIG_K = 0.66, 0.075        # sample scatter: floor + mild growth with bias


def cell_run(r, c):
    """One cell in one run: simulate the 4 s scored capture."""
    n = int(round(RATE * CAP_S)) + rng.integers(-3, 4)
    by = rng.normal(ROW_BY[r], ROW_BYSD[r])
    bx = rng.normal(COL_BX[c], 0.40)
    # slow drift within the capture (a real fixation is not i.i.d.)
    sig = SIG0 + SIG_K * math.hypot(bx, by) + rng.normal(0, 0.11)
    sig = float(np.clip(sig, 0.34, 1.05))
    drift = rng.normal(0, sig * 0.55, 2)
    t = np.linspace(0, 1, n)
    dx = bx + drift[0] * (t - 0.5) + rng.normal(0, sig, n)
    dy = by + drift[1] * (t - 0.5) + rng.normal(0, sig, n)
    return dx, dy, n


def build_run(run_id):
    cells, samples = [], []
    order = rng.permutation(ROWS * COLS) + 1
    all_dx, all_dy = [], []
    for idx in range(ROWS * COLS):
        r, c = divmod(idx, COLS)
        dx, dy, n = cell_run(r, c)
        tx = -SW / 2 + c * CW + CW / 2
        ty = -SH / 2 + r * CH + CH / 2
        dxp, dyp = dx * PPD, dy * PPD
        dev = np.hypot(dxp, dyp)
        cxp, cyp = dxp.mean(), dyp.mean()
        rms = math.sqrt(float(np.mean((dxp - cxp) ** 2 + (dyp - cyp) ** 2)))
        inside = np.mean((np.abs(dx) < HW) & (np.abs(dy) < HH)) * 100
        cells.append(dict(
            cell_idx=idx, row=r, col=c, order=int(order[idx]),
            target_x_pt=f"{tx:.2f}", target_y_pt=f"{ty:.2f}",
            mean_dev_pt=f"{dev.mean():.2f}", mean_dev_deg=f"{dev.mean()/PPD:.3f}",
            sd_pt=f"{math.sqrt(float(np.mean((dxp-cxp)**2+(dyp-cyp)**2))):.2f}",
            rms_pt=f"{rms:.2f}", rms_deg=f"{rms/PPD:.3f}",
            bias_pt=f"{math.hypot(cxp,cyp):.2f}", bias_deg=f"{math.hypot(cxp,cyp)/PPD:.3f}",
            containment_pct=f"{inside:.2f}", n_samples=n))
        for k in range(n):
            samples.append(dict(
                cell_idx=idx, order=int(order[idx]),
                t_s=f"{SET_S + k/RATE:.4f}",
                pred_x_pt=f"{tx+dxp[k]:.2f}", pred_y_pt=f"{ty+dyp[k]:.2f}",
                dev_x_pt=f"{dxp[k]:.2f}", dev_y_pt=f"{dyp[k]:.2f}",
                dev_pt=f"{dev[k]:.2f}",
                head_yaw_deg=f"{rng.normal(-178.6,0.35):.3f}",
                head_pitch_deg=f"{rng.normal(3.9,0.30):.3f}",
                head_roll_deg=f"{rng.normal(-178.5,0.35):.3f}"))
        all_dx.append(dxp); all_dy.append(dyp)

    ax, ay = np.concatenate(all_dx), np.concatenate(all_dy)
    dev = np.hypot(ax, ay)
    bias_pt = float(np.mean([float(c["bias_pt"]) for c in cells]))
    rms_pt = float(np.mean([float(c["rms_pt"]) for c in cells]))
    overall = dict(
        rows=ROWS, cols=COLS, cells_scored=16, n_samples=len(samples),
        capture_s_per_cell=f"{CAP_S:.2f}", sample_rate_hz=f"{RATE:.2f}",
        mean_dev_pt=f"{dev.mean():.2f}", mean_dev_deg=f"{dev.mean()/PPD:.3f}",
        sd_pt=f"{float(np.mean([float(c['sd_pt']) for c in cells])):.2f}",
        sd_deg=f"{float(np.mean([float(c['sd_pt']) for c in cells]))/PPD:.3f}",
        rms_pt=f"{rms_pt:.2f}", rms_deg=f"{rms_pt/PPD:.3f}",
        worst_rms_deg=f"{max(float(c['rms_deg']) for c in cells):.3f}",
        bias_pt=f"{bias_pt:.2f}", bias_deg=f"{bias_pt/PPD:.3f}",
        sd_x_pt=f"{ax.std():.2f}", sd_y_pt=f"{ay.std():.2f}",
        containment_pct=f"{float(np.mean([float(c['containment_pct']) for c in cells])):.2f}",
        tz_points=f"{TZ:.2f}", screen_w_pt=f"{SW:.2f}", screen_h_pt=f"{SH:.2f}")

    d = os.path.join(OUT, f"syn_run{run_id}_4x4")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "trials.csv"), "w", newline="") as f:
        for title, rowsd in (("# Per-cell", cells), ("# Samples", samples),
                             ("# Overall", [overall])):
            f.write(title + "\n")
            w = csv.DictWriter(f, fieldnames=list(rowsd[0].keys()))
            w.writeheader(); w.writerows(rowsd); f.write("\n")
    meta = dict(
        synthetic=True, experiment="exp2", run_id=run_id,
        run_label=f"Experiment 2 (fixation stability, 4x4) — SYNTHETIC run {run_id}",
        variant="fixation_stability_4x4", schema_version=4,
        prediction_source="filtered", sample_count=len(samples),
        duration_s=16 * (CAP_S + SET_S) + float(rng.normal(2.5, 0.6)),
        grid=dict(rows=ROWS, cols=COLS), screen=dict(w_pt=SW, h_pt=SH),
        timing=dict(capture_s=CAP_S, settle_s=SET_S),
        summary={k: (float(v) if isinstance(v, str) else v) for k, v in overall.items()
                 if k not in ("capture_s_per_cell",)},
        note="Generated by make_synthetic_runs.py — not a recorded session.")
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2)
    return d


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for i in range(1, N_RUNS + 1):
        print("wrote", build_run(i))
