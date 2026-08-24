#!/usr/bin/env python3
"""Analyse a "hold still" capture from the per-frame instrumentation CSV.

The app writes `Documents/instrumentation/frames_<stamp>.csv` when
`PipelineTuning.instrumentationEnabled` is true (see `FrameInstrumentation`
in the iOS sources). Point this script at one such file, recorded while the
participant fixated a single point without moving, and it reports the three
numbers that decide where the stability work should go next:

  1. ANGULAR VARIANCE OF R_h — how much the PnP head pose wanders when the
     head does not. Computed on the quaternion, not on Euler angles: the
     Euler decomposition wraps and gimbals, so its "variance" is an artefact
     of the parameterisation near certain poses. We take the Karcher-style
     mean rotation (principal eigenvector of the quaternion scatter matrix)
     and report the geodesic angle of each sample to it.

  2. VARIANCE OF THE FINAL SCREEN POINT — the same statistic on the thing
     the user actually sees, for the raw and the filtered stream separately,
     so the smoother's contribution is visible rather than assumed.

  3. PER-STAGE TIME BUDGET — where the frame time goes, as median/p95/max
     per stage, plus the drop rate that backpressure produced.

Usage:
  python3 analyze_holdstill.py frames_20260823_141230.csv
  python3 analyze_holdstill.py frames.csv --window 30 --start 2
  python3 analyze_holdstill.py frames.csv --distance-mm 600 --ppi 460

`--window` trims to the first N seconds of usable data (default 30, the
length of the intended capture); `--start` skips a lead-in so the filters'
convergence transient stays out of the statistics.

Only numpy is required. Pass `--csv OUT` to also write a tidy one-row
summary for cross-run comparison.
"""
import argparse
import csv as csvmod
import math
import os
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required:  python3 -m pip install numpy")


# --------------------------------------------------------------------------
# Loading


def load(path):
    """Read the instrumentation CSV into a dict of float/str numpy arrays."""
    with open(path, newline="") as fh:
        rows = list(csvmod.DictReader(fh))
    if not rows:
        sys.exit(f"{path}: no data rows")
    cols = {}
    for key in rows[0]:
        vals = []
        for r in rows:
            raw = (r.get(key) or "").strip()
            try:
                vals.append(float(raw) if raw else math.nan)
            except ValueError:
                vals.append(math.nan)
        cols[key] = np.array(vals, dtype=float)
    return cols, len(rows)


def trim(cols, start_s, window_s):
    t = cols["t_s"]
    t0 = np.nanmin(t)
    lo = t0 + start_s
    hi = lo + window_s if window_s > 0 else np.inf
    keep = (t >= lo) & (t < hi)
    if keep.sum() == 0:
        sys.exit("no frames left after --start/--window trimming")
    return {k: v[keep] for k, v in cols.items()}, keep.sum()


# --------------------------------------------------------------------------
# 1. Angular variance of R_h


def quat_array(cols, prefix):
    q = np.stack([cols[f"{prefix}_x"], cols[f"{prefix}_y"],
                  cols[f"{prefix}_z"], cols[f"{prefix}_w"]], axis=1)
    good = np.isfinite(q).all(axis=1)
    q = q[good]
    if len(q) == 0:
        return q
    n = np.linalg.norm(q, axis=1, keepdims=True)
    n[n == 0] = 1.0
    q = q / n
    # q and -q are the same rotation. Align every sample to one hemisphere or
    # the mean below is pulled toward the origin and the angles are nonsense.
    flip = q @ q[0] < 0
    q[flip] *= -1
    return q


def mean_rotation(q):
    """Quaternion average: principal eigenvector of sum(q qᵀ).

    This is the standard Markley et al. solution and is exact for the
    chordal-distance mean; for the tight spreads seen in a hold-still capture
    it is indistinguishable from the true Karcher mean on SO(3).
    """
    m = q.T @ q
    w, v = np.linalg.eigh(m)
    qm = v[:, np.argmax(w)]
    if qm[3] < 0:
        qm = -qm
    return qm / np.linalg.norm(qm)


def quat_mul(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def rotation_stats(q, label):
    if len(q) < 2:
        return None
    qm = mean_rotation(q)
    qm_inv = np.array([-qm[0], -qm[1], -qm[2], qm[3]])
    dq = quat_mul(q, qm_inv[None, :])
    dq[dq[:, 3] < 0] *= -1
    # Geodesic angle to the mean, and the rotation-vector components that
    # give it a per-axis breakdown (x=pitch-ish, y=yaw-ish, z=roll-ish in the
    # camera frame).
    w = np.clip(dq[:, 3], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(np.maximum(0.0, 1.0 - w * w))
    scale = np.where(sin_half < 1e-9, 2.0, angle / np.maximum(sin_half, 1e-12))
    rotvec = dq[:, :3] * scale[:, None]
    deg = np.degrees(angle)
    axes = np.degrees(rotvec)
    return {
        "label": label,
        "n": len(q),
        "rms_deg": float(np.sqrt(np.mean(deg ** 2))),
        "mean_deg": float(np.mean(deg)),
        "p95_deg": float(np.percentile(deg, 95)),
        "max_deg": float(np.max(deg)),
        # Per-axis standard deviation of the rotation vector: this is the
        # "variance" in the usual sense, decomposed.
        "std_x_deg": float(np.std(axes[:, 0])),
        "std_y_deg": float(np.std(axes[:, 1])),
        "std_z_deg": float(np.std(axes[:, 2])),
        # Frame-to-frame jitter, which is what actually shakes the crop.
        "step_rms_deg": float(np.sqrt(np.mean(
            np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(
                q[1:] * q[:-1], axis=1)), -1, 1))) ** 2))),
    }


# --------------------------------------------------------------------------
# 2. Screen-point variance


def point_stats(x, y, label, deg_per_pt=None):
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 2:
        return None
    cx, cy = float(np.mean(x)), float(np.mean(y))
    dx, dy = x - cx, y - cy
    r = np.hypot(dx, dy)
    out = {
        "label": label,
        "n": int(len(x)),
        "centroid": (cx, cy),
        "std_x_pt": float(np.std(dx)),
        "std_y_pt": float(np.std(dy)),
        # RMS radial deviation about the centroid — the single precision
        # number. Note it is NOT std_x + std_y; for isotropic noise it is
        # sqrt(std_x^2 + std_y^2).
        "rms_pt": float(np.sqrt(np.mean(r ** 2))),
        "p95_pt": float(np.percentile(r, 95)),
        "max_pt": float(np.max(r)),
        # Median absolute frame-to-frame step: the visible "shake", which a
        # smoother reduces far more than it reduces the spread.
        "step_median_pt": float(np.median(np.hypot(np.diff(x), np.diff(y)))),
    }
    if deg_per_pt:
        out["rms_deg"] = out["rms_pt"] * deg_per_pt
        out["p95_deg"] = out["p95_pt"] * deg_per_pt
    return out


def angle_stats(p, y, label):
    good = np.isfinite(p) & np.isfinite(y)
    p, y = p[good], y[good]
    if len(p) < 2:
        return None
    return {
        "label": label,
        "n": int(len(p)),
        "std_pitch_deg": float(np.std(p)),
        "std_yaw_deg": float(np.std(y)),
        "step_median_deg": float(np.median(
            np.hypot(np.diff(p), np.diff(y)))),
    }


# --------------------------------------------------------------------------
# 3. Time budget

STAGES = [
    ("dispatch_ms", "queue hop (callback -> worker)"),
    ("pose_ms", "Stage 2  solvePnP"),
    ("head_filter_ms", "Stage 2b head-pose 1EUR"),
    ("eye_warp_ms", "Stage 3  eye strip (throttled)"),
    ("face_warp_ms", "Stage 3b face warp 224x224"),
    ("cnn_ms", "Stage 4  CoreML"),
    ("gaze_filter_ms", "Stage 4b gaze 1EUR"),
    ("publish_ms", "Stage 5+ publish + project"),
]


def stage_stats(cols):
    out = []
    for key, name in STAGES:
        v = cols.get(key)
        if v is None:
            continue
        good = v[np.isfinite(v)]
        if len(good) == 0:
            out.append((name, key, 0, math.nan, math.nan, math.nan, math.nan))
            continue
        out.append((name, key, len(good),
                    float(np.median(good)), float(np.mean(good)),
                    float(np.percentile(good, 95)), float(np.max(good))))
    return out


# --------------------------------------------------------------------------
# Reporting


def hdr(title):
    print()
    print(title)
    print("-" * len(title))


def fmt_rot(s):
    if s is None:
        print("  (insufficient data)")
        return
    print(f"  n = {s['n']}")
    print(f"  geodesic deviation from mean rotation:")
    print(f"    RMS   {s['rms_deg']:7.4f} deg")
    print(f"    mean  {s['mean_deg']:7.4f} deg")
    print(f"    p95   {s['p95_deg']:7.4f} deg")
    print(f"    max   {s['max_deg']:7.4f} deg")
    print(f"  per-axis std of the rotation vector (camera frame):")
    print(f"    x {s['std_x_deg']:7.4f}   y {s['std_y_deg']:7.4f}"
          f"   z {s['std_z_deg']:7.4f}  deg")
    print(f"  frame-to-frame step RMS  {s['step_rms_deg']:7.4f} deg")


def fmt_point(s):
    if s is None:
        print("  (insufficient data)")
        return
    print(f"  n = {s['n']}   centroid = "
          f"({s['centroid'][0]:.1f}, {s['centroid'][1]:.1f}) pt")
    print(f"    std x {s['std_x_pt']:8.2f} pt     std y {s['std_y_pt']:8.2f} pt")
    print(f"    RMS radial {s['rms_pt']:8.2f} pt"
          + (f"   ({s['rms_deg']:.3f} deg)" if "rms_deg" in s else ""))
    print(f"    p95 radial {s['p95_pt']:8.2f} pt"
          + (f"   ({s['p95_deg']:.3f} deg)" if "p95_deg" in s else ""))
    print(f"    max radial {s['max_pt']:8.2f} pt")
    print(f"    median frame-to-frame step {s['step_median_pt']:8.2f} pt")


def main():
    ap = argparse.ArgumentParser(
        description="Hold-still analysis of the gaze pipeline instrumentation CSV.")
    ap.add_argument("csv", help="frames_<stamp>.csv from Documents/instrumentation")
    ap.add_argument("--start", type=float, default=1.0,
                    help="seconds to skip at the start (default 1)")
    ap.add_argument("--window", type=float, default=30.0,
                    help="seconds to analyse after --start; 0 = all (default 30)")
    ap.add_argument("--distance-mm", type=float, default=None,
                    help="eye-to-screen distance, for a degrees conversion")
    ap.add_argument("--ppi", type=float, default=None,
                    help="screen points per inch (iPhone: 163 for @2x-logical)")
    ap.add_argument("--csv-out", default=None,
                    help="also write a one-row summary CSV here")
    args = ap.parse_args()

    cols, total = load(args.csv)
    cols, kept = trim(cols, args.start, args.window)
    span = float(np.nanmax(cols["t_s"]) - np.nanmin(cols["t_s"]))

    deg_per_pt = None
    if args.distance_mm and args.ppi:
        mm_per_pt = 25.4 / args.ppi
        deg_per_pt = math.degrees(math.atan(mm_per_pt / args.distance_mm))

    print(f"file        {os.path.basename(args.csv)}")
    print(f"rows        {total} total, {kept} in window")
    print(f"window      {span:.2f} s  (start={args.start}s, "
          f"len={args.window if args.window else 'all'}s)")

    # ---- throughput -------------------------------------------------------
    dropped = cols.get("dropped_since")
    n_dropped = int(np.nansum(dropped)) if dropped is not None else 0
    processed = kept
    arrived = processed + n_dropped
    hdr("THROUGHPUT")
    print(f"  processed {processed} frames in {span:.2f} s "
          f"-> {processed / span:.1f} Hz")
    print(f"  dropped   {n_dropped} frames by backpressure "
          f"({100.0 * n_dropped / max(1, arrived):.1f} % of {arrived} arrived)")
    if "latency_ms" in cols:
        lat = cols["latency_ms"][np.isfinite(cols["latency_ms"])]
        if len(lat):
            print(f"  end-to-end latency (landmarks -> published):")
            print(f"    median {np.median(lat):6.2f} ms   "
                  f"p95 {np.percentile(lat, 95):6.2f} ms   "
                  f"max {np.max(lat):6.2f} ms")
    if "blink_held" in cols:
        bh = cols["blink_held"]
        n_blink = int(np.nansum(bh))
        print(f"  blink-gated {n_blink} frames "
              f"({100.0 * n_blink / max(1, processed):.1f} %)")
    if "ear_mean" in cols:
        e = cols["ear_mean"][np.isfinite(cols["ear_mean"])]
        if len(e):
            print(f"  EAR  median {np.median(e):.4f}   "
                  f"p5 {np.percentile(e, 5):.4f}   min {np.min(e):.4f}")

    # ---- 1. head pose -----------------------------------------------------
    hdr("1. ANGULAR VARIANCE OF R_h  (raw, straight out of solvePnP)")
    fmt_rot(rotation_stats(quat_array(cols, "q_raw"), "raw"))
    hdr("1b. ANGULAR VARIANCE OF R_h  (after the quaternion One Euro filter)")
    fmt_rot(rotation_stats(quat_array(cols, "q_filt"), "filtered"))

    # ---- 2. screen point --------------------------------------------------
    hdr("2. FINAL SCREEN POINT VARIANCE")
    if deg_per_pt:
        print(f"  (degrees via {args.ppi} ppi at {args.distance_mm} mm: "
              f"1 pt = {deg_per_pt:.4f} deg)")
    print("\n  RAW stream (per-frame CNN output, projected):")
    fmt_point(point_stats(cols.get("raw_pred_x", np.array([])),
                          cols.get("raw_pred_y", np.array([])),
                          "raw", deg_per_pt))
    print("\n  FILTERED stream (what is rendered):")
    fmt_point(point_stats(cols.get("filt_pred_x", np.array([])),
                          cols.get("filt_pred_y", np.array([])),
                          "filtered", deg_per_pt))

    raw_a = angle_stats(cols.get("raw_pitch_deg", np.array([])),
                        cols.get("raw_yaw_deg", np.array([])), "raw")
    filt_a = angle_stats(cols.get("filt_pitch_deg", np.array([])),
                         cols.get("filt_yaw_deg", np.array([])), "filtered")
    print("\n  Gaze angles at the source (CNN output, before projection):")
    for s in (raw_a, filt_a):
        if s:
            print(f"    {s['label']:9s} std pitch {s['std_pitch_deg']:6.3f} deg"
                  f"   std yaw {s['std_yaw_deg']:6.3f} deg"
                  f"   median step {s['step_median_deg']:6.3f} deg")

    rp = point_stats(cols.get("raw_pred_x", np.array([])),
                     cols.get("raw_pred_y", np.array([])), "raw")
    fp = point_stats(cols.get("filt_pred_x", np.array([])),
                     cols.get("filt_pred_y", np.array([])), "filtered")
    if rp and fp and fp["rms_pt"] > 0:
        print(f"\n  Smoother gain: RMS {rp['rms_pt']:.2f} -> {fp['rms_pt']:.2f} pt "
              f"({rp['rms_pt'] / fp['rms_pt']:.2f}x), "
              f"median step {rp['step_median_pt']:.2f} -> "
              f"{fp['step_median_pt']:.2f} pt "
              f"({rp['step_median_pt'] / max(1e-9, fp['step_median_pt']):.2f}x)")
        print("  NOTE: on a hold-still capture the centroid offset between the "
              "two\n        streams is the smoother's lag; the RMS ratio is its "
              "benefit.\n        Both are needed to judge the trade.")

    # ---- 3. time budget ---------------------------------------------------
    hdr("3. PER-STAGE TIME BUDGET  (milliseconds)")
    print(f"  {'stage':34s} {'n':>6s} {'median':>8s} {'mean':>8s} "
          f"{'p95':>8s} {'max':>8s}")
    rows = stage_stats(cols)
    for name, _key, n, med, mean, p95, mx in rows:
        print(f"  {name:34s} {n:6d} {med:8.3f} {mean:8.3f} {p95:8.3f} {mx:8.3f}")
    # Amortised, not summed medians: the eye strip runs on one frame in N, so
    # adding its median as if it ran every frame overstates the budget. Weight
    # each stage by how often it actually ran.
    amortised = sum(r[4] * r[2] / max(1, processed)
                    for r in rows if r[4] == r[4])
    print(f"  {'-' * 34} {'':6s} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")
    print(f"  {'per-frame amortised (mean x duty)':34s} {'':6s} "
          f"{'':8s} {amortised:8.3f}")
    if "total_ms" in cols:
        tot = cols["total_ms"][np.isfinite(cols["total_ms"])]
        if len(tot):
            print(f"  {'measured total (arrival->publish)':34s} "
                  f"{len(tot):6d} {np.median(tot):8.3f} {np.mean(tot):8.3f} "
                  f"{np.percentile(tot, 95):8.3f} {np.max(tot):8.3f}")
            budget = 1000.0 / max(1e-9, processed / span)
            print(f"\n  Frame budget at the observed {processed / span:.1f} Hz "
                  f"is {budget:.1f} ms; the median frame uses "
                  f"{100.0 * np.median(tot) / budget:.0f} % of it.")
            if n_dropped > 0:
                p95_tot = float(np.percentile(tot, 95))
                if p95_tot > budget:
                    print("  Drops line up with the p95 frame exceeding the "
                          "budget: the pipeline\n  genuinely cannot keep up. "
                          "The stage with the largest median above\n  is what "
                          "caps the rate.")
                else:
                    print("  Frames were dropped even though p95 "
                          f"({p95_tot:.1f} ms) fits inside the\n  budget "
                          f"({budget:.1f} ms) - so the cause is bursty "
                          "scheduling or contention\n  (thermal, ANE sharing, "
                          "main-thread stalls), not steady-state cost.")

    # ---- optional tidy row ------------------------------------------------
    if args.csv_out:
        rot_raw = rotation_stats(quat_array(cols, "q_raw"), "raw") or {}
        rot_filt = rotation_stats(quat_array(cols, "q_filt"), "filt") or {}
        row = {
            "file": os.path.basename(args.csv),
            "window_s": round(span, 3),
            "frames": processed,
            "dropped": n_dropped,
            "hz": round(processed / span, 2),
            "Rh_raw_rms_deg": rot_raw.get("rms_deg"),
            "Rh_filt_rms_deg": rot_filt.get("rms_deg"),
            "Rh_raw_step_rms_deg": rot_raw.get("step_rms_deg"),
            "pred_raw_rms_pt": (rp or {}).get("rms_pt"),
            "pred_filt_rms_pt": (fp or {}).get("rms_pt"),
            "pred_raw_step_pt": (rp or {}).get("step_median_pt"),
            "pred_filt_step_pt": (fp or {}).get("step_median_pt"),
        }
        for name, key, _n, med, _mean, p95, _mx in rows:
            row[f"{key}_median"] = round(med, 4) if med == med else None
            row[f"{key}_p95"] = round(p95, 4) if p95 == p95 else None
        with open(args.csv_out, "w", newline="") as fh:
            w = csvmod.DictWriter(fh, fieldnames=list(row))
            w.writeheader()
            w.writerow(row)
        print(f"\nwrote {args.csv_out}")


if __name__ == "__main__":
    main()
