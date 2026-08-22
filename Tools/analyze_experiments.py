#!/usr/bin/env python3
"""Aggregate analysis across many grid_experiment_log CSV files.

Each input file is one or more "runs" appended by the in-app grid
experiment (`GridExperimentModel.appendToMasterLog`). This tool ingests
ANY number of such files (different participants / sessions / experiments)
and produces combined breakdowns:

  1. Per file / participant   — one row per CSV: accuracy, mean error, n
  2. Per grid size            — 3x3, 5x4, ... aggregated over all files
  3. Distance vs accuracy     — binned by head_tz_mm (face-to-camera dist)
  4. Head pose vs accuracy    — binned by |head_yaw|, |head_pitch|

Outputs (written next to --outdir):
  - printed tables (stdout)
  - one tidy CSV per breakdown          (by_file.csv, by_grid.csv, ...)
  - one combined long CSV of every trial (all_trials.csv)
  - plots (PNG) for each breakdown       (requires matplotlib)

Usage:
  python3 analyze_experiments.py FILE_OR_DIR [FILE_OR_DIR ...] [-o OUTDIR]
  python3 analyze_experiments.py ~/Downloads/person-*/*.csv -o ~/Desktop/report

A directory argument is searched recursively for *.csv.
The participant label for each file defaults to the parent folder name
(e.g. ".../person-2/grid_experiment_log 4.csv" -> "person-2"); override
per file with "label=PATH" (e.g. p3=~/Downloads/run.csv).
"""
import argparse
import csv
import glob
import os
import re
import sys
from collections import defaultdict

# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------

RUN_RE = re.compile(r"=== Run \d+ @ .+ — (.+) ===")


def _f(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return float("nan")


def parse_file(path, label):
    """Return a list of trial dicts for every run in one CSV file."""
    with open(path, newline="") as f:
        lines = f.readlines()

    trials = []
    header = None
    run_label = None
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        m = RUN_RE.match(line)
        if m:
            run_label = m.group(1)
            header = None
            i += 1
            continue
        if line.startswith("trial,"):
            header = line.split(",")
            i += 1
            continue
        if line.startswith("# Overall"):
            header = None          # skip the 2-line overall block
            i += 3
            continue
        if header and line and line[0].isdigit():
            d = dict(zip(header, line.split(",")))

            # hit: grid logs have a 0/1 "hit" column; experiment2/3 logs
            # instead carry an "outcome" string (hit / miss / timeout).
            if "hit" in d:
                hit = int(_f(d.get("hit")) or 0)
            elif "outcome" in d:
                hit = 1 if d["outcome"].strip().lower() == "hit" else 0
            else:
                hit = 0

            # err_pt: present in grid logs; for experiment2/3 derive it from
            # the prediction vs. the cell center (cell_*_min/max columns).
            err = _f(d.get("err_pt"))
            if err != err and "pred_x" in d and "cell_x_min" in d:
                cx = 0.5 * (_f(d["cell_x_min"]) + _f(d["cell_x_max"]))
                cy = 0.5 * (_f(d["cell_y_min"]) + _f(d["cell_y_max"]))
                px, py = _f(d.get("pred_x")), _f(d.get("pred_y"))
                err = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5

            trials.append({
                "file": label,
                "run_label": run_label,
                "row": int(_f(d.get("row", -1)) or -1),
                "col": int(_f(d.get("col", -1)) or -1),
                "hit": hit,
                "err_pt": err,
                "head_tz_mm": _f(d.get("head_tz_mm")),
                "head_yaw_deg": _f(d.get("head_yaw_deg")),
                "head_pitch_deg": _f(d.get("head_pitch_deg")),
                "n_samples": int(_f(d.get("n_samples", 0)) or 0),
            })
        i += 1
    return trials


def collect_inputs(args):
    """Expand file/dir/glob args into [(label, path), ...]."""
    out = []
    for a in args:
        label = None
        if "=" in a and not os.path.exists(a):
            label, a = a.split("=", 1)
        a = os.path.expanduser(a)
        paths = []
        if os.path.isdir(a):
            paths = sorted(glob.glob(os.path.join(a, "**", "*.csv"), recursive=True))
        elif any(c in a for c in "*?[]"):
            paths = sorted(glob.glob(a, recursive=True))
        elif os.path.exists(a):
            paths = [a]
        else:
            print(f"  ! skipped (not found): {a}", file=sys.stderr)
        for p in paths:
            lbl = label or os.path.basename(os.path.dirname(os.path.abspath(p))) or os.path.basename(p)
            out.append((lbl, p))
    return out


# ----------------------------------------------------------------------------
# Aggregation helpers
# ----------------------------------------------------------------------------

def _normalize_yaw(y):
    """Front-camera yaw sits near +-180; fold to deviation-from-frontal."""
    if y != y:  # nan
        return y
    return abs(abs(y) - 180.0) if abs(y) > 90 else abs(y)


def summarize(trials):
    n = len(trials)
    hits = sum(t["hit"] for t in trials)
    errs = [t["err_pt"] for t in trials if t["err_pt"] == t["err_pt"]]
    mean_err = sum(errs) / len(errs) if errs else float("nan")
    acc = 100.0 * hits / n if n else float("nan")
    return n, hits, acc, mean_err


def group_by(trials, keyfn):
    g = defaultdict(list)
    for t in trials:
        k = keyfn(t)
        if k is not None:
            g[k].append(t)
    return g


def bin_label(v, edges):
    """Return 'lo-hi' label for value v given ascending bin edges."""
    if v != v:
        return None
    for lo, hi in zip(edges, edges[1:]):
        if lo <= v < hi:
            return f"{lo:g}-{hi:g}"
    if v >= edges[-1]:
        return f"{edges[-1]:g}+"
    return f"<{edges[0]:g}"


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------

def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def print_table(title, header, rows):
    print(f"\n### {title}")
    widths = [len(h) for h in header]
    srows = [[str(c) for c in r] for r in rows]
    for r in srows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    fmt = "  ".join("{:<%d}" % w for w in widths)
    print(fmt.format(*header))
    print(fmt.format(*["-" * w for w in widths]))
    for r in srows:
        print(fmt.format(*r))


def maybe_plot_bars(outdir, name, title, labels, accs, ns):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, ax = plt.subplots(figsize=(max(5, 0.7 * len(labels) + 2), 4))
    bars = ax.bar(range(len(labels)), accs, color="#3b78c3")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("accuracy %")
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=11)
    for b, a, nn in zip(bars, accs, ns):
        ax.text(b.get_x() + b.get_width() / 2, a + 1.5,
                f"{a:.0f}%\nn={nn}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    p = os.path.join(outdir, name)
    fig.savefig(p, dpi=130)
    plt.close(fig)
    return p


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="CSV files, directories (recursed), globs, or label=path")
    ap.add_argument("-o", "--outdir", default="experiment_report",
                    help="output directory (default: ./experiment_report)")
    ap.add_argument("--dist-edges", default="100,130,140,150,160,180",
                    help="head_tz_mm bin edges, comma-separated")
    ap.add_argument("--angle-edges", default="0,2,4,6,8,10",
                    help="|yaw|/|pitch| deg bin edges, comma-separated")
    args = ap.parse_args()

    inputs = collect_inputs(args.inputs)
    if not inputs:
        print("No input CSVs found.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.outdir, exist_ok=True)

    all_trials = []
    for label, path in inputs:
        t = parse_file(path, label)
        all_trials.extend(t)
        print(f"  loaded {len(t):4d} trials  [{label}]  {path}")

    n, hits, acc, merr = summarize(all_trials)
    print("\n" + "=" * 64)
    print(f"COMBINED: {len(inputs)} file(s), {n} trials, "
          f"{hits} hits = {acc:.1f}% accuracy, mean error {merr:.1f} pt")
    print("=" * 64)

    dist_edges = [float(x) for x in args.dist_edges.split(",")]
    angle_edges = [float(x) for x in args.angle_edges.split(",")]

    # 1) combined long CSV ----------------------------------------------------
    long_cols = ["file", "run_label", "row", "col", "hit", "err_pt",
                 "head_tz_mm", "head_yaw_deg", "head_pitch_deg", "n_samples"]
    write_csv(os.path.join(args.outdir, "all_trials.csv"),
              long_cols, [[t[c] for c in long_cols] for t in all_trials])

    # 2) per file -------------------------------------------------------------
    g = group_by(all_trials, lambda t: t["file"])
    rows = []
    for k in sorted(g):
        nn, hh, aa, ee = summarize(g[k])
        rows.append([k, nn, hh, f"{aa:.1f}", f"{ee:.1f}"])
    hdr = ["file", "n", "hits", "accuracy_%", "mean_err_pt"]
    print_table("Per file / participant", hdr, rows)
    write_csv(os.path.join(args.outdir, "by_file.csv"), hdr, rows)
    maybe_plot_bars(args.outdir, "by_file.png", "Accuracy per file",
                    [r[0] for r in rows], [float(r[3]) for r in rows],
                    [r[1] for r in rows])

    # 3) per grid size --------------------------------------------------------
    g = group_by(all_trials, lambda t: t["run_label"])
    rows = []
    for k in sorted(g, key=lambda s: (len(s), s)):
        nn, hh, aa, ee = summarize(g[k])
        rows.append([k, nn, hh, f"{aa:.1f}", f"{ee:.1f}"])
    hdr = ["grid", "n", "hits", "accuracy_%", "mean_err_pt"]
    print_table("Per grid size (across all files)", hdr, rows)
    write_csv(os.path.join(args.outdir, "by_grid.csv"), hdr, rows)
    maybe_plot_bars(args.outdir, "by_grid.png", "Accuracy per grid size",
                    [r[0] for r in rows], [float(r[3]) for r in rows],
                    [r[1] for r in rows])

    # 4) distance vs accuracy -------------------------------------------------
    g = group_by(all_trials, lambda t: bin_label(t["head_tz_mm"], dist_edges))
    order = sorted(g, key=lambda s: _f(re.split("[-+<]", s.lstrip("<"))[0]))
    rows = []
    for k in order:
        nn, hh, aa, ee = summarize(g[k])
        rows.append([k, nn, hh, f"{aa:.1f}", f"{ee:.1f}"])
    hdr = ["dist_mm", "n", "hits", "accuracy_%", "mean_err_pt"]
    print_table("Distance (head_tz_mm) vs accuracy", hdr, rows)
    write_csv(os.path.join(args.outdir, "by_distance.csv"), hdr, rows)
    maybe_plot_bars(args.outdir, "by_distance.png",
                    "Accuracy vs face-to-camera distance (mm)",
                    [r[0] for r in rows], [float(r[3]) for r in rows],
                    [r[1] for r in rows])

    # 5) head pose vs accuracy ------------------------------------------------
    for axis, key in (("yaw", "head_yaw_deg"), ("pitch", "head_pitch_deg")):
        normf = _normalize_yaw if axis == "yaw" else (lambda v: abs(v))
        g = group_by(all_trials, lambda t: bin_label(normf(t[key]), angle_edges))
        order = sorted(g, key=lambda s: _f(re.split("[-+<]", s.lstrip("<"))[0]))
        rows = []
        for k in order:
            nn, hh, aa, ee = summarize(g[k])
            rows.append([k, nn, hh, f"{aa:.1f}", f"{ee:.1f}"])
        hdr = [f"|{axis}|_deg", "n", "hits", "accuracy_%", "mean_err_pt"]
        print_table(f"Head {axis} vs accuracy", hdr, rows)
        write_csv(os.path.join(args.outdir, f"by_{axis}.csv"), hdr, rows)
        maybe_plot_bars(args.outdir, f"by_{axis}.png",
                        f"Accuracy vs head {axis} (deg from frontal)",
                        [r[0] for r in rows], [float(r[3]) for r in rows],
                        [r[1] for r in rows])

    print(f"\nWrote tables + plots to: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()
