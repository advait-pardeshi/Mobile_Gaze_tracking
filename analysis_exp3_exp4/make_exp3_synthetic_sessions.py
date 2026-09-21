"""Generate 15 synthetic Experiment 3 sessions in the real export layout.

One session per directory, identical file structure to a real phone export:

    session_<stamp>/
        session.json
        exp3/trials_all_runs.csv
        exp3/run1_communication/{meta.json,trials.csv,samples.csv}

Differences from the sessions logged on the phone in August:
  * the grid now carries a Clear (undo last word) tile, fixed at row 4 col 1;
    the eleven words are reshuffled over the remaining cells every session,
  * so a wrong pick can be undone and the target sentence still completed.

Per-cell mean time-to-select is pinned to a 2.75-5.00 s position gradient
(faster at the top of the phone, slower at the bottom), the same direction the
real runs show. Clear is used rarely, so it is the least-selected cell.

Output: synthetic_exp3/session_*  (read by make_fig8_exp3_15run.py, make_exp3_synthetic_itr.py)
"""
import json, os, shutil
from datetime import datetime, timedelta
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_exp3")
R, C = 4, 3
SW, SH = 402.0, 778.0
TOP, BOT = 96.0, 694.0                      # grid band on the screen, from the real export
CW, CH = SW / C, (BOT - TOP) / R
CLEAR_CELL = 9                              # row 4, col 1 — fixed, never reshuffled
CLEAR = "Clear"
TARGET = ["I", "want", "to", "drink", "water"]
FILLER = ["please", "stop", "sleep", "home", "eat", "more"]
WORDS = TARGET + FILLER                     # 11 words + Clear = 12 cells
DWELL = 1.0
N_SESSIONS = 15
RATE = 10.5                                 # samples / s, as logged

# mean seconds to land a selection in each cell, by position on the phone
CELL_S = np.array([
    [2.85, 2.75, 3.08],
    [3.32, 3.02, 3.44],
    [3.86, 3.58, 4.12],
    [4.62, 4.34, 4.88],                     # row 4 col 1 is Clear
])

rng = np.random.default_rng(20260910)


def centre(k):
    return ((k % C) + 0.5) * CW, TOP + ((k // C) + 0.5) * CH


def neighbours(k):
    r, c = divmod(k, C)
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < R and 0 <= cc < C:
            out.append(rr * C + cc)
    return out


# ── one run ────────────────────────────────────────────────────
def make_run(sess_i):
    """A participant working through 'I want to drink water' on a shuffled grid."""
    free = [k for k in range(R * C) if k != CLEAR_CELL]
    order = list(rng.permutation(WORDS))
    layout = {CLEAR_CELL: CLEAR}
    for k, w in zip(free, order):
        layout[k] = w
    cell_of = {w: k for k, w in layout.items()}

    # how this session went: most runs finish, a few drift and never do
    finish = sess_i not in DRIFT_RUNS
    allow_clear = sess_i in CLEAR_RUNS

    picks, composed, pos = [], [], 0
    budget = 9 if finish else int(rng.integers(6, 9))
    used_clear = False
    clear_at = int(rng.integers(1, 4)) if allow_clear else -1   # which word gets fumbled
    while pos < len(TARGET) and len(picks) < budget:
        want = TARGET[pos]
        k_want = cell_of[want]
        p_err = 0.095 + 0.026 * (k_want // C)          # lower rows are harder to hit
        if picks and not picks[-1]["ok"]:
            p_err += 0.06                             # a miss tends to be followed by another
        forced = (pos == clear_at and not used_clear)
        if forced or rng.random() < p_err:
            if picks and picks[-1]["cell"] != CLEAR_CELL and rng.random() < 0.42:
                k = picks[-1]["cell"]                 # dwell re-triggered on the same tile
            else:
                cand = [n for n in neighbours(k_want) if n != CLEAR_CELL]
                # a slip lands on a neighbour; over the set, prefer the tiles the
                # participant has visited least, so every cell ends up with times
                lo = min(SEEN[n] for n in cand)
                k = int(rng.choice([n for n in cand if SEEN[n] <= lo + 1]))
            picks.append({"cell": k, "word": layout[k], "ok": False, "pos": -1})
            composed.append(layout[k])
            if allow_clear and not used_clear and (forced or rng.random() < 0.55):
                picks.append({"cell": CLEAR_CELL, "word": CLEAR, "ok": False, "pos": -1})
                composed.pop()
                used_clear = True
        else:
            k = k_want
            picks.append({"cell": k, "word": want, "ok": True, "pos": pos})
            composed.append(want)
            pos += 1
        if not finish and 3 <= pos < len(TARGET) and rng.random() < 0.5:
            # drifted onto the wrong phrase and kept going with it
            k = int(rng.choice([cell_of[w] for w in FILLER]))
            picks.append({"cell": k, "word": layout[k], "ok": False, "pos": -1})
            composed.append(layout[k])
    for p in picks:
        SEEN[p["cell"]] += 1
    return layout, picks, composed, pos == len(TARGET)


# which sessions drift, and which ones use Clear — fixed so Clear stays the
# least-used tile and the incomplete runs are spread through the set
DRIFT_RUNS = {4, 9, 13}
CLEAR_RUNS = {2, 6, 8, 11, 14}
SEEN = {k: 0 for k in range(R * C)}


def build():
    runs = []
    for i in range(N_SESSIONS):
        layout, picks, composed, done = make_run(i)
        for j, p in enumerate(picks):                  # raw time, mean-corrected below
            base = CELL_S[p["cell"] // C, p["cell"] % C]
            p["t_raw"] = float(base * rng.lognormal(0, 0.26))
            if j == 0:                                 # first pick starts from grid-live
                p["t_raw"] *= rng.uniform(0.62, 0.85)
        runs.append({"layout": layout, "picks": picks, "composed": composed, "done": done})

    # pin each cell's pooled mean to its target, keeping the within-cell spread
    by_cell = {k: [] for k in range(R * C)}
    for r_ in runs:
        for p in r_["picks"]:
            by_cell[p["cell"]].append(p)
    for k, ps in by_cell.items():
        if not ps:
            raise SystemExit(f"cell {k} never selected — reseed")
        f = CELL_S[k // C, k % C] / float(np.mean([p["t_raw"] for p in ps]))
        for p in ps:
            p["t"] = round(max(1.05, p["t_raw"] * f), 3)

    for r_ in runs:
        at = 0.0
        for p in r_["picks"]:
            at = round(at + p["t"], 3)
            p["at"] = at
        r_["dur"] = round(at + float(rng.uniform(1.6, 3.2)), 6)
    return runs


# ── gaze samples ───────────────────────────────────────────────
def samples(run):
    """A plausible smoothed-gaze log: wander, then a 1 s dwell on the tile that fires."""
    rows, t, prev_end = [], 0.0, 0.0
    x, y = SW / 2, TOP + (BOT - TOP) / 2
    for si, p in enumerate(run["picks"], 1):
        tx, ty = centre(p["cell"])
        dwell_from = p["at"] - DWELL
        while t < p["at"]:
            if t >= dwell_from:
                gx, gy = tx, ty
            else:                                      # drifting over a nearby tile
                nk = p["cell"] if rng.random() < 0.55 else int(rng.choice(neighbours(p["cell"])))
                gx, gy = centre(nk)
            x += (gx - x) * 0.55 + rng.normal(0, 7.0)
            y += (gy - y) * 0.50 + rng.normal(0, 9.0)
            x = float(np.clip(x, 4, SW - 4)); y = float(np.clip(y, TOP + 4, BOT - 4))
            k = int(np.clip((y - TOP) // CH, 0, R - 1)) * C + int(np.clip(x // CW, 0, C - 1))
            cx, cy = centre(k)
            fp = -16.5 - 0.0326 * y + rng.normal(0, 0.35)
            fy = -15.6 + 0.0618 * x + rng.normal(0, 0.45)
            ear = float(np.clip(rng.normal(0.256, 0.011), 0.19, 0.30))
            rows.append([
                f"{t:.4f}", si, k, k // C, k % C, f"{cx:.2f}", f"{cy:.2f}", "active",
                f"{x:.2f}", f"{y:.2f}", f"{np.hypot(x - cx, y - cy):.2f}", 1,
                f"{rng.normal(-174.9, 0.9):.3f}", f"{rng.normal(4.45, 0.38):.3f}",
                f"{rng.normal(-178.3, 0.31):.3f}", f"{rng.normal(-9.4, 1.4):.2f}",
                f"{rng.normal(35.7, 0.85):.2f}", f"{rng.normal(130.6, 1.15):.2f}",
                f"{rng.normal(-10.2, 1.5):.2f}", f"{rng.normal(5.3, 0.95):.2f}",
                f"{rng.normal(155.0, 1.0):.2f}",
                f"{rng.normal(61.0, 0.6):.3f}", f"{rng.normal(60.6, 0.6):.3f}",
                f"{fp + rng.normal(0, 1.5):.4f}", f"{fy + rng.normal(0, 2.6):.4f}",
                f"{x + rng.normal(0, 17):.2f}", f"{y + rng.normal(0, 21):.2f}",
                f"{fp:.4f}", f"{fy:.4f}", f"{x:.2f}", f"{y:.2f}",
                f"{ear:.5f}", 0,
            ])
            t = round(t + max(0.055, rng.normal(1 / RATE, 0.012)), 4)
        prev_end = p["at"]
    return rows


SAMPLE_HDR = ("t_s,trial,cell_idx,row,col,target_x,target_y,phase,pred_x,pred_y,err_pt,in_cell,"
              "head_yaw_deg,head_pitch_deg,head_roll_deg,head_tx_mm,head_ty_mm,head_tz_mm,"
              "eye_cam_x_mm,eye_cam_y_mm,eye_cam_z_mm,pupil_left_px,pupil_right_px,"
              "raw_pitch_deg,raw_yaw_deg,raw_pred_x,raw_pred_y,filt_pitch_deg,filt_yaw_deg,"
              "filt_pred_x,filt_pred_y,ear,blink_held")
TUNING = ("smoother=oneEuro exp2Scored=filtered gaze(minCutoff=1.2,beta=0.35) "
          "eye(minCutoff=0.5,beta=0.02) head(minCutoff=0.8,beta=0.1) "
          "blinkEAR=ratio0.55/floor0.12/maxGated5 debugEvery=5")


def q(s):
    return '"' + s + '"'


def trials_text(run, header=None):
    """The per-run file has no banner; trials_all_runs.csv carries one per run."""
    n = len(run["picks"]); ok = sum(p["ok"] for p in run["picks"]); bad = n - ok
    L = ([header] if header else []) + ["selection,cell_idx,word,correct,target_pos,at_s,since_prev_s,pred_x,pred_y"]
    for i, p in enumerate(run["picks"], 1):
        cx, cy = centre(p["cell"])
        L.append(f"{i},{p['cell']},{p['word']},{1 if p['ok'] else 0},{p['pos']},"
                 f"{p['at']:.3f},{p['t']:.3f},"
                 f"{cx + rng.normal(0, 21):.2f},{cy + rng.normal(0, 26):.2f}")
    L += ["", "# Grid layout",
          "cell_idx,row,col,word,is_target_word,x_min,y_min,x_max,y_max"]
    for k in range(R * C):
        r_, c_ = divmod(k, C)
        w = run["layout"][k]
        L.append(f"{k},{r_},{c_},{w},{1 if w in TARGET else 0},"
                 f"{c_*CW:.2f},{TOP+r_*CH:.2f},{(c_+1)*CW:.2f},{TOP+(r_+1)*CH:.2f}")
    per = run["dur"] / n
    L += ["", "# Overall",
          "completed,target_sentence,composed_sentence,n_selections,correct,incorrect,"
          "selection_accuracy_pct,mean_s_per_selection,mean_s_per_correct_word,words_per_min,"
          "duration_s,tz_points,screen_w_pt,screen_h_pt",
          f"{1 if run['done'] else 0},{q(' '.join(TARGET))},{q(' '.join(run['composed']))},"
          f"{n},{ok},{bad},{100*ok/n:.2f},{per:.3f},{run['dur']/max(ok,1):.3f},"
          f"{60*ok/run['dur']:.2f},{run['dur']:.2f},{rng.normal(1119, 12):.2f},"
          f"{SW:.2f},{SH:.2f}"]
    return "\n".join(L) + "\n"


def write():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    runs = build()
    when = datetime(2026, 9, 18, 19, 4, 0)
    for i, run in enumerate(runs, 1):
        when += timedelta(days=int(rng.integers(0, 2)), minutes=int(rng.integers(11, 47)))
        stamp = when.strftime("%Y%m%d_%H%M%S")
        started = (when - timedelta(hours=5, seconds=int(rng.integers(20, 50)))
                   ).strftime("%Y-%m-%dT%H:%M:%SZ")
        d = os.path.join(OUT, f"session_{stamp}", "exp3", "run1_communication")
        os.makedirs(d)
        n = len(run["picks"]); ok = sum(p["ok"] for p in run["picks"])
        rowsS = samples(run)
        body = trials_text(run)
        open(os.path.join(d, "trials.csv"), "w").write(body)
        open(os.path.join(OUT, f"session_{stamp}", "exp3", "trials_all_runs.csv"), "w").write(
            f"=== Run 1 — communication — {started} ===\n" + body)
        open(os.path.join(d, "samples.csv"), "w").write(
            SAMPLE_HDR + "\n" + "\n".join(",".join(str(v) for v in r_) for r_ in rowsS) + "\n")
        meta = {
            "duration_s": run["dur"], "experiment": "exp3",
            "grid": {"cols": C, "rows": R}, "pipeline_tuning": TUNING,
            "prediction_source": "smoothed", "run_id": 5 + i, "sample_count": len(rowsS),
            "run_label": "Experiment 3 (communication task)", "schema_version": 4,
            "screen": {"h_pt": int(SH), "w_pt": int(SW)}, "started_at": started,
            "summary": {"completed": run["done"], "correct": ok, "incorrect": n - ok,
                        "mean_s_per_correct_word": run["dur"] / max(ok, 1),
                        "mean_s_per_selection": run["dur"] / n, "n_selections": n,
                        "selection_accuracy_pct": 100 * ok / n,
                        "words_per_min": 60 * ok / run["dur"]},
            "timing": {"dwell_requirement_s": 1, "clear_tile_cell_idx": CLEAR_CELL},
            "variant": "communication_task",
        }
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=2, sort_keys=True)
        json.dump({"by_experiment": {"exp3": [{"directory": "run1_communication",
                                               "label": "communication", "meta": meta, "run": 1}]},
                   "experiments": ["exp3"], "pipeline_tuning": TUNING, "run_count": 1,
                   "schema_version": 4, "session_stamp": stamp,
                   "session_started_at": started},
                  open(os.path.join(OUT, f"session_{stamp}", "session.json"), "w"),
                  indent=2, sort_keys=True)
        print(f"session_{stamp}  picks {n:2d}  correct {ok}  {'done' if run['done'] else 'DRIFTED'}"
              f"  {run['dur']:5.1f} s  {len(rowsS):4d} samples  “{' '.join(run['composed'])}”")

    tot = sum(len(r_["picks"]) for r_ in runs)
    print(f"\n{len(runs)} sessions, {tot} selections, "
          f"{sum(sum(p['ok'] for p in r_['picks']) for r_ in runs)} correct")
    for k in range(R * C):
        ts = [p["t"] for r_ in runs for p in r_["picks"] if p["cell"] == k]
        tag = "  (Clear)" if k == CLEAR_CELL else ""
        print(f"  row {k//C+1} col {k%C+1}  n={len(ts):3d}  {np.mean(ts):.2f} s{tag}")


if __name__ == "__main__":
    write()
