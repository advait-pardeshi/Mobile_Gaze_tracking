"""Experiment 4 — time taken to select each word, as an Excel workbook.

Sheets
  Every selection  one row per selection, in order, with its time
  By word          each distinct word, how often and how long on average
  By position      the 3x3 grid position map
  By answer        one row per answer given
  Summary          headline numbers
"""
import csv, os, glob
import numpy as np
import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

ROOT = "/Users/advait/Downloads/EXP 3"
OUT  = "/Users/advait/Desktop/GazeTracking/Mobile_Gaze_tracking-main/analysis_exp3_exp4"
XL   = os.path.join(OUT, "exp4_word_times.xlsx")
S = sorted(glob.glob(os.path.join(ROOT, "session_*")))

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

FN = {"word": "word", "back": "BACK (control)", "done": "DONE (control)"}
sel_rows, ans_rows = [], []

for s in S:
    sess = os.path.basename(s).replace("session_", "")
    blocks = sect(os.path.join(s, "exp4/run1_predictive_cued/trials.csv"))
    sel, states, geom, trials = blocks[0], blocks[1], blocks[2], blocks[3]
    kind = {int(g["cell_idx"]): g["kind_at_start"] for g in geom}
    qtext = {t["question"]: t["question_text"] for t in trials}
    # candidates offered at each grid step, to recover the word's rank
    cand = {g["grid_step"]: g["candidates"].split() for g in states}

    n_in_trial, cur_trial = 0, None
    for r in sel:
        if r["trial"] != cur_trial:
            cur_trial, n_in_trial = r["trial"], 0
        n_in_trial += 1
        k = int(r["cell_idx"])
        secs = float(r["since_prev_s"])
        offered = cand.get(r["grid_step"], [])
        sel_rows.append({
            "Session": sess,
            "Question": qtext.get(r["question"], r["question"]),
            "Answer_word_no": n_in_trial,
            "Word": r["word"] if r["word"] else FN[kind[k]],
            "Cell": k + 1,
            "Grid_row": k // 3 + 1,
            "Grid_col": k % 3 + 1,
            "Cell_type": FN[kind[k]],
            "Rank_on_grid": int(r["chosen_rank"]) + 1 if int(r["chosen_rank"]) >= 0 else None,
            "Words_offered": len(offered),
            "Seconds_to_select": round(secs, 3),
            "Milliseconds": round(secs * 1000),
            "Elapsed_in_run_s": round(float(r["at_s"]), 3),
        })
    for t in trials:
        ans_rows.append({
            "Session": sess,
            "Question": t["question_text"],
            "Answer_composed": t["composed"],
            "Words": int(t["word_selections"]),
            "Answer_time_s": round(float(t["response_s"]), 3),
            "Seconds_per_word": round(float(t["response_s"]) / int(t["word_selections"]), 3)
                                if int(t["word_selections"]) else None,
            "Corrections": int(t["corrections"]),
        })

sel_df = pd.DataFrame(sel_rows)
ans_df = pd.DataFrame(ans_rows)
words = sel_df[sel_df["Cell_type"] == "word"]

by_word = (words.groupby("Word")
           .agg(Times_selected=("Seconds_to_select", "size"),
                Mean_s=("Seconds_to_select", "mean"),
                Median_s=("Seconds_to_select", "median"),
                Fastest_s=("Seconds_to_select", "min"),
                Slowest_s=("Seconds_to_select", "max"),
                Usual_cell=("Cell", lambda c: int(c.mode().iloc[0])))
           .round(3).sort_values(["Times_selected", "Mean_s"], ascending=[False, True])
           .reset_index())

by_pos = (sel_df.groupby(["Grid_row", "Grid_col"])
          .agg(Selections=("Seconds_to_select", "size"),
               Mean_s=("Seconds_to_select", "mean"),
               Median_s=("Seconds_to_select", "median"),
               Fastest_s=("Seconds_to_select", "min"),
               Slowest_s=("Seconds_to_select", "max"))
          .round(3).reset_index())
by_pos["Cell"] = (by_pos["Grid_row"] - 1) * 3 + by_pos["Grid_col"]
missing = [(r, c) for r in (1, 2, 3) for c in (1, 2, 3)
           if not ((by_pos["Grid_row"] == r) & (by_pos["Grid_col"] == c)).any()]
for r, c in missing:
    by_pos.loc[len(by_pos)] = {"Grid_row": r, "Grid_col": c, "Selections": 0,
                               "Mean_s": None, "Median_s": None, "Fastest_s": None,
                               "Slowest_s": None, "Cell": (r - 1) * 3 + c}
by_pos = by_pos.sort_values("Cell")[["Cell", "Grid_row", "Grid_col", "Selections",
                                     "Mean_s", "Median_s", "Fastest_s", "Slowest_s"]]

summary = pd.DataFrame([
    ("Sessions", len(S)),
    ("Answers given", len(ans_df)),
    ("Word selections", len(words)),
    ("All selections (words + DONE)", len(sel_df)),
    ("Distinct words selected", words["Word"].nunique()),
    ("Mean seconds per word", round(words["Seconds_to_select"].mean(), 3)),
    ("Median seconds per word", round(words["Seconds_to_select"].median(), 3)),
    ("Fastest single word (s)", round(words["Seconds_to_select"].min(), 3)),
    ("Slowest single word (s)", round(words["Seconds_to_select"].max(), 3)),
    ("Mean seconds per answer", round(ans_df["Answer_time_s"].mean(), 3)),
    ("Mean words per answer", round(ans_df["Words"].mean(), 2)),
    ("Corrections (BACK presses)", int(ans_df["Corrections"].sum())),
    ("Word dwell threshold (s)", 1.0),
    ("Control dwell threshold (s)", 1.5),
    ("Grid refresh lockout (s)", 0.4),
], columns=["Measure", "Value"])

with pd.ExcelWriter(XL, engine="openpyxl") as w:
    sel_df.to_excel(w, sheet_name="Every selection", index=False)
    by_word.to_excel(w, sheet_name="By word", index=False)
    by_pos.to_excel(w, sheet_name="By position", index=False)
    ans_df.to_excel(w, sheet_name="By answer", index=False)
    summary.to_excel(w, sheet_name="Summary", index=False)

# ── formatting (matches the other workbooks) ───────────────────
from openpyxl import load_workbook
wb = load_workbook(XL)
HDR_FILL = PatternFill("solid", fgColor="12161B")
HDR_FONT = Font(color="FFFFFF", bold=True, size=10, name="Calibri")
BAND = PatternFill("solid", fgColor="F2F5F8")
thin = Side(style="thin", color="D6DBE0")
for ws in wb.worksheets:
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for c in ws[1]:
        c.fill = HDR_FILL; c.font = HDR_FONT
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[1].height = 30
    for col in ws.columns:
        L = max((len(str(c.value)) for c in col if c.value is not None), default=8)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max(L + 3, 11), 44)
    for i, row in enumerate(ws.iter_rows(min_row=2)):
        for c in row:
            c.border = Border(bottom=thin)
            if isinstance(c.value, float): c.number_format = "0.000"
            if i % 2: c.fill = BAND
    for c in ws[1]:
        if str(c.value) in ("Question", "Answer_composed", "Word"):
            ws.column_dimensions[c.column_letter].width = 30
wb.save(XL)

print(f"wrote {XL}")
print(f"  {len(sel_df)} selections, {len(words)} words, {len(ans_df)} answers")
print(f"  mean {words['Seconds_to_select'].mean():.2f} s per word "
      f"(median {words['Seconds_to_select'].median():.2f}, "
      f"range {words['Seconds_to_select'].min():.2f}–{words['Seconds_to_select'].max():.2f})")
