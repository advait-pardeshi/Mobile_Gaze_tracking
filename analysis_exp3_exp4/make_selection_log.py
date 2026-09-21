"""Build exp3_exp4_selection_log.xlsx — the per-selection word log for Experiments 3 and 4.

Reads the derived tables in exp3_exp4_results.xlsx (T5 / T7 / T8) and re-emits them as a
reader-facing workbook: what word was selected, in what order, and how long each pick took.
No re-scoring — every value is carried across unchanged.
"""
import os
from openpyxl import load_workbook, Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "exp3_exp4_results.xlsx")
OUT = os.path.join(HERE, "exp3_exp4_selection_log.xlsx")

# house palette, matching make_figures.py
INK = "12161B"
BLUE = "2A78D6"
ORANGE = "EB6834"
GREEN = "1BAF7A"
BLUE_SOFT = "E3EEFB"
ORANGE_SOFT = "FDEAE1"
GREEN_SOFT = "E0F5EE"
GREY_SOFT = "EEF1F4"
BAND = "F5F8FB"

HEAD_FILL = PatternFill("solid", fgColor=INK)
HEAD_FONT = Font(name="Calibri", bold=True, size=10, color="FFFFFF")
TITLE_FONT = Font(name="Calibri", bold=True, size=13, color=INK)
NOTE_FONT = Font(name="Calibri", italic=True, size=9, color="4C5763")
THIN = Side(style="thin", color="D6DBE0")
EDGE = Border(bottom=THIN)


def read(sheet):
    ws = load_workbook(SRC)[sheet]
    rows = list(ws.iter_rows(values_only=True))
    head = list(rows[0])
    return head, [list(r) for r in rows[1:] if r[0] is not None]


def col(head, name):
    return head.index(name)


def write_sheet(wb, title, headers, rows, notes, fills=None, widths=None, freeze="A4"):
    ws = wb.create_sheet(title)
    ws.sheet_view.showGridLines = False

    ws["A1"] = notes[0]
    ws["A1"].font = TITLE_FONT
    ws["A2"] = notes[1]
    ws["A2"].font = NOTE_FONT

    hrow = 3
    for j, h in enumerate(headers, start=1):
        c = ws.cell(row=hrow, column=j, value=h)
        c.fill = HEAD_FILL
        c.font = HEAD_FONT
        c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
    ws.row_dimensions[hrow].height = 28

    for i, r in enumerate(rows):
        rn = hrow + 1 + i
        fill = fills(r) if fills else None
        for j, v in enumerate(r, start=1):
            c = ws.cell(row=rn, column=j, value=v)
            c.border = EDGE
            if isinstance(v, float):
                c.number_format = "0.000" if abs(v) < 100 else "0.00"
                c.alignment = Alignment(horizontal="right")
            elif isinstance(v, int):
                c.alignment = Alignment(horizontal="right")
            if fill:
                c.fill = fill

    last = hrow + len(rows)
    ws.auto_filter.ref = f"A{hrow}:{get_column_letter(len(headers))}{last}"
    ws.freeze_panes = freeze
    for j, w in enumerate(widths or [], start=1):
        ws.column_dimensions[get_column_letter(j)].width = w
    return ws


wb = Workbook()
wb.remove(wb.active)

# ── Sheet 1: Experiment 3, every selection ────────────────────────────
h, rows = read("T5 Exp3 selections")
i = lambda n: col(h, n)
e3 = [[
    r[i("Session")], r[i("Selection")], r[i("Word")], r[i("Cell_idx")],
    "yes" if r[i("Correct")] == 1 else "no",
    r[i("Target_pos")] if r[i("Target_pos")] >= 0 else None,
    r[i("At_s")], r[i("Since_prev_s")],
    r[i("Offset_from_centre_pt")], r[i("Offset_norm_halfcell")],
    r[i("Outcome_class")],
] for r in rows]

E3_FILL = {
    "correct": PatternFill("solid", fgColor=BLUE_SOFT),
    "spatial substitution": PatternFill("solid", fgColor=ORANGE_SOFT),
    "unintended re-selection": PatternFill("solid", fgColor=GREEN_SOFT),
}
write_sheet(
    wb, "Exp3 selections",
    ["Run (session)", "#", "Word", "Cell", "Correct", "Target position",
     "At (s)", "Gap since prev (s)", "Offset from centre (pt)",
     "Offset (1.0 = border)", "Outcome"],
    e3,
    ("Experiment 3 — every word selected, static 4x3 grid, 1.0 s dwell",
     "Target sentence 'I want to drink water' in all 5 runs; tiles 134 x 149 pt, reshuffled per run. "
     "Offset is Chebyshev-normalised: 1.0 means the gaze landed on the nearest tile border."),
    fills=lambda r: E3_FILL.get(r[10]),
    widths=[14, 5, 11, 6, 9, 15, 9, 17, 21, 20, 22],
)

# ── Sheet 2: Experiment 4, trial-level question -> answer ─────────────
h, rows = read("T8 Exp4 trials")
i = lambda n: col(h, n)
e4t = [[
    r[i("Session")], r[i("Trial")], r[i("Question_text")], r[i("Cued_answer")],
    r[i("Composed")], "yes" if r[i("Completed")] == 1 else "no",
    r[i("Word_selections")], r[i("Ideal_selections")], r[i("Correct")],
    r[i("Predictor_hits")], r[i("Predictor_opportunities")],
    r[i("Prompt_s")], r[i("Response_s")], r[i("First_divergence")],
] for r in rows]

write_sheet(
    wb, "Exp4 trials",
    ["Session", "Trial", "Question asked", "Cued answer", "Answer composed",
     "Completed", "Words selected", "Ideal words", "Correct",
     "Predictor hits", "Predictor chances", "Prompt (s)", "Response (s)",
     "First divergence"],
    e4t,
    ("Experiment 4 — what was asked and what was answered, predictive 3x3 grid",
     "0 of 15 trials reproduced the cue. Every first divergence is 'word present, other cell taken' — "
     "a deliberate branch choice, not an absent word. Response (s) is the scored response phase."),
    fills=lambda r: PatternFill("solid", fgColor=ORANGE_SOFT),
    widths=[11, 6, 30, 24, 34, 11, 15, 12, 9, 14, 16, 11, 12, 42],
)

# ── Sheet 3: Experiment 4, every activation ───────────────────────────
h, rows = read("T7 Exp4 selections")
i = lambda n: col(h, n)
e4s = [[
    r[i("Session")], r[i("Trial")], r[i("Question")], r[i("Grid_step")],
    r[i("Kind")], r[i("Word_chosen")] or "(done)",
    "yes" if r[i("Correct")] == 1 else "no",
    r[i("Intended_word")],
    r[i("Chosen_rank")] if r[i("Chosen_rank")] >= 0 else None,
    r[i("Cell_idx")], r[i("Cell_distance")],
    r[i("At_s")], r[i("Since_prev_s")],
    r[i("Scorability")], r[i("Attribution")],
] for r in rows]

E4_FILL = {
    "correct": PatternFill("solid", fgColor=BLUE_SOFT),
    "adjacent cell (tracker-plausible)": PatternFill("solid", fgColor=ORANGE_SOFT),
    "two cells away (NOT tracker)": PatternFill("solid", fgColor=ORANGE_SOFT),
    "downstream of divergence": PatternFill("solid", fgColor=GREY_SOFT),
    "control activation": PatternFill("solid", fgColor=BAND),
}
write_sheet(
    wb, "Exp4 selections",
    ["Session", "Trial", "Question", "Step", "Kind", "Word chosen", "Correct",
     "Word needed", "Rank offered", "Cell", "Cells from target",
     "At (s)", "Gap since prev (s)", "Scorability", "Attribution"],
    e4s,
    ("Experiment 4 — every activation, 65 words + 15 'done' controls",
     "Cells are 134 x 199 pt. Two cells apart is >=268 pt of displacement — beyond any 1.5 deg "
     "estimator, so those picks are deliberate. Word dwell 1.0 s; 'done' control dwell 1.5 s."),
    fills=lambda r: E4_FILL.get(r[14]),
    widths=[11, 6, 10, 6, 7, 13, 9, 13, 13, 6, 17, 9, 17, 24, 32],
)

# ── Sheet 4: timing summary ───────────────────────────────────────────
h, rows = read("T9b Timing summary")
ws = write_sheet(
    wb, "Timing",
    list(h), [list(r) for r in rows],
    ("Interval between confirmed selections",
     "Search overhead is the interval minus the dwell the system charges. Note: Exp 4 also carries a "
     "0.4 s grid-refresh lockout, which is currently counted inside the search figure."),
    widths=[26] + [11] * (len(h) - 1),
)

# ── Sheet 5: provenance ───────────────────────────────────────────────
ws = wb.create_sheet("Notes")
ws.sheet_view.showGridLines = False
ws.column_dimensions["A"].width = 22
ws.column_dimensions["B"].width = 110
ws["A1"] = "Selection log — Experiments 3 and 4"
ws["A1"].font = TITLE_FONT
notes = [
    ("Sessions", "5 sessions on 2026-08-26, 20:08:10-20:36:31. The export records no participant "
                 "identifier — these are 5 sessions, not 5 participants."),
    ("Source", "Derived from exp3_exp4_results.xlsx sheets T5, T7, T8, T9b, which are themselves "
               "computed from EXP 3/session_20260826_* (schema v4). No re-scoring, no exclusions."),
    ("Times", "All times are seconds from the start of that run. 'Gap since prev' is the interval "
              "from the previous confirmed selection, so it includes the 1.0 s dwell."),
    ("Cells", "Cell indices are row-major. Exp 3 tiles 134 x 149 pt; Exp 4 cells 134 x 199 pt."),
    ("Exp 3 caution", "The 5 picks classed 'spatial substitution' average 0.479 normalised offset — "
                      "more central than the 21 correct picks at 0.592. Two runs composed the coherent "
                      "alternative 'I want to sleep more'. These may be deliberate choices rather than "
                      "tracker error, in which case 75% understates per-selection reliability."),
    ("Exp 4 caution", "Selection accuracy as logged (16.9%) is NOT valid — the cued condition did not "
                      "hold. Use conditional accuracy (42.3%, 11 of 26 scorable) instead."),
    ("Pipeline", "One-Euro smoother: gaze (min_cutoff 1.2, beta 0.35), eye (0.5, 0.02), head (0.8, 0.1); "
                 "blink EAR ratio 0.55 / floor 0.12 / max 5 gated frames."),
]
for k, (a, b) in enumerate(notes, start=3):
    ws.cell(row=k, column=1, value=a).font = Font(bold=True, size=10, color=INK)
    c = ws.cell(row=k, column=2, value=b)
    c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[k].height = 42

wb.save(OUT)
print("written:", OUT)
for s in wb.sheetnames:
    print("  %-18s %d rows" % (s, wb[s].max_row))
