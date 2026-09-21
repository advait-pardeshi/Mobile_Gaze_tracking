"""Information Transfer Rate (ITR) for Experiments 3 and 4.

ITR is the bits-per-minute metric standard in the BCI / gaze-speller
literature (Wolpaw et al. 1998, 2000). It exists because words-per-minute
alone is not comparable across interfaces: it ignores how many alternatives
a selection chose *between*, and it does not charge for errors. Exp 3 picks
1-of-12 and Exp 4 picks 1-of-9, so their WPM figures are not on the same
axis; their ITRs are.

    B   = log2(N) + P*log2(P) + (1-P)*log2((1-P)/(N-1))      bits / selection
    ITR = B * (60 / T)                                       bits / minute

  N  number of equiprobable selectable targets on screen
  P  probability a selection is the intended one
  T  mean seconds between consecutive selections (dwell + saccade + search)

Assumptions, all of which this pipeline violates to some degree (see the
Method sheet): equiprobable targets, errors spread uniformly over the N-1
non-targets, and independent selections.

Reads:  exp3_exp4_results.xlsx   (built by make_workbook.py)
Writes: exp3_exp4_itr.xlsx, fig9_itr.png
"""
import os
from math import log2

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

OUT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(OUT, "exp3_exp4_results.xlsx")
XL  = os.path.join(OUT, "exp3_exp4_itr.xlsx")
FIG = os.path.join(OUT, "fig9_itr.png")

N3, N4 = 12, 9          # 4x3 word tiles (Exp 3); 3x3 cells (Exp 4)
DWELL3, DWELL4 = 1.0, 1.0

INK, INK2, INK3 = "#12161b", "#4c5763", "#78838f"
LINE = "#d6dbe0"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": LINE, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9.5,
    "xtick.color": INK3, "ytick.color": INK3,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.frameon": False, "legend.fontsize": 8.5,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def tidy(ax, xgrid=False, ygrid=True):
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.xaxis.grid(xgrid, **({"color": "#e6eaed", "lw": 0.7} if xgrid else {}))
    ax.yaxis.grid(ygrid, **({"color": "#e6eaed", "lw": 0.7} if ygrid else {}))
    ax.set_axisbelow(True)


def cap(fig, text):
    fig.text(0.0, -0.02, text, ha="left", va="top", fontsize=7.8,
             color=INK3, wrap=True, transform=fig.transFigure)


def bits(N, P):
    """Wolpaw bits per selection. Chance or worse carries no information."""
    if not np.isfinite(P) or P <= 1.0 / N:
        return 0.0
    if P >= 1.0:
        return log2(N)
    return log2(N) + P * log2(P) + (1 - P) * log2((1 - P) / (N - 1))


def itr(N, P, T):
    return bits(N, P) * 60.0 / T if np.isfinite(T) and T > 0 else np.nan


# ── Experiment 3 ──────────────────────────────────────────────────────
T4 = pd.read_excel(SRC, sheet_name="T4 Exp3 runs")
T5 = pd.read_excel(SRC, sheet_name="T5 Exp3 selections")

e3 = []
for _, r in T4.iterrows():
    sel = T5[T5.Session == r.Session]
    P, T = r.Selection_accuracy_pct / 100.0, sel.Since_prev_s.mean()
    e3.append(dict(
        Session=r.Session, N_targets=N3, Selections=int(r.N_selections),
        Correct=int(r.Correct), Accuracy_P=round(P, 4),
        Mean_s_per_selection=round(T, 3),
        Selections_per_min=round(60 / T, 2),
        Bits_per_selection=round(bits(N3, P), 3),
        ITR_bits_per_min=round(itr(N3, P, T), 2),
        Words_per_min=r.Words_per_min,
        ITR_ceiling_at_P1=round(log2(N3) * 60 / T, 2)))
E3 = pd.DataFrame(e3)

P3 = T5.Correct.sum() / len(T5)
T3, T3m = T5.Since_prev_s.mean(), T5.Since_prev_s.median()
E3_pool = dict(Session="POOLED", N_targets=N3, Selections=len(T5),
               Correct=int(T5.Correct.sum()), Accuracy_P=round(P3, 4),
               Mean_s_per_selection=round(T3, 3),
               Selections_per_min=round(60 / T3, 2),
               Bits_per_selection=round(bits(N3, P3), 3),
               ITR_bits_per_min=round(itr(N3, P3, T3), 2),
               Words_per_min=round(T4.Words_per_min.mean(), 2),
               ITR_ceiling_at_P1=round(log2(N3) * 60 / T3, 2))
E3 = pd.concat([E3, pd.DataFrame([E3_pool])], ignore_index=True)

# ── Experiment 4 ──────────────────────────────────────────────────────
# The cued condition did not hold (PAPER.md 5.6): the raw accuracy is not a
# selection accuracy. Only the conditional accuracy — picks made while the
# needed word was actually on the grid — is reportable, so the conditional
# ITR is the headline and the as-logged one is carried only as a floor.
T6 = pd.read_excel(SRC, sheet_name="T6 Exp4 runs")
T7 = pd.read_excel(SRC, sheet_name="T7 Exp4 selections")
W7 = T7[T7.Kind == "word"]

e4 = []
for _, r in T6.iterrows():
    w = W7[W7.Session == r.Session]
    sc = w[w.Scorability == "scorable"]
    P = sc.Correct.sum() / len(sc) if len(sc) else np.nan
    T = w.Since_prev_s.mean()
    e4.append(dict(
        Session=r.Session, N_targets=N4,
        Word_selections=int(r.Word_selections), Scorable=len(sc),
        Conditional_P=round(P, 4), Mean_s_per_selection=round(T, 3),
        Selections_per_min=round(60 / T, 2),
        Bits_per_selection=round(bits(N4, P), 3),
        ITR_conditional=round(itr(N4, P, T), 2),
        ITR_as_logged=round(itr(N4, r.Selection_accuracy_pct / 100.0, T), 2),
        ITR_ceiling_at_P1=round(log2(N4) * 60 / T, 2)))
E4 = pd.DataFrame(e4)

SC = W7[W7.Scorability == "scorable"]
P4 = SC.Correct.sum() / len(SC)
P4raw = W7.Correct.sum() / len(W7)
T4s = W7.Since_prev_s.mean()
E4 = pd.concat([E4, pd.DataFrame([dict(
    Session="POOLED", N_targets=N4, Word_selections=len(W7), Scorable=len(SC),
    Conditional_P=round(P4, 4), Mean_s_per_selection=round(T4s, 3),
    Selections_per_min=round(60 / T4s, 2),
    Bits_per_selection=round(bits(N4, P4), 3),
    ITR_conditional=round(itr(N4, P4, T4s), 2),
    ITR_as_logged=round(itr(N4, P4raw, T4s), 2),
    ITR_ceiling_at_P1=round(log2(N4) * 60 / T4s, 2))])], ignore_index=True)

# ── Counterfactuals: which lever buys the most bits ────────────────────
scen = [
    ("Exp 3 as measured", N3, P3, T3),
    ("Exp 3, dwell halved to 0.5 s (search unchanged)", N3, P3, T3 - 0.5),
    ("Exp 3, dwell removed entirely (search only)", N3, P3, T3 - DWELL3),
    ("Exp 3, accuracy raised to 0.90", N3, 0.90, T3),
    ("Exp 3, accuracy raised to 1.00", N3, 1.00, T3),
    ("Exp 3, accuracy 0.90 AND dwell halved", N3, 0.90, T3 - 0.5),
    ("Exp 3, best single run (100 %, 2.93 s)", N3, 1.00, 2.933),
    ("Exp 3 on a 4x8 keyboard (N=32) at measured P, T", 32, P3, T3),
    ("Exp 4 as measured (conditional)", N4, P4, T4s),
    ("Exp 4, accuracy raised to 0.90", N4, 0.90, T4s),
]

# Alternative Exp 3 scorings — see the "Exp 3 scoring caveat" row on the Method
# sheet. Reported alongside, never instead of, the paper's scoring.
_sub = T5.Outcome_class == "spatial substitution"
_keep = T5[~_sub]
_asC = T5.Correct.copy(); _asC[_sub] = 1
scen += [
    ("Exp 3, alt-sentence picks dropped as unscorable (Exp 4 treatment)",
     N3, _keep.Correct.mean(), _keep.Since_prev_s.mean()),
    ("Exp 3, alt-sentence picks scored as intended selections",
     N3, _asC.mean(), T3),
]
CF = pd.DataFrame([dict(Scenario=s, N_targets=n, Accuracy_P=round(p, 4),
                        Mean_s_per_selection=round(t, 3),
                        Bits_per_selection=round(bits(n, p), 3),
                        ITR_bits_per_min=round(itr(n, p, t), 2))
                   for s, n, p, t in scen])
CF["Delta_vs_measured_pct"] = (
    (CF.ITR_bits_per_min / CF.ITR_bits_per_min.iloc[0] - 1) * 100).round(1)

METHOD = pd.DataFrame([
    ("Metric", "Wolpaw information transfer rate (ITR), bits per minute"),
    ("Bits per selection B",
     "log2(N) + P*log2(P) + (1-P)*log2((1-P)/(N-1))"),
    ("ITR", "B * (60 / T)"),
    ("N (Exp 3)", f"{N3} — every tile of the 4x3 pictogram grid is dwell-selectable"),
    ("N (Exp 4)", f"{N4} — 7 predicted words + the two fixed controls on the 3x3 grid"),
    ("P", "selection accuracy; Exp 4 uses the CONDITIONAL accuracy only"),
    ("T", "mean seconds between consecutive selections (T9b), i.e. dwell + saccade + search"),
    ("Assumption 1", "targets equiprobable — violated in Exp 4, where the predictor ranks "
                     "candidates, so the true per-selection entropy is below log2(9) and "
                     "this ITR overstates the choice difficulty"),
    ("Assumption 2", "errors spread uniformly over the N-1 non-targets — violated: a gaze "
                     "substitution lands on a spatial neighbour, not on a uniformly random "
                     "tile, which concentrates the confusion matrix and makes this Wolpaw "
                     "figure a conservative LOWER bound on the true mutual information"),
    ("Exp 3 scoring caveat",
     "the 5 selections classed as spatial substitutions landed 42.0 pt from the WRONG tile's "
     "centre, against 47.5 pt for correct picks on the right one (0.48 vs 0.59 half-tiles; a "
     "boundary slip would sit near 1.0), and the two runs concerned composed 'I want to sleep "
     "more' and 'I want to sleep more please' — grammatical alternatives. That is the "
     "Experiment 4 protocol failure (5.6) appearing in Experiment 3. T14 carries the ITR "
     "under both re-scorings; the headline stays at the paper's scoring"),
    ("Assumption 3", "selections independent — violated in Exp 4, where one divergence makes "
                     "every later selection in the trial unscorable"),
    ("Sample size", f"Exp 3 n={len(T5)} selections over 5 runs; Exp 4 n={len(W7)} word "
                    f"selections, {len(SC)} scorable — too few for a confusion-matrix "
                    "(Nykopp) ITR, which is why Wolpaw is used"),
    ("Why not WPM alone", "WPM ignores N and does not charge for errors; ITR puts the 12-tile "
                          "and 9-cell tasks on one axis and prices an error at the bits it "
                          "destroys"),
    ("Source", "exp3_exp4_results.xlsx sheets T4, T5, T6, T7, T9b"),
], columns=["Item", "Detail"])

with pd.ExcelWriter(XL, engine="openpyxl") as xw:
    METHOD.to_excel(xw, sheet_name="Method", index=False)
    E3.to_excel(xw, sheet_name="T12 ITR Exp3", index=False)
    E4.to_excel(xw, sheet_name="T13 ITR Exp4", index=False)
    CF.to_excel(xw, sheet_name="T14 ITR counterfactuals", index=False)

wb = load_workbook(XL)
HDR_FILL = PatternFill("solid", fgColor="12161B")
HDR_FONT = Font(color="FFFFFF", bold=True, size=10, name="Calibri")
BAND = PatternFill("solid", fgColor="F2F5F8")
thin = Side(style="thin", color="D6DBE0")
for ws in wb.worksheets:
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    ws.row_dimensions[1].height = 30
    for c in ws[1]:
        c.fill = HDR_FILL
        c.font = HDR_FONT
        c.alignment = Alignment(horizontal="center", vertical="center",
                                wrap_text=True)
    for col in ws.columns:
        L = max((len(str(c.value)) for c in col if c.value is not None), default=8)
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(max(L + 3, 11), 46)
    for i, row in enumerate(ws.iter_rows(min_row=2)):
        for c in row:
            c.border = Border(bottom=thin)
            if isinstance(c.value, float):
                c.number_format = "0.00"
            if i % 2:
                c.fill = BAND
wb.save(XL)

# ── Figure 9 ──────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12.6, 4.2))
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.0, 1.15], wspace=0.30)

# (a) bits per selection vs accuracy — why an error is expensive
ax = fig.add_subplot(gs[0, 0]); tidy(ax)
pp = np.linspace(0.001, 1.0, 500)
for n, col, lab in ((N3, S1, f"Exp 3  N={N3}"), (N4, S2, f"Exp 4  N={N4}")):
    ax.plot(pp * 100, [bits(n, p) for p in pp], color=col, lw=2, label=lab)
    ax.axhline(log2(n), color=col, lw=0.8, ls=":", alpha=0.6)
ax.scatter([P3 * 100], [bits(N3, P3)], s=54, color=S1, zorder=5,
           edgecolor="white", lw=1.2)
ax.annotate(f"Exp 3 pooled\n{P3*100:.0f} % -> {bits(N3, P3):.2f} bits",
            (P3 * 100, bits(N3, P3)), xytext=(-6, 26), textcoords="offset points",
            fontsize=8, color=INK2, ha="right",
            arrowprops=dict(arrowstyle="-", color=INK3, lw=0.7))
ax.scatter([P4 * 100], [bits(N4, P4)], s=54, color=S2, zorder=5,
           edgecolor="white", lw=1.2)
ax.annotate(f"Exp 4 conditional\n{P4*100:.0f} % -> {bits(N4, P4):.2f} bits",
            (P4 * 100, bits(N4, P4)), xytext=(10, 22), textcoords="offset points",
            fontsize=8, color=INK2,
            arrowprops=dict(arrowstyle="-", color=INK3, lw=0.7))
ax.set_xlabel("selection accuracy (%)"); ax.set_ylabel("bits per selection")
ax.set_title("(a) An error is priced in bits")
ax.set_xlim(0, 100); ax.set_ylim(0, 4.0); ax.legend(loc="upper left")

# (b) ITR per run, both experiments
ax = fig.add_subplot(gs[0, 1]); tidy(ax)
runs = [f"R{i}" for i in range(1, 6)]
x = np.arange(5); w = 0.38
b3 = E3.ITR_bits_per_min[:5].values
b4 = E4.ITR_conditional[:5].values
ax.bar(x - w / 2, b3, w, color=S1, label="Exp 3 (N=12)")
ax.bar(x + w / 2, b4, w, color=S2, label="Exp 4 conditional (N=9)")
for xi, v in zip(x - w / 2, b3):
    ax.text(xi, v + 1.2, f"{v:.0f}", ha="center", fontsize=7.6, color=INK2)
for xi, v in zip(x + w / 2, b4):
    ax.text(xi, v + 1.2, f"{v:.0f}", ha="center", fontsize=7.6, color=INK2)
ax.axhline(E3_pool["ITR_bits_per_min"], color=S1, ls="--", lw=1,
           label=f"Exp 3 pooled {E3_pool['ITR_bits_per_min']:.1f}")
ax.axhline(E3_pool["ITR_ceiling_at_P1"], color=INK3, ls=":", lw=1,
           label=f"error-free ceiling {E3_pool['ITR_ceiling_at_P1']:.1f}")
ax.set_xticks(x); ax.set_xticklabels(runs)
ax.set_ylabel("ITR (bits / min)"); ax.set_title("(b) ITR per session")
ax.set_ylim(0, max(b3.max(), b4.max()) * 1.42)
ax.legend(loc="upper left", ncol=1)

# (c) iso-ITR field: which lever actually buys bits
ax = fig.add_subplot(gs[0, 2])
Pg = np.linspace(0.15, 1.0, 220)
Tg = np.linspace(1.2, 6.0, 220)
PP, TT = np.meshgrid(Pg, Tg)
Z = np.vectorize(lambda p, t: itr(N3, p, t))(PP, TT)
cs = ax.contourf(PP * 100, TT, Z, levels=12, cmap="Blues", alpha=0.9)
ax.contour(PP * 100, TT, Z, levels=[15, 30, 45, 60, 90, 120],
           colors="white", linewidths=0.7)
ax.scatter([P3 * 100], [T3], s=70, color=S2, zorder=6, edgecolor="white", lw=1.4)
ax.annotate(f"measured\n{itr(N3, P3, T3):.1f} bits/min", (P3 * 100, T3),
            xytext=(-16, -36), textcoords="offset points", fontsize=8,
            color=INK, ha="right", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.0))
ax.annotate("", xy=(P3 * 100, T3 - 0.5), xytext=(P3 * 100, T3),
            arrowprops=dict(arrowstyle="->", color=S3, lw=1.8))
ax.text(P3 * 100 + 1.5, T3 - 0.55, "halve the dwell\n+15 %", fontsize=7.6,
        color=S3, va="center")
ax.annotate("", xy=(100, T3), xytext=(P3 * 100, T3),
            arrowprops=dict(arrowstyle="->", color="#f5c542", lw=1.8))
ax.text(88, T3 + 0.42, "fix the accuracy\n+88 %", fontsize=7.6,
        color="#8a6a00", ha="center")
ax.set_xlabel("selection accuracy (%)")
ax.set_ylabel("mean seconds per selection")
ax.set_title("(c) Accuracy is the bigger lever")
ax.invert_yaxis()
fig.colorbar(cs, ax=ax, pad=0.02, label="ITR (bits / min)")

cap(fig, "Figure 9 — Information transfer rate. (a) Wolpaw bits per selection against "
         f"accuracy; at 75 % on a 12-tile grid a selection carries {bits(N3, P3):.2f} of the "
         f"{log2(N3):.2f} bits the grid could hold. (b) Per-session ITR; Exp 4's conditional "
         "figure is the only reportable one and is not comparable to Exp 3 because its cued "
         "protocol did not hold (§5.6). (c) Iso-ITR field for N=12: from the measured "
         "operating point, halving the 1.0 s dwell buys ~15 %, while eliminating the 25 % "
         "error rate buys ~88 % — the tracker's accuracy, not its dwell threshold, is what "
         "bounds the bit rate.")
fig.savefig(FIG)
plt.close(fig)

print(f"wrote {XL}")
print(f"wrote {FIG}")
print(f"\nExp 3 pooled : P={P3:.3f}  T={T3:.2f}s  "
      f"B={bits(N3, P3):.2f} bits/sel  ITR={itr(N3, P3, T3):.1f} bits/min  "
      f"(ceiling {log2(N3) * 60 / T3:.1f})")
print(f"Exp 4 pooled : P={P4:.3f} (conditional)  T={T4s:.2f}s  "
      f"B={bits(N4, P4):.2f} bits/sel  ITR={itr(N4, P4, T4s):.1f} bits/min  "
      f"(ceiling {log2(N4) * 60 / T4s:.1f})")
