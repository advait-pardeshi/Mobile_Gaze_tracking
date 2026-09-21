"""ITR for the fifteen Clear-tile Experiment 3 sessions.

Same Wolpaw formulation as make_itr.py, on the sessions in synthetic_exp3/:

    B   = log2(N) + P*log2(P) + (1-P)*log2((1-P)/(N-1))   bits / selection
    ITR = B * (60 / T)                                    bits / minute

N = 12 tiles (eleven words + Clear), P = selection accuracy, T = mean seconds
per selection. Two accuracies are reported per session: as-logged (every pick
that did not advance the sentence counts as an error, Clear included) and
sentence-level (whether the target sentence came out right).

Writes exp3_synthetic_itr.csv, prints the table.
"""
import csv, glob, os
from math import log2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "synthetic_exp3")
N = 12
CLEAR_CELL = 9


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


def bits(n, P):
    if P >= 1: return log2(n)
    if P <= 1.0 / n: return 0.0
    return log2(n) + P * log2(P) + (1 - P) * log2((1 - P) / (n - 1))


def itr(n, P, T):
    return bits(n, P) * 60.0 / T


rows, all_t, all_ok, clears = [], [], 0, 0
for i, s in enumerate(sorted(glob.glob(os.path.join(ROOT, "session_*"))), 1):
    sel, geom, ov = sect(os.path.join(s, "exp3/run1_communication/trials.csv"))
    ov = ov[0]
    t = [float(r["since_prev_s"]) for r in sel]
    ok = sum(r["correct"] == "1" for r in sel)
    ncl = sum(int(r["cell_idx"]) == CLEAR_CELL for r in sel)
    all_t += t; all_ok += ok; clears += ncl
    P, T = ok / len(sel), float(np.mean(t))
    rows.append(dict(Session=f"S{i}", Selections=len(sel), Correct=ok, Clears=ncl,
                     Completed=int(ov["completed"]), Accuracy_P=round(P, 4),
                     Mean_s_per_selection=round(T, 3), Selections_per_min=round(60 / T, 2),
                     Bits_per_selection=round(bits(N, P), 3),
                     ITR_bits_per_min=round(itr(N, P, T), 2),
                     Words_per_min=float(ov["words_per_min"]),
                     ITR_ceiling_at_P1=round(log2(N) * 60 / T, 2)))

P, T = all_ok / len(all_t), float(np.mean(all_t))
pool = dict(Session="POOLED", Selections=len(all_t), Correct=all_ok, Clears=clears,
            Completed=sum(r["Completed"] for r in rows), Accuracy_P=round(P, 4),
            Mean_s_per_selection=round(T, 3), Selections_per_min=round(60 / T, 2),
            Bits_per_selection=round(bits(N, P), 3),
            ITR_bits_per_min=round(itr(N, P, T), 2),
            Words_per_min=round(float(np.mean([r["Words_per_min"] for r in rows])), 2),
            ITR_ceiling_at_P1=round(log2(N) * 60 / T, 2))
rows.append(pool)

out = os.path.join(HERE, "exp3_synthetic_itr.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

hdr = f"{'sess':<7}{'sel':>5}{'ok':>4}{'clr':>5}{'done':>6}{'P':>7}{'T s':>7}{'bits/sel':>10}{'ITR':>8}{'ceil':>8}{'wpm':>7}"
print(hdr); print("-" * len(hdr))
for r in rows:
    print(f"{r['Session']:<7}{r['Selections']:5d}{r['Correct']:4d}{r['Clears']:5d}"
          f"{r['Completed']:6d}{r['Accuracy_P']:7.3f}{r['Mean_s_per_selection']:7.2f}"
          f"{r['Bits_per_selection']:10.2f}{r['ITR_bits_per_min']:8.1f}"
          f"{r['ITR_ceiling_at_P1']:8.1f}{r['Words_per_min']:7.1f}")
sent = sum(r["Completed"] for r in rows[:-1]) / (len(rows) - 1)
print(f"\nN = {N} tiles, log2(N) = {log2(N):.2f} bits.  Sentence-level completion "
      f"{sent*100:.0f} % ({sum(r['Completed'] for r in rows[:-1])}/{len(rows)-1} sessions).")
print(f"Pooled: P = {P*100:.1f} %, T = {T:.2f} s/selection, "
      f"B = {bits(N,P):.2f} bits/selection, ITR = {itr(N,P,T):.1f} bits/min "
      f"(error-free ceiling {log2(N)*60/T:.1f}).")
print("wrote", out)
