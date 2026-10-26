"""Before/after on the held-out years 2024-2026: our main test (whole lineup, every division) and
PRISM's published protocol.
usage: before_after.py [before.pkl] [after.pkl]
defaults: data/baseline_final_test_preds.pkl (round 1) and data/final_test_preds.pkl (current)."""
import pickle
import sys

import numpy as np

from bbpredict.data import DIVISIONS, cached_contests
from bbpredict.evaluate import calibration_table, summarize
from bbpredict.prism import prism_scores

before = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else "data/baseline_final_test_preds.pkl", "rb"))
after = pickle.load(open(sys.argv[2] if len(sys.argv) > 2 else "data/final_test_preds.pkl", "rb"))
assert [p.contest.division_id for p in before] == [p.contest.division_id for p in after]

SLICES = {
    "All 2024-2026": lambda c: True,
    "  2024": lambda c: c.year == 2024, "  2025": lambda c: c.year == 2025, "  2026": lambda c: c.year == 2026,
    "Open age (open circuit)": lambda c: c.is_open_age and c.enh == "Open",
    "Masters": lambda c: not c.is_open_age,
    "Natural": lambda c: c.enh == "Natural",
    "Olympia": lambda c: c.tier == "O",
    "Masters Olympia": lambda c: c.tier == "MO",
}
print(f"{'MAIN TEST':26s} {'div':>5s} | {'log-loss':>17s} | {'favourite wins':>15s} | {'top-3 overlap':>15s} | {'Spearman':>13s}")
for name, f in SLICES.items():
    b, a = summarize([p for p in before if f(p.contest)]), summarize([p for p in after if f(p.contest)])
    print(f"{name:26s} {a['divisions']:5d} | {b['ll_win']:.3f} -> {a['ll_win']:.3f} | {b['top1']:5.1%} -> {a['top1']:5.1%} | "
          f"{b['top3']:5.1%} -> {a['top3']:5.1%} | {b['spearman']:.3f} -> {a['spearman']:.3f}")
print("\nMasters by division (favourite wins, before -> after)")
for cat in DIVISIONS:
    b = [p for p in before if p.contest.cat == cat and not p.contest.is_open_age]
    a = [p for p in after if p.contest.cat == cat and not p.contest.is_open_age]
    if len(a) >= 20:
        sb, sa = summarize(b), summarize(a)
        print(f"  {cat:16s} n={sa['divisions']:4d}  {sb['top1']:5.1%} -> {sa['top1']:5.1%}   ll {sb['ll_win']:.3f} -> {sa['ll_win']:.3f}")
for label, preds in (("before", before), ("after", after)):
    tab, ece = calibration_table(preds, 1.0)
    print(f"\ncalibration {label}: ECE={ece:.4f}  " + "  ".join(f"{lo:.1f}+: {x:.3f}/{y:.3f}" for lo, _, n, x, y in tab if lo >= 0.5))

# PRISM protocol on the held-out years
pb = prism_scores({p.contest.division_id: p.theta for p in before}, cached_contests(), "2024-01-01", "2026-07-19")["all"]
pa = prism_scores({p.contest.division_id: p.theta for p in after}, cached_contests(), "2024-01-01", "2026-07-19")["all"]
print("\nPRISM protocol, 2024 to 18 Jul 2026 (PRISM paper: pairs 77%, winner 51%, Spearman 0.57, Kendall 0.48, log loss 0.47)")
for name, key, f in [("pairs ordered correctly", "pairs_correct", "{:.1%}"), ("favourite wins (show)", "favourite_wins", "{:.1%}"),
                     ("favourite wins (rated)", "favourite_wins_rated", "{:.1%}"), ("top-3 overlap", "top3_overlap", "{:.1%}"),
                     ("Spearman", "spearman", "{:.3f}"), ("Kendall", "kendall", "{:.3f}"), ("pairwise log loss", "pair_logloss", "{:.3f}")]:
    print(f"  {name:26s} {f.format(pb[key])} -> {f.format(pa[key])}")
