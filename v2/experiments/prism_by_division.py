"""Our model on PRISM's published test, per division (its white paper, section 6), as JSON lines.

The rules are bbpredict.prism. Our predictions are leak-free walk-forward ones from 2024 to 18 Jul 2026
(PRISM's paper date); PRISM's window opens in 2023, but 2023 helped choose our settings, so we start in 2024.
usage: prism_by_division.py [predictions.pkl]     (default data/final_test_preds.pkl)
"""
import json
import pickle
import sys

from bbpredict.data import cached_contests
from bbpredict.prism import DIVISIONS, prism_scores

# PRISM's white paper table, section 6
PAPER_COLUMNS = ("contests", "pairs", "pairs_correct", "favourite_wins", "top3_overlap", "spearman", "kendall")
PAPER = {
    "mensbb": (65, 1039, .79, .56, .62, .63, .55), "classic": (34, 253, .75, .71, .62, .50, .44),
    "mensphysique": (51, 370, .77, .46, .65, .39, .32), "bikini": (45, 503, .78, .55, .62, .62, .51),
    "fitness": (16, 166, .80, .58, .74, .80, .69), "wellness": (53, 279, .70, .29, .59, .54, .44),
    "all": (264, 2610, .77, .51, .63, .57, .48),
}
preds = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else "data/final_test_preds.pkl", "rb"))
ours = prism_scores({p.contest.division_id: p.theta for p in preds}, cached_contests(), "2024-01-01", "2026-07-19")
for division in DIVISIONS + ("all",):
    row = dict(division=division, **ours[division])
    row.update({f"prism_{column}": value for column, value in zip(PAPER_COLUMNS, PAPER[division])})
    print(json.dumps(row))
