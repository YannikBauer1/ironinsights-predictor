"""Round-2 candidates B and C, walked like walk.py but with the main rating changed (2018-2023 choices):
  C: re-tuned global weights (Olympia 2.535, tier 2 1.3, Masters Olympia 0.394, prior 0.45) only
  B: C plus per-division decay and prior strength where the per-division grid gained >= 0.008
usage: walk_candidate_b.py b|c   -> data/walk_cand_<b|c>.pkl  (candidate A is walk.py's final config)"""
import pickle
import sys
import time

from bbpredict.data import cached_contests
from bbpredict.evaluate import walk_forward
from bbpredict.final import PL, recorder
from bbpredict.plfit import TimeDecayPL

RETUNED = dict(PL, lam=0.45, tier_w={"O": 2.535, "2": 1.3, "MO": 0.394})
TAU = {"classic": 1400.0, "womensbb": 1000.0, "womensphysique": 1000.0, "bikini": 365.0, "fitness": 365.0}
LAM = {"classic": 0.15, "womensbb": 0.6, "womensphysique": 0.15, "bikini": 0.15, "fitness": 0.15}


class PerDivisionPL(TimeDecayPL):
    """Main rating with its own decay and prior strength in the divisions listed above."""

    def fit(self, pool, date):
        cat = pool.split("|")[0]
        self.tau, self.lam = TAU.get(cat, RETUNED["tau"]), LAM.get(cat, RETUNED["lam"])
        return super().fit(pool, date)


which = sys.argv[1]
t = time.time()
rec = recorder()
rec.base = (PerDivisionPL if which == "b" else TimeDecayPL)(**RETUNED, masters_w=2 / 3)
for name, view in rec.extra.items():
    if isinstance(view, TimeDecayPL):          # same views, re-tuned weights and prior
        view.lam, view.tier_w = RETUNED["lam"], RETUNED["tier_w"]
preds = walk_forward(rec, cached_contests(), "2018-01-01", "2026-12-31")
pickle.dump((preds, rec.snap, rec.names), open(f"data/walk_cand_{which}.pkl", "wb"))
print(f"walked candidate {which.upper()}: {len(preds)} divisions in {time.time() - t:.0f}s", flush=True)
