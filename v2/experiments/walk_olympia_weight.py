"""Walk 2018 to today with a heavier Olympia in every Plackett-Luce view (final.TIER_W["O"]).

usage: walk_olympia_weight.py <weight> <out.pkl>    e.g. 3 data/walk_oly3.pkl  (final config: 1.5)
"""
import pickle
import sys
import time

import bbpredict.final as final
from bbpredict.data import cached_contests
from bbpredict.evaluate import walk_forward

final.TIER_W["O"] = float(sys.argv[1])          # PL holds this same dict, so every view sees it
t = time.time()
rec = final.recorder()
preds = walk_forward(rec, cached_contests(), "2018-01-01", "2026-12-31")
pickle.dump((preds, rec.snap, rec.names), open(sys.argv[2], "wb"))
print(f"Olympia weight {sys.argv[1]}: walked {len(preds)} divisions in {time.time() - t:.0f}s -> {sys.argv[2]}")
