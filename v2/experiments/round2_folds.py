"""Rolling stage-2 predictions with the current final stage 2 (final.make_blend).
usage: round2_folds.py <walk.pkl> <out.pkl> <years comma-separated> [baseline]
'baseline' uses the round-1 columns in both layers (the before model of round 2)."""
import pickle
import sys
import warnings

import numpy as np

from bbpredict.blend import SplitBlend
from bbpredict.evaluate import Prediction
from bbpredict.final import make_blend

warnings.filterwarnings("ignore")

walk, out_path, years = sys.argv[1], sys.argv[2], [int(y) for y in sys.argv[3].split(",")]
baseline = len(sys.argv) > 4 and sys.argv[4] == "baseline"
preds, snap, names = pickle.load(open(walk, "rb"))
out = []
for year in years:
    lo, hi = np.datetime64(f"{year}-01-01"), np.datetime64(f"{year + 1}-01-01")
    model = SplitBlend(names) if baseline else make_blend(names)
    model.fit([(snap[p.contest.division_id], p.contest) for p in preds if p.contest.date < lo])
    out += [Prediction(p.contest, model.utility(snap[p.contest.division_id], p.contest)) for p in preds if lo <= p.contest.date < hi]
    print(year, "done", flush=True)
pickle.dump(out, open(out_path, "wb"))
