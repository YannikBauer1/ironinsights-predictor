"""Rolling stage-2 predictions with the round-2 split columns (Open: unchanged, masters: + new groups).
usage: round2_folds.py <walk.pkl> <out.pkl> <years comma-separated> [baseline]
'baseline' uses the old columns in both layers (the before model)."""
import pickle
import sys
import warnings

import numpy as np

from bbpredict.blend import MASTERS_COLUMNS, SplitBlend
from bbpredict.evaluate import Prediction

warnings.filterwarnings("ignore")

walk, out_path, years = sys.argv[1], sys.argv[2], [int(y) for y in sys.argv[3].split(",")]
baseline = len(sys.argv) > 4 and sys.argv[4] == "baseline"
preds, snap, names = pickle.load(open(walk, "rb"))
out = []
for year in years:
    lo, hi = np.datetime64(f"{year}-01-01"), np.datetime64(f"{year + 1}-01-01")
    model = SplitBlend(names) if baseline else SplitBlend(names, masters_columns=MASTERS_COLUMNS)
    model.fit([(snap[p.contest.division_id], p.contest) for p in preds if p.contest.date < lo])
    out += [Prediction(p.contest, model.utility(snap[p.contest.division_id], p.contest)) for p in preds if lo <= p.contest.date < hi]
    print(year, "done", flush=True)
pickle.dump(out, open(out_path, "wb"))
