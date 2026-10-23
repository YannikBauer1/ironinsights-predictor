"""Re-check the two stage-2 choices on the tuning window only (2022 and 2023, each fitted on earlier years):
one shared stage 2 vs separate Open-age and Masters layers, and the LightGBM blend weight."""
import pickle
import warnings

import numpy as np

from bbpredict.blend import Blend, SplitBlend
from bbpredict.evaluate import Prediction, fmt, summarize

warnings.filterwarnings("ignore")
with open("data/walk.pkl", "rb") as f:
    preds, snap, names = pickle.load(f)


def rolling(make, years=(2022, 2023)):
    out = []
    for year in years:
        lo, hi = np.datetime64(f"{year}-01-01"), np.datetime64(f"{year + 1}-01-01")
        model = make().fit([(snap[p.contest.division_id], p.contest) for p in preds if p.contest.date < lo])
        out += [Prediction(p.contest, model.utility(snap[p.contest.division_id], p.contest))
                for p in preds if lo <= p.contest.date < hi]
    return out


print("shared w=0.5        ", fmt(summarize(rolling(lambda: Blend(names)))))
for w in (0.3, 0.5, 0.7):
    print(f"split  w={w}        ", fmt(summarize(rolling(lambda: SplitBlend(names, w_lgbm=w)))))
