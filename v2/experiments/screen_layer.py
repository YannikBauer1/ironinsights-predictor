"""One stage-2 layer at a time: which round-2 groups help it on the tuning years (2022, 2023)?
usage: screen_layer.py open|masters   (reads data/walk.pkl from walk.py)"""
import pickle
import sys
import warnings

import numpy as np

from bbpredict.blend import AGE_GROUP as AGE, AGE_GROUP_VIEWS as VIEWS, COLUMNS, HEAD_TO_HEAD as H2H, QUALIFICATION as QUAL, Blend
from bbpredict.evaluate import Prediction, summarize

warnings.filterwarnings("ignore")
CANDIDATES = {
    "open": [("baseline", COLUMNS), ("+ qualification", COLUMNS + QUAL), ("+ head to head", COLUMNS + H2H),
             ("+ both", COLUMNS + QUAL + H2H), ("+ both + age views", COLUMNS + QUAL + H2H + VIEWS)],
    "masters": [("baseline", COLUMNS), ("+ age features", COLUMNS + AGE), ("+ age features + views", COLUMNS + AGE + VIEWS),
                ("+ age + views + h2h", COLUMNS + AGE + VIEWS + H2H), ("+ all four groups", COLUMNS + AGE + VIEWS + H2H + QUAL)],
}

layer = sys.argv[1]
open_age = layer == "open"
preds, snap, names = pickle.load(open("data/walk.pkl", "rb"))
for label, cols in CANDIDATES[layer]:
    out = []
    for year in (2022, 2023):
        lo, hi = np.datetime64(f"{year}-01-01"), np.datetime64(f"{year + 1}-01-01")
        m = Blend(names, columns=cols).fit([(snap[p.contest.division_id], p.contest) for p in preds
                                            if p.contest.date < lo and p.contest.is_open_age == open_age])
        out += [Prediction(p.contest, m.utility(snap[p.contest.division_id], p.contest)) for p in preds
                if lo <= p.contest.date < hi and p.contest.is_open_age == open_age]
    s = summarize(out)
    print(f"{label:24s} {layer} ll={s['ll_win']:.4f} top1={s['top1']:.3f} top3={s['top3']:.3f} rho={s['spearman']:.3f}", flush=True)
