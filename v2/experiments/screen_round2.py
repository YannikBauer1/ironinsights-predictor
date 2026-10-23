"""Round 2 screening on the tuning years only: does each new idea help predictions of 2022 and 2023
(each year predicted by a stage 2 fitted on the years before it)?
Fast pass with the logit layer (split Open-age / masters), then the full blend for the finalists."""
import pickle
import sys
import warnings

import numpy as np

from bbpredict.blend import AGE_GROUP, AGE_GROUP_VIEWS, COLUMNS, HEAD_TO_HEAD, QUALIFICATION, SplitBlend
from bbpredict.data import DIVISIONS
from bbpredict.evaluate import Prediction, summarize
from bbpredict.stage2 import Stage2

warnings.filterwarnings("ignore")
WALK = sys.argv[1] if len(sys.argv) > 1 else "data/walk.pkl"
preds, snap, names = pickle.load(open(WALK, "rb"))
YEARS = (2022, 2023)

GROUPS = {
    "per-division scale": [f"base_x_{c}" for c in DIVISIONS],
    "qualification": QUALIFICATION,
    "head to head": HEAD_TO_HEAD,
    "age-group features": AGE_GROUP,
    "age-group views": AGE_GROUP_VIEWS,
}


def logit_rolling(cols):
    idx = [names.index(c) for c in cols]
    out = []
    for year in YEARS:
        lo, hi = np.datetime64(f"{year}-01-01"), np.datetime64(f"{year + 1}-01-01")
        for open_age in (True, False):
            train = [(snap[p.contest.division_id][:, idx], p.contest.places) for p in preds
                     if p.contest.date < lo and p.contest.is_open_age == open_age]
            m = Stage2(k_terms=1, l2=1.0).fit(train)
            out += [Prediction(p.contest, m.utility(snap[p.contest.division_id][:, idx])) for p in preds
                    if lo <= p.contest.date < hi and p.contest.is_open_age == open_age]
    return out


def blend_rolling(cols):
    out = []
    for year in YEARS:
        lo, hi = np.datetime64(f"{year}-01-01"), np.datetime64(f"{year + 1}-01-01")
        m = SplitBlend(names, columns=cols, masters_columns=cols).fit([(snap[p.contest.division_id], p.contest) for p in preds if p.contest.date < lo])
        out += [Prediction(p.contest, m.utility(snap[p.contest.division_id], p.contest)) for p in preds if lo <= p.contest.date < hi]
    return out


def show(label, out):
    s_all = summarize(out)
    s_m = summarize([p for p in out if not p.contest.is_open_age])
    s_o = summarize([p for p in out if p.contest.is_open_age])
    print(f"{label:34s} all ll={s_all['ll_win']:.4f} top1={s_all['top1']:.3f} rho={s_all['spearman']:.3f} | "
          f"open ll={s_o['ll_win']:.4f} | masters ll={s_m['ll_win']:.4f}", flush=True)
    return s_all["ll_win"]


mode = sys.argv[2] if len(sys.argv) > 2 else "logit"
if mode == "logit":
    base = show("baseline columns", logit_rolling(COLUMNS))
    kept = []
    for name, cols in GROUPS.items():
        ll = show(f"+ {name}", logit_rolling(COLUMNS + cols))
        if ll < base - 0.002:
            kept.append(name)
    print("groups that help alone:", kept)
    show("+ all helpful groups", logit_rolling(COLUMNS + sum((GROUPS[k] for k in kept), [])))
    show("+ every group", logit_rolling(COLUMNS + sum(GROUPS.values(), [])))
else:
    picks = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else []
    show("blend: baseline columns", blend_rolling(COLUMNS))
    show("blend: + " + "+".join(picks), blend_rolling(COLUMNS + sum((GROUPS[k] for k in picks), [])))
