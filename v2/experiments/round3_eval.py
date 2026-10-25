"""Round 3 evaluation harness: rolling stage-2 folds on 2021-2023 (each year fitted on the years before).

usage:
  round3_eval.py params  <config>            stage-2 parameter sweep, final.make_blend columns
  round3_eval.py open    <feature-group>     Open-age layer: the round-2 columns + one candidate group
  round3_eval.py masters <feature-group>     masters layer: the round-2 masters columns + one group
Each run mutates module settings (LGBM_PARAMS, HOLDOUT, Blend._label) for its own process only.
Prints log-loss, favourite wins and Spearman, overall and per layer. Reads data/walk.pkl.
"""
import pickle
import sys
import warnings

import numpy as np

import bbpredict.blend as blend
from bbpredict.blend import COLUMNS, MASTERS_COLUMNS, TIME_AWAY, TRAVEL, Blend
from bbpredict.final import make_blend
from bbpredict.evaluate import Prediction, summarize

warnings.filterwarnings("ignore")
YEARS = (2021, 2022, 2023)
blend.LGBM_PARAMS["num_threads"] = 2          # several configs run side by side

PARAMS = {
    "base": {},
    "iter300": dict(iterations=300), "iter1000": dict(iterations=1000),
    "leaves15": dict(lgbm=dict(num_leaves=15)), "leaves4": dict(lgbm=dict(num_leaves=4)),
    "minleaf40": dict(lgbm=dict(min_data_in_leaf=40)), "minleaf200": dict(lgbm=dict(min_data_in_leaf=200)),
    "lgbm_l2_20": dict(lgbm=dict(lambda_l2=20.0)),
    "logit_l2_0.3": dict(l2=0.3), "logit_l2_3": dict(l2=3.0), "logit_l2_10": dict(l2=10.0),
    "w_lgbm_0.4": dict(w_lgbm=0.4), "w_lgbm_0.6": dict(w_lgbm=0.6),
    "recent4y": dict(recent_years=4), "recent6y": dict(recent_years=6),
    "holdout730": dict(holdout=730),
    "graded10": dict(graded=10),
}
GROUPS = {
    "none": [],
    "time away": TIME_AWAY,
    "travel": TRAVEL,
    "head to head": ["h2h_net", "h2h_meet", "h2h_vs_top"],
    "h2h net only": ["h2h_net"],
    "travel + time away": TRAVEL + TIME_AWAY,
    "travel + years away": TRAVEL + ["years_away", "years_away_pool"],
}

mode, name = sys.argv[1], sys.argv[2]
if mode not in ("params", "open", "masters"):
    sys.exit(f"unknown mode {mode!r}")
preds, snap, names = pickle.load(open("data/walk.pkl", "rb"))


def show(label, out):
    s = summarize(out)
    o = summarize([p for p in out if p.contest.is_open_age])
    m = summarize([p for p in out if not p.contest.is_open_age])
    line = f"{label:34s} all ll={s['ll_win']:.4f} top1={s['top1']:.3f} rho={s['spearman']:.3f}"
    if o and m:
        line += f" | open ll={o['ll_win']:.4f} top1={o['top1']:.3f} | masters ll={m['ll_win']:.4f} top1={m['top1']:.3f}"
    print(line, flush=True)


out = []
for year in YEARS:
    lo, hi = np.datetime64(f"{year}-01-01"), np.datetime64(f"{year + 1}-01-01")
    if mode == "params":
        cfg = dict(PARAMS[name])
        blend.LGBM_PARAMS.update(cfg.pop("lgbm", {}))
        if "holdout" in cfg:
            blend.HOLDOUT = np.timedelta64(cfg.pop("holdout"), "D")
        if "graded" in cfg:          # LightGBM label: 1st = graded ... graded-th = 1 instead of 1st = 5
            g = cfg.pop("graded")
            blend.Blend._label = staticmethod(lambda pl, g=g: np.clip(g + 1 - pl, 0, g))
        start = lo - np.timedelta64(int(cfg.pop("recent_years", 99) * 365), "D")
        model = make_blend(names)
        for layer in (model.open, model.masters):
            for key, value in cfg.items():
                setattr(layer, {"w_lgbm": "w"}.get(key, key), value)
        model.fit([(snap[p.contest.division_id], p.contest) for p in preds if start <= p.contest.date < lo])
        out += [Prediction(p.contest, model.utility(snap[p.contest.division_id], p.contest)) for p in preds if lo <= p.contest.date < hi]
    else:
        open_age = mode == "open"
        cols = (COLUMNS if open_age else MASTERS_COLUMNS) + [c for c in GROUPS[name] if c not in (COLUMNS if open_age else MASTERS_COLUMNS)]
        model = Blend(names, columns=cols).fit([(snap[p.contest.division_id], p.contest) for p in preds
                                                if p.contest.date < lo and p.contest.is_open_age == open_age])
        out += [Prediction(p.contest, model.utility(snap[p.contest.division_id], p.contest)) for p in preds
                if lo <= p.contest.date < hi and p.contest.is_open_age == open_age]
show(f"{mode}:{name}", out)
