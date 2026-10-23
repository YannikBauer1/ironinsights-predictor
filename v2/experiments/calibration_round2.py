"""Round 2: can a calibration step fix the over-confident favourites? Tuning years only.
For year Y in 2022 and 2023 the stage 2 is fitted on the years before Y; the calibration is fitted on
year Y-1 as predicted by a stage 2 fitted on the years before Y-1, so it never sees Y.
  none: as is
  size: one extra temperature per lineup-size bucket (<=4, 5-8, 9-15, 16+)
  mix:  a share eps of uniform probability blended in
Neither "size" nor "mix" helped; the final model has no extra calibration step.
usage: calibration_round2.py [walk.pkl]"""
import pickle
import sys
import warnings

import numpy as np

from bbpredict.evaluate import Prediction, calibration_table, fit_scale, softmax, summarize, winner_logloss
from bbpredict.final import make_blend

warnings.filterwarnings("ignore")
preds, snap, names = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else "data/walk.pkl", "rb"))
BUCKETS = (4, 8, 15)
EPS_GRID = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12)


def predict_year(year):
    lo, hi = np.datetime64(f"{year}-01-01"), np.datetime64(f"{year + 1}-01-01")
    m = make_blend(names).fit([(snap[p.contest.division_id], p.contest) for p in preds if p.contest.date < lo])
    return [Prediction(p.contest, m.utility(snap[p.contest.division_id], p.contest)) for p in preds if lo <= p.contest.date < hi]


def bucket(p):
    return int(np.searchsorted(BUCKETS, len(p.contest.persons)))


def mixed(p, eps):
    return Prediction(p.contest, np.log((1 - eps) * softmax(p.theta) + eps / len(p.theta)))


results = {"none": [], "size": [], "mix": []}
for year in (2022, 2023):
    calib, target = predict_year(year - 1), predict_year(year)
    results["none"] += target
    temps = {k: fit_scale([p for p in calib if bucket(p) == k], grid=np.exp(np.linspace(-1, 1, 41))) for k in range(4)}
    results["size"] += [Prediction(p.contest, p.theta * temps[bucket(p)]) for p in target]
    eps = min(EPS_GRID, key=lambda e: winner_logloss([mixed(p, e) for p in calib], 1.0))
    results["mix"] += [mixed(p, eps) for p in target]
    print(f"  {year}: size temperatures {[round(temps[k], 2) for k in range(4)]}, mix eps {eps}", flush=True)
for name, out in results.items():
    s = summarize(out)
    tab, ece = calibration_table(out, 1.0)
    top = "  ".join(f"{lo:.1f}+: {a:.3f}/{b:.3f} (n={n})" for lo, hi, n, a, b in tab if lo >= 0.5)
    print(f"{name:5s} ll={s['ll_win']:.4f} top1={s['top1']:.3f} ECE={ece:.4f} | predicted/observed {top}", flush=True)
