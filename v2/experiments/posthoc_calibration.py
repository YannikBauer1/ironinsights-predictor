"""Round 3: can a correction applied after the model fix the shape of its probabilities?
Each target year is corrected with parameters fitted on the earlier years' out-of-sample predictions.
  rank:     u + offset[rank bucket]  (buckets 1, 2, 3, 4-5, 6-10, 11+; rank 1 fixed at 0)
  rank+T:   a * u + offset[rank bucket]
  division: u * T[division, Open/masters], T shrunk towards 1 (100 divisions of evidence = halfway)
usage: posthoc_calibration.py <predictions.pkl covering 2020-2023>"""
import pickle
import sys

import numpy as np
from scipy.optimize import minimize

from bbpredict.evaluate import Prediction, calibration_table, fit_scale, lineup_rank, summarize

preds = pickle.load(open(sys.argv[1], "rb"))
preds = [p for p in preds if p.contest.winner is not None]
EDGES = (1, 2, 3, 5, 10)            # rank <=1, <=2, <=3, <=5, <=10, >10


def buckets(p):
    return np.searchsorted(EDGES, lineup_rank(p.theta, p.contest.persons), side="left")


def fit_offsets(train, with_scale):
    data = [(p.theta, buckets(p), p.contest.winner) for p in train]

    def nll(x):
        a = x[0] if with_scale else 1.0
        g = np.r_[0.0, x[1:] if with_scale else x]
        tot = 0.0
        for u, b, w in data:
            z = a * u + g[b]
            m = z.max()
            tot += m + np.log(np.exp(z - m).sum()) - z[w]
        return tot / len(data)

    x0 = np.r_[1.0, np.zeros(5)] if with_scale else np.zeros(5)
    x = minimize(nll, x0, method="L-BFGS-B").x
    return (x[0], np.r_[0.0, x[1:]]) if with_scale else (1.0, np.r_[0.0, x])


def seg(c):
    return (c.cat, c.is_open_age)


results = {"raw": [], "rank": [], "rank+T": [], "division": []}
for year in (2021, 2022, 2023):
    train = [p for p in preds if p.contest.year < year]
    target = [p for p in preds if p.contest.year == year]
    results["raw"] += target
    for name, with_scale in (("rank", False), ("rank+T", True)):
        a, g = fit_offsets(train, with_scale)
        results[name] += [Prediction(p.contest, a * p.theta + g[buckets(p)]) for p in target]
        print(f"  {year} {name}: scale {a:.3f}, offsets by rank bucket {np.round(g, 3).tolist()}", flush=True)
    temps = {}
    for key in {seg(p.contest) for p in target}:
        sub = [p for p in train if seg(p.contest) == key]
        t = fit_scale(sub, grid=np.exp(np.linspace(-1, 1, 41))) if len(sub) >= 20 else 1.0
        temps[key] = 1 + (t - 1) * len(sub) / (len(sub) + 100)
    results["division"] += [Prediction(p.contest, p.theta * temps[seg(p.contest)]) for p in target]
for name, out in results.items():
    s = summarize(out)
    tab, ece = calibration_table(out, 1.0)
    print(f"{name:9s} ll={s['ll_win']:.4f} top1={s['top1']:.3f} top3={s['top3']:.3f} rho={s['spearman']:.3f} ECE={ece:.4f}")
