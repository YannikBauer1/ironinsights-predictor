"""Pick one probability shrink on the tuning-year predictions: a temperature k on the log-strengths,
or a uniform mix eps. usage: round2_shrink.py <tuning preds.pkl>"""
import pickle
import sys

import numpy as np

from bbpredict.evaluate import Prediction, calibration_table, softmax, summarize

preds = pickle.load(open(sys.argv[1], "rb"))
for k in (0.8, 0.85, 0.9, 0.95, 1.0, 1.05):
    s = summarize(preds, k)
    tab, ece = calibration_table(preds, k)
    hi = [(a, b, n) for lo, _, n, a, b in tab if lo >= 0.8]
    print(f"temperature k={k:.2f}: ll={s['ll_win']:.4f} brier={s['brier']:.4f} ECE={ece:.4f} "
          + " ".join(f"[{a:.3f}->{b:.3f} n={n}]" for a, b, n in hi))
for eps in (0.0, 0.01, 0.02, 0.03, 0.05):
    mixed = [Prediction(p.contest, np.log((1 - eps) * softmax(p.theta) + eps / len(p.theta))) for p in preds]
    s = summarize(mixed)
    tab, ece = calibration_table(mixed, 1.0)
    hi = [(a, b, n) for lo, _, n, a, b in tab if lo >= 0.8]
    print(f"uniform mix eps={eps:.2f}: ll={s['ll_win']:.4f} brier={s['brier']:.4f} ECE={ece:.4f} "
          + " ".join(f"[{a:.3f}->{b:.3f} n={n}]" for a, b, n in hi))
