"""Final backtest: stage 2 refitted every January on everything before it (as production would),
scored on every division of 2024, 2025 and 2026. Prints overall, per year, per slice and per division
metrics plus win and top-3 calibration tables. Reads data/walk.pkl (run walk.py first) and writes
data/final_test_preds.pkl for case_studies.py.
"""
import pickle

import numpy as np

from bbpredict.blend import SplitBlend
from bbpredict.data import DIVISIONS
from bbpredict.evaluate import Prediction, calibration_table, fmt, summarize
from bbpredict.montecarlo import place_probabilities

SLICES = {
    "Open age (open circuit)": lambda c: c.is_open_age and c.enh == "Open",
    "Masters": lambda c: not c.is_open_age,
    "Natural": lambda c: c.enh == "Natural",
    "Olympia": lambda c: c.tier == "O",
    "Tier 2 (Arnold, NY, Pitt, Romania)": lambda c: c.tier == "2",
    "Masters Olympia": lambda c: c.tier == "MO",
}


def rolling_predictions(preds, snap, names, years=(2024, 2025, 2026)):
    out = []
    for year in years:
        lo, hi = np.datetime64(f"{year}-01-01"), np.datetime64(f"{year + 1}-01-01")
        train = [(snap[p.contest.division_id], p.contest) for p in preds if p.contest.date < lo]
        model = SplitBlend(names).fit(train)
        out += [Prediction(p.contest, model.utility(snap[p.contest.division_id], p.contest))
                for p in preds if lo <= p.contest.date < hi]
        print(year, "trained on", len(train), "divisions", flush=True)
    return out


def top3_calibration(out):
    ps, ys = [], []
    for p in out:
        c = p.contest
        if c.n_placed < 4:
            continue
        P = place_probabilities(p.theta, sims=4000)
        ps.extend(P[:, :3].sum(1))
        ys.extend((c.places <= 3).astype(float))     # nan places compare False
    return np.array(ps), np.array(ys)


def main():
    with open("data/walk.pkl", "rb") as f:
        preds, snap, names = pickle.load(f)
    out = rolling_predictions(preds, snap, names)
    with open("data/final_test_preds.pkl", "wb") as f:
        pickle.dump(out, f)

    print("\nALL 2024-2026          ", fmt(summarize(out)))
    for y in (2024, 2025, 2026):
        print(f"  {y}                 ", fmt(summarize([p for p in out if p.contest.year == y])))
    for name, keep in SLICES.items():
        print(f"{name:34s}", fmt(summarize([p for p in out if keep(p.contest)])))
    print("\nby division, Open age, open circuit")
    for cat in DIVISIONS:
        s = [p for p in out if p.contest.cat == cat and p.contest.is_open_age and p.contest.enh == "Open"]
        if s:
            print(f"  {cat:16s}", fmt(summarize(s)))

    tab, ece = calibration_table(out, 1.0)
    print(f"\nwin-probability calibration (ECE={ece:.4f})")
    for lo, hi, n, a, b in tab:
        print(f"  {lo:.2f}-{hi:.2f}  n={n:5d}  predicted={a:.3f}  observed={b:.3f}")
    ps, ys = top3_calibration(out)
    print("\ntop-3 probability calibration")
    for lo, hi in [(0, .1), (.1, .2), (.2, .4), (.4, .6), (.6, .8), (.8, .9), (.9, 1.01)]:
        k = (ps >= lo) & (ps < hi)
        print(f"  {lo:.1f}-{hi:.1f} n={k.sum():5d} predicted={ps[k].mean():.3f} observed={ys[k].mean():.3f}")


if __name__ == "__main__":
    main()
