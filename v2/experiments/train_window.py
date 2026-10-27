"""How many past years should stage 2 learn from? Rolling folds, each year fitted on the years before,
with only the training window changed (backtest_final.rolling_predictions does the fitting).

Variants, for test year Y:
  from2006 .. from2018   every year from the start year up to Y (production today: from2018)
  last3 .. last8         only the K years before Y
usage: train_window.py <walk.pkl> <out.pkl> <years> <base variant> [<variant> ...]
       e.g. data/walk_2006.pkl data/tw_a.pkl 2015,2016,...,2026 last6 from2012 from2006
Writes {variant: [Prediction]}, the first variant under "base", for division_compare.py.
"""
import pickle
import sys
import warnings

import bbpredict.blend as blend
from backtest_final import rolling_predictions

warnings.filterwarnings("ignore")
blend.LGBM_PARAMS["num_threads"] = 3          # several runs side by side
walk, out_path, years = sys.argv[1], sys.argv[2], [int(y) for y in sys.argv[3].split(",")]
variants = sys.argv[4:]
preds, snap, names = pickle.load(open(walk, "rb"))


def first_year(variant, year):
    start = int(variant.removeprefix("from")) if variant.startswith("from") else year - int(variant.removeprefix("last"))
    if start >= year:
        sys.exit(f"{variant} has no years to learn from before {year}")
    return start


results = {}
for v in variants:
    out = []
    for year in years:
        start = first_year(v, year)
        out += rolling_predictions([p for p in preds if p.contest.year >= start], snap, names, years=(year,))
    results["base" if v == variants[0] else v] = out
    print(v, "done", flush=True)
pickle.dump(results, open(out_path, "wb"))
