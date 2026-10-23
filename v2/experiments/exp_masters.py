"""Does a separate stage 2 for masters classes beat one shared blend? Rolling yearly, 2024-2026."""
import pickle, warnings
import numpy as np
from bbpredict.blend import Blend
from bbpredict.evaluate import Prediction, summarize, fmt
warnings.filterwarnings("ignore")
preds, snap, names = pickle.load(open("data/walk.pkl", "rb"))
final = pickle.load(open("data/final_test_preds.pkl", "rb"))
is_m = lambda c: not c.is_open_age
sep = []
for year in [2024, 2025, 2026]:
    lo, hi = np.datetime64(f"{year}-01-01"), np.datetime64(f"{year + 1}-01-01")
    for grp in [True, False]:
        train = [(snap[p.contest.division_id], p.contest) for p in preds if p.contest.date < lo and is_m(p.contest) == grp]
        model = Blend(names).fit(train)
        sep += [Prediction(p.contest, model.utility(snap[p.contest.division_id], p.contest))
                for p in preds if lo <= p.contest.date < hi and is_m(p.contest) == grp]
for name, f in [("masters", is_m), ("open age", lambda c: not is_m(c)), ("all", lambda c: True)]:
    print(f"{name:9s} shared  ", fmt(summarize([p for p in final if f(p.contest)])))
    print(f"{name:9s} separate", fmt(summarize([p for p in sep if f(p.contest)])))
