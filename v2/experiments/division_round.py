"""Division round: should divisions be handled differently? Rolling stage-2 folds, each year fitted on the years before.

Variants (the final config, champion bonus included, is "base"):
  cons_open, cons_both   athlete consistency inputs in the Open-age layer / in both layers:
                         cons_known (3+ earlier placed results in the division), cons_sd (spread of the latest
                         6 placing percentiles), cons_low (worst of the latest 5); leak-free, from earlier dates only
  groups2, groups2_mix   one Open-age layer per division group (men / women), alone or averaged 50/50 with the
                         shared layer; groups3 / groups3_mix: muscle / shape / women's muscle (see GROUPS)
  bikini_t1 .. bikini_t5 base, with Bikini's Open-age win % sharpened or softened by a temperature fitted on the
                         previous k years of out-of-sample Bikini predictions (fewer while the replay is young;
                         the first 3 replay years get none)
usage: division_round.py [--walk walk.pkl] <out.pkl> <years> <variant> [<variant> ...]
       e.g. --walk data/walk_2012.pkl data/dr_tune_a.pkl 2015,2016,...,2023 base cons_open
Writes {variant: [Prediction]} for division_compare.py; reads data/walk.pkl unless --walk is given.
"""
import pickle
import sys
import warnings
from collections import defaultdict
from itertools import groupby

import numpy as np

import bbpredict.blend as blend
from bbpredict.blend import MASTERS_COLUMNS, OPEN_COLUMNS, SplitBlend
from bbpredict.data import cached_contests
from bbpredict.evaluate import Prediction, fit_scale, summarize
from bbpredict.features import HistoryEntry, placing_percentile
from bbpredict.final import CHAMPION_BONUS, make_blend

warnings.filterwarnings("ignore")
blend.LGBM_PARAMS["num_threads"] = 3          # several variants run side by side
walk = "data/walk.pkl"
if "--walk" in sys.argv:
    i = sys.argv.index("--walk")
    walk = sys.argv[i + 1]
    del sys.argv[i:i + 2]
out_path, years = sys.argv[1], tuple(int(y) for y in sys.argv[2].split(","))
variants = sys.argv[3:]
preds, snap, names = pickle.load(open(walk, "rb"))
CONS = ["cons_known", "cons_sd", "cons_low"]
names = names + CONS

# consistency inputs, from placed results on strictly earlier dates in the same pool
history = defaultdict(list)                   # (person, pool) -> [placing percentile]
cons = {}
for _, day in groupby(cached_contests(), key=lambda c: c.date):
    day = list(day)
    for c in day:
        inputs = np.zeros((len(c.persons), 3))
        for i, person in enumerate(c.persons):
            h = history[(person, c.pool)]
            if len(h) >= 3:
                inputs[i] = [1.0, float(np.std(h[-6:])), float(min(h[-5:]))]
        cons[c.division_id] = inputs
    for c in day:
        for person, place in zip(c.persons, c.places):
            if not np.isnan(place):
                history[(person, c.pool)].append(placing_percentile(
                    HistoryEntry(c.date, c.pool, c.comp, c.tier, c.is_open_age, place, c.n_placed)))
rows = [(np.hstack([snap[p.contest.division_id], cons[p.contest.division_id]]), p.contest) for p in preds]

# division -> group; divisions not listed fall into the default group
GROUPS = {
    "groups2": ({"mensbb": "men", "classic": "men", "202_212": "men", "mensphysique": "men", "wheelchair": "men"},
                "women"),
    "groups3": ({"mensbb": "muscle", "classic": "muscle", "202_212": "muscle", "wheelchair": "muscle",
                 "womensbb": "womens_muscle", "womensphysique": "womens_muscle"}, "shape"),
}


class GroupBlend:
    """make_blend, but Open-age classes of each division group get their own layer (optionally mixed with the shared one)."""

    def __init__(self, group_of, default, mix):
        self.shared, self.group_of, self.default, self.mix, self.layers = make_blend(names), group_of, default, mix, {}

    def fit(self, rows):
        self.shared.fit(rows)
        for g in set(self.group_of.values()) | {self.default}:
            layer = make_blend(names).open                               # the final Open-age layer, unfitted
            self.layers[g] = layer.fit([(x, c) for x, c in rows if c.is_open_age and self.group_of.get(c.cat, self.default) == g])
        return self

    def utility(self, x, c):
        if not c.is_open_age:
            return self.shared.utility(x, c)
        own = self.layers[self.group_of.get(c.cat, self.default)].utility(x, c)
        return 0.5 * own + 0.5 * self.shared.utility(x, c) if self.mix else own


def model_for(variant):
    if variant.startswith("cons"):
        masters = MASTERS_COLUMNS + CONS if variant == "cons_both" else MASTERS_COLUMNS
        return SplitBlend(names, columns=OPEN_COLUMNS + CONS, masters_columns=masters, champion_bonus=CHAMPION_BONUS)
    if variant.startswith("groups"):
        return GroupBlend(*GROUPS[variant.removesuffix("_mix")], variant.endswith("_mix"))
    return make_blend(names)


fold_cache = {}


def folds(variant, fold_years):
    out = []
    for year in fold_years:
        if (variant, year) not in fold_cache:
            lo, hi = np.datetime64(f"{year}-01-01"), np.datetime64(f"{year + 1}-01-01")
            model = model_for(variant).fit([(x, c) for x, c in rows if c.date < lo])
            fold_cache[(variant, year)] = [Prediction(c, model.utility(x, c)) for x, c in rows if lo <= c.date < hi]
        out += fold_cache[(variant, year)]
    return out


def is_bikini(c):
    return c.cat == "bikini" and c.is_open_age and c.enh == "Open"


def bikini_tempered(base_all, k):
    """Each year's Bikini win % rescaled by a temperature fitted on the k years before (out-of-sample folds)."""
    bikini = [p for p in base_all if is_bikini(p.contest)]
    temperature = {}
    for year in years:
        past = [p for p in bikini if year - k <= p.contest.year < year]
        temperature[year] = fit_scale(past, grid=np.exp(np.linspace(-0.7, 0.7, 29))) if past else 1.0
    return [Prediction(p.contest, p.theta * temperature[p.contest.year]) if is_bikini(p.contest) else p
            for p in base_all if p.contest.year in years]


def show(label, out):
    every = summarize(out)
    open_age = summarize([p for p in out if p.contest.is_open_age])
    masters = summarize([p for p in out if not p.contest.is_open_age])
    bikini = summarize([p for p in out if is_bikini(p.contest)])
    print(f"{label:12s} all ll={every['ll_win']:.4f} top1={every['top1']:.3f} rho={every['spearman']:.3f}"
          f" | open ll={open_age['ll_win']:.4f} top1={open_age['top1']:.3f} | masters ll={masters['ll_win']:.4f}"
          f" | bikini favourite said {bikini['p_fav']:.1%} won {bikini['fav_won']:.1%}", flush=True)


results = {}
for v in variants:
    if v.startswith("bikini_t"):
        k = int(v.removeprefix("bikini_t"))
        first = max(min(years) - k, min(p.contest.year for p in preds) + 3)   # a fold needs 3 years of training rows;
                                                                              # earlier years get no correction
        base_all = folds("base", tuple(range(first, max(years) + 1)))
        results[v] = bikini_tempered(base_all, k)
    else:
        results[v] = folds(v, years)
    show(v, results[v])
pickle.dump(results, open(out_path, "wb"))
