"""Can the model itself keep the reigning Olympia champion first? Open-age layer, rolling folds 2021-2023.

Two leak-free inputs are added to the walk snapshot, computed from earlier contests only, with the same
definition as features.champ_unbeaten (named x_ here so they never clash with a walk that has it):
  x_champ_unbeaten  won the division's latest Olympia (until the next one, at most CHAMPION_DAYS) and nobody
                    placed above them in an Open-age class of the division since
  x_champ_beaten    won it, but was beaten since
Variants: the final columns (base), plus the two inputs learned (learned), and either of those with a fixed
bonus of k on the unbeaten champion's utility (base+k, learned+k), or on every reigning champion,
beaten or not (base+kall).
Besides log-loss and favourite wins, it reports how often an unbeaten champion was the favourite at the
Olympia (the power ranking is a hypothetical Olympia) and their expected vs actual wins at every show.
usage: champion_weight.py <walk.pkl> [--years Y,Y,..] <variant> [<variant> ...]     variants: base, learned, base+1.0, learned+0.5, ...
"""
import pickle
import sys
import warnings

import numpy as np

from bbpredict.blend import OPEN_COLUMNS, Blend
from bbpredict.data import cached_contests
from bbpredict.evaluate import Prediction, softmax, summarize
from bbpredict.features import CHAMPION_DAYS

warnings.filterwarnings("ignore")
YEARS = (2021, 2022, 2023)
if "--years" in sys.argv:                      # e.g. --years 2024,2025,2026 for the confirmation
    i = sys.argv.index("--years")
    YEARS = tuple(int(y) for y in sys.argv[i + 1].split(","))
    del sys.argv[i:i + 2]
EXTRA = ["x_champ_unbeaten", "x_champ_beaten"]
walk = sys.argv[1]
preds, snap, names = pickle.load(open(walk, "rb"))
names = names + EXTRA

cs = [c for c in cached_contests() if c.enh == "Open" and c.is_open_age]
olys = {}
for c in cs:
    if c.tier == "O" and c.winner is not None:
        olys.setdefault(c.cat, []).append(c)


def champion_status(c):
    """(champion id, beaten since?) for the latest Olympia of c's division, if it is under CHAMPION_DAYS old."""
    past = [o for o in olys.get(c.cat, []) if o.date < c.date]
    if not past:
        return None, False
    o = max(past, key=lambda o: o.date)
    if (c.date - o.date).astype(int) > CHAMPION_DAYS:
        return None, False
    champ = o.persons[o.winner]
    for s in cs:
        if s.cat == c.cat and o.date < s.date < c.date and champ in s.persons:
            mine = s.places[s.persons.index(champ)]
            if not np.isnan(mine) and np.nanmin(s.places) < mine:
                return champ, True
    return champ, False


X = {}
status = {}
for p in preds:
    c = p.contest
    if not c.is_open_age or c.year < 2012:
        continue
    champ, beaten = champion_status(c) if c.enh == "Open" else (None, False)
    extra = np.zeros((len(c.persons), 2))
    if champ in c.persons:
        extra[c.persons.index(champ), 1 if beaten else 0] = 1.0
        status[c.division_id] = (c.persons.index(champ), beaten)
    X[c.division_id] = np.hstack([snap[c.division_id], extra])
rows = [p for p in preds if p.contest.division_id in X]


def run(variant):
    kind, _, k = variant.partition("+")
    every = k.endswith("all")
    k = float(k.removesuffix("all") or 0)
    cols = OPEN_COLUMNS + (EXTRA if kind == "learned" else [])
    out = []
    for year in YEARS:
        lo, hi = np.datetime64(f"{year}-01-01"), np.datetime64(f"{year + 1}-01-01")
        model = Blend(names, columns=cols).fit([(X[p.contest.division_id], p.contest) for p in rows if p.contest.date < lo])
        for p in rows:
            c = p.contest
            if lo <= c.date < hi:
                champ = X[c.division_id][:, -2] + (X[c.division_id][:, -1] if every else 0)
                u = model.utility(X[c.division_id], c) + k * champ
                out.append(Prediction(c, u))
    s = summarize(out)
    oly_first = oly_n = 0
    e_win, o_win = [0.0, 0.0], [0.0, 0.0]              # [unbeaten, beaten] champions
    for p in out:
        st = status.get(p.contest.division_id)
        if st is None or p.contest.winner is None:
            continue
        i, beaten = st
        prob = softmax(p.theta)
        e_win[beaten] += prob[i]
        o_win[beaten] += float(p.contest.winner == i)
        if p.contest.tier == "O" and not beaten:
            oly_n += 1
            oly_first += int(np.argmax(prob) == i)
    print(f"{variant:14s} ll={s['ll_win']:.4f} top1={s['top1']:.3f} rho={s['spearman']:.3f} | unbeaten champion favourite at the"
          f" Olympia {oly_first}/{oly_n} | expected / actual wins: unbeaten champion {e_win[0]:.1f} / {o_win[0]:.0f},"
          f" beaten champion {e_win[1]:.1f} / {o_win[1]:.0f}", flush=True)


print(walk)
for v in sys.argv[2:]:
    run(v)
