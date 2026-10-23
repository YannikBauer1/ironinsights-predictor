"""Second tuning round for the main Plackett-Luce rating, with whole-lineup scoring, on 2018-2023 only.
Starts from the configuration in bbpredict/final.py."""
import json

from bbpredict.data import cached_contests
from bbpredict.plfit import TimeDecayPL
from bbpredict.tune import around, coordinate_search, plus


def make(t_o=1.5, t_2=1.0, t_mo=2 / 3, **kw):
    return TimeDecayPL(reverse=True, tier_w={"O": t_o, "2": t_2, "MO": t_mo}, **kw)


start = dict(tau=730.0, lam=0.3, rookie=-0.2, masters_w=2 / 3, window_years=6.0, t_o=1.5, t_2=1.0, t_mo=2 / 3)
steps = dict(tau=around(1.3), lam=around(1.5), rookie=plus(0.2), masters_w=around(1.3), t_o=around(1.3),
             t_2=around(1.3), t_mo=around(1.3), window_years=lambda v: [4.0, 8.0, 10.0])
best, ll, scale = coordinate_search(make, start, steps, cached_contests(), rounds=3, log=lambda s: print(s, flush=True))
print(json.dumps(dict(params=best, ll=ll, scale=scale)))
