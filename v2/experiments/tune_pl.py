import json
from bbpredict.data import cached_contests
from bbpredict.plfit import TimeDecayPL
from bbpredict.tune import coordinate_search, around, plus
cs = cached_contests()
def mk(t_o=1.0, t_2=1.0, t_mo=1.0, t_n=1.0, **kw):
    return TimeDecayPL(tier_w={"O": t_o, "2": t_2, "MO": t_mo, "N": t_n}, **kw)
start = dict(tau=730.0, lam=0.3, rookie=-0.5, reverse=True, masters_w=1.0, window_years=6.0, t_o=1.0, t_2=1.0, t_mo=1.0, t_n=1.0, max_terms=None)
steps = dict(tau=around(1.4), lam=around(1.7), rookie=plus(0.3), masters_w=around(1.5),
             t_o=around(1.5), t_2=around(1.5), t_mo=around(1.5), window_years=lambda v: [4.0, 8.0, 10.0], max_terms=lambda v: [None, 5, 10])
best = coordinate_search(mk, start, steps, cs, rounds=4, log=lambda s: print(s, flush=True))
print(json.dumps(dict(params=best[0], ll=best[1], scale=best[2]), default=str))
