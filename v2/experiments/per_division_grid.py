"""Per-division decay (tau) and prior strength (lam) for the main rating, on 2018-2023 only.
Divisions are rated independently, so one walk per setting scores every division at once.
Each division's log-loss uses its own best temperature (stage 2 sets the final scale anyway)."""
import itertools
import json
from collections import defaultdict

from bbpredict.data import DIVISIONS, cached_contests
from bbpredict.evaluate import fit_scale, walk_forward, winner_logloss
from bbpredict.plfit import TimeDecayPL
from bbpredict.tune import TUNE

cs = cached_contests()
rows = defaultdict(dict)
for tau, lam in itertools.product([365.0, 550.0, 730.0, 1000.0, 1400.0], [0.15, 0.3, 0.6]):
    m = TimeDecayPL(tau=tau, lam=lam, reverse=True, rookie=-0.2, masters_w=2 / 3, tier_w={"O": 1.5, "MO": 2 / 3})
    preds = walk_forward(m, cs, *TUNE)
    for cat in DIVISIONS:
        sub = [p for p in preds if p.contest.cat == cat and p.contest.winner is not None]
        if sub:
            rows[cat][f"{tau:.0f}|{lam}"] = (winner_logloss(sub, fit_scale(sub)), len(sub))
    print(tau, lam, flush=True)
json.dump(rows, open("logs/per_division_grid.json", "w"), indent=1)
for cat, res in rows.items():
    best = min(res, key=lambda k: res[k][0])
    print(f"{cat:16s} n={res[best][1]:4d} best={best:10s} ll={res[best][0]:.4f}  current(730|0.3)={res['730|0.3'][0]:.4f}")
