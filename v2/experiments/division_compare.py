"""Compare division-round variants with base, year by year (division_round.py outputs).

For each variant: overall and Open-age log-loss against base, and in how many years it beat base; for
Bikini also the favourite's claimed vs actual win rate per year. A variant is worth keeping only if it
wins on the total and in most years.
usage: division_compare.py <out.pkl> [<out.pkl> ...]
"""
import pickle
import sys
from collections import defaultdict

from bbpredict.evaluate import summarize

results = {}
for path in sys.argv[1:]:
    results.update(pickle.load(open(path, "rb")))
base = results.pop("base")
years = sorted({p.contest.year for p in base})


def is_bikini(c):
    return c.cat == "bikini" and c.is_open_age and c.enh == "Open"


def by_year(preds, keep=lambda c: True):
    out = defaultdict(list)
    for p in preds:
        if keep(p.contest):
            out[p.contest.year].append(p)
    return out


def table(label, keep):
    b = by_year(base, keep)
    print(f"\n{label}: log-loss change against base (negative = better)")
    print(f"  {'variant':12s} {'total':>8s}  years better  " + " ".join(f"{y:>6d}" for y in years))
    tb = summarize([p for p in base if keep(p.contest)])["ll_win"]
    print(f"  {'base':12s} {tb:8.4f}" + " " * 16 + " ".join(f"{summarize(b[y])['ll_win']:6.3f}" for y in years))
    for v, preds in results.items():
        r = by_year(preds, keep)
        d = [summarize(r[y])["ll_win"] - summarize(b[y])["ll_win"] for y in years]
        total = summarize([p for p in preds if keep(p.contest)])["ll_win"] - tb
        print(f"  {v:12s} {total:+8.4f}  {sum(x < 0 for x in d):3d} of {len(years):<5d} " + " ".join(f"{x:+6.3f}" for x in d))


table("All divisions", lambda c: True)
table("Open age", lambda c: c.is_open_age)
table("Bikini, Open age", is_bikini)

print("\nBikini favourite, said / won, by year (base)")
for y, ps in sorted(by_year(base, is_bikini).items()):
    s = summarize(ps)
    print(f"  {y} n={s['divisions']:3d}  said {s['p_fav']:.1%}  won {s['fav_won']:.1%}  gap {s['fav_won'] - s['p_fav']:+.1%}")
