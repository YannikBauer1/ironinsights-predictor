"""Round 3: where does the model go wrong, and is there a pattern?

For every entrant of every analysed division: the predicted win probability (over the whole lineup)
and whether they won. For a group of entrants, expected wins = sum of their win probabilities;
observed / expected far from 1 in a large group is a systematic miss, not an outlier.
usage: error_analysis.py <predictions.pkl> [walk.pkl]   (predictions from round2_folds.py)
Also writes the entrant table to data/r3_entrants.pkl.
"""
import math
import pickle
import re
import sys
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd

from bbpredict.data import age_number, cached_contests
from bbpredict.evaluate import lineup_rank, softmax

preds = pickle.load(open(sys.argv[1], "rb"))
_, snap, names = pickle.load(open(sys.argv[2] if len(sys.argv) > 2 else "data/walk.pkl", "rb"))
COLS = ["base", "elo", "base_known", "rookie_pro", "switcher", "log_shows_pool", "years_pro", "olympia_last_top3",
        "olympia_last_top6", "olympia_last_in", "olympia_champ", "defending_show", "prev_place_show_inv", "log_pro_wins",
        "won_last", "last_pct", "trend", "gap_b2b", "gap_short", "gap_year", "gap_long", "shows_365", "home",
        "masters_only", "h2h_net", "h2h_meet", "qualified", "agegrp_shows", "agegrp_new", "age_moved_up"]
idx = {c: names.index(c) for c in COLS}

rows = []
for p in preds:
    c = p.contest
    if c.winner is None:
        continue
    prob = softmax(p.theta)
    rank = lineup_rank(p.theta, c.persons)
    X = snap[c.division_id]
    for i, person in enumerate(c.persons):
        r = dict(div=c.division_id, date=c.date, year=c.year, month=int(str(c.date)[5:7]), cat=c.cat, age=c.age,
                 enh=c.enh, tier=c.tier, comp=c.comp, n=len(c.persons), person=person, name=c.names[i],
                 country=c.countries[i], event_country=c.event_country, place=c.places[i], won=float(i == c.winner),
                 p=float(prob[i]), rank=int(rank[i]), fav_p=float(prob.max()))
        r.update({k: X[i, j] for k, j in idx.items()})
        rows.append(r)
df = pd.DataFrame(rows)
df["open_age"] = df.age.eq("Open")
df["shows"] = np.expm1(df.log_shows_pool).round()
df["top3"] = (df.place <= 3).astype(float)
print(f"{df['div'].nunique()} divisions, {len(df)} entrants, {int(df.won.sum())} winners\n")


def oe(sub, label, min_e=5.0):
    e, o = sub.p.sum(), sub.won.sum()
    if e < min_e:
        return
    z = (o - e) / math.sqrt(max(e * (1 - e / max(len(sub), 1)), 1e-9))
    flag = "  <<<" if abs(z) >= 3 else ("  <" if abs(z) >= 2 else "")
    print(f"  {label:44s} n={len(sub):6d}  expected wins {e:7.1f}  observed {o:5.0f}  O/E {o / e:5.2f}  z={z:+5.1f}{flag}")


def section(title):
    print(f"\n=== {title}")


section("A. Entrant groups: is a kind of athlete systematically over- or underrated?")
groups = {
    "rookie (no pro result anywhere)": df.rookie_pro == 1,
    "division switcher": df.switcher == 1,
    "1-2 earlier shows in division": df.shows.between(1, 2),
    "3-9 earlier shows": df.shows.between(3, 9),
    "10+ earlier shows": df.shows >= 10,
    "years pro < 1": (df.years_pro < 1) & (df.rookie_pro == 0),
    "years pro 1-3": df.years_pro.between(1, 3, inclusive="left"),
    "years pro 3-8": df.years_pro.between(3, 8, inclusive="left"),
    "years pro 8+": df.years_pro >= 8,
    "layoff: back to back (<=21 d)": df.gap_b2b == 1,
    "layoff: 1-2 years": df.gap_year == 1,
    "layoff: 2+ years": df.gap_long == 1,
    "shows last 365 d: 4+": df.shows_365 >= 4,
    "home country": df.home == 1,
    "athlete from USA": df.country.eq("US"),
    "athlete not from USA": ~df.country.eq("US") & df.country.ne(""),
    "country unknown (filled in later for notable athletes)": df.country.eq(""),
    "won their last show": df.won_last == 1,
    "Olympia top 3 last year": df.olympia_last_top3 == 1,
    "Olympia top 6 last year": df.olympia_last_top6 == 1,
    "defending this show": df.defending_show == 1,
    "already qualified (Open Olympia)": df.qualified == 1,
    "masters-only athlete": df.masters_only == 1,
    "trend up (> +0.2)": df.trend > 0.2,
    "trend down (< -0.2)": df.trend < -0.2,
    "beat this lineup before (h2h > 0.2)": df.h2h_net > 0.2,
    "lost to this lineup before (h2h < -0.2)": df.h2h_net < -0.2,
}
for scope, mask in (("all", df.n > 0), ("Open age", df.open_age), ("masters", ~df.open_age)):
    print(f" [{scope}]")
    for label, g in groups.items():
        oe(df[mask & g], label)

section("B. By predicted rank in the lineup")
for lo, hi in ((1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 99)):
    oe(df[df["rank"].between(lo, hi)], f"model rank {lo}" + (f"-{hi}" if hi != lo else ""))

section("C. Favourites by strength: do strong favourites win as often as claimed?")
fav = df[df["rank"] == 1]
for lo, hi in ((0, .2), (.2, .35), (.35, .5), (.5, .65), (.65, .8), (.8, .9), (.9, 1.01)):
    s = fav[(fav.p >= lo) & (fav.p < hi)]
    if len(s) >= 20:
        print(f"  favourite at {lo:.2f}-{min(hi, 1):.2f}: n={len(s):4d} predicted {s.p.mean():.3f} observed {s.won.mean():.3f}")
print(" [favourites above 60%: which kind of favourite loses?]")
strong = fav[fav.p >= 0.6]
for label, g in groups.items():
    oe(strong[g.loc[strong.index]], label, min_e=3)

section("D. Show-level segments")
seg = {
    "field <=4": df.n <= 4, "field 5-8": df.n.between(5, 8), "field 9-15": df.n.between(9, 15), "field 16+": df.n >= 16,
    "Olympia": df.tier == "O", "tier 2": df.tier == "2", "tier 3": df.tier == "3", "Masters Olympia": df.tier == "MO",
    "natural": df.tier == "N", "event in USA": df.event_country.eq("US"), "event outside USA": ~df.event_country.isin(["US", ""]),
    "Jan-Apr": df.month <= 4, "May-Aug": df.month.between(5, 8), "Sep-Dec": df.month >= 9,
}
for label, g in seg.items():
    s = df[g]
    favs = s[s["rank"] == 1]
    print(f"  {label:20s} divisions={s['div'].nunique():4d}  favourite claimed {favs.p.mean():.3f} won {favs.won.mean():.3f}"
          f"  | rank 2-3 claimed {s[s['rank'].between(2, 3)].p.sum() / max(s['div'].nunique(), 1):.3f}"
          f" won {s[s['rank'].between(2, 3)].won.sum() / max(s['div'].nunique(), 1):.3f}")
print(" [by division, Open age]")
for cat, s in df[df.open_age & df.enh.eq("Open")].groupby("cat"):
    favs = s[s["rank"] == 1]
    if len(favs) >= 25:
        print(f"  {cat:16s} divisions={len(favs):4d} favourite claimed {favs.p.mean():.3f} won {favs.won.mean():.3f}")
print(" [by masters bracket]")
for ag, s in df[~df.open_age].groupby(df.age.map(age_number)):
    favs = s[s["rank"] == 1]
    if len(favs) >= 25:
        print(f"  {ag:3d}+ divisions={len(favs):4d} favourite claimed {favs.p.mean():.3f} won {favs.won.mean():.3f}")

section("E. Upsets: winners the model gave under 5%")
low = df[df.p < 0.05]
ups = low[low.won == 1]
print(f"  {len(ups)} upset winners out of {int(df.won.sum())} ({len(ups) / df.won.sum():.1%}); "
      f"low-probability entrants expected {low.p.sum():.0f} wins, got {len(ups)}")
for label, g in groups.items():
    oe(low[g.loc[low.index]], label, min_e=3)

section("F. Possible data problems: 'rookie' winners whose name already exists under another person")


def norm(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", "", s).split()


seen = defaultdict(set)          # (first, last) -> person ids, from all contests before the analysis start
first_date = df.date.min()
for c in cached_contests():
    if c.date >= first_date:
        break
    for p, nm in zip(c.persons, c.names):
        t = norm(nm)
        if len(t) >= 2:
            seen[(t[0], t[-1])].add(p)
rookie_winners = df[(df.won == 1) & (df.rookie_pro == 1)]
hits = []
for _, r in rookie_winners.iterrows():
    t = norm(r["name"])
    if len(t) >= 2 and seen.get((t[0], t[-1]), set()) - {r.person}:
        hits.append(f"{r['name']} ({r.date}, {r.comp}, {r.cat})")
print(f"  {len(rookie_winners)} rookie winners; {len(hits)} share first+last name with an earlier, different person")
for h in hits[:15]:
    print("   ", h)
pd.to_pickle(df, "data/r3_entrants.pkl")
