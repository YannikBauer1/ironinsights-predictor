"""Reigning Olympia champions between two Olympias: how often were they beaten, and did it matter?

For every Olympia champion (open circuit, Open age) and the next Olympia of that division:
the shows in between where the champion competed, who placed above them, and what happened at the
next Olympia (competed? kept the title? did the athletes who beat them finish above them again?).
Writes data/champion_seasons.pkl for champion_rules.py.
usage: champion_between_olympias.py [first_year]
"""
import sys
from collections import Counter

import numpy as np
import pandas as pd

from bbpredict.data import cached_contests

FIRST = int(sys.argv[1]) if len(sys.argv) > 1 else 1965
cs = [c for c in cached_contests() if c.enh == "Open" and c.is_open_age]
olympias = {}
for c in cs:
    if c.tier == "O" and c.winner is not None:
        olympias.setdefault(c.cat, []).append(c)

rows = []
for cat, olys in olympias.items():
    olys.sort(key=lambda c: c.date)
    for this, nxt in zip(olys, olys[1:]):
        if this.year < FIRST:
            continue
        champ = this.persons[this.winner]
        between = [c for c in cs if c.cat == cat and this.date < c.date < nxt.date and champ in c.persons]
        shows, losses, beaten_by = 0, [], set()
        for c in between:
            i = c.persons.index(champ)
            if np.isnan(c.places[i]):
                continue
            shows += 1
            above = [p for p, pl in zip(c.persons, c.places) if not np.isnan(pl) and pl < c.places[i]]
            if above:
                losses.append(f"{c.comp} {c.date} ({int(c.places[i])}.)")
                beaten_by |= set(above)
        at_next = champ in nxt.persons and not np.isnan(nxt.places[nxt.persons.index(champ)])
        place_next = int(nxt.places[nxt.persons.index(champ)]) if at_next else None
        # did those who beat the champion in between also finish above the champion at the next Olympia?
        beaters_above = beaters_at = 0
        if at_next:
            for b in beaten_by:
                if b in nxt.persons and not np.isnan(nxt.places[nxt.persons.index(b)]):
                    beaters_at += 1
                    beaters_above += nxt.places[nxt.persons.index(b)] < place_next
        rows.append(dict(cat=cat, year=this.year, champ=this.names[this.winner], champ_id=champ, oly_date=this.date,
                         next_date=nxt.date, shows=shows, losses=len(losses), loss_list="; ".join(losses),
                         beaten_by=sorted(beaten_by), at_next=at_next, place_next=place_next,
                         kept=place_next == 1, beaters_at=beaters_at, beaters_above=beaters_above,
                         next_winner=nxt.names[nxt.winner]))

df = pd.DataFrame(rows)
df.to_pickle("data/champion_seasons.pkl")
print(f"{len(df)} champion seasons since {FIRST} ({df.cat.nunique()} divisions)\n")


def line(label, s):
    at = s[s.at_next]
    print(f"  {label:44s} seasons {len(s):4d}  defended {len(at):4d}  kept the title {int(at.kept.sum()):4d}"
          f"  ({at.kept.mean():.0%})" if len(at) else f"  {label:44s} seasons {len(s):4d}  defended 0")


print("A. What the champion did between the two Olympias")
line("no other show", df[df.shows == 0])
line("competed, never beaten", df[(df.shows > 0) & (df.losses == 0)])
line("beaten at least once", df[df.losses > 0])
line("  beaten once", df[df.losses == 1])
line("  beaten twice or more", df[df.losses >= 2])
line("all", df)

print("\nB. When the champion was beaten and both met again at the next Olympia")
b = df[(df.losses > 0) & df.at_next & (df.beaters_at > 0)]
print(f"  {len(b)} seasons; the athletes who had beaten the champion: {int(b.beaters_at.sum())} met again,"
      f" {int(b.beaters_above.sum())} finished above the champion again ({b.beaters_above.sum() / max(b.beaters_at.sum(), 1):.0%})")

print("\nC. By era")
for lo, hi in ((1965, 1999), (2000, 2014), (2015, 2026)):
    s = df[df.year.between(lo, hi)]
    print(f" [{lo}-{hi}]")
    line("never beaten (incl. no show)", s[s.losses == 0])
    line("beaten at least once", s[s.losses > 0])

print("\nD. By division, since 2010")
for cat, s in df[df.year >= 2010].groupby("cat"):
    beaten = s[s.losses > 0]
    print(f"  {cat:15s} seasons {len(s):3d}  beaten in between {len(beaten):3d}"
          f"  beaten & defended {int(beaten.at_next.sum()):3d} -> kept {int(beaten[beaten.at_next].kept.sum()):3d}"
          f"  | unbeaten & defended {int(s[(s.losses == 0) & s.at_next].shape[0]):3d} -> kept {int(s[(s.losses == 0) & s.at_next].kept.sum()):3d}")

print("\nE. Every season where the champion was beaten in between")
pd.set_option("display.width", 250)
for r in df[df.losses > 0].sort_values(["cat", "year"]).itertuples():
    nxt = "skipped next Olympia" if not r.at_next else f"next Olympia {r.place_next}." + (" (kept)" if r.kept else f", won by {r.next_winner}")
    print(f"  {r.cat:14s} {r.year} {r.champ:24s} shows {r.shows} lost {r.losses}: {r.loss_list[:90]:90s} | {nxt}")
print("\nShows per champion season:", Counter(df.shows.clip(upper=4)).most_common())
