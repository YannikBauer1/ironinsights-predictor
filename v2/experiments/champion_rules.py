"""Should the reigning Olympia champion be put first? Three rules scored on the next Olympia.

model:      the model's own favourite (leak-free walk-forward predictions)
unbeaten:   the champion first if nobody beat them since the last Olympia, else the model's favourite
always:     the champion first whenever they compete
Only Olympias where the reigning champion is on stage. Needs data/champion_seasons.pkl
(champion_between_olympias.py) and prediction files from round2_folds.py / backtest_final.py.
usage: champion_rules.py <predictions.pkl> [<predictions.pkl> ...]
"""
import pickle
import sys

import numpy as np
import pandas as pd

from bbpredict.evaluate import softmax

seasons = pd.read_pickle("data/champion_seasons.pkl")
season_by_next = {(r.cat, str(r.next_date)[:10]): r for r in seasons.itertuples()}
preds = {}
for path in sys.argv[1:]:
    for p in pickle.load(open(path, "rb")):
        c = p.contest
        if c.tier == "O" and c.is_open_age and c.winner is not None:
            preds[(c.cat, str(c.date)[:10])] = p          # later files win: give the cleanest last

rows = []
for key, p in sorted(preds.items(), key=lambda kv: kv[0][1]):
    s = season_by_next.get(key)
    if s is None or not s.at_next:
        continue
    c = p.contest
    prob = softmax(p.theta)
    fav = int(np.argmax(prob))
    champ = c.persons.index(s.champ_id)
    unbeaten = s.losses == 0
    rows.append(dict(cat=c.cat, year=c.year, champ=s.champ, unbeaten=unbeaten, losses=s.losses,
                     model_fav=c.names[fav], model_fav_p=prob[fav], champ_p=prob[champ], champ_rank=int((prob > prob[champ]).sum()) + 1,
                     winner=c.names[c.winner], model=fav == c.winner, rule_unbeaten=(champ if unbeaten else fav) == c.winner,
                     rule_always=champ == c.winner))
df = pd.DataFrame(rows)
print(f"{len(df)} Olympias with the reigning champion on stage, {df.year.min()} to {df.year.max()}\n")
for label, s in (("all", df), ("champion unbeaten since", df[df.unbeaten]), ("champion beaten since", df[~df.unbeaten])):
    print(f"  {label:26s} n={len(s):3d}  first place won the Olympia:  model {int(s.model.sum()):3d}"
          f"  unbeaten rule {int(s.rule_unbeaten.sum()):3d}  always champion {int(s.rule_always.sum()):3d}"
          f"  | model had the champion first {int((s.champ_rank == 1).sum()):3d}")
print("\nWhere the model did NOT put the reigning champion first:")
for r in df[df.champ_rank > 1].itertuples():
    print(f"  {r.cat:14s} {r.year}  champion {r.champ:22s} ({'unbeaten' if r.unbeaten else f'beaten {r.losses}x'}) rank {r.champ_rank} {r.champ_p:.0%}"
          f" | model first {r.model_fav} {r.model_fav_p:.0%} | won: {r.winner}")
