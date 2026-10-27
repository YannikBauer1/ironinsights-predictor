"""The v2 predictor end to end: replay history, predict lineups and rank athletes.

Read-only by design: it consumes the results snapshot and returns predictions; nothing is written back.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .blend import SplitBlend
from .data import Contest
from .evaluate import softmax, walk_forward
from .features import FEATURES, FeatureRecorder
from .montecarlo import summary

RANKING_FIELD = 30   # a hypothetical Olympia: the top 30 by strength share the win probability


def replay(rec: FeatureRecorder, contests: list[Contest], start: str, end: str):
    """Feed every finished contest before `end` to the recorder; record features from `start` on."""
    return walk_forward(rec, [c for c in contests if c.n_placed >= 1 and c.date < np.datetime64(end)], start, end)


def predict_contest(rec: FeatureRecorder, blend: SplitBlend, c: Contest, sims: int = 20000) -> list[dict]:
    u = blend.utility(rec.features(c), c)
    s = summary(u, sims)
    rows = [dict(person_id=p, name=nm, country=ctry, strength=float(u[i]), win=float(s["win"][i]),
                 p_second=float(s["P"][i, 1]) if len(u) > 1 else 0.0,
                 p_third=float(s["P"][i, 2]) if len(u) > 2 else 0.0,
                 top3=float(s["top3"][i]), top5=float(s["top5"][i]), expected_place=float(s["expected_place"][i]))
            for i, (p, nm, ctry) in enumerate(zip(c.persons, c.names, c.countries))]
    rows.sort(key=lambda r: -r["strength"])
    for k, r in enumerate(rows, 1):
        r["predicted_place"] = k
    return rows


@dataclass
class RankingField:
    """A hypothetical Olympia: the synthetic contest, its feature matrix and each candidate's latest result."""
    contest: Contest
    X: np.ndarray
    last: dict


ACTIVE_DAYS = 913   # 30 months: an athlete who skips one whole Olympia season stays in the list


def ranking_field(rec: FeatureRecorder, contests: list[Contest], cat: str, date: str,
                  active_days: int = ACTIVE_DAYS, inactive: set | None = None) -> RankingField | None:
    """Every open-circuit pro who placed in an Open class of `cat` in the last `active_days` and is not
    marked inactive (retired), entered into an Olympia on `date`. Features come from the recorder's state
    at that moment; how long ago they last competed is in each entry's `months_away`."""
    d = np.datetime64(date)
    lo = d - np.timedelta64(active_days, "D")
    inactive = inactive or set()
    last: dict = {}
    for c in contests:
        if c.cat != cat or c.enh != "Open" or not c.is_open_age or not (lo <= c.date < d):
            continue
        for p, nm, pl, ctry in zip(c.persons, c.names, c.places, c.countries):
            if not np.isnan(pl) and p not in inactive:
                last[p] = dict(name=nm, country=ctry, last_date=str(c.date), last_show=c.comp, last_place=int(pl),
                               months_away=round((d - c.date).astype(int) / 30.44))
    if not last:
        return None
    persons = list(last)
    synth = Contest(
        division_id=f"ranking|{cat}|{d}", event_id="ranking", date=d, year=int(str(d)[:4]),
        cat=cat, age="Open", enh="Open", comp="olympia", tier="O", weight_class="",
        persons=persons, places=np.full(len(persons), np.nan), prejudge=np.full(len(persons), np.nan),
        total=np.full(len(persons), np.nan), names=[last[p]["name"] for p in persons],
        countries=[last[p]["country"] for p in persons], event_country="US")   # the Olympia is held in the USA
    return RankingField(synth, rec.features(synth).copy(), last)


def rank(blend: SplitBlend, f: RankingField, field: int = RANKING_FIELD) -> list[dict]:
    """Order the candidates by strength; win share = softmax over the top `field`."""
    u = blend.utility(f.X, f.contest, field_size=field)
    order = np.argsort(-u)
    top = order[:field]
    share = dict(zip(top.tolist(), softmax(u[top]).tolist()))
    persons, champ = f.contest.persons, f.X[:, FEATURES.index("reigning_champ")]
    return [dict(rank=k, person_id=persons[i], strength=float(u[i]), olympia_win=share.get(int(i), 0.0),
                 reigning_champion=bool(champ[i]), **f.last[persons[i]]) for k, i in enumerate(order, 1)]
