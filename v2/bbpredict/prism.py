"""PRISM's published test (its white paper of 18 July 2026, section 6 and appendix), applied to our predictions.

Its six divisions, Open age, open circuit. Only "rated" athletes count: 3+ placed shows in the division
since 2023, or a pro win in a men's division. Per contest: pairs of rated athletes ordered correctly, the
favourite winning the show, top-3 overlap, and Spearman and Kendall over contests with 3+ rated athletes.
Ties in predicted strength never count as right, and ties are broken by athlete id, never by list
position (stored lineups are in finishing order).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from scipy.stats import kendalltau, spearmanr

from .data import Contest
from .evaluate import lineup_rank

DIVISIONS = ("mensbb", "classic", "mensphysique", "bikini", "fitness", "wellness")
MEN = {"mensbb", "classic", "mensphysique"}
WINDOW_START = np.datetime64("2023-01-01")   # PRISM's measurement window opens here


def prism_scores(theta_by_division: dict, contests: list[Contest], start: str, end: str) -> dict[str, dict]:
    """Scores per division and for "all", over contests dated start <= date < end that have a prediction."""
    lo, hi = np.datetime64(start), np.datetime64(end)
    scope = [c for c in contests if c.cat in DIVISIONS and c.is_open_age and c.enh == "Open" and WINDOW_START <= c.date < hi]
    placings = defaultdict(list)
    for c in scope:
        for person, place in zip(c.persons, c.places):
            if not np.isnan(place):
                placings[(c.cat, person)].append((c.date, place))

    def rated(c, person):
        earlier = [place for date, place in placings[(c.cat, person)] if date < c.date]
        return len(earlier) >= 3 or (c.cat in MEN and 1 in earlier)

    tally = defaultdict(lambda: defaultdict(list))
    for c in scope:
        if c.date < lo or c.division_id not in theta_by_division:
            continue
        idx = [i for i, person in enumerate(c.persons) if not np.isnan(c.places[i]) and rated(c, person)]
        if len(idx) < 2:
            continue
        theta, places = theta_by_division[c.division_id][idx], c.places[idx]
        persons = [c.persons[i] for i in idx]
        predicted, actual = lineup_rank(theta, persons), lineup_rank(-places, persons)
        for key in (c.cat, "all"):
            t = tally[key]
            t["contests"].append(1.0)
            for x in range(len(idx)):
                for y in range(x + 1, len(idx)):
                    if places[x] != places[y]:
                        w, l = (x, y) if places[x] < places[y] else (y, x)
                        t["pairs"].append(float(theta[w] > theta[l]))
                        t["pair_logloss"].append(math.log1p(math.exp(theta[l] - theta[w])))
            favourite = int(np.argmin(predicted))
            t["favourite_wins"].append(float(places[favourite] == 1))
            t["favourite_wins_rated"].append(float(places[favourite] == places.min()))
            if len(idx) >= 3:
                t["top3_overlap"].append(float(np.sum((predicted <= 3) & (actual <= 3))) / 3)
                if np.ptp(theta) > 0 and np.ptp(places) > 0:
                    t["spearman"].append(spearmanr(-theta, places).statistic)
                    t["kendall"].append(kendalltau(-theta, places).statistic)

    def mean(xs):
        return round(float(np.nanmean(xs)), 4) if xs else None

    return {key: dict(contests=len(t["contests"]), pairs=len(t["pairs"]), pairs_correct=mean(t["pairs"]),
                      favourite_wins=mean(t["favourite_wins"]), favourite_wins_rated=mean(t["favourite_wins_rated"]),
                      top3_overlap=mean(t["top3_overlap"]), spearman=mean(t["spearman"]), kendall=mean(t["kendall"]),
                      pair_logloss=mean(t["pair_logloss"]))
            for key, t in tally.items()}
