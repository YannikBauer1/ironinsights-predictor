"""Stage-2 features: everything known about an entrant before the show, beyond the base rating.

FeatureRecorder sits in the walk-forward loop like a model: on predict() it snapshots
features from strictly earlier history, on update() it adds the contest to the history.
"""
from __future__ import annotations

import math
import zlib
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .data import DIVISIONS, Contest, age_number
from .models import Elo

FEATURES = [
    "base", "base_known",           # base log-strength, and an indicator that the athlete has pool history
    "elo",                          # pool Elo, centred on a rookie, per 100 points
    "rookie_pro",                   # no pro result in any pool before
    "switcher",                     # no history in this pool, but in another pool
    "switch_elo",                   # their best other-pool Elo, centred (0 if not a switcher)
    "log_shows_pool",
    "years_pro", "years_pro_sq",    # career length (any pool), capped at 15
    "olympia_last_top3", "olympia_last_top6", "olympia_last_in",   # placing at the latest Olympia (13 months) in this pool
    "olympia_champ",                # won the latest Olympia in this pool
    "olympia_ever_top3",            # top 3 at any Olympia in this pool
    "defending_show",               # won this competition's previous edition in this pool
    "prev_place_show_inv",          # 1/place at this competition's previous edition (0 if absent)
    "log_pro_wins",
    "won_last",                     # won their most recent contest in this pool
    "last_pct",                     # percentile of their most recent placing in this pool (1 = won)
    "trend",                        # mean percentile of the last 2 pool results minus the 3 before
    "gap_b2b", "gap_short", "gap_year", "gap_long",   # layoff since last show (any pool)
    "shows_365",                    # contests in the last 365 days (any pool)
    "home",                         # athlete country == event country
    "masters_only",                 # in this pool they have only ever done masters classes
    "score_resid_last",             # average judge placing minus final place, latest scored pool contest (<0 = closer to the next place up)
    "score_resid_mean",             # the same, mean over the latest 3 scored pool contests
    "win_margin_last",              # when they won their latest scored pool contest: runner-up minus winner average placing
    # contest-context interactions
    "oly_x_base", "oly_x_top3", "oly_x_champ", "oly_x_rookie",
    "mas_x_base", "mas_x_masters_only",
] + [f"base_x_{c}" for c in DIVISIONS] + [
    # Olympia qualification (the Open Olympia of this division, also used in masters classes as a strength signal):
    # top 3 at the latest Olympia, or an Open-age pro win since it. Masters Olympia qualification is not modelled.
    "qualified", "qual_x_t3",       # already qualified; and that, at a regular (tier 3) show
    "near_oly", "near_oly_x_qual", "near_oly_x_top6",   # within 8 weeks before the next Olympia (assumed a year after the last)
    # head to head against this lineup, pool results of the last 3 years, recency weighted
    "h2h_net",                      # (wins - losses) / (1 + meetings) against everyone in this lineup
    "h2h_meet",                     # log(1 + meetings) with this lineup
    "h2h_vs_top",                   # net record (-1..1) against their highest-rated rival in this lineup
    # masters age groups
    "agegrp_shows",                 # log(1 + earlier placed contests in this exact age group)
    "agegrp_wins",                  # log(1 + earlier wins in this exact age group)
    "agegrp_last_pct",              # percentile of their latest result in this age group
    "agegrp_new",                   # masters class, first time in this age group but has history in the division
    "age_moved_up",                 # masters class above every bracket they did before (just aged into it)
    # round 3: time away, travel, and where the athlete stands within this lineup
    # (an unknown country counts as the event's own: see _country)
    "years_away",                   # years since their last show in any division, capped at 10 (0 also for rookies,
                                    # whom rookie_pro marks)
    "years_away_pool",              # years since their last placed result in this division, capped at 10
    "stale_x_base",                 # base rating times min(years_away_pool, 5), so the model can discount old ratings
    "away",                         # athlete's country differs from the event's
    "foreign_at_us",                # athlete from outside the USA at a show in the USA
]
H2H_DAYS, H2H_TAU = 3 * 365, 730.0


@dataclass
class HistoryEntry:
    """One past contest of one athlete, as the features need it."""
    date: np.datetime64
    pool: str
    comp: str
    tier: str
    open_age: bool
    place: float
    n: int
    avg_jp: float = math.nan        # average counted-judge placing at prejudging
    margin: float = math.nan        # for a winner: runner-up minus winner average placing
    age: str = "Open"               # the class's age group


def judge_placings(c: Contest) -> np.ndarray:
    """judging_1 is the sum of the counted judges' placings; divide by the judge count to get an average placing."""
    out = np.full(len(c.persons), math.nan)
    m = c.placed_mask & ~np.isnan(c.prejudge)
    k = int(m.sum())
    if k < 2:
        return out
    j = round(c.prejudge[m].sum() / (k * (k + 1) / 2))
    if j >= 1:
        out[m] = c.prejudge[m] / j
    return out


def _country(c: Contest, i: int) -> str:
    """Entrant i's country, an unknown one counting as the event's own. person.country is filled in
    after the fact, mostly for athletes who became notable, so "unknown" must not be visible to the model."""
    return c.countries[i] or c.event_country


def placing_percentile(r: HistoryEntry) -> float:
    """1 for a win, 0 for last place."""
    return 1 - (r.place - 1) / max(r.n - 1, 1) if r.n > 1 else 1.0


class FeatureRecorder:
    def __init__(self, base, extra: dict | None = None):
        self.base = base
        self.extra = extra or {}                 # name -> further base model, recorded as column b_<name>
        self.names = FEATURES + [f"b_{n}" for n in self.extra]
        self.elo = Elo()
        self.hist: dict = defaultdict(list)      # person -> [HistoryEntry]
        self.meetings: dict = defaultdict(list)  # (pool, person) -> [(date, {person: place})] of placed contests
        self.olympias: dict = defaultdict(list)  # pool -> dates of its Open-age Olympias
        self.snap: dict = {}                     # division_id -> X

    def predict(self, c: Contest):
        theta = np.asarray(self.base.predict(c), float)
        X = np.zeros((len(c.persons), len(self.names)))
        # ties broken by a hash of the person id (stable across runs, unlike hash()), never by list
        # position: stored lineups are in finishing order, so position would leak the result
        by_rating = sorted(range(len(theta)), key=lambda k: (-theta[k], zlib.crc32(c.persons[k].encode())))
        for i, p in enumerate(c.persons):
            f = self._features(p, c, theta[i])
            country = _country(c, i)
            f["home"] = float(bool(c.event_country) and country == c.event_country)
            rival = c.persons[by_rating[1] if by_rating[0] == i else by_rating[0]] if len(theta) > 1 else None
            f.update(self._head_to_head(p, c, rival))
            f["away"] = float(bool(c.event_country) and country != c.event_country)
            f["foreign_at_us"] = float(c.event_country == "US" and country != "US")
            X[i, :len(FEATURES)] = [f[k] for k in FEATURES]
        for j, m in enumerate(self.extra.values()):
            X[:, len(FEATURES) + j] = m.predict(c)
        self.snap[c.division_id] = X
        return theta

    def features(self, c: Contest) -> np.ndarray:
        """The feature matrix of a contest (one row per entrant, columns = self.names)."""
        self.predict(c)
        return self.snap[c.division_id]

    def update(self, c: Contest):
        self.base.update(c)
        self.elo.update(c)
        for m in self.extra.values():
            m.update(c)
        n = c.n_placed
        ajp = judge_placings(c)
        margin = math.nan
        if np.isfinite(ajp).sum() >= 2:
            order = np.argsort(np.where(np.isnan(c.places), np.inf, c.places), kind="stable")
            a1, a2 = ajp[order[0]], ajp[order[1]]
            if np.isfinite(a1) and np.isfinite(a2) and c.places[order[0]] == 1:
                margin = a2 - a1
        for p, pl, a in zip(c.persons, c.places, ajp):
            self.hist[p].append(HistoryEntry(c.date, c.pool, c.comp, c.tier, c.is_open_age, pl, n, a,
                                    margin if pl == 1 else math.nan, c.age))
        placed = {p: pl for p, pl in zip(c.persons, c.places) if not math.isnan(pl)}
        for p in placed:
            self.meetings[(c.pool, p)].append((c.date, placed))
        if c.tier == "O" and c.is_open_age:
            self.olympias[c.pool].append(c.date)

    def _head_to_head(self, p, c: Contest, rival) -> dict:
        """Record against this lineup, and against `rival` (the highest-rated other entrant)."""
        field = set(c.persons) - {p}
        net = meet = vs_top = top_meet = 0.0
        for d, places in self.meetings[(c.pool, p)]:
            days_ago = (c.date - d).astype(int)
            if days_ago > H2H_DAYS:
                continue
            w = math.exp(-days_ago / H2H_TAU)
            mine = places[p]
            for q in field & places.keys():
                if places[q] == mine:
                    continue
                s = 1.0 if mine < places[q] else -1.0
                net += w * s
                meet += w
                if q == rival:
                    vs_top += w * s
                    top_meet += w
        return dict(h2h_net=net / (1 + meet), h2h_meet=math.log1p(meet),
                    h2h_vs_top=vs_top / top_meet if top_meet else 0.0)

    def _features(self, p, c: Contest, base) -> dict:
        h = self.hist[p]
        f = dict.fromkeys(FEATURES, 0.0)
        pool_rows = [r for r in h if r.pool == c.pool]
        placed_pool = [r for r in pool_rows if not math.isnan(r.place)]
        f["base"] = base
        f["base_known"] = float(bool(placed_pool))
        f["elo"] = (self.elo.r.get((p, c.pool), self.elo.rookie) - self.elo.rookie) / 100
        f["rookie_pro"] = float(not h)
        if not pool_rows and h:
            f["switcher"] = 1.0
            others = {r.pool for r in h}
            best = max(self.elo.r.get((p, o), self.elo.rookie) for o in others)
            f["switch_elo"] = (best - self.elo.rookie) / 100
        f["log_shows_pool"] = math.log1p(len(placed_pool))
        if h:
            yp = min((c.date - h[0].date).astype(int) / 365, 15.0)
            f["years_pro"], f["years_pro_sq"] = yp, yp * yp / 15
        olympia = [r for r in placed_pool if r.tier == "O"]
        recent_o = [r for r in olympia if (c.date - r.date).astype(int) <= 400]
        if recent_o:
            last = recent_o[-1]
            f["olympia_last_in"] = 1.0
            f["olympia_last_top6"] = float(last.place <= 6)
            f["olympia_last_top3"] = float(last.place <= 3)
            f["olympia_champ"] = float(last.place == 1)
        f["olympia_ever_top3"] = float(any(r.place <= 3 for r in olympia))
        same = [r for r in pool_rows if r.comp == c.comp and (c.date - r.date).astype(int) <= 550]
        if same and not math.isnan(same[-1].place):
            f["defending_show"] = float(same[-1].place == 1)
            f["prev_place_show_inv"] = 1.0 / same[-1].place
        f["log_pro_wins"] = math.log1p(sum(1 for r in placed_pool if r.place == 1 and r.open_age))
        if placed_pool:
            last = placed_pool[-1]
            f["won_last"] = float(last.place == 1)
            f["last_pct"] = placing_percentile(last)
            if len(placed_pool) >= 3:
                f["trend"] = np.mean([placing_percentile(r) for r in placed_pool[-2:]]) - np.mean([placing_percentile(r) for r in placed_pool[-5:-2]])
        if h:
            gap = (c.date - h[-1].date).astype(int)
            f["gap_b2b"] = float(gap <= 21)
            f["gap_short"] = float(21 < gap <= 120)
            f["gap_year"] = float(365 < gap <= 730)
            f["gap_long"] = float(gap > 730)
            f["shows_365"] = sum(1 for r in h if (c.date - r.date).astype(int) <= 365)
            f["years_away"] = min(gap / 365, 10.0)
        if placed_pool:
            f["years_away_pool"] = min((c.date - placed_pool[-1].date).astype(int) / 365, 10.0)
            f["stale_x_base"] = base * min(f["years_away_pool"], 5.0)
        f["masters_only"] = float(bool(pool_rows) and not any(r.open_age for r in pool_rows))
        scored = [r for r in placed_pool if not math.isnan(r.avg_jp)]
        if scored:
            f["score_resid_last"] = scored[-1].avg_jp - scored[-1].place
            f["score_resid_mean"] = float(np.mean([r.avg_jp - r.place for r in scored[-3:]]))
            if scored[-1].place == 1 and not math.isnan(scored[-1].margin):
                f["win_margin_last"] = min(scored[-1].margin, 3.0)
        oly, mas = float(c.tier == "O"), float(not c.is_open_age)
        f["oly_x_base"], f["oly_x_top3"] = oly * base, oly * f["olympia_last_top3"]
        f["oly_x_champ"], f["oly_x_rookie"] = oly * f["olympia_champ"], oly * (1 - f["base_known"])
        f["mas_x_base"], f["mas_x_masters_only"] = mas * base, mas * f["masters_only"]
        if c.cat in DIVISIONS:
            f[f"base_x_{c.cat}"] = base
        f.update(self._qualification(c, placed_pool, f["olympia_last_top6"]))
        f.update(self._age_group(c, pool_rows, placed_pool))
        return f

    def _qualification(self, c: Contest, placed_pool, top6_last_olympia: float) -> dict:
        f: dict = {}
        past = [d for d in self.olympias[c.pool] if d < c.date]
        if c.enh != "Open" or not past:
            return f
        last_oly = past[-1]
        top3_there = any(r.tier == "O" and r.date == last_oly and r.place <= 3 for r in placed_pool)
        won_since = any(r.place == 1 and r.open_age and r.tier in ("O", "2", "3") and r.date > last_oly
                        for r in placed_pool)
        f["qualified"] = float(top3_there or won_since)
        f["qual_x_t3"] = f["qualified"] * float(c.tier == "3")
        days_to = (last_oly + np.timedelta64(365, "D") - c.date).astype(int)
        f["near_oly"] = float(c.tier != "O" and 0 <= days_to <= 56)
        f["near_oly_x_qual"] = f["near_oly"] * f["qualified"]
        f["near_oly_x_top6"] = f["near_oly"] * top6_last_olympia
        return f

    def _age_group(self, c: Contest, pool_rows, placed_pool) -> dict:
        f: dict = {}
        if c.is_open_age:
            return f
        group = [r for r in placed_pool if r.age == c.age]
        f["agegrp_shows"] = math.log1p(len(group))
        f["agegrp_wins"] = math.log1p(sum(1 for r in group if r.place == 1))
        if group:
            f["agegrp_last_pct"] = placing_percentile(group[-1])
        f["agegrp_new"] = float(not group and bool(pool_rows))
        brackets_before = [age_number(r.age) for r in pool_rows if not r.open_age]
        f["age_moved_up"] = float(bool(brackets_before) and age_number(c.age) > max(brackets_before))
        return f
