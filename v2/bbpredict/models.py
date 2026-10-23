"""Rating models. Each one predicts log-strengths for a contest, then learns from its result.

Every model keys an athlete by (person, pool): a pool is a category type plus the open or
natural circuit, so masters classes share the open rating of the same category.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np

from .data import Contest, contest_weight, default_pool


def placed_order(c: Contest):
    """Indices and places of the athletes who placed, best first."""
    idx = np.flatnonzero(c.placed_mask)
    idx = idx[np.argsort(c.places[idx], kind="stable")]
    return idx, c.places[idx]


class Uniform:
    def predict(self, c): return np.zeros(len(c.persons))
    def update(self, c): pass


class RecentPlace:
    """Heuristic baseline in the spirit of v1: average of tier bonus minus log(place) over recent shows."""

    def __init__(self, k=5, years=3.0, tier_bonus=None, rookie=-2.5):
        self.k, self.days, self.rookie = k, years * 365, rookie
        self.tier_bonus = tier_bonus or {"O": 2.0, "2": 0.7, "MO": 0.0, "3": 0.0, "N": 0.0}
        self.hist = defaultdict(lambda: deque(maxlen=self.k))

    def predict(self, c):
        out = []
        for p in c.persons:
            h = [v for d, v in self.hist[(p, c.pool)] if (c.date - d).astype(int) <= self.days]
            out.append(np.mean(h) if h else self.rookie)
        return np.array(out)

    def update(self, c):
        bonus = self.tier_bonus.get(c.tier, 0.0) + (0.0 if c.is_open_age else -1.0)
        for i in np.flatnonzero(c.placed_mask):
            self.hist[(c.persons[i], c.pool)].append((c.date, bonus - math.log(c.places[i])))


class Elo:
    """Multiplayer Elo: every pair in the final order is a game, K shared over the field.

    Defaults are the values tuned on 2018 to 2023 (K 48, field exponent 0.25, tier-2 shows 1.5x,
    Masters Olympia 2/3x).
    """

    def __init__(self, k=48.0, field_exp=0.25, rookie=-150.0, tier_k=None, masters_k=1.0, pool_fn=None):
        self.k, self.field_exp, self.rookie = k, field_exp, rookie
        self.tier_k = {"2": 1.5, "MO": 2 / 3} if tier_k is None else tier_k
        self.masters_k = masters_k
        self.r: dict = {}
        self.pool_of = pool_fn or default_pool

    def rating(self, key):
        return self.r.get(key, self.rookie)

    def predict(self, c):
        return np.array([self.rating((p, self.pool_of(c))) for p in c.persons]) * math.log(10) / 400

    def update(self, c):
        idx, pl = placed_order(c)
        n = len(idx)
        if n < 2:
            return
        keys = [(c.persons[i], self.pool_of(c)) for i in idx]
        r = np.array([self.rating(k) for k in keys])
        expected = 1 / (1 + 10 ** ((r[None, :] - r[:, None]) / 400))      # [i, j] = P(i beats j)
        scored = (pl[:, None] < pl[None, :]).astype(float) + 0.5 * (pl[:, None] == pl[None, :])
        np.fill_diagonal(scored, 0)
        np.fill_diagonal(expected, 0)
        k = self.k * contest_weight(c, self.tier_k, self.masters_k)
        delta = k / (n - 1) ** self.field_exp * (scored - expected).sum(1)
        for key, ri, di in zip(keys, r, delta):
            self.r[key] = ri + di


class WengLin:
    """Bayesian Plackett-Luce rating (Weng and Lin 2011, the OpenSkill PL model).

    mu/sigma per athlete; sigma grows with time away (tau per sqrt(day)), so a long layoff
    makes the next result count more.
    """

    def __init__(self, sigma0=1.5, beta=1.0, tau=0.02, kappa=1e-4, rookie=-0.8,
                 tier_w=None, masters_w=1.0, pred_sigma=0.0):
        self.sigma0, self.beta, self.tau, self.kappa = sigma0, beta, tau, kappa
        self.rookie, self.tier_w, self.masters_w, self.pred_sigma = rookie, tier_w or {}, masters_w, pred_sigma
        self.mu: dict = {}
        self.s2: dict = {}
        self.last: dict = {}

    def state(self, key, date):
        if key not in self.mu:
            return self.rookie, self.sigma0 ** 2
        dt = max((date - self.last[key]).astype(int), 0)
        return self.mu[key], min(self.s2[key] + self.tau ** 2 * dt, self.sigma0 ** 2)

    def predict(self, c):
        out = []
        for p in c.persons:
            mu, s2 = self.state((p, c.pool), c.date)
            out.append(mu - self.pred_sigma * math.sqrt(s2))
        return np.array(out)

    def update(self, c):
        idx, pl = placed_order(c)
        n = len(idx)
        if n < 2:
            return
        keys = [(c.persons[i], c.pool) for i in idx]
        st = [self.state(k, c.date) for k in keys]
        mu = np.array([s[0] for s in st])
        s2 = np.array([s[1] for s in st])
        w = contest_weight(c, self.tier_w, self.masters_w)
        c_norm = math.sqrt((s2 + self.beta ** 2).sum())
        ex = np.exp((mu - mu.max()) / c_norm)
        # sum over everyone placed at or below each rank; a = size of each tie group
        first = np.searchsorted(pl, pl, side="left")
        tail = np.cumsum(ex[::-1])[::-1]
        sum_q = tail[first]
        a = np.array([(pl == x).sum() for x in pl], float)
        omega = np.zeros(n)
        delta = np.zeros(n)
        for i in range(n):
            for q in range(n):
                if pl[q] > pl[i]:
                    continue
                quot = ex[i] / sum_q[q]
                omega[i] += ((1 - quot) if q == i else -quot) / a[q]
                delta[i] += quot * (1 - quot) / a[q]
        gamma = np.sqrt(s2) / c_norm
        new_mu = mu + w * s2 / c_norm * omega
        new_s2 = s2 * np.maximum(1 - w * gamma * s2 / c_norm ** 2 * delta, self.kappa)
        for k, m, v in zip(keys, new_mu, new_s2):
            self.mu[k], self.s2[k], self.last[k] = m, v, c.date
