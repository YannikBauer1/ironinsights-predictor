"""Time-weighted Plackett-Luce fitted by MAP: the "everything we know up to the day before" rating.

For a show on date d every earlier contest of the pool counts with weight
exp(-(d - t) / tau) * tier weight, and every athlete gets a normal prior around 0.
Unlike online systems, a result is reinterpreted when we learn more about the opponents.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .data import Contest, contest_weight, default_pool


@dataclass
class Encoded:
    """One contest flattened into selection order for a forward or reverse PL likelihood."""
    ids: np.ndarray     # athlete (or row) ids in selection order
    first: np.ndarray   # offset of each position's tie group start
    term: np.ndarray    # does this position contribute a selection term
    sign: float         # +1 forward (winner chosen first), -1 reverse (last place chosen first)


def encode(ids: np.ndarray, places: np.ndarray, reverse: bool, max_terms: int | None = None) -> Encoded:
    order = np.argsort(-places if reverse else places, kind="stable")
    ids, pl = ids[order], places[order]
    key = -pl if reverse else pl
    first = np.searchsorted(key, key, side="left")
    term = pl != pl[-1]                              # the final tie group is unordered
    if max_terms is not None and not reverse:
        term &= np.arange(len(pl)) < max_terms
    return Encoded(ids, first, term, -1.0 if reverse else 1.0)


@dataclass
class Stacked:
    """Encoded contests laid end to end, the flat layout neg_loglik works on."""
    ids: np.ndarray
    first: np.ndarray       # global position where each position's tie group starts
    seg_start: np.ndarray   # global position where each position's contest starts
    seg_end: np.ndarray     # global position just after each position's contest
    term: np.ndarray
    weight: np.ndarray
    sign: np.ndarray


def stack(parts: list[tuple[Encoded, float]]) -> Stacked:
    """parts: (encoded contest, weight of all its terms)."""
    cols = defaultdict(list)
    off = 0
    for e, w in parts:
        m = len(e.ids)
        for name, values in (("ids", e.ids), ("first", e.first + off), ("term", e.term),
                             ("seg_start", np.full(m, off)), ("seg_end", np.full(m, off + m)),
                             ("weight", np.full(m, w)), ("sign", np.full(m, e.sign))):
            cols[name].append(values)
        off += m
    return Stacked(**{k: np.concatenate(v) for k, v in cols.items()})


def neg_loglik(theta: np.ndarray, s: Stacked, lam: float, prior: np.ndarray):
    """Weighted negative PL log-likelihood of every stacked contest plus an L2 pull towards `prior`.

    Position k picks its athlete out of everyone at or after its tie group start (Breslow ties).
    Returns (value, gradient with respect to theta).
    """
    v = s.sign * theta[s.ids]
    vmax = v.max()
    e = np.exp(v - vmax)
    tail = np.concatenate([np.cumsum(e[::-1])[::-1], [0.0]])     # sum of e from a position to the end
    denom = tail[s.first] - tail[s.seg_end]
    t = s.term
    ll = np.sum(s.weight[t] * (v[t] - vmax - np.log(denom[t])))
    a = np.bincount(s.first[t], weights=s.weight[t] / denom[t], minlength=len(v))
    cum = np.concatenate([[0.0], np.cumsum(a)])
    exposure = cum[1:] - cum[s.seg_start]       # sum of w/denom over the terms whose choice set holds this position
    grad_v = s.weight * t - e * exposure
    g = np.bincount(s.ids, weights=s.sign * grad_v, minlength=len(theta))
    d = theta - prior
    return -ll + 0.5 * lam * d @ d, -g + lam * d


class TimeDecayPL:
    def __init__(self, tau=365.0, window_years=6.0, lam=1.0, tier_w=None, masters_w=1.0,
                 reverse=False, max_terms=None, rookie=-0.5, pool_fn=None):
        self.tau, self.window = tau, np.timedelta64(int(window_years * 365), "D")
        self.lam, self.tier_w, self.masters_w = lam, tier_w or {}, masters_w
        self.reverse, self.max_terms, self.rookie = reverse, max_terms, rookie
        self.pool_of = pool_fn or default_pool
        self.pool_ids: dict = defaultdict(dict)        # pool -> person -> id
        self.store: dict = defaultdict(list)           # pool -> [(date, weight, Encoded)]
        self.theta: dict = {}                          # pool -> last fitted vector (warm start)
        self.cache: dict = {}                          # (pool, date) -> theta
        self.last_seen: dict = {}                      # (pool, person) -> date

    def _id(self, pool, person):
        ids = self.pool_ids[pool]
        if person not in ids:
            ids[person] = len(ids)
        return ids[person]

    def update(self, c: Contest):
        pool = self.pool_of(c)
        mask = c.placed_mask
        if mask.sum() < 2:
            return
        persons = np.array(c.persons)[mask]
        ids = np.array([self._id(pool, p) for p in persons])
        enc = encode(ids, c.places[mask], self.reverse, self.max_terms)
        self.store[pool].append((c.date, contest_weight(c, self.tier_w, self.masters_w), enc))
        for p in persons:
            self.last_seen[(pool, p)] = c.date

    def fit(self, pool, date):
        key = (pool, date)
        if key in self.cache:
            return self.cache[key]
        n = len(self.pool_ids[pool])
        theta0 = np.zeros(n)
        prev = self.theta.get(pool)
        if prev is not None:
            theta0[:len(prev)] = prev
        parts = [(e, w * np.exp(-(date - d).astype(int) / self.tau))
                 for d, w, e in self.store[pool] if date - self.window <= d < date]
        if not parts:
            self.cache[key] = theta0
            return theta0
        res = minimize(neg_loglik, theta0, args=(stack(parts), self.lam, np.zeros(n)), jac=True,
                       method="L-BFGS-B", options=dict(maxiter=500, gtol=1e-6))
        self.theta[pool] = self.cache[key] = res.x
        return res.x

    def predict(self, c: Contest):
        pool = self.pool_of(c)
        th = self.fit(pool, c.date)
        ids = self.pool_ids[pool]
        return np.array([th[ids[p]] if (pool, p) in self.last_seen and ids[p] < len(th) else self.rookie
                         for p in c.persons])
