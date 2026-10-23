"""Placing probabilities from strengths: sample performances u + Gumbel noise (exactly Plackett-Luce)."""
from __future__ import annotations

import numpy as np

from .evaluate import softmax


def place_probabilities(u: np.ndarray, sims: int = 20000, seed: int = 0) -> np.ndarray:
    """P[i, k] = probability that entrant i finishes in place k + 1."""
    rng = np.random.default_rng(seed)
    n = len(u)
    perf = u[None, :] + rng.gumbel(size=(sims, n))
    order = np.argsort(-perf, axis=1)
    P = np.zeros((n, n))
    for k in range(n):
        P[:, k] = np.bincount(order[:, k], minlength=n) / sims
    return P


def summary(u: np.ndarray, sims: int = 20000) -> dict:
    """Win (closed form), top-3, top-5 and expected placing per entrant."""
    win = softmax(u)
    P = place_probabilities(u, sims)
    places = np.arange(1, len(u) + 1)
    return dict(win=win, top3=P[:, :3].sum(1), top5=P[:, :5].sum(1), expected_place=P @ places, P=P)
