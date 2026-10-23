"""Coordinate search over model hyperparameters on the tuning window."""
from __future__ import annotations

from typing import Callable

from .evaluate import _winner_arrays, fit_scale, walk_forward, winner_logloss

TUNE = ("2018-01-01", "2024-01-01")


def score(make: Callable, params: dict, contests, window=TUNE, keep=None) -> tuple[float, float]:
    preds = walk_forward(make(**params), contests, *window, keep=keep)
    arr = _winner_arrays(preds)
    s = fit_scale(preds)
    return winner_logloss(arr, s), s


def coordinate_search(make, params: dict, steps: dict, contests, rounds=3, log=print, keep=None):
    """steps: name -> list of candidate values, or a function of the current value returning that list."""
    best = dict(params)
    best_ll, best_s = score(make, best, contests, keep=keep)
    log(f"start ll={best_ll:.4f} {best}")
    for r in range(rounds):
        improved = False
        for name, cands in steps.items():
            if callable(cands):
                cands = cands(best[name])
            for v in cands:
                if v == best[name]:
                    continue
                trial = dict(best, **{name: v})
                ll, s = score(make, trial, contests, keep=keep)
                if ll < best_ll - 1e-5:
                    best, best_ll, best_s, improved = trial, ll, s, True
                    log(f"  r{r} {name}={v!r:>10} ll={ll:.4f} scale={s:.3f}")
        if not improved:
            break
    return best, best_ll, best_s


def around(f=1.5):
    return lambda v: [v / f, v * f]


def plus(d):
    return lambda v: [v - d, v + d]
