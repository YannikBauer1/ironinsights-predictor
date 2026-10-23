"""Walk-forward backtesting: predict every contest from strictly earlier results only.

All contests that start on the same date are predicted before any of them is learned from,
so an athlete doing the Open and the Masters class of one weekend never leaks into himself.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from itertools import groupby
from typing import Callable, Iterable, Protocol

import numpy as np
from scipy.stats import spearmanr

from .data import Contest


class Model(Protocol):
    def predict(self, c: Contest) -> np.ndarray: ...   # log-strengths, one per entry (higher = better)
    def update(self, c: Contest) -> None: ...


@dataclass
class Prediction:
    contest: Contest
    theta: np.ndarray  # log-strengths before any calibration


def walk_forward(model: Model, contests: list[Contest], start: str, end: str,
                 keep: Callable[[Contest], bool] | None = None) -> list[Prediction]:
    s, e = np.datetime64(start), np.datetime64(end)
    out = []
    for date, group in groupby(contests, key=lambda c: c.date):
        if date >= e:
            break
        group = list(group)
        if date >= s:
            for c in group:
                if c.n_placed >= 2 and (keep is None or keep(c)):
                    out.append(Prediction(c, np.asarray(model.predict(c), float)))
        for c in group:
            model.update(c)
    return out


def softmax(x: np.ndarray) -> np.ndarray:
    z = np.exp(x - x.max())
    return z / z.sum()


def pl_loglik_topk(theta: np.ndarray, places: np.ndarray, k: int) -> float:
    """Plackett-Luce log-likelihood of the observed top-k order (ties: Breslow)."""
    order = np.argsort(places, kind="stable")
    th, pl = theta[order], places[order]
    ll, used = 0.0, 0
    for i in range(len(th)):
        if used >= k or pl[i] == pl[-1]:   # the bottom tie group is unordered
            break
        first = np.searchsorted(pl, pl[i], side="left")
        rest = th[first:]
        m = rest.max()
        ll += th[i] - (m + np.log(np.exp(rest - m).sum()))
        used += 1
    return ll


def contest_metrics(theta: np.ndarray, c: Contest, scale: float = 1.0) -> dict | None:
    """Win metrics over the whole lineup (pre-show, no-shows included); order metrics over who placed."""
    w = c.winner
    if w is None:
        return None
    th = theta * scale
    n = len(th)
    p = softmax(th)
    top = np.flatnonzero(th == th.max())
    rng = np.random.default_rng(zlib.crc32(c.division_id.encode()))   # stable tie-breaks across runs
    jitter = th + rng.uniform(0, 1e-9, n)
    mask = c.placed_mask
    placed = np.flatnonzero(mask)
    act_top3 = set(placed[np.argsort(c.places[mask], kind="stable")[:3]])
    pred_top3 = set(np.argsort(-jitter)[:3])
    thp, pl = th[mask], c.places[mask]
    rho = spearmanr(-thp, pl).statistic if np.ptp(thp) > 0 and np.ptp(pl) > 0 else 0.0
    y = np.eye(n)[w]
    return dict(
        n=n,
        ll_win=-np.log(max(p[w], 1e-12)),
        ll_unif=np.log(n),
        top1=(1.0 / len(top)) if w in top else 0.0,
        top3=len(pred_top3 & act_top3) / len(act_top3),
        spearman=0.0 if np.isnan(rho) else rho,
        pl5=pl_loglik_topk(thp, pl, 5),
        pl5_unif=pl_loglik_topk(np.zeros(len(pl)), pl, 5),
        brier=float(((p - y) ** 2).sum()),
        p_fav=float(p.max()),
        fav_won=float(np.argmax(jitter) == w),
    )


def summarize(preds: Iterable[Prediction], scale: float = 1.0) -> dict:
    rows = [m for pr in preds if (m := contest_metrics(pr.theta, pr.contest, scale))]
    if not rows:
        return {}
    keys = rows[0].keys()
    agg = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    agg["divisions"] = len(rows)
    agg["skill_win"] = 1 - agg["ll_win"] / agg["ll_unif"]
    agg["skill_pl5"] = 1 - agg["pl5"] / agg["pl5_unif"]
    return agg


def _winner_arrays(preds: list[Prediction]):
    pairs = [(pr.theta, pr.contest.winner) for pr in preds if pr.contest.winner is not None]
    return [t for t, _ in pairs], [w for _, w in pairs]


def winner_logloss(preds_or_arrays, scale: float) -> float:
    ths, ws = preds_or_arrays if isinstance(preds_or_arrays, tuple) else _winner_arrays(preds_or_arrays)
    tot = 0.0
    for th, w in zip(ths, ws):
        z = th * scale
        m = z.max()
        tot += m + np.log(np.exp(z - m).sum()) - z[w]
    return tot / len(ths)


def fit_scale(preds: list[Prediction], grid=np.exp(np.linspace(-3, 3, 61))) -> float:
    """One global temperature that minimises the winner log-loss on the rows given."""
    arr = _winner_arrays(preds)
    best = min(grid, key=lambda s: winner_logloss(arr, s))
    fine = best * np.exp(np.linspace(-0.1, 0.1, 21))
    return float(min(fine, key=lambda s: winner_logloss(arr, s)))


def calibration_table(preds: list[Prediction], scale: float, bins=(0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001)):
    ps, ys = [], []
    for pr in preds:
        w = pr.contest.winner
        if w is None:
            continue
        p = softmax(pr.theta * scale)
        ps.extend(p)
        ys.extend(np.eye(len(p))[w])
    ps, ys = np.array(ps), np.array(ys)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (ps >= lo) & (ps < hi)
        if m.sum():
            out.append((lo, hi, int(m.sum()), float(ps[m].mean()), float(ys[m].mean())))
    ece = sum(n * abs(a - b) for _, _, n, a, b in out) / len(ps)
    return out, ece


def fmt(agg: dict) -> str:
    return (f"div={agg['divisions']:5d} llwin={agg['ll_win']:.4f} skill={agg['skill_win']:.4f} "
            f"top1={agg['top1']:.3f} top3={agg['top3']:.3f} rho={agg['spearman']:.3f} "
            f"pl5skill={agg['skill_pl5']:.4f} brier={agg['brier']:.4f}")
