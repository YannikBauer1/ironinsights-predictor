"""v1 vs v2 on the 2026 divisions v1 scored before the show, and the Olympia 2026 pre-show predictions.

Reads data/final_test_preds.pkl (written by backtest_final.py).
"""
import pickle

import numpy as np
from scipy.stats import spearmanr

from bbpredict.evaluate import softmax
from bbpredict.montecarlo import summary


def rank_metrics(score, places):
    """score: higher = better. Returns top1 (ties shared), top-3 overlap, Spearman."""
    w = np.argmin(places)
    fav = np.flatnonzero(score == score.max())
    top1 = (1 / len(fav)) if w in fav else 0.0
    top3 = len(set(np.argsort(-score, kind="stable")[:3]) & set(np.argsort(places, kind="stable")[:3])) / min(3, len(places))
    rho = spearmanr(-score, places).statistic if np.ptp(score) > 0 else 0.0
    return top1, top3, 0.0 if np.isnan(rho) else rho


def v1_comparison(out):
    """Both models judged on the same athletes: those v1 gave a predicted place and who placed."""
    v1, v2, ll = [], [], []
    for p in out:
        c = p.contest
        if c.old_pred_place is None or c.winner is None:
            continue
        m = c.placed_mask & np.isfinite(c.old_pred_place)
        if m.sum() < 2 or not m[c.winner]:
            continue
        pl = c.places[m]
        v1.append(rank_metrics(-c.old_pred_place[m], pl))
        v2.append(rank_metrics(p.theta[m], pl))
        if c.old_pred_win is not None and np.isfinite(c.old_pred_win[m]).all():
            q = c.old_pred_win[m] / c.old_pred_win[m].sum()
            w = np.argmin(pl)
            ll.append((-np.log(max(q[w], 1e-6)), -np.log(softmax(p.theta[m])[w]), np.log(len(pl))))
    a, b, ll = np.array(v1), np.array(v2), np.array(ll)
    print(f"2026 divisions scored by v1 before the show: {len(a)}")
    print(f"  v1: winner {a[:, 0].mean():.1%}  top-3 overlap {a[:, 1].mean():.1%}  Spearman {a[:, 2].mean():.3f}")
    print(f"  v2: winner {b[:, 0].mean():.1%}  top-3 overlap {b[:, 1].mean():.1%}  Spearman {b[:, 2].mean():.3f}")
    print(f"  win-prob log-loss where v1 stored win %, {len(ll)} divisions: "
          f"v1 {ll[:, 0].mean():.3f}  v2 {ll[:, 1].mean():.3f}  uniform {ll[:, 2].mean():.3f}")


def olympia_2026(out):
    """Win % over the whole lineup, as it would have been shown before the show."""
    print("\nOLYMPIA 2026 (stage 2 trained on results up to 2025-12-31, ratings up to the day before)")
    whole_top3 = 0
    shows = [p for p in out if p.contest.comp == "olympia" and p.contest.year == 2026 and p.contest.winner is not None]
    for p in shows:
        c = p.contest
        s = summary(p.theta, 20000)
        order = list(np.argsort(-p.theta))
        w = c.winner
        placed = np.flatnonzero(c.placed_mask)
        act3 = set(placed[np.argsort(c.places[c.placed_mask], kind="stable")[:3]])
        hits = len(act3 & set(order[:5]))
        whole_top3 += hits == 3
        print(f"\n{c.cat} ({len(c.persons)} in lineup): winner {c.names[w]} had {s['win'][w]:.0%} "
              f"(model rank {order.index(w) + 1}); {hits} of the top 3 in the model's top 5")
        for k in order[:5]:
            actual = "-" if np.isnan(c.places[k]) else int(c.places[k])
            print(f"   {c.names[k]:28s} win {s['win'][k]:5.1%}  top3 {s['top3'][k]:5.1%}  "
                  f"exp.place {s['expected_place'][k]:4.1f}  actual {actual}")
    called = sum(int(np.argmax(p.theta) == p.contest.winner) for p in shows)
    print(f"\nwinners called: {called} of {len(shows)}; whole top 3 inside the model's top 5: {whole_top3} of {len(shows)}")


def main():
    with open("data/final_test_preds.pkl", "rb") as f:
        out = pickle.load(f)
    v1_comparison(out)
    olympia_2026(out)


if __name__ == "__main__":
    main()
