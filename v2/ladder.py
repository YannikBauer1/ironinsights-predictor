"""Every step of the model ladder on the same 2024-2026 test divisions.

Rating models get one temperature fitted on 2018-2023; the stage-2 steps are fitted on 2018-2023
rows; the final step is backtest_final.py's yearly refit. Reads data/walk.pkl and
data/final_test_preds.pkl.
"""
import pickle

import numpy as np

from bbpredict.data import cached_contests
from bbpredict.evaluate import Prediction, fit_scale, fmt, summarize, walk_forward
from bbpredict.models import Elo, RecentPlace, Uniform, WengLin
from bbpredict.stage2 import Stage2

CUT = np.datetime64("2024-01-01")
VIEWS = ["base", "elo", "b_pl_msep", "b_pl_fwd", "b_pl_short", "b_pl_long", "b_elo_msep"]
FEATURES = ["base_known", "rookie_pro", "switcher", "switch_elo", "log_shows_pool", "olympia_last_top3",
            "olympia_last_top6", "olympia_last_in", "olympia_champ", "defending_show", "prev_place_show_inv",
            "log_pro_wins", "won_last", "last_pct", "gap_b2b", "gap_short", "gap_year", "gap_long", "shows_365",
            "home", "masters_only", "years_pro", "trend", "olympia_ever_top3"]
WENG_LIN_TUNED = dict(sigma0=7.6, beta=0.0625, tau=0.0025, rookie=-0.8, masters_w=1.5, pred_sigma=-1.2,
                      tier_w={"O": 3.375, "2": 5.06, "MO": 3.375})


def report(name, preds, scale=None):
    tune = [p for p in preds if p.contest.date < CUT]
    test = [p for p in preds if p.contest.date >= CUT]
    s = fit_scale(tune) if scale is None else scale
    print(f"{name:62s}", fmt(summarize(test, s)), flush=True)


def stage2_step(name, preds, snap, names, cols):
    idx = [names.index(c) for c in cols]
    m = Stage2(k_terms=1, l2=1.0).fit([(snap[p.contest.division_id][:, idx], p.contest.places)
                                       for p in preds if p.contest.date < CUT])
    report(name, [Prediction(p.contest, m.utility(snap[p.contest.division_id][:, idx])) for p in preds], 1.0)


def main():
    cs = cached_contests()
    walk = lambda model: walk_forward(model, cs, "2018-01-01", "2026-12-31")
    report("Uniform (everyone equal)", walk(Uniform()), 1.0)
    report("Recent placings heuristic (v1 style)", walk(RecentPlace()))
    report("Weng-Lin Bayesian PL (tuned)", walk(WengLin(**WENG_LIN_TUNED)))
    report("Multiplayer Elo (tuned)", walk(Elo()))
    with open("data/walk.pkl", "rb") as f:
        preds, snap, names = pickle.load(f)
    report("Time-weighted reverse Plackett-Luce", preds)
    stage2_step("+ Elo and 5 more rating views (blend)", preds, snap, names, VIEWS)
    stage2_step("+ status, layoff, switch features (logit)", preds, snap, names, VIEWS + FEATURES)
    with open("data/final_test_preds.pkl", "rb") as f:
        report("+ boosted ranker, split Open/Masters, yearly refit (final)", pickle.load(f), 1.0)


if __name__ == "__main__":
    main()
