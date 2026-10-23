import pickle
import numpy as np
from bbpredict.evaluate import Prediction, summarize, fit_scale, fmt
from bbpredict.stage2 import Stage2

preds, snap, names = pickle.load(open("data/walk.pkl", "rb"))
cut = np.datetime64("2024-01-01")
tune = [p for p in preds if p.contest.date < cut]
test = [p for p in preds if p.contest.date >= cut]

def single(col):
    j = names.index(col)
    w = lambda ps: [Prediction(p.contest, snap[p.contest.division_id][:, j]) for p in ps]
    s = fit_scale(w(tune))
    print(f"{col:18s} alone       test", fmt(summarize(w(test), s)))

def run(name, cols, l2=1.0, k=1, show=False):
    idx = [names.index(c) for c in cols]
    m = Stage2(k_terms=k, l2=l2).fit([(snap[p.contest.division_id][:, idx], p.contest.places) for p in tune])
    wrap = lambda ps: [Prediction(p.contest, m.utility(snap[p.contest.division_id][:, idx])) for p in ps]
    a, b = summarize(wrap(tune)), summarize(wrap(test))
    print(f"{name:30s} tune={a['ll_win']:.4f} test", fmt(b))
    if show:
        for nm, per_unit, std_beta in m.coef_table(cols):
            print(f"     {nm:22s} {per_unit:+.3f}  (std {std_beta:+.3f})")
    return m

for col in ["base", "elo", "b_pl_msep", "b_pl_fwd", "b_pl_short", "b_pl_long", "b_elo_msep"]:
    single(col)
feat = ["base_known", "rookie_pro", "switcher", "switch_elo", "log_shows_pool", "olympia_last_top3", "olympia_last_top6",
        "olympia_last_in", "olympia_champ", "defending_show", "prev_place_show_inv", "log_pro_wins", "won_last", "last_pct",
        "gap_b2b", "gap_short", "gap_year", "gap_long", "shows_365", "home", "masters_only"]
score = ["score_resid_last", "score_resid_mean", "win_margin_last"]
inter = ["oly_x_base", "mas_x_base", "mas_x_masters_only"]
run("bases: base+elo", ["base", "elo"])
run("bases: all", ["base", "elo", "b_pl_msep", "b_pl_fwd", "b_pl_short", "b_pl_long", "b_elo_msep"])
run("base+elo + feat", ["base", "elo"] + feat)
run("all bases + feat", ["base", "elo", "b_pl_msep", "b_pl_fwd", "b_pl_short", "b_pl_long", "b_elo_msep"] + feat)
run("all bases + feat + score", ["base", "elo", "b_pl_msep", "b_pl_fwd", "b_pl_short", "b_pl_long", "b_elo_msep"] + feat + score)
m = run("all bases + feat + score + int", ["base", "elo", "b_pl_msep", "b_pl_fwd", "b_pl_short", "b_pl_long", "b_elo_msep"] + feat + score + inter, show=True)
