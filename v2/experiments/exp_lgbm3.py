"""LightGBM stage 2 on the multi-base features, iterations picked on a time split inside the tuning window."""
import pickle, warnings
import numpy as np
import lightgbm as lgb
from bbpredict.evaluate import Prediction, summarize, fit_scale, fmt
from bbpredict.stage2 import Stage2
warnings.filterwarnings("ignore")

preds, snap, names = pickle.load(open("data/walk.pkl", "rb"))
cut, vcut = np.datetime64("2024-01-01"), np.datetime64("2022-01-01")
tune = [p for p in preds if p.contest.date < cut]
test = [p for p in preds if p.contest.date >= cut]
tr = [p for p in tune if p.contest.date < vcut]
va = [p for p in tune if p.contest.date >= vcut]
cols = ["base", "elo", "b_pl_msep", "b_pl_fwd", "b_pl_short", "b_pl_long", "b_elo_msep",
        "base_known", "rookie_pro", "switcher", "switch_elo", "log_shows_pool", "olympia_last_top3", "olympia_last_top6",
        "olympia_last_in", "olympia_champ", "defending_show", "prev_place_show_inv", "log_pro_wins", "won_last", "last_pct",
        "gap_b2b", "gap_short", "gap_year", "gap_long", "shows_365", "home", "masters_only",
        "years_pro", "trend", "olympia_ever_top3"]
idx = [names.index(c) for c in cols]
tier_code = {"O": 0, "2": 1, "3": 2, "MO": 3, "N": 4}
cats = sorted({p.contest.cat for p in preds})

def X_of(p):
    c = p.contest
    X = snap[c.division_id][:, idx]
    ctx = np.tile([tier_code[c.tier], float(c.is_open_age), cats.index(c.cat), c.n_placed], (len(X), 1))
    return np.hstack([X, ctx])

def mats(ps):
    X, y, g = [], [], []
    for p in ps:
        m = p.contest.placed_mask
        X.append(X_of(p)[m]); y.append(np.clip(6 - p.contest.places[m], 0, 5)); g.append(m.sum())
    return np.vstack(X), np.concatenate(y), np.array(g)

params = dict(objective="lambdarank", learning_rate=0.03, num_leaves=7, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, label_gain=list(range(32)),
              lambdarank_truncation_level=10)
Xtr, ytr, gtr = mats(tr)
b = lgb.train(params, lgb.Dataset(Xtr, ytr, group=gtr, categorical_feature=[len(cols) + 2]), 1500)
best = None
for it in [200, 400, 600, 800, 1000, 1500]:
    sc = lambda ps: [Prediction(p.contest, b.predict(X_of(p), num_iteration=it)) for p in ps]
    s = fit_scale(sc(tr), grid=np.exp(np.linspace(-4, 6, 101)))
    ll = summarize(sc(va), s)["ll_win"]
    print(f"it={it} val ll={ll:.4f}")
    if best is None or ll < best[1]:
        best = (it, ll)
it = best[0]
Xt, yt, gt = mats(tune)
b = lgb.train(params, lgb.Dataset(Xt, yt, group=gt, categorical_feature=[len(cols) + 2]), it)
sc = lambda ps: [Prediction(p.contest, b.predict(X_of(p))) for p in ps]
s = fit_scale(sc(tune), grid=np.exp(np.linspace(-4, 6, 101)))
lg_test = sc(test)
print(f"LGBM it={it}            test", fmt(summarize(lg_test, s)))
# conditional logit on the same columns (no ctx) for the blend
m = Stage2(k_terms=1, l2=1.0).fit([(snap[p.contest.division_id][:, idx], p.contest.places) for p in tune])
cl = lambda ps: [Prediction(p.contest, m.utility(snap[p.contest.division_id][:, idx])) for p in ps]
print("logit                   test", fmt(summarize(cl(test))))
# blend: average of the two calibrated utilities, weight picked on the tuning window
lt, ct = sc(tune), cl(tune)
for w in [0.3, 0.5, 0.7]:
    bl = lambda L, C: [Prediction(a.contest, w * a.theta * s + (1 - w) * c.theta) for a, c in zip(L, C)]
    print(f"blend w_lgbm={w} tune ll={summarize(bl(lt, ct))['ll_win']:.4f} test", fmt(summarize(bl(lg_test, cl(test)))))
