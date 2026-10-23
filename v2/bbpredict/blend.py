"""The final stage 2: a conditional logit and a LightGBM ranker on the same features, blended 50/50.

The logit is the interpretable backbone; the ranker adds interactions (division x tier x field size).
Both are fitted on walk-forward rows only, so no row ever saw its own result. The ranker's scale and
the blend's temperature are fitted on the last year of those rows by models that did not train on it.
"""
from __future__ import annotations

import numpy as np

from .data import DIVISIONS
from .evaluate import Prediction, fit_scale
from .stage2 import Stage2

COLUMNS = ["base", "elo", "b_pl_msep", "b_pl_fwd", "b_pl_short", "b_pl_long", "b_elo_msep",
           "base_known", "rookie_pro", "switcher", "switch_elo", "log_shows_pool", "olympia_last_top3",
           "olympia_last_top6", "olympia_last_in", "olympia_champ", "defending_show", "prev_place_show_inv",
           "log_pro_wins", "won_last", "last_pct", "gap_b2b", "gap_short", "gap_year", "gap_long", "shows_365",
           "home", "masters_only", "years_pro", "trend", "olympia_ever_top3"]
# round 2 input groups; they helped masters classes only, so only the masters layer gets them
AGE_GROUP = ["agegrp_shows", "agegrp_wins", "agegrp_last_pct", "agegrp_new", "age_moved_up"]
AGE_GROUP_VIEWS = ["b_pl_age", "b_elo_age"]
HEAD_TO_HEAD = ["h2h_net", "h2h_meet", "h2h_vs_top"]
QUALIFICATION = ["qualified", "qual_x_t3", "near_oly", "near_oly_x_qual", "near_oly_x_top6"]
MASTERS_COLUMNS = COLUMNS + AGE_GROUP + AGE_GROUP_VIEWS + HEAD_TO_HEAD + QUALIFICATION
TIERS = {"O": 0, "2": 1, "3": 2, "MO": 3, "N": 4}
LGBM_PARAMS = dict(objective="lambdarank", learning_rate=0.03, num_leaves=7, min_data_in_leaf=100,
                   feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1,
                   label_gain=list(range(32)), lambdarank_truncation_level=10, seed=7)
HOLDOUT = np.timedelta64(365, "D")


class Blend:
    def __init__(self, names: list[str], columns=COLUMNS, iterations=600, w_lgbm=0.5, l2=1.0):
        self.idx = [names.index(c) for c in columns]
        self.columns, self.iterations, self.w, self.l2 = columns, iterations, w_lgbm, l2
        self.logit = self.booster = None
        self.scale = self.temperature = 1.0

    def _with_context(self, X, c, field_size=None):
        """Feature columns plus show tier, Open-age flag, division and field size (the lineup's size)."""
        n = field_size or len(c.persons)
        cat = DIVISIONS.index(c.cat) if c.cat in DIVISIONS else -1   # -1 = missing for LightGBM
        ctx = np.tile([TIERS[c.tier], float(c.is_open_age), cat, n], (len(X), 1))
        return np.hstack([X[:, self.idx], ctx])

    def _train(self, rows):
        import lightgbm as lgb
        logit = Stage2(k_terms=1, l2=self.l2).fit([(X[:, self.idx], c.places) for X, c in rows])
        Xs, ys, gs = [], [], []
        for X, c in rows:
            m = c.placed_mask
            Xs.append(self._with_context(X, c)[m])
            ys.append(np.clip(6 - c.places[m], 0, 5))
            gs.append(m.sum())
        data = lgb.Dataset(np.vstack(Xs), np.concatenate(ys), group=np.array(gs),
                           categorical_feature=[len(self.idx) + 2])
        return logit, lgb.train(LGBM_PARAMS, data, self.iterations)

    def _raw(self, logit, booster, X, c, field_size=None):
        return logit.utility(X[:, self.idx]), booster.predict(self._with_context(X, c, field_size))

    def fit(self, rows):
        """rows: [(X_full, contest)] from the FeatureRecorder snapshot."""
        rows = [(X, c) for X, c in rows if c.n_placed >= 2]
        cut = max(c.date for _, c in rows) - HOLDOUT
        early = [(X, c) for X, c in rows if c.date < cut]
        late = [(X, c) for X, c in rows if c.date >= cut]
        logit, booster = self._train(early)
        raw = [(self._raw(logit, booster, X, c), c) for X, c in late]
        self.scale = fit_scale([Prediction(c, b) for (_, b), c in raw], grid=np.exp(np.linspace(-4, 6, 101)))
        mixed = [Prediction(c, (1 - self.w) * a + self.w * self.scale * b) for (a, b), c in raw]
        self.temperature = fit_scale(mixed, grid=np.exp(np.linspace(-1, 1, 41)))
        self.logit, self.booster = self._train(rows)
        return self

    def utility(self, X, c, field_size=None):
        a, b = self._raw(self.logit, self.booster, X, c, field_size)
        return self.temperature * ((1 - self.w) * a + self.w * self.scale * b)


class SplitBlend:
    """One Blend for Open-age classes and one for masters classes: the two judge and mix differently."""

    def __init__(self, names: list[str], masters_columns=COLUMNS, **kw):
        self.open = Blend(names, **kw)
        self.masters = Blend(names, **dict(kw, columns=masters_columns))

    def fit(self, rows):
        self.open.fit([(X, c) for X, c in rows if c.is_open_age])
        self.masters.fit([(X, c) for X, c in rows if not c.is_open_age])
        return self

    def utility(self, X, c, field_size=None):
        return (self.open if c.is_open_age else self.masters).utility(X, c, field_size)
