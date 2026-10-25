import numpy as np
import pytest
from scipy.optimize import check_grad

from bbpredict.data import Contest
from bbpredict.evaluate import contest_metrics, walk_forward
from bbpredict.features import judge_placings
from bbpredict.montecarlo import summary
from bbpredict.plfit import TimeDecayPL, encode, neg_loglik, stack


def contest(div, date, persons, places, prejudge=None, age="Open"):
    n = len(persons)
    return Contest(division_id=div, event_id=f"e{date}", date=np.datetime64(date), year=int(date[:4]), cat="bikini",
                   age=age, enh="Open", comp="show", tier="3", weight_class="", persons=list(persons),
                   places=np.array(places, float),
                   prejudge=np.array(prejudge if prejudge is not None else [np.nan] * n, float),
                   total=np.full(n, np.nan), names=list(persons), countries=[""] * n)


class Spy:
    """Records, at every predict, the latest contest date it had been updated with."""
    def __init__(self):
        self.seen = None
        self.log = []

    def predict(self, c):
        self.log.append((c.date, self.seen))
        return np.zeros(len(c.persons))

    def update(self, c):
        self.seen = c.date if self.seen is None else max(self.seen, c.date)


def test_walk_forward_never_learns_from_the_same_day_or_later():
    cs = [contest("a", "2020-01-01", "xy", [1, 2]), contest("b", "2020-02-01", "xy", [2, 1]),
          contest("c", "2020-02-01", "xz", [1, 2], age="40+"), contest("d", "2020-03-01", "yz", [1, 2])]
    spy = Spy()
    walk_forward(spy, cs, "2020-01-15", "2021-01-01")
    assert [str(d) for d, _ in spy.log] == ["2020-02-01", "2020-02-01", "2020-03-01"]
    for date, seen in spy.log:
        assert seen < date


def test_pl_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    parts = []
    for _ in range(6):
        n = rng.integers(3, 8)
        ids = rng.choice(10, n, replace=False)
        pl = np.sort(rng.integers(1, 6, n)).astype(float)
        parts += [(encode(ids, pl, False), rng.uniform(0.5, 2)), (encode(ids, pl, True), rng.uniform(0.5, 2))]
    s = stack(parts)
    f = lambda th: neg_loglik(th, s, 0.3, np.zeros(10))[0]
    g = lambda th: neg_loglik(th, s, 0.3, np.zeros(10))[1]
    assert check_grad(f, g, rng.normal(size=10), epsilon=1e-6) < 1e-4


def test_unordered_tail_contributes_no_terms():
    e = encode(np.arange(5), np.array([1, 2, 3, 16, 16.0]), False, None)
    assert e.term.tolist() == [True, True, True, False, False]


def test_pl_fit_ranks_a_repeat_winner_first():
    cs = [contest(f"d{i}", f"2020-0{i + 1}-01", "abc", [1, 2, 3]) for i in range(5)]
    cs.append(contest("t", "2020-08-01", "abc", [np.nan] * 3))
    m = TimeDecayPL()
    for c in cs[:-1]:
        m.update(c)
    th = m.predict(cs[-1])
    assert th[0] > th[1] > th[2]


def test_judge_placings_infers_three_counted_judges():
    c = contest("j", "2020-01-01", "abcd", [1, 2, 3, 4], prejudge=[3, 7, 8, 12])
    assert np.allclose(judge_placings(c), [1.0, 7 / 3, 8 / 3, 4.0])


def test_place_probabilities_are_distributions_and_win_matches_softmax():
    u = np.array([1.5, 0.5, 0.0, -1.0])
    s = summary(u, sims=40000)
    assert np.allclose(s["P"].sum(0), 1) and np.allclose(s["P"].sum(1), 1)
    assert np.allclose(s["P"][:, 0], s["win"], atol=0.01)
    assert s["expected_place"][0] < s["expected_place"][-1]


def test_win_metrics_count_the_whole_lineup_not_only_who_placed():
    c = contest("w", "2020-01-01", "abcd", [1, 2, 3, np.nan])
    m = contest_metrics(np.zeros(4), c)
    assert m["ll_win"] == pytest.approx(np.log(4))
    assert m["top1"] == pytest.approx(0.25)


def _features_after(history, target):
    from bbpredict.features import FEATURES, FeatureRecorder
    from bbpredict.models import Uniform
    rec = FeatureRecorder(Uniform())
    for c in history:
        rec.update(c)
    X = rec.features(target)
    return {name: X[:, FEATURES.index(name)] for name in FEATURES}


def test_head_to_head_counts_earlier_meetings_with_this_lineup():
    history = [contest("h1", "2020-01-01", "abc", [1, 2, 3]), contest("h2", "2020-03-01", "ab", [1, 2])]
    f = _features_after(history, contest("t", "2020-06-01", "ab", [np.nan, np.nan]))
    assert f["h2h_net"][0] > 0 > f["h2h_net"][1]
    # under Uniform the first entrant counts as top rated: a's rival is b, everyone else's rival is a
    assert f["h2h_vs_top"].tolist() == pytest.approx([1.0, -1.0])


def test_age_group_features_see_a_move_up_in_bracket():
    history = [contest("m1", "2020-01-01", "ab", [1, 2], age="40+")]
    f = _features_after(history, contest("t", "2021-01-01", "ac", [np.nan, np.nan], age="50+"))
    assert f["age_moved_up"].tolist() == [1.0, 0.0]
    assert f["agegrp_new"].tolist() == [1.0, 0.0]


def test_a_pro_win_after_the_olympia_counts_as_qualified():
    olympia = contest("o", "2020-09-01", "abc", [1, 2, 3])
    olympia.tier = "O"
    win = contest("w", "2021-03-01", "d", [1])
    f = _features_after([olympia, win], contest("t", "2021-05-01", "adx", [np.nan] * 3))
    assert f["qualified"].tolist() == [1.0, 1.0, 0.0]


def test_pl_likelihood_stays_finite_when_contests_sit_far_apart():
    # contest 1 scores near +1000, contest 2 near 0: one global shift would underflow contest 2 to 0/0
    s = stack([(encode(np.array([0, 1]), np.array([1.0, 2.0]), False), 1.0),
               (encode(np.array([2, 3]), np.array([1.0, 2.0]), False), 1.0)])
    val, grad = neg_loglik(np.array([1000.0, 999.0, 0.0, -1.0]), s, 0.0, np.zeros(4))
    assert np.isfinite(val) and np.isfinite(grad).all()
    assert val == pytest.approx(2 * np.log1p(np.exp(-1.0)))


def test_pl_likelihood_stays_exact_with_a_wide_spread_inside_one_contest():
    # 2nd and 3rd place sit 40 log units below the winner; a later contest must not swamp their sums
    s = stack([(encode(np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0]), False), 1.0),
               (encode(np.array([3, 4]), np.array([1.0, 2.0]), False), 1.0)])
    theta = np.array([0.0, -40.0, -45.0, 0.0, 0.0])
    val, grad = neg_loglik(theta, s, 0.0, np.zeros(5))
    expected = -(0 - np.logaddexp.reduce([0, -40, -45]) + (-40 - np.logaddexp(-40, -45)) + (0 - np.log(2)))
    assert np.isfinite(val) and np.isfinite(grad).all()
    assert val == pytest.approx(expected, rel=1e-9)


def test_stage2_refuses_a_fit_that_breaks_down():
    from bbpredict.stage2 import Stage2
    with pytest.raises(RuntimeError), np.errstate(all="ignore"):
        Stage2().fit([(np.array([[np.inf], [0.0], [1.0]]), np.array([1.0, 2.0, 3.0]))])


def test_features_do_not_depend_on_the_order_of_the_lineup():
    # historical lineups are stored in finishing order: any order dependence would leak the result
    from bbpredict.features import FeatureRecorder
    from bbpredict.models import Uniform
    history = [contest("h1", "2020-01-01", "abc", [1, 2, 3]), contest("h2", "2020-03-01", "bd", [1, 2])]
    rec = FeatureRecorder(Uniform())
    for c in history:
        rec.update(c)
    first = rec.features(contest("t1", "2020-06-01", "abcdxy", [np.nan] * 6)).copy()
    reversed_ = rec.features(contest("t2", "2020-06-01", "yxdcba", [np.nan] * 6))
    assert np.allclose(first, reversed_[::-1])


def test_an_unknown_country_looks_exactly_like_a_home_athlete():
    # person.country is filled in later, mostly for athletes who became notable: "unknown" must not be visible
    from bbpredict.features import FeatureRecorder
    from bbpredict.models import Uniform
    c = contest("t", "2020-06-01", "abc", [np.nan] * 3)
    c.countries, c.event_country = ["US", "", "BR"], "US"
    X = FeatureRecorder(Uniform()).features(c)
    assert np.array_equal(X[0], X[1])
    assert not np.array_equal(X[0], X[2])


def test_power_ranking_field_keeps_30_months_skips_retired_and_says_how_long_away():
    from bbpredict.features import FeatureRecorder
    from bbpredict.models import Uniform
    from bbpredict.pipeline import ranking_field
    shows = [contest("old", "2024-05-01", "ab", [1, 2]), contest("new", "2026-09-01", "cd", [1, 2]),
             contest("gone", "2023-01-01", "e", [1])]
    f = ranking_field(FeatureRecorder(Uniform()), shows, "bikini", "2026-10-01", inactive={"b"})
    assert sorted(f.contest.persons) == ["a", "c", "d"]          # b retired, e beyond 30 months
    assert f.last["a"]["months_away"] == 29
    assert f.last["c"]["months_away"] == 1


def test_the_olympia_champion_is_marked_unbeaten_until_someone_beats_them():
    olympia = contest("o", "2025-10-01", "abc", [1, 2, 3])
    olympia.tier = "O"
    target = lambda date: contest("t", date, "ab", [np.nan, np.nan])
    win = contest("w", "2026-03-01", "ac", [1, 2])
    masters_loss = contest("m", "2026-04-01", "ca", [1, 2], age="40+")
    assert _features_after([olympia, win, masters_loss], target("2026-06-01"))["champ_unbeaten"].tolist() == [1.0, 0.0]
    loss = contest("l", "2026-04-01", "ca", [1, 2])
    f = _features_after([olympia, win, loss], target("2026-06-01"))
    assert f["champ_unbeaten"].tolist() == [0.0, 0.0]
    assert f["reigning_champ"].tolist() == [1.0, 0.0]                  # beaten, but still the champion


def test_the_champion_reigns_until_the_next_olympia_and_not_after_it():
    first = contest("o1", "2019-09-15", "abc", [1, 2, 3])
    second = contest("o2", "2020-12-18", "bc", [1, 2])                # a skipped it: b is the new champion
    first.tier = second.tier = "O"
    target = lambda date: contest("t", date, "ab", [np.nan, np.nan])
    assert _features_after([first], target("2020-12-10"))["champ_unbeaten"].tolist() == [1.0, 0.0]   # 452 days on
    f = _features_after([first, second], target("2021-01-10"))
    assert f["champ_unbeaten"].tolist() == [0.0, 1.0]
    assert f["reigning_champ"].tolist() == [0.0, 1.0]
    assert _features_after([first], target("2021-10-01"))["reigning_champ"].tolist() == [0.0, 0.0]   # no Olympia for 2 years


def test_the_open_layer_adds_a_fixed_bonus_for_the_unbeaten_champion():
    from bbpredict.blend import SplitBlend
    from bbpredict.features import FEATURES

    class Zero:
        def utility(self, X):
            return np.zeros(len(X))

        def predict(self, X):
            return np.zeros(len(X))

    model = SplitBlend(FEATURES, columns=["base"], masters_columns=["base"], champion_bonus=0.5)
    for layer in (model.open, model.masters):
        layer.logit = layer.booster = Zero()
    X = np.zeros((2, len(FEATURES)))
    X[0, FEATURES.index("champ_unbeaten")] = 1.0
    assert model.utility(X, contest("t", "2026-06-01", "ab", [np.nan] * 2)).tolist() == [0.5, 0.0]
    assert model.utility(X, contest("t", "2026-06-01", "ab", [np.nan] * 2, age="40+")).tolist() == [0.0, 0.0]
