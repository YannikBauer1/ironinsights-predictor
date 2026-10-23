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
    assert check_grad(f, g, rng.normal(size=10)) < 1e-4


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
