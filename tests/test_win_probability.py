import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import dataclass, field
from typing import Dict

from predictor.win_probability import apply_win_probabilities


@dataclass
class FakePrediction:
    factors: Dict[str, float] = field(default_factory=dict)
    win_probability: float = 0.0


def test_single_athlete_win_probability_is_one():
    p = FakePrediction(factors={"composite": 50.0})
    apply_win_probabilities([p])
    assert p.win_probability == 1.0


def test_win_probabilities_sum_to_one():
    preds = [
        FakePrediction(factors={"composite": 80.0}),
        FakePrediction(factors={"composite": 70.0}),
        FakePrediction(factors={"composite": 60.0}),
    ]
    apply_win_probabilities(preds)
    total = sum(p.win_probability for p in preds)
    assert math.isclose(total, 1.0, rel_tol=1e-9)
    assert preds[0].win_probability > preds[1].win_probability > preds[2].win_probability
