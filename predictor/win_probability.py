"""Map composite scores to a win probability distribution (sums to 1 per division)."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from predictor.predictor import Prediction


def apply_win_probabilities(
    predictions: Sequence["Prediction"],
    *,
    temperature: float = 12.0,
) -> None:
    """
    Softmax over composite score — relative P(finish 1st) within the field.

    Not calibrated to historical win rates; use alongside validation winner accuracy.
    """
    if not predictions:
        return
    if len(predictions) == 1:
        predictions[0].win_probability = 1.0
        return

    scores = [float(p.factors.get("composite", 0.0)) for p in predictions]
    max_score = max(scores)
    temp = max(temperature, 1e-6)
    weights = [math.exp((s - max_score) / temp) for s in scores]
    total = sum(weights)
    if total <= 0:
        uniform = 1.0 / len(predictions)
        for p in predictions:
            p.win_probability = uniform
        return
    for p, w in zip(predictions, weights):
        p.win_probability = w / total
