"""
Validation pipeline for the competition predictor.

Layers:
1. run_validation      - pick N random events and aggregate scores
2. validate_event       - validate every division in one event
3. validate_division    - hide results, predict division, compare to actual
"""

import time
from typing import Dict, List, Optional

# Fixed validation size for auto-improve and benchmarking
VALIDATION_EVENT_COUNT = 100

from predictor.data_fetcher import (
    fetch_division_actual_results,
    fetch_division_ids_for_event,
    fetch_event_metadata,
    fetch_random_validation_events,
)
from predictor.predict_division import predict_division_as_of
from validator.run_logger import log_division_validation, log_validation_run
from validator.metrics import ActualResult, ValidationMetrics, compare_prediction_to_actual


def validate_division(
    division_id: str,
    event_id: str,
    as_of_date: str,
    *,
    log_results: bool = True,
    event_name: str = "",
) -> Optional[ValidationMetrics]:
    """
    Validate one division: predict without seeing results, then compare.

    The predictor only receives lineup IDs and fetches history with a strict
    before_date cutoff (no same-day or future data).
    """
    actual_rows = fetch_division_actual_results(division_id)
    if len(actual_rows) < 3:
        return None

    target_division, predictions = predict_division_as_of(
        division_id, as_of_date=as_of_date
    )
    if not predictions:
        return None

    pred_list = [
        {'athlete_id': p.athlete_id, 'predicted_place': p.predicted_place}
        for p in predictions
    ]
    actual_objects = [
        ActualResult(row['athlete_id'], row['athlete_name'], row['place'])
        for row in actual_rows
    ]

    metrics = compare_prediction_to_actual(pred_list, actual_objects)
    if predictions:
        metrics.predicted_winner_win_probability = float(predictions[0].win_probability)
    metrics.event_id = event_id
    metrics.division = (
        target_division.category_type if target_division else division_id
    )
    metrics.division_id = division_id
    metrics.event_name = event_name

    if log_results:
        log_division_validation(
            metrics,
            event_name=event_name or event_id,
            as_of_date=as_of_date,
        )

    return metrics


def validate_event(
    event_id: str,
    *,
    log_results: bool = True,
    open_only: bool = True,
) -> List[ValidationMetrics]:
    """Validate divisions in one event (Open/Open only by default)."""
    metadata = fetch_event_metadata(event_id)
    if not metadata or not metadata.get('cutoff_date'):
        print(f"  Skipping event {event_id}: no cutoff date")
        return []

    event_name = metadata['event_name']
    as_of_date = metadata['cutoff_date']
    division_ids = fetch_division_ids_for_event(event_id, open_only=open_only)

    scope = "open " if open_only else ""
    print(f"  Event: {event_name} ({len(division_ids)} {scope}division(s))")
    metrics_list: List[ValidationMetrics] = []

    for division_id in division_ids:
        try:
            metrics = validate_division(
                division_id,
                event_id,
                as_of_date,
                event_name=event_name,
                log_results=log_results,
            )
            if metrics:
                metrics_list.append(metrics)
                p1_win = metrics.predicted_winner_win_probability
                print(
                    f"    Division {metrics.division}: "
                    f"exact={metrics.exact_match_rate:.1%}, "
                    f"top3={metrics.top3_accuracy:.1%}, "
                    f"#1 win chance={p1_win:.1%}, "
                    f"mae={metrics.mae:.1f}"
                )
        except Exception as exc:
            print(f"    Division {division_id} failed: {exc}")

    return metrics_list


def _aggregate_metrics(all_metrics: List[ValidationMetrics]) -> Dict:
    """Roll up per-division metrics into overall summary."""
    if not all_metrics:
        return {}

    total_exact_matches = sum(m.exact_matches for m in all_metrics)
    total_winner_correct = sum(1 for m in all_metrics if m.winner_correct)
    total_places = sum(m.num_athletes for m in all_metrics)
    total_top3_correct = sum(m.top3_correct for m in all_metrics)
    total_top5_correct = sum(m.top5_correct for m in all_metrics)
    total_top10_correct = sum(m.top10_correct for m in all_metrics)
    total_top3_attempts = sum(min(3, m.num_athletes) for m in all_metrics)
    total_top5_attempts = sum(min(5, m.num_athletes) for m in all_metrics)
    total_top10_attempts = sum(min(10, m.num_athletes) for m in all_metrics)
    total_mae = sum(m.mae for m in all_metrics)
    winner_win_probs = [
        m.predicted_winner_win_probability for m in all_metrics
        if m.predicted_winner_win_probability > 0
    ]
    avg_p1_win_probability = (
        sum(winner_win_probs) / len(winner_win_probs) if winner_win_probs else 0.0
    )
    total_mse = sum(
        sum(c['error'] ** 2 for c in m.comparisons)
        for m in all_metrics
    )
    successful = len(all_metrics)

    return {
        'total_events': len({m.event_id for m in all_metrics}),
        'total_divisions': successful,
        'total_places_predicted': total_places,
        'overall_exact_match_rate': total_exact_matches / total_places if total_places else 0,
        'overall_winner_accuracy': total_winner_correct / successful if successful else 0,
        'overall_avg_p1_win_probability': avg_p1_win_probability,
        'overall_top3_accuracy': total_top3_correct / total_top3_attempts if total_top3_attempts else 0,
        'overall_top5_accuracy': total_top5_correct / total_top5_attempts if total_top5_attempts else 0,
        'overall_top10_accuracy': total_top10_correct / total_top10_attempts if total_top10_attempts else 0,
        'overall_mae': total_mae / successful if successful else 0,
        'overall_mse': total_mse / total_places if total_places else 0,
        'division_metrics': all_metrics,
        'event_metrics': all_metrics,  # backward compat for generate_report.py
    }


def composite_score(aggregate: Dict) -> float:
    """Weighted composite score (same formula as autonomous_agent)."""
    return (
        aggregate.get('overall_winner_accuracy', 0) * 35
        + aggregate.get('overall_top3_accuracy', 0) * 30
        + aggregate.get('overall_top5_accuracy', 0) * 20
        + aggregate.get('overall_exact_match_rate', 0) * 10
        - aggregate.get('overall_mae', 0) * 3
        - aggregate.get('overall_mse', 0) * 0.05
    )


def run_validation(
    *,
    log_results: bool = True,
    open_only: bool = True,
    event_count: Optional[int] = None,
) -> Dict:
    """
    Run validation on random historical events.

    Validates event_count (default VALIDATION_EVENT_COUNT) random events with
    Open/Open divisions, validates each division independently, and returns metrics.
    """
    if event_count is None:
        event_count = VALIDATION_EVENT_COUNT
    print(f"Running validation on {event_count} random events...")
    if open_only:
        print("Scope: Open enhancement + Open age divisions only")
    print("=" * 60)

    events = fetch_random_validation_events(count=event_count)
    if not events:
        print("Error: No validation events found.")
        return {}

    all_metrics: List[ValidationMetrics] = []
    failed_events = 0

    for i, event in enumerate(events, 1):
        event_start = time.time()
        event_id = event['event_id']
        print(f"\nEvent {i}/{len(events)}: {event['event_name']}")

        try:
            event_metrics = validate_event(
                event_id,
                log_results=log_results,
                open_only=open_only,
            )
            all_metrics.extend(event_metrics)
            print(f"  Done in {time.time() - event_start:.1f}s")
        except Exception as exc:
            failed_events += 1
            print(f"  Failed: {exc}")

    if not all_metrics:
        print("No successful division validations.")
        return {}

    aggregate = _aggregate_metrics(all_metrics)

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Events processed:     {aggregate['total_events']}")
    print(f"Divisions validated:  {aggregate['total_divisions']}")
    print(f"Failed events:        {failed_events}")
    print(f"Exact match rate:     {aggregate['overall_exact_match_rate']:.1%}")
    print(f"Winner accuracy:      {aggregate['overall_winner_accuracy']:.1%}")
    print(
        f"Avg #1 win chance:    {aggregate.get('overall_avg_p1_win_probability', 0):.1%}"
    )
    print(f"Top 3 accuracy:       {aggregate['overall_top3_accuracy']:.1%}")
    print(f"Top 5 accuracy:       {aggregate['overall_top5_accuracy']:.1%}")
    print(f"Mean error:           {aggregate['overall_mae']:.1f} places")
    print(f"Composite score:      {composite_score(aggregate):.2f}")
    print()

    if log_results:
        log_validation_run(aggregate, event_count=event_count)

    return aggregate
