"""
Analyze validation mismatches to understand why predictions failed.

Focuses on winner / top-3 / top-5 accuracy and the largest placing errors.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from predictor.data_fetcher import create_athlete_profile, fetch_event_metadata
from predictor.predict_division import predict_division_as_of
from validator.metrics import ValidationMetrics


@dataclass
class MismatchReport:
    """Structured analysis of a validation run."""
    winner_accuracy: float = 0.0
    top3_accuracy: float = 0.0
    top5_accuracy: float = 0.0
    mae: float = 0.0
    mse: float = 0.0
    exact_match_rate: float = 0.0
    worst_errors: List[Dict] = field(default_factory=list)
    winner_misses: List[Dict] = field(default_factory=list)
    top3_misses: List[Dict] = field(default_factory=list)
    top5_misses: List[Dict] = field(default_factory=list)
    overpredicted_count: int = 0
    underpredicted_count: int = 0
    factor_investigations: List[Dict] = field(default_factory=list)
    hypotheses: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


def _comparison_row(metrics: ValidationMetrics, comp: Dict) -> Dict:
    return {
        'event_id': metrics.event_id,
        'event_name': metrics.event_name,
        'division': metrics.division,
        'division_id': metrics.division_id,
        'athlete': comp['name'],
        'athlete_id': comp['athlete_id'],
        'predicted': comp['predicted'],
        'actual': comp['actual'],
        'error': comp['error'],
    }


def analyze_validation_results(results: Dict, *, investigate_limit: int = 5) -> MismatchReport:
    """Build a mismatch report from aggregated validation output."""
    division_metrics: List[ValidationMetrics] = results.get('division_metrics') or []
    report = MismatchReport(
        winner_accuracy=results.get('overall_winner_accuracy', 0),
        top3_accuracy=results.get('overall_top3_accuracy', 0),
        top5_accuracy=results.get('overall_top5_accuracy', 0),
        mae=results.get('overall_mae', 0),
        mse=results.get('overall_mse', 0),
        exact_match_rate=results.get('overall_exact_match_rate', 0),
    )

    all_errors: List[Dict] = []
    for metrics in division_metrics:
        for comp in metrics.comparisons:
            row = _comparison_row(metrics, comp)
            all_errors.append(row)
            if comp['predicted'] < comp['actual']:
                report.overpredicted_count += 1
            elif comp['predicted'] > comp['actual']:
                report.underpredicted_count += 1

        if not metrics.winner_correct:
            actual_winner = next(
                (c for c in metrics.comparisons if c['actual'] == 1), None
            )
            predicted_winner = next(
                (c for c in metrics.comparisons if c['predicted'] == 1), None
            )
            report.winner_misses.append({
                'event_name': metrics.event_name,
                'division': metrics.division,
                'division_id': metrics.division_id,
                'predicted_winner': predicted_winner['name'] if predicted_winner else '?',
                'actual_winner': actual_winner['name'] if actual_winner else '?',
                'actual_winner_pred': actual_winner['predicted'] if actual_winner else None,
            })

        for comp in metrics.comparisons:
            in_pred_top3 = comp['predicted'] <= 3
            in_actual_top3 = comp['actual'] <= 3
            if in_pred_top3 != in_actual_top3:
                report.top3_misses.append(_comparison_row(metrics, comp))

            in_pred_top5 = comp['predicted'] <= 5
            in_actual_top5 = comp['actual'] <= 5
            if in_pred_top5 != in_actual_top5:
                report.top5_misses.append(_comparison_row(metrics, comp))

    report.worst_errors = sorted(all_errors, key=lambda r: r['error'], reverse=True)[:15]
    report.factor_investigations = _investigate_mismatches(
        report.worst_errors[:investigate_limit]
    )
    report.hypotheses = _build_hypotheses(report)
    return report


def _investigate_mismatches(rows: List[Dict]) -> List[Dict]:
    """Re-run prediction factors for the worst mismatches."""
    investigations: List[Dict] = []
    seen_divisions = set()

    for row in rows:
        division_id = row.get('division_id')
        if not division_id or division_id in seen_divisions:
            continue
        seen_divisions.add(division_id)

        metadata = fetch_event_metadata(row['event_id'])
        if not metadata or not metadata.get('cutoff_date'):
            continue

        try:
            target_division, predictions = predict_division_as_of(
                division_id, as_of_date=metadata['cutoff_date']
            )
        except Exception as exc:
            investigations.append({
                'division_id': division_id,
                'error': str(exc),
            })
            continue

        pred_by_id = {p.athlete_id: p for p in predictions}
        athlete_id = row['athlete_id']
        pred = pred_by_id.get(athlete_id)
        if not pred:
            continue

        profile = create_athlete_profile(athlete_id, before_date=metadata['cutoff_date'])
        recent = profile.recent_placings[:6] if profile else []

        investigations.append({
            'athlete': row['athlete'],
            'event_name': row['event_name'],
            'division': row['division'],
            'predicted': row['predicted'],
            'actual': row['actual'],
            'error': row['error'],
            'factors': {
                k: round(v, 1) if isinstance(v, (int, float)) else v
                for k, v in pred.factors.items()
                if k != 'weights_used'
            },
            'recent_placings': recent,
            'show_count': profile.total_shows if profile else 0,
        })

    return investigations


def _build_hypotheses(report: MismatchReport) -> List[str]:
    """Turn patterns into human-readable improvement hypotheses."""
    hypotheses: List[str] = []

    if report.winner_accuracy < 0.5:
        hypotheses.append(
            "Winner accuracy is weak: review recent_form and head_to_head weighting "
            "for athletes who finish 1st but were ranked outside the top 3."
        )

    if report.top3_accuracy < 0.65:
        hypotheses.append(
            "Top-3 overlap is low: podium athletes are being swapped — check momentum "
            "and H2H scores among the top composite cluster."
        )

    if report.top5_accuracy < 0.7:
        hypotheses.append(
            "Top-5 overlap is low: mid-podium separation may need stronger recent-form "
            "weighting or weaker momentum influence."
        )

    if report.underpredicted_count > report.overpredicted_count * 1.2:
        hypotheses.append(
            "More under-predictions than over-predictions: actual top placers are ranked "
            "too low — boost recent_form and/or reduce momentum penalty."
        )
    elif report.overpredicted_count > report.underpredicted_count * 1.2:
        hypotheses.append(
            "More over-predictions than under-predictions: athletes are ranked too high — "
            "reduce momentum and H2H influence or tighten recent-form scoring."
        )

    for inv in report.factor_investigations:
        factors = inv.get('factors') or {}
        if inv.get('error', 0) >= 5 and factors.get('head_to_head', 50) <= 50:
            hypotheses.append(
                f"{inv.get('athlete')}: large error with neutral H2H (50) — athlete may "
                "lack lineup history; regional show strength may be undervalued."
            )
        if inv.get('actual', 99) <= 3 and factors.get('momentum', 70) >= 90:
            hypotheses.append(
                f"{inv.get('athlete')}: high momentum ({factors.get('momentum')}) but "
                "missed top placement — momentum may be rewarding the wrong trend."
            )
        if inv.get('actual', 99) <= 5 and factors.get('momentum', 70) <= 40:
            hypotheses.append(
                f"{inv.get('athlete')}: low momentum ({factors.get('momentum')}) despite "
                "strong actual result — improving athletes may be penalized by momentum."
            )

    if not hypotheses:
        hypotheses.append(
            "No dominant mismatch pattern detected; try small weight rebalance toward "
            "recent_form and head_to_head."
        )

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for item in hypotheses:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique[:8]


def load_prior_learnings(memory_file: str = "agent_memory.json") -> List[str]:
    """Load condensed learnings from prior agent runs."""
    if not os.path.exists(memory_file):
        return []

    import json
    with open(memory_file) as handle:
        data = json.load(handle)

    lines: List[str] = []
    for exp in data.get('experiments', [])[-10:]:
        status = "kept" if exp.get('kept') else "reverted"
        lines.append(
            f"[{status}] {exp.get('idea_name')}: {exp.get('learnings', '')}"
        )

    for idea, learning in list(data.get('learnings', {}).items())[-5:]:
        detail = learning.get('what_worked') or learning.get('what_didnt_work', '')
        lines.append(f"{idea}: {detail}")

    return lines
