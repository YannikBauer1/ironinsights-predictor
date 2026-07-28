#!/usr/bin/env python3
"""
Batch validation CLI — test the prediction algorithm on historical events.

  python validator/cli.py --count 100 --save

Pipeline: run_validation → validate_event → validate_division → predict_division
Logs: logs/validation/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predictor.data_fetcher import fetch_event_metadata
from validator.validation import (
    VALIDATION_EVENT_COUNT,
    run_validation,
    validate_division,
    validate_event,
)


def run_batch_validation(
    *,
    log_results: bool = True,
    open_only: bool = True,
    event_count: Optional[int] = None,
) -> Dict:
    return run_validation(
        log_results=log_results,
        open_only=open_only,
        event_count=event_count,
    )


def analyze_failures(metrics: List) -> Dict:
    failures = []
    for m in metrics:
        for comp in m.comparisons:
            if comp['error'] > 2:
                failures.append({
                    'event_id': m.event_id,
                    'athlete': comp['name'],
                    'predicted': comp['predicted'],
                    'actual': comp['actual'],
                    'error': comp['error'],
                })
    large_errors = [f for f in failures if f['error'] >= 3]
    return {
        'total_significant_errors': len(failures),
        'large_errors': len(large_errors),
        'error_rate': len(failures) / sum(m.num_athletes for m in metrics) if metrics else 0,
        'common_failure_patterns': {
            'overpredicted': len([f for f in failures if f['predicted'] < f['actual']]),
            'underpredicted': len([f for f in failures if f['predicted'] > f['actual']]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Batch validate prediction algorithm')
    parser.add_argument('--event', type=str, help='Validate a single event by ID')
    parser.add_argument('--division', type=str, help='Validate a single division by ID (requires --event)')
    parser.add_argument('--save', action='store_true', help='Save results JSON to reports/')
    parser.add_argument('--no-log', action='store_true', help='Skip writing comparison logs')
    parser.add_argument(
        '--all-divisions', action='store_true',
        help='Include masters/natural divisions (default: Open/Open only)',
    )
    parser.add_argument(
        '--count', type=int, default=None,
        help=f'Number of random events to validate (default: {VALIDATION_EVENT_COUNT})',
    )
    args = parser.parse_args()

    open_only = not args.all_divisions

    if args.division:
        if not args.event:
            parser.error('--division requires --event')
        meta = fetch_event_metadata(args.event)
        if not meta or not meta.get('cutoff_date'):
            print(f"Error: no cutoff date for event {args.event}")
            sys.exit(1)
        validate_division(
            args.division,
            args.event,
            meta['cutoff_date'],
            log_results=not args.no_log,
            event_name=meta['event_name'],
        )
        sys.exit(0)

    if args.event:
        validate_event(args.event, log_results=not args.no_log, open_only=open_only)
        sys.exit(0)

    event_count = args.count if args.count is not None else VALIDATION_EVENT_COUNT
    results = run_batch_validation(
        log_results=not args.no_log,
        open_only=open_only,
        event_count=event_count,
    )

    if results:
        print(f"\nValidated {event_count} random events.")
        print("\nAnalysis complete!")
        if args.save:
            from validator.save_results import save_validation_results
            filepath = save_validation_results(results)
            print(f"\nResults saved to: {filepath}")


if __name__ == '__main__':
    main()
