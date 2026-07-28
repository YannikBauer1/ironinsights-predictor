"""
Persist validation output after each run.

Writes human-readable predicted-vs-actual logs under logs/validation/ so
results are available even if terminal output is lost or interrupted.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from validator.metrics import ValidationMetrics

_REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = str(_REPO_ROOT / "logs" / "validation")
LATEST_DIVISION_LOG = os.path.join(LOG_DIR, "latest_division.log")
LATEST_RUN_LOG = os.path.join(LOG_DIR, "latest_run.log")


def _ensure_log_dir() -> None:
    os.makedirs(LOG_DIR, exist_ok=True)


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _slug(text: str, max_len: int = 40) -> str:
    slug = "".join(c if c.isalnum() else "_" for c in text.lower()).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug[:max_len] or "unknown"


def format_division_comparison(
    metrics: ValidationMetrics,
    event_name: str = "",
    division_name: str = "",
    as_of_date: str = "",
) -> str:
    """Build a predicted vs actual comparison report."""
    lines: List[str] = []
    header = event_name or metrics.event_id or "Unknown event"
    div = division_name or metrics.division or "Unknown division"

    lines.append("=" * 88)
    lines.append(f"VALIDATION: {header}")
    lines.append(f"Division:   {div}")
    if as_of_date:
        lines.append(f"Cutoff:     {as_of_date}")
    lines.append(f"Timestamp:  {datetime.now().isoformat(timespec='seconds')}")
    lines.append("=" * 88)
    lines.append("")
    lines.append(
        f"Exact: {metrics.exact_matches}/{metrics.num_athletes} "
        f"({metrics.exact_match_rate:.1%})  |  "
        f"Winner: {'✓' if metrics.winner_correct else '✗'}  |  "
        f"Top 3: {metrics.top3_correct}/3 ({metrics.top3_accuracy:.1%})  |  "
        f"MAE: {metrics.mae:.2f}"
    )
    lines.append("")

    by_pred = sorted(metrics.comparisons, key=lambda c: c["predicted"])
    lines.append("BY PREDICTED PLACE")
    lines.append(f"{'Pred':>4}  {'Actual':>6}  {'Δ':>3}  {'Athlete':<36}  {'Match':>5}")
    lines.append("-" * 62)
    for comp in by_pred:
        mark = "✓" if comp["exact_match"] else ""
        lines.append(
            f"{comp['predicted']:>4}  {comp['actual']:>6}  {comp['error']:>3}  "
            f"{comp['name'][:36]:<36}  {mark:>5}"
        )

    lines.append("")
    lines.append("BY ACTUAL PLACE")
    lines.append(f"{'Actual':>6}  {'Pred':>4}  {'Δ':>3}  {'Athlete':<36}  {'Match':>5}")
    lines.append("-" * 62)
    for comp in sorted(metrics.comparisons, key=lambda c: c["actual"]):
        mark = "✓" if comp["exact_match"] else ""
        lines.append(
            f"{comp['actual']:>6}  {comp['predicted']:>4}  {comp['error']:>3}  "
            f"{comp['name'][:36]:<36}  {mark:>5}"
        )

    lines.append("")
    lines.append("PODIUM COMPARISON")
    lines.append(f"{'':>8}  {'Predicted':<36}  {'Actual':<36}")
    lines.append("-" * 88)
    for place in range(1, 4):
        pred_name = next(
            (c["name"] for c in by_pred if c["predicted"] == place),
            "(none)",
        )
        act_name = next(
            (c["name"] for c in metrics.comparisons if c["actual"] == place),
            "(none)",
        )
        match = "✓" if pred_name == act_name else ""
        lines.append(f"  #{place:<3}    {pred_name[:36]:<36}  {act_name[:36]:<36}  {match}")

    lines.append("")
    return "\n".join(lines)


def _write_log(content: str, filename: str, latest_path: Optional[str] = None) -> str:
    _ensure_log_dir()
    filepath = os.path.join(LOG_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as handle:
        handle.write(content)
        if not content.endswith("\n"):
            handle.write("\n")

    if latest_path:
        with open(latest_path, "w", encoding="utf-8") as handle:
            handle.write(content)
            if not content.endswith("\n"):
                handle.write("\n")

    return filepath


def log_division_validation(
    metrics: ValidationMetrics,
    event_name: str = "",
    as_of_date: str = "",
    *,
    print_report: bool = True,
) -> str:
    """Write division comparison log. Returns filepath."""
    report = format_division_comparison(
        metrics,
        event_name=event_name,
        division_name=metrics.division,
        as_of_date=as_of_date,
    )

    if print_report:
        print(report)

    filename = (
        f"division_{_slug(event_name or metrics.event_id)}_"
        f"{_slug(metrics.division)}_{_timestamp()}.log"
    )
    filepath = _write_log(report, filename, latest_path=LATEST_DIVISION_LOG)
    print(f"\nDivision log saved: {filepath}")
    print(f"Latest division log:  {LATEST_DIVISION_LOG}")
    return filepath


def format_validation_run_summary(aggregate: Dict, event_count: int) -> str:
    """Build summary text for a full validation run."""
    lines: List[str] = []
    lines.append("=" * 88)
    lines.append(f"BATCH VALIDATION RUN ({event_count} events requested)")
    lines.append(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("=" * 88)
    lines.append("")
    lines.append(f"Events processed:     {aggregate.get('total_events', 0)}")
    lines.append(f"Divisions validated:  {aggregate.get('total_divisions', 0)}")
    lines.append(
        f"Exact match rate:     {aggregate.get('overall_exact_match_rate', 0):.1%}"
    )
    lines.append(
        f"Winner accuracy:      {aggregate.get('overall_winner_accuracy', 0):.1%}"
    )
    lines.append(
        f"Top 3 accuracy:       {aggregate.get('overall_top3_accuracy', 0):.1%}"
    )
    lines.append(f"Mean error (MAE):     {aggregate.get('overall_mae', 0):.2f}")
    lines.append("")

    division_metrics = aggregate.get("division_metrics") or []
    if division_metrics:
        lines.append("PER-DIVISION SUMMARY")
        lines.append(
            f"{'Event/Division':<50}  {'Exact':>7}  {'Winner':>6}  {'Top3':>6}  {'MAE':>5}"
        )
        lines.append("-" * 88)
        for m in division_metrics:
            label = f"{m.event_id} / {m.division}"[:50]
            lines.append(
                f"{label:<50}  {m.exact_match_rate:>6.1%}  "
                f"{'✓' if m.winner_correct else '✗':>6}  "
                f"{m.top3_accuracy:>5.1%}  {m.mae:>5.2f}"
            )
        lines.append("")

    return "\n".join(lines)


def log_validation_run(aggregate: Dict, event_count: int, *, print_summary: bool = True) -> str:
    """Write batch validation summary log. Returns filepath."""
    report = format_validation_run_summary(aggregate, event_count)

    if print_summary:
        print(report)

    filename = f"run_{_timestamp()}.log"
    filepath = _write_log(report, filename, latest_path=LATEST_RUN_LOG)
    print(f"\nRun log saved:   {filepath}")
    print(f"Latest run log:  {LATEST_RUN_LOG}")
    return filepath
