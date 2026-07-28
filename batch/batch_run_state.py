"""Persist last successful competition-predictor batch run (UTC)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Tuple

DEFAULT_STATE_PATH = (
    Path.home() / ".openclaw/workspace/state/competition_predictor_batch.json"
)


def state_file_path() -> Path:
    raw = os.environ.get("COMPETITION_PREDICTOR_BATCH_STATE_PATH", "").strip()
    return Path(raw) if raw else DEFAULT_STATE_PATH


def parse_utc_datetime(value: str) -> datetime:
    s = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def load_last_run_at() -> Optional[datetime]:
    path = state_file_path()
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        ts = raw.get("last_run_at")
        if not isinstance(ts, str) or not ts.strip():
            return None
        return parse_utc_datetime(ts)
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def save_last_run_at(run_started_at: datetime) -> None:
    path = state_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_run_at": run_started_at.astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def result_created_window(
    run_started_at: datetime,
    last_run_at: Optional[datetime],
    *,
    bootstrap_hours: int = 24,
) -> Tuple[datetime, datetime, bool]:
    """Returns (start, end, exclusive_start). end is run_started_at."""
    end = run_started_at.astimezone(timezone.utc)
    if last_run_at is not None:
        start = last_run_at.astimezone(timezone.utc)
        return start, end, True
    start = end - timedelta(hours=bootstrap_hours)
    return start, end, False
