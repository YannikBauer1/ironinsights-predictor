#!/usr/bin/env python3
"""
Batch-fetch upcoming competition lineups from Supabase, run ``predict_division_live``
per division (same pipeline as validation), and POST predictions to the
``update_result_predictions`` Edge Function.

By default only Open enhancement + Open age divisions are predicted (masters and
natural divisions are excluded). Use ``--all-divisions`` to include everything.

Uses algorithm weights from ``algorithm_state.json`` (tuned by the auto-improver).

Environment (.env loading)
---------------------------
- Loads ``.env`` from the competition-predictor directory, then optionally the
  workspace parent directory.

Required / optional vars
^^^^^^^^^^^^^^^^^^^^^^
- ``SUPABASE_URL``: required base URL.

- Database client key (first available wins): ``SUPABASE_SERVICE_ROLE_KEY``, else
  ``SUPABASE_ANON_KEY``.

Edge Function Bearer token (JWT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- ``SUPABASE_FUNCTIONS_JWT``: if set, used as the ``Authorization: Bearer …`` token.
- Else ``SUPABASE_ANON_KEY``, else ``SUPABASE_SERVICE_ROLE_KEY``.

Edge Function URL
^^^^^^^^^^^^^^^^^^
- ``UPDATE_RESULT_PREDICTIONS_URL``: full HTTPS URL override.
- Else: ``{SUPABASE_URL.rstrip("/")}/functions/v1/update_result_predictions``.

Each POST includes ``predicted_place``, ``predicted_confidence`` (data-quality 0–1),
``predicted_win_percentage`` (model win chance 0–100), and optional ``tier``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import ssl
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import certifi
from dotenv import load_dotenv

_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_ROOT))

_LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )


def _load_dotenv_files() -> None:
    load_dotenv(_ROOT / ".env")
    parent = _ROOT.parent / ".env"
    if parent.is_file():
        load_dotenv(parent)


def _supabase_keys() -> Tuple[str, str]:
    url = os.environ.get("SUPABASE_URL", "").strip()
    if not url:
        raise RuntimeError("SUPABASE_URL is required")
    db_key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip() or os.environ.get(
        "SUPABASE_ANON_KEY", ""
    ).strip()
    if not db_key:
        raise RuntimeError(
            "Set SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY for database access."
        )
    return url, db_key


def _configure_supabase_env(base_url: str, db_key: str) -> None:
    """Ensure src.data_fetcher uses the same credentials as this script."""
    os.environ["SUPABASE_URL"] = base_url
    os.environ["SUPABASE_ANON_KEY"] = db_key


def _functions_bearer_jwt() -> str:
    explicit = os.environ.get("SUPABASE_FUNCTIONS_JWT", "").strip()
    if explicit:
        return explicit
    anon = os.environ.get("SUPABASE_ANON_KEY", "").strip()
    if anon:
        return anon
    sr = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if sr:
        return sr
    raise RuntimeError(
        "Need SUPABASE_FUNCTIONS_JWT, SUPABASE_ANON_KEY, or SUPABASE_SERVICE_ROLE_KEY "
        "for Edge Function Bearer token."
    )


def _functions_url(base_url: str) -> str:
    custom = os.environ.get("UPDATE_RESULT_PREDICTIONS_URL", "").strip()
    if custom:
        return custom
    return f'{base_url.rstrip("/")}/functions/v1/update_result_predictions'


def _create_supabase(url: str, key: str):
    from supabase import create_client

    return create_client(url, key)


def _load_algorithm_version() -> Tuple[int, Dict[str, float]]:
    state_path = _SCRIPT_DIR / "algorithm_state.json"
    try:
        with open(state_path) as f:
            state = json.load(f)
        return int(state.get("version", 0)), state.get("weights", {})
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0, {}


def _utc_cutoff_iso(minutes: int) -> str:
    dt = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _parse_start_date(raw: Optional[str]):
    if raw is None or raw == "":
        return None
    s = str(raw).strip()
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s.replace(" ", "T"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).date()
    except ValueError:
        pass
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _filter_upcoming(rows: Iterable[Dict[str, Any]], today_utc) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        division = row.get("division") or {}
        ev = division.get("event") or {}
        d = _parse_start_date(ev.get("start_date"))
        if d is None:
            continue
        if d >= today_utc:
            out.append(row)
    return out


def _division_label(division_row: Dict[str, Any]) -> str:
    cat = division_row.get("category") or {}
    ct = cat.get("category_type") or {}
    return ct.get("name") or "Unknown"


def _event_id_from_row(row: Dict[str, Any]) -> Optional[str]:
    division = row.get("division") or {}
    ev = division.get("event") or {}
    return ev.get("id") or division.get("event_id")


def get_competitors_for_division(client, division_id: str) -> List[Dict[str, Any]]:
    result = (
        client.table("result")
        .select(
            """
id, athlete_id, division_id,
athlete:athlete_id(person:person_id(id, name, instagram))
""".strip()
        )
        .eq("division_id", division_id)
        .is_("place", "null")
        .execute()
    )
    return result.data or []


def post_json(url: str, bearer: str, payload: Dict[str, Any], timeout_s: int = 120) -> Tuple[int, str]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(url, data=data, method="POST")
    req.add_header("Authorization", f"Bearer {bearer}")
    req.add_header("Content-Type", "application/json")
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(req, timeout=timeout_s, context=ssl_context) as resp:
        body = resp.read().decode("utf-8", errors="replace")
        return resp.status, body


def win_percentage_for_storage(win_probability: float) -> float:
    """Map model P(1st) in [0,1] to result.predicted_win_percentage (0–100)."""
    clamped = max(0.0, min(1.0, float(win_probability)))
    return round(clamped * 100.0, 2)


def run_batch(
    dry_run: bool,
    minutes: Optional[int],
    *,
    all_upcoming: bool = False,
    since_last_run: bool = False,
    open_only: bool = True,
) -> int:
    _load_dotenv_files()
    _configure_logging()
    run_started_at = datetime.now(timezone.utc)

    try:
        base_url, db_key = _supabase_keys()
        bearer = _functions_bearer_jwt()
        fn_url = _functions_url(base_url)
    except Exception as e:
        _LOGGER.critical("Configuration error: %s", e)
        return 1

    _configure_supabase_env(base_url, db_key)

    version, weights = _load_algorithm_version()
    if version:
        _LOGGER.info(
            "Using algorithm_state.json v%s (recent_form=%.3f, h2h=%.3f, momentum=%.3f)",
            version,
            weights.get("recent_form", 0),
            weights.get("head_to_head", 0),
            weights.get("momentum", 0),
        )
    else:
        _LOGGER.warning("Could not read algorithm_state.json; predictor will use defaults")

    if open_only:
        _LOGGER.info("Division filter: Open enhancement + Open age only (excludes masters/natural)")

    try:
        client = _create_supabase(base_url, db_key)
    except Exception as e:
        _LOGGER.critical("Failed to create Supabase client: %s", e)
        return 1

    today_utc = datetime.now(timezone.utc).date()

    from batch.batch_run_state import (
        load_last_run_at,
        result_created_window,
        save_last_run_at,
    )
    from predictor.data_fetcher import fetch_upcoming_lineup_rows, is_open_validation_division
    from predictor.predict_division import predict_division_live

    created_since: Optional[str] = None
    created_until: Optional[str] = None
    since_exclusive = False
    previous_run = load_last_run_at() if since_last_run else None

    if all_upcoming:
        pass
    elif since_last_run:
        start, end, since_exclusive = result_created_window(
            run_started_at, previous_run
        )
        created_since = start.isoformat()
        created_until = end.isoformat()
        if previous_run:
            _LOGGER.info(
                "Result window: after %s through %s (since last batch run)",
                created_since,
                created_until,
            )
        else:
            _LOGGER.info(
                "No previous batch run — bootstrap window %s through %s",
                created_since,
                created_until,
            )
    else:
        created_since = _utc_cutoff_iso(minutes or 30)

    try:
        lineup_rows = fetch_upcoming_lineup_rows(
            supabase_client=client,
            created_since_iso=created_since,
            created_until_iso=created_until,
            created_since_exclusive=since_exclusive,
        )
    except Exception as e:
        _LOGGER.critical("Failed to fetch upcoming lineups: %s", e)
        return 1

    if all_upcoming:
        _LOGGER.info("Fetched %s unfinalized lineup row(s) (all upcoming)", len(lineup_rows))
    elif since_last_run:
        _LOGGER.info(
            "Fetched %s lineup row(s) in created_at window",
            len(lineup_rows),
        )
    else:
        _LOGGER.info(
            "Fetched %s lineup row(s) with created_at >= %s (last %s min)",
            len(lineup_rows),
            created_since,
            minutes,
        )

    upcoming = _filter_upcoming(lineup_rows, today_utc)
    _LOGGER.info(
        "After start_date >= today (UTC %s): %s row(s)",
        today_utc.isoformat(),
        len(upcoming),
    )

    if not upcoming:
        _LOGGER.info("No lineup rows to predict in this window")
        if not dry_run and since_last_run:
            save_last_run_at(run_started_at)
            _LOGGER.info("Saved last_run_at=%s", run_started_at.isoformat())
        return 0

    by_division: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in upcoming:
        div_id = row.get("division_id")
        if not div_id:
            continue
        by_division[str(div_id)].append(row)

    _LOGGER.info("Division groups to process: %s", len(by_division))

    divisions_failed = 0
    divisions_ok = 0
    divisions_skipped = 0

    for division_id, div_rows in by_division.items():
        try:
            sample = div_rows[0]
            division = sample.get("division") or {}
            division_name = _division_label(division)
            event_id = _event_id_from_row(sample)
            if not event_id:
                _LOGGER.warning("division_id=%s: missing event_id; skipping", division_id)
                divisions_failed += 1
                continue

            if open_only:
                from predictor.data_fetcher import fetch_division_info

                division_info = fetch_division_info(division_id)
                if not is_open_validation_division(division_info):
                    _LOGGER.info(
                        "division_id=%s (%s): skipping — not Open/Open (masters/natural excluded)",
                        division_id,
                        division_name,
                    )
                    divisions_skipped += 1
                    continue

            competitors = get_competitors_for_division(client, division_id)
            result_id_map = {
                str(c["athlete_id"]): str(c["id"])
                for c in competitors
                if c.get("athlete_id") and c.get("id")
            }
            _LOGGER.info(
                "division_id=%s event_id=%s (%s): %s lineup athlete(s)",
                division_id,
                event_id,
                division_name,
                len(result_id_map),
            )

            if not result_id_map:
                _LOGGER.warning("division_id=%s: no lineup athletes; skipping", division_id)
                divisions_failed += 1
                continue

            target_division, predictions = predict_division_live(
                division_id,
                supabase_client=client,
            )
            if not predictions:
                _LOGGER.warning("division_id=%s: predictor returned no results", division_id)
                divisions_failed += 1
                continue

            if target_division:
                _LOGGER.info(
                    "division_id=%s: predicted %s athletes (%s / %s)",
                    division_id,
                    len(predictions),
                    target_division.enhancement,
                    target_division.age_category,
                )

            successful_updates = 0
            failed_updates = 0

            if predictions:
                top = predictions[0]
                _LOGGER.info(
                    "division_id=%s: predicted #1 win chance=%.1f%% (data conf=%.2f)",
                    division_id,
                    top.win_probability * 100,
                    top.confidence,
                )

            for p in predictions:
                result_id = result_id_map.get(str(p.athlete_id))
                if not result_id:
                    _LOGGER.warning(
                        "division_id=%s: no result_id for athlete_id=%s; skipping",
                        division_id,
                        p.athlete_id,
                    )
                    failed_updates += 1
                    continue

                tier_num = None
                if p.tier is not None:
                    try:
                        if isinstance(p.tier, str) and p.tier.startswith("TIER_"):
                            tier_num = int(p.tier.split("_")[1])
                        else:
                            tier_num = int(p.tier)
                    except (ValueError, IndexError):
                        _LOGGER.warning(
                            "Could not parse tier '%s' for athlete_id=%s",
                            p.tier,
                            p.athlete_id,
                        )
                        tier_num = None

                body = {
                    "result_id": result_id,
                    "predicted_place": int(p.predicted_place),
                    "predicted_confidence": float(p.confidence),
                    "predicted_win_percentage": win_percentage_for_storage(p.win_probability),
                    "tier": tier_num,
                }

                if dry_run:
                    _LOGGER.info(
                        "DRY-RUN result_id=%s athlete_id=%s place=%s conf=%.2f win=%.1f%%",
                        result_id,
                        p.athlete_id,
                        body["predicted_place"],
                        body["predicted_confidence"],
                        p.win_probability * 100,
                    )
                    successful_updates += 1
                else:
                    try:
                        status, resp_text = post_json(fn_url, bearer, body)
                        if status >= 400:
                            _LOGGER.error(
                                "result_id=%s athlete_id=%s: HTTP %s — %s",
                                result_id,
                                p.athlete_id,
                                status,
                                resp_text[:200] if len(resp_text) > 200 else resp_text,
                            )
                            failed_updates += 1
                        else:
                            successful_updates += 1
                    except Exception as e:
                        _LOGGER.error(
                            "result_id=%s athlete_id=%s: Error %s",
                            result_id,
                            p.athlete_id,
                            e,
                        )
                        failed_updates += 1

            _LOGGER.info(
                "division_id=%s: %s successful updates, %s failed",
                division_id,
                successful_updates,
                failed_updates,
            )

            if failed_updates == 0:
                divisions_ok += 1
            else:
                divisions_failed += 1
        except HTTPError as e:
            divisions_failed += 1
            try:
                err_body = e.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            _LOGGER.error(
                "division_id=%s: HTTPError %s %s — %s",
                division_id,
                e.code,
                e.reason,
                (err_body[:2000] + "…") if len(err_body) > 2000 else err_body,
            )
        except URLError as e:
            divisions_failed += 1
            _LOGGER.error("division_id=%s: URLError %s", division_id, e)
        except Exception as e:
            divisions_failed += 1
            _LOGGER.exception("division_id=%s: unexpected error: %s", division_id, e)

    _LOGGER.info(
        "Done. divisions_ok=%s divisions_failed=%s divisions_skipped=%s (dry_run=%s, algo_v=%s)",
        divisions_ok,
        divisions_failed,
        divisions_skipped,
        dry_run,
        version or "?",
    )

    if not dry_run and since_last_run:
        save_last_run_at(run_started_at)
        _LOGGER.info("Saved last_run_at=%s", run_started_at.isoformat())

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch predict upcoming divisions and POST to update_result_predictions"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute predictions but do not POST to the Edge Function.",
    )
    parser.add_argument(
        "--minutes",
        type=int,
        default=30,
        help="Only consider lineup rows created in the last N minutes (default: 30).",
    )
    parser.add_argument(
        "--days-ago",
        type=int,
        default=None,
        help="Alternative to --minutes: consider lineups added since N days ago.",
    )
    parser.add_argument(
        "--all-upcoming",
        action="store_true",
        help="Predict every upcoming division with an unfinalized lineup (full refresh).",
    )
    parser.add_argument(
        "--since-last-run",
        action="store_true",
        help="Only result rows created since the last successful batch (state file).",
    )
    parser.add_argument(
        "--all-divisions",
        action="store_true",
        help="Include masters and natural divisions (default: Open enhancement + Open age only).",
    )
    args = parser.parse_args()

    minutes = args.minutes
    if args.days_ago is not None:
        minutes = args.days_ago * 24 * 60

    code = run_batch(
        dry_run=args.dry_run,
        minutes=minutes,
        all_upcoming=args.all_upcoming,
        since_last_run=args.since_last_run,
        open_only=not args.all_divisions,
    )
    raise SystemExit(code)


if __name__ == "__main__":
    main()
