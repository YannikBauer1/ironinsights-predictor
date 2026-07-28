"""
Division-level prediction entry point.

Used by live prediction jobs and historical validation. The predictor only sees
lineup athlete IDs; all history and head-to-head data is fetched here with an
optional temporal cutoff (as_of_date).
"""

from typing import List, Optional, Tuple

from predictor.data_fetcher import (
    create_athlete_profile,
    fetch_division_event_start_date,
    fetch_division_info,
    fetch_division_lineup_athlete_ids,
    get_supabase_client,
)
from predictor.predictor import AthleteProfile, DivisionInfo, Prediction, predict_event


def predict_division(
    division_id: str,
    as_of_date: Optional[str] = None,
    supabase_client=None,
    include_head_to_head: bool = True,
) -> Tuple[Optional[DivisionInfo], List[Prediction]]:
    """
    Predict placements for one division.

    Args:
        division_id: Division to predict.
        as_of_date: When set (validation), only use data from events strictly
            before this date. When None (live), use all available history.
        supabase_client: Optional Supabase client reuse.
        include_head_to_head: Whether to fetch H2H data (disabled for speed).

    Returns:
        (target_division, predictions)
    """
    supabase = supabase_client or get_supabase_client()
    target_division = fetch_division_info(division_id)
    if not target_division:
        return None, []

    for_validation = as_of_date is not None
    athlete_ids = fetch_division_lineup_athlete_ids(
        division_id, for_validation=for_validation
    )
    if not athlete_ids:
        return target_division, []

    profiles: List[AthleteProfile] = []
    for athlete_id in athlete_ids:
        profile = create_athlete_profile(athlete_id, before_date=as_of_date)
        if profile:
            profiles.append(profile)

    if not profiles:
        return target_division, []

    h2h_client = supabase if include_head_to_head else None
    predictions = predict_event(
        profiles,
        target_division,
        supabase_client=h2h_client,
        before_date=as_of_date,
    )
    return target_division, predictions


def predict_division_live(division_id: str, supabase_client=None) -> Tuple[Optional[DivisionInfo], List[Prediction]]:
    """Predict an upcoming division (no temporal cutoff)."""
    return predict_division(
        division_id,
        as_of_date=None,
        supabase_client=supabase_client,
        include_head_to_head=True,
    )


def predict_division_as_of(
    division_id: str,
    as_of_date: Optional[str] = None,
    supabase_client=None,
) -> Tuple[Optional[DivisionInfo], List[Prediction]]:
    """
    Predict a historical division as if we only knew data before the event.

    If as_of_date is omitted, the parent event start_date is used.
    """
    cutoff = as_of_date or fetch_division_event_start_date(division_id)
    if not cutoff:
        return None, []
    return predict_division(
        division_id,
        as_of_date=cutoff,
        supabase_client=supabase_client,
        include_head_to_head=True,
    )
