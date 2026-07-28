"""
Data fetcher module - connect to Supabase and fetch event/athlete data with division awareness.
"""

import os
from datetime import datetime, timezone
from typing import List, Dict, Optional
import random

from predictor.predictor import DivisionInfo, AthleteProfile, ShowRecord


# Supabase configuration
SUPABASE_URL = os.getenv('SUPABASE_URL', '')
SUPABASE_KEY = os.getenv('SUPABASE_ANON_KEY', '')


def get_supabase_client():
    """Create and return Supabase client."""
    try:
        from supabase import create_client
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except ImportError:
        print("Error: supabase package not installed")
        print("Run: pip install supabase")
        return None


# Cache division metadata within a process (validation hits the same divisions repeatedly)
_division_info_cache: Dict[str, Optional[DivisionInfo]] = {}


def _division_info_from_nested(division_row: Optional[Dict]) -> Optional[DivisionInfo]:
    """Build DivisionInfo from a nested Supabase division/category select."""
    if not division_row:
        return None
    cat = division_row.get('category') or {}
    enhancement = (cat.get('category_enhancement') or {}).get('name', 'Unknown')
    age = (cat.get('category_age') or {}).get('name', 'Unknown')
    cat_type = (cat.get('category_type') or {}).get('name', 'Unknown')
    return DivisionInfo(enhancement, age, cat_type)


def fetch_division_info(division_id: str) -> Optional[DivisionInfo]:
    """Fetch division metadata from Supabase."""
    if division_id in _division_info_cache:
        return _division_info_cache[division_id]

    supabase = get_supabase_client()
    if not supabase:
        return None
    
    div_resp = supabase.table('division').select(
        'id, category:category_id('
        'category_enhancement:category_enhancement_id(name),'
        'category_age:category_age_id(name),'
        'category_type:category_type_id(name)'
        ')'
    ).eq('id', division_id).single().execute()
    
    if not div_resp.data:
        _division_info_cache[division_id] = None
        return None

    info = _division_info_from_nested(div_resp.data)
    _division_info_cache[division_id] = info
    return info


def is_open_validation_division(division_info: Optional[DivisionInfo]) -> bool:
    """
    True for Open enhancement + Open age only.

    Used by validation to exclude masters (40+, etc.) and natural divisions.
    """
    if not division_info:
        return False
    enhancement = division_info.enhancement.strip().lower()
    age = division_info.age_category.strip().lower()
    return enhancement == "open" and age == "open"


def fetch_events_with_results(limit: int = 100) -> List[Dict]:
    """Fetch events that have results/placings."""
    supabase = get_supabase_client()
    if not supabase:
        return []
    
    # Get sample of results to find events
    results_response = supabase.table('result').select('division_id').limit(5000).execute()
    
    if not results_response.data:
        return []
    
    # Get unique division IDs
    division_ids = list(set(r['division_id'] for r in results_response.data if r.get('division_id')))
    
    # Get unique event IDs from those divisions
    event_ids = set()
    for div_id in division_ids[:200]:
        div_resp = supabase.table('division').select('event_id').eq('id', div_id).execute()
        if div_resp.data:
            event_ids.add(div_resp.data[0]['event_id'])
    
    # Fetch event details
    events = []
    for event_id in list(event_ids)[:limit]:
        event_resp = supabase.table('event').select(
            'id, start_date, end_date, year, competition(name)'
        ).eq('id', event_id).single().execute()
        
        if event_resp.data:
            events.append(event_resp.data)
    
    return events


def parse_event_date(raw: Optional[str]) -> Optional[datetime]:
    """Parse event start_date to UTC datetime. Returns None if unparseable."""
    if raw is None or raw == "":
        return None
    s = str(raw).strip()
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s.replace(" ", "T"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass
    try:
        dt = datetime.strptime(s[:10], "%Y-%m-%d")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def resolve_event_cutoff_date(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    year: Optional[int] = None,
) -> Optional[str]:
    """Best available date string for temporal cutoff (start > end > year-end)."""
    if start_date:
        return start_date
    if end_date:
        return end_date
    if year:
        return f"{int(year)}-12-31"
    return None


def is_strictly_before(event_date_raw: Optional[str], cutoff_raw: Optional[str]) -> bool:
    """True when event_date is strictly before cutoff (same day excluded)."""
    event_dt = parse_event_date(event_date_raw)
    cutoff_dt = parse_event_date(cutoff_raw)
    if event_dt is None or cutoff_dt is None:
        return False
    return event_dt.date() < cutoff_dt.date()


def is_event_before_cutoff(
    event_start: Optional[str],
    event_end: Optional[str],
    event_year: Optional[int],
    cutoff_raw: Optional[str],
) -> bool:
    """Compare event timing to cutoff using start/end/year fallbacks."""
    event_cutoff = resolve_event_cutoff_date(event_start, event_end, event_year)
    return is_strictly_before(event_cutoff, cutoff_raw)


def fetch_event_metadata(event_id: str) -> Optional[Dict]:
    """Fetch event id, name, and start_date."""
    supabase = get_supabase_client()
    if not supabase:
        return None

    event_resp = supabase.table('event').select(
        'id, start_date, end_date, year, competition:competition_id(name)'
    ).eq('id', event_id).single().execute()

    if not event_resp.data:
        return None

    data = event_resp.data
    cutoff = resolve_event_cutoff_date(
        data.get('start_date'), data.get('end_date'), data.get('year')
    )
    return {
        'event_id': data['id'],
        'event_name': data.get('competition', {}).get('name', f'Event {event_id}'),
        'start_date': data.get('start_date'),
        'end_date': data.get('end_date'),
        'year': data.get('year'),
        'cutoff_date': cutoff,
    }


def fetch_division_ids_for_event(event_id: str, *, open_only: bool = False) -> List[str]:
    """Return division IDs for an event that have finalized results."""
    supabase = get_supabase_client()
    if not supabase:
        return []

    select = """
id,
category:category_id(
  category_enhancement:category_enhancement_id(name),
  category_age:category_age_id(name)
)
""".strip()
    div_resp = supabase.table('division').select(select).eq('event_id', event_id).execute()

    candidate_ids: List[str] = []
    for div in div_resp.data or []:
        div_id = div.get('id')
        if not div_id:
            continue
        if open_only:
            cat = div.get('category') or {}
            enhancement = (cat.get('category_enhancement') or {}).get('name', 'Unknown')
            age = (cat.get('category_age') or {}).get('name', 'Unknown')
            if not is_open_validation_division(
                DivisionInfo(enhancement, age, 'Unknown')
            ):
                continue
        candidate_ids.append(div_id)

    if not candidate_ids:
        return []

    division_ids: List[str] = []
    batch_size = 50
    for i in range(0, len(candidate_ids), batch_size):
        batch = candidate_ids[i:i + batch_size]
        results = (
            supabase.table('result')
            .select('division_id')
            .in_('division_id', batch)
            .not_.is_('place', 'null')
            .execute()
        )
        with_results = {r['division_id'] for r in (results.data or []) if r.get('division_id')}
        division_ids.extend(div_id for div_id in batch if div_id in with_results)

    return division_ids


def fetch_upcoming_lineup_rows(
    *,
    supabase_client=None,
    created_since_iso: Optional[str] = None,
    created_until_iso: Optional[str] = None,
    created_since_exclusive: bool = False,
) -> List[Dict]:
    """
    Result rows for upcoming events with unfinalized lineups (place IS NULL).

    Optional created_at bounds (ISO timestamps). When created_since_exclusive is
    True, uses strictly greater than created_since_iso (avoids re-processing the
    previous run's upper bound).
    """
    supabase = supabase_client or get_supabase_client()
    if not supabase:
        return []

    select = """
id, division_id, athlete_id, created_at,
division:division_id(
  id,
  event_id,
  category:category_id(
    category_type:category_type_id(name),
    category_enhancement:category_enhancement_id(name),
    category_age:category_age_id(name)
  ),
  event:event_id(start_date, id, competition:competition_id(name))
)
""".strip()

    query = supabase.table("result").select(select).is_("place", "null")
    if created_since_iso:
        if created_since_exclusive:
            query = query.gt("created_at", created_since_iso)
        else:
            query = query.gte("created_at", created_since_iso)
    if created_until_iso:
        query = query.lte("created_at", created_until_iso)
    response = query.execute()
    return response.data or []


def fetch_division_event_start_date(division_id: str) -> Optional[str]:
    """Return the parent event start_date for a division."""
    supabase = get_supabase_client()
    if not supabase:
        return None

    div_resp = supabase.table('division').select('event_id').eq('id', division_id).single().execute()
    if not div_resp.data or not div_resp.data.get('event_id'):
        return None

    event_resp = supabase.table('event').select('start_date, end_date, year').eq(
        'id', div_resp.data['event_id']
    ).single().execute()
    if not event_resp.data:
        return None
    return resolve_event_cutoff_date(
        event_resp.data.get('start_date'),
        event_resp.data.get('end_date'),
        event_resp.data.get('year'),
    )


def fetch_division_lineup_athlete_ids(division_id: str, *, for_validation: bool) -> List[str]:
    """
    Return athlete IDs in a division lineup.

    for_validation=True: athletes with finalized places (historical backtest).
    for_validation=False: athletes entered but not yet placed (live prediction).
    """
    supabase = get_supabase_client()
    if not supabase:
        return []

    query = supabase.table('result').select('athlete_id').eq('division_id', division_id)
    if for_validation:
        query = query.not_.is_('place', 'null')
    else:
        query = query.is_('place', 'null')

    response = query.execute()
    athlete_ids = []
    seen = set()
    for row in response.data or []:
        athlete_id = row.get('athlete_id')
        if athlete_id and athlete_id not in seen:
            seen.add(athlete_id)
            athlete_ids.append(athlete_id)
    return athlete_ids


def fetch_division_actual_results(division_id: str) -> List[Dict]:
    """Fetch finalized places for a division (used only after prediction)."""
    supabase = get_supabase_client()
    if not supabase:
        return []

    results = (
        supabase.table('result')
        .select('athlete_id, place')
        .eq('division_id', division_id)
        .not_.is_('place', 'null')
        .order('place')
        .execute()
    )

    actual = []
    for row in results.data or []:
        if not row.get('athlete_id') or not row.get('place'):
            continue
        athlete_id = row['athlete_id']
        name = 'Unknown'
        ath_resp = supabase.table('athlete').select('person_id').eq('id', athlete_id).single().execute()
        if ath_resp.data and ath_resp.data.get('person_id'):
            person_resp = supabase.table('person').select('name').eq(
                'id', ath_resp.data['person_id']
            ).single().execute()
            if person_resp.data:
                name = person_resp.data.get('name', 'Unknown')
        actual.append({
            'athlete_id': athlete_id,
            'athlete_name': name,
            'place': row['place'],
        })
    return actual


def fetch_random_validation_events(count: int = 20) -> List[Dict]:
    """Pick random past events that have at least one division with results."""
    supabase = get_supabase_client()
    if not supabase:
        print("Error: Could not connect to Supabase")
        return []

    print(f"Selecting {count} random validation events...")

    # Paginate results (Supabase caps at 1000 rows per request)
    division_ids: set = set()
    offset = 0
    page_size = 1000
    min_divisions = max(1000, count * 20)

    while offset < 100_000:
        resp = (
            supabase.table('result')
            .select('division_id')
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = resp.data or []
        if not batch:
            break
        for row in batch:
            if row.get('division_id'):
                division_ids.add(row['division_id'])
        offset += page_size
        if len(batch) < page_size:
            break
        if len(division_ids) >= min_divisions:
            break

    if not division_ids:
        return []

    division_list = list(division_ids)
    random.shuffle(division_list)

    event_ids: set = set()
    batch_size = 100
    for i in range(0, len(division_list), batch_size):
        batch = division_list[i:i + batch_size]
        div_resp = (
            supabase.table('division')
            .select('event_id')
            .in_('id', batch)
            .execute()
        )
        for row in div_resp.data or []:
            if row.get('event_id'):
                event_ids.add(row['event_id'])

    candidate_event_ids = list(event_ids)
    random.shuffle(candidate_event_ids)
    print(f"Candidate pool: {len(candidate_event_ids)} events from {len(division_ids)} divisions")

    events = []
    for event_id in candidate_event_ids:
        if len(events) >= count:
            break
        metadata = fetch_event_metadata(event_id)
        if not metadata or not metadata.get('cutoff_date'):
            continue
        if fetch_division_ids_for_event(event_id, open_only=True):
            events.append(metadata)

    print(f"Found {len(events)} events for validation")
    if len(events) < count:
        print(f"Warning: only {len(events)}/{count} events available in database")
    return events


def fetch_results_for_event(event_id: str) -> List[Dict]:
    """Fetch all results (placings) for a specific event."""
    supabase = get_supabase_client()
    if not supabase:
        return []
    
    # Get divisions for this event
    divisions_response = supabase.table('division').select('id').eq('event_id', event_id).execute()
    division_ids = [d['id'] for d in divisions_response.data]
    
    if not division_ids:
        return []
    
    # Query results for these divisions
    all_results = []
    for div_id in division_ids:
        # Query with proper relationships: result -> athlete -> person
        response = supabase.table('result').select(
            'id, place, athlete_id, division_id'
        ).eq('division_id', div_id).order('place').execute()
        
        for result in response.data:
            # Get athlete name
            if result.get('athlete_id'):
                ath_resp = supabase.table('athlete').select('person_id').eq('id', result['athlete_id']).single().execute()
                if ath_resp.data and ath_resp.data.get('person_id'):
                    person_resp = supabase.table('person').select('name').eq('id', ath_resp.data['person_id']).single().execute()
                    if person_resp.data:
                        result['athlete_name'] = person_resp.data.get('name', 'Unknown')
            
            # Get division info
            result['division_info'] = fetch_division_info(result.get('division_id'))
            all_results.append(result)
    
    return all_results


def fetch_athlete_history(athlete_id: str, before_date: Optional[str] = None) -> Dict:
    """
    Fetch complete history for an athlete with division grouping.

    When before_date is set, only results from events strictly before that date
    are included (same-day events are excluded).
    """
    supabase = get_supabase_client()
    if not supabase:
        return {}
    
    # Get athlete basic info
    athlete_response = supabase.table('athlete').select('id, person_id').eq('id', athlete_id).single().execute()
    
    if not athlete_response.data:
        return {}
    
    athlete = athlete_response.data
    
    # Get person name
    name = 'Unknown'
    if athlete.get('person_id'):
        person_resp = supabase.table('person').select('name').eq('id', athlete['person_id']).single().execute()
        if person_resp.data:
            name = person_resp.data.get('name', 'Unknown')
    
    # Get results with nested event dates and category info for temporal filtering
    results_response = supabase.table('result').select(
        'id, place, division_id, created_at, '
        'division:division_id('
        'event:event_id(start_date, end_date, year, competition:competition_id(name)),'
        'category:category_id('
        'category_enhancement:category_enhancement_id(name),'
        'category_age:category_age_id(name),'
        'category_type:category_type_id(name)'
        ')'
        ')'
    ).eq('athlete_id', athlete_id).order('created_at', desc=True).limit(100).execute()
    
    # Group placings by division characteristics
    placings_by_division: Dict[str, List[int]] = {}
    show_records_by_division: Dict[str, List[Dict]] = {}
    recent_placings: List[int] = []
    last_division_info = None
    last_show_date = None
    division_info_cache: Dict[str, Optional[DivisionInfo]] = {}
    
    for result in results_response.data or []:
        if not result.get('place'):
            continue

        division = result.get('division') or {}
        event = division.get('event') or {}
        if before_date:
            if not is_event_before_cutoff(
                event.get('start_date'),
                event.get('end_date'),
                event.get('year'),
                before_date,
            ):
                continue
        
        div_id = result.get('division_id')
        if div_id not in division_info_cache:
            div_info = _division_info_from_nested(division)
            if div_info is None and div_id:
                div_info = fetch_division_info(div_id)
            division_info_cache[div_id] = div_info
        div_info = division_info_cache.get(div_id)

        competition = event.get('competition') or {}
        event_name = competition.get('name', '') if isinstance(competition, dict) else ''
        
        if div_info:
            # Create key: "Enhancement_AgeCategory"
            key = f"{div_info.enhancement}_{div_info.age_category}"
            
            if key not in placings_by_division:
                placings_by_division[key] = []
                show_records_by_division[key] = []
            
            placings_by_division[key].append(result['place'])
            show_records_by_division[key].append({
                'place': result['place'],
                'event_name': event_name or '',
            })
            
            # Keep most recent as fallback
            if len(recent_placings) < 5:
                recent_placings.append(result['place'])
            
            if not last_division_info:
                last_division_info = div_info
            if last_show_date is None:
                last_show_date = result.get('created_at')
    
    # Calculate totals
    all_placings = []
    for placings in placings_by_division.values():
        all_placings.extend(placings)
    
    total_shows = len(all_placings)
    wins = sum(1 for p in all_placings if p == 1)
    
    return {
        'id': athlete_id,
        'name': name,
        'division_info': last_division_info or DivisionInfo("Unknown", "Unknown", "Unknown"),
        'recent_placings': recent_placings[:5],
        'placings_by_division': placings_by_division,
        'show_records_by_division': show_records_by_division,
        'total_shows': total_shows,
        'wins': wins,
        'last_show_date': last_show_date
    }


def create_athlete_profile(athlete_id: str, before_date: Optional[str] = None) -> Optional[AthleteProfile]:
    """Create a complete AthleteProfile with division-aware history."""
    history = fetch_athlete_history(athlete_id, before_date=before_date)
    
    if not history:
        return None
    
    show_records_by_division = {}
    for key, records in history.get('show_records_by_division', {}).items():
        show_records_by_division[key] = [
            ShowRecord(place=r['place'], event_name=r.get('event_name', ''))
            for r in records
        ]

    return AthleteProfile(
        id=history['id'],
        name=history['name'],
        division=history['division_info'].category_type if history['division_info'] else "Unknown",
        division_info=history['division_info'],
        recent_placings=history['recent_placings'],
        placings_by_division=history['placings_by_division'],
        show_records_by_division=show_records_by_division,
        total_shows=history['total_shows'],
        wins=history['wins'],
        last_show_date=history['last_show_date']
    )


def fetch_event_competitors(event_id: str) -> List[AthleteProfile]:
    """Fetch all competitors for an event with division-aware profiles."""
    supabase = get_supabase_client()
    if not supabase:
        return []
    
    # Get divisions for this event
    divisions_response = supabase.table('division').select('id').eq('event_id', event_id).execute()
    division_ids = [d['id'] for d in divisions_response.data]
    
    if not division_ids:
        return []
    
    # Get unique athletes from results
    athlete_ids = set()
    for div_id in division_ids:
        results = supabase.table('result').select('athlete_id').eq('division_id', div_id).execute()
        for r in results.data:
            if r.get('athlete_id'):
                athlete_ids.add(r['athlete_id'])
    
    # Create profiles for each athlete
    profiles = []
    for athlete_id in athlete_ids:
        profile = create_athlete_profile(athlete_id)
        if profile:
            profiles.append(profile)
    
    return profiles


def fetch_event_division_info(event_id: str) -> Optional[DivisionInfo]:
    """Fetch division info for an event (uses first division)."""
    supabase = get_supabase_client()
    if not supabase:
        return None
    
    divisions_response = supabase.table('division').select('id').eq('event_id', event_id).limit(1).execute()
    
    if divisions_response.data:
        return fetch_division_info(divisions_response.data[0]['id'])
    
    return None


def fetch_test_events(count: int = 100) -> List[Dict]:
    """
    Fetch random events with complete results for testing/validation.
    
    Args:
        count: Number of events to fetch
    
    Returns:
        List of events with competitor data and actual results
    """
    supabase = get_supabase_client()
    if not supabase:
        print("Error: Could not connect to Supabase")
        return []
    
    # Get events that have results
    print(f"Fetching {count} test events with results...")
    
    # Get results with division info
    results_response = supabase.table('result').select('division_id').limit(5000).execute()
    
    if not results_response.data:
        print("No results found")
        return []
    
    # Get unique division IDs
    division_ids = list(set(r['division_id'] for r in results_response.data if r.get('division_id')))
    
    # Get event IDs from divisions
    event_ids = set()
    for div_id in division_ids[:500]:
        div_resp = supabase.table('division').select('event_id').eq('id', div_id).execute()
        if div_resp.data and div_resp.data[0].get('event_id'):
            event_ids.add(div_resp.data[0]['event_id'])
    
    # Convert to list and shuffle for randomness
    event_ids = list(event_ids)
    random.shuffle(event_ids)
    event_ids = event_ids[:count]
    
    print(f"Found {len(event_ids)} events to test")
    
    # Build test events
    test_events = []
    for event_id in event_ids:
        # Get event details
        event_resp = supabase.table('event').select(
            'id, start_date, end_date, year, competition:competition_id(name)'
        ).eq('id', event_id).single().execute()
        
        if not event_resp.data:
            continue
        
        event_data = event_resp.data
        event_name = event_data.get('competition', {}).get('name', f'Event {event_id}')
        
        # Get divisions for this event
        div_resp = supabase.table('division').select('id').eq('event_id', event_id).execute()
        event_division_ids = [d['id'] for d in div_resp.data]
        
        if not event_division_ids:
            continue
        
        # Get division info for the first division (target division)
        target_division = fetch_division_info(event_division_ids[0])
        
        # Fetch competitors with history
        competitors = []
        actual_results = []
        
        for div_id in event_division_ids:
            # Get results for this division
            results = supabase.table('result').select(
                'id, place, athlete_id'
            ).eq('division_id', div_id).order('place').execute()
            
            for result in results.data:
                if not result.get('athlete_id') or not result.get('place'):
                    continue
                
                athlete_id = result['athlete_id']
                
                # Create full athlete profile with history
                profile = create_athlete_profile(athlete_id)
                
                if profile:
                    competitors.append({
                        'id': profile.id,
                        'name': profile.name,
                        'division': profile.division,
                        'division_info': {
                            'enhancement': profile.division_info.enhancement if profile.division_info else 'Unknown',
                            'age_category': profile.division_info.age_category if profile.division_info else 'Unknown',
                            'category_type': profile.division_info.category_type if profile.division_info else 'Unknown'
                        },
                        'recent_placings': profile.recent_placings,
                        'placings_by_division': profile.placings_by_division,
                        'total_shows': profile.total_shows,
                        'wins': profile.wins,
                        'last_show_date': profile.last_show_date
                    })
                    
                    actual_results.append({
                        'athlete_id': athlete_id,
                        'athlete_name': profile.name,
                        'place': result['place']
                    })
        
        if len(competitors) >= 3:  # Only include events with enough competitors
            test_events.append({
                'event_id': event_id,
                'event_name': event_name,
                'division': target_division.category_type if target_division else 'Unknown',
                'division_info': {
                    'enhancement': target_division.enhancement if target_division else 'Unknown',
                    'age_category': target_division.age_category if target_division else 'Unknown'
                },
                'competitors': competitors,
                'actual_results': actual_results
            })
    
    print(f"Successfully prepared {len(test_events)} events for testing")
    return test_events


if __name__ == "__main__":
    print("Division-aware data fetcher loaded")
    print(f"Supabase URL: {SUPABASE_URL[:20]}...")
