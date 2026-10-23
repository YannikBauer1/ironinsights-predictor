"""Load the results snapshot and turn it into chronologically ordered contests.

A contest is one division of one event: who stood on stage and how they placed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "results.csv"

# Tier 2 as in the database's qualification rulesets (tier_2): the shows whose winners qualify outright.
TIER2 = {"arnold_classic", "new_york_pro", "pittsburgh_pro", "romania_muscle_fest_pro"}
DIVISIONS = ["mensbb", "classic", "202_212", "mensphysique", "wheelchair", "womensbb", "womensphysique",
             "figure", "wellness", "bikini", "fitness", "fitmodel"]


def show_tier(competition: str, enh: str) -> str:
    """Olympia 'O', tier-2 majors '2', Masters Olympia 'MO', natural 'N', everything else '3'."""
    if enh == "Natural":
        return "N"
    if competition == "olympia":
        return "O"
    if competition == "masters_olympia":
        return "MO"
    if competition in TIER2:
        return "2"
    return "3"


@dataclass
class Contest:
    division_id: str
    event_id: str
    date: np.datetime64
    year: int
    cat: str            # category type: mensbb, classic, bikini, ...
    age: str            # Open, 40+, Masters, ...
    enh: str            # Open or Natural
    comp: str           # competition slug
    tier: str           # O, 2, 3, MO, N
    weight_class: str
    persons: list[str]
    places: np.ndarray  # float, nan = no place (DNF/DQ)
    prejudge: np.ndarray  # judging_1 (sum of judges' placings, lower is better), nan if unknown
    total: np.ndarray
    names: list[str] = field(default_factory=list)
    countries: list[str] = field(default_factory=list)
    event_country: str = ""
    old_pred_place: np.ndarray | None = None   # the v1 batch predictor's pre-show place, when it ran
    old_pred_win: np.ndarray | None = None

    @property
    def pool(self) -> str:
        """Rating pool: natural pros are a separate circuit; masters share the open pool."""
        return f"{self.cat}|{'nat' if self.enh == 'Natural' else 'open'}"

    @property
    def is_open_age(self) -> bool:
        return self.age == "Open"

    @property
    def placed_mask(self) -> np.ndarray:
        return ~np.isnan(self.places)

    @property
    def n_placed(self) -> int:
        return int(self.placed_mask.sum())

    @property
    def winner(self) -> int | None:
        """Entry index of the single winner; None when nobody placed or first place is shared."""
        if self.n_placed < 2:
            return None
        best = np.nanmin(self.places)
        idx = np.flatnonzero(self.places == best)
        return int(idx[0]) if len(idx) == 1 else None


def _impute_dates(df: pd.DataFrame) -> pd.Series:
    """Pre-1997 events have no start date: Olympia in mid September, everything else on 1 June."""
    fallback = pd.to_datetime(df["year"].astype(str) + np.where(df["competition"].eq("olympia"), "-09-15", "-06-01"))
    return df["start_date"].fillna(df["end_date"]).fillna(fallback)


def load_frame(path: Path = DATA) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, parse_dates=["start_date", "end_date"])
    df["date"] = _impute_dates(df)
    df["tier"] = [show_tier(c, e) for c, e in zip(df["competition"], df["enh"])]
    return df


def load_contests(path: Path = DATA) -> list[Contest]:
    df = load_frame(path)
    df = df.sort_values(["date", "event_id", "division_id", "place"], na_position="last")
    contests = []
    for div_id, g in df.groupby("division_id", sort=False):
        r = g.iloc[0]
        # one person twice in a division would be a data error; keep the better row
        g = g.drop_duplicates("person_id", keep="first")
        old_place = g["predicted_place"].to_numpy(float)
        contests.append(Contest(
            division_id=div_id, event_id=r.event_id, date=np.datetime64(r.date, "D"), year=int(r.year),
            cat=r.cat_type, age=r.age, enh=r.enh, comp=r.competition, tier=r.tier,
            weight_class=r.weight_class,
            persons=g["person_id"].tolist(), places=g["place"].to_numpy(float),
            prejudge=g["judging_1"].to_numpy(float), total=g["total"].to_numpy(float),
            names=g["person_name"].tolist(), countries=g["person_country"].fillna("").tolist(),
            event_country=r.event_country if isinstance(r.event_country, str) else "",
            old_pred_place=old_place if np.isfinite(old_place).any() else None,
            old_pred_win=g["predicted_win_percentage"].to_numpy(float) if np.isfinite(old_place).any() else None,
        ))
    contests.sort(key=lambda c: (c.date, c.event_id, c.division_id))
    return contests


def cached_contests(path: Path = DATA) -> list[Contest]:
    """load_contests, memoised in data/contests.pkl until results.csv changes."""
    import pickle
    cache = path.with_name("contests.pkl")
    if cache.exists() and cache.stat().st_mtime > path.stat().st_mtime:
        return pickle.loads(cache.read_bytes())
    cs = load_contests(path)
    cache.write_bytes(pickle.dumps(cs))
    return cs


def default_pool(c: Contest) -> str:
    return c.pool


def masters_split_pool(c: Contest) -> str:
    """Alternative pool that keeps masters classes apart from the Open class of the same division."""
    return f"{c.pool}|{'o' if c.is_open_age else 'm'}"


def contest_weight(c: Contest, tier_w: dict, masters_w: float) -> float:
    """How much a contest counts for a rating: its tier's weight, times masters_w for masters classes."""
    return tier_w.get(c.tier, 1.0) * (1.0 if c.is_open_age else masters_w)
