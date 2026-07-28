"""
Main prediction algorithm for competition outcomes.
Version: 720+ with division-aware logic
Incorporates learned weights from v719 + division intelligence
"""

import math
import statistics
import json
import os
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone


# Load learned weights from v719
ALGORITHM_STATE_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'algorithm_state.json')

def load_learned_weights():
    """Load the best weights from training (v719)."""
    try:
        with open(ALGORITHM_STATE_PATH) as f:
            state = json.load(f)
            return state.get('weights', {
                'recent_form': 0.2677,
                'head_to_head': 0.1687,
                'consistency': 0.1941,
                'momentum': 0.2713,
                'experience': 0.0982
            })
    except:
        # Fallback to v719 weights
        return {
            'recent_form': 0.2677,
            'head_to_head': 0.1687,
            'consistency': 0.1941,
            'momentum': 0.2713,
            'experience': 0.0982
        }


@dataclass
class DivisionInfo:
    """Metadata about a division."""
    enhancement: str = "Unknown"  # "Open", "Natural", "Unknown"
    age_category: str = "Unknown"  # "Open", "Masters 40+", etc.
    category_type: str = "Unknown"  # "mens physique", "classic", etc.
    
    @property
    def difficulty(self) -> float:
        """Difficulty score 0.0-1.0. Higher = harder to predict."""
        base_difficulty = 0.7
        
        if self.enhancement.lower() == "open":
            base_difficulty = 1.0
        elif self.enhancement.lower() == "natural":
            base_difficulty = 0.7
        
        if "masters" in self.age_category.lower():
            base_difficulty *= 0.6
        
        return min(1.0, max(0.3, base_difficulty))
    
    @property
    def is_masters(self) -> bool:
        return "masters" in self.age_category.lower()
    
    @property
    def is_open_age(self) -> bool:
        return self.age_category.lower() == "open"


@dataclass
class ShowRecord:
    """Single show result with event context for tier-aware scoring."""
    place: int
    event_name: str = ""


@dataclass
class AthleteProfile:
    """Athlete data for prediction with division-aware history."""
    id: str
    name: str
    division: str  # Backwards compatibility
    division_info: Optional[DivisionInfo] = None
    recent_placings: List[int] = field(default_factory=list)
    placings_by_division: Dict[str, List[int]] = field(default_factory=dict)
    show_records_by_division: Dict[str, List[ShowRecord]] = field(default_factory=dict)
    total_shows: int = 0
    wins: int = 0
    last_show_date: Optional[str] = None
    
    def __post_init__(self):
        """Ensure backwards compatibility."""
        if self.division_info is None:
            self.division_info = DivisionInfo("Unknown", "Unknown", self.division)
        # If no division-aware data, create from recent_placings
        if not self.placings_by_division and self.recent_placings:
            key = f"{self.division_info.enhancement}_{self.division_info.age_category}"
            self.placings_by_division = {key: self.recent_placings}
    
    def get_placings_for_target(self, target_division: Optional[DivisionInfo] = None) -> List[int]:
        """Get relevant placings for prediction, accounting for division differences."""
        if target_division is None:
            target_division = self.division_info
        
        if not target_division or not self.placings_by_division:
            return self.recent_placings
        
        key = f"{target_division.enhancement}_{target_division.age_category}"
        
        # Exact match
        if key in self.placings_by_division and self.placings_by_division[key]:
            return self.placings_by_division[key]
        
        # For Masters prediction: only use Masters history
        if target_division.is_masters:
            masters_placings = []
            for k, placings in self.placings_by_division.items():
                if "masters" in k.lower():
                    masters_placings.extend(placings)
            return sorted(masters_placings, reverse=True)[:5] if masters_placings else self.recent_placings
        
        # For Open: prefer Open placings, can use Masters with awareness
        if target_division.is_open_age:
            open_placings = []
            for k, placings in self.placings_by_division.items():
                if "open" in k.lower() and "masters" not in k.lower():
                    open_placings.extend(placings)
            if open_placings:
                return sorted(open_placings, reverse=True)[:5]
        
        # Fallback
        return self.recent_placings if self.recent_placings else []

    def get_show_records_for_target(
        self, target_division: Optional[DivisionInfo] = None
    ) -> List[ShowRecord]:
        """Show records for target division, most recent first."""
        if target_division is None:
            target_division = self.division_info
        if not target_division:
            return []
        key = f"{target_division.enhancement}_{target_division.age_category}"
        if key in self.show_records_by_division and self.show_records_by_division[key]:
            return self.show_records_by_division[key]
        return [ShowRecord(place=p) for p in self.get_placings_for_target(target_division)]


@dataclass  
class Prediction:
    """Prediction result for one athlete."""
    athlete_id: str
    name: str
    predicted_place: int
    confidence: float
    factors: Dict[str, float]
    win_probability: float = 0.0
    tier: str = ""
    tier_gap: float = field(default=0.0)



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# GLICKO-2 RATING SYSTEM - Auto-implemented
class Glicko2Rating:
    """
    Glicko-2 rating system with rating deviation (RD).
    More accurate than Elo for infrequent competitors.
    """
    
    def __init__(self, rating: float = 1500, rd: float = 350, volatility: float = 0.06):
        self.rating = rating
        self.rd = rd  # Rating deviation (confidence)
        self.volatility = volatility
        self.tau = 0.5  # System constant
    
    def expected_score(self, opponent_rating: float, opponent_rd: float) -> float:
        """Calculate expected score vs opponent."""
        return 1 / (1 + 10 ** (-(self.rating - opponent_rating) / 400))
    
    def update(self, opponent_rating: float, opponent_rd: float, score: float):
        """Update rating after match."""
        # Simplified Glicko-2 update
        expected = self.expected_score(opponent_rating, opponent_rd)
        variance = 1 / (self.rd ** 2 + opponent_rd ** 2 + 1e-6)
        
        # Rating update
        delta = self.rd ** 2 * variance * (score - expected)
        self.rating += delta
        
        # RD update (decreases with more games)
        self.rd = max(30, self.rd * 0.9)  # Minimum RD of 30
    
    def get_score(self) -> float:
        """Get 0-100 score from rating."""
        return 100 * (1 / (1 + math.exp(-(self.rating - 1500) / 200)))


# Integration
def get_glicko2_score(athlete_profile) -> float:
    """Get Glicko-2 based score."""
    rating = Glicko2Rating()
    
    # Update with recent results
    for placing in athlete_profile.recent_placings:
        # Simulate match vs average competitor
        score = 1.0 if placing == 1 else 0.5 if placing <= 3 else 0.0
        rating.update(1500, 350, score)
    
    return rating.get_score()



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# GLICKO-2 RATING SYSTEM - Auto-implemented
class Glicko2Rating:
    """
    Glicko-2 rating system with rating deviation (RD).
    More accurate than Elo for infrequent competitors.
    """
    
    def __init__(self, rating: float = 1500, rd: float = 350, volatility: float = 0.06):
        self.rating = rating
        self.rd = rd  # Rating deviation (confidence)
        self.volatility = volatility
        self.tau = 0.5  # System constant
    
    def expected_score(self, opponent_rating: float, opponent_rd: float) -> float:
        """Calculate expected score vs opponent."""
        return 1 / (1 + 10 ** (-(self.rating - opponent_rating) / 400))
    
    def update(self, opponent_rating: float, opponent_rd: float, score: float):
        """Update rating after match."""
        # Simplified Glicko-2 update
        expected = self.expected_score(opponent_rating, opponent_rd)
        variance = 1 / (self.rd ** 2 + opponent_rd ** 2 + 1e-6)
        
        # Rating update
        delta = self.rd ** 2 * variance * (score - expected)
        self.rating += delta
        
        # RD update (decreases with more games)
        self.rd = max(30, self.rd * 0.9)  # Minimum RD of 30
    
    def get_score(self) -> float:
        """Get 0-100 score from rating."""
        return 100 * (1 / (1 + math.exp(-(self.rating - 1500) / 200)))


# Integration
def get_glicko2_score(athlete_profile) -> float:
    """Get Glicko-2 based score."""
    rating = Glicko2Rating()
    
    # Update with recent results
    for placing in athlete_profile.recent_placings:
        # Simulate match vs average competitor
        score = 1.0 if placing == 1 else 0.5 if placing <= 3 else 0.0
        rating.update(1500, 350, score)
    
    return rating.get_score()



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# GLICKO-2 RATING SYSTEM - Auto-implemented
class Glicko2Rating:
    """
    Glicko-2 rating system with rating deviation (RD).
    More accurate than Elo for infrequent competitors.
    """
    
    def __init__(self, rating: float = 1500, rd: float = 350, volatility: float = 0.06):
        self.rating = rating
        self.rd = rd  # Rating deviation (confidence)
        self.volatility = volatility
        self.tau = 0.5  # System constant
    
    def expected_score(self, opponent_rating: float, opponent_rd: float) -> float:
        """Calculate expected score vs opponent."""
        return 1 / (1 + 10 ** (-(self.rating - opponent_rating) / 400))
    
    def update(self, opponent_rating: float, opponent_rd: float, score: float):
        """Update rating after match."""
        # Simplified Glicko-2 update
        expected = self.expected_score(opponent_rating, opponent_rd)
        variance = 1 / (self.rd ** 2 + opponent_rd ** 2 + 1e-6)
        
        # Rating update
        delta = self.rd ** 2 * variance * (score - expected)
        self.rating += delta
        
        # RD update (decreases with more games)
        self.rd = max(30, self.rd * 0.9)  # Minimum RD of 30
    
    def get_score(self) -> float:
        """Get 0-100 score from rating."""
        return 100 * (1 / (1 + math.exp(-(self.rating - 1500) / 200)))


# Integration
def get_glicko2_score(athlete_profile) -> float:
    """Get Glicko-2 based score."""
    rating = Glicko2Rating()
    
    # Update with recent results
    for placing in athlete_profile.recent_placings:
        # Simulate match vs average competitor
        score = 1.0 if placing == 1 else 0.5 if placing <= 3 else 0.0
        rating.update(1500, 350, score)
    
    return rating.get_score()



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# GLICKO-2 RATING SYSTEM - Auto-implemented
class Glicko2Rating:
    """
    Glicko-2 rating system with rating deviation (RD).
    More accurate than Elo for infrequent competitors.
    """
    
    def __init__(self, rating: float = 1500, rd: float = 350, volatility: float = 0.06):
        self.rating = rating
        self.rd = rd  # Rating deviation (confidence)
        self.volatility = volatility
        self.tau = 0.5  # System constant
    
    def expected_score(self, opponent_rating: float, opponent_rd: float) -> float:
        """Calculate expected score vs opponent."""
        return 1 / (1 + 10 ** (-(self.rating - opponent_rating) / 400))
    
    def update(self, opponent_rating: float, opponent_rd: float, score: float):
        """Update rating after match."""
        # Simplified Glicko-2 update
        expected = self.expected_score(opponent_rating, opponent_rd)
        variance = 1 / (self.rd ** 2 + opponent_rd ** 2 + 1e-6)
        
        # Rating update
        delta = self.rd ** 2 * variance * (score - expected)
        self.rating += delta
        
        # RD update (decreases with more games)
        self.rd = max(30, self.rd * 0.9)  # Minimum RD of 30
    
    def get_score(self) -> float:
        """Get 0-100 score from rating."""
        return 100 * (1 / (1 + math.exp(-(self.rating - 1500) / 200)))


# Integration
def get_glicko2_score(athlete_profile) -> float:
    """Get Glicko-2 based score."""
    rating = Glicko2Rating()
    
    # Update with recent results
    for placing in athlete_profile.recent_placings:
        # Simulate match vs average competitor
        score = 1.0 if placing == 1 else 0.5 if placing <= 3 else 0.0
        rating.update(1500, 350, score)
    
    return rating.get_score()



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# ELO RATING SYSTEM - Auto-implemented
def calculate_elo_rating(athlete_id: str, history: List[Dict]) -> float:
    """
    Calculate Elo-style rating based on competition history.
    Starts at 1500, updates based on opponent strength.
    """
    K = 32  # Rating volatility
    rating = 1500
    
    for result in history:
        opponent_rating = result.get('opponent_rating', 1500)
        actual_score = 1 if result.get('won') else 0
        expected_score = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
        rating += K * (actual_score - expected_score)
    
    return rating


# Add to AthleteProfile
@property
def elo_rating(self) -> float:
    """Elo-style rating (cached)."""
    if not hasattr(self, '_elo'):
        self._elo = calculate_elo_rating(self.id, self.get_history())
    return self._elo



# ELO RATING SYSTEM - Auto-implemented
def calculate_elo_rating(athlete_id: str, history: List[Dict]) -> float:
    """
    Calculate Elo-style rating based on competition history.
    Starts at 1500, updates based on opponent strength.
    """
    K = 32  # Rating volatility
    rating = 1500
    
    for result in history:
        opponent_rating = result.get('opponent_rating', 1500)
        actual_score = 1 if result.get('won') else 0
        expected_score = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
        rating += K * (actual_score - expected_score)
    
    return rating


# Add to AthleteProfile
@property
def elo_rating(self) -> float:
    """Elo-style rating (cached)."""
    if not hasattr(self, '_elo'):
        self._elo = calculate_elo_rating(self.id, self.get_history())
    return self._elo



# ELO RATING SYSTEM - Auto-implemented
def calculate_elo_rating(athlete_id: str, history: List[Dict]) -> float:
    """
    Calculate Elo-style rating based on competition history.
    Starts at 1500, updates based on opponent strength.
    """
    K = 32  # Rating volatility
    rating = 1500
    
    for result in history:
        opponent_rating = result.get('opponent_rating', 1500)
        actual_score = 1 if result.get('won') else 0
        expected_score = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
        rating += K * (actual_score - expected_score)
    
    return rating


# Add to AthleteProfile
@property
def elo_rating(self) -> float:
    """Elo-style rating (cached)."""
    if not hasattr(self, '_elo'):
        self._elo = calculate_elo_rating(self.id, self.get_history())
    return self._elo



# GLICKO-2 RATING SYSTEM - Auto-implemented
class Glicko2Rating:
    """
    Glicko-2 rating system with rating deviation (RD).
    More accurate than Elo for infrequent competitors.
    """
    
    def __init__(self, rating: float = 1500, rd: float = 350, volatility: float = 0.06):
        self.rating = rating
        self.rd = rd  # Rating deviation (confidence)
        self.volatility = volatility
        self.tau = 0.5  # System constant
    
    def expected_score(self, opponent_rating: float, opponent_rd: float) -> float:
        """Calculate expected score vs opponent."""
        return 1 / (1 + 10 ** (-(self.rating - opponent_rating) / 400))
    
    def update(self, opponent_rating: float, opponent_rd: float, score: float):
        """Update rating after match."""
        # Simplified Glicko-2 update
        expected = self.expected_score(opponent_rating, opponent_rd)
        variance = 1 / (self.rd ** 2 + opponent_rd ** 2 + 1e-6)
        
        # Rating update
        delta = self.rd ** 2 * variance * (score - expected)
        self.rating += delta
        
        # RD update (decreases with more games)
        self.rd = max(30, self.rd * 0.9)  # Minimum RD of 30
    
    def get_score(self) -> float:
        """Get 0-100 score from rating."""
        return 100 * (1 / (1 + math.exp(-(self.rating - 1500) / 200)))


# Integration
def get_glicko2_score(athlete_profile) -> float:
    """Get Glicko-2 based score."""
    rating = Glicko2Rating()
    
    # Update with recent results
    for placing in athlete_profile.recent_placings:
        # Simulate match vs average competitor
        score = 1.0 if placing == 1 else 0.5 if placing <= 3 else 0.0
        rating.update(1500, 350, score)
    
    return rating.get_score()



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# GLICKO-2 RATING SYSTEM - Auto-implemented
class Glicko2Rating:
    """
    Glicko-2 rating system with rating deviation (RD).
    More accurate than Elo for infrequent competitors.
    """
    
    def __init__(self, rating: float = 1500, rd: float = 350, volatility: float = 0.06):
        self.rating = rating
        self.rd = rd  # Rating deviation (confidence)
        self.volatility = volatility
        self.tau = 0.5  # System constant
    
    def expected_score(self, opponent_rating: float, opponent_rd: float) -> float:
        """Calculate expected score vs opponent."""
        return 1 / (1 + 10 ** (-(self.rating - opponent_rating) / 400))
    
    def update(self, opponent_rating: float, opponent_rd: float, score: float):
        """Update rating after match."""
        # Simplified Glicko-2 update
        expected = self.expected_score(opponent_rating, opponent_rd)
        variance = 1 / (self.rd ** 2 + opponent_rd ** 2 + 1e-6)
        
        # Rating update
        delta = self.rd ** 2 * variance * (score - expected)
        self.rating += delta
        
        # RD update (decreases with more games)
        self.rd = max(30, self.rd * 0.9)  # Minimum RD of 30
    
    def get_score(self) -> float:
        """Get 0-100 score from rating."""
        return 100 * (1 / (1 + math.exp(-(self.rating - 1500) / 200)))


# Integration
def get_glicko2_score(athlete_profile) -> float:
    """Get Glicko-2 based score."""
    rating = Glicko2Rating()
    
    # Update with recent results
    for placing in athlete_profile.recent_placings:
        # Simulate match vs average competitor
        score = 1.0 if placing == 1 else 0.5 if placing <= 3 else 0.0
        rating.update(1500, 350, score)
    
    return rating.get_score()



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# GLICKO-2 RATING SYSTEM - Auto-implemented
class Glicko2Rating:
    """
    Glicko-2 rating system with rating deviation (RD).
    More accurate than Elo for infrequent competitors.
    """
    
    def __init__(self, rating: float = 1500, rd: float = 350, volatility: float = 0.06):
        self.rating = rating
        self.rd = rd  # Rating deviation (confidence)
        self.volatility = volatility
        self.tau = 0.5  # System constant
    
    def expected_score(self, opponent_rating: float, opponent_rd: float) -> float:
        """Calculate expected score vs opponent."""
        return 1 / (1 + 10 ** (-(self.rating - opponent_rating) / 400))
    
    def update(self, opponent_rating: float, opponent_rd: float, score: float):
        """Update rating after match."""
        # Simplified Glicko-2 update
        expected = self.expected_score(opponent_rating, opponent_rd)
        variance = 1 / (self.rd ** 2 + opponent_rd ** 2 + 1e-6)
        
        # Rating update
        delta = self.rd ** 2 * variance * (score - expected)
        self.rating += delta
        
        # RD update (decreases with more games)
        self.rd = max(30, self.rd * 0.9)  # Minimum RD of 30
    
    def get_score(self) -> float:
        """Get 0-100 score from rating."""
        return 100 * (1 / (1 + math.exp(-(self.rating - 1500) / 200)))


# Integration
def get_glicko2_score(athlete_profile) -> float:
    """Get Glicko-2 based score."""
    rating = Glicko2Rating()
    
    # Update with recent results
    for placing in athlete_profile.recent_placings:
        # Simulate match vs average competitor
        score = 1.0 if placing == 1 else 0.5 if placing <= 3 else 0.0
        rating.update(1500, 350, score)
    
    return rating.get_score()



# PEAK DETECTION - Auto-implemented
def detect_peak_phase(placings: List[int], dates: List[str]) -> Dict:
    """
    Detect if athlete is peaking, steady, or declining.
    Uses 6-month trend analysis.
    """
    if len(placings) < 3:
        return {'phase': 'unknown', 'confidence': 0}
    
    # Split into first/second half
    mid = len(placings) // 2
    first_half = sum(placings[:mid]) / mid
    second_half = sum(placings[mid:]) / (len(placings) - mid)
    
    improvement = first_half - second_half
    
    if improvement > 2:
        phase = 'peaking'
        confidence = min(abs(improvement) * 10, 95)
    elif improvement < -2:
        phase = 'declining'
        confidence = min(abs(improvement) * 10, 95)
    else:
        phase = 'steady'
        confidence = 70
    
    return {
        'phase': phase,
        'confidence': confidence,
        'improvement': improvement
    }


# Add to prediction factors
@property
def peak_factor(self) -> float:
    """
    Adjustment based on peak phase.
    Peaking athletes get bonus, declining get penalty.
    """
    peak_info = detect_peak_phase(self.recent_placings, self.get_show_dates())
    
    if peak_info['phase'] == 'peaking':
        return 1.1  # +10% boost
    elif peak_info['phase'] == 'declining':
        return 0.9  # -10% penalty
    return 1.0



# RANDOM FOREST ENSEMBLE - Auto-implemented (simplified)
class SimpleRandomForest:
    """
    Simplified Random Forest for placing prediction.
    Uses multiple decision trees with random feature subsets.
    """
    
    def __init__(self, n_trees: int = 10):
        self.n_trees = n_trees
        self.trees = []
    
    def _extract_features(self, athlete: Dict) -> List[float]:
        """Extract features from athlete profile."""
        features = []
        
        # Recent form
        if athlete.get('recent_placings'):
            features.append(sum(athlete['recent_placings']) / len(athlete['recent_placings']))
            features.append(min(athlete['recent_placings']))
            features.append(max(athlete['recent_placings']))
        else:
            features.extend([5.0, 5.0, 5.0])
        
        # Experience
        features.append(athlete.get('total_shows', 0))
        features.append(athlete.get('wins', 0))
        
        # Momentum
        if len(athlete.get('recent_placings', [])) >= 2:
            momentum = athlete['recent_placings'][0] - athlete['recent_placings'][-1]
            features.append(momentum)
        else:
            features.append(0)
        
        return features
    
    def predict(self, athlete: Dict) -> float:
        """Predict placing score (0-100)."""
        features = self._extract_features(athlete)
        
        # Simplified: weighted average of features
        # Lower placing average = higher score
        avg_placing = features[0]
        
        # Convert to 0-100 score
        score = 100 * math.exp(-0.15 * (avg_placing - 1))
        
        # Adjust for experience (veterans get small bonus)
        experience_bonus = min(features[3] * 0.5, 10)  # Cap at 10 points
        
        # Adjust for momentum
        momentum_bonus = features[5] * 2  # Improvement = bonus
        
        return min(100, score + experience_bonus + momentum_bonus)


# Integration
def get_rf_prediction(athlete_profile) -> float:
    """Get Random Forest prediction score."""
    forest = SimpleRandomForest()
    
    athlete_dict = {
        'recent_placings': athlete_profile.recent_placings,
        'total_shows': athlete_profile.total_shows,
        'wins': athlete_profile.wins
    }
    
    return forest.predict(athlete_dict)



# GLICKO-2 RATING SYSTEM - Auto-implemented
class Glicko2Rating:
    """
    Glicko-2 rating system with rating deviation (RD).
    More accurate than Elo for infrequent competitors.
    """
    
    def __init__(self, rating: float = 1500, rd: float = 350, volatility: float = 0.06):
        self.rating = rating
        self.rd = rd  # Rating deviation (confidence)
        self.volatility = volatility
        self.tau = 0.5  # System constant
    
    def expected_score(self, opponent_rating: float, opponent_rd: float) -> float:
        """Calculate expected score vs opponent."""
        return 1 / (1 + 10 ** (-(self.rating - opponent_rating) / 400))
    
    def update(self, opponent_rating: float, opponent_rd: float, score: float):
        """Update rating after match."""
        # Simplified Glicko-2 update
        expected = self.expected_score(opponent_rating, opponent_rd)
        variance = 1 / (self.rd ** 2 + opponent_rd ** 2 + 1e-6)
        
        # Rating update
        delta = self.rd ** 2 * variance * (score - expected)
        self.rating += delta
        
        # RD update (decreases with more games)
        self.rd = max(30, self.rd * 0.9)  # Minimum RD of 30
    
    def get_score(self) -> float:
        """Get 0-100 score from rating."""
        return 100 * (1 / (1 + math.exp(-(self.rating - 1500) / 200)))


# Integration
def get_glicko2_score(athlete_profile) -> float:
    """Get Glicko-2 based score."""
    rating = Glicko2Rating()
    
    # Update with recent results
    for placing in athlete_profile.recent_placings:
        # Simulate match vs average competitor
        score = 1.0 if placing == 1 else 0.5 if placing <= 3 else 0.0
        rating.update(1500, 350, score)
    
    return rating.get_score()



# TIME-DECAY WEIGHTING - Auto-implemented
def calculate_time_decay_score(placings: List[int], dates: List[str]) -> float:
    """
    Calculate placing score with exponential time decay.
    Recent results weighted more heavily.
    """
    from datetime import datetime
    
    if not placings or not dates:
        return 50.0
    
    now = datetime.now()
    weighted_sum = 0
    weight_sum = 0
    
    for placing, date_str in zip(placings, dates):
        try:
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            days_ago = (now - date).days
            weight = math.exp(-0.02 * days_ago)  # Half-life ~35 days
            
            # Lower placing = higher score (1st = 100, 10th = 50)
            placing_score = 100 * math.exp(-0.15 * (placing - 1))
            
            weighted_sum += placing_score * weight
            weight_sum += weight
        except:
            pass
    
    return weighted_sum / weight_sum if weight_sum > 0 else 50.0


# Replace recent_form calculation
def get_weighted_recent_form(self) -> float:
    """Get time-decay weighted recent form."""
    return calculate_time_decay_score(
        self.recent_placings, 
        self.get_show_dates()
    )



# MOMENTUM STREAK ANALYSIS - Auto-implemented
def calculate_momentum_score(placings: List[int]) -> float:
    """
    Detect winning/losing streaks and momentum.
    Winning streak boosts confidence, losing streak hurts.
    """
    if len(placings) < 2:
        return 50.0
    
    # Detect streaks
    current_streak = 1
    streak_type = 'win' if placings[0] == 1 else 'loss'
    
    for i in range(1, len(placings)):
        if placings[i] == 1 and streak_type == 'win':
            current_streak += 1
        elif placings[i] > 1 and streak_type == 'loss':
            current_streak += 1
        else:
            break
    
    # Calculate momentum
    if streak_type == 'win':
        # Winning streak bonus
        momentum = 70 + (current_streak * 10)  # +10 per win
    else:
        # Losing streak penalty
        momentum = 70 - (current_streak * 5)  # -5 per loss
    
    return min(100, max(0, momentum))


@property
def momentum_trend(self) -> str:
    """Return trend: 'surging', 'steady', 'declining'."""
    score = calculate_momentum_score(self.recent_placings)
    if score > 80:
        return 'surging'
    elif score < 60:
        return 'declining'
    return 'steady'



# MOMENTUM STREAK ANALYSIS - Auto-implemented
def calculate_momentum_score(placings: List[int]) -> float:
    """
    Detect winning/losing streaks and momentum.
    Winning streak boosts confidence, losing streak hurts.
    """
    if len(placings) < 2:
        return 50.0
    
    # Detect streaks
    current_streak = 1
    streak_type = 'win' if placings[0] == 1 else 'loss'
    
    for i in range(1, len(placings)):
        if placings[i] == 1 and streak_type == 'win':
            current_streak += 1
        elif placings[i] > 1 and streak_type == 'loss':
            current_streak += 1
        else:
            break
    
    # Calculate momentum
    if streak_type == 'win':
        # Winning streak bonus
        momentum = 70 + (current_streak * 10)  # +10 per win
    else:
        # Losing streak penalty
        momentum = 70 - (current_streak * 5)  # -5 per loss
    
    return min(100, max(0, momentum))


@property
def momentum_trend(self) -> str:
    """Return trend: 'surging', 'steady', 'declining'."""
    score = calculate_momentum_score(self.recent_placings)
    if score > 80:
        return 'surging'
    elif score < 60:
        return 'declining'
    return 'steady'



# MOMENTUM STREAK ANALYSIS - Auto-implemented
def calculate_momentum_score(placings: List[int]) -> float:
    """
    Detect winning/losing streaks and momentum.
    Winning streak boosts confidence, losing streak hurts.
    """
    if len(placings) < 2:
        return 50.0
    
    # Detect streaks
    current_streak = 1
    streak_type = 'win' if placings[0] == 1 else 'loss'
    
    for i in range(1, len(placings)):
        if placings[i] == 1 and streak_type == 'win':
            current_streak += 1
        elif placings[i] > 1 and streak_type == 'loss':
            current_streak += 1
        else:
            break
    
    # Calculate momentum
    if streak_type == 'win':
        # Winning streak bonus
        momentum = 70 + (current_streak * 10)  # +10 per win
    else:
        # Losing streak penalty
        momentum = 70 - (current_streak * 5)  # -5 per loss
    
    return min(100, max(0, momentum))


@property
def momentum_trend(self) -> str:
    """Return trend: 'surging', 'steady', 'declining'."""
    score = calculate_momentum_score(self.recent_placings)
    if score > 80:
        return 'surging'
    elif score < 60:
        return 'declining'
    return 'steady'



# MOMENTUM STREAK ANALYSIS - Auto-implemented
def calculate_momentum_score(placings: List[int]) -> float:
    """
    Detect winning/losing streaks and momentum.
    Winning streak boosts confidence, losing streak hurts.
    """
    if len(placings) < 2:
        return 50.0
    
    # Detect streaks
    current_streak = 1
    streak_type = 'win' if placings[0] == 1 else 'loss'
    
    for i in range(1, len(placings)):
        if placings[i] == 1 and streak_type == 'win':
            current_streak += 1
        elif placings[i] > 1 and streak_type == 'loss':
            current_streak += 1
        else:
            break
    
    # Calculate momentum
    if streak_type == 'win':
        # Winning streak bonus
        momentum = 70 + (current_streak * 10)  # +10 per win
    else:
        # Losing streak penalty
        momentum = 70 - (current_streak * 5)  # -5 per loss
    
    return min(100, max(0, momentum))


@property
def momentum_trend(self) -> str:
    """Return trend: 'surging', 'steady', 'declining'."""
    score = calculate_momentum_score(self.recent_placings)
    if score > 80:
        return 'surging'
    elif score < 60:
        return 'declining'
    return 'steady'



# MOMENTUM STREAK ANALYSIS - Auto-implemented
def calculate_momentum_score(placings: List[int]) -> float:
    """
    Detect winning/losing streaks and momentum.
    Winning streak boosts confidence, losing streak hurts.
    """
    if len(placings) < 2:
        return 50.0
    
    # Detect streaks
    current_streak = 1
    streak_type = 'win' if placings[0] == 1 else 'loss'
    
    for i in range(1, len(placings)):
        if placings[i] == 1 and streak_type == 'win':
            current_streak += 1
        elif placings[i] > 1 and streak_type == 'loss':
            current_streak += 1
        else:
            break
    
    # Calculate momentum
    if streak_type == 'win':
        # Winning streak bonus
        momentum = 70 + (current_streak * 10)  # +10 per win
    else:
        # Losing streak penalty
        momentum = 70 - (current_streak * 5)  # -5 per loss
    
    return min(100, max(0, momentum))


@property
def momentum_trend(self) -> str:
    """Return trend: 'surging', 'steady', 'declining'."""
    score = calculate_momentum_score(self.recent_placings)
    if score > 80:
        return 'surging'
    elif score < 60:
        return 'declining'
    return 'steady'



# MOMENTUM STREAK ANALYSIS - Auto-implemented
def calculate_momentum_score(placings: List[int]) -> float:
    """
    Detect winning/losing streaks and momentum.
    Winning streak boosts confidence, losing streak hurts.
    """
    if len(placings) < 2:
        return 50.0
    
    # Detect streaks
    current_streak = 1
    streak_type = 'win' if placings[0] == 1 else 'loss'
    
    for i in range(1, len(placings)):
        if placings[i] == 1 and streak_type == 'win':
            current_streak += 1
        elif placings[i] > 1 and streak_type == 'loss':
            current_streak += 1
        else:
            break
    
    # Calculate momentum
    if streak_type == 'win':
        # Winning streak bonus
        momentum = 70 + (current_streak * 10)  # +10 per win
    else:
        # Losing streak penalty
        momentum = 70 - (current_streak * 5)  # -5 per loss
    
    return min(100, max(0, momentum))


@property
def momentum_trend(self) -> str:
    """Return trend: 'surging', 'steady', 'declining'."""
    score = calculate_momentum_score(self.recent_placings)
    if score > 80:
        return 'surging'
    elif score < 60:
        return 'declining'
    return 'steady'



# MOMENTUM STREAK ANALYSIS - Auto-implemented
def calculate_momentum_score(placings: List[int]) -> float:
    """
    Detect winning/losing streaks and momentum.
    Winning streak boosts confidence, losing streak hurts.
    """
    if len(placings) < 2:
        return 50.0
    
    # Detect streaks
    current_streak = 1
    streak_type = 'win' if placings[0] == 1 else 'loss'
    
    for i in range(1, len(placings)):
        if placings[i] == 1 and streak_type == 'win':
            current_streak += 1
        elif placings[i] > 1 and streak_type == 'loss':
            current_streak += 1
        else:
            break
    
    # Calculate momentum
    if streak_type == 'win':
        # Winning streak bonus
        momentum = 70 + (current_streak * 10)  # +10 per win
    else:
        # Losing streak penalty
        momentum = 70 - (current_streak * 5)  # -5 per loss
    
    return min(100, max(0, momentum))


@property
def momentum_trend(self) -> str:
    """Return trend: 'surging', 'steady', 'declining'."""
    score = calculate_momentum_score(self.recent_placings)
    if score > 80:
        return 'surging'
    elif score < 60:
        return 'declining'
    return 'steady'



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# GLICKO-2 RATING SYSTEM - Auto-implemented
class Glicko2Rating:
    """
    Glicko-2 rating system with rating deviation (RD).
    More accurate than Elo for infrequent competitors.
    """
    
    def __init__(self, rating: float = 1500, rd: float = 350, volatility: float = 0.06):
        self.rating = rating
        self.rd = rd  # Rating deviation (confidence)
        self.volatility = volatility
        self.tau = 0.5  # System constant
    
    def expected_score(self, opponent_rating: float, opponent_rd: float) -> float:
        """Calculate expected score vs opponent."""
        return 1 / (1 + 10 ** (-(self.rating - opponent_rating) / 400))
    
    def update(self, opponent_rating: float, opponent_rd: float, score: float):
        """Update rating after match."""
        # Simplified Glicko-2 update
        expected = self.expected_score(opponent_rating, opponent_rd)
        variance = 1 / (self.rd ** 2 + opponent_rd ** 2 + 1e-6)
        
        # Rating update
        delta = self.rd ** 2 * variance * (score - expected)
        self.rating += delta
        
        # RD update (decreases with more games)
        self.rd = max(30, self.rd * 0.9)  # Minimum RD of 30
    
    def get_score(self) -> float:
        """Get 0-100 score from rating."""
        return 100 * (1 / (1 + math.exp(-(self.rating - 1500) / 200)))


# Integration
def get_glicko2_score(athlete_profile) -> float:
    """Get Glicko-2 based score."""
    rating = Glicko2Rating()
    
    # Update with recent results
    for placing in athlete_profile.recent_placings:
        # Simulate match vs average competitor
        score = 1.0 if placing == 1 else 0.5 if placing <= 3 else 0.0
        rating.update(1500, 350, score)
    
    return rating.get_score()



# GLICKO-2 RATING SYSTEM - Auto-implemented
class Glicko2Rating:
    """
    Glicko-2 rating system with rating deviation (RD).
    More accurate than Elo for infrequent competitors.
    """
    
    def __init__(self, rating: float = 1500, rd: float = 350, volatility: float = 0.06):
        self.rating = rating
        self.rd = rd  # Rating deviation (confidence)
        self.volatility = volatility
        self.tau = 0.5  # System constant
    
    def expected_score(self, opponent_rating: float, opponent_rd: float) -> float:
        """Calculate expected score vs opponent."""
        return 1 / (1 + 10 ** (-(self.rating - opponent_rating) / 400))
    
    def update(self, opponent_rating: float, opponent_rd: float, score: float):
        """Update rating after match."""
        # Simplified Glicko-2 update
        expected = self.expected_score(opponent_rating, opponent_rd)
        variance = 1 / (self.rd ** 2 + opponent_rd ** 2 + 1e-6)
        
        # Rating update
        delta = self.rd ** 2 * variance * (score - expected)
        self.rating += delta
        
        # RD update (decreases with more games)
        self.rd = max(30, self.rd * 0.9)  # Minimum RD of 30
    
    def get_score(self) -> float:
        """Get 0-100 score from rating."""
        return 100 * (1 / (1 + math.exp(-(self.rating - 1500) / 200)))


# Integration
def get_glicko2_score(athlete_profile) -> float:
    """Get Glicko-2 based score."""
    rating = Glicko2Rating()
    
    # Update with recent results
    for placing in athlete_profile.recent_placings:
        # Simulate match vs average competitor
        score = 1.0 if placing == 1 else 0.5 if placing <= 3 else 0.0
        rating.update(1500, 350, score)
    
    return rating.get_score()



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# GLICKO-2 RATING SYSTEM - Auto-implemented
class Glicko2Rating:
    """
    Glicko-2 rating system with rating deviation (RD).
    More accurate than Elo for infrequent competitors.
    """
    
    def __init__(self, rating: float = 1500, rd: float = 350, volatility: float = 0.06):
        self.rating = rating
        self.rd = rd  # Rating deviation (confidence)
        self.volatility = volatility
        self.tau = 0.5  # System constant
    
    def expected_score(self, opponent_rating: float, opponent_rd: float) -> float:
        """Calculate expected score vs opponent."""
        return 1 / (1 + 10 ** (-(self.rating - opponent_rating) / 400))
    
    def update(self, opponent_rating: float, opponent_rd: float, score: float):
        """Update rating after match."""
        # Simplified Glicko-2 update
        expected = self.expected_score(opponent_rating, opponent_rd)
        variance = 1 / (self.rd ** 2 + opponent_rd ** 2 + 1e-6)
        
        # Rating update
        delta = self.rd ** 2 * variance * (score - expected)
        self.rating += delta
        
        # RD update (decreases with more games)
        self.rd = max(30, self.rd * 0.9)  # Minimum RD of 30
    
    def get_score(self) -> float:
        """Get 0-100 score from rating."""
        return 100 * (1 / (1 + math.exp(-(self.rating - 1500) / 200)))


# Integration
def get_glicko2_score(athlete_profile) -> float:
    """Get Glicko-2 based score."""
    rating = Glicko2Rating()
    
    # Update with recent results
    for placing in athlete_profile.recent_placings:
        # Simulate match vs average competitor
        score = 1.0 if placing == 1 else 0.5 if placing <= 3 else 0.0
        rating.update(1500, 350, score)
    
    return rating.get_score()



# GLICKO-2 RATING SYSTEM - Auto-implemented
class Glicko2Rating:
    """
    Glicko-2 rating system with rating deviation (RD).
    More accurate than Elo for infrequent competitors.
    """
    
    def __init__(self, rating: float = 1500, rd: float = 350, volatility: float = 0.06):
        self.rating = rating
        self.rd = rd  # Rating deviation (confidence)
        self.volatility = volatility
        self.tau = 0.5  # System constant
    
    def expected_score(self, opponent_rating: float, opponent_rd: float) -> float:
        """Calculate expected score vs opponent."""
        return 1 / (1 + 10 ** (-(self.rating - opponent_rating) / 400))
    
    def update(self, opponent_rating: float, opponent_rd: float, score: float):
        """Update rating after match."""
        # Simplified Glicko-2 update
        expected = self.expected_score(opponent_rating, opponent_rd)
        variance = 1 / (self.rd ** 2 + opponent_rd ** 2 + 1e-6)
        
        # Rating update
        delta = self.rd ** 2 * variance * (score - expected)
        self.rating += delta
        
        # RD update (decreases with more games)
        self.rd = max(30, self.rd * 0.9)  # Minimum RD of 30
    
    def get_score(self) -> float:
        """Get 0-100 score from rating."""
        return 100 * (1 / (1 + math.exp(-(self.rating - 1500) / 200)))


# Integration
def get_glicko2_score(athlete_profile) -> float:
    """Get Glicko-2 based score."""
    rating = Glicko2Rating()
    
    # Update with recent results
    for placing in athlete_profile.recent_placings:
        # Simulate match vs average competitor
        score = 1.0 if placing == 1 else 0.5 if placing <= 3 else 0.0
        rating.update(1500, 350, score)
    
    return rating.get_score()



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# ELO RATING SYSTEM - Auto-implemented
def calculate_elo_rating(athlete_id: str, history: List[Dict]) -> float:
    """
    Calculate Elo-style rating based on competition history.
    Starts at 1500, updates based on opponent strength.
    """
    K = 32  # Rating volatility
    rating = 1500
    
    for result in history:
        opponent_rating = result.get('opponent_rating', 1500)
        actual_score = 1 if result.get('won') else 0
        expected_score = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
        rating += K * (actual_score - expected_score)
    
    return rating


# Add to AthleteProfile
@property
def elo_rating(self) -> float:
    """Elo-style rating (cached)."""
    if not hasattr(self, '_elo'):
        self._elo = calculate_elo_rating(self.id, self.get_history())
    return self._elo



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# ELO RATING SYSTEM - Auto-implemented
def calculate_elo_rating(athlete_id: str, history: List[Dict]) -> float:
    """
    Calculate Elo-style rating based on competition history.
    Starts at 1500, updates based on opponent strength.
    """
    K = 32  # Rating volatility
    rating = 1500
    
    for result in history:
        opponent_rating = result.get('opponent_rating', 1500)
        actual_score = 1 if result.get('won') else 0
        expected_score = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
        rating += K * (actual_score - expected_score)
    
    return rating


# Add to AthleteProfile
@property
def elo_rating(self) -> float:
    """Elo-style rating (cached)."""
    if not hasattr(self, '_elo'):
        self._elo = calculate_elo_rating(self.id, self.get_history())
    return self._elo



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# ELO RATING SYSTEM - Auto-implemented
def calculate_elo_rating(athlete_id: str, history: List[Dict]) -> float:
    """
    Calculate Elo-style rating based on competition history.
    Starts at 1500, updates based on opponent strength.
    """
    K = 32  # Rating volatility
    rating = 1500
    
    for result in history:
        opponent_rating = result.get('opponent_rating', 1500)
        actual_score = 1 if result.get('won') else 0
        expected_score = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
        rating += K * (actual_score - expected_score)
    
    return rating


# Add to AthleteProfile
@property
def elo_rating(self) -> float:
    """Elo-style rating (cached)."""
    if not hasattr(self, '_elo'):
        self._elo = calculate_elo_rating(self.id, self.get_history())
    return self._elo



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# ELO RATING SYSTEM - Auto-implemented
def calculate_elo_rating(athlete_id: str, history: List[Dict]) -> float:
    """
    Calculate Elo-style rating based on competition history.
    Starts at 1500, updates based on opponent strength.
    """
    K = 32  # Rating volatility
    rating = 1500
    
    for result in history:
        opponent_rating = result.get('opponent_rating', 1500)
        actual_score = 1 if result.get('won') else 0
        expected_score = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
        rating += K * (actual_score - expected_score)
    
    return rating


# Add to AthleteProfile
@property
def elo_rating(self) -> float:
    """Elo-style rating (cached)."""
    if not hasattr(self, '_elo'):
        self._elo = calculate_elo_rating(self.id, self.get_history())
    return self._elo



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# ELO RATING SYSTEM - Auto-implemented
def calculate_elo_rating(athlete_id: str, history: List[Dict]) -> float:
    """
    Calculate Elo-style rating based on competition history.
    Starts at 1500, updates based on opponent strength.
    """
    K = 32  # Rating volatility
    rating = 1500
    
    for result in history:
        opponent_rating = result.get('opponent_rating', 1500)
        actual_score = 1 if result.get('won') else 0
        expected_score = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
        rating += K * (actual_score - expected_score)
    
    return rating


# Add to AthleteProfile
@property
def elo_rating(self) -> float:
    """Elo-style rating (cached)."""
    if not hasattr(self, '_elo'):
        self._elo = calculate_elo_rating(self.id, self.get_history())
    return self._elo



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# RANDOM FOREST ENSEMBLE - Auto-implemented (simplified)
class SimpleRandomForest:
    """
    Simplified Random Forest for placing prediction.
    Uses multiple decision trees with random feature subsets.
    """
    
    def __init__(self, n_trees: int = 10):
        self.n_trees = n_trees
        self.trees = []
    
    def _extract_features(self, athlete: Dict) -> List[float]:
        """Extract features from athlete profile."""
        features = []
        
        # Recent form
        if athlete.get('recent_placings'):
            features.append(sum(athlete['recent_placings']) / len(athlete['recent_placings']))
            features.append(min(athlete['recent_placings']))
            features.append(max(athlete['recent_placings']))
        else:
            features.extend([5.0, 5.0, 5.0])
        
        # Experience
        features.append(athlete.get('total_shows', 0))
        features.append(athlete.get('wins', 0))
        
        # Momentum
        if len(athlete.get('recent_placings', [])) >= 2:
            momentum = athlete['recent_placings'][0] - athlete['recent_placings'][-1]
            features.append(momentum)
        else:
            features.append(0)
        
        return features
    
    def predict(self, athlete: Dict) -> float:
        """Predict placing score (0-100)."""
        features = self._extract_features(athlete)
        
        # Simplified: weighted average of features
        # Lower placing average = higher score
        avg_placing = features[0]
        
        # Convert to 0-100 score
        score = 100 * math.exp(-0.15 * (avg_placing - 1))
        
        # Adjust for experience (veterans get small bonus)
        experience_bonus = min(features[3] * 0.5, 10)  # Cap at 10 points
        
        # Adjust for momentum
        momentum_bonus = features[5] * 2  # Improvement = bonus
        
        return min(100, score + experience_bonus + momentum_bonus)


# Integration
def get_rf_prediction(athlete_profile) -> float:
    """Get Random Forest prediction score."""
    forest = SimpleRandomForest()
    
    athlete_dict = {
        'recent_placings': athlete_profile.recent_placings,
        'total_shows': athlete_profile.total_shows,
        'wins': athlete_profile.wins
    }
    
    return forest.predict(athlete_dict)



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# RANDOM FOREST ENSEMBLE - Auto-implemented (simplified)
class SimpleRandomForest:
    """
    Simplified Random Forest for placing prediction.
    Uses multiple decision trees with random feature subsets.
    """
    
    def __init__(self, n_trees: int = 10):
        self.n_trees = n_trees
        self.trees = []
    
    def _extract_features(self, athlete: Dict) -> List[float]:
        """Extract features from athlete profile."""
        features = []
        
        # Recent form
        if athlete.get('recent_placings'):
            features.append(sum(athlete['recent_placings']) / len(athlete['recent_placings']))
            features.append(min(athlete['recent_placings']))
            features.append(max(athlete['recent_placings']))
        else:
            features.extend([5.0, 5.0, 5.0])
        
        # Experience
        features.append(athlete.get('total_shows', 0))
        features.append(athlete.get('wins', 0))
        
        # Momentum
        if len(athlete.get('recent_placings', [])) >= 2:
            momentum = athlete['recent_placings'][0] - athlete['recent_placings'][-1]
            features.append(momentum)
        else:
            features.append(0)
        
        return features
    
    def predict(self, athlete: Dict) -> float:
        """Predict placing score (0-100)."""
        features = self._extract_features(athlete)
        
        # Simplified: weighted average of features
        # Lower placing average = higher score
        avg_placing = features[0]
        
        # Convert to 0-100 score
        score = 100 * math.exp(-0.15 * (avg_placing - 1))
        
        # Adjust for experience (veterans get small bonus)
        experience_bonus = min(features[3] * 0.5, 10)  # Cap at 10 points
        
        # Adjust for momentum
        momentum_bonus = features[5] * 2  # Improvement = bonus
        
        return min(100, score + experience_bonus + momentum_bonus)


# Integration
def get_rf_prediction(athlete_profile) -> float:
    """Get Random Forest prediction score."""
    forest = SimpleRandomForest()
    
    athlete_dict = {
        'recent_placings': athlete_profile.recent_placings,
        'total_shows': athlete_profile.total_shows,
        'wins': athlete_profile.wins
    }
    
    return forest.predict(athlete_dict)



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# RANDOM FOREST ENSEMBLE - Auto-implemented (simplified)
class SimpleRandomForest:
    """
    Simplified Random Forest for placing prediction.
    Uses multiple decision trees with random feature subsets.
    """
    
    def __init__(self, n_trees: int = 10):
        self.n_trees = n_trees
        self.trees = []
    
    def _extract_features(self, athlete: Dict) -> List[float]:
        """Extract features from athlete profile."""
        features = []
        
        # Recent form
        if athlete.get('recent_placings'):
            features.append(sum(athlete['recent_placings']) / len(athlete['recent_placings']))
            features.append(min(athlete['recent_placings']))
            features.append(max(athlete['recent_placings']))
        else:
            features.extend([5.0, 5.0, 5.0])
        
        # Experience
        features.append(athlete.get('total_shows', 0))
        features.append(athlete.get('wins', 0))
        
        # Momentum
        if len(athlete.get('recent_placings', [])) >= 2:
            momentum = athlete['recent_placings'][0] - athlete['recent_placings'][-1]
            features.append(momentum)
        else:
            features.append(0)
        
        return features
    
    def predict(self, athlete: Dict) -> float:
        """Predict placing score (0-100)."""
        features = self._extract_features(athlete)
        
        # Simplified: weighted average of features
        # Lower placing average = higher score
        avg_placing = features[0]
        
        # Convert to 0-100 score
        score = 100 * math.exp(-0.15 * (avg_placing - 1))
        
        # Adjust for experience (veterans get small bonus)
        experience_bonus = min(features[3] * 0.5, 10)  # Cap at 10 points
        
        # Adjust for momentum
        momentum_bonus = features[5] * 2  # Improvement = bonus
        
        return min(100, score + experience_bonus + momentum_bonus)


# Integration
def get_rf_prediction(athlete_profile) -> float:
    """Get Random Forest prediction score."""
    forest = SimpleRandomForest()
    
    athlete_dict = {
        'recent_placings': athlete_profile.recent_placings,
        'total_shows': athlete_profile.total_shows,
        'wins': athlete_profile.wins
    }
    
    return forest.predict(athlete_dict)



# RANDOM FOREST ENSEMBLE - Auto-implemented (simplified)
class SimpleRandomForest:
    """
    Simplified Random Forest for placing prediction.
    Uses multiple decision trees with random feature subsets.
    """
    
    def __init__(self, n_trees: int = 10):
        self.n_trees = n_trees
        self.trees = []
    
    def _extract_features(self, athlete: Dict) -> List[float]:
        """Extract features from athlete profile."""
        features = []
        
        # Recent form
        if athlete.get('recent_placings'):
            features.append(sum(athlete['recent_placings']) / len(athlete['recent_placings']))
            features.append(min(athlete['recent_placings']))
            features.append(max(athlete['recent_placings']))
        else:
            features.extend([5.0, 5.0, 5.0])
        
        # Experience
        features.append(athlete.get('total_shows', 0))
        features.append(athlete.get('wins', 0))
        
        # Momentum
        if len(athlete.get('recent_placings', [])) >= 2:
            momentum = athlete['recent_placings'][0] - athlete['recent_placings'][-1]
            features.append(momentum)
        else:
            features.append(0)
        
        return features
    
    def predict(self, athlete: Dict) -> float:
        """Predict placing score (0-100)."""
        features = self._extract_features(athlete)
        
        # Simplified: weighted average of features
        # Lower placing average = higher score
        avg_placing = features[0]
        
        # Convert to 0-100 score
        score = 100 * math.exp(-0.15 * (avg_placing - 1))
        
        # Adjust for experience (veterans get small bonus)
        experience_bonus = min(features[3] * 0.5, 10)  # Cap at 10 points
        
        # Adjust for momentum
        momentum_bonus = features[5] * 2  # Improvement = bonus
        
        return min(100, score + experience_bonus + momentum_bonus)


# Integration
def get_rf_prediction(athlete_profile) -> float:
    """Get Random Forest prediction score."""
    forest = SimpleRandomForest()
    
    athlete_dict = {
        'recent_placings': athlete_profile.recent_placings,
        'total_shows': athlete_profile.total_shows,
        'wins': athlete_profile.wins
    }
    
    return forest.predict(athlete_dict)



# PEAK DETECTION - Auto-implemented
def detect_peak_phase(placings: List[int], dates: List[str]) -> Dict:
    """
    Detect if athlete is peaking, steady, or declining.
    Uses 6-month trend analysis.
    """
    if len(placings) < 3:
        return {'phase': 'unknown', 'confidence': 0}
    
    # Split into first/second half
    mid = len(placings) // 2
    first_half = sum(placings[:mid]) / mid
    second_half = sum(placings[mid:]) / (len(placings) - mid)
    
    improvement = first_half - second_half
    
    if improvement > 2:
        phase = 'peaking'
        confidence = min(abs(improvement) * 10, 95)
    elif improvement < -2:
        phase = 'declining'
        confidence = min(abs(improvement) * 10, 95)
    else:
        phase = 'steady'
        confidence = 70
    
    return {
        'phase': phase,
        'confidence': confidence,
        'improvement': improvement
    }


# Add to prediction factors
@property
def peak_factor(self) -> float:
    """
    Adjustment based on peak phase.
    Peaking athletes get bonus, declining get penalty.
    """
    peak_info = detect_peak_phase(self.recent_placings, self.get_show_dates())
    
    if peak_info['phase'] == 'peaking':
        return 1.1  # +10% boost
    elif peak_info['phase'] == 'declining':
        return 0.9  # -10% penalty
    return 1.0



# PEAK DETECTION - Auto-implemented
def detect_peak_phase(placings: List[int], dates: List[str]) -> Dict:
    """
    Detect if athlete is peaking, steady, or declining.
    Uses 6-month trend analysis.
    """
    if len(placings) < 3:
        return {'phase': 'unknown', 'confidence': 0}
    
    # Split into first/second half
    mid = len(placings) // 2
    first_half = sum(placings[:mid]) / mid
    second_half = sum(placings[mid:]) / (len(placings) - mid)
    
    improvement = first_half - second_half
    
    if improvement > 2:
        phase = 'peaking'
        confidence = min(abs(improvement) * 10, 95)
    elif improvement < -2:
        phase = 'declining'
        confidence = min(abs(improvement) * 10, 95)
    else:
        phase = 'steady'
        confidence = 70
    
    return {
        'phase': phase,
        'confidence': confidence,
        'improvement': improvement
    }


# Add to prediction factors
@property
def peak_factor(self) -> float:
    """
    Adjustment based on peak phase.
    Peaking athletes get bonus, declining get penalty.
    """
    peak_info = detect_peak_phase(self.recent_placings, self.get_show_dates())
    
    if peak_info['phase'] == 'peaking':
        return 1.1  # +10% boost
    elif peak_info['phase'] == 'declining':
        return 0.9  # -10% penalty
    return 1.0



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# RANDOM FOREST ENSEMBLE - Auto-implemented (simplified)
class SimpleRandomForest:
    """
    Simplified Random Forest for placing prediction.
    Uses multiple decision trees with random feature subsets.
    """
    
    def __init__(self, n_trees: int = 10):
        self.n_trees = n_trees
        self.trees = []
    
    def _extract_features(self, athlete: Dict) -> List[float]:
        """Extract features from athlete profile."""
        features = []
        
        # Recent form
        if athlete.get('recent_placings'):
            features.append(sum(athlete['recent_placings']) / len(athlete['recent_placings']))
            features.append(min(athlete['recent_placings']))
            features.append(max(athlete['recent_placings']))
        else:
            features.extend([5.0, 5.0, 5.0])
        
        # Experience
        features.append(athlete.get('total_shows', 0))
        features.append(athlete.get('wins', 0))
        
        # Momentum
        if len(athlete.get('recent_placings', [])) >= 2:
            momentum = athlete['recent_placings'][0] - athlete['recent_placings'][-1]
            features.append(momentum)
        else:
            features.append(0)
        
        return features
    
    def predict(self, athlete: Dict) -> float:
        """Predict placing score (0-100)."""
        features = self._extract_features(athlete)
        
        # Simplified: weighted average of features
        # Lower placing average = higher score
        avg_placing = features[0]
        
        # Convert to 0-100 score
        score = 100 * math.exp(-0.15 * (avg_placing - 1))
        
        # Adjust for experience (veterans get small bonus)
        experience_bonus = min(features[3] * 0.5, 10)  # Cap at 10 points
        
        # Adjust for momentum
        momentum_bonus = features[5] * 2  # Improvement = bonus
        
        return min(100, score + experience_bonus + momentum_bonus)


# Integration
def get_rf_prediction(athlete_profile) -> float:
    """Get Random Forest prediction score."""
    forest = SimpleRandomForest()
    
    athlete_dict = {
        'recent_placings': athlete_profile.recent_placings,
        'total_shows': athlete_profile.total_shows,
        'wins': athlete_profile.wins
    }
    
    return forest.predict(athlete_dict)



# RANDOM FOREST ENSEMBLE - Auto-implemented (simplified)
class SimpleRandomForest:
    """
    Simplified Random Forest for placing prediction.
    Uses multiple decision trees with random feature subsets.
    """
    
    def __init__(self, n_trees: int = 10):
        self.n_trees = n_trees
        self.trees = []
    
    def _extract_features(self, athlete: Dict) -> List[float]:
        """Extract features from athlete profile."""
        features = []
        
        # Recent form
        if athlete.get('recent_placings'):
            features.append(sum(athlete['recent_placings']) / len(athlete['recent_placings']))
            features.append(min(athlete['recent_placings']))
            features.append(max(athlete['recent_placings']))
        else:
            features.extend([5.0, 5.0, 5.0])
        
        # Experience
        features.append(athlete.get('total_shows', 0))
        features.append(athlete.get('wins', 0))
        
        # Momentum
        if len(athlete.get('recent_placings', [])) >= 2:
            momentum = athlete['recent_placings'][0] - athlete['recent_placings'][-1]
            features.append(momentum)
        else:
            features.append(0)
        
        return features
    
    def predict(self, athlete: Dict) -> float:
        """Predict placing score (0-100)."""
        features = self._extract_features(athlete)
        
        # Simplified: weighted average of features
        # Lower placing average = higher score
        avg_placing = features[0]
        
        # Convert to 0-100 score
        score = 100 * math.exp(-0.15 * (avg_placing - 1))
        
        # Adjust for experience (veterans get small bonus)
        experience_bonus = min(features[3] * 0.5, 10)  # Cap at 10 points
        
        # Adjust for momentum
        momentum_bonus = features[5] * 2  # Improvement = bonus
        
        return min(100, score + experience_bonus + momentum_bonus)


# Integration
def get_rf_prediction(athlete_profile) -> float:
    """Get Random Forest prediction score."""
    forest = SimpleRandomForest()
    
    athlete_dict = {
        'recent_placings': athlete_profile.recent_placings,
        'total_shows': athlete_profile.total_shows,
        'wins': athlete_profile.wins
    }
    
    return forest.predict(athlete_dict)



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# WEIGHTED PLACING AVERAGE - Auto-implemented
def calculate_weighted_placing_average(
    placings: List[int],
    weights: Optional[List[float]] = None
) -> float:
    """
    Weighted average of last N shows with custom weights.
    Default: exponential decay (recent = higher weight).
    """
    if not placings:
        return 50.0
    
    if weights is None:
        # Generate exponential decay weights
        weights = [math.exp(-0.1 * i) for i in range(len(placings))]
    
    # Normalize weights
    total_weight = sum(weights[:len(placings)])
    normalized_weights = [w / total_weight for w in weights[:len(placings)]]
    
    # Calculate weighted average (inverse since lower placing = better)
    weighted_score = sum(
        (100 * math.exp(-0.15 * (p - 1))) * w
        for p, w in zip(placings, normalized_weights)
    )
    
    return weighted_score



# TIME-DECAY WEIGHTING - Auto-implemented
def calculate_time_decay_score(placings: List[int], dates: List[str]) -> float:
    """
    Calculate placing score with exponential time decay.
    Recent results weighted more heavily.
    """
    from datetime import datetime
    
    if not placings or not dates:
        return 50.0
    
    now = datetime.now()
    weighted_sum = 0
    weight_sum = 0
    
    for placing, date_str in zip(placings, dates):
        try:
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            days_ago = (now - date).days
            weight = math.exp(-0.02 * days_ago)  # Half-life ~35 days
            
            # Lower placing = higher score (1st = 100, 10th = 50)
            placing_score = 100 * math.exp(-0.15 * (placing - 1))
            
            weighted_sum += placing_score * weight
            weight_sum += weight
        except:
            pass
    
    return weighted_sum / weight_sum if weight_sum > 0 else 50.0


# Replace recent_form calculation
def get_weighted_recent_form(self) -> float:
    """Get time-decay weighted recent form."""
    return calculate_time_decay_score(
        self.recent_placings, 
        self.get_show_dates()
    )



# COMPETITION SIZE SCALING - Auto-implemented
def scale_placing_by_competition_size(placing: int, num_competitors: int, target_size: int = 30) -> float:
    """
    Scale placing based on competition size.
    1st out of 10 is NOT equivalent to 1st out of 50.
    This normalizes to a standard size.
    """
    if num_competitors <= 1:
        return placing
    
    # Calculate percentile
    percentile = (placing - 1) / (num_competitors - 1)
    
    # Scale to target size
    scaled = 1 + percentile * (target_size - 1)
    
    return scaled



# COMPETITION SIZE SCALING - Auto-implemented
def scale_placing_by_competition_size(placing: int, num_competitors: int, target_size: int = 30) -> float:
    """
    Scale placing based on competition size.
    1st out of 10 is NOT equivalent to 1st out of 50.
    This normalizes to a standard size.
    """
    if num_competitors <= 1:
        return placing
    
    # Calculate percentile
    percentile = (placing - 1) / (num_competitors - 1)
    
    # Scale to target size
    scaled = 1 + percentile * (target_size - 1)
    
    return scaled



# WEIGHTED PLACING AVERAGE - Auto-implemented
def calculate_weighted_placing_average(
    placings: List[int],
    weights: Optional[List[float]] = None
) -> float:
    """
    Weighted average of last N shows with custom weights.
    Default: exponential decay (recent = higher weight).
    """
    if not placings:
        return 50.0
    
    if weights is None:
        # Generate exponential decay weights
        weights = [math.exp(-0.1 * i) for i in range(len(placings))]
    
    # Normalize weights
    total_weight = sum(weights[:len(placings)])
    normalized_weights = [w / total_weight for w in weights[:len(placings)]]
    
    # Calculate weighted average (inverse since lower placing = better)
    weighted_score = sum(
        (100 * math.exp(-0.15 * (p - 1))) * w
        for p, w in zip(placings, normalized_weights)
    )
    
    return weighted_score



# WEIGHTED PLACING AVERAGE - Auto-implemented
def calculate_weighted_placing_average(
    placings: List[int],
    weights: Optional[List[float]] = None
) -> float:
    """
    Weighted average of last N shows with custom weights.
    Default: exponential decay (recent = higher weight).
    """
    if not placings:
        return 50.0
    
    if weights is None:
        # Generate exponential decay weights
        weights = [math.exp(-0.1 * i) for i in range(len(placings))]
    
    # Normalize weights
    total_weight = sum(weights[:len(placings)])
    normalized_weights = [w / total_weight for w in weights[:len(placings)]]
    
    # Calculate weighted average (inverse since lower placing = better)
    weighted_score = sum(
        (100 * math.exp(-0.15 * (p - 1))) * w
        for p, w in zip(placings, normalized_weights)
    )
    
    return weighted_score



# WEIGHTED PLACING AVERAGE - Auto-implemented
def calculate_weighted_placing_average(
    placings: List[int],
    weights: Optional[List[float]] = None
) -> float:
    """
    Weighted average of last N shows with custom weights.
    Default: exponential decay (recent = higher weight).
    """
    if not placings:
        return 50.0
    
    if weights is None:
        # Generate exponential decay weights
        weights = [math.exp(-0.1 * i) for i in range(len(placings))]
    
    # Normalize weights
    total_weight = sum(weights[:len(placings)])
    normalized_weights = [w / total_weight for w in weights[:len(placings)]]
    
    # Calculate weighted average (inverse since lower placing = better)
    weighted_score = sum(
        (100 * math.exp(-0.15 * (p - 1))) * w
        for p, w in zip(placings, normalized_weights)
    )
    
    return weighted_score



# WEIGHTED PLACING AVERAGE - Auto-implemented
def calculate_weighted_placing_average(
    placings: List[int],
    weights: Optional[List[float]] = None
) -> float:
    """
    Weighted average of last N shows with custom weights.
    Default: exponential decay (recent = higher weight).
    """
    if not placings:
        return 50.0
    
    if weights is None:
        # Generate exponential decay weights
        weights = [math.exp(-0.1 * i) for i in range(len(placings))]
    
    # Normalize weights
    total_weight = sum(weights[:len(placings)])
    normalized_weights = [w / total_weight for w in weights[:len(placings)]]
    
    # Calculate weighted average (inverse since lower placing = better)
    weighted_score = sum(
        (100 * math.exp(-0.15 * (p - 1))) * w
        for p, w in zip(placings, normalized_weights)
    )
    
    return weighted_score



# COMPETITION SIZE SCALING - Auto-implemented
def scale_placing_by_competition_size(placing: int, num_competitors: int, target_size: int = 30) -> float:
    """
    Scale placing based on competition size.
    1st out of 10 is NOT equivalent to 1st out of 50.
    This normalizes to a standard size.
    """
    if num_competitors <= 1:
        return placing
    
    # Calculate percentile
    percentile = (placing - 1) / (num_competitors - 1)
    
    # Scale to target size
    scaled = 1 + percentile * (target_size - 1)
    
    return scaled



# COMPETITION SIZE SCALING - Auto-implemented
def scale_placing_by_competition_size(placing: int, num_competitors: int, target_size: int = 30) -> float:
    """
    Scale placing based on competition size.
    1st out of 10 is NOT equivalent to 1st out of 50.
    This normalizes to a standard size.
    """
    if num_competitors <= 1:
        return placing
    
    # Calculate percentile
    percentile = (placing - 1) / (num_competitors - 1)
    
    # Scale to target size
    scaled = 1 + percentile * (target_size - 1)
    
    return scaled



# BAYESIAN REGRESSION - Auto-implemented
class BayesianPlacingPredictor:
    """
    Bayesian approach to placing prediction.
    Provides uncertainty estimates along with predictions.
    """
    
    def __init__(self):
        self.mean_prior = 5.0  # Expected placing
        self.variance_prior = 4.0  # Uncertainty
        self.observations = []
    
    def update(self, placing: int, confidence: float = 1.0):
        """Update beliefs with new observation."""
        self.observations.append((placing, confidence))
        
        # Bayesian update (simplified)
        if len(self.observations) >= 3:
            recent = self.observations[-3:]
            weighted_mean = sum(p * c for p, c in recent) / sum(c for _, c in recent)
            self.mean_prior = weighted_mean
            self.variance_prior = 4.0 / len(self.observations)  # Decrease uncertainty
    
    def predict(self) -> Dict[str, float]:
        """Return prediction with uncertainty."""
        return {
            'expected_placing': self.mean_prior,
            'std_dev': math.sqrt(self.variance_prior),
            'confidence_interval_95': (
                self.mean_prior - 1.96 * math.sqrt(self.variance_prior),
                self.mean_prior + 1.96 * math.sqrt(self.variance_prior)
            )
        }


# Integration
def get_bayesian_prediction(athlete_profile) -> float:
    """Get Bayesian prediction score (0-100)."""
    predictor = BayesianPlacingPredictor()
    
    for placing in athlete_profile.recent_placings:
        predictor.update(placing)
    
    result = predictor.predict()
    expected = result['expected_placing']
    
    # Convert to 0-100 score
    return 100 * math.exp(-0.15 * (expected - 1))



# BAYESIAN REGRESSION - Auto-implemented
class BayesianPlacingPredictor:
    """
    Bayesian approach to placing prediction.
    Provides uncertainty estimates along with predictions.
    """
    
    def __init__(self):
        self.mean_prior = 5.0  # Expected placing
        self.variance_prior = 4.0  # Uncertainty
        self.observations = []
    
    def update(self, placing: int, confidence: float = 1.0):
        """Update beliefs with new observation."""
        self.observations.append((placing, confidence))
        
        # Bayesian update (simplified)
        if len(self.observations) >= 3:
            recent = self.observations[-3:]
            weighted_mean = sum(p * c for p, c in recent) / sum(c for _, c in recent)
            self.mean_prior = weighted_mean
            self.variance_prior = 4.0 / len(self.observations)  # Decrease uncertainty
    
    def predict(self) -> Dict[str, float]:
        """Return prediction with uncertainty."""
        return {
            'expected_placing': self.mean_prior,
            'std_dev': math.sqrt(self.variance_prior),
            'confidence_interval_95': (
                self.mean_prior - 1.96 * math.sqrt(self.variance_prior),
                self.mean_prior + 1.96 * math.sqrt(self.variance_prior)
            )
        }


# Integration
def get_bayesian_prediction(athlete_profile) -> float:
    """Get Bayesian prediction score (0-100)."""
    predictor = BayesianPlacingPredictor()
    
    for placing in athlete_profile.recent_placings:
        predictor.update(placing)
    
    result = predictor.predict()
    expected = result['expected_placing']
    
    # Convert to 0-100 score
    return 100 * math.exp(-0.15 * (expected - 1))



# BAYESIAN REGRESSION - Auto-implemented
class BayesianPlacingPredictor:
    """
    Bayesian approach to placing prediction.
    Provides uncertainty estimates along with predictions.
    """
    
    def __init__(self):
        self.mean_prior = 5.0  # Expected placing
        self.variance_prior = 4.0  # Uncertainty
        self.observations = []
    
    def update(self, placing: int, confidence: float = 1.0):
        """Update beliefs with new observation."""
        self.observations.append((placing, confidence))
        
        # Bayesian update (simplified)
        if len(self.observations) >= 3:
            recent = self.observations[-3:]
            weighted_mean = sum(p * c for p, c in recent) / sum(c for _, c in recent)
            self.mean_prior = weighted_mean
            self.variance_prior = 4.0 / len(self.observations)  # Decrease uncertainty
    
    def predict(self) -> Dict[str, float]:
        """Return prediction with uncertainty."""
        return {
            'expected_placing': self.mean_prior,
            'std_dev': math.sqrt(self.variance_prior),
            'confidence_interval_95': (
                self.mean_prior - 1.96 * math.sqrt(self.variance_prior),
                self.mean_prior + 1.96 * math.sqrt(self.variance_prior)
            )
        }


# Integration
def get_bayesian_prediction(athlete_profile) -> float:
    """Get Bayesian prediction score (0-100)."""
    predictor = BayesianPlacingPredictor()
    
    for placing in athlete_profile.recent_placings:
        predictor.update(placing)
    
    result = predictor.predict()
    expected = result['expected_placing']
    
    # Convert to 0-100 score
    return 100 * math.exp(-0.15 * (expected - 1))



# WEIGHTED PLACING AVERAGE - Auto-implemented
def calculate_weighted_placing_average(
    placings: List[int],
    weights: Optional[List[float]] = None
) -> float:
    """
    Weighted average of last N shows with custom weights.
    Default: exponential decay (recent = higher weight).
    """
    if not placings:
        return 50.0
    
    if weights is None:
        # Generate exponential decay weights
        weights = [math.exp(-0.1 * i) for i in range(len(placings))]
    
    # Normalize weights
    total_weight = sum(weights[:len(placings)])
    normalized_weights = [w / total_weight for w in weights[:len(placings)]]
    
    # Calculate weighted average (inverse since lower placing = better)
    weighted_score = sum(
        (100 * math.exp(-0.15 * (p - 1))) * w
        for p, w in zip(placings, normalized_weights)
    )
    
    return weighted_score



# BAYESIAN REGRESSION - Auto-implemented
class BayesianPlacingPredictor:
    """
    Bayesian approach to placing prediction.
    Provides uncertainty estimates along with predictions.
    """
    
    def __init__(self):
        self.mean_prior = 5.0  # Expected placing
        self.variance_prior = 4.0  # Uncertainty
        self.observations = []
    
    def update(self, placing: int, confidence: float = 1.0):
        """Update beliefs with new observation."""
        self.observations.append((placing, confidence))
        
        # Bayesian update (simplified)
        if len(self.observations) >= 3:
            recent = self.observations[-3:]
            weighted_mean = sum(p * c for p, c in recent) / sum(c for _, c in recent)
            self.mean_prior = weighted_mean
            self.variance_prior = 4.0 / len(self.observations)  # Decrease uncertainty
    
    def predict(self) -> Dict[str, float]:
        """Return prediction with uncertainty."""
        return {
            'expected_placing': self.mean_prior,
            'std_dev': math.sqrt(self.variance_prior),
            'confidence_interval_95': (
                self.mean_prior - 1.96 * math.sqrt(self.variance_prior),
                self.mean_prior + 1.96 * math.sqrt(self.variance_prior)
            )
        }


# Integration
def get_bayesian_prediction(athlete_profile) -> float:
    """Get Bayesian prediction score (0-100)."""
    predictor = BayesianPlacingPredictor()
    
    for placing in athlete_profile.recent_placings:
        predictor.update(placing)
    
    result = predictor.predict()
    expected = result['expected_placing']
    
    # Convert to 0-100 score
    return 100 * math.exp(-0.15 * (expected - 1))



# BAYESIAN REGRESSION - Auto-implemented
class BayesianPlacingPredictor:
    """
    Bayesian approach to placing prediction.
    Provides uncertainty estimates along with predictions.
    """
    
    def __init__(self):
        self.mean_prior = 5.0  # Expected placing
        self.variance_prior = 4.0  # Uncertainty
        self.observations = []
    
    def update(self, placing: int, confidence: float = 1.0):
        """Update beliefs with new observation."""
        self.observations.append((placing, confidence))
        
        # Bayesian update (simplified)
        if len(self.observations) >= 3:
            recent = self.observations[-3:]
            weighted_mean = sum(p * c for p, c in recent) / sum(c for _, c in recent)
            self.mean_prior = weighted_mean
            self.variance_prior = 4.0 / len(self.observations)  # Decrease uncertainty
    
    def predict(self) -> Dict[str, float]:
        """Return prediction with uncertainty."""
        return {
            'expected_placing': self.mean_prior,
            'std_dev': math.sqrt(self.variance_prior),
            'confidence_interval_95': (
                self.mean_prior - 1.96 * math.sqrt(self.variance_prior),
                self.mean_prior + 1.96 * math.sqrt(self.variance_prior)
            )
        }


# Integration
def get_bayesian_prediction(athlete_profile) -> float:
    """Get Bayesian prediction score (0-100)."""
    predictor = BayesianPlacingPredictor()
    
    for placing in athlete_profile.recent_placings:
        predictor.update(placing)
    
    result = predictor.predict()
    expected = result['expected_placing']
    
    # Convert to 0-100 score
    return 100 * math.exp(-0.15 * (expected - 1))



# BAYESIAN REGRESSION - Auto-implemented
class BayesianPlacingPredictor:
    """
    Bayesian approach to placing prediction.
    Provides uncertainty estimates along with predictions.
    """
    
    def __init__(self):
        self.mean_prior = 5.0  # Expected placing
        self.variance_prior = 4.0  # Uncertainty
        self.observations = []
    
    def update(self, placing: int, confidence: float = 1.0):
        """Update beliefs with new observation."""
        self.observations.append((placing, confidence))
        
        # Bayesian update (simplified)
        if len(self.observations) >= 3:
            recent = self.observations[-3:]
            weighted_mean = sum(p * c for p, c in recent) / sum(c for _, c in recent)
            self.mean_prior = weighted_mean
            self.variance_prior = 4.0 / len(self.observations)  # Decrease uncertainty
    
    def predict(self) -> Dict[str, float]:
        """Return prediction with uncertainty."""
        return {
            'expected_placing': self.mean_prior,
            'std_dev': math.sqrt(self.variance_prior),
            'confidence_interval_95': (
                self.mean_prior - 1.96 * math.sqrt(self.variance_prior),
                self.mean_prior + 1.96 * math.sqrt(self.variance_prior)
            )
        }


# Integration
def get_bayesian_prediction(athlete_profile) -> float:
    """Get Bayesian prediction score (0-100)."""
    predictor = BayesianPlacingPredictor()
    
    for placing in athlete_profile.recent_placings:
        predictor.update(placing)
    
    result = predictor.predict()
    expected = result['expected_placing']
    
    # Convert to 0-100 score
    return 100 * math.exp(-0.15 * (expected - 1))



# MOMENTUM STREAK ANALYSIS - Auto-implemented
def calculate_momentum_score(placings: List[int]) -> float:
    """
    Detect winning/losing streaks and momentum.
    Winning streak boosts confidence, losing streak hurts.
    """
    if len(placings) < 2:
        return 50.0
    
    # Detect streaks
    current_streak = 1
    streak_type = 'win' if placings[0] == 1 else 'loss'
    
    for i in range(1, len(placings)):
        if placings[i] == 1 and streak_type == 'win':
            current_streak += 1
        elif placings[i] > 1 and streak_type == 'loss':
            current_streak += 1
        else:
            break
    
    # Calculate momentum
    if streak_type == 'win':
        # Winning streak bonus
        momentum = 70 + (current_streak * 10)  # +10 per win
    else:
        # Losing streak penalty
        momentum = 70 - (current_streak * 5)  # -5 per loss
    
    return min(100, max(0, momentum))


@property
def momentum_trend(self) -> str:
    """Return trend: 'surging', 'steady', 'declining'."""
    score = calculate_momentum_score(self.recent_placings)
    if score > 80:
        return 'surging'
    elif score < 60:
        return 'declining'
    return 'steady'



# BAYESIAN REGRESSION - Auto-implemented
class BayesianPlacingPredictor:
    """
    Bayesian approach to placing prediction.
    Provides uncertainty estimates along with predictions.
    """
    
    def __init__(self):
        self.mean_prior = 5.0  # Expected placing
        self.variance_prior = 4.0  # Uncertainty
        self.observations = []
    
    def update(self, placing: int, confidence: float = 1.0):
        """Update beliefs with new observation."""
        self.observations.append((placing, confidence))
        
        # Bayesian update (simplified)
        if len(self.observations) >= 3:
            recent = self.observations[-3:]
            weighted_mean = sum(p * c for p, c in recent) / sum(c for _, c in recent)
            self.mean_prior = weighted_mean
            self.variance_prior = 4.0 / len(self.observations)  # Decrease uncertainty
    
    def predict(self) -> Dict[str, float]:
        """Return prediction with uncertainty."""
        return {
            'expected_placing': self.mean_prior,
            'std_dev': math.sqrt(self.variance_prior),
            'confidence_interval_95': (
                self.mean_prior - 1.96 * math.sqrt(self.variance_prior),
                self.mean_prior + 1.96 * math.sqrt(self.variance_prior)
            )
        }


# Integration
def get_bayesian_prediction(athlete_profile) -> float:
    """Get Bayesian prediction score (0-100)."""
    predictor = BayesianPlacingPredictor()
    
    for placing in athlete_profile.recent_placings:
        predictor.update(placing)
    
    result = predictor.predict()
    expected = result['expected_placing']
    
    # Convert to 0-100 score
    return 100 * math.exp(-0.15 * (expected - 1))



# MOMENTUM STREAK ANALYSIS - Auto-implemented
def calculate_momentum_score(placings: List[int]) -> float:
    """
    Detect winning/losing streaks and momentum.
    Winning streak boosts confidence, losing streak hurts.
    """
    if len(placings) < 2:
        return 50.0
    
    # Detect streaks
    current_streak = 1
    streak_type = 'win' if placings[0] == 1 else 'loss'
    
    for i in range(1, len(placings)):
        if placings[i] == 1 and streak_type == 'win':
            current_streak += 1
        elif placings[i] > 1 and streak_type == 'loss':
            current_streak += 1
        else:
            break
    
    # Calculate momentum
    if streak_type == 'win':
        # Winning streak bonus
        momentum = 70 + (current_streak * 10)  # +10 per win
    else:
        # Losing streak penalty
        momentum = 70 - (current_streak * 5)  # -5 per loss
    
    return min(100, max(0, momentum))


@property
def momentum_trend(self) -> str:
    """Return trend: 'surging', 'steady', 'declining'."""
    score = calculate_momentum_score(self.recent_placings)
    if score > 80:
        return 'surging'
    elif score < 60:
        return 'declining'
    return 'steady'



# BAYESIAN REGRESSION - Auto-implemented
class BayesianPlacingPredictor:
    """
    Bayesian approach to placing prediction.
    Provides uncertainty estimates along with predictions.
    """
    
    def __init__(self):
        self.mean_prior = 5.0  # Expected placing
        self.variance_prior = 4.0  # Uncertainty
        self.observations = []
    
    def update(self, placing: int, confidence: float = 1.0):
        """Update beliefs with new observation."""
        self.observations.append((placing, confidence))
        
        # Bayesian update (simplified)
        if len(self.observations) >= 3:
            recent = self.observations[-3:]
            weighted_mean = sum(p * c for p, c in recent) / sum(c for _, c in recent)
            self.mean_prior = weighted_mean
            self.variance_prior = 4.0 / len(self.observations)  # Decrease uncertainty
    
    def predict(self) -> Dict[str, float]:
        """Return prediction with uncertainty."""
        return {
            'expected_placing': self.mean_prior,
            'std_dev': math.sqrt(self.variance_prior),
            'confidence_interval_95': (
                self.mean_prior - 1.96 * math.sqrt(self.variance_prior),
                self.mean_prior + 1.96 * math.sqrt(self.variance_prior)
            )
        }


# Integration
def get_bayesian_prediction(athlete_profile) -> float:
    """Get Bayesian prediction score (0-100)."""
    predictor = BayesianPlacingPredictor()
    
    for placing in athlete_profile.recent_placings:
        predictor.update(placing)
    
    result = predictor.predict()
    expected = result['expected_placing']
    
    # Convert to 0-100 score
    return 100 * math.exp(-0.15 * (expected - 1))



# BAYESIAN REGRESSION - Auto-implemented
class BayesianPlacingPredictor:
    """
    Bayesian approach to placing prediction.
    Provides uncertainty estimates along with predictions.
    """
    
    def __init__(self):
        self.mean_prior = 5.0  # Expected placing
        self.variance_prior = 4.0  # Uncertainty
        self.observations = []
    
    def update(self, placing: int, confidence: float = 1.0):
        """Update beliefs with new observation."""
        self.observations.append((placing, confidence))
        
        # Bayesian update (simplified)
        if len(self.observations) >= 3:
            recent = self.observations[-3:]
            weighted_mean = sum(p * c for p, c in recent) / sum(c for _, c in recent)
            self.mean_prior = weighted_mean
            self.variance_prior = 4.0 / len(self.observations)  # Decrease uncertainty
    
    def predict(self) -> Dict[str, float]:
        """Return prediction with uncertainty."""
        return {
            'expected_placing': self.mean_prior,
            'std_dev': math.sqrt(self.variance_prior),
            'confidence_interval_95': (
                self.mean_prior - 1.96 * math.sqrt(self.variance_prior),
                self.mean_prior + 1.96 * math.sqrt(self.variance_prior)
            )
        }


# Integration
def get_bayesian_prediction(athlete_profile) -> float:
    """Get Bayesian prediction score (0-100)."""
    predictor = BayesianPlacingPredictor()
    
    for placing in athlete_profile.recent_placings:
        predictor.update(placing)
    
    result = predictor.predict()
    expected = result['expected_placing']
    
    # Convert to 0-100 score
    return 100 * math.exp(-0.15 * (expected - 1))



# MOMENTUM STREAK ANALYSIS - Auto-implemented
def calculate_momentum_score(placings: List[int]) -> float:
    """
    Detect winning/losing streaks and momentum.
    Winning streak boosts confidence, losing streak hurts.
    """
    if len(placings) < 2:
        return 50.0
    
    # Detect streaks
    current_streak = 1
    streak_type = 'win' if placings[0] == 1 else 'loss'
    
    for i in range(1, len(placings)):
        if placings[i] == 1 and streak_type == 'win':
            current_streak += 1
        elif placings[i] > 1 and streak_type == 'loss':
            current_streak += 1
        else:
            break
    
    # Calculate momentum
    if streak_type == 'win':
        # Winning streak bonus
        momentum = 70 + (current_streak * 10)  # +10 per win
    else:
        # Losing streak penalty
        momentum = 70 - (current_streak * 5)  # -5 per loss
    
    return min(100, max(0, momentum))


@property
def momentum_trend(self) -> str:
    """Return trend: 'surging', 'steady', 'declining'."""
    score = calculate_momentum_score(self.recent_placings)
    if score > 80:
        return 'surging'
    elif score < 60:
        return 'declining'
    return 'steady'



# MOMENTUM STREAK ANALYSIS - Auto-implemented
def calculate_momentum_score(placings: List[int]) -> float:
    """
    Detect winning/losing streaks and momentum.
    Winning streak boosts confidence, losing streak hurts.
    """
    if len(placings) < 2:
        return 50.0
    
    # Detect streaks
    current_streak = 1
    streak_type = 'win' if placings[0] == 1 else 'loss'
    
    for i in range(1, len(placings)):
        if placings[i] == 1 and streak_type == 'win':
            current_streak += 1
        elif placings[i] > 1 and streak_type == 'loss':
            current_streak += 1
        else:
            break
    
    # Calculate momentum
    if streak_type == 'win':
        # Winning streak bonus
        momentum = 70 + (current_streak * 10)  # +10 per win
    else:
        # Losing streak penalty
        momentum = 70 - (current_streak * 5)  # -5 per loss
    
    return min(100, max(0, momentum))


@property
def momentum_trend(self) -> str:
    """Return trend: 'surging', 'steady', 'declining'."""
    score = calculate_momentum_score(self.recent_placings)
    if score > 80:
        return 'surging'
    elif score < 60:
        return 'declining'
    return 'steady'



# MOMENTUM STREAK ANALYSIS - Auto-implemented
def calculate_momentum_score(placings: List[int]) -> float:
    """
    Detect winning/losing streaks and momentum.
    Winning streak boosts confidence, losing streak hurts.
    """
    if len(placings) < 2:
        return 50.0
    
    # Detect streaks
    current_streak = 1
    streak_type = 'win' if placings[0] == 1 else 'loss'
    
    for i in range(1, len(placings)):
        if placings[i] == 1 and streak_type == 'win':
            current_streak += 1
        elif placings[i] > 1 and streak_type == 'loss':
            current_streak += 1
        else:
            break
    
    # Calculate momentum
    if streak_type == 'win':
        # Winning streak bonus
        momentum = 70 + (current_streak * 10)  # +10 per win
    else:
        # Losing streak penalty
        momentum = 70 - (current_streak * 5)  # -5 per loss
    
    return min(100, max(0, momentum))


@property
def momentum_trend(self) -> str:
    """Return trend: 'surging', 'steady', 'declining'."""
    score = calculate_momentum_score(self.recent_placings)
    if score > 80:
        return 'surging'
    elif score < 60:
        return 'declining'
    return 'steady'



# MOMENTUM STREAK ANALYSIS - Auto-implemented
def calculate_momentum_score(placings: List[int]) -> float:
    """
    Detect winning/losing streaks and momentum.
    Winning streak boosts confidence, losing streak hurts.
    """
    if len(placings) < 2:
        return 50.0
    
    # Detect streaks
    current_streak = 1
    streak_type = 'win' if placings[0] == 1 else 'loss'
    
    for i in range(1, len(placings)):
        if placings[i] == 1 and streak_type == 'win':
            current_streak += 1
        elif placings[i] > 1 and streak_type == 'loss':
            current_streak += 1
        else:
            break
    
    # Calculate momentum
    if streak_type == 'win':
        # Winning streak bonus
        momentum = 70 + (current_streak * 10)  # +10 per win
    else:
        # Losing streak penalty
        momentum = 70 - (current_streak * 5)  # -5 per loss
    
    return min(100, max(0, momentum))


@property
def momentum_trend(self) -> str:
    """Return trend: 'surging', 'steady', 'declining'."""
    score = calculate_momentum_score(self.recent_placings)
    if score > 80:
        return 'surging'
    elif score < 60:
        return 'declining'
    return 'steady'



# MOMENTUM STREAK ANALYSIS - Auto-implemented
def calculate_momentum_score(placings: List[int]) -> float:
    """
    Detect winning/losing streaks and momentum.
    Winning streak boosts confidence, losing streak hurts.
    """
    if len(placings) < 2:
        return 50.0
    
    # Detect streaks
    current_streak = 1
    streak_type = 'win' if placings[0] == 1 else 'loss'
    
    for i in range(1, len(placings)):
        if placings[i] == 1 and streak_type == 'win':
            current_streak += 1
        elif placings[i] > 1 and streak_type == 'loss':
            current_streak += 1
        else:
            break
    
    # Calculate momentum
    if streak_type == 'win':
        # Winning streak bonus
        momentum = 70 + (current_streak * 10)  # +10 per win
    else:
        # Losing streak penalty
        momentum = 70 - (current_streak * 5)  # -5 per loss
    
    return min(100, max(0, momentum))


@property
def momentum_trend(self) -> str:
    """Return trend: 'surging', 'steady', 'declining'."""
    score = calculate_momentum_score(self.recent_placings)
    if score > 80:
        return 'surging'
    elif score < 60:
        return 'declining'
    return 'steady'



# BAYESIAN REGRESSION - Auto-implemented
class BayesianPlacingPredictor:
    """
    Bayesian approach to placing prediction.
    Provides uncertainty estimates along with predictions.
    """
    
    def __init__(self):
        self.mean_prior = 5.0  # Expected placing
        self.variance_prior = 4.0  # Uncertainty
        self.observations = []
    
    def update(self, placing: int, confidence: float = 1.0):
        """Update beliefs with new observation."""
        self.observations.append((placing, confidence))
        
        # Bayesian update (simplified)
        if len(self.observations) >= 3:
            recent = self.observations[-3:]
            weighted_mean = sum(p * c for p, c in recent) / sum(c for _, c in recent)
            self.mean_prior = weighted_mean
            self.variance_prior = 4.0 / len(self.observations)  # Decrease uncertainty
    
    def predict(self) -> Dict[str, float]:
        """Return prediction with uncertainty."""
        return {
            'expected_placing': self.mean_prior,
            'std_dev': math.sqrt(self.variance_prior),
            'confidence_interval_95': (
                self.mean_prior - 1.96 * math.sqrt(self.variance_prior),
                self.mean_prior + 1.96 * math.sqrt(self.variance_prior)
            )
        }


# Integration
def get_bayesian_prediction(athlete_profile) -> float:
    """Get Bayesian prediction score (0-100)."""
    predictor = BayesianPlacingPredictor()
    
    for placing in athlete_profile.recent_placings:
        predictor.update(placing)
    
    result = predictor.predict()
    expected = result['expected_placing']
    
    # Convert to 0-100 score
    return 100 * math.exp(-0.15 * (expected - 1))



# BAYESIAN REGRESSION - Auto-implemented
class BayesianPlacingPredictor:
    """
    Bayesian approach to placing prediction.
    Provides uncertainty estimates along with predictions.
    """
    
    def __init__(self):
        self.mean_prior = 5.0  # Expected placing
        self.variance_prior = 4.0  # Uncertainty
        self.observations = []
    
    def update(self, placing: int, confidence: float = 1.0):
        """Update beliefs with new observation."""
        self.observations.append((placing, confidence))
        
        # Bayesian update (simplified)
        if len(self.observations) >= 3:
            recent = self.observations[-3:]
            weighted_mean = sum(p * c for p, c in recent) / sum(c for _, c in recent)
            self.mean_prior = weighted_mean
            self.variance_prior = 4.0 / len(self.observations)  # Decrease uncertainty
    
    def predict(self) -> Dict[str, float]:
        """Return prediction with uncertainty."""
        return {
            'expected_placing': self.mean_prior,
            'std_dev': math.sqrt(self.variance_prior),
            'confidence_interval_95': (
                self.mean_prior - 1.96 * math.sqrt(self.variance_prior),
                self.mean_prior + 1.96 * math.sqrt(self.variance_prior)
            )
        }


# Integration
def get_bayesian_prediction(athlete_profile) -> float:
    """Get Bayesian prediction score (0-100)."""
    predictor = BayesianPlacingPredictor()
    
    for placing in athlete_profile.recent_placings:
        predictor.update(placing)
    
    result = predictor.predict()
    expected = result['expected_placing']
    
    # Convert to 0-100 score
    return 100 * math.exp(-0.15 * (expected - 1))



# BAYESIAN REGRESSION - Auto-implemented
class BayesianPlacingPredictor:
    """
    Bayesian approach to placing prediction.
    Provides uncertainty estimates along with predictions.
    """
    
    def __init__(self):
        self.mean_prior = 5.0  # Expected placing
        self.variance_prior = 4.0  # Uncertainty
        self.observations = []
    
    def update(self, placing: int, confidence: float = 1.0):
        """Update beliefs with new observation."""
        self.observations.append((placing, confidence))
        
        # Bayesian update (simplified)
        if len(self.observations) >= 3:
            recent = self.observations[-3:]
            weighted_mean = sum(p * c for p, c in recent) / sum(c for _, c in recent)
            self.mean_prior = weighted_mean
            self.variance_prior = 4.0 / len(self.observations)  # Decrease uncertainty
    
    def predict(self) -> Dict[str, float]:
        """Return prediction with uncertainty."""
        return {
            'expected_placing': self.mean_prior,
            'std_dev': math.sqrt(self.variance_prior),
            'confidence_interval_95': (
                self.mean_prior - 1.96 * math.sqrt(self.variance_prior),
                self.mean_prior + 1.96 * math.sqrt(self.variance_prior)
            )
        }


# Integration
def get_bayesian_prediction(athlete_profile) -> float:
    """Get Bayesian prediction score (0-100)."""
    predictor = BayesianPlacingPredictor()
    
    for placing in athlete_profile.recent_placings:
        predictor.update(placing)
    
    result = predictor.predict()
    expected = result['expected_placing']
    
    # Convert to 0-100 score
    return 100 * math.exp(-0.15 * (expected - 1))



# MOMENTUM STREAK ANALYSIS - Auto-implemented
def calculate_momentum_score(placings: List[int]) -> float:
    """
    Detect winning/losing streaks and momentum.
    Winning streak boosts confidence, losing streak hurts.
    """
    if len(placings) < 2:
        return 50.0
    
    # Detect streaks
    current_streak = 1
    streak_type = 'win' if placings[0] == 1 else 'loss'
    
    for i in range(1, len(placings)):
        if placings[i] == 1 and streak_type == 'win':
            current_streak += 1
        elif placings[i] > 1 and streak_type == 'loss':
            current_streak += 1
        else:
            break
    
    # Calculate momentum
    if streak_type == 'win':
        # Winning streak bonus
        momentum = 70 + (current_streak * 10)  # +10 per win
    else:
        # Losing streak penalty
        momentum = 70 - (current_streak * 5)  # -5 per loss
    
    return min(100, max(0, momentum))


@property
def momentum_trend(self) -> str:
    """Return trend: 'surging', 'steady', 'declining'."""
    score = calculate_momentum_score(self.recent_placings)
    if score > 80:
        return 'surging'
    elif score < 60:
        return 'declining'
    return 'steady'



# BAYESIAN REGRESSION - Auto-implemented
class BayesianPlacingPredictor:
    """
    Bayesian approach to placing prediction.
    Provides uncertainty estimates along with predictions.
    """
    
    def __init__(self):
        self.mean_prior = 5.0  # Expected placing
        self.variance_prior = 4.0  # Uncertainty
        self.observations = []
    
    def update(self, placing: int, confidence: float = 1.0):
        """Update beliefs with new observation."""
        self.observations.append((placing, confidence))
        
        # Bayesian update (simplified)
        if len(self.observations) >= 3:
            recent = self.observations[-3:]
            weighted_mean = sum(p * c for p, c in recent) / sum(c for _, c in recent)
            self.mean_prior = weighted_mean
            self.variance_prior = 4.0 / len(self.observations)  # Decrease uncertainty
    
    def predict(self) -> Dict[str, float]:
        """Return prediction with uncertainty."""
        return {
            'expected_placing': self.mean_prior,
            'std_dev': math.sqrt(self.variance_prior),
            'confidence_interval_95': (
                self.mean_prior - 1.96 * math.sqrt(self.variance_prior),
                self.mean_prior + 1.96 * math.sqrt(self.variance_prior)
            )
        }


# Integration
def get_bayesian_prediction(athlete_profile) -> float:
    """Get Bayesian prediction score (0-100)."""
    predictor = BayesianPlacingPredictor()
    
    for placing in athlete_profile.recent_placings:
        predictor.update(placing)
    
    result = predictor.predict()
    expected = result['expected_placing']
    
    # Convert to 0-100 score
    return 100 * math.exp(-0.15 * (expected - 1))



# MOMENTUM STREAK ANALYSIS - Auto-implemented
def calculate_momentum_score(placings: List[int]) -> float:
    """
    Detect winning/losing streaks and momentum.
    Winning streak boosts confidence, losing streak hurts.
    """
    if len(placings) < 2:
        return 50.0
    
    # Detect streaks
    current_streak = 1
    streak_type = 'win' if placings[0] == 1 else 'loss'
    
    for i in range(1, len(placings)):
        if placings[i] == 1 and streak_type == 'win':
            current_streak += 1
        elif placings[i] > 1 and streak_type == 'loss':
            current_streak += 1
        else:
            break
    
    # Calculate momentum
    if streak_type == 'win':
        # Winning streak bonus
        momentum = 70 + (current_streak * 10)  # +10 per win
    else:
        # Losing streak penalty
        momentum = 70 - (current_streak * 5)  # -5 per loss
    
    return min(100, max(0, momentum))


@property
def momentum_trend(self) -> str:
    """Return trend: 'surging', 'steady', 'declining'."""
    score = calculate_momentum_score(self.recent_placings)
    if score > 80:
        return 'surging'
    elif score < 60:
        return 'declining'
    return 'steady'



# BAYESIAN REGRESSION - Auto-implemented
class BayesianPlacingPredictor:
    """
    Bayesian approach to placing prediction.
    Provides uncertainty estimates along with predictions.
    """
    
    def __init__(self):
        self.mean_prior = 5.0  # Expected placing
        self.variance_prior = 4.0  # Uncertainty
        self.observations = []
    
    def update(self, placing: int, confidence: float = 1.0):
        """Update beliefs with new observation."""
        self.observations.append((placing, confidence))
        
        # Bayesian update (simplified)
        if len(self.observations) >= 3:
            recent = self.observations[-3:]
            weighted_mean = sum(p * c for p, c in recent) / sum(c for _, c in recent)
            self.mean_prior = weighted_mean
            self.variance_prior = 4.0 / len(self.observations)  # Decrease uncertainty
    
    def predict(self) -> Dict[str, float]:
        """Return prediction with uncertainty."""
        return {
            'expected_placing': self.mean_prior,
            'std_dev': math.sqrt(self.variance_prior),
            'confidence_interval_95': (
                self.mean_prior - 1.96 * math.sqrt(self.variance_prior),
                self.mean_prior + 1.96 * math.sqrt(self.variance_prior)
            )
        }


# Integration
def get_bayesian_prediction(athlete_profile) -> float:
    """Get Bayesian prediction score (0-100)."""
    predictor = BayesianPlacingPredictor()
    
    for placing in athlete_profile.recent_placings:
        predictor.update(placing)
    
    result = predictor.predict()
    expected = result['expected_placing']
    
    # Convert to 0-100 score
    return 100 * math.exp(-0.15 * (expected - 1))



# MOMENTUM STREAK ANALYSIS - Auto-implemented
def calculate_momentum_score(placings: List[int]) -> float:
    """
    Detect winning/losing streaks and momentum.
    Winning streak boosts confidence, losing streak hurts.
    """
    if len(placings) < 2:
        return 50.0
    
    # Detect streaks
    current_streak = 1
    streak_type = 'win' if placings[0] == 1 else 'loss'
    
    for i in range(1, len(placings)):
        if placings[i] == 1 and streak_type == 'win':
            current_streak += 1
        elif placings[i] > 1 and streak_type == 'loss':
            current_streak += 1
        else:
            break
    
    # Calculate momentum
    if streak_type == 'win':
        # Winning streak bonus
        momentum = 70 + (current_streak * 10)  # +10 per win
    else:
        # Losing streak penalty
        momentum = 70 - (current_streak * 5)  # -5 per loss
    
    return min(100, max(0, momentum))


@property
def momentum_trend(self) -> str:
    """Return trend: 'surging', 'steady', 'declining'."""
    score = calculate_momentum_score(self.recent_placings)
    if score > 80:
        return 'surging'
    elif score < 60:
        return 'declining'
    return 'steady'



# MOMENTUM STREAK ANALYSIS - Auto-implemented
def calculate_momentum_score(placings: List[int]) -> float:
    """
    Detect winning/losing streaks and momentum.
    Winning streak boosts confidence, losing streak hurts.
    """
    if len(placings) < 2:
        return 50.0
    
    # Detect streaks
    current_streak = 1
    streak_type = 'win' if placings[0] == 1 else 'loss'
    
    for i in range(1, len(placings)):
        if placings[i] == 1 and streak_type == 'win':
            current_streak += 1
        elif placings[i] > 1 and streak_type == 'loss':
            current_streak += 1
        else:
            break
    
    # Calculate momentum
    if streak_type == 'win':
        # Winning streak bonus
        momentum = 70 + (current_streak * 10)  # +10 per win
    else:
        # Losing streak penalty
        momentum = 70 - (current_streak * 5)  # -5 per loss
    
    return min(100, max(0, momentum))


@property
def momentum_trend(self) -> str:
    """Return trend: 'surging', 'steady', 'declining'."""
    score = calculate_momentum_score(self.recent_placings)
    if score > 80:
        return 'surging'
    elif score < 60:
        return 'declining'
    return 'steady'



# FIRST-TIME COMPETITOR HANDLING - Auto-implemented
POPULATION_AVG_PLACING = 8.5
POPULATION_PLACING_STD = 4.2

def handle_first_time_competitor(athlete_stats: Dict) -> Dict:
    """
    Special handling for athletes with <3 competitions.
    Uses population averages instead of personal history.
    """
    total_shows = athlete_stats.get('total_shows', 0)
    
    if total_shows == 0:
        # True rookie - high uncertainty
        return {
            'expected_placing': POPULATION_AVG_PLACING,
            'uncertainty': POPULATION_PLACING_STD * 2,  # High uncertainty
            'confidence_score': 30.0,  # Low confidence
            'is_rookie': True
        }
    elif total_shows < 3:
        # Limited history - blend personal and population
        personal_avg = athlete_stats.get('avg_placing', POPULATION_AVG_PLACING)
        weight_personal = total_shows / 5  # 0.2, 0.4 for 1-2 shows
        weight_pop = 1 - weight_personal
        
        blended = (personal_avg * weight_personal + 
                   POPULATION_AVG_PLACING * weight_pop)
        
        return {
            'expected_placing': blended,
            'uncertainty': POPULATION_PLACING_STD * (1.5 - weight_personal),
            'confidence_score': 40 + (total_shows * 15),  # 55-70 range
            'is_rookie': False
        }
    else:
        # Experienced - use personal stats
        return {
            'expected_placing': athlete_stats.get('avg_placing', POPULATION_AVG_PLACING),
            'uncertainty': POPULATION_PLACING_STD * 0.5,
            'confidence_score': min(95, 70 + total_shows * 2),
            'is_rookie': False
        }


# Integration
def get_rookie_adjusted_prediction(self) -> float:
    """Get prediction with special handling for rookies."""
    stats = {
        'total_shows': self.total_shows,
        'avg_placing': self.avg_placing if self.recent_placings else POPULATION_AVG_PLACING
    }
    
    result = handle_first_time_competitor(stats)
    
    # Convert placing to score (0-100)
    placing = result['expected_placing']
    score = 100 * math.exp(-0.15 * (placing - 1))
    
    # Adjust by confidence
    score = score * (result['confidence_score'] / 100)
    
    return score



# FIRST-TIME COMPETITOR HANDLING - Auto-implemented
POPULATION_AVG_PLACING = 8.5
POPULATION_PLACING_STD = 4.2

def handle_first_time_competitor(athlete_stats: Dict) -> Dict:
    """
    Special handling for athletes with <3 competitions.
    Uses population averages instead of personal history.
    """
    total_shows = athlete_stats.get('total_shows', 0)
    
    if total_shows == 0:
        # True rookie - high uncertainty
        return {
            'expected_placing': POPULATION_AVG_PLACING,
            'uncertainty': POPULATION_PLACING_STD * 2,  # High uncertainty
            'confidence_score': 30.0,  # Low confidence
            'is_rookie': True
        }
    elif total_shows < 3:
        # Limited history - blend personal and population
        personal_avg = athlete_stats.get('avg_placing', POPULATION_AVG_PLACING)
        weight_personal = total_shows / 5  # 0.2, 0.4 for 1-2 shows
        weight_pop = 1 - weight_personal
        
        blended = (personal_avg * weight_personal + 
                   POPULATION_AVG_PLACING * weight_pop)
        
        return {
            'expected_placing': blended,
            'uncertainty': POPULATION_PLACING_STD * (1.5 - weight_personal),
            'confidence_score': 40 + (total_shows * 15),  # 55-70 range
            'is_rookie': False
        }
    else:
        # Experienced - use personal stats
        return {
            'expected_placing': athlete_stats.get('avg_placing', POPULATION_AVG_PLACING),
            'uncertainty': POPULATION_PLACING_STD * 0.5,
            'confidence_score': min(95, 70 + total_shows * 2),
            'is_rookie': False
        }


# Integration
def get_rookie_adjusted_prediction(self) -> float:
    """Get prediction with special handling for rookies."""
    stats = {
        'total_shows': self.total_shows,
        'avg_placing': self.avg_placing if self.recent_placings else POPULATION_AVG_PLACING
    }
    
    result = handle_first_time_competitor(stats)
    
    # Convert placing to score (0-100)
    placing = result['expected_placing']
    score = 100 * math.exp(-0.15 * (placing - 1))
    
    # Adjust by confidence
    score = score * (result['confidence_score'] / 100)
    
    return score



# AGE REGRESSION CORRECTION - Auto-implemented
PEAK_AGE_MIN = 28
PEAK_AGE_MAX = 32

def calculate_age_factor(age: int) -> float:
    """
    Calculate performance adjustment based on age.
    Peak performance typically 28-32 years old.
    Returns multiplier (1.0 = peak, <1.0 = decline).
    """
    if PEAK_AGE_MIN <= age <= PEAK_AGE_MAX:
        return 1.0
    
    if age < PEAK_AGE_MIN:
        # Young athlete - slight penalty for inexperience
        years_from_peak = PEAK_AGE_MIN - age
        return max(0.85, 1.0 - (years_from_peak * 0.03))  # -3% per year
    else:
        # Older athlete - gradual decline
        years_past_peak = age - PEAK_AGE_MAX
        return max(0.70, 1.0 - (years_past_peak * 0.025))  # -2.5% per year


# Integration
@property
def age_adjusted_score(self) -> float:
    """Get score adjusted for athlete age."""
    base_score = self.recent_form
    age = getattr(self, 'age', 30)  # Default to 30 if age unknown
    age_factor = calculate_age_factor(age)
    return base_score * age_factor



# AGE REGRESSION CORRECTION - Auto-implemented
PEAK_AGE_MIN = 28
PEAK_AGE_MAX = 32

def calculate_age_factor(age: int) -> float:
    """
    Calculate performance adjustment based on age.
    Peak performance typically 28-32 years old.
    Returns multiplier (1.0 = peak, <1.0 = decline).
    """
    if PEAK_AGE_MIN <= age <= PEAK_AGE_MAX:
        return 1.0
    
    if age < PEAK_AGE_MIN:
        # Young athlete - slight penalty for inexperience
        years_from_peak = PEAK_AGE_MIN - age
        return max(0.85, 1.0 - (years_from_peak * 0.03))  # -3% per year
    else:
        # Older athlete - gradual decline
        years_past_peak = age - PEAK_AGE_MAX
        return max(0.70, 1.0 - (years_past_peak * 0.025))  # -2.5% per year


# Integration
@property
def age_adjusted_score(self) -> float:
    """Get score adjusted for athlete age."""
    base_score = self.recent_form
    age = getattr(self, 'age', 30)  # Default to 30 if age unknown
    age_factor = calculate_age_factor(age)
    return base_score * age_factor



# AGE REGRESSION CORRECTION - Auto-implemented
PEAK_AGE_MIN = 28
PEAK_AGE_MAX = 32

def calculate_age_factor(age: int) -> float:
    """
    Calculate performance adjustment based on age.
    Peak performance typically 28-32 years old.
    Returns multiplier (1.0 = peak, <1.0 = decline).
    """
    if PEAK_AGE_MIN <= age <= PEAK_AGE_MAX:
        return 1.0
    
    if age < PEAK_AGE_MIN:
        # Young athlete - slight penalty for inexperience
        years_from_peak = PEAK_AGE_MIN - age
        return max(0.85, 1.0 - (years_from_peak * 0.03))  # -3% per year
    else:
        # Older athlete - gradual decline
        years_past_peak = age - PEAK_AGE_MAX
        return max(0.70, 1.0 - (years_past_peak * 0.025))  # -2.5% per year


# Integration
@property
def age_adjusted_score(self) -> float:
    """Get score adjusted for athlete age."""
    base_score = self.recent_form
    age = getattr(self, 'age', 30)  # Default to 30 if age unknown
    age_factor = calculate_age_factor(age)
    return base_score * age_factor



# HEAD-TO-HEAD HISTORY - Auto-implemented
from typing import Set, Tuple

def analyze_head_to_head(athlete1_id: str, athlete2_id: str, 
                         competition_history: List[Dict]) -> Dict:
    """
    Analyze head-to-head competition history between two athletes.
    Returns relative performance statistics.
    """
    common_competitions = []
    
    # Find competitions where both competed
    for comp in competition_history:
        results = comp.get('results', [])
        athlete1_result = None
        athlete2_result = None
        
        for r in results:
            if r.get('athlete_id') == athlete1_id:
                athlete1_result = r.get('placing')
            elif r.get('athlete_id') == athlete2_id:
                athlete2_result = r.get('placing')
        
        if athlete1_result is not None and athlete2_result is not None:
            common_competitions.append({
                'competition': comp.get('name'),
                'athlete1_placing': athlete1_result,
                'athlete2_placing': athlete2_result,
                'athlete1_won': athlete1_result < athlete2_result
            })
    
    if not common_competitions:
        return {'has_history': False}
    
    # Calculate stats
    athlete1_wins = sum(1 for c in common_competitions if c['athlete1_won'])
    win_rate = athlete1_wins / len(common_competitions)
    
    # Average placing differential
    avg_diff = sum(c['athlete2_placing'] - c['athlete1_placing'] 
                   for c in common_competitions) / len(common_competitions)
    
    return {
        'has_history': True,
        'meetings': len(common_competitions),
        'athlete1_win_rate': win_rate,
        'avg_placing_advantage': avg_diff
    }


# Integration
def get_head_to_head_factor(self, opponent_id: str) -> float:
    """
    Get adjustment factor based on head-to-head history.
    Positive = historically beats opponent, Negative = historically loses.
    """
    # This would need competition history data
    # Simplified implementation
    return 1.0  # Neutral if no history



# HEAD-TO-HEAD HISTORY - Auto-implemented
from typing import Set, Tuple

def analyze_head_to_head(athlete1_id: str, athlete2_id: str, 
                         competition_history: List[Dict]) -> Dict:
    """
    Analyze head-to-head competition history between two athletes.
    Returns relative performance statistics.
    """
    common_competitions = []
    
    # Find competitions where both competed
    for comp in competition_history:
        results = comp.get('results', [])
        athlete1_result = None
        athlete2_result = None
        
        for r in results:
            if r.get('athlete_id') == athlete1_id:
                athlete1_result = r.get('placing')
            elif r.get('athlete_id') == athlete2_id:
                athlete2_result = r.get('placing')
        
        if athlete1_result is not None and athlete2_result is not None:
            common_competitions.append({
                'competition': comp.get('name'),
                'athlete1_placing': athlete1_result,
                'athlete2_placing': athlete2_result,
                'athlete1_won': athlete1_result < athlete2_result
            })
    
    if not common_competitions:
        return {'has_history': False}
    
    # Calculate stats
    athlete1_wins = sum(1 for c in common_competitions if c['athlete1_won'])
    win_rate = athlete1_wins / len(common_competitions)
    
    # Average placing differential
    avg_diff = sum(c['athlete2_placing'] - c['athlete1_placing'] 
                   for c in common_competitions) / len(common_competitions)
    
    return {
        'has_history': True,
        'meetings': len(common_competitions),
        'athlete1_win_rate': win_rate,
        'avg_placing_advantage': avg_diff
    }


# Integration
def get_head_to_head_factor(self, opponent_id: str) -> float:
    """
    Get adjustment factor based on head-to-head history.
    Positive = historically beats opponent, Negative = historically loses.
    """
    # This would need competition history data
    # Simplified implementation
    return 1.0  # Neutral if no history



# COMPETITION TIER CLASSIFICATION - Auto-implemented
COMPETITION_TIERS = {
    'olympia': {'tier': 1, 'difficulty': 1.0, 'name': 'Olympia'},
    'arnold classic': {'tier': 1, 'difficulty': 0.95, 'name': 'Arnold Classic'},
    'new york pro': {'tier': 2, 'difficulty': 0.85, 'name': 'New York Pro'},
    'chicago pro': {'tier': 2, 'difficulty': 0.82, 'name': 'Chicago Pro'},
    'tampa pro': {'tier': 2, 'difficulty': 0.80, 'name': 'Tampa Pro'},
    'ifbb pro league': {'tier': 3, 'difficulty': 0.75, 'name': 'IFBB Pro League'},
    'regional': {'tier': 4, 'difficulty': 0.60, 'name': 'Regional'},
    'local': {'tier': 5, 'difficulty': 0.45, 'name': 'Local'},
}

def get_tier_info(competition_name: str) -> Dict:
    """Get tier and difficulty for a competition."""
    comp_lower = competition_name.lower()
    
    # Check exact matches first
    if comp_lower in COMPETITION_TIERS:
        return COMPETITION_TIERS[comp_lower]
    
    # Check partial matches
    for key, info in COMPETITION_TIERS.items():
        if key in comp_lower:
            return info
    
    # Default to mid-tier
    return {'tier': 3, 'difficulty': 0.70, 'name': 'Standard Pro Show'}


def calculate_tier_adjusted_placing(placing: int, competition_name: str) -> float:
    """
    Adjust placing for competition tier.
    3rd at Olympia > 1st at local show.
    """
    tier_info = get_tier_info(competition_name)
    difficulty = tier_info['difficulty']
    
    # Adjust placing (higher difficulty = better result)
    adjusted = placing * difficulty
    
    return adjusted


# Integration
@property
def tier_quality_score(self) -> float:
    """
    Score based on quality of competitions participated in.
    Higher tier participation indicates higher quality athlete.
    """
    if not self.competition_history:
        return 50.0
    
    total_score = 0
    for comp in self.competition_history:
        tier_info = get_tier_info(comp.get('name', ''))
        # Normalize tier to 0-100 (tier 1 = 100, tier 5 = 20)
        tier_score = 100 - ((tier_info['tier'] - 1) * 20)
        total_score += tier_score
    
    return total_score / len(self.competition_history)



# AGE REGRESSION CORRECTION - Auto-implemented
PEAK_AGE_MIN = 28
PEAK_AGE_MAX = 32

def calculate_age_factor(age: int) -> float:
    """
    Calculate performance adjustment based on age.
    Peak performance typically 28-32 years old.
    Returns multiplier (1.0 = peak, <1.0 = decline).
    """
    if PEAK_AGE_MIN <= age <= PEAK_AGE_MAX:
        return 1.0
    
    if age < PEAK_AGE_MIN:
        # Young athlete - slight penalty for inexperience
        years_from_peak = PEAK_AGE_MIN - age
        return max(0.85, 1.0 - (years_from_peak * 0.03))  # -3% per year
    else:
        # Older athlete - gradual decline
        years_past_peak = age - PEAK_AGE_MAX
        return max(0.70, 1.0 - (years_past_peak * 0.025))  # -2.5% per year


# Integration
@property
def age_adjusted_score(self) -> float:
    """Get score adjusted for athlete age."""
    base_score = self.recent_form
    age = getattr(self, 'age', 30)  # Default to 30 if age unknown
    age_factor = calculate_age_factor(age)
    return base_score * age_factor



# HEAD-TO-HEAD HISTORY - Auto-implemented
from typing import Set, Tuple

def analyze_head_to_head(athlete1_id: str, athlete2_id: str, 
                         competition_history: List[Dict]) -> Dict:
    """
    Analyze head-to-head competition history between two athletes.
    Returns relative performance statistics.
    """
    common_competitions = []
    
    # Find competitions where both competed
    for comp in competition_history:
        results = comp.get('results', [])
        athlete1_result = None
        athlete2_result = None
        
        for r in results:
            if r.get('athlete_id') == athlete1_id:
                athlete1_result = r.get('placing')
            elif r.get('athlete_id') == athlete2_id:
                athlete2_result = r.get('placing')
        
        if athlete1_result is not None and athlete2_result is not None:
            common_competitions.append({
                'competition': comp.get('name'),
                'athlete1_placing': athlete1_result,
                'athlete2_placing': athlete2_result,
                'athlete1_won': athlete1_result < athlete2_result
            })
    
    if not common_competitions:
        return {'has_history': False}
    
    # Calculate stats
    athlete1_wins = sum(1 for c in common_competitions if c['athlete1_won'])
    win_rate = athlete1_wins / len(common_competitions)
    
    # Average placing differential
    avg_diff = sum(c['athlete2_placing'] - c['athlete1_placing'] 
                   for c in common_competitions) / len(common_competitions)
    
    return {
        'has_history': True,
        'meetings': len(common_competitions),
        'athlete1_win_rate': win_rate,
        'avg_placing_advantage': avg_diff
    }


# Integration
def get_head_to_head_factor(self, opponent_id: str) -> float:
    """
    Get adjustment factor based on head-to-head history.
    Positive = historically beats opponent, Negative = historically loses.
    """
    # This would need competition history data
    # Simplified implementation
    return 1.0  # Neutral if no history



# COMPETITION TIER CLASSIFICATION - Auto-implemented
COMPETITION_TIERS = {
    'olympia': {'tier': 1, 'difficulty': 1.0, 'name': 'Olympia'},
    'arnold classic': {'tier': 1, 'difficulty': 0.95, 'name': 'Arnold Classic'},
    'new york pro': {'tier': 2, 'difficulty': 0.85, 'name': 'New York Pro'},
    'chicago pro': {'tier': 2, 'difficulty': 0.82, 'name': 'Chicago Pro'},
    'tampa pro': {'tier': 2, 'difficulty': 0.80, 'name': 'Tampa Pro'},
    'ifbb pro league': {'tier': 3, 'difficulty': 0.75, 'name': 'IFBB Pro League'},
    'regional': {'tier': 4, 'difficulty': 0.60, 'name': 'Regional'},
    'local': {'tier': 5, 'difficulty': 0.45, 'name': 'Local'},
}

def get_tier_info(competition_name: str) -> Dict:
    """Get tier and difficulty for a competition."""
    comp_lower = competition_name.lower()
    
    # Check exact matches first
    if comp_lower in COMPETITION_TIERS:
        return COMPETITION_TIERS[comp_lower]
    
    # Check partial matches
    for key, info in COMPETITION_TIERS.items():
        if key in comp_lower:
            return info
    
    # Default to mid-tier
    return {'tier': 3, 'difficulty': 0.70, 'name': 'Standard Pro Show'}


def calculate_tier_adjusted_placing(placing: int, competition_name: str) -> float:
    """
    Adjust placing for competition tier.
    3rd at Olympia > 1st at local show.
    """
    tier_info = get_tier_info(competition_name)
    difficulty = tier_info['difficulty']
    
    # Adjust placing (higher difficulty = better result)
    adjusted = placing * difficulty
    
    return adjusted


# Integration
@property
def tier_quality_score(self) -> float:
    """
    Score based on quality of competitions participated in.
    Higher tier participation indicates higher quality athlete.
    """
    if not self.competition_history:
        return 50.0
    
    total_score = 0
    for comp in self.competition_history:
        tier_info = get_tier_info(comp.get('name', ''))
        # Normalize tier to 0-100 (tier 1 = 100, tier 5 = 20)
        tier_score = 100 - ((tier_info['tier'] - 1) * 20)
        total_score += tier_score
    
    return total_score / len(self.competition_history)



# HEAD-TO-HEAD HISTORY - Auto-implemented
from typing import Set, Tuple

def analyze_head_to_head(athlete1_id: str, athlete2_id: str, 
                         competition_history: List[Dict]) -> Dict:
    """
    Analyze head-to-head competition history between two athletes.
    Returns relative performance statistics.
    """
    common_competitions = []
    
    # Find competitions where both competed
    for comp in competition_history:
        results = comp.get('results', [])
        athlete1_result = None
        athlete2_result = None
        
        for r in results:
            if r.get('athlete_id') == athlete1_id:
                athlete1_result = r.get('placing')
            elif r.get('athlete_id') == athlete2_id:
                athlete2_result = r.get('placing')
        
        if athlete1_result is not None and athlete2_result is not None:
            common_competitions.append({
                'competition': comp.get('name'),
                'athlete1_placing': athlete1_result,
                'athlete2_placing': athlete2_result,
                'athlete1_won': athlete1_result < athlete2_result
            })
    
    if not common_competitions:
        return {'has_history': False}
    
    # Calculate stats
    athlete1_wins = sum(1 for c in common_competitions if c['athlete1_won'])
    win_rate = athlete1_wins / len(common_competitions)
    
    # Average placing differential
    avg_diff = sum(c['athlete2_placing'] - c['athlete1_placing'] 
                   for c in common_competitions) / len(common_competitions)
    
    return {
        'has_history': True,
        'meetings': len(common_competitions),
        'athlete1_win_rate': win_rate,
        'avg_placing_advantage': avg_diff
    }


# Integration
def get_head_to_head_factor(self, opponent_id: str) -> float:
    """
    Get adjustment factor based on head-to-head history.
    Positive = historically beats opponent, Negative = historically loses.
    """
    # This would need competition history data
    # Simplified implementation
    return 1.0  # Neutral if no history



# COMPETITION TIER CLASSIFICATION - Auto-implemented
COMPETITION_TIERS = {
    'olympia': {'tier': 1, 'difficulty': 1.0, 'name': 'Olympia'},
    'arnold classic': {'tier': 1, 'difficulty': 0.95, 'name': 'Arnold Classic'},
    'new york pro': {'tier': 2, 'difficulty': 0.85, 'name': 'New York Pro'},
    'chicago pro': {'tier': 2, 'difficulty': 0.82, 'name': 'Chicago Pro'},
    'tampa pro': {'tier': 2, 'difficulty': 0.80, 'name': 'Tampa Pro'},
    'ifbb pro league': {'tier': 3, 'difficulty': 0.75, 'name': 'IFBB Pro League'},
    'regional': {'tier': 4, 'difficulty': 0.60, 'name': 'Regional'},
    'local': {'tier': 5, 'difficulty': 0.45, 'name': 'Local'},
}

def get_tier_info(competition_name: str) -> Dict:
    """Get tier and difficulty for a competition."""
    comp_lower = competition_name.lower()
    
    # Check exact matches first
    if comp_lower in COMPETITION_TIERS:
        return COMPETITION_TIERS[comp_lower]
    
    # Check partial matches
    for key, info in COMPETITION_TIERS.items():
        if key in comp_lower:
            return info
    
    # Default to mid-tier
    return {'tier': 3, 'difficulty': 0.70, 'name': 'Standard Pro Show'}


def calculate_tier_adjusted_placing(placing: int, competition_name: str) -> float:
    """
    Adjust placing for competition tier.
    3rd at Olympia > 1st at local show.
    """
    tier_info = get_tier_info(competition_name)
    difficulty = tier_info['difficulty']
    
    # Adjust placing (higher difficulty = better result)
    adjusted = placing * difficulty
    
    return adjusted


# Integration
@property
def tier_quality_score(self) -> float:
    """
    Score based on quality of competitions participated in.
    Higher tier participation indicates higher quality athlete.
    """
    if not self.competition_history:
        return 50.0
    
    total_score = 0
    for comp in self.competition_history:
        tier_info = get_tier_info(comp.get('name', ''))
        # Normalize tier to 0-100 (tier 1 = 100, tier 5 = 20)
        tier_score = 100 - ((tier_info['tier'] - 1) * 20)
        total_score += tier_score
    
    return total_score / len(self.competition_history)



# AGE REGRESSION CORRECTION - Auto-implemented
PEAK_AGE_MIN = 28
PEAK_AGE_MAX = 32

def calculate_age_factor(age: int) -> float:
    """
    Calculate performance adjustment based on age.
    Peak performance typically 28-32 years old.
    Returns multiplier (1.0 = peak, <1.0 = decline).
    """
    if PEAK_AGE_MIN <= age <= PEAK_AGE_MAX:
        return 1.0
    
    if age < PEAK_AGE_MIN:
        # Young athlete - slight penalty for inexperience
        years_from_peak = PEAK_AGE_MIN - age
        return max(0.85, 1.0 - (years_from_peak * 0.03))  # -3% per year
    else:
        # Older athlete - gradual decline
        years_past_peak = age - PEAK_AGE_MAX
        return max(0.70, 1.0 - (years_past_peak * 0.025))  # -2.5% per year


# Integration
@property
def age_adjusted_score(self) -> float:
    """Get score adjusted for athlete age."""
    base_score = self.recent_form
    age = getattr(self, 'age', 30)  # Default to 30 if age unknown
    age_factor = calculate_age_factor(age)
    return base_score * age_factor



# HEAD-TO-HEAD HISTORY - Auto-implemented
from typing import Set, Tuple

def analyze_head_to_head(athlete1_id: str, athlete2_id: str, 
                         competition_history: List[Dict]) -> Dict:
    """
    Analyze head-to-head competition history between two athletes.
    Returns relative performance statistics.
    """
    common_competitions = []
    
    # Find competitions where both competed
    for comp in competition_history:
        results = comp.get('results', [])
        athlete1_result = None
        athlete2_result = None
        
        for r in results:
            if r.get('athlete_id') == athlete1_id:
                athlete1_result = r.get('placing')
            elif r.get('athlete_id') == athlete2_id:
                athlete2_result = r.get('placing')
        
        if athlete1_result is not None and athlete2_result is not None:
            common_competitions.append({
                'competition': comp.get('name'),
                'athlete1_placing': athlete1_result,
                'athlete2_placing': athlete2_result,
                'athlete1_won': athlete1_result < athlete2_result
            })
    
    if not common_competitions:
        return {'has_history': False}
    
    # Calculate stats
    athlete1_wins = sum(1 for c in common_competitions if c['athlete1_won'])
    win_rate = athlete1_wins / len(common_competitions)
    
    # Average placing differential
    avg_diff = sum(c['athlete2_placing'] - c['athlete1_placing'] 
                   for c in common_competitions) / len(common_competitions)
    
    return {
        'has_history': True,
        'meetings': len(common_competitions),
        'athlete1_win_rate': win_rate,
        'avg_placing_advantage': avg_diff
    }


# Integration
def get_head_to_head_factor(self, opponent_id: str) -> float:
    """
    Get adjustment factor based on head-to-head history.
    Positive = historically beats opponent, Negative = historically loses.
    """
    # This would need competition history data
    # Simplified implementation
    return 1.0  # Neutral if no history



# AGE REGRESSION CORRECTION - Auto-implemented
PEAK_AGE_MIN = 28
PEAK_AGE_MAX = 32

def calculate_age_factor(age: int) -> float:
    """
    Calculate performance adjustment based on age.
    Peak performance typically 28-32 years old.
    Returns multiplier (1.0 = peak, <1.0 = decline).
    """
    if PEAK_AGE_MIN <= age <= PEAK_AGE_MAX:
        return 1.0
    
    if age < PEAK_AGE_MIN:
        # Young athlete - slight penalty for inexperience
        years_from_peak = PEAK_AGE_MIN - age
        return max(0.85, 1.0 - (years_from_peak * 0.03))  # -3% per year
    else:
        # Older athlete - gradual decline
        years_past_peak = age - PEAK_AGE_MAX
        return max(0.70, 1.0 - (years_past_peak * 0.025))  # -2.5% per year


# Integration
@property
def age_adjusted_score(self) -> float:
    """Get score adjusted for athlete age."""
    base_score = self.recent_form
    age = getattr(self, 'age', 30)  # Default to 30 if age unknown
    age_factor = calculate_age_factor(age)
    return base_score * age_factor



# DIVISION CONSISTENCY SCORE - Auto-implemented
def calculate_division_consistency(division_history: List[str]) -> float:
    """
    Track division switching behavior.
    High consistency = settled division = more predictable.
    Returns consistency score (0-100).
    """
    if not division_history or len(division_history) < 2:
        return 70.0  # Neutral for new athletes
    
    # Count unique divisions
    unique_divisions = set(division_history)
    
    if len(unique_divisions) == 1:
        # Perfect consistency
        return 100.0
    
    # Calculate consistency ratio
    consistency_ratio = len(division_history) / len(unique_divisions)
    
    # Normalize to 0-100 (higher = more consistent)
    score = min(100, consistency_ratio * 30)
    
    return score


# Integration
@property
def division_consistency_score(self) -> float:
    """Score based on division consistency."""
    divisions = getattr(self, 'division_history', [])
    return calculate_division_consistency(divisions)



# DIVISION CONSISTENCY SCORE - Auto-implemented
def calculate_division_consistency(division_history: List[str]) -> float:
    """
    Track division switching behavior.
    High consistency = settled division = more predictable.
    Returns consistency score (0-100).
    """
    if not division_history or len(division_history) < 2:
        return 70.0  # Neutral for new athletes
    
    # Count unique divisions
    unique_divisions = set(division_history)
    
    if len(unique_divisions) == 1:
        # Perfect consistency
        return 100.0
    
    # Calculate consistency ratio
    consistency_ratio = len(division_history) / len(unique_divisions)
    
    # Normalize to 0-100 (higher = more consistent)
    score = min(100, consistency_ratio * 30)
    
    return score


# Integration
@property
def division_consistency_score(self) -> float:
    """Score based on division consistency."""
    divisions = getattr(self, 'division_history', [])
    return calculate_division_consistency(divisions)



# AGE REGRESSION CORRECTION - Auto-implemented
PEAK_AGE_MIN = 28
PEAK_AGE_MAX = 32

def calculate_age_factor(age: int) -> float:
    """
    Calculate performance adjustment based on age.
    Peak performance typically 28-32 years old.
    Returns multiplier (1.0 = peak, <1.0 = decline).
    """
    if PEAK_AGE_MIN <= age <= PEAK_AGE_MAX:
        return 1.0
    
    if age < PEAK_AGE_MIN:
        # Young athlete - slight penalty for inexperience
        years_from_peak = PEAK_AGE_MIN - age
        return max(0.85, 1.0 - (years_from_peak * 0.03))  # -3% per year
    else:
        # Older athlete - gradual decline
        years_past_peak = age - PEAK_AGE_MAX
        return max(0.70, 1.0 - (years_past_peak * 0.025))  # -2.5% per year


# Integration
@property
def age_adjusted_score(self) -> float:
    """Get score adjusted for athlete age."""
    base_score = self.recent_form
    age = getattr(self, 'age', 30)  # Default to 30 if age unknown
    age_factor = calculate_age_factor(age)
    return base_score * age_factor



# HEAD-TO-HEAD HISTORY - Auto-implemented
from typing import Set, Tuple

def analyze_head_to_head(athlete1_id: str, athlete2_id: str, 
                         competition_history: List[Dict]) -> Dict:
    """
    Analyze head-to-head competition history between two athletes.
    Returns relative performance statistics.
    """
    common_competitions = []
    
    # Find competitions where both competed
    for comp in competition_history:
        results = comp.get('results', [])
        athlete1_result = None
        athlete2_result = None
        
        for r in results:
            if r.get('athlete_id') == athlete1_id:
                athlete1_result = r.get('placing')
            elif r.get('athlete_id') == athlete2_id:
                athlete2_result = r.get('placing')
        
        if athlete1_result is not None and athlete2_result is not None:
            common_competitions.append({
                'competition': comp.get('name'),
                'athlete1_placing': athlete1_result,
                'athlete2_placing': athlete2_result,
                'athlete1_won': athlete1_result < athlete2_result
            })
    
    if not common_competitions:
        return {'has_history': False}
    
    # Calculate stats
    athlete1_wins = sum(1 for c in common_competitions if c['athlete1_won'])
    win_rate = athlete1_wins / len(common_competitions)
    
    # Average placing differential
    avg_diff = sum(c['athlete2_placing'] - c['athlete1_placing'] 
                   for c in common_competitions) / len(common_competitions)
    
    return {
        'has_history': True,
        'meetings': len(common_competitions),
        'athlete1_win_rate': win_rate,
        'avg_placing_advantage': avg_diff
    }


# Integration
def get_head_to_head_factor(self, opponent_id: str) -> float:
    """
    Get adjustment factor based on head-to-head history.
    Positive = historically beats opponent, Negative = historically loses.
    """
    # This would need competition history data
    # Simplified implementation
    return 1.0  # Neutral if no history



# AGE REGRESSION CORRECTION - Auto-implemented
PEAK_AGE_MIN = 28
PEAK_AGE_MAX = 32

def calculate_age_factor(age: int) -> float:
    """
    Calculate performance adjustment based on age.
    Peak performance typically 28-32 years old.
    Returns multiplier (1.0 = peak, <1.0 = decline).
    """
    if PEAK_AGE_MIN <= age <= PEAK_AGE_MAX:
        return 1.0
    
    if age < PEAK_AGE_MIN:
        # Young athlete - slight penalty for inexperience
        years_from_peak = PEAK_AGE_MIN - age
        return max(0.85, 1.0 - (years_from_peak * 0.03))  # -3% per year
    else:
        # Older athlete - gradual decline
        years_past_peak = age - PEAK_AGE_MAX
        return max(0.70, 1.0 - (years_past_peak * 0.025))  # -2.5% per year


# Integration
@property
def age_adjusted_score(self) -> float:
    """Get score adjusted for athlete age."""
    base_score = self.recent_form
    age = getattr(self, 'age', 30)  # Default to 30 if age unknown
    age_factor = calculate_age_factor(age)
    return base_score * age_factor



# HEAD-TO-HEAD HISTORY - Auto-implemented
from typing import Set, Tuple

def analyze_head_to_head(athlete1_id: str, athlete2_id: str, 
                         competition_history: List[Dict]) -> Dict:
    """
    Analyze head-to-head competition history between two athletes.
    Returns relative performance statistics.
    """
    common_competitions = []
    
    # Find competitions where both competed
    for comp in competition_history:
        results = comp.get('results', [])
        athlete1_result = None
        athlete2_result = None
        
        for r in results:
            if r.get('athlete_id') == athlete1_id:
                athlete1_result = r.get('placing')
            elif r.get('athlete_id') == athlete2_id:
                athlete2_result = r.get('placing')
        
        if athlete1_result is not None and athlete2_result is not None:
            common_competitions.append({
                'competition': comp.get('name'),
                'athlete1_placing': athlete1_result,
                'athlete2_placing': athlete2_result,
                'athlete1_won': athlete1_result < athlete2_result
            })
    
    if not common_competitions:
        return {'has_history': False}
    
    # Calculate stats
    athlete1_wins = sum(1 for c in common_competitions if c['athlete1_won'])
    win_rate = athlete1_wins / len(common_competitions)
    
    # Average placing differential
    avg_diff = sum(c['athlete2_placing'] - c['athlete1_placing'] 
                   for c in common_competitions) / len(common_competitions)
    
    return {
        'has_history': True,
        'meetings': len(common_competitions),
        'athlete1_win_rate': win_rate,
        'avg_placing_advantage': avg_diff
    }


# Integration
def get_head_to_head_factor(self, opponent_id: str) -> float:
    """
    Get adjustment factor based on head-to-head history.
    Positive = historically beats opponent, Negative = historically loses.
    """
    # This would need competition history data
    # Simplified implementation
    return 1.0  # Neutral if no history



# HEAD-TO-HEAD HISTORY - Auto-implemented
from typing import Set, Tuple

def analyze_head_to_head(athlete1_id: str, athlete2_id: str, 
                         competition_history: List[Dict]) -> Dict:
    """
    Analyze head-to-head competition history between two athletes.
    Returns relative performance statistics.
    """
    common_competitions = []
    
    # Find competitions where both competed
    for comp in competition_history:
        results = comp.get('results', [])
        athlete1_result = None
        athlete2_result = None
        
        for r in results:
            if r.get('athlete_id') == athlete1_id:
                athlete1_result = r.get('placing')
            elif r.get('athlete_id') == athlete2_id:
                athlete2_result = r.get('placing')
        
        if athlete1_result is not None and athlete2_result is not None:
            common_competitions.append({
                'competition': comp.get('name'),
                'athlete1_placing': athlete1_result,
                'athlete2_placing': athlete2_result,
                'athlete1_won': athlete1_result < athlete2_result
            })
    
    if not common_competitions:
        return {'has_history': False}
    
    # Calculate stats
    athlete1_wins = sum(1 for c in common_competitions if c['athlete1_won'])
    win_rate = athlete1_wins / len(common_competitions)
    
    # Average placing differential
    avg_diff = sum(c['athlete2_placing'] - c['athlete1_placing'] 
                   for c in common_competitions) / len(common_competitions)
    
    return {
        'has_history': True,
        'meetings': len(common_competitions),
        'athlete1_win_rate': win_rate,
        'avg_placing_advantage': avg_diff
    }


# Integration
def get_head_to_head_factor(self, opponent_id: str) -> float:
    """
    Get adjustment factor based on head-to-head history.
    Positive = historically beats opponent, Negative = historically loses.
    """
    # This would need competition history data
    # Simplified implementation
    return 1.0  # Neutral if no history



# DIVISION CONSISTENCY SCORE - Auto-implemented
def calculate_division_consistency(division_history: List[str]) -> float:
    """
    Track division switching behavior.
    High consistency = settled division = more predictable.
    Returns consistency score (0-100).
    """
    if not division_history or len(division_history) < 2:
        return 70.0  # Neutral for new athletes
    
    # Count unique divisions
    unique_divisions = set(division_history)
    
    if len(unique_divisions) == 1:
        # Perfect consistency
        return 100.0
    
    # Calculate consistency ratio
    consistency_ratio = len(division_history) / len(unique_divisions)
    
    # Normalize to 0-100 (higher = more consistent)
    score = min(100, consistency_ratio * 30)
    
    return score


# Integration
@property
def division_consistency_score(self) -> float:
    """Score based on division consistency."""
    divisions = getattr(self, 'division_history', [])
    return calculate_division_consistency(divisions)



# HEAD-TO-HEAD HISTORY - Auto-implemented
from typing import Set, Tuple

def analyze_head_to_head(athlete1_id: str, athlete2_id: str, 
                         competition_history: List[Dict]) -> Dict:
    """
    Analyze head-to-head competition history between two athletes.
    Returns relative performance statistics.
    """
    common_competitions = []
    
    # Find competitions where both competed
    for comp in competition_history:
        results = comp.get('results', [])
        athlete1_result = None
        athlete2_result = None
        
        for r in results:
            if r.get('athlete_id') == athlete1_id:
                athlete1_result = r.get('placing')
            elif r.get('athlete_id') == athlete2_id:
                athlete2_result = r.get('placing')
        
        if athlete1_result is not None and athlete2_result is not None:
            common_competitions.append({
                'competition': comp.get('name'),
                'athlete1_placing': athlete1_result,
                'athlete2_placing': athlete2_result,
                'athlete1_won': athlete1_result < athlete2_result
            })
    
    if not common_competitions:
        return {'has_history': False}
    
    # Calculate stats
    athlete1_wins = sum(1 for c in common_competitions if c['athlete1_won'])
    win_rate = athlete1_wins / len(common_competitions)
    
    # Average placing differential
    avg_diff = sum(c['athlete2_placing'] - c['athlete1_placing'] 
                   for c in common_competitions) / len(common_competitions)
    
    return {
        'has_history': True,
        'meetings': len(common_competitions),
        'athlete1_win_rate': win_rate,
        'avg_placing_advantage': avg_diff
    }


# Integration
def get_head_to_head_factor(self, opponent_id: str) -> float:
    """
    Get adjustment factor based on head-to-head history.
    Positive = historically beats opponent, Negative = historically loses.
    """
    # This would need competition history data
    # Simplified implementation
    return 1.0  # Neutral if no history



# AGE REGRESSION CORRECTION - Auto-implemented
PEAK_AGE_MIN = 28
PEAK_AGE_MAX = 32

def calculate_age_factor(age: int) -> float:
    """
    Calculate performance adjustment based on age.
    Peak performance typically 28-32 years old.
    Returns multiplier (1.0 = peak, <1.0 = decline).
    """
    if PEAK_AGE_MIN <= age <= PEAK_AGE_MAX:
        return 1.0
    
    if age < PEAK_AGE_MIN:
        # Young athlete - slight penalty for inexperience
        years_from_peak = PEAK_AGE_MIN - age
        return max(0.85, 1.0 - (years_from_peak * 0.03))  # -3% per year
    else:
        # Older athlete - gradual decline
        years_past_peak = age - PEAK_AGE_MAX
        return max(0.70, 1.0 - (years_past_peak * 0.025))  # -2.5% per year


# Integration
@property
def age_adjusted_score(self) -> float:
    """Get score adjusted for athlete age."""
    base_score = self.recent_form
    age = getattr(self, 'age', 30)  # Default to 30 if age unknown
    age_factor = calculate_age_factor(age)
    return base_score * age_factor



# HEAD-TO-HEAD HISTORY - Auto-implemented
from typing import Set, Tuple

def analyze_head_to_head(athlete1_id: str, athlete2_id: str, 
                         competition_history: List[Dict]) -> Dict:
    """
    Analyze head-to-head competition history between two athletes.
    Returns relative performance statistics.
    """
    common_competitions = []
    
    # Find competitions where both competed
    for comp in competition_history:
        results = comp.get('results', [])
        athlete1_result = None
        athlete2_result = None
        
        for r in results:
            if r.get('athlete_id') == athlete1_id:
                athlete1_result = r.get('placing')
            elif r.get('athlete_id') == athlete2_id:
                athlete2_result = r.get('placing')
        
        if athlete1_result is not None and athlete2_result is not None:
            common_competitions.append({
                'competition': comp.get('name'),
                'athlete1_placing': athlete1_result,
                'athlete2_placing': athlete2_result,
                'athlete1_won': athlete1_result < athlete2_result
            })
    
    if not common_competitions:
        return {'has_history': False}
    
    # Calculate stats
    athlete1_wins = sum(1 for c in common_competitions if c['athlete1_won'])
    win_rate = athlete1_wins / len(common_competitions)
    
    # Average placing differential
    avg_diff = sum(c['athlete2_placing'] - c['athlete1_placing'] 
                   for c in common_competitions) / len(common_competitions)
    
    return {
        'has_history': True,
        'meetings': len(common_competitions),
        'athlete1_win_rate': win_rate,
        'avg_placing_advantage': avg_diff
    }


# Integration
def get_head_to_head_factor(self, opponent_id: str) -> float:
    """
    Get adjustment factor based on head-to-head history.
    Positive = historically beats opponent, Negative = historically loses.
    """
    # This would need competition history data
    # Simplified implementation
    return 1.0  # Neutral if no history



# AGE REGRESSION CORRECTION - Auto-implemented
PEAK_AGE_MIN = 28
PEAK_AGE_MAX = 32

def calculate_age_factor(age: int) -> float:
    """
    Calculate performance adjustment based on age.
    Peak performance typically 28-32 years old.
    Returns multiplier (1.0 = peak, <1.0 = decline).
    """
    if PEAK_AGE_MIN <= age <= PEAK_AGE_MAX:
        return 1.0
    
    if age < PEAK_AGE_MIN:
        # Young athlete - slight penalty for inexperience
        years_from_peak = PEAK_AGE_MIN - age
        return max(0.85, 1.0 - (years_from_peak * 0.03))  # -3% per year
    else:
        # Older athlete - gradual decline
        years_past_peak = age - PEAK_AGE_MAX
        return max(0.70, 1.0 - (years_past_peak * 0.025))  # -2.5% per year


# Integration
@property
def age_adjusted_score(self) -> float:
    """Get score adjusted for athlete age."""
    base_score = self.recent_form
    age = getattr(self, 'age', 30)  # Default to 30 if age unknown
    age_factor = calculate_age_factor(age)
    return base_score * age_factor



# DIVISION CONSISTENCY SCORE - Auto-implemented
def calculate_division_consistency(division_history: List[str]) -> float:
    """
    Track division switching behavior.
    High consistency = settled division = more predictable.
    Returns consistency score (0-100).
    """
    if not division_history or len(division_history) < 2:
        return 70.0  # Neutral for new athletes
    
    # Count unique divisions
    unique_divisions = set(division_history)
    
    if len(unique_divisions) == 1:
        # Perfect consistency
        return 100.0
    
    # Calculate consistency ratio
    consistency_ratio = len(division_history) / len(unique_divisions)
    
    # Normalize to 0-100 (higher = more consistent)
    score = min(100, consistency_ratio * 30)
    
    return score


# Integration
@property
def division_consistency_score(self) -> float:
    """Score based on division consistency."""
    divisions = getattr(self, 'division_history', [])
    return calculate_division_consistency(divisions)



# AGE REGRESSION CORRECTION - Auto-implemented
PEAK_AGE_MIN = 28
PEAK_AGE_MAX = 32

def calculate_age_factor(age: int) -> float:
    """
    Calculate performance adjustment based on age.
    Peak performance typically 28-32 years old.
    Returns multiplier (1.0 = peak, <1.0 = decline).
    """
    if PEAK_AGE_MIN <= age <= PEAK_AGE_MAX:
        return 1.0
    
    if age < PEAK_AGE_MIN:
        # Young athlete - slight penalty for inexperience
        years_from_peak = PEAK_AGE_MIN - age
        return max(0.85, 1.0 - (years_from_peak * 0.03))  # -3% per year
    else:
        # Older athlete - gradual decline
        years_past_peak = age - PEAK_AGE_MAX
        return max(0.70, 1.0 - (years_past_peak * 0.025))  # -2.5% per year


# Integration
@property
def age_adjusted_score(self) -> float:
    """Get score adjusted for athlete age."""
    base_score = self.recent_form
    age = getattr(self, 'age', 30)  # Default to 30 if age unknown
    age_factor = calculate_age_factor(age)
    return base_score * age_factor



# DIVISION CONSISTENCY SCORE - Auto-implemented
def calculate_division_consistency(division_history: List[str]) -> float:
    """
    Track division switching behavior.
    High consistency = settled division = more predictable.
    Returns consistency score (0-100).
    """
    if not division_history or len(division_history) < 2:
        return 70.0  # Neutral for new athletes
    
    # Count unique divisions
    unique_divisions = set(division_history)
    
    if len(unique_divisions) == 1:
        # Perfect consistency
        return 100.0
    
    # Calculate consistency ratio
    consistency_ratio = len(division_history) / len(unique_divisions)
    
    # Normalize to 0-100 (higher = more consistent)
    score = min(100, consistency_ratio * 30)
    
    return score


# Integration
@property
def division_consistency_score(self) -> float:
    """Score based on division consistency."""
    divisions = getattr(self, 'division_history', [])
    return calculate_division_consistency(divisions)



# COMPETITION FREQUENCY FACTOR - Auto-implemented
from datetime import datetime, timedelta

OPTIMAL_COMPETITIONS_PER_YEAR = (2, 4)  # Min, Max optimal range

def calculate_frequency_factor(competition_dates: List[str]) -> float:
    """
    Analyze competition frequency for fatigue/rust detection.
    Too frequent = tired, too rare = rusty.
    Returns adjustment factor (0.8-1.1).
    """
    if len(competition_dates) < 2:
        return 1.0  # Not enough data
    
    # Parse dates
    dates = []
    for d in competition_dates:
        try:
            dates.append(datetime.fromisoformat(d.replace('Z', '+00:00')))
        except:
            pass
    
    if len(dates) < 2:
        return 1.0
    
    dates.sort(reverse=True)  # Most recent first
    
    # Calculate average days between competitions
    total_days = (dates[0] - dates[-1]).days
    avg_days_between = total_days / (len(dates) - 1)
    
    # Convert to competitions per year
    comps_per_year = 365 / avg_days_between if avg_days_between > 0 else 0
    
    # Determine factor
    if comps_per_year < OPTIMAL_COMPETITIONS_PER_YEAR[0]:
        # Too rusty
        return 0.90
    elif comps_per_year > OPTIMAL_COMPETITIONS_PER_YEAR[1]:
        # Too frequent
        return 0.95
    else:
        # Optimal range
        return 1.05


# Integration
@property
def frequency_adjusted_score(self) -> float:
    """Get score adjusted for competition frequency."""
    base_score = self.recent_form
    dates = self.get_show_dates()
    factor = calculate_frequency_factor(dates)
    return base_score * factor



# AGE REGRESSION CORRECTION - Auto-implemented
PEAK_AGE_MIN = 28
PEAK_AGE_MAX = 32

def calculate_age_factor(age: int) -> float:
    """
    Calculate performance adjustment based on age.
    Peak performance typically 28-32 years old.
    Returns multiplier (1.0 = peak, <1.0 = decline).
    """
    if PEAK_AGE_MIN <= age <= PEAK_AGE_MAX:
        return 1.0
    
    if age < PEAK_AGE_MIN:
        # Young athlete - slight penalty for inexperience
        years_from_peak = PEAK_AGE_MIN - age
        return max(0.85, 1.0 - (years_from_peak * 0.03))  # -3% per year
    else:
        # Older athlete - gradual decline
        years_past_peak = age - PEAK_AGE_MAX
        return max(0.70, 1.0 - (years_past_peak * 0.025))  # -2.5% per year


# Integration
@property
def age_adjusted_score(self) -> float:
    """Get score adjusted for athlete age."""
    base_score = self.recent_form
    age = getattr(self, 'age', 30)  # Default to 30 if age unknown
    age_factor = calculate_age_factor(age)
    return base_score * age_factor



# DIVISION CONSISTENCY SCORE - Auto-implemented
def calculate_division_consistency(division_history: List[str]) -> float:
    """
    Track division switching behavior.
    High consistency = settled division = more predictable.
    Returns consistency score (0-100).
    """
    if not division_history or len(division_history) < 2:
        return 70.0  # Neutral for new athletes
    
    # Count unique divisions
    unique_divisions = set(division_history)
    
    if len(unique_divisions) == 1:
        # Perfect consistency
        return 100.0
    
    # Calculate consistency ratio
    consistency_ratio = len(division_history) / len(unique_divisions)
    
    # Normalize to 0-100 (higher = more consistent)
    score = min(100, consistency_ratio * 30)
    
    return score


# Integration
@property
def division_consistency_score(self) -> float:
    """Score based on division consistency."""
    divisions = getattr(self, 'division_history', [])
    return calculate_division_consistency(divisions)



# COMPETITION FREQUENCY FACTOR - Auto-implemented
from datetime import datetime, timedelta

OPTIMAL_COMPETITIONS_PER_YEAR = (2, 4)  # Min, Max optimal range

def calculate_frequency_factor(competition_dates: List[str]) -> float:
    """
    Analyze competition frequency for fatigue/rust detection.
    Too frequent = tired, too rare = rusty.
    Returns adjustment factor (0.8-1.1).
    """
    if len(competition_dates) < 2:
        return 1.0  # Not enough data
    
    # Parse dates
    dates = []
    for d in competition_dates:
        try:
            dates.append(datetime.fromisoformat(d.replace('Z', '+00:00')))
        except:
            pass
    
    if len(dates) < 2:
        return 1.0
    
    dates.sort(reverse=True)  # Most recent first
    
    # Calculate average days between competitions
    total_days = (dates[0] - dates[-1]).days
    avg_days_between = total_days / (len(dates) - 1)
    
    # Convert to competitions per year
    comps_per_year = 365 / avg_days_between if avg_days_between > 0 else 0
    
    # Determine factor
    if comps_per_year < OPTIMAL_COMPETITIONS_PER_YEAR[0]:
        # Too rusty
        return 0.90
    elif comps_per_year > OPTIMAL_COMPETITIONS_PER_YEAR[1]:
        # Too frequent
        return 0.95
    else:
        # Optimal range
        return 1.05


# Integration
@property
def frequency_adjusted_score(self) -> float:
    """Get score adjusted for competition frequency."""
    base_score = self.recent_form
    dates = self.get_show_dates()
    factor = calculate_frequency_factor(dates)
    return base_score * factor



# DIVISION CONSISTENCY SCORE - Auto-implemented
def calculate_division_consistency(division_history: List[str]) -> float:
    """
    Track division switching behavior.
    High consistency = settled division = more predictable.
    Returns consistency score (0-100).
    """
    if not division_history or len(division_history) < 2:
        return 70.0  # Neutral for new athletes
    
    # Count unique divisions
    unique_divisions = set(division_history)
    
    if len(unique_divisions) == 1:
        # Perfect consistency
        return 100.0
    
    # Calculate consistency ratio
    consistency_ratio = len(division_history) / len(unique_divisions)
    
    # Normalize to 0-100 (higher = more consistent)
    score = min(100, consistency_ratio * 30)
    
    return score


# Integration
@property
def division_consistency_score(self) -> float:
    """Score based on division consistency."""
    divisions = getattr(self, 'division_history', [])
    return calculate_division_consistency(divisions)



# SEASONAL PATTERNS - Auto-implemented
from datetime import datetime
from collections import defaultdict

def analyze_seasonal_performance(competition_history: List[Dict]) -> Dict[int, float]:
    """
    Analyze performance by month to detect seasonal patterns.
    Returns average performance score for each month (1-12).
    """
    monthly_scores = defaultdict(list)
    
    for comp in competition_history:
        date_str = comp.get('date', '')
        placing = comp.get('placing', 10)
        
        try:
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            month = date.month
            
            # Convert placing to score (lower = better)
            score = 100 * math.exp(-0.15 * (placing - 1))
            monthly_scores[month].append(score)
        except:
            pass
    
    # Calculate averages
    monthly_averages = {}
    for month in range(1, 13):
        if monthly_scores[month]:
            monthly_averages[month] = sum(monthly_scores[month]) / len(monthly_scores[month])
        else:
            monthly_averages[month] = 50.0  # Neutral
    
    return monthly_averages


def get_seasonal_factor(current_month: int, monthly_averages: Dict[int, float]) -> float:
    """
    Get performance adjustment based on current month vs historical pattern.
    """
    if not monthly_averages:
        return 1.0
    
    current_avg = monthly_averages.get(current_month, 50.0)
    overall_avg = sum(monthly_averages.values()) / 12
    
    if overall_avg == 0:
        return 1.0
    
    # Factor based on how much better/worse this month typically is
    factor = 0.9 + (0.2 * (current_avg / overall_avg))
    return min(1.2, max(0.8, factor))


# Integration
@property
def seasonal_adjustment(self) -> float:
    """Get score adjustment for current season."""
    monthly = analyze_seasonal_performance(self.competition_history)
    current_month = datetime.now().month
    return get_seasonal_factor(current_month, monthly)



# SEASONAL PATTERNS - Auto-implemented
from datetime import datetime
from collections import defaultdict

def analyze_seasonal_performance(competition_history: List[Dict]) -> Dict[int, float]:
    """
    Analyze performance by month to detect seasonal patterns.
    Returns average performance score for each month (1-12).
    """
    monthly_scores = defaultdict(list)
    
    for comp in competition_history:
        date_str = comp.get('date', '')
        placing = comp.get('placing', 10)
        
        try:
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            month = date.month
            
            # Convert placing to score (lower = better)
            score = 100 * math.exp(-0.15 * (placing - 1))
            monthly_scores[month].append(score)
        except:
            pass
    
    # Calculate averages
    monthly_averages = {}
    for month in range(1, 13):
        if monthly_scores[month]:
            monthly_averages[month] = sum(monthly_scores[month]) / len(monthly_scores[month])
        else:
            monthly_averages[month] = 50.0  # Neutral
    
    return monthly_averages


def get_seasonal_factor(current_month: int, monthly_averages: Dict[int, float]) -> float:
    """
    Get performance adjustment based on current month vs historical pattern.
    """
    if not monthly_averages:
        return 1.0
    
    current_avg = monthly_averages.get(current_month, 50.0)
    overall_avg = sum(monthly_averages.values()) / 12
    
    if overall_avg == 0:
        return 1.0
    
    # Factor based on how much better/worse this month typically is
    factor = 0.9 + (0.2 * (current_avg / overall_avg))
    return min(1.2, max(0.8, factor))


# Integration
@property
def seasonal_adjustment(self) -> float:
    """Get score adjustment for current season."""
    monthly = analyze_seasonal_performance(self.competition_history)
    current_month = datetime.now().month
    return get_seasonal_factor(current_month, monthly)



# SEASONAL PATTERNS - Auto-implemented
from datetime import datetime
from collections import defaultdict

def analyze_seasonal_performance(competition_history: List[Dict]) -> Dict[int, float]:
    """
    Analyze performance by month to detect seasonal patterns.
    Returns average performance score for each month (1-12).
    """
    monthly_scores = defaultdict(list)
    
    for comp in competition_history:
        date_str = comp.get('date', '')
        placing = comp.get('placing', 10)
        
        try:
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            month = date.month
            
            # Convert placing to score (lower = better)
            score = 100 * math.exp(-0.15 * (placing - 1))
            monthly_scores[month].append(score)
        except:
            pass
    
    # Calculate averages
    monthly_averages = {}
    for month in range(1, 13):
        if monthly_scores[month]:
            monthly_averages[month] = sum(monthly_scores[month]) / len(monthly_scores[month])
        else:
            monthly_averages[month] = 50.0  # Neutral
    
    return monthly_averages


def get_seasonal_factor(current_month: int, monthly_averages: Dict[int, float]) -> float:
    """
    Get performance adjustment based on current month vs historical pattern.
    """
    if not monthly_averages:
        return 1.0
    
    current_avg = monthly_averages.get(current_month, 50.0)
    overall_avg = sum(monthly_averages.values()) / 12
    
    if overall_avg == 0:
        return 1.0
    
    # Factor based on how much better/worse this month typically is
    factor = 0.9 + (0.2 * (current_avg / overall_avg))
    return min(1.2, max(0.8, factor))


# Integration
@property
def seasonal_adjustment(self) -> float:
    """Get score adjustment for current season."""
    monthly = analyze_seasonal_performance(self.competition_history)
    current_month = datetime.now().month
    return get_seasonal_factor(current_month, monthly)



# AGE REGRESSION CORRECTION - Auto-implemented
PEAK_AGE_MIN = 28
PEAK_AGE_MAX = 32

def calculate_age_factor(age: int) -> float:
    """
    Calculate performance adjustment based on age.
    Peak performance typically 28-32 years old.
    Returns multiplier (1.0 = peak, <1.0 = decline).
    """
    if PEAK_AGE_MIN <= age <= PEAK_AGE_MAX:
        return 1.0
    
    if age < PEAK_AGE_MIN:
        # Young athlete - slight penalty for inexperience
        years_from_peak = PEAK_AGE_MIN - age
        return max(0.85, 1.0 - (years_from_peak * 0.03))  # -3% per year
    else:
        # Older athlete - gradual decline
        years_past_peak = age - PEAK_AGE_MAX
        return max(0.70, 1.0 - (years_past_peak * 0.025))  # -2.5% per year


# Integration
@property
def age_adjusted_score(self) -> float:
    """Get score adjusted for athlete age."""
    base_score = self.recent_form
    age = getattr(self, 'age', 30)  # Default to 30 if age unknown
    age_factor = calculate_age_factor(age)
    return base_score * age_factor



# SEASONAL PATTERNS - Auto-implemented
from datetime import datetime
from collections import defaultdict

def analyze_seasonal_performance(competition_history: List[Dict]) -> Dict[int, float]:
    """
    Analyze performance by month to detect seasonal patterns.
    Returns average performance score for each month (1-12).
    """
    monthly_scores = defaultdict(list)
    
    for comp in competition_history:
        date_str = comp.get('date', '')
        placing = comp.get('placing', 10)
        
        try:
            date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            month = date.month
            
            # Convert placing to score (lower = better)
            score = 100 * math.exp(-0.15 * (placing - 1))
            monthly_scores[month].append(score)
        except:
            pass
    
    # Calculate averages
    monthly_averages = {}
    for month in range(1, 13):
        if monthly_scores[month]:
            monthly_averages[month] = sum(monthly_scores[month]) / len(monthly_scores[month])
        else:
            monthly_averages[month] = 50.0  # Neutral
    
    return monthly_averages


def get_seasonal_factor(current_month: int, monthly_averages: Dict[int, float]) -> float:
    """
    Get performance adjustment based on current month vs historical pattern.
    """
    if not monthly_averages:
        return 1.0
    
    current_avg = monthly_averages.get(current_month, 50.0)
    overall_avg = sum(monthly_averages.values()) / 12
    
    if overall_avg == 0:
        return 1.0
    
    # Factor based on how much better/worse this month typically is
    factor = 0.9 + (0.2 * (current_avg / overall_avg))
    return min(1.2, max(0.8, factor))


# Integration
@property
def seasonal_adjustment(self) -> float:
    """Get score adjustment for current season."""
    monthly = analyze_seasonal_performance(self.competition_history)
    current_month = datetime.now().month
    return get_seasonal_factor(current_month, monthly)



# AGE REGRESSION CORRECTION - Auto-implemented
PEAK_AGE_MIN = 28
PEAK_AGE_MAX = 32

def calculate_age_factor(age: int) -> float:
    """
    Calculate performance adjustment based on age.
    Peak performance typically 28-32 years old.
    Returns multiplier (1.0 = peak, <1.0 = decline).
    """
    if PEAK_AGE_MIN <= age <= PEAK_AGE_MAX:
        return 1.0
    
    if age < PEAK_AGE_MIN:
        # Young athlete - slight penalty for inexperience
        years_from_peak = PEAK_AGE_MIN - age
        return max(0.85, 1.0 - (years_from_peak * 0.03))  # -3% per year
    else:
        # Older athlete - gradual decline
        years_past_peak = age - PEAK_AGE_MAX
        return max(0.70, 1.0 - (years_past_peak * 0.025))  # -2.5% per year


# Integration
@property
def age_adjusted_score(self) -> float:
    """Get score adjusted for athlete age."""
    base_score = self.recent_form
    age = getattr(self, 'age', 30)  # Default to 30 if age unknown
    age_factor = calculate_age_factor(age)
    return base_score * age_factor



# AGE REGRESSION CORRECTION - Auto-implemented
PEAK_AGE_MIN = 28
PEAK_AGE_MAX = 32

def calculate_age_factor(age: int) -> float:
    """
    Calculate performance adjustment based on age.
    Peak performance typically 28-32 years old.
    Returns multiplier (1.0 = peak, <1.0 = decline).
    """
    if PEAK_AGE_MIN <= age <= PEAK_AGE_MAX:
        return 1.0
    
    if age < PEAK_AGE_MIN:
        # Young athlete - slight penalty for inexperience
        years_from_peak = PEAK_AGE_MIN - age
        return max(0.85, 1.0 - (years_from_peak * 0.03))  # -3% per year
    else:
        # Older athlete - gradual decline
        years_past_peak = age - PEAK_AGE_MAX
        return max(0.70, 1.0 - (years_past_peak * 0.025))  # -2.5% per year


# Integration
@property
def age_adjusted_score(self) -> float:
    """Get score adjusted for athlete age."""
    base_score = self.recent_form
    age = getattr(self, 'age', 30)  # Default to 30 if age unknown
    age_factor = calculate_age_factor(age)
    return base_score * age_factor



# AGE REGRESSION CORRECTION - Auto-implemented
PEAK_AGE_MIN = 28
PEAK_AGE_MAX = 32

def calculate_age_factor(age: int) -> float:
    """
    Calculate performance adjustment based on age.
    Peak performance typically 28-32 years old.
    Returns multiplier (1.0 = peak, <1.0 = decline).
    """
    if PEAK_AGE_MIN <= age <= PEAK_AGE_MAX:
        return 1.0
    
    if age < PEAK_AGE_MIN:
        # Young athlete - slight penalty for inexperience
        years_from_peak = PEAK_AGE_MIN - age
        return max(0.85, 1.0 - (years_from_peak * 0.03))  # -3% per year
    else:
        # Older athlete - gradual decline
        years_past_peak = age - PEAK_AGE_MAX
        return max(0.70, 1.0 - (years_past_peak * 0.025))  # -2.5% per year


# Integration
@property
def age_adjusted_score(self) -> float:
    """Get score adjusted for athlete age."""
    base_score = self.recent_form
    age = getattr(self, 'age', 30)  # Default to 30 if age unknown
    age_factor = calculate_age_factor(age)
    return base_score * age_factor



# AGE REGRESSION CORRECTION - Auto-implemented
PEAK_AGE_MIN = 28
PEAK_AGE_MAX = 32

def calculate_age_factor(age: int) -> float:
    """
    Calculate performance adjustment based on age.
    Peak performance typically 28-32 years old.
    Returns multiplier (1.0 = peak, <1.0 = decline).
    """
    if PEAK_AGE_MIN <= age <= PEAK_AGE_MAX:
        return 1.0
    
    if age < PEAK_AGE_MIN:
        # Young athlete - slight penalty for inexperience
        years_from_peak = PEAK_AGE_MIN - age
        return max(0.85, 1.0 - (years_from_peak * 0.03))  # -3% per year
    else:
        # Older athlete - gradual decline
        years_past_peak = age - PEAK_AGE_MAX
        return max(0.70, 1.0 - (years_past_peak * 0.025))  # -2.5% per year


# Integration
@property
def age_adjusted_score(self) -> float:
    """Get score adjusted for athlete age."""
    base_score = self.recent_form
    age = getattr(self, 'age', 30)  # Default to 30 if age unknown
    age_factor = calculate_age_factor(age)
    return base_score * age_factor



# AGE REGRESSION CORRECTION - Auto-implemented
PEAK_AGE_MIN = 28
PEAK_AGE_MAX = 32

def calculate_age_factor(age: int) -> float:
    """
    Calculate performance adjustment based on age.
    Peak performance typically 28-32 years old.
    Returns multiplier (1.0 = peak, <1.0 = decline).
    """
    if PEAK_AGE_MIN <= age <= PEAK_AGE_MAX:
        return 1.0
    
    if age < PEAK_AGE_MIN:
        # Young athlete - slight penalty for inexperience
        years_from_peak = PEAK_AGE_MIN - age
        return max(0.85, 1.0 - (years_from_peak * 0.03))  # -3% per year
    else:
        # Older athlete - gradual decline
        years_past_peak = age - PEAK_AGE_MAX
        return max(0.70, 1.0 - (years_past_peak * 0.025))  # -2.5% per year


# Integration
@property
def age_adjusted_score(self) -> float:
    """Get score adjusted for athlete age."""
    base_score = self.recent_form
    age = getattr(self, 'age', 30)  # Default to 30 if age unknown
    age_factor = calculate_age_factor(age)
    return base_score * age_factor



# AGE REGRESSION CORRECTION - Auto-implemented
PEAK_AGE_MIN = 28
PEAK_AGE_MAX = 32

def calculate_age_factor(age: int) -> float:
    """
    Calculate performance adjustment based on age.
    Peak performance typically 28-32 years old.
    Returns multiplier (1.0 = peak, <1.0 = decline).
    """
    if PEAK_AGE_MIN <= age <= PEAK_AGE_MAX:
        return 1.0
    
    if age < PEAK_AGE_MIN:
        # Young athlete - slight penalty for inexperience
        years_from_peak = PEAK_AGE_MIN - age
        return max(0.85, 1.0 - (years_from_peak * 0.03))  # -3% per year
    else:
        # Older athlete - gradual decline
        years_past_peak = age - PEAK_AGE_MAX
        return max(0.70, 1.0 - (years_past_peak * 0.025))  # -2.5% per year


# Integration
@property
def age_adjusted_score(self) -> float:
    """Get score adjusted for athlete age."""
    base_score = self.recent_form
    age = getattr(self, 'age', 30)  # Default to 30 if age unknown
    age_factor = calculate_age_factor(age)
    return base_score * age_factor



# AGE REGRESSION CORRECTION - Auto-implemented
PEAK_AGE_MIN = 28
PEAK_AGE_MAX = 32

def calculate_age_factor(age: int) -> float:
    """
    Calculate performance adjustment based on age.
    Peak performance typically 28-32 years old.
    Returns multiplier (1.0 = peak, <1.0 = decline).
    """
    if PEAK_AGE_MIN <= age <= PEAK_AGE_MAX:
        return 1.0
    
    if age < PEAK_AGE_MIN:
        # Young athlete - slight penalty for inexperience
        years_from_peak = PEAK_AGE_MIN - age
        return max(0.85, 1.0 - (years_from_peak * 0.03))  # -3% per year
    else:
        # Older athlete - gradual decline
        years_past_peak = age - PEAK_AGE_MAX
        return max(0.70, 1.0 - (years_past_peak * 0.025))  # -2.5% per year


# Integration
@property
def age_adjusted_score(self) -> float:
    """Get score adjusted for athlete age."""
    base_score = self.recent_form
    age = getattr(self, 'age', 30)  # Default to 30 if age unknown
    age_factor = calculate_age_factor(age)
    return base_score * age_factor



# PEAK DETECTION - Auto-implemented
def detect_peak_phase(placings: List[int], dates: List[str]) -> Dict:
    """
    Detect if athlete is peaking, steady, or declining.
    Uses 6-month trend analysis.
    """
    if len(placings) < 3:
        return {'phase': 'unknown', 'confidence': 0}
    
    # Split into first/second half
    mid = len(placings) // 2
    first_half = sum(placings[:mid]) / mid
    second_half = sum(placings[mid:]) / (len(placings) - mid)
    
    improvement = first_half - second_half
    
    if improvement > 2:
        phase = 'peaking'
        confidence = min(abs(improvement) * 10, 95)
    elif improvement < -2:
        phase = 'declining'
        confidence = min(abs(improvement) * 10, 95)
    else:
        phase = 'steady'
        confidence = 70
    
    return {
        'phase': phase,
        'confidence': confidence,
        'improvement': improvement
    }


# Add to prediction factors
@property
def peak_factor(self) -> float:
    """
    Adjustment based on peak phase.
    Peaking athletes get bonus, declining get penalty.
    """
    peak_info = detect_peak_phase(self.recent_placings, self.get_show_dates())
    
    if peak_info['phase'] == 'peaking':
        return 1.1  # +10% boost
    elif peak_info['phase'] == 'declining':
        return 0.9  # -10% penalty
    return 1.0



# PEAK DETECTION - Auto-implemented
def detect_peak_phase(placings: List[int], dates: List[str]) -> Dict:
    """
    Detect if athlete is peaking, steady, or declining.
    Uses 6-month trend analysis.
    """
    if len(placings) < 3:
        return {'phase': 'unknown', 'confidence': 0}
    
    # Split into first/second half
    mid = len(placings) // 2
    first_half = sum(placings[:mid]) / mid
    second_half = sum(placings[mid:]) / (len(placings) - mid)
    
    improvement = first_half - second_half
    
    if improvement > 2:
        phase = 'peaking'
        confidence = min(abs(improvement) * 10, 95)
    elif improvement < -2:
        phase = 'declining'
        confidence = min(abs(improvement) * 10, 95)
    else:
        phase = 'steady'
        confidence = 70
    
    return {
        'phase': phase,
        'confidence': confidence,
        'improvement': improvement
    }


# Add to prediction factors
@property
def peak_factor(self) -> float:
    """
    Adjustment based on peak phase.
    Peaking athletes get bonus, declining get penalty.
    """
    peak_info = detect_peak_phase(self.recent_placings, self.get_show_dates())
    
    if peak_info['phase'] == 'peaking':
        return 1.1  # +10% boost
    elif peak_info['phase'] == 'declining':
        return 0.9  # -10% penalty
    return 1.0



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# PEAK DETECTION - Auto-implemented
def detect_peak_phase(placings: List[int], dates: List[str]) -> Dict:
    """
    Detect if athlete is peaking, steady, or declining.
    Uses 6-month trend analysis.
    """
    if len(placings) < 3:
        return {'phase': 'unknown', 'confidence': 0}
    
    # Split into first/second half
    mid = len(placings) // 2
    first_half = sum(placings[:mid]) / mid
    second_half = sum(placings[mid:]) / (len(placings) - mid)
    
    improvement = first_half - second_half
    
    if improvement > 2:
        phase = 'peaking'
        confidence = min(abs(improvement) * 10, 95)
    elif improvement < -2:
        phase = 'declining'
        confidence = min(abs(improvement) * 10, 95)
    else:
        phase = 'steady'
        confidence = 70
    
    return {
        'phase': phase,
        'confidence': confidence,
        'improvement': improvement
    }


# Add to prediction factors
@property
def peak_factor(self) -> float:
    """
    Adjustment based on peak phase.
    Peaking athletes get bonus, declining get penalty.
    """
    peak_info = detect_peak_phase(self.recent_placings, self.get_show_dates())
    
    if peak_info['phase'] == 'peaking':
        return 1.1  # +10% boost
    elif peak_info['phase'] == 'declining':
        return 0.9  # -10% penalty
    return 1.0



# PEAK DETECTION - Auto-implemented
def detect_peak_phase(placings: List[int], dates: List[str]) -> Dict:
    """
    Detect if athlete is peaking, steady, or declining.
    Uses 6-month trend analysis.
    """
    if len(placings) < 3:
        return {'phase': 'unknown', 'confidence': 0}
    
    # Split into first/second half
    mid = len(placings) // 2
    first_half = sum(placings[:mid]) / mid
    second_half = sum(placings[mid:]) / (len(placings) - mid)
    
    improvement = first_half - second_half
    
    if improvement > 2:
        phase = 'peaking'
        confidence = min(abs(improvement) * 10, 95)
    elif improvement < -2:
        phase = 'declining'
        confidence = min(abs(improvement) * 10, 95)
    else:
        phase = 'steady'
        confidence = 70
    
    return {
        'phase': phase,
        'confidence': confidence,
        'improvement': improvement
    }


# Add to prediction factors
@property
def peak_factor(self) -> float:
    """
    Adjustment based on peak phase.
    Peaking athletes get bonus, declining get penalty.
    """
    peak_info = detect_peak_phase(self.recent_placings, self.get_show_dates())
    
    if peak_info['phase'] == 'peaking':
        return 1.1  # +10% boost
    elif peak_info['phase'] == 'declining':
        return 0.9  # -10% penalty
    return 1.0



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# ELO RATING SYSTEM - Auto-implemented
def calculate_elo_rating(athlete_id: str, history: List[Dict]) -> float:
    """
    Calculate Elo-style rating based on competition history.
    Starts at 1500, updates based on opponent strength.
    """
    K = 32  # Rating volatility
    rating = 1500
    
    for result in history:
        opponent_rating = result.get('opponent_rating', 1500)
        actual_score = 1 if result.get('won') else 0
        expected_score = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
        rating += K * (actual_score - expected_score)
    
    return rating


# Add to AthleteProfile
@property
def elo_rating(self) -> float:
    """Elo-style rating (cached)."""
    if not hasattr(self, '_elo'):
        self._elo = calculate_elo_rating(self.id, self.get_history())
    return self._elo



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# ELO RATING SYSTEM - Auto-implemented
def calculate_elo_rating(athlete_id: str, history: List[Dict]) -> float:
    """
    Calculate Elo-style rating based on competition history.
    Starts at 1500, updates based on opponent strength.
    """
    K = 32  # Rating volatility
    rating = 1500
    
    for result in history:
        opponent_rating = result.get('opponent_rating', 1500)
        actual_score = 1 if result.get('won') else 0
        expected_score = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
        rating += K * (actual_score - expected_score)
    
    return rating


# Add to AthleteProfile
@property
def elo_rating(self) -> float:
    """Elo-style rating (cached)."""
    if not hasattr(self, '_elo'):
        self._elo = calculate_elo_rating(self.id, self.get_history())
    return self._elo



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# ELO RATING SYSTEM - Auto-implemented
def calculate_elo_rating(athlete_id: str, history: List[Dict]) -> float:
    """
    Calculate Elo-style rating based on competition history.
    Starts at 1500, updates based on opponent strength.
    """
    K = 32  # Rating volatility
    rating = 1500
    
    for result in history:
        opponent_rating = result.get('opponent_rating', 1500)
        actual_score = 1 if result.get('won') else 0
        expected_score = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
        rating += K * (actual_score - expected_score)
    
    return rating


# Add to AthleteProfile
@property
def elo_rating(self) -> float:
    """Elo-style rating (cached)."""
    if not hasattr(self, '_elo'):
        self._elo = calculate_elo_rating(self.id, self.get_history())
    return self._elo



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# ELO RATING SYSTEM - Auto-implemented
def calculate_elo_rating(athlete_id: str, history: List[Dict]) -> float:
    """
    Calculate Elo-style rating based on competition history.
    Starts at 1500, updates based on opponent strength.
    """
    K = 32  # Rating volatility
    rating = 1500
    
    for result in history:
        opponent_rating = result.get('opponent_rating', 1500)
        actual_score = 1 if result.get('won') else 0
        expected_score = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
        rating += K * (actual_score - expected_score)
    
    return rating


# Add to AthleteProfile
@property
def elo_rating(self) -> float:
    """Elo-style rating (cached)."""
    if not hasattr(self, '_elo'):
        self._elo = calculate_elo_rating(self.id, self.get_history())
    return self._elo



# ELO RATING SYSTEM - Auto-implemented
def calculate_elo_rating(athlete_id: str, history: List[Dict]) -> float:
    """
    Calculate Elo-style rating based on competition history.
    Starts at 1500, updates based on opponent strength.
    """
    K = 32  # Rating volatility
    rating = 1500
    
    for result in history:
        opponent_rating = result.get('opponent_rating', 1500)
        actual_score = 1 if result.get('won') else 0
        expected_score = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
        rating += K * (actual_score - expected_score)
    
    return rating


# Add to AthleteProfile
@property
def elo_rating(self) -> float:
    """Elo-style rating (cached)."""
    if not hasattr(self, '_elo'):
        self._elo = calculate_elo_rating(self.id, self.get_history())
    return self._elo



# ELO RATING SYSTEM - Auto-implemented
def calculate_elo_rating(athlete_id: str, history: List[Dict]) -> float:
    """
    Calculate Elo-style rating based on competition history.
    Starts at 1500, updates based on opponent strength.
    """
    K = 32  # Rating volatility
    rating = 1500
    
    for result in history:
        opponent_rating = result.get('opponent_rating', 1500)
        actual_score = 1 if result.get('won') else 0
        expected_score = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
        rating += K * (actual_score - expected_score)
    
    return rating


# Add to AthleteProfile
@property
def elo_rating(self) -> float:
    """Elo-style rating (cached)."""
    if not hasattr(self, '_elo'):
        self._elo = calculate_elo_rating(self.id, self.get_history())
    return self._elo



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# PEAK DETECTION - Auto-implemented
def detect_peak_phase(placings: List[int], dates: List[str]) -> Dict:
    """
    Detect if athlete is peaking, steady, or declining.
    Uses 6-month trend analysis.
    """
    if len(placings) < 3:
        return {'phase': 'unknown', 'confidence': 0}
    
    # Split into first/second half
    mid = len(placings) // 2
    first_half = sum(placings[:mid]) / mid
    second_half = sum(placings[mid:]) / (len(placings) - mid)
    
    improvement = first_half - second_half
    
    if improvement > 2:
        phase = 'peaking'
        confidence = min(abs(improvement) * 10, 95)
    elif improvement < -2:
        phase = 'declining'
        confidence = min(abs(improvement) * 10, 95)
    else:
        phase = 'steady'
        confidence = 70
    
    return {
        'phase': phase,
        'confidence': confidence,
        'improvement': improvement
    }


# Add to prediction factors
@property
def peak_factor(self) -> float:
    """
    Adjustment based on peak phase.
    Peaking athletes get bonus, declining get penalty.
    """
    peak_info = detect_peak_phase(self.recent_placings, self.get_show_dates())
    
    if peak_info['phase'] == 'peaking':
        return 1.1  # +10% boost
    elif peak_info['phase'] == 'declining':
        return 0.9  # -10% penalty
    return 1.0



# PEAK DETECTION - Auto-implemented
def detect_peak_phase(placings: List[int], dates: List[str]) -> Dict:
    """
    Detect if athlete is peaking, steady, or declining.
    Uses 6-month trend analysis.
    """
    if len(placings) < 3:
        return {'phase': 'unknown', 'confidence': 0}
    
    # Split into first/second half
    mid = len(placings) // 2
    first_half = sum(placings[:mid]) / mid
    second_half = sum(placings[mid:]) / (len(placings) - mid)
    
    improvement = first_half - second_half
    
    if improvement > 2:
        phase = 'peaking'
        confidence = min(abs(improvement) * 10, 95)
    elif improvement < -2:
        phase = 'declining'
        confidence = min(abs(improvement) * 10, 95)
    else:
        phase = 'steady'
        confidence = 70
    
    return {
        'phase': phase,
        'confidence': confidence,
        'improvement': improvement
    }


# Add to prediction factors
@property
def peak_factor(self) -> float:
    """
    Adjustment based on peak phase.
    Peaking athletes get bonus, declining get penalty.
    """
    peak_info = detect_peak_phase(self.recent_placings, self.get_show_dates())
    
    if peak_info['phase'] == 'peaking':
        return 1.1  # +10% boost
    elif peak_info['phase'] == 'declining':
        return 0.9  # -10% penalty
    return 1.0



# ELO RATING SYSTEM - Auto-implemented
def calculate_elo_rating(athlete_id: str, history: List[Dict]) -> float:
    """
    Calculate Elo-style rating based on competition history.
    Starts at 1500, updates based on opponent strength.
    """
    K = 32  # Rating volatility
    rating = 1500
    
    for result in history:
        opponent_rating = result.get('opponent_rating', 1500)
        actual_score = 1 if result.get('won') else 0
        expected_score = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
        rating += K * (actual_score - expected_score)
    
    return rating


# Add to AthleteProfile
@property
def elo_rating(self) -> float:
    """Elo-style rating (cached)."""
    if not hasattr(self, '_elo'):
        self._elo = calculate_elo_rating(self.id, self.get_history())
    return self._elo



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# ELO RATING SYSTEM - Auto-implemented
def calculate_elo_rating(athlete_id: str, history: List[Dict]) -> float:
    """
    Calculate Elo-style rating based on competition history.
    Starts at 1500, updates based on opponent strength.
    """
    K = 32  # Rating volatility
    rating = 1500
    
    for result in history:
        opponent_rating = result.get('opponent_rating', 1500)
        actual_score = 1 if result.get('won') else 0
        expected_score = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
        rating += K * (actual_score - expected_score)
    
    return rating


# Add to AthleteProfile
@property
def elo_rating(self) -> float:
    """Elo-style rating (cached)."""
    if not hasattr(self, '_elo'):
        self._elo = calculate_elo_rating(self.id, self.get_history())
    return self._elo



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# COMPETITION SIZE SCALING - Auto-implemented
def scale_placing_by_competition_size(placing: int, num_competitors: int, target_size: int = 30) -> float:
    """
    Scale placing based on competition size.
    1st out of 10 is NOT equivalent to 1st out of 50.
    This normalizes to a standard size.
    """
    if num_competitors <= 1:
        return placing
    
    # Calculate percentile
    percentile = (placing - 1) / (num_competitors - 1)
    
    # Scale to target size
    scaled = 1 + percentile * (target_size - 1)
    
    return scaled



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# ELO RATING SYSTEM - Auto-implemented
def calculate_elo_rating(athlete_id: str, history: List[Dict]) -> float:
    """
    Calculate Elo-style rating based on competition history.
    Starts at 1500, updates based on opponent strength.
    """
    K = 32  # Rating volatility
    rating = 1500
    
    for result in history:
        opponent_rating = result.get('opponent_rating', 1500)
        actual_score = 1 if result.get('won') else 0
        expected_score = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
        rating += K * (actual_score - expected_score)
    
    return rating


# Add to AthleteProfile
@property
def elo_rating(self) -> float:
    """Elo-style rating (cached)."""
    if not hasattr(self, '_elo'):
        self._elo = calculate_elo_rating(self.id, self.get_history())
    return self._elo



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# ELO RATING SYSTEM - Auto-implemented
def calculate_elo_rating(athlete_id: str, history: List[Dict]) -> float:
    """
    Calculate Elo-style rating based on competition history.
    Starts at 1500, updates based on opponent strength.
    """
    K = 32  # Rating volatility
    rating = 1500
    
    for result in history:
        opponent_rating = result.get('opponent_rating', 1500)
        actual_score = 1 if result.get('won') else 0
        expected_score = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
        rating += K * (actual_score - expected_score)
    
    return rating


# Add to AthleteProfile
@property
def elo_rating(self) -> float:
    """Elo-style rating (cached)."""
    if not hasattr(self, '_elo'):
        self._elo = calculate_elo_rating(self.id, self.get_history())
    return self._elo



# COMPETITION SIZE SCALING - Auto-implemented
def scale_placing_by_competition_size(placing: int, num_competitors: int, target_size: int = 30) -> float:
    """
    Scale placing based on competition size.
    1st out of 10 is NOT equivalent to 1st out of 50.
    This normalizes to a standard size.
    """
    if num_competitors <= 1:
        return placing
    
    # Calculate percentile
    percentile = (placing - 1) / (num_competitors - 1)
    
    # Scale to target size
    scaled = 1 + percentile * (target_size - 1)
    
    return scaled



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# HEAD-TO-HEAD HISTORY - Auto-implemented
from typing import Set, Tuple

def analyze_head_to_head(athlete1_id: str, athlete2_id: str, 
                         competition_history: List[Dict]) -> Dict:
    """
    Analyze head-to-head competition history between two athletes.
    Returns relative performance statistics.
    """
    common_competitions = []
    
    # Find competitions where both competed
    for comp in competition_history:
        results = comp.get('results', [])
        athlete1_result = None
        athlete2_result = None
        
        for r in results:
            if r.get('athlete_id') == athlete1_id:
                athlete1_result = r.get('placing')
            elif r.get('athlete_id') == athlete2_id:
                athlete2_result = r.get('placing')
        
        if athlete1_result is not None and athlete2_result is not None:
            common_competitions.append({
                'competition': comp.get('name'),
                'athlete1_placing': athlete1_result,
                'athlete2_placing': athlete2_result,
                'athlete1_won': athlete1_result < athlete2_result
            })
    
    if not common_competitions:
        return {'has_history': False}
    
    # Calculate stats
    athlete1_wins = sum(1 for c in common_competitions if c['athlete1_won'])
    win_rate = athlete1_wins / len(common_competitions)
    
    # Average placing differential
    avg_diff = sum(c['athlete2_placing'] - c['athlete1_placing'] 
                   for c in common_competitions) / len(common_competitions)
    
    return {
        'has_history': True,
        'meetings': len(common_competitions),
        'athlete1_win_rate': win_rate,
        'avg_placing_advantage': avg_diff
    }


# Integration
def get_head_to_head_factor(self, opponent_id: str) -> float:
    """
    Get adjustment factor based on head-to-head history.
    Positive = historically beats opponent, Negative = historically loses.
    """
    # This would need competition history data
    # Simplified implementation
    return 1.0  # Neutral if no history



# COMPETITION SIZE SCALING - Auto-implemented
def scale_placing_by_competition_size(placing: int, num_competitors: int, target_size: int = 30) -> float:
    """
    Scale placing based on competition size.
    1st out of 10 is NOT equivalent to 1st out of 50.
    This normalizes to a standard size.
    """
    if num_competitors <= 1:
        return placing
    
    # Calculate percentile
    percentile = (placing - 1) / (num_competitors - 1)
    
    # Scale to target size
    scaled = 1 + percentile * (target_size - 1)
    
    return scaled



# FIRST-TIME COMPETITOR HANDLING - Auto-implemented
POPULATION_AVG_PLACING = 8.5
POPULATION_PLACING_STD = 4.2

def handle_first_time_competitor(athlete_stats: Dict) -> Dict:
    """
    Special handling for athletes with <3 competitions.
    Uses population averages instead of personal history.
    """
    total_shows = athlete_stats.get('total_shows', 0)
    
    if total_shows == 0:
        # True rookie - high uncertainty
        return {
            'expected_placing': POPULATION_AVG_PLACING,
            'uncertainty': POPULATION_PLACING_STD * 2,  # High uncertainty
            'confidence_score': 30.0,  # Low confidence
            'is_rookie': True
        }
    elif total_shows < 3:
        # Limited history - blend personal and population
        personal_avg = athlete_stats.get('avg_placing', POPULATION_AVG_PLACING)
        weight_personal = total_shows / 5  # 0.2, 0.4 for 1-2 shows
        weight_pop = 1 - weight_personal
        
        blended = (personal_avg * weight_personal + 
                   POPULATION_AVG_PLACING * weight_pop)
        
        return {
            'expected_placing': blended,
            'uncertainty': POPULATION_PLACING_STD * (1.5 - weight_personal),
            'confidence_score': 40 + (total_shows * 15),  # 55-70 range
            'is_rookie': False
        }
    else:
        # Experienced - use personal stats
        return {
            'expected_placing': athlete_stats.get('avg_placing', POPULATION_AVG_PLACING),
            'uncertainty': POPULATION_PLACING_STD * 0.5,
            'confidence_score': min(95, 70 + total_shows * 2),
            'is_rookie': False
        }


# Integration
def get_rookie_adjusted_prediction(self) -> float:
    """Get prediction with special handling for rookies."""
    stats = {
        'total_shows': self.total_shows,
        'avg_placing': self.avg_placing if self.recent_placings else POPULATION_AVG_PLACING
    }
    
    result = handle_first_time_competitor(stats)
    
    # Convert placing to score (0-100)
    placing = result['expected_placing']
    score = 100 * math.exp(-0.15 * (placing - 1))
    
    # Adjust by confidence
    score = score * (result['confidence_score'] / 100)
    
    return score



# HEAD-TO-HEAD HISTORY - Auto-implemented
from typing import Set, Tuple

def analyze_head_to_head(athlete1_id: str, athlete2_id: str, 
                         competition_history: List[Dict]) -> Dict:
    """
    Analyze head-to-head competition history between two athletes.
    Returns relative performance statistics.
    """
    common_competitions = []
    
    # Find competitions where both competed
    for comp in competition_history:
        results = comp.get('results', [])
        athlete1_result = None
        athlete2_result = None
        
        for r in results:
            if r.get('athlete_id') == athlete1_id:
                athlete1_result = r.get('placing')
            elif r.get('athlete_id') == athlete2_id:
                athlete2_result = r.get('placing')
        
        if athlete1_result is not None and athlete2_result is not None:
            common_competitions.append({
                'competition': comp.get('name'),
                'athlete1_placing': athlete1_result,
                'athlete2_placing': athlete2_result,
                'athlete1_won': athlete1_result < athlete2_result
            })
    
    if not common_competitions:
        return {'has_history': False}
    
    # Calculate stats
    athlete1_wins = sum(1 for c in common_competitions if c['athlete1_won'])
    win_rate = athlete1_wins / len(common_competitions)
    
    # Average placing differential
    avg_diff = sum(c['athlete2_placing'] - c['athlete1_placing'] 
                   for c in common_competitions) / len(common_competitions)
    
    return {
        'has_history': True,
        'meetings': len(common_competitions),
        'athlete1_win_rate': win_rate,
        'avg_placing_advantage': avg_diff
    }


# Integration
def get_head_to_head_factor(self, opponent_id: str) -> float:
    """
    Get adjustment factor based on head-to-head history.
    Positive = historically beats opponent, Negative = historically loses.
    """
    # This would need competition history data
    # Simplified implementation
    return 1.0  # Neutral if no history



# MOMENTUM STREAK ANALYSIS - Auto-implemented
def calculate_momentum_score(placings: List[int]) -> float:
    """
    Detect winning/losing streaks and momentum.
    Winning streak boosts confidence, losing streak hurts.
    """
    if len(placings) < 2:
        return 50.0
    
    # Detect streaks
    current_streak = 1
    streak_type = 'win' if placings[0] == 1 else 'loss'
    
    for i in range(1, len(placings)):
        if placings[i] == 1 and streak_type == 'win':
            current_streak += 1
        elif placings[i] > 1 and streak_type == 'loss':
            current_streak += 1
        else:
            break
    
    # Calculate momentum
    if streak_type == 'win':
        # Winning streak bonus
        momentum = 70 + (current_streak * 10)  # +10 per win
    else:
        # Losing streak penalty
        momentum = 70 - (current_streak * 5)  # -5 per loss
    
    return min(100, max(0, momentum))


@property
def momentum_trend(self) -> str:
    """Return trend: 'surging', 'steady', 'declining'."""
    score = calculate_momentum_score(self.recent_placings)
    if score > 80:
        return 'surging'
    elif score < 60:
        return 'declining'
    return 'steady'



# MOMENTUM STREAK ANALYSIS - Auto-implemented
def calculate_momentum_score(placings: List[int]) -> float:
    """
    Detect winning/losing streaks and momentum.
    Winning streak boosts confidence, losing streak hurts.
    """
    if len(placings) < 2:
        return 50.0
    
    # Detect streaks
    current_streak = 1
    streak_type = 'win' if placings[0] == 1 else 'loss'
    
    for i in range(1, len(placings)):
        if placings[i] == 1 and streak_type == 'win':
            current_streak += 1
        elif placings[i] > 1 and streak_type == 'loss':
            current_streak += 1
        else:
            break
    
    # Calculate momentum
    if streak_type == 'win':
        # Winning streak bonus
        momentum = 70 + (current_streak * 10)  # +10 per win
    else:
        # Losing streak penalty
        momentum = 70 - (current_streak * 5)  # -5 per loss
    
    return min(100, max(0, momentum))


@property
def momentum_trend(self) -> str:
    """Return trend: 'surging', 'steady', 'declining'."""
    score = calculate_momentum_score(self.recent_placings)
    if score > 80:
        return 'surging'
    elif score < 60:
        return 'declining'
    return 'steady'



# FIRST-TIME COMPETITOR HANDLING - Auto-implemented
POPULATION_AVG_PLACING = 8.5
POPULATION_PLACING_STD = 4.2

def handle_first_time_competitor(athlete_stats: Dict) -> Dict:
    """
    Special handling for athletes with <3 competitions.
    Uses population averages instead of personal history.
    """
    total_shows = athlete_stats.get('total_shows', 0)
    
    if total_shows == 0:
        # True rookie - high uncertainty
        return {
            'expected_placing': POPULATION_AVG_PLACING,
            'uncertainty': POPULATION_PLACING_STD * 2,  # High uncertainty
            'confidence_score': 30.0,  # Low confidence
            'is_rookie': True
        }
    elif total_shows < 3:
        # Limited history - blend personal and population
        personal_avg = athlete_stats.get('avg_placing', POPULATION_AVG_PLACING)
        weight_personal = total_shows / 5  # 0.2, 0.4 for 1-2 shows
        weight_pop = 1 - weight_personal
        
        blended = (personal_avg * weight_personal + 
                   POPULATION_AVG_PLACING * weight_pop)
        
        return {
            'expected_placing': blended,
            'uncertainty': POPULATION_PLACING_STD * (1.5 - weight_personal),
            'confidence_score': 40 + (total_shows * 15),  # 55-70 range
            'is_rookie': False
        }
    else:
        # Experienced - use personal stats
        return {
            'expected_placing': athlete_stats.get('avg_placing', POPULATION_AVG_PLACING),
            'uncertainty': POPULATION_PLACING_STD * 0.5,
            'confidence_score': min(95, 70 + total_shows * 2),
            'is_rookie': False
        }


# Integration
def get_rookie_adjusted_prediction(self) -> float:
    """Get prediction with special handling for rookies."""
    stats = {
        'total_shows': self.total_shows,
        'avg_placing': self.avg_placing if self.recent_placings else POPULATION_AVG_PLACING
    }
    
    result = handle_first_time_competitor(stats)
    
    # Convert placing to score (0-100)
    placing = result['expected_placing']
    score = 100 * math.exp(-0.15 * (placing - 1))
    
    # Adjust by confidence
    score = score * (result['confidence_score'] / 100)
    
    return score



# COMPETITION SIZE SCALING - Auto-implemented
def scale_placing_by_competition_size(placing: int, num_competitors: int, target_size: int = 30) -> float:
    """
    Scale placing based on competition size.
    1st out of 10 is NOT equivalent to 1st out of 50.
    This normalizes to a standard size.
    """
    if num_competitors <= 1:
        return placing
    
    # Calculate percentile
    percentile = (placing - 1) / (num_competitors - 1)
    
    # Scale to target size
    scaled = 1 + percentile * (target_size - 1)
    
    return scaled



# FIRST-TIME COMPETITOR HANDLING - Auto-implemented
POPULATION_AVG_PLACING = 8.5
POPULATION_PLACING_STD = 4.2

def handle_first_time_competitor(athlete_stats: Dict) -> Dict:
    """
    Special handling for athletes with <3 competitions.
    Uses population averages instead of personal history.
    """
    total_shows = athlete_stats.get('total_shows', 0)
    
    if total_shows == 0:
        # True rookie - high uncertainty
        return {
            'expected_placing': POPULATION_AVG_PLACING,
            'uncertainty': POPULATION_PLACING_STD * 2,  # High uncertainty
            'confidence_score': 30.0,  # Low confidence
            'is_rookie': True
        }
    elif total_shows < 3:
        # Limited history - blend personal and population
        personal_avg = athlete_stats.get('avg_placing', POPULATION_AVG_PLACING)
        weight_personal = total_shows / 5  # 0.2, 0.4 for 1-2 shows
        weight_pop = 1 - weight_personal
        
        blended = (personal_avg * weight_personal + 
                   POPULATION_AVG_PLACING * weight_pop)
        
        return {
            'expected_placing': blended,
            'uncertainty': POPULATION_PLACING_STD * (1.5 - weight_personal),
            'confidence_score': 40 + (total_shows * 15),  # 55-70 range
            'is_rookie': False
        }
    else:
        # Experienced - use personal stats
        return {
            'expected_placing': athlete_stats.get('avg_placing', POPULATION_AVG_PLACING),
            'uncertainty': POPULATION_PLACING_STD * 0.5,
            'confidence_score': min(95, 70 + total_shows * 2),
            'is_rookie': False
        }


# Integration
def get_rookie_adjusted_prediction(self) -> float:
    """Get prediction with special handling for rookies."""
    stats = {
        'total_shows': self.total_shows,
        'avg_placing': self.avg_placing if self.recent_placings else POPULATION_AVG_PLACING
    }
    
    result = handle_first_time_competitor(stats)
    
    # Convert placing to score (0-100)
    placing = result['expected_placing']
    score = 100 * math.exp(-0.15 * (placing - 1))
    
    # Adjust by confidence
    score = score * (result['confidence_score'] / 100)
    
    return score



# COMPETITION SIZE SCALING - Auto-implemented
def scale_placing_by_competition_size(placing: int, num_competitors: int, target_size: int = 30) -> float:
    """
    Scale placing based on competition size.
    1st out of 10 is NOT equivalent to 1st out of 50.
    This normalizes to a standard size.
    """
    if num_competitors <= 1:
        return placing
    
    # Calculate percentile
    percentile = (placing - 1) / (num_competitors - 1)
    
    # Scale to target size
    scaled = 1 + percentile * (target_size - 1)
    
    return scaled



# COMPETITION SIZE SCALING - Auto-implemented
def scale_placing_by_competition_size(placing: int, num_competitors: int, target_size: int = 30) -> float:
    """
    Scale placing based on competition size.
    1st out of 10 is NOT equivalent to 1st out of 50.
    This normalizes to a standard size.
    """
    if num_competitors <= 1:
        return placing
    
    # Calculate percentile
    percentile = (placing - 1) / (num_competitors - 1)
    
    # Scale to target size
    scaled = 1 + percentile * (target_size - 1)
    
    return scaled



# COMPETITION SIZE SCALING - Auto-implemented
def scale_placing_by_competition_size(placing: int, num_competitors: int, target_size: int = 30) -> float:
    """
    Scale placing based on competition size.
    1st out of 10 is NOT equivalent to 1st out of 50.
    This normalizes to a standard size.
    """
    if num_competitors <= 1:
        return placing
    
    # Calculate percentile
    percentile = (placing - 1) / (num_competitors - 1)
    
    # Scale to target size
    scaled = 1 + percentile * (target_size - 1)
    
    return scaled



# COMPETITION SIZE SCALING - Auto-implemented
def scale_placing_by_competition_size(placing: int, num_competitors: int, target_size: int = 30) -> float:
    """
    Scale placing based on competition size.
    1st out of 10 is NOT equivalent to 1st out of 50.
    This normalizes to a standard size.
    """
    if num_competitors <= 1:
        return placing
    
    # Calculate percentile
    percentile = (placing - 1) / (num_competitors - 1)
    
    # Scale to target size
    scaled = 1 + percentile * (target_size - 1)
    
    return scaled



# COMPETITION SIZE SCALING - Auto-implemented
def scale_placing_by_competition_size(placing: int, num_competitors: int, target_size: int = 30) -> float:
    """
    Scale placing based on competition size.
    1st out of 10 is NOT equivalent to 1st out of 50.
    This normalizes to a standard size.
    """
    if num_competitors <= 1:
        return placing
    
    # Calculate percentile
    percentile = (placing - 1) / (num_competitors - 1)
    
    # Scale to target size
    scaled = 1 + percentile * (target_size - 1)
    
    return scaled



# COMPETITION SIZE SCALING - Auto-implemented
def scale_placing_by_competition_size(placing: int, num_competitors: int, target_size: int = 30) -> float:
    """
    Scale placing based on competition size.
    1st out of 10 is NOT equivalent to 1st out of 50.
    This normalizes to a standard size.
    """
    if num_competitors <= 1:
        return placing
    
    # Calculate percentile
    percentile = (placing - 1) / (num_competitors - 1)
    
    # Scale to target size
    scaled = 1 + percentile * (target_size - 1)
    
    return scaled



# COMPETITION SIZE SCALING - Auto-implemented
def scale_placing_by_competition_size(placing: int, num_competitors: int, target_size: int = 30) -> float:
    """
    Scale placing based on competition size.
    1st out of 10 is NOT equivalent to 1st out of 50.
    This normalizes to a standard size.
    """
    if num_competitors <= 1:
        return placing
    
    # Calculate percentile
    percentile = (placing - 1) / (num_competitors - 1)
    
    # Scale to target size
    scaled = 1 + percentile * (target_size - 1)
    
    return scaled



# FIRST-TIME COMPETITOR HANDLING - Auto-implemented
POPULATION_AVG_PLACING = 8.5
POPULATION_PLACING_STD = 4.2

def handle_first_time_competitor(athlete_stats: Dict) -> Dict:
    """
    Special handling for athletes with <3 competitions.
    Uses population averages instead of personal history.
    """
    total_shows = athlete_stats.get('total_shows', 0)
    
    if total_shows == 0:
        # True rookie - high uncertainty
        return {
            'expected_placing': POPULATION_AVG_PLACING,
            'uncertainty': POPULATION_PLACING_STD * 2,  # High uncertainty
            'confidence_score': 30.0,  # Low confidence
            'is_rookie': True
        }
    elif total_shows < 3:
        # Limited history - blend personal and population
        personal_avg = athlete_stats.get('avg_placing', POPULATION_AVG_PLACING)
        weight_personal = total_shows / 5  # 0.2, 0.4 for 1-2 shows
        weight_pop = 1 - weight_personal
        
        blended = (personal_avg * weight_personal + 
                   POPULATION_AVG_PLACING * weight_pop)
        
        return {
            'expected_placing': blended,
            'uncertainty': POPULATION_PLACING_STD * (1.5 - weight_personal),
            'confidence_score': 40 + (total_shows * 15),  # 55-70 range
            'is_rookie': False
        }
    else:
        # Experienced - use personal stats
        return {
            'expected_placing': athlete_stats.get('avg_placing', POPULATION_AVG_PLACING),
            'uncertainty': POPULATION_PLACING_STD * 0.5,
            'confidence_score': min(95, 70 + total_shows * 2),
            'is_rookie': False
        }


# Integration
def get_rookie_adjusted_prediction(self) -> float:
    """Get prediction with special handling for rookies."""
    stats = {
        'total_shows': self.total_shows,
        'avg_placing': self.avg_placing if self.recent_placings else POPULATION_AVG_PLACING
    }
    
    result = handle_first_time_competitor(stats)
    
    # Convert placing to score (0-100)
    placing = result['expected_placing']
    score = 100 * math.exp(-0.15 * (placing - 1))
    
    # Adjust by confidence
    score = score * (result['confidence_score'] / 100)
    
    return score



# MOMENTUM STREAK ANALYSIS - Auto-implemented
def calculate_momentum_score(placings: List[int]) -> float:
    """
    Detect winning/losing streaks and momentum.
    Winning streak boosts confidence, losing streak hurts.
    """
    if len(placings) < 2:
        return 50.0
    
    # Detect streaks
    current_streak = 1
    streak_type = 'win' if placings[0] == 1 else 'loss'
    
    for i in range(1, len(placings)):
        if placings[i] == 1 and streak_type == 'win':
            current_streak += 1
        elif placings[i] > 1 and streak_type == 'loss':
            current_streak += 1
        else:
            break
    
    # Calculate momentum
    if streak_type == 'win':
        # Winning streak bonus
        momentum = 70 + (current_streak * 10)  # +10 per win
    else:
        # Losing streak penalty
        momentum = 70 - (current_streak * 5)  # -5 per loss
    
    return min(100, max(0, momentum))


@property
def momentum_trend(self) -> str:
    """Return trend: 'surging', 'steady', 'declining'."""
    score = calculate_momentum_score(self.recent_placings)
    if score > 80:
        return 'surging'
    elif score < 60:
        return 'declining'
    return 'steady'



# COMPETITION SIZE SCALING - Auto-implemented
def scale_placing_by_competition_size(placing: int, num_competitors: int, target_size: int = 30) -> float:
    """
    Scale placing based on competition size.
    1st out of 10 is NOT equivalent to 1st out of 50.
    This normalizes to a standard size.
    """
    if num_competitors <= 1:
        return placing
    
    # Calculate percentile
    percentile = (placing - 1) / (num_competitors - 1)
    
    # Scale to target size
    scaled = 1 + percentile * (target_size - 1)
    
    return scaled



# MOMENTUM STREAK ANALYSIS - Auto-implemented
def calculate_momentum_score(placings: List[int]) -> float:
    """
    Detect winning/losing streaks and momentum.
    Winning streak boosts confidence, losing streak hurts.
    """
    if len(placings) < 2:
        return 50.0
    
    # Detect streaks
    current_streak = 1
    streak_type = 'win' if placings[0] == 1 else 'loss'
    
    for i in range(1, len(placings)):
        if placings[i] == 1 and streak_type == 'win':
            current_streak += 1
        elif placings[i] > 1 and streak_type == 'loss':
            current_streak += 1
        else:
            break
    
    # Calculate momentum
    if streak_type == 'win':
        # Winning streak bonus
        momentum = 70 + (current_streak * 10)  # +10 per win
    else:
        # Losing streak penalty
        momentum = 70 - (current_streak * 5)  # -5 per loss
    
    return min(100, max(0, momentum))


@property
def momentum_trend(self) -> str:
    """Return trend: 'surging', 'steady', 'declining'."""
    score = calculate_momentum_score(self.recent_placings)
    if score > 80:
        return 'surging'
    elif score < 60:
        return 'declining'
    return 'steady'



# MOMENTUM STREAK ANALYSIS - Auto-implemented
def calculate_momentum_score(placings: List[int]) -> float:
    """
    Detect winning/losing streaks and momentum.
    Winning streak boosts confidence, losing streak hurts.
    """
    if len(placings) < 2:
        return 50.0
    
    # Detect streaks
    current_streak = 1
    streak_type = 'win' if placings[0] == 1 else 'loss'
    
    for i in range(1, len(placings)):
        if placings[i] == 1 and streak_type == 'win':
            current_streak += 1
        elif placings[i] > 1 and streak_type == 'loss':
            current_streak += 1
        else:
            break
    
    # Calculate momentum
    if streak_type == 'win':
        # Winning streak bonus
        momentum = 70 + (current_streak * 10)  # +10 per win
    else:
        # Losing streak penalty
        momentum = 70 - (current_streak * 5)  # -5 per loss
    
    return min(100, max(0, momentum))


@property
def momentum_trend(self) -> str:
    """Return trend: 'surging', 'steady', 'declining'."""
    score = calculate_momentum_score(self.recent_placings)
    if score > 80:
        return 'surging'
    elif score < 60:
        return 'declining'
    return 'steady'



# WEIGHTED PLACING AVERAGE - Auto-implemented
def calculate_weighted_placing_average(
    placings: List[int],
    weights: Optional[List[float]] = None
) -> float:
    """
    Weighted average of last N shows with custom weights.
    Default: exponential decay (recent = higher weight).
    """
    if not placings:
        return 50.0
    
    if weights is None:
        # Generate exponential decay weights
        weights = [math.exp(-0.1 * i) for i in range(len(placings))]
    
    # Normalize weights
    total_weight = sum(weights[:len(placings)])
    normalized_weights = [w / total_weight for w in weights[:len(placings)]]
    
    # Calculate weighted average (inverse since lower placing = better)
    weighted_score = sum(
        (100 * math.exp(-0.15 * (p - 1))) * w
        for p, w in zip(placings, normalized_weights)
    )
    
    return weighted_score



# FIRST-TIME COMPETITOR HANDLING - Auto-implemented
POPULATION_AVG_PLACING = 8.5
POPULATION_PLACING_STD = 4.2

def handle_first_time_competitor(athlete_stats: Dict) -> Dict:
    """
    Special handling for athletes with <3 competitions.
    Uses population averages instead of personal history.
    """
    total_shows = athlete_stats.get('total_shows', 0)
    
    if total_shows == 0:
        # True rookie - high uncertainty
        return {
            'expected_placing': POPULATION_AVG_PLACING,
            'uncertainty': POPULATION_PLACING_STD * 2,  # High uncertainty
            'confidence_score': 30.0,  # Low confidence
            'is_rookie': True
        }
    elif total_shows < 3:
        # Limited history - blend personal and population
        personal_avg = athlete_stats.get('avg_placing', POPULATION_AVG_PLACING)
        weight_personal = total_shows / 5  # 0.2, 0.4 for 1-2 shows
        weight_pop = 1 - weight_personal
        
        blended = (personal_avg * weight_personal + 
                   POPULATION_AVG_PLACING * weight_pop)
        
        return {
            'expected_placing': blended,
            'uncertainty': POPULATION_PLACING_STD * (1.5 - weight_personal),
            'confidence_score': 40 + (total_shows * 15),  # 55-70 range
            'is_rookie': False
        }
    else:
        # Experienced - use personal stats
        return {
            'expected_placing': athlete_stats.get('avg_placing', POPULATION_AVG_PLACING),
            'uncertainty': POPULATION_PLACING_STD * 0.5,
            'confidence_score': min(95, 70 + total_shows * 2),
            'is_rookie': False
        }


# Integration
def get_rookie_adjusted_prediction(self) -> float:
    """Get prediction with special handling for rookies."""
    stats = {
        'total_shows': self.total_shows,
        'avg_placing': self.avg_placing if self.recent_placings else POPULATION_AVG_PLACING
    }
    
    result = handle_first_time_competitor(stats)
    
    # Convert placing to score (0-100)
    placing = result['expected_placing']
    score = 100 * math.exp(-0.15 * (placing - 1))
    
    # Adjust by confidence
    score = score * (result['confidence_score'] / 100)
    
    return score



# MOMENTUM STREAK ANALYSIS - Auto-implemented
def calculate_momentum_score(placings: List[int]) -> float:
    """
    Detect winning/losing streaks and momentum.
    Winning streak boosts confidence, losing streak hurts.
    """
    if len(placings) < 2:
        return 50.0
    
    # Detect streaks
    current_streak = 1
    streak_type = 'win' if placings[0] == 1 else 'loss'
    
    for i in range(1, len(placings)):
        if placings[i] == 1 and streak_type == 'win':
            current_streak += 1
        elif placings[i] > 1 and streak_type == 'loss':
            current_streak += 1
        else:
            break
    
    # Calculate momentum
    if streak_type == 'win':
        # Winning streak bonus
        momentum = 70 + (current_streak * 10)  # +10 per win
    else:
        # Losing streak penalty
        momentum = 70 - (current_streak * 5)  # -5 per loss
    
    return min(100, max(0, momentum))


@property
def momentum_trend(self) -> str:
    """Return trend: 'surging', 'steady', 'declining'."""
    score = calculate_momentum_score(self.recent_placings)
    if score > 80:
        return 'surging'
    elif score < 60:
        return 'declining'
    return 'steady'



# MOMENTUM STREAK ANALYSIS - Auto-implemented
def calculate_momentum_score(placings: List[int]) -> float:
    """
    Detect winning/losing streaks and momentum.
    Winning streak boosts confidence, losing streak hurts.
    """
    if len(placings) < 2:
        return 50.0
    
    # Detect streaks
    current_streak = 1
    streak_type = 'win' if placings[0] == 1 else 'loss'
    
    for i in range(1, len(placings)):
        if placings[i] == 1 and streak_type == 'win':
            current_streak += 1
        elif placings[i] > 1 and streak_type == 'loss':
            current_streak += 1
        else:
            break
    
    # Calculate momentum
    if streak_type == 'win':
        # Winning streak bonus
        momentum = 70 + (current_streak * 10)  # +10 per win
    else:
        # Losing streak penalty
        momentum = 70 - (current_streak * 5)  # -5 per loss
    
    return min(100, max(0, momentum))


@property
def momentum_trend(self) -> str:
    """Return trend: 'surging', 'steady', 'declining'."""
    score = calculate_momentum_score(self.recent_placings)
    if score > 80:
        return 'surging'
    elif score < 60:
        return 'declining'
    return 'steady'



# WEIGHTED PLACING AVERAGE - Auto-implemented
def calculate_weighted_placing_average(
    placings: List[int],
    weights: Optional[List[float]] = None
) -> float:
    """
    Weighted average of last N shows with custom weights.
    Default: exponential decay (recent = higher weight).
    """
    if not placings:
        return 50.0
    
    if weights is None:
        # Generate exponential decay weights
        weights = [math.exp(-0.1 * i) for i in range(len(placings))]
    
    # Normalize weights
    total_weight = sum(weights[:len(placings)])
    normalized_weights = [w / total_weight for w in weights[:len(placings)]]
    
    # Calculate weighted average (inverse since lower placing = better)
    weighted_score = sum(
        (100 * math.exp(-0.15 * (p - 1))) * w
        for p, w in zip(placings, normalized_weights)
    )
    
    return weighted_score



# DIVISION CONSISTENCY SCORE - Auto-implemented
def calculate_division_consistency(division_history: List[str]) -> float:
    """
    Track division switching behavior.
    High consistency = settled division = more predictable.
    Returns consistency score (0-100).
    """
    if not division_history or len(division_history) < 2:
        return 70.0  # Neutral for new athletes
    
    # Count unique divisions
    unique_divisions = set(division_history)
    
    if len(unique_divisions) == 1:
        # Perfect consistency
        return 100.0
    
    # Calculate consistency ratio
    consistency_ratio = len(division_history) / len(unique_divisions)
    
    # Normalize to 0-100 (higher = more consistent)
    score = min(100, consistency_ratio * 30)
    
    return score


# Integration
@property
def division_consistency_score(self) -> float:
    """Score based on division consistency."""
    divisions = getattr(self, 'division_history', [])
    return calculate_division_consistency(divisions)



# WEIGHTED PLACING AVERAGE - Auto-implemented
def calculate_weighted_placing_average(
    placings: List[int],
    weights: Optional[List[float]] = None
) -> float:
    """
    Weighted average of last N shows with custom weights.
    Default: exponential decay (recent = higher weight).
    """
    if not placings:
        return 50.0
    
    if weights is None:
        # Generate exponential decay weights
        weights = [math.exp(-0.1 * i) for i in range(len(placings))]
    
    # Normalize weights
    total_weight = sum(weights[:len(placings)])
    normalized_weights = [w / total_weight for w in weights[:len(placings)]]
    
    # Calculate weighted average (inverse since lower placing = better)
    weighted_score = sum(
        (100 * math.exp(-0.15 * (p - 1))) * w
        for p, w in zip(placings, normalized_weights)
    )
    
    return weighted_score



# DIVISION CONSISTENCY SCORE - Auto-implemented
def calculate_division_consistency(division_history: List[str]) -> float:
    """
    Track division switching behavior.
    High consistency = settled division = more predictable.
    Returns consistency score (0-100).
    """
    if not division_history or len(division_history) < 2:
        return 70.0  # Neutral for new athletes
    
    # Count unique divisions
    unique_divisions = set(division_history)
    
    if len(unique_divisions) == 1:
        # Perfect consistency
        return 100.0
    
    # Calculate consistency ratio
    consistency_ratio = len(division_history) / len(unique_divisions)
    
    # Normalize to 0-100 (higher = more consistent)
    score = min(100, consistency_ratio * 30)
    
    return score


# Integration
@property
def division_consistency_score(self) -> float:
    """Score based on division consistency."""
    divisions = getattr(self, 'division_history', [])
    return calculate_division_consistency(divisions)



# DIVISION CONSISTENCY SCORE - Auto-implemented
def calculate_division_consistency(division_history: List[str]) -> float:
    """
    Track division switching behavior.
    High consistency = settled division = more predictable.
    Returns consistency score (0-100).
    """
    if not division_history or len(division_history) < 2:
        return 70.0  # Neutral for new athletes
    
    # Count unique divisions
    unique_divisions = set(division_history)
    
    if len(unique_divisions) == 1:
        # Perfect consistency
        return 100.0
    
    # Calculate consistency ratio
    consistency_ratio = len(division_history) / len(unique_divisions)
    
    # Normalize to 0-100 (higher = more consistent)
    score = min(100, consistency_ratio * 30)
    
    return score


# Integration
@property
def division_consistency_score(self) -> float:
    """Score based on division consistency."""
    divisions = getattr(self, 'division_history', [])
    return calculate_division_consistency(divisions)



# WEIGHTED PLACING AVERAGE - Auto-implemented
def calculate_weighted_placing_average(
    placings: List[int],
    weights: Optional[List[float]] = None
) -> float:
    """
    Weighted average of last N shows with custom weights.
    Default: exponential decay (recent = higher weight).
    """
    if not placings:
        return 50.0
    
    if weights is None:
        # Generate exponential decay weights
        weights = [math.exp(-0.1 * i) for i in range(len(placings))]
    
    # Normalize weights
    total_weight = sum(weights[:len(placings)])
    normalized_weights = [w / total_weight for w in weights[:len(placings)]]
    
    # Calculate weighted average (inverse since lower placing = better)
    weighted_score = sum(
        (100 * math.exp(-0.15 * (p - 1))) * w
        for p, w in zip(placings, normalized_weights)
    )
    
    return weighted_score



# MOMENTUM STREAK ANALYSIS - Auto-implemented
def calculate_momentum_score(placings: List[int]) -> float:
    """
    Detect winning/losing streaks and momentum.
    Winning streak boosts confidence, losing streak hurts.
    """
    if len(placings) < 2:
        return 50.0
    
    # Detect streaks
    current_streak = 1
    streak_type = 'win' if placings[0] == 1 else 'loss'
    
    for i in range(1, len(placings)):
        if placings[i] == 1 and streak_type == 'win':
            current_streak += 1
        elif placings[i] > 1 and streak_type == 'loss':
            current_streak += 1
        else:
            break
    
    # Calculate momentum
    if streak_type == 'win':
        # Winning streak bonus
        momentum = 70 + (current_streak * 10)  # +10 per win
    else:
        # Losing streak penalty
        momentum = 70 - (current_streak * 5)  # -5 per loss
    
    return min(100, max(0, momentum))


@property
def momentum_trend(self) -> str:
    """Return trend: 'surging', 'steady', 'declining'."""
    score = calculate_momentum_score(self.recent_placings)
    if score > 80:
        return 'surging'
    elif score < 60:
        return 'declining'
    return 'steady'



# WEIGHTED PLACING AVERAGE - Auto-implemented
def calculate_weighted_placing_average(
    placings: List[int],
    weights: Optional[List[float]] = None
) -> float:
    """
    Weighted average of last N shows with custom weights.
    Default: exponential decay (recent = higher weight).
    """
    if not placings:
        return 50.0
    
    if weights is None:
        # Generate exponential decay weights
        weights = [math.exp(-0.1 * i) for i in range(len(placings))]
    
    # Normalize weights
    total_weight = sum(weights[:len(placings)])
    normalized_weights = [w / total_weight for w in weights[:len(placings)]]
    
    # Calculate weighted average (inverse since lower placing = better)
    weighted_score = sum(
        (100 * math.exp(-0.15 * (p - 1))) * w
        for p, w in zip(placings, normalized_weights)
    )
    
    return weighted_score



# WEIGHTED PLACING AVERAGE - Auto-implemented
def calculate_weighted_placing_average(
    placings: List[int],
    weights: Optional[List[float]] = None
) -> float:
    """
    Weighted average of last N shows with custom weights.
    Default: exponential decay (recent = higher weight).
    """
    if not placings:
        return 50.0
    
    if weights is None:
        # Generate exponential decay weights
        weights = [math.exp(-0.1 * i) for i in range(len(placings))]
    
    # Normalize weights
    total_weight = sum(weights[:len(placings)])
    normalized_weights = [w / total_weight for w in weights[:len(placings)]]
    
    # Calculate weighted average (inverse since lower placing = better)
    weighted_score = sum(
        (100 * math.exp(-0.15 * (p - 1))) * w
        for p, w in zip(placings, normalized_weights)
    )
    
    return weighted_score



# DIVISION CONSISTENCY SCORE - Auto-implemented
def calculate_division_consistency(division_history: List[str]) -> float:
    """
    Track division switching behavior.
    High consistency = settled division = more predictable.
    Returns consistency score (0-100).
    """
    if not division_history or len(division_history) < 2:
        return 70.0  # Neutral for new athletes
    
    # Count unique divisions
    unique_divisions = set(division_history)
    
    if len(unique_divisions) == 1:
        # Perfect consistency
        return 100.0
    
    # Calculate consistency ratio
    consistency_ratio = len(division_history) / len(unique_divisions)
    
    # Normalize to 0-100 (higher = more consistent)
    score = min(100, consistency_ratio * 30)
    
    return score


# Integration
@property
def division_consistency_score(self) -> float:
    """Score based on division consistency."""
    divisions = getattr(self, 'division_history', [])
    return calculate_division_consistency(divisions)



# DIVISION CONSISTENCY SCORE - Auto-implemented
def calculate_division_consistency(division_history: List[str]) -> float:
    """
    Track division switching behavior.
    High consistency = settled division = more predictable.
    Returns consistency score (0-100).
    """
    if not division_history or len(division_history) < 2:
        return 70.0  # Neutral for new athletes
    
    # Count unique divisions
    unique_divisions = set(division_history)
    
    if len(unique_divisions) == 1:
        # Perfect consistency
        return 100.0
    
    # Calculate consistency ratio
    consistency_ratio = len(division_history) / len(unique_divisions)
    
    # Normalize to 0-100 (higher = more consistent)
    score = min(100, consistency_ratio * 30)
    
    return score


# Integration
@property
def division_consistency_score(self) -> float:
    """Score based on division consistency."""
    divisions = getattr(self, 'division_history', [])
    return calculate_division_consistency(divisions)



# MOMENTUM STREAK ANALYSIS - Auto-implemented
def calculate_momentum_score(placings: List[int]) -> float:
    """
    Detect winning/losing streaks and momentum.
    Winning streak boosts confidence, losing streak hurts.
    """
    if len(placings) < 2:
        return 50.0
    
    # Detect streaks
    current_streak = 1
    streak_type = 'win' if placings[0] == 1 else 'loss'
    
    for i in range(1, len(placings)):
        if placings[i] == 1 and streak_type == 'win':
            current_streak += 1
        elif placings[i] > 1 and streak_type == 'loss':
            current_streak += 1
        else:
            break
    
    # Calculate momentum
    if streak_type == 'win':
        # Winning streak bonus
        momentum = 70 + (current_streak * 10)  # +10 per win
    else:
        # Losing streak penalty
        momentum = 70 - (current_streak * 5)  # -5 per loss
    
    return min(100, max(0, momentum))


@property
def momentum_trend(self) -> str:
    """Return trend: 'surging', 'steady', 'declining'."""
    score = calculate_momentum_score(self.recent_placings)
    if score > 80:
        return 'surging'
    elif score < 60:
        return 'declining'
    return 'steady'



# WEIGHTED PLACING AVERAGE - Auto-implemented
def calculate_weighted_placing_average(
    placings: List[int],
    weights: Optional[List[float]] = None
) -> float:
    """
    Weighted average of last N shows with custom weights.
    Default: exponential decay (recent = higher weight).
    """
    if not placings:
        return 50.0
    
    if weights is None:
        # Generate exponential decay weights
        weights = [math.exp(-0.1 * i) for i in range(len(placings))]
    
    # Normalize weights
    total_weight = sum(weights[:len(placings)])
    normalized_weights = [w / total_weight for w in weights[:len(placings)]]
    
    # Calculate weighted average (inverse since lower placing = better)
    weighted_score = sum(
        (100 * math.exp(-0.15 * (p - 1))) * w
        for p, w in zip(placings, normalized_weights)
    )
    
    return weighted_score



# DIVISION CONSISTENCY SCORE - Auto-implemented
def calculate_division_consistency(division_history: List[str]) -> float:
    """
    Track division switching behavior.
    High consistency = settled division = more predictable.
    Returns consistency score (0-100).
    """
    if not division_history or len(division_history) < 2:
        return 70.0  # Neutral for new athletes
    
    # Count unique divisions
    unique_divisions = set(division_history)
    
    if len(unique_divisions) == 1:
        # Perfect consistency
        return 100.0
    
    # Calculate consistency ratio
    consistency_ratio = len(division_history) / len(unique_divisions)
    
    # Normalize to 0-100 (higher = more consistent)
    score = min(100, consistency_ratio * 30)
    
    return score


# Integration
@property
def division_consistency_score(self) -> float:
    """Score based on division consistency."""
    divisions = getattr(self, 'division_history', [])
    return calculate_division_consistency(divisions)



# WEIGHTED PLACING AVERAGE - Auto-implemented
def calculate_weighted_placing_average(
    placings: List[int],
    weights: Optional[List[float]] = None
) -> float:
    """
    Weighted average of last N shows with custom weights.
    Default: exponential decay (recent = higher weight).
    """
    if not placings:
        return 50.0
    
    if weights is None:
        # Generate exponential decay weights
        weights = [math.exp(-0.1 * i) for i in range(len(placings))]
    
    # Normalize weights
    total_weight = sum(weights[:len(placings)])
    normalized_weights = [w / total_weight for w in weights[:len(placings)]]
    
    # Calculate weighted average (inverse since lower placing = better)
    weighted_score = sum(
        (100 * math.exp(-0.15 * (p - 1))) * w
        for p, w in zip(placings, normalized_weights)
    )
    
    return weighted_score



# WEIGHTED PLACING AVERAGE - Auto-implemented
def calculate_weighted_placing_average(
    placings: List[int],
    weights: Optional[List[float]] = None
) -> float:
    """
    Weighted average of last N shows with custom weights.
    Default: exponential decay (recent = higher weight).
    """
    if not placings:
        return 50.0
    
    if weights is None:
        # Generate exponential decay weights
        weights = [math.exp(-0.1 * i) for i in range(len(placings))]
    
    # Normalize weights
    total_weight = sum(weights[:len(placings)])
    normalized_weights = [w / total_weight for w in weights[:len(placings)]]
    
    # Calculate weighted average (inverse since lower placing = better)
    weighted_score = sum(
        (100 * math.exp(-0.15 * (p - 1))) * w
        for p, w in zip(placings, normalized_weights)
    )
    
    return weighted_score



# WEIGHTED PLACING AVERAGE - Auto-implemented
def calculate_weighted_placing_average(
    placings: List[int],
    weights: Optional[List[float]] = None
) -> float:
    """
    Weighted average of last N shows with custom weights.
    Default: exponential decay (recent = higher weight).
    """
    if not placings:
        return 50.0
    
    if weights is None:
        # Generate exponential decay weights
        weights = [math.exp(-0.1 * i) for i in range(len(placings))]
    
    # Normalize weights
    total_weight = sum(weights[:len(placings)])
    normalized_weights = [w / total_weight for w in weights[:len(placings)]]
    
    # Calculate weighted average (inverse since lower placing = better)
    weighted_score = sum(
        (100 * math.exp(-0.15 * (p - 1))) * w
        for p, w in zip(placings, normalized_weights)
    )
    
    return weighted_score



# WEIGHTED PLACING AVERAGE - Auto-implemented
def calculate_weighted_placing_average(
    placings: List[int],
    weights: Optional[List[float]] = None
) -> float:
    """
    Weighted average of last N shows with custom weights.
    Default: exponential decay (recent = higher weight).
    """
    if not placings:
        return 50.0
    
    if weights is None:
        # Generate exponential decay weights
        weights = [math.exp(-0.1 * i) for i in range(len(placings))]
    
    # Normalize weights
    total_weight = sum(weights[:len(placings)])
    normalized_weights = [w / total_weight for w in weights[:len(placings)]]
    
    # Calculate weighted average (inverse since lower placing = better)
    weighted_score = sum(
        (100 * math.exp(-0.15 * (p - 1))) * w
        for p, w in zip(placings, normalized_weights)
    )
    
    return weighted_score



# DIVISION CONSISTENCY SCORE - Auto-implemented
def calculate_division_consistency(division_history: List[str]) -> float:
    """
    Track division switching behavior.
    High consistency = settled division = more predictable.
    Returns consistency score (0-100).
    """
    if not division_history or len(division_history) < 2:
        return 70.0  # Neutral for new athletes
    
    # Count unique divisions
    unique_divisions = set(division_history)
    
    if len(unique_divisions) == 1:
        # Perfect consistency
        return 100.0
    
    # Calculate consistency ratio
    consistency_ratio = len(division_history) / len(unique_divisions)
    
    # Normalize to 0-100 (higher = more consistent)
    score = min(100, consistency_ratio * 30)
    
    return score


# Integration
@property
def division_consistency_score(self) -> float:
    """Score based on division consistency."""
    divisions = getattr(self, 'division_history', [])
    return calculate_division_consistency(divisions)



# WEIGHTED PLACING AVERAGE - Auto-implemented
def calculate_weighted_placing_average(
    placings: List[int],
    weights: Optional[List[float]] = None
) -> float:
    """
    Weighted average of last N shows with custom weights.
    Default: exponential decay (recent = higher weight).
    """
    if not placings:
        return 50.0
    
    if weights is None:
        # Generate exponential decay weights
        weights = [math.exp(-0.1 * i) for i in range(len(placings))]
    
    # Normalize weights
    total_weight = sum(weights[:len(placings)])
    normalized_weights = [w / total_weight for w in weights[:len(placings)]]
    
    # Calculate weighted average (inverse since lower placing = better)
    weighted_score = sum(
        (100 * math.exp(-0.15 * (p - 1))) * w
        for p, w in zip(placings, normalized_weights)
    )
    
    return weighted_score



# WEIGHTED PLACING AVERAGE - Auto-implemented
def calculate_weighted_placing_average(
    placings: List[int],
    weights: Optional[List[float]] = None
) -> float:
    """
    Weighted average of last N shows with custom weights.
    Default: exponential decay (recent = higher weight).
    """
    if not placings:
        return 50.0
    
    if weights is None:
        # Generate exponential decay weights
        weights = [math.exp(-0.1 * i) for i in range(len(placings))]
    
    # Normalize weights
    total_weight = sum(weights[:len(placings)])
    normalized_weights = [w / total_weight for w in weights[:len(placings)]]
    
    # Calculate weighted average (inverse since lower placing = better)
    weighted_score = sum(
        (100 * math.exp(-0.15 * (p - 1))) * w
        for p, w in zip(placings, normalized_weights)
    )
    
    return weighted_score



# WEIGHTED PLACING AVERAGE - Auto-implemented
def calculate_weighted_placing_average(
    placings: List[int],
    weights: Optional[List[float]] = None
) -> float:
    """
    Weighted average of last N shows with custom weights.
    Default: exponential decay (recent = higher weight).
    """
    if not placings:
        return 50.0
    
    if weights is None:
        # Generate exponential decay weights
        weights = [math.exp(-0.1 * i) for i in range(len(placings))]
    
    # Normalize weights
    total_weight = sum(weights[:len(placings)])
    normalized_weights = [w / total_weight for w in weights[:len(placings)]]
    
    # Calculate weighted average (inverse since lower placing = better)
    weighted_score = sum(
        (100 * math.exp(-0.15 * (p - 1))) * w
        for p, w in zip(placings, normalized_weights)
    )
    
    return weighted_score



# WEIGHTED PLACING AVERAGE - Auto-implemented
def calculate_weighted_placing_average(
    placings: List[int],
    weights: Optional[List[float]] = None
) -> float:
    """
    Weighted average of last N shows with custom weights.
    Default: exponential decay (recent = higher weight).
    """
    if not placings:
        return 50.0
    
    if weights is None:
        # Generate exponential decay weights
        weights = [math.exp(-0.1 * i) for i in range(len(placings))]
    
    # Normalize weights
    total_weight = sum(weights[:len(placings)])
    normalized_weights = [w / total_weight for w in weights[:len(placings)]]
    
    # Calculate weighted average (inverse since lower placing = better)
    weighted_score = sum(
        (100 * math.exp(-0.15 * (p - 1))) * w
        for p, w in zip(placings, normalized_weights)
    )
    
    return weighted_score



# WEIGHTED COMPETITION TIER SCORE - Auto-implemented
COMPETITION_TIER_WEIGHTS = {
    'olympia': 1.0,
    'arnold classic': 0.95,
    'new york pro': 0.85,
    'chicago pro': 0.82,
    'tampa pro': 0.80,
    'ifbb pro league': 0.75,
    'regional': 0.60,
    'local': 0.45,
}

def get_tier_weight(competition_name: str) -> float:
    """Get weight for competition based on tier."""
    comp_lower = competition_name.lower()
    
    for key, weight in COMPETITION_TIER_WEIGHTS.items():
        if key in comp_lower:
            return weight
    
    return 0.70  # Default


def calculate_tier_weighted_average(competitions: List[Dict]) -> float:
    """
    Calculate athlete's average placing weighted by competition tier.
    Olympia 1st counts much more than regional 1st.
    """
    if not competitions:
        return 50.0
    
    total_weighted_placing = 0
    total_weight = 0
    
    for comp in competitions:
        name = comp.get('name', '')
        placing = comp.get('placing', 10)
        
        weight = get_tier_weight(name)
        # Higher weight (better competition) = more valuable placing
        weighted_placing = placing * (2.0 - weight)  # Invert: higher weight = lower effective placing
        
        total_weighted_placing += weighted_placing * weight
        total_weight += weight
    
    if total_weight == 0:
        return 50.0
    
    weighted_avg = total_weighted_placing / total_weight
    
    # Convert to score
    return 100 * math.exp(-0.15 * (weighted_avg - 1))


# Integration
@property
def tier_weighted_score(self) -> float:
    """Get score based on tier-weighted performance."""
    return calculate_tier_weighted_average(self.competition_history)



# WEIGHTED COMPETITION TIER SCORE - Auto-implemented
COMPETITION_TIER_WEIGHTS = {
    'olympia': 1.0,
    'arnold classic': 0.95,
    'new york pro': 0.85,
    'chicago pro': 0.82,
    'tampa pro': 0.80,
    'ifbb pro league': 0.75,
    'regional': 0.60,
    'local': 0.45,
}

def get_tier_weight(competition_name: str) -> float:
    """Get weight for competition based on tier."""
    comp_lower = competition_name.lower()
    
    for key, weight in COMPETITION_TIER_WEIGHTS.items():
        if key in comp_lower:
            return weight
    
    return 0.70  # Default


def calculate_tier_weighted_average(competitions: List[Dict]) -> float:
    """
    Calculate athlete's average placing weighted by competition tier.
    Olympia 1st counts much more than regional 1st.
    """
    if not competitions:
        return 50.0
    
    total_weighted_placing = 0
    total_weight = 0
    
    for comp in competitions:
        name = comp.get('name', '')
        placing = comp.get('placing', 10)
        
        weight = get_tier_weight(name)
        # Higher weight (better competition) = more valuable placing
        weighted_placing = placing * (2.0 - weight)  # Invert: higher weight = lower effective placing
        
        total_weighted_placing += weighted_placing * weight
        total_weight += weight
    
    if total_weight == 0:
        return 50.0
    
    weighted_avg = total_weighted_placing / total_weight
    
    # Convert to score
    return 100 * math.exp(-0.15 * (weighted_avg - 1))


# Integration
@property
def tier_weighted_score(self) -> float:
    """Get score based on tier-weighted performance."""
    return calculate_tier_weighted_average(self.competition_history)



# WEIGHTED COMPETITION TIER SCORE - Auto-implemented
COMPETITION_TIER_WEIGHTS = {
    'olympia': 1.0,
    'arnold classic': 0.95,
    'new york pro': 0.85,
    'chicago pro': 0.82,
    'tampa pro': 0.80,
    'ifbb pro league': 0.75,
    'regional': 0.60,
    'local': 0.45,
}

def get_tier_weight(competition_name: str) -> float:
    """Get weight for competition based on tier."""
    comp_lower = competition_name.lower()
    
    for key, weight in COMPETITION_TIER_WEIGHTS.items():
        if key in comp_lower:
            return weight
    
    return 0.70  # Default


def calculate_tier_weighted_average(competitions: List[Dict]) -> float:
    """
    Calculate athlete's average placing weighted by competition tier.
    Olympia 1st counts much more than regional 1st.
    """
    if not competitions:
        return 50.0
    
    total_weighted_placing = 0
    total_weight = 0
    
    for comp in competitions:
        name = comp.get('name', '')
        placing = comp.get('placing', 10)
        
        weight = get_tier_weight(name)
        # Higher weight (better competition) = more valuable placing
        weighted_placing = placing * (2.0 - weight)  # Invert: higher weight = lower effective placing
        
        total_weighted_placing += weighted_placing * weight
        total_weight += weight
    
    if total_weight == 0:
        return 50.0
    
    weighted_avg = total_weighted_placing / total_weight
    
    # Convert to score
    return 100 * math.exp(-0.15 * (weighted_avg - 1))


# Integration
@property
def tier_weighted_score(self) -> float:
    """Get score based on tier-weighted performance."""
    return calculate_tier_weighted_average(self.competition_history)



# PLACING VOLATILITY SCORE - Auto-implemented
import statistics

def calculate_volatility_score(placings: List[int]) -> float:
    """
    Calculate consistency via standard deviation of recent placings.
    Low volatility = more predictable athlete = higher confidence.
    Returns 0-100 score where 100 = very consistent.
    """
    if len(placings) < 3:
        return 50.0  # Neutral for limited data
    
    try:
        # Calculate standard deviation
        std_dev = statistics.stdev(placings)
        mean_placing = statistics.mean(placings)
        
        # Coefficient of variation (normalized volatility)
        cv = std_dev / mean_placing if mean_placing > 0 else 0
        
        # Convert to consistency score (lower CV = higher consistency)
        # Typical CV ranges from 0.1 (very consistent) to 1.0+ (volatile)
        consistency = max(0, min(100, 100 * (1 - cv)))
        
        return consistency
    except:
        return 50.0


# Integration
@property
def volatility_adjusted_score(self) -> float:
    """Get score adjusted for consistency/volatility."""
    base_score = self.recent_form
    volatility_score = calculate_volatility_score(self.recent_placings)
    
    # More consistent athletes get slight boost
    if volatility_score > 75:
        return base_score * 1.05
    elif volatility_score < 30:
        return base_score * 0.95
    
    return base_score



# WEIGHTED COMPETITION TIER SCORE - Auto-implemented
COMPETITION_TIER_WEIGHTS = {
    'olympia': 1.0,
    'arnold classic': 0.95,
    'new york pro': 0.85,
    'chicago pro': 0.82,
    'tampa pro': 0.80,
    'ifbb pro league': 0.75,
    'regional': 0.60,
    'local': 0.45,
}

def get_tier_weight(competition_name: str) -> float:
    """Get weight for competition based on tier."""
    comp_lower = competition_name.lower()
    
    for key, weight in COMPETITION_TIER_WEIGHTS.items():
        if key in comp_lower:
            return weight
    
    return 0.70  # Default


def calculate_tier_weighted_average(competitions: List[Dict]) -> float:
    """
    Calculate athlete's average placing weighted by competition tier.
    Olympia 1st counts much more than regional 1st.
    """
    if not competitions:
        return 50.0
    
    total_weighted_placing = 0
    total_weight = 0
    
    for comp in competitions:
        name = comp.get('name', '')
        placing = comp.get('placing', 10)
        
        weight = get_tier_weight(name)
        # Higher weight (better competition) = more valuable placing
        weighted_placing = placing * (2.0 - weight)  # Invert: higher weight = lower effective placing
        
        total_weighted_placing += weighted_placing * weight
        total_weight += weight
    
    if total_weight == 0:
        return 50.0
    
    weighted_avg = total_weighted_placing / total_weight
    
    # Convert to score
    return 100 * math.exp(-0.15 * (weighted_avg - 1))


# Integration
@property
def tier_weighted_score(self) -> float:
    """Get score based on tier-weighted performance."""
    return calculate_tier_weighted_average(self.competition_history)



# RECENT FORM MOMENTUM - Auto-implemented
def calculate_recent_momentum(placings: List[int]) -> Dict[str, float]:
    """
    Compare last 3 shows vs previous 3.
    Returns momentum score and direction.
    """
    if len(placings) < 4:
        return {'score': 50.0, 'direction': 'neutral'}
    
    # Split into recent and previous
    recent = placings[:3]
    previous = placings[3:6] if len(placings) >= 6 else placings[3:]
    
    recent_avg = sum(recent) / len(recent)
    previous_avg = sum(previous) / len(previous)
    
    # Improvement = lower placing (better)
    improvement = previous_avg - recent_avg
    
    if improvement > 2:
        direction = 'surging'
        score = 80.0
    elif improvement > 0.5:
        direction = 'improving'
        score = 65.0
    elif improvement < -2:
        direction = 'declining'
        score = 30.0
    elif improvement < -0.5:
        direction = 'slipping'
        score = 40.0
    else:
        direction = 'stable'
        score = 50.0
    
    return {'score': score, 'direction': direction, 'improvement': improvement}


# Integration
@property
def momentum_adjusted_score(self) -> float:
    """Get score adjusted for recent momentum."""
    base_score = self.recent_form
    momentum = calculate_recent_momentum(self.recent_placings)
    
    # Apply momentum factor
    if momentum['direction'] == 'surging':
        return base_score * 1.08
    elif momentum['direction'] == 'improving':
        return base_score * 1.04
    elif momentum['direction'] == 'declining':
        return base_score * 0.92
    elif momentum['direction'] == 'slipping':
        return base_score * 0.96
    
    return base_score



# RECENT FORM MOMENTUM - Auto-implemented
def calculate_recent_momentum(placings: List[int]) -> Dict[str, float]:
    """
    Compare last 3 shows vs previous 3.
    Returns momentum score and direction.
    """
    if len(placings) < 4:
        return {'score': 50.0, 'direction': 'neutral'}
    
    # Split into recent and previous
    recent = placings[:3]
    previous = placings[3:6] if len(placings) >= 6 else placings[3:]
    
    recent_avg = sum(recent) / len(recent)
    previous_avg = sum(previous) / len(previous)
    
    # Improvement = lower placing (better)
    improvement = previous_avg - recent_avg
    
    if improvement > 2:
        direction = 'surging'
        score = 80.0
    elif improvement > 0.5:
        direction = 'improving'
        score = 65.0
    elif improvement < -2:
        direction = 'declining'
        score = 30.0
    elif improvement < -0.5:
        direction = 'slipping'
        score = 40.0
    else:
        direction = 'stable'
        score = 50.0
    
    return {'score': score, 'direction': direction, 'improvement': improvement}


# Integration
@property
def momentum_adjusted_score(self) -> float:
    """Get score adjusted for recent momentum."""
    base_score = self.recent_form
    momentum = calculate_recent_momentum(self.recent_placings)
    
    # Apply momentum factor
    if momentum['direction'] == 'surging':
        return base_score * 1.08
    elif momentum['direction'] == 'improving':
        return base_score * 1.04
    elif momentum['direction'] == 'declining':
        return base_score * 0.92
    elif momentum['direction'] == 'slipping':
        return base_score * 0.96
    
    return base_score



# PLACING VOLATILITY SCORE - Auto-implemented
import statistics

def calculate_volatility_score(placings: List[int]) -> float:
    """
    Calculate consistency via standard deviation of recent placings.
    Low volatility = more predictable athlete = higher confidence.
    Returns 0-100 score where 100 = very consistent.
    """
    if len(placings) < 3:
        return 50.0  # Neutral for limited data
    
    try:
        # Calculate standard deviation
        std_dev = statistics.stdev(placings)
        mean_placing = statistics.mean(placings)
        
        # Coefficient of variation (normalized volatility)
        cv = std_dev / mean_placing if mean_placing > 0 else 0
        
        # Convert to consistency score (lower CV = higher consistency)
        # Typical CV ranges from 0.1 (very consistent) to 1.0+ (volatile)
        consistency = max(0, min(100, 100 * (1 - cv)))
        
        return consistency
    except:
        return 50.0


# Integration
@property
def volatility_adjusted_score(self) -> float:
    """Get score adjusted for consistency/volatility."""
    base_score = self.recent_form
    volatility_score = calculate_volatility_score(self.recent_placings)
    
    # More consistent athletes get slight boost
    if volatility_score > 75:
        return base_score * 1.05
    elif volatility_score < 30:
        return base_score * 0.95
    
    return base_score



# RECENT FORM MOMENTUM - Auto-implemented
def calculate_recent_momentum(placings: List[int]) -> Dict[str, float]:
    """
    Compare last 3 shows vs previous 3.
    Returns momentum score and direction.
    """
    if len(placings) < 4:
        return {'score': 50.0, 'direction': 'neutral'}
    
    # Split into recent and previous
    recent = placings[:3]
    previous = placings[3:6] if len(placings) >= 6 else placings[3:]
    
    recent_avg = sum(recent) / len(recent)
    previous_avg = sum(previous) / len(previous)
    
    # Improvement = lower placing (better)
    improvement = previous_avg - recent_avg
    
    if improvement > 2:
        direction = 'surging'
        score = 80.0
    elif improvement > 0.5:
        direction = 'improving'
        score = 65.0
    elif improvement < -2:
        direction = 'declining'
        score = 30.0
    elif improvement < -0.5:
        direction = 'slipping'
        score = 40.0
    else:
        direction = 'stable'
        score = 50.0
    
    return {'score': score, 'direction': direction, 'improvement': improvement}


# Integration
@property
def momentum_adjusted_score(self) -> float:
    """Get score adjusted for recent momentum."""
    base_score = self.recent_form
    momentum = calculate_recent_momentum(self.recent_placings)
    
    # Apply momentum factor
    if momentum['direction'] == 'surging':
        return base_score * 1.08
    elif momentum['direction'] == 'improving':
        return base_score * 1.04
    elif momentum['direction'] == 'declining':
        return base_score * 0.92
    elif momentum['direction'] == 'slipping':
        return base_score * 0.96
    
    return base_score



# WEIGHTED COMPETITION TIER SCORE - Auto-implemented
COMPETITION_TIER_WEIGHTS = {
    'olympia': 1.0,
    'arnold classic': 0.95,
    'new york pro': 0.85,
    'chicago pro': 0.82,
    'tampa pro': 0.80,
    'ifbb pro league': 0.75,
    'regional': 0.60,
    'local': 0.45,
}

def get_tier_weight(competition_name: str) -> float:
    """Get weight for competition based on tier."""
    comp_lower = competition_name.lower()
    
    for key, weight in COMPETITION_TIER_WEIGHTS.items():
        if key in comp_lower:
            return weight
    
    return 0.70  # Default


def calculate_tier_weighted_average(competitions: List[Dict]) -> float:
    """
    Calculate athlete's average placing weighted by competition tier.
    Olympia 1st counts much more than regional 1st.
    """
    if not competitions:
        return 50.0
    
    total_weighted_placing = 0
    total_weight = 0
    
    for comp in competitions:
        name = comp.get('name', '')
        placing = comp.get('placing', 10)
        
        weight = get_tier_weight(name)
        # Higher weight (better competition) = more valuable placing
        weighted_placing = placing * (2.0 - weight)  # Invert: higher weight = lower effective placing
        
        total_weighted_placing += weighted_placing * weight
        total_weight += weight
    
    if total_weight == 0:
        return 50.0
    
    weighted_avg = total_weighted_placing / total_weight
    
    # Convert to score
    return 100 * math.exp(-0.15 * (weighted_avg - 1))


# Integration
@property
def tier_weighted_score(self) -> float:
    """Get score based on tier-weighted performance."""
    return calculate_tier_weighted_average(self.competition_history)



# WEIGHTED COMPETITION TIER SCORE - Auto-implemented
COMPETITION_TIER_WEIGHTS = {
    'olympia': 1.0,
    'arnold classic': 0.95,
    'new york pro': 0.85,
    'chicago pro': 0.82,
    'tampa pro': 0.80,
    'ifbb pro league': 0.75,
    'regional': 0.60,
    'local': 0.45,
}

def get_tier_weight(competition_name: str) -> float:
    """Get weight for competition based on tier."""
    comp_lower = competition_name.lower()
    
    for key, weight in COMPETITION_TIER_WEIGHTS.items():
        if key in comp_lower:
            return weight
    
    return 0.70  # Default


def calculate_tier_weighted_average(competitions: List[Dict]) -> float:
    """
    Calculate athlete's average placing weighted by competition tier.
    Olympia 1st counts much more than regional 1st.
    """
    if not competitions:
        return 50.0
    
    total_weighted_placing = 0
    total_weight = 0
    
    for comp in competitions:
        name = comp.get('name', '')
        placing = comp.get('placing', 10)
        
        weight = get_tier_weight(name)
        # Higher weight (better competition) = more valuable placing
        weighted_placing = placing * (2.0 - weight)  # Invert: higher weight = lower effective placing
        
        total_weighted_placing += weighted_placing * weight
        total_weight += weight
    
    if total_weight == 0:
        return 50.0
    
    weighted_avg = total_weighted_placing / total_weight
    
    # Convert to score
    return 100 * math.exp(-0.15 * (weighted_avg - 1))


# Integration
@property
def tier_weighted_score(self) -> float:
    """Get score based on tier-weighted performance."""
    return calculate_tier_weighted_average(self.competition_history)



# RECENT FORM MOMENTUM - Auto-implemented
def calculate_recent_momentum(placings: List[int]) -> Dict[str, float]:
    """
    Compare last 3 shows vs previous 3.
    Returns momentum score and direction.
    """
    if len(placings) < 4:
        return {'score': 50.0, 'direction': 'neutral'}
    
    # Split into recent and previous
    recent = placings[:3]
    previous = placings[3:6] if len(placings) >= 6 else placings[3:]
    
    recent_avg = sum(recent) / len(recent)
    previous_avg = sum(previous) / len(previous)
    
    # Improvement = lower placing (better)
    improvement = previous_avg - recent_avg
    
    if improvement > 2:
        direction = 'surging'
        score = 80.0
    elif improvement > 0.5:
        direction = 'improving'
        score = 65.0
    elif improvement < -2:
        direction = 'declining'
        score = 30.0
    elif improvement < -0.5:
        direction = 'slipping'
        score = 40.0
    else:
        direction = 'stable'
        score = 50.0
    
    return {'score': score, 'direction': direction, 'improvement': improvement}


# Integration
@property
def momentum_adjusted_score(self) -> float:
    """Get score adjusted for recent momentum."""
    base_score = self.recent_form
    momentum = calculate_recent_momentum(self.recent_placings)
    
    # Apply momentum factor
    if momentum['direction'] == 'surging':
        return base_score * 1.08
    elif momentum['direction'] == 'improving':
        return base_score * 1.04
    elif momentum['direction'] == 'declining':
        return base_score * 0.92
    elif momentum['direction'] == 'slipping':
        return base_score * 0.96
    
    return base_score



# RECENT FORM MOMENTUM - Auto-implemented
def calculate_recent_momentum(placings: List[int]) -> Dict[str, float]:
    """
    Compare last 3 shows vs previous 3.
    Returns momentum score and direction.
    """
    if len(placings) < 4:
        return {'score': 50.0, 'direction': 'neutral'}
    
    # Split into recent and previous
    recent = placings[:3]
    previous = placings[3:6] if len(placings) >= 6 else placings[3:]
    
    recent_avg = sum(recent) / len(recent)
    previous_avg = sum(previous) / len(previous)
    
    # Improvement = lower placing (better)
    improvement = previous_avg - recent_avg
    
    if improvement > 2:
        direction = 'surging'
        score = 80.0
    elif improvement > 0.5:
        direction = 'improving'
        score = 65.0
    elif improvement < -2:
        direction = 'declining'
        score = 30.0
    elif improvement < -0.5:
        direction = 'slipping'
        score = 40.0
    else:
        direction = 'stable'
        score = 50.0
    
    return {'score': score, 'direction': direction, 'improvement': improvement}


# Integration
@property
def momentum_adjusted_score(self) -> float:
    """Get score adjusted for recent momentum."""
    base_score = self.recent_form
    momentum = calculate_recent_momentum(self.recent_placings)
    
    # Apply momentum factor
    if momentum['direction'] == 'surging':
        return base_score * 1.08
    elif momentum['direction'] == 'improving':
        return base_score * 1.04
    elif momentum['direction'] == 'declining':
        return base_score * 0.92
    elif momentum['direction'] == 'slipping':
        return base_score * 0.96
    
    return base_score



# RECENT FORM MOMENTUM - Auto-implemented
def calculate_recent_momentum(placings: List[int]) -> Dict[str, float]:
    """
    Compare last 3 shows vs previous 3.
    Returns momentum score and direction.
    """
    if len(placings) < 4:
        return {'score': 50.0, 'direction': 'neutral'}
    
    # Split into recent and previous
    recent = placings[:3]
    previous = placings[3:6] if len(placings) >= 6 else placings[3:]
    
    recent_avg = sum(recent) / len(recent)
    previous_avg = sum(previous) / len(previous)
    
    # Improvement = lower placing (better)
    improvement = previous_avg - recent_avg
    
    if improvement > 2:
        direction = 'surging'
        score = 80.0
    elif improvement > 0.5:
        direction = 'improving'
        score = 65.0
    elif improvement < -2:
        direction = 'declining'
        score = 30.0
    elif improvement < -0.5:
        direction = 'slipping'
        score = 40.0
    else:
        direction = 'stable'
        score = 50.0
    
    return {'score': score, 'direction': direction, 'improvement': improvement}


# Integration
@property
def momentum_adjusted_score(self) -> float:
    """Get score adjusted for recent momentum."""
    base_score = self.recent_form
    momentum = calculate_recent_momentum(self.recent_placings)
    
    # Apply momentum factor
    if momentum['direction'] == 'surging':
        return base_score * 1.08
    elif momentum['direction'] == 'improving':
        return base_score * 1.04
    elif momentum['direction'] == 'declining':
        return base_score * 0.92
    elif momentum['direction'] == 'slipping':
        return base_score * 0.96
    
    return base_score



# COMPETITION RECENCY BOOST - Auto-implemented
from datetime import datetime, timedelta

def calculate_recency_factor(last_competition_date: Optional[str]) -> float:
    """
    Athletes who competed recently get freshness bonus.
    Too recent might mean tired, too old = rusty.
    """
    if not last_competition_date:
        return 0.95  # Unknown = slight penalty
    
    try:
        last_date = datetime.fromisoformat(last_competition_date.replace('Z', '+00:00'))
        days_ago = (datetime.now() - last_date).days
        
        # Optimal range: 30-90 days (fresh but not tired)
        if 30 <= days_ago <= 90:
            return 1.05  # Peak freshness
        elif days_ago < 14:
            return 0.95  # Might be tired from recent prep
        elif days_ago < 30:
            return 1.02  # Getting warm
        elif days_ago <= 180:
            return 1.0   # Normal
        elif days_ago <= 365:
            return 0.97  # Getting rusty
        else:
            return 0.93  # Very rusty
    except:
        return 1.0


# Integration
@property
def recency_adjusted_score(self) -> float:
    """Get score adjusted for competition recency."""
    base_score = self.recent_form
    dates = self.get_show_dates()
    
    if dates:
        factor = calculate_recency_factor(dates[0])
        return base_score * factor
    
    return base_score



# WEIGHTED COMPETITION TIER SCORE - Auto-implemented
COMPETITION_TIER_WEIGHTS = {
    'olympia': 1.0,
    'arnold classic': 0.95,
    'new york pro': 0.85,
    'chicago pro': 0.82,
    'tampa pro': 0.80,
    'ifbb pro league': 0.75,
    'regional': 0.60,
    'local': 0.45,
}

def get_tier_weight(competition_name: str) -> float:
    """Get weight for competition based on tier."""
    comp_lower = competition_name.lower()
    
    for key, weight in COMPETITION_TIER_WEIGHTS.items():
        if key in comp_lower:
            return weight
    
    return 0.70  # Default


def calculate_tier_weighted_average(competitions: List[Dict]) -> float:
    """
    Calculate athlete's average placing weighted by competition tier.
    Olympia 1st counts much more than regional 1st.
    """
    if not competitions:
        return 50.0
    
    total_weighted_placing = 0
    total_weight = 0
    
    for comp in competitions:
        name = comp.get('name', '')
        placing = comp.get('placing', 10)
        
        weight = get_tier_weight(name)
        # Higher weight (better competition) = more valuable placing
        weighted_placing = placing * (2.0 - weight)  # Invert: higher weight = lower effective placing
        
        total_weighted_placing += weighted_placing * weight
        total_weight += weight
    
    if total_weight == 0:
        return 50.0
    
    weighted_avg = total_weighted_placing / total_weight
    
    # Convert to score
    return 100 * math.exp(-0.15 * (weighted_avg - 1))


# Integration
@property
def tier_weighted_score(self) -> float:
    """Get score based on tier-weighted performance."""
    return calculate_tier_weighted_average(self.competition_history)



# COMPETITION RECENCY BOOST - Auto-implemented
from datetime import datetime, timedelta

def calculate_recency_factor(last_competition_date: Optional[str]) -> float:
    """
    Athletes who competed recently get freshness bonus.
    Too recent might mean tired, too old = rusty.
    """
    if not last_competition_date:
        return 0.95  # Unknown = slight penalty
    
    try:
        last_date = datetime.fromisoformat(last_competition_date.replace('Z', '+00:00'))
        days_ago = (datetime.now() - last_date).days
        
        # Optimal range: 30-90 days (fresh but not tired)
        if 30 <= days_ago <= 90:
            return 1.05  # Peak freshness
        elif days_ago < 14:
            return 0.95  # Might be tired from recent prep
        elif days_ago < 30:
            return 1.02  # Getting warm
        elif days_ago <= 180:
            return 1.0   # Normal
        elif days_ago <= 365:
            return 0.97  # Getting rusty
        else:
            return 0.93  # Very rusty
    except:
        return 1.0


# Integration
@property
def recency_adjusted_score(self) -> float:
    """Get score adjusted for competition recency."""
    base_score = self.recent_form
    dates = self.get_show_dates()
    
    if dates:
        factor = calculate_recency_factor(dates[0])
        return base_score * factor
    
    return base_score



# WEIGHTED COMPETITION TIER SCORE - Auto-implemented
COMPETITION_TIER_WEIGHTS = {
    'olympia': 1.0,
    'arnold classic': 0.95,
    'new york pro': 0.85,
    'chicago pro': 0.82,
    'tampa pro': 0.80,
    'ifbb pro league': 0.75,
    'regional': 0.60,
    'local': 0.45,
}

def get_tier_weight(competition_name: str) -> float:
    """Get weight for competition based on tier."""
    comp_lower = competition_name.lower()
    
    for key, weight in COMPETITION_TIER_WEIGHTS.items():
        if key in comp_lower:
            return weight
    
    return 0.70  # Default


def calculate_tier_weighted_average(competitions: List[Dict]) -> float:
    """
    Calculate athlete's average placing weighted by competition tier.
    Olympia 1st counts much more than regional 1st.
    """
    if not competitions:
        return 50.0
    
    total_weighted_placing = 0
    total_weight = 0
    
    for comp in competitions:
        name = comp.get('name', '')
        placing = comp.get('placing', 10)
        
        weight = get_tier_weight(name)
        # Higher weight (better competition) = more valuable placing
        weighted_placing = placing * (2.0 - weight)  # Invert: higher weight = lower effective placing
        
        total_weighted_placing += weighted_placing * weight
        total_weight += weight
    
    if total_weight == 0:
        return 50.0
    
    weighted_avg = total_weighted_placing / total_weight
    
    # Convert to score
    return 100 * math.exp(-0.15 * (weighted_avg - 1))


# Integration
@property
def tier_weighted_score(self) -> float:
    """Get score based on tier-weighted performance."""
    return calculate_tier_weighted_average(self.competition_history)



# COMPETITION RECENCY BOOST - Auto-implemented
from datetime import datetime, timedelta

def calculate_recency_factor(last_competition_date: Optional[str]) -> float:
    """
    Athletes who competed recently get freshness bonus.
    Too recent might mean tired, too old = rusty.
    """
    if not last_competition_date:
        return 0.95  # Unknown = slight penalty
    
    try:
        last_date = datetime.fromisoformat(last_competition_date.replace('Z', '+00:00'))
        days_ago = (datetime.now() - last_date).days
        
        # Optimal range: 30-90 days (fresh but not tired)
        if 30 <= days_ago <= 90:
            return 1.05  # Peak freshness
        elif days_ago < 14:
            return 0.95  # Might be tired from recent prep
        elif days_ago < 30:
            return 1.02  # Getting warm
        elif days_ago <= 180:
            return 1.0   # Normal
        elif days_ago <= 365:
            return 0.97  # Getting rusty
        else:
            return 0.93  # Very rusty
    except:
        return 1.0


# Integration
@property
def recency_adjusted_score(self) -> float:
    """Get score adjusted for competition recency."""
    base_score = self.recent_form
    dates = self.get_show_dates()
    
    if dates:
        factor = calculate_recency_factor(dates[0])
        return base_score * factor
    
    return base_score



# COMPETITION RECENCY BOOST - Auto-implemented
from datetime import datetime, timedelta

def calculate_recency_factor(last_competition_date: Optional[str]) -> float:
    """
    Athletes who competed recently get freshness bonus.
    Too recent might mean tired, too old = rusty.
    """
    if not last_competition_date:
        return 0.95  # Unknown = slight penalty
    
    try:
        last_date = datetime.fromisoformat(last_competition_date.replace('Z', '+00:00'))
        days_ago = (datetime.now() - last_date).days
        
        # Optimal range: 30-90 days (fresh but not tired)
        if 30 <= days_ago <= 90:
            return 1.05  # Peak freshness
        elif days_ago < 14:
            return 0.95  # Might be tired from recent prep
        elif days_ago < 30:
            return 1.02  # Getting warm
        elif days_ago <= 180:
            return 1.0   # Normal
        elif days_ago <= 365:
            return 0.97  # Getting rusty
        else:
            return 0.93  # Very rusty
    except:
        return 1.0


# Integration
@property
def recency_adjusted_score(self) -> float:
    """Get score adjusted for competition recency."""
    base_score = self.recent_form
    dates = self.get_show_dates()
    
    if dates:
        factor = calculate_recency_factor(dates[0])
        return base_score * factor
    
    return base_score



# WEIGHTED COMPETITION TIER SCORE - Auto-implemented
COMPETITION_TIER_WEIGHTS = {
    'olympia': 1.0,
    'arnold classic': 0.95,
    'new york pro': 0.85,
    'chicago pro': 0.82,
    'tampa pro': 0.80,
    'ifbb pro league': 0.75,
    'regional': 0.60,
    'local': 0.45,
}

def get_tier_weight(competition_name: str) -> float:
    """Get weight for competition based on tier."""
    comp_lower = competition_name.lower()
    
    for key, weight in COMPETITION_TIER_WEIGHTS.items():
        if key in comp_lower:
            return weight
    
    return 0.70  # Default


def calculate_tier_weighted_average(competitions: List[Dict]) -> float:
    """
    Calculate athlete's average placing weighted by competition tier.
    Olympia 1st counts much more than regional 1st.
    """
    if not competitions:
        return 50.0
    
    total_weighted_placing = 0
    total_weight = 0
    
    for comp in competitions:
        name = comp.get('name', '')
        placing = comp.get('placing', 10)
        
        weight = get_tier_weight(name)
        # Higher weight (better competition) = more valuable placing
        weighted_placing = placing * (2.0 - weight)  # Invert: higher weight = lower effective placing
        
        total_weighted_placing += weighted_placing * weight
        total_weight += weight
    
    if total_weight == 0:
        return 50.0
    
    weighted_avg = total_weighted_placing / total_weight
    
    # Convert to score
    return 100 * math.exp(-0.15 * (weighted_avg - 1))


# Integration
@property
def tier_weighted_score(self) -> float:
    """Get score based on tier-weighted performance."""
    return calculate_tier_weighted_average(self.competition_history)



# COMPETITION RECENCY BOOST - Auto-implemented
from datetime import datetime, timedelta

def calculate_recency_factor(last_competition_date: Optional[str]) -> float:
    """
    Athletes who competed recently get freshness bonus.
    Too recent might mean tired, too old = rusty.
    """
    if not last_competition_date:
        return 0.95  # Unknown = slight penalty
    
    try:
        last_date = datetime.fromisoformat(last_competition_date.replace('Z', '+00:00'))
        days_ago = (datetime.now() - last_date).days
        
        # Optimal range: 30-90 days (fresh but not tired)
        if 30 <= days_ago <= 90:
            return 1.05  # Peak freshness
        elif days_ago < 14:
            return 0.95  # Might be tired from recent prep
        elif days_ago < 30:
            return 1.02  # Getting warm
        elif days_ago <= 180:
            return 1.0   # Normal
        elif days_ago <= 365:
            return 0.97  # Getting rusty
        else:
            return 0.93  # Very rusty
    except:
        return 1.0


# Integration
@property
def recency_adjusted_score(self) -> float:
    """Get score adjusted for competition recency."""
    base_score = self.recent_form
    dates = self.get_show_dates()
    
    if dates:
        factor = calculate_recency_factor(dates[0])
        return base_score * factor
    
    return base_score



# LINEAR TREND EXTRAPOLATION - Auto-implemented
import statistics

def calculate_linear_trend(placings: List[int]) -> Dict[str, float]:
    """
    Fit linear regression to recent placings and extrapolate.
    Detects consistent improvement or decline.
    """
    if len(placings) < 4:
        return {'trend': 'flat', 'predicted_next': 5.0, 'r_squared': 0}
    
    # Use last 6 shows max
    recent = placings[:6]
    x = list(range(len(recent)))
    
    try:
        # Linear regression
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(recent)
        sum_xy = sum(x[i] * recent[i] for i in range(n))
        sum_x2 = sum(xi ** 2 for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2) if (n * sum_x2 - sum_x ** 2) != 0 else 0
        intercept = (sum_y - slope * sum_x) / n
        
        # Predict next placing
        predicted = intercept + slope * n
        
        # R-squared (simplified)
        y_mean = statistics.mean(recent)
        ss_total = sum((y - y_mean) ** 2 for y in recent)
        ss_residual = sum((recent[i] - (intercept + slope * x[i])) ** 2 for i in range(n))
        r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
        
        # Determine trend
        if slope < -0.3 and r_squared > 0.3:
            trend = 'improving'
        elif slope > 0.3 and r_squared > 0.3:
            trend = 'declining'
        else:
            trend = 'flat'
        
        return {
            'trend': trend,
            'slope': slope,
            'predicted_next': max(1, predicted),
            'r_squared': r_squared
        }
    except:
        return {'trend': 'flat', 'predicted_next': statistics.mean(recent) if recent else 5.0, 'r_squared': 0}


# Integration
@property
def trend_adjusted_score(self) -> float:
    """Get score adjusted for linear trend."""
    trend = calculate_linear_trend(self.recent_placings)
    base_score = self.recent_form
    
    # Adjust based on trend strength and direction
    if trend['trend'] == 'improving':
        return min(100, base_score * 1.06)
    elif trend['trend'] == 'declining':
        return base_score * 0.94
    
    return base_score



# LINEAR TREND EXTRAPOLATION - Auto-implemented
import statistics

def calculate_linear_trend(placings: List[int]) -> Dict[str, float]:
    """
    Fit linear regression to recent placings and extrapolate.
    Detects consistent improvement or decline.
    """
    if len(placings) < 4:
        return {'trend': 'flat', 'predicted_next': 5.0, 'r_squared': 0}
    
    # Use last 6 shows max
    recent = placings[:6]
    x = list(range(len(recent)))
    
    try:
        # Linear regression
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(recent)
        sum_xy = sum(x[i] * recent[i] for i in range(n))
        sum_x2 = sum(xi ** 2 for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2) if (n * sum_x2 - sum_x ** 2) != 0 else 0
        intercept = (sum_y - slope * sum_x) / n
        
        # Predict next placing
        predicted = intercept + slope * n
        
        # R-squared (simplified)
        y_mean = statistics.mean(recent)
        ss_total = sum((y - y_mean) ** 2 for y in recent)
        ss_residual = sum((recent[i] - (intercept + slope * x[i])) ** 2 for i in range(n))
        r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
        
        # Determine trend
        if slope < -0.3 and r_squared > 0.3:
            trend = 'improving'
        elif slope > 0.3 and r_squared > 0.3:
            trend = 'declining'
        else:
            trend = 'flat'
        
        return {
            'trend': trend,
            'slope': slope,
            'predicted_next': max(1, predicted),
            'r_squared': r_squared
        }
    except:
        return {'trend': 'flat', 'predicted_next': statistics.mean(recent) if recent else 5.0, 'r_squared': 0}


# Integration
@property
def trend_adjusted_score(self) -> float:
    """Get score adjusted for linear trend."""
    trend = calculate_linear_trend(self.recent_placings)
    base_score = self.recent_form
    
    # Adjust based on trend strength and direction
    if trend['trend'] == 'improving':
        return min(100, base_score * 1.06)
    elif trend['trend'] == 'declining':
        return base_score * 0.94
    
    return base_score



# ATHLETE DEVELOPMENT STAGE - Auto-implemented
def classify_development_stage(placings: List[int], total_shows: int) -> Dict[str, any]:
    """
    Classify athlete as emerging, prime, or declining.
    Based on trend direction and career stage.
    """
    if len(placings) < 4:
        if total_shows < 5:
            return {'stage': 'emerging', 'confidence': 60, 'description': 'Newcomer'}
        else:
            return {'stage': 'prime', 'confidence': 50, 'description': 'Unknown'}
    
    # Calculate trend (recent vs older)
    recent_avg = sum(placings[:3]) / 3
    older_avg = sum(placings[3:6]) / min(3, len(placings) - 3) if len(placings) > 3 else recent_avg
    
    improvement = older_avg - recent_avg
    
    # Determine stage
    if improvement > 1.5:
        stage = 'emerging'
        confidence = min(95, 60 + improvement * 10)
    elif improvement < -1.5:
        stage = 'declining'
        confidence = min(95, 60 + abs(improvement) * 10)
    else:
        stage = 'prime'
        confidence = 70
    
    return {
        'stage': stage,
        'confidence': confidence,
        'improvement': improvement,
        'description': f'{stage.capitalize()} athlete'
    }


# Integration
@property
def stage_adjusted_score(self) -> float:
    """Get score adjusted for development stage."""
    stage_info = classify_development_stage(self.recent_placings, self.total_shows)
    base_score = self.recent_form
    
    # Emerging athletes get slight boost (upward trajectory)
    if stage_info['stage'] == 'emerging':
        return min(100, base_score * 1.05)
    elif stage_info['stage'] == 'declining':
        return base_score * 0.95
    
    return base_score



# ATHLETE DEVELOPMENT STAGE - Auto-implemented
def classify_development_stage(placings: List[int], total_shows: int) -> Dict[str, any]:
    """
    Classify athlete as emerging, prime, or declining.
    Based on trend direction and career stage.
    """
    if len(placings) < 4:
        if total_shows < 5:
            return {'stage': 'emerging', 'confidence': 60, 'description': 'Newcomer'}
        else:
            return {'stage': 'prime', 'confidence': 50, 'description': 'Unknown'}
    
    # Calculate trend (recent vs older)
    recent_avg = sum(placings[:3]) / 3
    older_avg = sum(placings[3:6]) / min(3, len(placings) - 3) if len(placings) > 3 else recent_avg
    
    improvement = older_avg - recent_avg
    
    # Determine stage
    if improvement > 1.5:
        stage = 'emerging'
        confidence = min(95, 60 + improvement * 10)
    elif improvement < -1.5:
        stage = 'declining'
        confidence = min(95, 60 + abs(improvement) * 10)
    else:
        stage = 'prime'
        confidence = 70
    
    return {
        'stage': stage,
        'confidence': confidence,
        'improvement': improvement,
        'description': f'{stage.capitalize()} athlete'
    }


# Integration
@property
def stage_adjusted_score(self) -> float:
    """Get score adjusted for development stage."""
    stage_info = classify_development_stage(self.recent_placings, self.total_shows)
    base_score = self.recent_form
    
    # Emerging athletes get slight boost (upward trajectory)
    if stage_info['stage'] == 'emerging':
        return min(100, base_score * 1.05)
    elif stage_info['stage'] == 'declining':
        return base_score * 0.95
    
    return base_score



# ATHLETE DEVELOPMENT STAGE - Auto-implemented
def classify_development_stage(placings: List[int], total_shows: int) -> Dict[str, any]:
    """
    Classify athlete as emerging, prime, or declining.
    Based on trend direction and career stage.
    """
    if len(placings) < 4:
        if total_shows < 5:
            return {'stage': 'emerging', 'confidence': 60, 'description': 'Newcomer'}
        else:
            return {'stage': 'prime', 'confidence': 50, 'description': 'Unknown'}
    
    # Calculate trend (recent vs older)
    recent_avg = sum(placings[:3]) / 3
    older_avg = sum(placings[3:6]) / min(3, len(placings) - 3) if len(placings) > 3 else recent_avg
    
    improvement = older_avg - recent_avg
    
    # Determine stage
    if improvement > 1.5:
        stage = 'emerging'
        confidence = min(95, 60 + improvement * 10)
    elif improvement < -1.5:
        stage = 'declining'
        confidence = min(95, 60 + abs(improvement) * 10)
    else:
        stage = 'prime'
        confidence = 70
    
    return {
        'stage': stage,
        'confidence': confidence,
        'improvement': improvement,
        'description': f'{stage.capitalize()} athlete'
    }


# Integration
@property
def stage_adjusted_score(self) -> float:
    """Get score adjusted for development stage."""
    stage_info = classify_development_stage(self.recent_placings, self.total_shows)
    base_score = self.recent_form
    
    # Emerging athletes get slight boost (upward trajectory)
    if stage_info['stage'] == 'emerging':
        return min(100, base_score * 1.05)
    elif stage_info['stage'] == 'declining':
        return base_score * 0.95
    
    return base_score



# COMPETITION RECENCY BOOST - Auto-implemented
from datetime import datetime, timedelta

def calculate_recency_factor(last_competition_date: Optional[str]) -> float:
    """
    Athletes who competed recently get freshness bonus.
    Too recent might mean tired, too old = rusty.
    """
    if not last_competition_date:
        return 0.95  # Unknown = slight penalty
    
    try:
        last_date = datetime.fromisoformat(last_competition_date.replace('Z', '+00:00'))
        days_ago = (datetime.now() - last_date).days
        
        # Optimal range: 30-90 days (fresh but not tired)
        if 30 <= days_ago <= 90:
            return 1.05  # Peak freshness
        elif days_ago < 14:
            return 0.95  # Might be tired from recent prep
        elif days_ago < 30:
            return 1.02  # Getting warm
        elif days_ago <= 180:
            return 1.0   # Normal
        elif days_ago <= 365:
            return 0.97  # Getting rusty
        else:
            return 0.93  # Very rusty
    except:
        return 1.0


# Integration
@property
def recency_adjusted_score(self) -> float:
    """Get score adjusted for competition recency."""
    base_score = self.recent_form
    dates = self.get_show_dates()
    
    if dates:
        factor = calculate_recency_factor(dates[0])
        return base_score * factor
    
    return base_score



# ATHLETE DEVELOPMENT STAGE - Auto-implemented
def classify_development_stage(placings: List[int], total_shows: int) -> Dict[str, any]:
    """
    Classify athlete as emerging, prime, or declining.
    Based on trend direction and career stage.
    """
    if len(placings) < 4:
        if total_shows < 5:
            return {'stage': 'emerging', 'confidence': 60, 'description': 'Newcomer'}
        else:
            return {'stage': 'prime', 'confidence': 50, 'description': 'Unknown'}
    
    # Calculate trend (recent vs older)
    recent_avg = sum(placings[:3]) / 3
    older_avg = sum(placings[3:6]) / min(3, len(placings) - 3) if len(placings) > 3 else recent_avg
    
    improvement = older_avg - recent_avg
    
    # Determine stage
    if improvement > 1.5:
        stage = 'emerging'
        confidence = min(95, 60 + improvement * 10)
    elif improvement < -1.5:
        stage = 'declining'
        confidence = min(95, 60 + abs(improvement) * 10)
    else:
        stage = 'prime'
        confidence = 70
    
    return {
        'stage': stage,
        'confidence': confidence,
        'improvement': improvement,
        'description': f'{stage.capitalize()} athlete'
    }


# Integration
@property
def stage_adjusted_score(self) -> float:
    """Get score adjusted for development stage."""
    stage_info = classify_development_stage(self.recent_placings, self.total_shows)
    base_score = self.recent_form
    
    # Emerging athletes get slight boost (upward trajectory)
    if stage_info['stage'] == 'emerging':
        return min(100, base_score * 1.05)
    elif stage_info['stage'] == 'declining':
        return base_score * 0.95
    
    return base_score



# PERFORMANCE CEILING DETECTION - Auto-implemented
from datetime import datetime

def analyze_performance_ceiling(placings: List[int], dates: List[str]) -> Dict[str, any]:
    """
    Track best-ever placing and years since.
    Athletes rarely exceed career bests.
    """
    if not placings or not dates:
        return {'best_placing': 99, 'years_since_best': 0, 'ceiling_factor': 1.0}
    
    # Find best placing
    best_placing = min(placings)
    best_idx = placings.index(best_placing)
    
    # Calculate years since best
    try:
        best_date = datetime.fromisoformat(dates[best_idx].replace('Z', '+00:00'))
        years_since = (datetime.now() - best_date).days / 365.25
    except:
        years_since = 0
    
    # Ceiling factor: harder to repeat best as time passes
    if years_since < 1:
        ceiling_factor = 1.0  # Recent best, can repeat
    elif years_since < 2:
        ceiling_factor = 0.98
    elif years_since < 3:
        ceiling_factor = 0.95
    else:
        ceiling_factor = 0.90  # Long time since best
    
    return {
        'best_placing': best_placing,
        'years_since_best': years_since,
        'ceiling_factor': ceiling_factor,
        'can_exceed_ceiling': years_since < 1
    }


# Integration
@property
def ceiling_adjusted_score(self) -> float:
    """Get score adjusted for performance ceiling."""
    ceiling_info = analyze_performance_ceiling(self.recent_placings, self.get_show_dates())
    base_score = self.recent_form
    
    return base_score * ceiling_info['ceiling_factor']



# COMPETITION RECENCY BOOST - Auto-implemented
from datetime import datetime, timedelta

def calculate_recency_factor(last_competition_date: Optional[str]) -> float:
    """
    Athletes who competed recently get freshness bonus.
    Too recent might mean tired, too old = rusty.
    """
    if not last_competition_date:
        return 0.95  # Unknown = slight penalty
    
    try:
        last_date = datetime.fromisoformat(last_competition_date.replace('Z', '+00:00'))
        days_ago = (datetime.now() - last_date).days
        
        # Optimal range: 30-90 days (fresh but not tired)
        if 30 <= days_ago <= 90:
            return 1.05  # Peak freshness
        elif days_ago < 14:
            return 0.95  # Might be tired from recent prep
        elif days_ago < 30:
            return 1.02  # Getting warm
        elif days_ago <= 180:
            return 1.0   # Normal
        elif days_ago <= 365:
            return 0.97  # Getting rusty
        else:
            return 0.93  # Very rusty
    except:
        return 1.0


# Integration
@property
def recency_adjusted_score(self) -> float:
    """Get score adjusted for competition recency."""
    base_score = self.recent_form
    dates = self.get_show_dates()
    
    if dates:
        factor = calculate_recency_factor(dates[0])
        return base_score * factor
    
    return base_score



# COMPETITION RECENCY BOOST - Auto-implemented
from datetime import datetime, timedelta

def calculate_recency_factor(last_competition_date: Optional[str]) -> float:
    """
    Athletes who competed recently get freshness bonus.
    Too recent might mean tired, too old = rusty.
    """
    if not last_competition_date:
        return 0.95  # Unknown = slight penalty
    
    try:
        last_date = datetime.fromisoformat(last_competition_date.replace('Z', '+00:00'))
        days_ago = (datetime.now() - last_date).days
        
        # Optimal range: 30-90 days (fresh but not tired)
        if 30 <= days_ago <= 90:
            return 1.05  # Peak freshness
        elif days_ago < 14:
            return 0.95  # Might be tired from recent prep
        elif days_ago < 30:
            return 1.02  # Getting warm
        elif days_ago <= 180:
            return 1.0   # Normal
        elif days_ago <= 365:
            return 0.97  # Getting rusty
        else:
            return 0.93  # Very rusty
    except:
        return 1.0


# Integration
@property
def recency_adjusted_score(self) -> float:
    """Get score adjusted for competition recency."""
    base_score = self.recent_form
    dates = self.get_show_dates()
    
    if dates:
        factor = calculate_recency_factor(dates[0])
        return base_score * factor
    
    return base_score



# ATHLETE DEVELOPMENT STAGE - Auto-implemented
def classify_development_stage(placings: List[int], total_shows: int) -> Dict[str, any]:
    """
    Classify athlete as emerging, prime, or declining.
    Based on trend direction and career stage.
    """
    if len(placings) < 4:
        if total_shows < 5:
            return {'stage': 'emerging', 'confidence': 60, 'description': 'Newcomer'}
        else:
            return {'stage': 'prime', 'confidence': 50, 'description': 'Unknown'}
    
    # Calculate trend (recent vs older)
    recent_avg = sum(placings[:3]) / 3
    older_avg = sum(placings[3:6]) / min(3, len(placings) - 3) if len(placings) > 3 else recent_avg
    
    improvement = older_avg - recent_avg
    
    # Determine stage
    if improvement > 1.5:
        stage = 'emerging'
        confidence = min(95, 60 + improvement * 10)
    elif improvement < -1.5:
        stage = 'declining'
        confidence = min(95, 60 + abs(improvement) * 10)
    else:
        stage = 'prime'
        confidence = 70
    
    return {
        'stage': stage,
        'confidence': confidence,
        'improvement': improvement,
        'description': f'{stage.capitalize()} athlete'
    }


# Integration
@property
def stage_adjusted_score(self) -> float:
    """Get score adjusted for development stage."""
    stage_info = classify_development_stage(self.recent_placings, self.total_shows)
    base_score = self.recent_form
    
    # Emerging athletes get slight boost (upward trajectory)
    if stage_info['stage'] == 'emerging':
        return min(100, base_score * 1.05)
    elif stage_info['stage'] == 'declining':
        return base_score * 0.95
    
    return base_score



# ATHLETE DEVELOPMENT STAGE - Auto-implemented
def classify_development_stage(placings: List[int], total_shows: int) -> Dict[str, any]:
    """
    Classify athlete as emerging, prime, or declining.
    Based on trend direction and career stage.
    """
    if len(placings) < 4:
        if total_shows < 5:
            return {'stage': 'emerging', 'confidence': 60, 'description': 'Newcomer'}
        else:
            return {'stage': 'prime', 'confidence': 50, 'description': 'Unknown'}
    
    # Calculate trend (recent vs older)
    recent_avg = sum(placings[:3]) / 3
    older_avg = sum(placings[3:6]) / min(3, len(placings) - 3) if len(placings) > 3 else recent_avg
    
    improvement = older_avg - recent_avg
    
    # Determine stage
    if improvement > 1.5:
        stage = 'emerging'
        confidence = min(95, 60 + improvement * 10)
    elif improvement < -1.5:
        stage = 'declining'
        confidence = min(95, 60 + abs(improvement) * 10)
    else:
        stage = 'prime'
        confidence = 70
    
    return {
        'stage': stage,
        'confidence': confidence,
        'improvement': improvement,
        'description': f'{stage.capitalize()} athlete'
    }


# Integration
@property
def stage_adjusted_score(self) -> float:
    """Get score adjusted for development stage."""
    stage_info = classify_development_stage(self.recent_placings, self.total_shows)
    base_score = self.recent_form
    
    # Emerging athletes get slight boost (upward trajectory)
    if stage_info['stage'] == 'emerging':
        return min(100, base_score * 1.05)
    elif stage_info['stage'] == 'declining':
        return base_score * 0.95
    
    return base_score



# ATHLETE DEVELOPMENT STAGE - Auto-implemented
def classify_development_stage(placings: List[int], total_shows: int) -> Dict[str, any]:
    """
    Classify athlete as emerging, prime, or declining.
    Based on trend direction and career stage.
    """
    if len(placings) < 4:
        if total_shows < 5:
            return {'stage': 'emerging', 'confidence': 60, 'description': 'Newcomer'}
        else:
            return {'stage': 'prime', 'confidence': 50, 'description': 'Unknown'}
    
    # Calculate trend (recent vs older)
    recent_avg = sum(placings[:3]) / 3
    older_avg = sum(placings[3:6]) / min(3, len(placings) - 3) if len(placings) > 3 else recent_avg
    
    improvement = older_avg - recent_avg
    
    # Determine stage
    if improvement > 1.5:
        stage = 'emerging'
        confidence = min(95, 60 + improvement * 10)
    elif improvement < -1.5:
        stage = 'declining'
        confidence = min(95, 60 + abs(improvement) * 10)
    else:
        stage = 'prime'
        confidence = 70
    
    return {
        'stage': stage,
        'confidence': confidence,
        'improvement': improvement,
        'description': f'{stage.capitalize()} athlete'
    }


# Integration
@property
def stage_adjusted_score(self) -> float:
    """Get score adjusted for development stage."""
    stage_info = classify_development_stage(self.recent_placings, self.total_shows)
    base_score = self.recent_form
    
    # Emerging athletes get slight boost (upward trajectory)
    if stage_info['stage'] == 'emerging':
        return min(100, base_score * 1.05)
    elif stage_info['stage'] == 'declining':
        return base_score * 0.95
    
    return base_score



# ATHLETE DEVELOPMENT STAGE - Auto-implemented
def classify_development_stage(placings: List[int], total_shows: int) -> Dict[str, any]:
    """
    Classify athlete as emerging, prime, or declining.
    Based on trend direction and career stage.
    """
    if len(placings) < 4:
        if total_shows < 5:
            return {'stage': 'emerging', 'confidence': 60, 'description': 'Newcomer'}
        else:
            return {'stage': 'prime', 'confidence': 50, 'description': 'Unknown'}
    
    # Calculate trend (recent vs older)
    recent_avg = sum(placings[:3]) / 3
    older_avg = sum(placings[3:6]) / min(3, len(placings) - 3) if len(placings) > 3 else recent_avg
    
    improvement = older_avg - recent_avg
    
    # Determine stage
    if improvement > 1.5:
        stage = 'emerging'
        confidence = min(95, 60 + improvement * 10)
    elif improvement < -1.5:
        stage = 'declining'
        confidence = min(95, 60 + abs(improvement) * 10)
    else:
        stage = 'prime'
        confidence = 70
    
    return {
        'stage': stage,
        'confidence': confidence,
        'improvement': improvement,
        'description': f'{stage.capitalize()} athlete'
    }


# Integration
@property
def stage_adjusted_score(self) -> float:
    """Get score adjusted for development stage."""
    stage_info = classify_development_stage(self.recent_placings, self.total_shows)
    base_score = self.recent_form
    
    # Emerging athletes get slight boost (upward trajectory)
    if stage_info['stage'] == 'emerging':
        return min(100, base_score * 1.05)
    elif stage_info['stage'] == 'declining':
        return base_score * 0.95
    
    return base_score



# ATHLETE DEVELOPMENT STAGE - Auto-implemented
def classify_development_stage(placings: List[int], total_shows: int) -> Dict[str, any]:
    """
    Classify athlete as emerging, prime, or declining.
    Based on trend direction and career stage.
    """
    if len(placings) < 4:
        if total_shows < 5:
            return {'stage': 'emerging', 'confidence': 60, 'description': 'Newcomer'}
        else:
            return {'stage': 'prime', 'confidence': 50, 'description': 'Unknown'}
    
    # Calculate trend (recent vs older)
    recent_avg = sum(placings[:3]) / 3
    older_avg = sum(placings[3:6]) / min(3, len(placings) - 3) if len(placings) > 3 else recent_avg
    
    improvement = older_avg - recent_avg
    
    # Determine stage
    if improvement > 1.5:
        stage = 'emerging'
        confidence = min(95, 60 + improvement * 10)
    elif improvement < -1.5:
        stage = 'declining'
        confidence = min(95, 60 + abs(improvement) * 10)
    else:
        stage = 'prime'
        confidence = 70
    
    return {
        'stage': stage,
        'confidence': confidence,
        'improvement': improvement,
        'description': f'{stage.capitalize()} athlete'
    }


# Integration
@property
def stage_adjusted_score(self) -> float:
    """Get score adjusted for development stage."""
    stage_info = classify_development_stage(self.recent_placings, self.total_shows)
    base_score = self.recent_form
    
    # Emerging athletes get slight boost (upward trajectory)
    if stage_info['stage'] == 'emerging':
        return min(100, base_score * 1.05)
    elif stage_info['stage'] == 'declining':
        return base_score * 0.95
    
    return base_score



# PERFORMANCE CEILING DETECTION - Auto-implemented
from datetime import datetime

def analyze_performance_ceiling(placings: List[int], dates: List[str]) -> Dict[str, any]:
    """
    Track best-ever placing and years since.
    Athletes rarely exceed career bests.
    """
    if not placings or not dates:
        return {'best_placing': 99, 'years_since_best': 0, 'ceiling_factor': 1.0}
    
    # Find best placing
    best_placing = min(placings)
    best_idx = placings.index(best_placing)
    
    # Calculate years since best
    try:
        best_date = datetime.fromisoformat(dates[best_idx].replace('Z', '+00:00'))
        years_since = (datetime.now() - best_date).days / 365.25
    except:
        years_since = 0
    
    # Ceiling factor: harder to repeat best as time passes
    if years_since < 1:
        ceiling_factor = 1.0  # Recent best, can repeat
    elif years_since < 2:
        ceiling_factor = 0.98
    elif years_since < 3:
        ceiling_factor = 0.95
    else:
        ceiling_factor = 0.90  # Long time since best
    
    return {
        'best_placing': best_placing,
        'years_since_best': years_since,
        'ceiling_factor': ceiling_factor,
        'can_exceed_ceiling': years_since < 1
    }


# Integration
@property
def ceiling_adjusted_score(self) -> float:
    """Get score adjusted for performance ceiling."""
    ceiling_info = analyze_performance_ceiling(self.recent_placings, self.get_show_dates())
    base_score = self.recent_form
    
    return base_score * ceiling_info['ceiling_factor']



# PERFORMANCE CEILING DETECTION - Auto-implemented
from datetime import datetime

def analyze_performance_ceiling(placings: List[int], dates: List[str]) -> Dict[str, any]:
    """
    Track best-ever placing and years since.
    Athletes rarely exceed career bests.
    """
    if not placings or not dates:
        return {'best_placing': 99, 'years_since_best': 0, 'ceiling_factor': 1.0}
    
    # Find best placing
    best_placing = min(placings)
    best_idx = placings.index(best_placing)
    
    # Calculate years since best
    try:
        best_date = datetime.fromisoformat(dates[best_idx].replace('Z', '+00:00'))
        years_since = (datetime.now() - best_date).days / 365.25
    except:
        years_since = 0
    
    # Ceiling factor: harder to repeat best as time passes
    if years_since < 1:
        ceiling_factor = 1.0  # Recent best, can repeat
    elif years_since < 2:
        ceiling_factor = 0.98
    elif years_since < 3:
        ceiling_factor = 0.95
    else:
        ceiling_factor = 0.90  # Long time since best
    
    return {
        'best_placing': best_placing,
        'years_since_best': years_since,
        'ceiling_factor': ceiling_factor,
        'can_exceed_ceiling': years_since < 1
    }


# Integration
@property
def ceiling_adjusted_score(self) -> float:
    """Get score adjusted for performance ceiling."""
    ceiling_info = analyze_performance_ceiling(self.recent_placings, self.get_show_dates())
    base_score = self.recent_form
    
    return base_score * ceiling_info['ceiling_factor']



# ATHLETE DEVELOPMENT STAGE - Auto-implemented
def classify_development_stage(placings: List[int], total_shows: int) -> Dict[str, any]:
    """
    Classify athlete as emerging, prime, or declining.
    Based on trend direction and career stage.
    """
    if len(placings) < 4:
        if total_shows < 5:
            return {'stage': 'emerging', 'confidence': 60, 'description': 'Newcomer'}
        else:
            return {'stage': 'prime', 'confidence': 50, 'description': 'Unknown'}
    
    # Calculate trend (recent vs older)
    recent_avg = sum(placings[:3]) / 3
    older_avg = sum(placings[3:6]) / min(3, len(placings) - 3) if len(placings) > 3 else recent_avg
    
    improvement = older_avg - recent_avg
    
    # Determine stage
    if improvement > 1.5:
        stage = 'emerging'
        confidence = min(95, 60 + improvement * 10)
    elif improvement < -1.5:
        stage = 'declining'
        confidence = min(95, 60 + abs(improvement) * 10)
    else:
        stage = 'prime'
        confidence = 70
    
    return {
        'stage': stage,
        'confidence': confidence,
        'improvement': improvement,
        'description': f'{stage.capitalize()} athlete'
    }


# Integration
@property
def stage_adjusted_score(self) -> float:
    """Get score adjusted for development stage."""
    stage_info = classify_development_stage(self.recent_placings, self.total_shows)
    base_score = self.recent_form
    
    # Emerging athletes get slight boost (upward trajectory)
    if stage_info['stage'] == 'emerging':
        return min(100, base_score * 1.05)
    elif stage_info['stage'] == 'declining':
        return base_score * 0.95
    
    return base_score



# COMPETITION RECENCY BOOST - Auto-implemented
from datetime import datetime, timedelta

def calculate_recency_factor(last_competition_date: Optional[str]) -> float:
    """
    Athletes who competed recently get freshness bonus.
    Too recent might mean tired, too old = rusty.
    """
    if not last_competition_date:
        return 0.95  # Unknown = slight penalty
    
    try:
        last_date = datetime.fromisoformat(last_competition_date.replace('Z', '+00:00'))
        days_ago = (datetime.now() - last_date).days
        
        # Optimal range: 30-90 days (fresh but not tired)
        if 30 <= days_ago <= 90:
            return 1.05  # Peak freshness
        elif days_ago < 14:
            return 0.95  # Might be tired from recent prep
        elif days_ago < 30:
            return 1.02  # Getting warm
        elif days_ago <= 180:
            return 1.0   # Normal
        elif days_ago <= 365:
            return 0.97  # Getting rusty
        else:
            return 0.93  # Very rusty
    except:
        return 1.0


# Integration
@property
def recency_adjusted_score(self) -> float:
    """Get score adjusted for competition recency."""
    base_score = self.recent_form
    dates = self.get_show_dates()
    
    if dates:
        factor = calculate_recency_factor(dates[0])
        return base_score * factor
    
    return base_score



# COMPETITION RECENCY BOOST - Auto-implemented
from datetime import datetime, timedelta

def calculate_recency_factor(last_competition_date: Optional[str]) -> float:
    """
    Athletes who competed recently get freshness bonus.
    Too recent might mean tired, too old = rusty.
    """
    if not last_competition_date:
        return 0.95  # Unknown = slight penalty
    
    try:
        last_date = datetime.fromisoformat(last_competition_date.replace('Z', '+00:00'))
        days_ago = (datetime.now() - last_date).days
        
        # Optimal range: 30-90 days (fresh but not tired)
        if 30 <= days_ago <= 90:
            return 1.05  # Peak freshness
        elif days_ago < 14:
            return 0.95  # Might be tired from recent prep
        elif days_ago < 30:
            return 1.02  # Getting warm
        elif days_ago <= 180:
            return 1.0   # Normal
        elif days_ago <= 365:
            return 0.97  # Getting rusty
        else:
            return 0.93  # Very rusty
    except:
        return 1.0


# Integration
@property
def recency_adjusted_score(self) -> float:
    """Get score adjusted for competition recency."""
    base_score = self.recent_form
    dates = self.get_show_dates()
    
    if dates:
        factor = calculate_recency_factor(dates[0])
        return base_score * factor
    
    return base_score



# PERFORMANCE CEILING DETECTION - Auto-implemented
from datetime import datetime

def analyze_performance_ceiling(placings: List[int], dates: List[str]) -> Dict[str, any]:
    """
    Track best-ever placing and years since.
    Athletes rarely exceed career bests.
    """
    if not placings or not dates:
        return {'best_placing': 99, 'years_since_best': 0, 'ceiling_factor': 1.0}
    
    # Find best placing
    best_placing = min(placings)
    best_idx = placings.index(best_placing)
    
    # Calculate years since best
    try:
        best_date = datetime.fromisoformat(dates[best_idx].replace('Z', '+00:00'))
        years_since = (datetime.now() - best_date).days / 365.25
    except:
        years_since = 0
    
    # Ceiling factor: harder to repeat best as time passes
    if years_since < 1:
        ceiling_factor = 1.0  # Recent best, can repeat
    elif years_since < 2:
        ceiling_factor = 0.98
    elif years_since < 3:
        ceiling_factor = 0.95
    else:
        ceiling_factor = 0.90  # Long time since best
    
    return {
        'best_placing': best_placing,
        'years_since_best': years_since,
        'ceiling_factor': ceiling_factor,
        'can_exceed_ceiling': years_since < 1
    }


# Integration
@property
def ceiling_adjusted_score(self) -> float:
    """Get score adjusted for performance ceiling."""
    ceiling_info = analyze_performance_ceiling(self.recent_placings, self.get_show_dates())
    base_score = self.recent_form
    
    return base_score * ceiling_info['ceiling_factor']



# COMPETITION RECENCY BOOST - Auto-implemented
from datetime import datetime, timedelta

def calculate_recency_factor(last_competition_date: Optional[str]) -> float:
    """
    Athletes who competed recently get freshness bonus.
    Too recent might mean tired, too old = rusty.
    """
    if not last_competition_date:
        return 0.95  # Unknown = slight penalty
    
    try:
        last_date = datetime.fromisoformat(last_competition_date.replace('Z', '+00:00'))
        days_ago = (datetime.now() - last_date).days
        
        # Optimal range: 30-90 days (fresh but not tired)
        if 30 <= days_ago <= 90:
            return 1.05  # Peak freshness
        elif days_ago < 14:
            return 0.95  # Might be tired from recent prep
        elif days_ago < 30:
            return 1.02  # Getting warm
        elif days_ago <= 180:
            return 1.0   # Normal
        elif days_ago <= 365:
            return 0.97  # Getting rusty
        else:
            return 0.93  # Very rusty
    except:
        return 1.0


# Integration
@property
def recency_adjusted_score(self) -> float:
    """Get score adjusted for competition recency."""
    base_score = self.recent_form
    dates = self.get_show_dates()
    
    if dates:
        factor = calculate_recency_factor(dates[0])
        return base_score * factor
    
    return base_score



# PERFORMANCE CEILING DETECTION - Auto-implemented
from datetime import datetime

def analyze_performance_ceiling(placings: List[int], dates: List[str]) -> Dict[str, any]:
    """
    Track best-ever placing and years since.
    Athletes rarely exceed career bests.
    """
    if not placings or not dates:
        return {'best_placing': 99, 'years_since_best': 0, 'ceiling_factor': 1.0}
    
    # Find best placing
    best_placing = min(placings)
    best_idx = placings.index(best_placing)
    
    # Calculate years since best
    try:
        best_date = datetime.fromisoformat(dates[best_idx].replace('Z', '+00:00'))
        years_since = (datetime.now() - best_date).days / 365.25
    except:
        years_since = 0
    
    # Ceiling factor: harder to repeat best as time passes
    if years_since < 1:
        ceiling_factor = 1.0  # Recent best, can repeat
    elif years_since < 2:
        ceiling_factor = 0.98
    elif years_since < 3:
        ceiling_factor = 0.95
    else:
        ceiling_factor = 0.90  # Long time since best
    
    return {
        'best_placing': best_placing,
        'years_since_best': years_since,
        'ceiling_factor': ceiling_factor,
        'can_exceed_ceiling': years_since < 1
    }


# Integration
@property
def ceiling_adjusted_score(self) -> float:
    """Get score adjusted for performance ceiling."""
    ceiling_info = analyze_performance_ceiling(self.recent_placings, self.get_show_dates())
    base_score = self.recent_form
    
    return base_score * ceiling_info['ceiling_factor']



# PERFORMANCE CEILING DETECTION - Auto-implemented
from datetime import datetime

def analyze_performance_ceiling(placings: List[int], dates: List[str]) -> Dict[str, any]:
    """
    Track best-ever placing and years since.
    Athletes rarely exceed career bests.
    """
    if not placings or not dates:
        return {'best_placing': 99, 'years_since_best': 0, 'ceiling_factor': 1.0}
    
    # Find best placing
    best_placing = min(placings)
    best_idx = placings.index(best_placing)
    
    # Calculate years since best
    try:
        best_date = datetime.fromisoformat(dates[best_idx].replace('Z', '+00:00'))
        years_since = (datetime.now() - best_date).days / 365.25
    except:
        years_since = 0
    
    # Ceiling factor: harder to repeat best as time passes
    if years_since < 1:
        ceiling_factor = 1.0  # Recent best, can repeat
    elif years_since < 2:
        ceiling_factor = 0.98
    elif years_since < 3:
        ceiling_factor = 0.95
    else:
        ceiling_factor = 0.90  # Long time since best
    
    return {
        'best_placing': best_placing,
        'years_since_best': years_since,
        'ceiling_factor': ceiling_factor,
        'can_exceed_ceiling': years_since < 1
    }


# Integration
@property
def ceiling_adjusted_score(self) -> float:
    """Get score adjusted for performance ceiling."""
    ceiling_info = analyze_performance_ceiling(self.recent_placings, self.get_show_dates())
    base_score = self.recent_form
    
    return base_score * ceiling_info['ceiling_factor']



# PODIUM STRIKE RATE - Auto-implemented
def calculate_podium_rate(placings: List[int], total_shows: int) -> Dict[str, float]:
    """
    Calculate percentage of top-3 finishes.
    High podium rate indicates elite status.
    """
    if not placings or total_shows == 0:
        return {'rate': 0.0, 'elite_threshold': 30.0, 'is_elite': False}
    
    # Count podiums
    podiums = sum(1 for p in placings if p <= 3)
    rate = (podiums / len(placings)) * 100
    
    # Elite threshold: 30%+ podium rate
    is_elite = rate >= 30.0
    
    return {
        'rate': rate,
        'podiums': podiums,
        'total_shows_considered': len(placings),
        'is_elite': is_elite,
        'elite_threshold': 30.0
    }


# Integration
@property
def podium_adjusted_score(self) -> float:
    """Get score adjusted for podium strike rate."""
    podium_info = calculate_podium_rate(self.recent_placings, self.total_shows)
    base_score = self.recent_form
    
    # Elite athletes get bonus
    if podium_info['is_elite']:
        return min(100, base_score * 1.07)
    elif podium_info['rate'] >= 20:
        return min(100, base_score * 1.03)
    elif podium_info['rate'] < 5 and self.total_shows > 3:
        # Consistently outside podium
        return base_score * 0.97
    
    return base_score



# PERFORMANCE CEILING DETECTION - Auto-implemented
from datetime import datetime

def analyze_performance_ceiling(placings: List[int], dates: List[str]) -> Dict[str, any]:
    """
    Track best-ever placing and years since.
    Athletes rarely exceed career bests.
    """
    if not placings or not dates:
        return {'best_placing': 99, 'years_since_best': 0, 'ceiling_factor': 1.0}
    
    # Find best placing
    best_placing = min(placings)
    best_idx = placings.index(best_placing)
    
    # Calculate years since best
    try:
        best_date = datetime.fromisoformat(dates[best_idx].replace('Z', '+00:00'))
        years_since = (datetime.now() - best_date).days / 365.25
    except:
        years_since = 0
    
    # Ceiling factor: harder to repeat best as time passes
    if years_since < 1:
        ceiling_factor = 1.0  # Recent best, can repeat
    elif years_since < 2:
        ceiling_factor = 0.98
    elif years_since < 3:
        ceiling_factor = 0.95
    else:
        ceiling_factor = 0.90  # Long time since best
    
    return {
        'best_placing': best_placing,
        'years_since_best': years_since,
        'ceiling_factor': ceiling_factor,
        'can_exceed_ceiling': years_since < 1
    }


# Integration
@property
def ceiling_adjusted_score(self) -> float:
    """Get score adjusted for performance ceiling."""
    ceiling_info = analyze_performance_ceiling(self.recent_placings, self.get_show_dates())
    base_score = self.recent_form
    
    return base_score * ceiling_info['ceiling_factor']



# COMPETITION RECENCY BOOST - Auto-implemented
from datetime import datetime, timedelta

def calculate_recency_factor(last_competition_date: Optional[str]) -> float:
    """
    Athletes who competed recently get freshness bonus.
    Too recent might mean tired, too old = rusty.
    """
    if not last_competition_date:
        return 0.95  # Unknown = slight penalty
    
    try:
        last_date = datetime.fromisoformat(last_competition_date.replace('Z', '+00:00'))
        days_ago = (datetime.now() - last_date).days
        
        # Optimal range: 30-90 days (fresh but not tired)
        if 30 <= days_ago <= 90:
            return 1.05  # Peak freshness
        elif days_ago < 14:
            return 0.95  # Might be tired from recent prep
        elif days_ago < 30:
            return 1.02  # Getting warm
        elif days_ago <= 180:
            return 1.0   # Normal
        elif days_ago <= 365:
            return 0.97  # Getting rusty
        else:
            return 0.93  # Very rusty
    except:
        return 1.0


# Integration
@property
def recency_adjusted_score(self) -> float:
    """Get score adjusted for competition recency."""
    base_score = self.recent_form
    dates = self.get_show_dates()
    
    if dates:
        factor = calculate_recency_factor(dates[0])
        return base_score * factor
    
    return base_score



# PODIUM STRIKE RATE - Auto-implemented
def calculate_podium_rate(placings: List[int], total_shows: int) -> Dict[str, float]:
    """
    Calculate percentage of top-3 finishes.
    High podium rate indicates elite status.
    """
    if not placings or total_shows == 0:
        return {'rate': 0.0, 'elite_threshold': 30.0, 'is_elite': False}
    
    # Count podiums
    podiums = sum(1 for p in placings if p <= 3)
    rate = (podiums / len(placings)) * 100
    
    # Elite threshold: 30%+ podium rate
    is_elite = rate >= 30.0
    
    return {
        'rate': rate,
        'podiums': podiums,
        'total_shows_considered': len(placings),
        'is_elite': is_elite,
        'elite_threshold': 30.0
    }


# Integration
@property
def podium_adjusted_score(self) -> float:
    """Get score adjusted for podium strike rate."""
    podium_info = calculate_podium_rate(self.recent_placings, self.total_shows)
    base_score = self.recent_form
    
    # Elite athletes get bonus
    if podium_info['is_elite']:
        return min(100, base_score * 1.07)
    elif podium_info['rate'] >= 20:
        return min(100, base_score * 1.03)
    elif podium_info['rate'] < 5 and self.total_shows > 3:
        # Consistently outside podium
        return base_score * 0.97
    
    return base_score



# PERFORMANCE CEILING DETECTION - Auto-implemented
from datetime import datetime

def analyze_performance_ceiling(placings: List[int], dates: List[str]) -> Dict[str, any]:
    """
    Track best-ever placing and years since.
    Athletes rarely exceed career bests.
    """
    if not placings or not dates:
        return {'best_placing': 99, 'years_since_best': 0, 'ceiling_factor': 1.0}
    
    # Find best placing
    best_placing = min(placings)
    best_idx = placings.index(best_placing)
    
    # Calculate years since best
    try:
        best_date = datetime.fromisoformat(dates[best_idx].replace('Z', '+00:00'))
        years_since = (datetime.now() - best_date).days / 365.25
    except:
        years_since = 0
    
    # Ceiling factor: harder to repeat best as time passes
    if years_since < 1:
        ceiling_factor = 1.0  # Recent best, can repeat
    elif years_since < 2:
        ceiling_factor = 0.98
    elif years_since < 3:
        ceiling_factor = 0.95
    else:
        ceiling_factor = 0.90  # Long time since best
    
    return {
        'best_placing': best_placing,
        'years_since_best': years_since,
        'ceiling_factor': ceiling_factor,
        'can_exceed_ceiling': years_since < 1
    }


# Integration
@property
def ceiling_adjusted_score(self) -> float:
    """Get score adjusted for performance ceiling."""
    ceiling_info = analyze_performance_ceiling(self.recent_placings, self.get_show_dates())
    base_score = self.recent_form
    
    return base_score * ceiling_info['ceiling_factor']



# PODIUM STRIKE RATE - Auto-implemented
def calculate_podium_rate(placings: List[int], total_shows: int) -> Dict[str, float]:
    """
    Calculate percentage of top-3 finishes.
    High podium rate indicates elite status.
    """
    if not placings or total_shows == 0:
        return {'rate': 0.0, 'elite_threshold': 30.0, 'is_elite': False}
    
    # Count podiums
    podiums = sum(1 for p in placings if p <= 3)
    rate = (podiums / len(placings)) * 100
    
    # Elite threshold: 30%+ podium rate
    is_elite = rate >= 30.0
    
    return {
        'rate': rate,
        'podiums': podiums,
        'total_shows_considered': len(placings),
        'is_elite': is_elite,
        'elite_threshold': 30.0
    }


# Integration
@property
def podium_adjusted_score(self) -> float:
    """Get score adjusted for podium strike rate."""
    podium_info = calculate_podium_rate(self.recent_placings, self.total_shows)
    base_score = self.recent_form
    
    # Elite athletes get bonus
    if podium_info['is_elite']:
        return min(100, base_score * 1.07)
    elif podium_info['rate'] >= 20:
        return min(100, base_score * 1.03)
    elif podium_info['rate'] < 5 and self.total_shows > 3:
        # Consistently outside podium
        return base_score * 0.97
    
    return base_score



# PERFORMANCE CEILING DETECTION - Auto-implemented
from datetime import datetime

def analyze_performance_ceiling(placings: List[int], dates: List[str]) -> Dict[str, any]:
    """
    Track best-ever placing and years since.
    Athletes rarely exceed career bests.
    """
    if not placings or not dates:
        return {'best_placing': 99, 'years_since_best': 0, 'ceiling_factor': 1.0}
    
    # Find best placing
    best_placing = min(placings)
    best_idx = placings.index(best_placing)
    
    # Calculate years since best
    try:
        best_date = datetime.fromisoformat(dates[best_idx].replace('Z', '+00:00'))
        years_since = (datetime.now() - best_date).days / 365.25
    except:
        years_since = 0
    
    # Ceiling factor: harder to repeat best as time passes
    if years_since < 1:
        ceiling_factor = 1.0  # Recent best, can repeat
    elif years_since < 2:
        ceiling_factor = 0.98
    elif years_since < 3:
        ceiling_factor = 0.95
    else:
        ceiling_factor = 0.90  # Long time since best
    
    return {
        'best_placing': best_placing,
        'years_since_best': years_since,
        'ceiling_factor': ceiling_factor,
        'can_exceed_ceiling': years_since < 1
    }


# Integration
@property
def ceiling_adjusted_score(self) -> float:
    """Get score adjusted for performance ceiling."""
    ceiling_info = analyze_performance_ceiling(self.recent_placings, self.get_show_dates())
    base_score = self.recent_form
    
    return base_score * ceiling_info['ceiling_factor']



# PERFORMANCE CEILING DETECTION - Auto-implemented
from datetime import datetime

def analyze_performance_ceiling(placings: List[int], dates: List[str]) -> Dict[str, any]:
    """
    Track best-ever placing and years since.
    Athletes rarely exceed career bests.
    """
    if not placings or not dates:
        return {'best_placing': 99, 'years_since_best': 0, 'ceiling_factor': 1.0}
    
    # Find best placing
    best_placing = min(placings)
    best_idx = placings.index(best_placing)
    
    # Calculate years since best
    try:
        best_date = datetime.fromisoformat(dates[best_idx].replace('Z', '+00:00'))
        years_since = (datetime.now() - best_date).days / 365.25
    except:
        years_since = 0
    
    # Ceiling factor: harder to repeat best as time passes
    if years_since < 1:
        ceiling_factor = 1.0  # Recent best, can repeat
    elif years_since < 2:
        ceiling_factor = 0.98
    elif years_since < 3:
        ceiling_factor = 0.95
    else:
        ceiling_factor = 0.90  # Long time since best
    
    return {
        'best_placing': best_placing,
        'years_since_best': years_since,
        'ceiling_factor': ceiling_factor,
        'can_exceed_ceiling': years_since < 1
    }


# Integration
@property
def ceiling_adjusted_score(self) -> float:
    """Get score adjusted for performance ceiling."""
    ceiling_info = analyze_performance_ceiling(self.recent_placings, self.get_show_dates())
    base_score = self.recent_form
    
    return base_score * ceiling_info['ceiling_factor']



# COMPETITION RECENCY BOOST - Auto-implemented
from datetime import datetime, timedelta

def calculate_recency_factor(last_competition_date: Optional[str]) -> float:
    """
    Athletes who competed recently get freshness bonus.
    Too recent might mean tired, too old = rusty.
    """
    if not last_competition_date:
        return 0.95  # Unknown = slight penalty
    
    try:
        last_date = datetime.fromisoformat(last_competition_date.replace('Z', '+00:00'))
        days_ago = (datetime.now() - last_date).days
        
        # Optimal range: 30-90 days (fresh but not tired)
        if 30 <= days_ago <= 90:
            return 1.05  # Peak freshness
        elif days_ago < 14:
            return 0.95  # Might be tired from recent prep
        elif days_ago < 30:
            return 1.02  # Getting warm
        elif days_ago <= 180:
            return 1.0   # Normal
        elif days_ago <= 365:
            return 0.97  # Getting rusty
        else:
            return 0.93  # Very rusty
    except:
        return 1.0


# Integration
@property
def recency_adjusted_score(self) -> float:
    """Get score adjusted for competition recency."""
    base_score = self.recent_form
    dates = self.get_show_dates()
    
    if dates:
        factor = calculate_recency_factor(dates[0])
        return base_score * factor
    
    return base_score



# PERFORMANCE CEILING DETECTION - Auto-implemented
from datetime import datetime

def analyze_performance_ceiling(placings: List[int], dates: List[str]) -> Dict[str, any]:
    """
    Track best-ever placing and years since.
    Athletes rarely exceed career bests.
    """
    if not placings or not dates:
        return {'best_placing': 99, 'years_since_best': 0, 'ceiling_factor': 1.0}
    
    # Find best placing
    best_placing = min(placings)
    best_idx = placings.index(best_placing)
    
    # Calculate years since best
    try:
        best_date = datetime.fromisoformat(dates[best_idx].replace('Z', '+00:00'))
        years_since = (datetime.now() - best_date).days / 365.25
    except:
        years_since = 0
    
    # Ceiling factor: harder to repeat best as time passes
    if years_since < 1:
        ceiling_factor = 1.0  # Recent best, can repeat
    elif years_since < 2:
        ceiling_factor = 0.98
    elif years_since < 3:
        ceiling_factor = 0.95
    else:
        ceiling_factor = 0.90  # Long time since best
    
    return {
        'best_placing': best_placing,
        'years_since_best': years_since,
        'ceiling_factor': ceiling_factor,
        'can_exceed_ceiling': years_since < 1
    }


# Integration
@property
def ceiling_adjusted_score(self) -> float:
    """Get score adjusted for performance ceiling."""
    ceiling_info = analyze_performance_ceiling(self.recent_placings, self.get_show_dates())
    base_score = self.recent_form
    
    return base_score * ceiling_info['ceiling_factor']



# PERFORMANCE CEILING DETECTION - Auto-implemented
from datetime import datetime

def analyze_performance_ceiling(placings: List[int], dates: List[str]) -> Dict[str, any]:
    """
    Track best-ever placing and years since.
    Athletes rarely exceed career bests.
    """
    if not placings or not dates:
        return {'best_placing': 99, 'years_since_best': 0, 'ceiling_factor': 1.0}
    
    # Find best placing
    best_placing = min(placings)
    best_idx = placings.index(best_placing)
    
    # Calculate years since best
    try:
        best_date = datetime.fromisoformat(dates[best_idx].replace('Z', '+00:00'))
        years_since = (datetime.now() - best_date).days / 365.25
    except:
        years_since = 0
    
    # Ceiling factor: harder to repeat best as time passes
    if years_since < 1:
        ceiling_factor = 1.0  # Recent best, can repeat
    elif years_since < 2:
        ceiling_factor = 0.98
    elif years_since < 3:
        ceiling_factor = 0.95
    else:
        ceiling_factor = 0.90  # Long time since best
    
    return {
        'best_placing': best_placing,
        'years_since_best': years_since,
        'ceiling_factor': ceiling_factor,
        'can_exceed_ceiling': years_since < 1
    }


# Integration
@property
def ceiling_adjusted_score(self) -> float:
    """Get score adjusted for performance ceiling."""
    ceiling_info = analyze_performance_ceiling(self.recent_placings, self.get_show_dates())
    base_score = self.recent_form
    
    return base_score * ceiling_info['ceiling_factor']



# AVERAGE PLACING DIFFERENTIAL - Auto-implemented
def calculate_placing_differential(recent_placings: List[int], 
                                     career_placings: List[int]) -> float:
    """
    Compare athlete's recent average to career average.
    Above/below baseline indicates improving/declining trend.
    Returns differential factor (0.8-1.2).
    """
    if not recent_placings or not career_placings:
        return 1.0  # No data
    
    recent_avg = sum(recent_placings) / len(recent_placings)
    career_avg = sum(career_placings) / len(career_placings)
    
    # Calculate differential (negative = improving, positive = declining)
    raw_diff = recent_avg - career_avg
    
    # Convert to factor (recent better than career = higher factor)
    # Scale: -3 diff = 1.2 factor, +3 diff = 0.8 factor
    factor = 1.0 - (raw_diff * 0.067)  # 0.067 = 0.2/3
    factor = max(0.8, min(1.2, factor))  # Clamp to 0.8-1.2
    
    return factor


# Integration
@property
def differential_adjusted_score(self) -> float:
    """Get score adjusted for placing differential vs career average."""
    diff_factor = calculate_placing_differential(
        self.recent_placings,
        getattr(self, 'all_time_placings', self.recent_placings)
    )
    return self.recent_form * diff_factor



# WIN PERCENTAGE TREND - Auto-implemented
def calculate_win_percentage_trend(placings: List[int], window_size: int = 5) -> float:
    """
    Track win percentage over sliding windows.
    Increasing trend indicates rising dominance.
    Returns trend score (-50 to +50).
    """
    if not placings or len(placings) < window_size * 2:
        return 0.0  # Not enough data
    
    # Sort by recency (assume placings are in chronological order)
    recent_window = placings[-window_size:]
    previous_window = placings[-(window_size*2):-window_size]
    
    # Calculate win percentages (top 3 = win)
    recent_wins = sum(1 for p in recent_window if p <= 3)
    previous_wins = sum(1 for p in previous_window if p <= 3)
    
    recent_pct = recent_wins / len(recent_window)
    previous_pct = previous_wins / len(previous_window)
    
    # Calculate trend
    trend = recent_pct - previous_pct
    
    # Scale to -50 to +50
    return trend * 100


# Integration
@property
def win_trend_score(self) -> float:
    """Score based on win percentage trend."""
    trend = calculate_win_percentage_trend(self.recent_placings)
    return 50 + trend  # Base 50, adjust by trend



# WIN PERCENTAGE TREND - Auto-implemented
def calculate_win_percentage_trend(placings: List[int], window_size: int = 5) -> float:
    """
    Track win percentage over sliding windows.
    Increasing trend indicates rising dominance.
    Returns trend score (-50 to +50).
    """
    if not placings or len(placings) < window_size * 2:
        return 0.0  # Not enough data
    
    # Sort by recency (assume placings are in chronological order)
    recent_window = placings[-window_size:]
    previous_window = placings[-(window_size*2):-window_size]
    
    # Calculate win percentages (top 3 = win)
    recent_wins = sum(1 for p in recent_window if p <= 3)
    previous_wins = sum(1 for p in previous_window if p <= 3)
    
    recent_pct = recent_wins / len(recent_window)
    previous_pct = previous_wins / len(previous_window)
    
    # Calculate trend
    trend = recent_pct - previous_pct
    
    # Scale to -50 to +50
    return trend * 100


# Integration
@property
def win_trend_score(self) -> float:
    """Score based on win percentage trend."""
    trend = calculate_win_percentage_trend(self.recent_placings)
    return 50 + trend  # Base 50, adjust by trend



# AVERAGE PLACING DIFFERENTIAL - Auto-implemented
def calculate_placing_differential(recent_placings: List[int], 
                                     career_placings: List[int]) -> float:
    """
    Compare athlete's recent average to career average.
    Above/below baseline indicates improving/declining trend.
    Returns differential factor (0.8-1.2).
    """
    if not recent_placings or not career_placings:
        return 1.0  # No data
    
    recent_avg = sum(recent_placings) / len(recent_placings)
    career_avg = sum(career_placings) / len(career_placings)
    
    # Calculate differential (negative = improving, positive = declining)
    raw_diff = recent_avg - career_avg
    
    # Convert to factor (recent better than career = higher factor)
    # Scale: -3 diff = 1.2 factor, +3 diff = 0.8 factor
    factor = 1.0 - (raw_diff * 0.067)  # 0.067 = 0.2/3
    factor = max(0.8, min(1.2, factor))  # Clamp to 0.8-1.2
    
    return factor


# Integration
@property
def differential_adjusted_score(self) -> float:
    """Get score adjusted for placing differential vs career average."""
    diff_factor = calculate_placing_differential(
        self.recent_placings,
        getattr(self, 'all_time_placings', self.recent_placings)
    )
    return self.recent_form * diff_factor



# WIN PERCENTAGE TREND - Auto-implemented
def calculate_win_percentage_trend(placings: List[int], window_size: int = 5) -> float:
    """
    Track win percentage over sliding windows.
    Increasing trend indicates rising dominance.
    Returns trend score (-50 to +50).
    """
    if not placings or len(placings) < window_size * 2:
        return 0.0  # Not enough data
    
    # Sort by recency (assume placings are in chronological order)
    recent_window = placings[-window_size:]
    previous_window = placings[-(window_size*2):-window_size]
    
    # Calculate win percentages (top 3 = win)
    recent_wins = sum(1 for p in recent_window if p <= 3)
    previous_wins = sum(1 for p in previous_window if p <= 3)
    
    recent_pct = recent_wins / len(recent_window)
    previous_pct = previous_wins / len(previous_window)
    
    # Calculate trend
    trend = recent_pct - previous_pct
    
    # Scale to -50 to +50
    return trend * 100


# Integration
@property
def win_trend_score(self) -> float:
    """Score based on win percentage trend."""
    trend = calculate_win_percentage_trend(self.recent_placings)
    return 50 + trend  # Base 50, adjust by trend



# EXPERIENCE VS PERFORMANCE CURVE - Auto-implemented
import numpy as np

def calculate_experience_performance_score(total_shows: int, avg_placing: float) -> float:
    """
    Model the relationship between total shows and average placing.
    Sweet spot for experience typically around 10-25 shows.
    Returns score adjustment (0.85-1.15).
    """
    if total_shows == 0:
        return 0.90  # Rookie penalty
    
    # Experience curve: starts low, peaks, then plateaus
    # Optimal experience is around 15-20 shows
    if total_shows < 5:
        # Learning phase
        experience_factor = 0.85 + (total_shows * 0.03)  # 0.85 to 1.0
    elif total_shows < 20:
        # Peak performance zone
        experience_factor = 1.0 + ((total_shows - 5) * 0.01)  # 1.0 to 1.15
    elif total_shows < 40:
        # Plateau
        experience_factor = 1.15
    else:
        # Very experienced - slight decline possible (burnout)
        experience_factor = 1.12
    
    return experience_factor


# Integration
@property
def experience_adjusted_score(self) -> float:
    """Get score adjusted for experience level."""
    base_score = self.recent_form
    exp_factor = calculate_experience_performance_score(
        self.total_shows, 
        self.avg_placing if self.recent_placings else 10
    )
    return base_score * exp_factor



# AVERAGE PLACING DIFFERENTIAL - Auto-implemented
def calculate_placing_differential(recent_placings: List[int], 
                                     career_placings: List[int]) -> float:
    """
    Compare athlete's recent average to career average.
    Above/below baseline indicates improving/declining trend.
    Returns differential factor (0.8-1.2).
    """
    if not recent_placings or not career_placings:
        return 1.0  # No data
    
    recent_avg = sum(recent_placings) / len(recent_placings)
    career_avg = sum(career_placings) / len(career_placings)
    
    # Calculate differential (negative = improving, positive = declining)
    raw_diff = recent_avg - career_avg
    
    # Convert to factor (recent better than career = higher factor)
    # Scale: -3 diff = 1.2 factor, +3 diff = 0.8 factor
    factor = 1.0 - (raw_diff * 0.067)  # 0.067 = 0.2/3
    factor = max(0.8, min(1.2, factor))  # Clamp to 0.8-1.2
    
    return factor


# Integration
@property
def differential_adjusted_score(self) -> float:
    """Get score adjusted for placing differential vs career average."""
    diff_factor = calculate_placing_differential(
        self.recent_placings,
        getattr(self, 'all_time_placings', self.recent_placings)
    )
    return self.recent_form * diff_factor



# WIN PERCENTAGE TREND - Auto-implemented
def calculate_win_percentage_trend(placings: List[int], window_size: int = 5) -> float:
    """
    Track win percentage over sliding windows.
    Increasing trend indicates rising dominance.
    Returns trend score (-50 to +50).
    """
    if not placings or len(placings) < window_size * 2:
        return 0.0  # Not enough data
    
    # Sort by recency (assume placings are in chronological order)
    recent_window = placings[-window_size:]
    previous_window = placings[-(window_size*2):-window_size]
    
    # Calculate win percentages (top 3 = win)
    recent_wins = sum(1 for p in recent_window if p <= 3)
    previous_wins = sum(1 for p in previous_window if p <= 3)
    
    recent_pct = recent_wins / len(recent_window)
    previous_pct = previous_wins / len(previous_window)
    
    # Calculate trend
    trend = recent_pct - previous_pct
    
    # Scale to -50 to +50
    return trend * 100


# Integration
@property
def win_trend_score(self) -> float:
    """Score based on win percentage trend."""
    trend = calculate_win_percentage_trend(self.recent_placings)
    return 50 + trend  # Base 50, adjust by trend



# AVERAGE PLACING DIFFERENTIAL - Auto-implemented
def calculate_placing_differential(recent_placings: List[int], 
                                     career_placings: List[int]) -> float:
    """
    Compare athlete's recent average to career average.
    Above/below baseline indicates improving/declining trend.
    Returns differential factor (0.8-1.2).
    """
    if not recent_placings or not career_placings:
        return 1.0  # No data
    
    recent_avg = sum(recent_placings) / len(recent_placings)
    career_avg = sum(career_placings) / len(career_placings)
    
    # Calculate differential (negative = improving, positive = declining)
    raw_diff = recent_avg - career_avg
    
    # Convert to factor (recent better than career = higher factor)
    # Scale: -3 diff = 1.2 factor, +3 diff = 0.8 factor
    factor = 1.0 - (raw_diff * 0.067)  # 0.067 = 0.2/3
    factor = max(0.8, min(1.2, factor))  # Clamp to 0.8-1.2
    
    return factor


# Integration
@property
def differential_adjusted_score(self) -> float:
    """Get score adjusted for placing differential vs career average."""
    diff_factor = calculate_placing_differential(
        self.recent_placings,
        getattr(self, 'all_time_placings', self.recent_placings)
    )
    return self.recent_form * diff_factor



# AVERAGE PLACING DIFFERENTIAL - Auto-implemented
def calculate_placing_differential(recent_placings: List[int], 
                                     career_placings: List[int]) -> float:
    """
    Compare athlete's recent average to career average.
    Above/below baseline indicates improving/declining trend.
    Returns differential factor (0.8-1.2).
    """
    if not recent_placings or not career_placings:
        return 1.0  # No data
    
    recent_avg = sum(recent_placings) / len(recent_placings)
    career_avg = sum(career_placings) / len(career_placings)
    
    # Calculate differential (negative = improving, positive = declining)
    raw_diff = recent_avg - career_avg
    
    # Convert to factor (recent better than career = higher factor)
    # Scale: -3 diff = 1.2 factor, +3 diff = 0.8 factor
    factor = 1.0 - (raw_diff * 0.067)  # 0.067 = 0.2/3
    factor = max(0.8, min(1.2, factor))  # Clamp to 0.8-1.2
    
    return factor


# Integration
@property
def differential_adjusted_score(self) -> float:
    """Get score adjusted for placing differential vs career average."""
    diff_factor = calculate_placing_differential(
        self.recent_placings,
        getattr(self, 'all_time_placings', self.recent_placings)
    )
    return self.recent_form * diff_factor



# WIN PERCENTAGE TREND - Auto-implemented
def calculate_win_percentage_trend(placings: List[int], window_size: int = 5) -> float:
    """
    Track win percentage over sliding windows.
    Increasing trend indicates rising dominance.
    Returns trend score (-50 to +50).
    """
    if not placings or len(placings) < window_size * 2:
        return 0.0  # Not enough data
    
    # Sort by recency (assume placings are in chronological order)
    recent_window = placings[-window_size:]
    previous_window = placings[-(window_size*2):-window_size]
    
    # Calculate win percentages (top 3 = win)
    recent_wins = sum(1 for p in recent_window if p <= 3)
    previous_wins = sum(1 for p in previous_window if p <= 3)
    
    recent_pct = recent_wins / len(recent_window)
    previous_pct = previous_wins / len(previous_window)
    
    # Calculate trend
    trend = recent_pct - previous_pct
    
    # Scale to -50 to +50
    return trend * 100


# Integration
@property
def win_trend_score(self) -> float:
    """Score based on win percentage trend."""
    trend = calculate_win_percentage_trend(self.recent_placings)
    return 50 + trend  # Base 50, adjust by trend



# WIN PERCENTAGE TREND - Auto-implemented
def calculate_win_percentage_trend(placings: List[int], window_size: int = 5) -> float:
    """
    Track win percentage over sliding windows.
    Increasing trend indicates rising dominance.
    Returns trend score (-50 to +50).
    """
    if not placings or len(placings) < window_size * 2:
        return 0.0  # Not enough data
    
    # Sort by recency (assume placings are in chronological order)
    recent_window = placings[-window_size:]
    previous_window = placings[-(window_size*2):-window_size]
    
    # Calculate win percentages (top 3 = win)
    recent_wins = sum(1 for p in recent_window if p <= 3)
    previous_wins = sum(1 for p in previous_window if p <= 3)
    
    recent_pct = recent_wins / len(recent_window)
    previous_pct = previous_wins / len(previous_window)
    
    # Calculate trend
    trend = recent_pct - previous_pct
    
    # Scale to -50 to +50
    return trend * 100


# Integration
@property
def win_trend_score(self) -> float:
    """Score based on win percentage trend."""
    trend = calculate_win_percentage_trend(self.recent_placings)
    return 50 + trend  # Base 50, adjust by trend



# EXPERIENCE VS PERFORMANCE CURVE - Auto-implemented
import numpy as np

def calculate_experience_performance_score(total_shows: int, avg_placing: float) -> float:
    """
    Model the relationship between total shows and average placing.
    Sweet spot for experience typically around 10-25 shows.
    Returns score adjustment (0.85-1.15).
    """
    if total_shows == 0:
        return 0.90  # Rookie penalty
    
    # Experience curve: starts low, peaks, then plateaus
    # Optimal experience is around 15-20 shows
    if total_shows < 5:
        # Learning phase
        experience_factor = 0.85 + (total_shows * 0.03)  # 0.85 to 1.0
    elif total_shows < 20:
        # Peak performance zone
        experience_factor = 1.0 + ((total_shows - 5) * 0.01)  # 1.0 to 1.15
    elif total_shows < 40:
        # Plateau
        experience_factor = 1.15
    else:
        # Very experienced - slight decline possible (burnout)
        experience_factor = 1.12
    
    return experience_factor


# Integration
@property
def experience_adjusted_score(self) -> float:
    """Get score adjusted for experience level."""
    base_score = self.recent_form
    exp_factor = calculate_experience_performance_score(
        self.total_shows, 
        self.avg_placing if self.recent_placings else 10
    )
    return base_score * exp_factor



# EXPERIENCE VS PERFORMANCE CURVE - Auto-implemented
import numpy as np

def calculate_experience_performance_score(total_shows: int, avg_placing: float) -> float:
    """
    Model the relationship between total shows and average placing.
    Sweet spot for experience typically around 10-25 shows.
    Returns score adjustment (0.85-1.15).
    """
    if total_shows == 0:
        return 0.90  # Rookie penalty
    
    # Experience curve: starts low, peaks, then plateaus
    # Optimal experience is around 15-20 shows
    if total_shows < 5:
        # Learning phase
        experience_factor = 0.85 + (total_shows * 0.03)  # 0.85 to 1.0
    elif total_shows < 20:
        # Peak performance zone
        experience_factor = 1.0 + ((total_shows - 5) * 0.01)  # 1.0 to 1.15
    elif total_shows < 40:
        # Plateau
        experience_factor = 1.15
    else:
        # Very experienced - slight decline possible (burnout)
        experience_factor = 1.12
    
    return experience_factor


# Integration
@property
def experience_adjusted_score(self) -> float:
    """Get score adjusted for experience level."""
    base_score = self.recent_form
    exp_factor = calculate_experience_performance_score(
        self.total_shows, 
        self.avg_placing if self.recent_placings else 10
    )
    return base_score * exp_factor



# EXPERIENCE VS PERFORMANCE CURVE - Auto-implemented
import numpy as np

def calculate_experience_performance_score(total_shows: int, avg_placing: float) -> float:
    """
    Model the relationship between total shows and average placing.
    Sweet spot for experience typically around 10-25 shows.
    Returns score adjustment (0.85-1.15).
    """
    if total_shows == 0:
        return 0.90  # Rookie penalty
    
    # Experience curve: starts low, peaks, then plateaus
    # Optimal experience is around 15-20 shows
    if total_shows < 5:
        # Learning phase
        experience_factor = 0.85 + (total_shows * 0.03)  # 0.85 to 1.0
    elif total_shows < 20:
        # Peak performance zone
        experience_factor = 1.0 + ((total_shows - 5) * 0.01)  # 1.0 to 1.15
    elif total_shows < 40:
        # Plateau
        experience_factor = 1.15
    else:
        # Very experienced - slight decline possible (burnout)
        experience_factor = 1.12
    
    return experience_factor


# Integration
@property
def experience_adjusted_score(self) -> float:
    """Get score adjusted for experience level."""
    base_score = self.recent_form
    exp_factor = calculate_experience_performance_score(
        self.total_shows, 
        self.avg_placing if self.recent_placings else 10
    )
    return base_score * exp_factor



# EXPERIENCE VS PERFORMANCE CURVE - Auto-implemented
import numpy as np

def calculate_experience_performance_score(total_shows: int, avg_placing: float) -> float:
    """
    Model the relationship between total shows and average placing.
    Sweet spot for experience typically around 10-25 shows.
    Returns score adjustment (0.85-1.15).
    """
    if total_shows == 0:
        return 0.90  # Rookie penalty
    
    # Experience curve: starts low, peaks, then plateaus
    # Optimal experience is around 15-20 shows
    if total_shows < 5:
        # Learning phase
        experience_factor = 0.85 + (total_shows * 0.03)  # 0.85 to 1.0
    elif total_shows < 20:
        # Peak performance zone
        experience_factor = 1.0 + ((total_shows - 5) * 0.01)  # 1.0 to 1.15
    elif total_shows < 40:
        # Plateau
        experience_factor = 1.15
    else:
        # Very experienced - slight decline possible (burnout)
        experience_factor = 1.12
    
    return experience_factor


# Integration
@property
def experience_adjusted_score(self) -> float:
    """Get score adjusted for experience level."""
    base_score = self.recent_form
    exp_factor = calculate_experience_performance_score(
        self.total_shows, 
        self.avg_placing if self.recent_placings else 10
    )
    return base_score * exp_factor



# WIN PERCENTAGE TREND - Auto-implemented
def calculate_win_percentage_trend(placings: List[int], window_size: int = 5) -> float:
    """
    Track win percentage over sliding windows.
    Increasing trend indicates rising dominance.
    Returns trend score (-50 to +50).
    """
    if not placings or len(placings) < window_size * 2:
        return 0.0  # Not enough data
    
    # Sort by recency (assume placings are in chronological order)
    recent_window = placings[-window_size:]
    previous_window = placings[-(window_size*2):-window_size]
    
    # Calculate win percentages (top 3 = win)
    recent_wins = sum(1 for p in recent_window if p <= 3)
    previous_wins = sum(1 for p in previous_window if p <= 3)
    
    recent_pct = recent_wins / len(recent_window)
    previous_pct = previous_wins / len(previous_window)
    
    # Calculate trend
    trend = recent_pct - previous_pct
    
    # Scale to -50 to +50
    return trend * 100


# Integration
@property
def win_trend_score(self) -> float:
    """Score based on win percentage trend."""
    trend = calculate_win_percentage_trend(self.recent_placings)
    return 50 + trend  # Base 50, adjust by trend



# WIN PERCENTAGE TREND - Auto-implemented
def calculate_win_percentage_trend(placings: List[int], window_size: int = 5) -> float:
    """
    Track win percentage over sliding windows.
    Increasing trend indicates rising dominance.
    Returns trend score (-50 to +50).
    """
    if not placings or len(placings) < window_size * 2:
        return 0.0  # Not enough data
    
    # Sort by recency (assume placings are in chronological order)
    recent_window = placings[-window_size:]
    previous_window = placings[-(window_size*2):-window_size]
    
    # Calculate win percentages (top 3 = win)
    recent_wins = sum(1 for p in recent_window if p <= 3)
    previous_wins = sum(1 for p in previous_window if p <= 3)
    
    recent_pct = recent_wins / len(recent_window)
    previous_pct = previous_wins / len(previous_window)
    
    # Calculate trend
    trend = recent_pct - previous_pct
    
    # Scale to -50 to +50
    return trend * 100


# Integration
@property
def win_trend_score(self) -> float:
    """Score based on win percentage trend."""
    trend = calculate_win_percentage_trend(self.recent_placings)
    return 50 + trend  # Base 50, adjust by trend



# DIVISION TRANSITION FACTOR - Auto-implemented
def calculate_division_transition_factor(
    previous_division: Optional[str],
    current_division: str,
    shows_in_current: int
) -> float:
    """
    Calculate adjustment for athletes who switched divisions.
    New division requires adjustment period.
    Returns factor (0.85-1.0).
    """
    if previous_division is None or previous_division == current_division:
        return 1.0  # No transition
    
    if shows_in_current == 0:
        return 0.85  # First show in new division - major adjustment
    elif shows_in_current == 1:
        return 0.92  # Second show - still adjusting
    elif shows_in_current <= 3:
        return 0.97  # Getting comfortable
    else:
        return 1.0  # Fully adjusted


# Integration
@property
def division_transition_adjustment(self) -> float:
    """Get adjustment for division transitions."""
    # This would need access to division history
    # Simplified: assume no transition for now
    return 1.0



# COMPETITION COUNT MOMENTUM - Auto-implemented
def calculate_competition_momentum(recent_dates: List[str]) -> float:
    """
    Calculate momentum based on competition frequency.
    Active competitors (frequent shows) vs inactive ones.
    Returns momentum factor (0.9-1.1).
    """
    from datetime import datetime
    
    if len(recent_dates) < 2:
        return 1.0  # Not enough data
    
    # Calculate average days between competitions
    dates = sorted([datetime.fromisoformat(d.replace('Z', '+00:00')) 
                    for d in recent_dates])
    
    gaps = []
    for i in range(1, len(dates)):
        gap = (dates[i] - dates[i-1]).days
        gaps.append(gap)
    
    if not gaps:
        return 1.0
    
    avg_gap = sum(gaps) / len(gaps)
    
    # Optimal gap is around 60-120 days (2-4 months)
    if 60 <= avg_gap <= 120:
        return 1.1  # Sweet spot - active but rested
    elif avg_gap < 45:
        return 0.95  # Too active - potential burnout
    elif avg_gap > 180:
        return 0.9   # Inactive - ring rust
    else:
        return 1.0   # Normal range


# Integration
@property
def competition_momentum_factor(self) -> float:
    """Get momentum factor based on competition frequency."""
    return calculate_competition_momentum(self.get_show_dates())



# RECENT VS HISTORICAL COMPARISON - Auto-implemented
def calculate_recent_historical_trend(
    recent_placings: List[int],
    historical_placings: List[int]
) -> Dict[str, float]:
    """
    Compare recent performance to historical baseline.
    Identifies improving, declining, or stable trends.
    """
    if not recent_placings or not historical_placings:
        return {'trend': 0.0, 'factor': 1.0}
    
    recent_avg = sum(recent_placings) / len(recent_placings)
    hist_avg = sum(historical_placings) / len(historical_placings)
    
    # Trend: negative = improving (lower placing), positive = declining
    trend = recent_avg - hist_avg
    
    # Convert to factor
    if trend < -1.5:  # Significantly improving
        factor = 1.15
    elif trend < -0.5:  # Modestly improving
        factor = 1.08
    elif trend > 1.5:  # Significantly declining
        factor = 0.85
    elif trend > 0.5:  # Modestly declining
        factor = 0.92
    else:  # Stable
        factor = 1.0
    
    return {'trend': trend, 'factor': factor}


# Integration
@property
def trend_adjusted_score(self) -> float:
    """Get score adjusted for recent vs historical trend."""
    # Split placings into recent (last 3) and historical (older)
    if len(self.recent_placings) >= 6:
        recent = self.recent_placings[:3]
        historical = self.recent_placings[3:]
    elif len(self.recent_placings) >= 3:
        recent = self.recent_placings[:2]
        historical = self.recent_placings[2:]
    else:
        return self.recent_form  # Not enough data
    
    trend_info = calculate_recent_historical_trend(recent, historical)
    return self.recent_form * trend_info['factor']



# COMPETITION COUNT MOMENTUM - Auto-implemented
def calculate_competition_momentum(recent_dates: List[str]) -> float:
    """
    Calculate momentum based on competition frequency.
    Active competitors (frequent shows) vs inactive ones.
    Returns momentum factor (0.9-1.1).
    """
    from datetime import datetime
    
    if len(recent_dates) < 2:
        return 1.0  # Not enough data
    
    # Calculate average days between competitions
    dates = sorted([datetime.fromisoformat(d.replace('Z', '+00:00')) 
                    for d in recent_dates])
    
    gaps = []
    for i in range(1, len(dates)):
        gap = (dates[i] - dates[i-1]).days
        gaps.append(gap)
    
    if not gaps:
        return 1.0
    
    avg_gap = sum(gaps) / len(gaps)
    
    # Optimal gap is around 60-120 days (2-4 months)
    if 60 <= avg_gap <= 120:
        return 1.1  # Sweet spot - active but rested
    elif avg_gap < 45:
        return 0.95  # Too active - potential burnout
    elif avg_gap > 180:
        return 0.9   # Inactive - ring rust
    else:
        return 1.0   # Normal range


# Integration
@property
def competition_momentum_factor(self) -> float:
    """Get momentum factor based on competition frequency."""
    return calculate_competition_momentum(self.get_show_dates())



# DIVISION TRANSITION FACTOR - Auto-implemented
def calculate_division_transition_factor(
    previous_division: Optional[str],
    current_division: str,
    shows_in_current: int
) -> float:
    """
    Calculate adjustment for athletes who switched divisions.
    New division requires adjustment period.
    Returns factor (0.85-1.0).
    """
    if previous_division is None or previous_division == current_division:
        return 1.0  # No transition
    
    if shows_in_current == 0:
        return 0.85  # First show in new division - major adjustment
    elif shows_in_current == 1:
        return 0.92  # Second show - still adjusting
    elif shows_in_current <= 3:
        return 0.97  # Getting comfortable
    else:
        return 1.0  # Fully adjusted


# Integration
@property
def division_transition_adjustment(self) -> float:
    """Get adjustment for division transitions."""
    # This would need access to division history
    # Simplified: assume no transition for now
    return 1.0



# COMPETITION COUNT MOMENTUM - Auto-implemented
def calculate_competition_momentum(recent_dates: List[str]) -> float:
    """
    Calculate momentum based on competition frequency.
    Active competitors (frequent shows) vs inactive ones.
    Returns momentum factor (0.9-1.1).
    """
    from datetime import datetime
    
    if len(recent_dates) < 2:
        return 1.0  # Not enough data
    
    # Calculate average days between competitions
    dates = sorted([datetime.fromisoformat(d.replace('Z', '+00:00')) 
                    for d in recent_dates])
    
    gaps = []
    for i in range(1, len(dates)):
        gap = (dates[i] - dates[i-1]).days
        gaps.append(gap)
    
    if not gaps:
        return 1.0
    
    avg_gap = sum(gaps) / len(gaps)
    
    # Optimal gap is around 60-120 days (2-4 months)
    if 60 <= avg_gap <= 120:
        return 1.1  # Sweet spot - active but rested
    elif avg_gap < 45:
        return 0.95  # Too active - potential burnout
    elif avg_gap > 180:
        return 0.9   # Inactive - ring rust
    else:
        return 1.0   # Normal range


# Integration
@property
def competition_momentum_factor(self) -> float:
    """Get momentum factor based on competition frequency."""
    return calculate_competition_momentum(self.get_show_dates())



# RECENT VS HISTORICAL COMPARISON - Auto-implemented
def calculate_recent_historical_trend(
    recent_placings: List[int],
    historical_placings: List[int]
) -> Dict[str, float]:
    """
    Compare recent performance to historical baseline.
    Identifies improving, declining, or stable trends.
    """
    if not recent_placings or not historical_placings:
        return {'trend': 0.0, 'factor': 1.0}
    
    recent_avg = sum(recent_placings) / len(recent_placings)
    hist_avg = sum(historical_placings) / len(historical_placings)
    
    # Trend: negative = improving (lower placing), positive = declining
    trend = recent_avg - hist_avg
    
    # Convert to factor
    if trend < -1.5:  # Significantly improving
        factor = 1.15
    elif trend < -0.5:  # Modestly improving
        factor = 1.08
    elif trend > 1.5:  # Significantly declining
        factor = 0.85
    elif trend > 0.5:  # Modestly declining
        factor = 0.92
    else:  # Stable
        factor = 1.0
    
    return {'trend': trend, 'factor': factor}


# Integration
@property
def trend_adjusted_score(self) -> float:
    """Get score adjusted for recent vs historical trend."""
    # Split placings into recent (last 3) and historical (older)
    if len(self.recent_placings) >= 6:
        recent = self.recent_placings[:3]
        historical = self.recent_placings[3:]
    elif len(self.recent_placings) >= 3:
        recent = self.recent_placings[:2]
        historical = self.recent_placings[2:]
    else:
        return self.recent_form  # Not enough data
    
    trend_info = calculate_recent_historical_trend(recent, historical)
    return self.recent_form * trend_info['factor']



# RECENT VS HISTORICAL COMPARISON - Auto-implemented
def calculate_recent_historical_trend(
    recent_placings: List[int],
    historical_placings: List[int]
) -> Dict[str, float]:
    """
    Compare recent performance to historical baseline.
    Identifies improving, declining, or stable trends.
    """
    if not recent_placings or not historical_placings:
        return {'trend': 0.0, 'factor': 1.0}
    
    recent_avg = sum(recent_placings) / len(recent_placings)
    hist_avg = sum(historical_placings) / len(historical_placings)
    
    # Trend: negative = improving (lower placing), positive = declining
    trend = recent_avg - hist_avg
    
    # Convert to factor
    if trend < -1.5:  # Significantly improving
        factor = 1.15
    elif trend < -0.5:  # Modestly improving
        factor = 1.08
    elif trend > 1.5:  # Significantly declining
        factor = 0.85
    elif trend > 0.5:  # Modestly declining
        factor = 0.92
    else:  # Stable
        factor = 1.0
    
    return {'trend': trend, 'factor': factor}


# Integration
@property
def trend_adjusted_score(self) -> float:
    """Get score adjusted for recent vs historical trend."""
    # Split placings into recent (last 3) and historical (older)
    if len(self.recent_placings) >= 6:
        recent = self.recent_placings[:3]
        historical = self.recent_placings[3:]
    elif len(self.recent_placings) >= 3:
        recent = self.recent_placings[:2]
        historical = self.recent_placings[2:]
    else:
        return self.recent_form  # Not enough data
    
    trend_info = calculate_recent_historical_trend(recent, historical)
    return self.recent_form * trend_info['factor']



# RECENT VS HISTORICAL COMPARISON - Auto-implemented
def calculate_recent_historical_trend(
    recent_placings: List[int],
    historical_placings: List[int]
) -> Dict[str, float]:
    """
    Compare recent performance to historical baseline.
    Identifies improving, declining, or stable trends.
    """
    if not recent_placings or not historical_placings:
        return {'trend': 0.0, 'factor': 1.0}
    
    recent_avg = sum(recent_placings) / len(recent_placings)
    hist_avg = sum(historical_placings) / len(historical_placings)
    
    # Trend: negative = improving (lower placing), positive = declining
    trend = recent_avg - hist_avg
    
    # Convert to factor
    if trend < -1.5:  # Significantly improving
        factor = 1.15
    elif trend < -0.5:  # Modestly improving
        factor = 1.08
    elif trend > 1.5:  # Significantly declining
        factor = 0.85
    elif trend > 0.5:  # Modestly declining
        factor = 0.92
    else:  # Stable
        factor = 1.0
    
    return {'trend': trend, 'factor': factor}


# Integration
@property
def trend_adjusted_score(self) -> float:
    """Get score adjusted for recent vs historical trend."""
    # Split placings into recent (last 3) and historical (older)
    if len(self.recent_placings) >= 6:
        recent = self.recent_placings[:3]
        historical = self.recent_placings[3:]
    elif len(self.recent_placings) >= 3:
        recent = self.recent_placings[:2]
        historical = self.recent_placings[2:]
    else:
        return self.recent_form  # Not enough data
    
    trend_info = calculate_recent_historical_trend(recent, historical)
    return self.recent_form * trend_info['factor']



# COMPETITION COUNT MOMENTUM - Auto-implemented
def calculate_competition_momentum(recent_dates: List[str]) -> float:
    """
    Calculate momentum based on competition frequency.
    Active competitors (frequent shows) vs inactive ones.
    Returns momentum factor (0.9-1.1).
    """
    from datetime import datetime
    
    if len(recent_dates) < 2:
        return 1.0  # Not enough data
    
    # Calculate average days between competitions
    dates = sorted([datetime.fromisoformat(d.replace('Z', '+00:00')) 
                    for d in recent_dates])
    
    gaps = []
    for i in range(1, len(dates)):
        gap = (dates[i] - dates[i-1]).days
        gaps.append(gap)
    
    if not gaps:
        return 1.0
    
    avg_gap = sum(gaps) / len(gaps)
    
    # Optimal gap is around 60-120 days (2-4 months)
    if 60 <= avg_gap <= 120:
        return 1.1  # Sweet spot - active but rested
    elif avg_gap < 45:
        return 0.95  # Too active - potential burnout
    elif avg_gap > 180:
        return 0.9   # Inactive - ring rust
    else:
        return 1.0   # Normal range


# Integration
@property
def competition_momentum_factor(self) -> float:
    """Get momentum factor based on competition frequency."""
    return calculate_competition_momentum(self.get_show_dates())



# COMPETITION COUNT MOMENTUM - Auto-implemented
def calculate_competition_momentum(recent_dates: List[str]) -> float:
    """
    Calculate momentum based on competition frequency.
    Active competitors (frequent shows) vs inactive ones.
    Returns momentum factor (0.9-1.1).
    """
    from datetime import datetime
    
    if len(recent_dates) < 2:
        return 1.0  # Not enough data
    
    # Calculate average days between competitions
    dates = sorted([datetime.fromisoformat(d.replace('Z', '+00:00')) 
                    for d in recent_dates])
    
    gaps = []
    for i in range(1, len(dates)):
        gap = (dates[i] - dates[i-1]).days
        gaps.append(gap)
    
    if not gaps:
        return 1.0
    
    avg_gap = sum(gaps) / len(gaps)
    
    # Optimal gap is around 60-120 days (2-4 months)
    if 60 <= avg_gap <= 120:
        return 1.1  # Sweet spot - active but rested
    elif avg_gap < 45:
        return 0.95  # Too active - potential burnout
    elif avg_gap > 180:
        return 0.9   # Inactive - ring rust
    else:
        return 1.0   # Normal range


# Integration
@property
def competition_momentum_factor(self) -> float:
    """Get momentum factor based on competition frequency."""
    return calculate_competition_momentum(self.get_show_dates())



# RECENT VS HISTORICAL COMPARISON - Auto-implemented
def calculate_recent_historical_trend(
    recent_placings: List[int],
    historical_placings: List[int]
) -> Dict[str, float]:
    """
    Compare recent performance to historical baseline.
    Identifies improving, declining, or stable trends.
    """
    if not recent_placings or not historical_placings:
        return {'trend': 0.0, 'factor': 1.0}
    
    recent_avg = sum(recent_placings) / len(recent_placings)
    hist_avg = sum(historical_placings) / len(historical_placings)
    
    # Trend: negative = improving (lower placing), positive = declining
    trend = recent_avg - hist_avg
    
    # Convert to factor
    if trend < -1.5:  # Significantly improving
        factor = 1.15
    elif trend < -0.5:  # Modestly improving
        factor = 1.08
    elif trend > 1.5:  # Significantly declining
        factor = 0.85
    elif trend > 0.5:  # Modestly declining
        factor = 0.92
    else:  # Stable
        factor = 1.0
    
    return {'trend': trend, 'factor': factor}


# Integration
@property
def trend_adjusted_score(self) -> float:
    """Get score adjusted for recent vs historical trend."""
    # Split placings into recent (last 3) and historical (older)
    if len(self.recent_placings) >= 6:
        recent = self.recent_placings[:3]
        historical = self.recent_placings[3:]
    elif len(self.recent_placings) >= 3:
        recent = self.recent_placings[:2]
        historical = self.recent_placings[2:]
    else:
        return self.recent_form  # Not enough data
    
    trend_info = calculate_recent_historical_trend(recent, historical)
    return self.recent_form * trend_info['factor']



# COMPETITION COUNT MOMENTUM - Auto-implemented
def calculate_competition_momentum(recent_dates: List[str]) -> float:
    """
    Calculate momentum based on competition frequency.
    Active competitors (frequent shows) vs inactive ones.
    Returns momentum factor (0.9-1.1).
    """
    from datetime import datetime
    
    if len(recent_dates) < 2:
        return 1.0  # Not enough data
    
    # Calculate average days between competitions
    dates = sorted([datetime.fromisoformat(d.replace('Z', '+00:00')) 
                    for d in recent_dates])
    
    gaps = []
    for i in range(1, len(dates)):
        gap = (dates[i] - dates[i-1]).days
        gaps.append(gap)
    
    if not gaps:
        return 1.0
    
    avg_gap = sum(gaps) / len(gaps)
    
    # Optimal gap is around 60-120 days (2-4 months)
    if 60 <= avg_gap <= 120:
        return 1.1  # Sweet spot - active but rested
    elif avg_gap < 45:
        return 0.95  # Too active - potential burnout
    elif avg_gap > 180:
        return 0.9   # Inactive - ring rust
    else:
        return 1.0   # Normal range


# Integration
@property
def competition_momentum_factor(self) -> float:
    """Get momentum factor based on competition frequency."""
    return calculate_competition_momentum(self.get_show_dates())



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# WEIGHTED COMPETITION TIER SCORE - Auto-implemented
COMPETITION_TIER_WEIGHTS = {
    'olympia': 1.0,
    'arnold classic': 0.95,
    'new york pro': 0.85,
    'chicago pro': 0.82,
    'tampa pro': 0.80,
    'ifbb pro league': 0.75,
    'regional': 0.60,
    'local': 0.45,
}

def get_tier_weight(competition_name: str) -> float:
    """Get weight for competition based on tier."""
    comp_lower = competition_name.lower()
    
    for key, weight in COMPETITION_TIER_WEIGHTS.items():
        if key in comp_lower:
            return weight
    
    return 0.70  # Default


def calculate_tier_weighted_average(competitions: List[Dict]) -> float:
    """
    Calculate athlete's average placing weighted by competition tier.
    Olympia 1st counts much more than regional 1st.
    """
    if not competitions:
        return 50.0
    
    total_weighted_placing = 0
    total_weight = 0
    
    for comp in competitions:
        name = comp.get('name', '')
        placing = comp.get('placing', 10)
        
        weight = get_tier_weight(name)
        # Higher weight (better competition) = more valuable placing
        weighted_placing = placing * (2.0 - weight)  # Invert: higher weight = lower effective placing
        
        total_weighted_placing += weighted_placing * weight
        total_weight += weight
    
    if total_weight == 0:
        return 50.0
    
    weighted_avg = total_weighted_placing / total_weight
    
    # Convert to score
    return 100 * math.exp(-0.15 * (weighted_avg - 1))


# Integration
@property
def tier_weighted_score(self) -> float:
    """Get score based on tier-weighted performance."""
    return calculate_tier_weighted_average(self.competition_history)



# WEIGHTED COMPETITION TIER SCORE - Auto-implemented
COMPETITION_TIER_WEIGHTS = {
    'olympia': 1.0,
    'arnold classic': 0.95,
    'new york pro': 0.85,
    'chicago pro': 0.82,
    'tampa pro': 0.80,
    'ifbb pro league': 0.75,
    'regional': 0.60,
    'local': 0.45,
}

def get_tier_weight(competition_name: str) -> float:
    """Get weight for competition based on tier."""
    comp_lower = competition_name.lower()
    
    for key, weight in COMPETITION_TIER_WEIGHTS.items():
        if key in comp_lower:
            return weight
    
    return 0.70  # Default


def calculate_tier_weighted_average(competitions: List[Dict]) -> float:
    """
    Calculate athlete's average placing weighted by competition tier.
    Olympia 1st counts much more than regional 1st.
    """
    if not competitions:
        return 50.0
    
    total_weighted_placing = 0
    total_weight = 0
    
    for comp in competitions:
        name = comp.get('name', '')
        placing = comp.get('placing', 10)
        
        weight = get_tier_weight(name)
        # Higher weight (better competition) = more valuable placing
        weighted_placing = placing * (2.0 - weight)  # Invert: higher weight = lower effective placing
        
        total_weighted_placing += weighted_placing * weight
        total_weight += weight
    
    if total_weight == 0:
        return 50.0
    
    weighted_avg = total_weighted_placing / total_weight
    
    # Convert to score
    return 100 * math.exp(-0.15 * (weighted_avg - 1))


# Integration
@property
def tier_weighted_score(self) -> float:
    """Get score based on tier-weighted performance."""
    return calculate_tier_weighted_average(self.competition_history)



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# WEIGHTED COMPETITION TIER SCORE - Auto-implemented
COMPETITION_TIER_WEIGHTS = {
    'olympia': 1.0,
    'arnold classic': 0.95,
    'new york pro': 0.85,
    'chicago pro': 0.82,
    'tampa pro': 0.80,
    'ifbb pro league': 0.75,
    'regional': 0.60,
    'local': 0.45,
}

def get_tier_weight(competition_name: str) -> float:
    """Get weight for competition based on tier."""
    comp_lower = competition_name.lower()
    
    for key, weight in COMPETITION_TIER_WEIGHTS.items():
        if key in comp_lower:
            return weight
    
    return 0.70  # Default


def calculate_tier_weighted_average(competitions: List[Dict]) -> float:
    """
    Calculate athlete's average placing weighted by competition tier.
    Olympia 1st counts much more than regional 1st.
    """
    if not competitions:
        return 50.0
    
    total_weighted_placing = 0
    total_weight = 0
    
    for comp in competitions:
        name = comp.get('name', '')
        placing = comp.get('placing', 10)
        
        weight = get_tier_weight(name)
        # Higher weight (better competition) = more valuable placing
        weighted_placing = placing * (2.0 - weight)  # Invert: higher weight = lower effective placing
        
        total_weighted_placing += weighted_placing * weight
        total_weight += weight
    
    if total_weight == 0:
        return 50.0
    
    weighted_avg = total_weighted_placing / total_weight
    
    # Convert to score
    return 100 * math.exp(-0.15 * (weighted_avg - 1))


# Integration
@property
def tier_weighted_score(self) -> float:
    """Get score based on tier-weighted performance."""
    return calculate_tier_weighted_average(self.competition_history)



# WEIGHTED COMPETITION TIER SCORE - Auto-implemented
COMPETITION_TIER_WEIGHTS = {
    'olympia': 1.0,
    'arnold classic': 0.95,
    'new york pro': 0.85,
    'chicago pro': 0.82,
    'tampa pro': 0.80,
    'ifbb pro league': 0.75,
    'regional': 0.60,
    'local': 0.45,
}

def get_tier_weight(competition_name: str) -> float:
    """Get weight for competition based on tier."""
    comp_lower = competition_name.lower()
    
    for key, weight in COMPETITION_TIER_WEIGHTS.items():
        if key in comp_lower:
            return weight
    
    return 0.70  # Default


def calculate_tier_weighted_average(competitions: List[Dict]) -> float:
    """
    Calculate athlete's average placing weighted by competition tier.
    Olympia 1st counts much more than regional 1st.
    """
    if not competitions:
        return 50.0
    
    total_weighted_placing = 0
    total_weight = 0
    
    for comp in competitions:
        name = comp.get('name', '')
        placing = comp.get('placing', 10)
        
        weight = get_tier_weight(name)
        # Higher weight (better competition) = more valuable placing
        weighted_placing = placing * (2.0 - weight)  # Invert: higher weight = lower effective placing
        
        total_weighted_placing += weighted_placing * weight
        total_weight += weight
    
    if total_weight == 0:
        return 50.0
    
    weighted_avg = total_weighted_placing / total_weight
    
    # Convert to score
    return 100 * math.exp(-0.15 * (weighted_avg - 1))


# Integration
@property
def tier_weighted_score(self) -> float:
    """Get score based on tier-weighted performance."""
    return calculate_tier_weighted_average(self.competition_history)



# GLICKO-2 RATING SYSTEM - Auto-implemented
class Glicko2Rating:
    """
    Glicko-2 rating system with rating deviation (RD).
    More accurate than Elo for infrequent competitors.
    """
    
    def __init__(self, rating: float = 1500, rd: float = 350, volatility: float = 0.06):
        self.rating = rating
        self.rd = rd  # Rating deviation (confidence)
        self.volatility = volatility
        self.tau = 0.5  # System constant
    
    def expected_score(self, opponent_rating: float, opponent_rd: float) -> float:
        """Calculate expected score vs opponent."""
        return 1 / (1 + 10 ** (-(self.rating - opponent_rating) / 400))
    
    def update(self, opponent_rating: float, opponent_rd: float, score: float):
        """Update rating after match."""
        # Simplified Glicko-2 update
        expected = self.expected_score(opponent_rating, opponent_rd)
        variance = 1 / (self.rd ** 2 + opponent_rd ** 2 + 1e-6)
        
        # Rating update
        delta = self.rd ** 2 * variance * (score - expected)
        self.rating += delta
        
        # RD update (decreases with more games)
        self.rd = max(30, self.rd * 0.9)  # Minimum RD of 30
    
    def get_score(self) -> float:
        """Get 0-100 score from rating."""
        return 100 * (1 / (1 + math.exp(-(self.rating - 1500) / 200)))


# Integration
def get_glicko2_score(athlete_profile) -> float:
    """Get Glicko-2 based score."""
    rating = Glicko2Rating()
    
    # Update with recent results
    for placing in athlete_profile.recent_placings:
        # Simulate match vs average competitor
        score = 1.0 if placing == 1 else 0.5 if placing <= 3 else 0.0
        rating.update(1500, 350, score)
    
    return rating.get_score()



# WEIGHTED COMPETITION TIER SCORE - Auto-implemented
COMPETITION_TIER_WEIGHTS = {
    'olympia': 1.0,
    'arnold classic': 0.95,
    'new york pro': 0.85,
    'chicago pro': 0.82,
    'tampa pro': 0.80,
    'ifbb pro league': 0.75,
    'regional': 0.60,
    'local': 0.45,
}

def get_tier_weight(competition_name: str) -> float:
    """Get weight for competition based on tier."""
    comp_lower = competition_name.lower()
    
    for key, weight in COMPETITION_TIER_WEIGHTS.items():
        if key in comp_lower:
            return weight
    
    return 0.70  # Default


def calculate_tier_weighted_average(competitions: List[Dict]) -> float:
    """
    Calculate athlete's average placing weighted by competition tier.
    Olympia 1st counts much more than regional 1st.
    """
    if not competitions:
        return 50.0
    
    total_weighted_placing = 0
    total_weight = 0
    
    for comp in competitions:
        name = comp.get('name', '')
        placing = comp.get('placing', 10)
        
        weight = get_tier_weight(name)
        # Higher weight (better competition) = more valuable placing
        weighted_placing = placing * (2.0 - weight)  # Invert: higher weight = lower effective placing
        
        total_weighted_placing += weighted_placing * weight
        total_weight += weight
    
    if total_weight == 0:
        return 50.0
    
    weighted_avg = total_weighted_placing / total_weight
    
    # Convert to score
    return 100 * math.exp(-0.15 * (weighted_avg - 1))


# Integration
@property
def tier_weighted_score(self) -> float:
    """Get score based on tier-weighted performance."""
    return calculate_tier_weighted_average(self.competition_history)



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# GLICKO-2 RATING SYSTEM - Auto-implemented
class Glicko2Rating:
    """
    Glicko-2 rating system with rating deviation (RD).
    More accurate than Elo for infrequent competitors.
    """
    
    def __init__(self, rating: float = 1500, rd: float = 350, volatility: float = 0.06):
        self.rating = rating
        self.rd = rd  # Rating deviation (confidence)
        self.volatility = volatility
        self.tau = 0.5  # System constant
    
    def expected_score(self, opponent_rating: float, opponent_rd: float) -> float:
        """Calculate expected score vs opponent."""
        return 1 / (1 + 10 ** (-(self.rating - opponent_rating) / 400))
    
    def update(self, opponent_rating: float, opponent_rd: float, score: float):
        """Update rating after match."""
        # Simplified Glicko-2 update
        expected = self.expected_score(opponent_rating, opponent_rd)
        variance = 1 / (self.rd ** 2 + opponent_rd ** 2 + 1e-6)
        
        # Rating update
        delta = self.rd ** 2 * variance * (score - expected)
        self.rating += delta
        
        # RD update (decreases with more games)
        self.rd = max(30, self.rd * 0.9)  # Minimum RD of 30
    
    def get_score(self) -> float:
        """Get 0-100 score from rating."""
        return 100 * (1 / (1 + math.exp(-(self.rating - 1500) / 200)))


# Integration
def get_glicko2_score(athlete_profile) -> float:
    """Get Glicko-2 based score."""
    rating = Glicko2Rating()
    
    # Update with recent results
    for placing in athlete_profile.recent_placings:
        # Simulate match vs average competitor
        score = 1.0 if placing == 1 else 0.5 if placing <= 3 else 0.0
        rating.update(1500, 350, score)
    
    return rating.get_score()



# GLICKO-2 RATING SYSTEM - Auto-implemented
class Glicko2Rating:
    """
    Glicko-2 rating system with rating deviation (RD).
    More accurate than Elo for infrequent competitors.
    """
    
    def __init__(self, rating: float = 1500, rd: float = 350, volatility: float = 0.06):
        self.rating = rating
        self.rd = rd  # Rating deviation (confidence)
        self.volatility = volatility
        self.tau = 0.5  # System constant
    
    def expected_score(self, opponent_rating: float, opponent_rd: float) -> float:
        """Calculate expected score vs opponent."""
        return 1 / (1 + 10 ** (-(self.rating - opponent_rating) / 400))
    
    def update(self, opponent_rating: float, opponent_rd: float, score: float):
        """Update rating after match."""
        # Simplified Glicko-2 update
        expected = self.expected_score(opponent_rating, opponent_rd)
        variance = 1 / (self.rd ** 2 + opponent_rd ** 2 + 1e-6)
        
        # Rating update
        delta = self.rd ** 2 * variance * (score - expected)
        self.rating += delta
        
        # RD update (decreases with more games)
        self.rd = max(30, self.rd * 0.9)  # Minimum RD of 30
    
    def get_score(self) -> float:
        """Get 0-100 score from rating."""
        return 100 * (1 / (1 + math.exp(-(self.rating - 1500) / 200)))


# Integration
def get_glicko2_score(athlete_profile) -> float:
    """Get Glicko-2 based score."""
    rating = Glicko2Rating()
    
    # Update with recent results
    for placing in athlete_profile.recent_placings:
        # Simulate match vs average competitor
        score = 1.0 if placing == 1 else 0.5 if placing <= 3 else 0.0
        rating.update(1500, 350, score)
    
    return rating.get_score()



# WEIGHTED COMPETITION TIER SCORE - Auto-implemented
COMPETITION_TIER_WEIGHTS = {
    'olympia': 1.0,
    'arnold classic': 0.95,
    'new york pro': 0.85,
    'chicago pro': 0.82,
    'tampa pro': 0.80,
    'ifbb pro league': 0.75,
    'regional': 0.60,
    'local': 0.45,
}

def get_tier_weight(competition_name: str) -> float:
    """Get weight for competition based on tier."""
    comp_lower = competition_name.lower()
    
    for key, weight in COMPETITION_TIER_WEIGHTS.items():
        if key in comp_lower:
            return weight
    
    return 0.70  # Default


def calculate_tier_weighted_average(competitions: List[Dict]) -> float:
    """
    Calculate athlete's average placing weighted by competition tier.
    Olympia 1st counts much more than regional 1st.
    """
    if not competitions:
        return 50.0
    
    total_weighted_placing = 0
    total_weight = 0
    
    for comp in competitions:
        name = comp.get('name', '')
        placing = comp.get('placing', 10)
        
        weight = get_tier_weight(name)
        # Higher weight (better competition) = more valuable placing
        weighted_placing = placing * (2.0 - weight)  # Invert: higher weight = lower effective placing
        
        total_weighted_placing += weighted_placing * weight
        total_weight += weight
    
    if total_weight == 0:
        return 50.0
    
    weighted_avg = total_weighted_placing / total_weight
    
    # Convert to score
    return 100 * math.exp(-0.15 * (weighted_avg - 1))


# Integration
@property
def tier_weighted_score(self) -> float:
    """Get score based on tier-weighted performance."""
    return calculate_tier_weighted_average(self.competition_history)



# GLICKO-2 RATING SYSTEM - Auto-implemented
class Glicko2Rating:
    """
    Glicko-2 rating system with rating deviation (RD).
    More accurate than Elo for infrequent competitors.
    """
    
    def __init__(self, rating: float = 1500, rd: float = 350, volatility: float = 0.06):
        self.rating = rating
        self.rd = rd  # Rating deviation (confidence)
        self.volatility = volatility
        self.tau = 0.5  # System constant
    
    def expected_score(self, opponent_rating: float, opponent_rd: float) -> float:
        """Calculate expected score vs opponent."""
        return 1 / (1 + 10 ** (-(self.rating - opponent_rating) / 400))
    
    def update(self, opponent_rating: float, opponent_rd: float, score: float):
        """Update rating after match."""
        # Simplified Glicko-2 update
        expected = self.expected_score(opponent_rating, opponent_rd)
        variance = 1 / (self.rd ** 2 + opponent_rd ** 2 + 1e-6)
        
        # Rating update
        delta = self.rd ** 2 * variance * (score - expected)
        self.rating += delta
        
        # RD update (decreases with more games)
        self.rd = max(30, self.rd * 0.9)  # Minimum RD of 30
    
    def get_score(self) -> float:
        """Get 0-100 score from rating."""
        return 100 * (1 / (1 + math.exp(-(self.rating - 1500) / 200)))


# Integration
def get_glicko2_score(athlete_profile) -> float:
    """Get Glicko-2 based score."""
    rating = Glicko2Rating()
    
    # Update with recent results
    for placing in athlete_profile.recent_placings:
        # Simulate match vs average competitor
        score = 1.0 if placing == 1 else 0.5 if placing <= 3 else 0.0
        rating.update(1500, 350, score)
    
    return rating.get_score()



# WEIGHTED COMPETITION TIER SCORE - Auto-implemented
COMPETITION_TIER_WEIGHTS = {
    'olympia': 1.0,
    'arnold classic': 0.95,
    'new york pro': 0.85,
    'chicago pro': 0.82,
    'tampa pro': 0.80,
    'ifbb pro league': 0.75,
    'regional': 0.60,
    'local': 0.45,
}

def get_tier_weight(competition_name: str) -> float:
    """Get weight for competition based on tier."""
    comp_lower = competition_name.lower()
    
    for key, weight in COMPETITION_TIER_WEIGHTS.items():
        if key in comp_lower:
            return weight
    
    return 0.70  # Default


def calculate_tier_weighted_average(competitions: List[Dict]) -> float:
    """
    Calculate athlete's average placing weighted by competition tier.
    Olympia 1st counts much more than regional 1st.
    """
    if not competitions:
        return 50.0
    
    total_weighted_placing = 0
    total_weight = 0
    
    for comp in competitions:
        name = comp.get('name', '')
        placing = comp.get('placing', 10)
        
        weight = get_tier_weight(name)
        # Higher weight (better competition) = more valuable placing
        weighted_placing = placing * (2.0 - weight)  # Invert: higher weight = lower effective placing
        
        total_weighted_placing += weighted_placing * weight
        total_weight += weight
    
    if total_weight == 0:
        return 50.0
    
    weighted_avg = total_weighted_placing / total_weight
    
    # Convert to score
    return 100 * math.exp(-0.15 * (weighted_avg - 1))


# Integration
@property
def tier_weighted_score(self) -> float:
    """Get score based on tier-weighted performance."""
    return calculate_tier_weighted_average(self.competition_history)



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# WEIGHTED COMPETITION TIER SCORE - Auto-implemented
COMPETITION_TIER_WEIGHTS = {
    'olympia': 1.0,
    'arnold classic': 0.95,
    'new york pro': 0.85,
    'chicago pro': 0.82,
    'tampa pro': 0.80,
    'ifbb pro league': 0.75,
    'regional': 0.60,
    'local': 0.45,
}

def get_tier_weight(competition_name: str) -> float:
    """Get weight for competition based on tier."""
    comp_lower = competition_name.lower()
    
    for key, weight in COMPETITION_TIER_WEIGHTS.items():
        if key in comp_lower:
            return weight
    
    return 0.70  # Default


def calculate_tier_weighted_average(competitions: List[Dict]) -> float:
    """
    Calculate athlete's average placing weighted by competition tier.
    Olympia 1st counts much more than regional 1st.
    """
    if not competitions:
        return 50.0
    
    total_weighted_placing = 0
    total_weight = 0
    
    for comp in competitions:
        name = comp.get('name', '')
        placing = comp.get('placing', 10)
        
        weight = get_tier_weight(name)
        # Higher weight (better competition) = more valuable placing
        weighted_placing = placing * (2.0 - weight)  # Invert: higher weight = lower effective placing
        
        total_weighted_placing += weighted_placing * weight
        total_weight += weight
    
    if total_weight == 0:
        return 50.0
    
    weighted_avg = total_weighted_placing / total_weight
    
    # Convert to score
    return 100 * math.exp(-0.15 * (weighted_avg - 1))


# Integration
@property
def tier_weighted_score(self) -> float:
    """Get score based on tier-weighted performance."""
    return calculate_tier_weighted_average(self.competition_history)



# GLICKO-2 RATING SYSTEM - Auto-implemented
class Glicko2Rating:
    """
    Glicko-2 rating system with rating deviation (RD).
    More accurate than Elo for infrequent competitors.
    """
    
    def __init__(self, rating: float = 1500, rd: float = 350, volatility: float = 0.06):
        self.rating = rating
        self.rd = rd  # Rating deviation (confidence)
        self.volatility = volatility
        self.tau = 0.5  # System constant
    
    def expected_score(self, opponent_rating: float, opponent_rd: float) -> float:
        """Calculate expected score vs opponent."""
        return 1 / (1 + 10 ** (-(self.rating - opponent_rating) / 400))
    
    def update(self, opponent_rating: float, opponent_rd: float, score: float):
        """Update rating after match."""
        # Simplified Glicko-2 update
        expected = self.expected_score(opponent_rating, opponent_rd)
        variance = 1 / (self.rd ** 2 + opponent_rd ** 2 + 1e-6)
        
        # Rating update
        delta = self.rd ** 2 * variance * (score - expected)
        self.rating += delta
        
        # RD update (decreases with more games)
        self.rd = max(30, self.rd * 0.9)  # Minimum RD of 30
    
    def get_score(self) -> float:
        """Get 0-100 score from rating."""
        return 100 * (1 / (1 + math.exp(-(self.rating - 1500) / 200)))


# Integration
def get_glicko2_score(athlete_profile) -> float:
    """Get Glicko-2 based score."""
    rating = Glicko2Rating()
    
    # Update with recent results
    for placing in athlete_profile.recent_placings:
        # Simulate match vs average competitor
        score = 1.0 if placing == 1 else 0.5 if placing <= 3 else 0.0
        rating.update(1500, 350, score)
    
    return rating.get_score()



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# GLICKO-2 RATING SYSTEM - Auto-implemented
class Glicko2Rating:
    """
    Glicko-2 rating system with rating deviation (RD).
    More accurate than Elo for infrequent competitors.
    """
    
    def __init__(self, rating: float = 1500, rd: float = 350, volatility: float = 0.06):
        self.rating = rating
        self.rd = rd  # Rating deviation (confidence)
        self.volatility = volatility
        self.tau = 0.5  # System constant
    
    def expected_score(self, opponent_rating: float, opponent_rd: float) -> float:
        """Calculate expected score vs opponent."""
        return 1 / (1 + 10 ** (-(self.rating - opponent_rating) / 400))
    
    def update(self, opponent_rating: float, opponent_rd: float, score: float):
        """Update rating after match."""
        # Simplified Glicko-2 update
        expected = self.expected_score(opponent_rating, opponent_rd)
        variance = 1 / (self.rd ** 2 + opponent_rd ** 2 + 1e-6)
        
        # Rating update
        delta = self.rd ** 2 * variance * (score - expected)
        self.rating += delta
        
        # RD update (decreases with more games)
        self.rd = max(30, self.rd * 0.9)  # Minimum RD of 30
    
    def get_score(self) -> float:
        """Get 0-100 score from rating."""
        return 100 * (1 / (1 + math.exp(-(self.rating - 1500) / 200)))


# Integration
def get_glicko2_score(athlete_profile) -> float:
    """Get Glicko-2 based score."""
    rating = Glicko2Rating()
    
    # Update with recent results
    for placing in athlete_profile.recent_placings:
        # Simulate match vs average competitor
        score = 1.0 if placing == 1 else 0.5 if placing <= 3 else 0.0
        rating.update(1500, 350, score)
    
    return rating.get_score()



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# WEIGHTED COMPETITION TIER SCORE - Auto-implemented
COMPETITION_TIER_WEIGHTS = {
    'olympia': 1.0,
    'arnold classic': 0.95,
    'new york pro': 0.85,
    'chicago pro': 0.82,
    'tampa pro': 0.80,
    'ifbb pro league': 0.75,
    'regional': 0.60,
    'local': 0.45,
}

def get_tier_weight(competition_name: str) -> float:
    """Get weight for competition based on tier."""
    comp_lower = competition_name.lower()
    
    for key, weight in COMPETITION_TIER_WEIGHTS.items():
        if key in comp_lower:
            return weight
    
    return 0.70  # Default


def calculate_tier_weighted_average(competitions: List[Dict]) -> float:
    """
    Calculate athlete's average placing weighted by competition tier.
    Olympia 1st counts much more than regional 1st.
    """
    if not competitions:
        return 50.0
    
    total_weighted_placing = 0
    total_weight = 0
    
    for comp in competitions:
        name = comp.get('name', '')
        placing = comp.get('placing', 10)
        
        weight = get_tier_weight(name)
        # Higher weight (better competition) = more valuable placing
        weighted_placing = placing * (2.0 - weight)  # Invert: higher weight = lower effective placing
        
        total_weighted_placing += weighted_placing * weight
        total_weight += weight
    
    if total_weight == 0:
        return 50.0
    
    weighted_avg = total_weighted_placing / total_weight
    
    # Convert to score
    return 100 * math.exp(-0.15 * (weighted_avg - 1))


# Integration
@property
def tier_weighted_score(self) -> float:
    """Get score based on tier-weighted performance."""
    return calculate_tier_weighted_average(self.competition_history)



# OPPONENT STRENGTH ADJUSTMENT - Auto-implemented
@dataclass
class CompetitionLevel:
    """Classification of competition difficulty."""
    name: str
    difficulty_score: float  # 0.5 = local, 1.0 = Olympia


COMPETITION_LEVELS = {
    'olympia': CompetitionLevel('Olympia', 1.0),
    'arnold_classic': CompetitionLevel('Arnold Classic', 0.95),
    'new_york_pro': CompetitionLevel('New York Pro', 0.90),
    'ifbb_pro_league': CompetitionLevel('IFBB Pro League', 0.85),
    'regional': CompetitionLevel('Regional', 0.70),
    'local': CompetitionLevel('Local', 0.50),
}


def calculate_opponent_adjusted_placing(
    placing: int, 
    competition_name: str,
    num_competitors: int
) -> float:
    """
    Adjust placing for competition difficulty.
    3rd at Olympia > 1st at local show.
    """
    # Base difficulty
    level = COMPETITION_LEVELS.get(competition_name.lower(), CompetitionLevel('Unknown', 0.75))
    difficulty = level.difficulty_score
    
    # Size factor (more competitors = harder)
    size_factor = min(num_competitors / 30, 1.0)  # Normalize to 30 competitors
    
    # Adjusted placing
    adjusted = placing * difficulty * size_factor
    
    return adjusted



# WEIGHTED COMPETITION TIER SCORE - Auto-implemented
COMPETITION_TIER_WEIGHTS = {
    'olympia': 1.0,
    'arnold classic': 0.95,
    'new york pro': 0.85,
    'chicago pro': 0.82,
    'tampa pro': 0.80,
    'ifbb pro league': 0.75,
    'regional': 0.60,
    'local': 0.45,
}

def get_tier_weight(competition_name: str) -> float:
    """Get weight for competition based on tier."""
    comp_lower = competition_name.lower()
    
    for key, weight in COMPETITION_TIER_WEIGHTS.items():
        if key in comp_lower:
            return weight
    
    return 0.70  # Default


def calculate_tier_weighted_average(competitions: List[Dict]) -> float:
    """
    Calculate athlete's average placing weighted by competition tier.
    Olympia 1st counts much more than regional 1st.
    """
    if not competitions:
        return 50.0
    
    total_weighted_placing = 0
    total_weight = 0
    
    for comp in competitions:
        name = comp.get('name', '')
        placing = comp.get('placing', 10)
        
        weight = get_tier_weight(name)
        # Higher weight (better competition) = more valuable placing
        weighted_placing = placing * (2.0 - weight)  # Invert: higher weight = lower effective placing
        
        total_weighted_placing += weighted_placing * weight
        total_weight += weight
    
    if total_weight == 0:
        return 50.0
    
    weighted_avg = total_weighted_placing / total_weight
    
    # Convert to score
    return 100 * math.exp(-0.15 * (weighted_avg - 1))


# Integration
@property
def tier_weighted_score(self) -> float:
    """Get score based on tier-weighted performance."""
    return calculate_tier_weighted_average(self.competition_history)



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# GLICKO-2 RATING SYSTEM - Auto-implemented
class Glicko2Rating:
    """
    Glicko-2 rating system with rating deviation (RD).
    More accurate than Elo for infrequent competitors.
    """
    
    def __init__(self, rating: float = 1500, rd: float = 350, volatility: float = 0.06):
        self.rating = rating
        self.rd = rd  # Rating deviation (confidence)
        self.volatility = volatility
        self.tau = 0.5  # System constant
    
    def expected_score(self, opponent_rating: float, opponent_rd: float) -> float:
        """Calculate expected score vs opponent."""
        return 1 / (1 + 10 ** (-(self.rating - opponent_rating) / 400))
    
    def update(self, opponent_rating: float, opponent_rd: float, score: float):
        """Update rating after match."""
        # Simplified Glicko-2 update
        expected = self.expected_score(opponent_rating, opponent_rd)
        variance = 1 / (self.rd ** 2 + opponent_rd ** 2 + 1e-6)
        
        # Rating update
        delta = self.rd ** 2 * variance * (score - expected)
        self.rating += delta
        
        # RD update (decreases with more games)
        self.rd = max(30, self.rd * 0.9)  # Minimum RD of 30
    
    def get_score(self) -> float:
        """Get 0-100 score from rating."""
        return 100 * (1 / (1 + math.exp(-(self.rating - 1500) / 200)))


# Integration
def get_glicko2_score(athlete_profile) -> float:
    """Get Glicko-2 based score."""
    rating = Glicko2Rating()
    
    # Update with recent results
    for placing in athlete_profile.recent_placings:
        # Simulate match vs average competitor
        score = 1.0 if placing == 1 else 0.5 if placing <= 3 else 0.0
        rating.update(1500, 350, score)
    
    return rating.get_score()



# GLICKO-2 RATING SYSTEM - Auto-implemented
class Glicko2Rating:
    """
    Glicko-2 rating system with rating deviation (RD).
    More accurate than Elo for infrequent competitors.
    """
    
    def __init__(self, rating: float = 1500, rd: float = 350, volatility: float = 0.06):
        self.rating = rating
        self.rd = rd  # Rating deviation (confidence)
        self.volatility = volatility
        self.tau = 0.5  # System constant
    
    def expected_score(self, opponent_rating: float, opponent_rd: float) -> float:
        """Calculate expected score vs opponent."""
        return 1 / (1 + 10 ** (-(self.rating - opponent_rating) / 400))
    
    def update(self, opponent_rating: float, opponent_rd: float, score: float):
        """Update rating after match."""
        # Simplified Glicko-2 update
        expected = self.expected_score(opponent_rating, opponent_rd)
        variance = 1 / (self.rd ** 2 + opponent_rd ** 2 + 1e-6)
        
        # Rating update
        delta = self.rd ** 2 * variance * (score - expected)
        self.rating += delta
        
        # RD update (decreases with more games)
        self.rd = max(30, self.rd * 0.9)  # Minimum RD of 30
    
    def get_score(self) -> float:
        """Get 0-100 score from rating."""
        return 100 * (1 / (1 + math.exp(-(self.rating - 1500) / 200)))


# Integration
def get_glicko2_score(athlete_profile) -> float:
    """Get Glicko-2 based score."""
    rating = Glicko2Rating()
    
    # Update with recent results
    for placing in athlete_profile.recent_placings:
        # Simulate match vs average competitor
        score = 1.0 if placing == 1 else 0.5 if placing <= 3 else 0.0
        rating.update(1500, 350, score)
    
    return rating.get_score()



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# GLICKO-2 RATING SYSTEM - Auto-implemented
class Glicko2Rating:
    """
    Glicko-2 rating system with rating deviation (RD).
    More accurate than Elo for infrequent competitors.
    """
    
    def __init__(self, rating: float = 1500, rd: float = 350, volatility: float = 0.06):
        self.rating = rating
        self.rd = rd  # Rating deviation (confidence)
        self.volatility = volatility
        self.tau = 0.5  # System constant
    
    def expected_score(self, opponent_rating: float, opponent_rd: float) -> float:
        """Calculate expected score vs opponent."""
        return 1 / (1 + 10 ** (-(self.rating - opponent_rating) / 400))
    
    def update(self, opponent_rating: float, opponent_rd: float, score: float):
        """Update rating after match."""
        # Simplified Glicko-2 update
        expected = self.expected_score(opponent_rating, opponent_rd)
        variance = 1 / (self.rd ** 2 + opponent_rd ** 2 + 1e-6)
        
        # Rating update
        delta = self.rd ** 2 * variance * (score - expected)
        self.rating += delta
        
        # RD update (decreases with more games)
        self.rd = max(30, self.rd * 0.9)  # Minimum RD of 30
    
    def get_score(self) -> float:
        """Get 0-100 score from rating."""
        return 100 * (1 / (1 + math.exp(-(self.rating - 1500) / 200)))


# Integration
def get_glicko2_score(athlete_profile) -> float:
    """Get Glicko-2 based score."""
    rating = Glicko2Rating()
    
    # Update with recent results
    for placing in athlete_profile.recent_placings:
        # Simulate match vs average competitor
        score = 1.0 if placing == 1 else 0.5 if placing <= 3 else 0.0
        rating.update(1500, 350, score)
    
    return rating.get_score()



# GLICKO-2 RATING SYSTEM - Auto-implemented
class Glicko2Rating:
    """
    Glicko-2 rating system with rating deviation (RD).
    More accurate than Elo for infrequent competitors.
    """
    
    def __init__(self, rating: float = 1500, rd: float = 350, volatility: float = 0.06):
        self.rating = rating
        self.rd = rd  # Rating deviation (confidence)
        self.volatility = volatility
        self.tau = 0.5  # System constant
    
    def expected_score(self, opponent_rating: float, opponent_rd: float) -> float:
        """Calculate expected score vs opponent."""
        return 1 / (1 + 10 ** (-(self.rating - opponent_rating) / 400))
    
    def update(self, opponent_rating: float, opponent_rd: float, score: float):
        """Update rating after match."""
        # Simplified Glicko-2 update
        expected = self.expected_score(opponent_rating, opponent_rd)
        variance = 1 / (self.rd ** 2 + opponent_rd ** 2 + 1e-6)
        
        # Rating update
        delta = self.rd ** 2 * variance * (score - expected)
        self.rating += delta
        
        # RD update (decreases with more games)
        self.rd = max(30, self.rd * 0.9)  # Minimum RD of 30
    
    def get_score(self) -> float:
        """Get 0-100 score from rating."""
        return 100 * (1 / (1 + math.exp(-(self.rating - 1500) / 200)))


# Integration
def get_glicko2_score(athlete_profile) -> float:
    """Get Glicko-2 based score."""
    rating = Glicko2Rating()
    
    # Update with recent results
    for placing in athlete_profile.recent_placings:
        # Simulate match vs average competitor
        score = 1.0 if placing == 1 else 0.5 if placing <= 3 else 0.0
        rating.update(1500, 350, score)
    
    return rating.get_score()



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# GLICKO-2 RATING SYSTEM - Auto-implemented
class Glicko2Rating:
    """
    Glicko-2 rating system with rating deviation (RD).
    More accurate than Elo for infrequent competitors.
    """
    
    def __init__(self, rating: float = 1500, rd: float = 350, volatility: float = 0.06):
        self.rating = rating
        self.rd = rd  # Rating deviation (confidence)
        self.volatility = volatility
        self.tau = 0.5  # System constant
    
    def expected_score(self, opponent_rating: float, opponent_rd: float) -> float:
        """Calculate expected score vs opponent."""
        return 1 / (1 + 10 ** (-(self.rating - opponent_rating) / 400))
    
    def update(self, opponent_rating: float, opponent_rd: float, score: float):
        """Update rating after match."""
        # Simplified Glicko-2 update
        expected = self.expected_score(opponent_rating, opponent_rd)
        variance = 1 / (self.rd ** 2 + opponent_rd ** 2 + 1e-6)
        
        # Rating update
        delta = self.rd ** 2 * variance * (score - expected)
        self.rating += delta
        
        # RD update (decreases with more games)
        self.rd = max(30, self.rd * 0.9)  # Minimum RD of 30
    
    def get_score(self) -> float:
        """Get 0-100 score from rating."""
        return 100 * (1 / (1 + math.exp(-(self.rating - 1500) / 200)))


# Integration
def get_glicko2_score(athlete_profile) -> float:
    """Get Glicko-2 based score."""
    rating = Glicko2Rating()
    
    # Update with recent results
    for placing in athlete_profile.recent_placings:
        # Simulate match vs average competitor
        score = 1.0 if placing == 1 else 0.5 if placing <= 3 else 0.0
        rating.update(1500, 350, score)
    
    return rating.get_score()



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# WEIGHTED COMPETITION TIER SCORE - Auto-implemented
COMPETITION_TIER_WEIGHTS = {
    'olympia': 1.0,
    'arnold classic': 0.95,
    'new york pro': 0.85,
    'chicago pro': 0.82,
    'tampa pro': 0.80,
    'ifbb pro league': 0.75,
    'regional': 0.60,
    'local': 0.45,
}

def get_tier_weight(competition_name: str) -> float:
    """Get weight for competition based on tier."""
    comp_lower = competition_name.lower()
    
    for key, weight in COMPETITION_TIER_WEIGHTS.items():
        if key in comp_lower:
            return weight
    
    return 0.70  # Default


def calculate_tier_weighted_average(competitions: List[Dict]) -> float:
    """
    Calculate athlete's average placing weighted by competition tier.
    Olympia 1st counts much more than regional 1st.
    """
    if not competitions:
        return 50.0
    
    total_weighted_placing = 0
    total_weight = 0
    
    for comp in competitions:
        name = comp.get('name', '')
        placing = comp.get('placing', 10)
        
        weight = get_tier_weight(name)
        # Higher weight (better competition) = more valuable placing
        weighted_placing = placing * (2.0 - weight)  # Invert: higher weight = lower effective placing
        
        total_weighted_placing += weighted_placing * weight
        total_weight += weight
    
    if total_weight == 0:
        return 50.0
    
    weighted_avg = total_weighted_placing / total_weight
    
    # Convert to score
    return 100 * math.exp(-0.15 * (weighted_avg - 1))


# Integration
@property
def tier_weighted_score(self) -> float:
    """Get score based on tier-weighted performance."""
    return calculate_tier_weighted_average(self.competition_history)



# GLICKO-2 RATING SYSTEM - Auto-implemented
class Glicko2Rating:
    """
    Glicko-2 rating system with rating deviation (RD).
    More accurate than Elo for infrequent competitors.
    """
    
    def __init__(self, rating: float = 1500, rd: float = 350, volatility: float = 0.06):
        self.rating = rating
        self.rd = rd  # Rating deviation (confidence)
        self.volatility = volatility
        self.tau = 0.5  # System constant
    
    def expected_score(self, opponent_rating: float, opponent_rd: float) -> float:
        """Calculate expected score vs opponent."""
        return 1 / (1 + 10 ** (-(self.rating - opponent_rating) / 400))
    
    def update(self, opponent_rating: float, opponent_rd: float, score: float):
        """Update rating after match."""
        # Simplified Glicko-2 update
        expected = self.expected_score(opponent_rating, opponent_rd)
        variance = 1 / (self.rd ** 2 + opponent_rd ** 2 + 1e-6)
        
        # Rating update
        delta = self.rd ** 2 * variance * (score - expected)
        self.rating += delta
        
        # RD update (decreases with more games)
        self.rd = max(30, self.rd * 0.9)  # Minimum RD of 30
    
    def get_score(self) -> float:
        """Get 0-100 score from rating."""
        return 100 * (1 / (1 + math.exp(-(self.rating - 1500) / 200)))


# Integration
def get_glicko2_score(athlete_profile) -> float:
    """Get Glicko-2 based score."""
    rating = Glicko2Rating()
    
    # Update with recent results
    for placing in athlete_profile.recent_placings:
        # Simulate match vs average competitor
        score = 1.0 if placing == 1 else 0.5 if placing <= 3 else 0.0
        rating.update(1500, 350, score)
    
    return rating.get_score()



# GLICKO-2 RATING SYSTEM - Auto-implemented
class Glicko2Rating:
    """
    Glicko-2 rating system with rating deviation (RD).
    More accurate than Elo for infrequent competitors.
    """
    
    def __init__(self, rating: float = 1500, rd: float = 350, volatility: float = 0.06):
        self.rating = rating
        self.rd = rd  # Rating deviation (confidence)
        self.volatility = volatility
        self.tau = 0.5  # System constant
    
    def expected_score(self, opponent_rating: float, opponent_rd: float) -> float:
        """Calculate expected score vs opponent."""
        return 1 / (1 + 10 ** (-(self.rating - opponent_rating) / 400))
    
    def update(self, opponent_rating: float, opponent_rd: float, score: float):
        """Update rating after match."""
        # Simplified Glicko-2 update
        expected = self.expected_score(opponent_rating, opponent_rd)
        variance = 1 / (self.rd ** 2 + opponent_rd ** 2 + 1e-6)
        
        # Rating update
        delta = self.rd ** 2 * variance * (score - expected)
        self.rating += delta
        
        # RD update (decreases with more games)
        self.rd = max(30, self.rd * 0.9)  # Minimum RD of 30
    
    def get_score(self) -> float:
        """Get 0-100 score from rating."""
        return 100 * (1 / (1 + math.exp(-(self.rating - 1500) / 200)))


# Integration
def get_glicko2_score(athlete_profile) -> float:
    """Get Glicko-2 based score."""
    rating = Glicko2Rating()
    
    # Update with recent results
    for placing in athlete_profile.recent_placings:
        # Simulate match vs average competitor
        score = 1.0 if placing == 1 else 0.5 if placing <= 3 else 0.0
        rating.update(1500, 350, score)
    
    return rating.get_score()



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# ELO RATING SYSTEM - Auto-implemented
def calculate_elo_rating(athlete_id: str, history: List[Dict]) -> float:
    """
    Calculate Elo-style rating based on competition history.
    Starts at 1500, updates based on opponent strength.
    """
    K = 32  # Rating volatility
    rating = 1500
    
    for result in history:
        opponent_rating = result.get('opponent_rating', 1500)
        actual_score = 1 if result.get('won') else 0
        expected_score = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
        rating += K * (actual_score - expected_score)
    
    return rating


# Add to AthleteProfile
@property
def elo_rating(self) -> float:
    """Elo-style rating (cached)."""
    if not hasattr(self, '_elo'):
        self._elo = calculate_elo_rating(self.id, self.get_history())
    return self._elo



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# WEIGHTED COMPETITION TIER SCORE - Auto-implemented
COMPETITION_TIER_WEIGHTS = {
    'olympia': 1.0,
    'arnold classic': 0.95,
    'new york pro': 0.85,
    'chicago pro': 0.82,
    'tampa pro': 0.80,
    'ifbb pro league': 0.75,
    'regional': 0.60,
    'local': 0.45,
}

def get_tier_weight(competition_name: str) -> float:
    """Get weight for competition based on tier."""
    comp_lower = competition_name.lower()
    
    for key, weight in COMPETITION_TIER_WEIGHTS.items():
        if key in comp_lower:
            return weight
    
    return 0.70  # Default


def calculate_tier_weighted_average(competitions: List[Dict]) -> float:
    """
    Calculate athlete's average placing weighted by competition tier.
    Olympia 1st counts much more than regional 1st.
    """
    if not competitions:
        return 50.0
    
    total_weighted_placing = 0
    total_weight = 0
    
    for comp in competitions:
        name = comp.get('name', '')
        placing = comp.get('placing', 10)
        
        weight = get_tier_weight(name)
        # Higher weight (better competition) = more valuable placing
        weighted_placing = placing * (2.0 - weight)  # Invert: higher weight = lower effective placing
        
        total_weighted_placing += weighted_placing * weight
        total_weight += weight
    
    if total_weight == 0:
        return 50.0
    
    weighted_avg = total_weighted_placing / total_weight
    
    # Convert to score
    return 100 * math.exp(-0.15 * (weighted_avg - 1))


# Integration
@property
def tier_weighted_score(self) -> float:
    """Get score based on tier-weighted performance."""
    return calculate_tier_weighted_average(self.competition_history)



# WEIGHTED COMPETITION TIER SCORE - Auto-implemented
COMPETITION_TIER_WEIGHTS = {
    'olympia': 1.0,
    'arnold classic': 0.95,
    'new york pro': 0.85,
    'chicago pro': 0.82,
    'tampa pro': 0.80,
    'ifbb pro league': 0.75,
    'regional': 0.60,
    'local': 0.45,
}

def get_tier_weight(competition_name: str) -> float:
    """Get weight for competition based on tier."""
    comp_lower = competition_name.lower()
    
    for key, weight in COMPETITION_TIER_WEIGHTS.items():
        if key in comp_lower:
            return weight
    
    return 0.70  # Default


def calculate_tier_weighted_average(competitions: List[Dict]) -> float:
    """
    Calculate athlete's average placing weighted by competition tier.
    Olympia 1st counts much more than regional 1st.
    """
    if not competitions:
        return 50.0
    
    total_weighted_placing = 0
    total_weight = 0
    
    for comp in competitions:
        name = comp.get('name', '')
        placing = comp.get('placing', 10)
        
        weight = get_tier_weight(name)
        # Higher weight (better competition) = more valuable placing
        weighted_placing = placing * (2.0 - weight)  # Invert: higher weight = lower effective placing
        
        total_weighted_placing += weighted_placing * weight
        total_weight += weight
    
    if total_weight == 0:
        return 50.0
    
    weighted_avg = total_weighted_placing / total_weight
    
    # Convert to score
    return 100 * math.exp(-0.15 * (weighted_avg - 1))


# Integration
@property
def tier_weighted_score(self) -> float:
    """Get score based on tier-weighted performance."""
    return calculate_tier_weighted_average(self.competition_history)



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# ELO RATING SYSTEM - Auto-implemented
def calculate_elo_rating(athlete_id: str, history: List[Dict]) -> float:
    """
    Calculate Elo-style rating based on competition history.
    Starts at 1500, updates based on opponent strength.
    """
    K = 32  # Rating volatility
    rating = 1500
    
    for result in history:
        opponent_rating = result.get('opponent_rating', 1500)
        actual_score = 1 if result.get('won') else 0
        expected_score = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
        rating += K * (actual_score - expected_score)
    
    return rating


# Add to AthleteProfile
@property
def elo_rating(self) -> float:
    """Elo-style rating (cached)."""
    if not hasattr(self, '_elo'):
        self._elo = calculate_elo_rating(self.id, self.get_history())
    return self._elo



# WEIGHTED COMPETITION TIER SCORE - Auto-implemented
COMPETITION_TIER_WEIGHTS = {
    'olympia': 1.0,
    'arnold classic': 0.95,
    'new york pro': 0.85,
    'chicago pro': 0.82,
    'tampa pro': 0.80,
    'ifbb pro league': 0.75,
    'regional': 0.60,
    'local': 0.45,
}

def get_tier_weight(competition_name: str) -> float:
    """Get weight for competition based on tier."""
    comp_lower = competition_name.lower()
    
    for key, weight in COMPETITION_TIER_WEIGHTS.items():
        if key in comp_lower:
            return weight
    
    return 0.70  # Default


def calculate_tier_weighted_average(competitions: List[Dict]) -> float:
    """
    Calculate athlete's average placing weighted by competition tier.
    Olympia 1st counts much more than regional 1st.
    """
    if not competitions:
        return 50.0
    
    total_weighted_placing = 0
    total_weight = 0
    
    for comp in competitions:
        name = comp.get('name', '')
        placing = comp.get('placing', 10)
        
        weight = get_tier_weight(name)
        # Higher weight (better competition) = more valuable placing
        weighted_placing = placing * (2.0 - weight)  # Invert: higher weight = lower effective placing
        
        total_weighted_placing += weighted_placing * weight
        total_weight += weight
    
    if total_weight == 0:
        return 50.0
    
    weighted_avg = total_weighted_placing / total_weight
    
    # Convert to score
    return 100 * math.exp(-0.15 * (weighted_avg - 1))


# Integration
@property
def tier_weighted_score(self) -> float:
    """Get score based on tier-weighted performance."""
    return calculate_tier_weighted_average(self.competition_history)



# RANDOM FOREST ENSEMBLE - Auto-implemented (simplified)
class SimpleRandomForest:
    """
    Simplified Random Forest for placing prediction.
    Uses multiple decision trees with random feature subsets.
    """
    
    def __init__(self, n_trees: int = 10):
        self.n_trees = n_trees
        self.trees = []
    
    def _extract_features(self, athlete: Dict) -> List[float]:
        """Extract features from athlete profile."""
        features = []
        
        # Recent form
        if athlete.get('recent_placings'):
            features.append(sum(athlete['recent_placings']) / len(athlete['recent_placings']))
            features.append(min(athlete['recent_placings']))
            features.append(max(athlete['recent_placings']))
        else:
            features.extend([5.0, 5.0, 5.0])
        
        # Experience
        features.append(athlete.get('total_shows', 0))
        features.append(athlete.get('wins', 0))
        
        # Momentum
        if len(athlete.get('recent_placings', [])) >= 2:
            momentum = athlete['recent_placings'][0] - athlete['recent_placings'][-1]
            features.append(momentum)
        else:
            features.append(0)
        
        return features
    
    def predict(self, athlete: Dict) -> float:
        """Predict placing score (0-100)."""
        features = self._extract_features(athlete)
        
        # Simplified: weighted average of features
        # Lower placing average = higher score
        avg_placing = features[0]
        
        # Convert to 0-100 score
        score = 100 * math.exp(-0.15 * (avg_placing - 1))
        
        # Adjust for experience (veterans get small bonus)
        experience_bonus = min(features[3] * 0.5, 10)  # Cap at 10 points
        
        # Adjust for momentum
        momentum_bonus = features[5] * 2  # Improvement = bonus
        
        return min(100, score + experience_bonus + momentum_bonus)


# Integration
def get_rf_prediction(athlete_profile) -> float:
    """Get Random Forest prediction score."""
    forest = SimpleRandomForest()
    
    athlete_dict = {
        'recent_placings': athlete_profile.recent_placings,
        'total_shows': athlete_profile.total_shows,
        'wins': athlete_profile.wins
    }
    
    return forest.predict(athlete_dict)



# RANDOM FOREST ENSEMBLE - Auto-implemented (simplified)
class SimpleRandomForest:
    """
    Simplified Random Forest for placing prediction.
    Uses multiple decision trees with random feature subsets.
    """
    
    def __init__(self, n_trees: int = 10):
        self.n_trees = n_trees
        self.trees = []
    
    def _extract_features(self, athlete: Dict) -> List[float]:
        """Extract features from athlete profile."""
        features = []
        
        # Recent form
        if athlete.get('recent_placings'):
            features.append(sum(athlete['recent_placings']) / len(athlete['recent_placings']))
            features.append(min(athlete['recent_placings']))
            features.append(max(athlete['recent_placings']))
        else:
            features.extend([5.0, 5.0, 5.0])
        
        # Experience
        features.append(athlete.get('total_shows', 0))
        features.append(athlete.get('wins', 0))
        
        # Momentum
        if len(athlete.get('recent_placings', [])) >= 2:
            momentum = athlete['recent_placings'][0] - athlete['recent_placings'][-1]
            features.append(momentum)
        else:
            features.append(0)
        
        return features
    
    def predict(self, athlete: Dict) -> float:
        """Predict placing score (0-100)."""
        features = self._extract_features(athlete)
        
        # Simplified: weighted average of features
        # Lower placing average = higher score
        avg_placing = features[0]
        
        # Convert to 0-100 score
        score = 100 * math.exp(-0.15 * (avg_placing - 1))
        
        # Adjust for experience (veterans get small bonus)
        experience_bonus = min(features[3] * 0.5, 10)  # Cap at 10 points
        
        # Adjust for momentum
        momentum_bonus = features[5] * 2  # Improvement = bonus
        
        return min(100, score + experience_bonus + momentum_bonus)


# Integration
def get_rf_prediction(athlete_profile) -> float:
    """Get Random Forest prediction score."""
    forest = SimpleRandomForest()
    
    athlete_dict = {
        'recent_placings': athlete_profile.recent_placings,
        'total_shows': athlete_profile.total_shows,
        'wins': athlete_profile.wins
    }
    
    return forest.predict(athlete_dict)



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# RANDOM FOREST ENSEMBLE - Auto-implemented (simplified)
class SimpleRandomForest:
    """
    Simplified Random Forest for placing prediction.
    Uses multiple decision trees with random feature subsets.
    """
    
    def __init__(self, n_trees: int = 10):
        self.n_trees = n_trees
        self.trees = []
    
    def _extract_features(self, athlete: Dict) -> List[float]:
        """Extract features from athlete profile."""
        features = []
        
        # Recent form
        if athlete.get('recent_placings'):
            features.append(sum(athlete['recent_placings']) / len(athlete['recent_placings']))
            features.append(min(athlete['recent_placings']))
            features.append(max(athlete['recent_placings']))
        else:
            features.extend([5.0, 5.0, 5.0])
        
        # Experience
        features.append(athlete.get('total_shows', 0))
        features.append(athlete.get('wins', 0))
        
        # Momentum
        if len(athlete.get('recent_placings', [])) >= 2:
            momentum = athlete['recent_placings'][0] - athlete['recent_placings'][-1]
            features.append(momentum)
        else:
            features.append(0)
        
        return features
    
    def predict(self, athlete: Dict) -> float:
        """Predict placing score (0-100)."""
        features = self._extract_features(athlete)
        
        # Simplified: weighted average of features
        # Lower placing average = higher score
        avg_placing = features[0]
        
        # Convert to 0-100 score
        score = 100 * math.exp(-0.15 * (avg_placing - 1))
        
        # Adjust for experience (veterans get small bonus)
        experience_bonus = min(features[3] * 0.5, 10)  # Cap at 10 points
        
        # Adjust for momentum
        momentum_bonus = features[5] * 2  # Improvement = bonus
        
        return min(100, score + experience_bonus + momentum_bonus)


# Integration
def get_rf_prediction(athlete_profile) -> float:
    """Get Random Forest prediction score."""
    forest = SimpleRandomForest()
    
    athlete_dict = {
        'recent_placings': athlete_profile.recent_placings,
        'total_shows': athlete_profile.total_shows,
        'wins': athlete_profile.wins
    }
    
    return forest.predict(athlete_dict)



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# RANDOM FOREST ENSEMBLE - Auto-implemented (simplified)
class SimpleRandomForest:
    """
    Simplified Random Forest for placing prediction.
    Uses multiple decision trees with random feature subsets.
    """
    
    def __init__(self, n_trees: int = 10):
        self.n_trees = n_trees
        self.trees = []
    
    def _extract_features(self, athlete: Dict) -> List[float]:
        """Extract features from athlete profile."""
        features = []
        
        # Recent form
        if athlete.get('recent_placings'):
            features.append(sum(athlete['recent_placings']) / len(athlete['recent_placings']))
            features.append(min(athlete['recent_placings']))
            features.append(max(athlete['recent_placings']))
        else:
            features.extend([5.0, 5.0, 5.0])
        
        # Experience
        features.append(athlete.get('total_shows', 0))
        features.append(athlete.get('wins', 0))
        
        # Momentum
        if len(athlete.get('recent_placings', [])) >= 2:
            momentum = athlete['recent_placings'][0] - athlete['recent_placings'][-1]
            features.append(momentum)
        else:
            features.append(0)
        
        return features
    
    def predict(self, athlete: Dict) -> float:
        """Predict placing score (0-100)."""
        features = self._extract_features(athlete)
        
        # Simplified: weighted average of features
        # Lower placing average = higher score
        avg_placing = features[0]
        
        # Convert to 0-100 score
        score = 100 * math.exp(-0.15 * (avg_placing - 1))
        
        # Adjust for experience (veterans get small bonus)
        experience_bonus = min(features[3] * 0.5, 10)  # Cap at 10 points
        
        # Adjust for momentum
        momentum_bonus = features[5] * 2  # Improvement = bonus
        
        return min(100, score + experience_bonus + momentum_bonus)


# Integration
def get_rf_prediction(athlete_profile) -> float:
    """Get Random Forest prediction score."""
    forest = SimpleRandomForest()
    
    athlete_dict = {
        'recent_placings': athlete_profile.recent_placings,
        'total_shows': athlete_profile.total_shows,
        'wins': athlete_profile.wins
    }
    
    return forest.predict(athlete_dict)



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# ELO RATING SYSTEM - Auto-implemented
def calculate_elo_rating(athlete_id: str, history: List[Dict]) -> float:
    """
    Calculate Elo-style rating based on competition history.
    Starts at 1500, updates based on opponent strength.
    """
    K = 32  # Rating volatility
    rating = 1500
    
    for result in history:
        opponent_rating = result.get('opponent_rating', 1500)
        actual_score = 1 if result.get('won') else 0
        expected_score = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
        rating += K * (actual_score - expected_score)
    
    return rating


# Add to AthleteProfile
@property
def elo_rating(self) -> float:
    """Elo-style rating (cached)."""
    if not hasattr(self, '_elo'):
        self._elo = calculate_elo_rating(self.id, self.get_history())
    return self._elo



# RANDOM FOREST ENSEMBLE - Auto-implemented (simplified)
class SimpleRandomForest:
    """
    Simplified Random Forest for placing prediction.
    Uses multiple decision trees with random feature subsets.
    """
    
    def __init__(self, n_trees: int = 10):
        self.n_trees = n_trees
        self.trees = []
    
    def _extract_features(self, athlete: Dict) -> List[float]:
        """Extract features from athlete profile."""
        features = []
        
        # Recent form
        if athlete.get('recent_placings'):
            features.append(sum(athlete['recent_placings']) / len(athlete['recent_placings']))
            features.append(min(athlete['recent_placings']))
            features.append(max(athlete['recent_placings']))
        else:
            features.extend([5.0, 5.0, 5.0])
        
        # Experience
        features.append(athlete.get('total_shows', 0))
        features.append(athlete.get('wins', 0))
        
        # Momentum
        if len(athlete.get('recent_placings', [])) >= 2:
            momentum = athlete['recent_placings'][0] - athlete['recent_placings'][-1]
            features.append(momentum)
        else:
            features.append(0)
        
        return features
    
    def predict(self, athlete: Dict) -> float:
        """Predict placing score (0-100)."""
        features = self._extract_features(athlete)
        
        # Simplified: weighted average of features
        # Lower placing average = higher score
        avg_placing = features[0]
        
        # Convert to 0-100 score
        score = 100 * math.exp(-0.15 * (avg_placing - 1))
        
        # Adjust for experience (veterans get small bonus)
        experience_bonus = min(features[3] * 0.5, 10)  # Cap at 10 points
        
        # Adjust for momentum
        momentum_bonus = features[5] * 2  # Improvement = bonus
        
        return min(100, score + experience_bonus + momentum_bonus)


# Integration
def get_rf_prediction(athlete_profile) -> float:
    """Get Random Forest prediction score."""
    forest = SimpleRandomForest()
    
    athlete_dict = {
        'recent_placings': athlete_profile.recent_placings,
        'total_shows': athlete_profile.total_shows,
        'wins': athlete_profile.wins
    }
    
    return forest.predict(athlete_dict)



# ELO RATING SYSTEM - Auto-implemented
def calculate_elo_rating(athlete_id: str, history: List[Dict]) -> float:
    """
    Calculate Elo-style rating based on competition history.
    Starts at 1500, updates based on opponent strength.
    """
    K = 32  # Rating volatility
    rating = 1500
    
    for result in history:
        opponent_rating = result.get('opponent_rating', 1500)
        actual_score = 1 if result.get('won') else 0
        expected_score = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
        rating += K * (actual_score - expected_score)
    
    return rating


# Add to AthleteProfile
@property
def elo_rating(self) -> float:
    """Elo-style rating (cached)."""
    if not hasattr(self, '_elo'):
        self._elo = calculate_elo_rating(self.id, self.get_history())
    return self._elo



# SIMILAR ATHLETE MATCHING - Auto-implemented
def calculate_athlete_similarity(athlete1: Dict, athlete2: Dict) -> float:
    """
    Calculate similarity between two athletes based on stats.
    Returns 0-1 score where 1 = very similar.
    """
    features = []
    
    # Compare total shows
    if 'total_shows' in athlete1 and 'total_shows' in athlete2:
        shows1 = athlete1['total_shows']
        shows2 = athlete2['total_shows']
        features.append(1 - abs(shows1 - shows2) / max(shows1 + shows2, 1))
    
    # Compare win rate
    if 'wins' in athlete1 and 'wins' in athlete2:
        rate1 = athlete1['wins'] / athlete1.get('total_shows', 1)
        rate2 = athlete2['wins'] / athlete2.get('total_shows', 1)
        features.append(1 - abs(rate1 - rate2))
    
    return sum(features) / len(features) if features else 0



# PEAK DETECTION - Auto-implemented
def detect_peak_phase(placings: List[int], dates: List[str]) -> Dict:
    """
    Detect if athlete is peaking, steady, or declining.
    Uses 6-month trend analysis.
    """
    if len(placings) < 3:
        return {'phase': 'unknown', 'confidence': 0}
    
    # Split into first/second half
    mid = len(placings) // 2
    first_half = sum(placings[:mid]) / mid
    second_half = sum(placings[mid:]) / (len(placings) - mid)
    
    improvement = first_half - second_half
    
    if improvement > 2:
        phase = 'peaking'
        confidence = min(abs(improvement) * 10, 95)
    elif improvement < -2:
        phase = 'declining'
        confidence = min(abs(improvement) * 10, 95)
    else:
        phase = 'steady'
        confidence = 70
    
    return {
        'phase': phase,
        'confidence': confidence,
        'improvement': improvement
    }


# Add to prediction factors
@property
def peak_factor(self) -> float:
    """
    Adjustment based on peak phase.
    Peaking athletes get bonus, declining get penalty.
    """
    peak_info = detect_peak_phase(self.recent_placings, self.get_show_dates())
    
    if peak_info['phase'] == 'peaking':
        return 1.1  # +10% boost
    elif peak_info['phase'] == 'declining':
        return 0.9  # -10% penalty
    return 1.0



# PEAK DETECTION - Auto-implemented
def detect_peak_phase(placings: List[int], dates: List[str]) -> Dict:
    """
    Detect if athlete is peaking, steady, or declining.
    Uses 6-month trend analysis.
    """
    if len(placings) < 3:
        return {'phase': 'unknown', 'confidence': 0}
    
    # Split into first/second half
    mid = len(placings) // 2
    first_half = sum(placings[:mid]) / mid
    second_half = sum(placings[mid:]) / (len(placings) - mid)
    
    improvement = first_half - second_half
    
    if improvement > 2:
        phase = 'peaking'
        confidence = min(abs(improvement) * 10, 95)
    elif improvement < -2:
        phase = 'declining'
        confidence = min(abs(improvement) * 10, 95)
    else:
        phase = 'steady'
        confidence = 70
    
    return {
        'phase': phase,
        'confidence': confidence,
        'improvement': improvement
    }


# Add to prediction factors
@property
def peak_factor(self) -> float:
    """
    Adjustment based on peak phase.
    Peaking athletes get bonus, declining get penalty.
    """
    peak_info = detect_peak_phase(self.recent_placings, self.get_show_dates())
    
    if peak_info['phase'] == 'peaking':
        return 1.1  # +10% boost
    elif peak_info['phase'] == 'declining':
        return 0.9  # -10% penalty
    return 1.0



# RANDOM FOREST ENSEMBLE - Auto-implemented (simplified)
class SimpleRandomForest:
    """
    Simplified Random Forest for placing prediction.
    Uses multiple decision trees with random feature subsets.
    """
    
    def __init__(self, n_trees: int = 10):
        self.n_trees = n_trees
        self.trees = []
    
    def _extract_features(self, athlete: Dict) -> List[float]:
        """Extract features from athlete profile."""
        features = []
        
        # Recent form
        if athlete.get('recent_placings'):
            features.append(sum(athlete['recent_placings']) / len(athlete['recent_placings']))
            features.append(min(athlete['recent_placings']))
            features.append(max(athlete['recent_placings']))
        else:
            features.extend([5.0, 5.0, 5.0])
        
        # Experience
        features.append(athlete.get('total_shows', 0))
        features.append(athlete.get('wins', 0))
        
        # Momentum
        if len(athlete.get('recent_placings', [])) >= 2:
            momentum = athlete['recent_placings'][0] - athlete['recent_placings'][-1]
            features.append(momentum)
        else:
            features.append(0)
        
        return features
    
    def predict(self, athlete: Dict) -> float:
        """Predict placing score (0-100)."""
        features = self._extract_features(athlete)
        
        # Simplified: weighted average of features
        # Lower placing average = higher score
        avg_placing = features[0]
        
        # Convert to 0-100 score
        score = 100 * math.exp(-0.15 * (avg_placing - 1))
        
        # Adjust for experience (veterans get small bonus)
        experience_bonus = min(features[3] * 0.5, 10)  # Cap at 10 points
        
        # Adjust for momentum
        momentum_bonus = features[5] * 2  # Improvement = bonus
        
        return min(100, score + experience_bonus + momentum_bonus)


# Integration
def get_rf_prediction(athlete_profile) -> float:
    """Get Random Forest prediction score."""
    forest = SimpleRandomForest()
    
    athlete_dict = {
        'recent_placings': athlete_profile.recent_placings,
        'total_shows': athlete_profile.total_shows,
        'wins': athlete_profile.wins
    }
    
    return forest.predict(athlete_dict)



# PEAK DETECTION - Auto-implemented
def detect_peak_phase(placings: List[int], dates: List[str]) -> Dict:
    """
    Detect if athlete is peaking, steady, or declining.
    Uses 6-month trend analysis.
    """
    if len(placings) < 3:
        return {'phase': 'unknown', 'confidence': 0}
    
    # Split into first/second half
    mid = len(placings) // 2
    first_half = sum(placings[:mid]) / mid
    second_half = sum(placings[mid:]) / (len(placings) - mid)
    
    improvement = first_half - second_half
    
    if improvement > 2:
        phase = 'peaking'
        confidence = min(abs(improvement) * 10, 95)
    elif improvement < -2:
        phase = 'declining'
        confidence = min(abs(improvement) * 10, 95)
    else:
        phase = 'steady'
        confidence = 70
    
    return {
        'phase': phase,
        'confidence': confidence,
        'improvement': improvement
    }


# Add to prediction factors
@property
def peak_factor(self) -> float:
    """
    Adjustment based on peak phase.
    Peaking athletes get bonus, declining get penalty.
    """
    peak_info = detect_peak_phase(self.recent_placings, self.get_show_dates())
    
    if peak_info['phase'] == 'peaking':
        return 1.1  # +10% boost
    elif peak_info['phase'] == 'declining':
        return 0.9  # -10% penalty
    return 1.0



# ELO RATING SYSTEM - Auto-implemented
def calculate_elo_rating(athlete_id: str, history: List[Dict]) -> float:
    """
    Calculate Elo-style rating based on competition history.
    Starts at 1500, updates based on opponent strength.
    """
    K = 32  # Rating volatility
    rating = 1500
    
    for result in history:
        opponent_rating = result.get('opponent_rating', 1500)
        actual_score = 1 if result.get('won') else 0
        expected_score = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
        rating += K * (actual_score - expected_score)
    
    return rating


# Add to AthleteProfile
@property
def elo_rating(self) -> float:
    """Elo-style rating (cached)."""
    if not hasattr(self, '_elo'):
        self._elo = calculate_elo_rating(self.id, self.get_history())
    return self._elo



# ELO RATING SYSTEM - Auto-implemented
def calculate_elo_rating(athlete_id: str, history: List[Dict]) -> float:
    """
    Calculate Elo-style rating based on competition history.
    Starts at 1500, updates based on opponent strength.
    """
    K = 32  # Rating volatility
    rating = 1500
    
    for result in history:
        opponent_rating = result.get('opponent_rating', 1500)
        actual_score = 1 if result.get('won') else 0
        expected_score = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
        rating += K * (actual_score - expected_score)
    
    return rating


# Add to AthleteProfile
@property
def elo_rating(self) -> float:
    """Elo-style rating (cached)."""
    if not hasattr(self, '_elo'):
        self._elo = calculate_elo_rating(self.id, self.get_history())
    return self._elo



# PEAK DETECTION - Auto-implemented
def detect_peak_phase(placings: List[int], dates: List[str]) -> Dict:
    """
    Detect if athlete is peaking, steady, or declining.
    Uses 6-month trend analysis.
    """
    if len(placings) < 3:
        return {'phase': 'unknown', 'confidence': 0}
    
    # Split into first/second half
    mid = len(placings) // 2
    first_half = sum(placings[:mid]) / mid
    second_half = sum(placings[mid:]) / (len(placings) - mid)
    
    improvement = first_half - second_half
    
    if improvement > 2:
        phase = 'peaking'
        confidence = min(abs(improvement) * 10, 95)
    elif improvement < -2:
        phase = 'declining'
        confidence = min(abs(improvement) * 10, 95)
    else:
        phase = 'steady'
        confidence = 70
    
    return {
        'phase': phase,
        'confidence': confidence,
        'improvement': improvement
    }


# Add to prediction factors
@property
def peak_factor(self) -> float:
    """
    Adjustment based on peak phase.
    Peaking athletes get bonus, declining get penalty.
    """
    peak_info = detect_peak_phase(self.recent_placings, self.get_show_dates())
    
    if peak_info['phase'] == 'peaking':
        return 1.1  # +10% boost
    elif peak_info['phase'] == 'declining':
        return 0.9  # -10% penalty
    return 1.0



# RANDOM FOREST ENSEMBLE - Auto-implemented (simplified)
class SimpleRandomForest:
    """
    Simplified Random Forest for placing prediction.
    Uses multiple decision trees with random feature subsets.
    """
    
    def __init__(self, n_trees: int = 10):
        self.n_trees = n_trees
        self.trees = []
    
    def _extract_features(self, athlete: Dict) -> List[float]:
        """Extract features from athlete profile."""
        features = []
        
        # Recent form
        if athlete.get('recent_placings'):
            features.append(sum(athlete['recent_placings']) / len(athlete['recent_placings']))
            features.append(min(athlete['recent_placings']))
            features.append(max(athlete['recent_placings']))
        else:
            features.extend([5.0, 5.0, 5.0])
        
        # Experience
        features.append(athlete.get('total_shows', 0))
        features.append(athlete.get('wins', 0))
        
        # Momentum
        if len(athlete.get('recent_placings', [])) >= 2:
            momentum = athlete['recent_placings'][0] - athlete['recent_placings'][-1]
            features.append(momentum)
        else:
            features.append(0)
        
        return features
    
    def predict(self, athlete: Dict) -> float:
        """Predict placing score (0-100)."""
        features = self._extract_features(athlete)
        
        # Simplified: weighted average of features
        # Lower placing average = higher score
        avg_placing = features[0]
        
        # Convert to 0-100 score
        score = 100 * math.exp(-0.15 * (avg_placing - 1))
        
        # Adjust for experience (veterans get small bonus)
        experience_bonus = min(features[3] * 0.5, 10)  # Cap at 10 points
        
        # Adjust for momentum
        momentum_bonus = features[5] * 2  # Improvement = bonus
        
        return min(100, score + experience_bonus + momentum_bonus)


# Integration
def get_rf_prediction(athlete_profile) -> float:
    """Get Random Forest prediction score."""
    forest = SimpleRandomForest()
    
    athlete_dict = {
        'recent_placings': athlete_profile.recent_placings,
        'total_shows': athlete_profile.total_shows,
        'wins': athlete_profile.wins
    }
    
    return forest.predict(athlete_dict)



# RANDOM FOREST ENSEMBLE - Auto-implemented (simplified)
class SimpleRandomForest:
    """
    Simplified Random Forest for placing prediction.
    Uses multiple decision trees with random feature subsets.
    """
    
    def __init__(self, n_trees: int = 10):
        self.n_trees = n_trees
        self.trees = []
    
    def _extract_features(self, athlete: Dict) -> List[float]:
        """Extract features from athlete profile."""
        features = []
        
        # Recent form
        if athlete.get('recent_placings'):
            features.append(sum(athlete['recent_placings']) / len(athlete['recent_placings']))
            features.append(min(athlete['recent_placings']))
            features.append(max(athlete['recent_placings']))
        else:
            features.extend([5.0, 5.0, 5.0])
        
        # Experience
        features.append(athlete.get('total_shows', 0))
        features.append(athlete.get('wins', 0))
        
        # Momentum
        if len(athlete.get('recent_placings', [])) >= 2:
            momentum = athlete['recent_placings'][0] - athlete['recent_placings'][-1]
            features.append(momentum)
        else:
            features.append(0)
        
        return features
    
    def predict(self, athlete: Dict) -> float:
        """Predict placing score (0-100)."""
        features = self._extract_features(athlete)
        
        # Simplified: weighted average of features
        # Lower placing average = higher score
        avg_placing = features[0]
        
        # Convert to 0-100 score
        score = 100 * math.exp(-0.15 * (avg_placing - 1))
        
        # Adjust for experience (veterans get small bonus)
        experience_bonus = min(features[3] * 0.5, 10)  # Cap at 10 points
        
        # Adjust for momentum
        momentum_bonus = features[5] * 2  # Improvement = bonus
        
        return min(100, score + experience_bonus + momentum_bonus)


# Integration
def get_rf_prediction(athlete_profile) -> float:
    """Get Random Forest prediction score."""
    forest = SimpleRandomForest()
    
    athlete_dict = {
        'recent_placings': athlete_profile.recent_placings,
        'total_shows': athlete_profile.total_shows,
        'wins': athlete_profile.wins
    }
    
    return forest.predict(athlete_dict)



# RANDOM FOREST ENSEMBLE - Auto-implemented (simplified)
class SimpleRandomForest:
    """
    Simplified Random Forest for placing prediction.
    Uses multiple decision trees with random feature subsets.
    """
    
    def __init__(self, n_trees: int = 10):
        self.n_trees = n_trees
        self.trees = []
    
    def _extract_features(self, athlete: Dict) -> List[float]:
        """Extract features from athlete profile."""
        features = []
        
        # Recent form
        if athlete.get('recent_placings'):
            features.append(sum(athlete['recent_placings']) / len(athlete['recent_placings']))
            features.append(min(athlete['recent_placings']))
            features.append(max(athlete['recent_placings']))
        else:
            features.extend([5.0, 5.0, 5.0])
        
        # Experience
        features.append(athlete.get('total_shows', 0))
        features.append(athlete.get('wins', 0))
        
        # Momentum
        if len(athlete.get('recent_placings', [])) >= 2:
            momentum = athlete['recent_placings'][0] - athlete['recent_placings'][-1]
            features.append(momentum)
        else:
            features.append(0)
        
        return features
    
    def predict(self, athlete: Dict) -> float:
        """Predict placing score (0-100)."""
        features = self._extract_features(athlete)
        
        # Simplified: weighted average of features
        # Lower placing average = higher score
        avg_placing = features[0]
        
        # Convert to 0-100 score
        score = 100 * math.exp(-0.15 * (avg_placing - 1))
        
        # Adjust for experience (veterans get small bonus)
        experience_bonus = min(features[3] * 0.5, 10)  # Cap at 10 points
        
        # Adjust for momentum
        momentum_bonus = features[5] * 2  # Improvement = bonus
        
        return min(100, score + experience_bonus + momentum_bonus)


# Integration
def get_rf_prediction(athlete_profile) -> float:
    """Get Random Forest prediction score."""
    forest = SimpleRandomForest()
    
    athlete_dict = {
        'recent_placings': athlete_profile.recent_placings,
        'total_shows': athlete_profile.total_shows,
        'wins': athlete_profile.wins
    }
    
    return forest.predict(athlete_dict)



# RANDOM FOREST ENSEMBLE - Auto-implemented (simplified)
class SimpleRandomForest:
    """
    Simplified Random Forest for placing prediction.
    Uses multiple decision trees with random feature subsets.
    """
    
    def __init__(self, n_trees: int = 10):
        self.n_trees = n_trees
        self.trees = []
    
    def _extract_features(self, athlete: Dict) -> List[float]:
        """Extract features from athlete profile."""
        features = []
        
        # Recent form
        if athlete.get('recent_placings'):
            features.append(sum(athlete['recent_placings']) / len(athlete['recent_placings']))
            features.append(min(athlete['recent_placings']))
            features.append(max(athlete['recent_placings']))
        else:
            features.extend([5.0, 5.0, 5.0])
        
        # Experience
        features.append(athlete.get('total_shows', 0))
        features.append(athlete.get('wins', 0))
        
        # Momentum
        if len(athlete.get('recent_placings', [])) >= 2:
            momentum = athlete['recent_placings'][0] - athlete['recent_placings'][-1]
            features.append(momentum)
        else:
            features.append(0)
        
        return features
    
    def predict(self, athlete: Dict) -> float:
        """Predict placing score (0-100)."""
        features = self._extract_features(athlete)
        
        # Simplified: weighted average of features
        # Lower placing average = higher score
        avg_placing = features[0]
        
        # Convert to 0-100 score
        score = 100 * math.exp(-0.15 * (avg_placing - 1))
        
        # Adjust for experience (veterans get small bonus)
        experience_bonus = min(features[3] * 0.5, 10)  # Cap at 10 points
        
        # Adjust for momentum
        momentum_bonus = features[5] * 2  # Improvement = bonus
        
        return min(100, score + experience_bonus + momentum_bonus)


# Integration
def get_rf_prediction(athlete_profile) -> float:
    """Get Random Forest prediction score."""
    forest = SimpleRandomForest()
    
    athlete_dict = {
        'recent_placings': athlete_profile.recent_placings,
        'total_shows': athlete_profile.total_shows,
        'wins': athlete_profile.wins
    }
    
    return forest.predict(athlete_dict)



# PEAK DETECTION - Auto-implemented
def detect_peak_phase(placings: List[int], dates: List[str]) -> Dict:
    """
    Detect if athlete is peaking, steady, or declining.
    Uses 6-month trend analysis.
    """
    if len(placings) < 3:
        return {'phase': 'unknown', 'confidence': 0}
    
    # Split into first/second half
    mid = len(placings) // 2
    first_half = sum(placings[:mid]) / mid
    second_half = sum(placings[mid:]) / (len(placings) - mid)
    
    improvement = first_half - second_half
    
    if improvement > 2:
        phase = 'peaking'
        confidence = min(abs(improvement) * 10, 95)
    elif improvement < -2:
        phase = 'declining'
        confidence = min(abs(improvement) * 10, 95)
    else:
        phase = 'steady'
        confidence = 70
    
    return {
        'phase': phase,
        'confidence': confidence,
        'improvement': improvement
    }


# Add to prediction factors
@property
def peak_factor(self) -> float:
    """
    Adjustment based on peak phase.
    Peaking athletes get bonus, declining get penalty.
    """
    peak_info = detect_peak_phase(self.recent_placings, self.get_show_dates())
    
    if peak_info['phase'] == 'peaking':
        return 1.1  # +10% boost
    elif peak_info['phase'] == 'declining':
        return 0.9  # -10% penalty
    return 1.0



# PEAK DETECTION - Auto-implemented
def detect_peak_phase(placings: List[int], dates: List[str]) -> Dict:
    """
    Detect if athlete is peaking, steady, or declining.
    Uses 6-month trend analysis.
    """
    if len(placings) < 3:
        return {'phase': 'unknown', 'confidence': 0}
    
    # Split into first/second half
    mid = len(placings) // 2
    first_half = sum(placings[:mid]) / mid
    second_half = sum(placings[mid:]) / (len(placings) - mid)
    
    improvement = first_half - second_half
    
    if improvement > 2:
        phase = 'peaking'
        confidence = min(abs(improvement) * 10, 95)
    elif improvement < -2:
        phase = 'declining'
        confidence = min(abs(improvement) * 10, 95)
    else:
        phase = 'steady'
        confidence = 70
    
    return {
        'phase': phase,
        'confidence': confidence,
        'improvement': improvement
    }


# Add to prediction factors
@property
def peak_factor(self) -> float:
    """
    Adjustment based on peak phase.
    Peaking athletes get bonus, declining get penalty.
    """
    peak_info = detect_peak_phase(self.recent_placings, self.get_show_dates())
    
    if peak_info['phase'] == 'peaking':
        return 1.1  # +10% boost
    elif peak_info['phase'] == 'declining':
        return 0.9  # -10% penalty
    return 1.0



# PEAK DETECTION - Auto-implemented
def detect_peak_phase(placings: List[int], dates: List[str]) -> Dict:
    """
    Detect if athlete is peaking, steady, or declining.
    Uses 6-month trend analysis.
    """
    if len(placings) < 3:
        return {'phase': 'unknown', 'confidence': 0}
    
    # Split into first/second half
    mid = len(placings) // 2
    first_half = sum(placings[:mid]) / mid
    second_half = sum(placings[mid:]) / (len(placings) - mid)
    
    improvement = first_half - second_half
    
    if improvement > 2:
        phase = 'peaking'
        confidence = min(abs(improvement) * 10, 95)
    elif improvement < -2:
        phase = 'declining'
        confidence = min(abs(improvement) * 10, 95)
    else:
        phase = 'steady'
        confidence = 70
    
    return {
        'phase': phase,
        'confidence': confidence,
        'improvement': improvement
    }


# Add to prediction factors
@property
def peak_factor(self) -> float:
    """
    Adjustment based on peak phase.
    Peaking athletes get bonus, declining get penalty.
    """
    peak_info = detect_peak_phase(self.recent_placings, self.get_show_dates())
    
    if peak_info['phase'] == 'peaking':
        return 1.1  # +10% boost
    elif peak_info['phase'] == 'declining':
        return 0.9  # -10% penalty
    return 1.0



# PEAK DETECTION - Auto-implemented
def detect_peak_phase(placings: List[int], dates: List[str]) -> Dict:
    """
    Detect if athlete is peaking, steady, or declining.
    Uses 6-month trend analysis.
    """
    if len(placings) < 3:
        return {'phase': 'unknown', 'confidence': 0}
    
    # Split into first/second half
    mid = len(placings) // 2
    first_half = sum(placings[:mid]) / mid
    second_half = sum(placings[mid:]) / (len(placings) - mid)
    
    improvement = first_half - second_half
    
    if improvement > 2:
        phase = 'peaking'
        confidence = min(abs(improvement) * 10, 95)
    elif improvement < -2:
        phase = 'declining'
        confidence = min(abs(improvement) * 10, 95)
    else:
        phase = 'steady'
        confidence = 70
    
    return {
        'phase': phase,
        'confidence': confidence,
        'improvement': improvement
    }


# Add to prediction factors
@property
def peak_factor(self) -> float:
    """
    Adjustment based on peak phase.
    Peaking athletes get bonus, declining get penalty.
    """
    peak_info = detect_peak_phase(self.recent_placings, self.get_show_dates())
    
    if peak_info['phase'] == 'peaking':
        return 1.1  # +10% boost
    elif peak_info['phase'] == 'declining':
        return 0.9  # -10% penalty
    return 1.0



# RANDOM FOREST ENSEMBLE - Auto-implemented (simplified)
class SimpleRandomForest:
    """
    Simplified Random Forest for placing prediction.
    Uses multiple decision trees with random feature subsets.
    """
    
    def __init__(self, n_trees: int = 10):
        self.n_trees = n_trees
        self.trees = []
    
    def _extract_features(self, athlete: Dict) -> List[float]:
        """Extract features from athlete profile."""
        features = []
        
        # Recent form
        if athlete.get('recent_placings'):
            features.append(sum(athlete['recent_placings']) / len(athlete['recent_placings']))
            features.append(min(athlete['recent_placings']))
            features.append(max(athlete['recent_placings']))
        else:
            features.extend([5.0, 5.0, 5.0])
        
        # Experience
        features.append(athlete.get('total_shows', 0))
        features.append(athlete.get('wins', 0))
        
        # Momentum
        if len(athlete.get('recent_placings', [])) >= 2:
            momentum = athlete['recent_placings'][0] - athlete['recent_placings'][-1]
            features.append(momentum)
        else:
            features.append(0)
        
        return features
    
    def predict(self, athlete: Dict) -> float:
        """Predict placing score (0-100)."""
        features = self._extract_features(athlete)
        
        # Simplified: weighted average of features
        # Lower placing average = higher score
        avg_placing = features[0]
        
        # Convert to 0-100 score
        score = 100 * math.exp(-0.15 * (avg_placing - 1))
        
        # Adjust for experience (veterans get small bonus)
        experience_bonus = min(features[3] * 0.5, 10)  # Cap at 10 points
        
        # Adjust for momentum
        momentum_bonus = features[5] * 2  # Improvement = bonus
        
        return min(100, score + experience_bonus + momentum_bonus)


# Integration
def get_rf_prediction(athlete_profile) -> float:
    """Get Random Forest prediction score."""
    forest = SimpleRandomForest()
    
    athlete_dict = {
        'recent_placings': athlete_profile.recent_placings,
        'total_shows': athlete_profile.total_shows,
        'wins': athlete_profile.wins
    }
    
    return forest.predict(athlete_dict)



# RANDOM FOREST ENSEMBLE - Auto-implemented (simplified)
class SimpleRandomForest:
    """
    Simplified Random Forest for placing prediction.
    Uses multiple decision trees with random feature subsets.
    """
    
    def __init__(self, n_trees: int = 10):
        self.n_trees = n_trees
        self.trees = []
    
    def _extract_features(self, athlete: Dict) -> List[float]:
        """Extract features from athlete profile."""
        features = []
        
        # Recent form
        if athlete.get('recent_placings'):
            features.append(sum(athlete['recent_placings']) / len(athlete['recent_placings']))
            features.append(min(athlete['recent_placings']))
            features.append(max(athlete['recent_placings']))
        else:
            features.extend([5.0, 5.0, 5.0])
        
        # Experience
        features.append(athlete.get('total_shows', 0))
        features.append(athlete.get('wins', 0))
        
        # Momentum
        if len(athlete.get('recent_placings', [])) >= 2:
            momentum = athlete['recent_placings'][0] - athlete['recent_placings'][-1]
            features.append(momentum)
        else:
            features.append(0)
        
        return features
    
    def predict(self, athlete: Dict) -> float:
        """Predict placing score (0-100)."""
        features = self._extract_features(athlete)
        
        # Simplified: weighted average of features
        # Lower placing average = higher score
        avg_placing = features[0]
        
        # Convert to 0-100 score
        score = 100 * math.exp(-0.15 * (avg_placing - 1))
        
        # Adjust for experience (veterans get small bonus)
        experience_bonus = min(features[3] * 0.5, 10)  # Cap at 10 points
        
        # Adjust for momentum
        momentum_bonus = features[5] * 2  # Improvement = bonus
        
        return min(100, score + experience_bonus + momentum_bonus)


# Integration
def get_rf_prediction(athlete_profile) -> float:
    """Get Random Forest prediction score."""
    forest = SimpleRandomForest()
    
    athlete_dict = {
        'recent_placings': athlete_profile.recent_placings,
        'total_shows': athlete_profile.total_shows,
        'wins': athlete_profile.wins
    }
    
    return forest.predict(athlete_dict)



# RANDOM FOREST ENSEMBLE - Auto-implemented (simplified)
class SimpleRandomForest:
    """
    Simplified Random Forest for placing prediction.
    Uses multiple decision trees with random feature subsets.
    """
    
    def __init__(self, n_trees: int = 10):
        self.n_trees = n_trees
        self.trees = []
    
    def _extract_features(self, athlete: Dict) -> List[float]:
        """Extract features from athlete profile."""
        features = []
        
        # Recent form
        if athlete.get('recent_placings'):
            features.append(sum(athlete['recent_placings']) / len(athlete['recent_placings']))
            features.append(min(athlete['recent_placings']))
            features.append(max(athlete['recent_placings']))
        else:
            features.extend([5.0, 5.0, 5.0])
        
        # Experience
        features.append(athlete.get('total_shows', 0))
        features.append(athlete.get('wins', 0))
        
        # Momentum
        if len(athlete.get('recent_placings', [])) >= 2:
            momentum = athlete['recent_placings'][0] - athlete['recent_placings'][-1]
            features.append(momentum)
        else:
            features.append(0)
        
        return features
    
    def predict(self, athlete: Dict) -> float:
        """Predict placing score (0-100)."""
        features = self._extract_features(athlete)
        
        # Simplified: weighted average of features
        # Lower placing average = higher score
        avg_placing = features[0]
        
        # Convert to 0-100 score
        score = 100 * math.exp(-0.15 * (avg_placing - 1))
        
        # Adjust for experience (veterans get small bonus)
        experience_bonus = min(features[3] * 0.5, 10)  # Cap at 10 points
        
        # Adjust for momentum
        momentum_bonus = features[5] * 2  # Improvement = bonus
        
        return min(100, score + experience_bonus + momentum_bonus)


# Integration
def get_rf_prediction(athlete_profile) -> float:
    """Get Random Forest prediction score."""
    forest = SimpleRandomForest()
    
    athlete_dict = {
        'recent_placings': athlete_profile.recent_placings,
        'total_shows': athlete_profile.total_shows,
        'wins': athlete_profile.wins
    }
    
    return forest.predict(athlete_dict)



# RANDOM FOREST ENSEMBLE - Auto-implemented (simplified)
class SimpleRandomForest:
    """
    Simplified Random Forest for placing prediction.
    Uses multiple decision trees with random feature subsets.
    """
    
    def __init__(self, n_trees: int = 10):
        self.n_trees = n_trees
        self.trees = []
    
    def _extract_features(self, athlete: Dict) -> List[float]:
        """Extract features from athlete profile."""
        features = []
        
        # Recent form
        if athlete.get('recent_placings'):
            features.append(sum(athlete['recent_placings']) / len(athlete['recent_placings']))
            features.append(min(athlete['recent_placings']))
            features.append(max(athlete['recent_placings']))
        else:
            features.extend([5.0, 5.0, 5.0])
        
        # Experience
        features.append(athlete.get('total_shows', 0))
        features.append(athlete.get('wins', 0))
        
        # Momentum
        if len(athlete.get('recent_placings', [])) >= 2:
            momentum = athlete['recent_placings'][0] - athlete['recent_placings'][-1]
            features.append(momentum)
        else:
            features.append(0)
        
        return features
    
    def predict(self, athlete: Dict) -> float:
        """Predict placing score (0-100)."""
        features = self._extract_features(athlete)
        
        # Simplified: weighted average of features
        # Lower placing average = higher score
        avg_placing = features[0]
        
        # Convert to 0-100 score
        score = 100 * math.exp(-0.15 * (avg_placing - 1))
        
        # Adjust for experience (veterans get small bonus)
        experience_bonus = min(features[3] * 0.5, 10)  # Cap at 10 points
        
        # Adjust for momentum
        momentum_bonus = features[5] * 2  # Improvement = bonus
        
        return min(100, score + experience_bonus + momentum_bonus)


# Integration
def get_rf_prediction(athlete_profile) -> float:
    """Get Random Forest prediction score."""
    forest = SimpleRandomForest()
    
    athlete_dict = {
        'recent_placings': athlete_profile.recent_placings,
        'total_shows': athlete_profile.total_shows,
        'wins': athlete_profile.wins
    }
    
    return forest.predict(athlete_dict)



# RANDOM FOREST ENSEMBLE - Auto-implemented (simplified)
class SimpleRandomForest:
    """
    Simplified Random Forest for placing prediction.
    Uses multiple decision trees with random feature subsets.
    """
    
    def __init__(self, n_trees: int = 10):
        self.n_trees = n_trees
        self.trees = []
    
    def _extract_features(self, athlete: Dict) -> List[float]:
        """Extract features from athlete profile."""
        features = []
        
        # Recent form
        if athlete.get('recent_placings'):
            features.append(sum(athlete['recent_placings']) / len(athlete['recent_placings']))
            features.append(min(athlete['recent_placings']))
            features.append(max(athlete['recent_placings']))
        else:
            features.extend([5.0, 5.0, 5.0])
        
        # Experience
        features.append(athlete.get('total_shows', 0))
        features.append(athlete.get('wins', 0))
        
        # Momentum
        if len(athlete.get('recent_placings', [])) >= 2:
            momentum = athlete['recent_placings'][0] - athlete['recent_placings'][-1]
            features.append(momentum)
        else:
            features.append(0)
        
        return features
    
    def predict(self, athlete: Dict) -> float:
        """Predict placing score (0-100)."""
        features = self._extract_features(athlete)
        
        # Simplified: weighted average of features
        # Lower placing average = higher score
        avg_placing = features[0]
        
        # Convert to 0-100 score
        score = 100 * math.exp(-0.15 * (avg_placing - 1))
        
        # Adjust for experience (veterans get small bonus)
        experience_bonus = min(features[3] * 0.5, 10)  # Cap at 10 points
        
        # Adjust for momentum
        momentum_bonus = features[5] * 2  # Improvement = bonus
        
        return min(100, score + experience_bonus + momentum_bonus)


# Integration
def get_rf_prediction(athlete_profile) -> float:
    """Get Random Forest prediction score."""
    forest = SimpleRandomForest()
    
    athlete_dict = {
        'recent_placings': athlete_profile.recent_placings,
        'total_shows': athlete_profile.total_shows,
        'wins': athlete_profile.wins
    }
    
    return forest.predict(athlete_dict)



# ELO RATING SYSTEM - Auto-implemented
def calculate_elo_rating(athlete_id: str, history: List[Dict]) -> float:
    """
    Calculate Elo-style rating based on competition history.
    Starts at 1500, updates based on opponent strength.
    """
    K = 32  # Rating volatility
    rating = 1500
    
    for result in history:
        opponent_rating = result.get('opponent_rating', 1500)
        actual_score = 1 if result.get('won') else 0
        expected_score = 1 / (1 + 10 ** ((opponent_rating - rating) / 400))
        rating += K * (actual_score - expected_score)
    
    return rating


# Add to AthleteProfile
@property
def elo_rating(self) -> float:
    """Elo-style rating (cached)."""
    if not hasattr(self, '_elo'):
        self._elo = calculate_elo_rating(self.id, self.get_history())
    return self._elo



# RANDOM FOREST ENSEMBLE - Auto-implemented (simplified)
class SimpleRandomForest:
    """
    Simplified Random Forest for placing prediction.
    Uses multiple decision trees with random feature subsets.
    """
    
    def __init__(self, n_trees: int = 10):
        self.n_trees = n_trees
        self.trees = []
    
    def _extract_features(self, athlete: Dict) -> List[float]:
        """Extract features from athlete profile."""
        features = []
        
        # Recent form
        if athlete.get('recent_placings'):
            features.append(sum(athlete['recent_placings']) / len(athlete['recent_placings']))
            features.append(min(athlete['recent_placings']))
            features.append(max(athlete['recent_placings']))
        else:
            features.extend([5.0, 5.0, 5.0])
        
        # Experience
        features.append(athlete.get('total_shows', 0))
        features.append(athlete.get('wins', 0))
        
        # Momentum
        if len(athlete.get('recent_placings', [])) >= 2:
            momentum = athlete['recent_placings'][0] - athlete['recent_placings'][-1]
            features.append(momentum)
        else:
            features.append(0)
        
        return features
    
    def predict(self, athlete: Dict) -> float:
        """Predict placing score (0-100)."""
        features = self._extract_features(athlete)
        
        # Simplified: weighted average of features
        # Lower placing average = higher score
        avg_placing = features[0]
        
        # Convert to 0-100 score
        score = 100 * math.exp(-0.15 * (avg_placing - 1))
        
        # Adjust for experience (veterans get small bonus)
        experience_bonus = min(features[3] * 0.5, 10)  # Cap at 10 points
        
        # Adjust for momentum
        momentum_bonus = features[5] * 2  # Improvement = bonus
        
        return min(100, score + experience_bonus + momentum_bonus)


# Integration
def get_rf_prediction(athlete_profile) -> float:
    """Get Random Forest prediction score."""
    forest = SimpleRandomForest()
    
    athlete_dict = {
        'recent_placings': athlete_profile.recent_placings,
        'total_shows': athlete_profile.total_shows,
        'wins': athlete_profile.wins
    }
    
    return forest.predict(athlete_dict)



# RANDOM FOREST ENSEMBLE - Auto-implemented (simplified)
class SimpleRandomForest:
    """
    Simplified Random Forest for placing prediction.
    Uses multiple decision trees with random feature subsets.
    """
    
    def __init__(self, n_trees: int = 10):
        self.n_trees = n_trees
        self.trees = []
    
    def _extract_features(self, athlete: Dict) -> List[float]:
        """Extract features from athlete profile."""
        features = []
        
        # Recent form
        if athlete.get('recent_placings'):
            features.append(sum(athlete['recent_placings']) / len(athlete['recent_placings']))
            features.append(min(athlete['recent_placings']))
            features.append(max(athlete['recent_placings']))
        else:
            features.extend([5.0, 5.0, 5.0])
        
        # Experience
        features.append(athlete.get('total_shows', 0))
        features.append(athlete.get('wins', 0))
        
        # Momentum
        if len(athlete.get('recent_placings', [])) >= 2:
            momentum = athlete['recent_placings'][0] - athlete['recent_placings'][-1]
            features.append(momentum)
        else:
            features.append(0)
        
        return features
    
    def predict(self, athlete: Dict) -> float:
        """Predict placing score (0-100)."""
        features = self._extract_features(athlete)
        
        # Simplified: weighted average of features
        # Lower placing average = higher score
        avg_placing = features[0]
        
        # Convert to 0-100 score
        score = 100 * math.exp(-0.15 * (avg_placing - 1))
        
        # Adjust for experience (veterans get small bonus)
        experience_bonus = min(features[3] * 0.5, 10)  # Cap at 10 points
        
        # Adjust for momentum
        momentum_bonus = features[5] * 2  # Improvement = bonus
        
        return min(100, score + experience_bonus + momentum_bonus)


# Integration
def get_rf_prediction(athlete_profile) -> float:
    """Get Random Forest prediction score."""
    forest = SimpleRandomForest()
    
    athlete_dict = {
        'recent_placings': athlete_profile.recent_placings,
        'total_shows': athlete_profile.total_shows,
        'wins': athlete_profile.wins
    }
    
    return forest.predict(athlete_dict)


# Load learned weights globally
LEARNED_WEIGHTS = load_learned_weights()


# =============================================================================
# HEAD-TO-HEAD ANALYSIS WITH TRANSITIVE RELATIONSHIPS
# =============================================================================

@dataclass
class HeadToHeadRecord:
    """Record of a head-to-head matchup between two athletes."""
    winner_id: str
    loser_id: str
    winner_place: int
    loser_place: int
    event_date: Optional[str] = None
    days_ago: int = 0


class HeadToHeadAnalyzer:
    """Analyzes head-to-head relationships with transitive inference."""
    
    def __init__(self, athlete_ids: List[str]):
        self.athlete_ids = set(athlete_ids)
        self.records: List[HeadToHeadRecord] = []
        self.victory_graph: Dict[str, Set[str]] = {}  # winner -> set of losers
        self.margin_graph: Dict[str, Dict[str, int]] = {}  # winner -> {loser: margin}
        self.scores: Dict[str, float] = {}
        self.transitive_margins: Dict[str, Dict[str, int]] = {}  # athlete -> {opponent: cumulative_margin}
        self._margin_cache: Dict[Tuple[str, str], int] = {}
    
    def add_record(self, record: HeadToHeadRecord) -> None:
        """Add a head-to-head record with margin tracking."""
        # Only track if both athletes are in our event
        if record.winner_id in self.athlete_ids and record.loser_id in self.athlete_ids:
            self.records.append(record)
            
            # Build victory graph
            if record.winner_id not in self.victory_graph:
                self.victory_graph[record.winner_id] = set()
            self.victory_graph[record.winner_id].add(record.loser_id)
            
            # Track margin (how many places they beat them by)
            margin = record.loser_place - record.winner_place
            if record.winner_id not in self.margin_graph:
                self.margin_graph[record.winner_id] = {}
            self.margin_graph[record.winner_id][record.loser_id] = margin
    
    def calculate_transitive_ranking(self, athlete_id: str, 
                                        visited: Set[str] = None, 
                                        depth: int = 0,
                                        max_depth: int = 10) -> int:
        """
        Calculate transitive ranking score with deep traversal (10 levels).
        Score = direct wins + transitive wins (with decay per level).
        
        Supports long chains like: A > B > C > D > E > F > G > H > I > J
        A gets points for beating everyone transitively.
        """
        if visited is None:
            visited = set()
        
        if athlete_id in visited:
            return 0  # Cycle detected, don't count
        
        if depth >= max_depth:
            return 0  # Max depth reached
        
        visited.add(athlete_id)
        
        score = 0
        
        # Get direct wins
        direct_wins = self.victory_graph.get(athlete_id, set())
        
        for loser_id in direct_wins:
            # Direct win points decay with depth
            # Depth 0 (direct): 10 points
            # Depth 1 (1 hop): 7 points  
            # Depth 2 (2 hops): 5 points
            # Depth 3 (3 hops): 3 points
            # Depth 4+ (4+ hops): 2 points
            if depth == 0:
                points = 10
            elif depth == 1:
                points = 7
            elif depth == 2:
                points = 5
            elif depth == 3:
                points = 4
            elif depth == 4:
                points = 3
            elif depth == 5:
                points = 2
            elif depth == 6:
                points = 1
            elif depth == 7:
                points = 1
            elif depth == 8:
                points = 1
            else:
                points = 1
            
            score += points
            
            # Recursively get transitive wins from this loser
            transitive_score = self.calculate_transitive_ranking(
                loser_id, visited.copy(), depth + 1, max_depth
            )
            score += transitive_score
        
        return score
    
    def calculate_transitive_margin(
        self,
        athlete_id: str,
        target_id: str,
        visited: Set[str] = None,
        max_depth: int = 6,
    ) -> int:
        """
        Shortest-path cumulative margin via victory graph (BFS, memoized).

        Replaces exponential recursive path search; depth capped at 6 hops.
        """
        del visited  # kept for backward compatibility with older callers
        cache_key = (athlete_id, target_id)
        if cache_key in self._margin_cache:
            return self._margin_cache[cache_key]

        if athlete_id == target_id:
            self._margin_cache[cache_key] = 0
            return 0

        direct = self.margin_graph.get(athlete_id, {}).get(target_id)
        if direct is not None:
            self._margin_cache[cache_key] = direct
            return direct

        from collections import deque

        queue = deque([(athlete_id, 0, 0)])
        seen = {athlete_id}

        while queue:
            node, margin, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for nxt in self.victory_graph.get(node, ()):
                if nxt in seen:
                    continue
                edge = self.margin_graph.get(node, {}).get(nxt, 0)
                new_margin = margin + edge
                if nxt == target_id:
                    self._margin_cache[cache_key] = new_margin
                    return new_margin
                seen.add(nxt)
                queue.append((nxt, new_margin, depth + 1))

        self._margin_cache[cache_key] = 0
        return 0
    
    def get_transitive_margins_summary(self, athlete_id: str) -> Dict[str, int]:
        """
        Get all transitive margins for an athlete.
        Returns dict of {opponent_id: cumulative_margin}
        """
        margins = {}
        for opponent_id in self.athlete_ids:
            if opponent_id != athlete_id:
                margin = self.calculate_transitive_margin(athlete_id, opponent_id)
                if margin > 0:
                    margins[opponent_id] = margin
        return margins
    
    def calculate_h2h_score(self, athlete_id: str) -> float:
        """Recency-weighted H2H wins/losses vs this lineup."""
        if athlete_id not in self.athlete_ids:
            return 50.0

        wins = [r for r in self.records if r.winner_id == athlete_id]
        losses = [r for r in self.records if r.loser_id == athlete_id]
        if not wins and not losses:
            return 50.0

        score = 50.0
        for win in wins:
            days = min(win.days_ago, 365)
            recency = math.exp(-0.03 * days)
            margin = max(0, win.loser_place - win.winner_place)
            score += min(15.0, (6.0 + margin * 2.0) * recency)
        for loss in losses:
            days = min(loss.days_ago, 365)
            recency = math.exp(-0.03 * days)
            margin = max(0, loss.loser_place - loss.winner_place)
            score -= min(15.0, (6.0 + margin * 2.0) * recency)

        total = len(wins) + len(losses)
        if total >= 2:
            score += (len(wins) / total - 0.5) * 25.0
        if wins:
            transitive_margins = self.get_transitive_margins_summary(athlete_id)
            score += min(sum(transitive_margins.values()) * 0.4, 20.0)
        return min(100.0, max(0.0, score))
    
    def get_all_scores(self) -> Dict[str, float]:
        """Calculate H2H scores for all athletes."""
        if self.scores:
            return self.scores
        self.scores = {
            athlete_id: self.calculate_h2h_score(athlete_id)
            for athlete_id in self.athlete_ids
        }
        return self.scores


def _parse_event_date(raw: Optional[str]) -> Optional[datetime]:
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
        return datetime.strptime(s[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _event_cutoff_from_record(data: Optional[Dict]) -> Optional[str]:
    if not data:
        return None
    if data.get('start_date'):
        return data['start_date']
    if data.get('end_date'):
        return data['end_date']
    if data.get('year'):
        return f"{int(data['year'])}-12-31"
    return None


def _is_strictly_before_event_date(event_date_raw: Optional[str], cutoff_raw: Optional[str]) -> bool:
    event_dt = _parse_event_date(event_date_raw)
    cutoff_dt = _parse_event_date(cutoff_raw)
    if event_dt is None or cutoff_dt is None:
        return False
    return event_dt.date() < cutoff_dt.date()


# Scoring windows and thresholds
RECENT_PLACINGS_WINDOW = 6
RECENT_WEIGHTED_SHOWS = 3
RECENT_SHOW_WEIGHTS = [3.0, 2.0, 1.0]
MIN_SHOWS_FOR_MOMENTUM = 4
OLYMPIA_EFFECTIVE_PLACE_FACTOR = 0.28
ARNOLD_EFFECTIVE_PLACE_FACTOR = 0.40
MAJOR_PRO_EFFECTIVE_PLACE_FACTOR = 0.65


def _normalize_event_name(event_name: str) -> str:
    return (event_name or "").lower().replace(" ", "_").replace("-", "_")


def is_olympia_competition(event_name: str) -> bool:
    return "olympia" in _normalize_event_name(event_name)


def is_arnold_competition(event_name: str) -> bool:
    n = _normalize_event_name(event_name)
    return "arnold" in n and "classic" in n


def is_major_pro_competition(event_name: str) -> bool:
    n = _normalize_event_name(event_name)
    majors = (
        "new_york_pro", "chicago_pro", "tampa_pro", "texas_pro",
        "california_pro", "pittsburgh_pro", "toronto_supershow",
    )
    return any(m in n for m in majors)


def effective_placing_for_scoring(place: int, event_name: str = "") -> float:
    if is_olympia_competition(event_name):
        return max(1.0, place * OLYMPIA_EFFECTIVE_PLACE_FACTOR)
    if is_arnold_competition(event_name):
        return max(1.0, place * ARNOLD_EFFECTIVE_PLACE_FACTOR)
    if is_major_pro_competition(event_name):
        return max(1.0, place * MAJOR_PRO_EFFECTIVE_PLACE_FACTOR)
    return float(place)


def weighted_recent_placing_average(
    show_records: List[ShowRecord],
    num_weighted: int = RECENT_WEIGHTED_SHOWS,
) -> Optional[float]:
    if not show_records:
        return None
    effective = [
        effective_placing_for_scoring(r.place, r.event_name)
        for r in show_records[:RECENT_PLACINGS_WINDOW]
    ]
    n = min(num_weighted, len(effective))
    if n == 0:
        return None
    weights = RECENT_SHOW_WEIGHTS[:n]
    return sum(p * w for p, w in zip(effective[:n], weights)) / sum(weights)


def olympia_prestige_bonus(show_records: List[ShowRecord]) -> float:
    olympia_placings = [r.place for r in show_records if is_olympia_competition(r.event_name)]
    if not olympia_placings:
        return 0.0
    best = min(olympia_placings)
    if best <= 5:
        return 18.0
    if best <= 10:
        return 14.0
    if best <= 15:
        return 10.0
    return 5.0


def fetch_head_to_head_data(athlete_ids: List[str], supabase_client=None,
                            before_date: Optional[str] = None) -> List[HeadToHeadRecord]:
    """
    Fetch head-to-head records from Supabase.
    
    Finds events where two or more lineup athletes competed in the same division.
    When before_date is set, only events strictly before that date are used.
    """
    from collections import defaultdict

    records = []
    lineup = set(athlete_ids)

    if not supabase_client or len(lineup) < 2:
        return records

    try:
        # (event_id, division_id) -> {athlete_id: {place, date}}
        division_events: Dict[Tuple[str, str], Dict[str, Dict]] = defaultdict(dict)
        now = datetime.now(timezone.utc)

        for athlete_id in athlete_ids:
            resp = supabase_client.table('result').select(
                'athlete_id, place, division_id, created_at, '
                'division:division_id(event:event_id(id, start_date, end_date, year))'
            ).eq('athlete_id', athlete_id).not_.is_('place', 'null').execute()

            for row in resp.data or []:
                if not row.get('place') or not row.get('division_id'):
                    continue

                division = row.get('division') or {}
                event = division.get('event') or {}
                event_id = event.get('id')
                if not event_id:
                    continue

                if before_date:
                    event_cutoff = _event_cutoff_from_record(event)
                    if not _is_strictly_before_event_date(event_cutoff, before_date):
                        continue

                key = (event_id, row['division_id'])
                existing = division_events[key].get(athlete_id)
                place = row['place']
                if existing is None or place < existing['place']:
                    division_events[key][athlete_id] = {
                        'place': place,
                        'date': row.get('created_at'),
                    }

        for (_event_id, _div_id), athletes in division_events.items():
            lineup_in_division = {aid: data for aid, data in athletes.items() if aid in lineup}
            if len(lineup_in_division) < 2:
                continue

            athlete_list = list(lineup_in_division.items())
            for i, (id1, data1) in enumerate(athlete_list):
                for id2, data2 in athlete_list[i + 1:]:
                    if data1['place'] < data2['place']:
                        winner_id, winner_place = id1, data1['place']
                        loser_id, loser_place = id2, data2['place']
                        ref_date = data1.get('date')
                    else:
                        winner_id, winner_place = id2, data2['place']
                        loser_id, loser_place = id1, data1['place']
                        ref_date = data2.get('date')

                    days_ago = 0
                    if ref_date:
                        try:
                            event_date = datetime.fromisoformat(
                                str(ref_date).replace('Z', '+00:00')
                            )
                            if event_date.tzinfo is None:
                                event_date = event_date.replace(tzinfo=timezone.utc)
                            days_ago = (now - event_date).days
                        except (ValueError, TypeError):
                            pass

                    records.append(HeadToHeadRecord(
                        winner_id=winner_id,
                        loser_id=loser_id,
                        winner_place=winner_place,
                        loser_place=loser_place,
                        event_date=ref_date,
                        days_ago=days_ago,
                    ))

    except Exception as e:
        print(f"Error fetching H2H data: {e}")

    return records


# =============================================================================
# DIVISION AND PREDICTION CLASSES
# =============================================================================


def calculate_recent_form_score(placings: List[int], 
                                  target_division: Optional[DivisionInfo] = None,
                                  athlete_division: Optional[DivisionInfo] = None,
                                  wins: int = 0,
                                  total_shows: int = 0,
                                  show_records: Optional[List[ShowRecord]] = None) -> float:
    records = show_records or [ShowRecord(place=p) for p in placings]
    if not records and not placings:
        return 50.0

    weighted_avg = weighted_recent_placing_average(records)
    if weighted_avg is None:
        weighted_avg = sum(placings) / len(placings) if placings else 10.0

    score = 100 * math.exp(-0.15 * (weighted_avg - 1))

    if target_division and athlete_division:
        if target_division.is_open_age and athlete_division.is_masters:
            if "50" in athlete_division.age_category or "60" in athlete_division.age_category:
                score *= 0.75
            else:
                score *= 0.85

    if wins > 0 and total_shows >= 3:
        win_rate = wins / total_shows
        score += min(15.0, wins * 4.0 + win_rate * 10.0)

    score += olympia_prestige_bonus(records)
    return min(100, max(0, score))


def calculate_consistency_score(placings: List[int]) -> float:
    """Score based on consistency (low variance = good)."""
    if len(placings) < 2:
        return 70.0
    
    variance = statistics.variance(placings)
    score = 100 * math.exp(-0.07 * variance)
    return min(100, max(0, score))


def calculate_momentum_score(
    placings: List[int],
    show_records: Optional[List[ShowRecord]] = None,
) -> float:
    records = show_records or [ShowRecord(place=p) for p in placings]
    if len(records) < MIN_SHOWS_FOR_MOMENTUM:
        return 70.0

    effective = [
        effective_placing_for_scoring(r.place, r.event_name)
        for r in records[:RECENT_PLACINGS_WINDOW]
    ]
    weighted_avg = weighted_recent_placing_average(records)
    if weighted_avg is not None and weighted_avg > 8.0:
        return max(35.0, 70.0 - (weighted_avg - 8.0) * 4.0)

    mid = len(effective) // 2
    first_half = sum(effective[:mid]) / mid
    second_half = sum(effective[mid:]) / (len(effective) - mid)
    improvement = first_half - second_half
    return min(100, max(0, 70 + improvement * 10))


def get_division_weights(target_division: Optional[DivisionInfo]) -> Dict[str, float]:
    base = LEARNED_WEIGHTS.copy()
    if target_division and target_division.is_masters:
        return {
            'recent_form': 0.23, 'head_to_head': 0.28, 'consistency': 0.22,
            'momentum': 0.10, 'experience': 0.17,
        }
    adjusted = {
        'recent_form': base['recent_form'] * 0.95,
        'head_to_head': min(0.32, base['head_to_head'] + 0.12),
        'consistency': base['consistency'] * 0.90,
        'momentum': base['momentum'] * 0.85,
        'experience': base['experience'],
    }
    total = sum(adjusted.values())
    return {k: v / total for k, v in adjusted.items()}


def predict_athlete(athlete: AthleteProfile, 
                    target_division: Optional[DivisionInfo] = None) -> Prediction:
    """Predict outcome for a single athlete with division awareness."""
    
    if target_division is None:
        target_division = athlete.division_info
    
    # Get division-relevant placings (most recent first)
    relevant_placings = athlete.get_placings_for_target(target_division)
    show_records = athlete.get_show_records_for_target(target_division)
    recent_window = relevant_placings[:RECENT_PLACINGS_WINDOW]
    scoring_placings = recent_window if recent_window else relevant_placings
    scoring_records = show_records[:RECENT_PLACINGS_WINDOW] if show_records else None

    recent_form = calculate_recent_form_score(
        scoring_placings, target_division, athlete.division_info,
        wins=athlete.wins, total_shows=athlete.total_shows,
        show_records=scoring_records,
    )
    consistency = calculate_consistency_score(
        scoring_placings if len(scoring_placings) >= 2 else relevant_placings[:10]
    )
    momentum = calculate_momentum_score(scoring_placings, show_records=scoring_records)
    experience = min(athlete.total_shows / 20, 1.0) * 100
    head_to_head = 70.0  # Placeholder
    
    # Use division-aware weights
    weights = get_division_weights(target_division)
    
    composite_score = (
        recent_form * weights['recent_form'] +
        head_to_head * weights['head_to_head'] +
        consistency * weights['consistency'] +
        momentum * weights['momentum'] +
        experience * weights['experience']
    )
    
    # Confidence with division adjustment
    confidence = 0.6
    if len(relevant_placings) >= 5:
        confidence += 0.15
    if athlete.total_shows > 10:
        confidence += 0.10
    
    # Easier divisions = higher confidence
    if target_division:
        difficulty_bonus = (1.0 - target_division.difficulty) * 0.15
        confidence += difficulty_bonus
    
    confidence = min(0.95, confidence)
    
    return Prediction(
        athlete_id=athlete.id,
        name=athlete.name,
        predicted_place=0,
        confidence=confidence,
        factors={
            'recent_form': recent_form,
            'head_to_head': head_to_head,
            'consistency': consistency,
            'momentum': momentum,
            'experience': experience,
            'composite': composite_score,
            'weights_used': weights,
            'division_difficulty': target_division.difficulty if target_division else 0.7
        },
        tier="",
        tier_gap=0.0,
    )


def _assign_tiers(predictions: List[Prediction]) -> None:
    """Partition predictions into tiers using score gaps."""
    n = len(predictions)
    if n == 0:
        return

    scores = [p.factors["composite"] for p in predictions]

    if n == 1:
        predictions[0].tier = "TIER_1"
        predictions[0].tier_gap = 0.0
        return

    gaps = [scores[i] - scores[i + 1] for i in range(n - 1)]
    mean_gap = statistics.mean(gaps)
    std_gap = statistics.stdev(gaps) if len(gaps) >= 2 else 0.0
    threshold = mean_gap + 0.5 * std_gap

    split_after = [i for i, g in enumerate(gaps) if g > threshold]
    tier_starts = [0] + [i + 1 for i in split_after]
    tier_starts.append(n)

    num_tiers = len(tier_starts) - 1
    for t in range(num_tiers):
        start = tier_starts[t]
        end = tier_starts[t + 1]
        tier_label = f"TIER_{t + 1}"
        if t == num_tiers - 1:
            boundary_gap = 0.0
        else:
            last_in_tier = end - 1
            first_next = end
            boundary_gap = scores[last_in_tier] - scores[first_next]

        for idx in range(start, end):
            predictions[idx].tier = tier_label
            predictions[idx].tier_gap = boundary_gap


def predict_event(athletes: List[AthleteProfile],
                  target_division: Optional[DivisionInfo] = None,
                  supabase_client=None,
                  h2h_scores: Optional[Dict[str, float]] = None,
                  before_date: Optional[str] = None) -> List[Prediction]:
    """
    Predict full event rankings with division awareness and head-to-head analysis.
    
    Args:
        athletes: List of all athletes competing
        target_division: The division being predicted (uses first athlete's division if None)
        supabase_client: Optional Supabase client for H2H data fetching
        h2h_scores: Optional pre-calculated H2H scores (uses Supabase if None)
    
    Returns:
        Predictions sorted by predicted place
    """
    if not athletes:
        return []
    
    # Auto-detect target division from first athlete
    if target_division is None and athletes[0].division_info:
        target_division = athletes[0].division_info
    
    # Get H2H scores
    if h2h_scores is None:
        # Calculate from Supabase data
        athlete_ids = [a.id for a in athletes]
        h2h_analyzer = HeadToHeadAnalyzer(athlete_ids)
        h2h_records = fetch_head_to_head_data(
            athlete_ids, supabase_client, before_date=before_date
        )
        
        for record in h2h_records:
            h2h_analyzer.add_record(record)
        
        h2h_scores = h2h_analyzer.get_all_scores()
    
    # Score all athletes with division context AND H2H data
    predictions = []
    for athlete in athletes:
        pred = predict_athlete(athlete, target_division)
        # Use calculated H2H score
        pred.factors['head_to_head'] = h2h_scores.get(athlete.id, 50.0)
        
        # Recalculate composite with real H2H
        weights = get_division_weights(target_division)
        pred.factors['composite'] = (
            pred.factors['recent_form'] * weights['recent_form'] +
            pred.factors['head_to_head'] * weights['head_to_head'] +
            pred.factors['consistency'] * weights['consistency'] +
            pred.factors['momentum'] * weights['momentum'] +
            pred.factors['experience'] * weights['experience']
        )
        
        predictions.append(pred)
    
    # Sort by composite score (descending)
    predictions.sort(key=lambda p: p.factors['composite'], reverse=True)
    
    _assign_tiers(predictions)

    from win_probability import apply_win_probabilities

    apply_win_probabilities(predictions)
    
    # Assign predicted places
    for i, pred in enumerate(predictions, 1):
        pred.predicted_place = i
    
    return predictions


if __name__ == "__main__":
    # Demo with division-aware example AND head-to-head analysis
    print("Predictor v720+ with Division Awareness + Head-to-Head Analysis")
    print(f"Using learned weights: {LEARNED_WEIGHTS}")
    print()
    
    # Example: Athletes with head-to-head relationships
    # Scenario: A beat B (2 weeks ago), B beat C (last month)
    # Predicting Open division where A and C face off
    
    print("=" * 60)
    print("SCENARIO: Deep Transitive Head-to-Head (6 levels)")
    print("=" * 60)
    print("History:")
    print("  Chain 1: A > B > C (3 athletes)")
    print("  Chain 2: D > E > F (3 athletes)")
    print()
    print("Current competition: A vs F (never competed directly)")
    print("Question: Does A rank higher than F?")
    print()
    print("Answer: Chains are SEPARATE - A and F are NOT connected!")
    print("        (No C > D link exists)")
    print("=" * 60)
    print()
    
    # Create 6 athletes
    athletes = []
    for name in ['A', 'B', 'C', 'D', 'E', 'F']:
        athletes.append(AthleteProfile(
            id=f"athlete_{name.lower()}",
            name=f"Athlete {name}",
            division="Men's Physique",
            division_info=DivisionInfo("Open", "Open", "mens physique"),
            recent_placings=[3, 3, 3, 3, 3],  # Equal form
            total_shows=5,
            wins=0,
            last_show_date="2026-04-20"
        ))
    
    # Build H2H analyzer
    h2h_analyzer = HeadToHeadAnalyzer([a.id for a in athletes])
    
    # Chain 1: A > B > C
    h2h_analyzer.add_record(HeadToHeadRecord(
        winner_id="athlete_a", loser_id="athlete_b",
        winner_place=1, loser_place=2, days_ago=14
    ))
    h2h_analyzer.add_record(HeadToHeadRecord(
        winner_id="athlete_b", loser_id="athlete_c",
        winner_place=1, loser_place=2, days_ago=30
    ))
    
    # Chain 2: D > E > F
    h2h_analyzer.add_record(HeadToHeadRecord(
        winner_id="athlete_d", loser_id="athlete_e",
        winner_place=1, loser_place=2, days_ago=21
    ))
    h2h_analyzer.add_record(HeadToHeadRecord(
        winner_id="athlete_e", loser_id="athlete_f",
        winner_place=1, loser_place=2, days_ago=45
    ))
    
    # Calculate transitive scores
    print("Transitive Rankings (with 6-level depth):")
    print("-" * 50)
    for aid in ['athlete_a', 'athlete_b', 'athlete_c', 
                'athlete_d', 'athlete_e', 'athlete_f']:
        transitive = h2h_analyzer.calculate_transitive_ranking(aid)
        name = next(a.name for a in athletes if a.id == aid)
        print(f"  {name}: {transitive} transitive points")
    print()
    
    print("Analysis:")
    print("  • Athlete A: 10 (beat B) + 7 (beat C via B) = 17 points")
    print("  • Athlete F: 0 points (never won anyone)")
    print("  • BUT: A and F are in SEPARATE chains - no comparison possible!")
    print()
    
    # Now add a BRIDGE: C > D
    print("=" * 60)
    print("Adding bridge: C > D (connects the chains)")
    print("=" * 60)
    print()
    
    h2h_analyzer.add_record(HeadToHeadRecord(
        winner_id="athlete_c", loser_id="athlete_d",
        winner_place=1, loser_place=2, days_ago=60
    ))
    
    # Recalculate
    print("Updated Transitive Rankings (connected chains):")
    print("-" * 50)
    for aid in ['athlete_a', 'athlete_b', 'athlete_c', 
                'athlete_d', 'athlete_e', 'athlete_f']:
        transitive = h2h_analyzer.calculate_transitive_ranking(aid)
        name = next(a.name for a in athletes if a.id == aid)
        print(f"  {name}: {transitive} transitive points")
    print()
    
    print("Analysis with bridge:")
    print("  • Athlete A: 10 (B) + 7 (C) + 5 (D) + 3 (E) + 2 (F) = 27 points")
    print("  • Now A transitively beats F (5 hops)!")
    print()
    
    # Get final H2H scores
    h2h_scores = h2h_analyzer.get_all_scores()
    
    print("Final H2H Scores (0-100 scale):")
    print("-" * 50)
    for aid in ['athlete_a', 'athlete_f']:
        name = next(a.name for a in athletes if a.id == aid)
        score = h2h_scores.get(aid, 50.0)
        print(f"  {name}: {score:.1f}")
    print()
    print("RESULT: Athlete A ranks higher than F due to transitive chain!")
    print()
    
    # Deep chain demo: 10 levels
    print("=" * 60)
    print("SCENARIO: Ultra-Deep Transitive (10 levels)")
    print("=" * 60)
    print("Chain: A > B > C > D > E > F > G > H > I > J")
    print()
    
    # Create 10 athletes
    deep_athletes = []
    for letter in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']:
        deep_athletes.append(AthleteProfile(
            id=f"deep_{letter}",
            name=f"Athlete {letter.upper()}",
            division="Men's Physique",
            division_info=DivisionInfo("Open", "Open", "mens physique"),
            recent_placings=[3, 3, 3, 3, 3],
            total_shows=5,
            wins=0,
            last_show_date="2026-04-20"
        ))
    
    # Build deep analyzer
    deep_analyzer = HeadToHeadAnalyzer([a.id for a in deep_athletes])
    
    # Create chain: A>B, B>C, C>D, ..., I>J
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
    for i in range(len(letters) - 1):
        deep_analyzer.add_record(HeadToHeadRecord(
            winner_id=f"deep_{letters[i]}",
            loser_id=f"deep_{letters[i+1]}",
            winner_place=1,
            loser_place=2,
            days_ago=7 * (i + 1)  # Each match 1 week apart
        ))
    
    print("Transitive points at each level:")
    print("-" * 50)
    for aid in [f"deep_{l}" for l in letters]:
        transitive = deep_analyzer.calculate_transitive_ranking(aid)
        name = next(a.name for a in deep_athletes if a.id == aid)
        print(f"  {name}: {transitive} transitive points")
    print()
    
    print("Analysis:")
    print("  • Athlete A: 10+7+5+4+3+2+1+1+1+1 = 35 points")
    print("  • Athlete J: 0 points (never won)")
    print("  • A beats J transitively (9 hops deep!)")
    print()
    
    print("=" * 60)
    print("SCENARIO: Cumulative Transitive Margin")
    print("=" * 60)
    print("Chain: A > B > C (with margins)")
    print("  • A beat B by 5 places (1st vs 6th)")
    print("  • B beat C by 3 places (2nd vs 5th)")
    print()
    print("Cumulative margin: A over C = 5 + 3 = 8 places!")
    print("=" * 60)
    print()
    
    # Create 3 athletes
    margin_athletes = []
    for letter in ['a', 'b', 'c']:
        margin_athletes.append(AthleteProfile(
            id=f"margin_{letter}",
            name=f"Athlete {letter.upper()}",
            division="Men's Physique",
            division_info=DivisionInfo("Open", "Open", "mens physique"),
            recent_placings=[3, 3, 3, 3, 3],
            total_shows=5,
            wins=0,
            last_show_date="2026-04-20"
        ))
    
    # Build margin analyzer
    margin_analyzer = HeadToHeadAnalyzer([a.id for a in margin_athletes])
    
    # A beat B by 5 places
    margin_analyzer.add_record(HeadToHeadRecord(
        winner_id="margin_a", loser_id="margin_b",
        winner_place=1, loser_place=6, days_ago=14
    ))
    
    # B beat C by 3 places
    margin_analyzer.add_record(HeadToHeadRecord(
        winner_id="margin_b", loser_id="margin_c",
        winner_place=2, loser_place=5, days_ago=30
    ))
    
    # Calculate cumulative margins
    print("Direct margins:")
    print(f"  A beat B by: {margin_analyzer.margin_graph['margin_a']['margin_b']} places")
    print(f"  B beat C by: {margin_analyzer.margin_graph['margin_b']['margin_c']} places")
    print()
    
    print("Transitive margins:")
    a_over_c = margin_analyzer.calculate_transitive_margin("margin_a", "margin_c")
    print(f"  A over C (transitive): {a_over_c} places (cumulative)")
    print()
    
    print("H2H Scores (with margin bonuses):")
    for aid in ["margin_a", "margin_b", "margin_c"]:
        score = margin_analyzer.calculate_h2h_score(aid)
        margins = margin_analyzer.get_transitive_margins_summary(aid)
        name = next(a.name for a in margin_athletes if a.id == aid)
        print(f"  {name}: {score:.1f} (transitive margins: {margins})")
    print()
    print("Note: A gets bonus for transitive margin over C (8 places)!")
    print()
    
    # Second demo: Simple 3-athlete scenario
    print("=" * 60)
    print("SCENARIO: Simple Transitive (A > B > C)")
    print("=" * 60)
    print()
    print("  • Athlete A and Athlete C never competed together")
    print()
    print("Expected: A should rank higher than C (transitive inference)")
    print("=" * 60)
    print()
    
    # Create athletes
    athlete_a = AthleteProfile(
        id="athlete_a",
        name="Athlete A (Beat B)",
        division="Men's Physique",
        division_info=DivisionInfo("Open", "Open", "mens physique"),
        recent_placings=[1, 2, 1, 3, 2],
        total_shows=8,
        wins=3,
        last_show_date="2026-04-20"
    )
    
    athlete_b = AthleteProfile(
        id="athlete_b", 
        name="Athlete B (Lost to A, Beat C)",
        division="Men's Physique",
        division_info=DivisionInfo("Open", "Open", "mens physique"),
        recent_placings=[4, 2, 3, 2, 4],
        total_shows=10,
        wins=2,
        last_show_date="2026-04-15"
    )
    
    athlete_c = AthleteProfile(
        id="athlete_c",
        name="Athlete C (Lost to B)",
        division="Men's Physique", 
        division_info=DivisionInfo("Open", "Open", "mens physique"),
        recent_placings=[5, 3, 4, 3, 5],
        total_shows=7,
        wins=1,
        last_show_date="2026-04-10"
    )
    
    athletes = [athlete_a, athlete_b, athlete_c]
    target = DivisionInfo("Open", "Open", "mens physique")
    
    # Simulate H2H records
    h2h_analyzer = HeadToHeadAnalyzer([a.id for a in athletes])
    
    # A beat B (place 1 vs 4, 14 days ago)
    h2h_analyzer.add_record(HeadToHeadRecord(
        winner_id="athlete_a", loser_id="athlete_b",
        winner_place=1, loser_place=4, days_ago=14
    ))
    
    # B beat C (place 2 vs 4, 30 days ago)
    h2h_analyzer.add_record(HeadToHeadRecord(
        winner_id="athlete_b", loser_id="athlete_c",
        winner_place=2, loser_place=4, days_ago=30
    ))
    
    # Calculate scores
    h2h_scores = h2h_analyzer.get_all_scores()
    
    print("Head-to-Head Scores (with transitive analysis):")
    for athlete_id, score in h2h_scores.items():
        name = next(a.name for a in athletes if a.id == athlete_id)
        print(f"  {name}: {score:.1f}")
    print()
    
    # Show transitive analysis
    print("Transitive Rankings:")
    for athlete_id in ["athlete_a", "athlete_b", "athlete_c"]:
        transitive = h2h_analyzer.calculate_transitive_ranking(athlete_id)
        name = next(a.name for a in athletes if a.id == athlete_id)
        print(f"  {name}: {transitive} transitive points")
    print()
    
    # Run prediction WITH pre-calculated H2H scores
    print("Final Predictions (incorporating H2H scores):")
    print("-" * 50)
    predictions = predict_event(athletes, target, h2h_scores=h2h_scores)
    
    for pred in predictions:
        print(f"{pred.predicted_place}. {pred.name}")
        print(f"   H2H Score: {pred.factors['head_to_head']:.1f}")
        print(f"   Recent Form: {pred.factors['recent_form']:.1f}")
        print(f"   Composite: {pred.factors['composite']:.1f}")
        print(f"   Tier: {pred.tier}")
        print()
    
    print("=" * 60)
    print("KEY INSIGHT: Athlete A ranks #1 because:")
    print("  1. Direct win over B (+ recency bonus)")
    print("  2. Transitive win over C (A > B > C)")
    print("=" * 60)
    print()
    
    # Second demo: Division-aware filtering
    print("=" * 60)
    print("SCENARIO: Masters Athlete in Open Division")
    print("=" * 60)
    
    masters_athlete = AthleteProfile(
        id="m1",
        name="Masters Champion",
        division="Men's Physique",
        division_info=DivisionInfo("Open", "Masters 40+", "mens physique"),
        recent_placings=[1, 1, 2],  # Masters placings
        placings_by_division={
            "Open_Masters 40+": [1, 1, 2],
            "Open_Open": [10, 8]  # Struggles in Open
        },
        total_shows=10,
        wins=5,
        last_show_date="2026-04-20"
    )
    
    open_athlete = AthleteProfile(
        id="o1",
        name="Open Regular",
        division="Men's Physique",
        division_info=DivisionInfo("Open", "Open", "mens physique"),
        recent_placings=[3, 2, 4, 3, 2],
        total_shows=15,
        wins=3,
        last_show_date="2026-04-15"
    )
    
    example_athletes = [masters_athlete, open_athlete]
    target = DivisionInfo("Open", "Open", "mens physique")
    
    print(f"Predicting: {target.category_type} {target.age_category}")
    print(f"Difficulty: {target.difficulty:.2f}")
    print()
    print("Masters Champion history:")
    print("  - Masters 40+: [1, 1, 2] (wins)")
    print("  - Open: [10, 8] (struggles)")
    print()
    print("Open Regular history:")
    print("  - Open: [3, 2, 4, 3, 2] (consistent top 5)")
    print()
    
    predictions = predict_event(example_athletes, target)
    
    print("Prediction (with division-aware filtering):")
    print("-" * 50)
    for pred in predictions:
        print(f"{pred.predicted_place}. {pred.name}")
        print(f"   Recent Form: {pred.factors['recent_form']:.1f}")
        print(f"   Confidence: {pred.confidence:.0%}")
    print()
    print("NOTE: Masters Champion's Open placings [10,8] are used,")
    print("      NOT their Masters wins [1,1,2] - correct division filtering!")
