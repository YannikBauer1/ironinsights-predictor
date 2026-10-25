"""The chosen v2 configuration (see REPORT.md for how it was selected)."""
from __future__ import annotations

from .blend import MASTERS_COLUMNS, OPEN_COLUMNS, SplitBlend
from .data import age_group_pool, masters_split_pool
from .features import FeatureRecorder
from .models import Elo
from .plfit import TimeDecayPL

# Contest weights in the Plackett-Luce views. Field strength already makes a result against a strong
# field count for more; on top of that the Olympia counts 1.5x and the Masters Olympia 2/3x. Raising
# tier-2 shows (Arnold, NY Pro, Pittsburgh, Romania) above 1.0 did not improve predictions here; the
# Elo view and the stage-2 tier context still treat them separately.
TIER_W = {"O": 1.5, "MO": 2 / 3}
PL = dict(tau=730, lam=0.3, reverse=True, rookie=-0.2, tier_w=TIER_W)
# The reigning Olympia champion leads until someone beats them: +0.5 on their Open-age utility while
# unbeaten. Learned from data the edge is about +0.3; 0.5 made every unbeaten champion the favourite at
# the 49 Olympias of 2021-2026, with log-loss within noise, though unbeaten champions now win slightly
# less often than predicted (experiments/champion_weight.py). A heavier Olympia in TIER_W also lifts the
# runners-up and made predictions worse; a bonus for beaten champions too overrated them.
CHAMPION_BONUS = 0.5


def recorder() -> FeatureRecorder:
    """The eight rating views (main one first) wrapped in the stage-2 feature recorder."""
    return FeatureRecorder(
        TimeDecayPL(**PL, masters_w=2 / 3),
        extra={
            "pl_msep": TimeDecayPL(**PL, pool_fn=masters_split_pool),
            "pl_fwd": TimeDecayPL(**dict(PL, reverse=False), masters_w=2 / 3),
            "pl_short": TimeDecayPL(**dict(PL, tau=300), masters_w=2 / 3),
            "pl_long": TimeDecayPL(**dict(PL, tau=1500), window_years=10, masters_w=2 / 3),
            "elo_msep": Elo(pool_fn=masters_split_pool),
            "pl_age": TimeDecayPL(**PL, pool_fn=age_group_pool),
            "elo_age": Elo(pool_fn=age_group_pool),
        })


def make_blend(names: list[str]) -> SplitBlend:
    """The stage-2 ranking layer. Both layers also get time away (round 3); masters classes also get the
    age-group, head-to-head and qualification inputs (round 2) and travel (round 3)."""
    return SplitBlend(names, columns=OPEN_COLUMNS, masters_columns=MASTERS_COLUMNS, champion_bonus=CHAMPION_BONUS)
