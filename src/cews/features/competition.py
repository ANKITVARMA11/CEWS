"""Competition density: how crowded a topic is.

Three numbers, because "crowded" means different things: how many organizations are active at
all, how concentrated the activity is among them (the Herfindahl-Hirschman Index, 0 for a
fragmented field and 1 for a monopoly), and a 0-100 density that combines the two for the
opportunity score. A topic with ten equal players is crowded; a topic with ten players where one
holds 90% of the activity is effectively one player's field.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

DEFAULT_MIN_ACTIVITY = 1.0
DEFAULT_SATURATION = 8.0


@dataclass(frozen=True)
class CompetitionDensity:
    """How many organizations are active in a topic, and how concentrated they are."""

    active_competitors: int
    counted_competitors: int
    total_activity: float
    concentration: float
    effective_competitors: float
    density: float
    leader: str | None
    leader_share: float

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form for score components and evidence."""
        return {
            "active_competitors": self.active_competitors,
            "counted_competitors": self.counted_competitors,
            "total_activity": round(self.total_activity, 3),
            "concentration_hhi": round(self.concentration, 4),
            "effective_competitors": round(self.effective_competitors, 2),
            "density": round(self.density, 2),
            "leader": self.leader,
            "leader_share": round(self.leader_share, 4),
        }


def calculate_competition_density(
    activity_by_competitor: Mapping[str, float],
    *,
    min_activity: float = DEFAULT_MIN_ACTIVITY,
    saturation: float = DEFAULT_SATURATION,
) -> CompetitionDensity:
    """Measure how crowded a topic is from each organization's activity in it.

    Args:
        activity_by_competitor: activity per organization within the topic.
        min_activity: an organization must reach this much activity to count as active, so a
            single passing mention does not make a field look crowded.
        saturation: the number of active organizations at which density approaches 100.

    Raises:
        ValueError: for negative or non-finite activity, or a non-positive saturation.
    """
    if saturation <= 0 or not math.isfinite(saturation):
        raise ValueError("saturation must be a positive, finite number")
    for name, value in activity_by_competitor.items():
        if value < 0 or not math.isfinite(value):
            raise ValueError(f"activity for {name!r} must be finite and not negative")

    counted = {name: float(v) for name, v in activity_by_competitor.items() if v >= min_activity}
    total = sum(counted.values())
    if not counted or total <= 0:
        return CompetitionDensity(0, len(activity_by_competitor), 0.0, 0.0, 0.0, 0.0, None, 0.0)

    shares = {name: value / total for name, value in counted.items()}
    concentration = sum(share**2 for share in shares.values())
    leader, leader_share = max(shares.items(), key=lambda item: (item[1], item[0]))
    # The inverse of the concentration index is the "effective" number of competitors: ten equal
    # players count as ten, but ten players where one holds most of the activity count as barely
    # more than one. Density then saturates, so the 9th rival matters less than the 2nd.
    effective = 1.0 / concentration if concentration > 0 else 0.0
    density = 100.0 * (1.0 - math.exp(-effective / saturation))
    return CompetitionDensity(
        active_competitors=len(counted),
        counted_competitors=len(activity_by_competitor),
        total_activity=total,
        concentration=concentration,
        effective_competitors=effective,
        density=max(0.0, min(100.0, density)),
        leader=leader,
        leader_share=leader_share,
    )
