"""Source agreement: how many independent sources tell the same story.

A rise seen in trials, patents and publications at once is worth far more than the same rise in
one source, which is just as likely to be a collection artefact. Agreement is the share of the
source types that actually have data which are also growing.

Only source types with data are counted. A disabled or empty source neither supports nor
contradicts a trend, so it is reported as unavailable rather than counted as disagreement.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

MIN_SUPPORTING_SOURCES = 2


@dataclass(frozen=True)
class SourceAgreement:
    """Which sources support a trend, and which do not."""

    agreement: float
    supporting: tuple[str, ...]
    contradicting: tuple[str, ...]
    flat: tuple[str, ...]
    unavailable: tuple[str, ...]

    @property
    def available_count(self) -> int:
        """How many source types had data to judge."""
        return len(self.supporting) + len(self.contradicting) + len(self.flat)

    @property
    def multi_source(self) -> bool:
        """True when at least two independent sources support the trend."""
        return len(self.supporting) >= MIN_SUPPORTING_SOURCES

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form for score components and evidence."""
        return {
            "agreement": round(self.agreement, 4),
            "supporting": list(self.supporting),
            "contradicting": list(self.contradicting),
            "flat": list(self.flat),
            "unavailable": list(self.unavailable),
            "available_count": self.available_count,
            "multi_source": self.multi_source,
        }


def calculate_source_agreement(
    growth_by_source: Mapping[str, float | None], *, threshold: float = 0.0
) -> SourceAgreement:
    """Agreement across source types.

    Args:
        growth_by_source: growth value per source type; None means the source had no data.
        threshold: growth above this counts as support (0 by default: any increase).

    Raises:
        ValueError: if a growth value is not finite.
    """
    supporting: list[str] = []
    contradicting: list[str] = []
    flat: list[str] = []
    unavailable: list[str] = []
    for source in sorted(growth_by_source):
        value = growth_by_source[source]
        if value is None:
            unavailable.append(source)
            continue
        if not math.isfinite(value):
            raise ValueError(f"growth for {source!r} must be a finite number")
        if value > threshold:
            supporting.append(source)
        elif value < -threshold or value < 0:
            contradicting.append(source)
        else:
            flat.append(source)

    available = len(supporting) + len(contradicting) + len(flat)
    agreement = len(supporting) / available if available else 0.0
    return SourceAgreement(
        agreement=agreement,
        supporting=tuple(supporting),
        contradicting=tuple(contradicting),
        flat=tuple(flat),
        unavailable=tuple(unavailable),
    )
