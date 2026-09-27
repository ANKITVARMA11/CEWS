"""Threat scores within a therapeutic area.

A competitor that looks quiet overall can be moving hard in one field, and that is usually the
thing worth knowing. This builds a small set of features for each competitor-and-area pair
directly from the stored monthly activity, ranks each growth measure among the other pairs in
the same area, and scores them with the ordinary threat weights.

Areas with almost no activity are skipped rather than scored: ranking three records against
three other records produces a confident-looking number about nothing.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from cews.constants import EntityType, SourceType
from cews.database.models import Organization, TherapeuticArea, Topic
from cews.features.activity_counts import monthly_series
from cews.features.growth import growth_from_series
from cews.features.time_windows import month_range
from cews.scoring.config import ScoringConfig
from cews.scoring.inputs import EntityInputs
from cews.scoring.normalization import normalize_feature_cohort

LOGGER = logging.getLogger(__name__)

WINDOW_MONTHS = 12
GROWTH_SOURCES: tuple[str, ...] = (
    SourceType.CLINICAL_TRIAL.value,
    SourceType.PATENT.value,
    SourceType.PUBLICATION.value,
)
MIN_AREA_RECORDS = 10.0


@dataclass(frozen=True)
class AreaPair:
    """One competitor's activity inside one therapeutic area."""

    organization_id: int
    organization_name: str
    area_key: str
    area_name: str
    topic_ids: tuple[int, ...]
    activity: Mapping[str, float]
    growth: Mapping[str, float]
    records: float


def area_topics(session: Session) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Every therapeutic area, with the topic ids that belong to it.

    An area's own topic row counts too, so records tagged only with the area still appear.
    """
    areas: dict[str, tuple[str, list[int]]] = {}
    for area in session.scalars(select(TherapeuticArea)):
        areas[area.key] = (area.canonical_name, [])
    for topic in session.scalars(select(Topic).where(Topic.therapeutic_area_id.is_not(None))):
        owner = session.get(TherapeuticArea, topic.therapeutic_area_id)
        if owner is not None and owner.key in areas:
            areas[owner.key][1].append(topic.id)
    return {key: (name, tuple(sorted(ids))) for key, (name, ids) in areas.items() if ids}


def collect_area_pairs(
    session: Session,
    competitors: Sequence[Organization],
    *,
    as_of: date,
    window_months: int = WINDOW_MONTHS,
    minimum_records: float = MIN_AREA_RECORDS,
) -> list[AreaPair]:
    """Build the competitor-and-area pairs worth scoring."""
    months = month_range(as_of, window_months)
    areas = area_topics(session)
    pairs: list[AreaPair] = []
    for organization in competitors:
        for area_key, (area_name, topic_ids) in areas.items():
            totals: dict[str, list[float]] = {
                source: [0.0] * len(months) for source in GROWTH_SOURCES
            }
            for topic_id in topic_ids:
                series = monthly_series(
                    session,
                    months,
                    organization_id=organization.id,
                    topic_id=topic_id,
                    source_types=GROWTH_SOURCES,
                )
                for source, values in series.items():
                    running = totals[source]
                    for index, value in enumerate(values):
                        running[index] += value
            records = sum(sum(values) for values in totals.values())
            if records < minimum_records:
                continue
            pairs.append(
                AreaPair(
                    organization_id=organization.id,
                    organization_name=organization.canonical_name,
                    area_key=area_key,
                    area_name=area_name,
                    topic_ids=topic_ids,
                    activity={source: sum(values) for source, values in totals.items()},
                    growth={
                        source: growth_from_series(values).log_growth
                        for source, values in totals.items()
                    },
                    records=records,
                )
            )
    LOGGER.debug("built %d competitor/area pairs", len(pairs))
    return pairs


def area_inputs(
    pairs: Sequence[AreaPair], config: ScoringConfig, feature_date: date
) -> dict[tuple[int, str], EntityInputs]:
    """Normalize each growth measure within its area and return scoring inputs per pair.

    Each pair is ranked against the other competitors **in the same area**, so a busy area and a
    quiet one are never compared directly.
    """
    by_area: dict[str, list[AreaPair]] = {}
    for pair in pairs:
        by_area.setdefault(pair.area_key, []).append(pair)

    inputs: dict[tuple[int, str], EntityInputs] = {}
    for area_key, members in by_area.items():
        for source in GROWTH_SOURCES:
            cohort = f"competitor/{area_key}/growth_{source}/{feature_date.isoformat()}"
            values = {str(pair.organization_id): pair.growth[source] for pair in members}
            normalized = normalize_feature_cohort(values, method="percentile", cohort=cohort)
            for pair in members:
                key = (pair.organization_id, area_key)
                entry = inputs.get(key)
                if entry is None:
                    entry = EntityInputs(
                        entity_type=EntityType.COMPETITOR.value,
                        entity_id=pair.organization_id,
                        name=pair.organization_name,
                        sample_size=pair.records,
                    )
                    entry.raw.update(
                        {f"activity_{name}": value for name, value in pair.activity.items()}
                    )
                    inputs[key] = entry
                entry.raw[f"growth_{source}"] = pair.growth[source]
                entry.normalized[f"growth_{source}"] = normalized[
                    str(pair.organization_id)
                ].normalized_value
    return inputs
