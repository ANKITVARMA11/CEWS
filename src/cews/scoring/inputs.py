"""Reading the stored features that scores are built from.

Scoring works from ``feature_values``, not from the feature pass in memory, so a score can be
recomputed at any time from what is on disk, and so the same inputs can be shown next to the
score in the evidence view.

For each entity this gathers the raw and normalized feature values, how much evidence there was,
which source types actually had activity, how fresh the underlying records are, and how far the
values moved since the previous run (which is what model stability means).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from cews.constants import EntityType, SourceType
from cews.database.models import (
    FeatureValue,
    Organization,
    RecordOrganization,
    RecordTopic,
    SourceRecord,
    Topic,
)
from cews.features.sample_confidence import calculate_data_freshness

LOGGER = logging.getLogger(__name__)

STABILITY_FEATURES = ("velocity", "momentum", "consistency")
ACTIVITY_PREFIX = "activity_"


@dataclass
class EntityInputs:
    """Everything one entity brings to scoring."""

    entity_type: str
    entity_id: int
    name: str
    raw: dict[str, float] = field(default_factory=dict)
    normalized: dict[str, float] = field(default_factory=dict)
    sample_size: float = 0.0
    last_fetched_at: datetime | None = None
    previous_normalized: dict[str, float] = field(default_factory=dict)

    def value(self, feature: str, default: float | None = None) -> float | None:
        """The normalized 0-100 value of one feature, or ``default`` when it is missing."""
        return self.normalized.get(feature, default)

    def raw_value(self, feature: str, default: float | None = None) -> float | None:
        """The unnormalized value of one feature."""
        return self.raw.get(feature, default)

    @property
    def available_sources(self) -> tuple[str, ...]:
        """Source types that produced activity for this entity in the window.

        Only real source types count; the stored ``activity_total`` is a roll-up, not a source.
        """
        known = {source.value for source in SourceType}
        return tuple(
            name[len(ACTIVITY_PREFIX) :]
            for name, value in sorted(self.raw.items())
            if name.startswith(ACTIVITY_PREFIX)
            and value > 0
            and name[len(ACTIVITY_PREFIX) :] in known
        )

    @property
    def supporting_sources(self) -> int:
        """How many independent source types were growing."""
        return int(self.raw.get("supporting_sources", 0))

    @property
    def single_spike(self) -> bool:
        """True when most of the activity landed in one month and growth was not sustained."""
        return bool(self.raw.get("single_spike", 0.0) >= 1.0)

    def freshness(self, *, as_of: datetime | None = None, half_life_days: float = 30.0) -> float:
        """How current the records behind this entity are, from 0 to 1."""
        return calculate_data_freshness(
            self.last_fetched_at, as_of=as_of, half_life_days=half_life_days
        )

    def stability(self) -> float | None:
        """How steady the headline features are against the previous run, from 0 to 1.

        None when there is no earlier run to compare with, which is different from "unstable"
        and is treated as an unavailable input rather than a zero.
        """
        moves = [
            abs(self.normalized[name] - self.previous_normalized[name]) / 100.0
            for name in STABILITY_FEATURES
            if name in self.normalized and name in self.previous_normalized
        ]
        if not moves:
            return None
        return max(0.0, 1.0 - sum(moves) / len(moves))


def latest_feature_date(session: Session, *, on_or_before: date | None = None) -> date | None:
    """The most recent date features were computed for."""
    query = select(func.max(FeatureValue.feature_date))
    if on_or_before is not None:
        query = query.where(FeatureValue.feature_date <= on_or_before)
    return session.scalar(query)


def previous_feature_date(session: Session, feature_date: date) -> date | None:
    """The feature date before ``feature_date``, if the pass has ever run twice."""
    return session.scalar(
        select(func.max(FeatureValue.feature_date)).where(FeatureValue.feature_date < feature_date)
    )


def _names(session: Session) -> dict[tuple[str, int], str]:
    names: dict[tuple[str, int], str] = {}
    for topic in session.execute(select(Topic.id, Topic.canonical_name)).all():
        names[(EntityType.TOPIC.value, int(topic.id))] = str(topic.canonical_name)
    for organization in session.execute(select(Organization.id, Organization.canonical_name)).all():
        names[(EntityType.COMPETITOR.value, int(organization.id))] = str(
            organization.canonical_name
        )
    return names


def _last_fetched(session: Session) -> dict[tuple[str, int], datetime]:
    """The newest collection time per entity, used for freshness."""
    stamps: dict[tuple[str, int], datetime] = {}
    topic_rows = session.execute(
        select(RecordTopic.topic_id, func.max(SourceRecord.fetched_at))
        .join(SourceRecord, SourceRecord.id == RecordTopic.source_record_id)
        .group_by(RecordTopic.topic_id)
    ).all()
    for topic_id, stamp in topic_rows:
        if stamp is not None:
            stamps[(EntityType.TOPIC.value, int(topic_id))] = stamp
    organization_rows = session.execute(
        select(RecordOrganization.organization_id, func.max(SourceRecord.fetched_at))
        .join(SourceRecord, SourceRecord.id == RecordOrganization.source_record_id)
        .group_by(RecordOrganization.organization_id)
    ).all()
    for organization_id, stamp in organization_rows:
        if stamp is not None:
            stamps[(EntityType.COMPETITOR.value, int(organization_id))] = stamp
    return stamps


def load_entity_inputs(
    session: Session,
    feature_date: date,
    *,
    entity_types: Sequence[str] | None = None,
) -> dict[tuple[str, int], EntityInputs]:
    """Load every entity's features for one date, with the previous run for comparison.

    Returns a mapping of ``(entity_type, entity_id)`` to :class:`EntityInputs`.
    """
    wanted = set(entity_types) if entity_types else None
    names = _names(session)
    stamps = _last_fetched(session)
    entities: dict[tuple[str, int], EntityInputs] = {}

    rows = session.scalars(select(FeatureValue).where(FeatureValue.feature_date == feature_date))
    for row in rows:
        key = (row.entity_type, row.entity_id)
        if wanted is not None and row.entity_type not in wanted:
            continue
        entity = entities.get(key)
        if entity is None:
            entity = EntityInputs(
                entity_type=row.entity_type,
                entity_id=row.entity_id,
                name=names.get(key, f"{row.entity_type} {row.entity_id}"),
                last_fetched_at=stamps.get(key),
            )
            entities[key] = entity
        if row.raw_value is not None:
            entity.raw[row.feature_name] = float(row.raw_value)
        if row.normalized_value is not None:
            entity.normalized[row.feature_name] = float(row.normalized_value)
        entity.sample_size = max(entity.sample_size, float(row.sample_size or 0))

    earlier = previous_feature_date(session, feature_date)
    if earlier is not None:
        for row in session.scalars(
            select(FeatureValue).where(
                FeatureValue.feature_date == earlier,
                FeatureValue.feature_name.in_(STABILITY_FEATURES),
            )
        ):
            entity = entities.get((row.entity_type, row.entity_id))
            if entity is not None and row.normalized_value is not None:
                entity.previous_normalized[row.feature_name] = float(row.normalized_value)

    LOGGER.debug("loaded %d entities of features for %s", len(entities), feature_date)
    return entities


def source_growth(inputs: EntityInputs) -> dict[str, float | None]:
    """Growth per source type, with None where that source had no activity."""
    available = set(inputs.available_sources)
    return {
        source.value: (
            inputs.raw.get(f"growth_{source.value}") if source.value in available else None
        )
        for source in SourceType
    }


def as_evidence(inputs: EntityInputs) -> Mapping[str, Any]:
    """A compact record of the inputs, stored with the score."""
    return {
        "feature_values": {name: round(value, 4) for name, value in sorted(inputs.raw.items())},
        "normalized": {name: round(value, 2) for name, value in sorted(inputs.normalized.items())},
        "sample_size": round(inputs.sample_size, 3),
        "available_sources": list(inputs.available_sources),
        "last_collected": (
            inputs.last_fetched_at.astimezone(UTC).isoformat() if inputs.last_fetched_at else None
        ),
    }
