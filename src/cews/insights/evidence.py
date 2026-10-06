"""Finding the source records behind an insight.

Every insight must point at real records, never at a number alone: an insight with no evidence
is refused before it is ever stored (see ``rule_engine.py``). This module only reads
``record_topics`` / ``record_organizations``; it never computes anything.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from cews.constants import EntityType
from cews.database.models import RecordOrganization, RecordTopic, SourceRecord

DEFAULT_EVIDENCE_LIMIT = 10


def record_ids_for_entity(
    session: Session,
    entity_type: str,
    entity_id: int,
    *,
    source_type: str | None = None,
    limit: int = DEFAULT_EVIDENCE_LIMIT,
) -> list[int]:
    """The most recent source record ids linked to a topic or competitor.

    Args:
        source_type: restrict to one kind of record (e.g. ``"patent"``), for an insight that is
            specifically about one source. Omit for any type.
        limit: how many to return; an insight only needs enough to look verifiable, not every
            record behind the score.

    Raises:
        ValueError: for an unknown entity_type.
    """
    if entity_type == EntityType.TOPIC.value:
        query = (
            select(SourceRecord.id)
            .join(RecordTopic, RecordTopic.source_record_id == SourceRecord.id)
            .where(RecordTopic.topic_id == entity_id)
        )
    elif entity_type == EntityType.COMPETITOR.value:
        query = (
            select(SourceRecord.id)
            .join(RecordOrganization, RecordOrganization.source_record_id == SourceRecord.id)
            .where(RecordOrganization.organization_id == entity_id)
        )
    else:
        raise ValueError(f"unknown entity_type {entity_type!r}")

    if source_type is not None:
        query = query.where(SourceRecord.record_type == source_type)
    query = query.where(SourceRecord.duplicate_of_id.is_(None))
    query = query.order_by(SourceRecord.published_at.desc()).limit(limit)
    return [int(record_id) for record_id in session.scalars(query)]
