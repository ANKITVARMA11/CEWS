"""Finding the source records behind an insight.

Every insight must point at real records, never at a number alone: an insight with no evidence
is refused before it is ever stored (see ``rule_engine.py``). This module only reads
``record_topics`` / ``record_organizations``; it never computes anything.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

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
    topic_id: int | None = None,
    published_after: date | None = None,
    limit: int = DEFAULT_EVIDENCE_LIMIT,
) -> list[int]:
    """The most recent source record ids linked to a topic or competitor.

    Args:
        source_type: restrict to one kind of record (e.g. ``"patent"``), for an insight that is
            specifically about one source. Omit for any type.
        topic_id: additionally restrict to records tagged with this topic. Matters for a
            *competitor* entity whose claim is really about one specific topic (a new-market-entry
            insight says this organization entered this area) - without it, evidence for a
            competitor is every recent record linked to that organization in any topic, which can
            include records with nothing to do with the claim the insight is actually making, and
            is unfalsifiable from the evidence alone precisely because nothing in it says which
            topic each record belongs to. Confirmed as a real gap by someone fact-checking a real
            insight: 2 of 5 evidence records had no connection to the topic the insight claimed,
            and there was no way to tell which 2 from the evidence fields alone. Redundant (but
            harmless) when ``entity_type`` is already ``"topic"``, since that is already an
            implicit topic filter on ``entity_id`` itself.
        published_after: additionally restrict to records published on or after this date -
            for an insight whose claim is itself about a specific recent window (a
            new-market-entry's "N records in the last ``window_months``"), so the evidence shown
            cannot include records from further back than what was actually counted. Without
            this, evidence for such a claim could include - and, found on a real insight, did
            include - records years older than the window the count describes, making the count
            look wrong (more evidence shown than the claimed number) even though the count itself
            was correct for the window it was measuring.
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
        if topic_id is not None:
            query = query.join(RecordTopic, RecordTopic.source_record_id == SourceRecord.id).where(
                RecordTopic.topic_id == topic_id
            )
    else:
        raise ValueError(f"unknown entity_type {entity_type!r}")

    if source_type is not None:
        query = query.where(SourceRecord.record_type == source_type)
    if published_after is not None:
        # SourceRecord.published_at is timezone-aware; a bare date has no tzinfo at all and the
        # column's own bind-param check rejects it outright rather than silently assuming UTC.
        query = query.where(
            SourceRecord.published_at
            >= datetime(
                published_after.year, published_after.month, published_after.day, tzinfo=UTC
            )
        )
    query = query.where(SourceRecord.duplicate_of_id.is_(None))
    query = query.order_by(SourceRecord.published_at.desc()).limit(limit)
    return [int(record_id) for record_id in session.scalars(query)]
