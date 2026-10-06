"""Unit tests for finding the source records behind an insight."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from sqlalchemy.orm import Session, sessionmaker

from cews.database.connection import create_memory_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import Organization, RecordOrganization, RecordTopic, SourceRecord, Topic
from cews.insights.evidence import record_ids_for_entity

pytestmark = pytest.mark.unit


@pytest.fixture
def factory() -> Iterator[sessionmaker[Session]]:
    engine = create_memory_engine()
    upgrade_database(engine)
    yield create_session_factory(engine)
    engine.dispose()


def _record(
    session: Session, identifier: str, record_type: str, *, published: datetime
) -> SourceRecord:
    record = SourceRecord(
        source="test",
        source_record_id=identifier,
        record_type=record_type,
        fetched_at=published,
        published_at=published,
        content_hash=f"{identifier:0>64}"[:64],
    )
    session.add(record)
    session.flush()
    return record


def test_finds_records_linked_to_a_topic(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        topic = Topic(key="t", canonical_name="Topic", topic_type="technology")
        session.add(topic)
        session.flush()
        record = _record(session, "r1", "publication", published=datetime(2026, 1, 1, tzinfo=UTC))
        session.add(RecordTopic(source_record_id=record.id, topic_id=topic.id, confidence=1.0))
        session.flush()
        ids = record_ids_for_entity(session, "topic", topic.id)
    assert ids == [record.id]


def test_finds_records_linked_to_a_competitor(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        organization = Organization(
            canonical_name="Example Pharma", normalized_name="example pharma"
        )
        session.add(organization)
        session.flush()
        record = _record(session, "r1", "patent", published=datetime(2026, 1, 1, tzinfo=UTC))
        session.add(
            RecordOrganization(
                source_record_id=record.id,
                organization_id=organization.id,
                relationship_type="sponsor",
                confidence=1.0,
            )
        )
        session.flush()
        ids = record_ids_for_entity(session, "competitor", organization.id)
    assert ids == [record.id]


def test_newest_records_come_first(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        topic = Topic(key="t", canonical_name="Topic", topic_type="technology")
        session.add(topic)
        session.flush()
        older = _record(session, "old", "publication", published=datetime(2025, 1, 1, tzinfo=UTC))
        newer = _record(session, "new", "publication", published=datetime(2026, 6, 1, tzinfo=UTC))
        for record in (older, newer):
            session.add(RecordTopic(source_record_id=record.id, topic_id=topic.id, confidence=1.0))
        session.flush()
        ids = record_ids_for_entity(session, "topic", topic.id)
    assert ids == [newer.id, older.id]


def test_can_restrict_to_one_source_type(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        topic = Topic(key="t", canonical_name="Topic", topic_type="technology")
        session.add(topic)
        session.flush()
        patent = _record(session, "p", "patent", published=datetime(2026, 1, 1, tzinfo=UTC))
        paper = _record(session, "pub", "publication", published=datetime(2026, 1, 2, tzinfo=UTC))
        for record in (patent, paper):
            session.add(RecordTopic(source_record_id=record.id, topic_id=topic.id, confidence=1.0))
        session.flush()
        ids = record_ids_for_entity(session, "topic", topic.id, source_type="patent")
    assert ids == [patent.id]


def test_a_duplicate_record_is_not_offered_as_evidence(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        topic = Topic(key="t", canonical_name="Topic", topic_type="technology")
        session.add(topic)
        session.flush()
        original = _record(
            session, "orig", "publication", published=datetime(2026, 1, 1, tzinfo=UTC)
        )
        session.flush()
        duplicate = _record(
            session, "dup", "publication", published=datetime(2026, 1, 2, tzinfo=UTC)
        )
        duplicate.duplicate_of_id = original.id
        for record in (original, duplicate):
            session.add(RecordTopic(source_record_id=record.id, topic_id=topic.id, confidence=1.0))
        session.flush()
        ids = record_ids_for_entity(session, "topic", topic.id)
    assert ids == [original.id]


def test_respects_the_limit(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        topic = Topic(key="t", canonical_name="Topic", topic_type="technology")
        session.add(topic)
        session.flush()
        for index in range(5):
            record = _record(
                session,
                f"r{index}",
                "publication",
                published=datetime(2026, 1, index + 1, tzinfo=UTC),
            )
            session.add(RecordTopic(source_record_id=record.id, topic_id=topic.id, confidence=1.0))
        session.flush()
        ids = record_ids_for_entity(session, "topic", topic.id, limit=2)
    assert len(ids) == 2


def test_no_matching_records_is_an_empty_list_not_an_error(
    factory: sessionmaker[Session],
) -> None:
    with session_scope(factory) as session:
        assert record_ids_for_entity(session, "topic", 999999) == []


def test_an_unknown_entity_type_is_refused(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session, pytest.raises(ValueError, match="unknown entity_type"):
        record_ids_for_entity(session, "not-a-real-type", 1)
