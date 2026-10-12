"""Unit tests for finding the source records behind an insight."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime

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


# --------------------------------------------------------------------------------------
# topic_id: scoping a competitor's evidence to the specific topic a claim is actually about
#
# Caught fact-checking a real insight: a new_market_entry claim said an organization entered
# Oncology, but 2 of the 5 evidence records it was given had nothing to do with Oncology - they
# were just the organization's other, unrelated recent activity, included only because
# record_ids_for_entity's competitor branch had no way to filter by topic at all. Nothing in the
# evidence fields said which records were which, so the claim was unfalsifiable from what was
# shown even though the organization-level evidence itself was accurate.
# --------------------------------------------------------------------------------------
def test_a_competitors_evidence_can_be_scoped_to_one_topic(
    factory: sessionmaker[Session],
) -> None:
    with session_scope(factory) as session:
        organization = Organization(
            canonical_name="Example Pharma", normalized_name="example pharma"
        )
        oncology = Topic(key="oncology", canonical_name="Oncology", topic_type="technology")
        cardiology = Topic(key="cardiology", canonical_name="Cardiology", topic_type="technology")
        session.add_all([organization, oncology, cardiology])
        session.flush()

        for index, topic in enumerate([oncology, oncology, cardiology, cardiology, cardiology]):
            record = _record(
                session,
                f"r{index}",
                "publication",
                published=datetime(2026, 1, index + 1, tzinfo=UTC),
            )
            session.add(
                RecordOrganization(
                    source_record_id=record.id,
                    organization_id=organization.id,
                    relationship_type="affiliation",
                )
            )
            session.add(RecordTopic(source_record_id=record.id, topic_id=topic.id, confidence=1.0))
        session.flush()

        oncology_ids = {
            r.id
            for r in session.query(SourceRecord)
            .join(RecordTopic)
            .filter(RecordTopic.topic_id == oncology.id)
        }

        unscoped = record_ids_for_entity(session, "competitor", organization.id)
        scoped = record_ids_for_entity(session, "competitor", organization.id, topic_id=oncology.id)

    assert len(unscoped) == 5, "the default, unscoped behavior is unchanged for every other caller"
    assert set(scoped) == oncology_ids
    assert len(scoped) == 2


def test_topic_id_on_a_topic_entity_is_redundant_but_harmless(
    factory: sessionmaker[Session],
) -> None:
    with session_scope(factory) as session:
        topic = Topic(key="t", canonical_name="Topic", topic_type="technology")
        session.add(topic)
        session.flush()
        record = _record(session, "r1", "publication", published=datetime(2026, 1, 1, tzinfo=UTC))
        session.add(RecordTopic(source_record_id=record.id, topic_id=topic.id, confidence=1.0))
        session.flush()
        without = record_ids_for_entity(session, "topic", topic.id)
        with_redundant = record_ids_for_entity(session, "topic", topic.id, topic_id=topic.id)
    assert without == with_redundant == [record.id]


# --------------------------------------------------------------------------------------
# published_after: scoping evidence to the same window a claim's count actually describes
#
# Caught on the same real insight as the topic_id fix above: even once evidence was correctly
# scoped to the right topic, it still had no date bound - so it could include (and did include)
# records from years before the "N records in the last 6 months" window the claim's count was
# actually measuring, making the count look wrong even though it was correct for what it
# measured.
# --------------------------------------------------------------------------------------
def test_published_after_excludes_records_from_before_the_window(
    factory: sessionmaker[Session],
) -> None:
    with session_scope(factory) as session:
        organization = Organization(
            canonical_name="Example Pharma", normalized_name="example pharma"
        )
        topic = Topic(key="t", canonical_name="Topic", topic_type="technology")
        session.add_all([organization, topic])
        session.flush()

        old = _record(session, "old", "publication", published=datetime(2023, 1, 1, tzinfo=UTC))
        recent = _record(
            session, "recent", "publication", published=datetime(2026, 5, 1, tzinfo=UTC)
        )
        for record in (old, recent):
            session.add(
                RecordOrganization(
                    source_record_id=record.id,
                    organization_id=organization.id,
                    relationship_type="sponsor",
                )
            )
            session.add(RecordTopic(source_record_id=record.id, topic_id=topic.id, confidence=1.0))
        session.flush()

        unbounded = record_ids_for_entity(session, "competitor", organization.id)
        bounded = record_ids_for_entity(
            session, "competitor", organization.id, published_after=date(2026, 1, 1)
        )

    assert set(unbounded) == {old.id, recent.id}
    assert bounded == [recent.id]


def test_published_after_is_inclusive_of_its_own_date(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        topic = Topic(key="t", canonical_name="Topic", topic_type="technology")
        session.add(topic)
        session.flush()
        record = _record(session, "r1", "publication", published=datetime(2026, 3, 1, tzinfo=UTC))
        session.add(RecordTopic(source_record_id=record.id, topic_id=topic.id, confidence=1.0))
        session.flush()
        ids = record_ids_for_entity(session, "topic", topic.id, published_after=date(2026, 3, 1))
    assert ids == [record.id]
