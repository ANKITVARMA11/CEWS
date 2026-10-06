"""Every link-table row should point at something that still exists.

Foreign keys with ``ondelete=CASCADE``, enforced by the ``PRAGMA foreign_keys=ON`` this project
turns on for every SQLite connection, make an orphan impossible through ordinary use, and deleting
a referenced row cascades to its children rather than leaving them stranded - both confirmed
directly below. The deliberately orphaned cases are created through a raw connection with
enforcement switched off, the way an older database file, a manual edit, or a different backend
without the same safeguard could produce one in practice.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker
from tests.data_quality.conftest import make_record

from cews.database.connection import session_scope
from cews.database.models import (
    Insight,
    InsightEvidence,
    Organization,
    RecordOrganization,
    RecordTopic,
    Topic,
)
from cews.validation.data_quality import check_referential_integrity

pytestmark = pytest.mark.unit


def _bypass_and_delete(engine: Engine, table: str, row_id: int) -> None:
    """Delete a row with FK enforcement switched off on a raw connection.

    SQLite only honours a change to ``PRAGMA foreign_keys`` between transactions, so this uses
    the DBAPI connection directly rather than an ORM session, which begins a transaction as soon
    as it is used.
    """
    raw = engine.raw_connection()
    try:
        raw.execute("PRAGMA foreign_keys=OFF")
        raw.execute(
            f"DELETE FROM {table} WHERE id=?", (row_id,)
        )  # noqa: S608 - test-only, fixed table names
        raw.commit()
    finally:
        raw.execute("PRAGMA foreign_keys=ON")
        raw.close()


def test_a_clean_database_passes(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        record = make_record(session, "r1")
        topic = Topic(key="t", canonical_name="Topic", topic_type="technology")
        session.add(topic)
        session.flush()
        session.add(RecordTopic(source_record_id=record.id, topic_id=topic.id, confidence=1.0))
        result = check_referential_integrity(session)
    assert result.passed


def test_foreign_keys_prevent_an_orphan_by_construction(
    factory: sessionmaker[Session],
) -> None:
    """The safeguard this check exists in case of, demonstrated directly.

    The exception must propagate out of ``session_scope`` itself (rather than being caught
    inside it), so its own rollback path runs; catching it inside first would leave the session
    in a state ``session_scope`` then fails to commit cleanly on exit.
    """
    with (
        pytest.raises(Exception, match="FOREIGN KEY constraint failed"),
        session_scope(factory) as session,
    ):
        session.add(RecordTopic(source_record_id=999999, topic_id=999999, confidence=1.0))
        session.flush()


def test_deleting_a_topic_cascades_rather_than_orphaning_its_links(
    factory: sessionmaker[Session],
) -> None:
    """The normal path: removing a topic removes what pointed to it, leaving nothing stranded."""
    with session_scope(factory) as session:
        record = make_record(session, "r1")
        topic = Topic(key="t", canonical_name="Topic", topic_type="technology")
        session.add(topic)
        session.flush()
        session.add(RecordTopic(source_record_id=record.id, topic_id=topic.id, confidence=1.0))
        session.flush()
        session.delete(topic)
    with session_scope(factory) as session:
        assert session.query(RecordTopic).count() == 0
        assert check_referential_integrity(session).passed


def test_an_orphan_that_bypassed_enforcement_is_still_caught(
    factory: sessionmaker[Session],
) -> None:
    engine = factory.kw["bind"]
    with session_scope(factory) as session:
        record = make_record(session, "r1")
        topic = Topic(key="t", canonical_name="Topic", topic_type="technology")
        session.add(topic)
        session.flush()
        link = RecordTopic(source_record_id=record.id, topic_id=topic.id, confidence=1.0)
        session.add(link)
        session.flush()
        link_id, topic_id = link.id, topic.id

    _bypass_and_delete(engine, "topics", topic_id)

    with session_scope(factory) as session:
        result = check_referential_integrity(session)
    assert not result.passed
    assert f"record_topics.{link_id}" in result.examples[0]


def test_an_orphaned_organization_link_is_caught(factory: sessionmaker[Session]) -> None:
    engine = factory.kw["bind"]
    with session_scope(factory) as session:
        record = make_record(session, "r1")
        organization = Organization(
            canonical_name="Example Pharma", normalized_name="example pharma"
        )
        session.add(organization)
        session.flush()
        link = RecordOrganization(
            source_record_id=record.id,
            organization_id=organization.id,
            relationship_type="sponsor",
            confidence=1.0,
        )
        session.add(link)
        session.flush()
        link_id, organization_id = link.id, organization.id

    _bypass_and_delete(engine, "organizations", organization_id)

    with session_scope(factory) as session:
        result = check_referential_integrity(session)
    assert not result.passed
    assert f"record_organizations.{link_id}" in result.examples[0]


def test_insight_evidence_pointing_at_a_missing_source_record_is_caught(
    factory: sessionmaker[Session],
) -> None:
    engine = factory.kw["bind"]
    with session_scope(factory) as session:
        record = make_record(session, "r1")
        insight = Insight(
            insight_date=datetime(2026, 8, 1, tzinfo=UTC).date(),
            severity="watch",
            insight_type="emerging_trend",
            entity_type="topic",
            entity_id=1,
            title="T",
            observed_fact="fact",
            interpretation="interp",
            recommended_review="review",
            confidence_score=90.0,
        )
        session.add(insight)
        session.flush()
        session.add(InsightEvidence(insight_id=insight.id, source_record_id=record.id))
        session.flush()
        record_id = record.id

    _bypass_and_delete(engine, "source_records", record_id)

    with session_scope(factory) as session:
        result = check_referential_integrity(session)
    assert not result.passed
    assert any("missing source record" in example for example in result.examples)


def test_no_data_at_all_is_not_a_failure(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        result = check_referential_integrity(session)
    assert result.passed
