"""Every stored score, confidence and link weight must be on the scale it promises.

``scores.score_value``, ``scores.confidence_score``, ``record_topics.confidence`` and
``record_organizations.confidence`` all carry a database-level ``CHECK`` constraint already, so
this sweep is a defensive backstop rather than the primary guard - the same relationship
``check_referential_integrity`` has with foreign keys. The out-of-range rows here are created
through a raw connection with ``PRAGMA ignore_check_constraints`` on, the way a bulk import, a
schema change, or a different backend without the same constraint could let one through.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker
from tests.data_quality.conftest import make_record

from cews.database.connection import session_scope
from cews.database.models import Organization, Score, Topic
from cews.validation.data_quality import check_score_ranges

pytestmark = pytest.mark.unit


def _insert_score_bypassing_the_check(
    engine: Engine, *, score_value: float, confidence: float
) -> None:
    raw = engine.raw_connection()
    try:
        raw.execute("PRAGMA ignore_check_constraints=ON")
        raw.execute(
            "INSERT INTO scores (score_date, entity_type, entity_id, context_key, score_type, "
            "score_value, confidence_score, component_json, scoring_version, is_synthetic, created_at) "
            "VALUES ('2026-08-01','topic',1,'','trend',?,?,'{}','1.0.0',0,'2026-08-01T00:00:00+00:00')",
            (score_value, confidence),
        )
        raw.commit()
    finally:
        raw.execute("PRAGMA ignore_check_constraints=OFF")
        raw.close()


def test_a_normal_score_passes(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        session.add(
            Score(
                score_date=datetime(2026, 8, 1, tzinfo=UTC).date(),
                entity_type="topic",
                entity_id=1,
                score_type="trend",
                score_value=70.0,
                confidence_score=90.0,
                component_json={},
                scoring_version="1.0.0",
            )
        )
        result = check_score_ranges(session)
    assert result.passed


def test_the_database_itself_refuses_an_out_of_range_score(
    factory: sessionmaker[Session],
) -> None:
    """The safeguard this check backs up, demonstrated directly."""
    with (
        pytest.raises(Exception, match="CHECK constraint failed"),
        session_scope(factory) as session,
    ):
        session.add(
            Score(
                score_date=datetime(2026, 8, 1, tzinfo=UTC).date(),
                entity_type="topic",
                entity_id=1,
                score_type="trend",
                score_value=150.0,
                confidence_score=90.0,
                component_json={},
                scoring_version="1.0.0",
            )
        )
        session.flush()


def test_a_score_that_bypassed_the_constraint_is_still_caught(
    factory: sessionmaker[Session],
) -> None:
    engine = factory.kw["bind"]
    _insert_score_bypassing_the_check(engine, score_value=150.0, confidence=90.0)
    with session_scope(factory) as session:
        result = check_score_ranges(session)
    assert not result.passed
    assert "score_value=150.0" in result.examples[0]


def test_an_out_of_range_confidence_is_also_caught(factory: sessionmaker[Session]) -> None:
    engine = factory.kw["bind"]
    _insert_score_bypassing_the_check(engine, score_value=70.0, confidence=-5.0)
    with session_scope(factory) as session:
        result = check_score_ranges(session)
    assert not result.passed
    assert "confidence_score=-5.0" in result.examples[0]


def test_a_topic_link_confidence_out_of_range_is_caught(factory: sessionmaker[Session]) -> None:
    engine = factory.kw["bind"]
    with session_scope(factory) as session:
        record = make_record(session, "r1")
        topic = Topic(key="t", canonical_name="Topic", topic_type="technology")
        session.add(topic)
        session.flush()
        record_id, topic_id = record.id, topic.id

    raw = engine.raw_connection()
    try:
        raw.execute("PRAGMA ignore_check_constraints=ON")
        raw.execute(
            "INSERT INTO record_topics (source_record_id, topic_id, matching_method, confidence) "
            "VALUES (?, ?, 'keyword', 5.0)",
            (record_id, topic_id),
        )
        raw.commit()
    finally:
        raw.execute("PRAGMA ignore_check_constraints=OFF")
        raw.close()

    with session_scope(factory) as session:
        result = check_score_ranges(session)
    assert not result.passed
    assert any(
        "record_topics" in example and "confidence=5.0" in example for example in result.examples
    )


def test_an_organization_link_confidence_out_of_range_is_caught(
    factory: sessionmaker[Session],
) -> None:
    engine = factory.kw["bind"]
    with session_scope(factory) as session:
        record = make_record(session, "r1")
        organization = Organization(
            canonical_name="Example Pharma", normalized_name="example pharma"
        )
        session.add(organization)
        session.flush()
        record_id, organization_id = record.id, organization.id

    raw = engine.raw_connection()
    try:
        raw.execute("PRAGMA ignore_check_constraints=ON")
        raw.execute(
            "INSERT INTO record_organizations "
            "(source_record_id, organization_id, relationship_type, match_method, confidence) "
            "VALUES (?, ?, 'sponsor', 'deterministic', -1.0)",
            (record_id, organization_id),
        )
        raw.commit()
    finally:
        raw.execute("PRAGMA ignore_check_constraints=OFF")
        raw.close()

    with session_scope(factory) as session:
        result = check_score_ranges(session)
    assert not result.passed
    assert any(
        "record_organizations" in example and "confidence=-1.0" in example
        for example in result.examples
    )


def test_no_data_at_all_is_not_a_failure(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        result = check_score_ranges(session)
    assert result.passed
