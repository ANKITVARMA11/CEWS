"""Unit tests for ``components.review_panel``'s evidence-resolution helpers.

These run below Streamlit entirely - no ``AppTest`` - because the thing worth checking here
(resolving an organization name or topic id from a review item's payload into the real source
records behind it) is plain database logic, and testing it that way is both faster and more
direct than driving it through a rendered page.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from cews.database.connection import create_memory_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import (
    Organization,
    RecordOrganization,
    RecordTopic,
    ReviewQueueItem,
    SourceRecord,
    Topic,
)

STREAMLIT_DIR = Path(__file__).resolve().parents[2] / "dashboards" / "streamlit"
if str(STREAMLIT_DIR) not in sys.path:
    sys.path.insert(0, str(STREAMLIT_DIR))

from components.review_panel import _evidence_for, _named_organizations, _subject  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture
def factory() -> Iterator[sessionmaker[Session]]:
    engine = create_memory_engine()
    upgrade_database(engine)
    yield create_session_factory(engine)
    engine.dispose()


def test_named_organizations_collects_every_relevant_payload_field() -> None:
    payload = {
        "raw_name": "Orvexa Bio",
        "linked_to": "Orvexa Labs",
        "candidates": ["Orvexa Holdings", "Orvexa Inc"],
    }
    assert _named_organizations(payload) == {
        "Orvexa Bio",
        "Orvexa Labs",
        "Orvexa Holdings",
        "Orvexa Inc",
    }


def test_subject_prefers_organizations_then_raw_name_then_label() -> None:
    item = ReviewQueueItem(
        queue_type="organization_parent",
        subject_ref="fallback",
        payload_json={"organizations": ["A", "B"]},
        status="pending",
    )
    assert "A" in _subject(item) and "B" in _subject(item)

    item2 = ReviewQueueItem(
        queue_type="organization_parent",
        subject_ref="fallback",
        payload_json={"raw_name": "Just One Name"},
        status="pending",
    )
    assert _subject(item2) == "Just One Name"

    item3 = ReviewQueueItem(
        queue_type="organization_parent",
        subject_ref="fallback-ref",
        payload_json={},
        status="pending",
    )
    assert _subject(item3) == "fallback-ref"


def test_evidence_for_resolves_a_named_organization_to_its_real_records(
    factory: sessionmaker[Session],
) -> None:
    with session_scope(factory) as session:
        org = Organization(canonical_name="Orvexa Bio", normalized_name="orvexa bio")
        session.add(org)
        session.flush()
        record = SourceRecord(
            source="clinicaltrials_gov",
            source_record_id="NCT123",
            content_hash="hash-nct123",
            record_type="clinical_trial",
            title="A trial by Orvexa Bio",
            source_url="https://example.test/NCT123",
            published_at=datetime(2026, 1, 1, tzinfo=UTC),
            fetched_at=datetime(2026, 1, 2, tzinfo=UTC),
        )
        session.add(record)
        session.flush()
        session.add(
            RecordOrganization(
                source_record_id=record.id,
                organization_id=org.id,
                relationship_type="sponsor",
            )
        )
        item = ReviewQueueItem(
            queue_type="organization_parent",
            subject_ref="Orvexa Bio|Orvexa Labs",
            payload_json={"organizations": ["Orvexa Bio", "Orvexa Labs"]},
            status="pending",
        )
        session.add(item)
        session.flush()

    with session_scope(factory) as session:
        stored = session.get(ReviewQueueItem, 1)
        assert stored is not None
        evidence = _evidence_for(session, stored)
        assert "Orvexa Bio" in evidence
        assert len(evidence["Orvexa Bio"]) == 1
        assert evidence["Orvexa Bio"][0]["title"] == "A trial by Orvexa Bio"
        # "Orvexa Labs" was named but never resolved to a real Organization row - it gets no
        # section, which is correct (a brand-new raw name is not an error), not an empty list.
        assert "Orvexa Labs" not in evidence


def test_evidence_for_resolves_an_ai_topic_candidate_by_its_subject_ref(
    factory: sessionmaker[Session],
) -> None:
    with session_scope(factory) as session:
        topic = Topic(key="novel-topic", canonical_name="Novel Topic", topic_type="ai_candidate")
        session.add(topic)
        session.flush()
        record = SourceRecord(
            source="pubmed",
            source_record_id="PM1",
            content_hash="hash-pm1",
            record_type="publication",
            title="A paper about the novel topic",
            source_url="https://example.test/PM1",
            published_at=datetime(2026, 2, 1, tzinfo=UTC),
            fetched_at=datetime(2026, 2, 2, tzinfo=UTC),
        )
        session.add(record)
        session.flush()
        session.add(RecordTopic(source_record_id=record.id, topic_id=topic.id))
        item = ReviewQueueItem(
            queue_type="ai_topic_candidate",
            subject_ref=f"topic:{topic.id}",
            payload_json={"label": "Novel Topic"},
            status="pending",
        )
        session.add(item)
        session.flush()

    with session_scope(factory) as session:
        stored = session.get(ReviewQueueItem, 1)
        assert stored is not None
        evidence = _evidence_for(session, stored)
        assert "Novel Topic" in evidence
        assert evidence["Novel Topic"][0]["title"] == "A paper about the novel topic"


def test_evidence_for_a_name_with_no_matching_organization_is_empty(
    factory: sessionmaker[Session],
) -> None:
    with session_scope(factory) as session:
        item = ReviewQueueItem(
            queue_type="organization_parent",
            subject_ref="Nobody Yet|Nobody Else",
            payload_json={"organizations": ["Nobody Yet", "Nobody Else"]},
            status="pending",
        )
        session.add(item)
        session.flush()

    with session_scope(factory) as session:
        stored = session.get(ReviewQueueItem, 1)
        assert stored is not None
        assert _evidence_for(session, stored) == {}
