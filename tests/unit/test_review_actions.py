"""Unit tests for approving and rejecting review-queue items."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy.orm import Session, sessionmaker

from cews.constants import TopicStatus
from cews.database.connection import create_memory_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import ReviewQueueItem, Topic
from cews.normalization.review_actions import (
    ReviewActionError,
    approve_review_item,
    pending_review_items,
    reject_review_item,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def factory() -> Iterator[sessionmaker[Session]]:
    engine = create_memory_engine()
    upgrade_database(engine)
    yield create_session_factory(engine)
    engine.dispose()


def _candidate_topic(session: Session) -> Topic:
    topic = Topic(
        key="ai_candidate_protac",
        canonical_name="Protac, Disorders, Genetic",
        topic_type="ai_candidate",
        status=TopicStatus.PENDING_REVIEW.value,
        active=False,
    )
    session.add(topic)
    session.flush()
    session.add(
        ReviewQueueItem(
            queue_type="ai_topic_candidate",
            subject_ref=f"topic:{topic.id}",
            payload_json={"label": topic.canonical_name, "record_count": 10},
            status="pending",
        )
    )
    session.flush()
    return topic


def _org_item(session: Session) -> ReviewQueueItem:
    item = ReviewQueueItem(
        queue_type="organization_parent",
        subject_ref="Orvexa Bio|Orvexa Labs",
        payload_json={"organizations": ["Orvexa Bio", "Orvexa Labs"], "reason": "same prefix"},
        status="pending",
    )
    session.add(item)
    session.flush()
    return item


# --------------------------------------------------------------------------------------
# Approving an AI topic candidate
# --------------------------------------------------------------------------------------
def test_approving_a_topic_candidate_activates_it(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        topic = _candidate_topic(session)
        item_id = session.query(ReviewQueueItem).one().id
        result = approve_review_item(session, item_id)
    with session_scope(factory) as session:
        refreshed = session.get(Topic, topic.id)
        assert refreshed is not None
        assert refreshed.active is True
        assert refreshed.status == TopicStatus.ACTIVE.value
    assert result.decision == "accepted"
    assert "activated topic" in result.effect
    assert "Protac" in result.effect


def test_the_review_item_itself_is_marked_accepted(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        _candidate_topic(session)
        item_id = session.query(ReviewQueueItem).one().id
        approve_review_item(session, item_id)
    with session_scope(factory) as session:
        item = session.get(ReviewQueueItem, item_id)
        assert item is not None
        assert item.status == "accepted"
        assert item.resolved_at is not None


def test_rejecting_a_topic_candidate_leaves_it_inactive(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        topic = _candidate_topic(session)
        item_id = session.query(ReviewQueueItem).one().id
        result = reject_review_item(session, item_id)
    with session_scope(factory) as session:
        refreshed = session.get(Topic, topic.id)
        assert refreshed is not None
        assert refreshed.active is False  # unchanged
    assert result.decision == "rejected"
    assert "stays inactive" in result.effect


def test_approving_when_the_topic_no_longer_exists_is_refused(
    factory: sessionmaker[Session],
) -> None:
    with session_scope(factory) as session:
        topic = _candidate_topic(session)
        item_id = session.query(ReviewQueueItem).one().id
        session.delete(topic)
        session.flush()
        with pytest.raises(ReviewActionError, match="no longer exists"):
            approve_review_item(session, item_id)


# --------------------------------------------------------------------------------------
# Organization items: CEWS never merges automatically
# --------------------------------------------------------------------------------------
def test_approving_an_organization_item_takes_no_automatic_action(
    factory: sessionmaker[Session],
) -> None:
    with session_scope(factory) as session:
        item = _org_item(session)
        result = approve_review_item(session, item.id)
    assert result.decision == "accepted"
    assert "no automatic action" in result.effect


def test_rejecting_an_organization_item_takes_no_automatic_action(
    factory: sessionmaker[Session],
) -> None:
    with session_scope(factory) as session:
        item = _org_item(session)
        result = reject_review_item(session, item.id)
    assert result.decision == "rejected"
    assert "no data was changed" in result.effect


# --------------------------------------------------------------------------------------
# Guard rails
# --------------------------------------------------------------------------------------
def test_approving_a_nonexistent_item_is_refused(factory: sessionmaker[Session]) -> None:
    with (
        session_scope(factory) as session,
        pytest.raises(ReviewActionError, match="no review item"),
    ):
        approve_review_item(session, 999999)


def test_rejecting_a_nonexistent_item_is_refused(factory: sessionmaker[Session]) -> None:
    with (
        session_scope(factory) as session,
        pytest.raises(ReviewActionError, match="no review item"),
    ):
        reject_review_item(session, 999999)


def test_approving_an_already_accepted_item_is_refused(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        item = _org_item(session)
        approve_review_item(session, item.id)
        with pytest.raises(ReviewActionError, match="already accepted"):
            approve_review_item(session, item.id)


def test_rejecting_an_already_rejected_item_is_refused(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        item = _org_item(session)
        reject_review_item(session, item.id)
        with pytest.raises(ReviewActionError, match="already rejected"):
            reject_review_item(session, item.id)


def test_rejecting_an_already_accepted_item_is_also_refused(
    factory: sessionmaker[Session],
) -> None:
    """Approve then reject the same item must not silently flip the decision."""
    with session_scope(factory) as session:
        item = _org_item(session)
        approve_review_item(session, item.id)
        with pytest.raises(ReviewActionError, match="already accepted"):
            reject_review_item(session, item.id)


def test_a_malformed_topic_candidate_subject_ref_is_refused(
    factory: sessionmaker[Session],
) -> None:
    with session_scope(factory) as session:
        item = ReviewQueueItem(
            queue_type="ai_topic_candidate",
            subject_ref="not-a-topic-reference",
            payload_json={"label": "broken"},
            status="pending",
        )
        session.add(item)
        session.flush()
        with pytest.raises(ReviewActionError, match="malformed subject_ref"):
            approve_review_item(session, item.id)


# --------------------------------------------------------------------------------------
# Listing
# --------------------------------------------------------------------------------------
def test_pending_review_items_excludes_resolved_ones(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        _candidate_topic(session)
        org_item = _org_item(session)
        reject_review_item(session, org_item.id)
    with session_scope(factory) as session:
        pending = pending_review_items(session)
        assert len(pending) == 1
        assert pending[0].queue_type == "ai_topic_candidate"


def test_pending_review_items_respects_the_limit(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        for index in range(5):
            session.add(
                ReviewQueueItem(
                    queue_type="organization_parent",
                    subject_ref=f"pair-{index}",
                    payload_json={"organizations": [f"A{index}", f"B{index}"]},
                    status="pending",
                )
            )
        session.flush()
    with session_scope(factory) as session:
        assert len(pending_review_items(session, limit=3)) == 3
