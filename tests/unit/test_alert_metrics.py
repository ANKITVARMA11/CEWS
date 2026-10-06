"""Unit tests for expert alert review metrics."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, date, datetime

import pytest
from sqlalchemy.orm import Session, sessionmaker

from cews.database.connection import create_memory_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import ActivityAggregate, AlertReview, Insight, Topic
from cews.validation.alert_metrics import (
    AlertReviewError,
    compute_alert_metrics,
    confirmation_lead_days,
    latest_reviews,
    record_alert_review,
)

pytestmark = pytest.mark.unit

ALERT_DATE = date(2026, 3, 1)


@pytest.fixture
def factory() -> Iterator[sessionmaker[Session]]:
    engine = create_memory_engine()
    upgrade_database(engine)
    yield create_session_factory(engine)
    engine.dispose()


def _insight(session: Session, *, entity_id: int = 1, insight_date: date = ALERT_DATE) -> Insight:
    insight = Insight(
        insight_date=insight_date,
        severity="watch",
        insight_type="emerging_trend",
        entity_type="topic",
        entity_id=entity_id,
        title="T",
        observed_fact="fact",
        interpretation="meaning",
        recommended_review="review",
        confidence_score=90.0,
    )
    session.add(insight)
    session.flush()
    return insight


def _topic(session: Session) -> Topic:
    topic = Topic(key="t", canonical_name="Topic", topic_type="technology")
    session.add(topic)
    session.flush()
    return topic


def _activity(session: Session, topic_id: int, month: date, count: int) -> None:
    session.add(
        ActivityAggregate(
            period=month,
            period_type="month",
            topic_id=topic_id,
            source_type="publication",
            activity_count=count,
            weighted_activity=float(count),
            unique_record_count=count,
        )
    )
    session.flush()


def _review_many(session: Session, ratings: list[str]) -> list[Insight]:
    insights = []
    for index, rating in enumerate(ratings):
        insight = _insight(session, entity_id=index + 1)
        record_alert_review(session, insight.id, rating, reviewer="expert")
        insights.append(insight)
    return insights


# --------------------------------------------------------------------------------------
# Recording reviews
# --------------------------------------------------------------------------------------
def test_a_review_is_recorded(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        insight = _insight(session)
        review = record_alert_review(
            session, insight.id, "relevant", reviewer="ana", comment="good"
        )
    assert review.rating == "relevant" and review.reviewer == "ana" and review.comment == "good"


def test_an_unknown_rating_is_refused(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        insight = _insight(session)
        with pytest.raises(AlertReviewError, match="unknown rating"):
            record_alert_review(session, insight.id, "great")


def test_a_missing_insight_is_refused(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session, pytest.raises(AlertReviewError, match="no insight"):
        record_alert_review(session, 999, "relevant")


def test_only_the_latest_review_of_an_insight_counts(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        insight = _insight(session)
        first = record_alert_review(session, insight.id, "not_relevant")
        first.reviewed_at = datetime(2026, 1, 1, tzinfo=UTC)
        second = record_alert_review(session, insight.id, "relevant")
        second.reviewed_at = datetime(2026, 2, 1, tzinfo=UTC)
        session.flush()
        assert latest_reviews(session)[insight.id].rating == "relevant"
        metrics = compute_alert_metrics(session)
    assert metrics.reviewed == 1
    assert metrics.by_rating["relevant"] == 1 and metrics.by_rating["not_relevant"] == 0


def test_a_tie_on_time_goes_to_the_later_review(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        insight = _insight(session)
        stamp = datetime(2026, 1, 1, tzinfo=UTC)
        for rating in ("not_relevant", "relevant"):
            review = record_alert_review(session, insight.id, rating)
            review.reviewed_at = stamp
        session.flush()
        assert latest_reviews(session)[insight.id].rating == "relevant"


# --------------------------------------------------------------------------------------
# The four rates
# --------------------------------------------------------------------------------------
def test_precision_excludes_duplicates_and_undecided(factory: sessionmaker[Session]) -> None:
    """Judged = relevant, not relevant, too late, insufficient evidence."""
    with session_scope(factory) as session:
        _review_many(
            session,
            [
                "relevant",
                "relevant",
                "not_relevant",
                "too_late",
                "duplicate",
                "needs_investigation",
            ],
        )
        metrics = compute_alert_metrics(session)
    assert metrics.precision == pytest.approx(2 / 4)


def test_acceptance_counts_relevant_and_needs_investigation(
    factory: sessionmaker[Session],
) -> None:
    with session_scope(factory) as session:
        _review_many(
            session, ["relevant", "needs_investigation", "not_relevant", "duplicate", "too_late"]
        )
        metrics = compute_alert_metrics(session)
    assert metrics.acceptance_rate == pytest.approx(2 / 5)


def test_duplicate_rate_is_a_share_of_everything_reviewed(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        _review_many(session, ["duplicate", "relevant", "relevant", "not_relevant"])
        metrics = compute_alert_metrics(session)
    assert metrics.duplicate_rate == pytest.approx(1 / 4)


def test_unreviewed_insights_are_counted_not_assumed_bad(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        _review_many(session, ["relevant"])
        _insight(session, entity_id=50)
        _insight(session, entity_id=51)
        metrics = compute_alert_metrics(session)
    assert metrics.insights_total == 3 and metrics.reviewed == 1 and metrics.unreviewed == 2
    assert metrics.precision == 1.0


def test_nothing_reviewed_means_undefined_not_zero(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        _insight(session)
        metrics = compute_alert_metrics(session)
    assert metrics.precision is None and metrics.acceptance_rate is None
    assert metrics.duplicate_rate is None and metrics.median_lead_days is None
    assert any("no insight has been reviewed" in note for note in metrics.notes)


def test_only_undecided_reviews_leave_precision_undefined(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        _review_many(session, ["duplicate", "needs_investigation"])
        metrics = compute_alert_metrics(session)
    assert metrics.precision is None
    assert any("precision is undefined" in note for note in metrics.notes)


def test_every_rating_appears_in_the_breakdown(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        metrics = compute_alert_metrics(session)
    assert set(metrics.by_rating) == {
        "relevant",
        "not_relevant",
        "duplicate",
        "too_late",
        "insufficient_evidence",
        "needs_investigation",
    }


def test_a_review_of_a_deleted_insight_is_ignored(factory: sessionmaker[Session]) -> None:
    """Reviews cascade with their insight; a stray one must never inflate the counts."""
    with session_scope(factory) as session:
        insight = _insight(session)
        record_alert_review(session, insight.id, "relevant")
        session.delete(insight)
        session.flush()
        assert session.query(AlertReview).count() == 0
        assert compute_alert_metrics(session).reviewed == 0


# --------------------------------------------------------------------------------------
# Lead time
# --------------------------------------------------------------------------------------
def _flat_then_rise(session: Session, topic_id: int, *, rise_month: date, high: int) -> None:
    for step in (-2, -1, 0):
        month = date(2026, 3 + step, 1)
        _activity(session, topic_id, month, 10)
    for step in range(1, 7):
        month = date(2026 + (2 + step) // 12, (2 + step) % 12 + 1, 1)
        _activity(session, topic_id, month, high if month >= rise_month else 10)


def test_lead_time_is_days_until_activity_first_takes_off(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        topic = _topic(session)
        _flat_then_rise(session, topic.id, rise_month=date(2026, 5, 1), high=20)
        days = confirmation_lead_days(session, "topic", topic.id, ALERT_DATE)
    assert days == (date(2026, 5, 1) - ALERT_DATE).days


def test_activity_that_never_takes_off_has_no_lead_time(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        topic = _topic(session)
        _flat_then_rise(session, topic.id, rise_month=date(2027, 1, 1), high=20)
        assert confirmation_lead_days(session, "topic", topic.id, ALERT_DATE) is None


def test_a_modest_rise_below_the_ratio_is_not_confirmation(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        topic = _topic(session)
        _flat_then_rise(session, topic.id, rise_month=date(2026, 4, 1), high=13)  # 1.3x < 1.5x
        assert confirmation_lead_days(session, "topic", topic.id, ALERT_DATE) is None


def test_a_silent_entity_does_not_confirm_on_one_stray_record(
    factory: sessionmaker[Session],
) -> None:
    """The baseline floor of 1 record stops a single record from looking like a 50% jump."""
    with session_scope(factory) as session:
        topic = _topic(session)
        _activity(session, topic.id, date(2026, 4, 1), 1)
        assert confirmation_lead_days(session, "topic", topic.id, ALERT_DATE) is None


def test_lead_time_needs_a_sensible_ratio_and_horizon(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        with pytest.raises(ValueError, match="ratio"):
            confirmation_lead_days(session, "topic", 1, ALERT_DATE, ratio=1.0)
        with pytest.raises(ValueError, match="months_ahead"):
            confirmation_lead_days(session, "topic", 1, ALERT_DATE, months_ahead=0)


def test_median_lead_time_uses_only_relevant_and_measurable_alerts(
    factory: sessionmaker[Session],
) -> None:
    with session_scope(factory) as session:
        early = _topic(session)
        _flat_then_rise(session, early.id, rise_month=date(2026, 4, 1), high=20)
        late = Topic(key="t2", canonical_name="Late", topic_type="technology")
        session.add(late)
        session.flush()
        _flat_then_rise(session, late.id, rise_month=date(2026, 6, 1), high=20)
        never = Topic(key="t3", canonical_name="Never", topic_type="technology")
        session.add(never)
        session.flush()
        for topic, rating in ((early, "relevant"), (late, "relevant"), (never, "relevant")):
            insight = _insight(session, entity_id=topic.id)
            record_alert_review(session, insight.id, rating)
        ignored = _insight(session, entity_id=early.id)
        record_alert_review(session, ignored.id, "not_relevant")  # not counted for lead time
        metrics = compute_alert_metrics(session)
    expected = [(date(2026, 4, 1) - ALERT_DATE).days, (date(2026, 6, 1) - ALERT_DATE).days]
    assert metrics.median_lead_days == pytest.approx(sum(expected) / 2)
    assert metrics.lead_time_measurable == 2 and metrics.lead_time_relevant == 3


def test_the_metrics_are_json_friendly(factory: sessionmaker[Session]) -> None:
    import json

    with session_scope(factory) as session:
        _review_many(session, ["relevant", "duplicate"])
        json.dumps(compute_alert_metrics(session).as_dict())
