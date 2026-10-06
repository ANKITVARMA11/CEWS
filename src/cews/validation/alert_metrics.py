"""Judging alerts by what an expert made of them.

An insight is only worth having if a person who knows the field would act on it. Experts rate
each one (relevant, not relevant, duplicate, too late, insufficient evidence, needs
investigation); this module records those ratings and turns them into four numbers:

* **precision** - of the alerts an expert could judge, how many were relevant. Duplicates and
  "needs investigation" are left out of the denominator: a duplicate is redundant rather than
  wrong, and an alert still being investigated has not been judged yet.
* **acceptance rate** - of everything reviewed, how much was not rejected outright.
* **duplicate rate** - of everything reviewed, how much repeated something already reported.
* **lead time** - for relevant alerts, how long before the underlying activity visibly took off
  the alert was raised.

Only an insight's **latest** review counts, so an expert changing their mind replaces the earlier
rating rather than being averaged with it.

Lead time has no external ground truth in this system, so it is a documented proxy: the number
of days from the alert until the entity's monthly activity first reached ``ratio`` times its
level around the time of the alert. An alert whose activity never did (or that is too recent to
tell) has no measurable lead time, and is reported as such rather than counted as zero.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from cews.constants import AlertRating, EntityType
from cews.database.models import AlertReview, Insight
from cews.features.activity_counts import monthly_series
from cews.features.time_windows import add_months

DEFAULT_CONFIRMATION_RATIO = 1.5
DEFAULT_MONTHS_AHEAD = 6
BASELINE_MONTHS = 3
JUDGED_RATINGS = frozenset(
    {
        AlertRating.RELEVANT.value,
        AlertRating.NOT_RELEVANT.value,
        AlertRating.TOO_LATE.value,
        AlertRating.INSUFFICIENT_EVIDENCE.value,
    }
)
ACCEPTED_RATINGS = frozenset({AlertRating.RELEVANT.value, AlertRating.NEEDS_INVESTIGATION.value})


class AlertReviewError(ValueError):
    """Raised when a review cannot be recorded as asked."""


@dataclass(frozen=True)
class AlertMetrics:
    """Everything derived from the latest review of each insight."""

    insights_total: int
    reviewed: int
    unreviewed: int
    by_rating: dict[str, int]
    precision: float | None
    acceptance_rate: float | None
    duplicate_rate: float | None
    median_lead_days: float | None
    lead_time_measurable: int
    lead_time_relevant: int
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""

        def rounded(value: float | None) -> float | None:
            return None if value is None else round(value, 4)

        return {
            "insights_total": self.insights_total,
            "reviewed": self.reviewed,
            "unreviewed": self.unreviewed,
            "by_rating": dict(self.by_rating),
            "precision": rounded(self.precision),
            "acceptance_rate": rounded(self.acceptance_rate),
            "duplicate_rate": rounded(self.duplicate_rate),
            "median_lead_days": rounded(self.median_lead_days),
            "lead_time_measurable": self.lead_time_measurable,
            "lead_time_relevant": self.lead_time_relevant,
            "notes": list(self.notes),
        }


def record_alert_review(
    session: Session,
    insight_id: int,
    rating: str,
    *,
    reviewer: str | None = None,
    comment: str | None = None,
) -> AlertReview:
    """Record one expert's rating of one insight.

    Raises:
        AlertReviewError: if the rating is not one of :class:`AlertRating`, or the insight does
            not exist.
    """
    valid = {member.value for member in AlertRating}
    if rating not in valid:
        raise AlertReviewError(f"unknown rating {rating!r}; use one of: {', '.join(sorted(valid))}")
    if session.get(Insight, insight_id) is None:
        raise AlertReviewError(f"no insight with id {insight_id}")
    review = AlertReview(
        insight_id=insight_id,
        rating=rating,
        reviewer=reviewer,
        comment=comment,
        reviewed_at=datetime.now(UTC),
    )
    session.add(review)
    session.flush()
    return review


def latest_reviews(session: Session) -> dict[int, AlertReview]:
    """The most recent review of each reviewed insight (later id wins a same-instant tie)."""
    latest: dict[int, AlertReview] = {}
    for review in session.scalars(
        select(AlertReview).order_by(AlertReview.reviewed_at, AlertReview.id)
    ):
        latest[review.insight_id] = review
    return latest


def confirmation_lead_days(
    session: Session,
    entity_type: str,
    entity_id: int,
    alert_date: date,
    *,
    months_ahead: int = DEFAULT_MONTHS_AHEAD,
    ratio: float = DEFAULT_CONFIRMATION_RATIO,
) -> int | None:
    """Days from the alert until the entity's monthly activity first reached ``ratio`` times its
    level around the alert, or None if it never did within ``months_ahead`` months.

    The level around the alert is the mean of the ``BASELINE_MONTHS`` months up to and including
    the alert's own month, with a floor of 1 record so a previously silent entity does not
    "confirm" on a single stray record.

    Raises:
        ValueError: if ``ratio`` is not above 1 or ``months_ahead`` is not positive.
    """
    if ratio <= 1.0 or months_ahead < 1:
        raise ValueError("ratio must be above 1 and months_ahead positive")
    alert_month = date(alert_date.year, alert_date.month, 1)
    is_topic = entity_type == EntityType.TOPIC.value
    topic_id = entity_id if is_topic else None
    organization_id = None if is_topic else entity_id
    before = [add_months(alert_month, step) for step in range(-(BASELINE_MONTHS - 1), 1)]
    after = [add_months(alert_month, step) for step in range(1, months_ahead + 1)]

    def totals(months: list[date]) -> list[float]:
        by_source = monthly_series(
            session, months, organization_id=organization_id, topic_id=topic_id
        )
        return [sum(series[index] for series in by_source.values()) for index in range(len(months))]

    baseline = max(sum(totals(before)) / len(before), 1.0)
    for month, value in zip(after, totals(after), strict=True):
        if value >= ratio * baseline:
            return (month - alert_date).days
    return None


def compute_alert_metrics(
    session: Session,
    *,
    months_ahead: int = DEFAULT_MONTHS_AHEAD,
    ratio: float = DEFAULT_CONFIRMATION_RATIO,
) -> AlertMetrics:
    """Precision, acceptance, duplicate rate and lead time from the latest review of each insight."""
    insights = {row.id: row for row in session.scalars(select(Insight))}
    reviews = latest_reviews(session)
    reviews = {
        insight_id: review for insight_id, review in reviews.items() if insight_id in insights
    }
    by_rating = Counter(review.rating for review in reviews.values())
    reviewed = len(reviews)

    judged = sum(by_rating[rating] for rating in JUDGED_RATINGS)
    relevant = by_rating[AlertRating.RELEVANT.value]
    precision = relevant / judged if judged else None
    accepted = sum(by_rating[rating] for rating in ACCEPTED_RATINGS)
    acceptance = accepted / reviewed if reviewed else None
    duplicate = by_rating[AlertRating.DUPLICATE.value] / reviewed if reviewed else None

    leads: list[int] = []
    relevant_ids = [
        insight_id
        for insight_id, review in reviews.items()
        if review.rating == AlertRating.RELEVANT.value
    ]
    for insight_id in relevant_ids:
        insight = insights[insight_id]
        lead = confirmation_lead_days(
            session,
            insight.entity_type,
            insight.entity_id,
            insight.insight_date,
            months_ahead=months_ahead,
            ratio=ratio,
        )
        if lead is not None:
            leads.append(lead)

    notes: list[str] = []
    if not reviewed:
        notes.append("no insight has been reviewed yet; every rate is undefined")
    elif not judged:
        notes.append("nothing has been definitively judged yet; precision is undefined")
    if relevant_ids and not leads:
        notes.append(
            "no relevant alert has shown later confirming activity yet; lead time is undefined"
        )

    return AlertMetrics(
        insights_total=len(insights),
        reviewed=reviewed,
        unreviewed=len(insights) - reviewed,
        by_rating={rating.value: by_rating[rating.value] for rating in AlertRating},
        precision=precision,
        acceptance_rate=acceptance,
        duplicate_rate=duplicate,
        median_lead_days=float(statistics.median(leads)) if leads else None,
        lead_time_measurable=len(leads),
        lead_time_relevant=len(relevant_ids),
        notes=tuple(notes),
    )
