"""Approving or rejecting an item in the review queue.

The queue holds several kinds of decision (an AI-discovered topic, an uncertain organization
match), and what "approve" means depends on the kind:

* an **AI topic candidate** is activated, so it starts being scored like any other topic;
* an **organization match or parent/subsidiary question** has no automatic action - CEWS never
  merges organizations itself (see ``normalization/resolver.py``), so approving one only records
  that a person looked at it and decided what to do about it elsewhere.

Rejecting a topic candidate leaves it inactive; rejecting anything else is the same
record-keeping action as approving it, since neither ever touches other tables. Only a
**pending** item can be acted on: approving or rejecting an already-resolved item is refused
rather than silently repeated.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from cews.constants import TopicStatus
from cews.database.models import ReviewQueueItem, Topic

AI_TOPIC_CANDIDATE = "ai_topic_candidate"


class ReviewActionError(ValueError):
    """Raised when a review item cannot be approved or rejected as asked."""


@dataclass(frozen=True)
class ReviewActionResult:
    """What happened when a review item was resolved."""

    item_id: int
    queue_type: str
    subject_ref: str
    decision: str  # "accepted" or "rejected"
    effect: str  # a plain sentence describing what, if anything, changed


def _load_pending(session: Session, item_id: int) -> ReviewQueueItem:
    """
    Raises:
        ReviewActionError: if the id does not exist or the item is already resolved.
    """
    item = session.get(ReviewQueueItem, item_id)
    if item is None:
        raise ReviewActionError(f"no review item with id {item_id}")
    if item.status != "pending":
        raise ReviewActionError(
            f"review item {item_id} was already {item.status} on {item.resolved_at}; "
            "nothing left to decide"
        )
    return item


def approve_review_item(session: Session, item_id: int) -> ReviewActionResult:
    """Approve a pending review item, applying whatever effect its kind has.

    Raises:
        ReviewActionError: if the item does not exist, is already resolved, or (for an AI topic
            candidate) the topic it refers to no longer exists.
    """
    item = _load_pending(session, item_id)
    effect = "recorded as reviewed; this kind of item has no automatic action"

    if item.queue_type == AI_TOPIC_CANDIDATE:
        if not item.subject_ref.startswith("topic:"):
            raise ReviewActionError(
                f"malformed subject_ref for {AI_TOPIC_CANDIDATE}: {item.subject_ref!r}"
            )
        topic_id = int(item.subject_ref.removeprefix("topic:"))
        topic = session.get(Topic, topic_id)
        if topic is None:
            raise ReviewActionError(f"topic {topic_id} referenced by this item no longer exists")
        topic.active = True
        topic.status = TopicStatus.ACTIVE.value
        effect = f"activated topic {topic.canonical_name!r}; it will now be scored"

    item.status = "accepted"
    item.resolved_at = datetime.now(UTC)
    session.flush()
    return ReviewActionResult(item.id, item.queue_type, item.subject_ref, "accepted", effect)


def reject_review_item(session: Session, item_id: int) -> ReviewActionResult:
    """Reject a pending review item. Never activates or otherwise changes anything it refers to.

    Raises:
        ReviewActionError: if the item does not exist or is already resolved.
    """
    item = _load_pending(session, item_id)
    effect = "recorded as rejected; no data was changed"
    if item.queue_type == AI_TOPIC_CANDIDATE:
        effect = "recorded as rejected; the candidate topic stays inactive"

    item.status = "rejected"
    item.resolved_at = datetime.now(UTC)
    session.flush()
    return ReviewActionResult(item.id, item.queue_type, item.subject_ref, "rejected", effect)


def pending_review_items(
    session: Session,
    *,
    limit: int = 100,
    offset: int = 0,
    queue_type: str | None = None,
) -> list[ReviewQueueItem]:
    """One page of the items currently awaiting a decision, oldest first.

    ``limit``/``offset`` page through a queue that can be far larger than any one page - a real
    AUTO-mode pipeline run over months of continuous ingestion can leave tens of thousands of
    ambiguous-name items pending, and nothing here fits that in one call. Callers that need the
    true total (not just how many this page returned) use :func:`pending_review_count` instead -
    a bug once showed a page's own ``limit`` back as if it were the total, which looked like a
    second, smaller queue. ``queue_type`` narrows to one kind (e.g. ``"organization_parent"``),
    matching the same filter :func:`pending_review_count` accepts, so a filtered page and its
    filtered total always describe the same slice of the queue.
    """
    query = select(ReviewQueueItem).where(ReviewQueueItem.status == "pending")
    if queue_type:
        query = query.where(ReviewQueueItem.queue_type == queue_type)
    return list(
        session.scalars(
            query.order_by(ReviewQueueItem.queue_type, ReviewQueueItem.id)
            .offset(offset)
            .limit(limit)
        )
    )


def pending_review_count(session: Session, *, queue_type: str | None = None) -> int:
    """How many items are actually waiting - unbounded, unlike :func:`pending_review_items`'s
    page. See that function's docstring for why a page's own size must never stand in for this."""
    query = (
        select(func.count()).select_from(ReviewQueueItem).where(ReviewQueueItem.status == "pending")
    )
    if queue_type:
        query = query.where(ReviewQueueItem.queue_type == queue_type)
    return int(session.scalar(query) or 0)


def review_queue_types(session: Session) -> list[str]:
    """Every distinct kind of item currently pending, for a filter control - sorted, so the
    choices never reorder themselves between one page load and the next."""
    return sorted(
        str(value)
        for value in session.scalars(
            select(ReviewQueueItem.queue_type).where(ReviewQueueItem.status == "pending").distinct()
        )
    )
