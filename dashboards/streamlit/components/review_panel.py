"""The review-queue panel: shared by the dashboard's Review page and the chat page.

One real queue can hold far more than fits on a screen - a long-running AUTO-mode pipeline left
over 100,000 pending items in one real database, almost all the same handful of "ambiguous short
name" kind. This panel is built around that reality rather than assuming a queue of a dozen
items:

* the true total is always shown separately from how many rows are on screen (see
  ``cews.normalization.review_actions.pending_review_count`` - a page's own size was once echoed
  back as if it were the total, which looked like a second, smaller queue that disagreed with
  the Overview page and the chat agent);
* a kind filter and offset-based paging make the backlog actually reachable, a page at a time;
* each item's full detail - every payload field, not a curated subset, plus links to the actual
  source records behind any organization or topic it names - opens in a popup
  (:func:`streamlit.dialog`) rather than being guessed at from the one-line summary, so a
  reviewer can check a claim before approving or rejecting it, the same evidence-first principle
  as every other page in this dashboard.

Approving or rejecting here calls ``cews.normalization.review_actions`` directly, in this same
Streamlit process, with a normal (not read-only) database session - never through the MCP
server, which stays read-only on purpose, and never through the LLM, whose job is to answer
questions, not to be trusted to carry out a write it was asked to make.
"""

from __future__ import annotations

from typing import Any

import streamlit as st
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from cews.constants import EntityType
from cews.dashboard import queries
from cews.database.connection import session_scope
from cews.database.models import Organization, ReviewQueueItem, Topic
from cews.normalization.review_actions import (
    AI_TOPIC_CANDIDATE,
    ReviewActionError,
    approve_review_item,
    pending_review_count,
    pending_review_items,
    reject_review_item,
    review_queue_types,
)

PAGE_SIZE = 25


def _subject(item: ReviewQueueItem) -> str:
    """A short, human label for a review item - the same fields ``cews review`` prints."""
    payload = item.payload_json or {}
    return str(
        payload.get("organizations")
        or payload.get("raw_name")
        or payload.get("label")
        or item.subject_ref
    )


def _named_organizations(payload: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for key in ("raw_name", "linked_to", "similar_to"):
        if payload.get(key):
            names.add(str(payload[key]))
    for key in ("organizations", "candidates"):
        for name in payload.get(key) or ():
            names.add(str(name))
    return names


def _evidence_for(session: Session, item: ReviewQueueItem) -> dict[str, list[dict[str, Any]]]:
    """Recent, real source records for every organization or topic this item names - resolved by
    exact canonical-name match, or by the item's own topic id for an AI topic candidate. A name
    CEWS has not resolved to anything yet simply gets no section; that is not an error, since an
    item can legitimately name something that does not exist as its own entity (a brand-new raw
    name with no organization row of its own)."""
    payload = item.payload_json or {}
    evidence: dict[str, list[dict[str, Any]]] = {}

    if item.queue_type == AI_TOPIC_CANDIDATE and item.subject_ref.startswith("topic:"):
        topic_id = int(item.subject_ref.removeprefix("topic:"))
        topic = session.get(Topic, topic_id)
        if topic is not None:
            evidence[topic.canonical_name] = queries.evidence_records(
                session, EntityType.TOPIC.value, topic_id, limit=5
            )
        return evidence

    for name in _named_organizations(payload):
        org = session.scalar(select(Organization).where(Organization.canonical_name == name))
        if org is not None:
            evidence[name] = queries.evidence_records(
                session, EntityType.COMPETITOR.value, org.id, limit=5
            )
    return evidence


def _render_evidence_links(evidence: dict[str, list[dict[str, Any]]]) -> None:
    if not evidence:
        st.caption("Nothing in CEWS yet to show evidence for.")
        return
    for name, records in evidence.items():
        st.markdown(f"**Source records for {name}:**")
        if not records:
            st.caption("No source records found yet.")
            continue
        for record in records:
            published = record["published"] or "date unknown"
            kind = record["type"].replace("_", " ")
            title = record["title"] or record["identifier"]
            if record["url"]:
                st.markdown(f"- {published} · {kind} · [{title}]({record['url']})")
            else:
                st.markdown(f"- {published} · {kind} · {title} _(no link available)_")


def _act(
    factory: sessionmaker[Session], item_id: int, action: str
) -> tuple[str | None, str | None]:
    """Approve or reject, in its own committed transaction. Returns (toast, error) - never both -
    so the caller can show the right one without an exception crossing a dialog/rerun boundary."""
    function = approve_review_item if action == "approve" else reject_review_item
    try:
        with session_scope(factory) as session:
            result = function(session, item_id)
    except ReviewActionError as exc:
        return None, str(exc)
    return f"[{result.item_id}] {result.decision}: {result.effect}", None


def _open_details_dialog(factory: sessionmaker[Session], item_id: int, key_prefix: str) -> None:
    @st.dialog(f"Review item {item_id}", width="large")
    def _dialog() -> None:
        with session_scope(factory) as session:
            item = session.get(ReviewQueueItem, item_id)
            if item is None or item.status != "pending":
                st.info("This item was already resolved - nothing left to decide.")
                return
            st.markdown(f"**{item.queue_type.replace('_', ' ')}** — {_subject(item)}")
            st.caption(f"Raised {item.created_at:%Y-%m-%d %H:%M} UTC")
            st.markdown("**Full detail**")
            payload = item.payload_json or {}
            if payload:
                for field_name, value in payload.items():
                    st.markdown(f"- **{field_name.replace('_', ' ')}:** {value}")
            else:
                st.caption("No extra detail was recorded for this item.")
            st.markdown("**Evidence**")
            _render_evidence_links(_evidence_for(session, item))

        approve_col, reject_col = st.columns(2)
        if approve_col.button(
            "Approve", key=f"{key_prefix}_dialog_approve_{item_id}", type="primary"
        ):
            toast, error = _act(factory, item_id, "approve")
            if error:
                st.error(error)
            else:
                st.cache_data.clear()
                st.toast(toast)
                st.rerun()
        if reject_col.button("Reject", key=f"{key_prefix}_dialog_reject_{item_id}"):
            toast, error = _act(factory, item_id, "reject")
            if error:
                st.error(error)
            else:
                st.cache_data.clear()
                st.toast(toast)
                st.rerun()

    _dialog()


def render_review_panel(
    factory: sessionmaker[Session], *, key_prefix: str, show_inline_actions: bool = True
) -> None:
    """Draw the review queue: total + filter + page controls, then one row per item with a
    Details button that opens the full popup. ``key_prefix`` keeps two panels on the same page
    (or the dashboard's Review page and the chat page's panel, in the same process) from
    colliding over the same ``st.session_state`` keys or widget keys.
    """
    offset_key, filter_key = f"{key_prefix}_offset", f"{key_prefix}_filter"
    st.session_state.setdefault(offset_key, 0)
    st.session_state.setdefault(filter_key, "All kinds")

    with session_scope(factory) as session:
        kinds = ["All kinds", *review_queue_types(session)]
    chosen_kind = st.selectbox(
        "Kind", kinds, key=filter_key, help="Narrow to one kind of pending decision."
    )
    queue_type = None if chosen_kind == "All kinds" else chosen_kind

    with session_scope(factory) as session:
        total = pending_review_count(session, queue_type=queue_type)
        offset = min(st.session_state[offset_key], max(total - 1, 0))
        items = pending_review_items(session, limit=PAGE_SIZE, offset=offset, queue_type=queue_type)
        rows = [(item.id, item.queue_type, _subject(item), item.created_at) for item in items]

    if total == 0:
        st.success("Nothing is waiting for a decision.")
        return

    first = offset + 1
    last = offset + len(rows)
    st.caption(f"Showing {first}-{last} of {total} pending item(s).")

    prev_col, next_col, _ = st.columns([1, 1, 4])
    if prev_col.button("← Previous", key=f"{key_prefix}_prev", disabled=offset == 0):
        st.session_state[offset_key] = max(offset - PAGE_SIZE, 0)
        st.rerun()
    if next_col.button("Next →", key=f"{key_prefix}_next", disabled=last >= total):
        st.session_state[offset_key] = offset + PAGE_SIZE
        st.rerun()

    for item_id, queue_type_name, subject, created_at in rows:
        with st.container(border=True):
            left, right = st.columns([5, 2])
            with left:
                st.markdown(f"**[{item_id}] {queue_type_name.replace('_', ' ')}** — {subject}")
                st.caption(f"Raised {created_at:%Y-%m-%d %H:%M} UTC")
            with right:
                if st.button("Details", key=f"{key_prefix}_details_{item_id}"):
                    _open_details_dialog(factory, item_id, key_prefix)
                if show_inline_actions:
                    approve_col, reject_col = st.columns(2)
                    if approve_col.button(
                        "✓", key=f"{key_prefix}_approve_{item_id}", help="Approve"
                    ):
                        toast, error = _act(factory, item_id, "approve")
                        if error:
                            st.error(error)
                        else:
                            st.cache_data.clear()
                            st.toast(toast)
                            st.rerun()
                    if reject_col.button("✗", key=f"{key_prefix}_reject_{item_id}", help="Reject"):
                        toast, error = _act(factory, item_id, "reject")
                        if error:
                            st.error(error)
                        else:
                            st.cache_data.clear()
                            st.toast(toast)
                            st.rerun()
