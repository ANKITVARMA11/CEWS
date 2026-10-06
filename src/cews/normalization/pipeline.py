"""Turning collected records into linked entities.

For every record that has not been normalized yet this pass:

1. pulls the organization names out of it (trial sponsor, patent assignee, funding recipient,
   announcing company, publication affiliations) and resolves each to an organization;
2. assigns taxonomy topics by matching the record's text and structured metadata;
3. marks records that duplicate the same work collected from another source.

Relationship confidence reflects how strong the evidence is. A trial sponsor is close to
certain; an author's affiliation only says someone at that organization co-wrote a paper, not
that the organization sponsored the work, so it is stored at low confidence and weighted down
in competitor discovery.

Re-running is safe: links are keyed on (record, organization, relationship) and (record, topic,
method), so nothing is duplicated, and only records still marked ``new`` are processed unless
``reprocess`` is set.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from sqlalchemy import CursorResult, select, update
from sqlalchemy.orm import Session
from sqlalchemy.orm.util import identity_key

from cews.constants import ProcessingStatus, RelationshipType, SourceType
from cews.database.models import (
    DETAIL_MODELS,
    Announcement,
    ClinicalTrial,
    FundingAward,
    Patent,
    Publication,
    RecordOrganization,
    RecordTopic,
    SourceRecord,
    Topic,
)
from cews.normalization.deduplication import DeduplicationSummary, find_duplicates
from cews.normalization.organizations import split_affiliation
from cews.normalization.resolver import OrganizationResolver
from cews.normalization.topics import Taxonomy, TopicMatcher
from cews.settings import Settings

LOGGER = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 500
MAX_AFFILIATIONS = 10
# How much a mention of an organization says about its involvement.
RELATIONSHIP_CONFIDENCE: dict[RelationshipType, float] = {
    RelationshipType.SPONSOR: 0.95,
    RelationshipType.ASSIGNEE: 0.95,
    RelationshipType.RECIPIENT: 0.9,
    RelationshipType.ANNOUNCER: 0.9,
    RelationshipType.PARTNER: 0.7,
    RelationshipType.AFFILIATION: 0.5,  # authorship is not sponsorship
}


@dataclass
class NormalizationSummary:
    """What one normalization pass did."""

    records_processed: int = 0
    organization_links: int = 0
    organizations_created: int = 0
    topic_links: int = 0
    records_without_topic: int = 0
    duplicates_marked: int = 0
    review_items: int = 0
    failures: int = 0
    by_match_method: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly copy."""
        return {
            "records_processed": self.records_processed,
            "organization_links": self.organization_links,
            "organizations_created": self.organizations_created,
            "topic_links": self.topic_links,
            "records_without_topic": self.records_without_topic,
            "duplicates_marked": self.duplicates_marked,
            "review_items": self.review_items,
            "failures": self.failures,
            "by_match_method": dict(self.by_match_method),
        }


@dataclass(frozen=True)
class Mention:
    """One organization named by a record."""

    raw_name: str
    relationship: RelationshipType


def _texts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [part.strip() for part in value.split(";") if part.strip()]
    if isinstance(value, list | tuple):
        return [str(item).strip() for item in value if item]
    return []


def organization_mentions(record: SourceRecord, detail: Any) -> list[Mention]:
    """Organization names a record contains, with what each name means."""
    payload: Mapping[str, Any] = record.raw_payload_json or {}
    mentions: list[Mention] = []

    def add(name: Any, relationship: RelationshipType) -> None:
        if isinstance(name, str) and name.strip():
            mentions.append(Mention(name.strip(), relationship))

    if isinstance(detail, ClinicalTrial):
        add(detail.sponsor_name, RelationshipType.SPONSOR)
        collaborators = (
            payload.get("protocolSection", {})
            .get("sponsorCollaboratorsModule", {})
            .get("collaborators", [])
            if isinstance(payload.get("protocolSection"), Mapping)
            else []
        )
        for collaborator in collaborators if isinstance(collaborators, list) else []:
            if isinstance(collaborator, Mapping):
                add(collaborator.get("name"), RelationshipType.PARTNER)
    elif isinstance(detail, Publication):
        for affiliation in (detail.affiliations or [])[:MAX_AFFILIATIONS]:
            for candidate in split_affiliation(str(affiliation)):
                add(candidate, RelationshipType.AFFILIATION)
    elif isinstance(detail, Patent):
        add(detail.assignee_name, RelationshipType.ASSIGNEE)
    elif isinstance(detail, FundingAward):
        add(detail.recipient_name, RelationshipType.RECIPIENT)
    elif isinstance(detail, Announcement):
        add(detail.organization_name, RelationshipType.ANNOUNCER)
        for partner in detail.partner_organizations or []:
            add(partner, RelationshipType.PARTNER)
    return mentions


def topic_fields(record: SourceRecord, detail: Any) -> dict[str, Any]:
    """The text a record offers for topic matching, by field name."""
    payload: Mapping[str, Any] = record.raw_payload_json or {}
    fields: dict[str, Any] = {"title": record.title, "abstract": record.abstract}
    if isinstance(detail, ClinicalTrial):
        fields["conditions"] = _texts(detail.condition)
        fields["interventions"] = _texts(detail.intervention)
        protocol = payload.get("protocolSection")
        if isinstance(protocol, Mapping):
            conditions = protocol.get("conditionsModule")
            if isinstance(conditions, Mapping):
                fields["keywords"] = conditions.get("keywords") or []
    elif isinstance(detail, Publication):
        fields["keywords"] = payload.get("keywords") or []
        fields["mesh_terms"] = payload.get("mesh_terms") or []
    elif isinstance(detail, Patent):
        fields["keywords"] = detail.patent_classifications or []
    elif isinstance(detail, Announcement):
        fields["keywords"] = payload.get("tags") or []
    return fields


def _pending_ids(session: Session, reprocess: bool, limit: int | None) -> list[int]:
    """The ids of the records to normalize, without loading the records themselves."""
    query = select(SourceRecord.id).order_by(SourceRecord.id)
    if not reprocess:
        query = query.where(SourceRecord.processing_status == ProcessingStatus.NEW.value)
    if limit is not None:
        query = query.limit(limit)
    return list(session.scalars(query))


def _release(session: Session, batch: Sequence[SourceRecord]) -> None:
    """Drop a committed batch's records and their detail rows from the session's memory.

    Only these are released: the organizations and alias rows the resolver is still updating
    must stay attached so later changes to them are saved.
    """
    for record in batch:
        model = DETAIL_MODELS.get(record.record_type)
        if model is not None:
            detail = session.identity_map.get(identity_key(model, record.id))
            if detail is not None:
                session.expunge(detail)
        session.expunge(record)


def normalize_records(
    session: Session,
    settings: Settings,
    taxonomy: Taxonomy,
    *,
    limit: int | None = None,
    reprocess: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    deduplicate: bool = True,
    progress: Callable[[int, int], None] | None = None,
    commit_each_batch: bool = False,
) -> NormalizationSummary:
    """Normalize pending records. The caller owns the transaction.

    Args:
        limit: process at most this many records.
        reprocess: also process records already marked normalized.
        deduplicate: run the cross-source duplicate pass afterwards.
        progress: called with (done, total) after each batch.
        commit_each_batch: commit after every batch instead of leaving that to the caller, and
            release the batch's records from memory. A run that is interrupted then keeps
            everything it finished and the next run carries on from there, instead of starting
            again from nothing; it also lets go of the database's single write slot between
            batches. Off by default because the caller normally owns the transaction.

    Raises:
        ValueError: if ``batch_size`` is not positive.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    pending = _pending_ids(session, reprocess, limit)
    total = len(pending)
    summary = NormalizationSummary()
    if not pending:
        LOGGER.info("no records to normalize")
        return summary

    resolver = OrganizationResolver(session, settings)
    matcher = TopicMatcher(taxonomy)
    topic_ids = {topic.key: topic.id for topic in session.scalars(select(Topic))}
    known_org_links: set[tuple[int, int, str]] = set()
    known_topic_links: set[tuple[int, int, str]] = set()

    for start in range(0, total, batch_size):
        batch_ids = pending[start : start + batch_size]
        batch = list(
            session.scalars(
                select(SourceRecord).where(SourceRecord.id.in_(batch_ids)).order_by(SourceRecord.id)
            )
        )
        # Links already stored (from an earlier pass, or a reprocess) must not be inserted twice.
        record_ids = [record.id for record in batch]
        known_org_links.update(
            tuple(row)
            for row in session.execute(
                select(
                    RecordOrganization.source_record_id,
                    RecordOrganization.organization_id,
                    RecordOrganization.relationship_type,
                ).where(RecordOrganization.source_record_id.in_(record_ids))
            ).all()
        )
        known_topic_links.update(
            tuple(row)
            for row in session.execute(
                select(
                    RecordTopic.source_record_id, RecordTopic.topic_id, RecordTopic.matching_method
                ).where(RecordTopic.source_record_id.in_(record_ids))
            ).all()
        )
        for record in batch:
            try:
                _normalize_one(
                    session,
                    record,
                    resolver=resolver,
                    matcher=matcher,
                    topic_ids=topic_ids,
                    known_org_links=known_org_links,
                    known_topic_links=known_topic_links,
                    summary=summary,
                )
                record.processing_status = ProcessingStatus.NORMALIZED.value
                summary.records_processed += 1
            except Exception as exc:  # one bad record must not stop the pass
                LOGGER.exception("normalizing record %s failed", record.id)
                record.processing_status = ProcessingStatus.FAILED.value
                summary.failures += 1
                LOGGER.debug("failure detail: %s", exc)
        session.flush()
        done = summary.records_processed + summary.failures
        if commit_each_batch:
            session.commit()
            _release(session, batch)
        LOGGER.info(
            "normalized %s of %s records (%s organizations created so far)",
            done,
            total,
            resolver.stats.created,
        )
        if progress is not None:
            progress(done, total)

    summary.organizations_created = resolver.stats.created
    summary.review_items = resolver.stats.review_items
    summary.by_match_method = dict(resolver.stats.by_method)
    if deduplicate:
        summary.duplicates_marked = deduplicate_records(session).duplicates
    LOGGER.info("normalization finished: %s", summary.as_dict())
    return summary


def _normalize_one(
    session: Session,
    record: SourceRecord,
    *,
    resolver: OrganizationResolver,
    matcher: TopicMatcher,
    topic_ids: Mapping[str, int],
    known_org_links: set[tuple[int, int, str]],
    known_topic_links: set[tuple[int, int, str]],
    summary: NormalizationSummary,
) -> None:
    model = DETAIL_MODELS.get(record.record_type)
    detail = session.get(model, record.id) if model is not None else None

    for mention in organization_mentions(record, detail):
        match = resolver.resolve(mention.raw_name, synthetic=record.is_synthetic)
        if match is None:
            continue
        marker = (record.id, match.organization.id, mention.relationship.value)
        if marker in known_org_links:
            continue
        known_org_links.add(marker)
        session.add(
            RecordOrganization(
                source_record_id=record.id,
                organization_id=match.organization.id,
                relationship_type=mention.relationship.value,
                match_method=match.method,
                confidence=round(
                    RELATIONSHIP_CONFIDENCE[mention.relationship] * match.confidence, 3
                ),
            )
        )
        summary.organization_links += 1

    matches = matcher.match(topic_fields(record, detail))
    if not matches:
        summary.records_without_topic += 1
    for topic_match in matches:
        topic_id = topic_ids.get(topic_match.topic_key)
        if topic_id is None:
            continue
        marker = (record.id, topic_id, topic_match.method)
        if marker in known_topic_links:
            continue
        known_topic_links.add(marker)
        session.add(
            RecordTopic(
                source_record_id=record.id,
                topic_id=topic_id,
                matching_method=topic_match.method,
                confidence=topic_match.confidence,
            )
        )
        summary.topic_links += 1


def deduplicate_records(
    session: Session, record_types: Sequence[str] | None = None
) -> DeduplicationSummary:
    """Point records at the canonical copy of the same work. Safe to re-run."""
    types = list(record_types or [SourceType.PUBLICATION.value])
    records = list(
        session.scalars(
            select(SourceRecord)
            .where(SourceRecord.record_type.in_(types))
            .order_by(SourceRecord.id)
        )
    )
    links = find_duplicates(records)
    by_reason: dict[str, int] = {}
    linked = {link.record_id: link for link in links}
    changed = 0
    cleared = 0
    for record in records:
        link = linked.get(record.id)
        if link is not None:
            if record.duplicate_of_id != link.canonical_id:
                record.duplicate_of_id = link.canonical_id
                record.duplicate_reason = link.reason
                changed += 1
            by_reason[link.reason] = by_reason.get(link.reason, 0) + 1
        elif record.duplicate_of_id is not None:
            record.duplicate_of_id = None
            record.duplicate_reason = None
            cleared += 1
    session.flush()
    LOGGER.info(
        "de-duplication: %d duplicate(s) among %d record(s) %s",
        len(links),
        len(records),
        by_reason,
    )
    return DeduplicationSummary(len(records), len(links), by_reason, cleared)


def reset_normalization(session: Session) -> int:
    """Mark every record as needing normalization again (used by ``--reprocess``)."""
    result = cast(
        "CursorResult[Any]",
        session.execute(update(SourceRecord).values(processing_status=ProcessingStatus.NEW.value)),
    )
    session.flush()
    return int(result.rowcount or 0)
