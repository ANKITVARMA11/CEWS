"""Data-access helpers: idempotent record upserts, detail rows, and data-origin guards.

Upserts key on ``(source, source_record_id)`` and compare ``content_hash`` so re-running an
ingestion never creates duplicates and only rewrites records that actually changed.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, cast

from sqlalchemy import CursorResult, delete, func, inspect, select, tuple_
from sqlalchemy.orm import Session

from cews.constants import ProcessingStatus, SourceType
from cews.database.models import DETAIL_MODELS, SYNTHETIC_TABLES, Base, SourceRecord
from cews.normalization.identifiers import is_valid_url

LOGGER = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 500


class RecordValidationError(ValueError):
    """Raised when a record fails validation before being stored."""


class MixedDataError(RuntimeError):
    """Raised when synthetic and live data would be mixed without an explicit flag."""


class UpsertAction(StrEnum):
    """What an upsert did to a record."""

    INSERTED = "inserted"
    UPDATED = "updated"
    UNCHANGED = "unchanged"


@dataclass(frozen=True)
class SourceRecordData:
    """Input for :func:`upsert_source_record`."""

    source: str
    source_record_id: str
    record_type: str
    content_hash: str
    fetched_at: datetime
    title: str | None = None
    abstract: str | None = None
    source_url: str | None = None
    raw_payload: dict[str, Any] | None = None
    published_at: datetime | None = None
    updated_at_source: datetime | None = None
    is_synthetic: bool = False


@dataclass(frozen=True)
class UpsertResult:
    """The stored record and what happened to it."""

    record: SourceRecord
    action: UpsertAction


@dataclass
class UpsertSummary:
    """Counters over a batch of upserts."""

    inserted: int = 0
    updated: int = 0
    unchanged: int = 0
    results: list[UpsertResult] = field(default_factory=list)

    def add(self, result: UpsertResult) -> None:
        """Count one result."""
        self.results.append(result)
        if result.action is UpsertAction.INSERTED:
            self.inserted += 1
        elif result.action is UpsertAction.UPDATED:
            self.updated += 1
        else:
            self.unchanged += 1


def validate_source_record_data(data: SourceRecordData) -> None:
    """Validate a record before storing it.

    Raises:
        RecordValidationError: describing the first problem found.
    """
    if not data.source.strip():
        raise RecordValidationError("source must not be empty")
    if not data.source_record_id.strip():
        raise RecordValidationError(f"{data.source}: source_record_id must not be empty")
    if data.record_type not in {member.value for member in SourceType}:
        raise RecordValidationError(
            f"{data.source}/{data.source_record_id}: unknown record_type {data.record_type!r}"
        )
    if len(data.content_hash) != 64:
        raise RecordValidationError(
            f"{data.source}/{data.source_record_id}: content_hash must be a 64-character hex digest"
        )
    if data.source_url is not None and not is_valid_url(data.source_url):
        raise RecordValidationError(
            f"{data.source}/{data.source_record_id}: invalid source_url {data.source_url!r}"
        )
    for name in ("fetched_at", "published_at", "updated_at_source"):
        value = getattr(data, name)
        if value is not None and value.tzinfo is None:
            raise RecordValidationError(
                f"{data.source}/{data.source_record_id}: {name} must be timezone-aware"
            )


def _apply(row: SourceRecord, data: SourceRecordData) -> None:
    row.record_type = data.record_type
    row.source_url = data.source_url
    row.title = data.title
    row.abstract = data.abstract
    row.raw_payload_json = data.raw_payload
    row.published_at = data.published_at
    row.updated_at_source = data.updated_at_source
    row.fetched_at = data.fetched_at
    row.content_hash = data.content_hash
    row.is_synthetic = data.is_synthetic
    row.processing_status = ProcessingStatus.NEW.value


def upsert_source_record(session: Session, data: SourceRecordData) -> UpsertResult:
    """Insert or update one record. See :func:`upsert_source_records` for batches."""
    return upsert_source_records(session, [data]).results[0]


def _chunks(items: Sequence[SourceRecordData], size: int) -> Iterator[Sequence[SourceRecordData]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def upsert_source_records(
    session: Session, items: Sequence[SourceRecordData], batch_size: int = DEFAULT_BATCH_SIZE
) -> UpsertSummary:
    """Insert new records, update changed ones, and leave unchanged ones alone.

    Existing rows are looked up in batches, so the cost is one query per ``batch_size``
    records. Duplicate keys inside ``items`` are handled: the later item updates the earlier.
    The session is flushed (not committed) so callers get primary keys.

    Raises:
        RecordValidationError: if any item is invalid (nothing is written for that batch).
        ValueError: if ``batch_size`` is not positive.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for item in items:
        validate_source_record_data(item)

    summary = UpsertSummary()
    for chunk in _chunks(items, batch_size):
        keys = {(item.source, item.source_record_id) for item in chunk}
        existing: dict[tuple[str, str], SourceRecord] = {
            (row.source, row.source_record_id): row
            for row in session.scalars(
                select(SourceRecord).where(
                    tuple_(SourceRecord.source, SourceRecord.source_record_id).in_(keys)
                )
            )
        }
        for item in chunk:
            key = (item.source, item.source_record_id)
            row = existing.get(key)
            if row is None:
                row = SourceRecord(
                    source=item.source,
                    source_record_id=item.source_record_id,
                    first_seen_at=datetime.now(UTC),
                )
                _apply(row, item)
                session.add(row)
                existing[key] = row
                summary.add(UpsertResult(row, UpsertAction.INSERTED))
            elif row.content_hash == item.content_hash:
                summary.add(UpsertResult(row, UpsertAction.UNCHANGED))
            else:
                _apply(row, item)
                summary.add(UpsertResult(row, UpsertAction.UPDATED))
        session.flush()
    LOGGER.debug(
        "upserted %d records: %d inserted, %d updated, %d unchanged",
        len(items),
        summary.inserted,
        summary.updated,
        summary.unchanged,
    )
    return summary


def preview_source_records(
    session: Session, items: Sequence[SourceRecordData], batch_size: int = DEFAULT_BATCH_SIZE
) -> tuple[int, int, int]:
    """Return ``(would_insert, would_update, unchanged)`` without writing anything.

    Used by dry runs. Duplicate keys inside ``items`` are counted once, like the upsert.

    Raises:
        RecordValidationError: if any item is invalid.
        ValueError: if ``batch_size`` is not positive.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for item in items:
        validate_source_record_data(item)
    latest: dict[tuple[str, str], str] = {}
    for item in items:
        latest[(item.source, item.source_record_id)] = item.content_hash
    keys = list(latest)
    stored: dict[tuple[str, str], str] = {}
    for start in range(0, len(keys), batch_size):
        chunk = keys[start : start + batch_size]
        rows = session.execute(
            select(
                SourceRecord.source, SourceRecord.source_record_id, SourceRecord.content_hash
            ).where(tuple_(SourceRecord.source, SourceRecord.source_record_id).in_(chunk))
        )
        for source, record_id, content_hash in rows:
            stored[(source, record_id)] = content_hash
    new = sum(1 for key in keys if key not in stored)
    unchanged = sum(1 for key in keys if stored.get(key) == latest[key])
    return new, len(keys) - new - unchanged, unchanged


def upsert_detail(
    session: Session, record_type: str, source_record_id: int, values: dict[str, Any]
) -> Base:
    """Create or update the detail row (trial, publication, ...) for a source record.

    Raises:
        RecordValidationError: for an unknown record type or unknown column names.
    """
    model = DETAIL_MODELS.get(record_type)
    if model is None:
        raise RecordValidationError(f"no detail table for record_type {record_type!r}")
    columns = {column.key for column in inspect(model).columns}
    unknown = set(values) - columns
    if unknown:
        raise RecordValidationError(f"{model.__tablename__}: unknown columns {sorted(unknown)}")
    row = session.get(model, source_record_id)
    if row is None:
        row = model(source_record_id=source_record_id, **values)
        session.add(row)
    else:
        for name, value in values.items():
            setattr(row, name, value)
    return row


# --------------------------------------------------------------------------------------
# Data origin (synthetic vs live)
# --------------------------------------------------------------------------------------
def count_records_by_origin(session: Session) -> dict[str, int]:
    """Return ``{"synthetic": n, "live": m}`` counts of source records."""
    rows = session.execute(
        select(SourceRecord.is_synthetic, func.count()).group_by(SourceRecord.is_synthetic)
    ).all()
    counts = {"synthetic": 0, "live": 0}
    for is_synthetic, total in rows:
        counts["synthetic" if is_synthetic else "live"] = int(total)
    return counts


def assert_origin_homogeneous(session: Session, synthetic: bool, allow_mixed: bool = False) -> None:
    """Refuse to add ``synthetic`` (or live) data to a database holding the other kind.

    Raises:
        MixedDataError: unless ``allow_mixed`` is set.
    """
    if allow_mixed:
        return
    counts = count_records_by_origin(session)
    other = "live" if synthetic else "synthetic"
    if counts[other] > 0:
        adding = "synthetic demo" if synthetic else "live"
        raise MixedDataError(
            f"database already contains {counts[other]} {other} records; refusing to add {adding} "
            "records. Use a separate database (SQLITE_PATH), reset the demo data, or pass the "
            "explicit allow-mixed flag."
        )


def reset_synthetic_data(session: Session) -> dict[str, int]:
    """Delete every row flagged ``is_synthetic`` and return rows removed per table.

    Live rows are never touched. Child rows (details, evidence, links) are removed through
    foreign-key cascades. Deletion order respects foreign keys.
    """
    order = [
        "insights",
        "anomalies",
        "scores",
        "forecasts",
        "feature_values",
        "activity_aggregates",
        "evaluation_runs",
        "ingestion_runs",
        "source_records",
        "organizations",
    ]
    assert set(order) == set(SYNTHETIC_TABLES)
    tables = Base.metadata.tables
    removed: dict[str, int] = {}
    for name in order:
        table = tables[name]
        result = cast(
            CursorResult[Any], session.execute(delete(table).where(table.c.is_synthetic.is_(True)))
        )
        removed[name] = int(result.rowcount or 0)
    session.flush()
    return removed
