"""Load a synthetic demo dataset into the database.

Loading is idempotent: records are upserted on ``(source, source_record_id)`` so running the
seed twice leaves the row counts unchanged. A load refuses to run against a database that
already holds live records unless explicitly allowed, so synthetic and live data never mix
silently. Each load writes an ``ingestion_runs`` audit row.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from cews.constants import RunStatus
from cews.database.models import IngestionRun
from cews.database.repositories import (
    SourceRecordData,
    assert_origin_homogeneous,
    upsert_detail,
    upsert_source_records,
)
from cews.demo.generator import DEMO_FETCHED_AT, DemoDataset, DemoRecord
from cews.normalization.deduplication import create_record_hash

LOGGER = logging.getLogger(__name__)

LOADER_SOURCE = "synthetic_demo_loader"


@dataclass(frozen=True)
class LoadSummary:
    """Counts from loading a dataset."""

    total: int
    inserted: int
    updated: int
    unchanged: int
    job_id: str


def _to_record_data(record: DemoRecord) -> SourceRecordData:
    return SourceRecordData(
        source=record.source,
        source_record_id=record.source_record_id,
        record_type=record.record_type,
        title=record.title,
        abstract=record.abstract,
        source_url=record.source_url,
        raw_payload=record.payload,
        published_at=record.published_at,
        updated_at_source=record.published_at,
        fetched_at=DEMO_FETCHED_AT,
        content_hash=create_record_hash(
            source=record.source,
            source_record_id=record.source_record_id,
            title=record.title,
            abstract=record.abstract,
            published_at=record.published_at,
            payload=record.payload,
        ),
        is_synthetic=True,
    )


def load_demo_dataset(
    session: Session, dataset: DemoDataset, allow_mixed: bool = False
) -> LoadSummary:
    """Upsert every record (and its detail row) and write an audit row.

    The caller owns the transaction (use ``session_scope``).

    Raises:
        MixedDataError: if live records exist and ``allow_mixed`` is false.
    """
    assert_origin_homogeneous(session, synthetic=True, allow_mixed=allow_mixed)
    started = datetime.now(UTC)
    job_id = uuid.uuid4().hex

    items = [_to_record_data(record) for record in dataset.records]
    summary = upsert_source_records(session, items)
    for record, result in zip(dataset.records, summary.results, strict=True):
        upsert_detail(session, record.record_type, result.record.id, dict(record.detail))

    session.add(
        IngestionRun(
            job_id=job_id,
            source=LOADER_SOURCE,
            start_time=started,
            end_time=datetime.now(UTC),
            status=RunStatus.SUCCEEDED.value,
            records_requested=len(items),
            records_received=len(items),
            records_inserted=summary.inserted,
            records_updated=summary.updated,
            records_skipped=summary.unchanged,
            duplicate_count=0,
            error_count=0,
            checkpoint_json={
                "seed": dataset.config.seed,
                "end_month": dataset.config.end_month.isoformat(),
                "months": dataset.config.months,
                "scale": dataset.config.scale,
            },
            warnings_json=["synthetic demo data"],
            is_synthetic=True,
        )
    )
    session.flush()
    LOGGER.info(
        "loaded demo data: %d records (%d inserted, %d updated, %d unchanged)",
        len(items),
        summary.inserted,
        summary.updated,
        summary.unchanged,
    )
    return LoadSummary(len(items), summary.inserted, summary.updated, summary.unchanged, job_id)
