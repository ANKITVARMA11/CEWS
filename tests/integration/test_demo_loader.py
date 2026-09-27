"""Integration tests for loading the synthetic dataset into the database."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime

import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from cews.database.connection import create_memory_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import (
    Announcement,
    ClinicalTrial,
    FundingAward,
    IngestionRun,
    Patent,
    Publication,
    SourceRecord,
)
from cews.database.repositories import (
    MixedDataError,
    SourceRecordData,
    count_records_by_origin,
    reset_synthetic_data,
    upsert_source_record,
)
from cews.demo.generator import DemoConfig, DemoDataset, generate_demo_dataset
from cews.demo.loader import load_demo_dataset

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def small_dataset() -> DemoDataset:
    return generate_demo_dataset(DemoConfig(scale=0.25))


@pytest.fixture
def factory() -> Iterator[sessionmaker[Session]]:
    memory: Engine = create_memory_engine()
    upgrade_database(memory)
    yield create_session_factory(memory)
    memory.dispose()


def _count(session: Session, model: type) -> int:
    return int(session.scalar(select(func.count()).select_from(model)) or 0)


def test_load_inserts_records_and_details(
    factory: sessionmaker[Session], small_dataset: DemoDataset
) -> None:
    with session_scope(factory) as session:
        summary = load_demo_dataset(session, small_dataset)
    assert summary.total == summary.inserted == len(small_dataset.records)
    by_type = small_dataset.count_by_type()
    with session_scope(factory) as session:
        assert _count(session, SourceRecord) == len(small_dataset.records)
        assert _count(session, ClinicalTrial) == by_type["clinical_trial"]
        assert _count(session, Publication) == by_type["publication"]
        assert _count(session, Patent) == by_type["patent"]
        assert _count(session, FundingAward) == by_type["funding"]
        assert _count(session, Announcement) == by_type["announcement"]
        assert count_records_by_origin(session) == {
            "synthetic": len(small_dataset.records),
            "live": 0,
        }


def test_loading_twice_is_idempotent(
    factory: sessionmaker[Session], small_dataset: DemoDataset
) -> None:
    with session_scope(factory) as session:
        load_demo_dataset(session, small_dataset)
    with session_scope(factory) as session:
        second = load_demo_dataset(session, small_dataset)
    assert (second.inserted, second.updated, second.unchanged) == (0, 0, len(small_dataset.records))
    with session_scope(factory) as session:
        assert _count(session, SourceRecord) == len(small_dataset.records)
        assert _count(session, IngestionRun) == 2  # one audit row per load


def test_loaded_rows_keep_source_spelling_and_flags(
    factory: sessionmaker[Session], small_dataset: DemoDataset
) -> None:
    with session_scope(factory) as session:
        load_demo_dataset(session, small_dataset)
    with session_scope(factory) as session:
        trial = session.scalars(select(ClinicalTrial)).first()
        assert trial is not None and trial.sponsor_name and trial.sponsor_organization_id is None
        record = session.get(SourceRecord, trial.source_record_id)
        assert record is not None
        assert record.is_synthetic is True and record.processing_status == "new"
        assert record.fetched_at == datetime(2026, 9, 1, tzinfo=UTC)
        assert record.raw_payload_json is not None and record.raw_payload_json["synthetic"] is True
        assert len(record.content_hash) == 64


def test_audit_row_describes_the_load(
    factory: sessionmaker[Session], small_dataset: DemoDataset
) -> None:
    with session_scope(factory) as session:
        summary = load_demo_dataset(session, small_dataset)
    with session_scope(factory) as session:
        run = session.scalars(select(IngestionRun)).one()
        assert run.job_id == summary.job_id and run.status == "succeeded" and run.is_synthetic
        assert run.records_inserted == len(small_dataset.records)
        assert run.checkpoint_json is not None and run.checkpoint_json["seed"] == 42


def test_live_data_blocks_the_demo_load(
    factory: sessionmaker[Session], small_dataset: DemoDataset
) -> None:
    live = SourceRecordData(
        source="pubmed",
        source_record_id="1",
        record_type="publication",
        content_hash="f" * 64,
        fetched_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    with session_scope(factory) as session:
        upsert_source_record(session, live)
    with pytest.raises(MixedDataError), session_scope(factory) as session:
        load_demo_dataset(session, small_dataset)
    with session_scope(factory) as session:
        assert _count(session, SourceRecord) == 1  # nothing was added
    with session_scope(factory) as session:
        load_demo_dataset(session, small_dataset, allow_mixed=True)
        assert count_records_by_origin(session)["live"] == 1


def test_reset_removes_synthetic_data_only(
    factory: sessionmaker[Session], small_dataset: DemoDataset
) -> None:
    with session_scope(factory) as session:
        load_demo_dataset(session, small_dataset, allow_mixed=True)
        upsert_source_record(
            session,
            SourceRecordData(
                source="pubmed",
                source_record_id="live",
                record_type="publication",
                content_hash="a" * 64,
                fetched_at=datetime(2026, 9, 1, tzinfo=UTC),
            ),
        )
    with session_scope(factory) as session:
        removed = reset_synthetic_data(session)
        assert removed["source_records"] == len(small_dataset.records)
    with session_scope(factory) as session:
        assert count_records_by_origin(session) == {"synthetic": 0, "live": 1}
        assert _count(session, ClinicalTrial) == 0 and _count(session, Publication) == 0


def test_changed_content_is_updated_not_duplicated(
    factory: sessionmaker[Session], small_dataset: DemoDataset
) -> None:
    with session_scope(factory) as session:
        load_demo_dataset(session, small_dataset)
    modified = generate_demo_dataset(DemoConfig(scale=0.25))
    first = modified.records[0]
    modified.records[0] = type(first)(**{**first.__dict__, "title": first.title + " (edited)"})
    with session_scope(factory) as session:
        summary = load_demo_dataset(session, modified)
    assert (summary.inserted, summary.updated) == (0, 1)
