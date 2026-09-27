"""Integration tests for idempotent upserts, detail rows and data-origin guards."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from cews.database.models import ClinicalTrial, SourceRecord
from cews.database.repositories import (
    MixedDataError,
    RecordValidationError,
    SourceRecordData,
    UpsertAction,
    assert_origin_homogeneous,
    count_records_by_origin,
    reset_synthetic_data,
    upsert_detail,
    upsert_source_record,
    upsert_source_records,
)

pytestmark = pytest.mark.integration

NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def _data(key: str = "1", **overrides: object) -> SourceRecordData:
    base = SourceRecordData(
        source="pubmed",
        source_record_id=key,
        record_type="publication",
        content_hash="a" * 64,
        fetched_at=NOW,
        title="Title",
        abstract="Abstract",
        source_url=f"https://example.org/{key}",
        raw_payload={"k": key},
        published_at=NOW,
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


def _count(session: Session) -> int:
    return int(session.scalar(select(func.count()).select_from(SourceRecord)) or 0)


def test_insert_then_unchanged_then_update(session: Session) -> None:
    first = upsert_source_record(session, _data())
    assert first.action is UpsertAction.INSERTED and first.record.id is not None
    original_first_seen = first.record.first_seen_at

    again = upsert_source_record(session, _data())
    assert again.action is UpsertAction.UNCHANGED and again.record.id == first.record.id

    changed = upsert_source_record(session, _data(content_hash="b" * 64, title="New title"))
    assert changed.action is UpsertAction.UPDATED
    assert changed.record.title == "New title" and changed.record.content_hash == "b" * 64
    assert changed.record.first_seen_at == original_first_seen
    assert _count(session) == 1


def test_update_resets_processing_status(session: Session) -> None:
    record = upsert_source_record(session, _data()).record
    record.processing_status = "normalized"
    upsert_source_record(session, _data(content_hash="c" * 64))
    assert record.processing_status == "new"


def test_rerunning_a_batch_creates_no_duplicates(session: Session) -> None:
    items = [_data(str(i), content_hash=f"{i:064x}") for i in range(1200)]
    first = upsert_source_records(session, items, batch_size=500)
    assert (first.inserted, first.updated, first.unchanged) == (1200, 0, 0)
    second = upsert_source_records(session, items, batch_size=500)
    assert (second.inserted, second.updated, second.unchanged) == (0, 0, 1200)
    assert _count(session) == 1200


def test_duplicate_keys_inside_one_batch_collapse_to_one_row(session: Session) -> None:
    summary = upsert_source_records(session, [_data("x"), _data("x", content_hash="d" * 64)])
    assert [r.action for r in summary.results] == [UpsertAction.INSERTED, UpsertAction.UPDATED]
    assert _count(session) == 1


def test_same_key_in_different_sources_is_two_records(session: Session) -> None:
    upsert_source_records(session, [_data("x", source="pubmed"), _data("x", source="europe_pmc")])
    assert _count(session) == 2


def test_empty_batch_and_bad_batch_size(session: Session) -> None:
    assert upsert_source_records(session, []).results == []
    with pytest.raises(ValueError, match="batch_size"):
        upsert_source_records(session, [_data()], batch_size=0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"source": " "}, "source must not be empty"),
        ({"source_record_id": ""}, "source_record_id"),
        ({"record_type": "podcast"}, "record_type"),
        ({"content_hash": "short"}, "content_hash"),
        ({"source_url": "javascript:alert(1)"}, "source_url"),
        ({"published_at": datetime(2026, 1, 1)}, "timezone-aware"),
        ({"fetched_at": datetime(2026, 1, 1)}, "timezone-aware"),
    ],
)
def test_invalid_records_are_rejected_before_writing(
    session: Session, overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(RecordValidationError, match=message):
        upsert_source_records(session, [_data("ok"), _data("bad", **overrides)])
    assert _count(session) == 0


def test_record_without_optional_fields_is_accepted(session: Session) -> None:
    minimal = SourceRecordData(
        source="s",
        source_record_id="1",
        record_type="patent",
        content_hash="e" * 64,
        fetched_at=NOW,
    )
    assert upsert_source_record(session, minimal).action is UpsertAction.INSERTED


def test_detail_rows_are_created_and_updated(session: Session) -> None:
    record = upsert_source_record(session, _data(record_type="clinical_trial")).record
    upsert_detail(
        session, "clinical_trial", record.id, {"trial_identifier": "T1", "phase": "PHASE1"}
    )
    upsert_detail(
        session, "clinical_trial", record.id, {"trial_identifier": "T1", "phase": "PHASE2"}
    )
    session.flush()
    rows = session.scalars(select(ClinicalTrial)).all()
    assert len(rows) == 1 and rows[0].phase == "PHASE2"


def test_detail_validation(session: Session) -> None:
    record = upsert_source_record(session, _data()).record
    with pytest.raises(RecordValidationError, match="unknown columns"):
        upsert_detail(session, "publication", record.id, {"publication_identifier": "P", "nope": 1})
    with pytest.raises(RecordValidationError, match="no detail table"):
        upsert_detail(session, "podcast", record.id, {})


def test_origin_counts_and_mixed_data_guard(session: Session) -> None:
    upsert_source_records(session, [_data("live1"), _data("syn1", is_synthetic=True)])
    assert count_records_by_origin(session) == {"synthetic": 1, "live": 1}
    with pytest.raises(MixedDataError, match="live records"):
        assert_origin_homogeneous(session, synthetic=True)
    with pytest.raises(MixedDataError, match="synthetic records"):
        assert_origin_homogeneous(session, synthetic=False)
    assert_origin_homogeneous(session, synthetic=True, allow_mixed=True)


def test_guard_allows_matching_origin(session: Session) -> None:
    upsert_source_record(session, _data("syn", is_synthetic=True))
    assert_origin_homogeneous(session, synthetic=True)


def test_reset_removes_only_synthetic_rows(session: Session) -> None:
    live = upsert_source_record(session, _data("live")).record
    synthetic = upsert_source_record(
        session, _data("syn", is_synthetic=True, record_type="clinical_trial")
    ).record
    upsert_detail(session, "clinical_trial", synthetic.id, {"trial_identifier": "SYN-1"})
    session.flush()
    live_id, synthetic_id = live.id, synthetic.id
    removed = reset_synthetic_data(session)
    assert removed["source_records"] == 1
    session.expire_all()
    assert session.get(SourceRecord, live_id) is not None
    assert session.get(SourceRecord, synthetic_id) is None
    assert session.scalar(select(func.count()).select_from(ClinicalTrial)) == 0
