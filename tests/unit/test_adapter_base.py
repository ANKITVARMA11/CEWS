"""Unit tests for edge cases of the shared adapter logic in cews.ingestion.base."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import replace
from datetime import timedelta
from typing import Any

import pytest
from sqlalchemy import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from cews.constants import RunStatus, SourceType
from cews.database.connection import create_memory_engine, create_session_factory
from cews.database.repositories import UpsertSummary
from cews.ingestion import base
from cews.ingestion.base import SourceAdapter, build_http_client, register_adapter
from cews.ingestion.checkpoints import CheckpointState
from cews.ingestion.errors import AdapterConfigError
from cews.ingestion.results import (
    CollectionWindow,
    HealthStatus,
    NormalizedRecord,
    ParsedPage,
    RawPage,
)
from cews.settings import Settings, load_settings
from support_adapters import START, ExampleAdapter, FakeApi, example_config, make_items, no_sleep

pytestmark = pytest.mark.unit

WINDOW = CollectionWindow(START, START + timedelta(days=90), "full")


@pytest.fixture
def settings() -> Settings:
    return load_settings(env_file=None)


@pytest.fixture
def factory() -> Iterator[sessionmaker[Session]]:
    from cews.database.migrations import upgrade_database

    engine: Engine = create_memory_engine()
    upgrade_database(engine)
    yield create_session_factory(engine)
    engine.dispose()


def _make(
    cls: type[SourceAdapter], settings: Settings, api: FakeApi, **config: Any
) -> SourceAdapter:
    cfg = example_config(**config)
    http = build_http_client(settings, cfg, transport=api.transport(), sleep=no_sleep, jitter=False)
    return cls(settings, cfg, http)


def _collect(
    cls: type[SourceAdapter],
    settings: Settings,
    factory: sessionmaker[Session],
    api: FakeApi,
    **config: Any,
) -> Any:
    adapter = _make(cls, settings, api, **config)
    try:
        return adapter.collect(factory, WINDOW)
    finally:
        adapter.close()


# --------------------------------------------------------------------------------------
# Configuration and health
# --------------------------------------------------------------------------------------
def test_registry_entry_with_another_source_type_is_rejected(settings: Settings) -> None:
    with pytest.raises(AdapterConfigError, match="registry says clinical_trial"):
        ExampleAdapter(settings, example_config(source_type=SourceType.CLINICAL_TRIAL))


def test_missing_base_url_is_a_configuration_problem(settings: Settings) -> None:
    adapter = ExampleAdapter(settings, example_config(base_url=None))
    try:
        assert adapter.validate_configuration() == [
            "example_source: no base_url or feeds configured"
        ]
    finally:
        adapter.close()


def test_health_check_without_a_url_reports_down(settings: Settings) -> None:
    class NoUrl(ExampleAdapter):
        health_check_url = SourceAdapter.health_check_url  # type: ignore[assignment]

    adapter = NoUrl(settings, example_config(base_url=None))
    try:
        health = adapter.health_check()
    finally:
        adapter.close()
    assert health.status is HealthStatus.DOWN and health.message == "no URL configured"


# --------------------------------------------------------------------------------------
# Record checks and de-duplication
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"source": "someone_else"}, "record source"),
        ({"record_type": "patent"}, "record type"),
        ({"is_synthetic": True}, "synthetic"),
    ],
)
def test_records_that_misrepresent_their_origin_are_rejected(
    settings: Settings, factory: sessionmaker[Session], change: dict[str, Any], message: str
) -> None:
    class Liar(ExampleAdapter):
        def normalize_record(self, item: Mapping[str, Any]) -> NormalizedRecord:
            record = super().normalize_record(item)
            return NormalizedRecord(replace(record.data, **change), record.detail)

    result = _collect(Liar, settings, factory, FakeApi(items=make_items(3)))
    assert result.records_inserted == 0 and result.record_error_count == 3
    assert message in result.errors[0]


def test_duplicates_across_pages_are_counted_once(
    settings: Settings, factory: sessionmaker[Session]
) -> None:
    items = make_items(8)
    items[7] = {**items[1], "published": items[7]["published"]}  # same id, served on page 2
    result = _collect(
        ExampleAdapter, settings, factory, FakeApi(items=items), window_slice_days=None
    )
    assert result.duplicate_count == 1
    assert result.records_inserted == 7
    assert result.records_received == 8


def test_error_messages_are_capped(settings: Settings, factory: sessionmaker[Session]) -> None:
    items = make_items(25)
    api = FakeApi(items=items, malformed_ids={item["id"] for item in items})
    result = _collect(ExampleAdapter, settings, factory, api)
    assert result.record_error_count == 25
    assert len(result.errors) == base.MAX_MESSAGES + 1
    assert result.errors[-1] == "further messages suppressed"


# --------------------------------------------------------------------------------------
# Page-level failures
# --------------------------------------------------------------------------------------
def test_parse_response_must_return_a_parsed_page(
    settings: Settings, factory: sessionmaker[Session]
) -> None:
    class WrongType(ExampleAdapter):
        def parse_response(self, page: RawPage) -> ParsedPage:
            return {"items": []}  # type: ignore[return-value]

    result = _collect(WrongType, settings, factory, FakeApi(items=make_items(3)))
    assert result.status is RunStatus.FAILED
    assert "must return a ParsedPage" in result.errors[0]


def test_adapter_bug_while_parsing_is_contained(
    settings: Settings, factory: sessionmaker[Session]
) -> None:
    class Buggy(ExampleAdapter):
        def parse_response(self, page: RawPage) -> ParsedPage:
            raise ZeroDivisionError("oops")

    result = _collect(Buggy, settings, factory, FakeApi(items=make_items(3)))
    assert result.status is RunStatus.FAILED and "ZeroDivisionError" in result.errors[0]


def test_records_requested_falls_back_to_page_size_without_totals(
    settings: Settings, factory: sessionmaker[Session]
) -> None:
    class NoTotals(ExampleAdapter):
        def parse_response(self, page: RawPage) -> ParsedPage:
            parsed = super().parse_response(page)
            return ParsedPage(parsed.items, parsed.next_cursor, None)

    result = _collect(
        NoTotals, settings, factory, FakeApi(items=make_items(12)), window_slice_days=None
    )
    assert result.pages_fetched == 3
    assert result.records_requested == 3 * 5  # page_size per page


def test_database_error_stops_the_run_and_keeps_earlier_pages(
    settings: Settings, factory: sessionmaker[Session]
) -> None:
    class FlakyDatabase(ExampleAdapter):
        calls = 0

        def persist_records(
            self, session: Session, records: list[NormalizedRecord]
        ) -> UpsertSummary:
            type(self).calls += 1
            if type(self).calls == 2:
                raise SQLAlchemyError("disk full")
            return super().persist_records(session, records)

    result = _collect(
        FlakyDatabase, settings, factory, FakeApi(items=make_items(15)), window_slice_days=None
    )
    assert result.status is RunStatus.PARTIAL
    assert result.records_inserted == 5  # page 1 stored before the failure
    assert "disk full" in result.errors[0]
    assert result.checkpoint is None  # the only slice never finished


# --------------------------------------------------------------------------------------
# collect_incremental
# --------------------------------------------------------------------------------------
def test_collect_incremental_starts_with_a_full_collection(
    settings: Settings, factory: sessionmaker[Session]
) -> None:
    now = START + timedelta(days=90)
    adapter = _make(ExampleAdapter, settings, FakeApi(items=make_items(10)))
    try:
        result = adapter.collect_incremental(factory, CheckpointState("example_source"), now=now)
    finally:
        adapter.close()
    assert result.status is RunStatus.SUCCEEDED and result.mode == "full"
    assert result.records_inserted == 10
    assert any("initial full collection" in w for w in result.warnings)


def test_collect_incremental_skips_when_a_full_refresh_is_not_due(
    settings: Settings, factory: sessionmaker[Session]
) -> None:
    now = START + timedelta(days=90)
    api = FakeApi(items=make_items(10))
    adapter = _make(ExampleAdapter, settings, api, refresh_mode="full", full_refresh_window_days=7)
    state = CheckpointState(
        "example_source",
        checkpoint={"watermark": now.isoformat(), "complete": True},
        last_full_refresh_at=now - timedelta(days=1),
    )
    try:
        result = adapter.collect_incremental(factory, state, now=now)
    finally:
        adapter.close()
    assert result.status is RunStatus.SKIPPED
    assert "next full refresh due" in result.warnings[0]
    assert api.requests == []


# --------------------------------------------------------------------------------------
# Adapter registration
# --------------------------------------------------------------------------------------
@pytest.fixture
def empty_registry(monkeypatch: pytest.MonkeyPatch) -> dict[str, type[SourceAdapter]]:
    registry: dict[str, type[SourceAdapter]] = {}
    monkeypatch.setattr(base, "_ADAPTER_CLASSES", registry)
    monkeypatch.setattr(base, "_discovered", True)
    return registry


def test_register_adapter_makes_it_discoverable(empty_registry: dict[str, Any]) -> None:
    class Registered(ExampleAdapter):
        source_name = "registered_source"

    assert register_adapter(Registered) is Registered
    assert register_adapter(Registered) is Registered  # registering twice is harmless
    assert base.adapter_classes() == {"registered_source": Registered}


def test_register_adapter_rejects_name_clashes_and_missing_names(
    empty_registry: dict[str, Any],
) -> None:
    class First(ExampleAdapter):
        source_name = "clash"

    class Second(ExampleAdapter):
        source_name = "clash"

    class Nameless(ExampleAdapter):
        source_name = ""

    register_adapter(First)
    with pytest.raises(TypeError, match="already registered"):
        register_adapter(Second)
    with pytest.raises(TypeError, match="must define source_name"):
        register_adapter(Nameless)
