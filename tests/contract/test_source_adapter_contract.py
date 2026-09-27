"""Contract tests every source adapter must pass.

Each adapter contributes one :class:`ContractCase`: the adapter class, its registry entry (the
shipped one, so the real configuration is exercised), and a fake server that serves recorded
fixture pages from ``data/fixtures/sources``. The reference ``ExampleAdapter`` is included too.
Every registered adapter must have a case; ``test_every_registered_adapter_has_a_contract_case``
enforces it.

No test in this module can reach the network: all traffic goes through ``httpx.MockTransport``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import Engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from cews.constants import RunStatus, SourceType
from cews.database.connection import create_memory_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import SourceRecord
from cews.ingestion.adapters.clinical_trials_gov import ClinicalTrialsGovAdapter
from cews.ingestion.adapters.europe_pmc import EuropePmcAdapter
from cews.ingestion.adapters.generic_rss import GenericRssAdapter
from cews.ingestion.adapters.pubmed import PubMedAdapter
from cews.ingestion.base import SourceAdapter, adapter_classes, build_http_client
from cews.ingestion.errors import AdapterConfigError
from cews.ingestion.registry import SourceConfig, load_source_registry
from cews.ingestion.results import (
    CollectionWindow,
    HealthStatus,
    NormalizationResult,
    ParsedPage,
    RawPage,
    RequestSpec,
    SourceHealth,
    SourceResult,
    SourceStatistics,
)
from cews.normalization.identifiers import is_valid_url
from cews.settings import Settings, load_settings
from support import REPO_ROOT
from support_adapters import START, ExampleAdapter, FakeApi, example_config, make_items, no_sleep
from support_sources import (
    ClinicalTrialsServer,
    EuropePmcServer,
    PubMedServer,
    RssServer,
    registry_config,
    rss_config,
)

pytestmark = pytest.mark.contract

# The adapter interface from spec section 8.
REQUIRED_METHODS = (
    "validate_configuration",
    "health_check",
    "build_query",
    "fetch_page",
    "parse_response",
    "normalize_records",
    "deduplicate_records",
    "persist_records",
    "get_next_checkpoint",
    "collect",
    "collect_incremental",
    "get_source_statistics",
)
# The standard result fields from spec section 8.
RESULT_FIELDS = (
    "source_name",
    "collection_start",
    "collection_end",
    "status",
    "records_requested",
    "records_received",
    "records_inserted",
    "records_updated",
    "records_skipped",
    "duplicate_count",
    "error_count",
    "warnings",
    "checkpoint",
    "rate_limit",
)
SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class ContractCase:
    """Everything needed to exercise one adapter against fixtures."""

    adapter_cls: type[SourceAdapter]
    config: SourceConfig
    make_api: Callable[[], Any]  # FakeApi or a support_sources.FixtureServer
    window: CollectionWindow
    expected_records: int

    @property
    def name(self) -> str:
        return self.adapter_cls.source_name


def _window(start: datetime, days: int) -> CollectionWindow:
    return CollectionWindow(start, start + timedelta(days=days), "full")


AUGUST = datetime(2026, 8, 1, tzinfo=UTC)
CASES: tuple[ContractCase, ...] = (
    ContractCase(
        adapter_cls=ExampleAdapter,
        config=example_config(),
        make_api=lambda: FakeApi(items=make_items(23)),
        window=_window(START, 90),
        expected_records=23,
    ),
    ContractCase(
        adapter_cls=ClinicalTrialsGovAdapter,
        config=registry_config("clinical_trials_gov"),
        make_api=ClinicalTrialsServer,
        window=_window(AUGUST, 20),  # inside one 30-day slice
        expected_records=3,
    ),
    ContractCase(
        adapter_cls=PubMedAdapter,
        config=registry_config("pubmed", page_size=2),  # two esearch/efetch pages
        make_api=PubMedServer,
        window=_window(datetime(2026, 8, 10, tzinfo=UTC), 5),  # inside one 7-day slice
        expected_records=3,
    ),
    ContractCase(
        adapter_cls=EuropePmcAdapter,
        config=registry_config("europe_pmc", page_size=2),
        make_api=EuropePmcServer,
        window=_window(AUGUST, 20),
        expected_records=3,
    ),
    ContractCase(
        adapter_cls=GenericRssAdapter,
        config=rss_config(),
        make_api=RssServer,
        window=_window(AUGUST, 31),  # 2 + 2 entries in August; one July and one undated entry
        expected_records=4,
    ),
)


@pytest.fixture
def settings() -> Settings:
    return load_settings(env_file=None)


@pytest.fixture
def factory() -> Iterator[sessionmaker[Session]]:
    engine: Engine = create_memory_engine()
    upgrade_database(engine)
    yield create_session_factory(engine)
    engine.dispose()


def _adapter(case: ContractCase, settings: Settings, api: FakeApi) -> SourceAdapter:
    http = build_http_client(
        settings, case.config, transport=api.transport(), sleep=no_sleep, jitter=False
    )
    return case.adapter_cls(settings, case.config, http)


def _collect(
    case: ContractCase, settings: Settings, factory: sessionmaker[Session], api: FakeApi, **kw: bool
) -> SourceResult:
    adapter = _adapter(case, settings, api)
    try:
        return adapter.collect(factory, case.window, **kw)
    finally:
        adapter.close()


def _count(factory: sessionmaker[Session], source: str) -> int:
    with session_scope(factory) as session:
        query = select(func.count()).select_from(SourceRecord).where(SourceRecord.source == source)
        return int(session.scalar(query) or 0)


# --------------------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------------------
def test_every_registered_adapter_has_a_contract_case() -> None:
    missing = sorted(set(adapter_classes()) - {case.name for case in CASES})
    assert not missing, f"adapters without a contract case: {missing}"


def test_every_registered_adapter_is_in_the_source_registry() -> None:
    registry = load_source_registry(REPO_ROOT / "config" / "source_registry.yaml")
    for name, cls in adapter_classes().items():
        assert registry.get(name).source_type is cls.source_type, name


def test_contract_lists_match_the_framework() -> None:
    """Guards against a typo silently weakening the contract."""
    assert set(RESULT_FIELDS) <= set(SourceResult.__dataclass_fields__)
    assert all(hasattr(SourceAdapter, name) for name in REQUIRED_METHODS)


# --------------------------------------------------------------------------------------
# Per-adapter contract
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.name)
class TestAdapterContract:
    def test_identity(self, case: ContractCase) -> None:
        cls = case.adapter_cls
        assert SNAKE_CASE.match(cls.source_name)
        assert isinstance(cls.source_type, SourceType)
        assert case.config.id == cls.source_name
        assert case.config.source_type is cls.source_type
        assert case.config.env_flag.lower() in Settings.model_fields

    def test_implements_the_interface(self, case: ContractCase, settings: Settings) -> None:
        adapter = _adapter(case, settings, case.make_api())
        try:
            for name in REQUIRED_METHODS:
                assert callable(getattr(adapter, name, None)), f"missing {name}()"
            assert isinstance(adapter.enabled, bool)
            assert adapter.validate_configuration() == []
        finally:
            adapter.close()

    def test_query_fetch_parse_normalize(self, case: ContractCase, settings: Settings) -> None:
        adapter = _adapter(case, settings, case.make_api())
        try:
            first_slice = case.window.slices(case.config.window_slice_days)[0]
            query = adapter.build_query(first_slice, None)
            assert isinstance(query, RequestSpec) and is_valid_url(query.url)
            page = adapter.fetch_page(query)
            assert isinstance(page, RawPage) and page.status_code == 200
            parsed = adapter.parse_response(page)
            assert isinstance(parsed, ParsedPage) and parsed.items
            normalized = adapter.normalize_records(parsed.items)
            assert isinstance(normalized, NormalizationResult)
            assert normalized.errors == () and len(normalized.records) == len(parsed.items)
        finally:
            adapter.close()

    def test_collect_returns_a_standard_result(
        self, case: ContractCase, settings: Settings, factory: sessionmaker[Session]
    ) -> None:
        result = _collect(case, settings, factory, case.make_api())
        assert isinstance(result, SourceResult)
        for name in RESULT_FIELDS:
            assert hasattr(result, name), name
        assert result.status is RunStatus.SUCCEEDED, result.errors
        assert result.source_name == case.name
        assert result.collection_end is not None
        assert result.collection_end >= result.collection_start
        assert result.records_inserted == case.expected_records
        assert result.records_received == (
            result.records_inserted
            + result.records_updated
            + result.records_skipped
            + result.duplicate_count
            + result.record_error_count
        )
        assert result.checkpoint is not None and result.checkpoint["complete"] is True
        assert result.checkpoint_advanced is True
        assert result.rate_limit["requests"] >= 1
        json.dumps(result.to_dict())  # must be serializable for audit rows and the API

    def test_collect_is_idempotent(
        self, case: ContractCase, settings: Settings, factory: sessionmaker[Session]
    ) -> None:
        first = _collect(case, settings, factory, case.make_api())
        second = _collect(case, settings, factory, case.make_api())
        assert first.records_inserted == case.expected_records
        assert (second.records_inserted, second.records_updated) == (0, 0)
        assert second.records_skipped == case.expected_records
        assert _count(factory, case.name) == case.expected_records

    def test_stored_records_are_well_formed(
        self, case: ContractCase, settings: Settings, factory: sessionmaker[Session]
    ) -> None:
        _collect(case, settings, factory, case.make_api())
        with session_scope(factory) as session:
            rows = session.scalars(select(SourceRecord).where(SourceRecord.source == case.name))
            rows_list = list(rows)
        assert rows_list
        for row in rows_list:
            assert row.record_type == case.adapter_cls.source_type.value
            assert row.source_record_id.strip()
            assert len(row.content_hash) == 64
            assert row.is_synthetic is False
            assert row.fetched_at.tzinfo is not None
            assert row.source_url is None or is_valid_url(row.source_url)
            assert row.published_at is None or row.published_at.tzinfo is not None

    def test_dry_run_writes_nothing(
        self, case: ContractCase, settings: Settings, factory: sessionmaker[Session]
    ) -> None:
        result = _collect(case, settings, factory, case.make_api(), dry_run=True)
        assert result.dry_run is True and result.checkpoint_advanced is False
        assert result.records_inserted == case.expected_records  # "would insert"
        assert _count(factory, case.name) == 0

    def test_statistics_describe_stored_data(
        self, case: ContractCase, settings: Settings, factory: sessionmaker[Session]
    ) -> None:
        adapter = _adapter(case, settings, case.make_api())
        try:
            adapter.collect(factory, case.window)
            with session_scope(factory) as session:
                stats = adapter.get_source_statistics(session)
        finally:
            adapter.close()
        assert isinstance(stats, SourceStatistics)
        assert stats.total_records == case.expected_records
        assert stats.last_fetched_at is not None

    def test_health_check_reports_up_and_down_without_raising(
        self, case: ContractCase, settings: Settings
    ) -> None:
        broken_api = case.make_api()
        broken_api.fail_always = True
        healthy = _adapter(case, settings, case.make_api())
        broken = _adapter(case, settings, broken_api)
        try:
            up, down = healthy.health_check(), broken.health_check()
        finally:
            healthy.close()
            broken.close()
        assert isinstance(up, SourceHealth) and up.status is HealthStatus.OK
        assert up.latency_ms is not None and up.latency_ms >= 0
        assert down.status is HealthStatus.DOWN and down.message

    def test_source_failure_is_reported_not_raised(
        self, case: ContractCase, settings: Settings, factory: sessionmaker[Session]
    ) -> None:
        api = case.make_api()
        api.fail_always = True
        result = _collect(case, settings, factory, api)
        assert result.status is RunStatus.FAILED
        assert result.checkpoint is None and result.checkpoint_advanced is False
        assert result.errors
        assert _count(factory, case.name) == 0

    def test_adapter_rejects_a_foreign_registry_entry(
        self, case: ContractCase, settings: Settings
    ) -> None:
        with pytest.raises(AdapterConfigError):
            case.adapter_cls(settings, replace(case.config, id="someone_else"))
