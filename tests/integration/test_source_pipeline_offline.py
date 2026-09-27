"""The four real adapters running together through the orchestrator, entirely offline.

This is the live-mode pipeline with the shipped registry and the real adapter classes; only the
HTTP transport is replaced by fixture servers, so no test touches the network.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select

from cews.constants import RunStatus
from cews.database.connection import create_db_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import (
    Announcement,
    ClinicalTrial,
    IngestionRun,
    Publication,
    SourceRecord,
)
from cews.ingestion.base import SourceAdapter, adapter_classes, build_http_client
from cews.ingestion.checkpoints import read_checkpoint
from cews.ingestion.orchestrator import fetch_all_enabled_sources
from cews.ingestion.registry import SourceConfig, SourceRegistry, load_source_registry
from cews.settings import Settings, load_settings
from support import REGISTRY_FILE
from support_adapters import no_sleep
from support_sources import (
    FEED_A,
    FEED_B,
    ClinicalTrialsServer,
    EuropePmcServer,
    FixtureServer,
    PubMedServer,
    RssServer,
)

pytestmark = pytest.mark.integration

NOW = datetime(2026, 9, 1, tzinfo=UTC)
SERVERS: dict[str, type[FixtureServer]] = {
    "clinical_trials_gov": ClinicalTrialsServer,
    "pubmed": PubMedServer,
    "europe_pmc": EuropePmcServer,
    "generic_rss": RssServer,
}
EXPECTED = {"clinical_trials_gov": 3, "pubmed": 3, "europe_pmc": 3, "generic_rss": 4}


class Live:
    """The shipped registry and adapters, wired to fixture servers."""

    def __init__(self, tmp_path: Path, **overrides: Any) -> None:
        self.settings: Settings = load_settings(
            env_file=None,
            overrides={
                "source_registry_file": REGISTRY_FILE,
                "default_lookback_days": 40,  # two 30-day slices at most
                "incremental_lookback_days": 7,
                **overrides,
            },
        )
        self.engine = create_db_engine(f"sqlite:///{tmp_path / 'live.db'}")
        upgrade_database(self.engine)
        self.factory = create_session_factory(self.engine)
        self.servers: dict[str, FixtureServer] = {name: cls() for name, cls in SERVERS.items()}
        shipped = load_source_registry(REGISTRY_FILE)
        configs = []
        for config in shipped:
            if config.id in SERVERS:
                config = replace(config, requests_per_second=1000.0, burst=1000)
                if config.id == "generic_rss":
                    config = replace(config, feeds=(FEED_A, FEED_B))
            configs.append(config)
        self.registry = SourceRegistry(shipped.version, tuple(configs))

    def adapter(self, config: SourceConfig, settings: Settings) -> SourceAdapter | None:
        cls = adapter_classes().get(config.id)
        if cls is None:
            return None
        http = build_http_client(
            settings,
            config,
            transport=self.servers[config.id].transport(),
            sleep=no_sleep,
            jitter=False,
        )
        return cls(settings, config, http)

    def run(self, **kwargs: Any) -> Any:
        return fetch_all_enabled_sources(
            self.settings,
            self.factory,
            registry=self.registry,
            adapter_factory=self.adapter,
            now=kwargs.pop("now", NOW),
            **kwargs,
        )

    def count(self, model: type, **where: Any) -> int:
        with session_scope(self.factory) as session:
            query = select(func.count()).select_from(model)
            for name, value in where.items():
                query = query.where(getattr(model, name) == value)
            return int(session.scalar(query) or 0)

    def close(self) -> None:
        self.engine.dispose()


@pytest.fixture
def live(tmp_path: Path) -> Iterator[Live]:
    instance = Live(tmp_path)
    yield instance
    instance.close()


def test_all_four_sources_collect_together(live: Live) -> None:
    report = live.run()
    assert report.status is RunStatus.SUCCEEDED, [r.errors for r in report.results]
    by_source = {r.source_name: r for r in report.results}
    assert set(by_source) == set(EXPECTED)
    for name, expected in EXPECTED.items():
        assert by_source[name].records_inserted == expected, name
        assert live.count(SourceRecord, source=name) == expected
    assert live.count(ClinicalTrial) == 3
    assert live.count(Publication) == 6  # PubMed plus Europe PMC
    assert live.count(Announcement) == 4


def test_records_are_stored_with_their_details(live: Live) -> None:
    live.run()
    with session_scope(live.factory) as session:
        trial = session.scalars(
            select(ClinicalTrial).where(ClinicalTrial.trial_identifier == "NCT09000001")
        ).one()
        assert trial.sponsor_name == "Fixture Oncology, Inc." and trial.enrollment == 120
        record = session.get(SourceRecord, trial.source_record_id)
        assert record is not None and record.is_synthetic is False
        assert record.source_url == "https://clinicaltrials.gov/study/NCT09000001"
        publication = session.scalars(
            select(Publication).where(Publication.publication_identifier == "99000001")
        ).one()
        assert publication.journal == "Journal of Fixture Oncology"
        preprint = session.scalars(
            select(Publication).where(Publication.publication_identifier == "PPR:PPR990001")
        ).one()
        assert preprint.journal == "bioRxiv"


def test_no_personal_contact_data_is_stored(live: Live) -> None:
    live.run()
    with session_scope(live.factory) as session:
        blob = " ".join(
            str(r.raw_payload_json) + str(r.abstract) + str(r.title)
            for r in session.scalars(select(SourceRecord))
        )
    for leak in (
        "Placeholder Contact",
        "placeholder@example.invalid",
        "000-000-0000",
        "Placeholder Official",
        "site@example.invalid",
    ):
        assert leak not in blob


def test_second_run_adds_nothing_and_uses_incremental_windows(live: Live) -> None:
    live.run()
    later = NOW + timedelta(hours=2)
    report = live.run(now=later)
    assert report.status is RunStatus.SUCCEEDED
    totals = report.totals()
    assert totals["records_inserted"] == 0 and totals["records_updated"] == 0
    assert all(r.mode == "incremental" for r in report.results)
    by_source = {r.source_name: r for r in report.results}
    # The API fixtures answer any date range, so their records are seen again and recognised as
    # unchanged. The RSS feeds are filtered by entry date, and the fixture entries are older than
    # the 7-day incremental overlap, so that source correctly returns nothing the second time.
    for name in ("clinical_trials_gov", "pubmed", "europe_pmc"):
        assert by_source[name].records_skipped == EXPECTED[name], name
    assert by_source["generic_rss"].records_received == 0
    with session_scope(live.factory) as session:
        for name in EXPECTED:
            assert live.count(SourceRecord, source=name) == EXPECTED[name]
            assert read_checkpoint(session, name).watermark == later


def test_checkpoints_and_audit_rows_are_written(live: Live) -> None:
    report = live.run()
    with session_scope(live.factory) as session:
        for name in EXPECTED:
            state = read_checkpoint(session, name)
            assert state.watermark == NOW and state.complete
            assert state.last_status == "succeeded" and state.consecutive_failures == 0
        runs = list(
            session.scalars(select(IngestionRun).where(IngestionRun.job_id == report.job_id))
        )
    assert {run.source for run in runs} == set(EXPECTED)
    assert all(run.status == "succeeded" and run.collection_mode == "full" for run in runs)
    assert all(run.rate_limit_json and run.rate_limit_json["requests"] >= 1 for run in runs)


def test_one_source_failing_does_not_stop_the_rest(live: Live) -> None:
    live.servers["pubmed"].fail_always = True
    report = live.run()
    assert report.status is RunStatus.PARTIAL
    by_source = {r.source_name: r for r in report.results}
    assert by_source["pubmed"].status is RunStatus.FAILED
    assert live.count(SourceRecord, source="pubmed") == 0
    for name in ("clinical_trials_gov", "europe_pmc", "generic_rss"):
        assert by_source[name].status is RunStatus.SUCCEEDED
        assert live.count(SourceRecord, source=name) == EXPECTED[name]


def test_dry_run_touches_nothing(live: Live) -> None:
    report = live.run(dry_run=True)
    assert report.dry_run is True
    assert report.totals()["records_inserted"] == sum(EXPECTED.values())
    assert live.count(SourceRecord) == 0 and live.count(IngestionRun) == 0


def test_each_source_only_contacts_its_own_host(live: Live) -> None:
    live.run()
    hosts = {
        name: {request.url.host for request in server.requests}
        for name, server in live.servers.items()
    }
    assert hosts["clinical_trials_gov"] == {"clinicaltrials.gov"}
    assert hosts["pubmed"] == {"eutils.ncbi.nlm.nih.gov"}
    assert hosts["europe_pmc"] == {"www.ebi.ac.uk"}
    assert hosts["generic_rss"] == {"news.fixture-a.test", "ir.fixture-b.test"}


def test_robots_is_checked_once_per_feed_host(live: Live) -> None:
    live.run()
    robots = [p for p in live.servers["generic_rss"].paths() if p == "/robots.txt"]
    assert len(robots) == 2  # one per host, not per feed request


def test_selecting_one_source_leaves_the_others_alone(live: Live) -> None:
    report = live.run(sources=["clinical_trials_gov"])
    assert [r.source_name for r in report.results] == ["clinical_trials_gov"]
    assert live.count(SourceRecord) == EXPECTED["clinical_trials_gov"]
    assert live.servers["pubmed"].requests == []


def test_health_checks_report_every_source(live: Live) -> None:
    for name, server in live.servers.items():
        config = live.registry.get(name)
        adapter = live.adapter(config, live.settings)
        assert adapter is not None
        try:
            healthy = adapter.health_check()
            server.fail_always = True
            unhealthy = adapter.health_check()
        finally:
            adapter.close()
        assert healthy.status.value == "ok", name
        assert unhealthy.status.value == "down", name
