"""Unit tests for the RSS/Atom adapter: robots.txt, per-feed isolation, window filtering."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, date, datetime
from typing import Any

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from cews.constants import RunStatus
from cews.database.connection import create_memory_engine, create_session_factory
from cews.database.migrations import upgrade_database
from cews.ingestion.adapters.generic_rss import GenericRssAdapter
from cews.ingestion.errors import SourceParseError
from cews.ingestion.results import CollectionWindow
from support_sources import FEED_A, FEED_B, RssServer, build_adapter, rss_config

pytestmark = pytest.mark.unit

WINDOW = CollectionWindow(
    datetime(2026, 8, 1, tzinfo=UTC), datetime(2026, 9, 1, tzinfo=UTC), "incremental"
)


def adapter(server: RssServer | None = None, **config: Any) -> GenericRssAdapter:
    built = build_adapter(GenericRssAdapter, rss_config(**config), server or RssServer())
    assert isinstance(built, GenericRssAdapter)
    return built


@pytest.fixture
def factory() -> Iterator[sessionmaker[Session]]:
    engine: Engine = create_memory_engine()
    upgrade_database(engine)
    yield create_session_factory(engine)
    engine.dispose()


def collect(built: GenericRssAdapter, factory: sessionmaker[Session]) -> Any:
    return built.collect(factory, WINDOW)


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
def test_without_feeds_the_source_explains_what_to_configure() -> None:
    from support_sources import registry_config

    built = build_adapter(GenericRssAdapter, registry_config("generic_rss"), RssServer())
    problems = built.validate_configuration()
    assert len(problems) == 1
    assert "no feeds configured" in problems[0] and "source_registry.yaml" in problems[0]
    assert built.health_check_url() is None


def test_duplicate_feeds_and_too_many_feeds_are_reported() -> None:
    assert any("duplicates" in p for p in adapter(feeds=(FEED_A, FEED_A)).validate_configuration())
    crowded = adapter(feeds=(FEED_A, FEED_B), max_pages_per_run=1)
    assert any("max_pages_per_run" in p for p in crowded.validate_configuration())


def test_feed_organizations_option_must_map_urls_to_names() -> None:
    bad = adapter(options={"feed_organizations": {"url": 7}})
    assert any("feed_organizations" in p for p in bad.validate_configuration())


def test_configured_feeds_are_valid() -> None:
    assert adapter().validate_configuration() == []


# --------------------------------------------------------------------------------------
# robots.txt (RFC 9309)
# --------------------------------------------------------------------------------------
def test_missing_robots_file_allows_fetching() -> None:
    server = RssServer()  # host B has no robots.txt (404)
    assert adapter(server=server).robots_block_reason(FEED_B) is None


def test_robots_rules_are_followed() -> None:
    server = RssServer()
    server.robots["news.fixture-a.test"] = b"User-agent: *\nDisallow: /rss.xml\n"
    reason = adapter(server=server).robots_block_reason(FEED_A)
    assert reason is not None and "disallows /rss.xml" in reason


def test_robots_rules_for_other_paths_do_not_block_the_feed() -> None:
    server = RssServer()  # fixture robots.txt disallows /private/ only
    assert adapter(server=server).robots_block_reason(FEED_A) is None


def test_unreachable_robots_file_stops_the_fetch() -> None:
    server = RssServer()
    server.status_for[("news.fixture-a.test", "/robots.txt")] = 500
    reason = adapter(server=server).robots_block_reason(FEED_A)
    assert reason is not None and "could not be read" in reason


def test_robots_file_is_requested_once_per_host() -> None:
    server = RssServer()
    built = adapter(server=server)
    for _ in range(3):
        built.robots_block_reason(FEED_A)
    assert server.paths().count("/robots.txt") == 1


def test_robots_blocked_feed_is_an_error_not_a_crash(factory: sessionmaker[Session]) -> None:
    server = RssServer()
    server.robots["news.fixture-a.test"] = b"User-agent: *\nDisallow: /\n"
    result = collect(adapter(server=server), factory)
    assert result.status is RunStatus.PARTIAL
    assert any("robots.txt" in message for message in result.errors)
    assert result.records_inserted == 2  # the other feed still collected
    assert "/rss.xml" not in server.paths()


# --------------------------------------------------------------------------------------
# Parsing and window filtering
# --------------------------------------------------------------------------------------
def test_entries_inside_the_window_are_collected(factory: sessionmaker[Session]) -> None:
    result = collect(adapter(), factory)
    assert result.status is RunStatus.SUCCEEDED
    assert result.records_inserted == 4  # 2 per feed
    assert result.pages_fetched == 2  # one page per feed
    assert any("without a date were ignored" in w for w in result.warnings)


def test_entries_outside_the_window_are_skipped() -> None:
    built = adapter()
    page = built.fetch_page(built.build_query(WINDOW, None))
    parsed = built.parse_response(page)
    titles = [item["title"] for item in parsed.items]
    assert len(titles) == 2
    assert all("Older fixture release" not in t for t in titles)  # July entry
    assert all("Undated" not in t for t in titles)
    assert parsed.next_cursor == "1"  # the next feed


def test_entry_fields_are_cleaned() -> None:
    built = adapter()
    parsed = built.parse_response(built.fetch_page(built.build_query(WINDOW, None)))
    first = parsed.items[0]
    assert first["title"].startswith("Fixture Oncology doses first patient")
    assert first["summary"] is not None and "<b>" not in first["summary"]
    assert first["organization"] == "Fixture Oncology News"  # from the feed title
    assert first["tags"] == ["Clinical"]


def test_feed_organizations_option_names_the_company() -> None:
    built = adapter(options={"feed_organizations": {FEED_A: "Fixture Oncology, Inc."}})
    parsed = built.parse_response(built.fetch_page(built.build_query(WINDOW, None)))
    assert parsed.items[0]["organization"] == "Fixture Oncology, Inc."


def test_atom_feeds_are_supported() -> None:
    built = adapter()
    built.build_query(WINDOW, None)  # remembers the window
    page = built.fetch_page(built.build_query(WINDOW, "1"))
    parsed = built.parse_response(page)
    assert len(parsed.items) == 2 and parsed.next_cursor is None
    assert parsed.items[0]["organization"] == "Fixture Genetics Investor Relations"
    assert parsed.items[0]["updated"] == "2026-08-07T08:00:00+00:00"


def test_record_mapping(factory: sessionmaker[Session]) -> None:
    built = adapter()
    parsed = built.parse_response(built.fetch_page(built.build_query(WINDOW, None)))
    record = built.normalize_record(parsed.items[0])
    data, detail = record.data, record.detail
    assert data.record_type == "announcement"
    assert data.source_url == "https://news.fixture-a.test/releases/fx-101-phase-2"
    assert data.published_at == datetime(2026, 8, 11, 13, 0, tzinfo=UTC)
    assert detail["organization_name"] == "Fixture Oncology News"
    assert detail["announcement_date"] == date(2026, 8, 11)
    # classification happens later, so nothing is inferred here
    assert detail["announcement_type"] is None and detail["partner_organizations"] is None


def test_record_ids_are_stable_so_reruns_do_not_duplicate(factory: sessionmaker[Session]) -> None:
    first = collect(adapter(), factory)
    second = collect(adapter(), factory)
    assert first.records_inserted == 4
    assert (second.records_inserted, second.records_updated) == (0, 0)
    assert second.records_skipped == 4


def test_entries_without_a_title_or_date_are_rejected() -> None:
    with pytest.raises(SourceParseError, match="title and a date"):
        adapter().normalize_record({"feed_url": FEED_A, "title": None, "published": None})


def test_invalid_entry_links_are_dropped() -> None:
    item = {
        "feed_url": FEED_A,
        "entry_id": "x",
        "title": "Fixture",
        "published": "2026-08-11T13:00:00+00:00",
        "link": "javascript:alert(1)",
    }
    assert adapter().normalize_record(item).data.source_url is None


# --------------------------------------------------------------------------------------
# Failure handling
# --------------------------------------------------------------------------------------
def test_one_unreadable_feed_does_not_block_the_other(factory: sessionmaker[Session]) -> None:
    server = RssServer()
    server.feeds["news.fixture-a.test"] = b"<html><body>not a feed</body></html>"
    result = collect(adapter(server=server), factory)
    assert result.status is RunStatus.PARTIAL
    assert result.records_inserted == 2
    assert any("no RSS or Atom feed found" in message for message in result.errors)
    assert result.checkpoint is not None  # the window still completed


def test_one_failing_feed_host_does_not_block_the_other(factory: sessionmaker[Session]) -> None:
    server = RssServer()
    server.status_for[("news.fixture-a.test", "/rss.xml")] = 404
    result = collect(adapter(server=server), factory)
    assert result.status is RunStatus.PARTIAL
    assert result.records_inserted == 2
    assert result.page_error_count == 0  # per-feed problems do not trip the circuit breaker


def test_when_every_feed_fails_the_run_fails(factory: sessionmaker[Session]) -> None:
    server = RssServer()
    server.fail_always = True
    result = collect(adapter(server=server), factory)
    assert result.status is RunStatus.FAILED
    assert result.checkpoint is None and result.checkpoint_advanced is False
    assert result.page_error_count >= 2  # counted against the circuit breaker


def test_feed_index_out_of_range_is_reported() -> None:
    with pytest.raises(SourceParseError, match="out of range"):
        adapter().build_query(WINDOW, "9")


def test_pages_from_elsewhere_are_rejected() -> None:
    from cews.ingestion.results import RawPage, RequestSpec

    plain = RawPage(RequestSpec(FEED_A), 200, b"", {}, datetime.now(UTC))
    with pytest.raises(SourceParseError, match="GenericRssAdapter.fetch_page"):
        adapter().parse_response(plain)


def test_health_check_uses_the_first_feed() -> None:
    built = adapter()
    assert built.health_check_url() == FEED_A
    assert built.health_check().status.value == "ok"


def test_only_configured_feed_hosts_can_be_contacted() -> None:
    config = replace(rss_config(), feeds=(FEED_A,))
    built = build_adapter(GenericRssAdapter, config, RssServer())
    assert built.http.allowed_hosts == frozenset({"news.fixture-a.test"})
