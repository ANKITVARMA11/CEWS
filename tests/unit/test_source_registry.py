"""Unit tests for loading and validating config/source_registry.yaml."""

from __future__ import annotations

from pathlib import Path

import pytest

from cews.constants import SourceType
from cews.ingestion.registry import RegistryError, SourceConfig, load_source_registry
from cews.settings import load_settings
from support import REGISTRY_FILE

pytestmark = pytest.mark.unit

MINIMAL = """
registry_version: 2
defaults: {requests_per_second: 2.0, page_size: 50}
sources:
  - id: a
    source_type: publication
    env_flag: ENABLE_PUBMED
    base_url: https://api.example.test/v1/
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "registry.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_shipped_registry_is_valid() -> None:
    registry = load_source_registry(REGISTRY_FILE)
    assert registry.version == 2
    assert registry.ids() == [
        "clinical_trials_gov",
        "pubmed",
        "europe_pmc",
        "openalex",
        "nih_reporter",
        "generic_rss",
        "patents_uspto_bulk",
    ]
    assert registry.get("nih_reporter").refresh_mode == "full"
    assert registry.get("patents_uspto_bulk").full_refresh_window_days == 30
    assert registry.get("generic_rss").window_slice_days is None
    assert registry.get("generic_rss").feeds == ()


def test_enabled_sources_follow_settings() -> None:
    registry = load_source_registry(REGISTRY_FILE)
    defaults = load_settings(env_file=None)
    assert [s.id for s in registry.enabled(defaults)] == [
        "clinical_trials_gov",
        "pubmed",
        "europe_pmc",
        "generic_rss",
    ]
    changed = load_settings(
        env_file=None, overrides={"enable_pubmed": False, "enable_openalex": True}
    )
    assert [s.id for s in registry.enabled(changed)] == [
        "clinical_trials_gov",
        "europe_pmc",
        "openalex",
        "generic_rss",
    ]


def test_every_registry_flag_matches_settings_source_flags() -> None:
    from cews.constants import SOURCE_FLAGS

    registry = load_source_registry(REGISTRY_FILE)
    for config in registry:
        assert config.env_flag.lower() in SOURCE_FLAGS


def test_defaults_are_applied_and_overridable(tmp_path: Path) -> None:
    registry = load_source_registry(_write(tmp_path, MINIMAL))
    source = registry.get("a")
    assert source.requests_per_second == 2.0 and source.page_size == 50
    assert source.burst == 1 and source.max_pages_per_run == 50 and source.window_slice_days == 30
    assert source.base_url == "https://api.example.test/v1"  # trailing slash removed
    assert source.source_type is SourceType.PUBLICATION
    assert source.refresh_mode == "incremental"
    overridden = load_source_registry(
        _write(tmp_path, MINIMAL + "    page_size: 7\n    burst: 3\n")
    )
    assert overridden.get("a").page_size == 7 and overridden.get("a").burst == 3


def test_allowed_hosts_come_from_base_url_feeds_and_extras() -> None:
    config = SourceConfig(
        id="x",
        source_type=SourceType.ANNOUNCEMENT,
        env_flag="ENABLE_GENERIC_RSS",
        base_url="https://API.example.test/v1",
        feeds=("https://news.example.org/rss", "https://ir.example.com/feed"),
        extra_hosts=("cdn.example.net",),
    )
    assert config.allowed_hosts == {
        "api.example.test",
        "news.example.org",
        "ir.example.com",
        "cdn.example.net",
    }


def test_get_unknown_source_lists_known_ids(tmp_path: Path) -> None:
    registry = load_source_registry(_write(tmp_path, MINIMAL))
    with pytest.raises(KeyError, match="configured: a"):
        registry.get("zzz")


BAD_CASES = [
    (MINIMAL.replace("source_type: publication", "source_type: podcast"), "source_type"),
    (MINIMAL.replace("ENABLE_PUBMED", "ENABLE_NOTHING"), "env_flag"),
    (MINIMAL.replace("ENABLE_PUBMED", "FETCH_INTERVAL_MINUTES"), "env_flag"),
    (MINIMAL.replace("https://api.example.test/v1/", "ftp://x"), "base_url"),
    (MINIMAL + "    colour: blue\n", "unknown keys"),
    (MINIMAL + "    refresh: {mode: sometimes}\n", "refresh.mode"),
    (MINIMAL + "    refresh: {mode: full}\n", "full_refresh_window_days"),
    (MINIMAL + "    refresh: {mode: incremental, full_refresh_window_days: 3}\n", "only applies"),
    (MINIMAL + "    refresh: {mode: full, full_refresh_window_days: 0}\n", "at least 1"),
    (MINIMAL + "    refresh: {every: 2h}\n", "refresh accepts only"),
    (MINIMAL + "    page_size: 0\n", "page_size"),
    (MINIMAL + "    page_size: 2.5\n", "whole number"),
    (MINIMAL + "    requests_per_second: fast\n", "must be a number"),
    (MINIMAL + "    burst: true\n", "must be a number"),
    (MINIMAL + "    auth: magic\n", "auth"),
    (MINIMAL + "    feeds: https://one.example\n", "must be lists"),
    (MINIMAL + "    feeds: [not-a-url]\n", "feed"),
    (MINIMAL + "    options: [1]\n", "options"),
    (MINIMAL + "    allowed_hosts: ['']\n", "allowed_hosts"),
    (MINIMAL + "  - id: a\n    source_type: patent\n    env_flag: ENABLE_PATENTS\n", "duplicate"),
    (MINIMAL + "  - {source_type: patent, env_flag: ENABLE_PATENTS}\n", "missing 'id'"),
    (MINIMAL + "  - just a string\n", "must be a mapping"),
    (
        MINIMAL.replace(
            "defaults: {requests_per_second: 2.0, page_size: 50}", "defaults: {speed: 3}"
        ),
        "defaults",
    ),
    ("sources: {a: 1}\n", "must be a list"),
    ("- a\n- b\n", "mapping"),
    ("sources: [unclosed\n", "cannot read"),
]


@pytest.mark.parametrize(("text", "message"), BAD_CASES)
def test_invalid_registries_are_rejected(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(RegistryError, match=message):
        load_source_registry(_write(tmp_path, text))


def test_missing_registry_file(tmp_path: Path) -> None:
    with pytest.raises(RegistryError, match="not found"):
        load_source_registry(tmp_path / "nope.yaml")


def test_null_slice_days_and_overrides(tmp_path: Path) -> None:
    text = MINIMAL + "    window_slice_days: null\n    timeout_seconds: 12\n    max_retries: 0\n"
    source = load_source_registry(_write(tmp_path, text)).get("a")
    assert source.window_slice_days is None
    assert source.timeout_seconds == 12.0 and source.max_retries == 0
