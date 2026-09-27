"""Load and validate ``config/source_registry.yaml``.

The registry describes each source: its type, the ``ENABLE_*`` flag that switches it on, its
base URL, rate limit, page size, how far one run may page, how collection windows are sliced,
and its refresh policy. Adapters read their settings from here, so tuning a source never
requires a code change.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import yaml

from cews.constants import SourceType
from cews.normalization.identifiers import is_valid_url
from cews.settings import Settings

RefreshMode = Literal["incremental", "full"]

DEFAULTS: dict[str, Any] = {
    "requests_per_second": 1.0,
    "burst": 1,
    "page_size": 100,
    "max_pages_per_run": 50,
    "window_slice_days": 30,
}
_SOURCE_KEYS = frozenset(
    {
        "id",
        "source_type",
        "env_flag",
        "base_url",
        "auth",
        "mvp",
        "refresh",
        "feeds",
        "allowed_hosts",
        "timeout_seconds",
        "max_retries",
        "options",
        *DEFAULTS,
    }
)
_AUTH_VALUES = frozenset(
    {"none", "optional_api_key", "api_key_required", "api_key_may_be_required"}
)


class RegistryError(ValueError):
    """Raised when the source registry is missing or invalid."""


@dataclass(frozen=True)
class SourceConfig:
    """Validated configuration of one source."""

    id: str
    source_type: SourceType
    env_flag: str
    base_url: str | None = None
    auth: str = "none"
    mvp: bool = False
    refresh_mode: RefreshMode = "incremental"
    full_refresh_window_days: int | None = None
    requests_per_second: float = 1.0
    burst: int = 1
    page_size: int = 100
    max_pages_per_run: int = 50
    window_slice_days: int | None = 30
    timeout_seconds: float | None = None
    max_retries: int | None = None
    feeds: tuple[str, ...] = ()
    extra_hosts: tuple[str, ...] = ()
    options: Mapping[str, Any] = field(default_factory=dict)

    @property
    def allowed_hosts(self) -> frozenset[str]:
        """Hosts adapters may contact: the base URL host, feed hosts and extra hosts."""
        hosts = {h.lower() for h in self.extra_hosts}
        for url in (self.base_url, *self.feeds):
            if url:
                hosts.add((urlsplit(url).hostname or "").lower())
        hosts.discard("")
        return frozenset(hosts)

    def is_enabled(self, settings: Settings) -> bool:
        """True when this source's ``ENABLE_*`` flag is on."""
        return bool(getattr(settings, self.env_flag.lower()))


@dataclass(frozen=True)
class SourceRegistry:
    """All configured sources, in file order."""

    version: int
    sources: tuple[SourceConfig, ...]

    def __iter__(self) -> Iterator[SourceConfig]:
        return iter(self.sources)

    def __len__(self) -> int:
        return len(self.sources)

    def get(self, source_id: str) -> SourceConfig:
        """Return the source with ``source_id``.

        Raises:
            KeyError: if no such source is configured.
        """
        for source in self.sources:
            if source.id == source_id:
                return source
        raise KeyError(f"unknown source {source_id!r}; configured: {', '.join(self.ids())}")

    def ids(self) -> list[str]:
        """Source ids in file order."""
        return [source.id for source in self.sources]

    def enabled(self, settings: Settings) -> list[SourceConfig]:
        """Sources whose ``ENABLE_*`` flag is on."""
        return [source for source in self.sources if source.is_enabled(settings)]


def _number(value: Any, label: str, *, minimum: float, integer: bool = False) -> Any:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RegistryError(f"{label} must be a number")
    if integer and not float(value).is_integer():
        raise RegistryError(f"{label} must be a whole number")
    if value < minimum:
        raise RegistryError(f"{label} must be at least {minimum:g}")
    return int(value) if integer else float(value)


def _url(value: Any, label: str) -> str:
    if not isinstance(value, str) or not is_valid_url(value):
        raise RegistryError(f"{label} must be an http(s) URL, got {value!r}")
    return value.rstrip("/")


def _parse_source(entry: Any, index: int, defaults: dict[str, Any]) -> SourceConfig:
    if not isinstance(entry, dict):
        raise RegistryError(f"sources[{index}] must be a mapping")
    source_id = entry.get("id")
    if not isinstance(source_id, str) or not source_id.strip():
        raise RegistryError(f"sources[{index}]: missing 'id'")
    label = f"source '{source_id}'"
    unknown = set(entry) - _SOURCE_KEYS
    if unknown:
        raise RegistryError(f"{label}: unknown keys {sorted(unknown)}")

    try:
        source_type = SourceType(str(entry.get("source_type", "")))
    except ValueError as exc:
        allowed = ", ".join(t.value for t in SourceType)
        raise RegistryError(f"{label}: source_type must be one of {allowed}") from exc

    env_flag = entry.get("env_flag")
    field_info = Settings.model_fields.get(str(env_flag).lower()) if env_flag else None
    if field_info is None or field_info.annotation is not bool:
        raise RegistryError(f"{label}: env_flag {env_flag!r} is not a known ENABLE_* setting")

    auth = entry.get("auth", "none")
    if auth not in _AUTH_VALUES:
        raise RegistryError(f"{label}: auth must be one of {sorted(_AUTH_VALUES)}")

    refresh = entry.get("refresh") or {}
    if not isinstance(refresh, dict) or set(refresh) - {"mode", "full_refresh_window_days"}:
        raise RegistryError(f"{label}: refresh accepts only 'mode' and 'full_refresh_window_days'")
    mode = refresh.get("mode", "incremental")
    if mode not in ("incremental", "full"):
        raise RegistryError(f"{label}: refresh.mode must be 'incremental' or 'full'")
    window_days = refresh.get("full_refresh_window_days")
    if mode == "full":
        if window_days is None:
            raise RegistryError(f"{label}: refresh.mode 'full' needs full_refresh_window_days")
        window_days = _number(
            window_days, f"{label}: full_refresh_window_days", minimum=1, integer=True
        )
    elif window_days is not None:
        raise RegistryError(f"{label}: full_refresh_window_days only applies to mode 'full'")

    merged = {**defaults, **{k: entry[k] for k in DEFAULTS if k in entry}}
    slice_days = merged["window_slice_days"]
    feeds = entry.get("feeds") or []
    extra_hosts = entry.get("allowed_hosts") or []
    options = entry.get("options") or {}
    if not isinstance(feeds, list) or not isinstance(extra_hosts, list):
        raise RegistryError(f"{label}: feeds and allowed_hosts must be lists")
    if not isinstance(options, dict):
        raise RegistryError(f"{label}: options must be a mapping")
    if not all(isinstance(h, str) and h.strip() for h in extra_hosts):
        raise RegistryError(f"{label}: allowed_hosts must be non-empty strings")

    return SourceConfig(
        id=source_id.strip(),
        source_type=source_type,
        env_flag=str(env_flag).upper(),
        base_url=_url(entry["base_url"], f"{label}: base_url") if entry.get("base_url") else None,
        auth=auth,
        mvp=bool(entry.get("mvp", False)),
        refresh_mode=mode,
        full_refresh_window_days=window_days,
        requests_per_second=_number(
            merged["requests_per_second"], f"{label}: requests_per_second", minimum=0.001
        ),
        burst=_number(merged["burst"], f"{label}: burst", minimum=1, integer=True),
        page_size=_number(merged["page_size"], f"{label}: page_size", minimum=1, integer=True),
        max_pages_per_run=_number(
            merged["max_pages_per_run"], f"{label}: max_pages_per_run", minimum=1, integer=True
        ),
        window_slice_days=(
            None
            if slice_days is None
            else _number(slice_days, f"{label}: window_slice_days", minimum=1, integer=True)
        ),
        timeout_seconds=(
            None
            if entry.get("timeout_seconds") is None
            else _number(entry["timeout_seconds"], f"{label}: timeout_seconds", minimum=0.1)
        ),
        max_retries=(
            None
            if entry.get("max_retries") is None
            else _number(entry["max_retries"], f"{label}: max_retries", minimum=0, integer=True)
        ),
        feeds=tuple(_url(url, f"{label}: feed") for url in feeds),
        extra_hosts=tuple(h.strip().lower() for h in extra_hosts),
        options=dict(options),
    )


def load_source_registry(path: Path) -> SourceRegistry:
    """Load and validate the registry file.

    Raises:
        RegistryError: for a missing file, invalid YAML, or any invalid entry.
    """
    if not path.is_file():
        raise RegistryError(f"source registry not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise RegistryError(f"cannot read source registry {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RegistryError(f"{path} must contain a mapping")
    defaults_raw = data.get("defaults") or {}
    if not isinstance(defaults_raw, dict) or set(defaults_raw) - set(DEFAULTS):
        raise RegistryError(f"defaults accepts only: {', '.join(sorted(DEFAULTS))}")
    defaults = {**DEFAULTS, **defaults_raw}
    entries = data.get("sources") or []
    if not isinstance(entries, list):
        raise RegistryError("'sources' must be a list")
    sources = tuple(_parse_source(entry, i, defaults) for i, entry in enumerate(entries))
    ids = [s.id for s in sources]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise RegistryError(f"duplicate source ids: {', '.join(duplicates)}")
    return SourceRegistry(version=int(data.get("registry_version", 1)), sources=sources)
