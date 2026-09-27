"""Value types passed between the ingestion framework and source adapters.

* :class:`CollectionWindow` - the time range one run collects, split into slices.
* :class:`RequestSpec`, :class:`RawPage`, :class:`ParsedPage` - one page of a source response.
* :class:`NormalizedRecord` - a record ready to store (plus its detail row).
* :class:`SourceResult` - the standard outcome of one source collection (spec section 8).
* :class:`RunReport` - the outcome of a multi-source fetch.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Literal

from cews.constants import RunStatus
from cews.database.repositories import SourceRecordData
from cews.ingestion.errors import SourceParseError

WindowMode = Literal["full", "incremental", "resume"]


@dataclass(frozen=True)
class CollectionWindow:
    """A half-open time range ``[start, end)`` to collect, with the reason it was chosen."""

    start: datetime
    end: datetime
    mode: WindowMode = "incremental"

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("collection window bounds must be timezone-aware")
        if self.start >= self.end:
            raise ValueError("collection window start must be before its end")

    def slices(self, days: int | None) -> list[CollectionWindow]:
        """Split the window into consecutive slices of ``days`` (oldest first).

        ``None`` returns the window unchanged (for sources without date filtering).

        Raises:
            ValueError: if ``days`` is not positive.
        """
        if days is None:
            return [self]
        if days < 1:
            raise ValueError("slice length must be at least one day")
        step = timedelta(days=days)
        result: list[CollectionWindow] = []
        cursor = self.start
        while cursor < self.end:
            upper = min(cursor + step, self.end)
            result.append(CollectionWindow(cursor, upper, self.mode))
            cursor = upper
        return result


@dataclass(frozen=True)
class RequestSpec:
    """One HTTP GET an adapter wants to make."""

    url: str
    params: Mapping[str, Any] = field(default_factory=dict)
    headers: Mapping[str, str] = field(default_factory=dict)
    page_size: int | None = None


@dataclass(frozen=True)
class RawPage:
    """A response body exactly as the source returned it."""

    request: RequestSpec
    status_code: int
    content: bytes
    headers: Mapping[str, str]
    fetched_at: datetime

    @property
    def text(self) -> str:
        """The body decoded as UTF-8 (invalid bytes replaced)."""
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        """The body parsed as JSON.

        Raises:
            SourceParseError: if the body is not valid JSON.
        """
        try:
            return json.loads(self.content)
        except (ValueError, UnicodeDecodeError) as exc:
            raise SourceParseError(f"response is not valid JSON: {exc}") from exc


@dataclass(frozen=True)
class ParsedPage:
    """Items extracted from one page, the cursor for the next page, and the reported total.

    ``warnings`` are notes for the run result (for example "this slice hit the source's result
    cap"). ``errors`` are problems that lost data from this page without stopping the run (for
    example one RSS feed that could not be read); they are counted like record errors, so the
    run ends PARTIAL but the circuit breaker is not tripped.
    """

    items: tuple[Mapping[str, Any], ...]
    next_cursor: str | None = None
    total_available: int | None = None
    warnings: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class NormalizedRecord:
    """A validated record and the values for its type-specific detail row."""

    data: SourceRecordData
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        """The ``(source, source_record_id)`` identity of the record."""
        return (self.data.source, self.data.source_record_id)


@dataclass(frozen=True)
class NormalizationResult:
    """Records that normalized cleanly and one message per item that did not."""

    records: tuple[NormalizedRecord, ...]
    errors: tuple[str, ...] = ()


class HealthStatus(StrEnum):
    """Outcome of a source health check."""

    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"


@dataclass(frozen=True)
class SourceHealth:
    """Result of an adapter health check."""

    source_name: str
    status: HealthStatus
    checked_at: datetime
    latency_ms: float | None = None
    message: str = ""


@dataclass(frozen=True)
class SourceStatistics:
    """What the database holds for one source."""

    source_name: str
    total_records: int
    first_published_at: datetime | None
    last_published_at: datetime | None
    last_fetched_at: datetime | None
    last_run_status: str | None
    last_run_at: datetime | None


@dataclass
class SourceResult:
    """Standard outcome of one source collection.

    Counter invariant for non-dry runs: ``records_received == records_inserted +
    records_updated + records_skipped + duplicate_count + record_error_count``.
    In a dry run, inserted/updated/skipped mean *would* insert, update or skip.
    """

    source_name: str
    collection_start: datetime
    collection_end: datetime | None = None
    status: RunStatus = RunStatus.RUNNING
    mode: str | None = None
    dry_run: bool = False
    records_requested: int = 0
    records_received: int = 0
    records_inserted: int = 0
    records_updated: int = 0
    records_skipped: int = 0
    duplicate_count: int = 0
    error_count: int = 0
    page_error_count: int = 0
    pages_fetched: int = 0
    slices_completed: int = 0
    truncated: bool = False
    checkpoint: dict[str, Any] | None = None
    checkpoint_advanced: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    rate_limit: dict[str, Any] = field(default_factory=dict)

    @property
    def record_error_count(self) -> int:
        """Records dropped because they could not be normalized or validated."""
        return self.error_count - self.page_error_count

    @property
    def duration_seconds(self) -> float | None:
        """Elapsed time, once the run has finished."""
        if self.collection_end is None:
            return None
        return (self.collection_end - self.collection_start).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly representation (datetimes as ISO strings)."""
        data: dict[str, Any] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, datetime):
                value = value.isoformat()
            elif isinstance(value, StrEnum):
                value = value.value
            data[name] = value
        data["record_error_count"] = self.record_error_count
        return data


@dataclass
class RunReport:
    """Outcome of fetching several sources in one job."""

    job_id: str
    started_at: datetime
    trigger: str = "manual"
    dry_run: bool = False
    finished_at: datetime | None = None
    results: list[SourceResult] = field(default_factory=list)

    @property
    def status(self) -> RunStatus:
        """SUCCEEDED if nothing failed, FAILED if every source that ran failed, else PARTIAL.

        A run in which every source was skipped is SKIPPED.
        """
        ran = [r for r in self.results if r.status is not RunStatus.SKIPPED]
        if not ran:
            return RunStatus.SKIPPED
        if all(r.status is RunStatus.FAILED for r in ran):
            return RunStatus.FAILED
        if any(r.status in (RunStatus.FAILED, RunStatus.PARTIAL) for r in ran):
            return RunStatus.PARTIAL
        return RunStatus.SUCCEEDED

    def totals(self) -> dict[str, int]:
        """Summed counters across all sources."""
        keys = (
            "records_received",
            "records_inserted",
            "records_updated",
            "records_skipped",
            "duplicate_count",
            "error_count",
        )
        return {key: sum(int(getattr(r, key)) for r in self.results) for key in keys}

    def count_by_status(self) -> dict[str, int]:
        """Number of sources per result status."""
        counts: dict[str, int] = {}
        for result in self.results:
            counts[result.status.value] = counts.get(result.status.value, 0) + 1
        return counts
