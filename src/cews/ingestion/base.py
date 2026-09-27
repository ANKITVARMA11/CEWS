"""The source adapter contract and the shared collection logic.

Every source (ClinicalTrials.gov, PubMed, ...) subclasses :class:`SourceAdapter`, sets
``source_name`` and ``source_type``, and implements three methods:

* ``build_query(window, cursor)`` - the request for one page of one time slice;
* ``parse_response(page)`` - the items and next-page cursor in a response;
* ``normalize_record(item)`` - one item turned into a :class:`NormalizedRecord`
  (``build_record`` does the hashing and validation for you).

The base class does the rest: slicing the window, pagination, rate limiting and retries (via
:class:`~cews.ingestion.http_client.HttpClient`), per-record error isolation, de-duplication
within a run, per-page transactions, dry-run previews, and checkpoint computation.

Failure semantics:

* A record that cannot be normalized is dropped and counted; the run continues.
* A page that cannot be fetched or parsed (after retries) stops the run. Pages already stored
  stay stored, and the checkpoint advances only over slices that finished completely.
* Hitting ``max_pages_per_run`` is not an error: the run ends with ``truncated=True`` and the
  next run resumes from the checkpoint.
"""

from __future__ import annotations

import contextlib
import importlib
import logging
import pkgutil
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from typing import Any, ClassVar

import httpx
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from cews.constants import RunStatus, SourceType
from cews.database.connection import session_scope
from cews.database.models import IngestionRun, SourceRecord
from cews.database.repositories import (
    RecordValidationError,
    SourceRecordData,
    UpsertSummary,
    preview_source_records,
    upsert_detail,
    upsert_source_records,
    validate_source_record_data,
)
from cews.ingestion.checkpoints import CheckpointState, plan_window
from cews.ingestion.errors import AdapterConfigError, SourceError, SourceParseError
from cews.ingestion.http_client import HttpClient
from cews.ingestion.rate_limiter import RateLimiter
from cews.ingestion.registry import SourceConfig
from cews.ingestion.results import (
    CollectionWindow,
    HealthStatus,
    NormalizationResult,
    NormalizedRecord,
    ParsedPage,
    RawPage,
    RequestSpec,
    SourceHealth,
    SourceResult,
    SourceStatistics,
)
from cews.ingestion.stdlib_transport import STDLIB, StdlibTransport
from cews.normalization.deduplication import create_record_hash
from cews.settings import Settings

LOGGER = logging.getLogger(__name__)

WriteGate = AbstractContextManager[Any]
MAX_MESSAGES = 20
ADAPTERS_PACKAGE = "cews.ingestion.adapters"


def build_http_client(
    settings: Settings,
    config: SourceConfig,
    *,
    transport: httpx.BaseTransport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    jitter: bool = True,
) -> HttpClient:
    """Create the rate-limited, retrying client for one source.

    A source whose registry entry sets ``options.http_transport: stdlib`` sends its requests
    through :class:`~cews.ingestion.stdlib_transport.StdlibTransport`; see that module for why.
    An explicit ``transport`` (used by tests) always wins.
    """
    timeout = config.timeout_seconds or settings.source_request_timeout_seconds
    if transport is None and str(config.options.get("http_transport", "")).lower() == STDLIB:
        transport = StdlibTransport(timeout)
    return HttpClient(
        limiter=RateLimiter(config.requests_per_second, config.burst, clock=clock, sleep=sleep),
        timeout_seconds=timeout,
        max_retries=(
            config.max_retries if config.max_retries is not None else settings.source_max_retries
        ),
        allowed_hosts=config.allowed_hosts,
        transport=transport,
        sleep=sleep,
        jitter=jitter,
    )


def _append(messages: list[str], message: str) -> None:
    if len(messages) < MAX_MESSAGES:
        messages.append(message)
    elif len(messages) == MAX_MESSAGES:
        messages.append("further messages suppressed")


class SourceAdapter(ABC):
    """Base class for every source adapter."""

    source_name: ClassVar[str]
    source_type: ClassVar[SourceType]

    def __init__(
        self,
        settings: Settings,
        config: SourceConfig,
        http: HttpClient | None = None,
    ) -> None:
        """Bind the adapter to its settings and registry entry.

        Raises:
            AdapterConfigError: if the registry entry does not belong to this adapter.
        """
        if config.id != self.source_name:
            raise AdapterConfigError(
                f"registry entry '{config.id}' given to adapter '{self.source_name}'"
            )
        if config.source_type is not self.source_type:
            raise AdapterConfigError(
                f"{self.source_name}: registry says {config.source_type.value}, "
                f"adapter produces {self.source_type.value}"
            )
        self.settings = settings
        self.config = config
        self.http = http or build_http_client(settings, config)

    # ---- configuration and health ----------------------------------------------------
    @property
    def enabled(self) -> bool:
        """True when the source's ``ENABLE_*`` flag is on."""
        return self.config.is_enabled(self.settings)

    def validate_configuration(self) -> list[str]:
        """Return problems that prevent this source from running (empty means ready).

        Subclasses extend this, for example to require an API key.
        """
        problems: list[str] = []
        if self.config.base_url is None and not self.config.feeds:
            problems.append(f"{self.source_name}: no base_url or feeds configured")
        return problems

    def health_check_url(self) -> str | None:
        """URL requested by :meth:`health_check` (defaults to the base URL)."""
        return self.config.base_url or (self.config.feeds[0] if self.config.feeds else None)

    def health_check(self) -> SourceHealth:
        """Make one lightweight request and report whether the source is reachable."""
        now = datetime.now(UTC)
        url = self.health_check_url()
        if url is None:
            return SourceHealth(
                self.source_name, HealthStatus.DOWN, now, message="no URL configured"
            )
        started = time.perf_counter()
        try:
            response = self.http.get(url)
        except Exception as exc:  # a health check must never raise
            message = f"{type(exc).__name__}: {str(exc)[:200]}"
            return SourceHealth(self.source_name, HealthStatus.DOWN, now, message=message)
        latency = (time.perf_counter() - started) * 1000
        return SourceHealth(
            self.source_name,
            HealthStatus.OK,
            now,
            latency_ms=round(latency, 1),
            message=f"HTTP {response.status_code}",
        )

    # ---- source-specific steps --------------------------------------------------------
    @abstractmethod
    def build_query(self, window: CollectionWindow, cursor: str | None) -> RequestSpec:
        """Return the request for one page of ``window`` (``cursor`` is None for page one)."""

    def fetch_page(self, query: RequestSpec) -> RawPage:
        """Fetch one page through the shared HTTP client.

        Raises:
            TransientSourceError, PermanentSourceError: when the request fails.
        """
        response = self.http.get(query.url, params=query.params, headers=query.headers)
        return RawPage(
            request=query,
            status_code=response.status_code,
            content=response.content,
            headers=dict(response.headers),
            fetched_at=datetime.now(UTC),
        )

    @abstractmethod
    def parse_response(self, page: RawPage) -> ParsedPage:
        """Extract items and the next-page cursor.

        Raises:
            SourceParseError: if the response does not have the expected shape.
        """

    @abstractmethod
    def normalize_record(self, item: Mapping[str, Any]) -> NormalizedRecord:
        """Turn one item into a record. Raise any error to have the item skipped and counted."""

    # ---- shared steps ------------------------------------------------------------------
    def build_record(
        self,
        *,
        source_record_id: str,
        title: str | None = None,
        abstract: str | None = None,
        source_url: str | None = None,
        published_at: datetime | None = None,
        updated_at_source: datetime | None = None,
        payload: Mapping[str, Any] | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> NormalizedRecord:
        """Build a validated record with its content hash (helper for ``normalize_record``).

        Raises:
            RecordValidationError: if the record is invalid.
        """
        record_id = str(source_record_id).strip()
        data = SourceRecordData(
            source=self.source_name,
            source_record_id=record_id,
            record_type=self.source_type.value,
            content_hash=create_record_hash(
                source=self.source_name,
                source_record_id=record_id or "?",
                title=title,
                abstract=abstract,
                published_at=published_at,
                payload=payload,
            ),
            fetched_at=datetime.now(UTC),
            title=title,
            abstract=abstract,
            source_url=source_url,
            raw_payload=dict(payload) if payload is not None else None,
            published_at=published_at,
            updated_at_source=updated_at_source,
        )
        validate_source_record_data(data)
        return NormalizedRecord(data=data, detail=dict(detail or {}))

    def normalize_records(self, items: Iterable[Mapping[str, Any]]) -> NormalizationResult:
        """Normalize every item, isolating failures to the item that caused them."""
        records: list[NormalizedRecord] = []
        errors: list[str] = []
        for index, item in enumerate(items):
            try:
                record = self.normalize_record(item)
                self._check_record(record)
            except Exception as exc:  # one bad item must not stop the page
                errors.append(f"item {index}: {type(exc).__name__}: {str(exc)[:200]}")
                continue
            records.append(record)
        return NormalizationResult(tuple(records), tuple(errors))

    def _check_record(self, record: NormalizedRecord) -> None:
        data = record.data
        if data.source != self.source_name:
            raise RecordValidationError(f"record source {data.source!r} != {self.source_name!r}")
        if data.record_type != self.source_type.value:
            raise RecordValidationError(
                f"record type {data.record_type!r} != {self.source_type.value!r}"
            )
        if data.is_synthetic:
            raise RecordValidationError("live adapters must not produce synthetic records")
        validate_source_record_data(data)

    def deduplicate_records(
        self, records: Iterable[NormalizedRecord], seen: set[tuple[str, str]] | None = None
    ) -> tuple[list[NormalizedRecord], int]:
        """Drop records already seen in this run (by source and record id).

        Returns the unique records and the number of duplicates removed. ``seen`` is updated.
        """
        seen = seen if seen is not None else set()
        unique: list[NormalizedRecord] = []
        duplicates = 0
        for record in records:
            if record.key in seen:
                duplicates += 1
                continue
            seen.add(record.key)
            unique.append(record)
        return unique, duplicates

    def persist_records(self, session: Session, records: list[NormalizedRecord]) -> UpsertSummary:
        """Upsert records and their detail rows (idempotent)."""
        summary = upsert_source_records(session, [record.data for record in records])
        for record, result in zip(records, summary.results, strict=True):
            if record.detail:
                upsert_detail(
                    session, self.source_type.value, result.record.id, dict(record.detail)
                )
        session.flush()
        return summary

    def get_next_checkpoint(
        self, window: CollectionWindow, completed_until: datetime | None
    ) -> dict[str, Any] | None:
        """Return the checkpoint for a run that fully collected ``window`` up to ``completed_until``.

        None means no slice completed, so the checkpoint must not move.
        """
        if completed_until is None:
            return None
        return {
            "watermark": completed_until.isoformat(),
            "window_start": window.start.isoformat(),
            "window_end": window.end.isoformat(),
            "complete": completed_until >= window.end,
            "mode": window.mode,
        }

    # ---- collection --------------------------------------------------------------------
    def collect(
        self,
        session_factory: sessionmaker[Session],
        window: CollectionWindow,
        *,
        dry_run: bool = False,
        write_lock: WriteGate | None = None,
    ) -> SourceResult:
        """Collect ``window`` and store the records. Never raises for source problems.

        Args:
            session_factory: Used for one short transaction per page.
            window: The time range to collect; it is processed in slices, oldest first.
            dry_run: Fetch and parse, report what would change, write nothing.
            write_lock: Serializes database access when several sources run at once.
        """
        gate = write_lock if write_lock is not None else contextlib.nullcontext()
        result = SourceResult(
            source_name=self.source_name,
            collection_start=datetime.now(UTC),
            mode=window.mode,
            dry_run=dry_run,
        )
        self.http.reset_stats()
        seen: set[tuple[str, str]] = set()
        completed_until: datetime | None = None
        stopped = False

        for piece in window.slices(self.config.window_slice_days):
            cursor: str | None = None
            first_page = True
            while True:
                if result.pages_fetched >= self.config.max_pages_per_run:
                    result.truncated = True
                    stopped = True
                    break
                try:
                    query = self.build_query(piece, cursor)
                    page = self.fetch_page(query)
                    parsed = self.parse_response(page)
                    if not isinstance(parsed, ParsedPage):
                        raise SourceParseError("parse_response must return a ParsedPage")
                except (SourceError, AdapterConfigError) as exc:
                    self._page_failed(result, piece, exc)
                    stopped = True
                    break
                except Exception as exc:  # an adapter bug must not crash the whole fetch
                    self._page_failed(result, piece, exc)
                    stopped = True
                    break

                result.pages_fetched += 1
                LOGGER.info(
                    "%s %s page %d: %d record(s) (%d stored so far)",
                    self.source_name,
                    piece.start.date(),
                    result.pages_fetched,
                    len(parsed.items),
                    result.records_inserted + result.records_updated,
                )
                if first_page and parsed.total_available is not None:
                    result.records_requested += max(parsed.total_available, 0)
                elif parsed.total_available is None:
                    result.records_requested += query.page_size or len(parsed.items)
                first_page = False
                result.records_received += len(parsed.items)
                for note in parsed.warnings:
                    _append(result.warnings, note)
                for message in parsed.errors:
                    result.error_count += 1
                    _append(result.errors, f"{piece.start.date()}: {message}")

                normalized = self.normalize_records(parsed.items)
                for message in normalized.errors:
                    result.error_count += 1
                    _append(result.errors, f"{piece.start.date()}: {message}")
                unique, duplicates = self.deduplicate_records(normalized.records, seen)
                result.duplicate_count += duplicates

                try:
                    self._store(session_factory, unique, result, gate, dry_run)
                except (SQLAlchemyError, RecordValidationError) as exc:
                    result.records_received -= len(unique)
                    self._page_failed(result, piece, exc)
                    stopped = True
                    break

                cursor = parsed.next_cursor
                if not cursor:
                    completed_until = piece.end
                    result.slices_completed += 1
                    break
            if stopped:
                break

        self._finish(result, window, completed_until)
        return result

    def _store(
        self,
        session_factory: sessionmaker[Session],
        records: list[NormalizedRecord],
        result: SourceResult,
        gate: WriteGate,
        dry_run: bool,
    ) -> None:
        if not records:
            return
        with gate, session_scope(session_factory) as session:
            if dry_run:
                new, changed, unchanged = preview_source_records(session, [r.data for r in records])
                result.records_inserted += new
                result.records_updated += changed
                result.records_skipped += unchanged
                session.rollback()
                return
            summary = self.persist_records(session, records)
        result.records_inserted += summary.inserted
        result.records_updated += summary.updated
        result.records_skipped += summary.unchanged

    def _page_failed(self, result: SourceResult, piece: CollectionWindow, exc: Exception) -> None:
        result.error_count += 1
        result.page_error_count += 1
        message = (
            f"{piece.start.date()} to {piece.end.date()}: {type(exc).__name__}: {str(exc)[:300]}"
        )
        _append(result.errors, message)
        LOGGER.warning("%s page failed: %s", self.source_name, message)

    def _finish(
        self, result: SourceResult, window: CollectionWindow, completed_until: datetime | None
    ) -> None:
        checkpoint = self.get_next_checkpoint(window, completed_until)
        result.checkpoint = checkpoint
        result.checkpoint_advanced = checkpoint is not None and not result.dry_run
        if result.page_error_count and result.pages_fetched == 0:
            result.status = RunStatus.FAILED
        elif result.page_error_count or result.record_error_count:
            result.status = RunStatus.PARTIAL
        else:
            result.status = RunStatus.SUCCEEDED
        if result.truncated:
            _append(
                result.warnings,
                f"stopped after {self.config.max_pages_per_run} pages (max_pages_per_run); "
                "the next run continues from the checkpoint",
            )
        if result.dry_run:
            _append(result.warnings, "dry run: nothing was written")
        result.rate_limit = self.http.stats.as_dict()
        result.collection_end = datetime.now(UTC)
        LOGGER.info(
            "%s %s: %d received, %d inserted, %d updated, %d unchanged, %d errors",
            self.source_name,
            result.status.value,
            result.records_received,
            result.records_inserted,
            result.records_updated,
            result.records_skipped,
            result.error_count,
        )

    def collect_incremental(
        self,
        session_factory: sessionmaker[Session],
        state: CheckpointState,
        *,
        now: datetime | None = None,
        dry_run: bool = False,
        write_lock: WriteGate | None = None,
    ) -> SourceResult:
        """Collect from the checkpoint onwards (initial full collection if there is none)."""
        now = now or datetime.now(UTC)
        plan = plan_window(self.config, state, self.settings, now, requested="incremental")
        if plan.window is None:
            result = SourceResult(
                self.source_name, now, collection_end=now, status=RunStatus.SKIPPED, dry_run=dry_run
            )
            _append(result.warnings, plan.skip_reason or "nothing to collect")
            return result
        result = self.collect(session_factory, plan.window, dry_run=dry_run, write_lock=write_lock)
        for note in plan.notes:
            _append(result.warnings, note)
        return result

    def get_source_statistics(self, session: Session) -> SourceStatistics:
        """Summarize what the database holds for this source."""
        total, first, last, fetched = session.execute(
            select(
                func.count(),
                func.min(SourceRecord.published_at),
                func.max(SourceRecord.published_at),
                func.max(SourceRecord.fetched_at),
            ).where(SourceRecord.source == self.source_name)
        ).one()
        run = session.scalars(
            select(IngestionRun)
            .where(IngestionRun.source == self.source_name)
            .order_by(IngestionRun.start_time.desc())
        ).first()
        return SourceStatistics(
            source_name=self.source_name,
            total_records=int(total or 0),
            first_published_at=first,
            last_published_at=last,
            last_fetched_at=fetched,
            last_run_status=run.status if run else None,
            last_run_at=run.start_time if run else None,
        )

    def close(self) -> None:
        """Release network resources."""
        self.http.close()


# --------------------------------------------------------------------------------------
# Adapter discovery
# --------------------------------------------------------------------------------------
_ADAPTER_CLASSES: dict[str, type[SourceAdapter]] = {}
_discovered = False


def register_adapter(cls: type[SourceAdapter]) -> type[SourceAdapter]:
    """Class decorator that makes an adapter available to the orchestrator.

    Raises:
        TypeError: if two adapters claim the same ``source_name``.
    """
    name = getattr(cls, "source_name", None)
    if not name:
        raise TypeError(f"{cls.__name__} must define source_name")
    existing = _ADAPTER_CLASSES.get(name)
    if existing is not None and existing is not cls:
        raise TypeError(f"adapter name {name!r} already registered by {existing.__name__}")
    _ADAPTER_CLASSES[name] = cls
    return cls


def adapter_classes() -> dict[str, type[SourceAdapter]]:
    """Return registered adapters, importing every module in ``cews.ingestion.adapters`` once."""
    global _discovered
    if not _discovered:
        package = importlib.import_module(ADAPTERS_PACKAGE)
        for module in pkgutil.iter_modules(package.__path__):
            importlib.import_module(f"{ADAPTERS_PACKAGE}.{module.name}")
        _discovered = True
    return dict(_ADAPTER_CLASSES)
