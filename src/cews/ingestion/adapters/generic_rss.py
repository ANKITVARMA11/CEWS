"""Company announcements from official RSS/Atom feeds (press releases, investor relations).

Feeds are never built in. Add official feed URLs to ``generic_rss.feeds`` in
``config/source_registry.yaml``; ``options.feed_organizations`` can name the company behind each
feed (otherwise the feed's own title is used). Only listed feed hosts can be contacted.

Rules:

* **robots.txt** is honoured for every feed host, following RFC 9309: a missing robots.txt
  (HTTP 4xx) allows everything, an unreachable one (5xx or network error) means the feed is
  skipped for this run, and a ``Disallow`` that matches the feed path skips the feed.
* **One bad feed never blocks the others.** A feed that fails is recorded as an error (the run
  ends PARTIAL, the circuit breaker is not tripped). Only when every feed fails does the run
  count as FAILED.
* Entries are kept when their date falls inside the collection window. Entries without a date
  cannot be placed in time and are ignored (with a warning).
* Feed content is parsed from the downloaded bytes, so the parser never fetches anything itself.
  HTML in titles and summaries is reduced to plain text.

Announcement type and partner organizations are filled in later (Phase 5 / the optional AI
extraction layer); this adapter stores what the feed says, nothing inferred.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import feedparser
from sqlalchemy.orm import Session, sessionmaker

from cews.constants import RunStatus, SourceType
from cews.ingestion.adapters.common import clean_text, clip
from cews.ingestion.base import SourceAdapter, WriteGate, register_adapter
from cews.ingestion.errors import PermanentSourceError, SourceError, SourceParseError
from cews.ingestion.results import (
    CollectionWindow,
    NormalizedRecord,
    ParsedPage,
    RawPage,
    RequestSpec,
    SourceResult,
)
from cews.normalization.dates import to_utc_datetime
from cews.normalization.identifiers import is_valid_url

ROBOTS_AGENT = "CEWS"
MAX_SUMMARY_CHARS = 5000
FEED_ACCEPT = "application/rss+xml, application/atom+xml, application/xml;q=0.9, */*;q=0.5"


@dataclass(frozen=True)
class FeedPage(RawPage):
    """One downloaded feed (or the reason it could not be downloaded)."""

    feed_url: str = ""
    feed_index: int = 0
    error: str | None = None


@register_adapter
class GenericRssAdapter(SourceAdapter):
    """Announcements from the RSS/Atom feeds listed in the registry."""

    source_name = "generic_rss"
    source_type = SourceType.ANNOUNCEMENT

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Bind the adapter and start with an empty robots.txt cache."""
        super().__init__(*args, **kwargs)
        self._robots: dict[str, RobotFileParser | None] = {}
        self._window: CollectionWindow | None = None
        self._feeds_attempted = 0
        self._feeds_failed = 0

    # ---- configuration -------------------------------------------------------------------
    def validate_configuration(self) -> list[str]:
        """Feeds must be configured, fit in one run, and organization names must be strings."""
        if not self.config.feeds:
            return [
                f"{self.source_name}: no feeds configured; add official company or "
                "investor-relations RSS/Atom URLs under 'feeds' in config/source_registry.yaml"
            ]
        problems: list[str] = []
        if len(set(self.config.feeds)) != len(self.config.feeds):
            problems.append(f"{self.source_name}: the feeds list contains duplicates")
        if len(self.config.feeds) > self.config.max_pages_per_run:
            problems.append(
                f"{self.source_name}: {len(self.config.feeds)} feeds exceed max_pages_per_run="
                f"{self.config.max_pages_per_run}; raise it so every feed is read each run"
            )
        names = self.config.options.get("feed_organizations", {})
        if not isinstance(names, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in names.items()
        ):
            problems.append(f"{self.source_name}: options.feed_organizations must map URL to name")
        return problems

    def health_check_url(self) -> str | None:
        """The first configured feed."""
        return self.config.feeds[0] if self.config.feeds else None

    def _organization(self, feed_url: str, feed_title: str | None) -> str | None:
        names = self.config.options.get("feed_organizations") or {}
        return clean_text(names.get(feed_url)) or feed_title

    # ---- robots.txt ----------------------------------------------------------------------
    def _load_robots(self, origin: str) -> RobotFileParser | None:
        parser = RobotFileParser()
        try:
            response = self.http.get(f"{origin}/robots.txt")
        except PermanentSourceError as exc:
            if exc.status_code is not None and 400 <= exc.status_code < 500:
                parser.parse([])  # RFC 9309: "unavailable" allows everything
                return parser
            return None
        except SourceError:
            return None  # RFC 9309: "unreachable" means complete disallow
        parser.parse(response.text.splitlines())
        return parser

    def robots_block_reason(self, url: str) -> str | None:
        """Why robots.txt forbids fetching ``url``, or None when it is allowed."""
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._robots:
            self._robots[origin] = self._load_robots(origin)
        parser = self._robots[origin]
        if parser is None:
            return f"robots.txt for {parts.hostname} could not be read, so the feed was not fetched"
        if not parser.can_fetch(ROBOTS_AGENT, url):
            return f"robots.txt for {parts.hostname} disallows {parts.path or '/'}"
        return None

    # ---- request / response --------------------------------------------------------------
    def build_query(self, window: CollectionWindow, cursor: str | None) -> RequestSpec:
        """The feed at position ``cursor`` (feeds are the pages of this source)."""
        self._window = window
        index = int(cursor or 0)
        if not 0 <= index < len(self.config.feeds):
            raise SourceParseError(f"feed index {index} out of range")
        return RequestSpec(self.config.feeds[index], headers={"Accept": FEED_ACCEPT})

    def fetch_page(self, query: RequestSpec) -> RawPage:
        """Download one feed; a failure is returned as an error page, not raised."""
        index = self.config.feeds.index(query.url)
        self._feeds_attempted += 1
        now = datetime.now(UTC)
        try:
            reason = self.robots_block_reason(query.url)
            if reason:
                raise PermanentSourceError(reason)
            response = self.http.get(query.url, headers=query.headers)
        except SourceError as exc:
            self._feeds_failed += 1
            return FeedPage(
                query, 0, b"", {}, now, feed_url=query.url, feed_index=index, error=str(exc)
            )
        return FeedPage(
            query,
            response.status_code,
            response.content,
            dict(response.headers),
            now,
            feed_url=query.url,
            feed_index=index,
        )

    def parse_response(self, page: RawPage) -> ParsedPage:
        """Entries of one feed that fall inside the collection window."""
        if not isinstance(page, FeedPage):
            raise SourceParseError("feed pages must come from GenericRssAdapter.fetch_page")
        following = page.feed_index + 1
        next_cursor = str(following) if following < len(self.config.feeds) else None
        if page.error:
            return ParsedPage(
                (), next_cursor, errors=(f"feed {page.feed_url} skipped: {page.error}",)
            )

        parsed = feedparser.parse(page.content)
        # feedparser reports an HTML page or an empty body as a feed with no version rather than
        # as an error, so the version is what distinguishes "not a feed" from "feed with no items".
        problem: str | None = None
        if not parsed.get("version"):
            problem = "no RSS or Atom feed found (is the URL a web page?)"
        elif parsed.get("bozo") and not parsed.entries:
            problem = f"malformed feed ({type(parsed.get('bozo_exception')).__name__})"
        if problem:
            self._feeds_failed += 1
            return ParsedPage((), next_cursor, errors=(f"feed {page.feed_url}: {problem}",))
        feed_title = clean_text(parsed.feed.get("title"))
        organization = self._organization(page.feed_url, feed_title)
        items: list[dict[str, Any]] = []
        undated = 0
        for entry in parsed.entries:
            # dict.get bypasses feedparser's deprecated updated->published key fallback.
            published_raw = dict.get(entry, "published_parsed")
            updated_raw = dict.get(entry, "updated_parsed")
            when = to_utc_datetime(published_raw or updated_raw)
            if when is None:
                undated += 1
                continue
            if self._window is not None and not self._window.start <= when < self._window.end:
                continue
            updated = to_utc_datetime(updated_raw)
            items.append(
                {
                    "feed_url": page.feed_url,
                    "feed_title": feed_title,
                    "organization": organization,
                    "entry_id": clean_text(entry.get("id")),
                    "title": clean_text(entry.get("title")),
                    "summary": clip(clean_text(entry.get("summary")), MAX_SUMMARY_CHARS),
                    "link": entry.get("link"),
                    "published": when.isoformat(),
                    "updated": updated.isoformat() if updated else None,
                    "tags": [
                        t
                        for t in (clean_text(tag.get("term")) for tag in entry.get("tags", []))
                        if t
                    ],
                }
            )
        warnings = (
            (f"{page.feed_url}: {undated} entries without a date were ignored",) if undated else ()
        )
        return ParsedPage(tuple(items), next_cursor, None, warnings=warnings)

    def normalize_record(self, item: Mapping[str, Any]) -> NormalizedRecord:
        """One feed entry as an announcement record."""
        title = item.get("title")
        published = to_utc_datetime(item.get("published"))
        if not title or published is None:
            raise SourceParseError("feed entry needs a title and a date")
        key = item.get("entry_id") or item.get("link") or f"{title}|{item.get('published')}"
        record_id = hashlib.sha256(f"{item['feed_url']}\n{key}".encode()).hexdigest()[:40]
        link = item.get("link")
        return self.build_record(
            source_record_id=record_id,
            title=title,
            abstract=item.get("summary"),
            source_url=link if isinstance(link, str) and is_valid_url(link) else None,
            published_at=published,
            updated_at_source=to_utc_datetime(item.get("updated")),
            payload=dict(item),
            detail={
                "organization_name": clip(item.get("organization"), 255),
                "announcement_type": None,
                "announcement_date": published.date(),
                "partner_organizations": None,
                "detected_topics": None,
            },
        )

    # ---- run -----------------------------------------------------------------------------
    def collect(
        self,
        session_factory: sessionmaker[Session],
        window: CollectionWindow,
        *,
        dry_run: bool = False,
        write_lock: WriteGate | None = None,
    ) -> SourceResult:
        """Collect every feed; the run counts as FAILED only when every feed failed."""
        self._feeds_attempted = 0
        self._feeds_failed = 0
        result = super().collect(session_factory, window, dry_run=dry_run, write_lock=write_lock)
        if self._feeds_attempted and self._feeds_failed >= self._feeds_attempted:
            # Reclassify the per-feed errors as page errors so the circuit breaker counts them.
            result.page_error_count += self._feeds_failed
            result.status = RunStatus.FAILED
            result.checkpoint = None
            result.checkpoint_advanced = False
        return result
