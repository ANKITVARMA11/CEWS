"""Exception types shared by the ingestion framework and every source adapter.

Adapters must raise these types (not raw ``httpx`` or ``json`` errors) so the framework can
decide what to do: retry, stop pagination, skip one record, or skip the whole source.
"""

from __future__ import annotations


class SourceError(RuntimeError):
    """Base class for problems while talking to a source."""


class TransientSourceError(SourceError):
    """A failure worth retrying: timeout, connection error, HTTP 429 or 5xx.

    ``retry_after`` holds the server's requested wait in seconds when it sent one.
    """

    def __init__(
        self, message: str, *, status_code: int | None = None, retry_after: float | None = None
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class PermanentSourceError(SourceError):
    """A failure that retrying will not fix: HTTP 4xx (except 429), blocked URL, bad request."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class SourceParseError(SourceError):
    """A response (or one record inside it) could not be parsed."""


class AdapterConfigError(ValueError):
    """An adapter cannot run with the current configuration (for example a missing API key)."""
