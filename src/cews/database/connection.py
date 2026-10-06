"""Database engine and session helpers for SQLite (default) and PostgreSQL.

SQLite connections enable foreign keys (off by default in SQLite), WAL journaling for file
databases, and a busy timeout so the scheduler, API and dashboard can share one file.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event, inspect
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from cews.constants import DatabaseBackend
from cews.settings import Settings

LOGGER = logging.getLogger(__name__)

CORE_TABLE = "source_records"


class DatabaseError(RuntimeError):
    """Raised for database configuration or initialization problems."""


def build_database_url(settings: Settings) -> URL:
    """Return the SQLAlchemy URL for the configured backend.

    Raises:
        DatabaseError: if a PostgreSQL URL is required but cannot be parsed.
    """
    if settings.database_backend is DatabaseBackend.SQLITE:
        return URL.create("sqlite", database=str(settings.sqlite_path))
    try:
        return make_url(settings.database_url)
    except ArgumentError as exc:
        raise DatabaseError(f"DATABASE_URL is not a valid SQLAlchemy URL: {exc}") from exc


def describe_url(url: URL) -> str:
    """Return the URL as text with the password hidden (safe to log)."""
    return url.render_as_string(hide_password=True)


def _is_memory(url: URL) -> bool:
    return url.get_backend_name() == "sqlite" and url.database in (None, "", ":memory:")


def _install_sqlite_pragmas(engine: Engine, memory: bool) -> None:
    @event.listens_for(engine, "connect")
    def _on_connect(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        if not memory:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()


def create_db_engine(target: Settings | URL | str, echo: bool = False) -> Engine:
    """Create an engine from settings or a URL.

    File-based SQLite databases get their parent directory created. An in-memory SQLite URL
    uses a single shared connection so every session sees the same database.

    Raises:
        DatabaseError: if the URL is invalid or the SQLite directory cannot be created.
    """
    if isinstance(target, Settings):
        url = build_database_url(target)
    elif isinstance(target, str):
        try:
            url = make_url(target)
        except ArgumentError as exc:
            raise DatabaseError(f"invalid database URL: {exc}") from exc
    else:
        url = target

    kwargs: dict[str, Any] = {"echo": echo, "future": True}
    if url.get_backend_name() == "sqlite":
        memory = _is_memory(url)
        kwargs["connect_args"] = {"check_same_thread": False}
        if memory:
            kwargs["poolclass"] = StaticPool
        else:
            try:
                Path(str(url.database)).parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise DatabaseError(
                    f"cannot create directory for SQLite file {url.database}: {exc}"
                ) from exc
        engine = create_engine(url, **kwargs)
        _install_sqlite_pragmas(engine, memory)
    else:
        kwargs["pool_pre_ping"] = True
        engine = create_engine(url, **kwargs)
    LOGGER.debug("created engine for %s", describe_url(url))
    return engine


def create_memory_engine() -> Engine:
    """Create an in-memory SQLite engine (for tests and quick experiments)."""
    return create_db_engine("sqlite://")


def create_read_only_engine(settings: Settings) -> Engine:
    """An engine that cannot change the database, for anything that only ever reads it.

    Every connection is switched to read-only at the database itself (SQLite ``query_only``,
    PostgreSQL ``default_transaction_read_only``), so a bug or a hostile request cannot write
    even by accident: the write is refused by the database, not by application code that could
    be wrong. A missing SQLite file is an error rather than being quietly created empty.

    Raises:
        DatabaseError: if the SQLite file does not exist.
    """
    url = build_database_url(settings)
    is_file_db = url.get_backend_name() == "sqlite" and not _is_memory(url)
    if is_file_db and not Path(str(url.database)).is_file():
        raise DatabaseError(f"database not found at {url.database}; run: cews db-init")
    engine = create_db_engine(url)

    @event.listens_for(engine, "connect")
    def _read_only(dbapi_connection: Any, _record: Any) -> None:
        cursor = dbapi_connection.cursor()
        if engine.dialect.name == "sqlite":
            cursor.execute("PRAGMA query_only=ON")
        else:
            cursor.execute("SET default_transaction_read_only = on")
        cursor.close()

    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Return a session factory bound to ``engine`` (objects stay usable after commit)."""
    return sessionmaker(bind=engine, expire_on_commit=False, future=True)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Provide a transactional session: commit on success, roll back on any exception."""
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def database_is_initialized(engine: Engine) -> bool:
    """Return True when the core CEWS tables exist."""
    return inspect(engine).has_table(CORE_TABLE)
