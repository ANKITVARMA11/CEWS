"""Shared pytest fixtures for CEWS.

* Every test runs with CEWS-related environment variables removed, so a developer's real
  environment or ``.env`` never changes test outcomes.
* Logging configuration done by a test is undone afterwards.
* ``engine``/``session`` fixtures provide a migrated in-memory SQLite database.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from cews.database.connection import create_memory_engine, create_session_factory
from cews.database.migrations import upgrade_database
from cews.settings import Settings, load_settings
from support import REGISTRY_FILE, SCORING_FILE, TAXONOMY_FILE


@pytest.fixture(autouse=True)
def _isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every CEWS setting from the process environment for the duration of a test."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)


@pytest.fixture(autouse=True)
def _block_real_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly if any test tries to reach the internet.

    Tests must use ``httpx.MockTransport`` (see ``support_adapters.FakeApi``). Only httpx's real
    network transports are blocked, so mock transports keep working.
    """

    def refuse(self: object, request: httpx.Request) -> httpx.Response:
        raise RuntimeError(f"real network access attempted in a test: {request.url}")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    """Undo handler changes made by ``configure_logging`` (keeps pytest's own handlers)."""
    yield
    for logger in (logging.getLogger(), logging.getLogger("cews")):
        for handler in list(logger.handlers):
            if not type(handler).__module__.startswith("_pytest"):
                logger.removeHandler(handler)
                handler.close()
    cews_logger = logging.getLogger("cews")
    cews_logger.setLevel(logging.NOTSET)
    cews_logger.propagate = True


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Default settings rooted in a temporary directory (config files come from the repo)."""
    return load_settings(
        env_file=None,
        overrides={
            "project_root": tmp_path,
            "sqlite_path": tmp_path / "cews.db",
            "topic_taxonomy_file": TAXONOMY_FILE,
            "scoring_config_file": SCORING_FILE,
            "source_registry_file": REGISTRY_FILE,
        },
    )


@pytest.fixture
def engine() -> Iterator[Engine]:
    """A migrated in-memory SQLite engine."""
    memory_engine = create_memory_engine()
    upgrade_database(memory_engine)
    yield memory_engine
    memory_engine.dispose()


@pytest.fixture
def session_factory(engine: Engine) -> sessionmaker[Session]:
    """Session factory bound to the migrated in-memory engine."""
    return create_session_factory(engine)


@pytest.fixture
def session(session_factory: sessionmaker[Session]) -> Iterator[Session]:
    """A session that is rolled back at the end of the test."""
    with session_factory() as db_session:
        yield db_session
        db_session.rollback()
