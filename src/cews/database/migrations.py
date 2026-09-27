"""Programmatic Alembic helpers: upgrade, downgrade, and inspect revisions.

The Alembic environment lives in ``migrations/`` at the project root. These helpers build an
Alembic ``Config`` without reading ``alembic.ini`` so they never reconfigure application
logging, and they can run against an existing engine (including in-memory SQLite).
"""

from __future__ import annotations

import logging
from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Engine

from cews.constants import PROJECT_ROOT
from cews.database.connection import DatabaseError

LOGGER = logging.getLogger(__name__)

MIGRATIONS_DIR: Path = PROJECT_ROOT / "migrations"


def build_alembic_config(migrations_dir: Path | None = None) -> Config:
    """Return an Alembic config pointing at the CEWS migration scripts.

    Raises:
        DatabaseError: if the migrations directory does not exist.
    """
    directory = migrations_dir or MIGRATIONS_DIR
    if not (directory / "env.py").is_file():
        raise DatabaseError(f"Alembic environment not found in {directory}")
    config = Config()
    config.set_main_option("script_location", str(directory))
    return config


def head_revision(migrations_dir: Path | None = None) -> str | None:
    """Return the newest revision id defined in the migration scripts."""
    script = ScriptDirectory.from_config(build_alembic_config(migrations_dir))
    return script.get_current_head()


def current_revision(engine: Engine) -> str | None:
    """Return the revision currently applied to ``engine`` (None for an empty database)."""
    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def upgrade_database(
    engine: Engine, revision: str = "head", migrations_dir: Path | None = None
) -> str | None:
    """Apply migrations up to ``revision`` and return the resulting revision.

    Safe to call repeatedly: applying an already-applied revision does nothing.

    Raises:
        DatabaseError: if migrations fail.
    """
    config = build_alembic_config(migrations_dir)
    try:
        with engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, revision)
    except Exception as exc:  # Alembic raises many exception types
        raise DatabaseError(f"database migration failed: {exc}") from exc
    result = current_revision(engine)
    LOGGER.info("database at revision %s", result)
    return result


def downgrade_database(
    engine: Engine, revision: str = "base", migrations_dir: Path | None = None
) -> str | None:
    """Revert migrations down to ``revision`` (default: remove everything)."""
    config = build_alembic_config(migrations_dir)
    try:
        with engine.begin() as connection:
            config.attributes["connection"] = connection
            command.downgrade(config, revision)
    except Exception as exc:
        raise DatabaseError(f"database downgrade failed: {exc}") from exc
    return current_revision(engine)
