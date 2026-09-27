"""Alembic environment for CEWS.

The database URL comes from, in order: ``sqlalchemy.url`` in the Alembic config, an existing
connection passed through ``config.attributes["connection"]`` (used by
``cews.database.migrations.upgrade_database``), or the CEWS settings (``.env``).
"""

from __future__ import annotations

from logging.config import fileConfig
from typing import Any

from alembic import context
from sqlalchemy import create_engine, pool

from cews.database.models import Base, UTCDateTime

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _database_url() -> str:
    url = config.get_main_option("sqlalchemy.url")
    if url:
        return url
    from cews.database.connection import build_database_url
    from cews.settings import load_settings

    return build_database_url(load_settings()).render_as_string(hide_password=False)


def render_item(type_: str, obj: Any, autogen_context: Any) -> str | bool:
    """Render the custom UTC datetime type as a plain SQLAlchemy type in migrations."""
    if type_ == "type" and isinstance(obj, UTCDateTime):
        return "sa.DateTime(timezone=True)"
    return False


def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting to a database."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        render_item=render_item,
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database."""
    connection = config.attributes.get("connection")
    if connection is not None:
        _run(connection)
        return
    engine = create_engine(_database_url(), poolclass=pool.NullPool)
    with engine.connect() as new_connection:
        _run(new_connection)
    engine.dispose()


def _run(connection: Any) -> None:
    # SQLite cannot ALTER most constraints, so migrations use batch mode there. Pass
    # ``-x batch=false`` to alembic to render plain operations (used for the baseline).
    batch_arg = context.get_x_argument(as_dictionary=True).get("batch", "true")
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_item=render_item,
        compare_type=True,
        render_as_batch=connection.dialect.name == "sqlite" and batch_arg != "false",
    )
    with context.begin_transaction():
        context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
