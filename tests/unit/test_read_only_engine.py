"""The read-only engine must refuse writes at the database itself."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text

from cews.database.connection import (
    DatabaseError,
    create_db_engine,
    create_read_only_engine,
    create_session_factory,
    session_scope,
)
from cews.database.migrations import upgrade_database
from cews.database.models import Topic
from cews.settings import Settings, load_settings

pytestmark = pytest.mark.unit


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    settings = load_settings(
        env_file=None, overrides={"project_root": tmp_path, "sqlite_path": Path("./cews.db")}
    )
    engine = create_db_engine(settings)
    upgrade_database(engine)
    with session_scope(create_session_factory(engine)) as session:
        session.add(Topic(key="t", canonical_name="Topic", topic_type="technology"))
    engine.dispose()
    return settings


def test_reads_work(settings: Settings) -> None:
    engine = create_read_only_engine(settings)
    with engine.connect() as connection:
        assert connection.execute(text("SELECT COUNT(*) FROM topics")).scalar() == 1
    engine.dispose()


def test_an_insert_is_refused_by_the_database(settings: Settings) -> None:
    engine = create_read_only_engine(settings)
    with engine.connect() as connection, pytest.raises(Exception, match="readonly"):
        connection.execute(
            text(
                "INSERT INTO topics (key, canonical_name, topic_type, active, status) VALUES ('x','x','t',1,'active')"
            )
        )
    engine.dispose()


@pytest.mark.parametrize(
    "statement",
    ["UPDATE topics SET canonical_name = 'hacked'", "DELETE FROM topics", "DROP TABLE topics"],
)
def test_no_kind_of_write_gets_through(settings: Settings, statement: str) -> None:
    engine = create_read_only_engine(settings)
    with engine.connect() as connection, pytest.raises(Exception, match="readonly"):
        connection.execute(text(statement))
    engine.dispose()


def test_the_data_is_unchanged_after_the_attempts(settings: Settings) -> None:
    engine = create_read_only_engine(settings)
    with engine.connect() as connection:
        for statement in ("UPDATE topics SET canonical_name = 'hacked'", "DELETE FROM topics"):
            with pytest.raises(Exception, match="readonly"):
                connection.execute(text(statement))
    engine.dispose()
    check = create_db_engine(settings)
    with check.connect() as connection:
        assert connection.execute(text("SELECT canonical_name FROM topics")).scalar() == "Topic"
    check.dispose()


def test_a_session_through_the_read_only_engine_cannot_write_either(settings: Settings) -> None:
    factory = create_session_factory(create_read_only_engine(settings))
    with pytest.raises(Exception, match="readonly"), session_scope(factory) as session:
        session.add(Topic(key="u", canonical_name="Other", topic_type="technology"))


def test_a_missing_database_is_an_error_not_a_new_empty_file(tmp_path: Path) -> None:
    settings = load_settings(
        env_file=None, overrides={"project_root": tmp_path, "sqlite_path": Path("./absent.db")}
    )
    with pytest.raises(DatabaseError, match="cews db-init"):
        create_read_only_engine(settings)
    assert not (tmp_path / "absent.db").exists()


def test_the_normal_engine_can_still_write(settings: Settings) -> None:
    """The read-only setting belongs to one engine, not to the database."""
    engine = create_db_engine(settings)
    with session_scope(create_session_factory(engine)) as session:
        session.add(Topic(key="v", canonical_name="Writable", topic_type="technology"))
    engine.dispose()
