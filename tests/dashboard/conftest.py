"""A small scored database for the dashboard tests."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session, sessionmaker

from cews.database.connection import create_db_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.demo.generator import DemoConfig, generate_demo_dataset
from cews.demo.loader import load_demo_dataset
from cews.discovery.competitor_discovery import discover_competitors
from cews.features.activity_counts import aggregate_monthly_activity
from cews.features.pipeline import compute_features
from cews.forecasting.pipeline import run_forecasting
from cews.normalization.pipeline import normalize_records
from cews.normalization.topics import load_taxonomy, sync_taxonomy
from cews.scoring.pipeline import run_scoring
from cews.settings import Settings, load_settings
from support import SCORING_FILE, TAXONOMY_FILE

AS_OF = datetime(2026, 9, 1, tzinfo=UTC)


@pytest.fixture(scope="package")
def dashboard_database(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Path, Settings]]:
    """Demo data taken all the way through scoring and forecasting, built once."""
    directory = tmp_path_factory.mktemp("dashboard")
    database = directory / "cews.db"
    settings = load_settings(
        env_file=None,
        overrides={
            "project_root": directory,
            "sqlite_path": Path("./cews.db"),
            "topic_taxonomy_file": TAXONOMY_FILE,
            "scoring_config_file": SCORING_FILE,
            "competitor_mode": "AUTO",
            "min_competitor_evidence_count": 5,
        },
    )
    engine: Engine = create_db_engine(settings)
    upgrade_database(engine)
    factory = create_session_factory(engine)
    taxonomy = load_taxonomy(TAXONOMY_FILE)
    with session_scope(factory) as session:
        sync_taxonomy(session, taxonomy)
        load_demo_dataset(session, generate_demo_dataset(DemoConfig(scale=0.3)))
        normalize_records(session, settings, taxonomy)
        discover_competitors(session, settings, as_of=AS_OF)
        aggregate_monthly_activity(session, as_of=AS_OF, is_synthetic=True)
        compute_features(session, settings, as_of=AS_OF, is_synthetic=True)
        run_scoring(session, settings, as_of=AS_OF, is_synthetic=True)
        run_forecasting(
            session, settings, as_of=AS_OF, history_months=18, horizon=3, is_synthetic=True
        )
    engine.dispose()
    yield database, settings


@pytest.fixture(scope="package")
def dashboard_session_factory(
    dashboard_database: tuple[Path, Settings],
) -> Iterator[sessionmaker[Session]]:
    _, settings = dashboard_database
    engine: Engine = create_db_engine(settings)
    yield create_session_factory(engine)
    engine.dispose()


@pytest.fixture(scope="package")
def env_file(dashboard_database: tuple[Path, Settings]) -> Path:
    """An env file pointing the dashboard at the prepared database."""
    database, _ = dashboard_database
    path = database.parent / ".env"
    path.write_text(
        f"PROJECT_ROOT={database.parent}\nSQLITE_PATH=./cews.db\n"
        f"TOPIC_TAXONOMY_FILE={TAXONOMY_FILE}\nSCORING_CONFIG_FILE={SCORING_FILE}\n",
        encoding="utf-8",
    )
    return path
