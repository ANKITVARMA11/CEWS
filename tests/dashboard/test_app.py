"""Tests that the dashboard itself renders.

Streamlit's own test harness runs the app script exactly as the server would, so these catch the
errors that only appear when the pages are actually drawn.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from cews.database.connection import create_db_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.demo.generator import DemoConfig, generate_demo_dataset
from cews.demo.loader import load_demo_dataset
from cews.features.activity_counts import aggregate_monthly_activity
from cews.features.pipeline import compute_features
from cews.normalization.pipeline import normalize_records
from cews.normalization.topics import load_taxonomy, sync_taxonomy
from cews.scoring.pipeline import run_scoring
from cews.settings import load_settings
from support import SCORING_FILE, TAXONOMY_FILE

pytestmark = pytest.mark.integration

APP = Path(__file__).resolve().parents[2] / "dashboards" / "streamlit" / "app.py"
PAGES = (
    "Overview",
    "Trends",
    "Competitors",
    "Opportunities",
    "Anomalies",
    "Evidence",
    "Review",
    "Data sources",
)


def start(env_file: Path) -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.session_state["env_file"] = str(env_file)
    return app.run()


def test_the_dashboard_starts(env_file: Path) -> None:
    app = start(env_file)
    assert not app.exception
    assert app.title[0].value == "Competitor Early Warning System"


def test_demo_data_is_announced_before_anything_else(env_file: Path) -> None:
    """Nobody should present these numbers without knowing they are invented."""
    app = start(env_file)
    banners = [warning.value for warning in app.warning]
    assert any("SYNTHETIC DEMO DATA" in text for text in banners)
    assert any("invented for" in text for text in banners)


def test_the_overview_shows_the_headline_numbers(env_file: Path) -> None:
    app = start(env_file)
    labels = {metric.label for metric in app.metric}
    assert {"Competitors monitored", "Topics watched", "Records collected"} <= labels
    assert "Emerging trends" in labels
    assert "Flagged as low confidence" in labels


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_error(env_file: Path, page: str) -> None:
    app = start(env_file)
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, f"{page}: {[str(error.value) for error in app.exception]}"


def test_trends_can_be_opened_one_at_a_time(env_file: Path) -> None:
    app = start(env_file)
    app.sidebar.radio[0].set_value("Trends").run()
    assert app.selectbox, "a topic picker should be offered"
    first = app.selectbox[0].options[0]
    app.selectbox[0].set_value(first).run()
    assert not app.exception
    assert any("In words" in text.value for text in app.markdown)


def test_competitor_wording_never_claims_an_actual_threat(env_file: Path) -> None:
    app = start(env_file)
    app.sidebar.radio[0].set_value("Competitors").run()
    captions = " ".join(text.value for text in app.caption).lower()
    assert "not evidence of a legal, commercial or scientific threat" in captions


def test_opportunities_are_not_presented_as_recommendations(env_file: Path) -> None:
    app = start(env_file)
    app.sidebar.radio[0].set_value("Opportunities").run()
    captions = " ".join(text.value for text in app.caption).lower()
    assert "not an investment, commercial or scientific recommendation" in captions


def test_evidence_traces_a_score_back_to_records(env_file: Path) -> None:
    app = start(env_file)
    app.sidebar.radio[0].set_value("Evidence").run()
    assert not app.exception
    labels = {metric.label for metric in app.metric}
    assert {"Score", "Confidence", "Records", "Qualifies"} <= labels
    assert len(app.dataframe) >= 2  # components and the source records behind them


def test_the_competitors_page_survives_zero_discovered_competitors(tmp_path: Path) -> None:
    """Regression: topics scored, but ``cews competitors`` never run (so no organization is
    flagged ``manually_included``/``discovered_automatically``) - the exact state a person lands
    in by running normalize/features/score/forecast without that step in between. Both tabs on
    this page then render an identical empty-state chart (``charts.ranked_bars([], [])``), which
    used to crash with ``StreamlitDuplicateElementId`` because neither ``st.plotly_chart`` call
    passed an explicit ``key``: Streamlit auto-generates an id from a chart's content, and two
    calls producing the same empty figure collided. Every ``st.plotly_chart`` call in this file now
    has a unique key."""
    database = tmp_path / "cews.db"
    settings = load_settings(
        env_file=None,
        overrides={
            "project_root": tmp_path,
            "sqlite_path": Path("./cews.db"),
            "topic_taxonomy_file": TAXONOMY_FILE,
            "scoring_config_file": SCORING_FILE,
        },
    )
    as_of = datetime(2026, 9, 1, tzinfo=UTC)
    engine = create_db_engine(settings)
    upgrade_database(engine)
    factory = create_session_factory(engine)
    taxonomy = load_taxonomy(TAXONOMY_FILE)
    with session_scope(factory) as session:
        sync_taxonomy(session, taxonomy)
        load_demo_dataset(session, generate_demo_dataset(DemoConfig(scale=0.3)))
        normalize_records(session, settings, taxonomy)
        # Deliberately no discover_competitors() call here.
        aggregate_monthly_activity(session, as_of=as_of, is_synthetic=True)
        compute_features(session, settings, as_of=as_of, is_synthetic=True)
        run_scoring(session, settings, as_of=as_of, is_synthetic=True)
    engine.dispose()

    env_file = database.parent / ".env"
    env_file.write_text(
        f"PROJECT_ROOT={database.parent}\nSQLITE_PATH=./cews.db\n"
        f"TOPIC_TAXONOMY_FILE={TAXONOMY_FILE}\nSCORING_CONFIG_FILE={SCORING_FILE}\n",
        encoding="utf-8",
    )
    app = start(env_file)
    app.sidebar.radio[0].set_value("Competitors").run()
    assert not app.exception
    # Both charts rendered under their own distinct key: before the fix, this is exactly the
    # pair of calls (both `charts.ranked_bars([], [])`, i.e. the same empty-state figure) that
    # collided with `StreamlitDuplicateElementId` for lack of one.
    assert app.get_by_key("competitors_threat_bars") is not None
    assert app.get_by_key("competitors_innovation_bars") is not None


def test_an_uninitialized_database_is_explained(tmp_path: Path) -> None:
    missing = tmp_path / ".env"
    missing.write_text(f"PROJECT_ROOT={tmp_path}\nSQLITE_PATH=./absent.db\n", encoding="utf-8")
    app = start(missing)
    assert not app.exception
    assert any("db-init" in error.value for error in app.error)
