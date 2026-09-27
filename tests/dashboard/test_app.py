"""Tests that the dashboard itself renders.

Streamlit's own test harness runs the app script exactly as the server would, so these catch the
errors that only appear when the pages are actually drawn.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

pytestmark = pytest.mark.integration

APP = Path(__file__).resolve().parents[2] / "dashboards" / "streamlit" / "app.py"
PAGES = (
    "Overview",
    "Trends",
    "Competitors",
    "Opportunities",
    "Anomalies",
    "Evidence",
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


def test_an_uninitialized_database_is_explained(tmp_path: Path) -> None:
    missing = tmp_path / ".env"
    missing.write_text(f"PROJECT_ROOT={tmp_path}\nSQLITE_PATH=./absent.db\n", encoding="utf-8")
    app = start(missing)
    assert not app.exception
    assert any("db-init" in error.value for error in app.error)
