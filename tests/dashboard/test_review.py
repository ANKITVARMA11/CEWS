"""Tests for the dashboard's Review page.

Unlike every other page, Review writes to the database - an approve/reject decision - so these
use their own isolated database with hand-seeded review items, rather than the shared
``dashboard_database`` fixture every other dashboard test reads from: a package-scoped fixture
shared across many tests is the wrong place to mutate state that would leak into whichever test
runs next.

The page itself is now a thin wrapper around ``components.review_panel.render_review_panel`` (see
``test_review_panel.py`` for the panel's own unit-level tests: pagination, the true-total count,
evidence resolution). What belongs here is page-level wiring: the page shows up, is reachable
from the sidebar, and the "dash_review" key prefix is what is actually used.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from cews.database.connection import create_db_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import ReviewQueueItem
from cews.settings import load_settings
from support import SCORING_FILE, TAXONOMY_FILE

pytestmark = pytest.mark.integration

APP = Path(__file__).resolve().parents[2] / "dashboards" / "streamlit" / "app.py"


def _seeded_env(tmp_path: Path) -> Path:
    """A freshly initialized, empty database plus one organization-match review item."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"PROJECT_ROOT={tmp_path}\nSQLITE_PATH=./cews.db\n"
        f"TOPIC_TAXONOMY_FILE={TAXONOMY_FILE}\nSCORING_CONFIG_FILE={SCORING_FILE}\n",
        encoding="utf-8",
    )
    settings = load_settings(env_file=env_file)
    engine = create_db_engine(settings)
    upgrade_database(engine)
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        session.add(
            ReviewQueueItem(
                queue_type="organization_parent",
                subject_ref="Orvexa Bio|Orvexa Labs",
                payload_json={
                    "organizations": ["Orvexa Bio", "Orvexa Labs"],
                    "reason": "same prefix, uncertain which is the parent",
                },
                status="pending",
            )
        )
    engine.dispose()
    return env_file


def start(env_file: Path) -> AppTest:
    app = AppTest.from_file(str(APP), default_timeout=120)
    app.session_state["env_file"] = str(env_file)
    app.run()
    return app.sidebar.radio[0].set_value("Review").run()


def test_a_pending_item_is_shown_with_its_subject(tmp_path: Path) -> None:
    app = start(_seeded_env(tmp_path))
    assert not app.exception
    text = " ".join(m.value for m in app.markdown)
    assert "Orvexa Bio" in text and "Orvexa Labs" in text
    assert any("Showing 1-1 of 1 pending item(s)" in c.value for c in app.caption)


def test_an_empty_queue_says_so(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        f"PROJECT_ROOT={tmp_path}\nSQLITE_PATH=./cews.db\n"
        f"TOPIC_TAXONOMY_FILE={TAXONOMY_FILE}\nSCORING_CONFIG_FILE={SCORING_FILE}\n",
        encoding="utf-8",
    )
    settings = load_settings(env_file=env_file)
    upgrade_database(create_db_engine(settings))
    app = start(env_file)
    assert not app.exception
    assert any("Nothing is waiting" in m.value for m in app.success)


def test_opening_details_shows_the_full_payload_and_evidence_section(tmp_path: Path) -> None:
    """Regression-shaped: before this, a review item showed only its one-line subject, with no
    way to see the rest of its payload or any link back to the source records behind it."""
    env_file = _seeded_env(tmp_path)
    app = start(env_file)
    details = next(b for b in app.button if b.key == "dash_review_details_1")
    app = details.click().run()
    assert not app.exception
    text = " ".join(m.value for m in app.markdown)
    assert "same prefix, uncertain which is the parent" in text
    assert "Evidence" in text


def test_approving_resolves_the_item_in_the_real_database(tmp_path: Path) -> None:
    env_file = _seeded_env(tmp_path)
    app = start(env_file)
    button = next(b for b in app.button if b.key == "dash_review_approve_1")
    button.click().run()
    assert not app.exception

    settings = load_settings(env_file=env_file)
    factory = create_session_factory(create_db_engine(settings))
    with session_scope(factory) as session:
        item = session.get(ReviewQueueItem, 1)
        assert item is not None and item.status == "accepted" and item.resolved_at is not None


def test_rejecting_resolves_the_item_without_changing_anything_else(tmp_path: Path) -> None:
    env_file = _seeded_env(tmp_path)
    app = start(env_file)
    button = next(b for b in app.button if b.key == "dash_review_reject_1")
    button.click().run()
    assert not app.exception

    settings = load_settings(env_file=env_file)
    factory = create_session_factory(create_db_engine(settings))
    with session_scope(factory) as session:
        item = session.get(ReviewQueueItem, 1)
        assert item is not None and item.status == "rejected"


def test_an_already_resolved_item_cannot_be_acted_on_twice(tmp_path: Path) -> None:
    """Regression-shaped: two browser tabs (or a slow double click) must not double-apply the
    same decision - review_actions.py refuses a second action on an already-resolved item, and
    the page should show that refusal rather than crash."""
    env_file = _seeded_env(tmp_path)
    settings = load_settings(env_file=env_file)
    factory = create_session_factory(create_db_engine(settings))
    from cews.normalization.review_actions import approve_review_item

    with session_scope(factory) as session:
        approve_review_item(session, 1)

    app = start(env_file)
    assert not app.exception
    assert any("Nothing is waiting" in m.value for m in app.success)


def test_a_kind_filter_is_offered(tmp_path: Path) -> None:
    app = start(_seeded_env(tmp_path))
    assert not app.exception
    kinds = next(s for s in app.selectbox if s.key == "dash_review_filter")
    assert kinds.options == ["All kinds", "organization_parent"]
