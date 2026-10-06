"""Unit tests for the expert review sheet and reading it back."""

from __future__ import annotations

import csv
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from cews.database.connection import create_memory_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import Score, Topic
from cews.validation.expert_review import (
    COLUMNS,
    ExpertReviewError,
    export_expert_review,
    summarize_expert_ratings,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def factory() -> Iterator[sessionmaker[Session]]:
    engine = create_memory_engine()
    upgrade_database(engine)
    yield create_session_factory(engine)
    engine.dispose()


def _scored_topic(session: Session, name: str, value: float, day: date = date(2026, 8, 1)) -> Topic:
    topic = Topic(key=name.lower(), canonical_name=name, topic_type="technology")
    session.add(topic)
    session.flush()
    session.add(
        Score(
            score_date=day,
            entity_type="topic",
            entity_id=topic.id,
            score_type="trend",
            score_value=value,
            confidence_score=80.0,
            component_json={"qualified": value > 50, "explanation": f"{name} explained"},
            scoring_version="1.0.0",
        )
    )
    session.flush()
    return topic


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


# --------------------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------------------
def test_the_sheet_has_every_required_column(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_scope(factory) as session:
        _scored_topic(session, "Alpha", 70.0)
        export_expert_review(session, tmp_path / "s.csv")
    rows = _rows(tmp_path / "s.csv")
    assert tuple(rows[0]) == COLUMNS
    assert {
        "topic",
        "score",
        "confidence",
        "explanation",
        "evidence",
        "expert_rating",
        "expert_comment",
    } <= set(COLUMNS)


def test_topics_are_listed_highest_score_first(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_scope(factory) as session:
        _scored_topic(session, "Low", 20.0)
        _scored_topic(session, "High", 90.0)
        _scored_topic(session, "Mid", 55.0)
        written = export_expert_review(session, tmp_path / "s.csv")
    assert written == 3
    assert [row["topic"] for row in _rows(tmp_path / "s.csv")] == ["High", "Mid", "Low"]


def test_the_expert_columns_are_written_blank(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_scope(factory) as session:
        _scored_topic(session, "Alpha", 70.0)
        export_expert_review(session, tmp_path / "s.csv")
    row = _rows(tmp_path / "s.csv")[0]
    assert row["expert_rating"] == "" and row["expert_comment"] == ""


def test_the_explanation_and_qualification_travel_with_the_score(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_scope(factory) as session:
        _scored_topic(session, "Alpha", 70.0)
        _scored_topic(session, "Beta", 30.0)
        export_expert_review(session, tmp_path / "s.csv")
    by_topic = {row["topic"]: row for row in _rows(tmp_path / "s.csv")}
    assert (
        by_topic["Alpha"]["explanation"] == "Alpha explained"
        and by_topic["Alpha"]["qualifies"] == "yes"
    )
    assert by_topic["Beta"]["qualifies"] == "no"


def test_only_the_most_recent_scores_are_exported_by_default(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_scope(factory) as session:
        _scored_topic(session, "Old", 99.0, day=date(2026, 6, 1))
        _scored_topic(session, "New", 10.0, day=date(2026, 8, 1))
        written = export_expert_review(session, tmp_path / "s.csv")
    assert written == 1 and _rows(tmp_path / "s.csv")[0]["topic"] == "New"


def test_a_specific_date_and_a_limit_can_be_requested(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_scope(factory) as session:
        _scored_topic(session, "A", 10.0, day=date(2026, 6, 1))
        _scored_topic(session, "B", 20.0, day=date(2026, 6, 1))
        _scored_topic(session, "C", 30.0, day=date(2026, 8, 1))
        written = export_expert_review(
            session, tmp_path / "s.csv", score_date=date(2026, 6, 1), limit=1
        )
    assert written == 1 and _rows(tmp_path / "s.csv")[0]["topic"] == "B"


def test_nothing_scored_writes_a_header_only_sheet(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    with session_scope(factory) as session:
        assert export_expert_review(session, tmp_path / "s.csv") == 0
    assert _rows(tmp_path / "s.csv") == []


# --------------------------------------------------------------------------------------
# Reading a completed sheet back
# --------------------------------------------------------------------------------------
def _write(path: Path, rows: list[tuple[str, str, str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["topic", "score", "expert_rating"])
        writer.writerows(rows)
    return path


def test_precision_at_k_uses_the_highest_scoring_rated_topics(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "s.csv",
        [
            ("A", "90", "relevant"),
            ("B", "80", "not_relevant"),
            ("C", "70", "relevant"),
            ("D", "10", "relevant"),
        ],
    )
    summary = summarize_expert_ratings(path, k=3)
    assert summary.precision_at_k == pytest.approx(2 / 3)  # D is outside the top three
    assert summary.rated == 4 and summary.k == 3


def test_unrated_rows_are_counted_and_left_out(tmp_path: Path) -> None:
    path = _write(tmp_path / "s.csv", [("A", "90", "relevant"), ("B", "80", ""), ("C", "70", "")])
    summary = summarize_expert_ratings(path)
    assert summary.rows == 3 and summary.rated == 1 and summary.unrated == 2
    assert summary.precision_at_k == 1.0


def test_ratings_are_matched_leniently_on_case_and_spaces(tmp_path: Path) -> None:
    path = _write(tmp_path / "s.csv", [("A", "90", "Not Relevant"), ("B", "80", " RELEVANT ")])
    summary = summarize_expert_ratings(path)
    assert summary.by_rating["not_relevant"] == 1 and summary.by_rating["relevant"] == 1


def test_an_unrecognised_rating_is_reported_not_guessed(tmp_path: Path) -> None:
    path = _write(tmp_path / "s.csv", [("A", "90", "great"), ("B", "80", "relevant")])
    summary = summarize_expert_ratings(path)
    assert summary.invalid_ratings == ("great",)
    assert summary.rated == 1 and summary.unrated == 0


def test_mean_scores_compare_relevant_with_not_relevant(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "s.csv",
        [
            ("A", "80", "relevant"),
            ("B", "60", "relevant"),
            ("C", "40", "not_relevant"),
            ("D", "20", "not_relevant"),
        ],
    )
    summary = summarize_expert_ratings(path)
    assert summary.mean_score_relevant == pytest.approx(70.0)
    assert summary.mean_score_not_relevant == pytest.approx(30.0)


def test_nothing_rated_leaves_every_figure_undefined(tmp_path: Path) -> None:
    summary = summarize_expert_ratings(_write(tmp_path / "s.csv", [("A", "90", "")]))
    assert summary.precision_at_k is None and summary.mean_score_relevant is None


def test_a_sheet_missing_required_columns_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("topic,notes\nA,hello\n", encoding="utf-8")
    with pytest.raises(ExpertReviewError, match="missing column"):
        summarize_expert_ratings(path)


def test_a_missing_file_and_a_bad_k_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ExpertReviewError, match="not found"):
        summarize_expert_ratings(tmp_path / "absent.csv")
    with pytest.raises(ExpertReviewError, match="positive"):
        summarize_expert_ratings(_write(tmp_path / "s.csv", []), k=0)


def test_an_exported_sheet_can_be_filled_in_and_read_back(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    """The whole loop: export, an expert types ratings into the blank column, summarize."""
    path = tmp_path / "s.csv"
    with session_scope(factory) as session:
        _scored_topic(session, "Alpha", 90.0)
        _scored_topic(session, "Beta", 40.0)
        export_expert_review(session, path)
    rows = _rows(path)
    rows[0]["expert_rating"], rows[1]["expert_rating"] = "relevant", "not_relevant"
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_expert_ratings(path, k=1)
    assert summary.rated == 2 and summary.precision_at_k == 1.0
    assert summary.as_dict()["by_rating"]["relevant"] == 1
