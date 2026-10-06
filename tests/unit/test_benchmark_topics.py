"""Unit tests for benchmark topics: loading, evaluating, and never influencing a score."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy.orm import Session, sessionmaker

from cews.database.connection import create_memory_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import Topic
from cews.validation.backtest import BacktestFold, BacktestReport
from cews.validation.benchmark_topics import (
    BenchmarkConfigError,
    BenchmarkTopic,
    evaluate_benchmarks,
    load_benchmark_topics,
)
from cews.validation.ranking_metrics import CorrelationResult

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def factory() -> Iterator[sessionmaker[Session]]:
    engine = create_memory_engine()
    upgrade_database(engine)
    yield create_session_factory(engine)
    engine.dispose()


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "benchmarks.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _fold(cutoff: date, ranking: list[int]) -> BacktestFold:
    none = CorrelationResult(None, None, "spearman")
    return BacktestFold(
        cutoff=cutoff,
        horizon_months=3,
        k=5,
        predicted=tuple(
            (topic_id, f"t{topic_id}", 100.0 - index) for index, topic_id in enumerate(ranking)
        ),
        actual_growth={},
        precision_at_k=0.0,
        recall_at_k=0.0,
        ndcg_at_k=0.0,
        spearman=none,
        kendall=none,
    )


def _report(*folds: BacktestFold) -> BacktestReport:
    report = BacktestReport(horizon_months=3, k=5)
    report.folds.extend(folds)
    return report


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def test_the_shipped_configuration_loads() -> None:
    benchmarks = load_benchmark_topics(REPO_ROOT / "config" / "benchmark_topics.yaml")
    assert {b.taxonomy_ref for b in benchmarks} >= {"mrna_therapeutics", "crispr_gene_editing"}
    assert all(b.window is None for b in benchmarks)  # no window is invented for an expert


def test_every_shipped_benchmark_names_a_real_taxonomy_entry() -> None:
    taxonomy_text = (REPO_ROOT / "config" / "topic_taxonomy.yaml").read_text(encoding="utf-8")
    for benchmark in load_benchmark_topics(REPO_ROOT / "config" / "benchmark_topics.yaml"):
        assert f"id: {benchmark.taxonomy_ref}," in taxonomy_text


def test_a_window_is_parsed(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        "benchmark_topics:\n  - {name: A, taxonomy_ref: a, evaluation_window: [2025-01-01, 2025-12-01]}\n",
    )
    assert load_benchmark_topics(path)[0].window == (date(2025, 1, 1), date(2025, 12, 1))


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("benchmark_topics: nope\n", "expected a list"),
        ("benchmark_topics:\n  - {name: A}\n", "needs both"),
        ("benchmark_topics:\n  - {taxonomy_ref: a}\n", "needs both"),
        (
            "benchmark_topics:\n  - {name: A, taxonomy_ref: a}\n  - {name: B, taxonomy_ref: a}\n",
            "more than once",
        ),
        (
            "benchmark_topics:\n  - {name: A, taxonomy_ref: a, evaluation_window: [2025-05-01, 2025-01-01]}\n",
            "start before it ends",
        ),
        (
            "benchmark_topics:\n  - {name: A, taxonomy_ref: a, evaluation_window: [soon, later]}\n",
            "YYYY-MM-DD",
        ),
    ],
)
def test_a_bad_configuration_is_refused(tmp_path: Path, body: str, message: str) -> None:
    with pytest.raises(BenchmarkConfigError, match=message):
        load_benchmark_topics(write(tmp_path, body))


def test_a_missing_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BenchmarkConfigError, match="not found"):
        load_benchmark_topics(tmp_path / "absent.yaml")


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
def _topics(session: Session, count: int = 4) -> list[Topic]:
    rows = [
        Topic(key=f"k{i}", canonical_name=f"Topic {i}", topic_type="technology")
        for i in range(count)
    ]
    session.add_all(rows)
    session.flush()
    return rows


def test_ranks_and_percentiles_follow_the_predicted_order(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        topics = _topics(session)
        ids = [t.id for t in topics]
        report = _report(
            _fold(date(2025, 6, 1), [ids[0], ids[1], ids[2], ids[3]]),
            _fold(date(2025, 7, 1), [ids[1], ids[0], ids[2], ids[3]]),
        )
        result = evaluate_benchmarks(session, report, [BenchmarkTopic("Zero", "k0")]).results[0]
    assert result.resolved and result.ranks == (1, 2)
    assert result.mean_rank == pytest.approx(1.5)
    assert result.mean_percentile == pytest.approx((1.0 + (1 - 1 / 3)) / 2)


def test_the_window_restricts_which_cutoffs_are_looked_at(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        ids = [t.id for t in _topics(session)]
        report = _report(
            _fold(date(2025, 6, 1), ids),
            _fold(date(2025, 8, 1), list(reversed(ids))),
        )
        benchmark = BenchmarkTopic("Zero", "k0", window=(date(2025, 8, 1), date(2025, 9, 1)))
        result = evaluate_benchmarks(session, report, [benchmark]).results[0]
    assert result.ranks == (4,)


def test_a_benchmark_that_is_not_a_ranked_topic_is_reported_not_dropped(
    factory: sessionmaker[Session],
) -> None:
    with session_scope(factory) as session:
        _topics(session)
        result = evaluate_benchmarks(
            session, _report(), [BenchmarkTopic("CAR-T", "car_t")]
        ).results[0]
    assert not result.resolved
    assert "parent area" in result.note


def test_a_topic_that_never_appears_is_reported(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        topics = _topics(session)
        others = [t.id for t in topics[1:]]
        result = evaluate_benchmarks(
            session, _report(_fold(date(2025, 6, 1), others)), [BenchmarkTopic("Zero", "k0")]
        ).results[0]
    assert result.resolved and result.ranks == () and result.mean_rank is None
    assert "never scored" in result.note


def test_a_single_topic_cohort_has_a_defined_percentile(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        topics = _topics(session, count=1)
        result = evaluate_benchmarks(
            session,
            _report(_fold(date(2025, 6, 1), [topics[0].id])),
            [BenchmarkTopic("Zero", "k0")],
        ).results[0]
    assert result.mean_percentile == 1.0


def test_the_report_is_json_friendly(factory: sessionmaker[Session]) -> None:
    import json

    with session_scope(factory) as session:
        ids = [t.id for t in _topics(session)]
        json.dumps(
            evaluate_benchmarks(
                session, _report(_fold(date(2025, 6, 1), ids)), [BenchmarkTopic("Zero", "k0")]
            ).as_dict()
        )


# --------------------------------------------------------------------------------------
# The rule that matters: benchmarks are never a thumb on the scale
# --------------------------------------------------------------------------------------
# The only places outside validation/ that may mention benchmarks: the setting that says where
# the file lives, and the command that runs the evaluation and prints it. Neither computes a
# score; everything that does (scoring, features, forecasting, insights, discovery, ...) must
# not know the file exists.
BENCHMARK_AWARE_OUTSIDE_VALIDATION = {"settings.py", "cli.py"}


def test_no_computing_code_reads_or_mentions_the_benchmark_file() -> None:
    offenders = []
    for path in (REPO_ROOT / "src" / "cews").rglob("*.py"):
        relative = path.relative_to(REPO_ROOT / "src" / "cews")
        if relative.parts[0] == "validation" or str(relative) in BENCHMARK_AWARE_OUTSIDE_VALIDATION:
            continue
        if "benchmark" in path.read_text(encoding="utf-8").lower():
            offenders.append(str(relative))
    assert offenders == []


def test_the_two_permitted_mentions_do_no_computation_with_it() -> None:
    """settings.py only declares a path; cli.py only forwards to the evaluation."""
    settings_text = (REPO_ROOT / "src" / "cews" / "settings.py").read_text(encoding="utf-8")
    assert "load_benchmark_topics" not in settings_text
    cli_text = (REPO_ROOT / "src" / "cews" / "cli.py").read_text(encoding="utf-8")
    assert "load_benchmark_topics" not in cli_text and "evaluate_benchmarks" not in cli_text


def test_scoring_code_never_names_a_benchmark_topic() -> None:
    names = [
        b.taxonomy_ref
        for b in load_benchmark_topics(REPO_ROOT / "config" / "benchmark_topics.yaml")
    ]
    scoring = REPO_ROOT / "src" / "cews" / "scoring"
    text = "\n".join(path.read_text(encoding="utf-8") for path in scoring.rglob("*.py"))
    for name in names:
        assert name not in text


def test_the_scoring_configuration_does_not_favour_a_benchmark() -> None:
    text = (REPO_ROOT / "config" / "scoring_weights.yaml").read_text(encoding="utf-8")
    for benchmark in load_benchmark_topics(REPO_ROOT / "config" / "benchmark_topics.yaml"):
        assert benchmark.taxonomy_ref not in text
