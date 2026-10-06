"""Benchmark topics: well-established areas used to sanity-check the ranking.

These are **evaluation anchors only**. Nothing in scoring, ranking or insight generation reads
this file, and a benchmark topic is scored exactly like every other topic; the tests assert
that. Here they are only *looked at afterwards*: where did each one rank at each backtest
cutoff? An area everyone agrees has been established for years that the ranking keeps burying
is a signal worth investigating, not a result to be corrected by boosting it.

The configuration lists each benchmark by its taxonomy id. An optional ``evaluation_window``
(``[start, end]`` dates) restricts which backtest cutoffs are looked at, for a topic whose
period of interest an expert has specified; without one, every cutoff counts.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from cews.database.models import Topic
from cews.validation.backtest import BacktestReport

LOGGER = logging.getLogger(__name__)


class BenchmarkConfigError(ValueError):
    """Raised when the benchmark configuration cannot be used."""


@dataclass(frozen=True)
class BenchmarkTopic:
    """One configured anchor."""

    name: str
    taxonomy_ref: str
    window: tuple[date, date] | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    """How one benchmark topic ranked across the backtest cutoffs that were looked at."""

    name: str
    taxonomy_ref: str
    resolved: bool
    folds_considered: int
    ranks: tuple[int, ...]
    mean_rank: float | None
    mean_percentile: float | None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return {
            "name": self.name,
            "taxonomy_ref": self.taxonomy_ref,
            "resolved": self.resolved,
            "folds_considered": self.folds_considered,
            "ranks": list(self.ranks),
            "mean_rank": None if self.mean_rank is None else round(self.mean_rank, 2),
            "mean_percentile": (
                None if self.mean_percentile is None else round(self.mean_percentile, 4)
            ),
            "note": self.note,
        }


@dataclass
class BenchmarkReport:
    """Every benchmark, looked at against one backtest."""

    results: list[BenchmarkResult] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return {"benchmarks": [result.as_dict() for result in self.results]}


def load_benchmark_topics(path: Path) -> list[BenchmarkTopic]:
    """Read the benchmark configuration.

    Raises:
        BenchmarkConfigError: if the file is missing or unreadable, an entry lacks a name or
            taxonomy_ref, two entries share a taxonomy_ref, or a window is not a valid
            ``[start, end]`` pair of dates with start before end.
    """
    if not path.is_file():
        raise BenchmarkConfigError(f"benchmark file not found: {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise BenchmarkConfigError(f"cannot read {path}: {exc}") from exc
    entries = document.get("benchmark_topics")
    if not isinstance(entries, list):
        raise BenchmarkConfigError(f"{path}: expected a list under 'benchmark_topics'")

    benchmarks: list[BenchmarkTopic] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not entry.get("name") or not entry.get("taxonomy_ref"):
            raise BenchmarkConfigError(
                f"{path}: entry {index} needs both 'name' and 'taxonomy_ref'"
            )
        ref = str(entry["taxonomy_ref"])
        if ref in seen:
            raise BenchmarkConfigError(f"{path}: taxonomy_ref {ref!r} is listed more than once")
        seen.add(ref)
        benchmarks.append(BenchmarkTopic(str(entry["name"]), ref, _window(entry, path, ref)))
    return benchmarks


def _window(entry: dict[str, Any], path: Path, ref: str) -> tuple[date, date] | None:
    raw = entry.get("evaluation_window")
    if raw is None:
        return None
    try:
        start, end = raw
        start_date = start if isinstance(start, date) else date.fromisoformat(str(start))
        end_date = end if isinstance(end, date) else date.fromisoformat(str(end))
    except (TypeError, ValueError) as exc:
        raise BenchmarkConfigError(
            f"{path}: evaluation_window for {ref!r} must be [start, end] as YYYY-MM-DD dates"
        ) from exc
    if start_date >= end_date:
        raise BenchmarkConfigError(
            f"{path}: evaluation_window for {ref!r} must start before it ends"
        )
    return start_date, end_date


def evaluate_benchmarks(
    session: Session, backtest: BacktestReport, benchmarks: list[BenchmarkTopic]
) -> BenchmarkReport:
    """Where each benchmark topic ranked at each backtest cutoff.

    A benchmark whose taxonomy id is not an active topic in the database (a parent area, say,
    which is not ranked as a topic) is reported as unresolved, with the reason, rather than
    silently dropped or guessed at.
    """
    report = BenchmarkReport()
    for benchmark in benchmarks:
        topic = session.scalar(select(Topic).where(Topic.key == benchmark.taxonomy_ref))
        if topic is None:
            report.results.append(
                BenchmarkResult(
                    name=benchmark.name,
                    taxonomy_ref=benchmark.taxonomy_ref,
                    resolved=False,
                    folds_considered=0,
                    ranks=(),
                    mean_rank=None,
                    mean_percentile=None,
                    note="not a ranked topic in the database (perhaps a parent area)",
                )
            )
            continue

        ranks: list[int] = []
        percentiles: list[float] = []
        for fold in backtest.folds:
            if benchmark.window and not (benchmark.window[0] <= fold.cutoff <= benchmark.window[1]):
                continue
            order = [topic_id for topic_id, _name, _score in fold.predicted]
            if topic.id not in order:
                continue
            position = order.index(topic.id) + 1
            ranks.append(position)
            percentiles.append(1.0 - (position - 1) / max(len(order) - 1, 1))
        report.results.append(
            BenchmarkResult(
                name=benchmark.name,
                taxonomy_ref=benchmark.taxonomy_ref,
                resolved=True,
                folds_considered=len(ranks),
                ranks=tuple(ranks),
                mean_rank=statistics.fmean(ranks) if ranks else None,
                mean_percentile=statistics.fmean(percentiles) if percentiles else None,
                note="" if ranks else "never scored at any cutoff considered",
            )
        )
    return report
