"""Integration tests for the refresh cycle and its commands, on the real pipeline.

The sources are all switched off, so nothing here touches the network; the analysis half of a
cycle (normalize through the Power BI export) runs for real on the demo data.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import func, select

from cews.cli import EXIT_FAILURE, EXIT_OK, main
from cews.constants import RunStatus
from cews.database.connection import create_db_engine, create_session_factory, session_scope
from cews.database.models import Forecast, Insight, JobRun, Score
from cews.scheduler import jobs
from cews.scheduler.job_state import job_lock, lock_directory, lock_status
from cews.scheduler.jobs import run_refresh
from cews.settings import Settings, load_settings
from support import REGISTRY_FILE, SCORING_FILE, TAXONOMY_FILE

pytestmark = pytest.mark.integration

AS_OF = datetime(2026, 9, 1, tzinfo=UTC)
ENV_TEMPLATE = (
    "PROJECT_ROOT={root}\nSQLITE_PATH=./data/cews.db\nTOPIC_TAXONOMY_FILE={taxonomy}\n"
    "SCORING_CONFIG_FILE={scoring}\nSOURCE_REGISTRY_FILE={registry}\n"
    "ENABLE_CLINICAL_TRIALS_GOV=false\nENABLE_PUBMED=false\nENABLE_EUROPE_PMC=false\n"
)


def write_env(root: Path, extra: str = "") -> Path:
    env = root / ".env"
    env.write_text(
        ENV_TEMPLATE.format(
            root=root, taxonomy=TAXONOMY_FILE, scoring=SCORING_FILE, registry=REGISTRY_FILE
        )
        + extra,
        encoding="utf-8",
    )
    return env


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("scheduler_project")
    env = write_env(root)
    assert main(["db-init", "--env-file", str(env)]) == EXIT_OK
    assert main(["seed-demo", "--env-file", str(env), "--scale", "0.2"]) == EXIT_OK
    return env


def count(session: object, model: type) -> int:
    return int(session.scalar(select(func.count()).select_from(model)) or 0)  # type: ignore[attr-defined]


def open_project(env: Path) -> tuple[Settings, object]:
    settings = load_settings(env_file=env)
    return settings, create_session_factory(create_db_engine(settings))


@pytest.fixture(scope="module")
def first_cycle(project: Path) -> jobs.RefreshReport:
    """The full analysis chain, run once for real on a database that has never been analysed."""
    settings, factory = open_project(project)
    return run_refresh(factory, settings, skip_fetch=True, now=AS_OF)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------
# The real chain
# --------------------------------------------------------------------------------------
def test_a_first_cycle_builds_every_result_from_raw_records(
    first_cycle: jobs.RefreshReport,
) -> None:
    assert first_cycle.status is RunStatus.SUCCEEDED, first_cycle.error
    assert [(s.name, s.status) for s in first_cycle.steps] == [
        (n, "succeeded") for n in jobs.ANALYSIS_STEP_NAMES
    ]


def test_the_cycle_actually_stored_scores_forecasts_insights_and_exports(
    project: Path, first_cycle: jobs.RefreshReport
) -> None:
    settings, factory = open_project(project)
    with session_scope(factory) as session:  # type: ignore[arg-type]
        assert count(session, Score) > 0
        assert count(session, Forecast) > 0
        assert count(session, Insight) > 0
    export = settings.export_directory / "powerbi"
    assert (export / "refresh_metadata.json").is_file() and len(list(export.glob("*.csv"))) == 13


def test_the_cycle_left_an_audit_row_and_no_lock(
    project: Path, first_cycle: jobs.RefreshReport
) -> None:
    settings, factory = open_project(project)
    with session_scope(factory) as session:  # type: ignore[arg-type]
        row = session.scalars(select(JobRun).where(JobRun.job_id == first_cycle.job_id)).one()
        assert row.status == "succeeded" and row.finished_at is not None
        assert row.summary_json is not None and len(row.summary_json["steps"]) == 7
    assert lock_status(lock_directory(settings), "refresh") is None


def test_running_the_cycle_again_changes_nothing_that_should_not_change(
    project: Path, first_cycle: jobs.RefreshReport
) -> None:
    """Idempotent: a repeat cycle must not pile up duplicate scores, forecasts or insights."""
    settings, factory = open_project(project)

    def counts() -> tuple[int, int, int]:
        with session_scope(factory) as session:  # type: ignore[arg-type]
            return tuple(  # type: ignore[return-value]
                session.scalar(select(func.count()).select_from(model))
                for model in (Score, Forecast, Insight)
            )

    before = counts()
    second = run_refresh(factory, settings, skip_fetch=True, now=AS_OF)  # type: ignore[arg-type]
    assert second.status is RunStatus.SUCCEEDED
    assert counts() == before


def test_a_failing_step_stops_the_rest_but_keeps_earlier_work(
    project: Path, first_cycle: jobs.RefreshReport, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings, factory = open_project(project)

    def broken(*_a: object, **_k: object) -> dict[str, object]:
        raise RuntimeError("scoring blew up")

    monkeypatch.setitem(jobs.ANALYSIS_STEPS, "score", broken)
    report = run_refresh(factory, settings, skip_fetch=True, now=AS_OF)  # type: ignore[arg-type]
    statuses = {s.name: s.status for s in report.steps}
    assert statuses["features"] == "succeeded" and statuses["score"] == "failed"
    assert all(statuses[name] == "skipped" for name in ("forecast", "insights", "export"))
    assert report.status is RunStatus.PARTIAL and "scoring blew up" in (report.error or "")


# --------------------------------------------------------------------------------------
# The commands
# --------------------------------------------------------------------------------------
def test_refresh_with_no_sources_enabled_runs_the_analysis_and_reports_it(
    project: Path, first_cycle: jobs.RefreshReport, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    assert main(["refresh", "--env-file", str(project), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "succeeded" and payload["trigger"] == "manual"
    assert (
        payload["fetch"]["status"] == "skipped"
    )  # nothing enabled, and that is reported, not hidden
    assert [s["name"] for s in payload["steps"]] == list(jobs.ANALYSIS_STEP_NAMES)


def test_refresh_prints_a_readable_summary(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["refresh", "--env-file", str(project), "--skip-analysis"]) == EXIT_OK
    out = capsys.readouterr().out
    assert "Refresh" in out and "succeeded" in out and "fetch" in out


def test_a_dry_run_writes_nothing_and_skips_the_analysis(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    assert main(["refresh", "--env-file", str(project), "--dry-run", "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True and payload["steps"] == []


def test_refresh_says_which_source_is_unknown(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["refresh", "--env-file", str(project), "--source", "nonsense"]) == EXIT_FAILURE
    assert "nonsense" in capsys.readouterr().out


def test_refresh_is_refused_while_another_is_running(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings, _ = open_project(project)
    with job_lock(lock_directory(settings)):
        capsys.readouterr()
        assert main(["refresh", "--env-file", str(project), "--skip-analysis"]) == EXIT_FAILURE
        out = capsys.readouterr().out
        assert "already running" in out and f"process {os.getpid()}" in out
        assert main(["jobs", "--env-file", str(project)]) == EXIT_OK
        assert "running now : yes" in capsys.readouterr().out


def test_a_hand_typed_fetch_is_refused_while_a_cycle_is_running(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A manual fetch used to ignore the lock, so it could run beside a scheduled cycle."""
    settings, _ = open_project(project)
    with job_lock(lock_directory(settings)):
        capsys.readouterr()
        assert main(["fetch", "--env-file", str(project), "--dry-run"]) == EXIT_FAILURE
        assert "already running" in capsys.readouterr().out
    assert (
        main(["fetch", "--env-file", str(project), "--dry-run"]) == EXIT_OK
    )  # and fine afterwards
    capsys.readouterr()


def test_a_leftover_lock_file_from_a_dead_process_does_not_block_forever(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Only a live process holds the lock; what a dead one leaves behind is just a file."""
    settings, _ = open_project(project)
    folder = lock_directory(settings)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "refresh.info").write_text(
        '{"pid": 999999, "since": "2026-01-01T00:00:00+00:00"}', encoding="utf-8"
    )
    (folder / "refresh.lock").write_bytes(b"")
    assert main(["refresh", "--env-file", str(project), "--skip-analysis"]) == EXIT_OK
    capsys.readouterr()


def test_the_lock_does_not_depend_on_the_database_being_free(project: Path) -> None:
    """The reported failure: a long step held the database's single writer and the old,
    database-backed lock could not be renewed. The lock must work regardless."""
    import sqlite3

    settings, _ = open_project(project)
    writer = sqlite3.connect(settings.sqlite_path, isolation_level=None, timeout=0.1)
    writer.execute("BEGIN IMMEDIATE")  # hold the database's only write slot
    try:
        with job_lock(lock_directory(settings)):
            assert lock_status(lock_directory(settings), "refresh") is not None
    finally:
        writer.execute("ROLLBACK")
        writer.close()


def test_the_jobs_command_shows_the_last_success_and_recent_runs(
    project: Path, first_cycle: jobs.RefreshReport, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    assert main(["jobs", "--env-file", str(project)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "scheduler   : enabled, every 120 minute(s)" in out
    assert "running now : no" in out and "last success:" in out and "refresh" in out


def test_the_jobs_json_is_pure_json(
    project: Path, first_cycle: jobs.RefreshReport, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    assert main(["jobs", "--env-file", str(project), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["running"] is False and payload["interval_minutes"] == 120
    assert payload["recent_runs"] and payload["recent_runs"][0]["job"] == "refresh"


@pytest.mark.parametrize(("hours_ago", "overdue"), [(1, False), (3, False), (5, True), (48, True)])
def test_jobs_flags_a_last_success_older_than_twice_the_interval(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], hours_ago: int, overdue: bool
) -> None:
    """With the default 120-minute interval, more than four hours without a success is overdue."""
    from cews.scheduler.job_state import finish_job_run, start_job_run

    env = write_env(tmp_path)
    assert main(["db-init", "--env-file", str(env)]) == EXIT_OK
    _, factory = open_project(env)
    started = datetime.now(UTC) - timedelta(hours=hours_ago)
    job_id = start_job_run(factory, "refresh", trigger="schedule", now=started)  # type: ignore[arg-type]
    finish_job_run(factory, job_id, RunStatus.SUCCEEDED, now=started + timedelta(minutes=3))  # type: ignore[arg-type]
    capsys.readouterr()
    assert main(["jobs", "--env-file", str(env), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["overdue"] is overdue
    assert abs(payload["minutes_since_last_success"] - hours_ago * 60) < 2
    assert main(["jobs", "--env-file", str(env)]) == EXIT_OK
    assert ("WARNING" in capsys.readouterr().out) is overdue


def test_jobs_reports_a_failing_streak(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from cews.scheduler.job_state import finish_job_run, start_job_run

    env = write_env(tmp_path)
    assert main(["db-init", "--env-file", str(env)]) == EXIT_OK
    _, factory = open_project(env)
    for hours in (3, 2, 1):
        job_id = start_job_run(factory, "refresh", trigger="schedule", now=datetime.now(UTC) - timedelta(hours=hours))  # type: ignore[arg-type]
        finish_job_run(factory, job_id, RunStatus.FAILED, error="no network")  # type: ignore[arg-type]
    capsys.readouterr()
    assert main(["jobs", "--env-file", str(env)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "the last 3 refresh(es) failed" in out and "last success: never" in out


def test_jobs_flags_a_running_row_as_stale_once_the_lock_is_free(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression: a row left marked "running" by a process that died without finishing stays
    that way until the *next* refresh cycle starts (mark_interrupted_runs only runs then) -
    which could be hours or days away. `cews jobs` is read-only and does not fix the row, but
    must say plainly that it is stale rather than showing a bare, misleading "running"."""
    from cews.scheduler.job_state import start_job_run

    env = write_env(tmp_path)
    assert main(["db-init", "--env-file", str(env)]) == EXIT_OK
    _, factory = open_project(env)
    start_job_run(factory, "refresh", trigger="manual", now=datetime.now(UTC) - timedelta(days=3))  # type: ignore[arg-type]
    capsys.readouterr()
    assert main(["jobs", "--env-file", str(env)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "running now : no" in out
    assert "stale: process ended without finishing" in out
    assert "run `cews refresh` to clear them" in out


def test_jobs_json_marks_the_same_row_stale(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from cews.scheduler.job_state import start_job_run

    env = write_env(tmp_path)
    assert main(["db-init", "--env-file", str(env)]) == EXIT_OK
    _, factory = open_project(env)
    start_job_run(factory, "refresh", trigger="manual", now=datetime.now(UTC) - timedelta(days=3))  # type: ignore[arg-type]
    capsys.readouterr()
    assert main(["jobs", "--env-file", str(env), "--json"]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["recent_runs"][0]["status"] == "running"
    assert payload["recent_runs"][0]["stale"] is True


def test_jobs_does_not_call_a_genuinely_running_row_stale(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The other half of the same check: while the lock really is held, a "running" row is
    exactly what it says - not flagged."""
    settings, _ = open_project(project)
    with job_lock(lock_directory(settings)):
        capsys.readouterr()
        assert main(["jobs", "--env-file", str(project), "--json"]) == EXIT_OK
        payload = json.loads(capsys.readouterr().out)
        assert payload["running"] is True
        assert all(not row["stale"] for row in payload["recent_runs"])


def test_jobs_on_a_fresh_database_says_nothing_has_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = write_env(tmp_path)
    assert main(["db-init", "--env-file", str(env)]) == EXIT_OK
    capsys.readouterr()
    assert main(["jobs", "--env-file", str(env)]) == EXIT_OK
    assert "never" in capsys.readouterr().out


def test_the_scheduler_refuses_to_start_when_disabled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = write_env(tmp_path, "ENABLE_SCHEDULER=false\n")
    assert main(["db-init", "--env-file", str(env)]) == EXIT_OK
    capsys.readouterr()
    assert main(["run-scheduler", "--env-file", str(env)]) == EXIT_FAILURE
    out = capsys.readouterr().out
    assert "ENABLE_SCHEDULER=false" in out and "cews refresh" in out


def test_the_scheduler_needs_an_initialized_database(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = write_env(tmp_path)
    assert main(["run-scheduler", "--env-file", str(env)]) == EXIT_FAILURE
    assert "db-init" in capsys.readouterr().out


def test_refresh_needs_an_initialized_database(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    env = write_env(tmp_path)
    assert main(["refresh", "--env-file", str(env)]) == EXIT_FAILURE
    assert "db-init" in capsys.readouterr().out


# --------------------------------------------------------------------------------------
# A real scheduler process, interrupted with real signals
# --------------------------------------------------------------------------------------
SRC = Path(__file__).resolve().parents[2] / "src"
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="sends SIGINT to a child process")


def cews(env: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "cews.cli", *args, "--env-file", str(env)],
        capture_output=True, text=True, timeout=120, env={**os.environ, "PYTHONPATH": str(SRC)},
    )  # fmt: skip


def is_running(env: Path) -> bool:
    line = next(
        row for row in cews(env, "jobs").stdout.splitlines() if row.startswith("running now")
    )
    return line.split(":")[1].strip() == "yes"


def start_scheduler(env: Path) -> subprocess.Popen[str]:
    process = subprocess.Popen(
        [sys.executable, "-m", "cews.cli", "run-scheduler", "--now", "--env-file", str(env)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={**os.environ, "PYTHONPATH": str(SRC)},
    )  # fmt: skip
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if is_running(env):
            return process
        time.sleep(0.5)
    process.kill()
    raise AssertionError("the scheduler never started a cycle")


@posix_only
def test_one_interrupt_lets_the_running_cycle_finish_and_never_allows_an_overlap(
    project: Path,
) -> None:
    process = start_scheduler(project)
    try:
        process.send_signal(signal.SIGINT)
        time.sleep(2)
        assert process.poll() is None  # still finishing its cycle
        assert is_running(project)
        refused = cews(project, "refresh", "--skip-analysis")
        assert refused.returncode == EXIT_FAILURE and "already running" in refused.stdout
        assert process.wait(timeout=90) == 0
    finally:
        if process.poll() is None:
            process.kill()
    assert not is_running(project)
    assert cews(project, "refresh", "--skip-analysis").returncode == EXIT_OK


@posix_only
def test_a_second_interrupt_aborts_at_once_and_the_next_refresh_recovers(project: Path) -> None:
    process = start_scheduler(project)
    try:
        process.send_signal(signal.SIGINT)
        time.sleep(1.5)
        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=20) == 130
    finally:
        if process.poll() is None:
            process.kill()
    assert not is_running(project)  # the lock was freed by the abort
    assert cews(project, "refresh", "--skip-analysis").returncode == EXIT_OK
    out = cews(project, "jobs", "--json").stdout
    errors = [run["error"] for run in json.loads(out)["recent_runs"]]
    assert any(error and "interrupted" in error for error in errors)  # the aborted run was closed
