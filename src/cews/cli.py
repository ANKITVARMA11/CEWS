"""Command-line interface for CEWS.

Commands (also available as ``python scripts/<name>.py`` wrappers)::

    cews check-env                 validate configuration, packages and database access
    cews db-init                   create/upgrade the database schema and load the taxonomy
    cews db-status                 show schema revision and record counts
    cews seed-demo                 load deterministic synthetic demo data
    cews reset-demo --yes          delete all synthetic data (live data is never touched)
    cews sources [--health]        list sources, their adapters, checkpoints and circuit state
    cews fetch [--dry-run]         collect enabled sources once (the scheduler does this every
                                   FETCH_INTERVAL_MINUTES from Phase 12)

Exit codes: 0 success, 1 runtime failure, 2 configuration or usage problem.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, func, inspect, select

from cews import __version__
from cews.constants import DatabaseBackend, RunStatus
from cews.database.connection import (
    DatabaseError,
    build_database_url,
    create_db_engine,
    create_session_factory,
    database_is_initialized,
    describe_url,
    session_scope,
)
from cews.database.migrations import current_revision, head_revision, upgrade_database
from cews.database.models import IngestionRun, ReviewQueueItem, Topic
from cews.database.repositories import (
    MixedDataError,
    count_records_by_origin,
    reset_synthetic_data,
)
from cews.demo.generator import DemoConfig, generate_demo_dataset, write_fixtures
from cews.demo.loader import load_demo_dataset
from cews.discovery.competitor_discovery import discover_competitors
from cews.features.activity_counts import aggregate_monthly_activity
from cews.features.pipeline import compute_features
from cews.forecasting.pipeline import run_forecasting
from cews.ingestion.base import adapter_classes
from cews.ingestion.checkpoints import read_checkpoint
from cews.ingestion.orchestrator import create_adapter, fetch_all_enabled_sources
from cews.ingestion.registry import RegistryError, load_source_registry
from cews.logging_config import configure_logging
from cews.normalization.pipeline import normalize_records, reset_normalization
from cews.normalization.topics import TaxonomyError, load_taxonomy, sync_taxonomy
from cews.scoring.config import ScoringConfigError
from cews.scoring.pipeline import run_scoring
from cews.settings import (
    Settings,
    SettingsError,
    load_settings,
    resolve_competitor_mode,
    resolve_env_file,
    validate_settings,
)

EXIT_OK, EXIT_FAILURE, EXIT_CONFIG = 0, 1, 2
LOW_CONFIDENCE = 60.0
REQUIRED_PACKAGES = ("pydantic", "pydantic_settings", "sqlalchemy", "alembic", "yaml", "tzdata")
MIN_PYTHON = (3, 11)


def _load(args: argparse.Namespace) -> Settings:
    settings = load_settings(env_file=resolve_env_file(args.env_file))
    configure_logging(settings)
    if not args.verbose:
        for handler in logging.getLogger("cews").handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                handler.setLevel(logging.WARNING)
    return settings


_created_engines: list[Engine] = []


def _engine(settings: Settings) -> Engine:
    """Create an engine that ``main`` disposes when the command finishes."""
    engine = create_db_engine(settings)
    _created_engines.append(engine)
    return engine


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------
def cmd_check_env(args: argparse.Namespace) -> int:
    """Validate Python, packages, configuration and database reachability."""
    failures = 0

    def report(level: str, message: str) -> None:
        nonlocal failures
        if level == "fail":
            failures += 1
        print(f"[{level:>4}] {message}")

    version = sys.version_info
    if (version.major, version.minor) >= MIN_PYTHON:
        report("ok", f"Python {version.major}.{version.minor}.{version.micro}")
    else:
        report("fail", f"Python {version.major}.{version.minor} found; 3.11 or newer is required")
    for package in REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
        except ImportError:
            report(
                "fail",
                f"package '{package}' is not installed (python -m pip install -r requirements-dev.txt)",
            )
    if failures:
        return EXIT_CONFIG

    try:
        settings = _load(args)
        warnings = validate_settings(settings)
        plan = resolve_competitor_mode(settings)
    except SettingsError as exc:
        print(exc)
        return EXIT_CONFIG
    report("ok", "configuration is valid")
    report(
        "ok",
        f"fetch interval {settings.fetch_interval_minutes} minutes, scheduler "
        f"{'on' if settings.enable_scheduler else 'off'}",
    )
    report(
        "ok",
        f"competitor mode {plan.mode.value}: {len(plan.include)} fixed, "
        f"{plan.auto_slots} automatic slot(s)",
    )
    report("ok", f"enabled sources: {', '.join(settings.enabled_sources) or 'none'}")
    try:
        registry = load_source_registry(settings.source_registry_file)
        report("ok", f"source registry valid ({len(registry)} sources)")
    except RegistryError as exc:
        report("fail", str(exc))
    for warning in warnings:
        report("warn", warning)

    url = build_database_url(settings)
    if settings.database_backend is DatabaseBackend.SQLITE and not Path(str(url.database)).exists():
        report("warn", f"SQLite database not created yet ({describe_url(url)}); run db-init")
    else:
        try:
            engine = _engine(settings)
            with engine.connect():
                pass
            report("ok", f"database reachable ({describe_url(url)})")
        except Exception as exc:  # any driver error is a failed check
            report("fail", f"cannot connect to {describe_url(url)}: {exc}")
    return EXIT_FAILURE if failures else EXIT_OK


def cmd_db_init(args: argparse.Namespace) -> int:
    """Create or upgrade the schema, then load the topic taxonomy."""
    settings = _load(args)
    engine = _engine(settings)
    revision = upgrade_database(engine)
    print(f"database schema at revision {revision} ({describe_url(build_database_url(settings))})")
    if not args.no_taxonomy:
        taxonomy = load_taxonomy(settings.topic_taxonomy_file)
        with session_scope(create_session_factory(engine)) as session:
            summary = sync_taxonomy(session, taxonomy)
        print(
            f"taxonomy loaded: {summary.topics_created} topic(s) and {summary.areas_created} area(s) "
            f"created, {summary.aliases_created} alias(es) added"
        )
    return EXIT_OK


def cmd_db_status(args: argparse.Namespace) -> int:
    """Print schema revision and record counts."""
    settings = _load(args)
    engine = _engine(settings)
    url = describe_url(build_database_url(settings))
    if not database_is_initialized(engine):
        print(f"database not initialized ({url}); run db-init")
        return EXIT_FAILURE
    revision, head = current_revision(engine), head_revision()
    state = "up to date" if revision == head else f"behind head {head}; run db-init"
    print(f"database : {url}")
    print(f"revision : {revision} ({state})")
    print(f"tables   : {len(inspect(engine).get_table_names()) - 1}")
    with session_scope(create_session_factory(engine)) as session:
        counts = count_records_by_origin(session)
        print(f"records  : {counts['synthetic']} synthetic, {counts['live']} live")
        print(f"topics   : {session.scalar(select(func.count()).select_from(Topic))}")
        last = session.scalars(
            select(IngestionRun).order_by(IngestionRun.start_time.desc())
        ).first()
        if last is not None:
            print(f"last run : {last.source} {last.status} at {last.start_time.isoformat()}")
    return EXIT_OK


def cmd_seed_demo(args: argparse.Namespace) -> int:
    """Generate and load the synthetic demo dataset (idempotent)."""
    settings = _load(args)
    engine = _engine(settings)
    if not database_is_initialized(engine):
        print("database not initialized; run db-init first")
        return EXIT_FAILURE
    config = DemoConfig(seed=args.seed, months=args.months, scale=args.scale)
    dataset = generate_demo_dataset(config)
    taxonomy = load_taxonomy(settings.topic_taxonomy_file)
    try:
        with session_scope(create_session_factory(engine)) as session:
            sync_taxonomy(session, taxonomy)
            summary = load_demo_dataset(session, dataset, allow_mixed=args.allow_mixed)
    except MixedDataError as exc:
        print(f"error: {exc}")
        return EXIT_FAILURE
    print("SYNTHETIC demo data loaded (invented organizations and artificial patterns)")
    print(
        f"  records: {summary.total} ({summary.inserted} inserted, {summary.updated} updated, "
        f"{summary.unchanged} unchanged)"
    )
    print(f"  by type: {dataset.count_by_type()}")
    print(f"  months : {dataset.months[0].isoformat()} to {dataset.months[-1].isoformat()}")
    if args.write_fixtures is not None:
        directory = Path(args.write_fixtures).expanduser()
        if not directory.is_absolute():
            directory = settings.project_root / directory
        written = write_fixtures(dataset, directory)
        print(f"  fixtures: {len(written)} files written to {directory}")
    return EXIT_OK


def cmd_reset_demo(args: argparse.Namespace) -> int:
    """Delete all synthetic rows. Live data is never touched."""
    settings = _load(args)
    engine = _engine(settings)
    if not database_is_initialized(engine):
        print("database not initialized; nothing to reset")
        return EXIT_OK
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        counts = count_records_by_origin(session)
    if not args.yes:
        print(
            f"This deletes {counts['synthetic']} synthetic records and all derived synthetic rows "
            f"({counts['live']} live records are kept). Re-run with --yes to confirm."
        )
        return EXIT_CONFIG
    with session_scope(factory) as session:
        removed = reset_synthetic_data(session)
    print("removed:", ", ".join(f"{name}={count}" for name, count in removed.items() if count))
    return EXIT_OK


def _when(value: object) -> str:
    return value.strftime("%Y-%m-%d %H:%M") if hasattr(value, "strftime") else "-"


def cmd_sources(args: argparse.Namespace) -> int:
    """List configured sources with adapter, checkpoint and circuit-breaker state."""
    settings = _load(args)
    registry = load_source_registry(settings.source_registry_file)
    implemented = adapter_classes()
    engine = _engine(settings)
    states = {}
    if database_is_initialized(engine):
        with session_scope(create_session_factory(engine)) as session:
            states = {c.id: read_checkpoint(session, c.id) for c in registry}
    now = datetime.now(UTC)
    header = f"{'source':<20} {'type':<15} {'enabled':<8} {'adapter':<9} {'refresh':<12} {'last success':<17} state"
    print(header)
    print("-" * len(header))
    for config in registry:
        state = states.get(config.id)
        refresh = (
            config.refresh_mode
            if config.refresh_mode == "incremental"
            else f"full/{config.full_refresh_window_days}d"
        )
        if state is not None and state.circuit_open(now):
            condition = f"paused until {_when(state.circuit_open_until)}"
        elif state is not None and state.consecutive_failures:
            condition = f"{state.consecutive_failures} failed run(s)"
        elif state is not None and state.last_status:
            condition = state.last_status
        else:
            condition = "never run"
        print(
            f"{config.id:<20} {config.source_type.value:<15} "
            f"{'yes' if config.is_enabled(settings) else 'no':<8} "
            f"{'yes' if config.id in implemented else 'not yet':<9} {refresh:<12} "
            f"{_when(state.last_success_at if state else None):<17} {condition}"
        )
    if args.health:
        print()
        for config in registry.enabled(settings):
            adapter = create_adapter(config, settings)
            if adapter is None:
                print(f"{config.id:<20} health: no adapter yet")
                continue
            try:
                health = adapter.health_check()
            finally:
                adapter.close()
            latency = f" {health.latency_ms:.0f} ms" if health.latency_ms is not None else ""
            print(f"{config.id:<20} health: {health.status.value}{latency} {health.message}")
    return EXIT_OK


def cmd_fetch(args: argparse.Namespace) -> int:
    """Collect enabled sources once."""
    settings = _load(args)
    engine = _engine(settings)
    if not database_is_initialized(engine):
        print("database not initialized; run db-init first")
        return EXIT_FAILURE
    from cews.scheduler.job_state import (
        REFRESH_JOB,
        REFRESH_LOCK,
        JobAlreadyRunningError,
        job_lock,
        lock_directory,
        mark_interrupted_runs,
    )

    factory = create_session_factory(engine)
    try:
        # The same lock a refresh cycle takes: a fetch typed by hand must not run beside one.
        with job_lock(lock_directory(settings), REFRESH_LOCK):
            mark_interrupted_runs(factory, (REFRESH_JOB, "fetch"))
            report = fetch_all_enabled_sources(
                settings,
                factory,
                sources=args.source or None,
                mode="full" if args.full else "auto",
                dry_run=args.dry_run,
                allow_mixed=args.allow_mixed,
            )
    except JobAlreadyRunningError as exc:
        print(f"error: {exc}")
        return EXIT_FAILURE
    except MixedDataError as exc:
        print(f"error: {exc}")
        return EXIT_FAILURE
    except KeyError as exc:
        print(f"error: {exc.args[0]}")
        return EXIT_CONFIG
    title = "Fetch (dry run, nothing written)" if report.dry_run else f"Fetch {report.job_id[:8]}"
    print(title)
    verb = "would insert" if report.dry_run else "inserted"
    for result in report.results:
        if result.status is RunStatus.SKIPPED:
            detail = "; ".join(result.warnings)
        else:
            detail = (
                f"{result.records_received} received, {result.records_inserted} {verb}, "
                f"{result.records_updated} updated, {result.records_skipped} unchanged, "
                f"{result.error_count} errors"
            )
        print(f"  {result.source_name:<20} {result.status.value:<9} {detail}")
        for message in result.errors[:3]:
            print(f"  {'':<20} ! {message}")
    ran = sum(1 for r in report.results if r.status is not RunStatus.SKIPPED)
    print(f"Result: {report.status.value} ({ran} of {len(report.results)} source(s) ran)")
    return EXIT_FAILURE if report.status in (RunStatus.FAILED, RunStatus.PARTIAL) else EXIT_OK


def cmd_normalize(args: argparse.Namespace) -> int:
    """Resolve organizations, assign topics and mark cross-source duplicates."""
    settings = _load(args)
    engine = _engine(settings)
    if not database_is_initialized(engine):
        print("database not initialized; run db-init first")
        return EXIT_FAILURE
    taxonomy = load_taxonomy(settings.topic_taxonomy_file)
    factory = create_session_factory(engine)

    def progress(done: int, total: int) -> None:
        print(f"\r  {done}/{total} records...", end="", flush=True)

    with session_scope(factory) as session:
        sync_taxonomy(session, taxonomy)
        if args.reprocess:
            print(f"marking {reset_normalization(session)} record(s) for reprocessing")
        summary = normalize_records(
            session,
            settings,
            taxonomy,
            limit=args.limit,
            reprocess=args.reprocess,
            deduplicate=not args.no_dedupe,
            progress=progress,
            commit_each_batch=True,  # an interrupted run resumes instead of starting over
        )
    print("\r" + " " * 40)
    print(f"normalized {summary.records_processed} record(s)")
    print(
        f"  organizations   : {summary.organizations_created} new, "
        f"{summary.organization_links} link(s)"
    )
    print(
        f"  topics          : {summary.topic_links} assignment(s), "
        f"{summary.records_without_topic} record(s) matched no topic"
    )
    print(f"  duplicates      : {summary.duplicates_marked} marked as copies of another source")
    print(f"  needs review    : {summary.review_items} (see: cews review)")
    if summary.failures:
        print(f"  failures        : {summary.failures} (see the log)")
    if summary.by_match_method:
        methods = ", ".join(f"{k}={v}" for k, v in sorted(summary.by_match_method.items()))
        print(f"  name matching   : {methods}")
    return EXIT_OK


def cmd_discover_topics(args: argparse.Namespace) -> int:
    """Cluster unmatched record text and report topics the taxonomy has no name for."""
    from sqlalchemy import select

    from cews.ai.topic_discovery import discover_topic_candidates
    from cews.database.models import RecordTopic, SourceRecord

    settings = _load(args)
    engine = _engine(settings)
    if not database_is_initialized(engine):
        print("database not initialized; run db-init first")
        return EXIT_FAILURE
    factory = create_session_factory(engine)

    with session_scope(factory) as session:
        unmatched_query = (
            select(SourceRecord.id, SourceRecord.title, SourceRecord.abstract)
            .where(
                SourceRecord.duplicate_of_id.is_(None),
                SourceRecord.title.is_not(None),
                ~select(RecordTopic.id)
                .where(RecordTopic.source_record_id == SourceRecord.id)
                .exists(),
            )
            .limit(args.limit)
        )
        texts_by_record = {
            record_id: f"{title} {abstract or ''}".strip()
            for record_id, title, abstract in session.execute(unmatched_query).all()
        }
        run = discover_topic_candidates(
            session,
            settings,
            texts_by_record,
            min_cluster_size=args.min_cluster_size,
            max_distance=args.max_distance,
            store=not args.no_store,
        )

    if args.json:
        print(
            json.dumps(
                {"run": run.as_dict(), "candidates": [c.as_dict() for c in run.candidates]},
                indent=2,
            )
        )
        return EXIT_OK

    if not settings.enable_ai_topic_discovery:
        print(
            "note: ENABLE_AI_TOPIC_DISCOVERY=false in .env; running anyway since you asked directly"
        )
    for warning in run.warnings:
        print(f"note: {warning}")
    print(
        f"considered {run.records_considered} unmatched record(s); "
        f"{run.records_clustered} fell into a cluster"
    )
    print(f"  matched an existing topic : {run.clusters_matching_existing_topics}")
    print(f"  new candidate(s)          : {len(run.candidates)}")
    if run.candidates:
        print(f"\n{'label':<34} {'records':>7} {'novelty':>8}  nearest existing topic")
        print("-" * 90)
        for candidate in run.candidates:
            nearest = candidate.nearest_existing_topic or "(none close)"
            print(
                f"{candidate.label[:33]:<34} {len(candidate.record_ids):>7} "
                f"{candidate.novelty:>8.2f}  {nearest}"
            )
        if not args.no_store:
            print("\nnew candidates were saved inactive and pending review; see: cews review")
    return EXIT_OK


def cmd_extract_announcements(args: argparse.Namespace) -> int:
    """Classify company announcements: type, partner organizations, therapeutic area, modality."""
    from collections import Counter

    from sqlalchemy import select

    from cews.ai.announcement_extraction import extract_announcement
    from cews.ai.llm import LLMClient, llm_enabled
    from cews.constants import SourceType
    from cews.database.models import AIExtraction, SourceRecord

    settings = _load(args)
    engine = _engine(settings)
    if not database_is_initialized(engine):
        print("database not initialized; run db-init first")
        return EXIT_FAILURE
    factory = create_session_factory(engine)

    with session_scope(factory) as session:
        pending_query = (
            select(SourceRecord)
            .where(
                SourceRecord.record_type == SourceType.ANNOUNCEMENT.value,
                SourceRecord.duplicate_of_id.is_(None),
                ~select(AIExtraction.id)
                .where(
                    AIExtraction.source_record_id == SourceRecord.id,
                    AIExtraction.extraction_type == "announcement",
                )
                .exists(),
            )
            .limit(args.limit)
        )
        records = list(session.scalars(pending_query))
        if not records:
            if args.json:
                print(json.dumps([]))
            else:
                print("no unclassified announcements found")
            return EXIT_OK

        client = LLMClient(settings) if llm_enabled(settings) else None
        results = []
        try:
            for record in records:
                text = f"{record.title or ''} {record.abstract or ''}".strip()
                result = extract_announcement(
                    session, settings, record, text=text, llm_client=client, store=not args.no_store
                )
                results.append((record, result))
        finally:
            if client is not None:
                client.close()

    if args.json:
        print(
            json.dumps(
                [{"record_id": record.id, **result.as_dict()} for record, result in results],
                indent=2,
            )
        )
        return EXIT_OK

    if not settings.enable_ai_announcement_extraction:
        print(
            "note: ENABLE_AI_ANNOUNCEMENT_EXTRACTION=false in .env; running anyway since you "
            "asked directly"
        )
    counts = Counter(result.announcement_type.value for _, result in results)
    methods = Counter(result.method for _, result in results)
    ungrounded = sum(1 for _, result in results if not result.grounded)
    print(f"classified {len(results)} announcement(s)")
    print("  by type   : " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    print("  by method : " + ", ".join(f"{k}={v}" for k, v in sorted(methods.items())))
    if ungrounded:
        print(
            f"  flagged   : {ungrounded} result(s) had an unverifiable detail dropped (see --json)"
        )
    for record, result in results[: args.top]:
        partners = ", ".join(result.partner_organizations) or "-"
        flag = " [ungrounded detail dropped]" if not result.grounded else ""
        print(f"  #{record.id:<6} {result.announcement_type.value:<20} partners: {partners}{flag}")
    return EXIT_OK


def _parse_as_of(value: str | None) -> datetime | None:
    """Read an --as-of date, or None for "now".

    Raises:
        ValueError: with a readable message for anything that is not YYYY-MM-DD.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).replace(tzinfo=UTC)
    except ValueError as exc:
        raise ValueError(f"--as-of must be a date like 2026-08-31, got {value!r}") from exc


def cmd_features(args: argparse.Namespace) -> int:
    """Count activity into months and compute the features scoring will use."""
    settings = _load(args)
    engine = _engine(settings)
    if not database_is_initialized(engine):
        print("database not initialized; run db-init first")
        return EXIT_FAILURE
    as_of = _parse_as_of(args.as_of)
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        synthetic = count_records_by_origin(session)["synthetic"] > 0
        counted = aggregate_monthly_activity(
            session, as_of=as_of, rebuild=args.rebuild, is_synthetic=synthetic
        )
        run = compute_features(session, settings, as_of=as_of, is_synthetic=synthetic)

    if args.json:
        print(json.dumps({"activity": counted.as_dict(), "features": run.as_dict()}, indent=2))
        return EXIT_OK

    print(
        f"activity: {counted.records_counted} record(s) over {counted.months} month(s) "
        f"({counted.first_period} to {counted.last_period})"
    )
    print(
        f"  periods : {counted.rows_written} written, {counted.rows_updated} updated"
        + (f", {counted.rows_removed} removed" if counted.rows_removed else "")
    )
    for warning in [*counted.warnings, *run.warnings]:
        print(f"  note    : {warning}")
    if not run.entities:
        print("no topics or competitors with activity yet; run: cews normalize, cews competitors")
        return EXIT_OK

    print(
        f"\nfeatures for {run.feature_date} over {run.months} month(s): {run.values_written} written, "
        f"{run.values_updated} updated"
    )
    if run.skipped_no_data:
        print(
            f"  no activity : {', '.join(run.skipped_no_data[:6])}"
            + (" ..." if len(run.skipped_no_data) > 6 else "")
        )
    if run.skipped_low_sample:
        print(f"  thin evidence: {', '.join(run.skipped_low_sample)} (kept, but low confidence)")

    ranked = sorted(
        run.entities,
        key=lambda entity: (entity.velocity.slope if entity.velocity else 0.0),
        reverse=True,
    )[: args.top]
    print(
        f"\n{'entity':<34} {'kind':<10} {'velocity':>9} {'momentum':>9} {'steady':>7} {'records':>8} {'conf':>5}"
    )
    print("-" * 92)
    for entity in ranked:
        velocity = f"{entity.velocity.slope:+.3f}" if entity.velocity else "n/a"
        steady = f"{entity.consistency.consistency:.2f}" if entity.consistency else "n/a"
        print(
            f"{entity.name[:33]:<34} {entity.entity_type:<10} {velocity:>9} "
            f"{entity.momentum.momentum:>+9.3f} {steady:>7} {entity.total_records:>8.0f} "
            f"{entity.sample_confidence:>5.2f}"
        )
    print("\nvelocity is the fitted monthly trend; conf is how much the sample size supports it")
    return EXIT_OK


def cmd_forecast(args: argparse.Namespace) -> int:
    """Forecast each topic and competitor, and record unusual months."""
    settings = _load(args)
    engine = _engine(settings)
    if not database_is_initialized(engine):
        print("database not initialized; run db-init first")
        return EXIT_FAILURE
    as_of = _parse_as_of(args.as_of)
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        synthetic = count_records_by_origin(session)["synthetic"] > 0
        run = run_forecasting(
            session,
            settings,
            as_of=as_of,
            horizon=args.horizon,
            store=not args.no_store,
            is_synthetic=synthetic,
            detect_anomalies=not args.no_anomalies,
        )
        forecasts: list[dict[str, Any]] = [
            {
                "entity": entity.name,
                "kind": entity.entity_type,
                "model": entity.selection.model,
                "metric_name": entity.selection.metric_name,
                "metric": entity.selection.metric,
                "reason": entity.selection.reason,
                "months": [month.isoformat() for month in entity.forecast_months],
                "predictions": (
                    list(entity.selection.forecast.predictions) if entity.selection.forecast else []
                ),
                "lower": list(entity.selection.forecast.lower) if entity.selection.forecast else [],
                "upper": list(entity.selection.forecast.upper) if entity.selection.forecast else [],
            }
            for entity in run.entities
        ]
        unusual: list[dict[str, Any]] = [
            {
                "entity": entity.name,
                "kind": entity.entity_type,
                **found.as_dict(),
            }
            for entity, found in run.anomalies()
        ]

    if args.json:
        print(
            json.dumps(
                {"run": run.as_dict(), "forecasts": forecasts, "anomalies": unusual}, indent=2
            )
        )
        return EXIT_OK

    for warning in run.warnings[:5]:
        print(f"note: {warning}")
    if not run.entities:
        print(
            "nothing to forecast: no topic or competitor has any recorded activity.\n"
            "run seed-demo (or fetch) and then normalize and features first."
        )
        return EXIT_OK

    summary = run.as_dict()
    print(
        f"forecast from {run.as_of} over {args.horizon} month(s): {len(run.entities)} entit(ies), "
        f"{run.forecasts_written} written, {run.forecasts_updated} updated"
    )
    print(
        "  models chosen: "
        + ", ".join(f"{name} x{count}" for name, count in sorted(summary["models_chosen"].items()))
    )
    if run.skipped:
        print(f"  no activity   : {len(run.skipped)} entit(ies) skipped")

    print(f"\n{'entity':<30} {'model':<15} {'fit':>10}  next {min(args.horizon, 3)} month(s)")
    print("-" * 104)
    forecastable = [row for row in forecasts if row["predictions"]]
    for row in sorted(forecastable, key=lambda item: -sum(item["predictions"]))[: args.top]:
        metric = f"{row['metric_name']}={row['metric']:.2f}" if row["metric"] is not None else "n/a"
        ahead = ", ".join(
            f"{value:.0f} [{low:.0f}-{high:.0f}]"
            for value, low, high in list(
                zip(row["predictions"], row["lower"], row["upper"], strict=True)
            )[:3]
        )
        print(f"{row['entity'][:29]:<30} {row['model']:<15} {metric:>10}  {ahead}")

    if unusual and not args.no_anomalies:
        print(f"\n{'month':<12} {'entity':<30} {'observed':>9} {'usual':>7}  {'kind':<22} conf")
        print("-" * 104)
        for row in unusual[: args.top]:
            print(
                f"{str(row['period']):<12} {row['entity'][:29]:<30} {row['observed']:>9.0f} "
                f"{row['expected']:>7.0f}  {row['kind']:<22} {row['confidence']:.0f}"
            )
    print(
        "\nfit is how the chosen model scored when replayed on history it had not seen;\n"
        "below 1 means it beat 'next month looks like this month'. A spike is not a trend."
    )
    return EXIT_OK


def cmd_score(args: argparse.Namespace) -> int:
    """Score the stored features and save the results with their breakdown."""
    settings = _load(args)
    engine = _engine(settings)
    if not database_is_initialized(engine):
        print("database not initialized; run db-init first")
        return EXIT_FAILURE
    as_of = _parse_as_of(args.as_of)
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        synthetic = count_records_by_origin(session)["synthetic"] > 0
        run = run_scoring(
            session, settings, as_of=as_of, store=not args.no_store, is_synthetic=synthetic
        )
        rows: list[dict[str, Any]] = [
            {
                "entity": result.entity_name,
                "kind": result.entity_type,
                "score_type": result.score_type,
                "score": round(result.value, 2),
                "confidence": round(result.confidence, 2),
                "category": result.category
                or (
                    f"rank {int(result.components['rank'].raw_value or 0)}"
                    if "rank" in result.components
                    else ""
                ),
                "records": round(result.sample_size),
                "context": result.context_key,
                "modifiers": [
                    item["name"]
                    for item in (
                        result.components["modifiers"].detail.get("applied", [])
                        if "modifiers" in result.components
                        else []
                    )
                ],
                "qualifies": result.qualified,
                "failed_rules": list(result.failed_gates),
                "missing_inputs": list(result.unavailable),
                "explanation": result.explain(),
            }
            for result in run.results
        ]

    if args.json:
        print(json.dumps({"run": run.as_dict(), "results": rows}, indent=2))
        return EXIT_OK

    for warning in run.warnings:
        print(f"note: {warning}")
    if not run.results:
        return EXIT_OK

    print(
        f"scored {run.score_date} with weights version {run.version}: "
        f"{len(run.results)} result(s), {run.stored.written} written, "
        f"{run.stored.updated} updated"
        + (f", {run.stored.removed} removed" if run.stored.removed else "")
    )

    if args.low_confidence:
        doubtful = sorted(
            (row for row in rows if row["confidence"] < LOW_CONFIDENCE),
            key=lambda row: row["confidence"],
        )
        _print_score_table("Results that should not be acted on", doubtful[: args.top])
        return EXIT_OK

    wanted = args.type
    if wanted in ("all", "trend"):
        trends = [row for row in rows if row["score_type"] == "trend"]
        qualified = [row["entity"] for row in trends if row["qualifies"]]
        print(
            "\nEmerging trends (passed every rule): "
            + (", ".join(qualified) if qualified else "none")
        )
        _print_score_table(
            "Topics by trend score", sorted(trends, key=lambda row: -row["score"])[: args.top]
        )
    if wanted in ("all", "opportunity"):
        opportunities = [row for row in rows if row["score_type"] == "opportunity"]
        _print_score_table(
            "Topics by opportunity score (prioritization only, not a recommendation)",
            sorted(opportunities, key=lambda row: -row["score"])[: args.top],
        )
    if wanted in ("all", "innovation"):
        innovation = [row for row in rows if row["score_type"] == "innovation"]
        _print_score_table(
            "Competitors by innovation score (research output, not growth)",
            sorted(innovation, key=lambda row: -row["score"])[: args.top],
        )
    if wanted in ("all", "threat"):
        overall = [row for row in rows if row["score_type"] == "threat" and not row["context"]]
        _print_score_table(
            "Competitors by monitoring priority (activity, not a legal or commercial claim)",
            sorted(overall, key=lambda row: -row["score"])[: args.top],
        )
        by_area = [row for row in rows if row["score_type"] == "threat" and row["context"]]
        if by_area:
            _print_score_table(
                "Monitoring priority within a therapeutic area",
                sorted(by_area, key=lambda row: -row["score"])[: args.top],
            )
    if wanted == "confidence":
        _print_score_table(
            "Confidence in each result",
            sorted(
                (row for row in rows if row["score_type"] == "confidence"),
                key=lambda row: -row["score"],
            )[: args.top],
        )
    print(
        "\nA score alone is never a finding: it must also clear confidence, evidence,\n"
        "independent-source and single-spike rules. The last column names the rules a result\n"
        "failed, or the events that added points to a monitoring priority."
    )
    return EXIT_OK


def _print_score_table(title: str, rows: Sequence[dict[str, Any]]) -> None:
    """Print one block of scores with the reasons any of them do not qualify."""
    print(f"\n{title}")
    print(f"{'entity':<30} {'score':>6} {'conf':>6} {'category':<28} {'ok':<4} why not")
    print("-" * 110)
    if not rows:
        print("(none)")
        return
    for row in rows:
        why = ", ".join(rule.replace("_", " ") for rule in row["failed_rules"])
        if row["missing_inputs"]:
            missing = ", ".join(name.replace("_", " ") for name in row["missing_inputs"])
            why = (why + "; " if why else "") + f"no {missing}"
        if not why and row.get("modifiers"):
            why = "+ " + ", ".join(name.replace("_", " ") for name in row["modifiers"])
        mark = "yes" if row["qualifies"] else "no"
        print(
            f"{row['entity'][:29]:<30} {row['score']:>6.1f} {row['confidence']:>6.1f} "
            f"{row['category'][:27]:<28} {mark:<4} {why[:45]}"
        )


def cmd_competitors(args: argparse.Namespace) -> int:
    """Rank organizations and show the competitors CEWS is monitoring."""
    settings = _load(args)
    engine = _engine(settings)
    if not database_is_initialized(engine):
        print("database not initialized; run db-init first")
        return EXIT_FAILURE
    with session_scope(create_session_factory(engine)) as session:
        result = discover_competitors(
            session,
            settings,
            window_months=args.window_months,
            include_all_types=args.include_all_types,
            persist=not args.no_persist,
        )
        if args.json:
            print(json.dumps(result.as_dict(), indent=2))
            return EXIT_OK
        window = result.window_start.date() if result.window_start else "-"
        print(f"mode {result.plan.mode.value if result.plan else '-'}, activity since {window}")
        if result.weights:
            print("weights: " + ", ".join(f"{k} {v}" for k, v in result.weights.items()))
        for warning in result.warnings:
            print(f"  note: {warning}")
        if not result.monitored:
            print("\nno competitors yet; collect data and run: cews normalize")
            return EXIT_OK
        print(f"\n{'competitor':32} {'score':>6} {'records':>8}  source types")
        print("-" * 78)
        for entry in result.monitored:
            mark = "*" if entry.manually_included else " "
            sources = ", ".join(s.replace("clinical_trial", "trial") for s in entry.source_types)
            print(
                f"{mark} {entry.name[:30]:30} {entry.score:6.1f} {entry.evidence_count:8}  {sources}"
            )
        print("\n* from COMPETITOR_INCLUDE; the rest were discovered from the data")
    return EXIT_OK


def cmd_insights(args: argparse.Namespace) -> int:
    """Run the five deterministic insight rules against the most recent scores."""
    from cews.insights.rule_engine import generate_insights
    from cews.scoring.config import ScoringConfigError

    settings = _load(args)
    engine = _engine(settings)
    if not database_is_initialized(engine):
        print("database not initialized; run db-init first")
        return EXIT_FAILURE
    score_date = _parse_as_of(args.as_of)
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        synthetic = count_records_by_origin(session)["synthetic"] > 0
        try:
            run = generate_insights(
                session,
                settings,
                score_date=score_date.date() if score_date else None,
                store=not args.no_store,
                is_synthetic=synthetic,
            )
        except ScoringConfigError as exc:
            print(f"scoring configuration problem: {exc}")
            return EXIT_CONFIG
        from cews.database.models import Insight

        stored_rows = (
            list(
                session.scalars(
                    select(Insight)
                    .where(Insight.insight_date == run.insight_date)
                    .order_by(Insight.severity.desc(), Insight.insight_type)
                )
            )
            if not args.no_store
            else []
        )

    if args.json:
        print(
            json.dumps(
                {
                    "run": run.as_dict(),
                    "insights": [
                        {
                            "id": row.id,
                            "type": row.insight_type,
                            "severity": row.severity,
                            "title": row.title,
                            "observed_fact": row.observed_fact,
                            "interpretation": row.interpretation,
                            "recommended_review": row.recommended_review,
                            "confidence": row.confidence_score,
                        }
                        for row in stored_rows
                    ],
                },
                indent=2,
            )
        )
        return EXIT_OK

    if run.warnings:
        for warning in run.warnings:
            print(f"note: {warning}")
        if run.candidates_found == 0:
            return EXIT_OK
    elif run.candidates_found == 0:
        print(f"no insight found for {run.insight_date}")
        return EXIT_OK
    print(
        f"insights for {run.insight_date}: {run.candidates_found} found, {run.stored} stored, "
        f"{run.duplicates_suppressed} already known"
    )
    if run.rejected_no_evidence:
        print(f"  dropped without evidence: {run.rejected_no_evidence}")
    if args.no_store:
        return EXIT_OK
    for row in stored_rows[: args.top]:
        print(f"\n[{row.severity}] {row.title}")
        print(f"    fact      : {row.observed_fact}")
        print(f"    means     : {row.interpretation}")
        print(f"    do next   : {row.recommended_review}")
    return EXIT_OK


def _print_evaluation(report: Any) -> None:
    """A short human summary, under the three headings the evidence must stay apart under."""
    data = report.as_dict()
    print(
        f"evaluation {data['evaluation_id']}"
        + ("  [synthetic demo data]" if data["is_synthetic"] else "")
    )

    algorithm = data["algorithm"]
    print("\nALGORITHM  (what the system says about itself; not checked against outside truth)")
    if "data_quality" in algorithm:
        failed = algorithm["data_quality"]["failed"]
        total_checks = len(algorithm["data_quality"]["checks"])
        print(
            f"  data quality : {total_checks - len(failed)}/{total_checks} checks passed"
            + (f"; failed: {', '.join(failed)}" if failed else "")
        )
    if "forecast" in algorithm:
        forecast = algorithm["forecast"]
        print(
            f"  forecasts    : {forecast['entities']} entities, median MASE {forecast['median_mase']}"
        )
    if "robustness" in algorithm:
        checks = [
            c
            for c in algorithm["robustness"]["checks"]
            if c["dimension"] != "low_sample_confidence"
        ]
        if checks:
            worst = min(checks, key=lambda c: c["stability"]["spearman"]["coefficient"] or 0.0)
            print(
                f"  robustness   : {len(checks)} perturbations; least stable is {worst['dimension']} "
                f"(rank correlation {worst['stability']['spearman']['coefficient']}, "
                f"top-{worst['stability']['k']} overlap {worst['stability']['top_k_overlap']})"
            )

    backtest = data["backtest"]
    print("\nBACKTEST  (would the ranking have predicted what actually grew?)")
    if "backtest" in backtest:
        b = backtest["backtest"]
        print(
            f"  {b['folds']} folds, {b['horizon_months']}-month horizon: precision@{b['k']} "
            f"{b['mean_precision_at_k']}, recall@{b['k']} {b['mean_recall_at_k']}, "
            f"NDCG {b['mean_ndcg_at_k']}, rank correlation {b['mean_spearman']}"
        )
        for warning in b["warnings"][:3]:
            print(f"  note: {warning}")
    for item in backtest.get("benchmarks", {}).get("benchmarks", []):
        where = (
            f"mean rank {item['mean_rank']} (percentile {item['mean_percentile']})"
            if item["mean_rank"] is not None
            else item["note"]
        )
        print(f"  benchmark {item['name']:<28} {where}")

    expert = data["expert_validation"]
    print("\nEXPERT VALIDATION  (what people made of it)")
    if "alerts" in expert:
        a = expert["alerts"]
        print(
            f"  alerts       : {a['reviewed']}/{a['insights_total']} reviewed; precision {a['precision']}, "
            f"acceptance {a['acceptance_rate']}, duplicates {a['duplicate_rate']}, "
            f"median lead time {a['median_lead_days']} days"
        )
    if "ai_ablation" in expert:
        ablation = expert["ai_ablation"]
        announcements = ablation.get("announcement_extraction")
        if announcements:
            parts = [
                (
                    f"{m['method']} {m['accuracy']}"
                    if m["accuracy"] is not None
                    else f"{m['method']} n/a"
                )
                for m in announcements["methods"]
            ]
            print(
                f"  announcements: accuracy on {announcements['labelled_count']} labelled: "
                + ", ".join(parts)
            )
        topics = ablation.get("topic_discovery")
        if topics:
            print(
                f"  topic discovery: {topics['novel_clusters_with_discovery']} novel cluster(s), precision {topics['precision']}"
            )
        print(f"  org matching : {ablation['org_matching']}")
    if "expert_sheet" in expert:
        e = expert["expert_sheet"]
        print(
            f"  expert sheet : {e['rated']}/{e['rows']} rated; precision@{e['k']} {e['precision_at_k']}"
        )

    if data["errors"]:
        print("\ncould not run:")
        for name, reason in data["errors"].items():
            print(f"  {name}: {reason}")


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run the evaluation suite and print it under algorithm / backtest / expert headings."""
    from cews.ai.llm import LLMClient, llm_enabled
    from cews.validation.evaluation_report import run_evaluation

    settings = _load(args)
    engine = _engine(settings)
    if not database_is_initialized(engine):
        print("database not initialized; run db-init first")
        return EXIT_FAILURE
    as_of = _parse_as_of(args.as_of)
    factory = create_session_factory(engine)
    client = LLMClient(settings) if llm_enabled(settings) else None
    try:
        with session_scope(factory) as session:
            try:
                report = run_evaluation(
                    session,
                    settings,
                    sections=tuple(args.only) if args.only else None,
                    as_of=as_of,
                    horizon_months=args.horizon,
                    k=args.k,
                    cleanup_backtest=not args.keep_backtest_rows,
                    llm_client=client,
                    expert_file=Path(args.expert_file) if args.expert_file else None,
                    store=not args.no_store,
                )
            except ScoringConfigError as exc:
                print(f"scoring configuration problem: {exc}")
                return EXIT_CONFIG
            except ValueError as exc:
                print(f"error: {exc}")
                return EXIT_FAILURE
    finally:
        if client is not None:
            client.close()

    if args.json:
        print(json.dumps(report.as_dict(), indent=2, default=str))
    else:
        _print_evaluation(report)
    return EXIT_OK


def cmd_alerts(args: argparse.Namespace) -> int:
    """List insights with their latest expert rating, or record a rating."""
    from cews.database.models import Insight
    from cews.validation.alert_metrics import AlertReviewError, latest_reviews, record_alert_review

    settings = _load(args)
    engine = _engine(settings)
    if not database_is_initialized(engine):
        print("database not initialized; run db-init first")
        return EXIT_FAILURE
    factory = create_session_factory(engine)

    if args.action == "rate":
        if args.item_id is None or not args.rating:
            print("usage: cews alerts rate <insight-id> <rating>  (see: cews alerts)")
            return EXIT_FAILURE
        with session_scope(factory) as session:
            try:
                review = record_alert_review(
                    session, args.item_id, args.rating, reviewer=args.reviewer, comment=args.comment
                )
            except AlertReviewError as exc:
                print(f"error: {exc}")
                return EXIT_FAILURE
            print(f"[{review.insight_id}] rated {review.rating}")
        return EXIT_OK

    with session_scope(factory) as session:
        insights = list(
            session.scalars(
                select(Insight).order_by(Insight.insight_date.desc(), Insight.id).limit(args.limit)
            )
        )
        if not insights:
            print("no insights yet; run: cews insights")
            return EXIT_OK
        reviews = latest_reviews(session)
        print(f"{len(insights)} insight(s); rate with: cews alerts rate <id> <rating>\n")
        for insight in insights:
            latest = reviews.get(insight.id)
            rated = latest.rating if latest else "-"
            print(f"[{insight.id:>4}] {rated:<21} {insight.insight_type:<20} {insight.title}")
        print(
            "\nratings: relevant, not_relevant, duplicate, too_late, insufficient_evidence, needs_investigation"
        )
    return EXIT_OK


def cmd_expert_export(args: argparse.Namespace) -> int:
    """Write a sheet of scored topics for an expert to rate."""
    from cews.validation.expert_review import export_expert_review

    settings = _load(args)
    engine = _engine(settings)
    if not database_is_initialized(engine):
        print("database not initialized; run db-init first")
        return EXIT_FAILURE
    factory = create_session_factory(engine)
    with session_scope(factory) as session:
        written = export_expert_review(session, Path(args.out), limit=args.limit)
    if written == 0:
        print("no scores stored yet; run: cews score")
        return EXIT_OK
    print(f"wrote {written} topic(s) to {args.out}")
    print("fill in the expert_rating and expert_comment columns, then check it with:")
    print(f"  cews evaluate --only expert_sheet --expert-file {args.out}")
    return EXIT_OK


def cmd_mcp_serve(args: argparse.Namespace) -> int:
    """Run the read-only MCP server on stdio, for an MCP client to launch."""
    import sys

    settings = _load(args)
    # The stdio transport speaks its protocol on stdout, so nothing else may write there: move
    # every console log handler to stderr before anything can log.
    for name in ("", "cews"):
        for handler in logging.getLogger(name).handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(
                handler, logging.FileHandler
            ):
                handler.setStream(sys.stderr)

    try:
        from cews.database.connection import DatabaseError, create_read_only_engine
        from cews.mcp_server.server import build_server
    except ImportError:
        print(
            "the MCP server needs the 'mcp' package: python -m pip install -r requirements-mcp.txt",
            file=sys.stderr,
        )
        return EXIT_FAILURE
    try:
        engine = create_read_only_engine(settings)
    except DatabaseError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILURE
    if not database_is_initialized(engine):
        print("database not initialized; run db-init first", file=sys.stderr)
        return EXIT_FAILURE
    server = build_server(settings, create_session_factory(engine))
    print("CEWS MCP server ready (read-only, stdio)", file=sys.stderr)
    server.run(transport="stdio")
    return EXIT_OK


def cmd_export_powerbi(args: argparse.Namespace) -> int:
    """Write the star-schema CSV files that Power BI Desktop loads."""
    from cews.database.connection import DatabaseError, create_read_only_engine
    from cews.exports.csv_export import ExportError
    from cews.exports.powerbi_export import export_powerbi

    settings = _load(args)
    try:
        engine = create_read_only_engine(settings)  # an export only reads; it cannot write
    except DatabaseError as exc:
        print(f"error: {exc}")
        return EXIT_FAILURE
    if not database_is_initialized(engine):
        print("database not initialized; run db-init first")
        return EXIT_FAILURE
    target = Path(args.out) if args.out else settings.export_directory / "powerbi"
    factory = create_session_factory(engine)
    try:
        with session_scope(factory) as session:
            result = export_powerbi(session, target)
    except (ExportError, OSError) as exc:
        print(f"export failed: {exc}")
        return EXIT_FAILURE

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
        return EXIT_OK
    print(f"wrote {len(result.row_counts)} tables to {result.output_directory}")
    for name, count in result.row_counts.items():
        print(f"  {name:<24}{count:>8} row(s)")
    print(f"\n{result.data_origin}")
    print(
        "open Power BI Desktop > Get data > Folder, and follow dashboards/powerbi/refresh_instructions.md"
    )
    return EXIT_OK


def _print_refresh(report: Any) -> None:
    title = "Refresh (dry run, nothing written)" if report.dry_run else "Refresh"
    print(f"{title} {report.job_id[:8]} [{report.trigger}]: {report.status.value}")
    if report.fetch is not None:
        totals = report.fetch["totals"]
        print(
            f"  fetch        {report.fetch['status']:<10} {totals.get('records_received', 0)} received, "
            f"{totals.get('records_inserted', 0)} new, {totals.get('records_updated', 0)} updated"
        )
        for source, status in sorted(report.fetch["sources"].items()):
            print(f"    {source:<22}{status}")
    for step in report.steps:
        note = f"  ! {step.error}" if step.error else ""
        print(f"  {step.name:<12} {step.status:<10} {step.seconds:>6.1f}s{note}")
    if report.error:
        print(f"\n{report.error}")


def cmd_refresh(args: argparse.Namespace) -> int:
    """Run one refresh cycle now: fetch every enabled source, then re-run the analysis."""
    from cews.scheduler.job_state import JobAlreadyRunningError
    from cews.scheduler.jobs import default_analysis_steps, run_refresh

    settings = _load(args)
    engine = _engine(settings)
    if not database_is_initialized(engine):
        print("database not initialized; run db-init first")
        return EXIT_FAILURE
    factory = create_session_factory(engine)
    steps = None
    if args.no_export:
        steps = [step for step in default_analysis_steps(settings) if step[0] != "export"]
    try:
        report = run_refresh(
            factory,
            settings,
            trigger="manual",
            dry_run=args.dry_run,
            sources=args.source or None,
            skip_fetch=args.skip_fetch,
            skip_analysis=args.skip_analysis,
            allow_mixed=args.allow_mixed,
            analysis_steps=steps,
        )
    except JobAlreadyRunningError as exc:
        print(f"error: {exc}")
        return EXIT_FAILURE
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, default=str))
    else:
        _print_refresh(report)
    return EXIT_OK if report.status.value in ("succeeded", "skipped") else EXIT_FAILURE


def cmd_run_scheduler(args: argparse.Namespace) -> int:
    """Run refresh cycles on an interval until stopped with Ctrl+C."""
    from cews.scheduler.scheduler import SchedulerDisabledError, run_scheduler

    settings = _load(args)
    engine = _engine(settings)
    if not database_is_initialized(engine):
        print("database not initialized; run db-init first")
        return EXIT_FAILURE
    interval = settings.fetch_interval_minutes
    first = (
        "immediately" if (args.now or settings.run_fetch_on_startup) else f"in {interval} minute(s)"
    )
    print(
        f"scheduler running: a refresh every {interval} minute(s), the first {first}. Ctrl+C to stop."
    )
    print("progress is written to the log; see 'cews jobs' for results.")
    print("Ctrl+C lets a running cycle finish; Ctrl+C twice aborts it.")
    try:
        run_scheduler(settings, create_session_factory(engine), run_now=True if args.now else None)
    except SchedulerDisabledError as exc:
        print(f"error: {exc}")
        return EXIT_FAILURE
    print("scheduler stopped")
    return EXIT_OK


def cmd_jobs(args: argparse.Namespace) -> int:
    """Show whether a refresh is running, when one last succeeded, and recent runs."""
    from cews.scheduler.job_state import (
        REFRESH_JOB,
        REFRESH_LOCK,
        consecutive_failures,
        last_success,
        lock_directory,
        lock_status,
        recent_job_runs,
    )

    settings = _load(args)
    engine = _engine(settings)
    if not database_is_initialized(engine):
        print("database not initialized; run db-init first")
        return EXIT_FAILURE
    factory = create_session_factory(engine)
    now = datetime.now(UTC)
    holder = lock_status(lock_directory(settings), REFRESH_LOCK)
    with session_scope(factory) as session:
        success = last_success(session, REFRESH_JOB)
        failures = consecutive_failures(session, REFRESH_JOB)
        runs = recent_job_runs(session, limit=args.limit)
        success_at = success.started_at if success else None
        # A row still saying "running" when no process actually holds the lock (holder is None)
        # is unambiguously stale: whatever ran it died without a chance to update its own row,
        # and the row will be properly marked "interrupted" the next time a cycle starts (see
        # mark_interrupted_runs) - but that could be hours or days away, and until then this is
        # the only thing that would otherwise make a long-dead run look like it is still going.
        # Flagged here from data already fetched, with no extra lock check and no database
        # write: `cews jobs` stays a pure read.
        rows: list[dict[str, Any]] = [
            {
                "job": run.job_name,
                "status": run.status,
                "trigger": run.trigger,
                "dry_run": run.dry_run,
                "started_at": run.started_at.isoformat(),
                "finished_at": run.finished_at.isoformat() if run.finished_at else None,
                "error": run.error_summary,
                "stale": run.status == "running" and holder is None,
            }
            for run in runs
        ]
    age = (now - success_at).total_seconds() / 60 if success_at else None
    interval = settings.fetch_interval_minutes
    overdue = age is not None and age > 2 * interval
    payload = {
        "scheduler_enabled": settings.enable_scheduler,
        "interval_minutes": interval,
        "running": holder is not None,
        "lock": holder,
        "last_success": success_at.isoformat() if success_at else None,
        "minutes_since_last_success": None if age is None else round(age, 1),
        "consecutive_failures": failures,
        "overdue": overdue,
        "recent_runs": rows,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return EXIT_OK

    print(
        f"scheduler   : {'enabled' if settings.enable_scheduler else 'DISABLED'}, every {interval} minute(s)"
    )
    print(f"running now : {'yes' if holder else 'no'}")
    if success_at is None:
        print("last success: never (run: cews refresh)")
    else:
        print(f"last success: {success_at:%Y-%m-%d %H:%M} UTC ({age:.0f} minute(s) ago)")
    if failures:
        print(f"failing     : the last {failures} refresh(es) failed")
    if overdue:
        print("WARNING     : no success for over twice the interval; is the scheduler running?")
    if any(row["stale"] for row in rows):
        print(
            "note        : row(s) below marked 'running' are stale; run `cews refresh` to clear them"
        )
    if rows:
        print(f"\n{'started (UTC)':<18}{'job':<9}{'trigger':<10}{'status':<11}note")
        print("-" * 70)
        for row in rows:
            if row["stale"]:
                note = "stale: process ended without finishing (not actually running)"
            elif row["dry_run"]:
                note = "dry run"
            else:
                note = row["error"] or ""
            print(
                f"{row['started_at'][:16].replace('T', ' '):<18}{row['job']:<9}{row['trigger']:<10}{row['status']:<11}{note[:55]}"
            )
    return EXIT_OK


def _print_agent_reply(reply: Any, persona_name: str) -> None:
    from cews.agents.core import format_tool_call

    print(f"[{persona_name}]\n{reply.text}")
    if reply.tool_calls:
        print("\nchecked:")
        for call in reply.tool_calls:
            print(f"  {format_tool_call(call)}")
    if reply.stopped_early:
        print("\n(stopped early: hit the tool-call limit before reaching a final answer)")


def cmd_agent(args: argparse.Namespace) -> int:
    """Ask an agent a question, verify an insight, or draft the weekly briefing.

    Launches a real MCP connection (the same server `cews mcp-serve` runs) for this one
    question, then closes it; use the dashboard's chat widget instead for a conversation with
    several questions in a row, which keeps that connection open across all of them.
    """
    from cews.agents.personas import ANALYST, BRIEFING_WRITER, FACT_CHECKER
    from cews.agents.session import AgentRuntime, RuntimeNotReadyError
    from cews.ai.llm import LLMError, llm_enabled

    settings = _load(args)
    if not llm_enabled(settings):
        print("no LLM is configured (LLM_PROVIDER=none); set LLM_PROVIDER and LLM_MODEL in .env")
        return EXIT_FAILURE
    engine = _engine(settings)
    if not database_is_initialized(engine):
        print("database not initialized; run db-init first")
        return EXIT_FAILURE

    if args.action == "verify":
        if args.arg is None:
            print("usage: cews agent verify <insight-id>")
            return EXIT_FAILURE
        persona, message = FACT_CHECKER, f"Check insight id {args.arg}."
    elif args.action == "brief":
        persona, message = BRIEFING_WRITER, "Write this week's briefing."
    else:
        if not args.arg:
            print('usage: cews agent ask "<question>"')
            return EXIT_FAILURE
        persona, message = ANALYST, args.arg

    runtime = AgentRuntime(settings, env_file=args.env_file)
    try:
        runtime.start(timeout=60)
        reply = runtime.ask(persona, message, timeout=120)
    except RuntimeNotReadyError as exc:
        print(f"error: {exc}")
        return EXIT_FAILURE
    except TimeoutError:
        print("error: the agent did not answer in time")
        return EXIT_FAILURE
    except LLMError as exc:
        print(f"error: {exc}")
        return EXIT_FAILURE
    finally:
        runtime.stop()

    if args.json:
        print(json.dumps(reply.as_dict(), indent=2))
    else:
        _print_agent_reply(reply, persona.name)
    return EXIT_OK


def cmd_review(args: argparse.Namespace) -> int:
    """List review items, or approve/reject one by id."""
    from cews.normalization.review_actions import (
        ReviewActionError,
        approve_review_item,
        reject_review_item,
    )

    settings = _load(args)
    engine = _engine(settings)
    if not database_is_initialized(engine):
        print("database not initialized; run db-init first")
        return EXIT_FAILURE

    if args.action in ("approve", "reject"):
        if args.item_id is None:
            print(f"usage: cews review {args.action} <item-id>  (see: cews review)")
            return EXIT_FAILURE
        act = approve_review_item if args.action == "approve" else reject_review_item
        with session_scope(create_session_factory(engine)) as session:
            try:
                result = act(session, args.item_id)
            except ReviewActionError as exc:
                print(f"error: {exc}")
                return EXIT_FAILURE
        print(f"[{result.item_id}] {result.decision}: {result.effect}")
        return EXIT_OK

    with session_scope(create_session_factory(engine)) as session:
        items = list(
            session.scalars(
                select(ReviewQueueItem)
                .where(ReviewQueueItem.status == "pending")
                .order_by(ReviewQueueItem.queue_type, ReviewQueueItem.id)
                .limit(args.limit)
            )
        )
        if not items:
            print("nothing waiting for review")
            return EXIT_OK
        print(f"{len(items)} item(s) awaiting review\n")
        for item in items:
            payload = item.payload_json or {}
            subject = (
                payload.get("organizations")
                or payload.get("raw_name")
                or payload.get("label")
                or item.subject_ref
            )
            print(f"[{item.id}] [{item.queue_type}] {subject}")
            print(f"    reason : {payload.get('reason', '-')}")
            for key in (
                "candidates",
                "similar_to",
                "linked_to",
                "similarity",
                "action",
                "record_count",
                "novelty",
                "nearest_existing_topic",
            ):
                if payload.get(key) is not None:
                    print(f"    {key:7}: {payload[key]}")
        print("\ndecide with: cews review approve <id>   or   cews review reject <id>")
    return EXIT_OK


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser. ``--env-file`` and ``-v`` are accepted after the command."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--env-file", help="dotenv file to load (default: .env in the project root)"
    )
    common.add_argument(
        "-v", "--verbose", action="store_true", help="show informational log output"
    )

    parser = argparse.ArgumentParser(prog="cews", description="Competitor Early Warning System")
    parser.add_argument("--version", action="version", version=f"cews {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser(
        "check-env", parents=[common], help="validate configuration, packages and database"
    ).set_defaults(func=cmd_check_env)
    init = sub.add_parser(
        "db-init", parents=[common], help="create/upgrade the schema and load the taxonomy"
    )
    init.add_argument("--no-taxonomy", action="store_true", help="skip loading the taxonomy file")
    init.set_defaults(func=cmd_db_init)
    sub.add_parser(
        "db-status", parents=[common], help="show schema revision and record counts"
    ).set_defaults(func=cmd_db_status)
    seed = sub.add_parser("seed-demo", parents=[common], help="load synthetic demo data")
    seed.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    seed.add_argument("--months", type=int, default=30, help="months of history (12-120)")
    seed.add_argument("--scale", type=float, default=1.0, help="activity multiplier (0.05-10)")
    seed.add_argument(
        "--write-fixtures",
        nargs="?",
        const="data/samples/demo",
        metavar="DIR",
        help="also write JSONL fixture files (default dir: data/samples/demo)",
    )
    seed.add_argument(
        "--allow-mixed",
        action="store_true",
        help="allow loading demo data into a database that holds live records",
    )
    seed.set_defaults(func=cmd_seed_demo)
    sources = sub.add_parser(
        "sources", parents=[common], help="list sources, adapters, checkpoints and circuit state"
    )
    sources.add_argument("--health", action="store_true", help="also contact each enabled source")
    sources.set_defaults(func=cmd_sources)
    fetch = sub.add_parser("fetch", parents=[common], help="collect enabled sources once")
    fetch.add_argument(
        "--source", action="append", metavar="ID", help="only this source (repeatable)"
    )
    fetch.add_argument("--dry-run", action="store_true", help="fetch and parse but write nothing")
    fetch.add_argument(
        "--full", action="store_true", help="ignore checkpoints; collect the full lookback"
    )
    fetch.add_argument(
        "--allow-mixed",
        action="store_true",
        help="allow writing live records into a database that holds demo data",
    )
    fetch.set_defaults(func=cmd_fetch)
    normalize = sub.add_parser(
        "normalize", parents=[common], help="resolve organizations and assign topics"
    )
    normalize.add_argument("--limit", type=int, help="process at most this many records")
    normalize.add_argument(
        "--reprocess", action="store_true", help="redo records that were already normalized"
    )
    normalize.add_argument(
        "--no-dedupe", action="store_true", help="skip the cross-source duplicate pass"
    )
    normalize.set_defaults(func=cmd_normalize)

    features = sub.add_parser(
        "features", parents=[common], help="count activity and compute trend features"
    )
    features.add_argument("--as-of", help="compute as of this date (YYYY-MM-DD, default: today)")
    features.add_argument(
        "--rebuild", action="store_true", help="drop stored periods that no longer have records"
    )
    features.add_argument("--top", type=int, default=12, help="how many rows to show (default 12)")
    features.add_argument("--json", action="store_true", help="print the full summary as JSON")
    features.set_defaults(func=cmd_features)
    forecast = sub.add_parser(
        "forecast", parents=[common], help="forecast activity and flag unusual months"
    )
    forecast.add_argument("--as-of", help="forecast from this date (YYYY-MM-DD, default: today)")
    forecast.add_argument("--horizon", type=int, default=6, help="months ahead (default 6)")
    forecast.add_argument("--top", type=int, default=12, help="how many rows to show (default 12)")
    forecast.add_argument("--no-anomalies", action="store_true", help="skip the anomaly pass")
    forecast.add_argument("--no-store", action="store_true", help="compute without saving")
    forecast.add_argument("--json", action="store_true", help="print the full detail as JSON")
    forecast.set_defaults(func=cmd_forecast)
    score = sub.add_parser("score", parents=[common], help="score the stored features")
    score.add_argument("--as-of", help="score the features on or before this date (YYYY-MM-DD)")
    score.add_argument("--top", type=int, default=15, help="how many rows to show (default 15)")
    score.add_argument(
        "--type",
        choices=("all", "trend", "opportunity", "innovation", "threat", "confidence"),
        default="all",
        help="which scores to show (default all)",
    )
    score.add_argument(
        "--low-confidence", action="store_true", help="show only results below the threshold"
    )
    score.add_argument("--no-store", action="store_true", help="compute without saving")
    score.add_argument("--json", action="store_true", help="print the full breakdown as JSON")
    score.set_defaults(func=cmd_score)
    competitors = sub.add_parser(
        "competitors", parents=[common], help="rank and show the monitored competitors"
    )
    competitors.add_argument("--window-months", type=int, help="activity window (default 12)")
    competitors.add_argument(
        "--include-all-types",
        action="store_true",
        help="also rank universities, hospitals and agencies",
    )
    competitors.add_argument("--json", action="store_true", help="print the full breakdown as JSON")
    competitors.add_argument(
        "--no-persist", action="store_true", help="do not update the organization flags"
    )
    competitors.set_defaults(func=cmd_competitors)

    discover_topics = sub.add_parser(
        "discover-topics",
        parents=[common],
        help="cluster unmatched text and find topics the taxonomy has no name for",
    )
    discover_topics.add_argument(
        "--limit",
        type=int,
        default=2000,
        help="consider at most this many unmatched records (default 2000)",
    )
    discover_topics.add_argument(
        "--min-cluster-size",
        type=int,
        default=5,
        help="minimum records to form a cluster (default 5)",
    )
    discover_topics.add_argument(
        "--max-distance",
        type=float,
        default=0.35,
        help="DBSCAN eps in cosine distance (default 0.35)",
    )
    discover_topics.add_argument(
        "--no-store", action="store_true", help="find candidates without saving"
    )
    discover_topics.add_argument(
        "--json", action="store_true", help="print the full detail as JSON"
    )
    discover_topics.set_defaults(func=cmd_discover_topics)

    extract_announcements = sub.add_parser(
        "extract-announcements",
        parents=[common],
        help="classify company announcements (type, partners, therapeutic area, modality)",
    )
    extract_announcements.add_argument(
        "--limit",
        type=int,
        default=500,
        help="classify at most this many announcements (default 500)",
    )
    extract_announcements.add_argument(
        "--top", type=int, default=15, help="how many results to show (default 15)"
    )
    extract_announcements.add_argument(
        "--no-store", action="store_true", help="classify without saving"
    )
    extract_announcements.add_argument(
        "--json", action="store_true", help="print full detail as JSON"
    )
    extract_announcements.set_defaults(func=cmd_extract_announcements)

    insights = sub.add_parser(
        "insights", parents=[common], help="turn scores into plain-language findings"
    )
    insights.add_argument("--as-of", help="use scores on or before this date (YYYY-MM-DD)")
    insights.add_argument("--top", type=int, default=15, help="how many to show (default 15)")
    insights.add_argument("--no-store", action="store_true", help="find without saving")
    insights.add_argument("--json", action="store_true", help="print full detail as JSON")
    insights.set_defaults(func=cmd_insights)

    evaluate = sub.add_parser(
        "evaluate",
        parents=[common],
        help="evaluate the system: data quality, backtest, robustness, alerts, AI layers",
    )
    evaluate.add_argument("--only", nargs="+", metavar="SECTION", help="run only these sections")
    evaluate.add_argument("--as-of", help="evaluate as of this date (YYYY-MM-DD, default: today)")
    evaluate.add_argument("--horizon", type=int, default=3, help="backtest look-ahead in months")
    evaluate.add_argument("--k", type=int, default=5, help="Precision@K cutoff for the backtest")
    evaluate.add_argument(
        "--keep-backtest-rows",
        action="store_true",
        help="leave the historical feature/score rows the backtest writes (default: remove them)",
    )
    evaluate.add_argument("--expert-file", help="a completed expert review sheet to summarize")
    evaluate.add_argument("--no-store", action="store_true", help="do not save the results")
    evaluate.add_argument("--json", action="store_true", help="print the full result as JSON")
    evaluate.set_defaults(func=cmd_evaluate)

    alerts = sub.add_parser(
        "alerts", parents=[common], help="list insights with expert ratings, or rate one"
    )
    alerts.add_argument("action", nargs="?", default="list", choices=("list", "rate"))
    alerts.add_argument("item_id", nargs="?", type=int, help="insight id, for rate")
    alerts.add_argument("rating", nargs="?", help="the rating, for rate")
    alerts.add_argument("--reviewer", help="who is rating")
    alerts.add_argument("--comment", help="an optional comment")
    alerts.add_argument("--limit", type=int, default=30, help="how many insights to list")
    alerts.set_defaults(func=cmd_alerts)

    expert_export = sub.add_parser(
        "expert-export", parents=[common], help="write a topic sheet for an expert to rate"
    )
    expert_export.add_argument("--out", required=True, help="CSV file to write")
    expert_export.add_argument("--limit", type=int, help="export only the top N topics")
    expert_export.set_defaults(func=cmd_expert_export)

    mcp_serve = sub.add_parser(
        "mcp-serve",
        parents=[common],
        help="run the read-only MCP server (stdio) for an MCP client to launch",
    )
    mcp_serve.set_defaults(func=cmd_mcp_serve)

    export_powerbi_parser = sub.add_parser(
        "export-powerbi",
        parents=[common],
        help="write star-schema CSV files for Power BI Desktop",
    )
    export_powerbi_parser.add_argument(
        "--out", help="output folder (default: <export directory>/powerbi)"
    )
    export_powerbi_parser.add_argument("--json", action="store_true", help="print a JSON summary")
    export_powerbi_parser.set_defaults(func=cmd_export_powerbi)

    refresh = sub.add_parser(
        "refresh",
        parents=[common],
        help="run one refresh cycle now: fetch every enabled source, then re-run the analysis",
    )
    refresh.add_argument("--source", action="append", help="fetch only this source id (repeatable)")
    refresh.add_argument(
        "--dry-run", action="store_true", help="fetch without writing; skip analysis"
    )
    refresh.add_argument("--skip-fetch", action="store_true", help="only re-run the analysis")
    refresh.add_argument("--skip-analysis", action="store_true", help="only fetch")
    refresh.add_argument("--no-export", action="store_true", help="skip the Power BI export step")
    refresh.add_argument(
        "--allow-mixed", action="store_true", help="allow live data in a demo database"
    )
    refresh.add_argument("--json", action="store_true", help="print the full result as JSON")
    refresh.set_defaults(func=cmd_refresh)

    run_scheduler_parser = sub.add_parser(
        "run-scheduler",
        parents=[common],
        help="run refresh cycles every FETCH_INTERVAL_MINUTES until stopped (Ctrl+C)",
    )
    run_scheduler_parser.add_argument(
        "--now",
        action="store_true",
        help="run the first cycle immediately instead of after one interval",
    )
    run_scheduler_parser.set_defaults(func=cmd_run_scheduler)

    jobs = sub.add_parser(
        "jobs", parents=[common], help="show whether a refresh is running and recent runs"
    )
    jobs.add_argument("--limit", type=int, default=8, help="how many recent runs to show")
    jobs.add_argument("--json", action="store_true", help="print as JSON")
    jobs.set_defaults(func=cmd_jobs)

    agent = sub.add_parser(
        "agent",
        parents=[common],
        help="ask an agent a question, verify an insight, or draft the weekly briefing",
    )
    agent.add_argument("action", nargs="?", default="ask", choices=("ask", "verify", "brief"))
    agent.add_argument("arg", nargs="?", help="the question (ask) or insight id (verify)")
    agent.add_argument("--json", action="store_true", help="print the full result as JSON")
    agent.set_defaults(func=cmd_agent)

    review = sub.add_parser(
        "review", parents=[common], help="show, approve or reject items awaiting human review"
    )
    review.add_argument(
        "action",
        nargs="?",
        default="list",
        choices=("list", "approve", "reject"),
        help="list (default), approve <id>, or reject <id>",
    )
    review.add_argument(
        "item_id", nargs="?", type=int, help="the review item id, for approve/reject"
    )
    review.add_argument("--limit", type=int, default=20, help="maximum items to show (for list)")
    review.set_defaults(func=cmd_review)

    reset = sub.add_parser("reset-demo", parents=[common], help="delete all synthetic data")
    reset.add_argument("--yes", action="store_true", help="confirm deletion")
    reset.set_defaults(func=cmd_reset_demo)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except SettingsError as exc:
        print(exc)
        return EXIT_CONFIG
    except ScoringConfigError as exc:
        print(f"scoring configuration problem: {exc}")
        return EXIT_CONFIG
    except (DatabaseError, TaxonomyError, RegistryError, ValueError) as exc:
        print(f"error: {exc}")
        return EXIT_FAILURE
    except BrokenPipeError:  # output piped into head, less and friends
        with contextlib.suppress(OSError):
            sys.stdout.close()
        return EXIT_OK
    finally:
        for engine in _created_engines:
            engine.dispose()
        _created_engines.clear()


if __name__ == "__main__":
    sys.exit(main())
