#!/usr/bin/env python3
"""Bootstrap the CEWS (Competitor Early Warning System) repository scaffold.

Usage (this file lives at scripts/bootstrap_project.py):

    python scripts/bootstrap_project.py             # create missing files and folders
    python scripts/bootstrap_project.py --tree      # ... and print the resulting tree
    python scripts/bootstrap_project.py --dry-run   # show what would happen, write nothing
    python scripts/bootstrap_project.py --validate  # only check the scaffold is complete
    python scripts/bootstrap_project.py --force     # also overwrite non-empty files
    python scripts/bootstrap_project.py --verbose   # list every created/skipped item

Behaviour:
  * Safe to re-run. A non-empty existing file is never overwritten unless --force.
    An existing empty file is populated.
  * Root directory: --root if given; otherwise the parent of the "scripts" directory this
    file lives in; otherwise ./cews under the current working directory.
  * Exit status: 0 on success, 1 if critical scaffolding failed or is missing.

Standard library only. Requires Python 3.11+.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath

LOGGER = logging.getLogger("cews.bootstrap")

EXIT_OK = 0
EXIT_CRITICAL_FAILURE = 1

PROJECT_NAME = "cews"
DEFAULT_PHASE = 13
SUMMARY_SAMPLE_SIZE = 15


# --------------------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------------------
class Status(StrEnum):
    """Outcome of one scaffolding action."""

    CREATED = "created"
    OVERWRITTEN = "overwritten"
    SKIPPED = "skipped"
    FAILED = "failed"


class Kind(StrEnum):
    """Type of filesystem item."""

    DIR = "dir"
    FILE = "file"


@dataclass(frozen=True)
class Result:
    """Outcome of processing a single directory or file."""

    path: str
    kind: Kind
    status: Status
    critical: bool = False
    detail: str = ""


@dataclass
class Report:
    """Collected results of a bootstrap run."""

    root: Path
    dry_run: bool = False
    force: bool = False
    results: list[Result] = field(default_factory=list)

    def select(self, status: Status | None = None, kind: Kind | None = None) -> list[Result]:
        """Return results filtered by status and/or kind."""
        return [
            r
            for r in self.results
            if (status is None or r.status == status) and (kind is None or r.kind == kind)
        ]

    def count(self, status: Status, kind: Kind | None = None) -> int:
        """Count results with the given status (and kind)."""
        return len(self.select(status, kind))

    def failed(self) -> list[Result]:
        """Return all failed results."""
        return self.select(Status.FAILED)

    def critical_failures(self) -> list[Result]:
        """Return failed results flagged as critical."""
        return [r for r in self.failed() if r.critical]


@dataclass(frozen=True)
class FileSpec:
    """A file to create: repo-relative POSIX path, text content, criticality."""

    path: str
    content: str
    critical: bool = False


@dataclass(frozen=True)
class DirSpec:
    """A directory to create: repo-relative POSIX path and criticality."""

    path: str
    critical: bool = False


@dataclass(frozen=True)
class Manifest:
    """Everything the bootstrap script creates."""

    directories: tuple[DirSpec, ...]
    files: tuple[FileSpec, ...]


# --------------------------------------------------------------------------------------
# Phase mapping (used only for placeholder docstrings and skip reasons)
# --------------------------------------------------------------------------------------
_PHASE_RULES: tuple[tuple[str, int], ...] = (
    ("src/cews/ingestion/adapters/", 4),
    ("src/cews/ingestion/", 3),
    ("src/cews/normalization/", 5),
    ("src/cews/discovery/competitor_discovery", 5),
    ("src/cews/discovery/", 8),
    ("src/cews/features/", 6),
    ("src/cews/scoring/", 7),
    ("src/cews/forecasting/", 8),
    ("src/cews/ai/", 9),
    ("src/cews/ai/llm/", 9),
    ("src/cews/insights/", 10),
    ("src/cews/validation/", 11),
    ("src/cews/scheduler/", 12),
    ("src/cews/mcp_server/", 12),
    ("src/cews/exports/", 12),
    ("src/cews/api/", 12),
    ("src/cews/database/", 2),
    ("src/cews/", 2),
    ("scripts/fetch_all", 4),
    ("scripts/run_analysis", 7),
    ("scripts/run_evaluation", 11),
    ("scripts/run_scheduler", 12),
    ("scripts/export_powerbi", 12),
    ("scripts/", 2),
    ("dashboards/", 12),
    ("migrations/", 2),
)

TEST_PHASES: dict[str, int] = {
    "test_settings": 2,
    "test_database": 2,
    "test_retry": 3,
    "test_rate_limiter": 3,
    "test_checkpoints": 3,
    "test_source_adapter_contract": 3,
    "test_source_schemas": 4,
    "test_ingestion_pipeline": 4,
    "test_competitor_discovery": 5,
    "test_deduplication": 5,
    "test_normalization": 6,
    "test_growth": 6,
    "test_momentum": 6,
    "test_velocity": 6,
    "test_consistency": 6,
    "test_trend_score": 7,
    "test_innovation_score": 7,
    "test_threat_score": 7,
    "test_opportunity_score": 7,
    "test_confidence_score": 7,
    "test_analysis_pipeline": 7,
    "test_score_ranges": 7,
    "test_forecasting": 8,
    "test_anomaly_detection": 8,
    "test_ai_topic_discovery": 9,
    "test_ai_announcement_extraction": 9,
    "test_ai_org_matching": 9,
    "test_rule_engine": 10,
    "test_ranking_metrics": 11,
    "test_forecast_metrics": 11,
    "test_required_fields": 11,
    "test_uniqueness": 11,
    "test_referential_integrity": 11,
    "test_date_validity": 11,
    "test_scheduler": 12,
    "test_exports": 12,
    "test_api": 12,
    "test_demo_workflow": 12,
}


def phase_for(rel_path: str) -> int:
    """Return the implementation phase in which ``rel_path`` is planned to be filled in."""
    posix = PurePosixPath(rel_path).as_posix()
    if posix.startswith("tests/"):
        return TEST_PHASES.get(PurePosixPath(posix).stem, DEFAULT_PHASE)
    for prefix, phase in _PHASE_RULES:
        if posix.startswith(prefix):
            return phase
    return DEFAULT_PHASE


# --------------------------------------------------------------------------------------
# Repository layout
# --------------------------------------------------------------------------------------
SRC_MODULES: dict[str, tuple[str, ...]] = {
    "src/cews": ("cli", "settings", "constants", "logging_config"),
    "src/cews/database": ("connection", "models", "migrations", "repositories", "views"),
    "src/cews/demo": ("generator", "loader"),
    "src/cews/ingestion": (
        "base",
        "stdlib_transport",
        "errors",
        "results",
        "registry",
        "http_client",
        "rate_limiter",
        "retry",
        "checkpoints",
        "orchestrator",
    ),
    "src/cews/ingestion/adapters": (
        "common",
        "clinical_trials_gov",
        "pubmed",
        "europe_pmc",
        "openalex",
        "nih_reporter",
        "generic_rss",
        "patents_fixture",
        "patents_uspto_bulk",
    ),
    "src/cews/normalization": (
        "organizations",
        "resolver",
        "pipeline",
        "topics",
        "dates",
        "identifiers",
        "deduplication",
        "review_actions",
    ),
    "src/cews/ai": (
        "embeddings",
        "topic_discovery",
        "announcement_extraction",
        "org_matching",
    ),
    "src/cews/dashboard": (
        "queries",
        "charts",
    ),
    "src/cews/features": (
        "pipeline",
        "time_windows",
        "activity_counts",
        "growth",
        "momentum",
        "velocity",
        "consistency",
        "competition",
        "source_agreement",
        "sample_confidence",
    ),
    "src/cews/scoring": (
        "base",
        "config",
        "inputs",
        "store",
        "pipeline",
        "modifiers",
        "area_threat",
        "normalization",
        "trend_score",
        "innovation_score",
        "threat_score",
        "opportunity_score",
        "confidence_score",
        "explanations",
    ),
    "src/cews/forecasting": (
        "baseline",
        "pipeline",
        "exponential_smoothing",
        "arima",
        "backtesting",
        "model_selection",
    ),
    "src/cews/discovery": ("competitor_discovery", "new_market_entry", "anomaly_detection"),
    "src/cews/insights": (
        "rule_engine",
        "templates",
        "evidence",
        "alert_generator",
        "report_generator",
    ),
    "src/cews/validation": (
        "data_quality",
        "backtest",
        "forecast_metrics",
        "ranking_metrics",
        "alert_metrics",
        "ai_ablation",
        "benchmark_topics",
        "robustness",
        "expert_review",
        "evaluation_report",
    ),
    "src/cews/scheduler": ("jobs", "scheduler", "job_state"),
    "src/cews/mcp_server": ("tools", "server"),
    "src/cews/exports": ("csv_export", "powerbi_export", "report_export"),
    "src/cews/api": ("app", "dependencies"),
    "src/cews/api/routes": (
        "health",
        "sources",
        "competitors",
        "trends",
        "insights",
        "evidence",
        "evaluations",
        "refresh",
    ),
}

SCRIPT_NAMES: tuple[str, ...] = (
    "initialize_database",
    "seed_demo_data",
    "fetch_all",
    "run_scheduler",
    "run_analysis",
    "run_evaluation",
    "export_powerbi",
    "validate_environment",
    "reset_demo",
)

TEST_FILES: dict[str, tuple[str, ...]] = {
    "tests/unit": (
        "test_settings",
        "test_normalization",
        "test_growth",
        "test_momentum",
        "test_velocity",
        "test_consistency",
        "test_trend_score",
        "test_innovation_score",
        "test_threat_score",
        "test_opportunity_score",
        "test_confidence_score",
        "test_competitor_discovery",
        "test_deduplication",
        "test_rule_engine",
        "test_forecasting",
        "test_anomaly_detection",
        "test_retry",
        "test_rate_limiter",
        "test_checkpoints",
        "test_ranking_metrics",
        "test_forecast_metrics",
        "test_ai_topic_discovery",
        "test_ai_announcement_extraction",
        "test_ai_org_matching",
    ),
    "tests/contract": ("test_source_adapter_contract", "test_source_schemas"),
    "tests/integration": (
        "test_database",
        "test_ingestion_pipeline",
        "test_analysis_pipeline",
        "test_scheduler",
        "test_exports",
        "test_api",
    ),
    "tests/data_quality": (
        "test_required_fields",
        "test_uniqueness",
        "test_referential_integrity",
        "test_date_validity",
        "test_score_ranges",
    ),
    "tests/end_to_end": ("test_demo_workflow",),
}

DOC_FILES: dict[str, str] = {
    "architecture": "Architecture",
    "data_sources": "Data sources",
    "entity_resolution": "Organizations and topics",
    "demo_data": "Demo data",
    "scoring_methodology": "Scoring methodology",
    "ai_layers": "AI layers",
    "validation_methodology": "Validation methodology",
    "configuration_guide": "Configuration guide",
    "troubleshooting": "Troubleshooting",
    "demo_script": "Demo script",
}

POWERBI_DOCS: dict[str, str] = {
    "README": "Power BI",
    "data_model": "Power BI data model",
    "dashboard_specification": "Power BI dashboard specification",
    "dax_measures": "Power BI DAX measures",
    "refresh_instructions": "Power BI refresh instructions",
}

# Empty directories kept in git through a .gitkeep file.
GITKEEP_DIRS: tuple[str, ...] = (
    "data/raw",
    "data/staging",
    "data/processed",
    "data/exports",
    "data/exports/powerbi",
    "data/fixtures",
    "data/samples",
    "reports/generated",
    "reports/templates",
    "logs",
    "migrations/versions",
    "dashboards/streamlit/pages",
)

CRITICAL_DIRS: frozenset[str] = frozenset(
    {"src", "src/cews", "config", "scripts", "tests", "migrations", "migrations/sql", "data"}
)

# --------------------------------------------------------------------------------------
# Dependencies (single source of truth for pyproject.toml and requirements*.txt)
# --------------------------------------------------------------------------------------
RUNTIME_DEPS: tuple[str, ...] = (
    "pydantic>=2.6,<3",
    "pydantic-settings>=2.2,<3",
    "tzdata>=2024.1,<2100",
    "python-dotenv>=1.0,<2",
    "PyYAML>=6.0,<7",
    "httpx>=0.27,<1",
    "tenacity>=8.2,<10",
    "feedparser>=6.0,<7",
    "defusedxml>=0.7,<1",
    "SQLAlchemy>=2.0,<3",
    "alembic>=1.13,<2",
    "psycopg[binary]>=3.1,<4",
    "pandas>=2.1,<3",
    "numpy>=1.26,<3",
    "scipy>=1.11,<2",
    "scikit-learn>=1.3,<2",
    "statsmodels>=0.14,<1",
    "rapidfuzz>=3.6,<4",
    "APScheduler>=3.10,<4",
    "portalocker>=2.8,<4",
    "fastapi>=0.110,<1",
    "uvicorn[standard]>=0.29,<1",
    "streamlit>=1.32,<2",
    "matplotlib>=3.8,<4",  # charts are rendered server-side as images (no JavaScript)
    "Jinja2>=3.1,<4",
    "openpyxl>=3.1,<4",
)

AI_DEPS: tuple[str, ...] = ("sentence-transformers>=2.7,<6",)
MCP_DEPS: tuple[str, ...] = (
    "mcp>=1.2,<2",
    # For cews.agents.documents: whole-document reading for a chat attachment, no chunking, no
    # ML layout models (see the module for why not docling).
    "pypdf>=4.0,<6",
    "python-docx>=1.1,<2",
)

DEV_DEPS: tuple[str, ...] = (
    "pytest>=8.0,<10",
    "pytest-cov>=5.0,<8",
    "pytest-mock>=3.12,<4",
    "responses>=0.25,<1",
    "respx>=0.21,<1",
    "freezegun>=1.4,<2",
    "hypothesis>=6.100,<7",
    "ruff>=0.4,<1",
    "black>=24.0,<27",
    "mypy>=1.9,<2",
    "types-PyYAML>=6.0,<7",
)


def _toml_list(items: Sequence[str]) -> str:
    return "[\n" + "".join(f'    "{item}",\n' for item in items) + "]"


# --------------------------------------------------------------------------------------
# File templates
# --------------------------------------------------------------------------------------
ENV_EXAMPLE = """\
# CEWS configuration template.
# Copy to .env and edit. NEVER commit a real .env file. Values are validated at startup
# (src/cews/settings.py, Phase 2). Comments sit on their own lines on purpose.

# ---- Application ----
APP_ENV=development
LOG_LEVEL=INFO
TIMEZONE=Asia/Kolkata

# ---- Database ----
# sqlite (default, zero setup) or postgresql (start it with: docker compose up -d postgres)
DATABASE_BACKEND=sqlite
# Local development credentials only.
DATABASE_URL=postgresql+psycopg://cews:cews@localhost:5432/cews
SQLITE_PATH=./data/cews.db
POSTGRES_USER=cews
POSTGRES_PASSWORD=cews
POSTGRES_DB=cews
POSTGRES_PORT=5432

# ---- Scheduler and collection ----
# Fetch frequency in minutes. Two-hour fetching is for freshness only; trend features use
# monthly, quarterly or yearly windows.
FETCH_INTERVAL_MINUTES=120
RUN_FETCH_ON_STARTUP=false
ENABLE_SCHEDULER=true
MAX_CONCURRENT_SOURCE_JOBS=3
SOURCE_REQUEST_TIMEOUT_SECONDS=30
SOURCE_MAX_RETRIES=3
DEFAULT_LOOKBACK_DAYS=1095
INCREMENTAL_LOOKBACK_DAYS=7
# Source endpoints, rate limits, page sizes and refresh windows.
SOURCE_REGISTRY_FILE=./config/source_registry.yaml
# After this many failed runs in a row a source is paused for the cooldown period.
CIRCUIT_BREAKER_FAILURE_THRESHOLD=3
CIRCUIT_BREAKER_COOLDOWN_MINUTES=360

# ---- Competitors ----
# COMPETITOR_MODE: AUTO | MANUAL | HYBRID
COMPETITOR_MODE=HYBRID
TOP_COMPETITORS=20
COMPETITOR_INCLUDE=Pfizer,Roche,Novartis,Merck,AstraZeneca
COMPETITOR_EXCLUDE=
# Optional YAML file for long include/exclude lists. Include/exclude above still work.
COMPETITOR_CONFIG_FILE=
MIN_COMPETITOR_EVIDENCE_COUNT=5

# ---- Topics ----
THERAPEUTIC_AREAS=Oncology,Neurology,Immunology,Rare Diseases,Gene Therapy,RNA Therapeutics,Cancer Vaccines,CAR-T
TOPIC_TAXONOMY_FILE=./config/topic_taxonomy.yaml

# ---- Sources ----
ENABLE_CLINICAL_TRIALS_GOV=true
ENABLE_PUBMED=true
ENABLE_EUROPE_PMC=true
ENABLE_GENERIC_RSS=true
# OpenAlex requires a free API key since February 2026 (see docs/data_sources.md).
ENABLE_OPENALEX=false
ENABLE_NIH_REPORTER=false
# Patents: PatentsView moved to the USPTO Open Data Portal in March 2026 (see docs/data_sources.md).
ENABLE_PATENTS=false
ENABLE_EPO_OPS=false
# Optional credentials. Leave blank when unused. Never log or commit real values.
NCBI_API_KEY=
NCBI_EMAIL=
OPENALEX_API_KEY=
USPTO_ODP_API_KEY=
EPO_CONSUMER_KEY=
EPO_CONSUMER_SECRET=

# ---- AI layers (optional; everything works without them) ----
ENABLE_AI_TOPIC_DISCOVERY=false
ENABLE_AI_ANNOUNCEMENT_EXTRACTION=false
ENABLE_AI_ORG_MATCHING=false
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
# LLM provider for optional AI features (announcement extraction, Phase 9). Everything works
# with this left as "none" - the deterministic fallback runs instead.
#   none               - no LLM calls are made (default)
#   ollama             - a local server, e.g. Ollama running qwen2.5:3b
#   openai_compatible  - a hosted or self-hosted OpenAI-style endpoint: NVIDIA NIM, Groq,
#                        Together, or a company gateway. Set LLM_API_KEY for these.
LLM_PROVIDER=none
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen2.5:3b
LLM_API_KEY=
LLM_TIMEOUT_SECONDS=30
LLM_MAX_RETRIES=2
# NVIDIA NIM example (uncomment and fill in LLM_API_KEY):
#   LLM_PROVIDER=openai_compatible
#   LLM_BASE_URL=https://integrate.api.nvidia.com/v1
#   LLM_MODEL=meta/llama-3.1-8b-instruct
AI_TOPIC_NOVELTY_THRESHOLD=0.55
AI_ORG_MATCH_AUTO_THRESHOLD=0.92
AI_ORG_MATCH_REVIEW_THRESHOLD=0.80
AI_MAX_DOCS_PER_RUN=5000

# ---- Scoring ----
SCORING_CONFIG_FILE=./config/scoring_weights.yaml
BENCHMARK_TOPICS_FILE=./config/benchmark_topics.yaml
ANNOUNCEMENT_LABELS_FILE=./config/announcement_eval_labels.yaml
MIN_TOPIC_SAMPLE_SIZE=10
# percentile | winsorized_minmax | robust_zscore | minmax
NORMALIZATION_METHOD=percentile
NORMALIZATION_WINSOR_LOWER=0.05
NORMALIZATION_WINSOR_UPPER=0.95
CONFIDENCE_MIN_SCORE=0
CONFIDENCE_MAX_SCORE=100

# ---- Alerts ----
ALERT_TREND_THRESHOLD=75
ALERT_THREAT_THRESHOLD=75
ALERT_OPPORTUNITY_THRESHOLD=70
ALERT_MIN_CONFIDENCE=60

# ---- Backtesting ----
BACKTEST_TRAIN_MONTHS=24
BACKTEST_HORIZON_MONTHS=6
BACKTEST_TOP_K=10

# ---- Exports, reports and dashboard ----
EXPORT_DIRECTORY=./data/exports
GENERATE_REPORTS=true
GENERATE_POWERBI_EXPORTS=true
ENABLE_STREAMLIT_DASHBOARD=true
"""

GITIGNORE = """\
# Secrets and local configuration
.env
.env.*
!.env.example

# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
venv/
build/
dist/

# Tooling caches and reports
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
.coverage.*
htmlcov/
coverage.xml

# Local data, logs and generated output (folders are kept through .gitkeep)
data/*.db
data/*.db-*
data/samples/demo/
data/raw/*
data/staging/*
data/processed/*
data/exports/*
!data/exports/powerbi/
data/exports/powerbi/*
data/fixtures/live_*
logs/*
reports/generated/*
!**/.gitkeep

# Editors and OS
.idea/
.vscode/
.DS_Store
Thumbs.db
"""

LICENSE_TEXT = """\
MIT License

Copyright (c) 2026 CEWS contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

PYPROJECT_TEMPLATE = """\
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "cews"
version = "0.1.0"
description = "Competitor Early Warning System: local, explainable competitive intelligence for life sciences"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
dependencies = __RUNTIME__

[project.optional-dependencies]
ai = __AI__
mcp = __MCP__
dev = __DEV__

[project.scripts]
cews = "cews.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src", "scripts", "tests"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM"]
# Black wraps code; long user-facing strings and templates are allowed.
ignore = ["E501"]

[tool.ruff.lint.per-file-ignores]
# Alembic generates these files in its own style.
"migrations/versions/*.py" = ["UP007", "UP035", "I001"]

[tool.black]
line-length = 100
target-version = ["py311"]

[tool.mypy]
python_version = "3.11"
mypy_path = "src"
disallow_untyped_defs = true
warn_unused_ignores = true
warn_redundant_casts = true
no_implicit_optional = true
ignore_missing_imports = true

[tool.coverage.run]
source = ["src/cews"]
branch = true

[tool.coverage.report]
show_missing = true
skip_empty = true
"""

PYTEST_INI = """\
[pytest]
minversion = 8.0
testpaths = tests
pythonpath = src scripts tests
addopts = -ra --strict-markers --strict-config --import-mode=importlib
markers =
    unit: fast isolated tests
    contract: source adapter contract tests (fixtures only, no network)
    integration: tests that touch the database or several modules
    data_quality: data quality and score-range checks
    e2e: end-to-end smoke tests
    slow: slower tests
filterwarnings =
    ignore:Skipped unsupported reflection of expression-based index:sqlalchemy.exc.SAWarning
"""

ALEMBIC_INI = """\
# Alembic configuration. The database URL is supplied by migrations/env.py from the
# CEWS settings (Phase 2); do not put credentials here.
[alembic]
script_location = migrations
prepend_sys_path = src
sqlalchemy.url =

[loggers]
keys = root,sqlalchemy,alembic

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARNING
handlers = console
qualname =

[logger_sqlalchemy]
level = WARNING
handlers =
qualname = sqlalchemy.engine

[logger_alembic]
level = INFO
handlers =
qualname = alembic

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
datefmt = %H:%M:%S
"""

DOCKER_COMPOSE = """\
# Local PostgreSQL for CEWS. Optional: SQLite is the default backend.
# Start with: docker compose up -d postgres
services:
  postgres:
    image: postgres:16-alpine
    container_name: cews-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-cews}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-cews}
      POSTGRES_DB: ${POSTGRES_DB:-cews}
    ports:
      - "127.0.0.1:${POSTGRES_PORT:-5432}:5432"
    volumes:
      - cews_pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U $${POSTGRES_USER:-cews} -d $${POSTGRES_DB:-cews}"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  cews_pgdata:
"""

# The <TAB> token is replaced by a real tab character (Makefile recipes require tabs).
MAKEFILE_TEMPLATE = r"""# CEWS task runner. Without make, use the plain Python commands listed in README.md.
PYTHON ?= python
.DEFAULT_GOAL := help

.PHONY: help bootstrap check-scaffold check-env setup db-up db-down db-init db-status seed reset-demo sources normalize review competitors features score forecast dashboard fetch analyze ai evaluate export scheduler api dashboard demo test cov lint format typecheck clean

help: ## Show this help
<TAB>@awk -F ':.*## ' '/^[a-zA-Z_-]+:.*## / {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

bootstrap: ## Create missing scaffold files (safe to re-run) and print the tree
<TAB>$(PYTHON) scripts/bootstrap_project.py --tree

check-scaffold: ## Verify the scaffold is complete
<TAB>$(PYTHON) scripts/bootstrap_project.py --validate

check-env: ## Validate configuration, packages and database access
<TAB>$(PYTHON) scripts/validate_environment.py

setup: ## Install the package and development dependencies
<TAB>$(PYTHON) -m pip install -r requirements-dev.txt
<TAB>$(PYTHON) -m pip install -e .

db-up: ## Start local PostgreSQL (optional; SQLite is the default)
<TAB>docker compose up -d postgres

db-down: ## Stop local PostgreSQL
<TAB>docker compose down

db-init: ## Create database tables
<TAB>$(PYTHON) scripts/initialize_database.py

db-status: ## Show schema revision and record counts
<TAB>$(PYTHON) -m cews.cli db-status

seed: ## Load synthetic demo data
<TAB>$(PYTHON) scripts/seed_demo_data.py

reset-demo: ## Delete all synthetic demo data
<TAB>$(PYTHON) scripts/reset_demo.py --yes

sources: ## List data sources, adapters, checkpoints and circuit state
<TAB>$(PYTHON) -m cews.cli sources

fetch: ## Run one manual fetch across enabled sources
<TAB>$(PYTHON) scripts/fetch_all.py

normalize: ## Resolve organizations, assign topics, mark duplicates
<TAB>$(PYTHON) -m cews.cli normalize

review: ## Show entity decisions awaiting a human
<TAB>$(PYTHON) -m cews.cli review

competitors: ## Rank and show the monitored competitors
<TAB>$(PYTHON) -m cews.cli competitors

features: ## Count activity and compute trend features
<TAB>$(PYTHON) -m cews.cli features

score: ## Score the stored features
<TAB>$(PYTHON) -m cews.cli score

forecast: ## Forecast activity and flag unusual months
<TAB>$(PYTHON) -m cews.cli forecast

analyze: ## Run features, scores, forecasts and insights
<TAB>$(PYTHON) scripts/run_analysis.py

ai: ## Run analysis including the optional AI layers
<TAB>$(PYTHON) scripts/run_analysis.py --with-ai

evaluate: ## Run data-quality checks, backtests and AI ablation
<TAB>$(PYTHON) scripts/run_evaluation.py

export: ## Write Power BI CSV exports
<TAB>$(PYTHON) scripts/export_powerbi.py

scheduler: ## Start the background scheduler
<TAB>$(PYTHON) scripts/run_scheduler.py

api: ## Start the FastAPI service
<TAB>$(PYTHON) -m uvicorn cews.api.app:app --reload

dashboard: ## Start the Streamlit dashboard
<TAB>$(PYTHON) -m streamlit run dashboards/streamlit/app.py

demo: db-init seed analyze ai evaluate export ## Run the offline demo pipeline on synthetic data

test: ## Run the test suite
<TAB>$(PYTHON) -m pytest

cov: ## Run tests with a coverage report
<TAB>$(PYTHON) -m pytest --cov=src/cews --cov-report=term-missing --cov-report=html

lint: ## Ruff and Black checks
<TAB>$(PYTHON) -m ruff check .
<TAB>$(PYTHON) -m black --check .

format: ## Auto-format the code
<TAB>$(PYTHON) -m ruff check --fix .
<TAB>$(PYTHON) -m black .

typecheck: ## Run mypy
<TAB>$(PYTHON) -m mypy src

clean: ## Remove caches
<TAB>$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in list(pathlib.Path('.').rglob('__pycache__')) + [pathlib.Path(x) for x in ('.pytest_cache', '.mypy_cache', '.ruff_cache', 'htmlcov')]]"
"""

LOGGING_YAML = """\
# Python logging.config.dictConfig format. A filter that redacts API keys and secrets
# from log records is added with the logging setup in Phase 2.
version: 1
disable_existing_loggers: false
formatters:
  standard:
    format: "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt: "%Y-%m-%dT%H:%M:%S%z"
handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: standard
    stream: ext://sys.stdout
  file:
    class: logging.handlers.RotatingFileHandler
    level: INFO
    formatter: standard
    filename: logs/cews.log
    maxBytes: 5242880
    backupCount: 5
    encoding: utf-8
loggers:
  cews:
    level: INFO
    handlers: [console, file]
    propagate: false
root:
  level: WARNING
  handlers: [console]
"""

SCORING_WEIGHTS_YAML = """\
# Scoring configuration. Every weight set must sum to 1 (checked by tests and, from
# Phase 7, validate_weights() at startup). If a component is unavailable its weight is
# renormalized across the available components; it is never replaced by zero.
scoring_version: "1.0.0"

normalization:
  method: percentile        # percentile | winsorized_minmax | robust_zscore | minmax
  winsor_lower: 0.05
  winsor_upper: 0.95
  neutral_score: 50         # returned for constant or single-value cohorts

features:
  alpha: 1.0                # smoothing constant for log growth
  momentum_recent_months: 3
  momentum_previous_months: 3
  velocity_window_months: 12
  velocity_min_observations: 6      # assumption: the spec does not fix this value
  sample_saturation_k: 25           # sample confidence = 1 - exp(-N / k)
  freshness_half_life_days: 30      # assumption: the spec does not fix this value

weight_sets:
  # Change from the original spec: patents and funding are excluded from this composite
  # because they already enter the Trend Score as separate components (double counting).
  # Restore the original 0.30/0.30/0.20/0.10/0.10 split here if you prefer it.
  composite_activity: {publication: 0.40, clinical_trial: 0.40, announcement: 0.20}
  trend_score: {velocity: 0.30, momentum: 0.25, patent_growth: 0.20, funding_growth: 0.15, consistency: 0.10}
  innovation_score: {patent: 0.40, clinical_trial: 0.30, publication: 0.20, funding: 0.10}
  threat_score: {trial_growth: 0.50, patent_growth: 0.30, publication_growth: 0.20}
  opportunity_score: {trend: 0.60, low_competition: 0.25, source_agreement: 0.15}
  confidence_score: {sample: 0.35, source_agreement: 0.25, completeness: 0.20, freshness: 0.10, model_stability: 0.10}
  competitor_discovery: {trial: 0.30, patent: 0.25, publication: 0.20, funding: 0.15, announcement: 0.10}

competitor_discovery:
  window_months: 12

categories:
  trend:
    - {label: "Low activity or declining", min: 0, max: 39}
    - {label: "Watchlist", min: 40, max: 59}
    - {label: "Emerging", min: 60, max: 79}
    - {label: "High priority", min: 80, max: 100}
  opportunity:
    - {label: "Low current opportunity", min: 0, max: 39}
    - {label: "Monitor", min: 40, max: 59}
    - {label: "Potential opportunity", min: 60, max: 79}
    - {label: "High-priority opportunity for expert review", min: 80, max: 100}
  innovation:
    - {label: "Limited recent innovation", min: 0, max: 39}
    - {label: "Moderate innovation activity", min: 40, max: 59}
    - {label: "Strong innovation activity", min: 60, max: 79}
    - {label: "Innovation leader", min: 80, max: 100}
  # Threat wording is deliberate: this score measures activity, not intent or capability, and
  # must never read as an accusation about a named company.
  threat:
    - {label: "Routine activity", min: 0, max: 39}
    - {label: "Worth watching", min: 40, max: 59}
    - {label: "Elevated competitive activity", min: 60, max: 74}
    - {label: "High monitoring priority", min: 75, max: 89}
    - {label: "Potential strategic threat requiring review", min: 90, max: 100}


emerging_trend_rules:
  min_trend_score: 60               # assumption: 'configured threshold' is not numeric in the spec
  min_confidence: 60                # assumption; mirrors ALERT_MIN_CONFIDENCE
  min_source_types: 2
  exclude_single_spike: true

threat_modifiers:
  enabled: true
  total_cap_points: 15              # assumption: the spec requires caps but gives no values
  items:
    new_therapeutic_area_entry: {max_points: 5}
    phase_progression: {max_points: 5}
    large_enrollment_increase: {max_points: 5}
    monitored_topic_patent_activity: {max_points: 5}
    multiple_supporting_sources: {max_points: 5}
"""

TOPIC_TAXONOMY_YAML = """\
# Starter taxonomy. Deterministic keyword/phrase matching is the default method.
# Edit freely: no code change is needed. Topics discovered by the optional AI layer are
# stored in the database as candidates and never written back to this file.
taxonomy_version: 1

therapeutic_areas:
  - {id: oncology, name: Oncology, parent: null, synonyms: [cancer, tumor, tumour, neoplasm, carcinoma, leukemia, lymphoma]}
  - {id: cancer_vaccines, name: Cancer Vaccines, parent: oncology, synonyms: [cancer vaccine, tumor vaccine, neoantigen vaccine]}
  - {id: car_t, name: CAR-T, parent: oncology, synonyms: [CAR T, CAR-T cell, chimeric antigen receptor]}
  - {id: neurology, name: Neurology, parent: null, synonyms: [neurological, neurodegenerative, alzheimer, parkinson, multiple sclerosis]}
  - {id: immunology, name: Immunology, parent: null, synonyms: [autoimmune, immune-mediated, inflammation]}
  - {id: rare_diseases, name: Rare Diseases, parent: null, synonyms: [orphan disease, orphan drug, rare disease]}
  - {id: gene_therapy, name: Gene Therapy, parent: null, synonyms: [gene transfer, gene replacement, viral vector]}
  - {id: rna_therapeutics, name: RNA Therapeutics, parent: null, synonyms: [RNA-based therapy, oligonucleotide therapy]}

topics:
  - {id: mrna_therapeutics, name: mRNA therapeutics, type: modality, area: rna_therapeutics, synonyms: [mRNA vaccine, messenger RNA, mRNA-based]}
  - {id: sirna_rnai, name: siRNA and RNA interference, type: modality, area: rna_therapeutics, synonyms: [siRNA, RNAi, RNA interference, antisense oligonucleotide]}
  - {id: crispr_gene_editing, name: CRISPR gene editing, type: technology, area: gene_therapy, synonyms: [CRISPR, Cas9, base editing, prime editing, gene editing]}
  - {id: aav_vectors, name: AAV gene delivery, type: technology, area: gene_therapy, synonyms: [AAV, adeno-associated virus]}
  - {id: immune_checkpoint, name: Immune checkpoint inhibitors, type: modality, area: oncology, synonyms: [PD-1, PD-L1, checkpoint inhibitor, immune checkpoint]}
  - {id: bispecific_antibodies, name: Bispecific antibodies, type: modality, area: oncology, synonyms: [bispecific antibody, T-cell engager, T cell engager]}
  - {id: neurodegeneration, name: Neurodegeneration, type: disease_area, area: neurology, synonyms: [amyloid beta, tau protein, neurodegenerative disease]}
  - {id: ai_drug_discovery, name: AI-assisted drug discovery, type: technology, area: null, synonyms: [AI drug discovery, machine learning drug discovery, generative chemistry]}

matching:
  default_method: keyword           # keyword | tfidf | embedding (tfidf/embedding are optional)
  min_confidence: 0.5
"""

SOURCE_REGISTRY_YAML = """\
# Source registry. The four MVP sources were checked against their documentation on
# 2026-09-22 (see docs/data_sources.md). Rate limits are deliberately conservative.
# Company feeds are never hard-coded: add official RSS/IR feed URLs under generic_rss.feeds.
#
# Per-source keys (all optional except id, source_type, env_flag):
#   requests_per_second, burst   rate limit for this source
#   page_size                    records requested per page
#   max_pages_per_run            safety cap; a capped run resumes from its checkpoint next time
#   window_slice_days            date windows are collected in slices of this many days
#                                (null = one request window, e.g. RSS feeds)
#   timeout_seconds, max_retries override SOURCE_REQUEST_TIMEOUT_SECONDS / SOURCE_MAX_RETRIES
#   refresh.mode                 incremental (every cycle) or full (only when the window is due)
#   options                      adapter-specific settings, for example:
#     query_terms: [..]          search these terms instead of THERAPEUTIC_AREAS
#     query: "..."               a complete source-specific query (replaces the terms)
#     store_abstracts: false     PubMed / Europe PMC: do not store abstracts
#     study_type: ALL            ClinicalTrials.gov: default INTERVENTIONAL
#     sources: [PPR, MED]        Europe PMC source codes (default: PPR only while PubMed is on)
#     feed_organizations: {url: name}   generic_rss: company behind each feed
registry_version: 2
defaults:
  requests_per_second: 1.0
  burst: 1
  page_size: 100
  max_pages_per_run: 50
  window_slice_days: 30

sources:
  - id: clinical_trials_gov
    source_type: clinical_trial
    env_flag: ENABLE_CLINICAL_TRIALS_GOV
    base_url: https://clinicaltrials.gov/api/v2
    auth: none
    mvp: true
    requests_per_second: 0.5      # no published limit; community reports ~50/minute
    page_size: 500                # API maximum is 1000
    options:
      # Their firewall fingerprints TLS and answers 403 to httpx; the standard library is
      # accepted. See docs/data_sources.md.
      http_transport: stdlib
    refresh: {mode: incremental}
  - id: pubmed
    source_type: publication
    env_flag: ENABLE_PUBMED
    base_url: https://eutils.ncbi.nlm.nih.gov/entrez/eutils
    auth: optional_api_key
    mvp: true
    requests_per_second: 2.5      # NCBI: 3/s without a key, 10/s with NCBI_API_KEY
    page_size: 200                # PMIDs per esearch/efetch pair
    max_pages_per_run: 100
    window_slice_days: 7          # one search can only reach 10,000 records
    refresh: {mode: incremental}
  - id: europe_pmc
    source_type: publication
    env_flag: ENABLE_EUROPE_PMC
    base_url: https://www.ebi.ac.uk/europepmc/webservices/rest
    auth: none
    mvp: true
    requests_per_second: 2.0
    page_size: 500                # must stay constant while paging; maximum 1000
    refresh: {mode: incremental}
  - id: openalex
    source_type: publication
    env_flag: ENABLE_OPENALEX
    base_url: https://api.openalex.org
    auth: api_key_required        # required since February 2026; see docs/data_sources.md
    mvp: false
    refresh: {mode: incremental}
  - id: nih_reporter
    source_type: funding
    env_flag: ENABLE_NIH_REPORTER
    base_url: https://api.reporter.nih.gov/v2
    auth: none
    mvp: false
    refresh: {mode: full, full_refresh_window_days: 7}
  - id: generic_rss
    source_type: announcement
    env_flag: ENABLE_GENERIC_RSS
    auth: none
    mvp: true
    requests_per_second: 0.5
    window_slice_days: null
    feeds: []                     # add official company / investor-relations feed URLs here
    # options:
    #   feed_organizations:
    #     https://www.example.com/news/rss.xml: Example Pharma
    refresh: {mode: incremental}
  - id: patents_uspto_bulk
    source_type: patent
    env_flag: ENABLE_PATENTS
    base_url: https://data.uspto.gov
    auth: api_key_may_be_required # verify on the USPTO Open Data Portal
    mvp: false
    refresh: {mode: full, full_refresh_window_days: 30}
"""

BENCHMARK_TOPICS_YAML = """\
# Evaluation anchors only: historically established areas used to sanity-check backtests.
# They are NEVER hard-coded as high-scoring trends. Add expected evaluation windows with a
# subject-matter expert before using them in a backtest.
benchmark_topics:
  - {name: mRNA therapeutics, taxonomy_ref: mrna_therapeutics, evaluation_window: null}
  - {name: CAR-T, taxonomy_ref: car_t, evaluation_window: null}
  - {name: CRISPR gene editing, taxonomy_ref: crispr_gene_editing, evaluation_window: null}
  - {name: Cancer vaccines, taxonomy_ref: cancer_vaccines, evaluation_window: null}
  - {name: AI-assisted drug discovery, taxonomy_ref: ai_drug_discovery, evaluation_window: null}
"""

POWERBI_THEME_JSON = """\
{
  "name": "CEWS colour-blind friendly",
  "dataColors": ["#0072B2", "#E69F00", "#009E73", "#56B4E9", "#CC79A7", "#F0E442", "#D55E00", "#000000"],
  "background": "#FFFFFF",
  "foreground": "#1F2937",
  "tableAccent": "#0072B2"
}
"""

SQL_PLACEHOLDERS: dict[str, str] = {
    "create_views": "Core analytical views over the CEWS tables.",
    "dashboard_views": "Star-schema views consumed by Power BI and the Streamlit dashboard.",
    "evaluation_views": "Views over evaluation runs, backtests and alert reviews.",
    "data_quality_checks": "SQL data-quality checks (required fields, uniqueness, integrity).",
}

MIGRATIONS_README = """\
# Migrations

- `versions/` holds Alembic revisions (baseline migration added in Phase 2).
- `sql/` holds plain SQL for database views and data-quality checks.
- Alembic environment files (`env.py`, `script.py.mako`) are added in Phase 2. Until then,
  `alembic` commands are not expected to work.
"""

DATA_SOURCES_DOC = """\
# Data sources

Status: skeleton. Sections are filled in as adapters are built (Phases 3-4).

## Access risks checked on 2026-09-20

**Patents (PatentsView).** PatentsView migrated to the USPTO Open Data Portal (ODP,
https://data.uspto.gov) starting 2026-03-20. USPTO's transition guide
(https://data.uspto.gov/support/transition-guide/patentsview) says temporary interruptions
are expected for the PatentSearch API and that API functions will be reintroduced in updated
forms; bulk downloads are available on ODP. Some third-party tools report the legacy API as
shut down. Plan: do not build a live patents adapter on the old PatentsView endpoint. Start
with a fixture adapter, then target ODP bulk downloads, and re-check API status first.

**Publications (OpenAlex).** OpenAlex's deprecations page
(https://developers.openalex.org/guides/deprecations) states that the polite pool was
replaced by API keys in February 2026: `mailto` is ignored and all users need a free API key.
A secondary source also reports usage-based pricing with a small daily free allowance; verify
this against OpenAlex's official pricing page before enabling the adapter, because the project
must stay free. Plan: OpenAlex is optional and off by default; Europe PMC is the third
no-key publication source in the MVP.

## Not yet verified

ClinicalTrials.gov, PubMed, Europe PMC, NIH RePORTER and RSS endpoints, terms, and rate limits
are taken from memory of public documentation and must be re-checked against the official
documentation when each adapter is built.
"""

README_SECTIONS: tuple[str, ...] = (
    "Project overview",
    "Business problem",
    "Business value",
    "Features",
    "Architecture",
    "Data sources",
    "Technology stack",
    "Installation",
    "Configuration",
    "Database setup",
    "Running demo mode",
    "Running live data mode",
    "Running the scheduler",
    "Running tests",
    "Opening the dashboard",
    "Power BI connection",
    "Scoring formulas",
    "Validation methodology",
    "Known limitations",
    "Ethical and legal considerations",
    "Troubleshooting",
    "Future production enhancements",
)

COMMAND_TABLE: tuple[tuple[str, str], ...] = (
    ("make setup", "python -m pip install -r requirements-dev.txt && python -m pip install -e ."),
    ("make db-up", "docker compose up -d postgres"),
    ("make check-env", "python scripts/validate_environment.py"),
    ("make db-init", "python scripts/initialize_database.py"),
    ("make db-status", "python -m cews.cli db-status"),
    ("make reset-demo", "python scripts/reset_demo.py --yes"),
    ("make seed", "python scripts/seed_demo_data.py"),
    ("make sources", "python -m cews.cli sources"),
    ("make fetch", "python scripts/fetch_all.py"),
    ("make normalize", "python -m cews.cli normalize"),
    ("make competitors", "python -m cews.cli competitors"),
    ("make features", "python -m cews.cli features"),
    ("make score", "python -m cews.cli score"),
    ("make forecast", "python -m cews.cli forecast"),
    ("make review", "python -m cews.cli review"),
    ("make analyze", "python scripts/run_analysis.py"),
    ("make ai", "python scripts/run_analysis.py --with-ai"),
    ("make evaluate", "python scripts/run_evaluation.py"),
    ("make export", "python scripts/export_powerbi.py"),
    ("make scheduler", "python scripts/run_scheduler.py"),
    ("make api", "python -m uvicorn cews.api.app:app --reload"),
    ("make dashboard", "python -m streamlit run dashboards/streamlit/app.py"),
    ("make test", "python -m pytest"),
)


def readme_skeleton() -> str:
    """Render the README skeleton (all 22 required sections plus a Phase 1 quick start)."""
    lines = [
        "# CEWS: Competitor Early Warning System",
        "",
        "A local, explainable competitive-intelligence proof of concept for life-sciences",
        "organizations. It collects public data, computes evidence-linked scores, and shows the",
        "results on a dashboard. It is not a chatbot and uses no paid services.",
        "",
        "> Status: Dashboard (all five scores, forecasts, anomalies, and a Streamlit dashboard over them). Insights and the AI layers are next.",
        "",
        "## Quick start (Phase 5)",
        "",
        "```",
        "python -m pip install -r requirements-dev.txt",
        "python -m pip install -e .",
        "copy .env.example .env          # Windows; use cp on macOS/Linux",
        "python scripts/validate_environment.py",
        "python scripts/initialize_database.py",
        "python scripts/seed_demo_data.py",
        "python scripts/initialize_database.py   # safe to repeat; also upgrades the schema",
        "python -m cews.cli sources              # data sources and their state",
        "python -m cews.cli normalize            # resolve organizations and topics",
        "python -m cews.cli review               # matches a person should confirm",
        "python -m cews.cli competitors          # who CEWS is monitoring, and why",
        "python -m cews.cli features             # monthly activity, growth, momentum, velocity",
        "python -m cews.cli score                # score the features, with confidence",
        "python -m cews.cli forecast             # forecast activity and flag unusual months",
        "python -m cews.cli review               # decisions it left to a human",
        "python -m pytest",
        "```",
        "",
        "## Collecting live data",
        "",
        "Live data and demo data are never mixed, so give live collection its own database:",
        "",
        "```",
        "copy .env .env.live                     # Windows; use cp on macOS/Linux",
        "# in .env.live set SQLITE_PATH=./data/cews_live.db and DEFAULT_LOOKBACK_DAYS=30",
        "python scripts/initialize_database.py --env-file .env.live",
        "python -m cews.cli sources --health --env-file .env.live",
        "python scripts/fetch_all.py --env-file .env.live --dry-run",
        "python scripts/fetch_all.py --env-file .env.live",
        "```",
        "",
        "Company announcements need feed URLs: add official RSS or investor-relations feeds",
        "under `generic_rss.feeds` in `config/source_registry.yaml`. See `docs/data_sources.md`.",
        "",
        "The demo data is SYNTHETIC (invented organizations, artificial patterns).",
        "Scaffold tools: `python scripts/bootstrap_project.py --tree` / `--validate`.",
        "",
        "## Command reference (make and plain Python)",
        "",
        "| make | Python |",
        "|---|---|",
    ]
    lines += [f"| `{make}` | `{py}` |" for make, py in COMMAND_TABLE]
    lines += [""]
    for number, title in enumerate(README_SECTIONS, start=1):
        lines += [f"## {number}. {title}", "", "_Not written yet._", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Placeholder generators
# --------------------------------------------------------------------------------------
def _dotted(rel_path: str) -> str:
    """Return a dotted module name for a repo-relative Python path."""
    path = PurePosixPath(rel_path)
    parts = list(path.with_suffix("").parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def python_placeholder(rel_path: str) -> str:
    """Return a placeholder Python module containing only a docstring."""
    phase = phase_for(rel_path)
    posix = PurePosixPath(rel_path)
    dotted = _dotted(rel_path)
    if rel_path == "src/cews/__init__.py":
        return (
            '"""CEWS: Competitor Early Warning System.\n\n'
            "A local, explainable competitive-intelligence proof of concept for life-sciences\n"
            "organizations. Package placeholder created by scripts/bootstrap_project.py.\n"
            '"""\n\n__version__ = "0.1.0"\n'
        )
    if posix.name == "__init__.py":
        return (
            f'"""CEWS package: {dotted}.\n\n'
            "Placeholder created by scripts/bootstrap_project.py. "
            f'Planned for Phase {phase}.\n"""\n'
        )
    human = posix.stem.replace("_", " ")
    return (
        f'"""CEWS module {dotted}: {human}.\n\n'
        "Placeholder created by scripts/bootstrap_project.py; not implemented yet.\n"
        f'Planned for Phase {phase}.\n"""\n'
    )


def placeholder_test_module(rel_path: str) -> str:
    """Return a placeholder test module whose single test is reported as skipped."""
    phase = phase_for(rel_path)
    stem = PurePosixPath(rel_path).stem
    return (
        f'"""Placeholder tests for {stem} (planned for Phase {phase}).\n\n'
        'Created by scripts/bootstrap_project.py. Replace with real tests.\n"""\n\n'
        "import pytest\n\n"
        f'pytestmark = pytest.mark.skip(reason="Placeholder scaffold: implemented in Phase {phase}.")\n\n\n'
        "def test_placeholder() -> None:\n"
        '    """Placeholder so the module is collected and reported as skipped."""\n'
    )


def markdown_placeholder(rel_path: str, title: str) -> str:
    """Return a placeholder Markdown document."""
    phase = phase_for(rel_path)
    return f"# {title}\n\n_Placeholder created by scripts/bootstrap_project.py. Planned for Phase {phase}._\n"


def sql_placeholder(name: str, description: str) -> str:
    """Return a placeholder SQL file containing only a comment header."""
    return f"-- {name}.sql\n-- {description}\n-- Placeholder created by scripts/bootstrap_project.py; planned for Phase 2 onward.\n"


# --------------------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------------------
def build_manifest() -> Manifest:
    """Build the full list of directories and files this script creates."""
    files: dict[str, FileSpec] = {}

    def add(path: str, content: str, critical: bool = False) -> None:
        if path in files:
            raise ValueError(f"duplicate manifest path: {path}")
        files[path] = FileSpec(path, content, critical)

    # Root files (all critical)
    pyproject = (
        PYPROJECT_TEMPLATE.replace("__RUNTIME__", _toml_list(RUNTIME_DEPS))
        .replace("__AI__", _toml_list(AI_DEPS))
        .replace("__MCP__", _toml_list(MCP_DEPS))
        .replace("__DEV__", _toml_list(DEV_DEPS))
    )
    requirements = "# Runtime dependencies (bounded). Keep in sync with pyproject.toml.\n"
    requirements += "\n".join(RUNTIME_DEPS) + "\n"
    requirements_dev = (
        "# Development dependencies. Installs runtime dependencies too.\n-r requirements.txt\n"
        "-r requirements-mcp.txt\n"
    )
    requirements_dev += "\n".join(DEV_DEPS) + "\n"
    requirements_ai = (
        "# Optional local AI layers (CPU is enough). Installs a large PyTorch dependency.\n"
    )
    requirements_ai += "\n".join(AI_DEPS) + "\n"
    requirements_mcp = (
        "# Optional: the read-only MCP server (cews mcp-serve) and the agents built on it\n"
        "# (cews agent ..., the dashboard chat widget). See docs/agents.md.\n"
    )
    requirements_mcp += "\n".join(MCP_DEPS) + "\n"
    makefile = MAKEFILE_TEMPLATE.replace("<TAB>", "\t")

    add(".env.example", ENV_EXAMPLE, critical=True)
    add(".gitignore", GITIGNORE, critical=True)
    add("README.md", readme_skeleton(), critical=True)
    add("LICENSE", LICENSE_TEXT, critical=True)
    add("pyproject.toml", pyproject, critical=True)
    add("requirements.txt", requirements, critical=True)
    add("requirements-dev.txt", requirements_dev, critical=True)
    add("requirements-ai.txt", requirements_ai, critical=True)
    add("requirements-mcp.txt", requirements_mcp, critical=True)
    add("Makefile", makefile, critical=True)
    add("docker-compose.yml", DOCKER_COMPOSE, critical=True)
    add("pytest.ini", PYTEST_INI, critical=True)
    add("alembic.ini", ALEMBIC_INI, critical=True)

    # Configuration (critical)
    add("config/logging.yaml", LOGGING_YAML, critical=True)
    add("config/scoring_weights.yaml", SCORING_WEIGHTS_YAML, critical=True)
    add("config/topic_taxonomy.yaml", TOPIC_TAXONOMY_YAML, critical=True)
    add("config/source_registry.yaml", SOURCE_REGISTRY_YAML, critical=True)
    add("config/benchmark_topics.yaml", BENCHMARK_TOPICS_YAML, critical=True)

    # Application packages
    for package, modules in SRC_MODULES.items():
        init_path = f"{package}/__init__.py"
        add(init_path, python_placeholder(init_path), critical=(package == "src/cews"))
        for module in modules:
            module_path = f"{package}/{module}.py"
            add(module_path, python_placeholder(module_path))

    # src/cews/ai/llm is a small subpackage (a provider-agnostic chat client), not a flat module,
    # so it is not part of SRC_MODULES above.
    add("src/cews/ai/llm/__init__.py", python_placeholder("src/cews/ai/llm/__init__.py"))
    add("src/cews/ai/llm/client.py", python_placeholder("src/cews/ai/llm/client.py"))

    # Scripts (this script itself is not part of the manifest)
    for name in SCRIPT_NAMES:
        script_path = f"scripts/{name}.py"
        add(script_path, python_placeholder(script_path))

    # Migrations
    add("migrations/README.md", MIGRATIONS_README)
    add("migrations/env.py", python_placeholder("migrations/env.py"))
    for name, description in SQL_PLACEHOLDERS.items():
        add(f"migrations/sql/{name}.sql", sql_placeholder(name, description))

    # Dashboards
    add("dashboards/streamlit/app.py", python_placeholder("dashboards/streamlit/app.py"))
    add(
        "dashboards/streamlit/components/__init__.py",
        python_placeholder("dashboards/streamlit/components/__init__.py"),
    )
    for name, title in POWERBI_DOCS.items():
        add(
            f"dashboards/powerbi/{name}.md",
            markdown_placeholder(f"dashboards/powerbi/{name}.md", title),
        )
    add("dashboards/powerbi/powerbi_theme.json", POWERBI_THEME_JSON)

    # Docs
    for name, title in DOC_FILES.items():
        content = (
            DATA_SOURCES_DOC
            if name == "data_sources"
            else markdown_placeholder(f"docs/{name}.md", title)
        )
        add(f"docs/{name}.md", content)

    # Tests (test_bootstrap_project.py is a real test and is not part of the manifest)
    add("tests/conftest.py", '"""Shared pytest fixtures for CEWS (added in Phase 2)."""\n')
    for directory, names in TEST_FILES.items():
        for name in names:
            test_path = f"{directory}/{name}.py"
            add(test_path, placeholder_test_module(test_path))

    # .gitkeep markers keep otherwise-empty directories in git.
    for directory in GITKEEP_DIRS:
        add(f"{directory}/.gitkeep", "")

    directories: set[str] = set(GITKEEP_DIRS)
    for path in files:
        for parent in PurePosixPath(path).parents:
            if parent.as_posix() != ".":
                directories.add(parent.as_posix())
    dir_specs = tuple(DirSpec(path, critical=path in CRITICAL_DIRS) for path in sorted(directories))
    return Manifest(directories=dir_specs, files=tuple(files.values()))


# --------------------------------------------------------------------------------------
# Filesystem operations
# --------------------------------------------------------------------------------------
def _safe_join(root: Path, rel_path: str) -> Path:
    """Join ``rel_path`` to ``root``, refusing absolute paths and traversal outside root."""
    candidate = PurePosixPath(rel_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"unsafe relative path: {rel_path!r}")
    target = (root / Path(*candidate.parts)).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes the project root: {rel_path!r}")
    return target


def ensure_directory(root: Path, spec: DirSpec, dry_run: bool = False) -> Result:
    """Create one directory if it is missing."""
    try:
        target = _safe_join(root, spec.path)
    except ValueError as exc:
        return Result(spec.path, Kind.DIR, Status.FAILED, spec.critical, str(exc))
    if target.is_dir():
        return Result(spec.path, Kind.DIR, Status.SKIPPED, spec.critical, "already exists")
    if target.exists():
        return Result(
            spec.path, Kind.DIR, Status.FAILED, spec.critical, "exists but is not a directory"
        )
    if dry_run:
        return Result(spec.path, Kind.DIR, Status.CREATED, spec.critical, "dry-run")
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return Result(
            spec.path, Kind.DIR, Status.FAILED, spec.critical, f"{type(exc).__name__}: {exc}"
        )
    return Result(spec.path, Kind.DIR, Status.CREATED, spec.critical)


def write_file(root: Path, spec: FileSpec, force: bool = False, dry_run: bool = False) -> Result:
    """Write one file, never overwriting a non-empty file unless ``force`` is set."""
    try:
        target = _safe_join(root, spec.path)
    except ValueError as exc:
        return Result(spec.path, Kind.FILE, Status.FAILED, spec.critical, str(exc))
    if target.is_dir():
        return Result(
            spec.path, Kind.FILE, Status.FAILED, spec.critical, "a directory exists at this path"
        )

    status = Status.CREATED
    detail = ""
    if target.exists():
        try:
            size = target.stat().st_size
        except OSError as exc:
            return Result(
                spec.path, Kind.FILE, Status.FAILED, spec.critical, f"{type(exc).__name__}: {exc}"
            )
        if size > 0 and not force:
            return Result(
                spec.path,
                Kind.FILE,
                Status.SKIPPED,
                spec.critical,
                "non-empty file exists (use --force to overwrite)",
            )
        if size == 0 and spec.content == "":
            return Result(spec.path, Kind.FILE, Status.SKIPPED, spec.critical, "already exists")
        if size > 0:
            status, detail = Status.OVERWRITTEN, "overwritten (--force)"
        else:
            detail = "populated existing empty file"
    if dry_run:
        return Result(spec.path, Kind.FILE, status, spec.critical, "dry-run")

    tmp = target.with_name(target.name + ".tmp")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(spec.content)
        os.replace(tmp, target)
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            LOGGER.debug("could not remove temporary file %s", tmp)
        return Result(
            spec.path, Kind.FILE, Status.FAILED, spec.critical, f"{type(exc).__name__}: {exc}"
        )
    return Result(spec.path, Kind.FILE, status, spec.critical, detail)


def bootstrap(
    root: Path, force: bool = False, dry_run: bool = False, manifest: Manifest | None = None
) -> Report:
    """Create the CEWS scaffold under ``root`` and return a report of every action."""
    manifest = manifest or build_manifest()
    report = Report(root=root, dry_run=dry_run, force=force)
    try:
        if not dry_run:
            root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        report.results.append(
            Result(
                ".",
                Kind.DIR,
                Status.FAILED,
                True,
                f"cannot create root: {type(exc).__name__}: {exc}",
            )
        )
        return report
    root = root.resolve()
    report.root = root
    for dir_spec in manifest.directories:
        report.results.append(ensure_directory(root, dir_spec, dry_run=dry_run))
    for file_spec in manifest.files:
        report.results.append(write_file(root, file_spec, force=force, dry_run=dry_run))
    return report


def validate_scaffold(root: Path, manifest: Manifest | None = None) -> list[Result]:
    """Return one FAILED result for every expected item that is missing or empty."""
    manifest = manifest or build_manifest()
    problems: list[Result] = []
    for dir_spec in manifest.directories:
        if not (root / dir_spec.path).is_dir():
            problems.append(
                Result(
                    dir_spec.path, Kind.DIR, Status.FAILED, dir_spec.critical, "missing directory"
                )
            )
    for file_spec in manifest.files:
        target = root / file_spec.path
        if not target.is_file():
            problems.append(
                Result(file_spec.path, Kind.FILE, Status.FAILED, file_spec.critical, "missing file")
            )
        elif file_spec.content and target.stat().st_size == 0:
            problems.append(
                Result(
                    file_spec.path, Kind.FILE, Status.FAILED, file_spec.critical, "file is empty"
                )
            )
    return problems


# --------------------------------------------------------------------------------------
# Presentation
# --------------------------------------------------------------------------------------
_IGNORED_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "htmlcov",
        ".coverage",
    }
)


def _ignored(entry: Path, show_markers: bool) -> bool:
    name = entry.name
    if name in _IGNORED_NAMES or name.endswith((".pyc", ".tmp", ".egg-info")):
        return True
    return name == ".gitkeep" and not show_markers


def render_tree(
    root: Path, ascii_only: bool = False, show_markers: bool = False, max_depth: int | None = None
) -> str:
    """Render a directory tree as text (directories first, then files, alphabetical)."""
    if ascii_only:
        branch, last, pipe, space = "+-- ", "`-- ", "|   ", "    "
    else:
        branch, last, pipe, space = (
            "\u251c\u2500\u2500 ",
            "\u2514\u2500\u2500 ",
            "\u2502   ",
            "    ",
        )
    lines = [f"{root.name}/"]

    def walk(directory: Path, prefix: str, depth: int) -> None:
        if max_depth is not None and depth > max_depth:
            return
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError as exc:
            lines.append(f"{prefix}[unreadable: {exc}]")
            return
        entries = [e for e in entries if not _ignored(e, show_markers)]
        for index, entry in enumerate(entries):
            is_last = index == len(entries) - 1
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{prefix}{last if is_last else branch}{entry.name}{suffix}")
            if entry.is_dir():
                walk(entry, prefix + (space if is_last else pipe), depth + 1)

    walk(root, "", 1)
    return "\n".join(lines)


def _stdout_supports_unicode() -> bool:
    encoding = (getattr(sys.stdout, "encoding", None) or "").lower()
    return "utf" in encoding


def print_summary(report: Report, verbose: bool = False) -> None:
    """Print a summary of created, skipped and failed items."""
    verb = "would create" if report.dry_run else "created"
    print("CEWS bootstrap summary")
    print(f"  root : {report.root}")
    print(
        f"  mode : {'dry-run' if report.dry_run else 'apply'}   force: {'yes' if report.force else 'no'}"
    )
    for kind, label in ((Kind.DIR, "directories"), (Kind.FILE, "files      ")):
        print(
            f"  {label}: {verb} {report.count(Status.CREATED, kind)}, "
            f"overwritten {report.count(Status.OVERWRITTEN, kind)}, "
            f"skipped {report.count(Status.SKIPPED, kind)}, "
            f"failed {report.count(Status.FAILED, kind)}"
        )
    failed = report.failed()
    if failed:
        print("\nFailed:")
        for item in failed:
            tag = "CRITICAL" if item.critical else "warning"
            print(f"  [{tag}] {item.path}: {item.detail}")
    for status, title in (
        (Status.CREATED, verb),
        (Status.OVERWRITTEN, "overwritten"),
        (Status.SKIPPED, "skipped"),
    ):
        items = report.select(status, Kind.FILE)
        if not items:
            continue
        shown = items if verbose else items[:SUMMARY_SAMPLE_SIZE]
        print(
            f"\nFiles {title} ({len(items)}){'' if verbose or len(items) <= len(shown) else ', first ' + str(len(shown)) + ' shown; use --verbose for all'}:"
        )
        for item in shown:
            print(f"  {item.path}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def resolve_root(explicit: str | None, script_path: Path, cwd: Path) -> Path:
    """Decide the project root: --root, else the parent of ./scripts, else ./cews."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    script_dir = script_path.resolve().parent
    if script_dir.name == "scripts":
        return script_dir.parent
    return (cwd / PROJECT_NAME).resolve()


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="bootstrap_project.py",
        description="Create (or verify) the CEWS repository scaffold. Safe to re-run.",
    )
    parser.add_argument("--root", help="project root (default: parent of scripts/, else ./cews)")
    parser.add_argument("--force", action="store_true", help="overwrite existing non-empty files")
    parser.add_argument(
        "--dry-run", action="store_true", help="show what would happen; write nothing"
    )
    parser.add_argument(
        "--validate", action="store_true", help="only check that the scaffold is complete"
    )
    parser.add_argument("--tree", action="store_true", help="print the directory tree afterwards")
    parser.add_argument("--ascii", action="store_true", help="use ASCII characters for the tree")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="list every created/skipped file"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns 0 on success and 1 if critical scaffolding failed or is missing."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING, format="%(levelname)s %(message)s"
    )
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(errors="replace")

    root = resolve_root(args.root, Path(__file__), Path.cwd())
    manifest = build_manifest()
    ascii_tree = args.ascii or not _stdout_supports_unicode()

    if args.validate:
        problems = validate_scaffold(root, manifest)
        if problems:
            print(f"Scaffold INCOMPLETE under {root}: {len(problems)} problem(s)")
            for item in problems:
                print(
                    f"  [{'CRITICAL' if item.critical else 'warning'}] {item.path}: {item.detail}"
                )
            return EXIT_CRITICAL_FAILURE
        print(
            f"Scaffold OK under {root}: {len(manifest.directories)} directories, {len(manifest.files)} files"
        )
        if args.tree:
            print(render_tree(root, ascii_only=ascii_tree))
        return EXIT_OK

    report = bootstrap(root, force=args.force, dry_run=args.dry_run, manifest=manifest)
    print_summary(report, verbose=args.verbose)

    missing_critical: list[Result] = []
    if not args.dry_run:
        problems = validate_scaffold(root, manifest)
        missing_critical = [p for p in problems if p.critical]
        print(
            f"\nValidation: {'OK' if not problems else str(len(problems)) + ' expected item(s) missing'}"
        )
        for item in problems:
            print(f"  [{'CRITICAL' if item.critical else 'warning'}] {item.path}: {item.detail}")
    if args.tree and root.exists():
        print("\n" + render_tree(root, ascii_only=ascii_tree))

    if report.critical_failures() or missing_critical:
        print("\nResult: FAILED (critical scaffolding failed)")
        return EXIT_CRITICAL_FAILURE
    print("\nResult: SUCCESS" + (" (dry-run, nothing written)" if args.dry_run else ""))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
