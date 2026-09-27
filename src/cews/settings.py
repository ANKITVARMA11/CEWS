"""CEWS configuration: loading, validation, and competitor-mode resolution.

Settings come from (highest priority first) explicit overrides, real environment variables,
the ``.env`` file, and the defaults below. The defaults are identical to ``.env.example``
(a test enforces this), so CEWS runs without a ``.env`` file.

Key public functions:

* :func:`load_settings` builds a validated :class:`Settings` or raises :class:`SettingsError`.
* :func:`validate_settings` performs cross-cutting checks (files exist, credentials present)
  and returns human-readable warnings.
* :func:`parse_competitor_list` turns a raw include/exclude string into a clean list.
* :func:`resolve_competitor_mode` converts the settings into a :class:`CompetitorPlan`.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import (
    Field,
    SecretStr,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from cews.constants import (
    DEFAULT_ENV_FILE,
    DEFAULT_FETCH_INTERVAL_MINUTES,
    MAX_ORG_NAME_LENGTH,
    PROJECT_ROOT,
    SOURCE_FLAGS,
    CompetitorMode,
    DatabaseBackend,
    LLMProvider,
    NormalizationMethod,
)

LOGGER = logging.getLogger(__name__)

SECRET_FIELDS: tuple[str, ...] = (
    "ncbi_api_key",
    "openalex_api_key",
    "uspto_odp_api_key",
    "epo_consumer_key",
    "epo_consumer_secret",
    "llm_api_key",
)
_PATH_FIELDS: tuple[str, ...] = (
    "sqlite_path",
    "source_registry_file",
    "topic_taxonomy_file",
    "scoring_config_file",
    "export_directory",
    "competitor_config_file",
)
_LEGAL_SUFFIXES = frozenset(
    {
        "inc",
        "ltd",
        "limited",
        "plc",
        "gmbh",
        "ag",
        "corp",
        "corporation",
        "company",
        "co",
        "llc",
        "sa",
    }
)


class SettingsError(ValueError):
    """Raised when configuration is invalid. ``problems`` lists each individual issue."""

    def __init__(self, message: str, problems: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.problems: tuple[str, ...] = tuple(problems)


# --------------------------------------------------------------------------------------
# Settings model
# --------------------------------------------------------------------------------------
class Settings(BaseSettings):
    """All CEWS configuration. Environment variable names equal the upper-cased field names."""

    model_config = SettingsConfigDict(
        extra="ignore", case_sensitive=False, env_file_encoding="utf-8-sig"
    )

    # Application
    project_root: Path = PROJECT_ROOT
    app_env: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    timezone: str = "Asia/Kolkata"

    # Database
    database_backend: DatabaseBackend = DatabaseBackend.SQLITE
    database_url: str = "postgresql+psycopg://cews:cews@localhost:5432/cews"
    sqlite_path: Path = Path("./data/cews.db")

    # Scheduler and collection
    fetch_interval_minutes: int = Field(DEFAULT_FETCH_INTERVAL_MINUTES, ge=1, le=10080)
    run_fetch_on_startup: bool = False
    enable_scheduler: bool = True
    max_concurrent_source_jobs: int = Field(3, ge=1, le=16)
    source_request_timeout_seconds: float = Field(30, gt=0, le=600)
    source_max_retries: int = Field(3, ge=0, le=10)
    default_lookback_days: int = Field(1095, ge=1, le=36500)
    incremental_lookback_days: int = Field(7, ge=1, le=365)
    source_registry_file: Path = Path("./config/source_registry.yaml")
    circuit_breaker_failure_threshold: int = Field(3, ge=1, le=100)
    circuit_breaker_cooldown_minutes: int = Field(360, ge=1, le=10080)

    # Competitors (stored as raw strings; use the *_list properties or resolve_competitor_mode)
    competitor_mode: CompetitorMode = CompetitorMode.HYBRID
    top_competitors: int = Field(20, ge=1, le=500)
    competitor_include: str = "Pfizer,Roche,Novartis,Merck,AstraZeneca"
    competitor_exclude: str = ""
    competitor_config_file: Path | None = None
    min_competitor_evidence_count: int = Field(5, ge=0)

    # Topics
    therapeutic_areas: str = (
        "Oncology,Neurology,Immunology,Rare Diseases,Gene Therapy,RNA Therapeutics,"
        "Cancer Vaccines,CAR-T"
    )
    topic_taxonomy_file: Path = Path("./config/topic_taxonomy.yaml")

    # Sources
    enable_clinical_trials_gov: bool = True
    enable_pubmed: bool = True
    enable_europe_pmc: bool = True
    enable_generic_rss: bool = True
    enable_openalex: bool = False
    enable_nih_reporter: bool = False
    enable_patents: bool = False
    enable_epo_ops: bool = False
    ncbi_api_key: SecretStr | None = None
    ncbi_email: str = ""
    openalex_api_key: SecretStr | None = None
    uspto_odp_api_key: SecretStr | None = None
    epo_consumer_key: SecretStr | None = None
    epo_consumer_secret: SecretStr | None = None

    # AI layers
    enable_ai_topic_discovery: bool = False
    enable_ai_announcement_extraction: bool = False
    enable_ai_org_matching: bool = False
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # LLM provider: used by announcement extraction (Phase 9) and any future MCP-adjacent
    # feature that needs a model. Everything here is optional; ENABLE_* flags above decide
    # whether an LLM is called at all, and this only decides where the call goes.
    #   none               - no LLM calls are made; the deterministic fallback always runs.
    #   ollama             - a local server (llama.cpp/Ollama), reached over plain HTTP.
    #   openai_compatible  - a hosted or self-hosted endpoint speaking the OpenAI chat-completions
    #                        API: NVIDIA NIM, Groq, Together, a company-run gateway, and so on.
    llm_provider: LLMProvider = LLMProvider.NONE
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:3b"
    llm_api_key: SecretStr | None = None
    llm_timeout_seconds: float = Field(30.0, gt=0)
    llm_max_retries: int = Field(2, ge=0)
    ai_topic_novelty_threshold: float = Field(0.55, ge=0, le=1)
    ai_org_match_auto_threshold: float = Field(0.92, ge=0, le=1)
    ai_org_match_review_threshold: float = Field(0.80, ge=0, le=1)
    ai_max_docs_per_run: int = Field(5000, ge=1)

    # Scoring
    scoring_config_file: Path = Path("./config/scoring_weights.yaml")
    min_topic_sample_size: int = Field(10, ge=1)
    normalization_method: NormalizationMethod = NormalizationMethod.PERCENTILE
    normalization_winsor_lower: float = Field(0.05, ge=0, lt=1)
    normalization_winsor_upper: float = Field(0.95, gt=0, le=1)
    confidence_min_score: float = Field(0, ge=0, le=100)
    confidence_max_score: float = Field(100, ge=0, le=100)

    # Alerts
    alert_trend_threshold: float = Field(75, ge=0, le=100)
    alert_threat_threshold: float = Field(75, ge=0, le=100)
    alert_opportunity_threshold: float = Field(70, ge=0, le=100)
    alert_min_confidence: float = Field(60, ge=0, le=100)

    # Backtesting
    backtest_train_months: int = Field(24, ge=6, le=240)
    backtest_horizon_months: int = Field(6, ge=1, le=24)
    backtest_top_k: int = Field(10, ge=1, le=1000)

    # Exports, reports, dashboard
    export_directory: Path = Path("./data/exports")
    generate_reports: bool = True
    generate_powerbi_exports: bool = True
    enable_streamlit_dashboard: bool = True

    # ---- validators -------------------------------------------------------------------
    @field_validator("app_env", mode="before")
    @classmethod
    def _lower_app_env(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("log_level", "competitor_mode", mode="before")
    @classmethod
    def _upper(cls, value: Any) -> Any:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator("database_backend", "normalization_method", mode="before")
    @classmethod
    def _lower(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator(*SECRET_FIELDS, "competitor_config_file", mode="before")
    @classmethod
    def _blank_to_none(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        return value.strip() if isinstance(value, str) else value

    @field_validator("project_root", mode="after")
    @classmethod
    def _absolute_root(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @field_validator(*_PATH_FIELDS, mode="after")
    @classmethod
    def _resolve_relative_paths(cls, value: Path | None, info: ValidationInfo) -> Path | None:
        if value is None:
            return None
        root = info.data.get("project_root", PROJECT_ROOT)
        expanded = value.expanduser()
        return expanded if expanded.is_absolute() else Path(os.path.normpath(root / expanded))

    @field_validator("timezone", mode="after")
    @classmethod
    def _known_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(
                f"unknown time zone {value!r}. Use an IANA name such as 'Asia/Kolkata'. "
                "On Windows, also run: python -m pip install tzdata"
            ) from exc
        return value

    @field_validator("llm_base_url", mode="after")
    @classmethod
    def _valid_http_url(cls, value: str) -> str:
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                f"must be an http(s) URL such as http://localhost:11434/v1, got {value!r}"
            )
        return value.rstrip("/")

    @model_validator(mode="after")
    def _cross_field_checks(self) -> Settings:
        if self.incremental_lookback_days > self.default_lookback_days:
            raise ValueError("INCREMENTAL_LOOKBACK_DAYS must not exceed DEFAULT_LOOKBACK_DAYS")
        if self.normalization_winsor_lower >= self.normalization_winsor_upper:
            raise ValueError(
                "NORMALIZATION_WINSOR_LOWER must be smaller than NORMALIZATION_WINSOR_UPPER"
            )
        if self.confidence_min_score >= self.confidence_max_score:
            raise ValueError("CONFIDENCE_MIN_SCORE must be smaller than CONFIDENCE_MAX_SCORE")
        if self.ai_org_match_review_threshold >= self.ai_org_match_auto_threshold:
            raise ValueError(
                "AI_ORG_MATCH_REVIEW_THRESHOLD must be smaller than AI_ORG_MATCH_AUTO_THRESHOLD"
            )
        if (
            self.database_backend is DatabaseBackend.POSTGRESQL
            and not self.database_url.startswith("postgresql")
        ):
            raise ValueError(
                "DATABASE_URL must start with 'postgresql' when DATABASE_BACKEND=postgresql"
            )
        return self

    # ---- convenience ------------------------------------------------------------------
    @property
    def competitor_include_list(self) -> list[str]:
        """Names from COMPETITOR_INCLUDE (environment only; see resolve_competitor_mode)."""
        return parse_competitor_list(self.competitor_include)

    @property
    def competitor_exclude_list(self) -> list[str]:
        """Names from COMPETITOR_EXCLUDE (environment only; see resolve_competitor_mode)."""
        return parse_competitor_list(self.competitor_exclude)

    @property
    def therapeutic_area_list(self) -> list[str]:
        """Monitored therapeutic areas as a list."""
        return parse_competitor_list(self.therapeutic_areas)

    @property
    def fetch_interval(self) -> timedelta:
        """Fetch interval as a ``timedelta``."""
        return timedelta(minutes=self.fetch_interval_minutes)

    @property
    def enabled_sources(self) -> tuple[str, ...]:
        """Registry ids of sources whose ENABLE_* flag is true."""
        return tuple(source_id for flag, source_id in SOURCE_FLAGS.items() if getattr(self, flag))

    def secret_values(self) -> list[str]:
        """Non-empty secret values, for log redaction. Never log the result."""
        values: list[str] = []
        for name in SECRET_FIELDS:
            secret = getattr(self, name)
            if secret is not None and secret.get_secret_value():
                values.append(secret.get_secret_value())
        return values

    def safe_dump(self) -> dict[str, Any]:
        """Return settings as JSON-friendly data with secrets masked."""
        return self.model_dump(mode="json")


# --------------------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------------------
def _format_validation_error(exc: ValidationError) -> tuple[str, list[str]]:
    problems: list[str] = []
    for error in exc.errors():
        loc = ".".join(str(part) for part in error.get("loc", ()))
        name = loc.upper() if loc else "configuration"
        message = str(error.get("msg", "invalid value"))
        if message.startswith("Value error, "):
            message = message[len("Value error, ") :]
        detail = f"{name}: {message}"
        given = error.get("input")
        if loc.lower() not in SECRET_FIELDS and isinstance(given, str | int | float | bool):
            shown = repr(given) if len(repr(given)) <= 60 else repr(given)[:57] + "..."
            detail += f" (got {shown})"
        problems.append(detail)
    message = "Invalid configuration:\n" + "\n".join(f"  - {p}" for p in problems)
    return message, problems


def load_settings(
    env_file: str | Path | None = DEFAULT_ENV_FILE,
    overrides: Mapping[str, Any] | None = None,
) -> Settings:
    """Load and validate settings.

    Args:
        env_file: Path to a dotenv file, or ``None`` to ignore dotenv files. The default
            ``.env`` in the project root is optional; an explicitly given file must exist.
        overrides: Field-name -> value mapping that takes priority over everything else.

    Raises:
        SettingsError: if any value is invalid. The message lists every problem.
    """
    if env_file is not None:
        path = Path(env_file)
        if path != DEFAULT_ENV_FILE and not path.is_file():
            raise SettingsError(f"Environment file not found: {path}")
    try:
        kwargs: dict[str, Any] = {"_env_file": env_file, **dict(overrides or {})}
        return Settings(**kwargs)
    except ValidationError as exc:
        message, problems = _format_validation_error(exc)
        raise SettingsError(message, problems) from exc


def validate_settings(settings: Settings) -> list[str]:
    """Run cross-cutting checks and return warnings.

    Raises:
        SettingsError: if a required file is missing or the competitor configuration is
            unusable (for example MANUAL mode with no included competitors).
    """
    errors: list[str] = []
    for label, path in (
        ("SOURCE_REGISTRY_FILE", settings.source_registry_file),
        ("TOPIC_TAXONOMY_FILE", settings.topic_taxonomy_file),
        ("SCORING_CONFIG_FILE", settings.scoring_config_file),
    ):
        if not path.is_file():
            errors.append(f"{label}: file not found: {path}")
    if errors:
        raise SettingsError(
            "Invalid configuration:\n" + "\n".join(f"  - {e}" for e in errors), errors
        )

    plan = resolve_competitor_mode(settings)
    warnings = list(plan.warnings)

    if settings.circuit_breaker_cooldown_minutes < settings.fetch_interval_minutes:
        warnings.append(
            f"CIRCUIT_BREAKER_COOLDOWN_MINUTES={settings.circuit_breaker_cooldown_minutes} is shorter "
            f"than FETCH_INTERVAL_MINUTES={settings.fetch_interval_minutes}, so a failing source is "
            "retried every cycle anyway."
        )
    if settings.fetch_interval_minutes < 30:
        warnings.append(
            f"FETCH_INTERVAL_MINUTES={settings.fetch_interval_minutes} is very short; public APIs "
            "have rate limits and research trends do not change within minutes."
        )
    if settings.enable_openalex and settings.openalex_api_key is None:
        warnings.append(
            "ENABLE_OPENALEX=true but OPENALEX_API_KEY is empty; OpenAlex now requires a free key, "
            "so this source will be skipped."
        )
    if settings.enable_patents and settings.uspto_odp_api_key is None:
        warnings.append(
            "ENABLE_PATENTS=true but USPTO_ODP_API_KEY is empty; the USPTO Open Data Portal may "
            "require a key (verify in docs/data_sources.md)."
        )
    if (settings.enable_ai_topic_discovery or settings.enable_ai_org_matching) and (
        importlib.util.find_spec("sentence_transformers") is None
    ):
        warnings.append(
            "An embedding-based AI layer is enabled but sentence-transformers is not installed "
            "(python -m pip install -r requirements-ai.txt); the deterministic fallback will be used."
        )
    if (
        settings.enable_ai_announcement_extraction
        and settings.llm_provider is LLMProvider.OPENAI_COMPATIBLE
        and settings.llm_api_key is None
    ):
        warnings.append(
            "LLM_PROVIDER=openai_compatible but LLM_API_KEY is empty; most hosted endpoints "
            "(NVIDIA NIM included) will reject the request and extraction will fall back to the "
            "deterministic classifier."
        )
    if not settings.enabled_sources:
        warnings.append("No data sources are enabled; only demo data can be loaded.")
    return list(dict.fromkeys(warnings))


# --------------------------------------------------------------------------------------
# Competitor configuration
# --------------------------------------------------------------------------------------
def parse_competitor_list(raw: str | Sequence[str] | None) -> list[str]:
    """Parse a competitor (or area) list from a raw string or a sequence of strings.

    Names are separated by commas or new lines. If the string contains a semicolon, only
    semicolons and new lines separate names, so a name such as ``Merck & Co., Inc.`` can
    contain commas. Blank items are dropped and case-insensitive duplicates are removed,
    keeping the first spelling.

    Raises:
        TypeError: if ``raw`` is not a string, a sequence of strings, or ``None``.
        ValueError: if a name is longer than ``MAX_ORG_NAME_LENGTH``.
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.replace("\r", "\n")
        separator = ";" if ";" in text else ","
        parts = [piece for line in text.split("\n") for piece in line.split(separator)]
    elif isinstance(raw, Sequence):
        parts = []
        for item in raw:
            if not isinstance(item, str):
                raise TypeError(f"competitor names must be strings, got {type(item).__name__}")
            parts.append(item)
    else:
        raise TypeError(f"expected a string or sequence of strings, got {type(raw).__name__}")

    names: list[str] = []
    seen: set[str] = set()
    for part in parts:
        name = " ".join(part.split())
        if not name:
            continue
        if len(name) > MAX_ORG_NAME_LENGTH:
            raise ValueError(f"name longer than {MAX_ORG_NAME_LENGTH} characters: {name[:40]}...")
        key = name.casefold()
        if key not in seen:
            seen.add(key)
            names.append(name)
    return names


def name_key(name: str) -> str:
    """Return a light comparison key for a user-supplied organization name.

    Lower-cases, removes punctuation, and strips trailing legal suffixes so that ``Pfizer``
    and ``Pfizer Inc.`` compare equal. The full organization normalizer (aliases, fuzzy
    matching) arrives in Phase 5; this key is only used to reconcile include/exclude lists.
    """
    text = re.sub(r"[^\w\s]", " ", name.casefold().replace("&", " and "))
    tokens = text.split()
    while len(tokens) > 1 and tokens[-1] in _LEGAL_SUFFIXES:
        tokens.pop()
    while len(tokens) > 1 and tokens[-1] == "and":
        tokens.pop()
    return " ".join(tokens)


def load_competitor_config_file(path: Path) -> tuple[list[str], list[str]]:
    """Read ``include`` and ``exclude`` lists from a YAML file.

    Each key may hold a list of names or a single delimited string.

    Raises:
        SettingsError: if the file is missing, is not valid YAML, or has the wrong shape.
    """
    if not path.is_file():
        raise SettingsError(f"COMPETITOR_CONFIG_FILE: file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError) as exc:
        raise SettingsError(f"COMPETITOR_CONFIG_FILE: cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SettingsError(
            f"COMPETITOR_CONFIG_FILE: {path} must contain a mapping with 'include'/'exclude'"
        )
    unknown = set(data) - {"include", "exclude"}
    if unknown:
        raise SettingsError(
            f"COMPETITOR_CONFIG_FILE: unknown keys {sorted(unknown)}; use include/exclude"
        )
    try:
        return (
            parse_competitor_list(data.get("include")),
            parse_competitor_list(data.get("exclude")),
        )
    except (TypeError, ValueError) as exc:
        raise SettingsError(f"COMPETITOR_CONFIG_FILE: invalid list in {path}: {exc}") from exc


@dataclass(frozen=True)
class CompetitorPlan:
    """How many competitors are fixed by configuration and how many discovery must fill.

    ``include`` lists names that must always be monitored (already reduced by ``exclude``).
    ``exclude`` lists names that must never be monitored. ``auto_slots`` is the number of
    additional competitors automatic discovery should supply.
    """

    mode: CompetitorMode
    include: tuple[str, ...]
    exclude: tuple[str, ...]
    auto_slots: int
    top_n: int
    warnings: tuple[str, ...] = field(default_factory=tuple)


def resolve_competitor_mode(settings: Settings) -> CompetitorPlan:
    """Turn competitor settings into a :class:`CompetitorPlan`.

    Rules:

    * Names from ``COMPETITOR_INCLUDE``/``COMPETITOR_EXCLUDE`` and from the optional
      ``COMPETITOR_CONFIG_FILE`` are merged (duplicates removed).
    * A name that is both included and excluded is excluded: exclusions always win.
    * ``MANUAL``: only included names; no automatic slots. At least one name is required.
    * ``HYBRID``: included names first, then automatic discovery fills the remaining slots
      up to ``TOP_COMPETITORS``. Included names are never dropped, even if they exceed it.
    * ``AUTO``: every slot is filled by discovery. ``COMPETITOR_INCLUDE`` is ignored (with a
      warning); exclusions still apply.

    Raises:
        SettingsError: for MANUAL mode with no included names or an unreadable config file.
    """
    include = settings.competitor_include_list
    exclude = settings.competitor_exclude_list
    if settings.competitor_config_file is not None:
        file_include, file_exclude = load_competitor_config_file(settings.competitor_config_file)
        include = parse_competitor_list([*include, *file_include])
        exclude = parse_competitor_list([*exclude, *file_exclude])

    warnings: list[str] = []
    excluded_keys = {name_key(name) for name in exclude}
    kept: list[str] = []
    for name in include:
        if name_key(name) in excluded_keys:
            warnings.append(f"'{name}' is in both include and exclude lists; it will be excluded.")
        else:
            kept.append(name)

    # Coerce rather than trust: settings built with model_copy bypass validation, and silently
    # treating an unrecognised mode as AUTO would quietly ignore the configured competitors.
    mode = CompetitorMode(str(settings.competitor_mode).upper())
    top_n = settings.top_competitors
    if mode is CompetitorMode.MANUAL:
        if not kept:
            raise SettingsError(
                "COMPETITOR_MODE=MANUAL requires at least one name in COMPETITOR_INCLUDE "
                "(or the include list of COMPETITOR_CONFIG_FILE) that is not excluded."
            )
        auto_slots = 0
        if len(kept) > top_n:
            warnings.append(f"{len(kept)} included competitors exceed TOP_COMPETITORS={top_n}.")
    elif mode is CompetitorMode.HYBRID:
        auto_slots = max(0, top_n - len(kept))
        if len(kept) > top_n:
            warnings.append(
                f"{len(kept)} included competitors exceed TOP_COMPETITORS={top_n}; all are kept "
                "and no automatic slots remain."
            )
    else:  # AUTO
        if kept:
            warnings.append("COMPETITOR_INCLUDE is ignored in AUTO mode; use HYBRID to keep it.")
        kept = []
        auto_slots = top_n

    return CompetitorPlan(
        mode=mode,
        include=tuple(kept),
        exclude=tuple(exclude),
        auto_slots=auto_slots,
        top_n=top_n,
        warnings=tuple(warnings),
    )
