"""Constants and enumerations shared across CEWS.

Enumerations are ``StrEnum`` so members compare equal to (and serialize as) their string
values, which is what the database stores.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE: Path = PROJECT_ROOT / ".env"
CONFIG_DIR: Path = PROJECT_ROOT / "config"

DEFAULT_FETCH_INTERVAL_MINUTES = 120
DEMO_RANDOM_SEED = 42
MAX_URL_LENGTH = 1024
MAX_ORG_NAME_LENGTH = 200
SYNTHETIC_SOURCE_PREFIX = "synthetic_"
SCORE_MIN = 0.0
SCORE_MAX = 100.0


class CompetitorMode(StrEnum):
    """How the monitored competitor list is built."""

    AUTO = "AUTO"
    MANUAL = "MANUAL"
    HYBRID = "HYBRID"


class DatabaseBackend(StrEnum):
    """Supported database backends."""

    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"


class NormalizationMethod(StrEnum):
    """Feature normalization methods."""

    PERCENTILE = "percentile"
    WINSORIZED_MINMAX = "winsorized_minmax"
    ROBUST_ZSCORE = "robust_zscore"
    MINMAX = "minmax"


class LLMProvider(StrEnum):
    """Where a configured LLM call is sent.

    ``none`` disables every LLM-backed feature and is the default: nothing here is required for
    CEWS to work. ``ollama`` and ``openai_compatible`` both speak the OpenAI chat-completions
    format, so switching between a local model and a hosted one (NVIDIA NIM, or any other
    OpenAI-compatible endpoint) is a config change, never a code change.
    """

    NONE = "none"
    OLLAMA = "ollama"
    OPENAI_COMPATIBLE = "openai_compatible"


class AnnouncementType(StrEnum):
    """The categories a company announcement is classified into.

    ``OTHER`` is a legitimate answer, not a failure: a classifier that reports "I don't know
    which of these six it is" is more honest than one forced to guess.
    """

    LICENSING = "licensing"
    PARTNERSHIP = "partnership"
    ACQUISITION = "acquisition"
    FUNDING = "funding"
    CLINICAL_MILESTONE = "clinical_milestone"
    REGULATORY = "regulatory"
    OTHER = "other"


class SourceType(StrEnum):
    """Kinds of records CEWS collects (also used as ``record_type``)."""

    CLINICAL_TRIAL = "clinical_trial"
    PUBLICATION = "publication"
    PATENT = "patent"
    FUNDING = "funding"
    ANNOUNCEMENT = "announcement"


class ProcessingStatus(StrEnum):
    """Normalization state of a source record."""

    NEW = "new"
    NORMALIZED = "normalized"
    FAILED = "failed"


class RunStatus(StrEnum):
    """Outcome of an ingestion or job run."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class OrganizationType(StrEnum):
    """Coarse organization classes used to keep non-competitors out of discovery."""

    COMPANY = "company"
    UNIVERSITY = "university"
    HOSPITAL = "hospital"
    GOVERNMENT = "government"
    NONPROFIT = "nonprofit"
    INDIVIDUAL = "individual"
    UNKNOWN = "unknown"


class PeriodType(StrEnum):
    """Aggregation period granularity."""

    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"


class TopicStatus(StrEnum):
    """Review state of a topic."""

    ACTIVE = "active"
    PENDING_REVIEW = "pending_review"
    REJECTED = "rejected"


class RelationshipType(StrEnum):
    """How an organization relates to a source record."""

    SPONSOR = "sponsor"
    ASSIGNEE = "assignee"
    AFFILIATION = "affiliation"
    RECIPIENT = "recipient"
    ANNOUNCER = "announcer"
    PARTNER = "partner"


class EntityType(StrEnum):
    """Entity a score, feature, forecast or insight refers to."""

    TOPIC = "topic"
    COMPETITOR = "competitor"
    THERAPEUTIC_AREA = "therapeutic_area"


class ScoreType(StrEnum):
    """Kinds of scores CEWS computes."""

    TREND = "trend"
    INNOVATION = "innovation"
    THREAT = "threat"
    OPPORTUNITY = "opportunity"
    CONFIDENCE = "confidence"


class AlertRating(StrEnum):
    """Expert review outcomes for an alert."""

    RELEVANT = "relevant"
    NOT_RELEVANT = "not_relevant"
    DUPLICATE = "duplicate"
    TOO_LATE = "too_late"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NEEDS_INVESTIGATION = "needs_investigation"


# Environment-flag name -> source registry id (see config/source_registry.yaml).
SOURCE_FLAGS: dict[str, str] = {
    "enable_clinical_trials_gov": "clinical_trials_gov",
    "enable_pubmed": "pubmed",
    "enable_europe_pmc": "europe_pmc",
    "enable_openalex": "openalex",
    "enable_nih_reporter": "nih_reporter",
    "enable_generic_rss": "generic_rss",
    "enable_patents": "patents_uspto_bulk",
    "enable_epo_ops": "epo_ops",
}
