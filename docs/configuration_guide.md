# Configuration guide

CEWS reads its settings from environment variables and an optional `.env` file in the project
root. Copy `.env.example` to `.env` and edit only what you need: **every default in the code
equals the value in `.env.example`** (a test enforces this), so CEWS also runs with no `.env`.

Never commit a real `.env`. It is listed in `.gitignore`.

## Precedence (highest first)

1. Explicit overrides passed in code (`load_settings(overrides=...)`, used by tests)
2. Real environment variables
3. The `.env` file
4. Built-in defaults

Setting names are the upper-cased field names, for example `FETCH_INTERVAL_MINUTES`. Unknown
keys in `.env` (such as the `POSTGRES_*` values used only by Docker Compose) are ignored.

## Check your configuration

```
python scripts/validate_environment.py
```

This checks the Python version, the required packages, every setting, the config files, and
database access. Invalid values stop with one message that lists **every** problem, for example:

```
Invalid configuration:
  - FETCH_INTERVAL_MINUTES: Input should be greater than or equal to 1 (got '0')
  - ENABLE_SCHEDULER: Input should be a valid boolean, unable to interpret input (got 'maybe')
```

Secret values (API keys) are never printed. Exit code 2 means a configuration problem.

## Collection and scheduling

| Setting | Default | Notes |
|---|---|---|
| `FETCH_INTERVAL_MINUTES` | 120 | 1 to 10080. Values under 30 produce a warning. |
| `RUN_FETCH_ON_STARTUP` | false | Fetch once when the scheduler starts. |
| `ENABLE_SCHEDULER` | true | Turn the background scheduler off without code changes. |
| `MAX_CONCURRENT_SOURCE_JOBS` | 3 | 1 to 16. |
| `SOURCE_REQUEST_TIMEOUT_SECONDS` | 30 | |
| `SOURCE_MAX_RETRIES` | 3 | 0 to 10. |
| `DEFAULT_LOOKBACK_DAYS` | 1095 | History for a first full collection. |
| `INCREMENTAL_LOOKBACK_DAYS` | 7 | Must not exceed the default lookback. |

Two-hour fetching keeps data fresh. Trend features still use monthly and quarterly windows.

| Setting | Default | Notes |
|---|---|---|
| `SOURCE_REGISTRY_FILE` | `./config/source_registry.yaml` | Per-source rate limits, page sizes, refresh policy. |
| `CIRCUIT_BREAKER_FAILURE_THRESHOLD` | 3 | Failed runs in a row before a source is paused. |
| `CIRCUIT_BREAKER_COOLDOWN_MINUTES` | 360 | Pause length. A warning is shown if it is shorter than the fetch interval. |

A run counts as failed for the circuit breaker when a page could not be fetched. Individual bad
records do not count. See `docs/data_sources.md` for how collection works.

## Database

`DATABASE_BACKEND=sqlite` (default) needs no setup and stores data in `SQLITE_PATH`
(default `./data/cews.db`). For PostgreSQL run `docker compose up -d postgres`, then set
`DATABASE_BACKEND=postgresql` and `DATABASE_URL`. Relative paths are resolved against the
project root, not the current folder.

Create or upgrade the schema with `python scripts/initialize_database.py` (safe to repeat).

## Competitors

`COMPETITOR_MODE` is `AUTO`, `MANUAL` or `HYBRID` (default).

| Mode | Behaviour |
|---|---|
| `MANUAL` | Only names in `COMPETITOR_INCLUDE`. At least one is required. |
| `HYBRID` | Included names always appear; automatic discovery fills the remaining slots up to `TOP_COMPETITORS`. |
| `AUTO` | Discovery fills every slot. `COMPETITOR_INCLUDE` is ignored (with a warning). |

In every mode, names in `COMPETITOR_EXCLUDE` never appear, and exclusion wins when a name is in
both lists. Names are compared without case, punctuation or legal suffixes, so `Pfizer Inc.`
matches `Pfizer`.

Separators: commas or new lines. If a value contains a semicolon, only semicolons and new lines
separate names, so a name with commas works: `COMPETITOR_INCLUDE=Merck & Co., Inc.; Pfizer`.

For long lists set `COMPETITOR_CONFIG_FILE` to a YAML file. It is merged with the `.env` lists:

```yaml
include:
  - Sanofi
  - Bayer
exclude:
  - Some Company
```

## Sources and credentials

Each source has an `ENABLE_*` flag in `.env`; everything else about a source (rate limit, page
size, date-slice length, search terms, refresh policy) lives in `config/source_registry.yaml`.
Neither needs a code change. `docs/data_sources.md` explains each source in detail.

| Setting | Default | Notes |
|---|---|---|
| `ENABLE_CLINICAL_TRIALS_GOV` | true | No key needed. |
| `ENABLE_PUBMED` | true | `NCBI_API_KEY` and `NCBI_EMAIL` are optional; a key raises NCBI's rate limit. |
| `ENABLE_EUROPE_PMC` | true | No key. Collects preprints only while PubMed is on, to avoid double counting. |
| `ENABLE_GENERIC_RSS` | true | Does nothing until you add feed URLs to the registry. |
| `ENABLE_OPENALEX` | false | Needs a free `OPENALEX_API_KEY`; adapter not built. |
| `ENABLE_NIH_REPORTER` | false | Adapter not built. |
| `ENABLE_PATENTS` | false | Blocked upstream; see `docs/data_sources.md`. |

Useful registry `options` per source: `query_terms` (search these instead of
`THERAPEUTIC_AREAS`), `query` (a complete source-specific query), `store_abstracts: false`,
`study_type` (ClinicalTrials.gov), `sources` (Europe PMC codes) and `feed_organizations`
(RSS). `python -m cews.cli sources --health` checks that each enabled source answers.

## AI layers

`ENABLE_AI_TOPIC_DISCOVERY`, `ENABLE_AI_ANNOUNCEMENT_EXTRACTION` and `ENABLE_AI_ORG_MATCHING`
are all false by default. Everything works without them. Thresholds:
`AI_TOPIC_NOVELTY_THRESHOLD`, `AI_ORG_MATCH_AUTO_THRESHOLD` (must be greater than
`AI_ORG_MATCH_REVIEW_THRESHOLD`).

## Scoring, alerts, backtesting

`NORMALIZATION_METHOD` (`percentile`, `winsorized_minmax`, `robust_zscore`, `minmax`),
`NORMALIZATION_WINSOR_LOWER/UPPER` (lower must be smaller), `MIN_TOPIC_SAMPLE_SIZE`,
`ALERT_*` thresholds (0 to 100), and `BACKTEST_*` settings. Score weights live in
`config/scoring_weights.yaml`, and the topic taxonomy in `config/topic_taxonomy.yaml`. Neither
needs a code change to edit.

## Charts and JavaScript

CEWS contains no JavaScript. Charts are drawn with matplotlib and rendered server-side as
images, reports are plain HTML and CSS from Jinja2 templates, and Power BI does its own
rendering. Streamlit's own interface is built in JavaScript, but nothing in this project writes
or maintains any; if that ever becomes a problem, Power BI plus the static reports cover the same
ground without it.

## Windows notes

- Time zone names need the `tzdata` package (installed by `requirements.txt`).
- `.env` files saved with a byte-order mark (Notepad does this) are read correctly.
- Without `make`, use the plain Python commands in the README.

## Logging

`config/logging.yaml` defines the handlers. The level comes from `LOG_LEVEL`; the file is
`logs/cews.log`. API keys and other secrets are masked in log messages.
