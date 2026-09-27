# CEWS: Competitor Early Warning System

A local, explainable competitive-intelligence proof of concept for life-sciences
organizations. It collects public data, computes evidence-linked scores, and shows the
results on a dashboard. It is not a chatbot and uses no paid services.

> Status: Dashboard (all five scores, forecasts, anomalies, and a Streamlit dashboard over them). Insights and the AI layers are next.

## Quick start (Phase 5)

```
python -m pip install -r requirements-dev.txt
python -m pip install -e .
copy .env.example .env          # Windows; use cp on macOS/Linux
python scripts/validate_environment.py
python scripts/initialize_database.py
python scripts/seed_demo_data.py
python scripts/initialize_database.py   # safe to repeat; also upgrades the schema
python -m cews.cli sources              # data sources and their state
python -m cews.cli normalize            # resolve organizations and topics
python -m cews.cli review               # matches a person should confirm
python -m cews.cli competitors          # who CEWS is monitoring, and why
python -m cews.cli features             # monthly activity, growth, momentum, velocity
python -m cews.cli score                # score the features, with confidence
python -m cews.cli forecast             # forecast activity and flag unusual months
python -m cews.cli review               # decisions it left to a human
python -m pytest
```

## Collecting live data

Live data and demo data are never mixed, so give live collection its own database:

```
copy .env .env.live                     # Windows; use cp on macOS/Linux
# in .env.live set SQLITE_PATH=./data/cews_live.db and DEFAULT_LOOKBACK_DAYS=30
python scripts/initialize_database.py --env-file .env.live
python -m cews.cli sources --health --env-file .env.live
python scripts/fetch_all.py --env-file .env.live --dry-run
python scripts/fetch_all.py --env-file .env.live
```

Company announcements need feed URLs: add official RSS or investor-relations feeds
under `generic_rss.feeds` in `config/source_registry.yaml`. See `docs/data_sources.md`.

The demo data is SYNTHETIC (invented organizations, artificial patterns).
Scaffold tools: `python scripts/bootstrap_project.py --tree` / `--validate`.

## Command reference (make and plain Python)

| make | Python |
|---|---|
| `make setup` | `python -m pip install -r requirements-dev.txt && python -m pip install -e .` |
| `make db-up` | `docker compose up -d postgres` |
| `make check-env` | `python scripts/validate_environment.py` |
| `make db-init` | `python scripts/initialize_database.py` |
| `make db-status` | `python -m cews.cli db-status` |
| `make reset-demo` | `python scripts/reset_demo.py --yes` |
| `make seed` | `python scripts/seed_demo_data.py` |
| `make sources` | `python -m cews.cli sources` |
| `make fetch` | `python scripts/fetch_all.py` |
| `make normalize` | `python -m cews.cli normalize` |
| `make competitors` | `python -m cews.cli competitors` |
| `make features` | `python -m cews.cli features` |
| `make score` | `python -m cews.cli score` |
| `make forecast` | `python -m cews.cli forecast` |
| `make review` | `python -m cews.cli review` |
| `make analyze` | `python scripts/run_analysis.py` |
| `make ai` | `python scripts/run_analysis.py --with-ai` |
| `make evaluate` | `python scripts/run_evaluation.py` |
| `make export` | `python scripts/export_powerbi.py` |
| `make scheduler` | `python scripts/run_scheduler.py` |
| `make api` | `python -m uvicorn cews.api.app:app --reload` |
| `make dashboard` | `python -m streamlit run dashboards/streamlit/app.py` |
| `make test` | `python -m pytest` |

## 1. Project overview

_Not written yet._

## 2. Business problem

_Not written yet._

## 3. Business value

_Not written yet._

## 4. Features

_Not written yet._

## 5. Architecture

_Not written yet._

## 6. Data sources

_Not written yet._

## 7. Technology stack

_Not written yet._

## 8. Installation

_Not written yet._

## 9. Configuration

_Not written yet._

## 10. Database setup

_Not written yet._

## 11. Running demo mode

_Not written yet._

## 12. Running live data mode

_Not written yet._

## 13. Running the scheduler

_Not written yet._

## 14. Running tests

_Not written yet._

## 15. Opening the dashboard

_Not written yet._

## 16. Power BI connection

_Not written yet._

## 17. Scoring formulas

_Not written yet._

## 18. Validation methodology

_Not written yet._

## 19. Known limitations

_Not written yet._

## 20. Ethical and legal considerations

_Not written yet._

## 21. Troubleshooting

_Not written yet._

## 22. Future production enhancements

_Not written yet._
