# CEWS: Competitor Early Warning System

A local, explainable competitive-intelligence proof of concept for life-sciences
organizations. It collects public data, computes evidence-linked scores, and shows the
results on a dashboard. It is not a chatbot and uses no paid services.

> Status: Phase 1 (scaffolding). Most modules are placeholders.

## Quick start (Phase 1)

```
python scripts/bootstrap_project.py --tree     # create missing scaffold, print tree
python scripts/bootstrap_project.py --validate # verify the scaffold
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Command reference (make and plain Python)

| make | Python |
|---|---|
| `make setup` | `python -m pip install -r requirements-dev.txt && python -m pip install -e .` |
| `make db-up` | `docker compose up -d postgres` |
| `make db-init` | `python scripts/initialize_database.py` |
| `make seed` | `python scripts/seed_demo_data.py` |
| `make fetch` | `python scripts/fetch_all.py` |
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
