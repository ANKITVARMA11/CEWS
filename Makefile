# CEWS task runner. Without make, use the plain Python commands listed in README.md.
PYTHON ?= python
.DEFAULT_GOAL := help

.PHONY: help bootstrap check-scaffold check-env setup db-up db-down db-init db-status seed reset-demo sources normalize review competitors features score forecast dashboard fetch analyze ai evaluate export scheduler api dashboard demo test cov lint format typecheck clean

help: ## Show this help
	@awk -F ':.*## ' '/^[a-zA-Z_-]+:.*## / {printf "  %-16s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

bootstrap: ## Create missing scaffold files (safe to re-run) and print the tree
	$(PYTHON) scripts/bootstrap_project.py --tree

check-scaffold: ## Verify the scaffold is complete
	$(PYTHON) scripts/bootstrap_project.py --validate

check-env: ## Validate configuration, packages and database access
	$(PYTHON) scripts/validate_environment.py

setup: ## Install the package and development dependencies
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pip install -e .

db-up: ## Start local PostgreSQL (optional; SQLite is the default)
	docker compose up -d postgres

db-down: ## Stop local PostgreSQL
	docker compose down

db-init: ## Create database tables
	$(PYTHON) scripts/initialize_database.py

db-status: ## Show schema revision and record counts
	$(PYTHON) -m cews.cli db-status

seed: ## Load synthetic demo data
	$(PYTHON) scripts/seed_demo_data.py

reset-demo: ## Delete all synthetic demo data
	$(PYTHON) scripts/reset_demo.py --yes

sources: ## List data sources, adapters, checkpoints and circuit state
	$(PYTHON) -m cews.cli sources

fetch: ## Run one manual fetch across enabled sources
	$(PYTHON) scripts/fetch_all.py

normalize: ## Resolve organizations, assign topics, mark duplicates
	$(PYTHON) -m cews.cli normalize

review: ## Show entity decisions awaiting a human
	$(PYTHON) -m cews.cli review

competitors: ## Rank and show the monitored competitors
	$(PYTHON) -m cews.cli competitors

features: ## Count activity and compute trend features
	$(PYTHON) -m cews.cli features

score: ## Score the stored features
	$(PYTHON) -m cews.cli score

forecast: ## Forecast activity and flag unusual months
	$(PYTHON) -m cews.cli forecast

analyze: ## Run features, scores, forecasts and insights
	$(PYTHON) scripts/run_analysis.py

ai: ## Run analysis including the optional AI layers
	$(PYTHON) scripts/run_analysis.py --with-ai

evaluate: ## Run data-quality checks, backtests and AI ablation
	$(PYTHON) scripts/run_evaluation.py

export: ## Write Power BI CSV exports
	$(PYTHON) scripts/export_powerbi.py

scheduler: ## Start the background scheduler
	$(PYTHON) scripts/run_scheduler.py

api: ## Start the FastAPI service
	$(PYTHON) -m uvicorn cews.api.app:app --reload

dashboard: ## Start the Streamlit dashboard
	$(PYTHON) -m streamlit run dashboards/streamlit/app.py

demo: db-init seed analyze ai evaluate export ## Run the offline demo pipeline on synthetic data

test: ## Run the test suite
	$(PYTHON) -m pytest

cov: ## Run tests with a coverage report
	$(PYTHON) -m pytest --cov=src/cews --cov-report=term-missing --cov-report=html

lint: ## Ruff and Black checks
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check .

format: ## Auto-format the code
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m black .

typecheck: ## Run mypy
	$(PYTHON) -m mypy src

clean: ## Remove caches
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in list(pathlib.Path('.').rglob('__pycache__')) + [pathlib.Path(x) for x in ('.pytest_cache', '.mypy_cache', '.ruff_cache', 'htmlcov')]]"
