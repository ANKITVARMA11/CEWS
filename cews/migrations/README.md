# Migrations

- `versions/` holds Alembic revisions (baseline migration added in Phase 2).
- `sql/` holds plain SQL for database views and data-quality checks.
- Alembic environment files (`env.py`, `script.py.mako`) are added in Phase 2. Until then,
  `alembic` commands are not expected to work.
