# Demo data

The demo dataset is **synthetic**. Organizations are invented, and the activity patterns are
artificial. They do not describe real companies or real research, and no conclusion about the
real world should be drawn from them.

## How it is labelled

- Source names start with `synthetic_` and identifiers with `SYN-`.
- Abstracts start with `[SYNTHETIC]`. Source URLs use the reserved `.invalid` domain.
- Every record and derived row carries `is_synthetic = true`.
- A database holding live records refuses to load demo data (and the reverse) unless you pass
  `--allow-mixed`. Prefer a separate database: set a different `SQLITE_PATH` for live data.

## Commands

```
python scripts/initialize_database.py            # schema + taxonomy
python scripts/seed_demo_data.py                 # load 30 months (about 4,700 records)
python scripts/seed_demo_data.py --write-fixtures   # also write JSONL files to data/samples/demo
python scripts/reset_demo.py --yes               # delete synthetic rows only
```

Options: `--seed` (default 42), `--months` (12 to 120), `--scale` (0.05 to 10).
The same options always produce identical data, so scores and charts are repeatable.
Loading is idempotent: seeding twice leaves the row counts unchanged.

## Built-in scenarios

| Scenario | Topic / organization | What the data does |
|---|---|---|
| Sustained trend | CRISPR gene editing | Grows steadily in all five source types |
| Declining topic | Immune checkpoint inhibitors | Falls in all source types |
| One-time anomaly | AAV gene delivery | One month of patents from one organization |
| High growth, low competition | AI-assisted drug discovery | Strong growth, only two active companies |
| High-threat competitor | Zentavia Pharma in bispecific antibodies | Rising trials, larger enrollment, phase progression, more patents |
| New market entry | Zentavia Pharma in neurodegeneration | First significant activity in the last six months |
| Low-confidence false trend | siRNA / RNA interference | Large relative growth from about eight records, one source, one organization |
| Seasonal topic | mRNA therapeutics | Publications follow a yearly cycle |
| Topic missing from taxonomy | Targeted protein degradation | Growing topic the taxonomy does not contain (AI topic discovery target) |
| Messy organization names | All companies | Several spellings each, a subsidiary, and look-alikes (Orvexa Bio vs Orvexa Labs) |

The machine-readable version is written to `data/samples/demo/scenarios.json`.

## Ground truth

Each record's payload holds `synthetic_ground_truth` (true topic, organization, announcement
type). It exists only so evaluations can check the pipeline's answers. **The production
pipeline must never read it.**

## Limits

Backtests on synthetic data only show that the pipeline runs; they do not show that the scores
are valid. Validity claims must come from live-data backtests and expert review.
