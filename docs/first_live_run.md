# Your first live fetch

Everything so far has run on invented demo data. This is how to switch to real data safely.
It has to run on your machine: the sources are public web APIs, and the development environment
this was built in has no internet access.

## Before you start

- **Optional but recommended for PubMed:** put `NCBI_API_KEY=` (free from NCBI) and
  `NCBI_EMAIL=` in `.env`. Without a key PubMed is limited to about 3 requests a second.
- **Patents are not collected** (the USPTO source is blocked; see `docs/data_sources.md`), so the
  patent components of scores are marked unavailable and their weight is shared out. That is
  intended, not a fault.

## Steps

```powershell
cews reset-demo --yes            # remove the demo data (live and demo data are never mixed)
cews db-init
cews sources --health            # can we reach each enabled source? (contacts them)
cews fetch --dry-run --source clinical_trials_gov    # try one source, write nothing
```

If the dry run looks sensible, fetch for real, one source at a time at first:

```powershell
cews fetch --source clinical_trials_gov
cews fetch --source europe_pmc
cews fetch --source pubmed
cews refresh --skip-fetch        # normalize, score, forecast, insights, export
cews jobs
```

When that works, start the schedule:

```powershell
cews run-scheduler --now
```

## What to expect

- **The first fetch is slow.** It reaches back `DEFAULT_LOOKBACK_DAYS` (default 1,095 days, three
  years) across every therapeutic area in `THERAPEUTIC_AREAS`, in 30-day windows (7-day for
  PubMed), at the polite rate limits. Expect **tens of minutes to a few hours**, not seconds.
  For a first trial, set `DEFAULT_LOOKBACK_DAYS=365` in `.env`; you can widen it later with
  `cews fetch --full`.
- **It resumes.** Progress is checkpointed, so an interrupted fetch continues where it stopped.
  `cews normalize` also saves its work in batches of 500 records, so if it is interrupted (or the
  computer restarts) the records it finished stay finished and the next run carries on from
  there. Run it with `-v` to see progress; it logs a line per batch.
- **`cews normalize` is the slow step on big data.** It matches every organization name against
  the ones already known, and that grows with the number of *distinct* organizations, not just
  records. Tens of thousands of records naming thousands of distinct organizations can take
  several minutes; steady CPU use in Task Manager means it is working, not stuck.
- **Later cycles are fast.** They collect only the last `INCREMENTAL_LOOKBACK_DAYS` (7) and
  anything new.
- **Results will look thin at first.** Velocity needs at least 6 months of history and the
  backtest about 15 months plus its horizon. Expect low confidence and "does not qualify" until
  enough real history accumulates; that is the system being honest.
- **A source can fail.** Each is isolated, and a source that keeps failing pauses itself for
  `CIRCUIT_BREAKER_COOLDOWN_MINUTES`. `cews sources` shows which. If one returns `403` for no
  clear reason, see the note in `docs/data_sources.md`.

## If something looks wrong

`cews check-env` (configuration), `cews sources --health` (connectivity), `cews jobs` (what the
last cycles did), `cews db-status` (record counts). Tell me what you see; the adapters have only
ever been tested against recorded responses, so the first live run is where surprises will be.
