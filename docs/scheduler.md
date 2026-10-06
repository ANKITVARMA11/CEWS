# Running CEWS on a schedule

A **refresh cycle** does two things in order:

1. **Fetch** every enabled source (each isolated: one failing never stops the others).
2. **Analyse**: normalize, competitors, features, scores, forecasts, insights, and the Power BI
   export (if `GENERATE_POWERBI_EXPORTS=true`).

If every source that ran failed, the analysis is not re-run (it would only recompute old data).
If an analysis step fails, the steps after it are skipped (scoring on stale features would publish
old numbers as new); the steps before it keep their work.

## Commands

| Command | What it does |
| --- | --- |
| `cews refresh` | One cycle now. `--dry-run` fetches without writing and skips the analysis. `--skip-fetch` re-runs only the analysis; `--skip-analysis` only fetches. `--source ID` limits the fetch. `--no-export` skips the Power BI step. |
| `cews run-scheduler` | A cycle every `FETCH_INTERVAL_MINUTES` (default 120), until you press Ctrl+C. `--now` runs the first one immediately; otherwise it waits one interval (or set `RUN_FETCH_ON_STARTUP=true`). |
| `cews jobs` | Is a cycle running? When did one last succeed? Recent runs. Warns when the last success is over twice the interval old. |

## Settings

| Setting | Effect |
| --- | --- |
| `ENABLE_SCHEDULER` | `false` makes `run-scheduler` refuse to start (it says so). `cews refresh` still works. |
| `FETCH_INTERVAL_MINUTES` | Time between cycles. Under 30 minutes gives a warning: public APIs rate-limit, and research trends do not move that fast. |
| `RUN_FETCH_ON_STARTUP` | Run a cycle as soon as the scheduler starts. |
| `GENERATE_POWERBI_EXPORTS` | Include the export as the last step of every cycle. |
| `TIMEZONE` | Time zone the scheduler works in. |

## Safety

- **No overlap, three ways.** One instance of the job at a time; a slow cycle that misses its slot
  becomes one catch-up run, not a queue; and an operating-system **file lock** (in
  `data/locks/`, beside the database) stops a scheduled cycle overlapping a manual `cews refresh`
  typed in another terminal.
- **The lock is not in the database, on purpose.** SQLite allows one writer at a time, and a long
  step (a big normalize or forecast) can hold it for minutes. A lock kept in the database had to
  be renewed by writing to it, which failed with `database is locked` in the middle of a cycle.
  A file lock needs no expiry and no heartbeat: it is held exactly as long as the process is
  alive, and Windows releases it the instant the process dies, however it dies.
- **Long steps save as they go.** `normalize` commits every 500 records, so a cycle cut short (or a killed process) keeps what it finished and the next cycle continues from there.
- **A crashed cycle is cleaned up.** A run left marked "running" by a dead process is closed as
  interrupted at the next cycle.
- **A bad cycle never stops the scheduler.** Errors are logged and the next slot still happens.
- **Failed cycles retry sooner**, after 5, 10, 20... minutes (never later than the interval).
- **Live data cannot be mixed into a demo database.** Run `cews reset-demo --yes` first.

## Stopping

- **Ctrl+C once:** no new cycle starts, and a cycle already running is allowed to finish. Its lock
  releases itself when it does, so a manual `cews refresh` typed meanwhile is refused rather than
  overlapping it.
- **Ctrl+C twice:** abort now. The process exits immediately (exit code 130) and the operating
  system releases the lock as it goes. The half-finished run is recorded as "interrupted" by the next cycle, and every step that
  had already finished keeps its work.

## Keeping it running (Windows)

`cews run-scheduler` is an ordinary foreground process: it runs while its terminal is open. For a
long-lived setup, either leave a terminal open, or skip the scheduler and have **Windows Task
Scheduler** run `cews refresh` every two hours; the lock makes overlapping starts harmless.
Either way, `cews jobs` tells you whether it is working.
