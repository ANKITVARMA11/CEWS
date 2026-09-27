# Data sources

Status: the four MVP adapters are built and tested (Phase 4). Patents are still blocked; see
below. All endpoints, parameters and limits were checked against the sources' own documentation
on **2026-09-22** and are recorded per source below. Re-check them if a source starts failing.

## Sources at a glance

| Source | Data | On by default | Key needed | Status |
|---|---|---|---|---|
| ClinicalTrials.gov | Clinical trials | yes | no | **Working** |
| PubMed | Publications | yes | optional (`NCBI_API_KEY`) | **Working** |
| Europe PMC | Preprints (and journals) | yes | no | **Working** |
| Company / investor-relations RSS | Announcements | yes | no | **Working once you add feed URLs** |
| NIH RePORTER | Grants and funding | no | no | Adapter not built yet |
| OpenAlex | Publications | no | **yes, required since Feb 2026** | Off until the free allowance is confirmed |
| USPTO Open Data Portal | Patents | no | yes, plus an account | **Blocked** (see below) |
| EPO Open Patent Services | Patents | no | yes (free registration) | Not built |

Without patents and funding, the patent and funding parts of the scores simply have no data.
The scoring engine renormalizes the remaining weights and lowers confidence rather than
treating the missing sources as zero.

## Per-source notes

### ClinicalTrials.gov (`clinical_trials_gov`)

`GET /api/v2/studies`, no key. Cursor paging with `pageToken`; the last page is the one without
`nextPageToken`. `pageSize` is capped at 1000 (an over-large value is silently clamped). The
collection window is applied with `filter.advanced=AREA[LastUpdatePostDate]RANGE[from,to]`, and
only interventional studies are collected unless `options.study_type` says otherwise.

**Privacy.** Study records can contain the names, phone numbers and e-mail addresses of
investigators and site contacts. CEWS requests an explicit list of institutional fields that
excludes them, and strips `centralContacts`, `overallOfficials` and per-location `contacts` from
anything it stores. Only each location's country is kept. Do not widen the field list without
re-reading this note.

There is no published rate limit, so the registry uses a conservative 0.5 requests/second.

**Their firewall rejects httpx.** ClinicalTrials.gov fingerprints the TLS handshake and answers
`403 Forbidden` to httpx while accepting the identical request from Python's standard library
(confirmed on 2026-09-23, and reported by several other projects). The registry entry therefore
sets `options.http_transport: stdlib`, which routes just this source through
`cews.ingestion.stdlib_transport`. Rate limiting, retries, host allow-lists and statistics are
unchanged. If another source starts returning 403 for no clear reason, try the same option.

### PubMed (`pubmed`)

Two requests per page: `esearch.fcgi` (JSON) for the PMIDs added in the window (`datetype=edat`),
then `efetch.fcgi` (XML) for their citations. The XML is parsed with `defusedxml`, which refuses
entity-expansion and external-reference attacks.

* Rate limit: 3 requests/second without a key, 10 with `NCBI_API_KEY`. The registry stays below
  the unauthenticated limit; setting a key does not change it automatically.
* **A search can only reach its first 10,000 records.** When a date slice matches more, CEWS
  collects the first 10,000 and records a warning naming the slice. Shorten `window_slice_days`
  (7 by default) or narrow the query to capture everything.
* `NCBI_EMAIL` is sent so NCBI can contact you about usage; `tool=cews` identifies the client.
* Articles whose only publication date is a year are counted under the date PubMed indexed them,
  so they do not all pile up in January.
* Abstracts are publisher-copyrighted. They are stored for internal analysis by default; set
  `options.store_abstracts: false` if your organization's policy requires it.

### Europe PMC (`europe_pmc`)

`GET /search` with `resultType=core`, `format=json` and `cursorMark` paging (`*` first, then
`nextCursorMark`); `pageSize` is capped at 1000 and must stay constant for a whole walk. The
window is applied with `FIRST_PDATE:[from TO to]`.

**Overlap with PubMed.** Europe PMC indexes all of PubMed, so collecting both would count the
same article twice. While PubMed is enabled, this adapter collects **preprints only**
(`SRC:PPR`, which covers bioRxiv and medRxiv); if PubMed is switched off it collects everything.
Override with `options.sources`, for example `[PPR, MED]`.

Every response is HTTP 200, even for a malformed query, and an unknown source code silently
returns nothing, so responses and codes are validated in the adapter.

### Company RSS / Atom feeds (`generic_rss`)

No feeds are built in. Add official company or investor-relations feed URLs to
`generic_rss.feeds` in `config/source_registry.yaml`, and optionally name each company under
`options.feed_organizations`. Only listed feed hosts can be contacted.

```yaml
  - id: generic_rss
    feeds:
      - https://www.example-pharma.com/news/rss.xml
    options:
      feed_organizations:
        https://www.example-pharma.com/news/rss.xml: Example Pharma
```

* **robots.txt is honoured** for every feed host (RFC 9309): a missing robots.txt allows
  fetching, an unreachable one means the feed is skipped for that run, and a matching `Disallow`
  skips the feed.
* One bad feed never blocks the others: it is recorded as an error and the run ends PARTIAL. Only
  if every feed fails does the source count as failed.
* Entries are kept when their date falls inside the window; undated entries are ignored with a
  warning. Feed titles and summaries are reduced to plain text.
* Announcement type and partner companies are **not** guessed here; they are filled in later by
  topic assignment (Phase 5) or the optional AI extraction layer.

### Patents - still blocked (checked 2026-09-22)

The legacy PatentsView PatentSearch API was retired when PatentsView moved to the USPTO Open Data
Portal on 2026-03-20. USPTO says it plans to reintroduce those API functions "in updated forms"
but gives **no estimated date**. Since 2026-06-18 the portal also requires a USPTO.gov account
with multi-factor authentication, and from 2026-08-18 extra profile fields, with an ODP API key;
old PatentsView keys do not work. The APIs that do exist cover individual patent file wrappers,
not the assignee/classification search CEWS needs. Bulk dataset downloads are available through
the portal.

Plan: keep patents disabled. When the analytics phases are running, revisit in this order:
(1) re-check whether an ODP search API has launched; (2) otherwise build an adapter that ingests
a bulk dataset file the user has downloaded manually. Nothing here should be scraped.

### OpenAlex - off by default

OpenAlex replaced its "polite pool" with API keys in February 2026: `mailto` is ignored and every
user needs a free key. A secondary source reports usage-based pricing with a small daily free
allowance. Confirm the free allowance against OpenAlex's own pricing page before enabling it,
because this project must stay free. Europe PMC covers preprints in the meantime.

## Not yet verified

NIH RePORTER's endpoints and limits have not been re-checked; its adapter is not built.

## How collection works

Each source is one adapter class plugged into a shared framework (`src/cews/ingestion/`).
Configuration lives in `config/source_registry.yaml` (rate limit, page size, page cap, date-slice
length, refresh mode) and the `ENABLE_*` flags in `.env`. No code change is needed to tune them.

1. **Plan.** Each run reads the source's checkpoint. The first run collects the full lookback
   (`DEFAULT_LOOKBACK_DAYS`). Later runs start at the last watermark minus
   `INCREMENTAL_LOOKBACK_DAYS`, so late updates are caught; upserts make the overlap harmless.
   Sources with `refresh.mode: full` (patents, funding) are checked every cycle but only
   collected when their refresh window is due.
2. **Fetch.** The window is split into date slices and fetched page by page. Every request goes
   through a per-source rate limiter, retries timeouts, HTTP 429 and 5xx with exponential
   backoff (honouring `Retry-After`), and may only contact the hosts listed for that source.
3. **Store.** Each page is validated, de-duplicated and upserted in its own short transaction.
   One bad record is skipped and counted; it never stops the run.
4. **Checkpoint.** The watermark moves forward only over slices that finished completely. A
   failure, or reaching `max_pages_per_run`, leaves an unfinished checkpoint that the next run
   resumes, so no data is skipped.
5. **Protect.** After `CIRCUIT_BREAKER_FAILURE_THRESHOLD` failed runs in a row, a source is paused
   for `CIRCUIT_BREAKER_COOLDOWN_MINUTES`. One failing source never stops the others: sources run
   in parallel (`MAX_CONCURRENT_SOURCE_JOBS`) and each failure is isolated.
6. **Audit.** Every run writes an `ingestion_runs` row per source (counters, errors, checkpoint,
   rate-limit details) and a `job_runs` row for the whole fetch.

Commands:

```
python -m cews.cli sources            # each source: enabled, adapter ready, last success, state
python -m cews.cli sources --health   # also contact each enabled source
python scripts/fetch_all.py           # one manual fetch of all enabled sources
python scripts/fetch_all.py --dry-run # fetch and parse, write nothing
python scripts/fetch_all.py --source pubmed --full
```

## Live data needs its own database

Live data and demo data are never mixed: fetching into a database that holds synthetic demo data
is refused before any request is made. Keep a separate env file for live collection:

```
copy .env .env.live                  # Windows; use cp on macOS/Linux
# in .env.live: SQLITE_PATH=./data/cews_live.db
#               DEFAULT_LOOKBACK_DAYS=30   (start small; 1095 is a large first collection)
python scripts/initialize_database.py --env-file .env.live
python -m cews.cli sources --health --env-file .env.live
python scripts/fetch_all.py --env-file .env.live --dry-run
python scripts/fetch_all.py --env-file .env.live
```

A first collection is deliberately capped (`max_pages_per_run`): a run that hits the cap stops
cleanly and the next run resumes from its checkpoint, so the history fills in over several runs.

## Adding a source

Subclass `SourceAdapter`, set `source_name` and `source_type`, implement `build_query`,
`parse_response` and `normalize_record`, decorate the class with `@register_adapter`, add a
registry entry, and add a case to `tests/contract/test_source_adapter_contract.py` (a test fails
until you do). `tests/support_adapters.py` holds a complete reference adapter.

## Access risks checked on 2026-09-20

**Patents (PatentsView).** PatentsView migrated to the USPTO Open Data Portal (ODP,
https://data.uspto.gov) starting 2026-03-20. USPTO's transition guide
(https://data.uspto.gov/support/transition-guide/patentsview) says temporary interruptions
are expected for the PatentSearch API and that API functions will be reintroduced in updated
forms; bulk downloads are available on ODP. Some third-party tools report the legacy API as
shut down. Plan: do not build a live patents adapter on the old PatentsView endpoint. Start
with a fixture adapter, then target ODP bulk downloads, and re-check API status first.

**Publications (OpenAlex).** OpenAlex's deprecations page
(https://developers.openalex.org/guides/deprecations) states that the polite pool was
replaced by API keys in February 2026: `mailto` is ignored and all users need a free API key.
A secondary source also reports usage-based pricing with a small daily free allowance; verify
this against OpenAlex's official pricing page before enabling the adapter, because the project
must stay free. Plan: OpenAlex is optional and off by default; Europe PMC is the third
no-key publication source in the MVP.

## Not yet verified

ClinicalTrials.gov, PubMed, Europe PMC, NIH RePORTER and RSS endpoints, terms, and rate limits
are taken from memory of public documentation and must be re-checked against the official
documentation when each adapter is built.
