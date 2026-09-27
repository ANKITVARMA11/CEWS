# Source response fixtures

Test fixtures for the source adapters. Each file mirrors the **structure** of a real API
response as documented (and, where noted in `docs/data_sources.md`, verified against live
responses) on 2026-09-22. **All identifiers, names, organizations and text are invented.**
Identifiers use ranges no real record uses today (NCT09..., PMID 99..., PPR99...), and
organizations are fictional, so fixtures can never be mistaken for real data.

| Folder | Mirrors |
|---|---|
| `clinical_trials_gov/` | `GET /api/v2/studies` pages (with `fields=`), `/stats/size` |
| `pubmed/` | `esearch.fcgi` (JSON), `efetch.fcgi` (PubmedArticleSet XML), `einfo.fcgi` |
| `europe_pmc/` | `GET /search?resultType=core&format=json` pages |
| `generic_rss/` | an RSS 2.0 feed, an Atom feed, a robots.txt |

`clinical_trials_gov/studies_page_1.json` deliberately contains personal contact fields
(`centralContacts`, `overallOfficials`, location `contacts`) with placeholder values, so tests
can prove the adapter strips them before storing anything.
