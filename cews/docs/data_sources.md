# Data sources

Status: skeleton. Sections are filled in as adapters are built (Phases 3-4).

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
