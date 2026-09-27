# Organizations and topics

Collected records name organizations in whatever way the source happened to write them, and say
nothing about which research topic they belong to. `cews normalize` turns that raw text into
linked entities:

```
python -m cews.cli normalize          # resolve organizations, assign topics, mark duplicates
python -m cews.cli review             # decisions CEWS would not make on its own
```

Running it again processes only new records. `--reprocess` redoes everything without creating
duplicate organizations or links.

## Matching organization names

`Zentavia Pharma`, `Zentavia Pharma, Inc.`, `ZENTAVIA PHARMA INC` and
`Zentavia Pharmaceuticals Ltd` are one company. `Orvexa Bio` and `Orvexa Labs` are two. The
resolver works in layers, stopping at the first that answers:

1. **the name as written**, lower-cased;
2. **normalized** - accents folded, punctuation dropped, legal suffixes (`Inc`, `Ltd`, `GmbH`,
   `PLC`, ...) removed;
3. **expanded** - abbreviations spelled out, so `Pharma` meets `Pharmaceuticals`, `Tx` meets
   `Therapeutics` and `Labs` meets `Laboratories`;
4. **without spaces**, so `OrvexaBio` meets `Orvexa Bio`;
5. **prefix**, for short names (`Zentavia`), used only when exactly one company fits;
6. **similarity**, which catches `Univ. of Harrowgate` against `Harrowgate University`.

Two guards keep it honest. Similarity may not merge names that differ by a word marking a
different business (`Biosciences` vs `Laboratories`, `Oncology` vs `Pharmaceuticals`), and a
company is never merged into a university, hospital or agency. Anything below the automatic
threshold creates a **separate** organization plus a review item, so an uncertain name is never
folded into the wrong company.

Every spelling seen is stored as an alias with the method and confidence that produced it, and
the fullest spelling becomes the display name, so a dashboard shows `Nexoria Genetics Corp.`
rather than `Nexoria Gen.`.

### What needs a human

`cews review` lists them. Nothing here is decided automatically:

| Case | Example | What CEWS does |
|---|---|---|
| Ambiguous short name | `Zentavia` when both `Zentavia Pharmaceuticals` and `Zentavia Oncology` exist | keeps it separate and asks |
| Near-miss spelling | `Meridian Theraputics` | keeps it separate and reports the similarity |
| Possible parent or subsidiary | `Zentavia Oncology Ltd` and `Zentavia Pharmaceuticals Ltd` | suggests the relationship; never sets it |
| Similar but possibly unrelated | `Orvexa Bio` and `Orvexa Labs` | suggests a decision; never merges |

Thresholds are `AI_ORG_MATCH_AUTO_THRESHOLD` (default 0.92) and
`AI_ORG_MATCH_REVIEW_THRESHOLD` (0.80). They govern the similarity step today and the optional
embedding matcher in Phase 9.

### How strongly a record implicates an organization

| Relationship | From | Confidence |
|---|---|---|
| sponsor | trial lead sponsor | 0.95 |
| assignee | patent assignee | 0.95 |
| recipient | grant recipient | 0.9 |
| announcer | company feed | 0.9 |
| partner | trial collaborator, announcement partner | 0.7 |
| affiliation | publication author affiliation | 0.5 |

An author's affiliation says someone at that organization co-wrote a paper, **not** that the
organization sponsored the work, so it counts for half a record in ranking. Affiliation strings
are messy free text (`"Fixture Oncology, Inc., Boston, MA, USA."`), so addresses, departments
and e-mail addresses are stripped and the result is stored at low confidence.

## Assigning topics

Topics come from `config/topic_taxonomy.yaml`. A record's title, abstract and structured fields
(conditions, keywords, MeSH terms, patent classifications) are matched against each topic's name
and synonyms. Matching tolerates separators and plurals, so `CAR-T`, `CAR T` and `CAR-T cells`
all match, and never matches inside another word.

Confidence reflects where the match was found - a title or a MeSH term counts for more than an
abstract - and rises slightly when several distinct terms match. A record can hold several
topics. Each therapeutic area is stored as a topic too (key `area:<id>`), so a record about a
monitored area with no narrower topic still counts towards it.

Records matching nothing are counted, not forced into a topic. That count is the gap the
optional AI topic discovery closes in Phase 9.

## Duplicate works

The same paper can arrive from more than one source. Matching is by DOI, then PubMed ID, then an
exact title within the same year - and the title rule applies only **across** sources, because
two records from one source with the same title are usually different works. Duplicates keep
their rows as evidence of where the work was found, but point at a canonical record so activity
is never counted twice.

## Choosing competitors

> Running against demo data: the names shipped in `.env` (Pfizer, Roche, ...) are real
> companies, and the demo data is synthetic, so they appear as monitored with zero activity.
> That is the rule working, not a fault. For a cleaner demo set `COMPETITOR_MODE=AUTO`, or clear
> `COMPETITOR_INCLUDE`.

`cews competitors` ranks organizations over the last 12 months:

    100 x (0.30 trials + 0.25 patents + 0.20 publications + 0.15 funding + 0.10 announcements)

Each component is a percentile rank among the candidates for that source type, so counts from
different sources are comparable. Source types with no data anywhere share their weight out
rather than counting as zero. Universities, hospitals and government bodies are excluded unless
`--include-all-types` is given, and an organization needs `MIN_COMPETITOR_EVIDENCE_COUNT`
records before it is ranked.

The `.env` settings then decide the final list: `MANUAL` uses only your names, `HYBRID` (default)
puts your names first and fills the rest by rank up to `TOP_COMPETITORS`, and `AUTO` fills every
slot. `COMPETITOR_EXCLUDE` always wins. A configured competitor with no records yet is still
monitored, with a score of 0, so its absence is visible rather than silent.

`cews competitors --json` prints the full breakdown: every component's raw count, normalized
value, weight and contribution, plus example record ids for the evidence behind each ranking.

## What is stored

Discovery writes only which organizations are monitored (the `manually_included` and
`discovered_automatically` flags). The ranking itself is recomputed on each run from the records
in the window, so it always reflects the current data. Stored, dated score rows arrive with the
scoring engine in Phase 7. Use `cews competitors --no-persist` to preview without changing the
flags, and `--json` for the full breakdown of every component and its evidence.
