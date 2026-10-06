# Suggested report layout

Mirrors the Streamlit dashboard, so the two tell the same story. Build it on the measures in
`dax_measures.md` and apply `powerbi_theme.json`.

## On every page

- The `Data Warning` measure in a card at the top. Nobody should present these numbers without
  knowing whether they are invented.
- A **score is always shown with its confidence**, beside it or in the tooltip.
- Red means "needs attention", never "large". Do not use colour as the only signal.
- Wording: competitors are **monitoring priorities** ("based on observed activity, not evidence of
  a legal, commercial or scientific threat"); opportunities are **prioritisation signals for
  review**, not recommendations.

## Pages

| Page | Contents |
| --- | --- |
| Overview | Cards: Emerging Trends, Insights, Records Collected, Ingestion Success Rate. Bar of Trend Score by topic with a Trend Confidence marker. Records by month. |
| Trends | Ranked topics with score, confidence and `category`. Scatter of Trend Score against Trend Confidence (upper left = a high score on thin evidence). Drill-through to a topic page. |
| Topic detail | Records by month with the forecast and its band; `fact_score_components[points]` by `component`; unusual months; Score Explanation; evidence table. |
| Competitors | Monitoring Priority and Innovation Score, each with confidence. Records by organization. The careful-wording note. |
| Opportunities | Trend Score against competition (from `fact_score_components`, component `low_competition`); only `is_qualified = 1` highlighted. |
| Insights | Table of insights (title, severity, confidence, observed fact, interpretation, recommended review) with the evidence records for the selected one. |
| Data and evaluation | Source health, failed runs, freshness. Three separate visuals: algorithm results, backtest results, expert validation. |

## Interaction

- Slicers: date range (on `dim_date`), therapeutic area, topic, organization, score type.
- Sync the date and topic slicers across pages.
- Drill-through from any topic or competitor to its detail page.
