# Data model

A star schema: small dimension tables describing *what* (organizations, topics, sources, dates)
and fact tables holding *measurements* (activity counts, scores, forecasts, insights...).
Conventions in every file: dates `YYYY-MM-DD`, timestamps ISO 8601 UTC, flags `1`/`0`, a
missing value is an empty cell, decimal separator `.`, UTF-8. Every table always has its header.
Column types are also listed, machine-readably, in `refresh_metadata.json`.

Every fact table carries `is_synthetic` (1 = invented demo data).

## Relationships

```
dim_date[date] 1 ──► * fact_activity[period]           dim_organization[organization_id] 1 ──► * fact_activity
                  ├─► * fact_scores[score_date]                                              ├─► * fact_scores
                  ├─► * fact_forecasts[target_period]  (active)                              ├─► * fact_forecasts
                  ├─► * fact_forecasts[forecast_date]  (inactive: USERELATIONSHIP)           ├─► * fact_anomalies
                  ├─► * fact_anomalies[anomaly_date]                                         └─► * fact_insights
                  ├─► * fact_insights[insight_date]    dim_topic[topic_id] 1 ──► * (the same five fact tables)
                  └─► * fact_evaluations[evaluation_date]
fact_scores[score_id] 1 ──► * fact_score_components[score_id]
fact_insights[insight_id] 1 ──► * fact_evidence[insight_id]
dim_source[source_name] 1 ──► * fact_ingestion_runs[source_name]      and  * fact_evidence[source_name]
```

All relationships are many-to-one with a single filter direction from the "one" side. Mark
`dim_date` as the date table (**Table tools > Mark as date table**, column `date`).

`organization_id` and `topic_id` are **each blank on the rows they do not apply to** (a score is
about a topic *or* an organization, never both), so both can be related without a bridge table.
Blank means "not about a specific one", never zero.

`fact_ingestion_runs` has timestamps rather than dates; analyse it on its own.

### Read this before summing `fact_activity`

`fact_activity` stores **the same activity at several grains**, so adding everything up counts it
several times. Rows differ by `period_type` (`month`, `quarter`, `year`) and by which of
`organization_id` / `topic_id` are filled:

| organization_id | topic_id | Meaning |
| --- | --- | --- |
| blank | blank | Overall total |
| blank | filled | One topic, all organizations |
| filled | blank | One organization, all topics |
| filled | filled | One organization within one topic |

Always filter to one `period_type` and one grain. The measures in `dax_measures.md` do this.

## Dimensions

### dim_date
One row per calendar day, from the first to the last date in the data, in whole months.

| Column | Type | Meaning |
| --- | --- | --- |
| `date` | date | The day. Relationship key. |
| `date_key` | int | `YYYYMMDD`. |
| `year` | int | |
| `quarter` | int | 1 to 4. |
| `quarter_label` | text | For example `2026 Q3`. |
| `month_number` | int | 1 to 12. |
| `month_name` | text | English month name. Sort by `month_number`. |
| `year_month` | text | `YYYY-MM`. |
| `month_start` | date | First day of the month. |
| `day_of_month` | int | |
| `is_month_start` | flag | 1 on the first of the month. |

### dim_organization
| Column | Type | Meaning |
| --- | --- | --- |
| `organization_id` | int | Key. |
| `organization_name` | text | Canonical name after merging spelling variants. |
| `organization_type` | text | |
| `country` | text | |
| `website` | text | |
| `parent_organization_id` | int | A recorded parent, if any. CEWS never links parents itself. |
| `is_monitored` | flag | 1 if included or discovered, and not manually excluded. |
| `is_manually_included` | flag | |
| `is_manually_excluded` | flag | |
| `is_synthetic` | flag | |

### dim_topic
Includes therapeutic areas, which are stored as topics with `topic_type = therapeutic_area`.

| Column | Type | Meaning |
| --- | --- | --- |
| `topic_id` | int | Key. |
| `topic_key` | text | Stable taxonomy key. |
| `topic_name` | text | |
| `topic_type` | text | For example technology, modality, therapeutic_area. |
| `parent_topic_id` | int | |
| `therapeutic_area_key` | text | |
| `therapeutic_area` | text | |
| `parent_therapeutic_area` | text | |
| `status` | text | active, pending_review or rejected. |
| `is_active` | flag | Only active topics are scored. |
| `is_ai_candidate` | flag | 1 for a topic found by AI discovery. |

### dim_source
| Column | Type | Meaning |
| --- | --- | --- |
| `source_name` | text | Key. |
| `record_types` | text | Kinds of record it supplied, `; `-separated. |
| `records` | int | |
| `first_published` | date | |
| `last_published` | date | |
| `is_synthetic` | flag | |

## Facts

### fact_activity
Monthly, quarterly and yearly activity. See the warning above.

| Column | Type | Meaning |
| --- | --- | --- |
| `period` | date | First day of the period. |
| `period_type` | text | month, quarter or year. |
| `organization_id` | int | Blank = all organizations. |
| `topic_id` | int | Blank = all topics. |
| `source_type` | text | patent, publication, clinical_trial, funding or announcement. |
| `activity_count` | int | Records counted. |
| `weighted_activity` | float | Counted with each link's confidence as its weight. |
| `unique_record_count` | int | |
| `is_synthetic` | flag | |

### fact_scores
One row per stored score, all dates kept so you can chart history.
`score_type` is one of `trend`, `opportunity`, `innovation`, `threat` (shown as **monitoring
priority**) or `confidence`. `confidence` rows are the confidence scores themselves; filter by
`score_type` before averaging.

| Column | Type | Meaning |
| --- | --- | --- |
| `score_id` | int | Key. |
| `score_date` | date | The month scored. |
| `entity_type` | text | topic or competitor. |
| `organization_id` | int | |
| `topic_id` | int | |
| `context_key` | text | Blank for an overall score; a therapeutic area for an area-level one. |
| `score_type` | text | See above. |
| `score_value` | float | 0 to 100. |
| `confidence_score` | float | 0 to 100. How far to trust the score. |
| `category` | text | The label for the score band, for example Emerging. |
| `is_qualified` | flag | 1 only if it passed every rule to count as a finding. |
| `sample_size` | float | Weighted records behind it (not always whole). |
| `explanation` | text | The reasoning in plain language. |
| `scoring_version` | text | |
| `is_synthetic` | flag | |

### fact_score_components
What each score was made of: one row per component per score.

| Column | Type | Meaning |
| --- | --- | --- |
| `score_id` | int | Relates to `fact_scores`. |
| `component` | text | For example velocity, momentum, patent_growth. |
| `raw_value` | float | Before normalizing. |
| `normalized_value` | float | 0 to 100 among peers. |
| `weight` | float | Weight actually used (shares out when a component has no data). |
| `points` | float | Points contributed to the score. |
| `is_available` | flag | 0 if there was no data for it. |

### fact_forecasts
| Column | Type | Meaning |
| --- | --- | --- |
| `forecast_id` | int | Key. |
| `forecast_date` | date | When the forecast was made (last complete month). |
| `target_period` | date | The month forecast. |
| `entity_type` | text | |
| `organization_id` | int | |
| `topic_id` | int | |
| `source_type` | text | |
| `predicted_value` | float | |
| `lower_bound` | float | About 95% range, lower. |
| `upper_bound` | float | About 95% range, upper. |
| `model_name` | text | The model chosen by backtest. |
| `backtest_metric_name` | text | mase. |
| `backtest_metric` | float | The chosen model's error on unseen history. |
| `training_months` | int | |
| `is_synthetic` | flag | |

### fact_anomalies
Unusual months. Not part of the original table list; added because they are a core dashboard item.

| Column | Type | Meaning |
| --- | --- | --- |
| `anomaly_id` | int | Key. |
| `anomaly_date` | date | |
| `entity_type` | text | |
| `organization_id` | int | |
| `topic_id` | int | |
| `metric` | text | |
| `observed_value` | float | |
| `expected_lower` | float | |
| `expected_upper` | float | |
| `deviation` | float | |
| `method` | text | |
| `anomaly_class` | text | one_time_spike, persistent_momentum, seasonal_pattern, collection_gap, drop or emerging_trend. |
| `confidence` | float | |
| `explanation` | text | |
| `is_synthetic` | flag | |

### fact_insights
| Column | Type | Meaning |
| --- | --- | --- |
| `insight_id` | int | Key. |
| `insight_date` | date | |
| `insight_type` | text | emerging_trend, competitor_movement, new_market_entry, patent_surge or opportunity. |
| `severity` | text | watch or high. |
| `entity_type` | text | |
| `organization_id` | int | |
| `topic_id` | int | |
| `title` | text | |
| `observed_fact` | text | What the records show. |
| `interpretation` | text | What it may mean. A judgement. |
| `recommended_review` | text | A next step for a person. |
| `confidence_score` | float | |
| `status` | text | |
| `evidence_count` | int | Records behind it (never 0). |
| `is_synthetic` | flag | |

### fact_evidence
The records behind each insight, one row per link.

| Column | Type | Meaning |
| --- | --- | --- |
| `evidence_id` | int | Key. |
| `insight_id` | int | Relates to `fact_insights`. |
| `source_record_id` | int | |
| `source_name` | text | |
| `record_type` | text | |
| `source_identifier` | text | The record's id at its source. |
| `title` | text | Outside text. A leading `=`, `+`, `-` or `@` is prefixed with `'`. |
| `published_date` | date | |
| `url` | text | |
| `is_synthetic` | flag | |

### fact_ingestion_runs
| Column | Type | Meaning |
| --- | --- | --- |
| `run_id` | int | Key. |
| `job_id` | text | |
| `source_name` | text | |
| `start_time` | datetime | UTC. |
| `end_time` | datetime | UTC. |
| `duration_seconds` | float | |
| `status` | text | |
| `records_requested` | int | |
| `records_received` | int | |
| `records_inserted` | int | |
| `records_updated` | int | |
| `records_skipped` | int | |
| `duplicate_count` | int | |
| `error_count` | int | |
| `warning_count` | int | |
| `collection_mode` | text | |
| `error_summary` | text | |
| `is_synthetic` | flag | |

### fact_evaluations
Evaluation results in long form: one row per metric.
`metric_path` is dotted (for example `mean_precision_at_k` or `passed`); lists of structures are
left out.

| Column | Type | Meaning |
| --- | --- | --- |
| `evaluation_id` | text | One evaluation section of one run. |
| `evaluation_type` | text | data_quality, backtest, robustness, alerts, ai_ablation... |
| `evaluation_date` | date | |
| `period_start` | date | Backtest only. |
| `period_end` | date | Backtest only. |
| `metric_path` | text | |
| `value_number` | float | The value, when numeric. |
| `value_text` | text | The value, when not numeric (booleans are `true` or `false`). |
| `is_synthetic` | flag | |

Algorithm results, backtest performance and expert validation are **different kinds of
evidence**: filter `evaluation_type` and keep them on separate visuals.
