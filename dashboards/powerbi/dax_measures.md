# DAX measures

Create these in a dedicated table (**Home > Enter data**, name it `_Measures`, then delete its
placeholder column) so they are easy to find. They assume the model in `data_model.md`.

> **Not yet run in Power BI Desktop.** These were written against the export and reviewed by
> reading. If a measure errors on first use, the cause is most likely a table or column you
> named differently; the logic should carry over.

Three rules run through all of them:

1. **`fact_activity` is pinned to one period type and one grain.** Otherwise the same activity is
   counted several times (see `data_model.md`).
2. **Scores are pinned to one `score_type` and to the latest `score_date`.** `fact_scores`
   holds every score type and every date.
3. **A score is quoted with its confidence.** A high score on thin evidence is not a finding.

## Data warning banner (put this on every page)

```dax
Data Warning =
IF (
    CALCULATE ( MAX ( fact_scores[is_synthetic] ), ALL ( fact_scores ) ) = 1,
    "SYNTHETIC DEMO DATA: invented organizations and artificial activity",
    BLANK ()
)
```

## Scores

```dax
Latest Score Date =
CALCULATE ( MAX ( fact_scores[score_date] ), ALL ( fact_scores ) )

Trend Score =
CALCULATE (
    AVERAGE ( fact_scores[score_value] ),
    fact_scores[score_type] = "trend",
    fact_scores[context_key] = BLANK (),
    fact_scores[score_date] = [Latest Score Date]
)

Trend Confidence =
CALCULATE (
    AVERAGE ( fact_scores[confidence_score] ),
    fact_scores[score_type] = "trend",
    fact_scores[context_key] = BLANK (),
    fact_scores[score_date] = [Latest Score Date]
)

Opportunity Score =
CALCULATE (
    AVERAGE ( fact_scores[score_value] ),
    fact_scores[score_type] = "opportunity",
    fact_scores[score_date] = [Latest Score Date]
)

Innovation Score =
CALCULATE (
    AVERAGE ( fact_scores[score_value] ),
    fact_scores[score_type] = "innovation",
    fact_scores[score_date] = [Latest Score Date]
)

-- Shown to users as "monitoring priority", never as a threat.
Monitoring Priority =
CALCULATE (
    AVERAGE ( fact_scores[score_value] ),
    fact_scores[score_type] = "threat",
    fact_scores[context_key] = BLANK (),
    fact_scores[score_date] = [Latest Score Date]
)

Emerging Trends =
CALCULATE (
    DISTINCTCOUNT ( fact_scores[topic_id] ),
    fact_scores[score_type] = "trend",
    fact_scores[is_qualified] = 1,
    fact_scores[score_date] = [Latest Score Date]
)

Score Explanation =
CALCULATE (
    SELECTEDVALUE ( fact_scores[explanation] ),
    fact_scores[context_key] = BLANK (),
    fact_scores[score_date] = [Latest Score Date]
)
```

`Trend Score` and the others average over whatever is in filter context, so they are exact for a
single topic or competitor and an average for a group. Use `Score Explanation` in a tooltip.

To show why a score is what it is, chart `fact_score_components[points]` by `component` for the
selected topic.

## Activity

Pick the grain by what the visual slices on. `Records` chooses it for you.

```dax
Records - Overall =
CALCULATE (
    SUM ( fact_activity[activity_count] ),
    fact_activity[period_type] = "month",
    ISBLANK ( fact_activity[organization_id] ),
    ISBLANK ( fact_activity[topic_id] )
)

Records - By Topic =
CALCULATE (
    SUM ( fact_activity[activity_count] ),
    fact_activity[period_type] = "month",
    ISBLANK ( fact_activity[organization_id] ),
    NOT ISBLANK ( fact_activity[topic_id] )
)

Records - By Organization =
CALCULATE (
    SUM ( fact_activity[activity_count] ),
    fact_activity[period_type] = "month",
    NOT ISBLANK ( fact_activity[organization_id] ),
    ISBLANK ( fact_activity[topic_id] )
)

Records - Organization In Topic =
CALCULATE (
    SUM ( fact_activity[activity_count] ),
    fact_activity[period_type] = "month",
    NOT ISBLANK ( fact_activity[organization_id] ),
    NOT ISBLANK ( fact_activity[topic_id] )
)

Records =
VAR ByTopic = ISFILTERED ( dim_topic )
VAR ByOrganization = ISFILTERED ( dim_organization )
RETURN
    SWITCH (
        TRUE (),
        ByTopic && ByOrganization, [Records - Organization In Topic],
        ByTopic, [Records - By Topic],
        ByOrganization, [Records - By Organization],
        [Records - Overall]
    )

Last Data Month =
CALCULATE ( MAX ( fact_activity[period] ), ALL ( fact_activity ) )

Records Last 3 Months =
CALCULATE (
    [Records],
    DATESINPERIOD ( dim_date[date], [Last Data Month], -3, MONTH )
)

Records Previous 3 Months =
CALCULATE (
    [Records],
    DATESINPERIOD ( dim_date[date], EDATE ( [Last Data Month], -3 ), -3, MONTH )
)

Momentum % =
DIVIDE ( [Records Last 3 Months] - [Records Previous 3 Months], [Records Previous 3 Months] )
```

`Momentum %` is a simple percentage change for display. CEWS's own momentum feature is a
smoothed log ratio (so a move from 1 to 3 records is not a "200% surge"), and the score uses that.
A tiny base can make this percentage look dramatic: show `Records Last 3 Months` next to it.

## Forecasts

```dax
Latest Forecast Date =
CALCULATE ( MAX ( fact_forecasts[forecast_date] ), ALL ( fact_forecasts ) )

Forecast =
CALCULATE (
    SUM ( fact_forecasts[predicted_value] ),
    fact_forecasts[forecast_date] = [Latest Forecast Date]
)

Forecast Lower =
CALCULATE (
    SUM ( fact_forecasts[lower_bound] ),
    fact_forecasts[forecast_date] = [Latest Forecast Date]
)

Forecast Upper =
CALCULATE (
    SUM ( fact_forecasts[upper_bound] ),
    fact_forecasts[forecast_date] = [Latest Forecast Date]
)
```

These add across whatever entities are in view, so use them on a single topic or competitor.
Draw `Forecast Lower` and `Forecast Upper` as a band: a forecast without its range invites more
trust than it has earned. Use `dim_date` on `target_period` (the active relationship) for the axis.

## Insights

```dax
Insights = COUNTROWS ( fact_insights )

High Severity Insights =
CALCULATE ( COUNTROWS ( fact_insights ), fact_insights[severity] = "high" )

Average Insight Confidence = AVERAGE ( fact_insights[confidence_score] )

Insight Evidence Records = COUNTROWS ( fact_evidence )
```

## Unusual months

```dax
Unusual Months = COUNTROWS ( fact_anomalies )

One-Time Spikes =
CALCULATE ( COUNTROWS ( fact_anomalies ), fact_anomalies[anomaly_class] = "one_time_spike" )
```

## Data health

```dax
Ingestion Runs = COUNTROWS ( fact_ingestion_runs )

Failed Ingestion Runs =
CALCULATE ( COUNTROWS ( fact_ingestion_runs ), fact_ingestion_runs[status] = "failed" )

Ingestion Success Rate =
DIVIDE (
    CALCULATE ( COUNTROWS ( fact_ingestion_runs ), fact_ingestion_runs[status] = "succeeded" ),
    [Ingestion Runs]
)

Records Collected = SUM ( dim_source[records] )
```

## Evaluation

Each evaluation run adds rows, so pin to the latest run of a kind.

```dax
Latest Backtest Date =
CALCULATE (
    MAX ( fact_evaluations[evaluation_date] ),
    fact_evaluations[evaluation_type] = "backtest",
    ALL ( dim_date )
)

Backtest Precision at K =
CALCULATE (
    MAX ( fact_evaluations[value_number] ),
    fact_evaluations[evaluation_type] = "backtest",
    fact_evaluations[metric_path] = "mean_precision_at_k",
    fact_evaluations[evaluation_date] = [Latest Backtest Date],
    ALL ( dim_date )
)

Backtest Rank Correlation =
CALCULATE (
    MAX ( fact_evaluations[value_number] ),
    fact_evaluations[evaluation_type] = "backtest",
    fact_evaluations[metric_path] = "mean_spearman",
    fact_evaluations[evaluation_date] = [Latest Backtest Date],
    ALL ( dim_date )
)

Alert Precision =
CALCULATE (
    MAX ( fact_evaluations[value_number] ),
    fact_evaluations[evaluation_type] = "alerts",
    fact_evaluations[metric_path] = "precision",
    fact_evaluations[evaluation_date] = CALCULATE (
        MAX ( fact_evaluations[evaluation_date] ),
        fact_evaluations[evaluation_type] = "alerts",
        ALL ( dim_date )
    ),
    ALL ( dim_date )
)
```

Two evaluations on the same day both match the date filter, and `MAX` then picks the larger
value. Run `cews evaluate` once per day, or filter on `evaluation_id`.

Show algorithm results, backtest results and expert validation as **separate visuals**; they are
different kinds of evidence, and a blended number would hide which one is speaking.
