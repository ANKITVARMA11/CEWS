# Scoring methodology

Status: complete. Activity features (Phase 6), the scoring foundation and confidence score (7a),
topic scores - trend and opportunity (7b) - and competitor scores - innovation and threat with
capped modifiers (7c).

Run the feature pass with:

```
python -m cews.cli features                 # count activity, compute and store features
python -m cews.cli features --as-of 2026-08-31 --json
python -m cews.cli features --rebuild       # drop stored periods that no longer have records
```

## Counting activity

Records are counted into whole months per topic, per competitor, per competitor-within-topic and
overall, for each source type. Quarters and years are summed from the months, so the periods
always agree. **The month in progress is excluded**, because a part-finished month looks like a
collapse in activity.

Two numbers are stored per bucket:

- `activity_count` - how many records there were, which is what people recognise;
- `weighted_activity` - the sum of link confidence, so weaker evidence counts for less. A
  sponsored trial counts fully; an author's affiliation counts half, because a paper by someone
  at a company is not proof the company ran the work. **Scoring uses the weighted figure.**

Counting happens in the database, and re-running it rewrites the same rows rather than adding
more.

## The features

Every topic and monitored competitor gets a 12-month series per source type, and from it:

| Feature | What it answers | How |
|---|---|---|
| Growth | Did activity rise between two periods? | `ln((current + 1) / (previous + 1))` |
| Momentum | Is it heating up right now? | The same ratio, last 3 months against the 3 before |
| Velocity | Which way has it gone all year? | Slope of a line fitted to `log1p` monthly counts |
| Consistency | Sustained, or one big month? | Share of month-to-month changes that were increases |
| Recent surge | Any sharp short-term jump? | The same ratio over consecutive quarters |
| Competition density | How crowded is this topic? | Effective number of competitors (inverse HHI), saturating |
| Source agreement | Do independent sources agree? | Share of sources with data that are growing |
| Sample confidence | Is there enough evidence? | `1 - exp(-N / 25)` |

### Why growth is a log ratio

Raw percentages are unusable on small counts: 1 record to 3 is "+200%" and means nothing. The
smoothed log ratio is symmetric (a halving is exactly the negative of a doubling), stays finite
when a period has no records at all, and is far less excitable about small numbers. The familiar
percentage is still reported, for display only.

### Why velocity is refused sometimes

Fitting a line through three points produces a confident-looking number with nothing behind it.
Velocity returns nothing below `velocity_min_observations` months (6 by default) rather than
guessing. R-squared is reported alongside the slope: a single spike gives a slope near zero and
an R-squared near zero, which is how a spike is told apart from a trend.

### Why a one-player field is not "crowded"

Density uses the *effective* number of competitors (the inverse of the concentration index), not
the raw count. Ten equal players count as ten; ten players where one holds most of the activity
count as barely more than one.

## Normalization

Raw features from different sources cannot be added together, so each value is ranked inside a
**cohort**: the same feature, the same entity kind, the same date. Topics are ranked against
topics, competitors against competitors.

Four methods are available (`NORMALIZATION_METHOD`):

- **percentile** (default) - rank within the cohort. Immune to outliers, since only order counts.
- **winsorized_minmax** - clip the extremes at the configured quantiles, then scale.
- **robust_zscore** - centre on the median, scale by the median absolute deviation.
- **minmax** - plain linear scaling. Included for comparison; one outlier flattens everything else.

Awkward cohorts are handled explicitly rather than crashing or inventing a rank: a cohort where
every value is equal, a cohort of one, an empty cohort, and non-finite values all score the
neutral **50**. Negative growth is ranked like any other value.

Entities with no activity at all in the window are left out of the ranking entirely: including an
empty topic would shift everyone else's percentile.

## What is stored

Each value is written to `feature_values` with its raw value, its normalized 0-100 position, the
method, the cohort and cohort size, and the sample size behind it. That is what the
explainability view reads, and what Phase 7 combines into scores.

## Settings

In `config/scoring_weights.yaml` under `features`: `alpha`, `momentum_recent_months`,
`momentum_previous_months`, `velocity_window_months`, `velocity_min_observations`,
`sample_saturation_k`, `freshness_half_life_days`. In `.env`: `NORMALIZATION_METHOD`,
`NORMALIZATION_WINSOR_LOWER` / `_UPPER`, `MIN_TOPIC_SAMPLE_SIZE`.

## Limitations

- Features describe what the collected data shows. A source that is disabled, rate-limited or
  blocked leaves a real gap; agreement and confidence report that, but cannot fill it.
- Monthly counts on small topics are noisy. A topic with six records can show spectacular
  momentum; sample confidence is what stops that being read as a trend.
- Demo data is synthetic. Feature values computed from it prove the pipeline works, not that the
  scoring is valid.

## Scores

```
python -m cews.cli score                    # score the stored features and save the results
python -m cews.cli score --low-confidence   # only the results that should not be acted on
python -m cews.cli score --json             # the full breakdown, including the explanation
python -m cews.cli score --no-store         # compute without saving
```

Every score is a weighted sum of components that are already on a 0-100 scale, and is stored
with the working shown: each component's raw value, normalized value, weight and contribution,
which inputs were missing, the rules it passed or failed, the features it was built from, and the
scoring version. That is what lets a dashboard answer "why is this 78?".

### Missing inputs are never zero

If a component has no data, its weight is shared across the components that remain. With patents
unavailable, a topic is scored on the sources that exist rather than being pushed down as though
patent activity were zero. Which components were missing is recorded, and confidence falls
because the score rests on fewer legs.

### Confidence

Every score carries one, and the two are always shown together:

```
Confidence = 100 x (0.35 sample + 0.25 source agreement + 0.20 completeness
                    + 0.10 freshness + 0.10 model stability)
```

- **sample** - evidence volume, saturating, so 10 records count for much more than 2 while 300
  are barely better than 200;
- **source agreement** - the share of sources with data that point the same way;
- **completeness** - the share of expected months that have data;
- **freshness** - how recently the sources were collected, halving every 30 days;
- **model stability** - how far the value moved since the previous run.

On a first run there is nothing to compare against, so stability is **unknown rather than zero**:
its weight is shared across the other inputs. Otherwise every first run would look untrustworthy.

A result below the configured minimum (60) fails the `usable_confidence` rule and is marked as
something not to act on. The demo's six-record topic scores about 52 and is flagged, which is the
whole point: it has the highest momentum in the dataset.

### Configuration is checked, not trusted

`config/scoring_weights.yaml` is validated when it loads. A weight set that does not sum to 1 is
refused (a score could never reach 100), and so is a misspelled or missing component name (that
component would silently vanish from every score). Overlapping category bands and impossible
thresholds are refused too. `cews score` exits with a configuration error and names the problem.

### Score history

Scores are keyed by date, entity, context, type and scoring version. Re-running a date rewrites
its scores; running under **different weights** (a new `scoring_version`) stores a second set
alongside the first, so a change in methodology does not erase what was reported before.

## Trend score

```
Trend = 0.30 velocity + 0.25 momentum + 0.20 patent growth
        + 0.15 funding growth + 0.10 consistency
```

Each part is the topic's normalized position among the other topics, so a patent count and a
publication count can sit in the same sum. A component with no data has its weight shared out
and the gap recorded; "no patents anywhere for this topic" is not the same as "no patent growth".

### A high score is not a finding

Calling a topic an **emerging trend** requires all five rules to pass:

| Rule | Requirement |
|---|---|
| `score_threshold` | trend score at or above 60 |
| `confidence_threshold` | confidence at or above 60 |
| `minimum_evidence` | at least `MIN_TOPIC_SAMPLE_SIZE` records |
| `independent_sources` | at least 2 source types showing growth |
| `not_a_single_spike` | growth is sustained, not one isolated month |

The rules travel with the score, so a number can never be quoted without the reasons it did or
did not qualify. In the demo data the six-record topic scores **81.5** - the highest trend score
in the dataset - and fails four of the five rules. It is shown with its reasons rather than
hidden, because hiding it would make the ranking look better than the evidence is.

Which topics qualify shifts with sample size, since percentile ranking is relative. What does not
shift is what qualification means.

## Opportunity score

```
Opportunity = 100 x (0.60 trend + 0.25 (1 - competition density) + 0.15 source agreement)
```

High when a topic is moving, few organizations are in it, and independent sources agree. The
same evidence rules apply, plus one more: **competition must actually have been measured**. A
topic with no competition data is marked incomplete rather than scored as though the field were
empty, which would be the most flattering possible reading of missing data.

This is a prioritization signal for expert review. It is never described as an investment,
commercial or scientific recommendation, and the stored note says so on every result.

## Reading the output

```
python -m cews.cli score                      # trends and opportunities, with the rules
python -m cews.cli score --type trend         # one kind of score
python -m cews.cli score --low-confidence     # only results that should not be acted on
python -m cews.cli score --json               # full breakdown including the explanation
```

The `why not` column names the rules a result failed. An empty column means it qualified.

## Innovation score

```
Innovation = 0.40 patents + 0.30 trials + 0.20 publications + 0.10 funding
```

This measures **levels, not growth**: how much a competitor is producing compared with the
others. Patents carry the most weight because a patent is a costly, deliberate claim on an idea;
a publication is cheaper and often academic.

A competitor with no patents scores low on that component. That is a finding about the company,
not missing data, so it is scored as a genuine zero. A component is only treated as unavailable
when **nobody** has data for it, which means the source itself is off or empty. "They file no
patents" and "we cannot see patents" deserve opposite treatment.

Each competitor is also given a rank, and the change in rank since the previous run, because
positions are read before numbers: moving from 6th to 2nd says more than a score of 71. Rank is
stored at zero weight, so it is recorded for the dashboard without entering the arithmetic.

## Threat score

```
Threat = 0.50 trial growth + 0.30 patent growth + 0.20 publication growth
```

Growth, not size: a large company doing what it always does is not news. Trials carry the most
weight because starting one is the most expensive and most committing act on the list.

A score is produced for each competitor overall **and for each therapeutic area they work in**,
so "who is moving into this area" can be answered directly. An area is only scored once enough
records sit behind it; an area is not judged on a handful of records.

### Capped modifiers

Some events matter but cannot show up in a growth rate. Each adds a few points, is capped on its
own, and the total is capped again (15 points), so no combination can dominate the measurement:

| Modifier | Fires when |
|---|---|
| `new_therapeutic_area_entry` | first significant activity in a research area after a long silence |
| `phase_progression` | trials reached a later phase than before |
| `large_enrollment_increase` | average enrollment rose by half again or more |
| `monitored_topic_patent_activity` | patents landed in topics being watched |
| `multiple_supporting_sources` | three or more independent sources are growing |

Modifiers are stored separately from the base score and never folded into it invisibly. Each one
records the fact that triggered it, so the points trace back to something observable.

**A new entry needs a long silence.** A company is only reported as entering an area when it has
several records there now and *none at all* over a much longer stretch before. A one-record blip
in a quiet topic is not a company entering a field, and the rule lives in one place
(`detect_new_therapeutic_area_entry`) so there is a single definition of what an entry is.

### Wording

This score says a competitor is worth watching. It is **not** evidence of a legal, commercial or
scientific threat, and both the labels and the stored note say so: routine activity, worth
watching, elevated competitive activity, high monitoring priority, potential strategic threat
requiring review. The same evidence rules apply as elsewhere: a high score on thin evidence is
shown with the rules it failed, because a small company can show the fastest growth simply by
starting from almost nothing.

## Forecasting: the measures and the baselines (Phase 8a)

Forecasts answer "if nothing changes, where does this go next?" and each one is stored with an
interval, because a single predicted number invites more confidence than monthly counts deserve.

### How a forecast is judged

| Measure | What it says |
|---|---|
| MAE | average miss, in records |
| RMSE | same, but large misses count for more |
| **MASE** | the miss divided by what "next month looks like this month" would have scored |
| SMAPE | a percentage that stays finite when the truth is zero |
| Directional accuracy | how often the direction of change was right |

**MASE is the one that decides which model gets used.** Below 1 means the model beat the naive
baseline; above 1 means it did not, and saying so plainly is more useful than a confident number.
It is also comparable between a busy topic and a quiet one, and it survives months with no
activity.

**MAPE is deliberately absent.** Activity counts are frequently zero, and dividing by zero makes
a percentage either infinite or quietly misleading.

### The baselines

- **naive** - next month looks like this month.
- **seasonal naive** - next month looks like the same month last year. Refused with less than a
  full cycle of history, because a yearly pattern would then be invented rather than observed.

They exist to be beaten. A clever model that cannot beat "next month looks like this month" is
not adding anything, and on short or noisy histories that is often the honest answer. They are
also the fallback when there is too little history for anything else.

### Intervals

Bounds come from how wrong the same model was on its own history, widen with the square root of
how far ahead the period is, and are clipped at zero, since counts cannot be negative.

They also never narrow below the noise a count of that size carries anyway: monthly counts behave
roughly like a Poisson process, where the spread is about the square root of the level. Without
that floor a perfectly smooth history would produce a zero-width interval, claiming a certainty
that counting twenty things a month never gives.

## Choosing a forecasting model (Phase 8b)

Three more models join the baselines:

- **Holt** follows the level and the trend, so a climbing topic keeps climbing instead of
  flattening the way the naive baseline does.
- **Holt-Winters** adds a yearly seasonal term. It needs **two full cycles**; with one it fits
  the noise of a single year and calls it a season.
- **ARIMA** can follow patterns the others cannot, and is the easiest to over-fit, so the order
  search is deliberately small and it demands the longest history.

Any of them may report itself unavailable - too little history, a failed fit, or statsmodels not
installed - and selection simply carries on with what remains.

### The model is chosen by replay, not by fit

Every candidate is replayed over the history: at each origin it sees only the months before that
point, forecasts the next few, and is scored against what actually happened. The origin then
moves forward and it repeats.

**No future data leaks backwards.** Each fit receives a prefix of the series and nothing else,
which is what separates a backtest from fitting a curve to the whole history and admiring it. A
test asserts this directly by recording exactly what each fit was shown.

The winner is the lowest MASE, so a model must beat "next month looks like this month" to be
used. **Ties go to the simplest model**: one that is no better but harder to explain and easier
to overfit is worse.

When a history barely changes from month to month, MASE cannot be computed at all - it divides
by the naive error, which is then zero. Models are compared by average miss instead, and the
stored reason says so rather than reporting that nothing could be measured.

With less than 15 months there is no forecast at all. A forecast built on six months would look
identical on a dashboard to one built on six years, and nothing would say otherwise.

Every candidate's backtest is stored alongside the winner, so "why this model?" is answerable
from the record.

## Anomaly detection (Phase 8c)

Finding an unusual month is the easy half; saying what kind of unusual it is matters more,
because "patents tripled last month" can be any of these:

| Kind | What it means |
|---|---|
| `one_time_spike` | one busy month, back to normal afterwards (a patent family published at once) |
| `persistent_momentum` | the level stepped up and stayed up: a real change |
| `emerging_trend` | a rise that is still climbing; whether it lasts is not yet known |
| `seasonal_pattern` | it happens every year at this time |
| `collection_gap` | every source went quiet at once, which says more about our pipeline than the field |
| `drop` | activity fell well below its usual level |

Calling all six "an anomaly" would put noise in front of leadership, so each is labelled and the
label travels with the record.

### How unusual is measured

The default test is the **robust z-score**, `0.6745 (value - median) / MAD`. It uses the median
and the median absolute deviation rather than the mean and standard deviation, because one
enormous month drags a mean far enough to hide itself. Each month is compared against the twelve
before it, not the whole history, so a topic that grew steadily for two years does not flag every
recent month for beating the distant past.

Three fallbacks matter, in order: the median absolute deviation, then the ordinary spread, then
the count noise (about the square root of the level). Without that last step a perfectly steady
history would have no spread at all, and a collapse from thirty records to one could never be
flagged - the most obvious anomaly there is would be the one that went unnoticed.

### Small numbers are flagged quietly

Three records against a near-empty baseline is a large ratio and a small fact. Such months are
still reported, because in a quiet field they can be the earliest signal there is, but their
confidence is held down by how few records are involved. In the demo data the six-record topic is
flagged at under 40, while the designed patent spike is flagged at 100.

Seasonality is refused on small numbers for the same reason: two small numbers in the same month
of different years is not a season.

## Running forecasts and anomaly detection (Phase 8d)

```
python -m cews.cli forecast                       # forecast 6 months and flag unusual months
python -m cews.cli forecast --horizon 3 --top 20
python -m cews.cli forecast --no-anomalies        # forecasts only
python -m cews.cli forecast --json                # every candidate model and its backtest
```

Forecasts start from the **last complete month**: a part-finished month looks like a collapse
and would drag every forecast down. Entities with no activity at all are skipped rather than
forecast as a flat zero.

Each stored forecast carries its interval, the model that produced it, that model's backtest
score, and how many months it was trained on. Re-running a date rewrites those rows instead of
adding more.

The `fit` column is the chosen model's error when replayed on history it had not seen. It is
compared against the naive model's own replay rather than against a fixed number, because MASE
scales by a one-step-ahead forecast: at a three-month horizon a score above 1 is ordinary and
says nothing about whether the model beat the baseline.
