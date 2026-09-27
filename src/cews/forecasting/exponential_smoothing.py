"""Exponential smoothing forecasts: Holt's linear trend and Holt-Winters.

Holt follows the level and the trend, so a topic that has been climbing keeps climbing rather
than flattening the way the naive baseline does. Holt-Winters adds a yearly seasonal term, which
only makes sense with at least two full years of history; with less, it fits the noise of a
single cycle and calls it a season.

Both come from statsmodels. If that is not installed the models simply report themselves as
unavailable and selection falls back to the baselines, rather than the whole pass failing.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import Any

from cews.forecasting.baseline import ForecastResult, check_series, count_noise, interval

LOGGER = logging.getLogger(__name__)

HOLT = "holt_linear"
HOLT_WINTERS = "holt_winters"
MIN_HOLT_MONTHS = 6
SEASONS_REQUIRED = 2  # Holt-Winters needs two full cycles to tell a season from noise
DAMPING = True


def statsmodels_available() -> bool:
    """Whether the optional statsmodels dependency can be imported."""
    try:
        import statsmodels.tsa.holtwinters  # noqa: F401
    except ImportError:  # pragma: no cover - exercised by the unavailable-path test
        return False
    return True


def _fit(values: Sequence[float], horizon: int, seasonal_periods: int | None) -> tuple[Any, Any]:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    # Multiplicative seasonality cannot handle zero months, so everything stays additive.
    model = ExponentialSmoothing(
        list(values),
        trend="add",
        damped_trend=DAMPING,
        seasonal="add" if seasonal_periods else None,
        seasonal_periods=seasonal_periods,
        initialization_method="estimated",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # convergence chatter and log(0) on flat series
        fitted = model.fit(optimized=True)
        return fitted, fitted.forecast(horizon)


def build_exponential_smoothing_forecast(
    series: Sequence[float],
    horizon: int,
    *,
    season: int | None = None,
    minimum_months: int = MIN_HOLT_MONTHS,
) -> ForecastResult | None:
    """Fit Holt (or Holt-Winters when ``season`` is given) and forecast ahead.

    Returns None when statsmodels is missing, the history is too short, or the fit fails - all
    of which are ordinary outcomes that selection handles by choosing another model.

    Raises:
        ValueError: for an invalid series or a horizon below 1.
    """
    values = check_series(series, minimum=2)
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    needed = max(minimum_months, (season or 0) * SEASONS_REQUIRED)
    if len(values) < needed:
        return None
    if not statsmodels_available():
        return None

    try:
        fitted, forecast = _fit(values, horizon, season)
        predictions = tuple(max(0.0, float(value)) for value in forecast)
        residuals = [float(value) for value in fitted.resid]
    except Exception as exc:  # statsmodels raises many kinds on awkward data
        LOGGER.debug("exponential smoothing failed: %s", exc)
        return None
    if not all(value == value for value in predictions):  # NaN check
        return None

    lower, upper = interval(predictions, residuals, minimum_spread=count_noise(values))
    return ForecastResult(
        model=HOLT_WINTERS if season else HOLT,
        predictions=predictions,
        lower=lower,
        upper=upper,
        training_months=len(values),
        season=season,
        parameters={
            "damped_trend": DAMPING,
            "smoothing_level": round(float(fitted.params.get("smoothing_level", 0.0)), 4),
            "smoothing_trend": round(float(fitted.params.get("smoothing_trend", 0.0) or 0.0), 4),
        },
    )
