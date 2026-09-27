"""ARIMA forecasts.

ARIMA can follow patterns the simpler models cannot, at the cost of being easy to over-fit: with
thirty monthly points it will happily model noise. CEWS therefore keeps the search small (a
handful of low orders), demands a longer history than the other models, and lets model selection
decide by backtest rather than by how well anything fits its own training data.

Like the smoothing models, a missing statsmodels or a failed fit is an ordinary outcome: the
model reports itself unavailable and selection picks something else.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from typing import Any

from cews.forecasting.baseline import ForecastResult, check_series, count_noise, interval

LOGGER = logging.getLogger(__name__)

ARIMA = "arima"
MIN_ARIMA_MONTHS = 12
# Small, deliberately conservative orders: enough for trend and short memory, no seasonality.
CANDIDATE_ORDERS: tuple[tuple[int, int, int], ...] = (
    (1, 1, 0),
    (0, 1, 1),
    (1, 1, 1),
    (2, 1, 0),
    (1, 0, 0),
)


def statsmodels_available() -> bool:
    """Whether the optional statsmodels dependency can be imported."""
    try:
        import statsmodels.tsa.arima.model  # noqa: F401
    except ImportError:  # pragma: no cover - exercised by the unavailable-path test
        return False
    return True


def _fit(
    values: Sequence[float], order: tuple[int, int, int], horizon: int
) -> tuple[Any, Any, float]:
    from statsmodels.tsa.arima.model import ARIMA as ArimaModel

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ArimaModel(list(values), order=order).fit()
        return fitted, fitted.forecast(horizon), float(fitted.aic)


def build_arima_forecast(
    series: Sequence[float],
    horizon: int,
    *,
    minimum_months: int = MIN_ARIMA_MONTHS,
    orders: Sequence[tuple[int, int, int]] = CANDIDATE_ORDERS,
) -> ForecastResult | None:
    """Fit a small set of ARIMA orders and forecast with whichever fits best.

    "Best" here is the lowest AIC, which balances fit against complexity. Whether ARIMA is worth
    using at all is decided later, by backtesting it against the baselines.

    Returns None when statsmodels is missing, the history is too short, or every order fails.

    Raises:
        ValueError: for an invalid series or a horizon below 1.
    """
    values = check_series(series, minimum=2)
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if len(values) < minimum_months or not statsmodels_available():
        return None

    best: tuple[float, tuple[int, int, int], Any, Any] | None = None
    for order in orders:
        try:
            fitted, forecast, aic = _fit(values, order, horizon)
        except Exception as exc:  # non-stationary data, singular matrices, and friends
            LOGGER.debug("ARIMA%s failed: %s", order, exc)
            continue
        if any(float(value) != float(value) for value in forecast):  # NaN
            continue
        if best is None or aic < best[0]:
            best = (aic, order, fitted, forecast)
    if best is None:
        return None

    aic, order, fitted, forecast = best
    predictions = tuple(max(0.0, float(value)) for value in forecast)
    residuals = [float(value) for value in fitted.resid]
    lower, upper = interval(predictions, residuals, minimum_spread=count_noise(values))
    return ForecastResult(
        model=ARIMA,
        predictions=predictions,
        lower=lower,
        upper=upper,
        training_months=len(values),
        parameters={"order": list(order), "aic": round(aic, 2)},
    )
