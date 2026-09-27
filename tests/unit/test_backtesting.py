"""Unit tests for rolling-origin backtesting."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from cews.forecasting.backtesting import BacktestResult, rolling_origin_backtest
from cews.forecasting.baseline import ForecastResult, build_naive_forecast

pytestmark = pytest.mark.unit

COUNTING = [float(value) for value in range(1, 25)]  # 1..24: leakage would be obvious


def perfect(history: Sequence[float], horizon: int) -> ForecastResult:
    """A model that knows the series continues counting upward."""
    start = history[-1] + 1
    predictions = tuple(start + step for step in range(horizon))
    return ForecastResult(
        model="perfect",
        predictions=predictions,
        lower=predictions,
        upper=predictions,
        training_months=len(history),
    )


def broken(history: Sequence[float], horizon: int) -> ForecastResult | None:
    return None


def explodes(history: Sequence[float], horizon: int) -> ForecastResult:
    raise RuntimeError("this model cannot fit anything")


# --------------------------------------------------------------------------------------
# No future data leaks backwards
# --------------------------------------------------------------------------------------
def test_a_model_only_ever_sees_the_past() -> None:
    """The whole point of a backtest: no fold may contain data from after its origin."""
    seen: list[list[float]] = []

    def spy(history: Sequence[float], horizon: int) -> ForecastResult:
        seen.append(list(history))
        return build_naive_forecast(history, horizon)

    result = rolling_origin_backtest(COUNTING, spy, model="spy", horizon=3, minimum_training=12)
    assert len(seen) == len(result.folds)
    for history, fold in zip(seen, result.folds, strict=True):
        assert len(history) == fold.origin
        assert history == COUNTING[: fold.origin]
        assert not set(fold.actual) & set(history)


def test_each_replay_moves_one_month_forward() -> None:
    result = rolling_origin_backtest(COUNTING, build_naive_forecast, model="naive", horizon=3)
    assert [fold.origin for fold in result.folds] == list(range(12, 22))
    assert result.folds[0].actual == (13.0, 14.0, 15.0)
    assert result.folds[-1].actual == (22.0, 23.0, 24.0)


def test_the_step_can_be_widened() -> None:
    result = rolling_origin_backtest(
        COUNTING, build_naive_forecast, model="naive", horizon=3, step=4
    )
    assert [fold.origin for fold in result.folds] == [12, 16, 20]


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------
def test_a_model_that_knows_the_future_scores_zero_error() -> None:
    result = rolling_origin_backtest(COUNTING, perfect, model="perfect", horizon=3)
    assert result.usable and result.metrics is not None
    assert result.metrics.mae == 0.0 and result.metrics.mase == 0.0


def test_a_worse_model_scores_worse() -> None:
    good = rolling_origin_backtest(COUNTING, perfect, model="perfect", horizon=3)
    naive = rolling_origin_backtest(COUNTING, build_naive_forecast, model="naive", horizon=3)
    assert good.metrics is not None and naive.metrics is not None
    assert good.metrics.mae < naive.metrics.mae


def test_every_fold_is_kept_for_inspection() -> None:
    result = rolling_origin_backtest(COUNTING, build_naive_forecast, model="naive", horizon=2)
    fold = result.folds[0]
    assert len(fold.actual) == len(fold.predicted) == 2
    assert fold.training_months == 12
    assert set(fold.as_dict()) == {"origin", "training_months", "actual", "predicted"}


# --------------------------------------------------------------------------------------
# When it cannot be done
# --------------------------------------------------------------------------------------
def test_too_little_history_is_reported_not_guessed() -> None:
    result = rolling_origin_backtest(
        COUNTING[:10], build_naive_forecast, model="naive", horizon=3, minimum_training=12
    )
    assert not result.usable and result.metrics is None
    assert "too short" in result.warnings[0]


def test_one_replay_is_not_enough_to_judge_a_model() -> None:
    result = rolling_origin_backtest(
        COUNTING[:13], build_naive_forecast, model="naive", horizon=3, minimum_training=10
    )
    assert len(result.folds) == 1
    assert not result.usable
    assert "at least 2" in result.warnings[0]


def test_a_model_that_cannot_fit_is_counted_not_fatal() -> None:
    result = rolling_origin_backtest(COUNTING, broken, model="broken", horizon=3)
    assert result.failures == 10 and result.folds == [] and not result.usable


def test_a_model_that_raises_does_not_stop_the_backtest() -> None:
    """One model blowing up must not take the comparison down with it."""
    result = rolling_origin_backtest(COUNTING, explodes, model="explodes", horizon=3)
    assert result.failures > 0 and result.metrics is None


@pytest.mark.parametrize(
    ("horizon", "training", "step"), [(0, 12, 1), (3, 0, 1), (3, 12, 0), (-1, 12, 1)]
)
def test_invalid_settings_are_refused(horizon: int, training: int, step: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        rolling_origin_backtest(
            COUNTING,
            build_naive_forecast,
            model="naive",
            horizon=horizon,
            minimum_training=training,
            step=step,
        )


def test_the_result_summarises_itself() -> None:
    result: BacktestResult = rolling_origin_backtest(
        COUNTING, build_naive_forecast, model="naive", horizon=3
    )
    payload = result.as_dict()
    assert payload["model"] == "naive" and payload["usable"] is True
    assert payload["folds"] == 10 and payload["metrics"] is not None
