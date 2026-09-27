"""Tests for the dashboard charts.

Charts are images, so these check the things that can be wrong without being visible: that a
figure is produced at all, that empty input says why rather than drawing an empty frame, and
that the promises made in the docstrings hold.
"""

from __future__ import annotations

from datetime import date

import matplotlib
import pytest
from matplotlib.figure import Figure

from cews.dashboard import charts

pytestmark = pytest.mark.unit

MONTHS = [date(2026, month, 1) for month in range(1, 7)]
VALUES = [4.0, 6.0, 5.0, 9.0, 12.0, 14.0]


def test_charts_never_try_to_open_a_window() -> None:
    """A dashboard renders on a server with no display attached."""
    assert matplotlib.get_backend().lower() == "agg"


def test_activity_chart_is_drawn() -> None:
    figure = charts.activity_with_forecast(MONTHS, VALUES, title="A topic")
    assert isinstance(figure, Figure)
    axes = figure.axes[0]
    assert axes.get_ylim()[0] == 0  # counts start at zero, never a cropped axis
    assert axes.get_title(loc="left") == "A topic"


def test_a_forecast_is_drawn_with_its_range() -> None:
    """A single predicted line invites more confidence than monthly counts deserve."""
    figure = charts.activity_with_forecast(
        MONTHS,
        VALUES,
        forecast_months=[date(2026, 7, 1), date(2026, 8, 1)],
        predicted=[15.0, 16.0],
        lower=[10.0, 9.0],
        upper=[20.0, 23.0],
        model="holt",
    )
    axes = figure.axes[0]
    labels = [text.get_text() for text in axes.get_legend().get_texts()]
    assert "recorded" in labels
    assert any("forecast" in label for label in labels)
    assert "likely range" in labels
    assert axes.collections  # the shaded interval


def test_an_empty_series_explains_itself() -> None:
    figure = charts.activity_with_forecast([], [])
    texts = [text.get_text() for text in figure.axes[0].texts]
    assert any("No activity" in text for text in texts)


@pytest.mark.parametrize(
    "call",
    [
        lambda: charts.ranked_bars([], []),
        lambda: charts.score_against_confidence([], [], []),
        lambda: charts.opportunity_quadrant([], [], []),
        lambda: charts.records_by_type({}),
        lambda: charts.component_contributions({}),
    ],
)
def test_every_chart_handles_having_nothing_to_show(call: object) -> None:
    figure = call()  # type: ignore[operator]
    assert isinstance(figure, Figure)
    assert figure.axes[0].texts  # a message, not an empty frame


def test_ranked_bars_put_the_highest_at_the_top() -> None:
    figure = charts.ranked_bars(["A", "B", "C"], [90.0, 50.0, 20.0], highlight=[True, False, False])
    labels = [text.get_text() for text in figure.axes[0].get_yticklabels()]
    assert labels[-1] == "A"  # matplotlib draws the y axis upward


def test_qualified_entries_are_coloured_differently() -> None:
    """Scoring highly and being a finding are different things, and must look different."""
    figure = charts.ranked_bars(["A", "B"], [90.0, 80.0], highlight=[True, False])
    colours = {bar.get_facecolor() for bar in figure.axes[0].patches}
    assert len(colours) == 2


def test_the_confidence_threshold_is_marked() -> None:
    figure = charts.score_against_confidence(["A"], [90.0], [30.0], confidence_threshold=60.0)
    axes = figure.axes[0]
    assert any("not enough evidence" in text.get_text() for text in axes.texts)
    assert axes.get_xlim()[1] >= 100


def test_low_confidence_points_are_marked_in_red() -> None:
    figure = charts.score_against_confidence(["A", "B"], [90.0, 90.0], [20.0, 95.0])
    colours = figure.axes[0].collections[0].get_facecolor()
    assert len(colours) == 2 and not (colours[0] == colours[1]).all()


def test_the_opportunity_corner_is_labelled() -> None:
    figure = charts.opportunity_quadrant(["A"], [80.0], [10.0], qualified=[True])
    assert any("few competitors" in text.get_text() for text in figure.axes[0].texts)


def test_component_contributions_add_up_to_the_score() -> None:
    components = {
        "velocity": {"available": True, "weight": 0.3, "contribution": 24.0},
        "momentum": {"available": True, "weight": 0.25, "contribution": 15.0},
        "patent_growth": {"available": False, "weight": 0.0, "contribution": 0.0},
    }
    figure = charts.component_contributions(components)
    assert "total 39.0" in figure.axes[0].get_xlabel()
    assert len(figure.axes[0].patches) == 2  # the unavailable component is not drawn


def test_records_by_type_is_ordered_by_size() -> None:
    figure = charts.records_by_type({"patent": 10, "publication": 90, "funding": 50})
    labels = [text.get_text() for text in figure.axes[0].get_xticklabels()]
    assert labels[0] == "publication"
