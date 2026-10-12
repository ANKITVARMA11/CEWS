"""Tests for the dashboard charts.

These check the same promises the old matplotlib charts made - a figure is produced at all,
empty input says why rather than drawing an empty frame, a forecast shows its range, the highest
bar lands on top - against the Plotly figure objects that replaced them.
"""

from __future__ import annotations

from datetime import date

import plotly.graph_objects as go
import pytest

from cews.dashboard import charts

pytestmark = pytest.mark.unit

MONTHS = [date(2026, month, 1) for month in range(1, 7)]
VALUES = [4.0, 6.0, 5.0, 9.0, 12.0, 14.0]


def _annotation_texts(figure: go.Figure) -> list[str]:
    return [str(a.text) for a in figure.layout.annotations or ()]


def test_activity_chart_is_drawn() -> None:
    figure = charts.activity_with_forecast(MONTHS, VALUES, title="A topic")
    assert isinstance(figure, go.Figure)
    assert figure.layout.yaxis.rangemode == "tozero"  # counts start at zero, never a cropped axis
    assert figure.layout.title.text == "A topic"


# --------------------------------------------------------------------------------------
# Regression: `st.plotly_chart`'s default theme re-colours a figure's text to match the app's
# active light/dark theme, turning it white under a dark browser/OS theme even though the
# figure's own background stays the fixed white these charts set - every call site now passes
# `theme=None` to stop that, and every figure sets its own dark text explicitly too, as
# defense-in-depth. This checks the explicit colour is actually set on every styled figure.
# --------------------------------------------------------------------------------------
def test_every_chart_sets_an_explicit_dark_text_colour() -> None:
    figures = [
        charts.activity_with_forecast(MONTHS, VALUES, title="A topic"),
        charts.ranked_bars(["A", "B"], [1.0, 2.0], highlight=[True, False]),
        charts.score_against_confidence(["A", "B"], [1.0, 2.0], [0.5, 0.6], sizes=[5, 5]),
        charts.records_by_type({"clinical_trial": 3, "publication": 2}),
        charts.component_contributions(
            {"velocity": {"available": True, "weight": 0.3, "contribution": 24.0}}
        ),
        charts.empty_chart("nothing to show"),
    ]
    for figure in figures:
        assert figure.layout.font.color == charts.TEXT
        assert figure.layout.paper_bgcolor == charts.BACKGROUND


# --------------------------------------------------------------------------------------
# Regression: a fixed 10px margin on every side clipped axis titles ("trend score", "how
# crowded the field is") and long tick labels (long competitor names) right off the edge of the
# chart. `automargin=True` lets Plotly grow the margin to fit whatever is actually drawn there,
# instead of a hand-picked pixel count that is wrong for anything longer than the shortest label.
# --------------------------------------------------------------------------------------
def test_every_chart_lets_its_axes_grow_to_fit_their_labels() -> None:
    figures = [
        charts.activity_with_forecast(MONTHS, VALUES, title="A topic"),
        charts.ranked_bars(["A very long competitor name indeed", "B"], [1.0, 2.0]),
        charts.score_against_confidence(["A", "B"], [1.0, 2.0], [0.5, 0.6], sizes=[5, 5]),
        charts.opportunity_quadrant(["A"], [80.0], [10.0], qualified=[True]),
        charts.component_contributions(
            {"velocity": {"available": True, "weight": 0.3, "contribution": 24.0}}
        ),
        charts.records_by_type({"clinical_trial": 3, "publication": 2}),
    ]
    for figure in figures:
        assert figure.layout.xaxis.automargin is True
        assert figure.layout.yaxis.automargin is True


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
    names = [trace.name for trace in figure.data]
    assert "recorded" in names
    assert any("forecast" in name for name in names if name)
    assert "likely range" in names


def test_an_empty_series_explains_itself() -> None:
    figure = charts.activity_with_forecast([], [])
    assert any("No activity" in text for text in _annotation_texts(figure))


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
    assert isinstance(figure, go.Figure)
    assert _annotation_texts(figure)  # a message, not an empty frame


def test_ranked_bars_put_the_highest_at_the_top() -> None:
    figure = charts.ranked_bars(["A", "B", "C"], [90.0, 50.0, 20.0], highlight=[True, False, False])
    # Plotly draws the first category of a horizontal bar at the bottom, so the highest value
    # ("A") must be last in the trace's own category order to end up drawn at the top.
    assert list(figure.data[0].y)[-1] == "A"


def test_qualified_entries_are_coloured_differently() -> None:
    """Scoring highly and being a finding are different things, and must look different."""
    figure = charts.ranked_bars(["A", "B"], [90.0, 80.0], highlight=[True, False])
    colours = set(figure.data[0].marker.color)
    assert len(colours) == 2


def test_the_confidence_threshold_is_marked() -> None:
    figure = charts.score_against_confidence(["A"], [90.0], [30.0], confidence_threshold=60.0)
    assert any("not enough evidence" in text for text in _annotation_texts(figure))
    assert figure.layout.xaxis.range[1] >= 100


def test_low_confidence_points_are_marked_in_red() -> None:
    figure = charts.score_against_confidence(["A", "B"], [90.0, 90.0], [20.0, 95.0])
    colours = list(figure.data[0].marker.color)
    assert len(colours) == 2 and colours[0] != colours[1]


def test_the_opportunity_corner_is_labelled() -> None:
    figure = charts.opportunity_quadrant(["A"], [80.0], [10.0], qualified=[True])
    assert any("few competitors" in text for text in _annotation_texts(figure))


def test_component_contributions_add_up_to_the_score() -> None:
    components = {
        "velocity": {"available": True, "weight": 0.3, "contribution": 24.0},
        "momentum": {"available": True, "weight": 0.25, "contribution": 15.0},
        "patent_growth": {"available": False, "weight": 0.0, "contribution": 0.0},
    }
    figure = charts.component_contributions(components)
    assert "total 39.0" in figure.layout.xaxis.title.text
    assert len(figure.data) == 2  # the unavailable component is not drawn


def test_records_by_type_is_ordered_by_size() -> None:
    figure = charts.records_by_type({"patent": 10, "publication": 90, "funding": 50})
    assert list(figure.data[0].x)[0] == "publication"
