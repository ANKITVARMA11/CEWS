"""Charts, drawn server-side as images.

Everything here is matplotlib, rendered to a picture before it reaches the browser. No charting
library runs in the page, which keeps the whole project free of JavaScript we would have to
maintain.

Three habits run through the charts:

* **Uncertainty is drawn, not implied.** A forecast is shown with its interval, so nobody reads
  a single line as a promise.
* **Colour is never the only signal.** The palette is colour-blind safe, and anything important
  is also labelled, positioned or shaped differently.
* **Red is reserved.** It means "this needs attention", not "this is a large number", so a busy
  chart does not look like an emergency.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date

import matplotlib

matplotlib.use("Agg")  # render to an image; never open a window
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

LOGGER = logging.getLogger(__name__)

# Colour-blind safe (Okabe-Ito). Blue carries the data; orange marks the forecast; red is only
# ever used for something that needs attention.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
GREY = "#8C8C8C"
RED = "#D55E00"
PURPLE = "#CC79A7"
BACKGROUND = "#FFFFFF"
GRID = "#E6E6E6"

BASE_SIZE = (9.0, 4.0)


def _style(figure: Figure) -> None:
    for axes in figure.axes:
        axes.set_facecolor(BACKGROUND)
        axes.grid(True, color=GRID, linewidth=0.8, zorder=0)
        axes.set_axisbelow(True)
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axes.spines[side].set_color(GREY)
    figure.patch.set_facecolor(BACKGROUND)
    figure.tight_layout()


def empty_chart(message: str, *, size: tuple[float, float] = BASE_SIZE) -> Figure:
    """A chart that says why it has nothing to show, rather than an empty frame."""
    figure, axes = plt.subplots(figsize=size)
    axes.text(0.5, 0.5, message, ha="center", va="center", color=GREY, fontsize=11, wrap=True)
    axes.set_xticks([])
    axes.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        axes.spines[side].set_visible(False)
    figure.patch.set_facecolor(BACKGROUND)
    figure.tight_layout()
    return figure


def activity_with_forecast(
    months: Sequence[date],
    observed: Sequence[float],
    *,
    forecast_months: Sequence[date] = (),
    predicted: Sequence[float] = (),
    lower: Sequence[float] = (),
    upper: Sequence[float] = (),
    title: str = "",
    model: str = "",
    size: tuple[float, float] = BASE_SIZE,
) -> Figure:
    """Monthly activity, with the forecast and its interval if there is one.

    The interval is shaded rather than drawn as lines, so the eye reads a range instead of three
    competing predictions.
    """
    if not months:
        return empty_chart("No activity recorded for this entity yet", size=size)
    # Dates are turned into plot numbers explicitly; the axis formatters below turn them back
    # into month labels.
    observed_x = [float(mdates.date2num(month)) for month in months]
    figure, axes = plt.subplots(figsize=size)
    axes.plot(
        observed_x,
        [float(value) for value in observed],
        color=BLUE,
        linewidth=2.0,
        marker="o",
        markersize=3,
        label="recorded",
        zorder=3,
    )

    if forecast_months and predicted:
        # Join the forecast to the last observed point so the line is not left floating.
        forecast_x = [float(mdates.date2num(month)) for month in forecast_months]
        bridge_x = [observed_x[-1], *forecast_x]
        bridge_values = [float(observed[-1]), *[float(value) for value in predicted]]
        axes.plot(
            bridge_x,
            bridge_values,
            color=ORANGE,
            linewidth=2.0,
            linestyle="--",
            marker="s",
            markersize=3,
            label=f"forecast ({model})" if model else "forecast",
            zorder=3,
        )
        if lower and upper:
            axes.fill_between(
                forecast_x,
                [float(value) for value in lower],
                [float(value) for value in upper],
                color=ORANGE,
                alpha=0.18,
                label="likely range",
                zorder=1,
            )
        axes.axvline(observed_x[-1], color=GREY, linewidth=1.0, linestyle=":", zorder=2)

    axes.set_ylabel("records per month")
    axes.set_ylim(bottom=0)
    axes.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(months) // 8)))
    figure.autofmt_xdate(rotation=45, ha="right")
    if title:
        axes.set_title(title, loc="left", fontsize=12)
    axes.legend(frameon=False, fontsize=9, loc="upper left")
    _style(figure)
    return figure


def ranked_bars(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    highlight: Sequence[bool] = (),
    title: str = "",
    xlabel: str = "score",
    size: tuple[float, float] | None = None,
) -> Figure:
    """A ranked horizontal bar chart, highest at the top.

    ``highlight`` marks the entries that passed every rule; the rest are drawn in grey, so the
    difference between "scores highly" and "is a finding" is visible at a glance.
    """
    if not labels:
        return empty_chart("Nothing to rank yet")
    height = size or (9.0, max(2.4, 0.42 * len(labels) + 1.0))
    figure, axes = plt.subplots(figsize=height)
    order = list(range(len(labels)))[::-1]  # highest at the top
    marks = list(highlight) + [False] * (len(labels) - len(highlight))
    colours = [BLUE if marks[index] else GREY for index in order]
    axes.barh(
        [labels[index] for index in order],
        [values[index] for index in order],
        color=colours,
        zorder=3,
    )
    for position, index in enumerate(order):
        axes.text(
            values[index] + 1.0,
            position,
            f"{values[index]:.0f}",
            va="center",
            fontsize=9,
            color=GREY,
        )
    axes.set_xlabel(xlabel)
    axes.set_xlim(0, max(105.0, max(values) * 1.15))
    if title:
        axes.set_title(title, loc="left", fontsize=12)
    _style(figure)
    return figure


def score_against_confidence(
    labels: Sequence[str],
    scores: Sequence[float],
    confidences: Sequence[float],
    *,
    sizes: Sequence[float] = (),
    confidence_threshold: float = 60.0,
    title: str = "",
    size: tuple[float, float] = (9.0, 5.0),
) -> Figure:
    """Score against confidence, which is the chart that stops a number being believed too easily.

    Anything to the left of the threshold line scores well on evidence that does not support it.
    """
    if not labels:
        return empty_chart("No scores to plot yet", size=size)
    figure, axes = plt.subplots(figsize=size)
    bubble = list(sizes) or [40.0] * len(labels)
    largest = max(bubble) or 1.0
    areas = [40.0 + 260.0 * (value / largest) for value in bubble]
    colours = [BLUE if value >= confidence_threshold else RED for value in confidences]
    axes.scatter(confidences, scores, s=areas, c=colours, alpha=0.75, edgecolor="white", zorder=3)
    for label, x, y in zip(labels, confidences, scores, strict=True):
        axes.annotate(
            label[:26], (x, y), fontsize=8, xytext=(6, 4), textcoords="offset points", color=GREY
        )
    axes.axvline(
        confidence_threshold,
        color=RED,
        linewidth=1.2,
        linestyle="--",
        zorder=2,
    )
    axes.text(
        confidence_threshold - 1,
        102,
        "not enough evidence to act on",
        fontsize=9,
        color=RED,
        ha="right",
    )
    axes.set_xlabel("confidence")
    axes.set_ylabel("score")
    axes.set_xlim(0, 105)
    axes.set_ylim(0, 108)
    if title:
        axes.set_title(title, loc="left", fontsize=12)
    _style(figure)
    return figure


def opportunity_quadrant(
    labels: Sequence[str],
    trend_scores: Sequence[float],
    competition: Sequence[float],
    *,
    sizes: Sequence[float] = (),
    qualified: Sequence[bool] = (),
    title: str = "",
    size: tuple[float, float] = (9.0, 5.5),
) -> Figure:
    """Growth against how crowded the field is, with the open-and-growing corner marked.

    Bubble size is the weight of evidence, so a tempting position built on very little is
    visibly small.
    """
    if not labels:
        return empty_chart("No topics to place yet", size=size)
    figure, axes = plt.subplots(figsize=size)
    bubble = list(sizes) or [40.0] * len(labels)
    largest = max(bubble) or 1.0
    areas = [40.0 + 300.0 * (value / largest) for value in bubble]
    marks = list(qualified) + [False] * (len(labels) - len(qualified))
    colours = [GREEN if mark else GREY for mark in marks]
    axes.axhspan(60, 105, xmin=0.0, xmax=0.4, color=GREEN, alpha=0.06, zorder=0)
    axes.scatter(
        competition, trend_scores, s=areas, c=colours, alpha=0.8, edgecolor="white", zorder=3
    )
    for label, x, y in zip(labels, competition, trend_scores, strict=True):
        axes.annotate(
            label[:24], (x, y), fontsize=8, xytext=(6, 4), textcoords="offset points", color=GREY
        )
    axes.axhline(60, color=GREY, linewidth=1.0, linestyle=":", zorder=2)
    axes.axvline(40, color=GREY, linewidth=1.0, linestyle=":", zorder=2)
    axes.text(2, 101, "growing, few competitors", fontsize=9, color=GREEN)
    axes.set_xlabel("how crowded the field is")
    axes.set_ylabel("trend score")
    axes.set_xlim(0, 105)
    axes.set_ylim(0, 108)
    if title:
        axes.set_title(title, loc="left", fontsize=12)
    _style(figure)
    return figure


def component_contributions(
    components: dict[str, dict[str, float]],
    *,
    title: str = "",
    size: tuple[float, float] = (9.0, 3.2),
) -> Figure:
    """What each part contributed to one score, so the total can be checked by eye."""
    usable = {
        name: detail
        for name, detail in components.items()
        if detail.get("available") and detail.get("weight")
    }
    if not usable:
        return empty_chart("This score has no weighted components", size=size)
    names = list(usable)
    values = [float(usable[name].get("contribution", 0.0)) for name in names]
    figure, axes = plt.subplots(figsize=size)
    left = 0.0
    palette = [BLUE, ORANGE, GREEN, PURPLE, GREY]
    for index, (name, value) in enumerate(zip(names, values, strict=True)):
        axes.barh([0], [value], left=[left], color=palette[index % len(palette)], zorder=3)
        if value > 4:
            axes.text(
                left + value / 2,
                0,
                f"{name.replace('_', ' ')}\n{value:.1f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
            )
        left += value
    axes.set_yticks([])
    axes.set_xlim(0, max(100.0, left * 1.05))
    axes.set_xlabel(f"points contributed (total {left:.1f})")
    if title:
        axes.set_title(title, loc="left", fontsize=12)
    _style(figure)
    return figure


def records_by_type(
    counts: dict[str, int], *, title: str = "", size: tuple[float, float] = (9.0, 3.0)
) -> Figure:
    """How much of each kind of record has been collected."""
    if not counts:
        return empty_chart("No records collected yet", size=size)
    names = sorted(counts, key=lambda name: counts[name], reverse=True)
    figure, axes = plt.subplots(figsize=size)
    axes.bar(
        [name.replace("_", " ") for name in names],
        [counts[name] for name in names],
        color=BLUE,
        zorder=3,
    )
    for index, name in enumerate(names):
        axes.text(
            index,
            counts[name],
            f"{counts[name]:,}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=GREY,
        )
    axes.set_ylabel("records")
    if title:
        axes.set_title(title, loc="left", fontsize=12)
    _style(figure)
    return figure
