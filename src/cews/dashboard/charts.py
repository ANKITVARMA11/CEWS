"""Charts, drawn as interactive Plotly figures.

Everything here is Plotly (``plotly.graph_objects``), rendered with ``st.plotly_chart`` so a
person can hover a point for its exact value, zoom into a crowded cluster, or toggle a legend
entry — none of which a static image can offer. This replaced an earlier matplotlib version that
rendered every chart as a picture; the data contract (what each function takes and what it is
meant to show) is unchanged, only the figure object itself is now a ``go.Figure``.

Three habits run through the charts, unchanged from before:

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

import plotly.graph_objects as go

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
# Explicit dark text, set on every figure's layout and axes below. `st.plotly_chart` defaults to
# `theme="streamlit"`, which re-themes a figure's text to match the app's active light/dark
# theme and will turn this white again under a dark browser/OS theme even though our background
# stays the fixed white set above - that is what caused "white text on white background." Every
# `st.plotly_chart(...)` call in the dashboard now passes `theme=None` to stop that re-theming;
# setting the colour explicitly here too is defense-in-depth in case a figure is ever rendered
# somewhere that still applies a theme.
TEXT = "#1A1A1A"

BASE_HEIGHT = 360


def _style(figure: go.Figure, *, title: str = "", height: int | None = None) -> go.Figure:
    """Shared layout: a plain white background, a light grid, no chrome Plotly adds by default."""
    figure.update_layout(
        title=(
            dict(text=title, x=0.0, xanchor="left", font=dict(size=14, color=TEXT))
            if title
            else None
        ),
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        font=dict(color=TEXT),
        height=height or BASE_HEIGHT,
        # These are a floor, not the final margin: `automargin=True` below lets Plotly grow past
        # them to fit an axis title or long tick labels. A fixed, too-tight margin (10px on every
        # side, as this used to be) is what clipped axis titles like "trend score" or long
        # competitor names off the edge of the chart - fine for a short label, not for a long one.
        margin=dict(l=10, r=10, t=40 if title else 10, b=10),
        hovermode="closest",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0.0, font=dict(color=TEXT)
        ),
    )
    figure.update_xaxes(
        gridcolor=GRID,
        zeroline=False,
        color=TEXT,
        tickfont=dict(color=TEXT),
        title_font=dict(color=TEXT),
        automargin=True,
    )
    figure.update_yaxes(
        gridcolor=GRID,
        zeroline=False,
        color=TEXT,
        tickfont=dict(color=TEXT),
        title_font=dict(color=TEXT),
        automargin=True,
    )
    return figure


def empty_chart(message: str, *, height: int = BASE_HEIGHT) -> go.Figure:
    """A chart that says why it has nothing to show, rather than an empty frame."""
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=13, color=GREY),
        align="center",
    )
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    figure.update_layout(
        paper_bgcolor=BACKGROUND,
        plot_bgcolor=BACKGROUND,
        font=dict(color=TEXT),
        height=height,
        margin=dict(t=10, b=10),
    )
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
    height: int = BASE_HEIGHT,
) -> go.Figure:
    """Monthly activity, with the forecast and its interval if there is one.

    The interval is shaded rather than drawn as lines, so the eye reads a range instead of three
    competing predictions. Hovering any point shows its exact month and value.
    """
    if not months:
        return empty_chart("No activity recorded for this entity yet", height=height)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=list(months),
            y=[float(value) for value in observed],
            mode="lines+markers",
            name="recorded",
            line=dict(color=BLUE, width=2),
            marker=dict(size=5),
            hovertemplate="%{x|%b %Y}: %{y:.0f} records<extra></extra>",
        )
    )

    if forecast_months and predicted:
        # Join the forecast to the last observed point so the line is not left floating.
        bridge_x = [months[-1], *forecast_months]
        bridge_y = [float(observed[-1]), *[float(value) for value in predicted]]
        label = f"forecast ({model})" if model else "forecast"
        figure.add_trace(
            go.Scatter(
                x=bridge_x,
                y=bridge_y,
                mode="lines+markers",
                name=label,
                line=dict(color=ORANGE, width=2, dash="dash"),
                marker=dict(size=5, symbol="square"),
                hovertemplate="%{x|%b %Y}: %{y:.0f} (forecast)<extra></extra>",
            )
        )
        if lower and upper:
            figure.add_trace(
                go.Scatter(
                    x=[*forecast_months, *forecast_months[::-1]],
                    y=[*[float(v) for v in upper], *[float(v) for v in lower][::-1]],
                    fill="toself",
                    fillcolor="rgba(230, 159, 0, 0.18)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="likely range",
                    hoverinfo="skip",
                )
            )
        figure.add_vline(x=months[-1], line=dict(color=GREY, width=1, dash="dot"))

    figure.update_yaxes(title_text="records per month", rangemode="tozero")
    figure.update_xaxes(tickformat="%b %Y")
    return _style(figure, title=title, height=height)


def ranked_bars(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    highlight: Sequence[bool] = (),
    title: str = "",
    xlabel: str = "score",
    height: int | None = None,
) -> go.Figure:
    """A ranked horizontal bar chart, highest at the top.

    ``highlight`` marks the entries that passed every rule; the rest are drawn in grey, so the
    difference between "scores highly" and "is a finding" is visible at a glance.
    """
    if not labels:
        return empty_chart("Nothing to rank yet")
    marks = list(highlight) + [False] * (len(labels) - len(highlight))
    colours = [BLUE if mark else GREY for mark in marks]
    # Plotly draws the first category at the bottom of a horizontal bar chart, so the input is
    # reversed up front to put the highest value at the top, matching the old chart's layout.
    order = list(range(len(labels)))[::-1]
    figure = go.Figure(
        go.Bar(
            x=[values[index] for index in order],
            y=[labels[index] for index in order],
            orientation="h",
            marker=dict(color=[colours[index] for index in order]),
            text=[f"{values[index]:.0f}" for index in order],
            textposition="outside",
            hovertemplate="%{y}: %{x:.1f}<extra></extra>",
        )
    )
    figure.update_xaxes(title_text=xlabel, range=[0, max(105.0, max(values) * 1.15)])
    return _style(figure, title=title, height=height or max(240, int(42 * len(labels)) + 80))


def score_against_confidence(
    labels: Sequence[str],
    scores: Sequence[float],
    confidences: Sequence[float],
    *,
    sizes: Sequence[float] = (),
    confidence_threshold: float = 60.0,
    title: str = "",
    height: int = 420,
) -> go.Figure:
    """Score against confidence, which is the chart that stops a number being believed too easily.

    Anything to the left of the threshold line scores well on evidence that does not support it.
    """
    if not labels:
        return empty_chart("No scores to plot yet", height=height)
    bubble = list(sizes) or [40.0] * len(labels)
    largest = max(bubble) or 1.0
    diameters = [12.0 + 34.0 * (value / largest) ** 0.5 for value in bubble]
    colours = [BLUE if value >= confidence_threshold else RED for value in confidences]
    figure = go.Figure(
        go.Scatter(
            x=list(confidences),
            y=list(scores),
            mode="markers+text",
            text=[label[:26] for label in labels],
            textposition="top center",
            textfont=dict(size=9, color=GREY),
            marker=dict(
                size=diameters, color=colours, opacity=0.8, line=dict(color="white", width=1)
            ),
            customdata=list(zip(labels, bubble, strict=True)),
            hovertemplate="%{customdata[0]}<br>score %{y:.1f}, confidence %{x:.1f}"
            "<br>%{customdata[1]:.0f} records<extra></extra>",
        )
    )
    figure.add_vline(x=confidence_threshold, line=dict(color=RED, width=1.5, dash="dash"))
    figure.add_annotation(
        x=confidence_threshold - 1,
        y=102,
        text="not enough evidence to act on",
        showarrow=False,
        font=dict(size=11, color=RED),
        xanchor="right",
    )
    figure.update_xaxes(title_text="confidence", range=[0, 105])
    figure.update_yaxes(title_text="score", range=[0, 108])
    return _style(figure, title=title, height=height)


def opportunity_quadrant(
    labels: Sequence[str],
    trend_scores: Sequence[float],
    competition: Sequence[float],
    *,
    sizes: Sequence[float] = (),
    qualified: Sequence[bool] = (),
    title: str = "",
    height: int = 460,
) -> go.Figure:
    """Growth against how crowded the field is, with the open-and-growing corner marked.

    Bubble size is the weight of evidence, so a tempting position built on very little is
    visibly small.
    """
    if not labels:
        return empty_chart("No topics to place yet", height=height)
    bubble = list(sizes) or [40.0] * len(labels)
    largest = max(bubble) or 1.0
    diameters = [12.0 + 36.0 * (value / largest) ** 0.5 for value in bubble]
    marks = list(qualified) + [False] * (len(labels) - len(qualified))
    colours = [GREEN if mark else GREY for mark in marks]
    figure = go.Figure()
    figure.add_shape(
        type="rect",
        x0=0,
        x1=40,
        y0=60,
        y1=105,
        fillcolor=GREEN,
        opacity=0.06,
        line=dict(width=0),
        layer="below",
    )
    figure.add_trace(
        go.Scatter(
            x=list(competition),
            y=list(trend_scores),
            mode="markers+text",
            text=[label[:24] for label in labels],
            textposition="top center",
            textfont=dict(size=9, color=GREY),
            marker=dict(
                size=diameters, color=colours, opacity=0.85, line=dict(color="white", width=1)
            ),
            customdata=list(zip(labels, bubble, strict=True)),
            hovertemplate="%{customdata[0]}<br>trend %{y:.1f}, crowding %{x:.1f}"
            "<br>%{customdata[1]:.0f} records<extra></extra>",
        )
    )
    figure.add_hline(y=60, line=dict(color=GREY, width=1, dash="dot"))
    figure.add_vline(x=40, line=dict(color=GREY, width=1, dash="dot"))
    figure.add_annotation(
        x=2,
        y=101,
        text="growing, few competitors",
        showarrow=False,
        font=dict(size=11, color=GREEN),
        xanchor="left",
    )
    figure.update_xaxes(title_text="how crowded the field is", range=[0, 105])
    figure.update_yaxes(title_text="trend score", range=[0, 108])
    return _style(figure, title=title, height=height)


def component_contributions(
    components: dict[str, dict[str, float]], *, title: str = "", height: int = 220
) -> go.Figure:
    """What each part contributed to one score, so the total can be checked by eye."""
    usable = {
        name: detail
        for name, detail in components.items()
        if detail.get("available") and detail.get("weight")
    }
    if not usable:
        return empty_chart("This score has no weighted components", height=height)
    names = list(usable)
    values = [float(usable[name].get("contribution", 0.0)) for name in names]
    palette = [BLUE, ORANGE, GREEN, PURPLE, GREY]
    figure = go.Figure()
    for index, (name, value) in enumerate(zip(names, values, strict=True)):
        figure.add_trace(
            go.Bar(
                x=[value],
                y=[""],
                orientation="h",
                name=name.replace("_", " "),
                marker=dict(color=palette[index % len(palette)]),
                text=[f"{name.replace('_', ' ')} {value:.1f}"] if value > 4 else [""],
                textposition="inside",
                insidetextfont=dict(color="white", size=10),
                hovertemplate=f"{name.replace('_', ' ')}: %{{x:.1f}}<extra></extra>",
            )
        )
    total = sum(values)
    figure.update_layout(barmode="stack", showlegend=False)
    figure.update_xaxes(
        title_text=f"points contributed (total {total:.1f})", range=[0, max(100.0, total * 1.05)]
    )
    figure.update_yaxes(visible=False)
    return _style(figure, title=title, height=height)


def records_by_type(counts: dict[str, int], *, title: str = "", height: int = 260) -> go.Figure:
    """How much of each kind of record has been collected."""
    if not counts:
        return empty_chart("No records collected yet", height=height)
    names = sorted(counts, key=lambda name: counts[name], reverse=True)
    figure = go.Figure(
        go.Bar(
            x=[name.replace("_", " ") for name in names],
            y=[counts[name] for name in names],
            marker=dict(color=BLUE),
            text=[f"{counts[name]:,}" for name in names],
            textposition="outside",
            hovertemplate="%{x}: %{y:,}<extra></extra>",
        )
    )
    figure.update_yaxes(title_text="records")
    return _style(figure, title=title, height=height)
