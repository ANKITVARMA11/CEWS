"""The CEWS dashboard.

Run it with:

    streamlit run dashboards/streamlit/app.py

This file is presentation only. Every number comes from ``cews.dashboard.queries``, which reads
what the pipeline already computed and stored; nothing is calculated here. Charts are drawn by
matplotlib and arrive in the browser as images, so the project stays free of JavaScript.

Three things are deliberate:

* the banner at the top always says whether the figures come from synthetic demo data or real
  collection, because a dashboard that cannot tell you that is one you cannot trust in a meeting;
* a score is never shown without its confidence and the rules it passed or failed;
* every screen can be traced back to the source records underneath it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:  # running through streamlit, not as an installed package
    sys.path.insert(0, str(ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:  # so "components" resolves explicitly,
    sys.path.insert(
        0, str(Path(__file__).resolve().parent)
    )  # rather than relying on Streamlit's own path setup

from components import chat_widget  # noqa: E402

from cews.constants import EntityType, ScoreType  # noqa: E402
from cews.dashboard import charts, queries  # noqa: E402
from cews.database.connection import (  # noqa: E402
    create_db_engine,
    create_session_factory,
    database_is_initialized,
    session_scope,
)
from cews.logging_config import configure_logging  # noqa: E402
from cews.settings import SettingsError, load_settings, resolve_env_file  # noqa: E402

CONFIDENCE_THRESHOLD = 60.0
PAGES = (
    "Overview",
    "Trends",
    "Competitors",
    "Opportunities",
    "Anomalies",
    "Evidence",
    "Data sources",
)


@st.cache_resource
def _session_factory(env_file: str | None) -> Any:
    settings = load_settings(env_file=resolve_env_file(env_file))
    # Unlike every `cews` command, which sets this up through cli.py's own _load(), nothing
    # configures logging when the dashboard is launched directly with `streamlit run` - so
    # without this call, every cews.* logger (the LLM client's request/response lines included)
    # has no handler at all and is silently dropped, regardless of level. @st.cache_resource
    # means this runs once per distinct env_file, not on every rerun.
    configure_logging(settings)
    engine = create_db_engine(settings)
    return settings, create_session_factory(engine), database_is_initialized(engine)


@st.cache_data(ttl=120)
def _read(name: str, *args: Any, **kwargs: Any) -> Any:
    """Run one query function and cache its result briefly."""
    _, factory, _ = _session_factory(st.session_state.get("env_file", ""))
    with session_scope(factory) as session:
        return getattr(queries, name)(session, *args, **kwargs)


def _origin_banner() -> None:
    origin = _read("data_origin")
    if origin.is_mixed:
        st.error(
            f"**{origin.label}.** Scores mix invented and real activity; do not present these."
        )
    elif origin.is_demo:
        st.warning(
            f"**{origin.label}.** Every organization, trial and paper below is invented for "
            "demonstration. The numbers show that the pipeline works, not what the market is doing."
        )
    elif origin.live_records:
        st.caption(origin.label)
    else:
        st.info("No records collected yet. Run `cews seed-demo` or `cews fetch` first.")


def _metric_row(items: list[tuple[str, Any, str]]) -> None:
    for column, (label, value, help_text) in zip(st.columns(len(items)), items, strict=True):
        column.metric(label, value, help=help_text)


def _score_table(rows: list[dict[str, Any]], *, show_context: bool = False) -> None:
    if not rows:
        st.info("Nothing scored yet. Run `cews features` and then `cews score`.")
        return
    table = []
    for row in rows:
        entry = {
            "Entity": row["entity"],
            "Score": round(row["score"], 1),
            "Confidence": round(row["confidence"], 1),
            "Category": row["category"],
            "Records": int(row["sample_size"]),
            "Qualifies": "yes" if row["qualified"] else "no",
            "Rules failed": ", ".join(rule.replace("_", " ") for rule in row["failed_rules"]),
        }
        if show_context:
            entry = {"Area": row["context"] or "overall", **entry}
        table.append(entry)
    st.dataframe(table, width="stretch", hide_index=True)


def page_overview() -> None:
    st.subheader("Where things stand")
    summary = _read("overview")
    _metric_row(
        [
            ("Competitors monitored", summary.competitors, "Organizations being tracked"),
            ("Topics watched", summary.topics, "Active topics in the taxonomy"),
            (
                "Records collected",
                f"{summary.records:,}",
                "Trials, papers, patents, grants, announcements",
            ),
            (
                "Emerging trends",
                summary.emerging_trends,
                "Topics that passed every rule, not just the score",
            ),
        ]
    )
    _metric_row(
        [
            (
                "High monitoring priority",
                summary.high_priority_threats,
                "Competitors scoring 80 or above",
            ),
            ("Opportunities", summary.opportunities, "Growing topics that are not yet crowded"),
            (
                "Flagged as low confidence",
                summary.low_confidence,
                "Scores that should not be acted on",
            ),
            (
                "Waiting for review",
                summary.needs_review,
                "Entity decisions a person should confirm",
            ),
        ]
    )
    if summary.score_date:
        st.caption(
            f"Scores computed for {summary.score_date}. "
            f"Last collection: {summary.last_collected:%Y-%m-%d %H:%M} UTC."
            if summary.last_collected
            else f"Scores computed for {summary.score_date}."
        )

    trends = _read("scores_for", ScoreType.TREND.value)
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Topics by trend score**")
        st.pyplot(
            charts.ranked_bars(
                [row["entity"] for row in trends[:10]],
                [row["score"] for row in trends[:10]],
                highlight=[row["qualified"] for row in trends[:10]],
                xlabel="trend score",
            ),
            width="stretch",
        )
        st.caption("Blue passed every rule. Grey scored well but did not qualify as a finding.")
    with right:
        st.markdown("**What has been collected**")
        st.pyplot(charts.records_by_type(summary.records_by_type), width="stretch")

    st.markdown("**Score against the evidence behind it**")
    st.pyplot(
        charts.score_against_confidence(
            [row["entity"] for row in trends[:12]],
            [row["score"] for row in trends[:12]],
            [row["confidence"] for row in trends[:12]],
            sizes=[row["sample_size"] for row in trends[:12]],
            confidence_threshold=CONFIDENCE_THRESHOLD,
        ),
        width="stretch",
    )
    st.caption(
        "A high score on the left of the red line rests on evidence too thin to act on. "
        "Bubble size is how many records sit behind it."
    )


def page_trends() -> None:
    st.subheader("Trends")
    trends = _read("scores_for", ScoreType.TREND.value)
    if not trends:
        st.info("Nothing scored yet. Run `cews features` and then `cews score`.")
        return
    qualified = [row["entity"] for row in trends if row["qualified"]]
    if qualified:
        st.success("Emerging trends (passed every rule): " + ", ".join(qualified))
    else:
        st.info("No topic currently clears every rule for an emerging trend.")
    _score_table(trends)

    chosen = st.selectbox("Look at one topic", [row["entity"] for row in trends])
    row = next(item for item in trends if item["entity"] == chosen)
    months, series = _read("monthly_activity", EntityType.TOPIC.value, row["entity_id"])
    forecast = _read("forecast_for", EntityType.TOPIC.value, row["entity_id"]) or {}
    st.pyplot(
        charts.activity_with_forecast(
            months,
            series,
            forecast_months=forecast.get("months", []),
            predicted=forecast.get("predicted", []),
            lower=forecast.get("lower", []),
            upper=forecast.get("upper", []),
            title=f"{chosen}: recorded activity and forecast",
            model=forecast.get("model", ""),
        ),
        width="stretch",
    )
    if forecast:
        metric = forecast.get("metric")
        st.caption(
            f"Forecast by {forecast['model']}, chosen by replaying models on history they had "
            f"not seen ({forecast['metric_name']} {metric:.2f}), trained on "
            f"{forecast['training_months']} months."
            if metric is not None
            else f"Forecast by {forecast['model']}."
        )
    st.pyplot(
        charts.component_contributions(
            row["components"], title=f"How the {row['score']:.1f} was reached"
        ),
        width="stretch",
    )
    st.markdown(f"**In words:** {row['explanation']}")


def page_competitors() -> None:
    st.subheader("Competitors")
    st.caption(
        "Monitoring priorities based on observed activity. Not evidence of a legal, commercial "
        "or scientific threat."
    )
    innovation = _read("scores_for", ScoreType.INNOVATION.value)
    threats = _read("scores_for", ScoreType.THREAT.value, with_context=False)
    tab_threat, tab_innovation, tab_area = st.tabs(
        ["Monitoring priority", "Innovation output", "By therapeutic area"]
    )
    with tab_threat:
        st.pyplot(
            charts.ranked_bars(
                [row["entity"] for row in threats],
                [row["score"] for row in threats],
                highlight=[row["qualified"] for row in threats],
                xlabel="monitoring priority",
            ),
            width="stretch",
        )
        _score_table(threats)
        for row in threats[:3]:
            modifiers = row["components"].get("modifiers", {}).get("detail", {}).get("applied", [])
            if modifiers:
                st.markdown(
                    f"**{row['entity']}** — "
                    + "; ".join(
                        f"{item['name'].replace('_', ' ')}: {item['detail']}" for item in modifiers
                    )
                )
    with tab_innovation:
        st.pyplot(
            charts.ranked_bars(
                [row["entity"] for row in innovation],
                [row["score"] for row in innovation],
                highlight=[True] * len(innovation),
                xlabel="innovation score",
            ),
            width="stretch",
        )
        _score_table(innovation)
    with tab_area:
        by_area = _read("scores_for", ScoreType.THREAT.value, with_context=True)
        if not by_area:
            st.info("No area has enough records behind it to be scored separately yet.")
        else:
            _score_table(by_area, show_context=True)


def page_opportunities() -> None:
    st.subheader("Opportunities")
    st.caption(
        "Where activity is growing and the field is not yet crowded. A prioritization signal "
        "for expert review, not an investment, commercial or scientific recommendation."
    )
    opportunities = _read("scores_for", ScoreType.OPPORTUNITY.value)
    trends = {row["entity_id"]: row for row in _read("scores_for", ScoreType.TREND.value)}
    if not opportunities:
        st.info("Nothing scored yet. Run `cews features` and then `cews score`.")
        return
    density = []
    labels = []
    trend_values = []
    sizes = []
    qualified = []
    for row in opportunities:
        component = row["components"].get("low_competition", {})
        raw = component.get("raw_value")
        if raw is None:
            continue
        labels.append(row["entity"])
        density.append(float(raw))
        trend_values.append(trends.get(row["entity_id"], {}).get("score", 0.0))
        sizes.append(float(row["sample_size"]))
        qualified.append(row["qualified"])
    st.pyplot(
        charts.opportunity_quadrant(
            labels, trend_values, density, sizes=sizes, qualified=qualified
        ),
        width="stretch",
    )
    st.caption("Green passed every rule. Bubble size is the weight of evidence behind it.")
    _score_table(opportunities)


def page_anomalies() -> None:
    st.subheader("Unusual months")
    st.caption(
        "A spike is not a trend. Each unusual month is labelled with what kind of unusual it is."
    )
    rows = _read("recent_anomalies", limit=40)
    if not rows:
        st.info("No unusual months found yet. Run `cews forecast`.")
        return
    kinds = sorted({row["kind"] for row in rows})
    chosen = st.multiselect("Kinds to show", kinds, default=kinds)
    shown = [row for row in rows if row["kind"] in chosen]
    st.dataframe(
        [
            {
                "Month": row["date"],
                "Entity": row["entity"],
                "Observed": round(row["observed"]),
                "Usual range": f"{row['expected_lower']:.0f}-{row['expected_upper']:.0f}",
                "Kind": row["kind"].replace("_", " "),
                "Confidence": round(row["confidence"]),
                "What it means": row["explanation"],
            }
            for row in shown
        ],
        width="stretch",
        hide_index=True,
    )


def page_evidence() -> None:
    st.subheader("Evidence")
    st.caption("Every score traces back to the records underneath it.")
    kind = st.radio("Look at", ["Topics", "Competitors"], horizontal=True)
    if kind == "Topics":
        rows = _read("scores_for", ScoreType.TREND.value)
        entity_type = EntityType.TOPIC.value
    else:
        rows = _read("scores_for", ScoreType.INNOVATION.value)
        entity_type = EntityType.COMPETITOR.value
    if not rows:
        st.info("Nothing scored yet.")
        return
    chosen = st.selectbox("Entity", [row["entity"] for row in rows])
    row = next(item for item in rows if item["entity"] == chosen)

    _metric_row(
        [
            ("Score", round(row["score"], 1), "The stored score"),
            ("Confidence", round(row["confidence"], 1), "How much the evidence supports it"),
            ("Records", int(row["sample_size"]), "Evidence behind the score"),
            ("Qualifies", "yes" if row["qualified"] else "no", "Whether it passed every rule"),
        ]
    )
    st.markdown(f"**In words:** {row['explanation']}")
    if row["failed_rules"]:
        st.warning(
            "Rules failed: " + ", ".join(rule.replace("_", " ") for rule in row["failed_rules"])
        )
    if row["unavailable"]:
        st.info(
            "No data for: "
            + ", ".join(name.replace("_", " ") for name in row["unavailable"])
            + ". Their weight was shared across the remaining components."
        )
    st.pyplot(charts.component_contributions(row["components"]), width="stretch")
    st.markdown("**Components**")
    st.dataframe(
        [
            {
                "Component": name.replace("_", " "),
                "Raw value": detail.get("raw_value"),
                "Normalized": detail.get("normalized"),
                "Weight": detail.get("weight"),
                "Points": detail.get("contribution"),
                "Available": "yes" if detail.get("available") else "no",
            }
            for name, detail in row["components"].items()
        ],
        width="stretch",
        hide_index=True,
    )
    st.markdown("**Source records**")
    records = _read("evidence_records", entity_type, row["entity_id"], limit=15)
    st.dataframe(
        [
            {
                "Published": item["published"],
                "Type": item["type"].replace("_", " "),
                "Identifier": item["identifier"],
                "Title": (item["title"] or "")[:110],
                "Link": item["url"] or "",
            }
            for item in records
        ],
        width="stretch",
        hide_index=True,
    )


def page_sources() -> None:
    st.subheader("Data sources")
    health = _read("source_health")
    if health:
        st.dataframe(
            [
                {
                    "Source": row["source"],
                    "Last status": row["last_status"] or "never run",
                    "Last success": row["last_success"],
                    "Consecutive failures": row["consecutive_failures"],
                    "Paused": "yes" if row["paused"] else "no",
                    "Records last run": row["records_last_run"],
                }
                for row in health
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No source has run yet. Run `cews fetch`.")
    st.markdown("**Records by source**")
    st.dataframe(
        [
            {"Source": name, "Records": count}
            for name, count in sorted(_read("record_counts_by_source").items())
        ],
        width="stretch",
        hide_index=True,
    )
    st.markdown("**Waiting for review**")
    review = _read("review_items")
    if review:
        st.dataframe(
            [
                {
                    "Kind": row["kind"].replace("_", " "),
                    "Subject": row["subject"],
                    "Raised": row["raised"],
                }
                for row in review
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.caption("Nothing is waiting for a decision.")


def main() -> None:
    """Draw the dashboard."""
    st.set_page_config(page_title="CEWS", page_icon="🧭", layout="wide")
    st.title("Competitor Early Warning System")

    with st.sidebar:
        st.markdown("### CEWS")
        env_file = st.text_input("Configuration file", value=st.session_state.get("env_file", ""))
        st.session_state["env_file"] = env_file
        page = st.radio("View", PAGES)
        if st.button("Reload data"):
            st.cache_data.clear()
        st.caption(
            "Scores, forecasts and anomalies are read as stored. Nothing on this dashboard is "
            "calculated here."
        )

    try:
        _, _, initialized = _session_factory(env_file or "")
    except SettingsError as error:
        st.error(f"Configuration problem: {error}")
        return
    if not initialized:
        st.error("The database has not been created yet. Run `python -m cews.cli db-init` first.")
        return

    _origin_banner()
    chat_widget.render(env_file)
    {
        "Overview": page_overview,
        "Trends": page_trends,
        "Competitors": page_competitors,
        "Opportunities": page_opportunities,
        "Anomalies": page_anomalies,
        "Evidence": page_evidence,
        "Data sources": page_sources,
    }[page]()


main()
