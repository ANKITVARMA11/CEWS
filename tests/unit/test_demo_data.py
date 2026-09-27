"""Unit tests for the synthetic demo data generator (patterns, labelling, determinism)."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from statistics import median

import pytest
import yaml

from cews.demo.generator import (
    DEMO_ORGS,
    DEMO_TOPICS,
    SOURCE_NAMES,
    DemoConfig,
    DemoDataset,
    DemoRecord,
    add_months,
    generate_demo_dataset,
    write_fixtures,
)
from cews.normalization.identifiers import is_valid_url
from support import TAXONOMY_FILE

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def dataset() -> DemoDataset:
    return generate_demo_dataset()


def _truth(record: DemoRecord) -> dict[str, object]:
    return record.payload["synthetic_ground_truth"]


def _month(record: DemoRecord) -> tuple[int, int]:
    return (record.published_at.year, record.published_at.month)


def _monthly(dataset: DemoDataset, topic: str, record_type: str | None = None) -> list[int]:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for record in dataset.records:
        if _truth(record)["topic"] == topic and record_type in (None, record.record_type):
            counts[_month(record)] += 1
    return [counts.get((m.year, m.month), 0) for m in dataset.months]


# --------------------------------------------------------------------------------------
# Determinism, labelling, coverage
# --------------------------------------------------------------------------------------
def test_same_config_gives_identical_data(dataset: DemoDataset) -> None:
    again = generate_demo_dataset()
    assert [r.to_jsonable() for r in again.records] == [r.to_jsonable() for r in dataset.records]


def test_different_seed_gives_different_data(dataset: DemoDataset) -> None:
    other = generate_demo_dataset(DemoConfig(seed=7))
    assert [r.title for r in other.records] != [r.title for r in dataset.records]


def test_every_record_is_labelled_synthetic(dataset: DemoDataset) -> None:
    valid_sources = set(SOURCE_NAMES.values())
    for record in dataset.records:
        assert record.source in valid_sources and record.source.startswith("synthetic_")
        assert record.source_record_id.startswith("SYN-")
        assert record.abstract.startswith("[SYNTHETIC]")
        assert record.payload["synthetic"] is True
        assert is_valid_url(record.source_url) and ".invalid/" in record.source_url
        assert record.published_at.tzinfo is not None


def test_record_keys_are_unique(dataset: DemoDataset) -> None:
    keys = [(r.source, r.source_record_id) for r in dataset.records]
    assert len(keys) == len(set(keys))


def test_covers_all_source_types_and_enough_history(dataset: DemoDataset) -> None:
    counts = dataset.count_by_type()
    assert set(counts) == {"clinical_trial", "publication", "patent", "funding", "announcement"}
    assert all(count >= 100 for count in counts.values())
    assert len(dataset.months) >= 24
    assert len({_month(r) for r in dataset.records}) >= 24


def test_has_enough_competitors_and_topics(dataset: DemoDataset) -> None:
    companies = {o.canonical for o in DEMO_ORGS if o.org_type == "company"}
    used = {_truth(r)["organization"] for r in dataset.records}
    assert len(companies & used) >= 5
    assert len({_truth(r)["topic"] for r in dataset.records}) >= 8
    assert {o.org_type for o in DEMO_ORGS} >= {"company", "university", "hospital", "nonprofit"}


def test_demo_topics_agree_with_the_shipped_taxonomy() -> None:
    taxonomy = yaml.safe_load(TAXONOMY_FILE.read_text(encoding="utf-8"))
    by_key = {t["id"]: {s.casefold() for s in t.get("synonyms", [])} for t in taxonomy["topics"]}
    all_synonyms = set().union(*by_key.values())
    for topic in DEMO_TOPICS:
        if topic.in_taxonomy:
            assert topic.key in by_key, f"{topic.key} missing from taxonomy"
            assert all(p.casefold() in by_key[topic.key] for p in topic.phrases), topic.key
        else:
            assert topic.key not in by_key
            assert not {p.casefold() for p in topic.phrases} & all_synonyms


def test_detail_fields_are_consistent(dataset: DemoDataset) -> None:
    for record in dataset.records:
        detail = record.detail
        if record.record_type == "clinical_trial":
            assert detail["enrollment"] >= 0
            assert detail["start_date"] < detail["completion_date"]
            assert detail["phase"].startswith("PHASE")
        elif record.record_type == "patent":
            assert detail["application_date"] < detail["publication_date"]
            assert detail["patent_classifications"]
        elif record.record_type == "funding":
            assert detail["amount"] > 0 and detail["start_date"] < detail["end_date"]
        elif record.record_type == "publication":
            assert detail["affiliations"] and detail["authors"]


# --------------------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------------------
def test_sustained_trend_grows_across_sources(dataset: DemoDataset) -> None:
    for record_type in ("publication", "clinical_trial", "patent"):
        series = _monthly(dataset, "crispr_gene_editing", record_type)
        assert sum(series[-12:]) > 1.3 * sum(series[:12]), record_type
    everything = _monthly(dataset, "crispr_gene_editing")
    assert sum(1 for a, b in zip(everything, everything[1:], strict=False) if b > a) >= 12


def test_declining_topic_falls(dataset: DemoDataset) -> None:
    series = _monthly(dataset, "immune_checkpoint")
    assert sum(series[-12:]) < 0.75 * sum(series[:12])


def test_one_time_spike_is_isolated_and_single_organization(dataset: DemoDataset) -> None:
    series = _monthly(dataset, "aav_vectors", "patent")
    peak = max(series)
    assert peak >= 5 * median(series)
    assert sum(1 for value in series if value > 3 * median(series)) == 1
    spike = dataset.months[series.index(peak)]
    orgs = Counter(
        _truth(r)["organization"]
        for r in dataset.records
        if _truth(r)["topic"] == "aav_vectors"
        and r.record_type == "patent"
        and _month(r) == (spike.year, spike.month)
    )
    assert orgs.most_common(1)[0][0] == "Varethyn Labs" and orgs.most_common(1)[0][1] >= 20
    assert any(
        s["name"] == "one_time_anomaly" and s["month"] == spike.isoformat()
        for s in dataset.scenarios
    )


def test_opportunity_topic_grows_with_few_competitors(dataset: DemoDataset) -> None:
    series = _monthly(dataset, "ai_drug_discovery")
    assert sum(series[-12:]) > 2 * sum(series[:12])
    company_names = {o.canonical for o in DEMO_ORGS if o.org_type == "company"}
    active = {
        _truth(r)["organization"]
        for r in dataset.records
        if _truth(r)["topic"] == "ai_drug_discovery" and _truth(r)["organization"] in company_names
    }
    assert len(active) <= 2


def test_high_threat_competitor_dominates_and_progresses(dataset: DemoDataset) -> None:
    trials = [
        r
        for r in dataset.records
        if r.record_type == "clinical_trial" and _truth(r)["topic"] == "bispecific_antibodies"
    ]

    def share(records: list[DemoRecord]) -> float:
        zentavia = sum(1 for r in records if str(_truth(r)["organization"]).startswith("Zentavia"))
        return zentavia / max(len(records), 1)

    early = [
        r for r in trials if r.published_at < dataset.records[0].published_at.replace(year=2025)
    ]
    late = [r for r in trials if _month(r) >= (dataset.months[-6].year, dataset.months[-6].month)]
    assert share(late) > 0.6 and share(late) > share(early)
    zent = sorted(
        (r for r in trials if str(_truth(r)["organization"]).startswith("Zentavia")),
        key=lambda r: r.published_at,
    )
    first_third, last_third = zent[: len(zent) // 3], zent[-(len(zent) // 3) :]
    average = lambda rs: sum(r.detail["enrollment"] for r in rs) / len(rs)  # noqa: E731
    assert average(last_third) > 2 * average(first_third)
    assert all(r.detail["phase"] != "PHASE3" for r in first_third)
    assert any(r.detail["phase"] == "PHASE3" for r in last_third)


def test_new_market_entry_starts_late(dataset: DemoDataset) -> None:
    entry = next(s for s in dataset.scenarios if s["name"] == "new_market_entry")
    entry_month = date.fromisoformat(entry["first_month"])
    zentavia_neuro = [
        r
        for r in dataset.records
        if _truth(r)["topic"] == "neurodegeneration"
        and str(_truth(r)["organization"]) == "Zentavia Pharma"
    ]
    assert len(zentavia_neuro) >= 5
    assert min(r.published_at.date() for r in zentavia_neuro) >= entry_month


def test_false_trend_is_tiny_single_source_single_organization(dataset: DemoDataset) -> None:
    records = [r for r in dataset.records if _truth(r)["topic"] == "sirna_rnai"]
    assert 5 <= len(records) < 10
    assert {r.record_type for r in records} == {"publication"}
    assert len({_truth(r)["organization"] for r in records}) == 1
    series = _monthly(dataset, "sirna_rnai")
    assert sum(series[-3:]) >= 4 * max(sum(series[-6:-3]), 1)


def test_seasonal_topic_has_a_yearly_cycle(dataset: DemoDataset) -> None:
    series = _monthly(dataset, "mrna_therapeutics", "publication")
    by_calendar_month: dict[int, list[int]] = defaultdict(list)
    for month, value in zip(dataset.months, series, strict=True):
        by_calendar_month[month.month].append(value)
    means = {m: sum(v) / len(v) for m, v in by_calendar_month.items()}
    assert max(means.values()) > 1.4 * min(means.values())


def test_taxonomy_gap_topic_is_growing(dataset: DemoDataset) -> None:
    series = _monthly(dataset, "targeted_protein_degradation")
    assert sum(series[-12:]) > 3 * sum(series[:12])
    text = " ".join(
        r.title + r.abstract
        for r in dataset.records
        if _truth(r)["topic"] == "targeted_protein_degradation"
    )
    assert "PROTAC" in text or "protein degradation" in text


def test_messy_organization_names(dataset: DemoDataset) -> None:
    spellings: dict[str, set[str]] = defaultdict(set)
    for record in dataset.records:
        spellings[str(_truth(record)["organization"])].add(record.payload["raw_organization_name"])
    assert sum(1 for names in spellings.values() if len(names) >= 2) >= 6
    assert "Orvexa Bio" in spellings and "Orvexa Labs" in spellings  # look-alikes stay separate
    assert not spellings["Orvexa Bio"] & spellings["Orvexa Labs"]
    assert "Zentavia Oncology Ltd" in spellings  # subsidiary present


def test_announcement_ground_truth_covers_all_types(dataset: DemoDataset) -> None:
    truths = [_truth(r) for r in dataset.records if r.record_type == "announcement"]
    kinds = {t["announcement_type"] for t in truths}
    assert kinds == {
        "partnership",
        "licensing",
        "acquisition",
        "clinical_milestone",
        "funding",
        "regulatory",
    }
    for truth in truths:
        has_partner = truth["partner"] is not None
        assert has_partner == (
            truth["announcement_type"] in {"partnership", "licensing", "acquisition"}
        )


def test_scenarios_are_described(dataset: DemoDataset) -> None:
    names = {s["name"] for s in dataset.scenarios}
    assert names >= {
        "sustained_trend",
        "declining_topic",
        "one_time_anomaly",
        "high_growth_low_competition",
        "high_threat_competitor",
        "new_market_entry",
        "low_confidence_false_trend",
        "seasonal_topic",
        "topic_missing_from_taxonomy",
        "messy_organization_names",
    }


# --------------------------------------------------------------------------------------
# Configuration, helpers, fixture files
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "config",
    [
        DemoConfig(months=5),
        DemoConfig(months=500),
        DemoConfig(scale=0),
        DemoConfig(scale=50),
        DemoConfig(end_month=date(2026, 8, 15)),
    ],
)
def test_invalid_config_is_rejected(config: DemoConfig) -> None:
    with pytest.raises(ValueError):
        generate_demo_dataset(config)


def test_smaller_scale_gives_fewer_records(dataset: DemoDataset) -> None:
    small = generate_demo_dataset(DemoConfig(scale=0.2))
    assert 0 < len(small.records) < len(dataset.records) / 2


@pytest.mark.parametrize(
    ("start", "delta", "expected"),
    [
        (date(2026, 8, 1), 1, date(2026, 9, 1)),
        (date(2026, 12, 1), 1, date(2027, 1, 1)),
        (date(2026, 1, 1), -1, date(2025, 12, 1)),
        (date(2026, 8, 17), -29, date(2024, 3, 1)),
    ],
)
def test_add_months(start: date, delta: int, expected: date) -> None:
    assert add_months(start, delta) == expected


def test_write_fixtures_is_deterministic(dataset: DemoDataset, tmp_path: Path) -> None:
    first = write_fixtures(dataset, tmp_path / "a")
    second = write_fixtures(generate_demo_dataset(), tmp_path / "b")
    assert [p.name for p in first] == [p.name for p in second]
    assert len(first) == 9
    for a, b in zip(first, second, strict=True):
        assert (
            hashlib.sha256(a.read_bytes()).hexdigest() == hashlib.sha256(b.read_bytes()).hexdigest()
        )
    lines = (tmp_path / "a" / "synthetic_patents.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == dataset.count_by_type()["patent"]
    parsed = json.loads(lines[0])
    assert parsed["source"] == "synthetic_patents" and parsed["payload"]["synthetic"] is True
    config = json.loads((tmp_path / "a" / "config.json").read_text(encoding="utf-8"))
    assert config["synthetic"] is True and config["seed"] == 42
