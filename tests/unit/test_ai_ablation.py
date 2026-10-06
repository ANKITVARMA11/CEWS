"""Unit tests for the AI on-vs-off ablation."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy.orm import Session, sessionmaker

from cews.ai.llm import LLMClient
from cews.constants import AnnouncementType
from cews.database.connection import create_memory_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import RecordTopic, ReviewQueueItem, SourceRecord, Topic
from cews.settings import load_settings
from cews.validation.ai_ablation import (
    ORG_MATCHING_STATUS,
    AblationError,
    LabelledAnnouncement,
    evaluate_announcement_methods,
    evaluate_topic_discovery,
    load_labelled_announcements,
    run_ai_ablation,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS = REPO_ROOT / "config" / "announcement_eval_labels.yaml"


@pytest.fixture
def factory() -> Iterator[sessionmaker[Session]]:
    engine = create_memory_engine()
    upgrade_database(engine)
    yield create_session_factory(engine)
    engine.dispose()


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "labels.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _small_set() -> list[LabelledAnnouncement]:
    """Three obviously-worded classes, enough per class to cross-validate."""
    rows = []
    for index in range(6):
        rows.append(
            LabelledAnnouncement(
                f"A{index}",
                f"Company {index} to acquire Target {index}",
                AnnouncementType.ACQUISITION,
            )
        )
        rows.append(
            LabelledAnnouncement(
                f"F{index}",
                f"Company {index} closes Series B financing round",
                AnnouncementType.FUNDING,
            )
        )
        rows.append(
            LabelledAnnouncement(
                f"O{index}", f"Company {index} appoints new chief officer", AnnouncementType.OTHER
            )
        )
    return rows


# --------------------------------------------------------------------------------------
# The shipped fixture
# --------------------------------------------------------------------------------------
def test_the_shipped_fixture_has_at_least_fifty_labelled_announcements() -> None:
    labelled = load_labelled_announcements(LABELS)
    assert len(labelled) >= 50


def test_the_shipped_fixture_covers_every_category_reasonably() -> None:
    counts = Counter(item.label for item in load_labelled_announcements(LABELS))
    assert set(counts) == set(AnnouncementType)
    assert min(counts.values()) >= 5


def test_the_shipped_fixture_texts_are_unique() -> None:
    texts = [item.text for item in load_labelled_announcements(LABELS)]
    assert len(texts) == len(set(texts))


# --------------------------------------------------------------------------------------
# Loading validation
# --------------------------------------------------------------------------------------
def test_too_few_announcements_are_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, "announcements:\n  - {id: a, label: other, text: hello}\n")
    with pytest.raises(AblationError, match="at least 50"):
        load_labelled_announcements(path)


def test_an_unknown_label_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, "announcements:\n  - {id: a, label: nonsense, text: hello}\n")
    with pytest.raises(AblationError, match="nonsense"):
        load_labelled_announcements(path, minimum=1)


def test_a_repeated_id_is_refused(tmp_path: Path) -> None:
    body = "announcements:\n  - {id: a, label: other, text: one}\n  - {id: a, label: other, text: two}\n"
    with pytest.raises(AblationError, match="more than once"):
        load_labelled_announcements(_write(tmp_path, body), minimum=1)


@pytest.mark.parametrize(
    "body", ["announcements: x\n", "announcements:\n  - {id: a, label: other}\n"]
)
def test_malformed_entries_are_refused(tmp_path: Path, body: str) -> None:
    with pytest.raises(AblationError):
        load_labelled_announcements(_write(tmp_path, body), minimum=1)


def test_a_missing_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(AblationError, match="not found"):
        load_labelled_announcements(tmp_path / "absent.yaml")


# --------------------------------------------------------------------------------------
# Announcement methods
# --------------------------------------------------------------------------------------
def test_keyword_accuracy_is_computed_against_the_labels() -> None:
    result = evaluate_announcement_methods(_small_set())
    keyword = next(m for m in result.methods if m.method == "keyword")
    assert keyword.evaluated == 18 and keyword.errors == 0
    assert keyword.correct == 18  # every text was worded to match a rule or fall through to "other"
    assert keyword.accuracy == 1.0
    assert keyword.per_class_recall["acquisition"] == 1.0


def test_the_trained_classifier_is_never_scored_on_its_own_training_text() -> None:
    """Cross-validation: with only two classes of unique text, a model scored on training data
    would be perfect; held-out folds are what make the number honest."""
    labelled = [
        LabelledAnnouncement(
            f"X{i}",
            f"alpha{i} beta{i}",
            AnnouncementType.ACQUISITION if i % 2 else AnnouncementType.FUNDING,
        )
        for i in range(20)
    ]
    result = evaluate_announcement_methods(labelled)
    trained = next(m for m in result.methods if m.method == "trained_classifier")
    assert (
        trained.accuracy is not None and trained.accuracy < 1.0
    )  # unique tokens cannot generalize


def test_the_trained_classifier_is_skipped_when_a_class_is_too_thin() -> None:
    labelled = _small_set()[:9] + [
        LabelledAnnouncement("Z", "lone example", AnnouncementType.REGULATORY)
    ]
    result = evaluate_announcement_methods(labelled)
    trained = next(m for m in result.methods if m.method == "trained_classifier")
    assert trained.skipped_reason and trained.accuracy is None


def test_the_llm_is_skipped_when_none_is_configured() -> None:
    llm = next(m for m in evaluate_announcement_methods(_small_set()).methods if m.method == "llm")
    assert "no LLM provider" in llm.skipped_reason and llm.accuracy is None


def _llm_client(answer: Any) -> LLMClient:
    def handler(request: httpx.Request) -> httpx.Response:
        text = json.loads(request.content)["messages"][-1]["content"]
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(answer(text))}}]}
        )

    settings = load_settings(
        env_file=None, overrides={"llm_provider": "ollama", "llm_base_url": "http://x/v1"}
    )
    return LLMClient(settings, transport=httpx.MockTransport(handler))


def test_a_perfect_llm_scores_perfectly_and_grounded() -> None:
    truth = {item.text: item.label.value for item in _small_set()}

    def answer(prompt: str) -> dict[str, Any]:
        for text, label in truth.items():
            if text in prompt:
                return {
                    "announcement_type": label,
                    "partner_organizations": [],
                    "therapeutic_areas": [],
                    "modality": None,
                }
        raise AssertionError("unexpected prompt")

    with _llm_client(answer) as client:
        result = evaluate_announcement_methods(_small_set(), llm_client=client, llm_model_name="m")
    llm = next(m for m in result.methods if m.method == "llm")
    assert llm.accuracy == 1.0 and llm.grounded_rate == 1.0
    assert result.gain_over_keyword("llm") == pytest.approx(0.0)


def test_an_llm_that_invents_a_partner_lowers_the_grounding_rate() -> None:
    def answer(_prompt: str) -> dict[str, Any]:
        return {
            "announcement_type": "other",
            "partner_organizations": ["Invented Corp"],
            "therapeutic_areas": [],
            "modality": None,
        }

    with _llm_client(answer) as client:
        result = evaluate_announcement_methods(_small_set(), llm_client=client, llm_model_name="m")
    llm = next(m for m in result.methods if m.method == "llm")
    assert llm.grounded_rate == 0.0
    assert llm.accuracy == pytest.approx(6 / 18)  # only the six "other" items are right


def test_llm_failures_count_as_wrong_not_as_skipped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad")

    settings = load_settings(
        env_file=None, overrides={"llm_provider": "ollama", "llm_base_url": "http://x/v1"}
    )
    with LLMClient(settings, transport=httpx.MockTransport(handler)) as client:
        result = evaluate_announcement_methods(_small_set(), llm_client=client, llm_model_name="m")
    llm = next(m for m in result.methods if m.method == "llm")
    assert llm.errors == 18 and llm.accuracy == 0.0 and llm.grounded_rate is None


def test_gain_over_keyword_is_none_without_a_comparison() -> None:
    result = evaluate_announcement_methods(_small_set())
    assert result.gain_over_keyword("llm") is None
    assert result.gain_over_keyword("nonexistent") is None


def test_too_few_cv_folds_is_refused() -> None:
    with pytest.raises(AblationError, match="at least 2"):
        evaluate_announcement_methods(_small_set(), cv_folds=1)


# --------------------------------------------------------------------------------------
# Topic discovery
# --------------------------------------------------------------------------------------
def _candidate(session: Session, status: str, records: int, key: str) -> None:
    session.add(
        ReviewQueueItem(
            queue_type="ai_topic_candidate",
            subject_ref=f"topic:{key}",
            payload_json={"label": key, "record_count": records},
            status=status,
            created_at=datetime.now(UTC),
        )
    )


def test_topic_discovery_precision_comes_from_reviewed_candidates(
    factory: sessionmaker[Session],
) -> None:
    with session_scope(factory) as session:
        for status, records, key in (
            ("accepted", 50, "a"),
            ("accepted", 40, "b"),
            ("rejected", 30, "c"),
            ("pending", 20, "d"),
        ):
            _candidate(session, status, records, key)
        session.flush()
        result = evaluate_topic_discovery(session, top_n=2)
    assert result.candidates == 4 and result.pending == 1
    assert result.precision == pytest.approx(2 / 3)  # pending is excluded, not counted either way
    assert result.precision_top_n == 1.0  # the two largest candidates were both accepted
    assert result.records_in_candidates == 140


def test_precision_is_undefined_with_nothing_reviewed(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        _candidate(session, "pending", 10, "a")
        session.flush()
        result = evaluate_topic_discovery(session)
    assert result.precision is None and result.precision_top_n is None


def test_without_discovery_the_novel_cluster_count_is_zero_by_definition(
    factory: sessionmaker[Session],
) -> None:
    with session_scope(factory) as session:
        payload = evaluate_topic_discovery(session).as_dict()
    assert payload["novel_clusters_without_discovery"] == 0
    assert payload["novel_clusters_with_discovery"] == 0


def test_other_queue_types_are_not_counted_as_topic_candidates(
    factory: sessionmaker[Session],
) -> None:
    with session_scope(factory) as session:
        session.add(
            ReviewQueueItem(
                queue_type="organization_match", subject_ref="x", payload_json={}, status="accepted"
            )
        )
        session.flush()
        assert evaluate_topic_discovery(session).candidates == 0


def test_unmatched_records_are_counted_for_context(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session:
        topic = Topic(key="t", canonical_name="T", topic_type="technology")
        session.add(topic)
        matched = SourceRecord(
            source="s",
            source_record_id="m",
            record_type="publication",
            fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
            content_hash="a" * 64,
        )
        loose = SourceRecord(
            source="s",
            source_record_id="u",
            record_type="publication",
            fetched_at=datetime(2026, 1, 1, tzinfo=UTC),
            content_hash="b" * 64,
        )
        session.add_all([matched, loose])
        session.flush()
        session.add(RecordTopic(source_record_id=matched.id, topic_id=topic.id, confidence=1.0))
        session.flush()
        assert evaluate_topic_discovery(session).unmatched_records == 1


def test_top_n_must_be_positive(factory: sessionmaker[Session]) -> None:
    with session_scope(factory) as session, pytest.raises(ValueError, match="positive"):
        evaluate_topic_discovery(session, top_n=0)


# --------------------------------------------------------------------------------------
# The whole ablation
# --------------------------------------------------------------------------------------
def test_org_matching_is_reported_as_skipped(factory: sessionmaker[Session]) -> None:
    settings = load_settings(env_file=None)
    with session_scope(factory) as session:
        report = run_ai_ablation(session, settings, LABELS)
    assert report.org_matching == ORG_MATCHING_STATUS
    assert report.as_dict()["org_matching"].startswith("skipped")


def test_a_full_ablation_covers_both_implemented_layers(factory: sessionmaker[Session]) -> None:
    settings = load_settings(env_file=None)
    with session_scope(factory) as session:
        report = run_ai_ablation(session, settings, LABELS)
    assert report.announcements is not None and report.announcements.labelled_count >= 50
    assert report.topic_discovery is not None
    assert any("no topic candidates yet" in warning for warning in report.warnings)
    json.dumps(report.as_dict())


def test_a_bad_fixture_is_a_warning_not_a_crash(
    factory: sessionmaker[Session], tmp_path: Path
) -> None:
    settings = load_settings(env_file=None)
    with session_scope(factory) as session:
        report = run_ai_ablation(session, settings, tmp_path / "absent.yaml")
    assert report.announcements is None
    assert any("announcement extraction not evaluated" in warning for warning in report.warnings)
    assert report.topic_discovery is not None
