"""Tests for keyword classification, the trainable classifier, grounding, and orchestration."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from sqlalchemy.orm import Session

from cews.ai.announcement_extraction import (
    AnnouncementClassifier,
    classifier_model_path,
    classify_by_keywords,
    classify_with_model,
    extract_announcement,
    extract_with_llm,
    ground_partner_names,
)
from cews.ai.llm import LLMClient, LLMError
from cews.constants import AnnouncementType
from cews.database.connection import create_memory_engine, create_session_factory, session_scope
from cews.database.migrations import upgrade_database
from cews.database.models import AIExtraction, SourceRecord
from cews.settings import load_settings

pytestmark = pytest.mark.unit

KEYWORD_CASES = [
    ("Zentavia Pharma to acquire Nexoria Genetics for $1.2B", AnnouncementType.ACQUISITION),
    (
        "Halcyra Biosciences announces exclusive license agreement with Orvexa Labs",
        AnnouncementType.LICENSING,
    ),
    ("Varethyn Labs receives FDA approval for lead candidate", AnnouncementType.REGULATORY),
    ("Zentavia dosed the first patient in its Phase 3 trial", AnnouncementType.CLINICAL_MILESTONE),
    ("Nexoria Genetics closes $50M Series B financing round", AnnouncementType.FUNDING),
    (
        "Orvexa Labs and Halcyra announce strategic partnership to co-develop therapies",
        AnnouncementType.PARTNERSHIP,
    ),
    ("Zentavia publishes quarterly earnings call transcript", AnnouncementType.OTHER),
]


@pytest.mark.parametrize(("text", "expected"), KEYWORD_CASES)
def test_keyword_rules_cover_every_category(text: str, expected: AnnouncementType) -> None:
    result = classify_by_keywords(text)
    assert result.announcement_type is expected
    assert result.method == "keyword"
    assert result.grounded is True


def test_other_is_a_legitimate_answer_not_a_crash() -> None:
    result = classify_by_keywords("The board met to discuss routine governance matters.")
    assert result.announcement_type is AnnouncementType.OTHER


def test_keyword_rules_never_raise_on_odd_input() -> None:
    for text in ("", " ", "12345", "!@#$%^&*()"):
        result = classify_by_keywords(text)
        assert result.announcement_type is AnnouncementType.OTHER


# --------------------------------------------------------------------------------------
# Grounding
# --------------------------------------------------------------------------------------
def test_grounding_keeps_names_present_in_the_text() -> None:
    text = "Zentavia Pharma, Inc. today announced a deal with Halcyra Biosciences GmbH."
    kept, dropped = ground_partner_names(["Zentavia Pharma Inc", "Halcyra Biosciences GmbH"], text)
    assert set(kept) == {"Zentavia Pharma Inc", "Halcyra Biosciences GmbH"}
    assert dropped == []


def test_grounding_drops_an_invented_name() -> None:
    text = "Zentavia Pharma announced a partnership with Halcyra Biosciences."
    kept, dropped = ground_partner_names(["Halcyra Biosciences", "Nexoria Genetics"], text)
    assert kept == ["Halcyra Biosciences"]
    assert dropped == ["Nexoria Genetics"]


def test_grounding_ignores_punctuation_and_case() -> None:
    text = "A deal between ZENTAVIA-PHARMA and Halcyra..."
    kept, _ = ground_partner_names(["zentavia pharma"], text)
    assert kept == ["zentavia pharma"]


def test_an_empty_name_is_never_kept() -> None:
    kept, dropped = ground_partner_names([""], "some text")
    assert kept == [] and dropped == [""]


# --------------------------------------------------------------------------------------
# Trained classifier
# --------------------------------------------------------------------------------------
TRAIN_TEXTS = [
    "to acquire smaller biotech for cash",
    "completes acquisition of rival firm",
    "closes merger with competitor",
    "to be acquired by larger pharma",
    "strategic partnership announced today",
    "signs collaboration agreement with partner",
    "joint venture formed to co-develop drug",
    "partners with academic institute",
    "raised $40M in Series B",
    "closes financing round led by investors",
    "completes IPO on Nasdaq",
    "announces private placement of shares",
]
TRAIN_LABELS = (
    [AnnouncementType.ACQUISITION] * 4
    + [AnnouncementType.PARTNERSHIP] * 4
    + [AnnouncementType.FUNDING] * 4
)


def test_a_trained_classifier_predicts_a_known_category() -> None:
    classifier = AnnouncementClassifier.train(TRAIN_TEXTS, TRAIN_LABELS)
    predicted, confidence = classifier.predict(
        "the company completed its acquisition of a competitor"
    )
    assert predicted is AnnouncementType.ACQUISITION
    assert 0.0 < confidence <= 1.0


def test_classify_with_model_wraps_the_prediction() -> None:
    classifier = AnnouncementClassifier.train(TRAIN_TEXTS, TRAIN_LABELS)
    result = classify_with_model(classifier, "raised a new financing round")
    assert result.method == "trained_classifier"
    assert result.announcement_type is AnnouncementType.FUNDING
    assert result.partner_organizations == ()


def test_training_needs_at_least_two_classes() -> None:
    with pytest.raises(ValueError, match="at least two"):
        AnnouncementClassifier.train(["a", "b", "c"], [AnnouncementType.FUNDING] * 3)


def test_training_needs_enough_examples_per_class() -> None:
    with pytest.raises(ValueError, match="at least 3 examples"):
        AnnouncementClassifier.train(
            ["a", "b", "c"],
            [AnnouncementType.FUNDING, AnnouncementType.FUNDING, AnnouncementType.OTHER],
        )


def test_mismatched_texts_and_labels_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        AnnouncementClassifier.train(["a", "b"], [AnnouncementType.FUNDING])


def test_a_classifier_can_be_saved_and_reloaded(tmp_path: Path) -> None:
    classifier = AnnouncementClassifier.train(TRAIN_TEXTS, TRAIN_LABELS)
    path = tmp_path / "model.pkl"
    classifier.save(path)
    loaded = AnnouncementClassifier.load(path)
    assert loaded is not None
    original = classifier.predict("acquisition of a rival firm")
    reloaded = loaded.predict("acquisition of a rival firm")
    assert original == reloaded


def test_loading_a_missing_file_returns_none(tmp_path: Path) -> None:
    assert AnnouncementClassifier.load(tmp_path / "does-not-exist.pkl") is None


def test_loading_a_corrupt_file_returns_none_rather_than_raising(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.pkl"
    path.write_bytes(b"not a pickle at all")
    assert AnnouncementClassifier.load(path) is None


# --------------------------------------------------------------------------------------
# LLM tier
# --------------------------------------------------------------------------------------
def _llm(handler: object) -> LLMClient:
    settings = load_settings(
        env_file=None, overrides={"llm_provider": "ollama", "llm_base_url": "http://x/v1"}
    )
    return LLMClient(settings, transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def test_llm_extraction_parses_every_field() -> None:
    text = "Zentavia Pharma announced a partnership with Halcyra Biosciences to co-develop CAR-T therapies for oncology."

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "announcement_type": "partnership",
                                    "partner_organizations": ["Halcyra Biosciences"],
                                    "therapeutic_areas": ["oncology"],
                                    "modality": "CAR-T",
                                }
                            )
                        }
                    }
                ]
            },
        )

    with _llm(handler) as client:
        result = extract_with_llm(client, text, model_name="qwen2.5:3b")
    assert result.announcement_type is AnnouncementType.PARTNERSHIP
    assert result.partner_organizations == ("Halcyra Biosciences",)
    assert result.therapeutic_areas == ("oncology",)
    assert result.modality == "CAR-T"
    assert result.grounded is True
    assert result.method == "llm" and result.model == "qwen2.5:3b"


def test_llm_extraction_drops_an_invented_partner_and_flags_ungrounded() -> None:
    text = "Zentavia Pharma announced a partnership with Halcyra Biosciences."

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "announcement_type": "partnership",
                                    "partner_organizations": [
                                        "Halcyra Biosciences",
                                        "Made-Up Corp",
                                    ],
                                    "therapeutic_areas": [],
                                    "modality": None,
                                }
                            )
                        }
                    }
                ]
            },
        )

    with _llm(handler) as client:
        result = extract_with_llm(client, text, model_name="m")
    assert result.grounded is False
    assert result.dropped_ungrounded == ("Made-Up Corp",)
    assert result.partner_organizations == ("Halcyra Biosciences",)


def test_an_invalid_announcement_type_from_the_model_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "announcement_type": "not_a_real_type",
                                    "partner_organizations": [],
                                    "therapeutic_areas": [],
                                    "modality": None,
                                }
                            )
                        }
                    }
                ]
            },
        )

    with _llm(handler) as client, pytest.raises(LLMError, match="invalid announcement_type"):
        extract_with_llm(client, "text", model_name="m")


def test_a_non_object_response_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "[1, 2, 3]"}}]})

    with _llm(handler) as client, pytest.raises(LLMError, match="not a JSON object"):
        extract_with_llm(client, "text", model_name="m")


# --------------------------------------------------------------------------------------
# Orchestration and caching
# --------------------------------------------------------------------------------------
def _record(session: Session, identifier: str = "r1") -> SourceRecord:
    record = SourceRecord(
        source="generic_rss",
        source_record_id=identifier,
        record_type="announcement",
        fetched_at=datetime.now(UTC),
        content_hash=f"{identifier:0>64}"[:64],
    )
    session.add(record)
    session.flush()
    return record


@pytest.fixture
def factory() -> object:
    engine = create_memory_engine()
    upgrade_database(engine)
    yield create_session_factory(engine)
    engine.dispose()


def test_with_nothing_configured_the_keyword_tier_runs(factory: object) -> None:
    settings = load_settings(env_file=None)
    with session_scope(factory) as session:  # type: ignore[arg-type]
        record = _record(session)
        result = extract_announcement(session, settings, record, text="to acquire a rival for cash")
    assert result.method == "keyword"
    assert result.announcement_type is AnnouncementType.ACQUISITION


def test_a_trained_classifier_is_used_when_provided(factory: object) -> None:
    settings = load_settings(env_file=None)
    classifier = AnnouncementClassifier.train(TRAIN_TEXTS, TRAIN_LABELS)
    with session_scope(factory) as session:  # type: ignore[arg-type]
        record = _record(session)
        result = extract_announcement(
            session, settings, record, text="raised a new financing round", classifier=classifier
        )
    assert result.method == "trained_classifier"


def test_llm_failure_falls_back_to_the_next_tier(factory: object) -> None:
    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    settings = load_settings(
        env_file=None,
        overrides={
            "enable_ai_announcement_extraction": True,
            "llm_provider": "ollama",
            "llm_base_url": "http://x/v1",
            "llm_max_retries": 0,
        },
    )
    with (
        LLMClient(settings, transport=httpx.MockTransport(failing)) as client,
        session_scope(factory) as session,  # type: ignore[arg-type]
    ):
        record = _record(session)
        result = extract_announcement(
            session, settings, record, text="to acquire a rival for cash", llm_client=client
        )
    assert result.method == "keyword"  # fell all the way through to the floor
    assert result.announcement_type is AnnouncementType.ACQUISITION


def test_a_result_is_cached_and_reused(factory: object) -> None:
    settings = load_settings(env_file=None)
    with session_scope(factory) as session:  # type: ignore[arg-type]
        record = _record(session)
        first = extract_announcement(session, settings, record, text="to acquire a rival for cash")
        second = extract_announcement(session, settings, record, text="a completely different text")
        rows = list(session.scalars(AIExtraction.__table__.select()))  # type: ignore[arg-type]
    assert first.announcement_type == second.announcement_type == AnnouncementType.ACQUISITION
    assert len(rows) == 1  # the second call reused the cached row rather than reprocessing


def test_two_different_records_are_not_conflated(factory: object) -> None:
    settings = load_settings(env_file=None)
    with session_scope(factory) as session:  # type: ignore[arg-type]
        one = _record(session, "r1")
        two = _record(session, "r2")
        result_one = extract_announcement(session, settings, one, text="to acquire a rival")
        result_two = extract_announcement(session, settings, two, text="raised a Series B")
    assert result_one.announcement_type is AnnouncementType.ACQUISITION
    assert result_two.announcement_type is AnnouncementType.FUNDING


def test_store_false_does_not_write_a_row(factory: object) -> None:
    settings = load_settings(env_file=None)
    with session_scope(factory) as session:  # type: ignore[arg-type]
        record = _record(session)
        extract_announcement(session, settings, record, text="to acquire a rival", store=False)
        rows = list(session.scalars(AIExtraction.__table__.select()))  # type: ignore[arg-type]
    assert rows == []


def test_classifier_model_path_lives_under_project_root(tmp_path: Path) -> None:
    settings = load_settings(env_file=None, overrides={"project_root": tmp_path})
    path = classifier_model_path(settings)
    assert path.is_relative_to(tmp_path)
    assert path.suffix == ".pkl"
