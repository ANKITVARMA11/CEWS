"""Classifying company announcements: keyword rules, a trainable classifier, or an LLM.

Three tiers, each falling back to the one below it:

1. **An LLM** (if ``LLM_PROVIDER`` is configured) reads the announcement and extracts the type,
   partner organizations, therapeutic areas and modality as one JSON object.
2. **A trained classifier** (TF-IDF + logistic regression), if a model has been fit on labelled
   examples and saved to disk.
3. **Keyword rules**, always available, zero dependencies. The floor everything else falls back
   to, never the ceiling.

Whichever tier answers, the type is always one of :class:`~cews.constants.AnnouncementType`, with
``other`` a legitimate answer rather than a forced guess.

**Grounding.** An LLM (and, more rarely, a trained model) can name a partner company that is
simply not in the text. Every partner name a model returns is checked against the announcement's
own text before it is trusted; a name that cannot be found is dropped, and the whole extraction is
marked ungrounded so a person can look at it rather than an invented company entering the data.
"""

from __future__ import annotations

import logging
import pickle
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from cews.ai.llm import LLMClient, LLMError, llm_enabled
from cews.constants import AnnouncementType
from cews.database.models import AIExtraction, SourceRecord
from cews.settings import Settings

LOGGER = logging.getLogger(__name__)

EXTRACTION_TYPE = "announcement"
PROMPT_VERSION = "1"
MIN_TRAINING_EXAMPLES_PER_CLASS = 3
KEYWORD_CONFIDENCE = 0.6  # a rule match is a reasonable guess, not a measured probability

# Deterministic keyword rules. Order matters: earlier categories are checked first, so an
# announcement mentioning both "acquisition" and "clinical trial" is read as an acquisition
# (the more specific, less common event) rather than the generic milestone.
KeywordRule = tuple[AnnouncementType, tuple[str, ...]]
KEYWORD_RULES: tuple[KeywordRule, ...] = (
    (
        AnnouncementType.ACQUISITION,
        ("to acquire", "acquisition of", "has acquired", "merger", "to be acquired"),
    ),
    (
        AnnouncementType.LICENSING,
        (
            "license agreement",
            "licensing agreement",
            "exclusive license",
            "out-license",
            "in-license",
        ),
    ),
    (
        AnnouncementType.REGULATORY,
        (
            "fda approval",
            "fda clearance",
            "breakthrough therapy",
            "orphan drug designation",
            "ema approval",
            "marketing authorization",
        ),
    ),
    (
        AnnouncementType.CLINICAL_MILESTONE,
        (
            "topline results",
            "primary endpoint",
            "enrolled its first patient",
            "dosed the first patient",
            "phase 3 trial met",
            "interim analysis",
        ),
    ),
    (
        AnnouncementType.FUNDING,
        (
            "series a",
            "series b",
            "series c",
            "raised $",
            "financing round",
            "ipo",
            "private placement",
        ),
    ),
    (
        AnnouncementType.PARTNERSHIP,
        (
            "strategic partnership",
            "collaboration agreement",
            "joint venture",
            "partner with",
            "to co-develop",
        ),
    ),
)

JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "announcement_type": {
            "type": "string",
            "enum": [member.value for member in AnnouncementType],
        },
        "partner_organizations": {"type": "array", "items": {"type": "string"}},
        "therapeutic_areas": {"type": "array", "items": {"type": "string"}},
        "modality": {"type": ["string", "null"]},
    },
    "required": ["announcement_type", "partner_organizations", "therapeutic_areas", "modality"],
}

SYSTEM_PROMPT = (
    "You classify biotech and pharmaceutical company press releases. Read only the text given; "
    "never use outside knowledge of the company or the deal. Return only the fields asked for, "
    "and leave a field empty if the text does not state it. Every partner organization name you "
    "return must appear in the text, spelled the way the text spells it."
)


@dataclass(frozen=True)
class ExtractionResult:
    """One classification, whichever tier produced it."""

    announcement_type: AnnouncementType
    partner_organizations: tuple[str, ...]
    therapeutic_areas: tuple[str, ...]
    modality: str | None
    method: str
    model: str
    confidence: float | None
    grounded: bool
    dropped_ungrounded: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form, stored in ``ai_extractions.output_json``."""
        return {
            "announcement_type": self.announcement_type.value,
            "partner_organizations": list(self.partner_organizations),
            "therapeutic_areas": list(self.therapeutic_areas),
            "modality": self.modality,
            "method": self.method,
            "dropped_ungrounded": list(self.dropped_ungrounded),
        }


# --------------------------------------------------------------------------------------
# Tier 3: keyword rules (always available)
# --------------------------------------------------------------------------------------
def classify_by_keywords(text: str) -> ExtractionResult:
    """Classify announcement text by matching a small, ordered set of phrases.

    Never raises, never needs a model, and always returns something: ``other`` when nothing
    matches is a legitimate result, not a failure.
    """
    lowered = text.casefold()
    for announcement_type, phrases in KEYWORD_RULES:
        if any(phrase in lowered for phrase in phrases):
            return ExtractionResult(
                announcement_type=announcement_type,
                partner_organizations=(),
                therapeutic_areas=(),
                modality=None,
                method="keyword",
                model="keyword-rules-v1",
                confidence=KEYWORD_CONFIDENCE,
                grounded=True,  # nothing was extracted that could be ungrounded
            )
    return ExtractionResult(
        announcement_type=AnnouncementType.OTHER,
        partner_organizations=(),
        therapeutic_areas=(),
        modality=None,
        method="keyword",
        model="keyword-rules-v1",
        confidence=KEYWORD_CONFIDENCE,
        grounded=True,
    )


# --------------------------------------------------------------------------------------
# Tier 2: a trained classifier, once labelled examples exist
# --------------------------------------------------------------------------------------
class AnnouncementClassifier:
    """A TF-IDF + logistic regression classifier over announcement text.

    Needs no GPU and no download: both pieces are plain scikit-learn. It exists to be trained on
    real labelled announcements once a labelled set is available (see
    ``scripts/train_announcement_classifier.py``); until then, callers use the keyword rules.
    """

    def __init__(self, pipeline: Any, classes: tuple[AnnouncementType, ...]) -> None:
        self._pipeline = pipeline
        self._classes = classes

    @classmethod
    def train(cls, texts: list[str], labels: list[AnnouncementType]) -> AnnouncementClassifier:
        """Fit a classifier on labelled examples.

        Raises:
            ValueError: if there are fewer than two classes, or any class has fewer than
                :data:`MIN_TRAINING_EXAMPLES_PER_CLASS` examples (too little to learn from
                reliably, and too little to be honest about how well it will generalise).
        """
        from collections import Counter

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        if len(texts) != len(labels):
            raise ValueError("texts and labels must be the same length")
        counts = Counter(labels)
        if len(counts) < 2:
            raise ValueError("need examples from at least two announcement types to train on")
        thin = {
            label.value: count
            for label, count in counts.items()
            if count < MIN_TRAINING_EXAMPLES_PER_CLASS
        }
        if thin:
            raise ValueError(
                f"need at least {MIN_TRAINING_EXAMPLES_PER_CLASS} examples per class; "
                f"too few for: {thin}"
            )
        pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=1)),
                ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
        pipeline.fit(texts, [label.value for label in labels])
        return cls(pipeline, tuple(sorted(counts, key=lambda label: label.value)))

    def predict(self, text: str) -> tuple[AnnouncementType, float]:
        """The predicted type and the model's own confidence for it."""
        probabilities = self._pipeline.predict_proba([text])[0]
        classes = list(self._pipeline.named_steps["classifier"].classes_)
        best_index = int(probabilities.argmax())
        return AnnouncementType(classes[best_index]), float(probabilities[best_index])

    def save(self, path: Path) -> None:
        """Persist the fitted pipeline. Raises whatever ``pickle`` raises on failure."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self._pipeline, handle)

    @classmethod
    def load(cls, path: Path) -> AnnouncementClassifier | None:
        """Load a previously trained classifier, or None if the file does not exist.

        A corrupt or unreadable file is treated the same as a missing one — logged and skipped —
        because a broken cache file must never stop the pipeline from running the fallback.
        """
        if not path.is_file():
            return None
        try:
            with path.open("rb") as handle:
                pipeline = pickle.load(
                    handle
                )  # noqa: S301 - written only by this class, not user input
        except Exception as exc:  # corrupt file, incompatible scikit-learn version, ...
            LOGGER.warning("could not load announcement classifier from %s: %s", path, exc)
            return None
        classes = tuple(
            AnnouncementType(value) for value in pipeline.named_steps["classifier"].classes_
        )
        return cls(pipeline, classes)


def classify_with_model(classifier: AnnouncementClassifier, text: str) -> ExtractionResult:
    """Classify with a trained classifier. Extracts a type only; no partner/area extraction."""
    announcement_type, confidence = classifier.predict(text)
    return ExtractionResult(
        announcement_type=announcement_type,
        partner_organizations=(),
        therapeutic_areas=(),
        modality=None,
        method="trained_classifier",
        model="tfidf_logreg_v1",
        confidence=confidence,
        grounded=True,
    )


# --------------------------------------------------------------------------------------
# Grounding: nothing extracted is trusted until it is found in the source text
# --------------------------------------------------------------------------------------
def _normalize_for_matching(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def ground_partner_names(names: list[str], source_text: str) -> tuple[list[str], list[str]]:
    """Split extracted partner names into those found in the text and those that are not.

    Matching is on normalized text (case and punctuation-insensitive) so "Zentavia Pharma, Inc."
    matches "Zentavia Pharma Inc" in the source, but a name that is simply absent is never kept.

    Returns ``(kept, dropped)``.
    """
    haystack = _normalize_for_matching(source_text)
    kept: list[str] = []
    dropped: list[str] = []
    for name in names:
        needle = _normalize_for_matching(name)
        if needle and needle in haystack:
            kept.append(name)
        else:
            dropped.append(name)
    return kept, dropped


# --------------------------------------------------------------------------------------
# Tier 1: an LLM, when one is configured
# --------------------------------------------------------------------------------------
def extract_with_llm(client: LLMClient, text: str, *, model_name: str) -> ExtractionResult:
    """Ask the configured LLM to extract fields, then ground every partner name it returns.

    Raises:
        LLMError: if the call fails or the response cannot be parsed as the expected JSON. The
            caller (:func:`extract_announcement`) catches this and falls back to a lower tier.
    """
    response = client.complete(
        f"Announcement text:\n\n{text}",
        system=SYSTEM_PROMPT,
        json_schema=JSON_SCHEMA,
    )
    payload = response.json()
    if not isinstance(payload, dict):
        raise LLMError("extraction response was not a JSON object")
    try:
        announcement_type = AnnouncementType(str(payload["announcement_type"]))
    except (KeyError, ValueError) as exc:
        raise LLMError(f"extraction response had an invalid announcement_type: {exc}") from exc

    raw_partners = [str(name) for name in payload.get("partner_organizations") or []]
    kept, dropped = ground_partner_names(raw_partners, text)
    areas = [str(area) for area in payload.get("therapeutic_areas") or []]
    modality = payload.get("modality")
    return ExtractionResult(
        announcement_type=announcement_type,
        partner_organizations=tuple(kept),
        therapeutic_areas=tuple(areas),
        modality=str(modality) if modality else None,
        method="llm",
        model=model_name,
        confidence=None,  # chat models do not give a calibrated probability
        grounded=not dropped,
        dropped_ungrounded=tuple(dropped),
    )


# --------------------------------------------------------------------------------------
# Orchestration: pick a tier, cache the result
# --------------------------------------------------------------------------------------
def classifier_model_path(settings: Settings) -> Path:
    """Where a trained classifier is expected to live."""
    return settings.project_root / "data" / "models" / "announcement_classifier.pkl"


def extract_announcement(
    session: Session,
    settings: Settings,
    record: SourceRecord,
    *,
    text: str,
    llm_client: LLMClient | None = None,
    classifier: AnnouncementClassifier | None = None,
    store: bool = True,
) -> ExtractionResult:
    """Classify one announcement, trying the best available tier and caching the result.

    Args:
        record: the announcement's source record (used for its id and for caching).
        text: the announcement's own text (title plus body), used for classification and for
            grounding anything the model claims.
        llm_client: an already-constructed client; if omitted and an LLM is enabled, one is
            built for this call. Pass one in in a loop to reuse the connection.
        classifier: a trained classifier; if omitted, one is loaded from
            :func:`classifier_model_path` if present.
        store: cache the result in ``ai_extractions``.

    A cached result for the same record, extraction type, model and prompt version is reused
    rather than reprocessed.
    """
    cached = session.scalar(
        select(AIExtraction).where(
            AIExtraction.source_record_id == record.id,
            AIExtraction.extraction_type == EXTRACTION_TYPE,
            AIExtraction.prompt_version == PROMPT_VERSION,
        )
    )
    if cached is not None:
        payload = cached.output_json
        return ExtractionResult(
            announcement_type=AnnouncementType(payload["announcement_type"]),
            partner_organizations=tuple(payload.get("partner_organizations", [])),
            therapeutic_areas=tuple(payload.get("therapeutic_areas", [])),
            modality=payload.get("modality"),
            method=payload.get("method", "unknown"),
            model=cached.model,
            confidence=cached.confidence,
            grounded=cached.grounded,
            dropped_ungrounded=tuple(payload.get("dropped_ungrounded", [])),
        )

    result = _classify(session, settings, text, llm_client=llm_client, classifier=classifier)
    if store:
        session.add(
            AIExtraction(
                source_record_id=record.id,
                extraction_type=EXTRACTION_TYPE,
                model=result.model,
                prompt_version=PROMPT_VERSION,
                output_json=result.as_dict(),
                grounded=result.grounded,
                confidence=result.confidence,
                created_at=datetime.now(UTC),
            )
        )
        session.flush()
    return result


def _classify(
    session: Session,
    settings: Settings,
    text: str,
    *,
    llm_client: LLMClient | None,
    classifier: AnnouncementClassifier | None,
) -> ExtractionResult:
    if settings.enable_ai_announcement_extraction and llm_enabled(settings):
        client = llm_client
        owns_client = False
        try:
            if client is None:
                client = LLMClient(settings)
                owns_client = True
            return extract_with_llm(client, text, model_name=settings.llm_model)
        except LLMError as exc:
            LOGGER.warning("LLM extraction failed, falling back: %s", exc)
        finally:
            if owns_client and client is not None:
                client.close()

    model = classifier or AnnouncementClassifier.load(classifier_model_path(settings))
    if model is not None:
        return classify_with_model(model, text)

    return classify_by_keywords(text)
