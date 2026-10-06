"""Does each AI layer earn its place? Compare it on against it off.

"Off" is the deterministic fallback that runs whenever an AI layer is disabled or unavailable,
so the question for each layer is what it adds over that floor:

* **Announcement extraction** is scored against a labelled fixture set
  (``config/announcement_eval_labels.yaml``): keyword rules (the floor), a trained classifier
  (evaluated by deterministic cross-validation, so it is never scored on text it was trained on),
  and an LLM if one is configured. The LLM is also given a **grounding pass rate**: how often
  everything it named could be found in the announcement's own text.
* **Topic discovery** is judged by what a person made of its candidates: how many novel clusters
  it proposed, and what share of those an expert accepted, read from the review queue.
* **Organization matching** is not implemented yet and is reported as skipped.

Nothing here decides whether to enable a layer; it reports what the layer did.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from cews.ai.announcement_extraction import (
    AnnouncementClassifier,
    ExtractionResult,
    classify_by_keywords,
    classify_with_model,
    extract_with_llm,
)
from cews.ai.llm import LLMClient, LLMError
from cews.ai.topic_discovery import QUEUE_TYPE
from cews.constants import AnnouncementType
from cews.database.models import RecordTopic, ReviewQueueItem, SourceRecord
from cews.settings import Settings

LOGGER = logging.getLogger(__name__)

MIN_LABELLED_ANNOUNCEMENTS = 50
DEFAULT_CV_FOLDS = 5
ORG_MATCHING_STATUS = "skipped: organization matching is not implemented yet"


class AblationError(ValueError):
    """Raised when the labelled fixture set cannot be used."""


@dataclass(frozen=True)
class LabelledAnnouncement:
    """One announcement with its true category."""

    identifier: str
    text: str
    label: AnnouncementType


@dataclass(frozen=True)
class MethodResult:
    """How one classification method did on the labelled set."""

    method: str
    evaluated: int
    correct: int
    errors: int
    per_class_recall: dict[str, float]
    grounded_rate: float | None
    skipped_reason: str = ""

    @property
    def accuracy(self) -> float | None:
        """Share of the labelled set classified correctly (failures count as wrong)."""
        total = self.evaluated + self.errors
        return self.correct / total if total and not self.skipped_reason else None

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return {
            "method": self.method,
            "evaluated": self.evaluated,
            "correct": self.correct,
            "errors": self.errors,
            "accuracy": None if self.accuracy is None else round(self.accuracy, 4),
            "per_class_recall": {k: round(v, 4) for k, v in self.per_class_recall.items()},
            "grounded_rate": None if self.grounded_rate is None else round(self.grounded_rate, 4),
            "skipped_reason": self.skipped_reason,
        }


@dataclass(frozen=True)
class AnnouncementAblation:
    """Every method side by side, with what each adds over the keyword floor."""

    labelled_count: int
    methods: tuple[MethodResult, ...]

    def gain_over_keyword(self, method: str) -> float | None:
        """Accuracy of ``method`` minus the keyword rules' accuracy, or None if either is missing."""
        by_name = {result.method: result for result in self.methods}
        baseline = by_name.get("keyword")
        other = by_name.get(method)
        if baseline is None or other is None or baseline.accuracy is None or other.accuracy is None:
            return None
        return other.accuracy - baseline.accuracy

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return {
            "labelled_count": self.labelled_count,
            "methods": [result.as_dict() for result in self.methods],
            "gain_over_keyword": {
                result.method: (
                    None
                    if (gain := self.gain_over_keyword(result.method)) is None
                    else round(gain, 4)
                )
                for result in self.methods
                if result.method != "keyword"
            },
        }


@dataclass(frozen=True)
class TopicDiscoveryEvaluation:
    """What people made of the topics discovery proposed."""

    candidates: int
    pending: int
    accepted: int
    rejected: int
    precision: float | None
    precision_top_n: float | None
    top_n: int
    unmatched_records: int
    records_in_candidates: int

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form. With discovery off, novel clusters is 0 by definition."""
        return {
            "novel_clusters_with_discovery": self.candidates,
            "novel_clusters_without_discovery": 0,
            "pending_review": self.pending,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "precision": None if self.precision is None else round(self.precision, 4),
            "precision_top_n": (
                None if self.precision_top_n is None else round(self.precision_top_n, 4)
            ),
            "top_n": self.top_n,
            "unmatched_records": self.unmatched_records,
            "records_in_candidates": self.records_in_candidates,
        }


@dataclass
class AIAblationReport:
    """Every AI layer, on against off."""

    announcements: AnnouncementAblation | None = None
    topic_discovery: TopicDiscoveryEvaluation | None = None
    org_matching: str = ORG_MATCHING_STATUS
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """JSON-friendly form."""
        return {
            "announcement_extraction": (
                None if self.announcements is None else self.announcements.as_dict()
            ),
            "topic_discovery": (
                None if self.topic_discovery is None else self.topic_discovery.as_dict()
            ),
            "org_matching": self.org_matching,
            "warnings": list(self.warnings),
        }


def load_labelled_announcements(
    path: Path, *, minimum: int = MIN_LABELLED_ANNOUNCEMENTS
) -> list[LabelledAnnouncement]:
    """Read the labelled fixture set.

    Raises:
        AblationError: if the file is missing or unreadable, an entry is malformed or carries an
            unknown label, an id repeats, or there are fewer than ``minimum`` announcements (too
            few to say anything reliable about accuracy).
    """
    if not path.is_file():
        raise AblationError(f"labelled announcement file not found: {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise AblationError(f"cannot read {path}: {exc}") from exc
    entries = document.get("announcements")
    if not isinstance(entries, list):
        raise AblationError(f"{path}: expected a list under 'announcements'")

    valid = {member.value for member in AnnouncementType}
    labelled: list[LabelledAnnouncement] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or not entry.get("id") or not entry.get("text"):
            raise AblationError(f"{path}: entry {index} needs 'id' and 'text'")
        identifier = str(entry["id"])
        if identifier in seen:
            raise AblationError(f"{path}: id {identifier!r} appears more than once")
        seen.add(identifier)
        if entry.get("label") not in valid:
            raise AblationError(
                f"{path}: {identifier} has label {entry.get('label')!r}; use one of {sorted(valid)}"
            )
        labelled.append(
            LabelledAnnouncement(identifier, str(entry["text"]), AnnouncementType(entry["label"]))
        )
    if len(labelled) < minimum:
        raise AblationError(
            f"{path}: only {len(labelled)} labelled announcement(s); at least {minimum} are needed"
        )
    return labelled


def _score(
    method: str,
    labelled: list[LabelledAnnouncement],
    predict: Callable[[int, LabelledAnnouncement], ExtractionResult | None],
) -> MethodResult:
    """Run ``predict`` over the set. ``None`` from ``predict`` marks a failed classification."""
    correct = errors = evaluated = grounded = 0
    seen_by_class: Counter[str] = Counter()
    right_by_class: Counter[str] = Counter()
    for index, item in enumerate(labelled):
        seen_by_class[item.label.value] += 1
        result = predict(index, item)
        if result is None:
            errors += 1
            continue
        evaluated += 1
        grounded += 1 if result.grounded else 0
        if result.announcement_type is item.label:
            correct += 1
            right_by_class[item.label.value] += 1
    return MethodResult(
        method=method,
        evaluated=evaluated,
        correct=correct,
        errors=errors,
        per_class_recall={
            label: right_by_class[label] / count for label, count in sorted(seen_by_class.items())
        },
        grounded_rate=(grounded / evaluated) if evaluated and method == "llm" else None,
    )


def _cross_validated_predictions(
    labelled: list[LabelledAnnouncement], folds: int
) -> dict[int, ExtractionResult] | str:
    """Predict every item with a classifier that never saw it, or explain why that is not possible.

    Each class's items are dealt into folds in order, so the split is deterministic and every
    fold holds a share of every class. Returns the reason as a string if a training set would be
    too thin to learn from.
    """
    fold_of: dict[int, int] = {}
    counters: dict[str, int] = defaultdict(int)
    for index, item in enumerate(labelled):
        fold_of[index] = counters[item.label.value] % folds
        counters[item.label.value] += 1

    predictions: dict[int, ExtractionResult] = {}
    for fold in range(folds):
        train = [i for i in range(len(labelled)) if fold_of[i] != fold]
        test = [i for i in range(len(labelled)) if fold_of[i] == fold]
        try:
            classifier = AnnouncementClassifier.train(
                [labelled[i].text for i in train], [labelled[i].label for i in train]
            )
        except ValueError as exc:
            return f"cross-validation not possible: {exc}"
        for index in test:
            predictions[index] = classify_with_model(classifier, labelled[index].text)
    return predictions


def evaluate_announcement_methods(
    labelled: list[LabelledAnnouncement],
    *,
    llm_client: LLMClient | None = None,
    llm_model_name: str = "",
    cv_folds: int = DEFAULT_CV_FOLDS,
) -> AnnouncementAblation:
    """Score keyword rules, a cross-validated trained classifier, and (if given) an LLM.

    Raises:
        AblationError: if ``cv_folds`` is below 2.
    """
    if cv_folds < 2:
        raise AblationError("cv_folds must be at least 2")
    methods = [_score("keyword", labelled, lambda _i, item: classify_by_keywords(item.text))]

    cv = _cross_validated_predictions(labelled, cv_folds)
    if isinstance(cv, str):
        methods.append(MethodResult("trained_classifier", 0, 0, 0, {}, None, skipped_reason=cv))
    else:
        methods.append(_score("trained_classifier", labelled, lambda i, _item: cv[i]))

    if llm_client is None:
        methods.append(
            MethodResult("llm", 0, 0, 0, {}, None, skipped_reason="no LLM provider configured")
        )
    else:

        def call(_index: int, item: LabelledAnnouncement) -> ExtractionResult | None:
            try:
                return extract_with_llm(llm_client, item.text, model_name=llm_model_name)
            except LLMError as exc:
                LOGGER.warning("LLM failed on %s: %s", item.identifier, exc)
                return None

        methods.append(_score("llm", labelled, call))
    return AnnouncementAblation(labelled_count=len(labelled), methods=tuple(methods))


def evaluate_topic_discovery(session: Session, *, top_n: int = 5) -> TopicDiscoveryEvaluation:
    """Read the review queue to see what people made of the topics discovery proposed.

    Precision is accepted / (accepted + rejected); candidates still pending are excluded from
    it rather than counted either way. ``precision_top_n`` looks only at the ``top_n``
    candidates covering the most records, since those are the ones a reviewer sees first.

    Raises:
        ValueError: if ``top_n`` is not positive.
    """
    if top_n < 1:
        raise ValueError("top_n must be positive")
    items = list(
        session.scalars(select(ReviewQueueItem).where(ReviewQueueItem.queue_type == QUEUE_TYPE))
    )
    counts = Counter(item.status for item in items)
    accepted, rejected, pending = counts["accepted"], counts["rejected"], counts["pending"]

    def precision(subset: list[ReviewQueueItem]) -> float | None:
        yes = sum(1 for item in subset if item.status == "accepted")
        no = sum(1 for item in subset if item.status == "rejected")
        return yes / (yes + no) if (yes + no) else None

    top = sorted(
        items, key=lambda item: (item.payload_json or {}).get("record_count", 0), reverse=True
    )[:top_n]
    unmatched = int(
        session.scalar(
            select(func.count())
            .select_from(SourceRecord)
            .where(
                SourceRecord.duplicate_of_id.is_(None),
                ~select(RecordTopic.id)
                .where(RecordTopic.source_record_id == SourceRecord.id)
                .exists(),
            )
        )
        or 0
    )
    return TopicDiscoveryEvaluation(
        candidates=len(items),
        pending=pending,
        accepted=accepted,
        rejected=rejected,
        precision=precision(items),
        precision_top_n=precision(top),
        top_n=top_n,
        unmatched_records=unmatched,
        records_in_candidates=sum(
            (item.payload_json or {}).get("record_count", 0) for item in items
        ),
    )


def run_ai_ablation(
    session: Session,
    settings: Settings,
    labels_path: Path,
    *,
    llm_client: LLMClient | None = None,
    top_n: int = 5,
) -> AIAblationReport:
    """Evaluate every implemented AI layer on against off."""
    report = AIAblationReport()
    try:
        labelled = load_labelled_announcements(labels_path)
    except AblationError as exc:
        report.warnings.append(f"announcement extraction not evaluated: {exc}")
    else:
        report.announcements = evaluate_announcement_methods(
            labelled, llm_client=llm_client, llm_model_name=settings.llm_model
        )
    report.topic_discovery = evaluate_topic_discovery(session, top_n=top_n)
    if report.topic_discovery.candidates == 0:
        report.warnings.append(
            "no topic candidates yet; run: cews discover-topics, then review them with cews review"
        )
    return report
