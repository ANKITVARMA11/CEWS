"""Sentence embeddings for the unassigned-text pool.

One small pretrained model, used as-is (no fine-tuning): it turns text into vectors so that
"targeted protein degradation" and "PROTAC" land near each other without either word having been
written into the taxonomy. This is the only place `sentence-transformers` is imported, so the
whole optional dependency is contained here and the rest of the codebase never needs to know it
exists.

Encoding is cached on disk by content hash so a re-run does not re-embed text it has already
seen; embedding is the slow part of topic discovery, everything after it is cheap.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)

_MODEL_LOCK = threading.Lock()
_LOADED: dict[str, Any] = {}


class EmbeddingUnavailableError(RuntimeError):
    """Raised when embeddings cannot be produced: the library is missing or the model failed to load.

    Every caller is expected to catch this and fall back to a deterministic method rather than
    let it propagate to the pipeline.
    """


def embeddings_available() -> bool:
    """Whether the optional ``sentence-transformers`` dependency can be imported."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _load_model(model_name: str) -> Any:
    """Load (and cache in-process) the embedding model.

    Raises:
        EmbeddingUnavailableError: if the library is missing or the model cannot be loaded
            (typically no cached weights and no network to fetch them).
    """
    with _MODEL_LOCK:
        if model_name in _LOADED:
            return _LOADED[model_name]
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingUnavailableError(
                "sentence-transformers is not installed "
                "(python -m pip install -r requirements-ai.txt)"
            ) from exc
        try:
            model = SentenceTransformer(model_name)
        except Exception as exc:  # network failure, corrupt cache, unknown model name, ...
            raise EmbeddingUnavailableError(
                f"could not load embedding model {model_name!r}: {exc}"
            ) from exc
        _LOADED[model_name] = model
        return model


def _cache_key(model_name: str, text: str) -> str:
    digest = hashlib.sha256(f"{model_name}\x00{text}".encode()).hexdigest()
    return digest


class EmbeddingCache:
    """A disk cache of embeddings, keyed by model name and text content.

    Backed by a single ``.npz`` file so lookups are fast and the whole cache is one file to
    inspect or delete. Never grows unbounded on its own; call :meth:`prune` to drop entries for
    text that no longer exists.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._vectors: dict[str, np.ndarray] = {}
        self._dirty = False
        if path.is_file():
            with np.load(path) as data:
                self._vectors = {key: data[key] for key in data.files}

    def __len__(self) -> int:
        return len(self._vectors)

    def vectors_for(self, model_name: str, texts: list[str]) -> np.ndarray | None:
        """The vectors for every text, in order, or None if any is missing."""
        keys = [_cache_key(model_name, text) for text in texts]
        if any(key not in self._vectors for key in keys):
            return None
        return np.stack([self._vectors[key] for key in keys])

    def put_many(self, model_name: str, texts: list[str], vectors: np.ndarray) -> None:
        for text, vector in zip(texts, vectors, strict=True):
            self._vectors[_cache_key(model_name, text)] = vector
        self._dirty = True

    def save(self) -> None:
        """Write the cache to disk if anything changed."""
        if not self._dirty:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self._path, **self._vectors)  # type: ignore[arg-type]  # numpy stub gap
        self._dirty = False

    def prune(self, keep_texts: list[str], model_name: str) -> int:
        """Drop cached vectors for text that is no longer relevant. Returns how many were dropped."""
        keep = {_cache_key(model_name, text) for text in keep_texts}
        drop = [key for key in self._vectors if key not in keep]
        for key in drop:
            del self._vectors[key]
        if drop:
            self._dirty = True
        return len(drop)


def embed_texts(
    texts: list[str], *, model_name: str, cache: EmbeddingCache | None = None
) -> np.ndarray:
    """Embed a list of texts, using and populating ``cache`` if given.

    Returns an array of shape ``(len(texts), dimensions)``. Empty strings are embedded like any
    other text (the model handles them); callers filter empty text out beforehand if that is not
    wanted, since what counts as "not enough text to embed" is a judgement for the caller.

    Raises:
        EmbeddingUnavailableError: if the library or model cannot be used.
        ValueError: if ``texts`` is empty.
    """
    if not texts:
        raise ValueError("cannot embed an empty list of texts")
    if cache is None:
        model = _load_model(model_name)
        return np.asarray(model.encode(texts, show_progress_bar=False, convert_to_numpy=True))

    # Only the text the cache has not seen before is actually encoded, so a run that adds a
    # handful of new records to a large, mostly-unchanged corpus stays cheap.
    uncached = sorted({text for text in texts if cache.vectors_for(model_name, [text]) is None})
    if uncached:
        model = _load_model(model_name)
        fresh = np.asarray(model.encode(uncached, show_progress_bar=False, convert_to_numpy=True))
        cache.put_many(model_name, uncached, fresh)
    result = cache.vectors_for(model_name, texts)
    assert result is not None  # every text was just embedded or was already cached
    return result
