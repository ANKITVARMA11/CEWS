"""Building a CrewAI ``LLM`` from CEWS's own settings.

CrewAI's ``LLM`` class is a thin wrapper over ``litellm``, which expects a ``model`` string
prefixed with a routing hint (``openai/...``, ``ollama/...``, ...) telling it which API shape to
speak, separately from ``base_url``/``api_key`` saying where to actually send it.

CEWS's own :class:`~cews.ai.llm.client.LLMClient` does not make that same distinction: both
``LLM_PROVIDER=ollama`` and ``LLM_PROVIDER=openai_compatible`` are treated identically there -
the same base URL, the same optional Bearer header, no provider-specific request shape at all
(Ollama has supported an OpenAI-compatible endpoint for a long time, which is exactly what
CEWS's own default ``LLM_BASE_URL`` already points at). This module keeps that same choice
instead of inventing a second convention: every CEWS provider is treated as "some OpenAI-shaped
endpoint," and the model is always sent to litellm as ``openai/<model>``.
"""

from __future__ import annotations

from crewai import LLM

from cews.ai.llm import llm_enabled
from cews.settings import Settings


def build_crew_llm(settings: Settings) -> LLM:
    """A CrewAI ``LLM`` pointed at whatever CEWS's own settings already configure.

    Raises:
        ValueError: if no LLM is configured (``LLM_PROVIDER=none``) - callers should check
            :func:`cews.ai.llm.llm_enabled` first for a clearer message than litellm's own.
    """
    if not llm_enabled(settings):
        raise ValueError("no LLM is configured (LLM_PROVIDER=none); set LLM_PROVIDER and LLM_MODEL")
    api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else "not-needed"
    return LLM(
        model=f"openai/{settings.llm_model}",
        base_url=settings.llm_base_url,
        api_key=api_key,
    )
