"""A small, provider-agnostic LLM client.

CEWS calls at most one kind of thing here: "given this text and this JSON schema, extract these
fields." Nothing about that needs a heavyweight SDK, and depending on one would mean a different
dependency per provider. Instead this speaks the OpenAI chat-completions HTTP format directly,
which Ollama, NVIDIA NIM, Groq, Together and most other hosted or local servers all implement.

Switching provider is a configuration change (``LLM_PROVIDER`` in ``.env``), never a code change.
With ``LLM_PROVIDER=none`` (the default) this module is never imported by the pipeline; every
caller is expected to check :func:`llm_enabled` first and fall back to a deterministic method
when it is False.
"""

from __future__ import annotations

from cews.ai.llm.client import LLMClient, LLMError, LLMResponse, llm_enabled

__all__ = ["LLMClient", "LLMError", "LLMResponse", "llm_enabled"]
