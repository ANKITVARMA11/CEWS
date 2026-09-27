"""Unit tests for the provider-agnostic LLM client.

Every provider CEWS supports speaks the same OpenAI chat-completions HTTP shape, so these tests
exercise that shape once against a fixture transport rather than hitting a real network. No test
here makes a real HTTP call.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from cews.ai.llm import LLMClient, LLMError, LLMResponse, llm_enabled
from cews.settings import load_settings

pytestmark = pytest.mark.unit


def settings(**overrides: Any) -> Any:
    return load_settings(env_file=None, overrides=overrides)


def echo_handler(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "prompt": body["messages"][-1]["content"],
                                "system": (
                                    body["messages"][0]["content"]
                                    if len(body["messages"]) > 1
                                    else None
                                ),
                                "authorization": request.headers.get("authorization"),
                                "model": body["model"],
                                "temperature": body.get("temperature"),
                            }
                        )
                    }
                }
            ]
        },
    )


def client_for(overrides: dict[str, Any], handler: Any = echo_handler) -> LLMClient:
    return LLMClient(settings(**overrides), transport=httpx.MockTransport(handler))


# --------------------------------------------------------------------------------------
# llm_enabled and construction
# --------------------------------------------------------------------------------------
def test_disabled_by_default() -> None:
    assert llm_enabled(settings()) is False


def test_enabled_once_a_provider_is_chosen() -> None:
    assert llm_enabled(settings(llm_provider="ollama")) is True
    assert llm_enabled(settings(llm_provider="openai_compatible")) is True


def test_constructing_a_client_with_no_provider_is_refused() -> None:
    with pytest.raises(LLMError, match="check llm_enabled"):
        LLMClient(settings(llm_provider="none"))


# --------------------------------------------------------------------------------------
# The request shape
# --------------------------------------------------------------------------------------
def test_a_request_carries_the_configured_model_and_prompt() -> None:
    with client_for(
        {"llm_provider": "ollama", "llm_base_url": "http://x/v1", "llm_model": "qwen2.5:3b"}
    ) as client:
        response = client.complete("summarize this")
    payload = response.json()
    assert payload["prompt"] == "summarize this"
    assert payload["model"] == "qwen2.5:3b"
    assert payload["system"] is None


def test_a_system_message_is_sent_first() -> None:
    with client_for({"llm_provider": "ollama", "llm_base_url": "http://x/v1"}) as client:
        response = client.complete("extract fields", system="you are an extractor")
    assert response.json()["system"] == "you are an extractor"


def test_temperature_defaults_to_zero_for_deterministic_extraction() -> None:
    with client_for({"llm_provider": "ollama", "llm_base_url": "http://x/v1"}) as client:
        response = client.complete("x")
    assert response.json()["temperature"] == 0.0


# --------------------------------------------------------------------------------------
# Local vs hosted: same code path, different auth
# --------------------------------------------------------------------------------------
def test_ollama_sends_no_authorization_header() -> None:
    with client_for({"llm_provider": "ollama", "llm_base_url": "http://x/v1"}) as client:
        response = client.complete("x")
    assert response.json()["authorization"] is None


def test_a_hosted_provider_sends_a_bearer_token() -> None:
    with client_for(
        {
            "llm_provider": "openai_compatible",
            "llm_base_url": "https://integrate.api.nvidia.com/v1",
            "llm_api_key": "sk-abc123",
        }
    ) as client:
        response = client.complete("x")
    assert response.json()["authorization"] == "Bearer sk-abc123"


def test_response_provider_and_model_are_recorded() -> None:
    with client_for(
        {"llm_provider": "openai_compatible", "llm_base_url": "https://x/v1", "llm_model": "m"}
    ) as client:
        response = client.complete("x")
    assert response.provider == "openai_compatible" and response.model == "m"


# --------------------------------------------------------------------------------------
# Secrets never leak
# --------------------------------------------------------------------------------------
def test_an_api_key_never_appears_in_an_error_message() -> None:
    def leaky_failure(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="Unauthorized: key sk-super-secret-999 rejected")

    client = client_for(
        {
            "llm_provider": "openai_compatible",
            "llm_base_url": "https://x/v1",
            "llm_api_key": "sk-super-secret-999",
            "llm_max_retries": 0,
        },
        handler=leaky_failure,
    )
    with client, pytest.raises(LLMError) as excinfo:
        client.complete("x")
    assert "sk-super-secret-999" not in str(excinfo.value)


def test_a_network_error_message_redacts_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    def raising(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused: sk-super-secret-999 in url somehow")

    client = client_for(
        {
            "llm_provider": "openai_compatible",
            "llm_base_url": "https://x/v1",
            "llm_api_key": "sk-super-secret-999",
            "llm_max_retries": 0,
        },
        handler=raising,
    )
    with client, pytest.raises(LLMError) as excinfo:
        client.complete("x")
    assert "sk-super-secret-999" not in str(excinfo.value)


# --------------------------------------------------------------------------------------
# Retries
# --------------------------------------------------------------------------------------
def test_a_transient_failure_is_retried_until_it_recovers() -> None:
    calls = {"count": 0}

    def flaky(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] < 3:
            return httpx.Response(503, text="overloaded")
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = client_for(
        {"llm_provider": "ollama", "llm_base_url": "http://x/v1", "llm_max_retries": 3},
        handler=flaky,
    )
    with client:
        response = client.complete("x")
    assert response.text == "ok" and calls["count"] == 3


def test_a_client_error_is_not_retried() -> None:
    calls = {"count": 0}

    def bad_request(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(400, text="malformed request")

    client = client_for(
        {"llm_provider": "ollama", "llm_base_url": "http://x/v1", "llm_max_retries": 3},
        handler=bad_request,
    )
    with client, pytest.raises(LLMError, match="HTTP 400"):
        client.complete("x")
    assert calls["count"] == 1


def test_retries_are_exhausted_and_then_reported() -> None:
    calls = {"count": 0}

    def always_503(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(503, text="down")

    client = client_for(
        {"llm_provider": "ollama", "llm_base_url": "http://x/v1", "llm_max_retries": 2},
        handler=always_503,
    )
    with client, pytest.raises(LLMError, match="HTTP 503"):
        client.complete("x")
    assert calls["count"] == 3  # the first attempt plus two retries


def test_a_timeout_is_reported_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    def timing_out(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out")

    client = client_for(
        {"llm_provider": "ollama", "llm_base_url": "http://x/v1", "llm_max_retries": 1},
        handler=timing_out,
    )
    with client, pytest.raises(LLMError, match="did not respond in time"):
        client.complete("x")


# --------------------------------------------------------------------------------------
# Malformed responses
# --------------------------------------------------------------------------------------
def test_an_unexpected_response_shape_is_reported() -> None:
    def wrong_shape(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"nothing": "useful"})

    client = client_for(
        {"llm_provider": "ollama", "llm_base_url": "http://x/v1"}, handler=wrong_shape
    )
    with client, pytest.raises(LLMError, match="unexpected response shape"):
        client.complete("x")


def test_a_non_json_body_is_reported() -> None:
    def not_json(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    client = client_for({"llm_provider": "ollama", "llm_base_url": "http://x/v1"}, handler=not_json)
    with client, pytest.raises(LLMError, match="invalid JSON"):
        client.complete("x")


def test_response_json_helper_parses_the_content() -> None:
    response = LLMResponse(text='{"a": 1}', model="m", provider="ollama", raw={})
    assert response.json() == {"a": 1}


def test_response_json_helper_reports_bad_json() -> None:
    response = LLMResponse(text="not json", model="m", provider="ollama", raw={})
    with pytest.raises(LLMError, match="not valid JSON"):
        response.json()


# --------------------------------------------------------------------------------------
# JSON schema is optional and provider-agnostic
# --------------------------------------------------------------------------------------
def test_a_json_schema_is_passed_through_when_given() -> None:
    captured: dict[str, Any] = {}

    def capturing(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    client = client_for(
        {"llm_provider": "ollama", "llm_base_url": "http://x/v1"}, handler=capturing
    )
    schema = {"type": "object", "properties": {"kind": {"type": "string"}}}
    with client:
        client.complete("x", json_schema=schema)
    assert captured["response_format"]["json_schema"]["schema"] == schema


def test_no_schema_means_no_response_format_field() -> None:
    captured: dict[str, Any] = {}

    def capturing(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    client = client_for(
        {"llm_provider": "ollama", "llm_base_url": "http://x/v1"}, handler=capturing
    )
    with client:
        client.complete("x")
    assert "response_format" not in captured
