# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""End-to-end routing tests for the new sampling parameters.

Pins the per-provider gating contract added by the
expose-sampling-params PR: each of `frequency_penalty`, `seed`, `stop`
/ `stop_sequences`, `service_tier`, `parallel_tool_calls` only appears
on the outbound body when the upstream provider actually accepts it.

The provider matrix is captured per docs:
- Anthropic Messages: accepts stop_sequences, service_tier
  (auto|standard_only), disable_parallel_tool_use (inverted). REJECTS
  frequency_penalty, seed, logprobs (silently dropped client-side).
- OpenAI Chat Completions (default OAI-compat branch): accepts every
  field; OpenAI cloud uses `max_completion_tokens` rather than
  `max_tokens`.
- OpenAI Responses (gpt-5.x / o3): rejects temperature, top_p,
  frequency_penalty, seed, stop, logprobs. Accepts service_tier
  (auto|default|flex|priority) and parallel_tool_calls.
"""

import asyncio
import json

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _install_mock(monkeypatch, *, sse_payload: bytes | None = None) -> dict:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        try:
            captured["body"] = json.loads(request.content.decode("utf-8"))
        except json.JSONDecodeError:
            captured["body"] = None
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        return httpx.Response(
            200,
            content = sse_payload
            or (b'event: message_stop\ndata: {"type":"message_stop"}\n\n'),
            headers = {"content-type": "text/event-stream"},
        )

    monkeypatch.setattr(
        ep_mod,
        "_http_client",
        httpx.AsyncClient(transport = httpx.MockTransport(handler)),
    )
    return captured


# ── Anthropic ──────────────────────────────────────────────────────────


def _drive_anthropic(captured, **kwargs) -> dict:
    async def run():
        client = ExternalProviderClient(
            provider_type = "anthropic",
            base_url = "https://api.anthropic.com/v1",
            api_key = "sk-ant-test",
        )
        async for _ in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            model = "claude-opus-4-7",
            temperature = 0.7,
            top_p = 0.95,
            max_tokens = 64,
            **kwargs,
        ):
            pass
        await client.close()

    _drive(run())
    return captured["body"]


def test_anthropic_stop_sequences_forwarded_as_renamed_field(monkeypatch):
    captured = _install_mock(monkeypatch)
    body = _drive_anthropic(captured, stop = ["END", "DONE"])
    assert body.get("stop_sequences") == ["END", "DONE"], body
    # Anthropic does not have a `stop` field; the unrenamed key must not appear.
    assert "stop" not in body, body


def test_anthropic_single_string_stop_is_wrapped(monkeypatch):
    captured = _install_mock(monkeypatch)
    body = _drive_anthropic(captured, stop = "STOPHERE")
    assert body.get("stop_sequences") == ["STOPHERE"], body


def test_anthropic_empty_stop_omitted(monkeypatch):
    captured = _install_mock(monkeypatch)
    body = _drive_anthropic(captured, stop = [])
    assert "stop_sequences" not in body, body
    assert "stop" not in body, body


def test_anthropic_service_tier_forwarded_when_valid(monkeypatch):
    captured = _install_mock(monkeypatch)
    body = _drive_anthropic(captured, service_tier = "standard_only")
    assert body.get("service_tier") == "standard_only", body


@pytest.mark.parametrize(
    "bogus", ["flex", "priority", "scale", "default", "", "auto-foo"]
)
def test_anthropic_service_tier_unsupported_values_dropped(monkeypatch, bogus):
    captured = _install_mock(monkeypatch)
    body = _drive_anthropic(captured, service_tier = bogus)
    assert "service_tier" not in body, body


def test_anthropic_disable_parallel_tool_use_only_when_false(monkeypatch):
    captured = _install_mock(monkeypatch)
    body = _drive_anthropic(captured, parallel_tool_calls = False)
    assert body.get("disable_parallel_tool_use") is True, body
    # Anthropic has no `parallel_tool_calls` field.
    assert "parallel_tool_calls" not in body, body


def test_anthropic_parallel_tool_calls_default_not_sent(monkeypatch):
    captured = _install_mock(monkeypatch)
    body = _drive_anthropic(captured, parallel_tool_calls = True)
    # True is the upstream default; do not surface
    # `disable_parallel_tool_use: false` which would over-specify the request.
    assert "disable_parallel_tool_use" not in body, body
    assert "parallel_tool_calls" not in body, body


def test_anthropic_rejects_openai_only_knobs(monkeypatch):
    """frequency_penalty / seed are dropped at the dispatch layer.

    Anthropic has no equivalent; the keyword args are not even forwarded
    from stream_chat_completion to _stream_anthropic. This test pins
    that no such field reaches the Messages body.
    """
    captured = _install_mock(monkeypatch)
    body = _drive_anthropic(
        captured,
        frequency_penalty = 1.5,
        seed = 42,
    )
    assert "frequency_penalty" not in body, body
    assert "seed" not in body, body


# ── OpenAI Chat Completions (default OAI-compat) ─────────────────────────


def _drive_openai_compat(captured, **kwargs) -> dict:
    """Send through the default OAI-compat branch (NOT /v1/responses).

    Use a non-OpenAI provider_type so the dispatcher takes the default
    branch at the bottom of stream_chat_completion rather than the
    Responses translator path that routes provider_type=="openai".
    """

    async def run():
        client = ExternalProviderClient(
            provider_type = "mistral",
            base_url = "https://api.mistral.ai/v1",
            api_key = "test-key",
        )
        # mistral's OpenAI-compat /v1/chat/completions returns OpenAI
        # SSE; a single DONE frame is enough to drain the stream.
        async for _ in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            model = "mistral-small-latest",
            temperature = 0.5,
            top_p = 0.9,
            max_tokens = 64,
            **kwargs,
        ):
            pass
        await client.close()

    _drive(run())
    return captured["body"]


def _oai_done_payload() -> bytes:
    return b"data: [DONE]\n\n"


def test_openai_compat_forwards_frequency_penalty(monkeypatch):
    captured = _install_mock(monkeypatch, sse_payload = _oai_done_payload())
    body = _drive_openai_compat(captured, frequency_penalty = 1.25)
    assert body.get("frequency_penalty") == 1.25, body


def test_openai_compat_forwards_seed(monkeypatch):
    captured = _install_mock(monkeypatch, sse_payload = _oai_done_payload())
    body = _drive_openai_compat(captured, seed = 12345)
    assert body.get("seed") == 12345, body


def test_openai_compat_forwards_stop_array(monkeypatch):
    captured = _install_mock(monkeypatch, sse_payload = _oai_done_payload())
    body = _drive_openai_compat(captured, stop = ["END", "DONE"])
    assert body.get("stop") == ["END", "DONE"], body
    # The default OAI-compat branch does not rename to stop_sequences.
    assert "stop_sequences" not in body, body


def test_openai_compat_truncates_stop_to_four(monkeypatch):
    captured = _install_mock(monkeypatch, sse_payload = _oai_done_payload())
    body = _drive_openai_compat(captured, stop = ["a", "b", "c", "d", "e", "f"])
    assert body.get("stop") == ["a", "b", "c", "d"], body


def test_openai_compat_empty_stop_omitted(monkeypatch):
    captured = _install_mock(monkeypatch, sse_payload = _oai_done_payload())
    body = _drive_openai_compat(captured, stop = [])
    assert "stop" not in body, body


def test_openai_compat_forwards_service_tier(monkeypatch):
    captured = _install_mock(monkeypatch, sse_payload = _oai_done_payload())
    body = _drive_openai_compat(captured, service_tier = "flex")
    assert body.get("service_tier") == "flex", body


def test_openai_compat_forwards_parallel_tool_calls(monkeypatch):
    captured = _install_mock(monkeypatch, sse_payload = _oai_done_payload())
    body = _drive_openai_compat(captured, parallel_tool_calls = False)
    assert body.get("parallel_tool_calls") is False, body


def test_openai_compat_omits_unset_optionals(monkeypatch):
    captured = _install_mock(monkeypatch, sse_payload = _oai_done_payload())
    body = _drive_openai_compat(captured)
    # Optional knobs default to None / unset -> never appear.
    assert "frequency_penalty" not in body, body
    assert "seed" not in body, body
    assert "stop" not in body, body
    assert "service_tier" not in body, body
    assert "parallel_tool_calls" not in body, body


# ── OpenAI Responses (gpt-5.x via /v1/responses) ─────────────────────────


def _responses_done_payload() -> bytes:
    return (
        b"event: response.completed\n"
        b'data: {"type":"response.completed","response":{"usage":{}}}\n\n'
    )


def _drive_openai_responses(captured, **kwargs) -> dict:
    async def run():
        client = ExternalProviderClient(
            provider_type = "openai",
            base_url = "https://api.openai.com/v1",
            api_key = "sk-test",
        )
        async for _ in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            model = "gpt-5.5",
            temperature = 1.0,
            top_p = 1.0,
            max_tokens = 64,
            **kwargs,
        ):
            pass
        await client.close()

    _drive(run())
    return captured["body"]


def test_openai_responses_drops_temperature_top_p(monkeypatch):
    captured = _install_mock(monkeypatch, sse_payload = _responses_done_payload())
    body = _drive_openai_responses(captured)
    assert "temperature" not in body, body
    assert "top_p" not in body, body


def test_openai_responses_drops_frequency_penalty_seed_stop(monkeypatch):
    captured = _install_mock(monkeypatch, sse_payload = _responses_done_payload())
    body = _drive_openai_responses(
        captured,
        frequency_penalty = 1.5,
        seed = 99,
        stop = ["END"],
    )
    # Responses 400s on any of these; the dispatch must drop them
    # before they hit the wire.
    assert "frequency_penalty" not in body, body
    assert "seed" not in body, body
    assert "stop" not in body, body
    assert "stop_sequences" not in body, body


def test_openai_responses_forwards_service_tier(monkeypatch):
    captured = _install_mock(monkeypatch, sse_payload = _responses_done_payload())
    body = _drive_openai_responses(captured, service_tier = "priority")
    assert body.get("service_tier") == "priority", body


def test_openai_responses_rejects_chat_only_service_tier(monkeypatch):
    captured = _install_mock(monkeypatch, sse_payload = _responses_done_payload())
    body = _drive_openai_responses(captured, service_tier = "scale")
    # Responses only accepts auto|default|flex|priority -- `scale` is
    # silently dropped so a stale frontend cannot 400 the request.
    assert "service_tier" not in body, body


def test_openai_responses_forwards_parallel_tool_calls(monkeypatch):
    captured = _install_mock(monkeypatch, sse_payload = _responses_done_payload())
    body = _drive_openai_responses(captured, parallel_tool_calls = False)
    assert body.get("parallel_tool_calls") is False, body


def test_openai_responses_omits_unset_optionals(monkeypatch):
    captured = _install_mock(monkeypatch, sse_payload = _responses_done_payload())
    body = _drive_openai_responses(captured)
    assert "service_tier" not in body, body
    assert "parallel_tool_calls" not in body, body


# ── Schema-level smoke tests ─────────────────────────────────────────────


def test_chat_completion_request_accepts_new_sampling_fields():
    from models.inference import ChatCompletionRequest

    payload = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "frequency_penalty": -1.0,
            "seed": 0,
            "stop": ["END"],
            "service_tier": "auto",
            "parallel_tool_calls": True,
        }
    )
    assert payload.frequency_penalty == -1.0
    assert payload.seed == 0
    assert payload.stop == ["END"]
    assert payload.service_tier == "auto"
    assert payload.parallel_tool_calls is True


def test_chat_completion_request_rejects_bad_service_tier():
    import pydantic
    from models.inference import ChatCompletionRequest

    with pytest.raises(pydantic.ValidationError):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "service_tier": "bogus",
            }
        )


def test_chat_completion_request_clamps_frequency_penalty_range():
    import pydantic
    from models.inference import ChatCompletionRequest

    with pytest.raises(pydantic.ValidationError):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "frequency_penalty": 3.0,
            }
        )
    with pytest.raises(pydantic.ValidationError):
        ChatCompletionRequest.model_validate(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "frequency_penalty": -3.0,
            }
        )
