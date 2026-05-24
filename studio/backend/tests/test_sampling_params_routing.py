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


def test_anthropic_stop_sequences_dedup_and_drop_whitespace(monkeypatch):
    """Anthropic 400s on any stop sequence that contains no non-
    whitespace character (`stop_sequences: each stop sequence must
    contain non-whitespace`). Empty strings, " ", "\\n", "\\n\\n", and
    other whitespace-only chips are filtered out client-side so the
    request reaches the wire. Duplicates are deduped to avoid wasting
    slots against the cap.
    """
    captured = _install_mock(monkeypatch)
    body = _drive_anthropic(
        captured,
        stop = ["END", "", "END", "DONE", "  ", "END", "\n\n", "\t"],
    )
    # Order preserved on first sight, duplicates + every whitespace-only
    # entry dropped.
    assert body.get("stop_sequences") == ["END", "DONE"], body


def test_anthropic_single_whitespace_stop_string_dropped(monkeypatch):
    """Single-string stop="\\n\\n" must not reach the wire either."""
    captured = _install_mock(monkeypatch)
    body = _drive_anthropic(captured, stop = "\n\n")
    assert "stop_sequences" not in body, body


def test_anthropic_stop_sequences_truncated_to_16(monkeypatch):
    captured = _install_mock(monkeypatch)
    body = _drive_anthropic(captured, stop = [f"S{i}" for i in range(20)])
    assert len(body.get("stop_sequences", [])) == 16, body
    assert body["stop_sequences"][0] == "S0"
    assert body["stop_sequences"][-1] == "S15"


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


def _drive_anthropic_with_tools(captured, **kwargs) -> dict:
    """Same as `_drive_anthropic` but enables a server-side tool
    (`web_search`) so the request body carries `tools`. Needed to
    exercise the `disable_parallel_tool_use` nesting path, which only
    fires when there is at least one tool defined.
    """
    enabled_tools = kwargs.pop("enabled_tools", None) or ["web_search"]
    return _drive_anthropic(captured, enabled_tools = enabled_tools, **kwargs)


def test_anthropic_disable_parallel_tool_use_nested_under_tool_choice(monkeypatch):
    """`disable_parallel_tool_use` must be a property of `tool_choice`,
    NOT a top-level body field. Top-level placement is rejected with
    `extraneous key [disable_parallel_tool_use] is not permitted`. See
    https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use.
    """
    captured = _install_mock(monkeypatch)
    body = _drive_anthropic_with_tools(captured, parallel_tool_calls = False)
    # Top-level placement is rejected with 400.
    assert "disable_parallel_tool_use" not in body, body
    assert "parallel_tool_calls" not in body, body
    # Flag lives on tool_choice; default type is "auto".
    tc = body.get("tool_choice")
    assert isinstance(tc, dict), body
    assert tc.get("disable_parallel_tool_use") is True, body
    assert tc.get("type") == "auto", body


def test_anthropic_disable_parallel_tool_use_skipped_without_tools(monkeypatch):
    """Without tools the flag is a no-op upstream; keep the body
    minimal and never emit it at top level either.
    """
    captured = _install_mock(monkeypatch)
    body = _drive_anthropic(captured, parallel_tool_calls = False)
    assert "disable_parallel_tool_use" not in body, body
    assert "parallel_tool_calls" not in body, body
    assert "tool_choice" not in body, body


def test_anthropic_parallel_tool_calls_default_not_sent(monkeypatch):
    captured = _install_mock(monkeypatch)
    body = _drive_anthropic_with_tools(captured, parallel_tool_calls = True)
    # True is the upstream default; do not surface a tool_choice we
    # would otherwise not have set, and definitely no top-level
    # `disable_parallel_tool_use`.
    assert "disable_parallel_tool_use" not in body, body
    assert "parallel_tool_calls" not in body, body
    tc = body.get("tool_choice")
    if isinstance(tc, dict):
        assert "disable_parallel_tool_use" not in tc, body


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
    # Default OAI-compat provider (mistral here) renames seed to
    # random_seed via provider registry's seed_field.
    assert body.get("random_seed") == 12345, body
    assert "seed" not in body, body


def test_openai_compat_seed_field_default_is_seed(monkeypatch):
    """Providers without a seed_field override get the OpenAI default."""
    captured = _install_mock(monkeypatch, sse_payload = _oai_done_payload())

    async def run():
        client = ExternalProviderClient(
            provider_type = "deepseek",
            base_url = "https://api.deepseek.com/v1",
            api_key = "ds-test",
        )
        async for _ in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            model = "deepseek-chat",
            temperature = 0.5,
            top_p = 0.9,
            max_tokens = 64,
            seed = 7,
        ):
            pass
        await client.close()

    _drive(run())
    body = captured["body"]
    assert body.get("seed") == 7, body
    assert "random_seed" not in body, body


def test_openai_compat_deepseek_stop_cap_is_16(monkeypatch):
    """DeepSeek docs allow up to 16 stop sequences; the previous
    4-cap silently truncated valid configs."""
    captured = _install_mock(monkeypatch, sse_payload = _oai_done_payload())

    async def run():
        client = ExternalProviderClient(
            provider_type = "deepseek",
            base_url = "https://api.deepseek.com/v1",
            api_key = "ds-test",
        )
        async for _ in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            model = "deepseek-chat",
            temperature = 0.5,
            top_p = 0.9,
            max_tokens = 64,
            stop = [f"S{i}" for i in range(20)],
        ):
            pass
        await client.close()

    _drive(run())
    body = captured["body"]
    assert len(body.get("stop", [])) == 16, body


def test_openai_compat_forwards_stop_array(monkeypatch):
    captured = _install_mock(monkeypatch, sse_payload = _oai_done_payload())
    body = _drive_openai_compat(captured, stop = ["END", "DONE"])
    assert body.get("stop") == ["END", "DONE"], body
    # The default OAI-compat branch does not rename to stop_sequences.
    assert "stop_sequences" not in body, body


def test_openai_compat_truncates_stop_to_default_cap(monkeypatch):
    """Default OAI-compat cap is 16 (DeepSeek and Mistral both accept
    that many); only OpenAI Chat has a tighter 4-entry hard limit."""
    captured = _install_mock(monkeypatch, sse_payload = _oai_done_payload())
    body = _drive_openai_compat(captured, stop = [f"s{i}" for i in range(20)])
    assert len(body.get("stop", [])) == 16, body
    assert body["stop"][0] == "s0"
    assert body["stop"][-1] == "s15"


def test_openai_compat_stop_dedup_and_drop_empties(monkeypatch):
    """Duplicates and empties shouldn't eat into the cap."""
    captured = _install_mock(monkeypatch, sse_payload = _oai_done_payload())
    body = _drive_openai_compat(captured, stop = ["END", "", "END", "DONE", "FIN", "END"])
    assert body.get("stop") == ["END", "DONE", "FIN"], body


def test_openai_compat_empty_stop_omitted(monkeypatch):
    captured = _install_mock(monkeypatch, sse_payload = _oai_done_payload())
    body = _drive_openai_compat(captured, stop = [])
    assert "stop" not in body, body


def test_openai_compat_drops_service_tier_by_default(monkeypatch):
    """Generic OAI-compat providers (mistral, deepseek, openrouter, ...)
    do not document a `service_tier` field. The dispatcher must drop
    it unless the provider registry explicitly opts in with
    `accepts_service_tier=True`; otherwise a stale frontend could
    smuggle Anthropic/OpenAI-Responses-only values onto unrelated
    providers."""
    captured = _install_mock(monkeypatch, sse_payload = _oai_done_payload())
    body = _drive_openai_compat(captured, service_tier = "flex")
    assert "service_tier" not in body, body


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


@pytest.mark.parametrize("value", ["auto", "default", "flex", "priority"])
def test_openai_responses_forwards_documented_service_tiers(monkeypatch, value):
    """The live OpenAI Responses API reference lists `service_tier` as
    `auto|default|flex|priority` for /v1/responses. Pin that every value
    in the documented enum forwards untouched."""
    captured = _install_mock(monkeypatch, sse_payload = _responses_done_payload())
    body = _drive_openai_responses(captured, service_tier = value)
    assert body.get("service_tier") == value, body


@pytest.mark.parametrize("bogus", ["scale", "standard_only", "bogus", ""])
def test_openai_responses_drops_undocumented_service_tier(monkeypatch, bogus):
    """`scale` and `standard_only` are not in the documented Responses
    request enum; drop them client-side so a stale frontend never
    sends an upstream-rejected value."""
    captured = _install_mock(monkeypatch, sse_payload = _responses_done_payload())
    body = _drive_openai_responses(captured, service_tier = bogus)
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


# ── Kimi web-search bypass forwards new sampling fields ────────────────


def test_kimi_web_search_bypass_forwards_new_sampling_fields(monkeypatch):
    """The Kimi $web_search path takes an early return into
    `_stream_kimi_web_search` before the default OAI-compat body
    builder runs; forwarding here keeps Kimi-with-search and
    Kimi-without-search in lockstep."""
    captured = _install_mock(monkeypatch, sse_payload = _oai_done_payload())

    async def run():
        client = ExternalProviderClient(
            provider_type = "kimi",
            base_url = "https://api.moonshot.ai/v1",
            api_key = "kimi-test",
        )
        async for _ in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            model = "kimi-k2.6",
            temperature = 1.0,
            top_p = 1.0,
            max_tokens = 256,
            enabled_tools = ["web_search"],
            presence_penalty = 0.5,
            frequency_penalty = 1.25,
            seed = 7,
            stop = ["END"],
            parallel_tool_calls = False,
        ):
            pass
        await client.close()

    _drive(run())
    body = captured["body"]
    # Kimi locks these: stripped by body_omit in providers.py.
    assert "frequency_penalty" not in body, body
    assert "temperature" not in body, body
    assert "top_p" not in body, body
    assert "seed" not in body, body
    assert "parallel_tool_calls" not in body, body
    # Knobs not on Kimi's drop-list forward through the bypass.
    assert body.get("stop") == ["END"], body
    assert body.get("presence_penalty") == 0.5, body


def test_kimi_web_search_uses_kimi_stop_cap_5(monkeypatch):
    """Kimi documents a 5-stop max; the web-search bypass must honour
    `provider_info["stop_max"]` rather than the OpenAI 4-cap or the
    permissive default."""
    captured = _install_mock(monkeypatch, sse_payload = _oai_done_payload())

    async def run():
        client = ExternalProviderClient(
            provider_type = "kimi",
            base_url = "https://api.moonshot.ai/v1",
            api_key = "kimi-test",
        )
        async for _ in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            model = "kimi-k2.6",
            temperature = 1.0,
            top_p = 1.0,
            max_tokens = 256,
            enabled_tools = ["web_search"],
            stop = [f"S{i}" for i in range(10)],
        ):
            pass
        await client.close()

    _drive(run())
    body = captured["body"]
    assert len(body.get("stop", [])) == 5, body
    assert body["stop"] == ["S0", "S1", "S2", "S3", "S4"], body


def test_openrouter_stop_cap_is_4(monkeypatch):
    """OpenRouter normalises to OpenAI's chat schema and inherits the
    4-entry stop cap; the default 16-cap is too permissive for it."""
    captured = _install_mock(monkeypatch, sse_payload = _oai_done_payload())

    async def run():
        client = ExternalProviderClient(
            provider_type = "openrouter",
            base_url = "https://openrouter.ai/api/v1",
            api_key = "or-test",
        )
        async for _ in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            model = "openai/gpt-4o",
            temperature = 0.5,
            top_p = 0.9,
            max_tokens = 64,
            stop = [f"S{i}" for i in range(10)],
        ):
            pass
        await client.close()

    _drive(run())
    body = captured["body"]
    assert len(body.get("stop", [])) == 4, body
    assert body["stop"] == ["S0", "S1", "S2", "S3"], body


def test_kimi_drops_stop_strings_over_32_bytes(monkeypatch):
    """Kimi limits each stop string to <= 32 bytes per
    https://platform.kimi.ai/docs/api/chat. Drop overlong entries
    client-side so a stale UI cannot 400 the request."""
    captured = _install_mock(monkeypatch, sse_payload = _oai_done_payload())

    async def run():
        client = ExternalProviderClient(
            provider_type = "kimi",
            base_url = "https://api.moonshot.ai/v1",
            api_key = "kimi-test",
        )
        async for _ in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            model = "kimi-k2.6",
            temperature = 1.0,
            top_p = 1.0,
            max_tokens = 256,
            stop = ["END", "x" * 33, "DONE"],
        ):
            pass
        await client.close()

    _drive(run())
    body = captured["body"]
    assert body.get("stop") == ["END", "DONE"], body


def test_kimi_web_search_drops_stop_strings_over_32_bytes(monkeypatch):
    """Same byte cap applies to the Kimi web-search bypass."""
    captured = _install_mock(monkeypatch, sse_payload = _oai_done_payload())

    async def run():
        client = ExternalProviderClient(
            provider_type = "kimi",
            base_url = "https://api.moonshot.ai/v1",
            api_key = "kimi-test",
        )
        async for _ in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            model = "kimi-k2.6",
            temperature = 1.0,
            top_p = 1.0,
            max_tokens = 256,
            enabled_tools = ["web_search"],
            stop = ["END", "x" * 40, "DONE"],
        ):
            pass
        await client.close()

    _drive(run())
    body = captured["body"]
    assert body.get("stop") == ["END", "DONE"], body


def test_kimi_default_path_uses_kimi_stop_cap_5(monkeypatch):
    """The normal Kimi path must also honour the documented 5-cap."""
    captured = _install_mock(monkeypatch, sse_payload = _oai_done_payload())

    async def run():
        client = ExternalProviderClient(
            provider_type = "kimi",
            base_url = "https://api.moonshot.ai/v1",
            api_key = "kimi-test",
        )
        async for _ in client.stream_chat_completion(
            messages = [{"role": "user", "content": "hi"}],
            model = "kimi-k2.6",
            temperature = 1.0,
            top_p = 1.0,
            max_tokens = 256,
            stop = [f"S{i}" for i in range(10)],
        ):
            pass
        await client.close()

    _drive(run())
    body = captured["body"]
    assert len(body.get("stop", [])) == 5, body


# ── Local OpenAI passthrough forwards new sampling fields ──────────────


def test_local_openai_passthrough_forwards_new_sampling_fields():
    """`_build_openai_passthrough_body` forwards frequency_penalty,
    seed, stop, and parallel_tool_calls to llama-server."""
    from models.inference import ChatCompletionRequest
    from routes.inference import _build_openai_passthrough_body

    payload = ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "frequency_penalty": 1.25,
            "seed": 123,
            "stop": ["END"],
            "parallel_tool_calls": False,
        }
    )
    body = _build_openai_passthrough_body(payload, backend_ctx = 4096)
    assert body["frequency_penalty"] == 1.25, body
    assert body["seed"] == 123, body
    assert body["stop"] == ["END"], body
    assert body["parallel_tool_calls"] is False, body


# ── Responses → ChatCompletions bridge preserves parallel_tool_calls ──


def test_responses_to_chat_bridge_preserves_parallel_tool_calls():
    """`_build_chat_request` (the /v1/responses to /v1/chat/completions
    translator) must forward parallel_tool_calls so a Responses-API
    caller's preference reaches llama-server."""
    from models.inference import ChatMessage, ResponsesRequest
    from routes.inference import _build_chat_request, _build_openai_passthrough_body

    payload = ResponsesRequest(
        input = "hi",
        stream = True,
        parallel_tool_calls = False,
    )
    chat_req = _build_chat_request(
        payload,
        [ChatMessage(role = "user", content = "hi")],
        stream = True,
    )
    assert chat_req.parallel_tool_calls is False, chat_req
    body = _build_openai_passthrough_body(chat_req, backend_ctx = 4096)
    assert body["parallel_tool_calls"] is False, body


def test_responses_to_chat_bridge_omits_unset_parallel_tool_calls():
    """Unset parallel_tool_calls (None) must not appear on the
    translated body; the upstream default is true everywhere so
    forwarding None would over-specify."""
    from models.inference import ChatMessage, ResponsesRequest
    from routes.inference import _build_chat_request, _build_openai_passthrough_body

    payload = ResponsesRequest(input = "hi", stream = True)
    chat_req = _build_chat_request(
        payload,
        [ChatMessage(role = "user", content = "hi")],
        stream = True,
    )
    assert chat_req.parallel_tool_calls is None, chat_req
    body = _build_openai_passthrough_body(chat_req, backend_ctx = 4096)
    assert "parallel_tool_calls" not in body, body


# ── Backend ChatInferenceSettings schema accepts new fields ────────────


def test_chat_settings_payload_accepts_new_sampling_keys():
    """ChatSettingsPayload has extra="forbid" so the new keys must be
    listed explicitly; otherwise every settings save with any of them
    422s. Pin the round-trip."""
    from routes.chat_history import ChatSettingsPayload

    parsed = ChatSettingsPayload.model_validate(
        {
            "inferenceParams": {
                "frequencyPenalty": 0.7,
                "seed": 42,
                "stop": ["END"],
                "serviceTier": "standard_only",
                "parallelToolCalls": False,
            }
        }
    )
    ip = parsed.inferenceParams
    assert ip is not None
    assert ip.frequencyPenalty == 0.7
    assert ip.seed == 42
    assert ip.stop == ["END"]
    assert ip.serviceTier == "standard_only"
    assert ip.parallelToolCalls is False


# ── Local /v1/messages: disable_parallel_tool_use translation ──────────


def test_local_anthropic_disable_parallel_tool_use_translation():
    """Anthropic nests `disable_parallel_tool_use` under `tool_choice`
    (per docs.claude.com). The local /v1/messages GGUF tool path must
    invert it into OpenAI-shaped `parallel_tool_calls` so third-party
    clients (Claude SDK, LiteLLM in passthrough mode) opt out of
    parallel calls successfully even on the local model."""

    # Mirror the extraction logic in routes/inference.py:anthropic_messages.
    def _extract(tc):
        if isinstance(tc, dict):
            v = tc.get("disable_parallel_tool_use")
            if isinstance(v, bool):
                return not v
        return None

    assert _extract({"type": "auto", "disable_parallel_tool_use": True}) is False
    assert _extract({"type": "any", "disable_parallel_tool_use": False}) is True
    assert _extract({"type": "auto"}) is None
    assert _extract(None) is None
    assert _extract("auto") is None  # string form (non-dict) → no opinion
    assert _extract({"type": "auto", "disable_parallel_tool_use": "yes"}) is None
