# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the prompt-cache accounting chunk from the external-provider proxy.

The streaming Anthropic + OpenAI Responses paths emit one extra include_usage
SSE chunk (``choices: []`` with a ``usage`` block) before ``[DONE]`` so clients
see cache savings. Covers the helper directly plus the Anthropic stream and the
OpenAI Responses completed/incomplete streams.
"""

import asyncio
import json

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_provider import (
    ExternalProviderClient,
    _build_usage_chunk,
)


# ── _build_usage_chunk unit tests ───────────────────────────────────


def test_build_usage_chunk_anthropic_shape():
    line = _build_usage_chunk(
        "chatcmpl-x",
        "anthropic",
        {
            "input_tokens": 8,
            "output_tokens": 862,
            "cache_creation_input_tokens": 1367,
            "cache_read_input_tokens": 18901,
        },
    )
    assert line is not None
    assert line.startswith("data: ")
    payload = json.loads(line[len("data: ") :])
    assert payload["id"] == "chatcmpl-x"
    assert payload["object"] == "chat.completion.chunk"
    assert payload["choices"] == []
    usage = payload["usage"]
    # Anthropic's input_tokens excludes cache buckets; prompt_tokens must
    # sum all three input components so downstream context/cost displays
    # see the real prompt size.
    assert usage["prompt_tokens"] == 8 + 1367 + 18901
    assert usage["completion_tokens"] == 862
    assert usage["total_tokens"] == 8 + 1367 + 18901 + 862
    assert usage["cache_creation_input_tokens"] == 1367
    assert usage["cache_read_input_tokens"] == 18901
    # OpenAI-style mirror for clients that key off prompt_tokens_details.
    assert usage["prompt_tokens_details"]["cached_tokens"] == 18901


def test_build_usage_chunk_openai_shape():
    line = _build_usage_chunk(
        "chatcmpl-y",
        "openai",
        {
            "input_tokens": 5507,
            "output_tokens": 252,
            "input_tokens_details": {"cached_tokens": 4736},
        },
    )
    assert line is not None
    payload = json.loads(line[len("data: ") :])
    usage = payload["usage"]
    assert usage["prompt_tokens"] == 5507
    assert usage["completion_tokens"] == 252
    assert usage["total_tokens"] == 5759
    assert usage["prompt_tokens_details"]["cached_tokens"] == 4736
    # Anthropic-only keys must not leak onto the OpenAI shape.
    assert "cache_creation_input_tokens" not in usage
    assert "cache_read_input_tokens" not in usage


def test_build_usage_chunk_missing_fields_default_to_zero():
    # OpenAI Responses can omit input_tokens_details when prompt caching is
    # unused; the helper should still emit a chunk with cached_tokens=0.
    line = _build_usage_chunk(
        "chatcmpl-z",
        "openai",
        {"input_tokens": 42, "output_tokens": 7},
    )
    assert line is not None
    payload = json.loads(line[len("data: ") :])
    assert payload["usage"]["prompt_tokens_details"]["cached_tokens"] == 0


def test_build_usage_chunk_returns_none_when_all_zero():
    # If upstream errored before any usage event, suppress the chunk to
    # avoid a misleading "0 tokens" line.
    assert _build_usage_chunk("id", "anthropic", {}) is None
    assert _build_usage_chunk("id", "anthropic", None) is None
    assert _build_usage_chunk("id", "openai", {}) is None
    assert (
        _build_usage_chunk(
            "id",
            "openai",
            {
                "input_tokens": 0,
                "output_tokens": 0,
                "input_tokens_details": {"cached_tokens": 0},
            },
        )
        is None
    )


# ── streaming integration tests ─────────────────────────────────────


def _drive(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


async def _collect(agen):
    out = []
    async for line in agen:
        out.append(line)
    return out


def _mock_http_client(monkeypatch, handler):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(ep_mod, "_http_client", httpx.AsyncClient(transport = transport))


def _make_anthropic_client() -> ExternalProviderClient:
    return ExternalProviderClient(
        provider_type = "anthropic",
        base_url = "https://api.anthropic.com/v1",
        api_key = "sk-ant-test",
    )


def _make_openai_client() -> ExternalProviderClient:
    return ExternalProviderClient(
        provider_type = "openai",
        base_url = "https://api.openai.com/v1",
        api_key = "sk-openai-test",
    )


def _make_custom_client() -> ExternalProviderClient:
    return ExternalProviderClient(
        provider_type = "custom",
        base_url = "http://custom.example/v1",
        api_key = "",
    )


def _anthropic_sse(events: list[dict]) -> bytes:
    chunks: list[str] = []
    for event in events:
        chunks.append(f"event: {event['type']}")
        chunks.append(f"data: {json.dumps(event)}")
        chunks.append("")
    return ("\n".join(chunks) + "\n").encode("utf-8")


def _openai_sse(events: list[dict]) -> bytes:
    # Responses API ships one `event:` line per object plus the data line.
    chunks: list[str] = []
    for event in events:
        chunks.append(f"event: {event['type']}")
        chunks.append(f"data: {json.dumps(event)}")
        chunks.append("")
    return ("\n".join(chunks) + "\n").encode("utf-8")


def _usage_chunks(lines: list[str]) -> list[dict]:
    out: list[dict] = []
    for raw in lines:
        if not raw.startswith("data:"):
            continue
        payload = raw[len("data:") :].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and "usage" in parsed and parsed.get("choices") == []:
            out.append(parsed["usage"])
    return out


def test_custom_provider_registry_is_hidden():
    """Hidden entries stay filtered by default and are opt-in via include_hidden.

    They used to be dropped from /registry unconditionally, which is why the UI
    could never learn that the self-hosted presets run Studio tools. Exposing
    them by default would instead make a cached pre-change bundle render them as
    duplicate dropdown rows, since that bundle filters on a hardcoded name set
    rather than on ``hidden``. So the default is unchanged and the current UI
    asks for them, then filters the dropdown on the flag.
    """
    from core.inference.providers import get_provider_info, list_available_providers

    info = get_provider_info("custom")
    assert info is not None
    assert info["hidden"] is True
    assert all(p["provider_type"] != "custom" for p in list_available_providers())
    entry = next(
        p for p in list_available_providers(include_hidden = True) if p["provider_type"] == "custom"
    )
    assert entry["hidden"] is True
    assert entry["supports_studio_tools"] is True


def test_custom_provider_uses_chat_completions_without_auth_key(monkeypatch):
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_custom_client()
        lines = await _collect(
            client.stream_chat_completion(
                messages = [{"role": "user", "content": "ping"}],
                model = "Qwen/Qwen3-0.6B",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 64,
            )
        )
        await client.close()
        return lines

    lines = _drive(run())
    assert captured["url"] == "http://custom.example/v1/chat/completions"
    assert "authorization" not in {k.lower() for k in captured["headers"]}
    assert captured["body"]["model"] == "Qwen/Qwen3-0.6B"
    assert any("ok" in line for line in lines)


def test_custom_provider_test_endpoint_probes_chat_completion(monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    module_path = Path(__file__).resolve().parents[1] / "routes" / "providers.py"
    spec = importlib.util.spec_from_file_location("_providers_route_under_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    providers_route = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = providers_route
    spec.loader.exec_module(providers_route)

    captured: dict = {}

    class _FakeClient:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        async def chat_completion(self, **kwargs):
            captured["chat_completion"] = kwargs
            return {"choices": [{"message": {"content": "ok"}}]}

        async def list_models(self):
            raise AssertionError("custom provider test must not call /models")

        async def close(self):
            captured["closed"] = True

    monkeypatch.setattr(providers_route, "ExternalProviderClient", _FakeClient)

    async def run():
        return await providers_route.test_provider(
            providers_route.ProviderTestRequest(
                provider_type = "custom",
                base_url = "http://custom.example/v1",
                model_id = "Qwen/Qwen3-0.6B",
            ),
            _current_subject = "unsloth",
            via_api_key = False,
        )

    result = _drive(run())
    assert result.success is True
    assert result.models_count is None
    assert captured["init"]["provider_type"] == "custom"
    assert captured["chat_completion"]["model"] == "Qwen/Qwen3-0.6B"
    assert captured["chat_completion"]["max_tokens"] == 1
    assert captured["closed"] is True


def test_custom_provider_test_endpoint_requires_model_id(monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    module_path = Path(__file__).resolve().parents[1] / "routes" / "providers.py"
    spec = importlib.util.spec_from_file_location("_providers_route_under_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    providers_route = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = providers_route
    spec.loader.exec_module(providers_route)

    class _FakeClient:
        def __init__(self, **kwargs):
            pass

        async def close(self):
            pass

    monkeypatch.setattr(providers_route, "ExternalProviderClient", _FakeClient)

    async def run():
        return await providers_route.test_provider(
            providers_route.ProviderTestRequest(
                provider_type = "custom",
                base_url = "http://custom.example/v1",
            ),
            _current_subject = "unsloth",
            via_api_key = False,
        )

    result = _drive(run())
    assert result.success is False
    assert "model ID" in result.message


def test_anthropic_stream_emits_usage_chunk_before_done(monkeypatch):
    sse_events = [
        {
            "type": "message_start",
            "message": {
                "usage": {
                    "input_tokens": 7,
                    "output_tokens": 0,
                    "cache_creation_input_tokens": 6253,
                    "cache_read_input_tokens": 5713,
                }
            },
        },
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 1066},
        },
        {"type": "message_stop"},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = _anthropic_sse(sse_events),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_anthropic_client()
        return await _collect(
            client._stream_anthropic(
                messages = [{"role": "user", "content": "ping"}],
                model = "claude-opus-4-7",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 64,
            )
        )

    lines = _drive(run())
    usages = _usage_chunks(lines)
    assert len(usages) == 1, f"expected one usage chunk, got {len(usages)}: {usages}"
    u = usages[0]
    # Real prompt size = uncached input + cache writes + cache reads.
    assert u["prompt_tokens"] == 7 + 6253 + 5713
    assert u["completion_tokens"] == 1066
    assert u["total_tokens"] == 7 + 6253 + 5713 + 1066
    assert u["cache_creation_input_tokens"] == 6253
    assert u["cache_read_input_tokens"] == 5713
    assert u["prompt_tokens_details"]["cached_tokens"] == 5713

    # Usage chunk must come before [DONE].
    data_lines = [ln for ln in lines if ln.startswith("data:")]
    done_idx = next(i for i, ln in enumerate(data_lines) if ln.strip().endswith("[DONE]"))
    usage_idx = next(
        i for i, ln in enumerate(data_lines) if '"usage":' in ln and '"choices": []' in ln
    )
    assert usage_idx < done_idx


def test_openai_responses_stream_emits_usage_chunk_on_completed(monkeypatch):
    sse_events = [
        {"type": "response.created", "response": {"id": "resp_1"}},
        {
            "type": "response.completed",
            "response": {
                "id": "resp_1",
                "usage": {
                    "input_tokens": 5507,
                    "output_tokens": 252,
                    "input_tokens_details": {"cached_tokens": 4736},
                },
            },
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = _openai_sse(sse_events),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_openai_client()
        return await _collect(
            client._stream_openai_responses(
                messages = [{"role": "user", "content": "ping"}],
                model = "gpt-5.5",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 64,
                enable_thinking = None,
                reasoning_effort = None,
            )
        )

    lines = _drive(run())
    usages = _usage_chunks(lines)
    assert len(usages) == 1, f"expected one usage chunk, got {len(usages)}: {usages}"
    u = usages[0]
    assert u["prompt_tokens"] == 5507
    assert u["completion_tokens"] == 252
    assert u["prompt_tokens_details"]["cached_tokens"] == 4736
    # OpenAI shape must NOT carry Anthropic-only keys.
    assert "cache_creation_input_tokens" not in u
    assert "cache_read_input_tokens" not in u


def test_openai_responses_stream_emits_usage_chunk_on_incomplete(monkeypatch):
    sse_events = [
        {"type": "response.created", "response": {"id": "resp_2"}},
        {
            "type": "response.incomplete",
            "response": {
                "id": "resp_2",
                "usage": {
                    "input_tokens": 1234,
                    "output_tokens": 1024,
                    "input_tokens_details": {"cached_tokens": 768},
                },
            },
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content = _openai_sse(sse_events),
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_openai_client()
        return await _collect(
            client._stream_openai_responses(
                messages = [{"role": "user", "content": "ping"}],
                model = "gpt-5.5",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 1024,
                enable_thinking = None,
                reasoning_effort = None,
            )
        )

    lines = _drive(run())
    usages = _usage_chunks(lines)
    assert len(usages) == 1
    assert usages[0]["prompt_tokens_details"]["cached_tokens"] == 768


def _continuation_body(monkeypatch, provider_type: str, base_url: str) -> dict:
    """Send a continuation through one provider and return the upstream body."""
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            content = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = ExternalProviderClient(
            provider_type = provider_type,
            base_url = base_url,
            api_key = "k",
        )
        await _collect(
            client.stream_chat_completion(
                messages = [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "It is a bar"},
                ],
                model = "Qwen/Qwen3-0.6B",
                continue_final_message = True,
            )
        )
        await client.close()

    _drive(run())
    return captured


def test_self_hosted_providers_get_the_continuation_flags(monkeypatch):
    """These apply the template themselves, so a trailing assistant turn alone would
    render closed plus a fresh generation prompt and restart the answer."""
    for provider_type in ("llama_cpp", "vllm"):
        body = _continuation_body(monkeypatch, provider_type, "http://local.example/v1")
        assert body["continue_final_message"] is True, provider_type
        # A server rejects both being asked for at once.
        assert body["add_generation_prompt"] is False, provider_type


@pytest.mark.parametrize(
    ("provider_type", "base_url"),
    [
        # Prompt assembly is theirs, so the flag would just be an unknown field.
        ("openai", "https://api.openai.com/v1"),
        # Any user-supplied base_url, including a strict endpoint that would 400.
        ("custom", "http://custom.example/v1"),
        ("ollama", "http://localhost:11434/v1"),
    ],
)
def test_other_providers_do_not_get_the_continuation_flags(monkeypatch, provider_type, base_url):
    body = _continuation_body(monkeypatch, provider_type, base_url)
    assert "continue_final_message" not in body
    assert "add_generation_prompt" not in body


@pytest.mark.parametrize(
    "provider_type, expected",
    [
        ("vllm", True),
        ("openrouter", True),
        ("kimi", True),
        # Any user-supplied base_url: a strict endpoint 400s on an unknown field.
        ("custom", False),
        ("ollama", False),
        # "openai" is absent: it routes to /v1/responses, which reports usage itself.
    ],
)
def test_streamed_usage_is_requested_only_where_documented(monkeypatch, provider_type, expected):
    # An OAI-compatible stream omits usage without stream_options.include_usage, and
    # these providers report no llama.cpp timings, so the monitor has no token count to
    # derive a speed from and the row shows a blank Speed for every completed request.
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode("utf-8"))
        return httpx.Response(
            200,
            content = b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = ExternalProviderClient(
            provider_type = provider_type,
            base_url = "http://provider.example/v1",
            api_key = "sk-test",
        )
        await _collect(
            client.stream_chat_completion(
                messages = [{"role": "user", "content": "ping"}],
                model = "m",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 64,
            )
        )
        await client.close()

    _drive(run())
    assert captured["body"]["stream"] is True
    if expected:
        assert captured["body"]["stream_options"] == {"include_usage": True}
    else:
        assert "stream_options" not in captured["body"]


def test_kimi_no_search_fallback_requests_usage(monkeypatch):
    # The web-search path returns before the common body injection, and Kimi reports no
    # engine timings, so this fallback would leave tokens and speed blank.
    bodies: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        bodies.append(body)
        # First call: the model declines to invoke $web_search.
        return httpx.Response(
            200,
            content = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n',
            headers = {"content-type": "text/event-stream"},
        )

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = ExternalProviderClient(
            provider_type = "kimi",
            base_url = "http://kimi.example/v1",
            api_key = "sk-test",
        )
        await _collect(
            client.stream_chat_completion(
                messages = [{"role": "user", "content": "ping"}],
                model = "kimi-k2",
                max_tokens = 64,
                enabled_tools = ["web_search"],
            )
        )
        await client.close()

    _drive(run())
    assert len(bodies) >= 2, "the search call then the plain fallback"
    search_body, fallback_body = bodies[0], bodies[-1]
    assert "tools" in search_body
    assert "tools" not in fallback_body
    assert fallback_body["stream_options"] == {"include_usage": True}


def test_a_type_carried_only_by_the_sse_event_field_is_honoured(monkeypatch):
    """Only the SSE ``event:`` line carries the type here, and a type-less frame is
    skipped rather than fatal, so the usage would vanish silently. Built raw, since
    _openai_sse always repeats the type in data."""
    body = (
        b"event: response.completed\n"
        b'data: {"response":{"usage":{"input_tokens":7,"output_tokens":3}}}\n'
        b"\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content = body, headers = {"content-type": "text/event-stream"})

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_openai_client()
        return await _collect(
            client._stream_openai_responses(
                messages = [{"role": "user", "content": "ping"}],
                model = "gpt-5.5",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 1024,
                enable_thinking = None,
                reasoning_effort = None,
            )
        )

    usages = _usage_chunks(_drive(run()))
    assert len(usages) == 1, usages


def test_an_sse_event_name_does_not_carry_past_its_blank_line(monkeypatch):
    """Held past its blank line, a stale ``response.failed`` would claim the next
    type-less frame, emit a 502 and break the loop before the real usage."""
    body = (
        b"event: response.failed\n"
        b"data:\n"
        b"\n"
        b'data: {"choices":[{"delta":{"content":"ok"}}]}\n'
        b"\n"
        b"event: response.completed\n"
        b'data: {"response":{"usage":{"input_tokens":7,"output_tokens":3}}}\n'
        b"\n"
        b"data: [DONE]\n"
        b"\n"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content = body, headers = {"content-type": "text/event-stream"})

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = _make_openai_client()
        return await _collect(
            client._stream_openai_responses(
                messages = [{"role": "user", "content": "ping"}],
                model = "gpt-5.5",
                temperature = 0.7,
                top_p = 0.95,
                max_tokens = 1024,
                enable_thinking = None,
                reasoning_effort = None,
            )
        )

    lines = _drive(run())
    assert not [line for line in lines if '"provider_error"' in line], lines
    assert len(_usage_chunks(lines)) == 1, lines


def test_an_untyped_error_frame_is_surfaced_rather_than_skipped(monkeypatch):
    """An OpenAI-compatible proxy emits its errors as a bare ``{"error": {...}}`` with no
    ``type`` and no SSE event name, so skipping it returned zero chunks and no error."""
    body = b'data: {"error":{"message":"you are rate limited","type":"rate_limit_error"}}\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content = body, headers = {"content-type": "text/event-stream"})

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = ExternalProviderClient(
            provider_type = "openai",
            base_url = "https://api.openai.com/v1",
            api_key = "sk-openai-test",
        )
        out = await _collect(
            client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}], model = "gpt-5"
            )
        )
        await client.close()
        return out

    chunks = _drive(run())
    assert chunks, "the error frame was swallowed and the answer came back empty"
    assert any("rate limited" in str(chunk) for chunk in chunks), chunks


def test_a_chat_completions_frame_on_the_responses_path_is_still_skipped(monkeypatch):
    """The case the skip exists for stays skipped: no type, no event name, no error key."""
    body = (
        b'data: {"choices":[{"delta":{"content":"ok"}}]}\n\n'
        b"event: response.completed\n"
        b'data: {"type":"response.completed"}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content = body, headers = {"content-type": "text/event-stream"})

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = ExternalProviderClient(
            provider_type = "openai",
            base_url = "https://api.openai.com/v1",
            api_key = "sk-openai-test",
        )
        out = await _collect(
            client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}], model = "gpt-5"
            )
        )
        await client.close()
        return out

    chunks = _drive(run())
    assert not any("502" in str(chunk) for chunk in chunks), chunks


@pytest.mark.parametrize("payload", [b"null", b"[]", b'"text"', b"7"])
def test_a_valid_but_non_object_frame_is_skipped_not_fatal(monkeypatch, payload):
    """`data: null` and `data: []` are valid JSON but not dicts, so the error check must
    not call .get() on them: that raised AttributeError and killed the stream."""
    body = (
        b"data: "
        + payload
        + b'\n\nevent: response.completed\ndata: {"type":"response.completed"}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content = body, headers = {"content-type": "text/event-stream"})

    _mock_http_client(monkeypatch, handler)

    async def run():
        client = ExternalProviderClient(
            provider_type = "openai",
            base_url = "https://api.openai.com/v1",
            api_key = "sk-openai-test",
        )
        out = await _collect(
            client.stream_chat_completion(
                messages = [{"role": "user", "content": "hi"}], model = "gpt-5"
            )
        )
        await client.close()
        return out

    chunks = _drive(run())
    assert not any("502" in str(chunk) for chunk in chunks), chunks
