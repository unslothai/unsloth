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
        if (
            isinstance(parsed, dict)
            and "usage" in parsed
            and parsed.get("choices") == []
        ):
            out.append(parsed["usage"])
    return out


def test_custom_provider_registry_is_hidden():
    from core.inference.providers import get_provider_info, list_available_providers

    info = get_provider_info("custom")
    assert info is not None
    assert info["hidden"] is True
    assert "custom" not in {p["provider_type"] for p in list_available_providers()}


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
    spec = importlib.util.spec_from_file_location(
        "_providers_route_under_test", module_path
    )
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
            current_subject = "unsloth",
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
    spec = importlib.util.spec_from_file_location(
        "_providers_route_under_test", module_path
    )
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
            current_subject = "unsloth",
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
    done_idx = next(
        i for i, ln in enumerate(data_lines) if ln.strip().endswith("[DONE]")
    )
    usage_idx = next(
        i
        for i, ln in enumerate(data_lines)
        if '"usage":' in ln and '"choices": []' in ln
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
