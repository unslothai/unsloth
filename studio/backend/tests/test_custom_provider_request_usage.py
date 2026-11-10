# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Custom endpoints get stream_options.include_usage only when the connection opts in (request_usage)."""

import asyncio
import json

import httpx
import pytest

from core.inference import external_provider as ep_mod
from core.inference.external_provider import ExternalProviderClient, _reported_window

# The final chunk Inception's /v1/chat/completions sends when include_usage is set.
_INCEPTION_SSE = (
    'data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,'
    '"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}\n\n'
    'data: {"id":"c1","object":"chat.completion.chunk","choices":[{"index":0,'
    '"delta":{},"finish_reason":"stop"}]}\n\n'
    'data: {"id":"c1","object":"chat.completion.chunk","choices":[],'
    '"usage":{"prompt_tokens":9,"completion_tokens":28,"total_tokens":37,'
    '"prompt_tokens_details":{"cached_tokens":0},'
    '"completion_tokens_details":{"reasoning_tokens":13}},'
    '"server_timing":{"total_ms":120}}\n\n'
    "data: [DONE]\n\n"
).encode()


def _run(monkeypatch, provider_type, request_usage):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(
            200, content = _INCEPTION_SSE, headers = {"content-type": "text/event-stream"}
        )

    monkeypatch.setattr(
        ep_mod, "_http_client", httpx.AsyncClient(transport = httpx.MockTransport(handler))
    )

    async def run():
        client = ExternalProviderClient(
            provider_type = provider_type,
            base_url = "http://provider.example/v1",
            api_key = "sk-test",
            request_usage = request_usage,
        )
        lines = [
            line
            async for line in client.stream_chat_completion(
                messages = [{"role": "user", "content": "ping"}], model = "m"
            )
        ]
        await client.close()
        return lines

    return bodies, asyncio.new_event_loop().run_until_complete(run())


def _usage(lines):
    for line in lines:
        if line.startswith("data:") and "[DONE]" not in line:
            chunk = json.loads(line[len("data:") :])
            if chunk.get("choices") == [] and "usage" in chunk:
                return chunk["usage"]
    return None


def test_custom_without_the_opt_in_gets_no_stream_options(monkeypatch):
    [body], _ = _run(monkeypatch, "custom", request_usage = False)
    assert "stream_options" not in body


def test_custom_with_the_opt_in_gets_include_usage_and_parses_the_usage_chunk(monkeypatch):
    [body], lines = _run(monkeypatch, "custom", request_usage = True)
    assert body["stream_options"] == {"include_usage": True}
    usage = _usage(lines)
    assert (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) == (
        9,
        28,
        37,
    )


@pytest.mark.parametrize("provider_type", ["vllm", "openrouter", "qwen"])
@pytest.mark.parametrize("request_usage", [False, True])
def test_the_opt_in_has_no_effect_on_listed_providers(monkeypatch, provider_type, request_usage):
    [body], _ = _run(monkeypatch, provider_type, request_usage)
    assert body["stream_options"] == {"include_usage": True}


@pytest.mark.parametrize("provider_type", ["openai", "anthropic", "mistral"])
def test_the_opt_in_does_not_reach_unlisted_providers(provider_type):
    client = ExternalProviderClient(
        provider_type = provider_type,
        base_url = "http://provider.example/v1",
        api_key = "sk-test",
        request_usage = True,
    )
    assert client.request_usage is False


@pytest.mark.parametrize(
    ("entry", "window"),
    [
        ({"context_length": 260000}, 260000),
        ({"context_window": 131072}, 131072),
        ({"max_context_length": 32768}, 32768),
        ({"top_provider": {"context_length": 65536}}, 65536),
        ({"max_model_len": 8192, "context_length": 4096}, 8192),
        ({"meta": {"n_ctx": 4096}, "max_model_len": 8192, "context_length": 16384}, 4096),
        ({"context_length": 0, "context_window": "big", "max_context_length": True}, None),
        ({"context_length": -1, "top_provider": {"context_length": 2048}}, 2048),
        ({}, None),
    ],
)
def test_reported_window_fields(entry, window):
    assert _reported_window(entry) == window


def _route_chunks(
    monkeypatch,
    served,
    ui_events,
    bodies = None,
    **fields,
):
    import sys
    from pathlib import Path

    from starlette.requests import Request

    import routes.inference as ri
    from models.inference import ChatCompletionRequest

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_external_provider_sampling_over_the_wire import _run, _Server

    async def served_context_window(self, model):
        return served

    monkeypatch.setattr(ExternalProviderClient, "served_context_window", served_context_window)

    async def receive():
        return {"type": "http.disconnect"}

    headers = [(b"content-type", b"application/json")]
    if ui_events:
        headers.append((b"x-unsloth-events", b"1"))
    chunks = []
    with _Server() as server:
        payload = ChatCompletionRequest(
            provider_type = "custom",
            provider_base_url = server.base_url,
            messages = [{"role": "user", "content": "hi"}],
            model = "mercury-2.5",
            stream = True,
            **fields,
        )
        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/v1/chat/completions",
                "headers": headers,
                "query_string": b"",
            },
            receive,
        )

        async def go():
            response = await ri._proxy_to_external_provider(payload, request)
            async for part in response.body_iterator:
                text = part.decode() if isinstance(part, bytes) else part
                for line in text.splitlines():
                    body = line[len("data:") :].strip() if line.startswith("data:") else ""
                    if body and body != "[DONE]":
                        chunks.append(json.loads(body))

        _run(go)
        if bodies is not None:
            bodies.extend(server.bodies)
    return chunks


def _window_frames(chunks):
    return [c for c in chunks if c.get("type") == "context_usage" and "context_window" in c]


def test_the_served_window_reaches_the_ui_meter(monkeypatch):
    chunks = _route_chunks(monkeypatch, 260000, ui_events = True, context_overflow = "truncate_oldest")
    assert _window_frames(chunks) == [{"type": "context_usage", "context_window": 260000}]


def test_no_served_window_read_without_auto_compact(monkeypatch):
    # the meter asks the server only under the same opt-in as compaction
    assert _window_frames(_route_chunks(monkeypatch, 260000, ui_events = True)) == []


def test_the_catalogue_window_the_client_sent_wins(monkeypatch):
    chunks = _route_chunks(monkeypatch, 260000, ui_events = True, context_window = 128000)
    assert _window_frames(chunks) == [{"type": "context_usage", "context_window": 128000}]


@pytest.mark.parametrize(("served", "ui_events"), [(260000, False), (None, True)])
def test_no_window_frame_without_ui_events_or_a_window(monkeypatch, served, ui_events):
    chunks = _route_chunks(monkeypatch, served, ui_events, context_overflow = "truncate_oldest")
    assert _window_frames(chunks) == []


@pytest.mark.parametrize("opted_in", [False, True])
def test_the_request_flag_reaches_the_custom_upstream(monkeypatch, opted_in):
    bodies = []
    _route_chunks(monkeypatch, None, ui_events = True, bodies = bodies, provider_request_usage = opted_in)
    assert [b.get("stream_options") for b in bodies] == [
        {"include_usage": True} if opted_in else None
    ]
