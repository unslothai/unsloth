# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""/v1/messages client-tool passthrough honours context_overflow like /v1/chat/completions (#15417)."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading

import httpx
import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

import routes.inference as inf_mod
from routes.inference import (
    _anthropic_passthrough_non_streaming,
    _anthropic_passthrough_stream,
)

_OVERFLOW = json.dumps(
    {
        "error": {
            "code": 400,
            "message": "request (10000 tokens) exceeds the available context size (2048 tokens), try increasing it",
            "type": "exceed_context_size_error",
            "n_prompt_tokens": 10000,
            "n_ctx": 2048,
        }
    }
).encode()


class _Backend:
    base_url = "http://127.0.0.1:59999"
    context_length = 2048

    def count_chat_tokens(self, *_args, **_kwargs):
        return 2


class _Request:
    async def is_disconnected(self):
        return False


def _history():
    messages = [{"role": "system", "content": "system"}]
    for idx in range(8):
        messages.append({"role": "user", "content": f"question {idx} " + "x" * 1000})
        messages.append({"role": "assistant", "content": f"answer {idx} " + "y" * 1000})
    messages.append({"role": "user", "content": "latest question"})
    return messages


@pytest.fixture(autouse = True)
def _fresh_cut_memory():
    from core.inference import prefix_boundary

    prefix_boundary.clear()
    yield
    prefix_boundary.clear()


def _install_transport(monkeypatch, bodies, *, stream):
    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            return httpx.Response(400, content = _OVERFLOW)
        if stream:
            content = (
                f"data: {json.dumps({'choices': [{'delta': {'content': 'hi'}}]})}\n\n"
                "data: [DONE]\n\n"
            )
            return httpx.Response(
                200, content = content.encode(), headers = {"content-type": "text/event-stream"}
            )
        return httpx.Response(
            200,
            json = {
                "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1},
            },
        )

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def _client(*_args, **kwargs):
        return real_client(transport = transport, timeout = kwargs.get("timeout", 600))

    monkeypatch.setattr(inf_mod.httpx, "AsyncClient", _client)


async def _stream_text(**kwargs):
    response = await _anthropic_passthrough_stream(
        _Request(),
        threading.Event(),
        _Backend(),
        _history(),
        [],
        0.7,
        0.95,
        20,
        16,
        "msg_1",
        "test-model",
        **kwargs,
    )
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk.decode() if isinstance(chunk, (bytes, bytearray)) else chunk)
    return "".join(chunks)


async def _non_streaming(**kwargs):
    return await _anthropic_passthrough_non_streaming(
        _Backend(),
        _history(),
        [],
        0.7,
        0.95,
        20,
        16,
        "msg_1",
        "test-model",
        **kwargs,
    )


@pytest.mark.parametrize("policy", ["truncate_oldest", "truncate_middle"])
def test_stream_overflow_is_truncated_and_retried(monkeypatch, policy):
    bodies = []
    _install_transport(monkeypatch, bodies, stream = True)

    text = asyncio.run(_stream_text(context_overflow = policy))

    assert len(bodies) == 2
    assert len(bodies[1]["messages"]) < len(bodies[0]["messages"])
    assert bodies[1]["messages"][0] == {"role": "system", "content": "system"}
    assert bodies[1]["messages"][-1]["content"] == "latest question"
    assert "event: error" not in text
    assert '"text": "hi"' in text or '"text":"hi"' in text
    assert "message_stop" in text


def test_stream_overflow_without_opt_in_keeps_the_error(monkeypatch):
    bodies = []
    _install_transport(monkeypatch, bodies, stream = True)

    text = asyncio.run(_stream_text())

    assert len(bodies) == 1
    assert "event: error" in text


@pytest.mark.parametrize("policy", ["truncate_oldest", "truncate_middle"])
def test_non_streaming_overflow_is_truncated_and_retried(monkeypatch, policy):
    bodies = []
    _install_transport(monkeypatch, bodies, stream = False)

    result = asyncio.run(_non_streaming(context_overflow = policy))

    assert len(bodies) == 2
    assert len(bodies[1]["messages"]) < len(bodies[0]["messages"])
    assert bodies[1]["messages"][-1]["content"] == "latest question"
    assert json.loads(result.body)["content"][0]["text"] == "hi"


def test_non_streaming_overflow_without_opt_in_keeps_the_error(monkeypatch):
    from fastapi import HTTPException

    bodies = []
    _install_transport(monkeypatch, bodies, stream = False)

    with pytest.raises(HTTPException) as excinfo:
        asyncio.run(_non_streaming())

    assert excinfo.value.status_code == 400
    assert len(bodies) == 1


@pytest.mark.parametrize("stream", [True, False])
def test_messages_route_honours_the_server_overflow_default(monkeypatch, stream):
    from types import SimpleNamespace

    from fastapi.responses import StreamingResponse
    from models.inference import AnthropicMessagesRequest
    from routes.inference import anthropic_messages

    monkeypatch.setenv("UNSLOTH_CONTEXT_OVERFLOW", "truncate_oldest")
    monkeypatch.setattr(inf_mod, "current_date_prompt_line", lambda **_kwargs: "")
    backend = SimpleNamespace(
        is_loaded = True,
        is_vision = False,
        supports_tools = True,
        model_identifier = "test-model",
        base_url = _Backend.base_url,
        context_length = 2048,
        count_chat_tokens = lambda *args, **kwargs: 2,
    )
    monkeypatch.setattr(inf_mod, "get_llama_cpp_backend", lambda: backend)
    bodies = []
    _install_transport(monkeypatch, bodies, stream = stream)
    history = _history()
    payload = AnthropicMessagesRequest(
        max_tokens = 16,
        stream = stream,
        system = history[0]["content"],
        messages = history[1:],
        tools = [{"name": "lookup", "input_schema": {"type": "object"}}],
    )

    async def _run():
        response = await anthropic_messages(payload, request = _Request(), current_subject = "t")
        if not isinstance(response, StreamingResponse):
            return response.body.decode()
        chunks = []
        async for chunk in response.body_iterator:
            chunks.append(chunk.decode() if isinstance(chunk, (bytes, bytearray)) else chunk)
        return "".join(chunks)

    text = asyncio.run(_run())

    assert len(bodies) == 2
    assert len(bodies[1]["messages"]) < len(bodies[0]["messages"])
    assert "exceeds the available context size" not in text
    assert "hi" in text


@pytest.mark.parametrize("policy", [None, "truncate_middle"])
def test_only_truncate_oldest_pays_for_the_cut_replay(monkeypatch, policy):
    def _no_hop(*_args, **_kwargs):
        raise AssertionError("no thread hop without a policy that remembers cuts")

    monkeypatch.setattr(inf_mod.asyncio, "to_thread", _no_hop)
    body = {"messages": _history()}

    assert asyncio.run(inf_mod._prepare_passthrough_cut(_Backend(), body, policy)) == (
        None,
        None,
        0,
    )
    assert body["messages"] == _history()
