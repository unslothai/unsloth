# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for `stream:false` on the GGUF agentic tool path (#6570).

When server-side tools are enabled (e.g. `unsloth studio run --model ...`,
which forces the tool policy on process-wide), a plain chat request used to be
routed into the tool loop, which returned an SSE body *regardless* of
`stream:false` -- breaking non-streaming clients and health checks like
LiteLLM. These tests drive the real route with a fake tool-capable backend and
assert the non-streaming path now returns a single JSON `chat.completion`,
while `stream:true` still streams.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
import routes.inference as inference_route


class _ToolGgufBackend:
    is_loaded = True
    model_identifier = "test/model.gguf"
    _is_audio = False
    is_vision = False
    supports_tools = True

    def generate_chat_completion_with_tools(self, **kwargs):
        # The agentic loop runs one tool, then the model answers. Event shapes
        # mirror the real GGUF loop (tool_start/tool_end/content/metadata).
        yield {
            "type": "tool_start",
            "tool_name": "python",
            "tool_call_id": "call_1",
            "arguments": {"code": "print(6 * 7)"},
        }
        yield {
            "type": "tool_end",
            "tool_name": "python",
            "tool_call_id": "call_1",
            "result": "42\n",
        }
        yield {"type": "content", "text": "The answer is 42."}
        yield {
            "type": "metadata",
            "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
            "timings": {"prompt_n": 11, "predicted_n": 5},
            "finish_reason": "stop",
        }


def _client(monkeypatch, backend = None):
    monkeypatch.setattr(
        inference_route, "get_llama_cpp_backend", lambda: backend or _ToolGgufBackend()
    )
    # Tools forced on -- the same effect as the CLI `run --model` tool policy.
    monkeypatch.setattr(inference_route, "_effective_enable_tools", lambda payload: True)

    async def _fake_select(payload, **_kwargs):
        return [{"type": "function", "function": {"name": "python"}}]

    monkeypatch.setattr(inference_route, "_select_request_tools", _fake_select)

    app = FastAPI()
    app.include_router(inference_route.router)
    app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return TestClient(app)


def _payload(stream: bool):
    return {
        "messages": [{"role": "user", "content": "What is 6 * 7? Use python."}],
        "stream": stream,
        "enable_tools": True,
    }


def test_non_streaming_tool_call_returns_single_json(monkeypatch):
    response = _client(monkeypatch).post("/chat/completions", json = _payload(stream = False))

    assert response.status_code == 200
    # The bug returned text/event-stream here; it must be a single JSON object.
    assert response.headers["content-type"].startswith("application/json")

    body = response.json()
    assert body["object"] == "chat.completion"
    choice = body["choices"][0]
    assert choice["message"]["content"] == "The answer is 42."
    assert choice["finish_reason"] == "stop"
    assert body["usage"]["prompt_tokens"] == 11
    assert body["usage"]["completion_tokens"] == 5
    assert body["usage"]["total_tokens"] == 16


def test_streaming_tool_call_still_streams(monkeypatch):
    # The parallel path is untouched: stream:true keeps returning SSE.
    response = _client(monkeypatch).post("/chat/completions", json = _payload(stream = True))

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "The answer is 42." in response.text
    assert "data: [DONE]" in response.text


class _EventsBackend(_ToolGgufBackend):
    """Tool backend that yields a caller-supplied event list."""

    def __init__(self, events):
        self._events = events

    def generate_chat_completion_with_tools(self, **kwargs):
        yield from self._events


def test_non_streaming_missing_usage_defaults_to_zero(monkeypatch):
    # No metadata event at all: usage zero-defaults and finish_reason falls back.
    events = [{"type": "content", "text": "hi"}]
    response = _client(monkeypatch, _EventsBackend(events)).post(
        "/chat/completions", json = _payload(stream = False)
    )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["message"]["content"] == "hi"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["prompt_tokens"] == 0
    assert body["usage"]["completion_tokens"] == 0
    assert body["usage"]["total_tokens"] == 0


def test_non_streaming_preserves_length_finish_reason(monkeypatch):
    events = [
        {"type": "content", "text": "truncated"},
        {
            "type": "metadata",
            "usage": {"prompt_tokens": 3, "completion_tokens": 9},
            "finish_reason": "length",
        },
    ]
    response = _client(monkeypatch, _EventsBackend(events)).post(
        "/chat/completions", json = _payload(stream = False)
    )

    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["finish_reason"] == "length"
    # total_tokens is derived when the server omits it.
    assert body["usage"]["total_tokens"] == 12


def test_non_streaming_preserves_cached_tokens(monkeypatch):
    # KV-cache hit details from the metadata event must survive into the body
    # (the tool path used to drop them and always report cached_tokens=0).
    events = [
        {"type": "content", "text": "hi"},
        {
            "type": "metadata",
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 4,
                "prompt_tokens_details": {"cached_tokens": 16},
            },
            "finish_reason": "stop",
        },
    ]
    response = _client(monkeypatch, _EventsBackend(events)).post(
        "/chat/completions", json = _payload(stream = False)
    )

    assert response.status_code == 200
    assert response.json()["usage"]["prompt_tokens_details"]["cached_tokens"] == 16
