# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Focused tests for the GGUF llama.cpp agentic tool loop.

These tests drive ``LlamaCppBackend.generate_chat_completion_with_tools``
with fake llama-server SSE streams. They require no model, subprocess, GPU,
or network access.
"""

from __future__ import annotations

import contextlib
import copy
import json
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.llama_cpp import LlamaCppBackend


def _sse(delta: dict) -> str:
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta}]}) + "\n"


def _done() -> str:
    return "data: [DONE]\n"


def _make_backend(monkeypatch, streams: list[list[str]], payloads: list[dict]):
    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend._process = object()
    backend._healthy = True
    backend._port = 48847
    backend._api_key = None
    backend._effective_context_length = 4096
    backend._supports_reasoning = False
    backend._reasoning_always_on = False
    backend._reasoning_style = "enable_thinking"
    backend._supports_preserve_thinking = False

    @contextlib.contextmanager
    def fake_stream_with_retry(_client, _url, payload, _cancel_event, headers = None):
        payloads.append(copy.deepcopy(payload))
        yield type("FakeResponse", (), {"status_code": 200, "chunks": streams.pop(0)})()

    def fake_iter_text_cancellable(response, _cancel_event):
        yield from response.chunks

    monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
    monkeypatch.setattr(backend, "_iter_text_cancellable", fake_iter_text_cancellable)
    return backend


def test_structured_tool_call_after_visible_preface_is_executed(monkeypatch):
    """llama-server may emit content first and then native delta.tool_calls.

    Studio must not drop that tool call after it has streamed the preface.
    """

    tool_call_id = "call_render_late"
    first_stream = [
        _sse({"content": "Here is the artifact.\n\n"}),
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": "render_html",
                            "arguments": json.dumps(
                                {
                                    "code": "<html><body><div>red</div></body></html>",
                                    "title": "Simple Red Square",
                                }
                            ),
                        },
                    }
                ]
            }
        ),
        _done(),
    ]
    second_stream = [
        _sse({"content": "Done."}),
        _done(),
    ]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [first_stream, second_stream], payloads)

    calls: list[tuple[str, dict]] = []

    def fake_execute_tool(name, arguments, **_kwargs):
        calls.append((name, arguments))
        return "Rendered HTML artifact: Simple Red Square."

    monkeypatch.setattr("core.inference.tools.execute_tool", fake_execute_tool)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "render_html",
                "description": "Render HTML.",
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
        }
    ]

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "Make a red square."}],
            tools = tools,
            max_tool_iterations = 1,
        )
    )

    content_events = [e for e in events if e.get("type") == "content"]
    assert content_events[0]["text"] == "Here is the artifact.\n\n"

    first_content_index = next(
        i for i, event in enumerate(events) if event.get("type") == "content"
    )
    actual_tool_start_index = next(
        i
        for i, event in enumerate(events)
        if event.get("type") == "tool_start"
        and event.get("arguments", {}).get("code")
    )
    assert first_content_index < actual_tool_start_index

    assert calls == [
        (
            "render_html",
            {
                "code": "<html><body><div>red</div></body></html>",
                "title": "Simple Red Square",
            },
        )
    ]
    assert any(e.get("type") == "tool_end" and e.get("tool_name") == "render_html" for e in events)

    # The second llama-server request should include the assistant preface
    # plus the structured tool call, preserving OpenAI-compatible ordering.
    assert len(payloads) == 2
    assistant_messages = [
        m for m in payloads[1]["messages"] if m.get("role") == "assistant"
    ]
    assert assistant_messages[-1]["content"] == "Here is the artifact.\n\n"
    assert assistant_messages[-1]["tool_calls"][0]["id"] == tool_call_id
    assert (
        assistant_messages[-1]["tool_calls"][0]["function"]["name"]
        == "render_html"
    )
