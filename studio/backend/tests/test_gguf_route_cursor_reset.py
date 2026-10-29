# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cursor resets on an internal no-op tool turn; else the final answer is truncated by the preface."""

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
    backend._port = 48848
    backend._api_key = None
    backend._effective_context_length = 4096
    backend._supports_reasoning = False
    backend._reasoning_always_on = False
    backend._reasoning_style = "enable_thinking"
    backend._supports_preserve_thinking = False

    @contextlib.contextmanager
    def fake_stream_with_retry(
        _client,
        _url,
        payload,
        _cancel_event,
        headers = None,
        first_token_deadline = None,
    ):
        payloads.append(copy.deepcopy(payload))
        yield type("FakeResponse", (), {"status_code": 200, "chunks": streams.pop(0)})()

    def fake_iter_text_cancellable(
        response,
        _cancel_event,
        first_token_deadline = None,
    ):
        yield from response.chunks

    monkeypatch.setattr(backend, "_stream_with_retry", fake_stream_with_retry)
    monkeypatch.setattr(backend, "_iter_text_cancellable", fake_iter_text_cancellable)
    return backend


def _replay_route_cursor(events: list[dict]) -> dict:
    """Replays the route's cumulative cursor loop: reset prev_text on empty status and tool_start."""
    prev_text = ""
    visible_deltas: list[str] = []
    tool_starts: list[dict] = []
    statuses: list[str] = []
    for event in events:
        etype = event["type"]
        if etype == "status":
            if not event["text"]:
                prev_text = ""
            statuses.append(event["text"])
            continue
        if etype in ("tool_start", "tool_end"):
            if etype == "tool_start":
                prev_text = ""
                tool_starts.append(event)
            continue
        if etype == "metadata":
            continue
        clean_cumulative = event.get("text", "")
        new_text = clean_cumulative[len(prev_text) :]
        prev_text = clean_cumulative
        if not new_text:
            continue
        visible_deltas.append(new_text)
    return {
        "visible": "".join(visible_deltas),
        "tool_starts": tool_starts,
        "statuses": statuses,
    }


def _replay_route_cursor_without_status_reset(events: list[dict]) -> dict:
    """Pre-fix control: identical to the route loop but never resets the
    cursor on an empty status (only on ``tool_start``)."""
    prev_text = ""
    visible_deltas: list[str] = []
    for event in events:
        etype = event["type"]
        if etype == "status":
            continue
        if etype in ("tool_start", "tool_end"):
            if etype == "tool_start":
                prev_text = ""
            continue
        if etype == "metadata":
            continue
        clean_cumulative = event.get("text", "")
        new_text = clean_cumulative[len(prev_text) :]
        prev_text = clean_cumulative
        if not new_text:
            continue
        visible_deltas.append(new_text)
    return {"visible": "".join(visible_deltas)}


def _web_search_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }


def test_final_answer_survives_preface_then_disabled_tool_noop(monkeypatch):
    """A preface then a disabled-tool no-op must not truncate the shorter final answer."""
    preface = "Let me run a quick command to double-check."
    final = "All set."  # shorter than the preface so truncation is visible

    # terminal is not enabled, so the controller treats the call as an internal no-op.
    turn_stream = [
        _sse({"content": preface}),
        _sse(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_disabled",
                        "type": "function",
                        "function": {
                            "name": "terminal",
                            "arguments": json.dumps({"command": "ls"}),
                        },
                    }
                ]
            }
        ),
        _done(),
    ]
    final_stream = [_sse({"content": final}), _done()]
    payloads: list[dict] = []
    backend = _make_backend(monkeypatch, [turn_stream, final_stream], payloads)

    executed: list[str] = []
    monkeypatch.setattr(
        "core.inference.tools.execute_tool",
        lambda name, arguments, **_kw: executed.append(name) or "should-not-run",
    )

    events = list(
        backend.generate_chat_completion_with_tools(
            messages = [{"role": "user", "content": "answer me"}],
            tools = [_web_search_tool()],
            temperature = 0.0,
            max_tool_iterations = 5,
        )
    )

    replay = _replay_route_cursor(events)

    assert executed == []
    assert replay["tool_starts"] == []

    # An empty status must reset the route cursor, or the shorter `final` diffs to nothing.
    assert "" in replay["statuses"], "no cursor-resetting empty status emitted"

    assert preface in replay["visible"], replay["visible"]
    assert final in replay["visible"], replay["visible"]
    assert replay["visible"].index(preface) < replay["visible"].index(final)
    assert replay["visible"].count(preface) == 1

    # Negative control: without the reset `final` is dropped, proving the empty status matters.
    no_reset = _replay_route_cursor_without_status_reset(events)
    assert final not in no_reset["visible"], no_reset["visible"]
