# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF SSE cursor reset across an internal no-op tool decision.

When a GGUF turn streams visible preface text ("I'll render it again...")
and then hits an internal no-op (duplicate / disabled / repeat render_html),
no ``tool_start`` event is emitted. The GGUF SSE route diffs each cumulative
``content`` event against a ``prev_text`` cursor and resets that cursor on
``tool_start`` *and* on an empty ``status`` event. The generator must emit an
empty status between the preface turn and the final no-tools pass, otherwise
the final answer would be diffed against the stale preface and truncated or
dropped entirely.

This test drives the real generator and replays its events through a faithful
copy of the route's cursor loop (studio/backend/routes/inference.py), asserting
the final answer survives in full and the no-op produces no phantom tool card.
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
    """Replicate the GGUF SSE route's cumulative-cursor loop.

    Mirrors studio/backend/routes/inference.py: reset ``prev_text`` on empty
    status and on ``tool_start``; otherwise diff each cumulative ``content``
    snapshot against the cursor and stream the delta. The preface/final text
    here carry no tool XML, so the display strip is the identity -- the cursor
    reset is the behaviour under test.
    """
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
    """Preface text, then a call to a disabled tool (internal no-op).

    A disabled-tool decision emits no ``tool_start`` and forces the final
    no-tools pass. The route cursor must be reset before that pass so the
    short final answer is not diffed away against the longer preface.
    """
    preface = "Let me run a quick command to double-check."
    final = "All set."  # deliberately shorter than the preface -> truncation is visible

    # Single turn: visible preface + a call to `terminal`, which is NOT in the
    # enabled tool list, so the controller marks it disabled -> internal no-op.
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
            tools = [_web_search_tool()],  # terminal intentionally absent
            temperature = 0.0,
            max_tool_iterations = 5,
        )
    )

    replay = _replay_route_cursor(events)

    # Disabled tool is an internal no-op: never executed, no visible card.
    assert executed == []
    assert replay["tool_starts"] == []

    # The generator must emit an empty status that resets the route cursor
    # before the final pass; otherwise `final` (shorter than `preface`) would
    # be diffed to nothing and dropped.
    assert "" in replay["statuses"], "no cursor-resetting empty status emitted"

    # Both the preface and the final answer survive, in order, untruncated.
    assert preface in replay["visible"], replay["visible"]
    assert final in replay["visible"], replay["visible"]
    assert replay["visible"].index(preface) < replay["visible"].index(final)
    assert replay["visible"].count(preface) == 1

    # Negative control: a route loop that does NOT reset on empty status (the
    # pre-fix behaviour) would diff `final` against the stale preface cursor
    # and drop it -- proving the empty status is load-bearing here.
    no_reset = _replay_route_cursor_without_status_reset(events)
    assert final not in no_reset["visible"], no_reset["visible"]
