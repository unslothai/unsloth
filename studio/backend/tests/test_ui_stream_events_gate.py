# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unsloth's UI control frames are opt-in on OpenAI-compatible streams.

Frames like ``tool_status`` / ``reasoning_summary`` carry no ``choices``, so
strict OpenAI clients (openai-python, the Vercel AI SDK, opencode) fail schema
validation mid-stream when they arrive. /v1/chat/completions therefore emits a
clean OpenAI stream by default; the Studio UI opts in with X-Unsloth-Events: 1,
and durable runs (whose event log is replayed to that UI) opt in internally.
"""

from __future__ import annotations

import ast
import inspect
import threading

from routes.inference import (
    UI_STREAM_EVENTS_HEADER,
    _ui_stream_events_enabled,
    produce_openai_chat_completions,
)


def _request(headers: list[tuple[bytes, bytes]]):
    from starlette.requests import Request

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 0),
        "server": ("127.0.0.1", 0),
        "state": {},
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    return Request(scope, receive)


def test_no_header_means_clean_openai_stream():
    assert _ui_stream_events_enabled(_request([])) is False


def test_header_opts_in():
    req = _request([(UI_STREAM_EVENTS_HEADER.lower().encode(), b"1")])
    assert _ui_stream_events_enabled(req) is True


def test_other_header_values_do_not_opt_in():
    for value in (b"0", b"true", b"yes", b"", b" 1x"):
        req = _request([(UI_STREAM_EVENTS_HEADER.lower().encode(), value)])
        assert _ui_stream_events_enabled(req) is False, value


def test_none_request_is_refused():
    assert _ui_stream_events_enabled(None) is False


def test_background_generation_run_opts_into_control_frames():
    # Durable runs replay the producer's SSE lines (tool cards included) to the
    # Studio UI, so their synthetic request must carry the opt-in.
    from core.inference.chat_generation_runs import _background_request

    req = _background_request(app=None, run_id="run-1", cancel_event=threading.Event())
    assert _ui_stream_events_enabled(req) is True


def test_openai_stream_control_yields_are_gated():
    # Every raw control-frame yield in the OpenAI chat producer must sit behind
    # the per-request opt-in; keepalive/error chunks are plain SSE and exempt.
    src = inspect.getsource(produce_openai_chat_completions)
    lines = src.splitlines()
    control_yields = (
        'yield f"data: {json.dumps(event)}',
        'yield f"data: {json.dumps(cumulative)}',
        'yield f"data: {status_data}',
    )
    candidate_lines = {
        i + 1
        for i, line in enumerate(lines)
        if any(line.strip().startswith(p) for p in control_yields)
    }
    assert candidate_lines, "control-frame yields disappeared from the producer"

    guarded: set[int] = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.If) and "_ui_events" in ast.dump(node.test):
            guarded.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))

    ungated = sorted(candidate_lines - guarded)
    assert not ungated, f"ungated control-frame yields at producer lines {ungated}"
