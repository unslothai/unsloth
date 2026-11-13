# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""tool_choice "none" leaves a Llama 3.x bare-JSON call in the final-pass text; it must not be shown."""

from __future__ import annotations

import json

import pytest

from .test_llama_cpp_tool_loop import (
    _done,
    _finish,
    _make_backend,
    _record_tool_calls,
    _run_tool_loop,
    _sse,
    _tool_call_sse,
    _web_search_tool,
)


def _python_tool():
    return {
        "type": "function",
        "function": {
            "name": "python",
            "description": "Run Python.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string"}},
                "required": ["code"],
            },
        },
    }


def _content_stream(text: str, step: int = 3) -> list[str]:
    return [_sse({"content": text[i : i + step]}) for i in range(0, len(text), step)] + [
        _finish("stop"),
        _done(),
    ]


def _final_pass(
    monkeypatch,
    final_text: str,
    tools = None,
    step: int = 3,
):
    payloads: list[dict] = []
    streams = [
        [_tool_call_sse("web_search", {"query": "a"}, "c1"), _finish("tool_calls"), _done()],
        _content_stream(final_text, step),
    ]
    backend = _make_backend(monkeypatch, streams, payloads)
    _record_tool_calls(monkeypatch, "RESULT_OK")
    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Search."}],
        tools or [_web_search_tool()],
        max_tool_iterations = 1,
    )
    assert payloads[-1].get("tool_choice") == "none"
    return [e["text"] for e in events if e.get("type") == "content"]


@pytest.mark.parametrize(
    "text",
    [
        '{"name": "web_search", "parameters": {"query": "again"}}',
        '{"type": "function", "function": "web_search", "parameters": {"query": "again"}}',
        '{"name": "web_search", "parameters": {"query": "a"}}; {"name": "web_search", "parameters": {"query": "b"}}',
        '{"name": "web_search", "parameters": {"query": "a"}}\n{"name": "web_search", "parameters": {"query": "b"}}',
    ],
)
@pytest.mark.parametrize("step", [1, 3])
def test_final_pass_does_not_show_a_bare_json_call(monkeypatch, text, step):
    shown = _final_pass(monkeypatch, text, step = step)
    # No fragment either: a chained peer whose name has not arrived yet must not stream.
    assert not any("{" in s or "web_search" in s for s in shown), shown


@pytest.mark.parametrize(
    "text",
    [
        "The search found RESULT_OK.",
        '{"answer": 4, "unit": "apples"}',
        'Here is the JSON: {"name": "web_search", "parameters": {"query": "x"}}',
    ],
)
def test_final_pass_answers_are_unchanged(monkeypatch, text):
    shown = _final_pass(monkeypatch, text)
    assert shown[-1] == text


def test_final_pass_keeps_blocked_execution_call_visible(monkeypatch):
    text = '{"name": "python", "parameters": {"code": "print(1)"}}'
    shown = _final_pass(monkeypatch, text, tools = [_web_search_tool(), _python_tool()])
    assert shown[-1] == text


def test_final_pass_releases_unclosed_ordinary_json_at_end(monkeypatch):
    text = '{"answer": [1, 2'
    shown = _final_pass(monkeypatch, text)
    assert shown[-1] == text
