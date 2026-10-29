# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The final pass after the tool cap sends tool_choice "none", so llama-server leaves a Llama 3.x bare-JSON
call in the answer text (Llama-3.2-1B-Instruct writes one there). It must not be shown as the answer."""

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


def _content_stream(text: str) -> list[str]:
    return [_sse({"content": text[i : i + 3]}) for i in range(0, len(text), 3)] + [
        _finish("stop"),
        _done(),
    ]


def _final_pass(
    monkeypatch,
    final_text: str,
    tools = None,
):
    payloads: list[dict] = []
    streams = [
        [_tool_call_sse("web_search", {"query": "a"}, "c1"), _finish("tool_calls"), _done()],
        _content_stream(final_text),
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
    ],
)
def test_final_pass_does_not_show_a_bare_json_call(monkeypatch, text):
    shown = _final_pass(monkeypatch, text)
    assert not any('"web_search"' in s for s in shown), shown


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
    """Bare text never promotes an exec-class tool; such JSON stays visible as before."""
    text = '{"name": "python", "parameters": {"code": "print(1)"}}'
    shown = _final_pass(monkeypatch, text, tools = [_web_search_tool(), _python_tool()])
    assert shown[-1] == text


def test_final_pass_releases_unclosed_ordinary_json_at_end(monkeypatch):
    text = '{"answer": [1, 2'
    shown = _final_pass(monkeypatch, text)
    assert shown[-1] == text
