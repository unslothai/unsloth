# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for the two live post-compaction failure mechanisms."""

import json

import pytest

from core.inference import tools
from .test_llama_cpp_tool_loop import (
    _backend_and_payloads,
    _done,
    _finish,
    _record_tool_calls,
    _run_tool_loop,
    _sse,
    _tool_call_sse,
    _web_search_tool,
)


def _usage(prompt, completion):
    return (
        "data: "
        + json.dumps(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": prompt + completion,
                },
            }
        )
        + "\n"
    )


def _cut(
    reason = "length",
    prompt = 3000,
    completion = 1096,
):
    return [
        _tool_call_sse("terminal", {"command": "echo unsafe"}, "cut"),
        _finish(reason),
        _usage(prompt, completion),
        _done(),
    ]


def _answer(text = "The task is complete."):
    return [_sse({"content": text}), _finish("stop"), _usage(300, 10), _done()]


def test_context_full_tool_turn_retries_without_executing_any_cut_calls(monkeypatch):
    backend, payloads = _backend_and_payloads(
        monkeypatch,
        [
            _cut(),
            [
                _tool_call_sse("terminal", {"command": "echo safe"}, "retry"),
                _finish("tool_calls"),
                _usage(300, 20),
                _done(),
            ],
            _answer(),
        ],
    )
    monkeypatch.setattr(backend, "count_chat_tokens", lambda *a, **kw: 300)
    calls = _record_tool_calls(monkeypatch, "safe")
    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Write the report."}],
        [tools.TERMINAL_TOOL],
        permission_mode = "off",
        sandbox_level = "low",
    )
    assert calls == [("terminal", {"command": "echo safe"})]
    assert len(payloads) == 3
    assert [m["role"] for m in payloads[1]["messages"]] == ["user", "assistant", "user"]
    assert "NONE" in payloads[1]["messages"][-1]["content"]
    assert "echo unsafe" not in json.dumps(payloads[1]["messages"])
    assert events[-1]["usage"]["completion_tokens"] == 1126
    starts = {e["tool_call_id"] for e in events if e["type"] == "tool_start"}
    ends = {e["tool_call_id"] for e in events if e["type"] == "tool_end"}
    assert starts <= ends


@pytest.mark.parametrize(
    "reason,prompt,cap",
    [
        ("content_filter", 3000, None),
        ("length", 1000, None),  # A length stop without a full context is not this bug.
        ("length", 3000, 1096),  # Explicit caller cap is spent.
    ],
)
def test_cut_tool_retry_respects_filter_context_and_caller_cap(monkeypatch, reason, prompt, cap):
    backend, payloads = _backend_and_payloads(monkeypatch, [_cut(reason, prompt)])
    calls = _record_tool_calls(monkeypatch, "must not run")
    _run_tool_loop(
        backend,
        [{"role": "user", "content": "Write the report."}],
        [tools.TERMINAL_TOOL],
        permission_mode = "off",
        sandbox_level = "low",
        max_tokens = cap,
    )
    assert not calls
    assert len(payloads) == 1


def test_context_full_tool_retry_is_bounded(monkeypatch):
    backend, payloads = _backend_and_payloads(monkeypatch, [_cut(), _cut(), _cut()])
    monkeypatch.setattr(backend, "count_chat_tokens", lambda *a, **kw: 300)
    calls = _record_tool_calls(monkeypatch, "must not run")
    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Write the report."}],
        [tools.TERMINAL_TOOL],
        permission_mode = "off",
        sandbox_level = "low",
    )
    assert not calls
    assert len(payloads) == 3
    assert events[-1]["finish_reason"] == "length"
    assert events[-1]["usage"]["completion_tokens"] == 3288


@pytest.mark.parametrize("step", [1, 3, 10000])
@pytest.mark.parametrize("shape", ["xml", "json", "json_whitespace", "structured"])
def test_hidden_final_call_gets_one_answer_only_retry(monkeypatch, shape, step):
    hidden = {
        "xml": _answer(
            "<tool_call><function=web_search><parameter=query>again</parameter></function></tool_call>"
        ),
        "json": _answer('{"name":"web_search","parameters":{"query":"again"}}'),
        "json_whitespace": _answer(' \n{"name":"web_search","parameters":{"query":"again"}}\n'),
        "structured": [
            _tool_call_sse("web_search", {"query": "again"}, "hidden"),
            _finish("tool_calls"),
            _usage(300, 10),
            _done(),
        ],
    }[shape]
    if shape != "structured":
        text = json.loads(hidden[0][6:])["choices"][0]["delta"]["content"]
        hidden = [
            _sse({"content": text[i : i + step]}) for i in range(0, len(text), step)
        ] + hidden[1:]
    backend, payloads = _backend_and_payloads(
        monkeypatch,
        [
            [
                _tool_call_sse("web_search", {"query": "first"}, "first"),
                _finish("tool_calls"),
                _usage(300, 20),
                _done(),
            ],
            hidden,
            _answer("Verified result: safe."),
        ],
    )
    monkeypatch.setattr(backend, "count_chat_tokens", lambda *a, **kw: 300)
    calls = _record_tool_calls(monkeypatch, "safe")
    events = _run_tool_loop(backend, [{"role": "user", "content": "Search."}], [_web_search_tool()])
    assert calls == [("web_search", {"query": "first"})]
    assert len(payloads) == 3
    assert "tools" not in payloads[-1]
    assert "tool_choice" not in payloads[-1]
    assert "Give your final answer" in payloads[-1]["messages"][-1]["content"]
    assert events[-1]["usage"]["completion_tokens"] == 40
    shown = [e["text"] for e in events if e["type"] == "content"]
    assert shown[-1] == "Verified result: safe."
    # The route also sanitizes each snapshot; partial XML may be held there.
    from routes.inference import _strip_tool_xml_for_display

    assert all(
        "<tool_call>"
        not in _strip_tool_xml_for_display(
            t, auto_heal_tool_calls = True, enabled_tool_names = {"web_search"}
        )
        for t in shown
    )


def test_repeated_hidden_final_call_is_an_error_not_empty_success(monkeypatch):
    hidden = _answer('{"name":"web_search","parameters":{"query":"again"}}')
    backend, payloads = _backend_and_payloads(
        monkeypatch,
        [
            [
                _tool_call_sse("web_search", {"query": "first"}, "first"),
                _finish("tool_calls"),
                _done(),
            ],
            hidden,
            hidden,
        ],
    )
    monkeypatch.setattr(backend, "count_chat_tokens", lambda *a, **kw: 300)
    _record_tool_calls(monkeypatch, "safe")
    with pytest.raises(RuntimeError, match = "could not produce a final answer"):
        _run_tool_loop(backend, [{"role": "user", "content": "Search."}], [_web_search_tool()])
    assert len(payloads) == 3


@pytest.mark.parametrize("max_tokens,expected_payloads", [(30, 2), (100, 3)])
def test_final_recovery_keeps_the_callers_remaining_output_cap(
    monkeypatch, max_tokens, expected_payloads
):
    hidden = _answer('{"name":"web_search","parameters":{"query":"again"}}')
    backend, payloads = _backend_and_payloads(
        monkeypatch,
        [
            [
                _tool_call_sse("web_search", {"query": "first"}, "first"),
                _finish("tool_calls"),
                _usage(300, 20),
                _done(),
            ],
            hidden,
            _answer("Verified result: safe."),
        ],
    )
    monkeypatch.setattr(backend, "count_chat_tokens", lambda *a, **kw: 300)
    calls = _record_tool_calls(monkeypatch, "safe")
    if max_tokens == 30:
        with pytest.raises(RuntimeError, match = "could not produce a final answer"):
            _run_tool_loop(
                backend,
                [{"role": "user", "content": "Search."}],
                [_web_search_tool()],
                max_tokens = max_tokens,
            )
    else:
        events = _run_tool_loop(
            backend,
            [{"role": "user", "content": "Search."}],
            [_web_search_tool()],
            max_tokens = max_tokens,
        )
        assert payloads[-1]["max_tokens"] == 70
        assert events[-1]["usage"]["completion_tokens"] == 40
    assert len(payloads) == expected_payloads
    assert len(calls) == 1


def test_context_retry_does_not_send_an_unservable_prompt(monkeypatch):
    backend, payloads = _backend_and_payloads(monkeypatch, [_cut()])
    monkeypatch.setattr(backend, "count_chat_tokens", lambda *a, **kw: 4096)
    calls = _record_tool_calls(monkeypatch, "must not run")
    events = _run_tool_loop(
        backend,
        [{"role": "user", "content": "Write the report."}],
        [tools.TERMINAL_TOOL],
        permission_mode = "off",
        sandbox_level = "low",
    )
    assert not calls
    assert len(payloads) == 1
    assert events[-1]["finish_reason"] == "length"


def test_empty_answer_after_final_retry_is_not_success(monkeypatch):
    backend, payloads = _backend_and_payloads(
        monkeypatch,
        [
            [
                _tool_call_sse("web_search", {"query": "first"}, "first"),
                _finish("tool_calls"),
                _done(),
            ],
            _answer('{"name":"web_search","parameters":{"query":"again"}}'),
            _answer(""),
        ],
    )
    monkeypatch.setattr(backend, "count_chat_tokens", lambda *a, **kw: 300)
    _record_tool_calls(monkeypatch, "safe")
    with pytest.raises(RuntimeError, match = "could not produce a final answer"):
        _run_tool_loop(backend, [{"role": "user", "content": "Search."}], [_web_search_tool()])
    assert len(payloads) == 3


@pytest.mark.parametrize("charge_schemas", [False, True])
def test_final_retry_after_user_nudge_uses_servable_prompt(monkeypatch, charge_schemas):
    from core.inference.llama_cpp import _MAX_TOOL_CALLS_PER_TURN

    # Skipping two different tools creates a user notice after the tool results.
    calls = "".join(
        '<tool_call>{"name":"web_search","arguments":{"query":"q%d"}}</tool_call>' % i
        for i in range(_MAX_TOOL_CALLS_PER_TURN + 1)
    )
    calls += '<tool_call>{"name":"python","arguments":{"code":"print(1)"}}</tool_call>'
    backend, payloads = _backend_and_payloads(
        monkeypatch,
        [
            [_sse({"content": calls}), _done()],
            _answer('{"name":"web_search","parameters":{"query":"again"}}'),
            _answer("Verified result: safe."),
        ],
    )
    retry_counts = []

    def count(messages, _system, schemas, **kwargs):
        if "Give your final answer now" in messages[-1]["content"]:
            retry_counts.append((schemas, kwargs.get("continue_final_message")))
            if charge_schemas and schemas:
                return 4096
        return 300

    monkeypatch.setattr(backend, "count_chat_tokens", count)
    executed = _record_tool_calls(monkeypatch, "safe")
    _run_tool_loop(
        backend,
        [{"role": "user", "content": "Search."}],
        [_web_search_tool(), tools.PYTHON_TOOL],
    )

    assert len(executed) == _MAX_TOOL_CALLS_PER_TURN
    assert len(payloads) == 3
    original_nudge = payloads[1]["messages"][-1]
    assert original_nudge["role"] == "user"
    assert "more tool call(s)" in original_nudge["content"]
    retry = payloads[-1]
    assert retry["messages"][-3] == original_nudge
    assert [m["role"] for m in retry["messages"][-3:]] == ["user", "assistant", "user"]
    assert retry_counts == [(None, False)]
    assert "tools" not in retry
    assert retry["add_generation_prompt"] is True
