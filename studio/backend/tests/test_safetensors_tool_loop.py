# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the safetensors agentic tool loop.

Covers the ``tool_call_parser`` helpers and the cumulative-text state machine in
``run_safetensors_tool_loop``, run against fake single-turn generators (no model
load). Edge cases: plain answers, JSON and XML tool-call forms, truncated/unclosed
calls, tool-result feedback, bad-JSON heal, duplicate-call short-circuit,
``__IMAGES__`` sentinel stripping, executor errors, cancel, and the iteration cap.
"""

import threading
from typing import cast

import pytest

from core.inference import safetensors_agentic
from core.inference.safetensors_agentic import (
    _coerce_arguments,
    _detect_render_html_tool_start,
    run_safetensors_tool_loop,
    strip_tool_markup_streaming,
)
from core.inference.tool_call_parser import (
    RAG_MAX_SEARCHES_PER_TURN,
    has_tool_signal,
    parse_tool_calls_from_text,
    strip_tool_markup,
)
from state import tool_approvals
from state.tool_approvals import resolve_tool_decision
from utils.datasets import is_gpt_oss_model_name


# ────────────────────────────────────────────────────────────────────
# parse_tool_calls_from_text
# ────────────────────────────────────────────────────────────────────


class TestParser:
    def test_json_tool_call(self):
        text = '<tool_call>{"name":"web_search","arguments":{"query":"hello"}}</tool_call>'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        tc = result[0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "web_search"
        # Arguments must always be a JSON string.
        assert isinstance(tc["function"]["arguments"], str)
        assert "hello" in tc["function"]["arguments"]

    def test_json_tool_call_unclosed(self):
        # No </tool_call>; balanced-brace extractor must still close it.
        text = '<tool_call>{"name":"python","arguments":{"code":"print(1)"}}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "python"

    def test_json_tool_call_unclosed_requires_healing(self):
        text = '<tool_call>{"name":"python","arguments":{"code":"print(1)"}}'
        assert parse_tool_calls_from_text(text)[0]["function"]["name"] == "python"
        assert parse_tool_calls_from_text(text, allow_incomplete = False) == []

    def test_xml_function_call(self):
        text = "<function=python><parameter=code>print('hi')</parameter></function>"
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "python"
        assert "print('hi')" in result[0]["function"]["arguments"]

    def test_xml_unclosed(self):
        # Closing tags omitted; parser must still extract the value.
        text = "<function=terminal><parameter=command>ls -la"
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "terminal"
        assert "ls -la" in result[0]["function"]["arguments"]

    def test_xml_unclosed_requires_healing(self):
        text = "<function=terminal><parameter=command>ls -la"
        assert parse_tool_calls_from_text(text)[0]["function"]["name"] == "terminal"
        assert parse_tool_calls_from_text(text, allow_incomplete = False) == []

    def test_code_with_embedded_xml(self):
        # A code parameter with a literal </parameter> must not truncate: the
        # parser uses end-of-body as the only boundary for single-param calls.
        text = (
            "<function=python><parameter=code>html = '<a></a>'\nprint('hi')</parameter></function>"
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert "print('hi')" in result[0]["function"]["arguments"]

    def test_function_signal_inside_parameter_is_literal(self):
        text = (
            "<function=python>"
            "<parameter=code>print('<function=render_html>')</parameter>"
            "</function>"
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "python"
        assert "<function=render_html>" in result[0]["function"]["arguments"]

    def test_multiple_calls(self):
        text = (
            '<tool_call>{"name":"web_search","arguments":{"query":"a"}}</tool_call>'
            '<tool_call>{"name":"web_search","arguments":{"query":"b"}}</tool_call>'
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "web_search"
        assert result[1]["function"]["name"] == "web_search"

    def test_bad_json_does_not_raise(self):
        text = "<tool_call>{not valid json}</tool_call>"
        result = parse_tool_calls_from_text(text)
        # Bad JSON is dropped silently; caller can fall back to text.
        assert result == []

    def test_has_tool_signal(self):
        assert has_tool_signal("blah <tool_call> x")
        assert has_tool_signal("hi <function=foo>...")
        assert not has_tool_signal("hello world")

    def test_render_html_start_detector_uses_first_tool(self):
        assert _detect_render_html_tool_start("<function=render_html>")
        assert _detect_render_html_tool_start(
            '<tool_call>{"name":"render_html","arguments":{"code":"<html>"}'
        )
        assert not _detect_render_html_tool_start(
            "<function=python><parameter=code>'<function=render_html>'"
        )
        assert not _detect_render_html_tool_start(
            '<tool_call>{"name":"python","arguments":{"code":"<function=render_html>"}}'
        )

    def test_strip_markup_closed(self):
        text = "before <tool_call>{}</tool_call> after"
        assert strip_tool_markup(text) == "before  after"

    def test_strip_markup_unclosed_final(self):
        text = "before <tool_call>{partial"
        # final=True drops the trailing run.
        assert strip_tool_markup(text, final = True) == "before"
        # Without final=True the unclosed run is preserved.
        assert "partial" in strip_tool_markup(text)

    def test_streaming_strip_respects_disabled_healing(self):
        raw = 'before <tool_call>{"name":"web_search"'
        assert strip_tool_markup_streaming(raw, auto_heal_tool_calls = False) == raw
        assert strip_tool_markup_streaming(raw) == "before "

    def test_streaming_strip_respects_disabled_healing_without_tool_protocol(self):
        raw = 'before <tool_call>{"name":"web_search"'
        assert strip_tool_markup_streaming(raw, auto_heal_tool_calls = False) == raw
        assert (
            strip_tool_markup_streaming(
                raw,
                auto_heal_tool_calls = False,
                tool_protocol_active = True,
            )
            == "before "
        )


# ────────────────────────────────────────────────────────────────────
# run_safetensors_tool_loop
# ────────────────────────────────────────────────────────────────────


def _fake_stream(chunks):
    """Build a single-turn generator that yields cumulative snapshots."""

    def _gen(_messages):
        acc = ""
        for c in chunks:
            acc += c
            yield acc

    return _gen


def _const_stream(text):
    """A single-turn generator that yields one cumulative snapshot."""

    def _gen(_messages):
        yield text

    return _gen


class FakeExecuteTool:
    """Stand-in for ``core.inference.tools.execute_tool``."""

    def __init__(self, results):
        # ``results`` is a list of strings or RuntimeError instances.
        self.results = list(results)
        self.calls: list[tuple[str, dict]] = []

    def __call__(
        self,
        name,
        arguments,
        *,
        cancel_event = None,
        timeout = None,
        session_id = None,
        rag_scope = None,
        disable_sandbox = False,
    ):
        self.calls.append((name, arguments))
        result = self.results.pop(0) if self.results else "OK"
        if isinstance(result, Exception):
            raise result
        return result


def _collect_events(generator, max_events = 200):
    events = []
    for ev in generator:
        events.append(ev)
        if len(events) >= max_events:
            break
    return events


def _make_loop(
    *,
    turns,
    exec_results = None,
    **kwargs,
):
    """Build a configured loop with a multi-turn fake generator.

    ``turns`` is a list of chunk-lists; iteration N yields chunks from ``turns[N]``.
    """
    turn_iter = iter(turns)

    def _gen(_messages):
        try:
            chunks = next(turn_iter)
        except StopIteration:
            return
        acc = ""
        for c in chunks:
            acc += c
            yield acc

    exec_fn = FakeExecuteTool(exec_results or [])
    return run_safetensors_tool_loop(
        single_turn = _gen,
        messages = [{"role": "user", "content": "hi"}],
        tools = [
            {"type": "function", "function": {"name": "web_search"}},
            {"type": "function", "function": {"name": "python"}},
            {"type": "function", "function": {"name": "terminal"}},
        ],
        execute_tool = exec_fn,
        **kwargs,
    ), exec_fn


def test_active_tools_are_passed_to_single_turn_after_render_html_success():
    captured_tool_names: list[list[str]] = []
    exec_fn = FakeExecuteTool(["Rendered HTML canvas."])

    def fake_single_turn(_messages, *, active_tools = None):
        captured_tool_names.append(
            [
                (tool.get("function") or {}).get("name")
                for tool in (active_tools or [])
                if (tool.get("function") or {}).get("name")
            ]
        )
        if len(captured_tool_names) == 1:
            yield '<tool_call>{"name":"render_html","arguments":{"code":"<html>one</html>"}}</tool_call>'
        else:
            yield "Done."

    events = _collect_events(
        run_safetensors_tool_loop(
            single_turn = fake_single_turn,
            messages = [{"role": "user", "content": "make html"}],
            tools = [
                {"type": "function", "function": {"name": "render_html"}},
                {"type": "function", "function": {"name": "web_search"}},
            ],
            execute_tool = exec_fn,
            max_tool_iterations = 3,
        )
    )

    assert exec_fn.calls == [("render_html", {"code": "<html>one</html>"})]
    assert captured_tool_names == [["render_html", "web_search"], ["web_search"]]
    assert any(event.get("type") == "content" and event.get("text") == "Done." for event in events)


class TestLoopBasic:
    def test_plain_answer(self):
        # No tool XML; loop should yield content then status="".
        loop, _exec = _make_loop(
            turns = [["Hello", " world", "!"]],
            exec_results = [],
        )
        events = _collect_events(loop)
        contents = [e for e in events if e["type"] == "content"]
        statuses = [e for e in events if e["type"] == "status"]
        assert contents, "expected at least one content event"
        # Final cumulative content must contain the answer.
        final_text = contents[-1]["text"]
        assert "Hello world!" in final_text
        assert statuses and statuses[-1]["text"] == ""

    def test_single_tool_then_answer(self):
        loop, exec_fn = _make_loop(
            turns = [
                # Tool call only.
                [
                    '<tool_call>{"name":"web_search",',
                    '"arguments":{"query":"weather"}}',
                    "</tool_call>",
                ],
                # Final answer.
                ["The ", "weather is ", "sunny."],
            ],
            exec_results = ["Sunny and 22C"],
        )
        events = _collect_events(loop)
        kinds = [e["type"] for e in events]

        assert "tool_start" in kinds
        assert "tool_end" in kinds
        # Tool was called with the parsed arguments.
        assert exec_fn.calls == [("web_search", {"query": "weather"})]

        tool_start = next(e for e in events if e["type"] == "tool_start")
        assert tool_start["tool_name"] == "web_search"
        tool_end = next(e for e in events if e["type"] == "tool_end")
        assert tool_end["result"] == "Sunny and 22C"

        contents = [e for e in events if e["type"] == "content"]
        assert contents and "sunny" in contents[-1]["text"].lower()

    def test_function_xml_form(self):
        loop, exec_fn = _make_loop(
            turns = [
                ["<function=python><parameter=code>print(1)</parameter></function>"],
                ["Result: 1"],
            ],
            exec_results = ["1\n"],
        )
        events = _collect_events(loop)
        assert exec_fn.calls == [("python", {"code": "print(1)"})]
        contents = [e for e in events if e["type"] == "content"]
        assert "Result: 1" in contents[-1]["text"]

    def test_render_html_emits_provisional_tool_start(self):
        exec_fn = FakeExecuteTool(["Rendered HTML canvas."])
        turn_iter = iter(
            [
                [
                    "<function=render_html>",
                    "<parameter=code><!doctype html><html>",
                    "<body>Hi</body></html></parameter></function>",
                ],
                ["Done."],
            ]
        )

        def _gen(_messages):
            chunks = next(turn_iter)
            acc = ""
            for chunk in chunks:
                acc += chunk
                yield acc

        loop = run_safetensors_tool_loop(
            single_turn = _gen,
            messages = [{"role": "user", "content": "make html"}],
            tools = [{"type": "function", "function": {"name": "render_html"}}],
            execute_tool = exec_fn,
        )
        events = _collect_events(loop)
        tool_starts = [e for e in events if e["type"] == "tool_start"]

        assert len(tool_starts) == 2
        assert tool_starts[0]["tool_name"] == "render_html"
        assert tool_starts[0]["arguments"] == {}
        assert tool_starts[1]["tool_name"] == "render_html"
        assert "<!doctype html>" in tool_starts[1]["arguments"]["code"]
        assert exec_fn.calls[0][0] == "render_html"
        assert "<!doctype html>" in exec_fn.calls[0][1]["code"]

    def test_render_html_confirmation_gate_suppresses_early_provisional(self, monkeypatch):
        """When a human confirmation gate is active, render_html must not surface
        an early provisional tool_start: that card (keyed by tool_call_id, no
        approval) would show the tool 'running' before the user approves. The
        gated real tool_start is the first signal the UI receives instead."""
        monkeypatch.setattr(safetensors_agentic, "new_approval_id", lambda: "approval-rh")
        monkeypatch.setattr(safetensors_agentic, "begin_tool_decision", lambda *_a, **_k: object())
        monkeypatch.setattr(safetensors_agentic, "wait_tool_decision", lambda *_a, **_k: "allow")

        exec_fn = FakeExecuteTool(["Rendered HTML canvas."])
        turn_iter = iter(
            [
                [
                    "<function=render_html>",
                    "<parameter=code><!doctype html><html>",
                    "<body>Hi</body></html></parameter></function>",
                ],
                ["Done."],
            ]
        )

        def _gen(_messages):
            chunks = next(turn_iter)
            acc = ""
            for chunk in chunks:
                acc += chunk
                yield acc

        loop = run_safetensors_tool_loop(
            single_turn = _gen,
            messages = [{"role": "user", "content": "make html"}],
            tools = [{"type": "function", "function": {"name": "render_html"}}],
            execute_tool = exec_fn,
            confirm_tool_calls = True,
            session_id = "sess",
            max_tool_iterations = 3,
        )
        events = _collect_events(loop)
        tool_starts = [e for e in events if e["type"] == "tool_start"]

        # No early provisional (empty-args) card while confirmation is pending.
        assert [e for e in tool_starts if e.get("arguments") == {}] == []
        # The real, gated tool_start still surfaces with the full arguments.
        real = [e for e in tool_starts if e.get("arguments", {}).get("code")]
        assert len(real) == 1
        assert real[0].get("awaiting_confirmation") is True
        assert "<!doctype html>" in real[0]["arguments"]["code"]
        assert exec_fn.calls[0][0] == "render_html"

    def test_render_html_bypass_permissions_keeps_early_provisional(self, monkeypatch):
        """bypass_permissions wins over the confirm gate, so the early provisional
        card is preserved (no human approval is required)."""
        exec_fn = FakeExecuteTool(["Rendered HTML canvas."])
        turn_iter = iter(
            [
                [
                    "<function=render_html>",
                    "<parameter=code><!doctype html><html>",
                    "<body>Hi</body></html></parameter></function>",
                ],
                ["Done."],
            ]
        )

        def _gen(_messages):
            chunks = next(turn_iter)
            acc = ""
            for chunk in chunks:
                acc += chunk
                yield acc

        loop = run_safetensors_tool_loop(
            single_turn = _gen,
            messages = [{"role": "user", "content": "make html"}],
            tools = [{"type": "function", "function": {"name": "render_html"}}],
            execute_tool = exec_fn,
            confirm_tool_calls = True,
            bypass_permissions = True,
            session_id = "sess",
            max_tool_iterations = 3,
        )
        events = _collect_events(loop)
        tool_starts = [e for e in events if e["type"] == "tool_start"]

        assert len(tool_starts) == 2
        assert tool_starts[0]["arguments"] == {}
        assert "<!doctype html>" in tool_starts[1]["arguments"]["code"]

    def test_render_html_provisional_card_closed_on_generator_exception(self):
        """If the model generator raises mid-stream after a provisional render_html
        card was surfaced, the loop must close that card as errored before the
        exception propagates, so the UI never leaves a tool spinning forever."""
        exec_fn = FakeExecuteTool([])

        def _gen(_messages):
            acc = ""
            for chunk in ["<function=render_html>", "<parameter=code><!doctype html><html>"]:
                acc += chunk
                yield acc
            raise RuntimeError("model pipeline exploded")

        loop = run_safetensors_tool_loop(
            single_turn = _gen,
            messages = [{"role": "user", "content": "make html"}],
            tools = [{"type": "function", "function": {"name": "render_html"}}],
            execute_tool = exec_fn,
        )

        collected: list[dict] = []
        raised = False
        try:
            for event in loop:
                collected.append(event)
        except RuntimeError as exc:
            raised = True
            assert "exploded" in str(exc)

        assert raised
        provisional = [
            e for e in collected if e["type"] == "tool_start" and e.get("arguments") == {}
        ]
        assert len(provisional) == 1
        # The provisional card is closed (as an error) before the exception
        # propagates, so it never dangles.
        closing = [
            e
            for e in collected
            if e["type"] == "tool_end" and e.get("tool_call_id") == provisional[0]["tool_call_id"]
        ]
        assert len(closing) == 1
        assert "Error" in (closing[0].get("result") or "")

    def test_python_tool_containing_render_html_signal_does_not_emit_provisional_start(self):
        loop, exec_fn = _make_loop(
            turns = [
                [
                    "<function=python>",
                    "<parameter=code>print('<function=render_html>')",
                    "</parameter></function>",
                ],
                ["Done."],
            ],
            exec_results = ["ok"],
        )
        events = _collect_events(loop)
        tool_starts = [e for e in events if e["type"] == "tool_start"]

        assert len(tool_starts) == 1
        assert tool_starts[0]["tool_name"] == "python"
        assert exec_fn.calls == [("python", {"code": "print('<function=render_html>')"})]

    def test_render_html_success_blocks_second_canvas_call(self):
        exec_fn = FakeExecuteTool(["Rendered HTML canvas."])
        turn_iter = iter(
            [
                [
                    '<tool_call>{"name":"render_html",',
                    '"arguments":{"code":"<html>one</html>"}}',
                ],
                [
                    '<tool_call>{"name":"render_html",',
                    '"arguments":{"code":"<html>two</html>"}}',
                ],
                ["Done."],
            ]
        )

        def _gen(_messages):
            chunks = next(turn_iter)
            acc = ""
            for chunk in chunks:
                acc += chunk
                yield acc

        loop = run_safetensors_tool_loop(
            single_turn = _gen,
            messages = [{"role": "user", "content": "make html"}],
            tools = [{"type": "function", "function": {"name": "render_html"}}],
            execute_tool = exec_fn,
        )
        events = _collect_events(loop)
        tool_starts = [e for e in events if e["type"] == "tool_start"]

        assert exec_fn.calls == [("render_html", {"code": "<html>one</html>"})]
        assert [e["arguments"] for e in tool_starts] == [{}, {"code": "<html>one</html>"}]

    def test_truncated_unclosed_tool_call(self):
        loop, exec_fn = _make_loop(
            turns = [
                # No </tool_call>; balanced-brace parser still succeeds because
                # the JSON itself is balanced.
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}'],
                ["done"],
            ],
            exec_results = ["result"],
        )
        events = _collect_events(loop)
        assert exec_fn.calls == [("web_search", {"query": "x"})]

    def test_bad_json_healed_to_query(self):
        # Non-JSON string arguments heal to {"query": ...} under auto_heal_tool_calls.
        loop, exec_fn = _make_loop(
            turns = [
                # ``arguments`` is a string _coerce_arguments can't parse, so heal runs.
                ['<tool_call>{"name":"web_search","arguments":"hello world"}</tool_call>'],
                ["ok"],
            ],
            exec_results = ["..."],
        )
        events = _collect_events(loop)
        assert exec_fn.calls and exec_fn.calls[0][0] == "web_search"
        assert exec_fn.calls[0][1] == {"query": "hello world"}


class TestLoopBehaviour:
    def test_duplicate_tool_call_internal_noop(self):
        captured_messages: list[list[dict]] = []
        turns = iter(
            [
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                ["final"],
            ]
        )

        def fake_single_turn(messages):
            captured_messages.append([dict(message) for message in messages])
            chunks = next(turns)
            acc = ""
            for chunk in chunks:
                acc += chunk
                yield acc

        exec_fn = FakeExecuteTool(["search-result-1"])
        events = _collect_events(
            run_safetensors_tool_loop(
                single_turn = fake_single_turn,
                messages = [{"role": "user", "content": "hi"}],
                tools = [{"type": "function", "function": {"name": "web_search"}}],
                execute_tool = exec_fn,
                max_tool_iterations = 3,
            )
        )

        assert exec_fn.calls == [("web_search", {"query": "x"})]
        assert [e["tool_call_id"] for e in events if e["type"] == "tool_end"] == ["call_0"]
        assert not [
            e
            for e in events
            if e.get("tool_call_id") == "call_1" and e.get("type") in {"tool_start", "tool_end"}
        ]
        duplicate_nudges = [
            message
            for message in captured_messages[-1]
            if message.get("role") == "user"
            and "already completed successfully" in message.get("content", "")
        ]
        assert len(duplicate_nudges) == 1

    def test_duplicate_tool_call_internal_noop_allows_distinct_followup_tool(self):
        captured_messages: list[list[dict]] = []
        captured_tool_names: list[list[str]] = []
        turns = iter(
            [
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                ['<tool_call>{"name":"python","arguments":{"code":"print(1)"}}</tool_call>'],
                ["final"],
            ]
        )

        def fake_single_turn(messages, active_tools = None):
            captured_messages.append([dict(message) for message in messages])
            captured_tool_names.append(
                [
                    tool["function"]["name"]
                    for tool in (active_tools or [])
                    if tool.get("function", {}).get("name")
                ]
            )
            chunks = next(turns)
            acc = ""
            for chunk in chunks:
                acc += chunk
                yield acc

        exec_fn = FakeExecuteTool(["search-result-1", "python-result"])
        events = _collect_events(
            run_safetensors_tool_loop(
                single_turn = fake_single_turn,
                messages = [{"role": "user", "content": "hi"}],
                tools = [
                    {"type": "function", "function": {"name": "web_search"}},
                    {"type": "function", "function": {"name": "python"}},
                ],
                execute_tool = exec_fn,
                max_tool_iterations = 4,
            )
        )

        assert exec_fn.calls == [
            ("web_search", {"query": "x"}),
            ("python", {"code": "print(1)"}),
        ]
        assert [e["tool_call_id"] for e in events if e["type"] == "tool_end"] == [
            "call_0",
            "call_2",
        ]
        assert not [
            e
            for e in events
            if e.get("tool_call_id") == "call_1" and e.get("type") in {"tool_start", "tool_end"}
        ]
        duplicate_nudges = [
            message
            for message in captured_messages[2]
            if message.get("role") == "user"
            and "already completed successfully" in message.get("content", "")
        ]
        assert len(duplicate_nudges) == 1
        assert captured_tool_names[2] == ["web_search", "python"]

    def test_repeated_duplicate_noop_transitions_to_final_attempt(self):
        captured_tool_names: list[list[str]] = []
        turns = iter(
            [
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                ["final from first result"],
            ]
        )

        def fake_single_turn(messages, active_tools = None):
            captured_tool_names.append(
                [
                    (tool.get("function") or {}).get("name")
                    for tool in (active_tools or [])
                    if (tool.get("function") or {}).get("name")
                ]
            )
            chunks = next(turns)
            acc = ""
            for chunk in chunks:
                acc += chunk
                yield acc

        exec_fn = FakeExecuteTool(["search-result"])
        events = _collect_events(
            run_safetensors_tool_loop(
                single_turn = fake_single_turn,
                messages = [{"role": "user", "content": "hi"}],
                tools = [{"type": "function", "function": {"name": "web_search"}}],
                execute_tool = exec_fn,
                max_tool_iterations = 10,
            )
        )

        assert exec_fn.calls == [("web_search", {"query": "x"})]
        assert [
            event.get("tool_call_id") for event in events if event.get("type") == "tool_end"
        ] == ["call_0"]
        assert captured_tool_names[-1] == []
        assert any(
            event.get("type") == "content" and "final from first result" in event.get("text", "")
            for event in events
        )

    def test_kb_search_capped_per_turn(self):
        # Paraphrased KB searches differ by args (dup guard misses them); the
        # per-turn cap stops the runaway re-search loop.
        n = RAG_MAX_SEARCHES_PER_TURN
        queries = [f"paraphrase {i}" for i in range(n + 1)]
        turns = [
            [
                '<tool_call>{"name":"search_knowledge_base",'
                f'"arguments":{{"query":"{q}"}}}}</tool_call>'
            ]
            for q in queries
        ] + [["final answer"]]
        turn_iter = iter(turns)

        def _gen(_messages):
            try:
                chunks = next(turn_iter)
            except StopIteration:
                return
            acc = ""
            for c in chunks:
                acc += c
                yield acc

        exec_fn = FakeExecuteTool([f"chunk-{i}" for i in range(n)])
        loop = run_safetensors_tool_loop(
            single_turn = _gen,
            messages = [{"role": "user", "content": "hi"}],
            tools = [{"type": "function", "function": {"name": "search_knowledge_base"}}],
            execute_tool = exec_fn,
        )
        events = _collect_events(loop)
        assert len(exec_fn.calls) == n
        assert all(c[0] == "search_knowledge_base" for c in exec_fn.calls)
        tool_end_events = [e for e in events if e["type"] == "tool_end"]
        assert len(tool_end_events) == n + 1
        assert "do not search again" in tool_end_events[n]["result"].lower()

    def test_image_sentinel_stripped_from_model_feed(self):
        # The image sentinel is stripped before the next turn, but tool_end still
        # carries the raw result for the UI.
        loop, exec_fn = _make_loop(
            turns = [
                ['<tool_call>{"name":"python","arguments":{"code":"plot()"}}</tool_call>'],
                ["see chart"],
            ],
            exec_results = ["chart\n__IMAGES__:/tmp/chart.png"],
        )
        events = _collect_events(loop)
        tool_end = next(e for e in events if e["type"] == "tool_end")
        assert "__IMAGES__" in tool_end["result"]

    def test_image_sentinel_stripped_with_leading_marker(self):
        # Sentinel at start (no newline) must not leak to the model.
        from core.inference import safetensors_agentic as _sa

        captured: list[list[dict]] = []

        def fake_single_turn(messages, **_kw):
            captured.append([dict(m) for m in messages])
            if len(captured) == 1:
                yield '<tool_call>{"name":"python","arguments":{"code":"plot()"}}</tool_call>'
            else:
                yield "done"

        events = list(
            _sa.run_safetensors_tool_loop(
                single_turn = fake_single_turn,
                messages = [{"role": "user", "content": "plot please"}],
                tools = [{"function": {"name": "python"}}],
                execute_tool = lambda *_a, **_kw: "__IMAGES__:/tmp/x.png",
                cancel_event = threading.Event(),
                max_tool_iterations = 3,
                auto_heal_tool_calls = True,
            )
        )
        # The model's second turn must not see "__IMAGES__".
        assert len(captured) >= 2
        tool_msgs = [m for m in captured[1] if m.get("role") == "tool"]
        assert tool_msgs, "no tool message reached the model"
        for tm in tool_msgs:
            assert "__IMAGES__" not in tm["content"], f"sentinel leaked to model: {tm['content']!r}"

    def test_image_sentinel_stripped_with_multiple_markers(self):
        # Consecutive sentinels: cut at the first, nothing leaks.
        from core.inference import safetensors_agentic as _sa

        captured: list[list[dict]] = []

        def fake_single_turn(messages, **_kw):
            captured.append([dict(m) for m in messages])
            if len(captured) == 1:
                yield '<tool_call>{"name":"python","arguments":{"code":"plot()"}}</tool_call>'
            else:
                yield "done"

        multi = "panel\n__IMAGES__:/tmp/a.png\n__IMAGES__:/tmp/b.png"
        events = list(
            _sa.run_safetensors_tool_loop(
                single_turn = fake_single_turn,
                messages = [{"role": "user", "content": "plot please"}],
                tools = [{"function": {"name": "python"}}],
                execute_tool = lambda *_a, **_kw: multi,
                cancel_event = threading.Event(),
                max_tool_iterations = 3,
                auto_heal_tool_calls = True,
            )
        )
        tool_msgs = [m for m in captured[1] if m.get("role") == "tool"]
        assert tool_msgs
        for tm in tool_msgs:
            assert "__IMAGES__" not in tm["content"], f"second sentinel leaked: {tm['content']!r}"
            assert tm["content"] == "panel", f"expected payload-only 'panel', got {tm['content']!r}"

    def test_tool_execution_error_is_emitted_but_loop_continues(self):
        loop, exec_fn = _make_loop(
            turns = [
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                ["sorry, that failed"],
            ],
            exec_results = ["Error: network unreachable"],
        )
        events = _collect_events(loop)
        tool_end = next(e for e in events if e["type"] == "tool_end")
        assert tool_end["result"].startswith("Error")
        # The loop must still emit a content event after the failure.
        contents = [e for e in events if e["type"] == "content"]
        assert contents

    def test_exception_in_executor_does_not_raise(self):
        loop, exec_fn = _make_loop(
            turns = [
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                ["recovered"],
            ],
            exec_results = [RuntimeError("boom")],
        )
        events = _collect_events(loop)
        tool_end = next(e for e in events if e["type"] == "tool_end")
        assert "boom" in tool_end["result"]


class TestLoopControl:
    def test_cancel_event_breaks_loop(self):
        cancel = threading.Event()
        cancel.set()
        # With cancel set, the loop bails before invoking execute_tool.
        exec_fn = FakeExecuteTool([])
        events = list(
            run_safetensors_tool_loop(
                single_turn = _const_stream(
                    '<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'
                ),
                messages = [{"role": "user", "content": "hi"}],
                tools = [],
                execute_tool = exec_fn,
                cancel_event = cancel,
            )
        )
        assert events == []
        assert exec_fn.calls == []

    def test_max_iterations_caps_loop(self):
        # The loop stops after max_tool_iterations even if the model keeps
        # asking for tools, then emits a final-attempt round.
        loop, exec_fn = _make_loop(
            turns = [
                # Tool call (executes once).
                ['<tool_call>{"name":"web_search","arguments":{"query":"a"}}</tool_call>'],
                # Model gives a final answer when nudged.
                ["here is the final answer"],
            ],
            exec_results = ["result"],
            max_tool_iterations = 1,
        )
        events = _collect_events(loop)
        contents = [e for e in events if e["type"] == "content"]
        # Final content must contain the final answer.
        assert contents and "final answer" in contents[-1]["text"]


class TestStatusFormatting:
    def test_status_for_known_tools(self):
        # Call the private helper directly to verify status formatting.
        assert (
            safetensors_agentic._status_for_tool("web_search", {"query": "abc"}) == "Searching: abc"
        )
        assert (
            safetensors_agentic._status_for_tool("web_search", {"url": "https://www.example.com/x"})
            == "Reading: example.com"
        )
        assert safetensors_agentic._status_for_tool("python", {"code": "x = 1"}).startswith(
            "Running Python:"
        )
        assert safetensors_agentic._status_for_tool("terminal", {"command": "ls"}).startswith(
            "Running:"
        )
        assert safetensors_agentic._status_for_tool("unknown_tool", {}).startswith("Calling:")


class TestProseMentioningToolCall:
    def test_assistant_prose_with_literal_tool_call_text_survives(self):
        # Regression: prose that mentions a literal ``<tool_call>`` (no real call)
        # must surface in full, not be stripped past the marker.
        loop, exec_fn = _make_loop(
            turns = [
                # A real tool call so the loop advances a turn.
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                # Prose that mentions the literal text.
                ["the docs say <tool_call> means an LLM tool call wrapper"],
            ],
            exec_results = ["result"],
        )
        events = _collect_events(loop)
        contents = [e for e in events if e["type"] == "content"]
        assert contents, "expected at least one content event"
        final = contents[-1]["text"]
        assert (
            "LLM tool" in final
        ), f"prose mentioning <tool_call> should not be truncated; got {final!r}"

    def test_tool_result_with_tool_call_text_does_not_retrigger(self):
        # A literal ``<tool_call>`` in the tool result must not re-trigger: the
        # loop parses only model output, so exactly one call.
        loop, exec_fn = _make_loop(
            turns = [
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                ["the docs mention <tool_call> wrappers"],
            ],
            exec_results = ["Page text: <tool_call> appears here in the docs"],
        )
        events = _collect_events(loop)
        assert len(exec_fn.calls) == 1


class TestChatTemplateHelper:
    """Cover the dependency-light helper used by InferenceBackend."""

    def setup_method(self):
        from core.inference.chat_template_helpers import (
            apply_chat_template_for_generation,
        )
        self.apply = apply_chat_template_for_generation

    class _Tok:
        def __init__(self, accepted):
            self.accepted = accepted
            self.call_count = 0
            self.last_kwargs = None

        def apply_chat_template(
            self,
            messages,
            *,
            tokenize = False,
            add_generation_prompt = True,
            **kw,
        ):
            self.call_count += 1
            unknown = set(kw) - self.accepted
            if unknown:
                raise TypeError(f"unexpected kwargs: {sorted(unknown)}")
            self.last_kwargs = dict(kw)
            return "PROMPT"

    def test_richest_call_wins_when_template_supports_all(self):
        tok = self._Tok({"tools", "enable_thinking"})
        self.apply(tok, [], tools = [{}], enable_thinking = True)
        assert tok.call_count == 1
        assert tok.last_kwargs is not None
        assert "tools" in tok.last_kwargs
        assert "enable_thinking" in tok.last_kwargs

    def test_falls_back_when_template_rejects_reasoning_kwarg(self):
        tok = self._Tok({"tools"})
        self.apply(tok, [], tools = [{}], enable_thinking = True)
        assert tok.call_count >= 2
        assert tok.last_kwargs == {"tools": [{}]}

    def test_falls_back_to_bare_call(self):
        tok = self._Tok(set())
        self.apply(tok, [], tools = [{}], enable_thinking = True)
        assert tok.last_kwargs == {}

    def test_jinja_error_propagates(self):
        class Boom:
            def apply_chat_template(self, *a, **kw):
                raise ValueError("jinja: missing var")

        with pytest.raises(ValueError):
            self.apply(Boom(), [])

    def test_no_kwargs_single_call(self):
        tok = self._Tok(set())
        self.apply(tok, [])
        assert tok.call_count == 1


# ────────────────────────────────────────────────────────────────────
# Guardrails (allowlist, budget, streaming-leak, dedup, id offset,
# auto_heal=False, canonical healed-arg key)
# ────────────────────────────────────────────────────────────────────


class TestGuardrails:
    def test_disabled_tool_is_not_executed(self):
        captured_messages: list[list[dict]] = []

        def fake_single_turn(messages):
            captured_messages.append([dict(message) for message in messages])
            if len(captured_messages) == 1:
                yield '<tool_call>{"name":"terminal","arguments":{"command":"echo bypass"}}</tool_call>'
            else:
                yield "final"

        exec_fn = FakeExecuteTool([])
        events = _collect_events(
            run_safetensors_tool_loop(
                single_turn = fake_single_turn,
                messages = [{"role": "user", "content": "hi"}],
                tools = [{"type": "function", "function": {"name": "web_search"}}],
                execute_tool = exec_fn,
                max_tool_iterations = 2,
            )
        )

        assert exec_fn.calls == []
        assert not [event for event in events if event.get("type") in {"tool_start", "tool_end"}]
        disabled_nudges = [
            message
            for message in captured_messages[-1]
            if message.get("role") == "user" and "not enabled" in message.get("content", "")
        ]
        assert len(disabled_nudges) == 1

    def test_empty_tools_list_means_allow_all_in_core_loop(self):
        turns = iter(
            [
                ['<tool_call>{"name":"python","arguments":{"code":"print(1)"}}</tool_call>'],
                ["done"],
            ]
        )

        def fake_single_turn(_messages, active_tools = None):
            assert active_tools == []
            acc = ""
            for chunk in next(turns):
                acc += chunk
                yield acc

        exec_fn = FakeExecuteTool(["OK"])
        events = _collect_events(
            run_safetensors_tool_loop(
                single_turn = fake_single_turn,
                messages = [{"role": "user", "content": "hi"}],
                tools = [],
                execute_tool = exec_fn,
                max_tool_iterations = 2,
            )
        )
        assert exec_fn.calls == [("python", {"code": "print(1)"})]
        assert any(event.get("type") == "tool_end" for event in events)

    def test_max_iterations_zero_executes_no_tools(self):
        loop, exec_fn = _make_loop(
            turns = [['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>']],
            exec_results = ["OK"],
            max_tool_iterations = 0,
        )
        events = _collect_events(loop)
        assert exec_fn.calls == []
        assert events and events[-1] == {"type": "status", "text": ""}

    def test_streaming_clips_before_tool_signal_no_leak(self):
        loop, exec_fn = _make_loop(
            turns = [
                [
                    "I will look this up. ",
                    "Some more prose that's long enough to leave the buffer. ",
                    '<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>',
                ],
                ["all done"],
            ],
            exec_results = ["weather: sunny"],
            max_tool_iterations = 2,
        )
        events = _collect_events(loop)
        assert exec_fn.calls == [("web_search", {"query": "x"})]
        for e in events:
            if e["type"] == "content":
                assert "<tool_call>" not in e["text"]
                assert "web_search" not in e["text"]

    def test_auto_heal_disabled_still_parses_valid_tool_call(self):
        loop, exec_fn = _make_loop(
            turns = [
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                ["done"],
            ],
            exec_results = ["OK"],
            auto_heal_tool_calls = False,
            max_tool_iterations = 2,
        )
        _collect_events(loop)
        assert exec_fn.calls == [("web_search", {"query": "x"})]

    def test_confirm_tool_calls_close_after_prompt_cleans_slot(self, monkeypatch):
        approval_id = "approval-close-sf"
        monkeypatch.setattr(safetensors_agentic, "new_approval_id", lambda: approval_id)

        loop, exec_fn = _make_loop(
            turns = [['<tool_call>{"name":"python","arguments":{"code":"print(1)"}}</tool_call>']],
            exec_results = ["OK"],
            confirm_tool_calls = True,
            session_id = "sess",
            max_tool_iterations = 1,
        )

        with tool_approvals._lock:
            tool_approvals._pending.clear()

        try:
            assert next(loop)["type"] == "status"
            start = next(loop)
            assert start["type"] == "tool_start"
            assert start["approval_id"] == approval_id
            with tool_approvals._lock:
                assert approval_id in tool_approvals._pending
        finally:
            loop.close()

        with tool_approvals._lock:
            assert approval_id not in tool_approvals._pending
        assert resolve_tool_decision(approval_id, "allow", session_id = "sess") is False
        assert exec_fn.calls == []

    def test_confirm_tool_calls_skips_rag_autoinject(self, monkeypatch):
        def fail_autoinject(*_args, **_kwargs):
            raise AssertionError("RAG autoinject must not run before approval")

        monkeypatch.setattr("core.inference.tools.build_rag_autoinject", fail_autoinject)
        loop, exec_fn = _make_loop(
            turns = [["plain answer"]],
            confirm_tool_calls = True,
            rag_scope = {"thread_id": "t1"},
        )
        events = _collect_events(loop)
        assert any(e.get("type") == "content" and e.get("text") == "plain answer" for e in events)
        assert exec_fn.calls == []

    def test_auto_heal_disabled_preserves_xml_on_final_no_tools_pass(self):
        turns = iter(
            [
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'],
                ['<tool_call>{"name":"web_search","arguments":{"query":"literal"}}</tool_call>'],
            ]
        )

        def fake_single_turn(_messages, active_tools = None):
            acc = ""
            for chunk in next(turns):
                acc += chunk
                yield acc

        exec_fn = FakeExecuteTool(["OK"])
        events = _collect_events(
            run_safetensors_tool_loop(
                single_turn = fake_single_turn,
                messages = [{"role": "user", "content": "show literal"}],
                tools = [{"type": "function", "function": {"name": "web_search"}}],
                execute_tool = exec_fn,
                max_tool_iterations = 1,
                auto_heal_tool_calls = False,
            )
        )
        assert exec_fn.calls == [("web_search", {"query": "x"})]
        assert any(
            event.get("type") == "content" and "<tool_call>" in event.get("text", "")
            for event in events
        )

    def test_auto_heal_disabled_does_not_repair_unclosed_tool_call(self):
        loop, exec_fn = _make_loop(
            turns = [
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}'],
            ],
            exec_results = ["OK"],
            auto_heal_tool_calls = False,
            max_tool_iterations = 1,
        )
        events = _collect_events(loop)
        assert exec_fn.calls == []
        assert any(
            event.get("type") == "content" and "<tool_call>" in event.get("text", "")
            for event in events
        )

    def test_auto_heal_enabled_strips_unparseable_xml_tool_call(self):
        loop, exec_fn = _make_loop(
            turns = [["<tool_call>{not valid json}</tool_call>"]],
            exec_results = ["OK"],
            auto_heal_tool_calls = True,
            max_tool_iterations = 1,
        )
        events = _collect_events(loop)
        assert exec_fn.calls == []
        assert not any(
            event.get("type") == "content" and "<tool_call>" in event.get("text", "")
            for event in events
        )

    def test_non_consecutive_duplicate_is_short_circuited(self):
        loop, exec_fn = _make_loop(
            turns = [
                ['<tool_call>{"name":"web_search","arguments":{"query":"A"}}</tool_call>'],
                ['<tool_call>{"name":"web_search","arguments":{"query":"B"}}</tool_call>'],
                ['<tool_call>{"name":"web_search","arguments":{"query":"A"}}</tool_call>'],
                ["final"],
            ],
            exec_results = ["res-A", "res-B"],
            max_tool_iterations = 4,
        )
        events = _collect_events(loop)
        assert exec_fn.calls == [("web_search", {"query": "A"}), ("web_search", {"query": "B"})]
        assert [
            event.get("tool_call_id") for event in events if event.get("type") == "tool_end"
        ] == ["call_0", "call_1"]
        assert not [
            event
            for event in events
            if event.get("tool_call_id") == "call_2"
            and event.get("type") in {"tool_start", "tool_end"}
        ]

    def test_same_turn_duplicate_is_short_circuited(self):
        loop, exec_fn = _make_loop(
            turns = [
                [
                    '<tool_call>{"name":"web_search","arguments":{"query":"A"}}</tool_call>'
                    '<tool_call>{"name":"web_search","arguments":{"query":"A"}}</tool_call>'
                ],
                ["final"],
            ],
            exec_results = ["res-A"],
            max_tool_iterations = 2,
        )
        events = _collect_events(loop)
        assert exec_fn.calls == [("web_search", {"query": "A"})]
        assert [
            event.get("tool_call_id") for event in events if event.get("type") == "tool_end"
        ] == ["call_0"]
        assert not [
            event
            for event in events
            if event.get("tool_call_id") == "call_1"
            and event.get("type") in {"tool_start", "tool_end"}
        ]

    def test_coerce_string_args_python_uses_code_key(self):
        assert _coerce_arguments("print(1)", heal = True, tool_name = "python") == {"code": "print(1)"}

    def test_coerce_string_args_terminal_uses_command_key(self):
        assert _coerce_arguments("ls -la", heal = True, tool_name = "terminal") == {"command": "ls -la"}

    def test_tool_call_ids_unique_across_loop_iterations(self):
        loop, _exec = _make_loop(
            turns = [
                ['<tool_call>{"name":"web_search","arguments":{"query":"A"}}</tool_call>'],
                ['<tool_call>{"name":"web_search","arguments":{"query":"B"}}</tool_call>'],
                ["done"],
            ],
            exec_results = ["A", "B"],
            max_tool_iterations = 3,
        )
        events = _collect_events(loop)
        ids = [e["tool_call_id"] for e in events if e["type"] == "tool_start"]
        assert len(ids) == 2 and ids[0] != ids[1]


# ────────────────────────────────────────────────────────────────────
# Shared gpt-oss name detector
# ────────────────────────────────────────────────────────────────────


class TestGptOssNameDetection:
    def test_substring_match(self):
        assert is_gpt_oss_model_name("unsloth/gpt-oss-20b") is True

    def test_negative_known_non_oss_model(self):
        assert is_gpt_oss_model_name("meta-llama/Llama-3.1-8B-Instruct") is False

    def test_empty_or_none_returns_false(self):
        assert is_gpt_oss_model_name("") is False
        assert is_gpt_oss_model_name(cast(str, None)) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
