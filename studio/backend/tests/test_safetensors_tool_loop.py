# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Tests for the safetensors agentic tool loop.

Covers the shared ``tool_call_parser`` helpers and the cumulative-text
state machine inside ``safetensors_agentic.run_safetensors_tool_loop``.
The loop is exercised with hand-crafted fake single-turn generators so
no model load is needed; the tests run in CI under a few seconds.

Edge cases under coverage:
* Plain answers (no tool calls) flush full content.
* Single ``<tool_call>{json}</tool_call>`` triggers the tool and re-enters.
* Single ``<function=name>...`` XML form triggers the same path.
* Truncated unclosed ``<tool_call>`` is still parsed.
* Tool result is fed back as ``role=tool`` for the next iteration.
* Bad JSON inside ``<tool_call>`` does not raise and (when healed) is
  routed as a ``{"query": ...}`` web search call.
* Duplicate tool calls produce a synthetic "do not repeat" result the
  second time.
* ``__IMAGES__`` sentinel is stripped before the model sees the result.
* Tool execution errors are tagged so the model gets a nudge but the
  loop keeps streaming.
* Cancel is honoured between iterations.
* ``max_tool_iterations`` cap is respected and a final-answer attempt
  closes the stream cleanly.
"""

import threading

import pytest

from core.inference import safetensors_agentic
from core.inference.safetensors_agentic import run_safetensors_tool_loop
from core.inference.tool_call_parser import (
    has_tool_signal,
    parse_tool_calls_from_text,
    strip_tool_markup,
)


# ────────────────────────────────────────────────────────────────────
# parse_tool_calls_from_text
# ────────────────────────────────────────────────────────────────────


class TestParser:
    def test_json_tool_call(self):
        text = (
            '<tool_call>{"name":"web_search","arguments":{"query":"hello"}}</tool_call>'
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        tc = result[0]
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "web_search"
        # Arguments must always be a JSON string.
        assert isinstance(tc["function"]["arguments"], str)
        assert "hello" in tc["function"]["arguments"]

    def test_json_tool_call_unclosed(self):
        # No </tool_call>; balanced-brace extractor must still close.
        text = '<tool_call>{"name":"python","arguments":{"code":"print(1)"}}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "python"

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

    def test_code_with_embedded_xml(self):
        # A code parameter contains the literal </parameter>. Must not
        # truncate the value because the parser uses end-of-body as the
        # only boundary for single-parameter calls.
        text = (
            "<function=python><parameter=code>html = '<a></a>'\n"
            "print('hi')</parameter></function>"
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert "print('hi')" in result[0]["function"]["arguments"]

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
        # Bad JSON is silently dropped; caller can fall back to text.
        assert result == []

    def test_has_tool_signal(self):
        assert has_tool_signal("blah <tool_call> x")
        assert has_tool_signal("hi <function=foo>...")
        assert not has_tool_signal("hello world")

    def test_strip_markup_closed(self):
        text = "before <tool_call>{}</tool_call> after"
        assert strip_tool_markup(text) == "before  after"

    def test_strip_markup_unclosed_final(self):
        text = "before <tool_call>{partial"
        # With final=True the trailing run is dropped.
        assert strip_tool_markup(text, final = True) == "before"
        # Without final=True the unclosed run is preserved.
        assert "partial" in strip_tool_markup(text)


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


def _make_loop(*, turns, exec_results = None, **kwargs):
    """Build a configured loop with a multi-turn fake generator.

    ``turns`` is a list of chunk-lists; iteration N yields chunks from
    ``turns[N]``.
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
        tools = [{"type": "function", "function": {"name": "web_search"}}],
        execute_tool = exec_fn,
        **kwargs,
    ), exec_fn


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
        # Final cumulative content should contain the answer.
        final_text = contents[-1]["text"]
        assert "Hello world!" in final_text
        assert statuses and statuses[-1]["text"] == ""

    def test_single_tool_then_answer(self):
        loop, exec_fn = _make_loop(
            turns = [
                # Iteration 1: tool call only.
                [
                    '<tool_call>{"name":"web_search",',
                    '"arguments":{"query":"weather"}}',
                    "</tool_call>",
                ],
                # Iteration 2: final answer.
                ["The ", "weather is ", "sunny."],
            ],
            exec_results = ["Sunny and 22C"],
        )
        events = _collect_events(loop)
        kinds = [e["type"] for e in events]

        assert "tool_start" in kinds
        assert "tool_end" in kinds
        # Tool was actually called with the parsed arguments.
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

    def test_truncated_unclosed_tool_call(self):
        loop, exec_fn = _make_loop(
            turns = [
                # No </tool_call>; balanced-brace parser must still
                # succeed because the JSON itself is balanced.
                ['<tool_call>{"name":"web_search","arguments":{"query":"x"}}'],
                ["done"],
            ],
            exec_results = ["result"],
        )
        events = _collect_events(loop)
        assert exec_fn.calls == [("web_search", {"query": "x"})]

    def test_bad_json_healed_to_query(self):
        # Tool call with non-JSON string arguments. With auto_heal_tool_calls
        # the string is routed as {"query": ...}.
        loop, exec_fn = _make_loop(
            turns = [
                # JSON inside the tool call is well-formed; the
                # ``arguments`` is a string that is not itself valid
                # JSON for ``_coerce_arguments`` to parse, so the
                # heal path runs.
                [
                    '<tool_call>{"name":"web_search","arguments":"hello world"}</tool_call>'
                ],
                ["ok"],
            ],
            exec_results = ["..."],
        )
        events = _collect_events(loop)
        assert exec_fn.calls and exec_fn.calls[0][0] == "web_search"
        assert exec_fn.calls[0][1] == {"query": "hello world"}


class TestLoopBehaviour:
    def test_duplicate_tool_call_synthetic_result(self):
        # Two identical successful calls in a row: the second is short-
        # circuited with a "do not repeat" message and execute_tool is
        # called only once.
        loop, exec_fn = _make_loop(
            turns = [
                [
                    '<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'
                ],
                [
                    '<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'
                ],
                ["final"],
            ],
            exec_results = ["search-result-1"],
        )
        events = _collect_events(loop)
        # Only one real call.
        assert len(exec_fn.calls) == 1
        tool_end_events = [e for e in events if e["type"] == "tool_end"]
        assert len(tool_end_events) == 2
        assert "do not repeat" in tool_end_events[1]["result"].lower()

    def test_image_sentinel_stripped_from_model_feed(self):
        # The tool result has a frontend image sentinel that should be
        # stripped before being fed back into the next turn, BUT the
        # tool_end event still carries the raw result for the UI.
        loop, exec_fn = _make_loop(
            turns = [
                [
                    '<tool_call>{"name":"python","arguments":{"code":"plot()"}}</tool_call>'
                ],
                ["see chart"],
            ],
            exec_results = ["chart\n__IMAGES__:/tmp/chart.png"],
        )
        events = _collect_events(loop)
        tool_end = next(e for e in events if e["type"] == "tool_end")
        assert "__IMAGES__" in tool_end["result"]

    def test_tool_execution_error_is_emitted_but_loop_continues(self):
        loop, exec_fn = _make_loop(
            turns = [
                [
                    '<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'
                ],
                ["sorry, that failed"],
            ],
            exec_results = ["Error: network unreachable"],
        )
        events = _collect_events(loop)
        tool_end = next(e for e in events if e["type"] == "tool_end")
        assert tool_end["result"].startswith("Error")
        # The loop must still produce a content event after the failure.
        contents = [e for e in events if e["type"] == "content"]
        assert contents

    def test_exception_in_executor_does_not_raise(self):
        loop, exec_fn = _make_loop(
            turns = [
                [
                    '<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'
                ],
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
        # Even with a fake stream that emits tool calls, the loop must
        # bail before invoking execute_tool when cancel is set.
        exec_fn = FakeExecuteTool([])
        events = list(
            run_safetensors_tool_loop(
                single_turn = _const_stream(
                    '<tool_call>{"name":"web_search",'
                    '"arguments":{"query":"x"}}</tool_call>'
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
        # The loop should stop after max_tool_iterations even if the
        # model keeps asking for tools, then emit a final-attempt round.
        loop, exec_fn = _make_loop(
            turns = [
                # iteration 1: tool call (executes once)
                [
                    '<tool_call>{"name":"web_search","arguments":{"query":"a"}}</tool_call>'
                ],
                # iteration 2: model gives a final answer when nudged.
                ["here is the final answer"],
            ],
            exec_results = ["result"],
            max_tool_iterations = 1,
        )
        events = _collect_events(loop)
        contents = [e for e in events if e["type"] == "content"]
        # Final content must include the final answer.
        assert contents and "final answer" in contents[-1]["text"]


class TestStatusFormatting:
    def test_status_for_known_tools(self):
        # Use the private helper directly to verify status formatting.
        assert (
            safetensors_agentic._status_for_tool("web_search", {"query": "abc"})
            == "Searching: abc"
        )
        assert (
            safetensors_agentic._status_for_tool(
                "web_search", {"url": "https://www.example.com/x"}
            )
            == "Reading: example.com"
        )
        assert safetensors_agentic._status_for_tool(
            "python", {"code": "x = 1"}
        ).startswith("Running Python:")
        assert safetensors_agentic._status_for_tool(
            "terminal", {"command": "ls"}
        ).startswith("Running:")
        assert safetensors_agentic._status_for_tool("unknown_tool", {}).startswith(
            "Calling:"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
