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
from core.inference.safetensors_agentic import (
    _coerce_arguments,
    run_safetensors_tool_loop,
)
from core.inference.tool_call_parser import (
    has_tool_signal,
    parse_tool_calls_from_text,
    strip_tool_markup,
)
from utils.datasets import is_gpt_oss_model_name


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


class TestParserMultiFormat:
    """Parser coverage for Llama-3 / Mistral / Gemma 4 emission formats.

    Each model family upstream of GGUF emits a different tool-call
    shape. The shared parser must turn all of them into the same
    OpenAI ``{name, arguments}`` shape so the safetensors / MLX
    agentic loop is family-agnostic.
    """

    # ── Llama-3 ────────────────────────────────────────────────────

    def test_llama3_python_tag_dot_call(self):
        # Llama-3 built-in tools: <|python_tag|>NAME.call(k="v", ...).
        import json

        text = '<|python_tag|>brave_search.call(query="weather in Tokyo")'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "brave_search"
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"query": "weather in Tokyo"}

    def test_llama3_python_tag_dot_call_multi_arg(self):
        import json

        text = (
            "<|python_tag|>get_weather.call("
            'location="Tokyo", units="celsius", days=5)'
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"location": "Tokyo", "units": "celsius", "days": 5}

    def test_llama3_python_tag_json_form(self):
        import json

        text = (
            '<|python_tag|>{"name":"web_search",' '"parameters":{"query":"hi","n":5}}'
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"query": "hi", "n": 5}

    def test_llama3_python_tag_json_form_with_eom(self):
        # Llama-3 emits ``<|eom_id|>`` after the JSON; must not break parsing.
        import json

        text = (
            '<|python_tag|>{"name":"python",'
            '"parameters":{"code":"print(2+2)"}}<|eom_id|>'
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"code": "print(2+2)"}

    def test_llama3_strip_markup_final(self):
        text = '<|python_tag|>brave_search.call(query="x")'
        assert strip_tool_markup(text, final = True) == ""

    # ── Llama-3.2 bare JSON ``custom_tools`` ─────────────────────

    def test_llama3_2_bare_json_parameters(self):
        # Llama-3.2-Instruct emits bare JSON directly as content; no
        # <|python_tag|> prefix per its training template.
        import json

        text = '{"name":"web_search","parameters":{"query":"Tokyo weather"}}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"query": "Tokyo weather"}

    def test_llama3_2_bare_json_arguments_key(self):
        import json

        text = '{"name":"add","arguments":{"a":1,"b":2}}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"a": 1, "b": 2}

    def test_llama3_2_bare_json_multi_call(self):
        # Llama-3 may chain calls with ``; `` per training template.
        text = '{"name":"a","parameters":{}}; ' '{"name":"b","parameters":{}}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "a"
        assert result[1]["function"]["name"] == "b"

    def test_llama3_2_bare_json_with_eom_sentinel(self):
        text = '{"name":"x","parameters":{"y":1}}<|eom_id|>'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "x"

    def test_llama3_2_bare_json_leading_sentinel_skipped(self):
        # Sometimes prior <|eot_id|> leaks into the next turn.
        text = '<|eot_id|>{"name":"x","parameters":{}}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "x"

    def test_llama3_2_bare_json_plain_prose_does_not_fire(self):
        # Defensive: must NOT fire on plain assistant prose.
        text = "Hello world, how are you today?"
        assert parse_tool_calls_from_text(text) == []

    def test_llama3_2_bare_json_embedded_in_prose_does_not_fire(self):
        # Defensive: JSON embedded in prose must NOT fire (parser is
        # strict about content STARTING with `{`).
        text = 'The tool result was: {"name":"foo"}'
        assert parse_tool_calls_from_text(text) == []

    def test_llama3_2_bare_json_missing_name_does_not_fire(self):
        text = '{"result":"ok","data":[1,2,3]}'
        assert parse_tool_calls_from_text(text) == []

    def test_llama3_2_bare_json_missing_args_does_not_fire(self):
        text = '{"name":"x"}'
        assert parse_tool_calls_from_text(text) == []

    def test_llama3_2_bare_json_args_not_dict_does_not_fire(self):
        text = '{"name":"x","parameters":42}'
        assert parse_tool_calls_from_text(text) == []

    def test_llama3_2_bare_json_string_parameters_does_not_fire(self):
        # Llama-3 spec: parameters must be a dict. Prose like
        # ``{"name":"foo","parameters":"a sentence"}`` must NOT trigger.
        text = '{"name":"foo","parameters":"this is a sentence"}'
        assert parse_tool_calls_from_text(text) == []

    def test_llama3_2_bare_json_string_arguments_not_json_does_not_fire(self):
        # OpenAI ``arguments`` may be a JSON-string of a dict, but a
        # plain non-JSON string must not pass the guard.
        text = '{"name":"foo","arguments":"not json"}'
        assert parse_tool_calls_from_text(text) == []

    def test_llama3_2_bare_json_string_arguments_json_dict_fires(self):
        # OpenAI shape: arguments is a JSON-encoded string of a dict.
        text = '{"name":"foo","arguments":"{\\"q\\":\\"x\\"}"}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "foo"
        # arguments stays as the original JSON-string.
        assert result[0]["function"]["arguments"] == '{"q":"x"}'

    def test_llama3_2_bare_json_string_arguments_json_non_dict_does_not_fire(self):
        # JSON-string that parses to a list / scalar / null must NOT fire.
        for bad in (
            '{"name":"foo","arguments":"[1,2,3]"}',
            '{"name":"foo","arguments":"\\"plain\\""}',
            '{"name":"foo","arguments":"null"}',
            '{"name":"foo","arguments":"42"}',
        ):
            assert parse_tool_calls_from_text(bad) == [], bad

    # ── Mistral pre-v11 ───────────────────────────────────────────

    def test_mistral_pre_v11_array(self):
        import json

        text = (
            '[TOOL_CALLS] [{"name":"web_search",'
            '"arguments":{"query":"hello"},"id":"abc"}]'
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"
        # Mistral provides its own id; preserve it.
        assert result[0]["id"] == "abc"
        assert json.loads(result[0]["function"]["arguments"]) == {"query": "hello"}

    def test_mistral_pre_v11_array_multi(self):
        text = (
            '[TOOL_CALLS] [{"name":"a","arguments":{"x":1},"id":"id1"},'
            '{"name":"b","arguments":{"y":2},"id":"id2"}]'
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "a"
        assert result[1]["function"]["name"] == "b"

    def test_mistral_pre_v11_unclosed_array(self):
        # Closing ``]`` truncated -- parser must heal off individual objects.
        text = '[TOOL_CALLS] [{"name":"web_search","arguments":{"q":"x"},"id":"id"}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "web_search"

    # ── Mistral v11+ ───────────────────────────────────────────────

    def test_mistral_v11_single(self):
        # Magistral / Mistral Small 3.1: bare ``name{json}`` after trigger.
        import json

        text = '[TOOL_CALLS]add{"a":3.5,"b":4}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "add"
        assert json.loads(result[0]["function"]["arguments"]) == {"a": 3.5, "b": 4}

    def test_mistral_v11_parallel(self):
        # v11+ parallel: ``[TOOL_CALLS]a{...}[TOOL_CALLS]b{...}``.
        text = '[TOOL_CALLS]add{"a":1}[TOOL_CALLS]sub{"b":2}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "add"
        assert result[1]["function"]["name"] == "sub"

    def test_mistral_v11_with_args_marker(self):
        # Ministral / Mistral Large 3: ``[TOOL_CALLS]name[ARGS]{json}``.
        import json

        text = '[TOOL_CALLS]add[ARGS]{"a":1,"b":2}'
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "add"
        assert json.loads(result[0]["function"]["arguments"]) == {"a": 1, "b": 2}

    def test_mistral_strip_markup_v11(self):
        text = '[TOOL_CALLS]add{"a":1}'
        assert strip_tool_markup(text, final = True) == ""

    # ── Gemma 4 ───────────────────────────────────────────────────

    def test_gemma4_simple_call(self):
        import json

        text = (
            "<|tool_call>call:get_weather{"
            'location:<|"|>Tokyo<|"|>,units:<|"|>celsius<|"|>}<tool_call|>'
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_weather"
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {"location": "Tokyo", "units": "celsius"}

    def test_gemma4_with_primitives(self):
        import json

        text = (
            "<|tool_call>call:set_pref{"
            "enabled:true,attempts:5,threshold:1.5,nickname:null}<tool_call|>"
        )
        result = parse_tool_calls_from_text(text)
        args = json.loads(result[0]["function"]["arguments"])
        assert args == {
            "enabled": True,
            "attempts": 5,
            "threshold": 1.5,
            "nickname": None,
        }

    def test_gemma4_nested_args(self):
        # Gemma 4 nests dicts / lists with bare keys and ``<|"|>`` strings.
        import json

        text = (
            "<|tool_call>call:search{"
            'query:<|"|>foo<|"|>,filters:{site:<|"|>example.com<|"|>,recent:true},'
            'tags:[<|"|>a<|"|>,<|"|>b<|"|>]}<tool_call|>'
        )
        result = parse_tool_calls_from_text(text)
        args = json.loads(result[0]["function"]["arguments"])
        assert args["query"] == "foo"
        assert args["filters"] == {"site": "example.com", "recent": True}
        assert args["tags"] == ["a", "b"]

    def test_gemma4_multi_call(self):
        text = (
            "<|tool_call>call:a{x:1}<tool_call|>" "<|tool_call>call:b{y:2}<tool_call|>"
        )
        result = parse_tool_calls_from_text(text)
        assert len(result) == 2
        assert result[0]["function"]["name"] == "a"
        assert result[1]["function"]["name"] == "b"

    def test_gemma4_unclosed_does_not_raise(self):
        # Truncated mid-stream; must not raise.
        text = '<|tool_call>call:foo{x:<|"|>bar<|"|>'
        result = parse_tool_calls_from_text(text)
        assert isinstance(result, list)

    def test_gemma4_strip_markup_final(self):
        text = "<|tool_call>call:foo{x:1}<tool_call|>"
        assert strip_tool_markup(text, final = True) == ""

    # ── Cross-format sentinels ────────────────────────────────────

    def test_all_markers_in_tool_xml_signals(self):
        # Streaming buffer wakes up on every emission marker.
        from core.inference.tool_call_parser import TOOL_XML_SIGNALS

        for marker in (
            "<tool_call>",
            "<function=",
            "<|python_tag|>",
            "[TOOL_CALLS]",
            "<|tool_call>",
        ):
            assert (
                marker in TOOL_XML_SIGNALS
            ), f"streaming loop would not wake on {marker!r}"

    def test_has_tool_signal_for_all_formats(self):
        assert has_tool_signal('<|python_tag|>brave_search.call(q="x")')
        assert has_tool_signal('[TOOL_CALLS] [{"name":"x"}]')
        assert has_tool_signal('[TOOL_CALLS]add{"a":1}')
        assert has_tool_signal("<|tool_call>call:foo{}<tool_call|>")


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
        tools = [
            {"type": "function", "function": {"name": "web_search"}},
            {"type": "function", "function": {"name": "python"}},
            {"type": "function", "function": {"name": "terminal"}},
        ],
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
                # : tool call only.
                [
                    '<tool_call>{"name":"web_search",',
                    '"arguments":{"query":"weather"}}',
                    "</tool_call>",
                ],
                # : final answer.
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

    def test_llama3_python_tag_form(self):
        # The agentic loop must recognise Llama-3's <|python_tag|>
        # marker, drain the rest of the turn, and execute the call.
        loop, exec_fn = _make_loop(
            turns = [
                [
                    "<|python_tag|>web_search.call(",
                    'query="weather in Tokyo"',
                    ")",
                ],
                ["The weather is sunny."],
            ],
            exec_results = ["Sunny, 22C"],
        )
        events = _collect_events(loop)
        assert exec_fn.calls == [("web_search", {"query": "weather in Tokyo"})]
        contents = [e for e in events if e["type"] == "content"]
        assert "sunny" in contents[-1]["text"].lower()

    def test_mistral_pre_v11_form(self):
        # Pre-v11 Mistral emission: ``[TOOL_CALLS] [{...}]``.
        loop, exec_fn = _make_loop(
            turns = [
                [
                    '[TOOL_CALLS] [{"name":"web_search",',
                    '"arguments":{"query":"hi"},"id":"abc"}]',
                ],
                ["done"],
            ],
            exec_results = ["ok"],
        )
        events = _collect_events(loop)
        assert exec_fn.calls == [("web_search", {"query": "hi"})]
        # Mistral-provided ids must propagate to tool_start events.
        tool_start = next(e for e in events if e["type"] == "tool_start")
        assert tool_start["tool_call_id"] == "abc"

    def test_mistral_v11_form(self):
        # v11+ Mistral emission: bare ``name{json}`` after the trigger.
        loop, exec_fn = _make_loop(
            turns = [
                ['[TOOL_CALLS]web_search{"query":"hi"}'],
                ["done"],
            ],
            exec_results = ["ok"],
        )
        events = _collect_events(loop)
        assert exec_fn.calls == [("web_search", {"query": "hi"})]

    def test_gemma4_form(self):
        # Gemma 4 emission: ``<|tool_call>call:NAME{...}<tool_call|>``.
        loop, exec_fn = _make_loop(
            turns = [
                [
                    "<|tool_call>call:web_search{",
                    'query:<|"|>weather<|"|>',
                    "}<tool_call|>",
                ],
                ["sunny"],
            ],
            exec_results = ["Sunny, 22C"],
        )
        events = _collect_events(loop)
        assert exec_fn.calls == [("web_search", {"query": "weather"})]

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
        # Model's second turn must not see "__IMAGES__".
        assert len(captured) >= 2
        tool_msgs = [m for m in captured[1] if m.get("role") == "tool"]
        assert tool_msgs, "no tool message reached the model"
        for tm in tool_msgs:
            assert (
                "__IMAGES__" not in tm["content"]
            ), f"sentinel leaked to model: {tm['content']!r}"

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
            assert (
                "__IMAGES__" not in tm["content"]
            ), f"second sentinel leaked: {tm['content']!r}"
            assert (
                tm["content"] == "panel"
            ), f"expected payload-only 'panel', got {tm['content']!r}"

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
                # : tool call (executes once)
                [
                    '<tool_call>{"name":"web_search","arguments":{"query":"a"}}</tool_call>'
                ],
                # : model gives a final answer when nudged.
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


class TestProseMentioningToolCall:
    def test_assistant_prose_with_literal_tool_call_text_survives(self):
        # Regression: if the assistant text legitimately mentions
        # ``<tool_call>`` as a literal string and the parser finds no
        # actual call, the loop must surface the full content instead
        # of silently stripping everything past the literal marker.
        loop, exec_fn = _make_loop(
            turns = [
                # : a real tool call so the loop moves to
                # .
                [
                    '<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'
                ],
                # : prose that mentions the literal text.
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
        # Tool result text contains the literal ``<tool_call>`` string.
        # The loop must only parse the MODEL output, not the tool
        # result, so we should see exactly one call.
        loop, exec_fn = _make_loop(
            turns = [
                [
                    '<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'
                ],
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
            self, messages, *, tokenize = False, add_generation_prompt = True, **kw
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
        exec_fn = FakeExecuteTool([])
        loop = run_safetensors_tool_loop(
            single_turn = _fake_stream(
                [
                    '<tool_call>{"name":"terminal","arguments":{"command":"echo bypass"}}</tool_call>'
                ]
            ),
            messages = [{"role": "user", "content": "hi"}],
            tools = [{"type": "function", "function": {"name": "web_search"}}],
            execute_tool = exec_fn,
            max_tool_iterations = 2,
        )
        events = _collect_events(loop)
        assert exec_fn.calls == []
        tool_ends = [e for e in events if e["type"] == "tool_end"]
        assert tool_ends and "not enabled" in tool_ends[0]["result"].lower()

    def test_empty_tools_list_does_not_enforce_allowlist(self):
        exec_fn = FakeExecuteTool(["OK"])
        loop = run_safetensors_tool_loop(
            single_turn = _fake_stream(
                [
                    '<tool_call>{"name":"python","arguments":{"code":"print(1)"}}</tool_call>'
                ]
            ),
            messages = [{"role": "user", "content": "hi"}],
            tools = [],
            execute_tool = exec_fn,
            max_tool_iterations = 2,
        )
        _collect_events(loop)
        assert exec_fn.calls == [("python", {"code": "print(1)"})]

    def test_max_iterations_zero_executes_no_tools(self):
        loop, exec_fn = _make_loop(
            turns = [
                [
                    '<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'
                ]
            ],
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
                [
                    '<tool_call>{"name":"web_search","arguments":{"query":"x"}}</tool_call>'
                ],
                ["done"],
            ],
            exec_results = ["OK"],
            auto_heal_tool_calls = False,
            max_tool_iterations = 2,
        )
        _collect_events(loop)
        assert exec_fn.calls == [("web_search", {"query": "x"})]

    def test_non_consecutive_duplicate_is_short_circuited(self):
        loop, exec_fn = _make_loop(
            turns = [
                [
                    '<tool_call>{"name":"web_search","arguments":{"query":"A"}}</tool_call>'
                ],
                [
                    '<tool_call>{"name":"web_search","arguments":{"query":"B"}}</tool_call>'
                ],
                [
                    '<tool_call>{"name":"web_search","arguments":{"query":"A"}}</tool_call>'
                ],
                ["final"],
            ],
            exec_results = ["res-A", "res-B"],
            max_tool_iterations = 4,
        )
        events = _collect_events(loop)
        assert exec_fn.calls == [
            ("web_search", {"query": "A"}),
            ("web_search", {"query": "B"}),
        ]
        tool_ends = [e for e in events if e["type"] == "tool_end"]
        assert "already made this exact call" in tool_ends[-1]["result"]

    def test_coerce_string_args_python_uses_code_key(self):
        assert _coerce_arguments("print(1)", heal = True, tool_name = "python") == {
            "code": "print(1)"
        }

    def test_coerce_string_args_terminal_uses_command_key(self):
        assert _coerce_arguments("ls -la", heal = True, tool_name = "terminal") == {
            "command": "ls -la"
        }

    def test_tool_call_ids_unique_across_loop_iterations(self):
        loop, _exec = _make_loop(
            turns = [
                [
                    '<tool_call>{"name":"web_search","arguments":{"query":"A"}}</tool_call>'
                ],
                [
                    '<tool_call>{"name":"web_search","arguments":{"query":"B"}}</tool_call>'
                ],
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
        assert is_gpt_oss_model_name(None) is False


# ────────────────────────────────────────────────────────────────────
# Routes-level python_tag strip (multi-line; stop on next sentinel)
# ────────────────────────────────────────────────────────────────────


class TestRoutesPythonTagStrip:
    """Earlier revisions of ``_TOOL_XML_RE`` in
    ``studio.backend.routes.inference`` used either ``[^\\n<]*`` (5615 --
    leaked the tail of any tool call whose argument contained a literal
    ``<`` like ``code="if x < 10"``) or ``[^\\n]*`` (5620 round one --
    single-line only, so the second line of
    ``python.call(code="line1\\nline2")`` leaked). The current pattern
    ``(?:[^<]|<(?!\\|))*`` consumes any character that is not a Llama-3
    ``<|`` sentinel start, so multi-line code, embedded JSON, and bare
    ``<`` characters in code all stay inside the strip.

    The fully resolved strip is also exposed via
    ``strip_tool_markup(text, final=True)`` in the parser; the
    streaming path's routes-level strip is the regression-prone one
    because it runs on every cumulative emission while content is
    still arriving.
    """

    def _strip(self, text: str) -> str:
        # Import inside the test so a routes-module import error does
        # not blow up the entire test file at collection time.
        from routes.inference import _strip_tool_xml

        return _strip_tool_xml(text)

    def test_single_line_python_tag_stripped(self):
        # Floor: the original 5620 single-line behaviour still works.
        text = '<|python_tag|>brave_search.call(query="weather")'
        assert self._strip(text) == ""

    def test_python_tag_with_less_than_in_code(self):
        # 5615 regression: literal ``<`` inside code must NOT terminate
        # the strip early.
        text = '<|python_tag|>python.call(code="if x < 10: pass")'
        assert self._strip(text) == ""

    def test_python_tag_multiline_code_stripped(self):
        # 5620 round-1 regression: multi-line code's second line leaked.
        text = '<|python_tag|>python.call(code="line1\nline2\nline3")'
        assert self._strip(text) == ""

    def test_python_tag_multiline_with_less_than(self):
        # Combined: multi-line code AND literal ``<`` in code.
        text = (
            '<|python_tag|>python.call(code="for i in range(10):\n'
            "    if i < 5:\n"
            '        print(i)")'
        )
        assert self._strip(text) == ""

    def test_python_tag_stops_at_eom_sentinel(self):
        # Strip stops at the next Llama-3 ``<|`` sentinel so any
        # trailing assistant content survives.
        text = (
            '<|python_tag|>python.call(code="multi\nline")'
            "<|eom_id|>final answer text"
        )
        assert self._strip(text) == "<|eom_id|>final answer text"

    def test_python_tag_stops_at_eot_sentinel(self):
        text = '<|python_tag|>brave_search.call(query="x")' "<|eot_id|>after"
        assert self._strip(text) == "<|eot_id|>after"

    def test_python_tag_json_form_multiline_stripped(self):
        # The JSON form of python_tag with newlines inside string args.
        text = (
            '<|python_tag|>{"name":"python",'
            '"parameters":{"code":"a = 1\nb = 2\nprint(a+b)"}}'
        )
        assert self._strip(text) == ""

    def test_python_tag_with_eom_then_trailing_python_tag(self):
        # Two python_tag emissions back-to-back across a sentinel: both
        # should strip independently.
        text = (
            '<|python_tag|>brave_search.call(query="a")'
            "<|eom_id|>"
            '<|python_tag|>python.call(code="x=1")'
        )
        # ``<|eom_id|>`` between the two strips remains; both
        # python_tag blocks are fully consumed.
        assert self._strip(text) == "<|eom_id|>"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
