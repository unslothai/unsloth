# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Regression tests for PR #5624 (DeepSeek R1/V3.x, GLM 4.x, Kimi K2 tool
parsing). Each test pins a specific edge case surfaced during the
review:

* GLM string-vs-JSON-encoded value coercion (template emits strings
  raw and non-strings JSON-encoded; the parser must not coerce a
  bare string ``"42"`` into ``42``).
* GLM ``<arg_value>`` containing a literal ``<`` (e.g. ``if x < 10``).
* Kimi K2 dotted name ``functions.my.tool:0`` keeps its full name
  (``my.tool``) after stripping only the ``functions.`` prefix and
  ``:idx`` suffix, while the full id is preserved on the call.
* Kimi K2 bare-counter id (no ``functions.`` prefix, no ``:IDX``) is
  dropped rather than surfaced under a numeric name.
* DeepSeek V3.1 truncated mid-stream produces an empty result without
  raising.
* ``routes.inference._strip_tool_xml`` strips the DeepSeek envelope and
  the Kimi section markers added by this PR.
"""

import json

import pytest

from core.inference.tool_call_parser import (
    parse_tool_calls_from_text,
    strip_tool_markup,
)


# GLM string-vs-JSON-encoded value coercion (finding B in plan)


@pytest.mark.parametrize(
    "raw_val, expected_python",
    [
        # Bare numeric / bool / null shapes are still treated as JSON
        # literals (ambiguous with strings; the template doesn't tell us).
        ("42", 42),
        ("true", True),
        ("false", False),
        ("null", None),
        ("3.14", 3.14),
        ("-7", -7),
        ("1e3", 1000.0),
    ],
)
def test_glm_numeric_and_bool_literals_are_json_decoded(raw_val, expected_python):
    text = (
        "<tool_call>n\n"
        f"<arg_key>v</arg_key>\n"
        f"<arg_value>{raw_val}</arg_value>\n"
        "</tool_call>"
    )
    calls = parse_tool_calls_from_text(text)
    assert len(calls) == 1
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["v"] == expected_python


@pytest.mark.parametrize(
    "raw_val",
    [
        "hello world",  # plain prose
        "True",  # Python literal, NOT JSON -- no longer eaten by ast.literal_eval
        "None",  # Python literal, NOT JSON -- no longer eaten by ast.literal_eval
        "if x < 10: pass",  # code with literal < (well, < not in arg_value here)
        "{not valid json",  # looks like an object but is malformed -- must stay raw
        "[oops",  # looks like an array but is malformed
    ],
)
def test_glm_non_json_shapes_stay_raw(raw_val):
    text = (
        "<tool_call>n\n"
        f"<arg_key>v</arg_key>\n"
        f"<arg_value>{raw_val}</arg_value>\n"
        "</tool_call>"
    )
    calls = parse_tool_calls_from_text(text)
    assert len(calls) == 1
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["v"] == raw_val
    assert isinstance(args["v"], str)


def test_glm_json_object_arg_decoded():
    text = (
        "<tool_call>nest\n"
        "<arg_key>opts</arg_key>\n"
        '<arg_value>{"limit": 10}</arg_value>\n'
        "</tool_call>"
    )
    calls = parse_tool_calls_from_text(text)
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["opts"] == {"limit": 10}


def test_glm_json_array_arg_decoded():
    text = (
        "<tool_call>nest\n"
        "<arg_key>ids</arg_key>\n"
        "<arg_value>[1, 2, 3]</arg_value>\n"
        "</tool_call>"
    )
    calls = parse_tool_calls_from_text(text)
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["ids"] == [1, 2, 3]


def test_glm_arg_value_with_literal_less_than():
    text = (
        "<tool_call>run\n"
        "<arg_key>code</arg_key>\n"
        "<arg_value>if x < 10: pass</arg_value>\n"
        "</tool_call>"
    )
    calls = parse_tool_calls_from_text(text)
    assert len(calls) == 1
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["code"] == "if x < 10: pass"


# GLM 4.7 no-newline emission shape


def test_glm_4_7_no_newlines_between_name_and_arg_key():
    """GLM 4.7 strips the ``\\n`` after the name (``{{- ... -}}`` in the
    template) so ``<arg_key>`` follows directly. Parser must accept both."""
    text = (
        "<tool_call>get_weather"
        "<arg_key>city</arg_key><arg_value>London</arg_value>"
        "<arg_key>units</arg_key><arg_value>celsius</arg_value>"
        "</tool_call>"
    )
    calls = parse_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"city": "London", "units": "celsius"}


def test_glm_4_7_no_newlines_multi_call():
    """Back-to-back GLM 4.7 calls without intervening newlines."""
    text = (
        "<tool_call>a<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>"
        "<tool_call>b<arg_key>y</arg_key><arg_value>2</arg_value></tool_call>"
    )
    calls = parse_tool_calls_from_text(text)
    assert len(calls) == 2
    assert calls[0]["function"]["name"] == "a"
    assert calls[1]["function"]["name"] == "b"


def test_glm_4_7_does_not_break_qwen_path():
    """Qwen ``<tool_call>{json}`` still dispatches to Qwen; GLM's
    first-char ``[^\\n<{]`` excludes ``{``."""
    text = '<tool_call>{"name":"web_search","arguments":{"q":"x"}}</tool_call>'
    calls = parse_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "web_search"


# Kimi K2 dotted name + bare counter (finding C in plan)


def test_kimi_dotted_namespace_keeps_full_dotted_name():
    # A dotted Kimi id keeps its FULL name after stripping only the ``functions.``
    # prefix and ``:idx`` suffix -- matching current vLLM
    # (``tool_id.split(":")[0].removeprefix("functions.")``) and SGLang. Truncating to
    # the final dot-segment would corrupt dotted MCP tool names like ``mcp.server-list``.
    text = (
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.my.tool:0"
        "<|tool_call_argument_begin|>{}"
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )
    calls = parse_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "my.tool"
    assert calls[0]["id"] == "functions.my.tool:0"


def test_kimi_two_sections_in_one_stream_both_parse():
    """Outer loop walks every ``<|tool_calls_section_begin|>...end|>``
    so vLLM / SGLang parity holds even on multi-section streams."""
    text = (
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.a:0"
        '<|tool_call_argument_begin|>{"x":1}'
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
        " some prose between sections "
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.b:0"
        '<|tool_call_argument_begin|>{"y":2}'
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )
    calls = parse_tool_calls_from_text(text)
    assert len(calls) == 2
    assert calls[0]["function"]["name"] == "a"
    assert calls[1]["function"]["name"] == "b"
    assert calls[0]["id"] == "functions.a:0"
    assert calls[1]["id"] == "functions.b:0"


def test_kimi_bare_counter_id_is_dropped():
    """Bare-digit id (``3``) is dropped (matches vLLM); SGLang infers
    name from schema, which we don't have at parse time."""
    text = (
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>3"
        "<|tool_call_argument_begin|>{}"
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )
    calls = parse_tool_calls_from_text(text)
    assert calls == []


# DeepSeek truncated mid-stream


def test_deepseek_v3_1_huge_truncated_body_is_linear():
    """Adversarial input: DeepSeek envelope with no JSON brace and a
    50k-char body. A regex-based ``[^\\n<]+?`` name capture is O(N^2)
    here; the parser uses ``str.find`` on the sep marker so it stays
    linear. Budget 1s to flag any future regression."""
    import time as _time

    text = "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>fn<｜tool▁sep｜>" + "x" * 50_000
    start = _time.time()
    calls = parse_tool_calls_from_text(text)
    elapsed = _time.time() - start
    assert elapsed < 1.0, f"V3 path is non-linear: {elapsed:.2f}s"
    assert calls == []


def test_deepseek_r1_huge_fenceless_body_is_linear():
    """R1 detection used a greedy ``([^\\n]+)\\n```json`` regex that is O(N^2) on a
    fence-less body of repeated ``function<sep>`` tokens. The parser now scans with
    ``str.find``; budget 1s to flag any regression."""
    import time as _time

    text = "<｜tool▁calls▁begin｜>" + "function<｜tool▁sep｜>a" * 40_000
    start = _time.time()
    calls = parse_tool_calls_from_text(text)
    elapsed = _time.time() - start
    assert elapsed < 1.0, f"R1 path is non-linear: {elapsed:.2f}s"
    assert calls == []


def test_glm_unclosed_body_many_arg_keys_is_linear():
    """An unclosed GLM ``<tool_call>`` body runs to EOF; a lazy-group ``finditer``
    over many bare ``<arg_key>`` tokens was O(N^2). The parser now walks pairs with
    ``str.find``; budget 1s."""
    import time as _time

    text = "<tool_call>foo\n" + "<arg_key>k" * 40_000
    start = _time.time()
    parse_tool_calls_from_text(text)
    elapsed = _time.time() - start
    assert elapsed < 1.0, f"GLM path is non-linear: {elapsed:.2f}s"


def test_deepseek_r1_fenced_json_parses():
    """R1 wraps args in a ```json fence after ``function<sep>NAME``."""
    import json as _json

    text = (
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n"
        "```json\n"
        '{"city":"NYC","unit":"c"}\n'
        "```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    )
    calls = parse_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_weather"
    assert _json.loads(calls[0]["function"]["arguments"]) == {"city": "NYC", "unit": "c"}


def test_deepseek_v3_1_truncated_arguments_drops_call_without_crash():
    text = (
        "<｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>get_time"
        "<｜tool▁sep｜>"
        '{"city":"Tokyo"'  # no closing brace, no end markers
    )
    calls = parse_tool_calls_from_text(text)
    assert calls == []


def test_deepseek_v3_1_truncated_after_end_marker_still_yields_call():
    text = (
        "<｜tool▁calls▁begin｜>" "<｜tool▁call▁begin｜>get_time" "<｜tool▁sep｜>" '{"city":"Tokyo"}'
        # neither <｜tool▁call▁end｜> nor <｜tool▁calls▁end｜>
    )
    calls = parse_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_time"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Tokyo"}


# Routes-layer strip across the three new families


def test_routes_layer_strip_removes_deepseek_envelope():
    from routes.inference import _strip_tool_xml as _routes_strip

    text = (
        "before "
        "<｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>get_time"
        '<｜tool▁sep｜>{"city":"Tokyo"}'
        "<｜tool▁call▁end｜>"
        "<｜tool▁calls▁end｜>"
        " after"
    )
    stripped = _routes_strip(text)
    assert stripped == "before  after"


def test_routes_layer_strip_removes_kimi_section():
    from routes.inference import _strip_tool_xml as _routes_strip

    text = (
        "before "
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.web_search:0"
        '<|tool_call_argument_begin|>{"q":"x"}'
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
        " after"
    )
    stripped = _routes_strip(text)
    assert stripped == "before  after"


def test_routes_layer_strip_removes_glm_block():
    """``<tool_call>.*?</tool_call>`` covers GLM via the Qwen pattern."""
    from routes.inference import _strip_tool_xml as _routes_strip

    text = (
        "before "
        "<tool_call>web_search\n"
        "<arg_key>q</arg_key>\n<arg_value>x</arg_value>\n"
        "</tool_call>"
        " after"
    )
    stripped = _routes_strip(text)
    assert stripped == "before  after"


# strip_tool_markup (parser-level finalise path) over the new families


def test_strip_tool_markup_handles_deepseek_envelope():
    text = (
        "before "
        "<｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>get_time"
        '<｜tool▁sep｜>{"city":"Tokyo"}'
        "<｜tool▁call▁end｜>"
        "<｜tool▁calls▁end｜>"
        " after"
    )
    stripped = strip_tool_markup(text, final = True)
    assert "before" in stripped and "after" in stripped
    assert "｜tool▁" not in stripped
    assert "get_time" not in stripped and "Tokyo" not in stripped


def test_strip_tool_markup_handles_kimi_section():
    text = (
        "before "
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.web_search:0"
        '<|tool_call_argument_begin|>{"q":"x"}'
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
        " after"
    )
    stripped = strip_tool_markup(text, final = True)
    assert "before" in stripped and "after" in stripped
    assert "tool_calls_section_begin" not in stripped


# Round-2 review findings: GLM quoted-string / unclosed-arg, DeepSeek
# strict terminator, nested wrapper-less Gemma strip


def test_glm_quoted_string_arg_keeps_its_quotes():
    # A GLM string value emitted verbatim that itself begins with a quote (e.g. a
    # search query the user wants quoted) must NOT be JSON-decoded, which would
    # strip the meaningful quotes before the tool runs.
    text = (
        "<tool_call>web_search\n"
        "<arg_key>query</arg_key>\n"
        '<arg_value>"exact phrase"</arg_value>\n'
        "</tool_call>"
    )
    calls = parse_tool_calls_from_text(text)
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["query"] == '"exact phrase"'


def test_glm_unclosed_arg_value_is_rejected_in_strict_mode():
    # Closing </tool_call> present but a value never closes: strict mode must reject
    # the whole call rather than execute it with the argument silently dropped.
    text = (
        "<tool_call>web_search\n"
        "<arg_key>query</arg_key>\n"
        "<arg_value>Tokyo weather"  # no </arg_value>
        "</tool_call>"
    )
    assert parse_tool_calls_from_text(text, allow_incomplete = False) == []
    # With Auto-Heal the partial value is kept, not dropped to a no-arg call.
    healed = parse_tool_calls_from_text(text, allow_incomplete = True)
    assert len(healed) == 1
    args = json.loads(healed[0]["function"]["arguments"])
    assert "Tokyo weather" in args.get("query", "")


def test_deepseek_v3_missing_call_terminator_rejected_in_strict_mode():
    # Envelope closes but the per-call <｜tool▁call▁end｜> is absent. Strict mode
    # must reject (it is truncated/merged); Auto-Heal still parses it.
    text = (
        "<｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>get_time"
        '<｜tool▁sep｜>{"city":"Tokyo"}'
        "<｜tool▁calls▁end｜>"  # envelope end only, no per-call end
    )
    assert parse_tool_calls_from_text(text, allow_incomplete = False) == []
    healed = parse_tool_calls_from_text(text, allow_incomplete = True)
    assert len(healed) == 1
    assert healed[0]["function"]["name"] == "get_time"


def test_deepseek_v3_with_call_terminator_parses_in_strict_mode():
    # Sanity: a well-formed V3 call (with the per-call end marker) still parses
    # under strict mode after the terminator check.
    text = (
        "<｜tool▁calls▁begin｜>"
        "<｜tool▁call▁begin｜>get_time"
        '<｜tool▁sep｜>{"city":"Tokyo"}'
        "<｜tool▁call▁end｜>"
        "<｜tool▁calls▁end｜>"
    )
    calls = parse_tool_calls_from_text(text, allow_incomplete = False)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "get_time"


def test_strip_tool_markup_removes_nested_wrapperless_gemma_call():
    # Wrapper-less Gemma call with a NESTED object arg: the [^{}]* cleanup regex
    # cannot match it, so the balanced helper must remove the whole call instead of
    # leaving a trailing ``}`` visible after execution.
    text = "answer: call:f{loc:{city:NYC},n:3} done"
    stripped = strip_tool_markup(text, final = True)
    assert "call:f" not in stripped
    assert "}" not in stripped
    assert "answer:" in stripped and "done" in stripped


# Pass-3 review findings: bare-Kimi streaming (non-final) strip symmetry
# and the wrapper-less Gemma route-display strip


def test_strip_tool_markup_non_final_removes_bare_kimi_call():
    # The parser accepts a bare ``<|tool_call_begin|>...<|tool_call_end|>`` with no
    # section wrapper, so the CLOSED (streaming, final=False) strip must remove it
    # too -- otherwise the call's markup leaks into the live stream mid-generation
    # while only the finalise pass cleaned it. The surrounding prose must survive.
    text = (
        "before "
        "<|tool_call_begin|>functions.web_search:0"
        '<|tool_call_argument_begin|>{"q":"x"}'
        "<|tool_call_end|>"
        " after"
    )
    stripped = strip_tool_markup(text, final = False)
    assert "tool_call_begin" not in stripped
    assert "tool_call_end" not in stripped
    assert "before" in stripped and "after" in stripped


def test_routes_layer_strip_removes_wrapperless_gemma_call():
    # Gemma 4 (skip_special_tokens) emits a wrapper-less ``call:NAME{..}`` with no
    # XML markers. _strip_tool_xml now delegates to _strip_gemma_wrapperless_calls,
    # so the route display strip removes it instead of leaking ``call:web_search``.
    from routes.inference import _strip_tool_xml as _routes_strip

    text = 'before call:web_search{query:"weather in Sydney"} after'
    stripped = _routes_strip(text)
    assert "call:web_search" not in stripped
    assert "before" in stripped and "after" in stripped


def test_deepseek_envelope_end_inside_arg_string_is_not_a_truncation():
    # A DeepSeek V3.1 call whose argument string legitimately contains the literal
    # envelope-end token must not be dropped: a raw find on the end marker cut the
    # body before the balanced JSON closed, so the whole valid call vanished.
    content = (
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>web_search<｜tool▁sep｜>"
        '{"query":"what does <｜tool▁calls▁end｜> mean"}'
        "<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    )
    calls = parse_tool_calls_from_text(content)
    assert len(calls) == 1, calls
    assert calls[0]["function"]["name"] == "web_search"
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "query": "what does <｜tool▁calls▁end｜> mean"
    }


def test_glm_value_containing_literal_arg_value_close_is_preserved():
    # A GLM string argument may legitimately contain </arg_value> (e.g. code that
    # prints the tag). The first-match close truncated the value and executed the
    # tool with corrupted args; the real close is the one whose next token is the
    # next <arg_key> / </tool_call>, so the whole value must survive.
    content = (
        "<tool_call>run<arg_key>code</arg_key>"
        '<arg_value>print("</arg_value>")</arg_value></tool_call>'
    )
    calls = parse_tool_calls_from_text(content)
    assert len(calls) == 1, calls
    assert json.loads(calls[0]["function"]["arguments"]) == {"code": 'print("</arg_value>")'}


def test_attribute_form_function_with_embedded_marker_runs_outer_call():
    # <function name="outer"> is a supported envelope; a DeepSeek/Kimi marker inside
    # one of its parameter values is data, not a second call. The marker guard must
    # cover the attribute form (not only <function=NAME>), else the pre-pass hijacks
    # the embedded marker and runs the wrong tool.
    content = (
        '<function name="respond"><parameter name="answer">'
        "The Kimi format is <|tool_call_begin|>functions.delete_all:0"
        "<|tool_call_argument_begin|>{}<|tool_call_end|>"
        "</parameter></function>"
    )
    calls = parse_tool_calls_from_text(content)
    assert [c["function"]["name"] for c in calls] == ["respond"], calls


def test_wrapperless_gemma_call_gated_by_enabled_tools():
    # Once skip_special_tokens removes the <|tool_call> wrapper, call:NAME{...} is
    # indistinguishable from prose documenting the Gemma syntax. With a restricted
    # tool set, a disabled/example name must not be stolen as a call (which strips
    # the real answer and forces a disabled no-op) -- same gate as Llama bare JSON.
    prose = "Here is an example of the syntax: call:foo{x:1}. That shows how tools work."
    assert parse_tool_calls_from_text(prose, enabled_tool_names = {"web_search"}) == []
    # The display strip is gated the same way, so the example survives in the answer.
    assert "call:foo{x:1}" in strip_tool_markup(
        prose, final = True, enabled_tool_names = {"web_search"}
    )
    # An enabled name is still a real call (parsed, and stripped from display).
    real = "Answer. call:web_search{query:hi}"
    calls = parse_tool_calls_from_text(real, enabled_tool_names = {"web_search"})
    assert [c["function"]["name"] for c in calls] == ["web_search"], calls
    assert "call:web_search" not in strip_tool_markup(
        real, final = True, enabled_tool_names = {"web_search"}
    )


def test_kimi_section_end_inside_arg_string_is_not_a_truncation():
    # In a Kimi section with multiple calls, a LATER call whose argument string contains
    # the literal <|tool_calls_section_end|> token must not truncate the section: a raw
    # find cut the body there and, since an earlier call already parsed, the bare-call
    # fallback was skipped and the later valid call was dropped. Mirror of the DeepSeek
    # envelope-end fix.
    content = (
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.search:0<|tool_call_argument_begin|>"
        '{"q":"cats"}<|tool_call_end|>'
        "<|tool_call_begin|>functions.explain:1<|tool_call_argument_begin|>"
        '{"text":"the token <|tool_calls_section_end|> means end"}<|tool_call_end|>'
        "<|tool_calls_section_end|>"
    )
    calls = parse_tool_calls_from_text(content)
    assert [c["function"]["name"] for c in calls] == ["search", "explain"], calls
    assert json.loads(calls[1]["function"]["arguments"]) == {
        "text": "the token <|tool_calls_section_end|> means end"
    }


def test_closed_envelope_example_before_deepseek_block_runs_real_call():
    # A CLOSED <tool_call>/<function> example in prose that ends BEFORE a genuine
    # DeepSeek/Kimi block must not suppress the marker pre-pass: the real call has
    # to be parsed, not the phantom tool named inside the prose example.
    deepseek = (
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>search_web\n"
        "```json\n"
        '{"query":"weather in Paris"}\n'
        "```"
        "<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    )
    prose = (
        'A Qwen call looks like <tool_call>{"name":"example_tool","arguments":{}}</tool_call>.\n'
    )
    calls = parse_tool_calls_from_text(prose + deepseek)
    assert [c["function"]["name"] for c in calls] == ["search_web"], calls
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": "weather in Paris"}

    kimi = (
        "<|tool_calls_section_begin|><|tool_call_begin|>functions.lookup:0"
        '<|tool_call_argument_begin|>{"id":7}<|tool_call_end|><|tool_calls_section_end|>'
    )
    calls_k = parse_tool_calls_from_text("Example: <function=demo>{}</function> and now:\n" + kimi)
    assert [c["function"]["name"] for c in calls_k] == ["lookup"], calls_k


def test_marker_inside_closed_outer_envelope_still_runs_outer_call():
    # The guard must still fire when the marker genuinely sits INSIDE a closed outer
    # <function>/<tool_call> envelope's arguments (a user asking about the syntax):
    # the OUTER call is the real one, not the embedded marker.
    outer = (
        "<function=lookup><parameter=q>what does <｜tool▁calls▁begin｜> mean</parameter></function>"
    )
    calls = parse_tool_calls_from_text(outer)
    # The outer envelope is the real call; the embedded DeepSeek marker must not
    # hijack the parse into a spurious tool.
    assert [c["function"]["name"] for c in calls] == ["lookup"], calls
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "q": "what does <｜tool▁calls▁begin｜> mean"
    }
