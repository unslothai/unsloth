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
* Kimi K2 dotted name ``functions.my.tool:0`` resolves to the last
  segment (``tool``) while the full id is preserved on the call.
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


# ────────────────────────────────────────────────────────────────────
# GLM string-vs-JSON-encoded value coercion (finding B in plan)
# ────────────────────────────────────────────────────────────────────


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


# ────────────────────────────────────────────────────────────────────
# GLM 4.7 no-newline emission shape
# ────────────────────────────────────────────────────────────────────


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


# ────────────────────────────────────────────────────────────────────
# Kimi K2 dotted name + bare counter (finding C in plan)
# ────────────────────────────────────────────────────────────────────


def test_kimi_dotted_namespace_resolves_to_last_segment():
    text = (
        "<|tool_calls_section_begin|>"
        "<|tool_call_begin|>functions.my.tool:0"
        "<|tool_call_argument_begin|>{}"
        "<|tool_call_end|>"
        "<|tool_calls_section_end|>"
    )
    calls = parse_tool_calls_from_text(text)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "tool"
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


# ────────────────────────────────────────────────────────────────────
# DeepSeek truncated mid-stream
# ────────────────────────────────────────────────────────────────────


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


# ────────────────────────────────────────────────────────────────────
# Routes-layer strip across the three new families
# ────────────────────────────────────────────────────────────────────


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


# ────────────────────────────────────────────────────────────────────
# strip_tool_markup (parser-level finalise path) over the new families
# ────────────────────────────────────────────────────────────────────


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
    assert "tool" not in strip_tool_markup(
        text, final = True
    ).lower() or "after" in strip_tool_markup(text, final = True)
    stripped = strip_tool_markup(text, final = True)
    assert "before" in stripped and "after" in stripped
    assert "｜tool▁" not in stripped


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
