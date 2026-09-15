# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Focused coverage for the native IFM tool-call protocol.

These tests exercise the shared parser, the cumulative-text safetensors loop, and the
template-rendering boundary used when a tool result starts a second generation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.inference.chat_template_helpers import (
    _repair_ifm_tool_history,
    apply_chat_template_for_generation,
)
from core.inference.safetensors_agentic import run_safetensors_tool_loop
from core.inference.tool_call_parser import (
    StreamingMarkupStripper,
    TOOL_XML_SIGNALS,
    has_tool_signal,
    parse_tool_calls_from_text,
    strip_tool_markup,
)
from core.inference.tool_loop_controller import ToolLoopController


IFM_XML = (
    "<ifm|tool_calls>\n"
    "<ifm|tool_call>python\n"
    "<ifm|arg_key>code</ifm|arg_key>\n"
    "<ifm|arg_value>print(1234567 * 891011)</ifm|arg_value>\n"
    "</ifm|tool_call>\n"
    "</ifm|tool_calls>"
)


def _function(call):
    return call["function"]["name"], json.loads(call["function"]["arguments"])


def _tool(name):
    return {"type": "function", "function": {"name": name}}


def _turn_generator(turns, captured = None):
    turn_iter = iter(turns)

    def generate(messages, **_kwargs):
        if captured is not None:
            captured.append(messages)
        chunks = next(turn_iter)
        accumulated = ""
        for chunk in chunks:
            accumulated += chunk
            yield accumulated

    return generate


class _Executor:
    def __init__(self, results = None):
        self.calls = []
        self._results = iter(results or ())

    def __call__(self, name, arguments, **_kwargs):
        self.calls.append((name, arguments))
        return next(self._results, f"result:{name}")


def _run(
    turns,
    *,
    tools = None,
    captured = None,
    results = None,
    **kwargs,
):
    executor = _Executor(results)
    events = list(
        run_safetensors_tool_loop(
            single_turn = _turn_generator(turns, captured),
            messages = [{"role": "user", "content": "please use the tool"}],
            tools = tools if tools is not None else [_tool("python"), _tool("web_search")],
            execute_tool = executor,
            nudge_tool_calls = False,
            permission_mode = "off",
            max_tool_iterations = 2,
            **kwargs,
        )
    )
    return events, executor


def test_ifm_xml_parses_to_openai_shape_and_controller_arguments():
    calls = parse_tool_calls_from_text(IFM_XML, allow_incomplete = False)

    assert len(calls) == 1
    assert calls[0]["id"] == "call_0"
    assert calls[0]["type"] == "function"
    assert _function(calls[0]) == ("python", {"code": "print(1234567 * 891011)"})

    controller = ToolLoopController(tools = [_tool("python")])
    decision = controller.prepare_call(calls[0])
    assert decision.should_execute
    assert decision.arguments == {"code": "print(1234567 * 891011)"}


def test_ifm_xml_multiple_arguments_preserve_plain_text_values():
    text = (
        "<ifm|tool_calls><ifm|tool_call>python"
        "<ifm|arg_key>multiline</ifm|arg_key>"
        "<ifm|arg_value>line one\nline two</ifm|arg_value>"
        "<ifm|arg_key>numeric_text</ifm|arg_key>"
        "<ifm|arg_value>007</ifm|arg_value>"
        "<ifm|arg_key>number</ifm|arg_key>"
        "<ifm|arg_value>42</ifm|arg_value>"
        "<ifm|arg_key>boolean</ifm|arg_key>"
        "<ifm|arg_value>true</ifm|arg_value>"
        "<ifm|arg_key>null_value</ifm|arg_key>"
        "<ifm|arg_value>null</ifm|arg_value>"
        "<ifm|arg_key>array</ifm|arg_key>"
        '<ifm|arg_value>[1, "two"]</ifm|arg_value>'
        "<ifm|arg_key>object</ifm|arg_key>"
        '<ifm|arg_value>{"nested": {"ok": true}}</ifm|arg_value>'
        "</ifm|tool_call></ifm|tool_calls>"
    )

    calls = parse_tool_calls_from_text(text, allow_incomplete = False)

    assert len(calls) == 1
    assert _function(calls[0]) == (
        "python",
        {
            "multiline": "line one\nline two",
            "numeric_text": "007",
            "number": "42",
            "boolean": "true",
            "null_value": "null",
            "array": '[1, "two"]',
            "object": '{"nested": {"ok": true}}',
        },
    )


def test_ifm_typed_xml_decodes_declared_argument_types():
    text = (
        "<ifm|tool_calls><ifm|tool_call>python"
        "<ifm|arg_key>count</ifm|arg_key><ifm|arg_type>integer</ifm|arg_type>"
        "<ifm|arg_value>7</ifm|arg_value>"
        "<ifm|arg_key>enabled</ifm|arg_key><ifm|arg_type>boolean</ifm|arg_type>"
        "<ifm|arg_value>true</ifm|arg_value>"
        "<ifm|arg_key>nothing</ifm|arg_key><ifm|arg_type>null</ifm|arg_type>"
        "<ifm|arg_value>null</ifm|arg_value>"
        "<ifm|arg_key>items</ifm|arg_key><ifm|arg_type>array</ifm|arg_type>"
        "<ifm|arg_value>[1, 2]</ifm|arg_value>"
        "<ifm|arg_key>config</ifm|arg_key><ifm|arg_type>object</ifm|arg_type>"
        '<ifm|arg_value>{"mode": "safe"}</ifm|arg_value>'
        "<ifm|arg_key>literal</ifm|arg_key><ifm|arg_type>string</ifm|arg_type>"
        "<ifm|arg_value>007</ifm|arg_value>"
        "<ifm|arg_key>score</ifm|arg_key><ifm|arg_type>number</ifm|arg_type>"
        "<ifm|arg_value>3.5</ifm|arg_value>"
        "</ifm|tool_call></ifm|tool_calls>"
    )

    calls = parse_tool_calls_from_text(text, allow_incomplete = False)

    assert len(calls) == 1
    assert _function(calls[0]) == (
        "python",
        {
            "count": 7,
            "enabled": True,
            "nothing": None,
            "items": [1, 2],
            "config": {"mode": "safe"},
            "literal": "007",
            "score": 3.5,
        },
    )


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_ifm_json_non_finite_values_are_rejected_without_execution(literal):
    text = (
        "<ifm|tool_calls><ifm|tool_call>"
        '{"name":"python","arguments":{"value":' + literal + "}}</ifm|tool_call></ifm|tool_calls>"
    )

    assert parse_tool_calls_from_text(text, allow_incomplete = False) == []
    events, executor = _run([[text]], tools = [_tool("python")])

    assert executor.calls == []
    assert not any(event["type"] in {"tool_start", "tool_end"} for event in events)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_ifm_typed_xml_non_finite_structured_values_are_rejected_without_execution(literal):
    text = (
        "<ifm|tool_calls><ifm|tool_call>python"
        "<ifm|arg_key>config</ifm|arg_key><ifm|arg_type>object</ifm|arg_type>"
        '<ifm|arg_value>{"value":' + literal + "}</ifm|arg_value></ifm|tool_call></ifm|tool_calls>"
    )

    assert parse_tool_calls_from_text(text, allow_incomplete = False) == []
    events, executor = _run([[text]], tools = [_tool("python")])

    assert executor.calls == []
    assert not any(event["type"] in {"tool_start", "tool_end"} for event in events)


def test_ifm_json_and_multiple_calls_preserve_order_and_ids():
    text = (
        "<ifm|tool_calls>"
        '<ifm|tool_call>{"name":"python","arguments":{"code":"print(1)"}}</ifm|tool_call>'
        '<ifm|tool_call>{"name":"web_search","arguments":{"query":"IFM"}}</ifm|tool_call>'
        "</ifm|tool_calls>"
    )

    calls = parse_tool_calls_from_text(text, id_offset = 4, allow_incomplete = False)

    assert [_function(call) for call in calls] == [
        ("python", {"code": "print(1)"}),
        ("web_search", {"query": "IFM"}),
    ]
    assert [call["id"] for call in calls] == ["call_4", "call_5"]


def test_ifm_json_body_preserves_valid_json_types():
    text = (
        "<ifm|tool_calls>"
        '<ifm|tool_call>{"name":"python","arguments":'
        '{"integer":1,"float":2.5,"boolean":true,"nothing":null,'
        '"array":[1,"two"],"object":{"ok":true},"string":"42"}}'
        "</ifm|tool_call></ifm|tool_calls>"
    )

    calls = parse_tool_calls_from_text(text, allow_incomplete = False)

    assert [_function(call) for call in calls] == [
        (
            "python",
            {
                "integer": 1,
                "float": 2.5,
                "boolean": True,
                "nothing": None,
                "array": [1, "two"],
                "object": {"ok": True},
                "string": "42",
            },
        )
    ]


def test_ifm_json_string_may_contain_reserved_ifm_markers():
    marker_text = "literal <ifm|arg_key><ifm|arg_value>"
    text = (
        "<ifm|tool_calls><ifm|tool_call>"
        + json.dumps({"name": "python", "arguments": {"code": marker_text}})
        + "</ifm|tool_call></ifm|tool_calls>"
    )

    calls = parse_tool_calls_from_text(text, allow_incomplete = False)

    assert [_function(call) for call in calls] == [("python", {"code": marker_text})]


@pytest.mark.parametrize(
    "text",
    [
        IFM_XML.removesuffix("</ifm|tool_calls>"),
        IFM_XML.replace("</ifm|tool_call>", ""),
        IFM_XML[: IFM_XML.index("891011")],
        IFM_XML.replace("</ifm|arg_key>", "</ifm|arg_value>"),
        IFM_XML.replace(
            "<ifm|arg_key>code</ifm|arg_key>",
            "<ifm|arg_key>code</ifm|arg_key><ifm|arg_key>code</ifm|arg_key>",
        ),
        IFM_XML.replace("</ifm|arg_value>", ""),
    ],
    ids = [
        "missing_outer_close",
        "missing_call_close",
        "missing_value_close",
        "mismatched_tags",
        "duplicate_key",
        "truncated_value",
    ],
)
def test_ifm_malformed_or_incomplete_never_parses(text):
    assert parse_tool_calls_from_text(text) == []
    assert parse_tool_calls_from_text(text, allow_incomplete = False) == []
    assert strip_tool_markup(text, final = True) == text.strip()


@pytest.mark.parametrize(
    "text",
    [
        (
            "<ifm|tool_calls>"
            "<ifm|tool_call>python<ifm|arg_key>code</ifm|arg_key>"
            "<ifm|arg_value>print(1)</ifm|arg_value>"
            "<ifm|tool_call>web_search<ifm|arg_key>query</ifm|arg_key>"
            "<ifm|arg_value>cats</ifm|arg_value></ifm|tool_call>"
            "</ifm|tool_calls>"
        ),
        (
            "<ifm|tool_calls>"
            "<ifm|tool_call>python<ifm|arg_key>code</ifm|arg_key>"
            "<ifm|arg_value>print(1)"
            "<ifm|tool_call>web_search<ifm|arg_key>query</ifm|arg_key>"
            "<ifm|arg_value>cats</ifm|arg_value></ifm|tool_call>"
            "</ifm|tool_calls>"
        ),
    ],
    ids = ["missing_first_call_close", "missing_first_value_close"],
)
def test_ifm_malformed_sibling_boundaries_never_execute(text):
    events, executor = _run(
        [[text]],
        tools = [_tool("python"), _tool("web_search")],
    )

    assert executor.calls == []
    assert not any(event["type"] in {"tool_start", "tool_end"} for event in events)


def test_ifm_unterminated_value_cannot_borrow_later_sibling_fields():
    text = (
        "<ifm|tool_calls><ifm|tool_call>python"
        "<ifm|arg_key>a</ifm|arg_key><ifm|arg_value>x"
        "<ifm|arg_key>b</ifm|arg_key><ifm|arg_value>y</ifm|arg_value>"
        "</ifm|tool_call></ifm|tool_calls>"
    )

    assert parse_tool_calls_from_text(text, allow_incomplete = False) == []


def test_ifm_unterminated_value_never_reaches_tool_loop():
    text = (
        "<ifm|tool_calls><ifm|tool_call>python"
        "<ifm|arg_key>a</ifm|arg_key><ifm|arg_value>x"
        "<ifm|arg_key>b</ifm|arg_key><ifm|arg_value>y</ifm|arg_value>"
        "</ifm|tool_call></ifm|tool_calls>"
    )
    events, executor = _run([[text]], tools = [_tool("python")])

    assert executor.calls == []
    assert not any(event["type"] in {"tool_start", "tool_end"} for event in events)


def test_ifm_unterminated_value_cannot_be_repaired_by_later_sibling_fields():
    text = (
        "<ifm|tool_calls><ifm|tool_call>python"
        "<ifm|arg_key>a</ifm|arg_key><ifm|arg_value>x"
        "<ifm|arg_value>y</ifm|arg_value>"
        "<ifm|arg_key>b</ifm|arg_key><ifm|arg_value>z</ifm|arg_value>"
        "</ifm|tool_call></ifm|tool_calls>"
    )

    assert parse_tool_calls_from_text(text, allow_incomplete = False) == []
    events, executor = _run([[text], ["done"]], tools = [_tool("python")])
    assert executor.calls == []
    assert not any(event["type"] in {"tool_start", "tool_end"} for event in events)


def test_ifm_valid_two_argument_form_preserves_both_arguments():
    text = (
        "<ifm|tool_calls><ifm|tool_call>python"
        "<ifm|arg_key>a</ifm|arg_key><ifm|arg_value>x</ifm|arg_value>"
        "<ifm|arg_key>b</ifm|arg_key><ifm|arg_value>y</ifm|arg_value>"
        "</ifm|tool_call></ifm|tool_calls>"
    )

    calls = parse_tool_calls_from_text(text, allow_incomplete = False)
    assert [_function(call) for call in calls] == [("python", {"a": "x", "b": "y"})]


@pytest.mark.parametrize(
    "field_sequence",
    [
        "<ifm|arg_type>string</ifm|arg_type><ifm|arg_value>y</ifm|arg_value>",
        "<ifm|arg_value>y</ifm|arg_value>",
    ],
    ids = ["later_arg_type", "later_arg_value"],
)
def test_ifm_unterminated_value_rejects_later_field_openers(field_sequence):
    text = (
        "<ifm|tool_calls><ifm|tool_call>python"
        "<ifm|arg_key>a</ifm|arg_key><ifm|arg_value>x"
        + field_sequence
        + "</ifm|tool_call></ifm|tool_calls>"
    )

    assert parse_tool_calls_from_text(text, allow_incomplete = False) == []


@pytest.mark.parametrize(
    "marker",
    [
        "<ifm|arg_key>",
        "<ifm|arg_type>",
        "<ifm|arg_value>",
        "</ifm|arg_key>",
        "</ifm|arg_type>",
        "<ifm|tool_call>",
        "</ifm|tool_call>",
        "<ifm|tool_calls>",
        "</ifm|tool_calls>",
        "</ifm|arg_value>",
    ],
)
def test_ifm_reserved_markers_inside_xml_value_fail_closed(marker):
    text = (
        "<ifm|tool_calls><ifm|tool_call>python"
        "<ifm|arg_key>code</ifm|arg_key><ifm|arg_value>prefix "
        + marker
        + " suffix</ifm|arg_value></ifm|tool_call></ifm|tool_calls>"
    )

    assert parse_tool_calls_from_text(text, allow_incomplete = False) == []
    events, executor = _run([[text], ["done"]], tools = [_tool("python")])
    assert executor.calls == []
    assert not any(event["type"] in {"tool_start", "tool_end"} for event in events)


@pytest.mark.parametrize(
    "text",
    [
        "The docs say '\n" + IFM_XML + "\n'.",
        'The docs say "\n' + IFM_XML + '\n".',
        r"The docs say \'escaped \' quote " + IFM_XML + r"'.",
        r'The docs say "escaped \" quote ' + IFM_XML + r'".',
        f"```xml\n{IFM_XML}\n```",
    ],
    ids = [
        "multiline_single_quote",
        "multiline_double_quote",
        "escaped_single_quote",
        "escaped_double_quote",
        "markdown_fence",
    ],
)
def test_ifm_lookalikes_in_multiline_and_escaped_literals_never_execute(text):
    events, executor = _run(
        [[text]],
        tools = [_tool("python"), _tool("web_search")],
    )

    assert executor.calls == []
    assert not any(event["type"] in {"tool_start", "tool_end"} for event in events)


def test_ifm_truncated_outer_never_executes_in_production_loop():
    text = IFM_XML.removesuffix("</ifm|tool_calls>")
    events, executor = _run([[text]], tools = [_tool("python")])

    assert executor.calls == []
    assert not any(event["type"] == "tool_start" for event in events)


def test_ifm_real_call_after_closed_quoted_prose_executes_once():
    text = f'The docs say "not a tool call".\n{IFM_XML}'
    events, executor = _run(
        [[text], ["done"]],
        tools = [_tool("python")],
    )

    assert executor.calls == [("python", {"code": "print(1234567 * 891011)"})]
    assert len([event for event in events if event["type"] == "tool_start"]) == 1
    assert len([event for event in events if event["type"] == "tool_end"]) == 1


def test_ifm_real_call_after_closed_markdown_fence_executes_once():
    # The quote inside the fenced example is intentionally unclosed. Fence boundaries must
    # still prevent that unrelated lexical state from suppressing the real call afterward.
    text = f'```xml\nexample = "{IFM_XML}\n```\n{IFM_XML}'
    events, executor = _run(
        [[text], ["done"]],
        tools = [_tool("python")],
    )

    assert executor.calls == [("python", {"code": "print(1234567 * 891011)"})]
    assert len([event for event in events if event["type"] == "tool_start"]) == 1
    assert len([event for event in events if event["type"] == "tool_end"]) == 1


def test_ifm_structural_strip_removes_only_complete_envelopes():
    assert strip_tool_markup("before " + IFM_XML + " after", final = True) == "before  after"
    assert has_tool_signal(IFM_XML)
    assert "<ifm|tool_calls>" in TOOL_XML_SIGNALS


def test_bare_args_prose_does_not_block_following_ifm_call():
    text = "The [ARGS] marker denotes arguments. " + IFM_XML

    calls = parse_tool_calls_from_text(text, allow_incomplete = False)

    assert [_function(call) for call in calls] == [
        ("python", {"code": "print(1234567 * 891011)"}),
    ]


def test_enabled_rehearsal_retains_precedence_over_following_ifm_call():
    rehearsal = 'get_weather[ARGS]{"city": "Paris"}'
    text = rehearsal + " " + IFM_XML

    calls = parse_tool_calls_from_text(
        text,
        allow_incomplete = False,
        enabled_tool_names = {"get_weather", "python"},
    )

    assert [_function(call) for call in calls] == [("get_weather", {"city": "Paris"})]


def test_ifm_lookalikes_in_quotes_fences_reasoning_and_payloads_are_not_promoted():
    quoted = f'The protocol example is "{IFM_XML}".'
    single_quoted = f"The protocol example is '{IFM_XML}'."
    fenced = f"```xml\n{IFM_XML}\n```"
    canonical_reasoning = f"<think>planning {IFM_XML}</think>answer"
    native_reasoning = f"<ifm|think_fast>planning {IFM_XML}</ifm|think>answer"
    nested_in_xml_value = (
        "<ifm|tool_calls><ifm|tool_call>python"
        "<ifm|arg_key>code</ifm|arg_key>"
        f"<ifm|arg_value>literal payload: {IFM_XML}</ifm|arg_value>"
        "</ifm|tool_call></ifm|tool_calls>"
    )
    nested_in_existing_call = (
        '<tool_call>{"name":"python","arguments":{"code":' + json.dumps(IFM_XML) + "}}</tool_call>"
    )

    assert parse_tool_calls_from_text(quoted) == []
    assert parse_tool_calls_from_text(single_quoted) == []
    assert parse_tool_calls_from_text(fenced) == []
    assert parse_tool_calls_from_text(canonical_reasoning) == []
    assert parse_tool_calls_from_text(native_reasoning) == []
    assert parse_tool_calls_from_text(nested_in_xml_value) == []
    assert _function(parse_tool_calls_from_text(nested_in_existing_call)[0]) == (
        "python",
        {"code": IFM_XML},
    )


def test_ifm_streaming_stripper_matches_final_structural_strip_at_one_char_boundaries():
    stripper = StreamingMarkupStripper(None)
    for index in range(1, len(IFM_XML) + 1):
        stripper.strip(IFM_XML[:index])
    assert stripper.strip(IFM_XML) == strip_tool_markup(IFM_XML, final = True) == ""


def test_ifm_one_character_stream_executes_once_without_protocol_leak():
    captured = []
    events, executor = _run(
        [list(IFM_XML), ["final answer"]],
        captured = captured,
        results = ["123"],
    )

    assert executor.calls == [("python", {"code": "print(1234567 * 891011)"})]
    assert len([event for event in events if event["type"] == "tool_start"]) == 1
    assert len([event for event in events if event["type"] == "tool_end"]) == 1
    assert not any(
        "<ifm|" in event.get("text", "") for event in events if event["type"] == "content"
    )
    assert any(event["type"] == "content" and event["text"] == "final answer" for event in events)
    assert len(captured) == 2
    assert any(message.get("role") == "tool" for message in captured[1])


def test_ifm_partial_outer_marker_after_visible_text_stays_buffered():
    prefix = "Before response; "
    events, executor = _run(
        [list(prefix + IFM_XML), ["done"]],
        results = ["123"],
    )

    assert executor.calls == [("python", {"code": "print(1234567 * 891011)"})]
    content = "".join(event.get("text", "") for event in events if event["type"] == "content")
    assert prefix in content
    assert "<ifm|" not in content


@pytest.mark.parametrize(
    "prefix",
    [
        "Before response; ",
        "<think></think>",
    ],
    ids = ["visible_prefix", "normalized_reasoning_close"],
)
def test_ifm_atomic_outer_marker_after_safe_text_stays_buffered(prefix):
    opener = "<ifm|tool_calls>"
    first = prefix + opener + "\n"
    remainder = IFM_XML[len(opener) + 1 :]
    captured = []
    events, executor = _run(
        [[first, remainder], ["done"]],
        captured = captured,
        results = ["123"],
    )

    assert executor.calls == [("python", {"code": "print(1234567 * 891011)"})]
    contents = [event["text"] for event in events if event["type"] == "content"]
    assert any(prefix in content for content in contents)
    assert not any("ifm|" in content for content in contents)
    assert len([event for event in events if event["type"] == "tool_start"]) == 1
    assert len([event for event in events if event["type"] == "tool_end"]) == 1
    assert any(content == "done" for content in contents)
    assert len(captured) == 2
    assert any(message.get("role") == "tool" for message in captured[1])


def test_ifm_buffer_without_tool_signal_streams_ordinary_text():
    text = "ordinary visible response"
    events, executor = _run([[text]])

    assert executor.calls == []
    assert any(event["type"] == "content" and text in event["text"] for event in events)


def test_ifm_multiple_calls_execute_once_in_order():
    text = (
        "<ifm|tool_calls>"
        "<ifm|tool_call>python<ifm|arg_key>code</ifm|arg_key>"
        "<ifm|arg_value>print(1)</ifm|arg_value></ifm|tool_call>"
        "<ifm|tool_call>web_search<ifm|arg_key>query</ifm|arg_key>"
        "<ifm|arg_value>cats</ifm|arg_value></ifm|tool_call>"
        "</ifm|tool_calls>"
    )
    events, executor = _run(
        [[text], ["done"]],
        tools = [_tool("python"), _tool("web_search")],
        results = ["python result", "search result"],
    )

    assert executor.calls == [
        ("python", {"code": "print(1)"}),
        ("web_search", {"query": "cats"}),
    ]
    assert [event["tool_name"] for event in events if event["type"] == "tool_end"] == [
        "python",
        "web_search",
    ]


def test_ifm_malformed_stream_is_visible_at_eos_but_never_executes():
    malformed = IFM_XML.removesuffix("</ifm|tool_calls>")
    captured = []
    events, executor = _run([[malformed]], captured = captured)

    assert executor.calls == []
    assert len(captured) == 1
    assert any(
        event["type"] == "content" and event["text"] == malformed.strip() for event in events
    )
    assert not any(event["type"] == "tool_start" for event in events)


def test_ifm_unknown_tool_still_follows_controller_authorization():
    captured = []
    events, executor = _run(
        [[IFM_XML.replace("python", "terminal")], ["done"]],
        tools = [_tool("web_search")],
        captured = captured,
    )

    assert executor.calls == []
    assert not any(event["type"] in {"tool_start", "tool_end"} for event in events)
    assert len(captured) == 2


def test_ifm_quoted_prose_does_not_execute_in_the_tool_loop():
    quoted = f'The model documentation says "{IFM_XML}".'
    captured = []
    events, executor = _run([[quoted]], captured = captured)

    assert executor.calls == []
    assert any(event["type"] == "content" and quoted in event["text"] for event in events)


class _StrictIfmTokenizer:
    """Small native-template stand-in that requires IFM thinking fields on assistant history."""

    chat_template = "native <ifm|think> <ifm|tool_calls> <ifm|tool_call>"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize = False,
        add_generation_prompt = True,
        **kwargs,
    ):
        del tokenize, add_generation_prompt, kwargs
        rendered = []
        for message in messages:
            if message.get("role") == "assistant" and message.get("tool_calls"):
                thinking = next(
                    (
                        message[field]
                        for field in (
                            "think",
                            "think_fast",
                            "think_faster",
                            "reasoning_content",
                            "reasoning",
                        )
                        if isinstance(message.get(field), str)
                    ),
                    None,
                )
                if thinking is None:
                    raise ValueError("IFM assistant history requires thinking")
                rendered.append(f"IFM_THOUGHT={thinking};CONTENT={message.get('content', '')}")
                rendered.append(f"IFM_CALLS={message['tool_calls']!r}")
            elif message.get("role") == "tool":
                rendered.append(f"IFM_TOOL_RESULT={message.get('content', '')}")
            else:
                rendered.append(str(message.get("content", "")))
        return "\n".join(rendered)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("reasoning_content", None),
        ("think_fast", {"trace": "not text"}),
        ("reasoning", 7),
    ],
)
def test_ifm_tool_history_repair_replaces_invalid_reasoning_field(field, invalid_value):
    messages = [
        {"role": "user", "content": "calculate"},
        {
            "role": "assistant",
            "content": "answer",
            field: invalid_value,
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "python", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "123"},
    ]

    repaired = _repair_ifm_tool_history(messages, _StrictIfmTokenizer(), [_tool("python")])

    assert repaired is not messages
    assert repaired[0] is messages[0]
    assert repaired[2] is messages[2]
    assert repaired[1] is not messages[1]
    assert repaired[1][field] == ""
    assert messages[1][field] == invalid_value
    assert "IFM_THOUGHT=;CONTENT=answer" in apply_chat_template_for_generation(
        _StrictIfmTokenizer(), messages, tools = [_tool("python")]
    )


def test_ifm_tool_history_repair_preserves_valid_reasoning_string():
    messages = [
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "existing reasoning",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "python", "arguments": "{}"},
                }
            ],
        }
    ]

    repaired = _repair_ifm_tool_history(messages, _StrictIfmTokenizer(), [_tool("python")])

    assert repaired is messages
    assert repaired[0]["reasoning_content"] == "existing reasoning"


def test_ifm_tool_result_replay_repairs_generic_assistant_history_for_second_generation():
    tokenizer = _StrictIfmTokenizer()
    rendered_prompts = []
    turns = iter(
        [
            [
                "<think>planning</think>",
                IFM_XML,
            ],
            ["final after tool"],
        ]
    )

    def single_turn(messages, **_kwargs):
        rendered_prompts.append(
            apply_chat_template_for_generation(
                tokenizer,
                messages,
                tools = [_tool("python")],
            )
        )
        accumulated = ""
        for chunk in next(turns):
            accumulated += chunk
            yield accumulated

    executor = _Executor(["123"])
    events = list(
        run_safetensors_tool_loop(
            single_turn = single_turn,
            messages = [{"role": "user", "content": "calculate"}],
            tools = [_tool("python")],
            execute_tool = executor,
            nudge_tool_calls = False,
            permission_mode = "off",
            max_tool_iterations = 2,
        )
    )

    assert executor.calls == [("python", {"code": "print(1234567 * 891011)"})]
    assert len(rendered_prompts) == 2
    assert "IFM_THOUGHT=planning" in rendered_prompts[1]
    assert "IFM_TOOL_RESULT=123" in rendered_prompts[1]
    assert "<think>planning</think>" not in rendered_prompts[1]
    assert any(
        event["type"] == "content" and event["text"] == "final after tool" for event in events
    )


def test_ifm_xml_value_unmatched_quote_is_payload():
    import json

    text = (
        "<ifm|tool_calls><ifm|tool_call>web_search"
        "<ifm|arg_key>query</ifm|arg_key>"
        '<ifm|arg_value>find "unfinished</ifm|arg_value>'
        "</ifm|tool_call></ifm|tool_calls>"
    )

    calls = parse_tool_calls_from_text(text, allow_incomplete = False)

    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "web_search"
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": 'find "unfinished'}


def test_ifm_xml_value_unclosed_code_fence_is_payload():
    import json

    text = (
        "<ifm|tool_calls><ifm|tool_call>web_search"
        "<ifm|arg_key>query</ifm|arg_key>"
        "<ifm|arg_value>find ```python\\nunfinished"
        "</ifm|arg_value></ifm|tool_call></ifm|tool_calls>"
    )

    calls = parse_tool_calls_from_text(text, allow_incomplete = False)

    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "web_search"
    assert json.loads(calls[0]["function"]["arguments"]) == {"query": "find ```python\\nunfinished"}


def test_ifm_value_literal_spill_does_not_hide_later_envelope():
    import json

    text = (
        "<ifm|tool_calls><ifm|tool_call>first"
        "<ifm|arg_key>q</ifm|arg_key>"
        '<ifm|arg_value>find "unfinished</ifm|arg_value>'
        "</ifm|tool_call></ifm|tool_calls>"
        "\n"
        "<ifm|tool_calls><ifm|tool_call>second"
        "<ifm|arg_key>q</ifm|arg_key>"
        "<ifm|arg_value>normal</ifm|arg_value>"
        "</ifm|tool_call></ifm|tool_calls>"
    )

    calls = parse_tool_calls_from_text(text, allow_incomplete = False)

    assert [call["function"]["name"] for call in calls] == ["first", "second"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"q": 'find "unfinished'}
    assert json.loads(calls[1]["function"]["arguments"]) == {"q": "normal"}


def test_ifm_value_literal_spill_reset_keeps_later_quoted_example_protected():
    text = (
        "<ifm|tool_calls><ifm|tool_call>first"
        "<ifm|arg_key>q</ifm|arg_key>"
        '<ifm|arg_value>find "unfinished</ifm|arg_value>'
        "</ifm|tool_call></ifm|tool_calls>"
        "\n"
        '"quoted example: '
        "<ifm|tool_calls><ifm|tool_call>fake"
        "<ifm|arg_key>q</ifm|arg_key>"
        "<ifm|arg_value>do not run</ifm|arg_value>"
        "</ifm|tool_call></ifm|tool_calls>"
        '"\n'
        "<ifm|tool_calls><ifm|tool_call>second"
        "<ifm|arg_key>q</ifm|arg_key>"
        "<ifm|arg_value>normal</ifm|arg_value>"
        "</ifm|tool_call></ifm|tool_calls>"
    )

    calls = parse_tool_calls_from_text(text, allow_incomplete = False)

    assert [call["function"]["name"] for call in calls] == ["first", "second"]
