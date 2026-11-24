"""A think tag quoted inside a tool call's JSON arguments is argument data, so it must not be read as
reasoning that hides the call. The closed-pair span of ``<tool_call>`` ends at the first
``</tool_call>``, so a quoted closer followed by a quoted ``<think>`` used to mask the real closer and
drop the call; DeepSeek and ``<|python_tag|>`` calls had no span at all, so a quoted ``</think>`` read
as a prefilled-reasoning close and masked the opener."""

import json

import pytest

from core.inference.tool_call_parser import parse_tool_calls_from_text


def _hermes(name, args):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"


def _deepseek(name, args):
    return (
        "<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>"
        + name
        + "\n```json\n"
        + json.dumps(args)
        + "\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜>"
    )


def _python_tag(name, args):
    return "<|python_tag|>" + json.dumps({"name": name, "parameters": args})


_ARGS = [
    {"a": "</tool_call>", "b": "<think>"},
    {"a": "</think>", "b": "<tool_call>"},
    {"a": "</think>", "b": "</tool_call>"},
    {"path": "parse.py", "content": 'END = "</tool_call>"\nTHINK = "<think>"\n'},
]


@pytest.mark.parametrize("build", [_hermes, _deepseek, _python_tag])
@pytest.mark.parametrize("args", _ARGS)
@pytest.mark.parametrize("preface", ["", "Let me check.\n\n"])
@pytest.mark.parametrize("allow_incomplete", [True, False])
def test_quoted_think_tags_in_arguments_keep_the_call(build, args, preface, allow_incomplete):
    text = preface + build("write_file", args)
    calls = parse_tool_calls_from_text(text, allow_incomplete = allow_incomplete)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "write_file"
    assert json.loads(calls[0]["function"]["arguments"]) == args


def test_quoted_tags_after_real_reasoning_keep_the_call():
    args = {"code": 's.split("</tool_call>")\nt = "<think>"'}
    calls = parse_tool_calls_from_text("<think>plan</think>\n" + _hermes("python", args))
    assert [json.loads(c["function"]["arguments"]) for c in calls] == [args]


def test_a_call_rehearsed_in_reasoning_still_does_not_run():
    text = "<think>I could " + _hermes("bad", {}) + "</think>\n" + _hermes("good", {})
    assert [c["function"]["name"] for c in parse_tool_calls_from_text(text)] == ["good"]


def test_a_call_rehearsed_in_prefilled_reasoning_still_does_not_run():
    text = "plan " + _hermes("bad", {}) + " done</think>\n" + _hermes("good", {})
    assert [c["function"]["name"] for c in parse_tool_calls_from_text(text)] == ["good"]
