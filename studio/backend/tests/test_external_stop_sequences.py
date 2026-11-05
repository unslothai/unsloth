# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An explicit external provider is sent no stop sequences, so its stream is cut on the way through."""

import asyncio
import json

from routes.inference import _cut_studio_reply_at_stops


def _event(content, finish = None):
    delta = {} if content is None else {"content": content}
    chunk = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    return f"data: {json.dumps(chunk)}\n\n"


def _run(events, stops):
    closed = []

    async def upstream():
        try:
            for event in events:
                yield event.encode()
        finally:
            closed.append(True)

    async def collect():
        return "".join([part async for part in _cut_studio_reply_at_stops(upstream(), stops)])

    return asyncio.run(collect()), closed


def _text(stream):
    text, finishes = "", []
    for line in stream.splitlines():
        if line.startswith("data:") and line[5:].strip() != "[DONE]":
            for choice in json.loads(line[5:]).get("choices", []):
                text += choice["delta"].get("content") or ""
                finishes.append(choice.get("finish_reason"))
    return text, finishes


def test_a_stop_split_across_chunks_ends_the_reply_before_it():
    events = [_event("Sure.\nA"), _event("nn: hi"), _event(" more"), "data: [DONE]\n\n"]
    stream, closed = _run(events, ["\nAnn:"])
    text, finishes = _text(stream)
    assert text == "Sure." and finishes[-1] == "stop"
    assert stream.count("data: [DONE]") == 1 and closed == [True]


def test_reasoning_is_not_cut_and_a_held_back_tail_is_released():
    events = [
        _event("<think>maybe\nAnn: says no</think>"),
        _event("Wait\nA"),
        _event("nd then."),
        _event(None, "stop"),
        "data: [DONE]\n\n",
    ]
    text, _finishes = _text(_run(events, ["\nAnn:"])[0])
    assert text == "<think>maybe\nAnn: says no</think>Wait\nAnd then."
    text, _finishes = _text(_run([_event("Ends on \nAn"), "data: [DONE]\n\n"], ["\nAnn:"])[0])
    assert text == "Ends on \nAn"


def test_lines_without_choices_pass_through():
    tool_event = "data: " + json.dumps({"_toolEvent": {"type": "x"}}) + "\n\n"
    stream, _closed = _run(
        [": keep-alive\n\n", tool_event, _event("ok"), "data: [DONE]\n\n"], ["\nAnn:"]
    )
    assert stream.startswith(": keep-alive\n\n" + tool_event)
    assert _text(stream)[0] == "ok"


def test_think_tags_split_across_chunks_still_bound_the_reasoning():
    events = [
        _event("<thi"),
        _event("nk>plan\nAnn: no</th"),
        _event("ink>Fine.\nAnn: yes"),
        "data: [DONE]\n\n",
    ]
    text, finishes = _text(_run(events, ["\nAnn:"])[0])
    assert text == "<think>plan\nAnn: no</think>Fine." and finishes[-1] == "stop"
    text, _finishes = _text(
        _run([_event("Hi\nA"), _event("<think>x</think>"), _event("ok")], ["\nAnn:"])[0]
    )
    assert text == "Hi\nA<think>x</think>ok"


def test_held_text_goes_out_before_a_tool_event():
    tool_event = "data: " + json.dumps({"_toolEvent": {"type": "tool_start"}}) + "\n\n"
    stream, _closed = _run(
        [_event("Let me search for that."), tool_event, _event("Found."), "data: [DONE]\n\n"],
        ["\nUser:"],
    )
    before, after = stream.split(tool_event)
    assert _text(before)[0] == "Let me search for that." and _text(after)[0] == "Found."


def test_held_text_goes_out_before_a_hosted_tool_event_with_choices():
    event = {
        "choices": [{"index": 0, "delta": {}, "finish_reason": None}],
        "_toolEvent": {"type": "tool_start"},
    }
    tool_event = "data: " + json.dumps(event) + "\n\n"
    stream, _closed = _run(
        [_event("Let me search for that."), tool_event, _event("Found."), "data: [DONE]\n\n"],
        ["\nUser:"],
    )
    lines = [line for line in stream.splitlines() if line.startswith("data:")]
    at = next(i for i, line in enumerate(lines) if "_toolEvent" in line)
    assert _text("\n".join(lines[:at]))[0] == "Let me search for that."
    assert _text("\n".join(lines[at + 1 :]))[0] == "Found."
