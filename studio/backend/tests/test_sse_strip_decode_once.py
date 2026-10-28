# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ServerToolCallStripper decodes each relayed line once, and its outputs and debt are unchanged."""

import json

import pytest

import core.inference.sse_control_frames as frames
from core.inference.sse_control_frames import ServerToolCallStripper


def _chunk(
    delta,
    finish_reason = None,
    **extra,
):
    payload = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        **extra,
    }
    return "data: " + json.dumps(payload, separators = (",", ":"))


@pytest.fixture
def decodes(monkeypatch):
    calls = []
    real = json.loads

    def counting(*args, **kwargs):
        calls.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(frames.json, "loads", counting)
    return calls


@pytest.mark.parametrize(
    "line",
    [
        _chunk({"content": "hello"}),
        _chunk({}, "stop"),
        _chunk({"reasoning_content": "thinking"}),
        _chunk({}, None, usage = {"prompt_tokens": 3}),
    ],
)
def test_untouched_line_is_decoded_once(decodes, line):
    stripper = ServerToolCallStripper()
    assert stripper.strip(line) is line
    assert len(decodes) == 1


def test_rewritten_line_is_read_back_once(decodes):
    call = {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "f", "arguments": "{}"}}]}
    line = _chunk({"content": "x", **call}, "tool_calls")
    stripper = ServerToolCallStripper()
    out = stripper.strip(line)
    assert out is not None and "tool_calls" not in out
    # The input once, the rewritten line once to see whether it still closes the turn.
    assert len(decodes) == 2


def test_stream_outputs_and_owed_finish_unchanged():
    call = {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "f", "arguments": "{}"}}]}
    stripper = ServerToolCallStripper()
    lines = [
        _chunk({"role": "assistant", "content": ""}),
        _chunk(call),
        _chunk({}, "stop"),
        "data: [DONE]",
    ]
    outs = [stripper.strip(line) for line in lines]
    assert outs[0] is lines[0]
    assert outs[1] is None
    assert outs[2] is None
    assert outs[3] == "data: [DONE]"
    owed = json.loads(stripper.owed_terminal_chunk()[len("data: ") :])
    assert owed == {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    assert stripper.owed_terminal_chunk() is None
