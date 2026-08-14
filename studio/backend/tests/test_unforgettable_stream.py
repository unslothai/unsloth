# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for virtual-model inner-stream rewrite. No FastAPI, no GPU."""

from __future__ import annotations

import asyncio
import json
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.unforgettable_host import (
    VIRTUAL_MODEL_ID,
    StudioHost,
    _forward_inner_stream,
    _rewrite_inner_frame,
    union_unforgettable_enabled_tools,
)


def _data_json(frame: bytes) -> dict:
    text = frame.decode("utf-8")
    payload = text.split("data:", 1)[1].strip()
    return json.loads(payload)


def test_rewrite_tool_start_unchanged():
    frame = b'data: {"type":"tool_start","tool_name":"memory_write","tool_call_id":"c1"}\n\n'
    assert _rewrite_inner_frame(frame) == frame


def test_rewrite_nulls_finish_reason_and_remints_model():
    frame = (
        b'data: {"id":"chatcmpl-inner","object":"chat.completion.chunk",'
        b'"model":"qwen","choices":[{"index":0,"delta":{"content":"hi"},'
        b'"finish_reason":"stop"}]}\n\n'
    )
    out = _rewrite_inner_frame(frame)
    assert out is not None
    payload = _data_json(out)
    assert payload["model"] == VIRTUAL_MODEL_ID
    assert payload["choices"][0]["finish_reason"] is None
    assert payload["choices"][0]["delta"]["content"] == "hi"
    assert payload["id"] == "chatcmpl-inner"


def test_rewrite_drops_inner_done():
    assert _rewrite_inner_frame(b"data: [DONE]\n\n") is None
    assert _rewrite_inner_frame("data: [DONE]") is None


def test_rewrite_forwards_error_and_unknown_frames():
    err = b'data: {"error":{"message":"boom","type":"server_error"}}\n\n'
    assert _rewrite_inner_frame(err) == err
    other = b'data: {"type":"diffusion_frame","step":1}\n\n'
    assert _rewrite_inner_frame(other) == other


class _ClosingIter:
    def __init__(self, frames: list[bytes]):
        self.frames = list(frames)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.frames:
            raise StopAsyncIteration
        return self.frames.pop(0)

    async def aclose(self):
        self.closed = True


def test_forward_inner_stream_rewrites_and_acloses():
    iterator = _ClosingIter(
        [
            b'data: {"type":"tool_start","tool_name":"memory_search"}\n\n',
            (
                b'data: {"object":"chat.completion.chunk","model":"inner",'
                b'"choices":[{"index":0,"delta":{"content":"hel"},'
                b'"finish_reason":"stop"}]}\n\n'
            ),
            b"data: [DONE]\n\n",
            (
                b'data: {"object":"chat.completion.chunk","model":"inner",'
                b'"choices":[{"index":0,"delta":{"content":"lo"},'
                b'"finish_reason":null}]}\n\n'
            ),
        ]
    )
    forwarded: list[bytes] = []

    async def on_chunk(data: bytes) -> None:
        forwarded.append(data)

    text = asyncio.run(_forward_inner_stream(SimpleNamespace(body_iterator = iterator), on_chunk))
    assert iterator.closed
    assert text == "hello"
    assert len(forwarded) == 3
    assert _data_json(forwarded[0])["type"] == "tool_start"
    assert _data_json(forwarded[1])["model"] == VIRTUAL_MODEL_ID
    assert _data_json(forwarded[1])["choices"][0]["finish_reason"] is None
    assert _data_json(forwarded[2])["choices"][0]["delta"]["content"] == "lo"
    assert all(b"[DONE]" not in item for item in forwarded)


def test_forward_inner_stream_acloses_when_on_chunk_breaks():
    iterator = _ClosingIter(
        [
            b'data: {"object":"chat.completion.chunk","choices":[{"delta":{"content":"a"}}]}\n\n',
            b'data: {"object":"chat.completion.chunk","choices":[{"delta":{"content":"b"}}]}\n\n',
        ]
    )

    async def on_chunk(data: bytes) -> None:
        raise BrokenPipeError("client gone")

    text = asyncio.run(_forward_inner_stream(SimpleNamespace(body_iterator = iterator), on_chunk))
    assert iterator.closed
    assert text == "a"


def test_forward_inner_stream_acloses_on_cancel():
    iterator = _ClosingIter(
        [
            b'data: {"object":"chat.completion.chunk","choices":[{"delta":{"content":"x"}}]}\n\n',
        ]
    )

    async def on_chunk(data: bytes) -> None:
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_forward_inner_stream(SimpleNamespace(body_iterator = iterator), on_chunk))
    assert iterator.closed


def test_enabled_tools_union_keeps_pills_and_adds_apache_tools():
    unioned = union_unforgettable_enabled_tools(["python", "terminal"])
    assert unioned is not None
    assert unioned[:2] == ["python", "terminal"]
    assert "rims_enter_sim" in unioned
    assert "memory_write" in unioned
    assert union_unforgettable_enabled_tools(None) is None


def _install_execute_tool(monkeypatch, fn) -> None:
    name = "core.inference.tools"
    if name not in sys.modules:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setattr(sys.modules[name], "execute_tool", fn, raising = False)


def test_studio_host_run_action_emits_tool_start_and_end(monkeypatch):
    seen = {}

    def execute_tool(name, arguments, session_id = None, timeout = None, cancel_event = None, **kwargs):
        seen["name"] = name
        seen["arguments"] = arguments
        seen["session_id"] = session_id
        seen["timeout"] = timeout
        seen["cancel_event"] = cancel_event
        time.sleep(0.08)
        return "===== 3 passed in 0.01s =====\n"

    _install_execute_tool(monkeypatch, execute_tool)
    monkeypatch.setattr(
        "core.inference.tool_stream_exec.TOOL_HEARTBEAT_INTERVAL_S",
        0.02,
    )
    host = StudioHost(
        payload = None,
        request = None,
        current_subject = "u",
        inner = None,
        inner_model = "default",
    )
    frames: list[bytes] = []

    def on_chunk(data: bytes) -> None:
        frames.append(data)

    result = asyncio.run(
        host.run_action("sim-1", "terminal", {"command": "pytest"}, on_chunk = on_chunk)
    )
    assert result.startswith("=====")
    assert seen["name"] == "terminal"
    assert seen["session_id"] == "sim-1"
    assert seen["timeout"] == 300
    assert seen["cancel_event"] is host.cancel_event
    data_frames = [frame for frame in frames if frame.startswith(b"data: ")]
    assert len(data_frames) >= 2
    start = _data_json(data_frames[0])
    end = _data_json(data_frames[-1])
    assert start["type"] == "tool_start"
    assert start["tool_name"] == "terminal"
    assert start["approval_id"] == ""
    assert start["awaiting_confirmation"] is False
    assert start["tool_call_id"].startswith("rims-action-")
    assert len(start["tool_call_id"]) == len("rims-action-") + 16
    assert start["arguments"] == {"command": "pytest"}
    assert end["type"] == "tool_end"
    assert end["tool_name"] == "terminal"
    assert end["tool_call_id"] == start["tool_call_id"]
    assert end["result"].startswith("=====")
    assert b": keep-alive\n\n" in frames

    denied = asyncio.run(host.run_action("sim-1", "web_search", {}))
    assert denied == "Error: run_action supports python|terminal only, got 'web_search'"
