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
    _parse_sse_json,
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


def test_stream_tool_execution_copies_episode_context():
    from core.inference.tool_stream_exec import stream_tool_execution
    from unforgettable.loop.runtime import bind_episode, current_db_path, reset_episode

    seen = {}

    def invoke(_cb):
        seen["db"] = current_db_path()
        return "ok"

    tokens, _ = bind_episode(db_path = "/tmp/unforgettable-ctx.db", episode_id = "ep-ctx")
    try:
        gen = stream_tool_execution(invoke, tool_name = "memory_write")
        result = None
        try:
            while True:
                next(gen)
        except StopIteration as stop:
            result = stop.value
    finally:
        reset_episode(tokens)
    assert result == "ok"
    assert seen["db"] == "/tmp/unforgettable-ctx.db"


def test_enabled_tools_union_keeps_pills_and_adds_apache_tools():
    unioned = union_unforgettable_enabled_tools(["python", "terminal"])
    assert unioned is not None
    assert unioned[:2] == ["python", "terminal"]
    assert "rims_enter_sim" in unioned
    assert "memory_write" in unioned
    assert union_unforgettable_enabled_tools(None) is None


def test_studio_tools_list_does_not_ship_memory_specs():
    from core.inference.tools import ALL_TOOLS

    names = {tool["function"]["name"] for tool in ALL_TOOLS}
    assert "memory_write" not in names
    assert "rims_enter_sim" not in names


def test_execute_tool_patch_dispatches_memory_and_marks_safe(tmp_path):
    from core.unforgettable_patches import install
    from unforgettable.loop.runtime import bind_episode, reset_episode

    install()
    from core.inference.tools import execute_tool, is_always_safe_tool

    assert is_always_safe_tool("memory_search") is True
    tokens, _ = bind_episode(db_path = str(tmp_path / "memory.db"), episode_id = "ep-tools")
    try:
        result = execute_tool("memory_search", {"query": "nothing"})
    finally:
        reset_episode(tokens)
    assert isinstance(result, str)


def _install_execute_tool(monkeypatch, fn) -> None:
    name = "core.inference.tools"
    if name not in sys.modules:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setattr(sys.modules[name], "execute_tool", fn, raising = False)


def test_studio_host_run_action_emits_tool_start_and_end(monkeypatch):
    seen = {}

    def execute_tool(
        name,
        arguments,
        session_id = None,
        timeout = None,
        cancel_event = None,
        **kwargs,
    ):
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


def test_studio_host_confirm_emits_tool_start_and_end(monkeypatch):
    seen = {}

    def wait_tool_decision(
        slot,
        approval_id,
        cancel_event = None,
        timeout = None,
    ):
        seen["slot"] = slot
        seen["approval_id"] = approval_id
        seen["cancel_event"] = cancel_event
        time.sleep(0.08)
        return "allow"

    monkeypatch.setattr(
        "state.tool_approvals.begin_tool_decision", lambda session, approval: {"session": session}
    )
    monkeypatch.setattr("state.tool_approvals.wait_tool_decision", wait_tool_decision)
    monkeypatch.setattr("state.tool_approvals.new_approval_id", lambda: "approval-retry-1")
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

    allowed = asyncio.run(
        host.confirm(
            "Retry the repaired plan in the world?",
            kind = "retry_world",
            on_chunk = on_chunk,
            session_id = "world",
        )
    )
    assert allowed is True
    assert seen["approval_id"] == "approval-retry-1"
    assert seen["cancel_event"] is host.cancel_event
    data_frames = [frame for frame in frames if frame.startswith(b"data: ")]
    assert len(data_frames) >= 2
    start = _parse_sse_json(_rewrite_inner_frame(data_frames[0]).decode("utf-8"))
    end = _parse_sse_json(_rewrite_inner_frame(data_frames[-1]).decode("utf-8"))
    assert start is not None
    assert start["type"] == "tool_start"
    assert start["tool_name"] == "rims_retry_world"
    assert start["approval_id"] == "approval-retry-1"
    assert start["awaiting_confirmation"] is True
    assert start["tool_call_id"] == "approval-retry-1"
    assert start["arguments"] == {
        "prompt": "Retry the repaired plan in the world?",
        "kind": "retry_world",
    }
    assert end is not None
    assert end["type"] == "tool_end"
    assert end["tool_name"] == "rims_retry_world"
    assert end["tool_call_id"] == "approval-retry-1"
    assert end["result"] == "allowed"
    assert b": keep-alive\n\n" in frames

    assert asyncio.run(host.confirm("x")) is False
    host.cancel_event.set()
    assert asyncio.run(host.confirm("x", on_chunk = on_chunk)) is False


def _payload():
    class Payload:
        def __init__(self):
            self.model = "qwen"
            self.session_id = "world"
            self.thread_id = None
            self.enable_tools = False
            self.enabled_tools = None
            self.messages = []
            self.stream = False
            self.use_adapter = None

        def model_copy(self, deep = True):
            import copy
            return copy.deepcopy(self)

    return Payload()


def test_is_peft_adapter_dir_rejects_fake_and_accepts_peft(tmp_path):
    from core.unforgettable_host import is_peft_adapter_dir, peft_adapter_name

    missing = tmp_path / "nope"
    assert is_peft_adapter_dir(missing) is False
    fake = tmp_path / "fake-ada"
    fake.mkdir()
    (fake / "adapter_config.json").write_text(
        json.dumps({"fake": True, "recipe": "sft", "n": 4}),
        encoding = "utf-8",
    )
    assert is_peft_adapter_dir(fake) is False
    peft = tmp_path / "ada-uuid"
    peft.mkdir()
    (peft / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": "unsloth/Qwen3.5-4B",
            }
        ),
        encoding = "utf-8",
    )
    assert is_peft_adapter_dir(peft) is True
    assert peft_adapter_name(peft) == "ada-uuid"


def test_studio_host_generate_sets_use_adapter_for_peft_dir(tmp_path):
    from core.unforgettable_host import GenerateRequest
    from unforgettable.loop.runtime import bind_episode, reset_episode

    peft = tmp_path / "ada-live"
    peft.mkdir()
    (peft / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": "unsloth/Qwen3.5-4B"}),
        encoding = "utf-8",
    )
    seen = {}

    async def inner(payload, request, subject):
        seen["use_adapter"] = getattr(payload, "use_adapter", None)
        chunk = (
            b'data: {"id":"c","object":"chat.completion.chunk","model":"qwen",'
            b'"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
        )
        return SimpleNamespace(body_iterator = _ClosingIter([chunk]))

    host = StudioHost(
        payload = _payload(),
        request = None,
        current_subject = "u",
        inner = inner,
        inner_model = "qwen",
    )
    tokens, _ = bind_episode(db_path = str(tmp_path / "memory.db"), episode_id = "ep-ada")
    try:
        result = asyncio.run(
            host.generate(
                GenerateRequest(
                    messages = [{"role": "user", "content": "hi"}],
                    session_id = "world",
                    adapter_path = str(peft),
                )
            )
        )
    finally:
        reset_episode(tokens)
    assert seen["use_adapter"] == str(peft)
    assert result is not None


def test_studio_host_generate_skips_fake_adapter_dir(tmp_path):
    from core.unforgettable_host import GenerateRequest
    from unforgettable.loop.runtime import bind_episode, reset_episode

    fake = tmp_path / "ada-fake"
    fake.mkdir()
    (fake / "adapter_config.json").write_text(
        json.dumps({"fake": True, "n": 4}),
        encoding = "utf-8",
    )
    seen = {}

    async def inner(payload, request, subject):
        seen["use_adapter"] = getattr(payload, "use_adapter", None)
        chunk = (
            b'data: {"id":"c","object":"chat.completion.chunk","model":"qwen",'
            b'"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
        )
        return SimpleNamespace(body_iterator = _ClosingIter([chunk]))

    host = StudioHost(
        payload = _payload(),
        request = None,
        current_subject = "u",
        inner = inner,
        inner_model = "qwen",
    )
    tokens, _ = bind_episode(db_path = str(tmp_path / "memory.db"), episode_id = "ep-fake")
    try:
        asyncio.run(
            host.generate(
                GenerateRequest(
                    messages = [{"role": "user", "content": "hi"}],
                    session_id = "world",
                    adapter_path = str(fake),
                )
            )
        )
    finally:
        reset_episode(tokens)
    assert seen["use_adapter"] is None


def test_prepare_sidecar_adapter_loads_peft_directory(tmp_path):
    from core.unforgettable_host import prepare_sidecar_adapter

    peft = tmp_path / "ada-path"
    peft.mkdir()
    (peft / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": "unsloth/Qwen3.5-4B"}),
        encoding = "utf-8",
    )
    loaded = []

    class FakeBackend:
        active_model_name = "qwen"
        models = {"qwen": {"model": object(), "active_adapter": None}}

        def load_adapter(self, base, path, name):
            loaded.append((base, path, name))
            return True

    name, snap = prepare_sidecar_adapter(FakeBackend(), str(peft))
    assert name == "ada-path"
    assert loaded == [("qwen", str(peft), "ada-path")]
    assert snap is not None
