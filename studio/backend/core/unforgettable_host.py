# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Thin Studio adapter for the Apache ``unforgettable`` package.

This file is the AGPL side of the Host protocol: sandbox paths, inner generate,
and the virtual model id. Stream rewrite lives in ``unforgettable_stream``.
Policy, schema, and clone logic stay in ``unforgettable/``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from contextvars import ContextVar, copy_context
from pathlib import Path
from typing import Any, Callable, Optional

from unforgettable import VIRTUAL_MODEL_ID, inner_model_id, is_virtual_model
from unforgettable.sidecar.peft import is_peft_adapter_dir, peft_adapter_name
from unforgettable.supervisor import coerce_filter_flag, coerce_planner_flag
from unforgettable.host import (
    EXTRACT_MAX_TOKENS,
    RUN_ACTION_CLIP,
    RUN_ACTION_NAMES,
    RUN_ACTION_TIMEOUT_SEC,
    SUPERVISE_MAX_TOKENS,
    GenerateRequest,
    GenerateResult,
    Host,
)
from unforgettable.loop.context import EpisodeRequest
from unforgettable.loop.episode import run as run_episode
from unforgettable.loop.runtime import current_traces

from core.unforgettable_stream import (
    _as_sse_bytes,
    _choice_text,
    _emit_on_chunk,
    _forward_inner_stream,
    _parse_sse_json,
    _response_payload,
    _rewrite_inner_frame,
)

_INNER: ContextVar[bool] = ContextVar("unforgettable_inner_generate", default = False)
_log = logging.getLogger(__name__)

# Hold the first confirm keepalive back so the Allow / Deny card flushes alone.
_TOOL_APPROVAL_FLUSH_DELAY_S = 0.05


def in_inner_generate() -> bool:
    return bool(_INNER.get())


def catalog_entry(created: int | None = None, *, loaded: bool = False) -> dict:
    return {
        "id": VIRTUAL_MODEL_ID,
        "object": "model",
        "created": int(created or time.time()),
        "owned_by": "unforgettable",
        "loaded": loaded,
    }


def snapshot_adapter_state(backend) -> Optional[dict]:
    """Record which adapter was live so a sidecar attach can be undone."""
    from peft import PeftModel, PeftModelForCausalLM

    base = backend.active_model_name
    if not base or base not in backend.models:
        return None
    model = backend.models[base].get("model")
    if model is None:
        return None
    base_tuner = getattr(model, "base_model", None)
    return {
        "base": base,
        "was_peft": isinstance(model, (PeftModel, PeftModelForCausalLM)),
        "active": backend.models[base].get("active_adapter"),
        "disabled": bool(getattr(base_tuner, "_disable_adapters", False)),
    }


def restore_sidecar_adapter(backend, snap: Optional[dict]) -> None:
    from peft import PeftModel, PeftModelForCausalLM

    if not snap:
        return
    base = snap.get("base") or backend.active_model_name
    if not base or base not in backend.models:
        return
    model = backend.models[base].get("model")
    if model is None or not isinstance(model, (PeftModel, PeftModelForCausalLM)):
        return
    if not snap.get("was_peft") or snap.get("disabled"):
        model.base_model.disable_adapter_layers()
        return
    model.base_model.enable_adapter_layers()
    previous = snap.get("active")
    if previous:
        backend.set_active_adapter(base, previous)


def prepare_sidecar_adapter(backend, use_adapter):
    """Load a PEFT dir onto the live model and return a named adapter for upstream apply.

    Returns ``(use_adapter, snapshot)``. Snapshot is None when this is not a
    sidecar path, so the caller can skip restore. A failed load returns
    ``(None, snapshot)`` so generation keeps the previous adapter.
    """
    if not isinstance(use_adapter, str) or not is_peft_adapter_dir(use_adapter):
        return use_adapter, None
    snap = snapshot_adapter_state(backend)
    name = peft_adapter_name(use_adapter)
    base = backend.active_model_name
    if not base or not backend.load_adapter(base, use_adapter, name):
        _log.warning(
            "use_adapter path '%s' did not load; inner generate stays on the base",
            use_adapter,
        )
        return None, snap
    return name, snap


_MESSAGE_EXTRA_KEYS = ("name", "tool_call_id", "tool_calls")


def _messages_as_dicts(messages) -> list[dict]:
    out = []
    for message in messages:
        if isinstance(message, dict):
            item = dict(message)
            out.append(item)
            continue
        item = {
            "role": getattr(message, "role", None),
            "content": getattr(message, "content", None),
        }
        for key in _MESSAGE_EXTRA_KEYS:
            value = getattr(message, key, None)
            if value is not None:
                item[key] = value
        out.append(item)
    return out


def _as_chat_messages(messages: list[dict]):
    from models.inference import ChatMessage
    return [ChatMessage(role = m["role"], content = m.get("content")) for m in messages]


def union_unforgettable_enabled_tools(enabled_tools):
    # A list is a filter; None would re-enable omitted pills (web_search / render_html).
    if enabled_tools is None:
        return None
    from unforgettable.tools.specs import CONTACT_TOOL_NAMES, MEMORY_TOOL_NAMES
    return list(dict.fromkeys(list(enabled_tools) + list(MEMORY_TOOL_NAMES | CONTACT_TOOL_NAMES)))


def _clip_action_args(name: str, arguments: dict | None) -> dict:
    args = arguments or {}
    if name == "python":
        return {"code": (args.get("code") or "")[:RUN_ACTION_CLIP]}
    return {"command": (args.get("command") or "")[:RUN_ACTION_CLIP]}


class StudioHost:
    """Studio implementation of ``unforgettable.host.Host``."""

    def __init__(
        self, payload, request, current_subject: str, inner: Callable, inner_model: str
    ) -> None:
        self.payload = payload
        self.request = request
        self.current_subject = current_subject
        self.inner = inner
        self.inner_model = inner_model
        self._sim_n = 0
        self.cancel_event = threading.Event()

    def memory_db_path(self) -> Path:
        from utils.paths import studio_root
        return studio_root() / "memory" / "memory.db"

    def world_session_id(self, request) -> str:
        sid = getattr(self.payload, "session_id", None)
        if sid:
            return sid
        tid = getattr(self.payload, "thread_id", None)
        if tid:
            return tid
        if getattr(request, "world_session_id", None):
            return request.world_session_id
        return "default"

    def create_sim_session(self, episode_id: str) -> str:
        from core.inference.tools import get_sandbox_workdir

        self._sim_n += 1
        sid = f"sim-{episode_id[:8]}-{self._sim_n}"
        get_sandbox_workdir(sid)
        return sid

    def sandbox_path(self, session_id: str) -> Path:
        from core.inference.tools import get_sandbox_workdir
        return Path(get_sandbox_workdir(session_id))

    def remove_sim_session(self, session_id: str) -> None:
        from core.inference.tools import remove_session_sandbox
        remove_session_sandbox(session_id, delete_files = True)

    async def generate(self, req: GenerateRequest) -> GenerateResult:
        payload = self.payload.model_copy(deep = True)
        payload.model = req.inner_model or self.inner_model or "default"
        payload.session_id = req.session_id
        if req.thread_id:
            payload.thread_id = req.thread_id
        payload.enable_tools = True
        payload.enabled_tools = union_unforgettable_enabled_tools(payload.enabled_tools)
        payload.messages = _as_chat_messages(req.messages)
        want_stream = req.on_chunk is not None
        payload.stream = want_stream
        # Sidecar C: a PEFT dir on GenerateRequest becomes the existing
        # use_adapter string. Fake adapter dirs and GGUF inners fail open.
        # A GGUF LoRA is a serve artifact: load it with llama extra args --lora
        # and reload. Do not convert or restart the server here.
        if req.adapter_path and is_peft_adapter_dir(req.adapter_path):
            payload.use_adapter = req.adapter_path
        elif req.gguf_adapter_path:
            _log.warning(
                "GGUF LoRA is at '%s'; add --lora <path> to the GGUF load and reload. "
                "Mid-chat attach is PEFT-only.",
                req.gguf_adapter_path,
            )
        before = len(current_traces())
        token = _INNER.set(True)
        try:
            resp = await self.inner(payload, self.request, self.current_subject)
        finally:
            _INNER.reset(token)
        if want_stream:
            text = await _forward_inner_stream(resp, req.on_chunk)
        else:
            text = _choice_text(_response_payload(resp))
        return GenerateResult(text = text, tool_traces = current_traces()[before:])

    async def run_action(
        self,
        session_id,
        name,
        arguments,
        *,
        timeout = None,
        on_chunk = None,
    ) -> str:
        from core.inference.tool_stream_exec import TOOL_HEARTBEAT_INTERVAL_S
        from core.inference.tools import execute_tool

        if name not in RUN_ACTION_NAMES:
            return f"Error: run_action supports python|terminal only, got {name!r}"
        effective = RUN_ACTION_TIMEOUT_SEC if timeout is None else timeout
        tool_call_id = f"rims-action-{uuid.uuid4().hex[:16]}"
        if on_chunk is not None:
            start_event = {
                "type": "tool_start",
                "tool_name": name,
                "tool_call_id": tool_call_id,
                "arguments": _clip_action_args(name, arguments),
                "approval_id": "",
                "awaiting_confirmation": False,
            }
            await _emit_on_chunk(
                on_chunk,
                _as_sse_bytes("data: " + json.dumps(start_event, separators = (",", ":"))),
            )
        ctx = copy_context()

        def _run_action():
            return execute_tool(
                name,
                arguments or {},
                session_id = session_id,
                timeout = effective,
                cancel_event = self.cancel_event,
            )

        work = asyncio.create_task(asyncio.to_thread(ctx.run, _run_action))
        try:
            while True:
                done, _ = await asyncio.wait({work}, timeout = TOOL_HEARTBEAT_INTERVAL_S)
                if done:
                    break
                if on_chunk is not None:
                    await _emit_on_chunk(on_chunk, b": keep-alive\n\n")
            result = work.result()
        finally:
            if not work.done():
                work.cancel()
        if on_chunk is not None:
            end_event = {
                "type": "tool_end",
                "tool_name": name,
                "tool_call_id": tool_call_id,
                "result": (result or "")[:RUN_ACTION_CLIP],
            }
            await _emit_on_chunk(
                on_chunk,
                _as_sse_bytes("data: " + json.dumps(end_event, separators = (",", ":"))),
            )
        return result

    async def confirm(
        self,
        prompt: str,
        *,
        kind: str = "retry_world",
        on_chunk = None,
        session_id = None,
    ) -> bool:
        from core.inference.tool_stream_exec import TOOL_HEARTBEAT_INTERVAL_S
        from state.tool_approvals import (
            begin_tool_decision,
            new_approval_id,
            wait_tool_decision,
        )

        if on_chunk is None:
            return False
        if self.cancel_event.is_set():
            return False
        approval_id = new_approval_id()
        slot = begin_tool_decision(session_id or self.world_session_id(None), approval_id)
        start_event = {
            "type": "tool_start",
            "tool_name": "rims_retry_world",
            "tool_call_id": approval_id,
            "arguments": {"prompt": prompt, "kind": kind},
            "approval_id": approval_id,
            "awaiting_confirmation": True,
        }
        await _emit_on_chunk(
            on_chunk,
            _as_sse_bytes("data: " + json.dumps(start_event, separators = (",", ":"))),
        )
        waiter = asyncio.create_task(
            asyncio.to_thread(
                wait_tool_decision,
                slot,
                approval_id,
                self.cancel_event,
            )
        )
        verdict = "deny"
        try:
            done, _ = await asyncio.wait({waiter}, timeout = _TOOL_APPROVAL_FLUSH_DELAY_S)
            while not done:
                await _emit_on_chunk(on_chunk, b": keep-alive\n\n")
                done, _ = await asyncio.wait({waiter}, timeout = TOOL_HEARTBEAT_INTERVAL_S)
            verdict = waiter.result()
        finally:
            if not waiter.done():
                waiter.cancel()
        end_event = {
            "type": "tool_end",
            "tool_name": "rims_retry_world",
            "tool_call_id": approval_id,
            "result": "allowed" if verdict == "allow" else "denied",
        }
        await _emit_on_chunk(
            on_chunk,
            _as_sse_bytes("data: " + json.dumps(end_event, separators = (",", ":"))),
        )
        return verdict == "allow"

    async def _one_shot(
        self, messages: list[dict[str, Any]], *, model: str, max_tokens: int
    ) -> str:
        # Pin both token fields: Studio prefers max_completion_tokens when set.
        # Strip leftover tool surfaces; tools_force_disabled beats CLI --enable-tools.
        from state.tool_policy import tools_force_disabled

        payload = self.payload.model_copy(deep = True)
        payload.model = model or self.inner_model or "default"
        payload.stream = False
        payload.enable_tools = False
        payload.mcp_enabled = False
        payload.tools = None
        payload.tool_choice = "none"
        payload.max_tokens = max_tokens
        payload.max_completion_tokens = max_tokens
        payload.messages = _as_chat_messages(messages)
        token = _INNER.set(True)
        try:
            with tools_force_disabled():
                resp = await self.inner(payload, self.request, self.current_subject)
        finally:
            _INNER.reset(token)
        return _choice_text(_response_payload(resp))

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = EXTRACT_MAX_TOKENS,
    ) -> str:
        return await self._one_shot(
            messages,
            model = self.inner_model or "default",
            max_tokens = max_tokens,
        )

    async def supervise(
        self,
        purpose: str,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        max_tokens: int = SUPERVISE_MAX_TOKENS,
    ) -> str:
        chosen = model or self._supervisor_model(purpose) or self.inner_model or "default"
        return await self._one_shot(messages, model = chosen, max_tokens = max_tokens)

    def _supervisor_model(self, purpose: str) -> str | None:
        import os

        if purpose == "plan":
            return (
                getattr(self.payload, "planner_model", None)
                or os.environ.get("UNFORGETTABLE_PLANNER_MODEL")
                or None
            )
        if purpose == "filter":
            return (
                getattr(self.payload, "filter_model", None)
                or os.environ.get("UNFORGETTABLE_FILTER_MODEL")
                or None
            )
        if purpose == "judge":
            return (
                getattr(self.payload, "judge_model", None)
                or os.environ.get("UNFORGETTABLE_JUDGE_MODEL")
                or None
            )
        if purpose in {"vote", "mine"}:
            return (
                getattr(self.payload, "voter_model", None)
                or os.environ.get("UNFORGETTABLE_VOTER_MODEL")
                or None
            )
        return None


def _planner_from_payload(payload) -> str | None:
    flag = getattr(payload, "planner", None)
    if flag is None:
        flag = os.environ.get("UNFORGETTABLE_PLANNER")
    return coerce_planner_flag(flag)


def _filter_from_payload(payload) -> str | None:
    flag = getattr(payload, "filter", None)
    if flag is None:
        flag = os.environ.get("UNFORGETTABLE_FILTER")
    return coerce_filter_flag(flag)


async def handle_chat_completions(payload, request, current_subject: str, inner: Callable):
    """Run the middle wheel, then look like a normal chat completion."""
    from routes.inference import _sse_streaming_response

    model = inner_model_id(getattr(payload, "model", None))
    host = StudioHost(payload, request, current_subject, inner, model)
    episode = EpisodeRequest(
        messages = _messages_as_dicts(payload.messages),
        world_session_id = payload.session_id or payload.thread_id,
        thread_id = payload.thread_id,
        stream = bool(payload.stream),
        inner_model = model,
        stakes = getattr(payload, "stakes", None),
        test_command = getattr(payload, "test_command", None),
        confirm_retry = getattr(payload, "confirm_retry", None),
        permission_mode = getattr(payload, "permission_mode", None),
        max_clones = getattr(payload, "max_clones", None),
        max_sim_turns = getattr(payload, "max_sim_turns", None),
        adapter_id = getattr(payload, "adapter_id", None),
        skip_standing = bool(getattr(payload, "skip_standing", False)),
        planner = _planner_from_payload(payload),
        planner_model = (
            getattr(payload, "planner_model", None)
            or os.environ.get("UNFORGETTABLE_PLANNER_MODEL")
            or None
        ),
        filter = _filter_from_payload(payload),
        filter_model = (
            getattr(payload, "filter_model", None)
            or os.environ.get("UNFORGETTABLE_FILTER_MODEL")
            or None
        ),
        judge_model = (
            getattr(payload, "judge_model", None)
            or os.environ.get("UNFORGETTABLE_JUDGE_MODEL")
            or None
        ),
        user_label = getattr(payload, "user_label", None),
        twin_plugin = getattr(payload, "twin_plugin", None) or os.environ.get("UNFORGETTABLE_TWIN"),
    )
    if payload.stream:
        queue: asyncio.Queue = asyncio.Queue()

        async def on_chunk(data: bytes) -> None:
            await queue.put(data)

        episode.on_chunk = on_chunk

        async def produce() -> None:
            try:
                await run_episode(host, episode)
            except Exception as exc:
                err = {
                    "error": {
                        "message": str(exc),
                        "type": "unforgettable_error",
                    }
                }
                await queue.put(f"data: {json.dumps(err)}\n\n".encode("utf-8"))
            finally:
                await queue.put(b"data: [DONE]\n\n")
                await queue.put(None)

        task = asyncio.create_task(produce())

        async def gen():
            try:
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    yield item
            finally:
                host.cancel_event.set()
                await task

        return _sse_streaming_response(gen())

    outcome = await run_episode(host, episode)
    return {
        "id": f"chatcmpl-unforgettable-{outcome.state.episode_id[:8]}",
        "object": "chat.completion",
        "model": VIRTUAL_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": outcome.text},
                "finish_reason": "stop",
            }
        ],
    }


# Re-export for route checks without a second import path.
__all__ = [
    "Host",
    "StudioHost",
    "VIRTUAL_MODEL_ID",
    "catalog_entry",
    "handle_chat_completions",
    "in_inner_generate",
    "inner_model_id",
    "is_peft_adapter_dir",
    "is_virtual_model",
    "peft_adapter_name",
    "prepare_sidecar_adapter",
    "restore_sidecar_adapter",
    "union_unforgettable_enabled_tools",
]
