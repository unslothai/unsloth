# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Thin Studio adapter for the Apache ``unforgettable`` package.

This file is the AGPL side of the Host protocol: sandbox paths, inner generate,
and the virtual model id. Policy, schema, and clone logic stay in ``unforgettable/``.
"""

from __future__ import annotations

import asyncio
import inspect
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
from unforgettable.supervisor import coerce_planner_flag
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

_INNER: ContextVar[bool] = ContextVar("unforgettable_inner_generate", default = False)
_log = logging.getLogger(__name__)

# Fallback chunk id when the inner path is not a stream we can drain.
_BUFFERED_STREAM_CHUNK_ID = "chatcmpl-unforgettable"

# Hold the first confirm keepalive back so the Allow / Deny card flushes alone.
_TOOL_APPROVAL_FLUSH_DELAY_S = 0.05


def in_inner_generate() -> bool:
    return bool(_INNER.get())


def catalog_entry(created: int | None = None) -> dict:
    return {
        "id": VIRTUAL_MODEL_ID,
        "object": "model",
        "created": int(created or time.time()),
        "owned_by": "unforgettable",
        "loaded": True,
    }


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


def _response_payload(resp: Any) -> dict:
    if isinstance(resp, dict):
        return resp
    body = getattr(resp, "body", None)
    if body:
        if isinstance(body, (bytes, bytearray)):
            return json.loads(body.decode("utf-8"))
        return json.loads(body)
    return {}


def _choice_text(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content")
    return content if isinstance(content, str) else ""


def _sse_data_payload(text: str) -> str:
    parts = []
    for line in text.splitlines():
        if line.startswith("data:"):
            parts.append(line[5:].lstrip())
    return "\n".join(parts).strip()


def union_unforgettable_enabled_tools(enabled_tools):
    # A list is a filter; None would re-enable omitted pills (web_search / render_html).
    if enabled_tools is None:
        return None
    from unforgettable.tools.specs import CONTACT_TOOL_NAMES, MEMORY_TOOL_NAMES
    return list(dict.fromkeys(list(enabled_tools) + list(MEMORY_TOOL_NAMES | CONTACT_TOOL_NAMES)))


def _as_sse_bytes(frame: bytes | str) -> bytes:
    if isinstance(frame, (bytes, bytearray)):
        raw = bytes(frame)
        if not raw.endswith(b"\n\n"):
            raw = raw.rstrip(b"\r\n") + b"\n\n"
        return raw
    text = str(frame).replace("\r\n", "\n")
    if not text.endswith("\n\n"):
        text = text.rstrip("\n") + "\n\n"
    return text.encode("utf-8")


def _parse_sse_json(text: str) -> dict | None:
    payload = _sse_data_payload(text)
    if not payload or payload == "[DONE]":
        return None
    try:
        obj = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def _is_complete_data_event(text: str) -> bool:
    stripped = text.strip()
    if not stripped.startswith("data:"):
        return False
    payload = stripped[5:].strip()
    if payload == "[DONE]":
        return True
    if not payload:
        return False
    try:
        json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _is_openai_delta_chunk(obj: dict) -> bool:
    if obj.get("object") == "chat.completion.chunk":
        return True
    choices = obj.get("choices")
    if not isinstance(choices, list):
        return False
    return any(
        isinstance(choice, dict) and isinstance(choice.get("delta"), dict) for choice in choices
    )


def _openai_delta_content(frame: bytes) -> str:
    text = frame.decode("utf-8", errors = "replace")
    obj = _parse_sse_json(text)
    if not obj or not _is_openai_delta_chunk(obj):
        return ""
    parts = []
    for choice in obj.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        content = delta.get("content")
        if isinstance(content, str):
            parts.append(content)
    return "".join(parts)


def _split_sse_events(buf: str) -> tuple[list[str], str]:
    events: list[str] = []
    while True:
        idx_nn = buf.find("\n\n")
        idx_rr = buf.find("\r\n\r\n")
        candidates = []
        if idx_nn != -1:
            candidates.append((idx_nn, 2))
        if idx_rr != -1:
            candidates.append((idx_rr, 4))
        if not candidates:
            break
        idx, width = min(candidates, key = lambda item: item[0])
        event = buf[:idx]
        buf = buf[idx + width :]
        if event.strip():
            events.append(event)
    if _is_complete_data_event(buf):
        events.append(buf)
        buf = ""
    return events, buf


def _rewrite_inner_frame(frame: bytes | str) -> bytes | None:
    """Rewrite one inner SSE frame for the outer virtual-model stream.

    Returns ``None`` to drop the frame (inner ``[DONE]``). Studio tool frames
    and anything that is not an OpenAI chunk are returned unchanged.
    """
    if isinstance(frame, (bytes, bytearray)):
        original: bytes | None = bytes(frame)
        text = original.decode("utf-8", errors = "replace")
    else:
        original = None
        text = str(frame)

    payload = _sse_data_payload(text)
    # Outer produce() owns the real [DONE] after run() returns.
    if payload == "[DONE]":
        return None

    obj = _parse_sse_json(text)
    if obj is None:
        return _as_sse_bytes(original if original is not None else text)

    # Studio UI frames (tool_start / diffusion_frame / …) stay byte-identical.
    if isinstance(obj.get("type"), str) and obj["type"]:
        return _as_sse_bytes(original if original is not None else text)

    if _is_openai_delta_chunk(obj):
        obj["model"] = VIRTUAL_MODEL_ID
        # A non-null finish_reason would close the desktop chat round between rims.
        for choice in obj.get("choices") or []:
            if isinstance(choice, dict):
                choice["finish_reason"] = None
        return f"data: {json.dumps(obj)}\n\n".encode("utf-8")

    return _as_sse_bytes(original if original is not None else text)


def _clip_action_args(name: str, arguments: dict | None) -> dict:
    args = arguments or {}
    if name == "python":
        return {"code": (args.get("code") or "")[:RUN_ACTION_CLIP]}
    return {"command": (args.get("command") or "")[:RUN_ACTION_CLIP]}


async def _emit_on_chunk(on_chunk: Callable, data: bytes) -> bool:
    """Forward one frame. False means the client is gone; do not keep decoding."""
    try:
        maybe = on_chunk(data)
        if inspect.isawaitable(maybe):
            await maybe
        return True
    except asyncio.CancelledError:
        raise
    except Exception:
        return False


async def _aclose_iterator(iterator: Any) -> None:
    aclose = getattr(iterator, "aclose", None)
    if not callable(aclose):
        return
    try:
        maybe = aclose()
        if inspect.isawaitable(maybe):
            await maybe
    except Exception:
        pass


async def _forward_buffered_choice(resp: Any, on_chunk: Callable) -> str:
    data = _response_payload(resp)
    text = _choice_text(data)
    if text:
        chunk = {
            "id": data.get("id") or _BUFFERED_STREAM_CHUNK_ID,
            "object": "chat.completion.chunk",
            "model": VIRTUAL_MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": text},
                    "finish_reason": None,
                }
            ],
        }
        raw = f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
        await _emit_on_chunk(on_chunk, raw)
    return text


async def _forward_inner_stream(resp: Any, on_chunk: Callable) -> str:
    """Drain the inner Studio stream onto ``on_chunk``. Always aclose()s."""
    iterator = getattr(resp, "body_iterator", None)
    if iterator is None:
        _log.warning(
            "unforgettable inner drain: no body_iterator, falling back to buffered choice text"
        )
        return await _forward_buffered_choice(resp, on_chunk)

    parts: list[str] = []
    buf = ""
    try:
        async for chunk in iterator:
            if isinstance(chunk, (bytes, bytearray)):
                buf += chunk.decode("utf-8", errors = "replace")
            else:
                buf += str(chunk)
            events, buf = _split_sse_events(buf)
            for event in events:
                rewritten = _rewrite_inner_frame(event)
                if rewritten is None:
                    continue
                delta = _openai_delta_content(rewritten)
                if delta:
                    parts.append(delta)
                if not await _emit_on_chunk(on_chunk, rewritten):
                    return "".join(parts)
        leftover = buf.strip()
        if leftover:
            rewritten = _rewrite_inner_frame(leftover)
            if rewritten is not None:
                delta = _openai_delta_content(rewritten)
                if delta:
                    parts.append(delta)
                await _emit_on_chunk(on_chunk, rewritten)
    finally:
        await _aclose_iterator(iterator)
    return "".join(parts)


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
        self, session_id, name, arguments, *, timeout = None, on_chunk = None
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
        waiter = asyncio.create_task(asyncio.to_thread(
            wait_tool_decision,
            slot,
            approval_id,
            self.cancel_event,
        ))
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
        self,
        messages: list[dict[str, Any]],
        *,
        model: str,
        max_tokens: int,
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
        return await self._one_shot(
            messages, model = chosen, max_tokens = max_tokens
        )

    def _supervisor_model(self, purpose: str) -> str | None:
        import os

        if purpose == "plan":
            return (
                getattr(self.payload, "planner_model", None)
                or os.environ.get("UNFORGETTABLE_PLANNER_MODEL")
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
    "is_virtual_model",
    "union_unforgettable_enabled_tools",
]
