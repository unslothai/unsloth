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
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Optional

from unforgettable import VIRTUAL_MODEL_ID, inner_model_id, is_virtual_model
from unforgettable.host import GenerateRequest, GenerateResult, Host
from unforgettable.loop.context import EpisodeRequest
from unforgettable.loop.episode import run as run_episode
from unforgettable.loop.runtime import current_traces

_INNER: ContextVar[bool] = ContextVar("unforgettable_inner_generate", default = False)


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


def _messages_as_dicts(messages) -> list[dict]:
    out = []
    for message in messages:
        role = getattr(message, "role", None) or message.get("role")
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        out.append({"role": role, "content": content})
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


class StudioHost:
    """Studio implementation of ``unforgettable.host.Host``."""

    def __init__(
        self,
        payload,
        request,
        current_subject: str,
        inner: Callable,
        inner_model: str,
    ) -> None:
        self.payload = payload
        self.request = request
        self.current_subject = current_subject
        self.inner = inner
        self.inner_model = inner_model
        self._sim_n = 0

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
        payload.stream = False
        payload.enable_tools = True
        payload.messages = _as_chat_messages(req.messages)
        before = len(current_traces())
        token = _INNER.set(True)
        try:
            resp = await self.inner(payload, self.request, self.current_subject)
        finally:
            _INNER.reset(token)
        data = _response_payload(resp)
        text = _choice_text(data)
        if req.on_chunk and text:
            chunk = {
                "id": data.get("id") or "chatcmpl-unforgettable",
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
            maybe = req.on_chunk(raw)
            if inspect.isawaitable(maybe):
                await maybe
        return GenerateResult(text = text, tool_traces = current_traces()[before:])


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
]
