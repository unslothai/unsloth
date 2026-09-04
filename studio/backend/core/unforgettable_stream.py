# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Inner-stream rewrite for the Unforgettable virtual model.

Studio tool frames stay byte-identical. OpenAI chunks are reminted onto the
virtual model id with ``finish_reason`` nulled so the desktop chat round does
not close between rims. Inner ``[DONE]`` is dropped; the outer produce owns
the real terminator.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import Any, Callable

from unforgettable import VIRTUAL_MODEL_ID

_log = logging.getLogger(__name__)

# Fallback chunk id when the inner path is not a stream we can drain.
_BUFFERED_STREAM_CHUNK_ID = "chatcmpl-unforgettable"


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
