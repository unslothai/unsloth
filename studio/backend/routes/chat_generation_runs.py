# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Authenticated API for resumable, server-owned Studio chat generations."""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from auth.authentication import get_current_subject
from core.inference.llama_keepwarm import inference_lifecycle_gate
from models.inference import ChatCompletionRequest
from storage import chat_generation_runs_db as db
from utils.api_errors import safe_validation_errors
from utils import chat_history_policy

router = APIRouter()
_EVENT_WAIT_EXECUTOR = ThreadPoolExecutor(
    max_workers = 32,
    thread_name_prefix = "chat-generation-events",
)
_SENSITIVE_KEYS = {
    "accesskey",
    "authorization",
    "encryptionkey",
    "password",
    "privatekey",
    "secret",
    "secretkey",
    "signingkey",
    "sshkey",
    "token",
    "apikey",
    "credential",
    "credentials",
    "encryptedapikey",
}
_SENSITIVE_SUFFIXES = (
    "token",
    "secret",
    "password",
    "credential",
    "credentials",
)
_EXTERNAL_ROUTING_FIELDS = {
    "provider_id",
    "provider_type",
    "external_model",
    "encrypted_api_key",
    "provider_base_url",
}
# Attachments the composer sends inline. Durable replay has no representation for them.
_MEDIA_FIELDS = {
    "image_base64",
    "audio_base64",
    "video_base64",
}
_SQLITE_MAX_INTEGER = 9_223_372_036_854_775_807
_ENVELOPE_MAX_DEPTH = 64
_ENVELOPE_MAX_NODES = 20_000
_ENVELOPE_MAX_JSON_CHARS = 1_000_000


class CreateChatGenerationRun(BaseModel):
    model_config = ConfigDict(extra = "forbid")
    runId: str = Field(min_length = 1, max_length = 128, pattern = r"^[A-Za-z0-9_-]+$")
    threadId: str = Field(min_length = 1, max_length = 256)
    userMessageId: str = Field(min_length = 1, max_length = 256)
    assistantMessageId: str = Field(min_length = 1, max_length = 256)
    requestPayload: dict[str, Any]


def _normalized_key(key: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).casefold())


def _is_sensitive_key(key: object) -> bool:
    normalized = _normalized_key(key)
    return normalized in _SENSITIVE_KEYS or normalized.endswith(_SENSITIVE_SUFFIXES)


def _contains_sensitive_json_key_text(value: str) -> bool:
    for match in re.finditer(r'"((?:\\.|[^"\\])*)"\s*:', value):
        try:
            key = json.loads(f'"{match.group(1)}"')
        except (json.JSONDecodeError, TypeError):
            key = match.group(1)
        if _is_sensitive_key(key):
            return True
    return False


def _contains_sensitive_key(value: object) -> bool:
    stack: list[tuple[object, int]] = [(value, 0)]
    nodes = 0
    decoded_chars = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if depth > _ENVELOPE_MAX_DEPTH or nodes > _ENVELOPE_MAX_NODES:
            return True
        if isinstance(item, dict):
            for key, nested in item.items():
                if _is_sensitive_key(key):
                    return True
                stack.append((nested, depth + 1))
        elif isinstance(item, (list, tuple)):
            stack.extend((nested, depth + 1) for nested in item)
        elif isinstance(item, str):
            candidate = item.lstrip()
            if not candidate.startswith(("{", "[", '"')):
                continue
            decoded_chars += len(candidate)
            if decoded_chars > _ENVELOPE_MAX_JSON_CHARS:
                return True
            try:
                decoded = json.loads(candidate)
            except json.JSONDecodeError:
                if _contains_sensitive_json_key_text(candidate):
                    return True
                continue
            except (MemoryError, RecursionError):
                return True
            stack.append((decoded, depth + 1))
    return False


def _sanitize_request(payload: CreateChatGenerationRun) -> dict[str, Any]:
    raw = dict(payload.requestPayload)
    unknown = set(raw) - set(ChatCompletionRequest.model_fields)
    if unknown:
        raise HTTPException(
            status_code = 400,
            detail = f"Unsupported durable request fields: {', '.join(sorted(unknown))}",
        )
    try:
        request = ChatCompletionRequest.model_validate(raw)
    except ValidationError as exc:
        raise HTTPException(
            status_code = 422,
            detail = safe_validation_errors(exc.errors()),
        ) from exc
    # Message content/reasoning are user-authored data, not routing configuration. Scan every
    # other persisted field, including extra message-envelope fields, with the credential policy.
    durable_config = {
        key: value
        for key, value in raw.items()
        if key != "messages" and key not in _EXTERNAL_ROUTING_FIELDS and value is not None
    }
    message_envelopes = [
        {
            key: value
            for key, value in message.items()
            if key not in {"content", "reasoning_content"}
        }
        for message in raw.get("messages", [])
        if isinstance(message, dict)
    ]
    if _contains_sensitive_key(durable_config) or _contains_sensitive_key(message_envelopes):
        raise HTTPException(status_code = 400, detail = "Credentials cannot be persisted")
    if any(raw.get(field) not in (None, "") for field in _EXTERNAL_ROUTING_FIELDS):
        raise HTTPException(
            status_code = 400,
            detail = "Durable chat runs are available only for local inference",
        )
    # Recovery rebuilds text and reasoning deltas. A media turn has neither the same
    # chunk shape nor a replayable transcript, and its payload is persisted verbatim,
    # so a base64 blob would live in request_json for the life of the thread. Studio's
    # own composer already keeps these on the legacy stream; keep the server in step so
    # a stale tab or a direct caller cannot open a path nothing replays.
    if any(raw.get(field) not in (None, "") for field in _MEDIA_FIELDS):
        raise HTTPException(
            status_code = 400,
            detail = "Media chat runs use the legacy streaming path",
        )
    # Recovery currently rebuilds text and reasoning deltas, not server-side tool
    # events. Keep any request whose effective policy can enter the local tool loop
    # on the legacy subscriber-owned stream until those events are replayable.
    from routes.inference import _checkpoint_recall_may_enable_tools, _effective_enable_tools

    request = request.model_copy(update = {"thread_id": payload.threadId})

    if (
        raw.get("tools")
        or request.enable_tools is True
        or bool(request.mcp_enabled)
        or _effective_enable_tools(request) is True
        or _checkpoint_recall_may_enable_tools(request)
    ):
        raise HTTPException(
            status_code = 400,
            detail = "Tool-enabled chat runs use the legacy streaming path",
        )
    if (request.n or 1) != 1:
        raise HTTPException(status_code = 400, detail = "Durable chat runs require n=1")
    sanitized = request.model_dump(mode = "json", exclude_none = True)
    for field in _EXTERNAL_ROUTING_FIELDS:
        sanitized.pop(field, None)
    sanitized["stream"] = True
    sanitized["thread_id"] = payload.threadId
    sanitized["cancel_id"] = payload.runId
    sanitized["generation_run_id"] = payload.runId
    return sanitized


def _require_run(run_id: str) -> dict[str, Any]:
    run = db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code = 404, detail = "Chat generation run not found")
    return run


def _event_cursor(after: int | None, last_event_id: str | None) -> int:
    if after is not None and after > _SQLITE_MAX_INTEGER:
        raise HTTPException(status_code = 400, detail = "Event cursor is too large")
    header_after = 0
    if last_event_id:
        if re.fullmatch(r"[0-9]+", last_event_id, flags = re.ASCII) is None:
            raise HTTPException(status_code = 400, detail = "Last-Event-ID must be an integer")
        max_text = str(_SQLITE_MAX_INTEGER)
        normalized = last_event_id.lstrip("0") or "0"
        if len(normalized) > len(max_text) or (
            len(normalized) == len(max_text) and normalized > max_text
        ):
            raise HTTPException(status_code = 400, detail = "Event cursor is too large")
        header_after = int(normalized)
    return max(after or 0, header_after)


@router.post("", status_code = 202)
async def create_chat_generation_run(
    payload: CreateChatGenerationRun,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    chat_history_policy.require_enabled()
    sanitized = _sanitize_request(payload)
    # Serialize the off-loop commit with model lifecycle work. If create wins,
    # the run is registered before the gate opens; if unload/swap wins, the run
    # is admitted afterward. SSE and unrelated requests stay responsive while
    # SQLite waits on a lock.
    async with inference_lifecycle_gate():
        try:
            run, created = await asyncio.to_thread(
                db.create_run,
                run_id = payload.runId,
                owner_subject = current_subject,
                thread_id = payload.threadId,
                user_message_id = payload.userMessageId,
                assistant_message_id = payload.assistantMessageId,
                request_payload = sanitized,
            )
        except db.ChatGenerationConflictError as exc:
            raise HTTPException(status_code = 409, detail = str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code = 404, detail = "Thread not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code = 400, detail = str(exc)) from exc
        except sqlite3.IntegrityError as exc:
            raise HTTPException(status_code = 409, detail = "Generation run conflicts") from exc
        supervisor = getattr(request.app.state, "chat_generation_supervisor", None)
        if supervisor is not None and run["status"] == "queued":
            supervisor.start(
                run["id"],
                thread_id = run["threadId"],
                model = run["requestPayload"].get("model"),
            )
    return {**run, "created": created}


@router.get("/active")
def active_chat_generation_runs(
    thread_id: str = Query(alias = "threadId"), current_subject: str = Depends(get_current_subject)
):
    if chat_history_policy.disabled():
        return {"runs": []}
    return {"runs": db.list_active(thread_id)}


@router.get("/{run_id}")
def get_chat_generation_run(run_id: str, current_subject: str = Depends(get_current_subject)):
    if chat_history_policy.disabled():
        raise HTTPException(status_code = 404, detail = "Chat generation run not found")
    return _require_run(run_id)


@router.post("/{run_id}/cancel")
def cancel_chat_generation_run(
    run_id: str,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    _require_run(run_id)
    run = db.request_cancel(run_id)
    if run is None:
        raise HTTPException(status_code = 404, detail = "Chat generation run not found")
    supervisor = getattr(request.app.state, "chat_generation_supervisor", None)
    if supervisor is not None and run["status"] in {"cancelling", "cancelled"}:
        supervisor.cancel(run_id)
    if chat_history_policy.disabled():
        return {"id": run_id, "status": run["status"]}
    return run


@router.post("/{run_id}/events")
async def chat_generation_events(
    run_id: str,
    request: Request,
    after: int | None = Query(None, ge = 0, le = _SQLITE_MAX_INTEGER),
    last_event_id: str | None = Header(None, alias = "Last-Event-ID"),
    current_subject: str = Depends(get_current_subject),
):
    if chat_history_policy.disabled():
        raise HTTPException(status_code = 404, detail = "Chat generation run not found")
    _require_run(run_id)
    cursor = _event_cursor(after, last_event_id)

    async def stream():
        nonlocal cursor
        loop = asyncio.get_running_loop()
        # A client that reconnects already caught up on a settled run has nothing to replay,
        # and wait_for_events would hold it for the full timeout: the finished answer reads
        # as still generating and an event-wait worker is tied up meanwhile. Only the first
        # wait needs this guard, since every later pass already returns on the same test
        # against the snapshot it read after waiting.
        opening = await asyncio.to_thread(db.get_run, run_id)
        if opening is None:
            return
        if opening["status"] in db.TERMINAL_STATUSES and cursor >= int(opening["lastEventSeq"]):
            return
        while True:
            events = await loop.run_in_executor(
                _EVENT_WAIT_EXECUTOR,
                db.wait_for_events,
                run_id,
                cursor,
                15,
            )
            snapshot = await asyncio.to_thread(db.get_run, run_id)
            if snapshot is None:
                return
            for event in events:
                cursor = int(event["seq"])
                data = {
                    "seq": cursor,
                    "type": event["type"],
                    "payload": event["payload"],
                    "createdAt": event["createdAt"],
                }
                if event["type"] != "chunk":
                    data["run"] = snapshot
                encoded = json.dumps(data, ensure_ascii = False, separators = (",", ":"))
                yield f"id: {cursor}\nevent: {event['type']}\ndata: {encoded}\n\n"
            if snapshot["status"] in db.TERMINAL_STATUSES and cursor >= int(
                snapshot["lastEventSeq"]
            ):
                return
            if await request.is_disconnected():
                return
            if not events:
                yield ": keep-alive\n\n"

    return StreamingResponse(
        stream(),
        media_type = "text/event-stream",
        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
