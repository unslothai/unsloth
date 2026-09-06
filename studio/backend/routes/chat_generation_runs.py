# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Authenticated API for resumable, server-owned Studio chat generations."""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from auth.authentication import get_current_subject
from auth import policy
from state import active_generations
from utils.account_context import current_account, current_account_id, run_as
from core.inference.llama_keepwarm import inference_lifecycle_gate
from models.inference import ChatCompletionRequest
from storage import chat_generation_runs_db as db
from utils.api_errors import safe_validation_errors

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
    # A media turn has no replayable transcript and its payload persists verbatim, so a base64 blob would live in
    # request_json for the life of the thread.
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


def cancel_account_run(request: Request, run_id: str, *, supervisor_name: str) -> None:
    """Signal only the caller's registration when supervisor IDs are shared.

    Durable chat producers reserve their event before starting. Research workers
    also observe cancellation/deletion through their account's persisted lease.
    Never stash a bare cancel ID in a multi-account installation: a later request
    from another account can legitimately reuse it.
    """
    if policy.installation_is_multi_user():
        active_generations.cancel_run(run_id, account_id = current_account_id())
        return
    supervisor = getattr(request.app.state, supervisor_name, None)
    if supervisor is not None:
        supervisor.cancel(run_id)
    elif supervisor_name == "chat_generation_supervisor":
        from routes.inference import _cancel_by_cancel_id_or_stash
        active_generations.cancel_run(run_id)
        _cancel_by_cancel_id_or_stash(run_id)


def _require_available_supervisor_run_id(run_id: str) -> None:
    """A legacy supervisor keys tasks by bare ID; refuse a foreign active slot."""
    if policy.installation_is_multi_user():
        for entry in active_generations.snapshot():
            if entry["run_id"] == run_id:
                policy.require_account_scope(entry.get("account_id"))


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
    sanitized = _sanitize_request(payload)
    # Serialize the off-loop commit with model lifecycle work, so a run is registered either before the gate opens or
    # after an unload/swap, never mid-swap.
    async with inference_lifecycle_gate():
        _require_available_supervisor_run_id(payload.runId)
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
    return {"runs": db.list_active(thread_id)}


@router.get("/{run_id}")
def get_chat_generation_run(run_id: str, current_subject: str = Depends(get_current_subject)):
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
    if run["status"] in {"cancelling", "cancelled"} and (
        getattr(request.app.state, "chat_generation_supervisor", None) is not None
        or policy.installation_is_multi_user()
    ):
        cancel_account_run(request, run_id, supervisor_name = "chat_generation_supervisor")
    return run


@router.post("/{run_id}/events")
async def chat_generation_events(
    run_id: str,
    request: Request,
    after: int | None = Query(None, ge = 0, le = _SQLITE_MAX_INTEGER),
    last_event_id: str | None = Header(None, alias = "Last-Event-ID"),
    current_subject: str = Depends(get_current_subject),
):
    _require_run(run_id)
    cursor = _event_cursor(after, last_event_id)
    wait_for_events = db.wait_for_events
    if policy.installation_is_multi_user():
        # run_in_executor does not copy ContextVars, unlike asyncio.to_thread.
        wait_for_events = partial(run_as, current_account(), db.wait_for_events)

    async def stream():
        nonlocal cursor
        loop = asyncio.get_running_loop()
        # A reconnect to an already-settled run has nothing to replay, and wait_for_events would hold it for the full
        # timeout and tie up an event-wait worker.
        opening = await asyncio.to_thread(db.get_run, run_id)
        if opening is None:
            return
        if opening["status"] in db.TERMINAL_STATUSES and cursor >= int(opening["lastEventSeq"]):
            return
        while True:
            events = await loop.run_in_executor(
                _EVENT_WAIT_EXECUTOR,
                wait_for_events,
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
                # Carries the run's progress stamp, which the lease renewals move.
                # A bare keep-alive proves only that the CONNECTION is healthy, so a follower rearming its no-progress
                # deadline on one could never settle a wedged run while the socket stayed up, the one case that fallback
                # exists for.
                # Comment framing, so _SSEDecoder still drops it and no client parsing it as an event is affected.
                yield f": keep-alive {int(snapshot['updatedAt'])}\n\n"

    return StreamingResponse(
        stream(),
        media_type = "text/event-stream",
        headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
