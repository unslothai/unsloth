# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Chat history API routes backed by studio.db.
"""

import logging
from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from auth.authentication import get_current_subject
from storage.studio_db import (
    ChatMessageThreadMismatch,
    clear_chat_history,
    count_chat_threads,
    delete_chat_threads,
    get_chat_thread,
    get_chat_message,
    list_chat_settings,
    list_chat_messages,
    list_chat_messages_for_threads,
    list_chat_threads,
    sync_chat_messages,
    update_chat_thread,
    upsert_chat_message,
    upsert_chat_settings_merge,
    upsert_chat_thread,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatThread(BaseModel):
    id: str
    title: str = "New Chat"
    modelType: Literal["base", "lora", "model1", "model2"]
    modelId: str = ""
    pairId: Optional[str] = None
    archived: bool = False
    createdAt: int
    openaiCodeExecContainerId: Optional[str] = None
    anthropicCodeExecContainerId: Optional[str] = None


class ChatThreadPatch(BaseModel):
    title: Optional[str] = None
    modelType: Optional[Literal["base", "lora", "model1", "model2"]] = None
    modelId: Optional[str] = None
    pairId: Optional[str] = None
    archived: Optional[bool] = None
    createdAt: Optional[int] = None
    openaiCodeExecContainerId: Optional[str] = None
    anthropicCodeExecContainerId: Optional[str] = None


class ChatMessage(BaseModel):
    id: str
    threadId: str
    parentId: Optional[str] = None
    role: str
    content: Any = Field(default_factory = list)
    attachments: Optional[Any] = None
    metadata: Optional[dict[str, Any]] = None
    createdAt: int


class ChatThreadListResponse(BaseModel):
    threads: list[ChatThread]


class ChatMessageListResponse(BaseModel):
    messages: list[ChatMessage]


class ChatMessageSyncRequest(BaseModel):
    messages: list[ChatMessage]
    pruneMissing: bool = False


class ChatDeleteRequest(BaseModel):
    ids: list[str]


class ChatMessagesBatchRequest(BaseModel):
    thread_ids: list[str] = Field(default_factory = list)


class ChatMessagesBatchResponse(BaseModel):
    """{thread_id -> messages[]} for many threads in one HTTP call."""

    threads: dict[str, list[ChatMessage]]


class ChatCountResponse(BaseModel):
    count: int


class ChatExportResponse(BaseModel):
    exportedAt: str
    version: int
    threadCount: int
    threads: list[ChatThread]
    messages: list[ChatMessage]


class ChatInferenceSettings(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    temperature: Optional[float] = None
    topP: Optional[float] = None
    topK: Optional[float] = None
    minP: Optional[float] = None
    repetitionPenalty: Optional[float] = None
    presencePenalty: Optional[float] = None
    maxSeqLength: Optional[float] = None
    maxTokens: Optional[float] = None
    systemPrompt: Optional[str] = None
    trustRemoteCode: Optional[bool] = None


class ChatPreset(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    name: str
    params: ChatInferenceSettings


class ChatSettingsPayload(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    inferenceParams: Optional[ChatInferenceSettings] = None
    customPresets: Optional[list[ChatPreset]] = None
    activePreset: Optional[str] = None
    activePresetSource: Optional[Literal["builtin-default", "custom", "modified"]] = (
        None
    )
    autoTitle: Optional[bool] = None
    reasoningEffort: Optional[
        Literal["none", "minimal", "low", "medium", "high", "max", "xhigh"]
    ] = None
    preserveThinking: Optional[bool] = None
    autoHealToolCalls: Optional[bool] = None
    maxToolCallsPerMessage: Optional[int] = Field(default = None, ge = 1)
    toolCallTimeout: Optional[int] = Field(default = None, ge = 1)


class ChatSettingsResponse(BaseModel):
    settings: dict[str, Any]


@router.get("/threads", response_model = ChatThreadListResponse)
async def list_threads(
    model_type: Optional[str] = Query(None),
    pair_id: Optional[str] = Query(None),
    include_archived: bool = Query(True),
    current_subject: str = Depends(get_current_subject),
):
    threads = list_chat_threads(
        subject = current_subject,
        model_type = model_type,
        pair_id = pair_id,
        include_archived = include_archived,
    )
    return ChatThreadListResponse(threads = [ChatThread(**t) for t in threads])


@router.post("/threads", response_model = ChatThread)
async def save_thread(
    payload: ChatThread,
    current_subject: str = Depends(get_current_subject),
):
    return ChatThread(
        **upsert_chat_thread(payload.model_dump(), subject = current_subject)
    )


@router.get("/threads/{thread_id}", response_model = ChatThread)
async def get_thread(
    thread_id: str,
    current_subject: str = Depends(get_current_subject),
):
    thread = get_chat_thread(thread_id, subject = current_subject)
    if thread is None:
        raise HTTPException(status_code = 404, detail = f"Thread {thread_id} not found")
    return ChatThread(**thread)


@router.patch("/threads/{thread_id}", response_model = ChatThread)
async def patch_thread(
    thread_id: str,
    payload: ChatThreadPatch,
    current_subject: str = Depends(get_current_subject),
):
    patch = payload.model_dump(exclude_unset = True)
    for field in ("title", "modelType", "modelId", "archived", "createdAt"):
        if field in patch and patch[field] is None:
            raise HTTPException(status_code = 400, detail = f"{field} cannot be null")
    thread = update_chat_thread(
        thread_id,
        patch,
        subject = current_subject,
    )
    if thread is None:
        raise HTTPException(status_code = 404, detail = f"Thread {thread_id} not found")
    return ChatThread(**thread)


@router.delete("/threads")
async def delete_threads(
    payload: ChatDeleteRequest,
    current_subject: str = Depends(get_current_subject),
):
    deleted = delete_chat_threads(payload.ids, subject = current_subject)
    return {"status": "deleted", "count": deleted}


@router.get("/threads/{thread_id}/messages", response_model = ChatMessageListResponse)
async def get_thread_messages(
    thread_id: str,
    current_subject: str = Depends(get_current_subject),
):
    if get_chat_thread(thread_id, subject = current_subject) is None:
        raise HTTPException(status_code = 404, detail = f"Thread {thread_id} not found")
    return ChatMessageListResponse(
        messages = [
            ChatMessage(**m)
            for m in list_chat_messages(thread_id, subject = current_subject)
        ]
    )


@router.get("/threads/{thread_id}/messages/{message_id}", response_model = ChatMessage)
async def get_thread_message(
    thread_id: str,
    message_id: str,
    current_subject: str = Depends(get_current_subject),
):
    if get_chat_thread(thread_id, subject = current_subject) is None:
        raise HTTPException(status_code = 404, detail = f"Thread {thread_id} not found")
    message = get_chat_message(thread_id, message_id, subject = current_subject)
    if message is None:
        raise HTTPException(status_code = 404, detail = f"Message {message_id} not found")
    return ChatMessage(**message)


@router.put("/threads/{thread_id}/messages/{message_id}", response_model = ChatMessage)
async def save_thread_message(
    thread_id: str,
    message_id: str,
    payload: ChatMessage,
    current_subject: str = Depends(get_current_subject),
):
    if thread_id != payload.threadId or message_id != payload.id:
        raise HTTPException(status_code = 400, detail = "Message id mismatch")
    if get_chat_thread(thread_id, subject = current_subject) is None:
        raise HTTPException(status_code = 404, detail = f"Thread {thread_id} not found")
    try:
        message = upsert_chat_message(payload.model_dump(), subject = current_subject)
    except ChatMessageThreadMismatch as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    return ChatMessage(**message)


@router.put("/threads/{thread_id}/messages", response_model = ChatMessageListResponse)
async def replace_thread_messages(
    thread_id: str,
    payload: ChatMessageSyncRequest,
    current_subject: str = Depends(get_current_subject),
):
    if get_chat_thread(thread_id, subject = current_subject) is None:
        raise HTTPException(status_code = 404, detail = f"Thread {thread_id} not found")
    # Reject mismatched body threadId instead of silently rewriting.
    for message in payload.messages:
        if message.threadId != thread_id:
            raise HTTPException(
                status_code = 400,
                detail = (
                    f"message {message.id!r} threadId={message.threadId!r} "
                    f"does not match URL thread {thread_id!r}"
                ),
            )
    messages = [message.model_dump() for message in payload.messages]
    try:
        rows = sync_chat_messages(
            thread_id,
            messages,
            subject = current_subject,
            prune_missing = payload.pruneMissing,
        )
    except ChatMessageThreadMismatch as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    return ChatMessageListResponse(messages = [ChatMessage(**m) for m in rows])


@router.post("/messages:batch", response_model = ChatMessagesBatchResponse)
async def batch_get_messages(
    payload: ChatMessagesBatchRequest,
    current_subject: str = Depends(get_current_subject),
):
    """Batched message fetch: many threads in one HTTP call.

    Subject-scoped. Unknown ids return an empty list (not 404) so the
    sidebar/search caller can rebuild atomically.
    """
    if not payload.thread_ids:
        return ChatMessagesBatchResponse(threads = {})
    rows = list_chat_messages_for_threads(payload.thread_ids, subject = current_subject)
    bucket: dict[str, list[ChatMessage]] = {tid: [] for tid in payload.thread_ids}
    for row in rows:
        thread_id = row["threadId"]
        if thread_id in bucket:
            bucket[thread_id].append(ChatMessage(**row))
    return ChatMessagesBatchResponse(threads = bucket)


@router.get("/count", response_model = ChatCountResponse)
async def count_threads(current_subject: str = Depends(get_current_subject)):
    return ChatCountResponse(count = count_chat_threads(subject = current_subject))


@router.delete("")
async def clear_history(
    confirm: bool = Query(
        False,
        description = "Must be set to true to confirm the destructive clear.",
    ),
    current_subject: str = Depends(get_current_subject),
):
    """Require ?confirm=true so a misclick or stray request cannot wipe history."""
    if not confirm:
        raise HTTPException(
            status_code = 400,
            detail = "DELETE /api/chat requires ?confirm=true",
        )
    deleted = clear_chat_history(subject = current_subject)
    logger.warning("cleared %d chat threads for subject=%s", deleted, current_subject)
    return {"status": "deleted", "count": deleted}


@router.get("/settings", response_model = ChatSettingsResponse)
async def get_settings(current_subject: str = Depends(get_current_subject)):
    return ChatSettingsResponse(settings = list_chat_settings(subject = current_subject))


@router.put("/settings", response_model = ChatSettingsResponse)
async def put_settings(
    payload: dict[str, Any],
    current_subject: str = Depends(get_current_subject),
):
    """Deep-merge in storage under BEGIN IMMEDIATE; concurrent writers
    no longer drop one another's updates."""
    try:
        parsed = ChatSettingsPayload.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code = 400, detail = exc.errors()) from exc
    merged = upsert_chat_settings_merge(
        parsed.model_dump(exclude_unset = True),
        subject = current_subject,
    )
    return ChatSettingsResponse(settings = merged)


@router.get("/export", response_model = ChatExportResponse)
async def export_history(current_subject: str = Depends(get_current_subject)):
    from datetime import datetime, timezone

    threads = list_chat_threads(subject = current_subject, include_archived = True)
    messages = list_chat_messages_for_threads(
        [thread["id"] for thread in threads],
        subject = current_subject,
    )
    return ChatExportResponse(
        exportedAt = datetime.now(timezone.utc).isoformat(),
        version = 1,
        threadCount = len(threads),
        threads = [ChatThread(**thread) for thread in threads],
        messages = [ChatMessage(**message) for message in messages],
    )
