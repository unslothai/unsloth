# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Chat history API routes backed by studio.db.
"""

from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from auth.authentication import get_current_subject
from storage.studio_db import (
    clear_chat_history,
    count_chat_threads,
    delete_chat_threads,
    get_chat_thread,
    list_chat_messages,
    list_chat_messages_for_threads,
    list_chat_threads,
    sync_chat_messages,
    update_chat_thread,
    upsert_chat_message,
    upsert_chat_thread,
)

router = APIRouter()


class ChatThread(BaseModel):
    id: str
    title: str = "New Chat"
    modelType: Literal["base", "lora", "model1", "model2"]
    modelId: str = ""
    pairId: Optional[str] = None
    archived: bool = False
    createdAt: int


class ChatThreadPatch(BaseModel):
    title: Optional[str] = None
    modelType: Optional[Literal["base", "lora", "model1", "model2"]] = None
    modelId: Optional[str] = None
    pairId: Optional[str] = None
    archived: Optional[bool] = None
    createdAt: Optional[int] = None


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


class ChatDeleteRequest(BaseModel):
    ids: list[str]


class ChatCountResponse(BaseModel):
    count: int


class ChatExportResponse(BaseModel):
    exportedAt: str
    version: int
    threadCount: int
    threads: list[ChatThread]
    messages: list[ChatMessage]


@router.get("/threads", response_model = ChatThreadListResponse)
async def list_threads(
    model_type: Optional[str] = Query(None),
    pair_id: Optional[str] = Query(None),
    include_archived: bool = Query(True),
    current_subject: str = Depends(get_current_subject),
):
    threads = list_chat_threads(
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
    return ChatThread(**upsert_chat_thread(payload.model_dump()))


@router.get("/threads/{thread_id}", response_model = ChatThread)
async def get_thread(
    thread_id: str,
    current_subject: str = Depends(get_current_subject),
):
    thread = get_chat_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code = 404, detail = f"Thread {thread_id} not found")
    return ChatThread(**thread)


@router.patch("/threads/{thread_id}", response_model = ChatThread)
async def patch_thread(
    thread_id: str,
    payload: ChatThreadPatch,
    current_subject: str = Depends(get_current_subject),
):
    thread = update_chat_thread(
        thread_id,
        payload.model_dump(exclude_unset = True),
    )
    if thread is None:
        raise HTTPException(status_code = 404, detail = f"Thread {thread_id} not found")
    return ChatThread(**thread)


@router.delete("/threads")
async def delete_threads(
    payload: ChatDeleteRequest,
    current_subject: str = Depends(get_current_subject),
):
    delete_chat_threads(payload.ids)
    return {"status": "deleted"}


@router.get("/threads/{thread_id}/messages", response_model = ChatMessageListResponse)
async def get_thread_messages(
    thread_id: str,
    current_subject: str = Depends(get_current_subject),
):
    if get_chat_thread(thread_id) is None:
        raise HTTPException(status_code = 404, detail = f"Thread {thread_id} not found")
    return ChatMessageListResponse(
        messages = [ChatMessage(**m) for m in list_chat_messages(thread_id)]
    )


@router.put("/threads/{thread_id}/messages/{message_id}", response_model = ChatMessage)
async def save_thread_message(
    thread_id: str,
    message_id: str,
    payload: ChatMessage,
    current_subject: str = Depends(get_current_subject),
):
    if thread_id != payload.threadId or message_id != payload.id:
        raise HTTPException(status_code = 400, detail = "Message id mismatch")
    if get_chat_thread(thread_id) is None:
        raise HTTPException(status_code = 404, detail = f"Thread {thread_id} not found")
    return ChatMessage(**upsert_chat_message(payload.model_dump()))


@router.put("/threads/{thread_id}/messages", response_model = ChatMessageListResponse)
async def replace_thread_messages(
    thread_id: str,
    payload: ChatMessageSyncRequest,
    current_subject: str = Depends(get_current_subject),
):
    if get_chat_thread(thread_id) is None:
        raise HTTPException(status_code = 404, detail = f"Thread {thread_id} not found")
    messages = []
    for message in payload.messages:
        data = message.model_dump()
        data["threadId"] = thread_id
        messages.append(data)
    return ChatMessageListResponse(
        messages = [ChatMessage(**m) for m in sync_chat_messages(thread_id, messages)]
    )


@router.get("/count", response_model = ChatCountResponse)
async def count_threads(current_subject: str = Depends(get_current_subject)):
    return ChatCountResponse(count = count_chat_threads())


@router.delete("")
async def clear_history(current_subject: str = Depends(get_current_subject)):
    clear_chat_history()
    return {"status": "deleted"}


@router.get("/export", response_model = ChatExportResponse)
async def export_history(current_subject: str = Depends(get_current_subject)):
    from datetime import datetime, timezone

    threads = list_chat_threads(include_archived = True)
    messages = list_chat_messages_for_threads([thread["id"] for thread in threads])
    return ChatExportResponse(
        exportedAt = datetime.now(timezone.utc).isoformat(),
        version = 1,
        threadCount = len(threads),
        threads = [ChatThread(**thread) for thread in threads],
        messages = [ChatMessage(**message) for message in messages],
    )
