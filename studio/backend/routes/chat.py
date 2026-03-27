# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Optional

from auth.authentication import get_current_subject
from loggers import get_logger
from models.chat import (
    ChatHydrateResponse,
    ChatMessageCreate,
    ChatMessageResponse,
    ChatThreadCreate,
    ChatThreadResponse,
    ChatThreadUpdate,
)
from storage.studio_db import (
    create_chat_thread,
    delete_chat_thread,
    get_all_chat_data,
    get_chat_thread_messages,
    list_chat_threads,
    update_chat_thread,
    upsert_chat_message,
)

logger = get_logger(__name__)
router = APIRouter()


@router.post("/threads", response_model = ChatThreadResponse, status_code = 201)
async def create_thread(
    body: ChatThreadCreate,
    current_subject: str = Depends(get_current_subject),
):
    create_chat_thread(
        thread_id = body.id,
        title = body.title,
        model_type = body.model_type,
        model_id = body.model_id,
        pair_id = body.pair_id,
        created_at = body.created_at,
    )
    return ChatThreadResponse(
        id = body.id,
        title = body.title,
        model_type = body.model_type,
        model_id = body.model_id,
        pair_id = body.pair_id,
        archived = False,
        created_at = body.created_at,
        updated_at = body.created_at,
    )


@router.get("/threads", response_model = list[ChatThreadResponse])
async def list_threads(
    model_type: Optional[str] = Query(None),
    current_subject: str = Depends(get_current_subject),
):
    rows = list_chat_threads(model_type = model_type)
    return [
        ChatThreadResponse(
            id = r["id"],
            title = r["title"],
            model_type = r["model_type"],
            model_id = r["model_id"] or "",
            pair_id = r["pair_id"],
            archived = bool(r["archived"]),
            created_at = r["created_at"],
            updated_at = r["updated_at"],
        )
        for r in rows
    ]


@router.patch("/threads/{thread_id}", status_code = 204)
async def update_thread(
    thread_id: str,
    body: ChatThreadUpdate,
    current_subject: str = Depends(get_current_subject),
):
    updates = body.model_dump(exclude_none = True)
    if "archived" in updates:
        updates["archived"] = 1 if updates["archived"] else 0
    if not update_chat_thread(thread_id, **updates):
        raise HTTPException(status_code = 404, detail = f"Thread {thread_id} not found")


@router.delete("/threads/{thread_id}", status_code = 204)
async def delete_thread(
    thread_id: str,
    current_subject: str = Depends(get_current_subject),
):
    if not delete_chat_thread(thread_id):
        raise HTTPException(status_code = 404, detail = f"Thread {thread_id} not found")


@router.get("/threads/{thread_id}/messages", response_model = list[ChatMessageResponse])
async def get_messages(
    thread_id: str,
    current_subject: str = Depends(get_current_subject),
):
    rows = get_chat_thread_messages(thread_id)
    return [
        ChatMessageResponse(
            id = r["id"],
            thread_id = r["thread_id"],
            role = r["role"],
            content = r["content"],
            attachments = r["attachments"],
            metadata = r["metadata"],
            created_at = r["created_at"],
        )
        for r in rows
    ]


@router.post("/threads/{thread_id}/messages", response_model = ChatMessageResponse, status_code = 201)
async def create_message(
    thread_id: str,
    body: ChatMessageCreate,
    current_subject: str = Depends(get_current_subject),
):
    upsert_chat_message(
        message_id = body.id,
        thread_id = thread_id,
        role = body.role,
        content = body.content,
        attachments = body.attachments,
        metadata = body.metadata,
        created_at = body.created_at,
    )
    return ChatMessageResponse(
        id = body.id,
        thread_id = thread_id,
        role = body.role,
        content = body.content,
        attachments = body.attachments,
        metadata = body.metadata,
        created_at = body.created_at,
    )


@router.get("/hydrate", response_model = ChatHydrateResponse)
async def hydrate(
    current_subject: str = Depends(get_current_subject),
):
    data = get_all_chat_data()
    return ChatHydrateResponse(
        threads = [
            ChatThreadResponse(
                id = t["id"],
                title = t["title"],
                model_type = t["model_type"],
                model_id = t["model_id"] or "",
                pair_id = t["pair_id"],
                archived = bool(t["archived"]),
                created_at = t["created_at"],
                updated_at = t["updated_at"],
            )
            for t in data["threads"]
        ],
        messages = [
            ChatMessageResponse(
                id = m["id"],
                thread_id = m["thread_id"],
                role = m["role"],
                content = m["content"],
                attachments = m["attachments"],
                metadata = m["metadata"],
                created_at = m["created_at"],
            )
            for m in data["messages"]
        ],
    )
