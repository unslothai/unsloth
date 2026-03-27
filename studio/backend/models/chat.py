# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

from typing import Optional, List

from pydantic import BaseModel, Field


class ChatThreadCreate(BaseModel):
    id: str
    title: str = "New Chat"
    model_type: str
    model_id: str = ""
    pair_id: Optional[str] = None
    created_at: int


class ChatThreadUpdate(BaseModel):
    title: Optional[str] = None
    archived: Optional[bool] = None


class ChatThreadResponse(BaseModel):
    id: str
    title: str
    model_type: str
    model_id: str
    pair_id: Optional[str] = None
    archived: bool
    created_at: int
    updated_at: int


class ChatMessageCreate(BaseModel):
    id: str
    role: str
    content: str = Field(..., description="JSON-serialized content array")
    attachments: Optional[str] = None
    metadata: Optional[str] = None
    created_at: int


class ChatMessageResponse(BaseModel):
    id: str
    thread_id: str
    role: str
    content: str
    attachments: Optional[str] = None
    metadata: Optional[str] = None
    created_at: int


class ChatHydrateResponse(BaseModel):
    threads: List[ChatThreadResponse]
    messages: List[ChatMessageResponse]
