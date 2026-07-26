# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Installation-global CRUD endpoints for durable chat memories."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from auth.authentication import get_current_subject
from core.inference import memory
from storage.studio_db import delete_chat_memory, list_chat_memories

router = APIRouter(dependencies = [Depends(get_current_subject)])


class MemoryPayload(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    content: str = Field(min_length = 1, max_length = memory.MAX_CONTENT_CHARS)
    scope: Literal["global", "project"]
    projectId: str | None = None


class MemoryPatch(MemoryPayload):
    pass


class CapturePayload(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    threadId: str = Field(min_length = 1, max_length = 200)
    sourceMessageId: str = Field(min_length = 1, max_length = 200)
    rawOutput: str = Field(max_length = memory.MAX_CAPTURE_OUTPUT_CHARS)


def _error(exc: memory.MemoryValidationError) -> HTTPException:
    return HTTPException(status_code = 400, detail = str(exc))


@router.get("")
async def list_memories(
    scope: Literal["global", "project"] | None = None, project_id: str | None = Query(None)
):
    if scope == "global" and project_id is not None:
        raise HTTPException(status_code = 400, detail = "global memories cannot have a project")
    if scope == "project" and not project_id:
        raise HTTPException(status_code = 400, detail = "project memories require project_id")
    if scope is None and project_id is not None:
        raise HTTPException(status_code = 400, detail = "project_id requires scope=project")
    return {"memories": list_chat_memories(scope, project_id)}


@router.post("")
async def create_memory(payload: MemoryPayload):
    try:
        created = memory.create_memory(
            content = payload.content,
            scope = payload.scope,
            project_id = payload.projectId,
        )
    except memory.MemoryValidationError as exc:
        raise _error(exc) from exc
    # Exact duplicates are an idempotent success rather than a new write.
    return {"memory": created, "created": created is not None}


@router.get("/export")
async def export_memories():
    return {"version": 1, "memories": memory.export_memories()}


@router.post("/apply-capture")
async def apply_capture(payload: CapturePayload):
    _, auto_save_enabled = memory.get_memory_settings()
    if not auto_save_enabled:
        return {"memories": []}
    try:
        memories = memory.apply_capture(
            thread_id = payload.threadId,
            source_message_id = payload.sourceMessageId,
            raw_output = payload.rawOutput,
        )
    except memory.MemoryValidationError as exc:
        raise _error(exc) from exc
    return {"memories": memories}


# Static routes are registered before this dynamic identifier route.
@router.patch("/{memory_id}")
async def patch_memory(memory_id: str, payload: MemoryPatch):
    try:
        updated = memory.edit_memory(
            memory_id = memory_id,
            content = payload.content,
            scope = payload.scope,
            project_id = payload.projectId,
        )
    except memory.MemoryConflictError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc
    except memory.MemoryValidationError as exc:
        raise _error(exc) from exc
    if updated is None:
        raise HTTPException(status_code = 404, detail = "Memory not found")
    return {"memory": updated}


@router.delete("/{memory_id}")
async def remove_memory(memory_id: str):
    # Deleting an absent row is intentionally idempotent.
    return {"deleted": delete_chat_memory(memory_id)}


@router.delete("")
async def clear_memories(scope: Literal["global", "project"], project_id: str | None = Query(None)):
    try:
        deleted = memory.clear_scope(scope, project_id)
    except memory.MemoryValidationError as exc:
        raise _error(exc) from exc
    return {"deleted": deleted}
