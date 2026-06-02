# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Chat history API routes backed by studio.db.
"""

from typing import Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from auth.authentication import get_current_subject
from core.rag.ingestion import purge_all_thread_documents, purge_thread_documents
from storage.studio_db import (
    ChatMessageConflictError,
    CorruptSettingsError,
    clear_chat_history,
    count_chat_threads,
    delete_chat_threads,
    delete_chat_project,
    ensure_chat_project_workspace,
    get_chat_project,
    get_chat_thread,
    get_chat_message,
    list_chat_projects,
    list_chat_legacy_imports,
    list_chat_settings,
    list_chat_messages,
    list_chat_messages_for_threads,
    list_chat_threads,
    sync_chat_messages,
    update_chat_project,
    update_chat_thread,
    upsert_chat_project,
    upsert_chat_legacy_imports,
    upsert_chat_message,
    upsert_chat_settings_merge,
    upsert_chat_thread,
)

router = APIRouter()


class ChatThread(BaseModel):
    id: str
    title: str = "New Chat"
    modelType: Literal["base", "lora", "model1", "model2"]
    modelId: str = ""
    pairId: Optional[str] = None
    projectId: Optional[str] = None
    archived: bool = False
    createdAt: int
    openaiCodeExecContainerId: Optional[str] = None
    anthropicCodeExecContainerId: Optional[str] = None


class ChatThreadPatch(BaseModel):
    title: Optional[str] = None
    modelType: Optional[Literal["base", "lora", "model1", "model2"]] = None
    modelId: Optional[str] = None
    pairId: Optional[str] = None
    projectId: Optional[str] = None
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


class ChatProject(BaseModel):
    id: str
    name: str
    instructions: str = ""
    rootPath: Optional[str] = None
    sandboxPath: Optional[str] = None
    archived: bool = False
    createdAt: int
    updatedAt: int


class ChatProjectPatch(BaseModel):
    name: Optional[str] = None
    instructions: Optional[str] = None
    archived: Optional[bool] = None
    createdAt: Optional[int] = None
    updatedAt: Optional[int] = None


class ChatThreadListResponse(BaseModel):
    threads: list[ChatThread]


class ChatProjectListResponse(BaseModel):
    projects: list[ChatProject]


class ChatMessageListResponse(BaseModel):
    messages: list[ChatMessage]


class ChatMessageSyncRequest(BaseModel):
    messages: list[ChatMessage]
    pruneMissing: bool = False


class ChatDeleteRequest(BaseModel):
    ids: list[str]


class ChatCountResponse(BaseModel):
    count: int


class ChatExportResponse(BaseModel):
    exportedAt: str
    version: int
    threadCount: int
    projects: list[ChatProject] = Field(default_factory = list)
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
    collapseHtmlArtifacts: Optional[bool] = None
    allowArtifactNetworkAccess: Optional[bool] = None
    autoHealToolCalls: Optional[bool] = None
    maxToolCallsPerMessage: Optional[int] = Field(default = None, ge = 1)
    toolCallTimeout: Optional[int] = Field(default = None, ge = 1)


class ChatSettingsResponse(BaseModel):
    settings: dict[str, Any]


class ChatMessagesBatchRequest(BaseModel):
    threadIds: list[str]


class ChatMessagesBatchResponse(BaseModel):
    messagesByThreadId: dict[str, list[ChatMessage]]


class ChatImportLedgerResponse(BaseModel):
    # Plain list of legacy thread ids. Keeping the payload key-less keeps
    # the client diff to a single Set construction.
    threadIds: list[str]


class ChatImportLedgerRecordRequest(BaseModel):
    # 10k cap keeps the request body bounded; real users have << 1k threads.
    threadIds: list[str] = Field(default_factory = list, max_length = 10_000)


class ChatImportLedgerRecordResponse(BaseModel):
    # accepted: deduped non-empty input count. inserted: rows actually new
    # (ON CONFLICT DO NOTHING skips already-recorded ids). The client uses
    # `accepted >= 0` as the "endpoint exists" signal and ignores the split
    # otherwise.
    accepted: int
    inserted: int


@router.get("/threads", response_model = ChatThreadListResponse)
async def list_threads(
    model_type: Optional[str] = Query(None),
    pair_id: Optional[str] = Query(None),
    project_id: Optional[str] = Query(None),
    include_archived: bool = Query(True),
    current_subject: str = Depends(get_current_subject),
):
    threads = list_chat_threads(
        model_type = model_type,
        pair_id = pair_id,
        project_id = project_id,
        include_archived = include_archived,
    )
    return ChatThreadListResponse(threads = [ChatThread(**t) for t in threads])


@router.post("/threads", response_model = ChatThread)
async def save_thread(
    payload: ChatThread,
    current_subject: str = Depends(get_current_subject),
):
    if payload.projectId and get_chat_project(payload.projectId) is None:
        raise HTTPException(
            status_code = 404,
            detail = f"Project {payload.projectId} not found",
        )
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
    patch = payload.model_dump(exclude_unset = True)
    for field in ("title", "modelType", "modelId", "archived", "createdAt"):
        if field in patch and patch[field] is None:
            raise HTTPException(status_code = 400, detail = f"{field} cannot be null")
    if patch.get("projectId") and get_chat_project(patch["projectId"]) is None:
        raise HTTPException(
            status_code = 404,
            detail = f"Project {patch['projectId']} not found",
        )
    thread = update_chat_thread(
        thread_id,
        patch,
    )
    if thread is None:
        raise HTTPException(status_code = 404, detail = f"Thread {thread_id} not found")
    return ChatThread(**thread)


@router.delete("/threads")
async def delete_threads(
    payload: ChatDeleteRequest,
    current_subject: str = Depends(get_current_subject),
):
    # No FK cascade from rag_documents to chat_threads, so purge their files +
    # vectors + bm25 explicitly before deleting the threads.
    purge_thread_documents(payload.ids)
    delete_chat_threads(payload.ids)
    return {"status": "deleted"}


@router.get("/projects", response_model = ChatProjectListResponse)
async def list_projects(
    include_archived: bool = Query(False),
    current_subject: str = Depends(get_current_subject),
):
    return ChatProjectListResponse(
        projects = [
            ChatProject(**(ensure_chat_project_workspace(project["id"]) or project))
            for project in list_chat_projects(include_archived = include_archived)
        ]
    )


@router.post("/projects", response_model = ChatProject)
async def save_project(
    payload: ChatProject,
    current_subject: str = Depends(get_current_subject),
):
    return ChatProject(**upsert_chat_project(payload.model_dump()))


@router.get("/projects/{project_id}", response_model = ChatProject)
async def get_project(
    project_id: str,
    current_subject: str = Depends(get_current_subject),
):
    project = ensure_chat_project_workspace(project_id)
    if project is None:
        raise HTTPException(
            status_code = 404,
            detail = f"Project {project_id} not found",
        )
    return ChatProject(**project)


@router.patch("/projects/{project_id}", response_model = ChatProject)
async def patch_project(
    project_id: str,
    payload: ChatProjectPatch,
    current_subject: str = Depends(get_current_subject),
):
    patch = payload.model_dump(exclude_unset = True)
    for field in ("name", "archived", "createdAt", "updatedAt"):
        if field in patch and patch[field] is None:
            raise HTTPException(status_code = 400, detail = f"{field} cannot be null")
    project = update_chat_project(project_id, patch)
    if project is not None:
        project = ensure_chat_project_workspace(project_id)
    if project is None:
        raise HTTPException(
            status_code = 404,
            detail = f"Project {project_id} not found",
        )
    return ChatProject(**project)


@router.delete("/projects/{project_id}", response_model = ChatProject)
async def delete_project(
    project_id: str,
    delete_files: bool = Query(False),
    current_subject: str = Depends(get_current_subject),
):
    project = delete_chat_project(project_id, delete_files = delete_files)
    if project is None:
        raise HTTPException(
            status_code = 404,
            detail = f"Project {project_id} not found",
        )
    return ChatProject(**project)


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


@router.post("/messages:batch", response_model = ChatMessagesBatchResponse)
async def batch_thread_messages(
    payload: ChatMessagesBatchRequest,
    current_subject: str = Depends(get_current_subject),
):
    """One round-trip per sidebar/search rebuild instead of N. Unknown thread
    ids are returned as empty lists so callers don't need a pre-flight."""
    by_thread: dict[str, list[ChatMessage]] = {tid: [] for tid in payload.threadIds}
    for m in list_chat_messages_for_threads(payload.threadIds):
        tid = m["threadId"]
        if tid in by_thread:
            by_thread[tid].append(ChatMessage(**m))
    return ChatMessagesBatchResponse(messagesByThreadId = by_thread)


@router.get("/threads/{thread_id}/messages/{message_id}", response_model = ChatMessage)
async def get_thread_message(
    thread_id: str,
    message_id: str,
    current_subject: str = Depends(get_current_subject),
):
    if get_chat_thread(thread_id) is None:
        raise HTTPException(status_code = 404, detail = f"Thread {thread_id} not found")
    message = get_chat_message(thread_id, message_id)
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
    if get_chat_thread(thread_id) is None:
        raise HTTPException(status_code = 404, detail = f"Thread {thread_id} not found")
    try:
        return ChatMessage(**upsert_chat_message(payload.model_dump()))
    except ChatMessageConflictError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc


@router.put("/threads/{thread_id}/messages", response_model = ChatMessageListResponse)
async def replace_thread_messages(
    thread_id: str,
    payload: ChatMessageSyncRequest,
    current_subject: str = Depends(get_current_subject),
):
    mismatched_ids = [
        message.id for message in payload.messages if message.threadId != thread_id
    ]
    if mismatched_ids:
        preview = ", ".join(mismatched_ids[:5])
        suffix = (
            "" if len(mismatched_ids) <= 5 else f" (+{len(mismatched_ids) - 5} more)"
        )
        raise HTTPException(
            status_code = 400,
            detail = f"Message threadId mismatch: {preview}{suffix}",
        )
    if get_chat_thread(thread_id) is None:
        raise HTTPException(status_code = 404, detail = f"Thread {thread_id} not found")
    messages = [message.model_dump() for message in payload.messages]
    try:
        return ChatMessageListResponse(
            messages = [
                ChatMessage(**m)
                for m in sync_chat_messages(
                    thread_id,
                    messages,
                    prune_missing = payload.pruneMissing,
                )
            ]
        )
    except ChatMessageConflictError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc


@router.get("/count", response_model = ChatCountResponse)
async def count_threads(current_subject: str = Depends(get_current_subject)):
    return ChatCountResponse(count = count_chat_threads())


@router.get("/import-ledger", response_model = ChatImportLedgerResponse)
async def get_import_ledger(current_subject: str = Depends(get_current_subject)):
    """Legacy-Dexie import ledger. Returns the set of legacy thread ids
    already copied into chat_threads / chat_messages. The frontend
    uses this on every fresh tab open to decide whether to re-run the
    Dexie -> studio.db import. Source of truth lives inside studio.db
    so a studio.db wipe makes the import recoverable."""
    return ChatImportLedgerResponse(threadIds = list_chat_legacy_imports())


@router.post("/import-ledger", response_model = ChatImportLedgerRecordResponse)
async def record_import_ledger(
    payload: ChatImportLedgerRecordRequest,
    current_subject: str = Depends(get_current_subject),
):
    """Mark each legacy thread id as imported. Idempotent."""
    accepted, inserted = upsert_chat_legacy_imports(payload.threadIds)
    return ChatImportLedgerRecordResponse(accepted = accepted, inserted = inserted)


@router.delete("")
async def clear_history(current_subject: str = Depends(get_current_subject)):
    purge_all_thread_documents()
    clear_chat_history()
    return {"status": "deleted"}


@router.get("/settings", response_model = ChatSettingsResponse)
async def get_settings(current_subject: str = Depends(get_current_subject)):
    return ChatSettingsResponse(settings = list_chat_settings())


@router.put("/settings", response_model = ChatSettingsResponse)
async def put_settings(
    payload: dict[str, Any],
    current_subject: str = Depends(get_current_subject),
):
    try:
        parsed = ChatSettingsPayload.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code = 400, detail = exc.errors()) from exc
    # Atomic read + deep-merge + write inside one BEGIN IMMEDIATE so two
    # concurrent slider drags can't drop each other's updates.
    try:
        return ChatSettingsResponse(
            settings = upsert_chat_settings_merge(parsed.model_dump(exclude_unset = True))
        )
    except CorruptSettingsError as exc:
        raise HTTPException(status_code = 409, detail = str(exc)) from exc


@router.get("/export", response_model = ChatExportResponse)
async def export_history(current_subject: str = Depends(get_current_subject)):
    from datetime import datetime, timezone

    threads = list_chat_threads(include_archived = True)
    projects = list_chat_projects(include_archived = True)
    messages = list_chat_messages_for_threads([thread["id"] for thread in threads])
    return ChatExportResponse(
        exportedAt = datetime.now(timezone.utc).isoformat(),
        version = 1,
        threadCount = len(threads),
        projects = [ChatProject(**project) for project in projects],
        threads = [ChatThread(**thread) for thread in threads],
        messages = [ChatMessage(**message) for message in messages],
    )
