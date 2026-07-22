# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Chat history API routes backed by studio.db.
"""

from typing import Annotated, Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from auth.authentication import get_current_subject
from loggers import get_logger
from utils.utils import safe_curated_detail, log_and_http_error
from storage.studio_db import (
    ChatMessageConflictError,
    ChatMessageProtectedError,
    CorruptSettingsError,
    clear_chat_history,
    count_chat_threads,
    count_forks_for_message,
    delete_chat_attachment,
    delete_chat_threads,
    delete_chat_project,
    ensure_chat_project_workspace,
    fork_chat_thread,
    get_chat_attachment,
    get_chat_project,
    get_chat_thread,
    get_chat_message,
    list_chat_attachments_page,
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

logger = get_logger(__name__)


class ChatThread(BaseModel):
    id: str
    title: str = "New Chat"
    modelType: Literal["base", "lora", "model1", "model2"]
    modelId: str = ""
    pairId: Optional[str] = None
    projectId: Optional[str] = None
    archived: bool = False
    createdAt: int
    updatedAt: Optional[int] = None
    openaiCodeExecContainerId: Optional[str] = None
    anthropicCodeExecContainerId: Optional[str] = None
    forkedFromThreadId: Optional[str] = None
    forkedFromMessageId: Optional[str] = None


class ChatThreadPatch(BaseModel):
    title: Optional[str] = None
    modelType: Optional[Literal["base", "lora", "model1", "model2"]] = None
    modelId: Optional[str] = None
    pairId: Optional[str] = None
    projectId: Optional[str] = None
    archived: Optional[bool] = None
    createdAt: Optional[int] = None
    updatedAt: Optional[int] = None
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
    systemVariables: Optional[str] = None
    trustRemoteCode: Optional[bool] = None
    fastMode: Optional[bool] = None


class ChatPreset(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    name: str
    params: ChatInferenceSettings


class ChatSettingsPayload(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    inferenceParams: Optional[ChatInferenceSettings] = None
    customPresets: Optional[list[ChatPreset]] = None
    activePreset: Optional[str] = None
    activePresetSource: Optional[Literal["builtin-default", "custom", "modified"]] = None
    autoTitle: Optional[bool] = None
    reasoningEffort: Optional[
        Literal["none", "minimal", "low", "medium", "high", "max", "xhigh"]
    ] = None
    preserveThinking: Optional[bool] = None
    collapseHtmlArtifacts: Optional[bool] = None
    allowArtifactNetworkAccess: Optional[bool] = None
    autoHealToolCalls: Optional[bool] = None
    nudgeToolCalls: Optional[bool] = None
    maxToolCallsPerMessage: Optional[int] = Field(default = None, ge = 1)
    toolCallTimeout: Optional[int] = Field(default = None, ge = 1)


class ChatSettingsResponse(BaseModel):
    settings: dict[str, Any]


class ChatMessagesBatchRequest(BaseModel):
    threadIds: list[str]


class ChatMessagesBatchResponse(BaseModel):
    messagesByThreadId: dict[str, list[ChatMessage]]


class ChatImportLedgerResponse(BaseModel):
    threadIds: list[str]


class ChatImportLedgerRecordRequest(BaseModel):
    # 10k cap bounds the request body; real users have << 1k threads.
    threadIds: list[str] = Field(default_factory = list, max_length = 10_000)


class ChatImportLedgerRecordResponse(BaseModel):
    # accepted: deduped non-empty input count. inserted: rows actually new
    # (ON CONFLICT DO NOTHING skips already-recorded ids).
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
async def save_thread(payload: ChatThread, current_subject: str = Depends(get_current_subject)):
    if payload.projectId and get_chat_project(payload.projectId) is None:
        raise HTTPException(
            status_code = 404,
            detail = f"Project {payload.projectId} not found",
        )
    return ChatThread(**upsert_chat_thread(payload.model_dump()))


@router.get("/threads/{thread_id}", response_model = ChatThread)
async def get_thread(thread_id: str, current_subject: str = Depends(get_current_subject)):
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
    for field in ("title", "modelType", "modelId", "archived", "createdAt", "updatedAt"):
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


def _cancel_active_research(request: Request, thread_ids: list[str]) -> None:
    """Signal any active research runs on these threads to stop before their rows are deleted.

    Deleting a thread cascade-deletes its research_runs row, and the worker eventually notices via
    lease loss -- but only at its next lease check, so it can keep doing model/web/RAG work (up to a
    tool timeout) for a run that no longer exists. Setting the cancel event first shortens that
    orphaned window. Best-effort: never let cancellation bookkeeping break the deletion itself.
    """
    if not thread_ids:
        return
    try:
        from storage import research_runs_db
    except Exception:  # noqa: BLE001 - research storage optional/unavailable
        return
    supervisor = getattr(request.app.state, "research_supervisor", None)
    for thread_id in thread_ids:
        try:
            active = research_runs_db.list_active(thread_id)
        except Exception:  # noqa: BLE001
            continue
        for run in active:
            try:
                status = research_runs_db.request_cancel(run["id"])
                if supervisor is not None and status == "cancelling":
                    supervisor.cancel(run["id"])
            except Exception:  # noqa: BLE001
                logger.warning(
                    "chat_history.cancel_active_research_failed run_id=%s",
                    run.get("id"),
                    exc_info = True,
                )


@router.delete("/threads")
async def delete_threads(
    payload: ChatDeleteRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    _cancel_active_research(request, payload.ids)
    delete_chat_threads(payload.ids)
    return {"status": "deleted"}


@router.get("/attachments")
def list_attachments(
    limit: Annotated[int, Query(ge = 1, le = 100)] = 50,
    offset: Annotated[int, Query(ge = 0)] = 0,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    """One bounded page of chat uploads for the settings Data tab."""
    attachments, next_offset = list_chat_attachments_page(limit = limit, offset = offset)
    return {"attachments": attachments, "nextOffset": next_offset}


def _decode_attachment_base64(payload: str) -> bytes:
    """Strict base64 decode of a stored payload.

    Normalizes first: strips whitespace, fixes padding, accepts the URL-safe
    alphabet. validate=False would silently drop bad characters and serve
    corrupted bytes instead of failing, so raise 422 on anything else.
    """
    import base64

    normalized = "".join(payload.split())
    altchars = b"-_" if ("-" in normalized or "_" in normalized) else None
    normalized += "=" * (-len(normalized) % 4)
    try:
        return base64.b64decode(normalized, altchars = altchars, validate = True)
    except Exception as exc:  # noqa: BLE001 - corrupt stored payload
        raise HTTPException(status_code = 422, detail = "Attachment data is corrupt") from exc


_AUDIO_FORMAT_MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "ogg": "audio/ogg",
    "flac": "audio/flac",
}


def _safe_image_media_type(media_type: str) -> str:
    """Clamp a data-URL media type to something inert to render.

    Imported chats store image parts verbatim, so the embedded type can be
    text/html or image/svg+xml; echoing those would execute markup with the
    app origin when opened. Anything not a plain raster type downloads as
    bytes instead.
    """
    lowered = media_type.strip().lower()
    if lowered.startswith("image/") and lowered != "image/svg+xml":
        return lowered
    return "application/octet-stream"


@router.get("/attachments/{message_id}/{attachment_id}/file")
def get_attachment_file(
    message_id: str,
    attachment_id: str,
    current_subject: str = Depends(get_current_subject),
):
    """Serve one attachment's stored content: image or audio bytes, or
    extracted text."""
    import urllib.parse

    from fastapi.responses import Response

    attachment = get_chat_attachment(message_id, attachment_id)
    if attachment is None:
        raise HTTPException(status_code = 404, detail = "Attachment not found")

    attachment_content_type = attachment.get("contentType")
    texts: list[str] = []
    for part in attachment.get("content") or []:
        if not isinstance(part, dict):
            continue
        image = part.get("image")
        if isinstance(image, str) and image[:5].lower() == "data:":
            header, _, payload = image.partition(",")
            media_type = _safe_image_media_type(
                header[5:].split(";", 1)[0] or "application/octet-stream"
            )
            if "base64" not in header.lower():
                # RFC 2397 non-base64 form stores percent-encoded bytes.
                data = urllib.parse.unquote_to_bytes(payload)
                return Response(content = data, media_type = media_type)
            data = _decode_attachment_base64(payload)
            return Response(content = data, media_type = media_type)
        # Audio parts: the attachment adapter stores {data, format} with raw
        # base64; compare chats store a bare base64 string.
        audio = part.get("audio")
        if isinstance(audio, dict) or (isinstance(audio, str) and audio):
            if isinstance(audio, dict):
                payload = audio.get("data")
                audio_format = audio.get("format")
            else:
                payload = audio.rsplit(",", 1)[-1]
                audio_format = None
            if isinstance(payload, str) and payload:
                data = _decode_attachment_base64(payload)
                media_type = (
                    attachment_content_type
                    if isinstance(attachment_content_type, str)
                    and attachment_content_type.startswith("audio/")
                    else _AUDIO_FORMAT_MEDIA_TYPES.get(
                        str(audio_format or "").lower(), "application/octet-stream"
                    )
                )
                return Response(content = data, media_type = media_type)
        text = part.get("text")
        if isinstance(text, str) and text:
            texts.append(text)
    if texts:
        return Response(content = "\n".join(texts), media_type = "text/plain; charset=utf-8")
    raise HTTPException(status_code = 404, detail = "Attachment has no stored content")


@router.delete("/attachments/{message_id}/{attachment_id}")
def delete_attachment(
    message_id: str,
    attachment_id: str,
    current_subject: str = Depends(get_current_subject),
) -> dict:
    """Remove one attachment from its chat message."""
    try:
        deleted = delete_chat_attachment(message_id, attachment_id)
    except ChatMessageProtectedError as exc:
        raise log_and_http_error(
            exc,
            409,
            safe_curated_detail(exc),
            event = "chat_history.delete_attachment_conflict",
            log = logger,
        ) from exc
    if not deleted:
        raise HTTPException(status_code = 404, detail = "Attachment not found")
    return {"ok": True}


@router.get("/projects", response_model = ChatProjectListResponse)
async def list_projects(
    include_archived: bool = Query(False), current_subject: str = Depends(get_current_subject)
):
    return ChatProjectListResponse(
        projects = [
            ChatProject(**(ensure_chat_project_workspace(project["id"]) or project))
            for project in list_chat_projects(include_archived = include_archived)
        ]
    )


@router.post("/projects", response_model = ChatProject)
async def save_project(payload: ChatProject, current_subject: str = Depends(get_current_subject)):
    return ChatProject(**upsert_chat_project(payload.model_dump()))


@router.get("/projects/{project_id}", response_model = ChatProject)
async def get_project(project_id: str, current_subject: str = Depends(get_current_subject)):
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
    request: Request,
    delete_files: bool = Query(False),
    current_subject: str = Depends(get_current_subject),
):
    _cancel_active_research(
        request, [thread["id"] for thread in list_chat_threads(project_id = project_id)]
    )
    project = delete_chat_project(project_id, delete_files = delete_files)
    if project is None:
        raise HTTPException(
            status_code = 404,
            detail = f"Project {project_id} not found",
        )
    # Best-effort: drop the project's RAG sources (lazy import keeps RAG optional).
    try:
        import os

        from storage import rag_db
        if rag_db.RAG_AVAILABLE:
            from core.rag import store as rag_store
            from utils.paths import rag_uploads_root

            uploads = os.path.realpath(str(rag_uploads_root()))
            conn = rag_db.get_connection()
            try:
                scope = rag_store.project_scope(project_id)
                for doc in rag_store.list_documents(conn, scope):
                    full = rag_store.get_document(conn, doc["id"]) or {}
                    rag_store.delete_document(conn, doc["id"])
                    stored = full.get("stored_path")
                    # Also remove the uploaded file; confined to the uploads root.
                    if stored:
                        target = os.path.realpath(stored)
                        if (
                            os.path.isfile(target)
                            and os.path.commonpath([uploads, target]) == uploads
                        ):
                            os.remove(target)
            finally:
                conn.close()
    except Exception:  # noqa: BLE001 - source cleanup must not block project deletion
        logger.warning("failed to delete RAG sources for project %s", project_id, exc_info = True)
    return ChatProject(**project)


@router.get("/threads/{thread_id}/messages", response_model = ChatMessageListResponse)
async def get_thread_messages(thread_id: str, current_subject: str = Depends(get_current_subject)):
    if get_chat_thread(thread_id) is None:
        raise HTTPException(status_code = 404, detail = f"Thread {thread_id} not found")
    return ChatMessageListResponse(
        messages = [ChatMessage(**m) for m in list_chat_messages(thread_id)]
    )


@router.post("/messages:batch", response_model = ChatMessagesBatchResponse)
async def batch_thread_messages(
    payload: ChatMessagesBatchRequest, current_subject: str = Depends(get_current_subject)
):
    """One round-trip per sidebar/search rebuild instead of N. Unknown thread ids return empty lists."""
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
def save_thread_message(
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
    except (ChatMessageConflictError, ChatMessageProtectedError) as exc:
        raise log_and_http_error(
            exc,
            409,
            safe_curated_detail(exc),
            event = "chat_history.save_message_conflict",
            log = logger,
        ) from exc


@router.put("/threads/{thread_id}/messages", response_model = ChatMessageListResponse)
def replace_thread_messages(
    thread_id: str,
    payload: ChatMessageSyncRequest,
    current_subject: str = Depends(get_current_subject),
):
    mismatched_ids = [message.id for message in payload.messages if message.threadId != thread_id]
    if mismatched_ids:
        preview = ", ".join(mismatched_ids[:5])
        suffix = "" if len(mismatched_ids) <= 5 else f" (+{len(mismatched_ids) - 5} more)"
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
    except (ChatMessageConflictError, ChatMessageProtectedError) as exc:
        raise log_and_http_error(
            exc,
            409,
            safe_curated_detail(exc),
            event = "chat_history.replace_messages_conflict",
            log = logger,
        ) from exc


@router.get("/count", response_model = ChatCountResponse)
async def count_threads(current_subject: str = Depends(get_current_subject)):
    return ChatCountResponse(count = count_chat_threads())


@router.get("/import-ledger", response_model = ChatImportLedgerResponse)
async def get_import_ledger(current_subject: str = Depends(get_current_subject)):
    """Legacy-Dexie import ledger: legacy thread ids already copied into chat tables.

    The frontend checks this on tab open to decide whether to re-run the Dexie -> studio.db import.
    """
    return ChatImportLedgerResponse(threadIds = list_chat_legacy_imports())


@router.post("/import-ledger", response_model = ChatImportLedgerRecordResponse)
async def record_import_ledger(
    payload: ChatImportLedgerRecordRequest, current_subject: str = Depends(get_current_subject)
):
    """Mark each legacy thread id as imported. Idempotent."""
    accepted, inserted = upsert_chat_legacy_imports(payload.threadIds)
    return ChatImportLedgerRecordResponse(accepted = accepted, inserted = inserted)


@router.delete("")
async def clear_history(request: Request, current_subject: str = Depends(get_current_subject)):
    _cancel_active_research(request, [thread["id"] for thread in list_chat_threads()])
    clear_chat_history()
    return {"status": "deleted"}


@router.get("/settings", response_model = ChatSettingsResponse)
async def get_settings(current_subject: str = Depends(get_current_subject)):
    return ChatSettingsResponse(settings = list_chat_settings())


@router.put("/settings", response_model = ChatSettingsResponse)
async def put_settings(
    payload: dict[str, Any], current_subject: str = Depends(get_current_subject)
):
    try:
        parsed = ChatSettingsPayload.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code = 400, detail = exc.errors()) from exc
    # Atomic read + deep-merge + write in one BEGIN IMMEDIATE so concurrent updates don't clobber.
    try:
        return ChatSettingsResponse(
            settings = upsert_chat_settings_merge(parsed.model_dump(exclude_unset = True))
        )
    except CorruptSettingsError as exc:
        raise log_and_http_error(
            exc,
            409,
            safe_curated_detail(exc),
            event = "chat_history.put_settings_conflict",
            log = logger,
        ) from exc


class ChatForkRequest(BaseModel):
    messageId: str
    newThreadId: str
    createdAt: int


class ChatForkResponse(BaseModel):
    thread: ChatThread
    messages: list[ChatMessage]
    containerSnapshotWarning: Optional[str] = None


class ChatForkCountResponse(BaseModel):
    count: int


@router.post("/threads/{thread_id}/fork", response_model = ChatForkResponse)
async def fork_thread(
    thread_id: str,
    payload: ChatForkRequest,
    current_subject: str = Depends(get_current_subject),
):
    """Fork a thread at `messageId` -- creates a new thread with
    ancestor msgs [root..messageId] copied with fresh ids. Both
    code-exec container ids reset on the fork. OpenAI snapshot is a
    best-effort enhancement; failure surfaces as
    `containerSnapshotWarning` and the fork still succeeds with a
    clean sandbox.
    """
    import uuid

    source = get_chat_thread(thread_id)
    if source is None:
        raise HTTPException(status_code = 404, detail = f"Thread {thread_id} not found")
    if get_chat_message(thread_id, payload.messageId) is None:
        raise HTTPException(
            status_code = 404,
            detail = f"Message {payload.messageId} not found in thread {thread_id}",
        )
    base_title = source.get("title") or "New Chat"
    new_title = f"fork · {base_title}"
    forked = fork_chat_thread(
        source_thread_id = thread_id,
        branch_message_id = payload.messageId,
        new_thread_id = payload.newThreadId,
        new_title = new_title,
        created_at = payload.createdAt,
        id_factory = lambda: str(uuid.uuid4()),
    )
    if forked is None:
        raise HTTPException(status_code = 500, detail = "Fork failed")
    messages = list_chat_messages(payload.newThreadId)
    # Best-effort OpenAI container snapshot. Stub: a follow-up patch can
    # call /v1/containers list+download / create+upload here and patch
    # the new openaiCodeExecContainerId. For v1 we always start clean
    # and surface the same warning regardless of provider so the UI can
    # show a consistent "sandbox starts fresh" toast.
    warning: Optional[str] = None
    if source.get("openaiCodeExecContainerId") or source.get("anthropicCodeExecContainerId"):
        warning = "Sandbox starts fresh in fork; files from parent are not carried over."
    return ChatForkResponse(
        thread = ChatThread(**forked),
        messages = [ChatMessage(**m) for m in messages],
        containerSnapshotWarning = warning,
    )


@router.get(
    "/threads/{thread_id}/messages/{message_id}/forks",
    response_model = ChatForkCountResponse,
)
async def get_fork_count(
    thread_id: str,
    message_id: str,
    current_subject: str = Depends(get_current_subject),
):
    return ChatForkCountResponse(count = count_forks_for_message(thread_id, message_id))


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
