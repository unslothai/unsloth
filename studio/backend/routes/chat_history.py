# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Chat history API routes backed by studio.db.
"""

from typing import Annotated, Any, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from auth.authentication import get_current_subject
from core.inference.llama_server_args import BATCH_MAX, BATCH_MIN, PARALLEL_MAX, PARALLEL_MIN
from loggers import get_logger
from utils.utils import safe_curated_detail, log_and_http_error
from storage.studio_db import (
    ChatMessageConflictError,
    ChatMessageProtectedError,
    ChatThreadPreconditionFailed,
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
    # Apply only while the row still holds this title, so a rename beats a background rewrite.
    expectedTitle: Optional[str] = None
    # Apply only while this is still the opening user message, so a title from a deleted one is rejected.
    expectedOpeningMessageId: Optional[str] = None
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


class ChatProjectDeleted(ChatProject):
    """The deleted project, plus the member sandboxes that still hold files."""

    sandboxes_kept: list[str] = []


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
    # Files a tool call wrote. Off by default: they are the user's and the chat
    # card offers them as downloads. An empty sandbox is removed either way.
    delete_files: bool = False


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


class ChatPresetLoadConfig(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    customContextLength: Optional[int] = Field(default = None, gt = 0)
    maxSeqLength: Optional[float] = None
    kvCacheDtype: Optional[str] = None
    mlxKvBits: Optional[Literal[8, 6, 5, 4, 3, 2]] = None
    speculativeType: Optional[str] = None
    specDraftNMax: Optional[int] = Field(default = None, ge = 1, le = 16)
    nParallel: Optional[int] = Field(default = None, ge = PARALLEL_MIN, le = PARALLEL_MAX)
    # The normalizer emits both keys on every preset (null included) and this model is
    # extra="forbid", so without them PUT /api/chat/settings 400s the whole save for any
    # preset carrying a loadConfig, including one that only pinned nParallel.
    nBatch: Optional[int] = Field(default = None, ge = BATCH_MIN, le = BATCH_MAX)
    nUbatch: Optional[int] = Field(default = None, ge = BATCH_MIN, le = BATCH_MAX)
    tensorParallel: Optional[bool] = None
    gpuMemoryMode: Optional[Literal["manual"]] = None
    gpuLayers: Optional[int] = None
    nCpuMoe: Optional[int] = Field(default = None, ge = 0)

    @field_validator("nBatch", "nUbatch", mode = "before")
    @classmethod
    def _no_booleans(cls, value: Any) -> Any:
        # Same contract as LoadRequest: bool subclasses int, so lax mode would store
        # `true` as 1 here while /load 422s it.
        if isinstance(value, bool):
            raise ValueError("Expected a number, got a boolean.")
        return value


class ChatPreset(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    name: str
    params: ChatInferenceSettings
    loadConfig: Optional[ChatPresetLoadConfig] = None


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
    expected_title = patch.pop("expectedTitle", None)
    expected_opening_message_id = patch.pop("expectedOpeningMessageId", None)
    for field in ("title", "modelType", "modelId", "archived", "createdAt", "updatedAt"):
        if field in patch and patch[field] is None:
            raise HTTPException(status_code = 400, detail = f"{field} cannot be null")
    if patch.get("projectId") and get_chat_project(patch["projectId"]) is None:
        raise HTTPException(
            status_code = 404,
            detail = f"Project {patch['projectId']} not found",
        )
    try:
        thread = update_chat_thread(
            thread_id,
            patch,
            expected_title = expected_title,
            expected_opening_message_id = expected_opening_message_id,
        )
    except ChatThreadPreconditionFailed:
        raise HTTPException(
            status_code = 409,
            detail = f"Thread {thread_id} changed since it was read",
        )
    if thread is None:
        raise HTTPException(status_code = 404, detail = f"Thread {thread_id} not found")
    return ChatThread(**thread)


def _cancel_active_research(request: Request, thread_ids: list[str]) -> None:
    """Signal any active research runs on these threads to stop before their rows are deleted.

    Deleting a thread cascade-deletes its research_runs row, but the worker only notices at its
    next lease check, so it can keep doing model/web/RAG work (up to a tool timeout) for a run
    that no longer exists. Best-effort: cancellation bookkeeping must never break the deletion.
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


def _cancel_research_runs(request: Request, run_ids: list[str]) -> None:
    """Stop these research runs by id. Best effort, like every cleanup here."""
    if not run_ids:
        return
    try:
        from storage import research_runs_db
    except Exception:  # noqa: BLE001 - research storage optional/unavailable
        return
    supervisor = getattr(request.app.state, "research_supervisor", None)
    for run_id in run_ids:
        # The row is usually already gone here, which makes request_cancel raise:
        # the supervisor is what actually stops the worker, so it is told first
        # and the status update is the best-effort half.
        if supervisor is not None:
            try:
                supervisor.cancel(run_id)
            except Exception:  # noqa: BLE001
                logger.warning("Could not signal research run %s", run_id, exc_info = True)
        try:
            research_runs_db.request_cancel(run_id)
        except Exception:  # noqa: BLE001
            pass  # no row to update, which is the ordinary case after a delete


def _cancel_active_generations(thread_ids: list[str]) -> None:
    """Stop any generation still running for these threads.

    The sandbox goes with the thread, but a request that has not reached the
    executor yet would dispatch its tool call afterwards, recreate the folder,
    and write files no chat can reach. The in-flight guard only covers calls
    already inside the executor. Best effort: this must never break a delete.
    """
    if not thread_ids:
        return
    try:
        from state import active_generations
    except Exception:  # noqa: BLE001 - never block a delete on this
        return
    for thread_id in thread_ids:
        try:
            active_generations.cancel_thread(thread_id)
        except Exception:  # noqa: BLE001
            continue


@router.delete("/threads")
async def delete_threads(
    payload: ChatDeleteRequest,
    request: Request,
    current_subject: str = Depends(get_current_subject),
):
    _cancel_active_research(request, payload.ids)
    _cancel_active_generations(payload.ids)
    delete_chat_threads(payload.ids)
    # Keyed by thread id, so nothing can reference the folder once the thread
    # is gone. Clean it up rather than leaking one per chat.
    # In a worker: right after an upgrade this also runs the legacy move, and a
    # cross-filesystem copy on the event loop stops every other request.
    removed, kept = await _remove_sandboxes(payload.ids, payload.delete_files)
    return {"status": "deleted", "sandboxes_removed": removed, "sandboxes_kept": kept}


async def _remove_sandboxes(thread_ids, delete_files: bool) -> "tuple[int, list[str]]":
    """Drop each thread's sandbox off the event loop. Never raises.

    Returns how many went and which ids still have files. The chat is the only
    way to those files, so a caller that never offered the choice can offer it
    once it knows there was something to keep.
    """
    from starlette.concurrency import run_in_threadpool

    def _remove() -> "tuple[int, list[str]]":
        from core.inference.tools import (
            record_kept_sandbox,
            remove_session_sandbox,
            sandbox_removal_deferred,
            session_sandbox_has_files,
        )
        from storage.studio_db import sandbox_is_referenced_elsewhere

        removed, kept = 0, []
        for thread_id in thread_ids:
            # The row went first, and another tab can upsert the same id in the
            # meantime. That chat is alive, with a tool call possibly running in
            # here, so its folder is not this delete's to take.
            if get_chat_thread(thread_id) is not None:
                continue
            # A fork clones the message content, cards and all, so the source
            # chat's files are still on screen in a chat the user kept.
            if delete_files and sandbox_is_referenced_elsewhere(thread_id):
                if session_sandbox_has_files(thread_id):
                    kept.append(thread_id)
                    # The user asked for these files and the chat is gone, so
                    # nothing comes back to that folder: written down, and the
                    # collection below takes it once the last fork goes too.
                    record_kept_sandbox(thread_id)
                continue
            if remove_session_sandbox(thread_id, delete_files = delete_files):
                removed += 1
            # A removal that had to wait for a running tool call is reported as
            # kept: that call can still write a file, and this is the only
            # answer the caller gets.
            elif sandbox_removal_deferred(thread_id) or session_sandbox_has_files(thread_id):
                kept.append(thread_id)
        return removed, kept

    try:
        result = await run_in_threadpool(_remove)
    except Exception:
        logger.warning("chat_history.sandbox_cleanup_failed", exc_info = True)
        return 0, []
    # Whatever this delete asked for: the last chat referencing a workspace the
    # user already asked to delete can go through the plain path, and only the
    # records marked pending are ever collected.
    from core.inference.tools import collect_orphaned_project_workspaces

    await run_in_threadpool(collect_orphaned_project_workspaces)
    return result


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


@router.delete("/projects/{project_id}", response_model = ChatProjectDeleted)
async def delete_project(
    project_id: str,
    request: Request,
    delete_files: bool = Query(False),
    current_subject: str = Depends(get_current_subject),
):
    # Rows first, files last: a member chat can still be running a tool in the
    # workspace, and its cwd disappearing mid-call either kills the call or
    # leaves what it writes next in a directory no project owns.
    project = delete_chat_project(project_id, delete_files = False)
    if project is None:
        raise HTTPException(
            status_code = 404,
            detail = f"Project {project_id} not found",
        )
    # The transaction is the only authority on membership and it runs first, so
    # a chat moved in just before is deleted and one moved out survives. An
    # earlier listing would stop a chat that is still there.
    member_ids = list(project.get("memberIds") or [])
    # By run id: the rows are gone by now, so there is nothing left to look up.
    _cancel_research_runs(request, list(project.get("activeResearchRunIds") or []))
    _cancel_active_generations(member_ids)
    if project.get("sandboxPath"):
        from starlette.concurrency import run_in_threadpool

        from core.inference.tools import (
            finish_workspace_delete_when_idle,
            forget_orphaned_project_if_gone,
            project_session_id,
            record_orphaned_project,
            wait_for_sessions_idle,
        )
        from storage.studio_db import (
            delete_project_workspace,
            sandbox_is_referenced_elsewhere,
        )

        # Cancelling only asks: a call already in the executor still has its
        # cwd in there, and removing it kills the call or strands what it
        # writes next. The shared id first, since a call in a project runs as
        # `project-<id>` and waiting on the member ids alone returned at once.
        shared = project_session_id(project_id)
        idle = (
            await run_in_threadpool(wait_for_sessions_idle, [shared, *member_ids])
            if delete_files
            else True
        )
        # A chat forked out of the project still shows cards for the shared
        # workspace, and the fork is not one of the ids deleted here.
        referenced = await run_in_threadpool(sandbox_is_referenced_elsewhere, shared, None)
        # The row went first, so another client can create a project with this
        # id in the window. It resolves to the same default path, and a tool
        # call of its own may be writing in there right now.
        recreated = await run_in_threadpool(get_chat_project, project_id) is not None
        if not delete_files:
            # The files stay, so the only job here is making them reachable: the
            # row that held a custom path is gone, and a fork's cards still name
            # this session.
            await run_in_threadpool(
                record_orphaned_project,
                project_id,
                project["sandboxPath"],
                False,
                project.get("rootPath"),
            )
        elif recreated:
            logger.warning(
                "Kept project workspace %s: a project was created with that id",
                project_id,
            )
        elif not idle:
            # Still running after the wait. Removing a live tool call's working
            # directory is worse than keeping files the user asked to delete,
            # and the record below means the next delete can still collect them.
            logger.warning(
                "Kept project workspace %s: a tool call was still running in it",
                project_id,
            )
        elif referenced:
            logger.info(
                "Kept project workspace %s: a surviving chat still shows its files",
                project_id,
            )
        if delete_files and idle and not referenced and not recreated:
            # Written down first: the delete can decline an unexpected path or
            # stop at a locked file, and the row that knew where this workspace
            # lives has already gone. The record is the only way back to it.
            await run_in_threadpool(
                record_orphaned_project,
                project_id,
                project["sandboxPath"],
                True,
                project.get("rootPath"),
            )
            await run_in_threadpool(delete_project_workspace, project)
            await run_in_threadpool(
                forget_orphaned_project_if_gone,
                project_id,
                project["sandboxPath"],
                project.get("rootPath"),
            )
        elif delete_files and not recreated:
            # Written down so it can be resolved and later collected: the row
            # that knew where it lives is gone. The root as well, since the
            # deferred delete has to remove what the immediate one would. Not
            # for a recreated project: its own row knows where its files are.
            await run_in_threadpool(
                record_orphaned_project,
                project_id,
                project["sandboxPath"],
                True,
                project.get("rootPath"),
            )
            if not idle:
                # Nothing else would come back to it: the collection otherwise
                # waits for some later delete that may never happen.
                finish_workspace_delete_when_idle(project_id)
    # Each member chat had its own sandbox for anything it wrote before joining
    # the project, and deleting the project removes the only records of them.
    _, sandboxes_kept = await _remove_sandboxes(member_ids, delete_files)
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
    # Those folders are reachable from nothing now, so the caller is told which
    # ones survived and can offer the delete once.
    return ChatProjectDeleted(**project, sandboxes_kept = sandboxes_kept)


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
async def clear_history(
    request: Request,
    delete_files: bool = False,
    current_subject: str = Depends(get_current_subject),
):
    thread_ids = [thread["id"] for thread in list_chat_threads()]
    _cancel_active_research(request, thread_ids)
    _cancel_active_generations(thread_ids)
    # The clear reports what it deleted, which is what gets cleaned up: a thread
    # added between the listing above and the delete is gone too, and its
    # sandbox would otherwise be stranded.
    cleared, cleared_runs = clear_chat_history()
    # A chat started between the listing and the transaction is in `cleared`
    # but was never cancelled, and a generation still running would dispatch a
    # tool and rebuild the sandbox this call is about to remove.
    listed = set(thread_ids)
    late = [thread_id for thread_id in cleared if thread_id not in listed]
    if late:
        _cancel_active_generations(late)
    # By id: the rows went with the threads, so nothing can look them up now.
    _cancel_research_runs(request, cleared_runs)
    # "Clear all chats" is the common bulk delete, so it has to clean up the
    # same folders DELETE /threads does; otherwise every sandbox is stranded.
    # delete_files matches DELETE /threads: off by default, since the files are
    # the user's, but a caller clearing everything can ask for them too.
    removed, kept = await _remove_sandboxes(list(dict.fromkeys(thread_ids + cleared)), delete_files)
    return {"status": "deleted", "sandboxes_removed": removed, "sandboxes_kept": kept}


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
