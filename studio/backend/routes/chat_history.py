# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Chat history API routes backed by studio.db.

SQLite-backed handlers are sync defs unless they also perform asynchronous cleanup. Those
mixed handlers explicitly send their database transaction through Starlette's threadpool.
"""

import asyncio
import sqlite3
from typing import Annotated, Any, Literal, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from auth.authentication import get_current_subject
from core.inference.llama_server_args import BATCH_MAX, BATCH_MIN, PARALLEL_MAX, PARALLEL_MIN
from loggers import get_logger
from utils.api_errors import safe_validation_errors
from utils.utils import safe_curated_detail, log_and_http_error
from storage.studio_db import (
    ChatMessageConflictError,
    ChatMessageProtectedError,
    ChatThreadDeletedError,
    ChatThreadPreconditionFailed,
    CorruptSettingsError,
    ProjectWorkspaceError,
    build_chat_history_export,
    clear_chat_history,
    count_chat_threads,
    count_forks_for_message,
    delete_chat_attachment,
    delete_chat_project,
    delete_chat_threads,
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
    write_chat_thread_settings,
    upsert_chat_thread,
)

router = APIRouter()

logger = get_logger(__name__)


class ChatRagThreadSource(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    type: Literal["thread"]


class ChatRagKnowledgeBaseSource(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    type: Literal["kb"]
    kbId: str = Field(min_length = 1, max_length = 256)


class ChatThreadSettings(BaseModel):
    """The chat settings captured per thread; a thread storing none uses the global ones."""

    model_config = ConfigDict(extra = "forbid")

    reasoningEnabled: Optional[bool] = None
    reasoningEffort: Optional[
        Literal["none", "minimal", "low", "medium", "high", "max", "xhigh"]
    ] = None
    toolsEnabled: Optional[bool] = None
    codeToolsEnabled: Optional[bool] = None
    imageToolsEnabled: Optional[bool] = None
    webFetchToolsEnabled: Optional[bool] = None
    deepResearchEnabled: Optional[bool] = None
    artifactsEnabled: Optional[bool] = None
    mcpEnabledForChat: Optional[bool] = None
    # "full" (Full access) is session-only and never persisted, per thread or globally.
    permissionMode: Optional[Literal["ask", "auto", "off"]] = None
    ragEnabled: Optional[bool] = None
    ragSource: Optional[
        Annotated[
            Union[ChatRagThreadSource, ChatRagKnowledgeBaseSource],
            Field(discriminator = "type"),
        ]
    ] = None
    ragMode: Optional[Literal["hybrid", "lexical", "dense"]] = None
    # Matches the ge/le the retrieval endpoint enforces on its own top_k.
    ragTopK: Optional[int] = Field(default = None, ge = 1, le = 50)
    ragAutoInject: Optional[Literal["auto", "on", "off"]] = None
    ragAutoInjectMinScore: Optional[float] = Field(default = None, ge = 0, le = 1)


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
    settings: Optional[ChatThreadSettings] = None


def thread_from_row(row: dict) -> ChatThread:
    """Build a ChatThread from a DATABASE row, tolerating a snapshot it cannot read.

    `settings` is the first strictly validated nested model Studio builds out of the
    database rather than off the wire, and a stored snapshot outlives the build that
    wrote it: a newer Studio adding a seventeenth setting, widening an enum or
    raising a bound writes a blob this one rejects. Refusing it here 500s the chat on
    open and takes the entire history export with it, since the export validates
    every thread. `_json_loads` already shrugs off JSON that will not parse; JSON
    that parses but postdates this build deserves the same treatment.

    Only the read is forgiving. The wire contract stays strict in both directions, so
    a client still cannot invent a setting, and the row itself is left untouched,
    which means upgrading again restores whatever this build had to drop.
    """
    settings = row.get("settings")
    if isinstance(settings, dict):
        row = {**row, "settings": readable_thread_settings(settings)}
    elif settings is not None:
        row = {**row, "settings": None}
    return ChatThread(**row)


def readable_thread_settings(settings: dict) -> Optional[dict]:
    """The part of a stored snapshot this build can validate, or None if none of it is."""
    known = {k: v for k, v in settings.items() if k in ChatThreadSettings.model_fields}
    # Drop exactly the fields pydantic names and retry: a version gap usually
    # carries several at once, so removing one guess at a time gives up too early.
    for _ in range(len(known) + 1):
        try:
            ChatThreadSettings.model_validate(known)
            return known
        except ValidationError as exc:
            bad = {str(e["loc"][0]) for e in exc.errors() if e.get("loc")}
            if not bad or not bad & set(known):
                return None
            known = {k: v for k, v in known.items() if k not in bad}
    return None


def _unreadable_thread_settings(stored: dict) -> dict:
    """The part of a stored snapshot this build cannot validate, and so must not delete.

    An older Studio opening a database a newer one wrote drops the fields it cannot read.
    A blind replacement would make that loss permanent instead of temporary, so a write
    carries forward everything the writer could not have known about: unknown keys, and
    known keys holding values this build rejects.
    """
    readable = readable_thread_settings(stored) or {}
    return {k: v for k, v in stored.items() if k not in readable}


def _settings_write_from_patch(patch: dict) -> Optional[dict]:
    """Take `settings` / `settingsPatch` out of `patch` and describe the write they ask for.

    `settings` replaces the snapshot, `settingsPatch` applies only the fields it names,
    for a client that knows what changed but not what else the row holds. The result is
    handed to storage rather than executed here: the read, the merge and the guarded
    metadata write all have to be one transaction, or two tabs each build a replacement
    from the same stale row, or a rejected precondition returns 409 having already
    committed the settings.
    """
    replace = "settings" in patch
    merge = "settingsPatch" in patch
    seq = patch.pop("settingsSeq", None)
    writer = patch.pop("settingsWriter", None)
    if not (replace or merge):
        return None
    incoming = patch.pop("settingsPatch", None)
    if merge:
        # A merge is the more specific instruction; sending both is a client bug.
        patch.pop("settings", None)
    else:
        incoming = patch.pop("settings")
    if incoming is None:
        # Clearing is the one instruction that means the whole column. A merge of
        # nothing is not an instruction at all, so it leaves the row alone.
        if replace and not merge:
            return {"clear": True, "seq": seq, "writer": writer}
        return None
    return {
        "merge": incoming if merge else None,
        "replace": None if merge else incoming,
        "seq": seq,
        "writer": writer,
        "keep_unreadable": _unreadable_thread_settings,
    }


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
    # Replaces the whole snapshot, except for anything the client could not read.
    settings: Optional[ChatThreadSettings] = None
    # Applies just the fields it names. For the writer that knows what changed but not
    # what else the row holds, which is any write made before the row has been read.
    settingsPatch: Optional[ChatThreadSettings] = None
    # Orders this writer's snapshot writes against its OWN earlier ones, so a keepalive
    # sent on unload cannot be undone by a PATCH the server already had in hand. Never
    # compared across writers: two browsers' counters mean nothing to each other.
    settingsSeq: Optional[int] = None
    settingsWriter: Optional[str] = None


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


class ChatClearRequest(BaseModel):
    # The client fences every legacy Dexie thread it holds, so this bound has to sit above what
    # a migrated install can legitimately collect: a 422 here fails the whole clear, and the
    # identical retry fails with it, leaving backend history behind after Clear all.
    ids: list[str] = Field(default_factory = list, max_length = 200_000)
    operationId: Optional[str] = Field(default = None, min_length = 1, max_length = 128)


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
    # allow_inf_nan: json.loads accepts bare NaN and Infinity, and pydantic takes them
    # for a float, so `{"temperature": NaN}` used to be stored as a bare NaN token in
    # value_json. Python reads that back, so the row is never quarantined, while the
    # response model renders it as null: the value is silently lost and the row is not
    # valid JSON for any strict reader. Refuse it at the door instead, the way
    # models/training.py already does for every numeric training field.
    model_config = ConfigDict(extra = "forbid", allow_inf_nan = False)

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
    model_config = ConfigDict(extra = "forbid", allow_inf_nan = False)

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


class ChatResearchWebsitePolicy(BaseModel):
    model_config = ConfigDict(extra = "forbid")

    # 253 is the maximum length of a DNS name.
    allowedDomains: list[Annotated[str, Field(max_length = 253)]] = Field(
        default_factory = list, max_length = 1_000
    )
    blockedDomains: list[Annotated[str, Field(max_length = 253)]] = Field(
        default_factory = list, max_length = 1_000
    )


class ChatSettingsPayload(BaseModel):
    model_config = ConfigDict(extra = "forbid", allow_inf_nan = False)

    inferenceParams: Optional[ChatInferenceSettings] = None
    # Last-used params per checkpoint id. Deep-merged per key, so patching one
    # model cannot drop another's.
    inferenceParamsByModel: Optional[dict[str, ChatInferenceSettings]] = None
    rememberParamsPerModel: Optional[bool] = None
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

    # Composer and RAG toggles. They describe the installation, not the browser
    # that set them, so a second browser or a remote session reads them back here
    # instead of falling back to defaults.
    reasoningEnabled: Optional[bool] = None
    toolsEnabled: Optional[bool] = None
    codeToolsEnabled: Optional[bool] = None
    imageToolsEnabled: Optional[bool] = None
    webFetchToolsEnabled: Optional[bool] = None
    deepResearchEnabled: Optional[bool] = None
    researchWebsitePolicy: Optional[ChatResearchWebsitePolicy] = None
    artifactsEnabled: Optional[bool] = None
    showCanvasMenuItem: Optional[bool] = None
    mcpEnabledForChat: Optional[bool] = None
    confirmToolCalls: Optional[bool] = None
    # "full" (Full access) is session-only by design and never persisted.
    permissionMode: Optional[Literal["ask", "auto", "off"]] = None
    ragSource: Optional[
        Annotated[
            Union[ChatRagThreadSource, ChatRagKnowledgeBaseSource],
            Field(discriminator = "type"),
        ]
    ] = None
    ragMode: Optional[Literal["hybrid", "lexical", "dense"]] = None
    # Matches the ge/le the retrieval endpoint enforces on its own top_k.
    ragTopK: Optional[int] = Field(default = None, ge = 1, le = 50)
    ragAutoInject: Optional[Literal["auto", "on", "off"]] = None
    ragAutoInjectMinScore: Optional[float] = Field(default = None, ge = 0, le = 1)
    ragOcrScanned: Optional[bool] = None
    ragCaptionFigures: Optional[bool] = None
    # Standing load preferences the model-load path reads outside the store.
    speculativeType: Optional[Literal["auto", "ngram", "off"]] = None
    gpuMemoryMode: Optional[Literal["auto", "manual"]] = None
    expandQuantizations: Optional[bool] = None
    showAllQuantizations: Optional[bool] = None
    fitOnDeviceOnly: Optional[bool] = None


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
def list_threads(
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
    return ChatThreadListResponse(threads = [thread_from_row(t) for t in threads])


def _missing_project_error(project_id: Optional[str]) -> HTTPException:
    """The row references a project that is gone, whether the check or the write noticed it."""
    return HTTPException(status_code = 404, detail = f"Project {project_id} not found")


def _missing_thread_error(thread_id: str) -> HTTPException:
    return HTTPException(status_code = 404, detail = f"Thread {thread_id} not found")


def _deleted_thread_error(thread_id: str) -> HTTPException:
    return HTTPException(status_code = 410, detail = f"Thread {thread_id} was deleted")


@router.post("/threads", response_model = ChatThread)
def save_thread(payload: ChatThread, current_subject: str = Depends(get_current_subject)):
    if payload.projectId and get_chat_project(payload.projectId) is None:
        raise _missing_project_error(payload.projectId)
    try:
        return thread_from_row(upsert_chat_thread(payload.model_dump()))
    except ChatThreadDeletedError as exc:
        raise _deleted_thread_error(payload.id) from exc
    except sqlite3.IntegrityError as exc:
        # The project can be deleted between the check above and this insert, and the foreign key
        # then fails. Report the same 404 rather than surfacing a 500.
        if not payload.projectId:
            raise
        raise _missing_project_error(payload.projectId) from exc


@router.get("/threads/{thread_id}", response_model = ChatThread)
def get_thread(thread_id: str, current_subject: str = Depends(get_current_subject)):
    thread = get_chat_thread(thread_id)
    if thread is None:
        raise HTTPException(status_code = 404, detail = f"Thread {thread_id} not found")
    return thread_from_row(thread)


@router.patch("/threads/{thread_id}", response_model = ChatThread)
def patch_thread(
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
        raise _missing_project_error(patch["projectId"])
    settings_write = _settings_write_from_patch(patch)
    try:
        thread = update_chat_thread(
            thread_id,
            patch,
            expected_title = expected_title,
            expected_opening_message_id = expected_opening_message_id,
            settings_write = settings_write,
        )
    except sqlite3.IntegrityError as exc:
        # Same race as save_thread: the project can go away before this write lands.
        if not patch.get("projectId"):
            raise
        raise _missing_project_error(patch["projectId"]) from exc
    except ChatThreadPreconditionFailed:
        raise HTTPException(
            status_code = 409,
            detail = f"Thread {thread_id} changed since it was read",
        )
    if thread is None:
        raise HTTPException(status_code = 404, detail = f"Thread {thread_id} not found")
    return thread_from_row(thread)


def _cancel_deleted_research_runs(request: Request, run_ids: list[str]) -> None:
    """Signal workers for active runs captured by the deletion transaction."""
    supervisor = getattr(request.app.state, "research_supervisor", None)
    if supervisor is None:
        return
    for run_id in run_ids:
        try:
            supervisor.cancel(run_id)
        except Exception:  # noqa: BLE001 - cancellation is best-effort after commit
            logger.warning(
                "chat_history.cancel_deleted_research_failed run_id=%s",
                run_id,
                exc_info = True,
            )


def _cancel_active_research(request: Request, thread_ids: list[str]) -> None:
    """Compatibility path for callers that must cancel runs before deleting their rows."""
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
    from starlette.concurrency import run_in_threadpool

    deleted_research_run_ids = await run_in_threadpool(delete_chat_threads, payload.ids)
    _cancel_research_runs(request, deleted_research_run_ids)
    _cancel_active_generations(payload.ids)
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
            # Again, next to the removal: the reference scan above reads
            # every message, and another tab can recreate the chat while it runs.
            if get_chat_thread(thread_id) is not None:
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
def list_projects(
    include_archived: bool = Query(False), current_subject: str = Depends(get_current_subject)
):
    return ChatProjectListResponse(
        projects = [
            ChatProject(**(ensure_chat_project_workspace(project["id"]) or project))
            for project in list_chat_projects(include_archived = include_archived)
        ]
    )


@router.post("/projects", response_model = ChatProject)
def save_project(payload: ChatProject, current_subject: str = Depends(get_current_subject)):
    try:
        return ChatProject(**upsert_chat_project(payload.model_dump()))
    except ProjectWorkspaceError as exc:
        # A project is the only thing Studio writes to Documents, so a folder it
        # cannot create there fails here and nowhere else. Only this error, and
        # only its own path: the same upsert also opens the database, which
        # lives somewhere else entirely.
        raise log_and_http_error(
            exc,
            500,
            f"Could not create the project folder {exc.path}. Check that the "
            "folder is writable, or set UNSLOTH_STUDIO_PROJECTS_HOME to another "
            "location.",
            event = "chat_history.create_project_workspace_failed",
            log = logger,
        ) from exc


@router.get("/projects/{project_id}", response_model = ChatProject)
def get_project(project_id: str, current_subject: str = Depends(get_current_subject)):
    project = ensure_chat_project_workspace(project_id)
    if project is None:
        raise HTTPException(
            status_code = 404,
            detail = f"Project {project_id} not found",
        )
    return ChatProject(**project)


@router.patch("/projects/{project_id}", response_model = ChatProject)
def patch_project(
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


def _delete_project_rag_sources(project_id: str) -> None:
    """Retire an ownerless project scope and reap it when RAG is available."""
    from storage import rag_db
    from core.rag import folder_sync, store as rag_store

    scope = rag_store.project_scope(project_id)
    # a project id is reusable, and the tombstone outlives the scope, so retiring one that
    # another client already recreated would permanently disable RAG for the new project
    with folder_sync.scope_lock(scope):
        checked_at = folder_sync.now_iso()
        if get_chat_project(project_id) is not None:
            return
        folder_sync.retire_scope(scope, checked_at)
    if rag_db.rag_available():
        folder_sync.delete_retired_scope(scope)


@router.delete("/projects/{project_id}", response_model = ChatProjectDeleted)
async def delete_project(
    project_id: str,
    request: Request,
    delete_files: bool = Query(False),
    current_subject: str = Depends(get_current_subject),
):
    from starlette.concurrency import run_in_threadpool

    # Rows first, files last: a member chat can still be running a tool in the
    # workspace, and its cwd disappearing mid-call either kills the call or
    # leaves what it writes next in a directory no project owns.
    try:
        project = await run_in_threadpool(
            lambda: delete_chat_project(project_id, delete_files = False)
        )
    except Exception:
        # the row transaction may still have committed, and an ownerless scope has to be
        # retired by someone; periodic reconciliation is the fallback if this also fails
        try:
            if await run_in_threadpool(get_chat_project, project_id) is None:
                await run_in_threadpool(_delete_project_rag_sources, project_id)
        except Exception:  # noqa: BLE001 - preserve the original deletion error
            logger.warning(
                "failed to delete RAG sources for committed project %s", project_id, exc_info = True
            )
        raise
    if project is None:
        raise HTTPException(
            status_code = 404,
            detail = f"Project {project_id} not found",
        )
    # before any workspace work: the row is already gone, so a later failure must not
    # leave the scope owned by nothing
    try:
        await run_in_threadpool(_delete_project_rag_sources, project_id)
    except Exception:  # noqa: BLE001 - source cleanup must not block project deletion
        logger.warning("failed to delete RAG sources for project %s", project_id, exc_info = True)
    # The transaction is the only authority on membership and it runs first, so
    # a chat moved in just before is deleted and one moved out survives. An
    # earlier listing would stop a chat that is still there.
    member_ids = list(project.get("memberIds") or [])
    # By run id: the rows are gone by now, so there is nothing left to look up.
    _cancel_research_runs(request, list(project.get("activeResearchRunIds") or []))
    _cancel_active_generations(member_ids)
    if project.get("sandboxPath"):
        from core.inference.tools import (
            finish_workspace_delete_when_idle,
            forget_orphaned_project,
            forget_orphaned_project_if_gone,
            live_project_owns,
            project_session_id,
            record_orphaned_project,
            wait_for_sessions_idle,
        )
        from storage.studio_db import (
            delete_project_workspace,
            sandbox_is_referenced_elsewhere,
        )

        # Cancelling only asks: a call already in the executor still has its
        # cwd in there, and removing it strands what it writes next. The shared
        # id first, since a call in a project runs as `project-<id>`.
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
            # Once more, next to the delete itself: the record write above is
            # an await, and a project created in that window resolves to this
            # same path.
            if await run_in_threadpool(get_chat_project, project_id) is not None:
                logger.warning(
                    "Kept project workspace %s: a project was created with that id",
                    project_id,
                )
                # Only when the new row is about these folders: the default
                # root carries the project's name, so a project remade under
                # this id can sit elsewhere and the old one would be stranded.
                if await run_in_threadpool(
                    live_project_owns,
                    project_id,
                    project["sandboxPath"],
                    project.get("rootPath"),
                ):
                    await run_in_threadpool(forget_orphaned_project, project_id)
            else:
                await run_in_threadpool(delete_project_workspace, project)
                await run_in_threadpool(
                    forget_orphaned_project_if_gone,
                    project_id,
                    project["sandboxPath"],
                    project.get("rootPath"),
                )
        elif delete_files and not recreated:
            # Written down so it can be resolved and later collected: the row
            # that knew where it lives is gone. The root too, since the deferred
            # delete removes what the immediate one would; not for a live id.
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
    # Those folders are reachable from nothing now, so the caller is told which
    # ones survived and can offer the delete once.
    return ChatProjectDeleted(**project, sandboxes_kept = sandboxes_kept)


@router.get("/threads/{thread_id}/messages", response_model = ChatMessageListResponse)
def get_thread_messages(thread_id: str, current_subject: str = Depends(get_current_subject)):
    if get_chat_thread(thread_id) is None:
        raise HTTPException(status_code = 404, detail = f"Thread {thread_id} not found")
    return ChatMessageListResponse(
        messages = [ChatMessage(**m) for m in list_chat_messages(thread_id)]
    )


@router.post("/messages:batch", response_model = ChatMessagesBatchResponse)
def batch_thread_messages(
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
def get_thread_message(
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
    except sqlite3.IntegrityError as exc:
        if get_chat_thread(thread_id) is None:
            raise _missing_thread_error(thread_id) from exc
        raise
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
    except sqlite3.IntegrityError as exc:
        if get_chat_thread(thread_id) is None:
            raise _missing_thread_error(thread_id) from exc
        raise
    except (ChatMessageConflictError, ChatMessageProtectedError) as exc:
        raise log_and_http_error(
            exc,
            409,
            safe_curated_detail(exc),
            event = "chat_history.replace_messages_conflict",
            log = logger,
        ) from exc


@router.get("/count", response_model = ChatCountResponse)
def count_threads(current_subject: str = Depends(get_current_subject)):
    return ChatCountResponse(count = count_chat_threads())


@router.get("/import-ledger", response_model = ChatImportLedgerResponse)
def get_import_ledger(current_subject: str = Depends(get_current_subject)):
    """Legacy-Dexie import ledger: legacy thread ids already copied into chat tables.

    The frontend checks this on tab open to decide whether to re-run the Dexie -> studio.db import.
    """
    return ChatImportLedgerResponse(threadIds = list_chat_legacy_imports())


@router.post("/import-ledger", response_model = ChatImportLedgerRecordResponse)
def record_import_ledger(
    payload: ChatImportLedgerRecordRequest, current_subject: str = Depends(get_current_subject)
):
    """Mark each legacy thread id as imported. Idempotent."""
    accepted, inserted = upsert_chat_legacy_imports(payload.threadIds)
    return ChatImportLedgerRecordResponse(accepted = accepted, inserted = inserted)


@router.delete("")
async def clear_history(
    request: Request,
    payload: Optional[ChatClearRequest] = None,
    delete_files: bool = False,
    current_subject: str = Depends(get_current_subject),
):
    from starlette.concurrency import run_in_threadpool

    # Admission is already closed in the frontend. Include its pending and legacy ids in the
    # transaction's fence so a delayed POST cannot recreate a chat after this returns.
    thread_ids = (
        list(payload.ids)
        if payload is not None
        else [thread["id"] for thread in await run_in_threadpool(list_chat_threads)]
    )

    def _clear_rows() -> tuple[list[str], list[str]]:
        if payload is None:
            cleared, cleared_runs = clear_chat_history()
        else:
            cleared, cleared_runs = clear_chat_history(
                payload.ids,
                operation_id = payload.operationId,
            )
        return cleared, cleared_runs

    # The clear reports what it deleted, which is what gets cleaned up: a thread
    # added between the listing above and the delete is gone too, and its
    # sandbox would otherwise be stranded.
    cleared, cleared_runs = await run_in_threadpool(_clear_rows)
    # A chat started between the listing and the transaction is in `cleared`
    # but was never cancelled, and a generation still running would dispatch a
    # tool and rebuild the sandbox this call is about to remove.
    listed = set(thread_ids)
    late = [thread_id for thread_id in cleared if thread_id not in listed]
    _cancel_active_generations(thread_ids)
    if late:
        _cancel_active_generations(late)
    # By id: the rows went with the threads, so nothing can look them up now.
    _cancel_research_runs(request, cleared_runs)
    # "Clear all chats" is the common bulk delete, so it has to clean up the
    # same folders DELETE /threads does; otherwise every sandbox is stranded.
    # delete_files matches DELETE /threads: off by default, since the files are
    # the user's, but a caller clearing everything can ask for them too.
    removed, kept = await _remove_sandboxes(list(dict.fromkeys(thread_ids + cleared)), delete_files)
    return {
        "status": "deleted",
        "deletedThreadIds": cleared,
        "sandboxes_removed": removed,
        "sandboxes_kept": kept,
    }


@router.get("/settings", response_model = ChatSettingsResponse)
def get_settings(current_subject: str = Depends(get_current_subject)):
    return ChatSettingsResponse(settings = list_chat_settings())


@router.put("/settings", response_model = ChatSettingsResponse)
def put_settings(payload: dict[str, Any], current_subject: str = Depends(get_current_subject)):
    try:
        parsed = ChatSettingsPayload.model_validate(payload)
    except ValidationError as exc:
        # safe_validation_errors, not exc.errors(): the raw errors echo the offending
        # input, and Starlette's JSONResponse dumps with allow_nan = False, so a
        # rejected NaN or Infinity made the 400 handler itself unrenderable and the
        # caller got a 500 for a request the validator had already refused. It also
        # bounds a multi-megabyte value being quoted back.
        raise HTTPException(status_code = 400, detail = safe_validation_errors(exc.errors())) from exc
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
def fork_thread(
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
    try:
        forked = fork_chat_thread(
            source_thread_id = thread_id,
            branch_message_id = payload.messageId,
            new_thread_id = payload.newThreadId,
            new_title = new_title,
            created_at = payload.createdAt,
            id_factory = lambda: str(uuid.uuid4()),
        )
    except ChatThreadDeletedError as exc:
        raise _deleted_thread_error(payload.newThreadId) from exc
    if forked is None:
        # The source can be deleted between the reads above and the fork transaction, which the
        # threadpool lets run concurrently. Report it gone rather than as a server fault.
        raise HTTPException(status_code = 404, detail = f"Thread {thread_id} not found")
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
        thread = thread_from_row(forked),
        messages = [ChatMessage(**m) for m in messages],
        containerSnapshotWarning = warning,
    )


@router.get(
    "/threads/{thread_id}/messages/{message_id}/forks",
    response_model = ChatForkCountResponse,
)
def get_fork_count(
    thread_id: str,
    message_id: str,
    current_subject: str = Depends(get_current_subject),
):
    return ChatForkCountResponse(count = count_forks_for_message(thread_id, message_id))


@router.get("/export", response_model = ChatExportResponse)
def export_history(current_subject: str = Depends(get_current_subject)):
    from datetime import datetime, timezone
    projects, threads, messages = build_chat_history_export()
    return ChatExportResponse(
        exportedAt = datetime.now(timezone.utc).isoformat(),
        version = 1,
        threadCount = len(threads),
        projects = [ChatProject(**project) for project in projects],
        threads = [thread_from_row(thread) for thread in threads],
        messages = [ChatMessage(**message) for message in messages],
    )
