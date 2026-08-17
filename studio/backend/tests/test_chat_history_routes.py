# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import inspect
import os
import re
import sqlite3
import sys
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from core.inference.llama_server_args import BATCH_MAX
from routes import chat_history


def _message(message_id: str, thread_id: str) -> chat_history.ChatMessage:
    return chat_history.ChatMessage(
        id = message_id,
        threadId = thread_id,
        parentId = None,
        role = "user",
        content = [{"type": "text", "text": "hello"}],
        createdAt = 1_700_000_000_000,
    )


def test_async_delete_handlers_dispatch_sqlite_to_the_threadpool():
    """Cleanup handlers may be async, but their blocking sqlite work must leave the event loop."""
    coroutine_handlers = sorted(
        route.endpoint.__name__
        for route in chat_history.router.routes
        if inspect.iscoroutinefunction(route.endpoint)
    )
    assert coroutine_handlers == ["clear_history", "delete_project", "delete_threads"]
    for handler in (
        chat_history.clear_history,
        chat_history.delete_project,
        chat_history.delete_threads,
    ):
        assert "run_in_threadpool" in inspect.getsource(handler)


def test_replace_thread_messages_rejects_body_thread_mismatch(monkeypatch):
    called = False

    def fake_get_chat_thread(thread_id: str):
        return {"id": thread_id}

    def fake_sync_chat_messages(*args, **kwargs):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(chat_history, "get_chat_thread", fake_get_chat_thread)
    monkeypatch.setattr(chat_history, "sync_chat_messages", fake_sync_chat_messages)

    with pytest.raises(HTTPException) as exc_info:
        chat_history.replace_thread_messages(
            "thread-1",
            chat_history.ChatMessageSyncRequest(
                messages = [_message("msg-1", "thread-2")],
                pruneMissing = True,
            ),
            current_subject = "test-user",
        )

    assert exc_info.value.status_code == 400
    assert "Message threadId mismatch" in str(exc_info.value.detail)
    assert called is False


def test_replace_thread_messages_reports_protected_research_turn(monkeypatch):
    monkeypatch.setattr(chat_history, "get_chat_thread", lambda _thread_id: {"id": "thread-1"})

    def reject_prune(*_args, **_kwargs):
        raise chat_history.ChatMessageProtectedError(
            "Research prompts and responses cannot be deleted from their original thread"
        )

    monkeypatch.setattr(chat_history, "sync_chat_messages", reject_prune)

    with pytest.raises(HTTPException) as exc_info:
        chat_history.replace_thread_messages(
            "thread-1",
            chat_history.ChatMessageSyncRequest(messages = [], pruneMissing = True),
            current_subject = "test-user",
        )

    assert exc_info.value.status_code == 409
    assert "Research prompts and responses" in str(exc_info.value.detail)


def test_save_thread_message_returns_404_when_thread_is_deleted_during_write(monkeypatch):
    parent_reads = iter(({"id": "thread-1"}, None))
    monkeypatch.setattr(chat_history, "get_chat_thread", lambda _thread_id: next(parent_reads))

    def missing_parent(*_args, **_kwargs):
        raise sqlite3.IntegrityError("FOREIGN KEY constraint failed")

    monkeypatch.setattr(chat_history, "upsert_chat_message", missing_parent)

    with pytest.raises(HTTPException) as exc_info:
        chat_history.save_thread_message(
            "thread-1",
            "msg-1",
            _message("msg-1", "thread-1"),
            current_subject = "test-user",
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Thread thread-1 not found"


def test_replace_thread_messages_returns_404_when_thread_is_deleted_during_write(monkeypatch):
    parent_reads = iter(({"id": "thread-1"}, None))
    monkeypatch.setattr(chat_history, "get_chat_thread", lambda _thread_id: next(parent_reads))

    def missing_parent(*_args, **_kwargs):
        raise sqlite3.IntegrityError("FOREIGN KEY constraint failed")

    monkeypatch.setattr(chat_history, "sync_chat_messages", missing_parent)

    with pytest.raises(HTTPException) as exc_info:
        chat_history.replace_thread_messages(
            "thread-1",
            chat_history.ChatMessageSyncRequest(
                messages = [_message("msg-1", "thread-1")],
                pruneMissing = True,
            ),
            current_subject = "test-user",
        )

    assert exc_info.value.status_code == 404
    assert exc_info.value.detail == "Thread thread-1 not found"


def test_save_thread_message_does_not_mask_an_unrelated_integrity_error(monkeypatch):
    monkeypatch.setattr(
        chat_history,
        "get_chat_thread",
        lambda _thread_id: {"id": "thread-1"},
    )
    failure = sqlite3.IntegrityError("unrelated constraint")

    def raise_unrelated(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(chat_history, "upsert_chat_message", raise_unrelated)

    with pytest.raises(sqlite3.IntegrityError) as exc_info:
        chat_history.save_thread_message(
            "thread-1",
            "msg-1",
            _message("msg-1", "thread-1"),
            current_subject = "test-user",
        )

    assert exc_info.value is failure


def test_save_thread_distinguishes_a_tombstone_from_an_unknown_id(monkeypatch):
    def reject_deleted_thread(_thread):
        raise chat_history.ChatThreadDeletedError("thread-1")

    monkeypatch.setattr(chat_history, "upsert_chat_thread", reject_deleted_thread)
    payload = chat_history.ChatThread(
        id = "thread-1",
        title = "Deleted",
        modelType = "base",
        modelId = "model-1",
        createdAt = 1,
    )

    with pytest.raises(HTTPException) as exc_info:
        chat_history.save_thread(payload, current_subject = "test-user")

    assert exc_info.value.status_code == 410
    assert exc_info.value.detail == "Thread thread-1 was deleted"


def test_clear_history_fences_pending_thread_ids(monkeypatch):
    captured: list[str] = []
    captured_operation_ids: list[str | None] = []

    def clear_with_ids(thread_ids = (), operation_id = None):
        captured.extend(thread_ids)
        captured_operation_ids.append(operation_id)
        return list(thread_ids), []

    async def remove_sandboxes(_thread_ids, _delete_files):
        return 0, []

    monkeypatch.setattr(chat_history, "clear_chat_history", clear_with_ids)
    monkeypatch.setattr(chat_history, "_remove_sandboxes", remove_sandboxes)
    monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda _ids: None)
    monkeypatch.setattr(chat_history, "_cancel_research_runs", lambda _request, _ids: None)
    request = SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace()))

    response = asyncio.run(
        chat_history.clear_history(
            request,
            chat_history.ChatClearRequest(ids = ["pending-thread"], operationId = "clear-operation-1"),
            current_subject = "test-user",
        )
    )

    assert response == {
        "status": "deleted",
        "deletedThreadIds": ["pending-thread"],
        "sandboxes_removed": 0,
        "sandboxes_kept": [],
    }
    assert captured == ["pending-thread"]
    assert captured_operation_ids == ["clear-operation-1"]


def test_project_delete_cancels_research_before_workspace_cleanup(monkeypatch):
    project = {
        "id": "project-1",
        "name": "Project",
        "createdAt": 1,
        "updatedAt": 1,
        "memberIds": ["thread-1"],
        "activeResearchRunIds": ["run-1"],
    }
    cancelled: list[str] = []
    monkeypatch.setattr(
        chat_history,
        "delete_chat_project",
        lambda _project_id, delete_files = False: project,
    )
    monkeypatch.setattr(
        chat_history,
        "_cancel_research_runs",
        lambda _request, run_ids: cancelled.extend(run_ids),
    )
    monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda _ids: None)

    async def fail_workspace_cleanup(_ids, _delete_files):
        raise OSError("workspace is busy")

    monkeypatch.setattr(chat_history, "_remove_sandboxes", fail_workspace_cleanup)

    with pytest.raises(OSError, match = "workspace is busy"):
        asyncio.run(
            chat_history.delete_project(
                "project-1",
                SimpleNamespace(),
                delete_files = True,
                current_subject = "test-user",
            )
        )

    assert cancelled == ["run-1"]


# ---------------------------------------------------------------------------
# /api/chat/settings
# ---------------------------------------------------------------------------


def test_chat_settings_payload_accepts_fast_mode_presets():
    payload = chat_history.ChatSettingsPayload.model_validate(
        {
            "inferenceParams": {"fastMode": False},
            "customPresets": [
                {
                    "name": "Fast Opus",
                    "params": {
                        "temperature": 0.6,
                        "topP": 0.95,
                        "topK": 20,
                        "minP": 0.01,
                        "repetitionPenalty": 1.0,
                        "presencePenalty": 0.0,
                        "maxTokens": 8192,
                        "systemPrompt": "",
                        "trustRemoteCode": False,
                        "fastMode": True,
                    },
                },
            ],
        }
    )

    dumped = payload.model_dump(exclude_unset = True)
    assert dumped["inferenceParams"]["fastMode"] is False
    assert dumped["customPresets"][0]["params"]["fastMode"] is True


def test_chat_settings_payload_carries_per_model_params():
    """The payload is extra="forbid", so per-model memory only reaches the DB if the
    schema names both keys. A patch also has to survive exclude_unset with just the
    one model it touched, since that is what keeps other models' settings intact."""
    payload = chat_history.ChatSettingsPayload.model_validate(
        {
            "rememberParamsPerModel": True,
            "inferenceParamsByModel": {
                "unsloth/Qwen3.5-9B-GGUF": {"temperature": 0.2, "maxTokens": 4096},
            },
        }
    )

    dumped = payload.model_dump(exclude_unset = True)
    assert dumped["rememberParamsPerModel"] is True
    assert dumped["inferenceParamsByModel"] == {
        "unsloth/Qwen3.5-9B-GGUF": {"temperature": 0.2, "maxTokens": 4096},
    }
    # Nothing else is implied by the patch, so the merge cannot clobber it.
    assert "inferenceParams" not in dumped


def test_chat_settings_payload_rejects_junk_per_model_params():
    """Provider-qualified ids are opaque keys, but the values are still real
    inference settings -- an unknown field must 400 rather than reach the DB."""
    with pytest.raises(ValidationError):
        chat_history.ChatSettingsPayload.model_validate(
            {"inferenceParamsByModel": {"openai:gpt-x": {"notAParam": 1}}}
        )


def test_chat_settings_payload_accepts_preset_load_config():
    payload = chat_history.ChatSettingsPayload.model_validate(
        {
            "customPresets": [
                {
                    "name": "GGUF preset",
                    "params": {"temperature": 0.7, "maxTokens": 512},
                    "loadConfig": {
                        "customContextLength": 256,
                        "kvCacheDtype": "q8_0",
                        "tensorParallel": False,
                    },
                },
            ],
        }
    )

    dumped = payload.model_dump(exclude_unset = True)
    assert dumped["customPresets"][0]["loadConfig"]["customContextLength"] == 256
    assert dumped["customPresets"][0]["loadConfig"]["kvCacheDtype"] == "q8_0"


def test_chat_settings_payload_accepts_preset_batch_sizes():
    from pydantic import ValidationError

    # extra="forbid" 400s the whole settings write, and the normalizer emits both keys on
    # every preset (null included), so a preset that only pinned nParallel would stop saving.
    payload = chat_history.ChatSettingsPayload.model_validate(
        {
            "customPresets": [
                {
                    "name": "batch preset",
                    "params": {"temperature": 0.7},
                    "loadConfig": {"nParallel": 4, "nBatch": 4096, "nUbatch": 1024},
                },
            ],
        }
    )
    dumped = payload.model_dump(exclude_unset = True)
    assert dumped["customPresets"][0]["loadConfig"]["nBatch"] == 4096
    assert dumped["customPresets"][0]["loadConfig"]["nUbatch"] == 1024

    # The unset shape the normalizer sends alongside an untouched knob.
    chat_history.ChatPresetLoadConfig.model_validate(
        {"nParallel": 4, "nBatch": None, "nUbatch": None}
    )
    for bad in ({"nBatch": 0}, {"nUbatch": BATCH_MAX + 1}, {"nBatch": True}):
        with pytest.raises(ValidationError):
            chat_history.ChatPresetLoadConfig.model_validate(bad)


def test_chat_settings_payload_accepts_mlx_kv_bits():
    from pydantic import ValidationError

    # extra="forbid" rejects the whole settings write on an undeclared key.
    payload = chat_history.ChatSettingsPayload.model_validate(
        {
            "customPresets": [
                {
                    "name": "MLX preset",
                    "params": {"temperature": 0.7},
                    "loadConfig": {"mlxKvBits": 8},
                },
            ],
        }
    )
    dumped = payload.model_dump(exclude_unset = True)
    assert dumped["customPresets"][0]["loadConfig"]["mlxKvBits"] == 8

    for width in (4, None):
        chat_history.ChatPresetLoadConfig.model_validate({"mlxKvBits": width})
    # Only the widths MLX supports.
    with pytest.raises(ValidationError):
        chat_history.ChatPresetLoadConfig.model_validate({"mlxKvBits": 7})


def test_chat_settings_payload_accepts_nudge_tool_calls():
    # extra="forbid" 400s PUT /api/chat/settings on unknown keys, so the
    # frontend's persisted nudgeToolCalls needs a payload field (like
    # autoHealToolCalls).
    payload = chat_history.ChatSettingsPayload.model_validate(
        {"autoHealToolCalls": True, "nudgeToolCalls": False}
    )
    dumped = payload.model_dump(exclude_unset = True)
    assert dumped == {"autoHealToolCalls": True, "nudgeToolCalls": False}


def test_chat_inference_settings_covers_frontend_persisted_fields():
    # Drift guard: every InferenceParams field the UI persists (all but
    # checkpoint) must exist on ChatInferenceSettings, else extra="forbid"
    # 400s PUT /api/chat/settings on the next added field (issue #5862).
    runtime_ts = os.path.join(
        _backend,
        "..",
        "frontend",
        "src",
        "features",
        "chat",
        "types",
        "runtime.ts",
    )
    if not os.path.exists(runtime_ts):
        pytest.skip("frontend runtime.ts not present")

    with open(runtime_ts, encoding = "utf-8") as fh:
        block = re.search(r"interface InferenceParams \{(.*?)\n\}", fh.read(), re.DOTALL)
    assert block, "InferenceParams interface not found in runtime.ts"
    persisted = set(re.findall(r"^\s*(\w+)\??:", block.group(1), re.M)) - {"checkpoint"}

    backend = set(chat_history.ChatInferenceSettings.model_fields)
    assert (
        persisted == backend
    ), f"schema drift: frontend-only {persisted - backend}, backend-only {backend - persisted}"


# ---------------------------------------------------------------------------
# /api/chat/import-ledger
# ---------------------------------------------------------------------------


def test_get_import_ledger_round_trips_through_storage(monkeypatch):
    seen: list[str] = []

    def fake_list():
        return list(seen)

    monkeypatch.setattr(chat_history, "list_chat_legacy_imports", fake_list)

    response = chat_history.get_import_ledger(current_subject = "test-user")
    assert response.threadIds == []

    seen.extend(["legacy-a", "legacy-b"])
    response = chat_history.get_import_ledger(current_subject = "test-user")
    assert response.threadIds == ["legacy-a", "legacy-b"]


def test_record_import_ledger_returns_accepted_and_inserted(monkeypatch):
    captured: list[list[str]] = []

    def fake_upsert(thread_ids):
        captured.append(list(thread_ids))
        # Pretend two of the three were already in the ledger.
        return (len(thread_ids), max(0, len(thread_ids) - 2))

    monkeypatch.setattr(chat_history, "upsert_chat_legacy_imports", fake_upsert)

    response = chat_history.record_import_ledger(
        payload = chat_history.ChatImportLedgerRecordRequest(
            threadIds = ["a", "b", "c"],
        ),
        current_subject = "test-user",
    )
    assert response.accepted == 3
    assert response.inserted == 1
    assert captured == [["a", "b", "c"]]


def test_record_import_ledger_rejects_oversize_payload():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        chat_history.ChatImportLedgerRecordRequest(
            threadIds = [f"id-{i}" for i in range(10_001)],
        )


# ---------------------------------------------------------------------------
# /api/chat/threads/{id}/fork
# ---------------------------------------------------------------------------


def test_fork_thread_404_when_source_missing(monkeypatch):
    monkeypatch.setattr(chat_history, "get_chat_thread", lambda _id: None)
    with pytest.raises(HTTPException) as exc:
        chat_history.fork_thread(
            thread_id = "missing",
            payload = chat_history.ChatForkRequest(
                messageId = "m1",
                newThreadId = "new",
                createdAt = 1,
            ),
            current_subject = "test-user",
        )
    assert exc.value.status_code == 404


def test_fork_thread_404_when_branch_message_missing(monkeypatch):
    monkeypatch.setattr(chat_history, "get_chat_thread", lambda _id: {"id": _id, "title": "T"})
    monkeypatch.setattr(chat_history, "get_chat_message", lambda _t, _m: None)
    with pytest.raises(HTTPException) as exc:
        chat_history.fork_thread(
            thread_id = "src",
            payload = chat_history.ChatForkRequest(
                messageId = "missing",
                newThreadId = "new",
                createdAt = 1,
            ),
            current_subject = "test-user",
        )
    assert exc.value.status_code == 404


def test_fork_thread_happy_path(monkeypatch):
    source = {
        "id": "src",
        "title": "Original",
        "modelType": "base",
        "modelId": "m",
        "pairId": None,
        "archived": False,
        "createdAt": 1,
        "openaiCodeExecContainerId": None,
        "anthropicCodeExecContainerId": None,
        "forkedFromThreadId": None,
        "forkedFromMessageId": None,
    }
    forked = {
        **source,
        "id": "new",
        "title": "fork · Original",
        "createdAt": 2,
        "forkedFromThreadId": "src",
        "forkedFromMessageId": "m1",
    }
    monkeypatch.setattr(chat_history, "get_chat_thread", lambda _id: source)
    monkeypatch.setattr(
        chat_history,
        "get_chat_message",
        lambda _t, _m: {
            "id": _m,
            "threadId": _t,
            "role": "user",
            "content": [],
            "createdAt": 1,
        },
    )
    monkeypatch.setattr(chat_history, "fork_chat_thread", lambda **_: forked)
    monkeypatch.setattr(
        chat_history,
        "list_chat_messages",
        lambda _id: [
            {
                "id": "n1",
                "threadId": "new",
                "parentId": None,
                "role": "user",
                "content": [],
                "createdAt": 1,
            }
        ],
    )
    response = chat_history.fork_thread(
        thread_id = "src",
        payload = chat_history.ChatForkRequest(
            messageId = "m1",
            newThreadId = "new",
            createdAt = 2,
        ),
        current_subject = "test-user",
    )
    assert response.thread.id == "new"
    assert response.thread.title == "fork · Original"
    assert response.thread.forkedFromThreadId == "src"
    assert response.thread.forkedFromMessageId == "m1"
    assert len(response.messages) == 1
    assert response.containerSnapshotWarning is None


def test_fork_thread_warns_when_parent_had_container(monkeypatch):
    source = {
        "id": "src",
        "title": "T",
        "modelType": "base",
        "modelId": "",
        "pairId": None,
        "archived": False,
        "createdAt": 1,
        "openaiCodeExecContainerId": "cnt_123",
        "anthropicCodeExecContainerId": None,
        "forkedFromThreadId": None,
        "forkedFromMessageId": None,
    }
    monkeypatch.setattr(chat_history, "get_chat_thread", lambda _id: source)
    monkeypatch.setattr(
        chat_history,
        "get_chat_message",
        lambda _t, _m: {
            "id": _m,
            "threadId": _t,
            "role": "user",
            "content": [],
            "createdAt": 1,
        },
    )
    monkeypatch.setattr(
        chat_history,
        "fork_chat_thread",
        lambda **_: {
            **source,
            "id": "new",
            "title": "fork · T",
            "forkedFromThreadId": "src",
            "forkedFromMessageId": "m1",
            "openaiCodeExecContainerId": None,
        },
    )
    monkeypatch.setattr(chat_history, "list_chat_messages", lambda _id: [])
    response = chat_history.fork_thread(
        thread_id = "src",
        payload = chat_history.ChatForkRequest(
            messageId = "m1",
            newThreadId = "new",
            createdAt = 2,
        ),
        current_subject = "test-user",
    )
    assert response.containerSnapshotWarning is not None
    assert "fresh" in response.containerSnapshotWarning.lower()


def test_get_fork_count(monkeypatch):
    monkeypatch.setattr(chat_history, "count_forks_for_message", lambda _t, _m: 3)
    response = chat_history.get_fork_count(
        thread_id = "t",
        message_id = "m",
        current_subject = "test-user",
    )
    assert response.count == 3
