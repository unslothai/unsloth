# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import asyncio
import inspect
import os
import re
import sqlite3
import subprocess
import sys
import threading
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


def test_save_thread_message_forwards_explicit_generation_edit(monkeypatch):
    monkeypatch.setattr(chat_history, "get_chat_thread", lambda _thread_id: {"id": "thread-1"})
    captured = {}

    def save(message, *, allow_generation_edit = False):
        captured["allow_generation_edit"] = allow_generation_edit
        return message

    monkeypatch.setattr(chat_history, "upsert_chat_message", save)
    payload = _message("assistant-1", "thread-1").model_copy(update = {"role": "assistant"})
    chat_history.save_thread_message(
        "thread-1",
        "assistant-1",
        payload,
        allow_generation_edit = True,
        current_subject = "test-user",
    )
    assert captured == {"allow_generation_edit": True}


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


def test_chat_thread_payload_carries_gguf_variant():
    thread = chat_history.ChatThread(
        id = "thread-1",
        title = "GGUF chat",
        modelType = "base",
        modelId = "unsloth/Qwen3-GGUF",
        modelGgufVariant = "Q6_K",
        createdAt = 1,
    )
    patch = chat_history.ChatThreadPatch(modelGgufVariant = "Q8_0")

    assert thread.model_dump()["modelGgufVariant"] == "Q6_K"
    assert patch.model_dump(exclude_unset = True) == {"modelGgufVariant": "Q8_0"}


def test_clear_history_fences_pending_thread_ids(monkeypatch):
    captured: list[str] = []
    captured_operation_ids: list[str | None] = []

    def clear_with_ids(
        thread_ids = (),
        operation_id = None,
        include_chat_generation_runs = False,
    ):
        captured.extend(thread_ids)
        captured_operation_ids.append(operation_id)
        result = (list(thread_ids), [])
        return (*result, [], False) if include_chat_generation_runs else (*result, False)

    async def remove_sandboxes(_thread_ids, _delete_files):
        return 0, []

    monkeypatch.setattr(chat_history, "clear_chat_history_with_replay_status", clear_with_ids)
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


def test_clear_history_reaps_search_thumbnails_with_a_body(monkeypatch):
    """DELETE /api/chat is clear-all either way, and the frontend always sends a body.

    Gating the thumbnail reap on `payload is None` meant it never ran, so "Clear all
    chats" left every cached thumbnail (which says what was searched for) on disk.
    """
    from core.inference import search_images

    reaped: list[bool] = []
    monkeypatch.setattr(search_images, "clear_cache", lambda only_ids = None: reaped.append(True))

    async def remove_sandboxes(_thread_ids, _delete_files):
        return 0, []

    monkeypatch.setattr(
        chat_history,
        "clear_chat_history_with_replay_status",
        lambda thread_ids = (), operation_id = None, include_chat_generation_runs = False: (
            ([], [], [], False) if include_chat_generation_runs else ([], [], False)
        ),
    )
    monkeypatch.setattr(chat_history, "_remove_sandboxes", remove_sandboxes)
    monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda _ids: None)
    monkeypatch.setattr(chat_history, "_cancel_research_runs", lambda _request, _ids: None)
    monkeypatch.setattr(
        chat_history, "_remove_conversation_archives", lambda _ids, cutoff = None: None
    )
    request = SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace()))

    asyncio.run(
        chat_history.clear_history(
            request,
            chat_history.ChatClearRequest(ids = [], operationId = "clear-operation-2"),
            current_subject = "test-user",
        )
    )

    assert reaped == [True]


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


def test_project_delete_blocks_while_studio_worktree_is_active(tmp_path):
    from core.agent_workspace.state import save_worktree
    from storage import studio_db

    root = tmp_path / "repository"
    root.mkdir()
    project = studio_db.upsert_chat_project(
        {
            "id": "project-with-worktree",
            "name": "Project",
            "instructions": "",
            "rootPath": str(root),
            "workspaceKind": "folder",
            "workspaceDeviceId": str(root.stat().st_dev),
            "workspaceFileId": str(root.stat().st_ino),
            "archived": False,
            "createdAt": 1,
            "updatedAt": 1,
        }
    )
    save_worktree(
        {
            "id": "active-worktree",
            "projectId": project["id"],
            "gitRoot": str(root),
            "path": str(tmp_path / "owned-worktree"),
            "branch": "unsloth-studio/delete-safety",
            "baseRef": "HEAD",
            "markerPath": str(tmp_path / "owner.json"),
            "markerTokenHash": "proof",
            "backgroundTaskId": None,
            "status": "active",
            "createdAt": 1,
            "updatedAt": 1,
        }
    )

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            chat_history.delete_project(
                project["id"],
                SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace())),
                current_subject = "test-user",
            )
        )

    assert exc_info.value.status_code == 409
    assert "active and unresolved Studio worktrees" in exc_info.value.detail
    assert studio_db.get_chat_project(project["id"]) is not None


def test_project_delete_removes_owned_checkpoint_refs_before_row_cascade(tmp_path, monkeypatch):
    from core.agent_workspace.git_service import create_checkpoint
    from storage import studio_db

    root = tmp_path / "repository"
    root.mkdir()

    def run_git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd = root,
            check = True,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
        )
        return result.stdout.strip()

    run_git("init", "-q")
    run_git("config", "user.name", "Test")
    run_git("config", "user.email", "test@example.invalid")
    (root / "owned.txt").write_text("base\n", encoding = "utf-8")
    run_git("add", "owned.txt")
    run_git("commit", "-qm", "base")
    project = studio_db.upsert_chat_project(
        {
            "id": "project-with-checkpoint",
            "name": "Project",
            "instructions": "",
            "rootPath": str(root),
            "workspaceKind": "folder",
            "workspaceDeviceId": str(root.stat().st_dev),
            "workspaceFileId": str(root.stat().st_ino),
            "archived": False,
            "createdAt": 1,
            "updatedAt": 1,
        }
    )
    (root / "owned.txt").write_text("checkpoint\n", encoding = "utf-8")
    checkpoint = create_checkpoint(project["id"], ["owned.txt"])
    foreign_ref = "refs/unsloth-studio/checkpoints/foreign-user-ref"
    run_git("update-ref", foreign_ref, "HEAD")
    real_delete = chat_history.delete_chat_project

    def observed_delete(project_id, delete_files = False):
        owned_ref = subprocess.run(
            ["git", "show-ref", "--verify", checkpoint["refName"]],
            cwd = root,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            check = False,
        )
        assert owned_ref.returncode != 0
        assert run_git("show-ref", "--verify", "--hash", foreign_ref) == run_git(
            "rev-parse", "HEAD"
        )
        return real_delete(project_id, delete_files = delete_files)

    monkeypatch.setattr(chat_history, "delete_chat_project", observed_delete)
    monkeypatch.setattr(chat_history, "_delete_project_rag_sources", lambda _id: None)
    monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda _ids: None)
    monkeypatch.setattr(chat_history, "_cancel_research_runs", lambda _request, _ids: None)
    monkeypatch.setattr(chat_history, "_remove_conversation_archives", lambda *_a, **_k: None)

    async def remove_sandboxes(_ids, _delete_files):
        return 0, []

    monkeypatch.setattr(chat_history, "_remove_sandboxes", remove_sandboxes)
    deleted = asyncio.run(
        chat_history.delete_project(
            project["id"],
            SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace())),
            delete_files = True,
            current_subject = "test-user",
        )
    )

    assert deleted.id == project["id"]
    assert studio_db.get_chat_project(project["id"]) is None
    assert root.is_dir()
    assert run_git("show-ref", "--verify", "--hash", foreign_ref) == run_git("rev-parse", "HEAD")


def test_project_delete_stops_if_owned_checkpoint_ref_changed(tmp_path):
    from core.agent_workspace.git_service import create_checkpoint
    from core.agent_workspace.state import get_checkpoint
    from storage import studio_db

    root = tmp_path / "repository"
    root.mkdir()

    def run_git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd = root,
            check = True,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True,
        )
        return result.stdout.strip()

    run_git("init", "-q")
    run_git("config", "user.name", "Test")
    run_git("config", "user.email", "test@example.invalid")
    (root / "owned.txt").write_text("base\n", encoding = "utf-8")
    run_git("add", "owned.txt")
    run_git("commit", "-qm", "base")
    project = studio_db.upsert_chat_project(
        {
            "id": "project-with-changed-checkpoint",
            "name": "Project",
            "instructions": "",
            "rootPath": str(root),
            "workspaceKind": "folder",
            "workspaceDeviceId": str(root.stat().st_dev),
            "workspaceFileId": str(root.stat().st_ino),
            "archived": False,
            "createdAt": 1,
            "updatedAt": 1,
        }
    )
    (root / "owned.txt").write_text("checkpoint\n", encoding = "utf-8")
    checkpoint = create_checkpoint(project["id"], ["owned.txt"])
    head = run_git("rev-parse", "HEAD")
    run_git("update-ref", checkpoint["refName"], head)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            chat_history.delete_project(
                project["id"],
                SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace())),
                current_subject = "test-user",
            )
        )

    assert exc_info.value.status_code == 409
    assert "checkpoint ref changed" in str(exc_info.value.detail)
    assert studio_db.get_chat_project(project["id"]) is not None
    assert get_checkpoint(checkpoint["id"]) is not None
    assert run_git("show-ref", "--verify", "--hash", checkpoint["refName"]) == head


def test_project_delete_waits_for_agent_verification_worker(tmp_path, monkeypatch):
    from core.agent_workspace import background as background_module
    from storage import studio_db

    root = tmp_path / "repository"
    root.mkdir()
    project = studio_db.upsert_chat_project(
        {
            "id": "project-with-task",
            "name": "Project",
            "instructions": "",
            "rootPath": str(root),
            "workspaceKind": "folder",
            "workspaceDeviceId": str(root.stat().st_dev),
            "workspaceFileId": str(root.stat().st_ino),
            "archived": False,
            "createdAt": 1,
            "updatedAt": 1,
        }
    )
    entered = threading.Event()
    events = []

    def cancellable_verification(*args, cancel_event, **kwargs):
        entered.set()
        assert cancel_event.wait(timeout = 5)
        events.append("worker-stopped")
        return {"status": "cancelled"}

    real_delete = chat_history.delete_chat_project

    def observed_delete(project_id, delete_files = False):
        events.append("row-delete")
        return real_delete(project_id, delete_files = delete_files)

    monkeypatch.setattr(background_module, "run_project_verification", cancellable_verification)
    monkeypatch.setattr(chat_history, "delete_chat_project", observed_delete)
    monkeypatch.setattr(chat_history, "_delete_project_rag_sources", lambda _id: None)
    monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda _ids: None)
    monkeypatch.setattr(chat_history, "_cancel_research_runs", lambda _request, _ids: None)
    monkeypatch.setattr(chat_history, "_remove_conversation_archives", lambda *_a, **_k: None)

    async def remove_sandboxes(_ids, _delete_files):
        return 0, []

    monkeypatch.setattr(chat_history, "_remove_sandboxes", remove_sandboxes)
    background_module.manager.enqueue_verification(project["id"], start = True)
    assert entered.wait(timeout = 2)

    deleted = asyncio.run(
        chat_history.delete_project(
            project["id"],
            SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace())),
            current_subject = "test-user",
        )
    )

    assert deleted.id == project["id"]
    assert events == ["worker-stopped", "row-delete"]
    assert studio_db.get_chat_project(project["id"]) is None


def test_project_delete_cancels_and_waits_for_direct_verification_route(tmp_path, monkeypatch):
    from core.agent_workspace import verification as verification_module
    from core.agent_workspace.state import set_verification_config
    from routes import agent_workspace as agent_workspace_routes
    from storage import studio_db

    root = tmp_path / "repository"
    root.mkdir()
    project = studio_db.upsert_chat_project(
        {
            "id": "project-with-foreground-verification",
            "name": "Project",
            "instructions": "",
            "rootPath": str(root),
            "workspaceKind": "folder",
            "workspaceDeviceId": str(root.stat().st_dev),
            "workspaceFileId": str(root.stat().st_ino),
            "archived": False,
            "createdAt": 1,
            "updatedAt": 1,
        }
    )
    verification_config = set_verification_config(
        project["id"],
        [
            {
                "name": "test",
                "kind": "test",
                "command": "test-command",
                "required": True,
                "timeoutSeconds": 30,
                "logLimitBytes": 4096,
            }
        ],
    )
    entered = threading.Event()
    events = []
    route_result = []
    route_errors = []

    def cancellable_check(
        check,
        *,
        root,
        cancel_event,
        run_id,
        expected_root_identity = None,
    ):
        entered.set()
        assert cancel_event.wait(timeout = 5)
        events.append("foreground-stopped")
        return {
            "name": check["name"],
            "kind": check["kind"],
            "command": check["command"],
            "required": True,
            "status": "cancelled",
            "exitCode": None,
            "output": "",
            "outputBytes": 0,
            "outputTruncated": False,
            "timeoutSeconds": 30,
            "startedAt": 1,
            "completedAt": 2,
            "durationMs": 1,
        }

    real_delete = chat_history.delete_chat_project

    def observed_delete(project_id, delete_files = False):
        events.append("row-delete")
        return real_delete(project_id, delete_files = delete_files)

    def run_foreground_route():
        try:
            route_result.append(
                agent_workspace_routes.run_verification(
                    project["id"],
                    agent_workspace_routes.VerificationRunRequest(
                        configRevision = verification_config["revision"]
                    ),
                    current_subject = "test-user",
                )
            )
        except Exception as exc:
            route_errors.append(exc)

    monkeypatch.setattr(verification_module, "execute_check", cancellable_check)
    monkeypatch.setattr(agent_workspace_routes, "_require_execution_boundary", lambda: None)
    monkeypatch.setattr(chat_history, "delete_chat_project", observed_delete)
    monkeypatch.setattr(chat_history, "_delete_project_rag_sources", lambda _id: None)
    monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda _ids: None)
    monkeypatch.setattr(chat_history, "_cancel_research_runs", lambda _request, _ids: None)
    monkeypatch.setattr(chat_history, "_remove_conversation_archives", lambda *_a, **_k: None)

    async def remove_sandboxes(_ids, _delete_files):
        return 0, []

    monkeypatch.setattr(chat_history, "_remove_sandboxes", remove_sandboxes)
    foreground = threading.Thread(target = run_foreground_route)
    foreground.start()
    assert entered.wait(timeout = 2)

    deleted = asyncio.run(
        chat_history.delete_project(
            project["id"],
            SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace())),
            current_subject = "test-user",
        )
    )
    foreground.join(timeout = 2)

    assert not foreground.is_alive()
    assert route_errors == []
    assert route_result[0]["status"] == "cancelled"
    assert deleted.id == project["id"]
    assert events == ["foreground-stopped", "row-delete"]
    assert studio_db.get_chat_project(project["id"]) is None


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


def test_get_thread_fork_counts(monkeypatch):
    monkeypatch.setattr(chat_history, "fork_counts_for_thread", lambda _t: {"m1": 2, "m2": 1})
    response = chat_history.get_thread_fork_counts(
        thread_id = "t",
        current_subject = "test-user",
    )
    assert response.counts == {"m1": 2, "m2": 1}


def _clear_thread_row(thread_id: str) -> dict:
    return {
        "id": thread_id,
        "title": "Test Chat",
        "modelType": "base",
        "modelId": "test-model",
        "pairId": None,
        "archived": False,
        "createdAt": 1_700_000_000_000,
    }


def test_a_clear_does_not_reap_an_image_registered_while_it_was_running(tmp_path, monkeypatch):
    """The reap is global; the delete it accompanies is not.

    Between the transaction committing and the reap there is archive and sandbox cleanup
    that can run for seconds. A chat created in that window survives the delete, so
    wiping the whole registry afterwards took ITS thumbnails and left its cards 404ing
    out of thumbnail_bytes. Independent of the replay case: this one is a first clear.

    The snapshot is taken before the slow work, so an id registered during it is not the
    clear's to reap.
    """
    from core.inference import search_images
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    monkeypatch.setattr(search_images, "_registry", {})
    monkeypatch.setattr(search_images, "_cleared_unservable", set())
    monkeypatch.setattr(search_images, "_cache_dir", lambda: tmp_path / "thumbs")
    (tmp_path / "thumbs").mkdir(parents = True, exist_ok = True)

    old = search_images.register_images(
        [
            {
                "title": "before",
                "image": "https://img.example.com/a.jpg",
                "thumbnail": "https://tse1.mm.bing.net/th?id=a",
                "url": "https://example.com/a",
                "source": "Bing",
            }
        ]
    )
    assert old, "fixture must register: the whole test turns on this id being reapable"
    old_id = old[0]["id"]

    late: dict[str, str] = {}

    async def remove_sandboxes(_thread_ids, _delete_files):
        # Stands in for the concurrent client: another tab registers an image for a chat
        # created after the transaction, while this slow cleanup is still running.
        entries = search_images.register_images(
            [
                {
                    "title": "during",
                    "image": "https://img.example.com/b.jpg",
                    "thumbnail": "https://tse1.mm.bing.net/th?id=b",
                    "url": "https://example.com/b",
                    "source": "Bing",
                }
            ]
        )
        late["id"] = entries[0]["id"]
        return 0, []

    monkeypatch.setattr(chat_history, "_remove_sandboxes", remove_sandboxes)
    monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda _ids: None)
    monkeypatch.setattr(chat_history, "_cancel_research_runs", lambda _request, _ids: None)
    monkeypatch.setattr(
        chat_history, "_remove_conversation_archives", lambda _ids, cutoff = None: None
    )
    request = SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace()))

    studio_db.upsert_chat_thread(_clear_thread_row("before-clear"))
    asyncio.run(
        chat_history.clear_history(
            request,
            chat_history.ChatClearRequest(ids = [], operationId = "clear-operation-race"),
            current_subject = "test-user",
        )
    )

    assert late.get("id"), "the stand-in never registered, so this asserts nothing"
    assert search_images.lookup_image(late["id"]) is not None, (
        "an image registered while the clear was running belongs to a chat the clear "
        "kept, so reaping it 404s that chat's cards"
    )
    assert (
        search_images.lookup_image(old_id) is None
    ), "the clear still has to reap what it was responsible for"


def test_replayed_clear_keeps_the_thumbnails_of_a_chat_it_did_not_delete(tmp_path, monkeypatch):
    """A retry under a recorded operationId replays, so it must not reap the global cache.

    The frontend retries DELETE /chat once under the SAME operationId after its 30s
    abort, and Starlette does not cancel the first handler when the client hangs up, so
    the retry lands behind a transaction that already committed. That transaction
    deliberately leaves chats created since alone -- but the thumbnail registry is
    global, so reaping it again took the images of a chat this call is not deleting and
    left its cards 404ing.
    """
    from core.inference import search_images
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    reaped: list[str] = []
    monkeypatch.setattr(search_images, "clear_cache", lambda only_ids = None: reaped.append("reaped"))

    async def remove_sandboxes(_thread_ids, _delete_files):
        return 0, []

    monkeypatch.setattr(chat_history, "_remove_sandboxes", remove_sandboxes)
    monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda _ids: None)
    monkeypatch.setattr(chat_history, "_cancel_research_runs", lambda _request, _ids: None)
    monkeypatch.setattr(
        chat_history, "_remove_conversation_archives", lambda _ids, cutoff = None: None
    )
    request = SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace()))

    def clear():
        return asyncio.run(
            chat_history.clear_history(
                request,
                chat_history.ChatClearRequest(ids = [], operationId = "clear-operation-retry"),
                current_subject = "test-user",
            )
        )

    studio_db.upsert_chat_thread(_clear_thread_row("before-clear"))
    assert clear()["deletedThreadIds"] == ["before-clear"]
    assert reaped == ["reaped"], "the first clear has to reap: the thumbnails say what was searched"

    # The image-bearing chat the delayed retry must not touch.
    studio_db.upsert_chat_thread(_clear_thread_row("after-clear"))

    replay = clear()
    assert replay["deletedThreadIds"] == ["before-clear"]
    assert studio_db.get_chat_thread("after-clear") is not None
    assert reaped == ["reaped"], "the replay reaped a surviving chat's thumbnails"


def test_the_replay_bit_comes_from_the_clear_transaction(monkeypatch, tmp_path):
    """Two concurrent retries of one operationId: exactly one of them performed the clear.

    Establishing `replayed` with a read taken before the transaction is a guess. Both
    requests carrying the same operationId see the same unrecorded ledger, so both
    conclude they cleared; BEGIN IMMEDIATE then serialises them and the loser silently
    replays while still believing otherwise. It would go on to reap the thumbnail
    registry -- which is global, and so is not covered by the ids the transaction
    deliberately kept -- taking the images of chats created since the winner committed.
    This is the retry the operationId exists to make safe: the frontend reissues the
    same id after its 30s abort, and Starlette does not cancel the handler the client
    hung up on, so both really do run at once.
    """
    from core.inference import search_images
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)

    reaps: list[str] = []
    reap_lock = threading.Lock()

    def record_reap(only_ids = None):
        with reap_lock:
            reaps.append("reaped")

    monkeypatch.setattr(search_images, "clear_cache", record_reap)

    async def remove_sandboxes(_thread_ids, _delete_files):
        return 0, []

    monkeypatch.setattr(chat_history, "_remove_sandboxes", remove_sandboxes)
    monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda _ids: None)
    monkeypatch.setattr(chat_history, "_cancel_research_runs", lambda _request, _ids: None)
    monkeypatch.setattr(
        chat_history, "_remove_conversation_archives", lambda _ids, cutoff = None: None
    )
    request = SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace()))

    studio_db.upsert_chat_thread(_clear_thread_row("before-clear"))

    # Released together, so both are past the point the old code decided `replayed` at
    # before either transaction commits -- which is the whole race.
    start = threading.Barrier(2)
    failures: list[BaseException] = []

    def clear():
        try:
            start.wait(timeout = 10)
            asyncio.run(
                chat_history.clear_history(
                    request,
                    chat_history.ChatClearRequest(ids = [], operationId = "clear-operation-concurrent"),
                    current_subject = "test-user",
                )
            )
        except BaseException as exc:  # noqa: BLE001 -- re-raised on the main thread
            failures.append(exc)

    threads = [threading.Thread(target = clear) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout = 30)
    if failures:
        raise failures[0]

    assert reaps == ["reaped"], (
        "only the request that actually performed the clear may reap the global "
        f"thumbnail registry; got {len(reaps)} reaps"
    )


def test_clear_history_does_not_read_the_replay_ledger_outside_the_transaction():
    """The structural half of the race above, which no scheduling can hide.

    `replayed` has to be whatever the transaction did, so it is returned by the call
    that does the clear. A separate ledger read reintroduces the window even if the
    threads in the test above happen to serialise.
    """
    source = inspect.getsource(chat_history.clear_history)
    assert "clear_chat_history_with_replay_status" in source
    assert (
        "chat_clear_operation_is_recorded" not in source
    ), "a pre-transaction ledger read cannot tell a replay from a concurrent clear"


def test_a_chat_created_in_the_gap_after_the_clear_keeps_its_images(monkeypatch, tmp_path):
    """The snapshot has to be taken at the clear boundary, not one await later.

    `await run_in_threadpool(...)` is a yield point. With the clear and the snapshot in
    separate calls, the event loop can run another request in between: a chat created there
    survives the transaction (the clear only deletes what it saw), but its images register
    before the snapshot, so the reap that follows takes them and its cards 404 out of
    thumbnail_bytes. One threadpool call for both removes that gap.

    It does not make the two atomic -- another worker THREAD can still land between the
    commit and the read, and closing that would mean holding the image registry's lock across
    the whole transaction, stalling every search in the process for the length of a clear.
    This pins the gap that was worth removing.

    The interleave is forced rather than raced: `run_in_threadpool` is wrapped so the other
    tab registers its image immediately after the FIRST hop returns, which is exactly the
    window in question.
    """
    from core.inference import search_images
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    monkeypatch.setattr(search_images, "_registry", {})
    monkeypatch.setattr(search_images, "_cache_dir", lambda: tmp_path / "thumbs")
    (tmp_path / "thumbs").mkdir(parents = True, exist_ok = True)

    reaped: list = []
    monkeypatch.setattr(search_images, "clear_cache", lambda only_ids = None: reaped.append(only_ids))

    async def remove_sandboxes(_thread_ids, _delete_files):
        return 0, []

    monkeypatch.setattr(chat_history, "_remove_sandboxes", remove_sandboxes)
    monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda _ids: None)
    monkeypatch.setattr(chat_history, "_cancel_research_runs", lambda _request, _ids: None)
    monkeypatch.setattr(
        chat_history, "_remove_conversation_archives", lambda _ids, cutoff = None: None
    )

    # The other tab's image, registered in the gap. Straight into the registry: this is about
    # WHEN the id becomes visible to the snapshot, not about how it got there.
    late_image_id = "beefbeefbeef"
    hops = {"n": 0}
    # The route imports it inside the handler, so the patch has to land on the module it
    # imports FROM, not on routes.chat_history.
    import starlette.concurrency

    real_run_in_threadpool = starlette.concurrency.run_in_threadpool

    async def interleaving_run_in_threadpool(func, *args, **kwargs):
        result = await real_run_in_threadpool(func, *args, **kwargs)
        hops["n"] += 1
        if hops["n"] == 1:
            search_images._registry[late_image_id] = {
                "thumbnail": "https://example.invalid/x.jpg",
                "source": "https://example.invalid/",
                "created": 0.0,
                "policy": None,
            }
        return result

    monkeypatch.setattr(starlette.concurrency, "run_in_threadpool", interleaving_run_in_threadpool)
    request = SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace()))

    studio_db.upsert_chat_thread(_clear_thread_row("before-clear"))
    asyncio.run(
        chat_history.clear_history(
            request,
            chat_history.ChatClearRequest(ids = [], operationId = "clear-operation-gap"),
            current_subject = "test-user",
        )
    )

    assert reaped, "the clear still has to reap what it was responsible for"
    snapshot = reaped[0]
    assert snapshot is not None, "a real clear reaps a bounded set, not everything"
    assert late_image_id not in snapshot, (
        "an image registered after the clear committed belongs to a chat the clear kept, "
        "so the reap must not be allowed to take it"
    )


def test_the_clear_and_its_image_snapshot_share_one_threadpool_hop():
    """The structural half of the race above, which no test scheduling can hide."""
    source = inspect.getsource(chat_history.clear_history)
    assert source.count("run_in_threadpool(_clear_rows)") == 1
    assert (
        "run_in_threadpool(snapshot_and_fence_registrations)" not in source
    ), "a second hop for the snapshot reopens the gap the first one closed"
    body = source.split("def _clear_rows(", 1)[1].split("\n    # The clear reports", 1)[0]
    assert (
        "snapshot_and_fence_registrations()" in body
    ), "the snapshot belongs inside the clear's hop, and it carries the registration fence"


def test_a_replay_finishes_a_reap_the_original_clear_died_before_running(monkeypatch, tmp_path):
    """A crash between the clear's commit and its thumbnail reap must not lose the reap.

    The reap runs after the transaction, behind seconds of archive and sandbox cleanup. Killed
    in that window the operation is already recorded, so the retry the frontend sends replays
    -- and a replay deliberately reaps nothing, because the chats created since the original
    clear are not its to take. The thumbnails of every deleted chat then stay on disk for good,
    saying what was searched for, which is the worse of the two failures this path weighs.

    The ledger now carries the original clear's own snapshot and whether the reap finished, so
    the replay can complete exactly that set. The crash is simulated by making the first reap
    raise, which is the same state a SIGKILL leaves behind: committed, recorded, unreaped.
    """
    from core.inference import search_images
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    monkeypatch.setattr(search_images, "_registry", {})
    monkeypatch.setattr(search_images, "_cache_dir", lambda: tmp_path / "thumbs")
    (tmp_path / "thumbs").mkdir(parents = True, exist_ok = True)

    doomed_image_id = "aaaabbbbcccc"
    search_images._registry[doomed_image_id] = {
        "thumbnail": "https://example.invalid/x.jpg",
        "source": "https://example.invalid/",
        "created": 0.0,
        "policy": None,
    }

    reaps: list = []

    def reap(only_ids = None):
        reaps.append(only_ids)
        if len(reaps) == 1:
            raise RuntimeError("process died before the reap finished")

    monkeypatch.setattr(search_images, "clear_cache", reap)

    async def remove_sandboxes(_thread_ids, _delete_files):
        return 0, []

    monkeypatch.setattr(chat_history, "_remove_sandboxes", remove_sandboxes)
    monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda _ids: None)
    monkeypatch.setattr(chat_history, "_cancel_research_runs", lambda _request, _ids: None)
    monkeypatch.setattr(
        chat_history, "_remove_conversation_archives", lambda _ids, cutoff = None: None
    )
    request = SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace()))

    def clear():
        return asyncio.run(
            chat_history.clear_history(
                request,
                chat_history.ChatClearRequest(ids = [], operationId = "clear-operation-crash"),
                current_subject = "test-user",
            )
        )

    studio_db.upsert_chat_thread(_clear_thread_row("before-clear"))
    with pytest.raises(RuntimeError):
        clear()
    assert reaps == [{doomed_image_id}], "the first attempt got as far as its own reap"

    # A chat started after the crash, whose images the replay must NOT take.
    studio_db.upsert_chat_thread(_clear_thread_row("after-crash"))
    later_image_id = "ddddeeeeffff"
    search_images._registry[later_image_id] = {
        "thumbnail": "https://example.invalid/y.jpg",
        "source": "https://example.invalid/",
        "created": 0.0,
        "policy": None,
    }

    clear()
    assert len(reaps) == 2, "the replay has to finish the reap the crash interrupted"
    finished = reaps[1]
    assert finished == {
        doomed_image_id
    }, "bounded to the original clear's own snapshot, so a chat created since keeps its images"
    assert later_image_id not in finished

    # And a further retry has nothing left to do.
    clear()
    assert len(reaps) == 2, "the finished reap must be recorded, not repeated on every retry"


def test_a_plain_replay_with_nothing_outstanding_still_reaps_nothing(monkeypatch, tmp_path):
    """The ordinary retry. The first attempt completed its reap, so the replay must stay out
    of the global registry entirely -- that is what the replay branch is for."""
    from core.inference import search_images
    from storage import studio_db

    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    monkeypatch.setattr(search_images, "_registry", {})
    monkeypatch.setattr(search_images, "_cache_dir", lambda: tmp_path / "thumbs")
    (tmp_path / "thumbs").mkdir(parents = True, exist_ok = True)

    reaps: list = []
    monkeypatch.setattr(search_images, "clear_cache", lambda only_ids = None: reaps.append(only_ids))

    async def remove_sandboxes(_thread_ids, _delete_files):
        return 0, []

    monkeypatch.setattr(chat_history, "_remove_sandboxes", remove_sandboxes)
    monkeypatch.setattr(chat_history, "_cancel_active_generations", lambda _ids: None)
    monkeypatch.setattr(chat_history, "_cancel_research_runs", lambda _request, _ids: None)
    monkeypatch.setattr(
        chat_history, "_remove_conversation_archives", lambda _ids, cutoff = None: None
    )
    request = SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace()))

    def clear():
        return asyncio.run(
            chat_history.clear_history(
                request,
                chat_history.ChatClearRequest(ids = [], operationId = "clear-operation-plain"),
                current_subject = "test-user",
            )
        )

    studio_db.upsert_chat_thread(_clear_thread_row("before-clear"))
    clear()
    assert len(reaps) == 1
    clear()
    assert len(reaps) == 1, "a replay behind a completed reap must not touch the registry"
