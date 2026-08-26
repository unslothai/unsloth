# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from core import research_runs as research_worker
from core.agent_workspace.project_context import (
    ProjectContextSnapshotInvalid,
    resolve_project_context_snapshot,
)
from routes import inference
from routes import research_runs as research_routes
from routes.research_runs import CreateResearchRun, create_research_run
from storage import research_runs_db, studio_db


@pytest.fixture
def project_research_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio-home"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    root = tmp_path / "project"
    root.mkdir()
    (root / "AGENTS.md").write_text("Keep the original research rule.\n", encoding = "utf-8")
    (root / "target.py").write_text("VALUE = 1\n", encoding = "utf-8")
    metadata = root.stat()
    studio_db.upsert_chat_project(
        {
            "id": "project-1",
            "name": "Research project",
            "instructions": "Use the project policy.",
            "goal": "Explain target.py safely.",
            "goalStatus": "active",
            "goalUpdatedAt": 1,
            "rootPath": str(root),
            "workspaceKind": "folder",
            "workspaceDeviceId": str(metadata.st_dev),
            "workspaceFileId": str(metadata.st_ino),
            "archived": False,
            "createdAt": 1,
            "updatedAt": 1,
        }
    )
    studio_db.upsert_chat_thread(
        {
            "id": "thread-1",
            "title": "Research",
            "modelType": "base",
            "modelId": "local-model",
            "projectId": "project-1",
            "createdAt": 1,
        }
    )
    studio_db.upsert_chat_message(
        {
            "id": "user-1",
            "threadId": "thread-1",
            "role": "user",
            "content": [{"type": "text", "text": "Research target.py"}],
            "createdAt": 2,
        }
    )
    return root


def _create_project_run() -> dict:
    return create_research_run(
        CreateResearchRun(
            threadId = "thread-1",
            userMessageId = "user-1",
            inferenceRequest = {"model": "local-model"},
        ),
        SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace())),
        current_subject = "alice",
    )


def test_project_research_persists_only_server_snapshot_id_and_survives_restart(
    project_research_home: Path, monkeypatch: pytest.MonkeyPatch
):
    run = _create_project_run()
    snapshot_id = run["config"]["projectContextSnapshotId"]

    assert 32 <= len(snapshot_id) <= 128
    persisted_config = json.dumps(run["config"], ensure_ascii = False)
    assert "sessionId" not in run["config"]
    assert "projectId" not in run["config"]
    assert str(project_research_home) not in persisted_config
    assert "Keep the original research rule" not in persisted_config
    assert "Explain target.py safely" not in persisted_config

    snapshot = research_runs_db.get_project_context_snapshot(snapshot_id, run_id = run["id"])
    assert snapshot is not None
    assert snapshot["projectId"] == "project-1"
    assert snapshot["sessionId"] == "project-project-1"
    assert "Keep the original research rule" in snapshot["context"]["addition"]
    assert "Explain target.py safely" in snapshot["context"]["addition"]

    project_research_home.joinpath("AGENTS.md").write_text(
        "A changed rule that must not enter the active run.\n",
        encoding = "utf-8",
    )
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    with pytest.raises(HTTPException) as replay_error:
        inference._with_project_context_messages(
            [{"role": "user", "content": "Research target.py"}],
            "project-project-1",
            snapshot_id,
        )
    assert replay_error.value.status_code == 409
    with pytest.raises(ProjectContextSnapshotInvalid):
        resolve_project_context_snapshot(
            "project-project-1",
            snapshot_id,
            durable_research_run_id = "another-run",
            durable_owner_subject = "alice",
        )
    with pytest.raises(ProjectContextSnapshotInvalid):
        resolve_project_context_snapshot(
            "project-project-1",
            snapshot_id,
            durable_research_run_id = run["id"],
            durable_owner_subject = "mallory",
        )
    frozen = resolve_project_context_snapshot(
        "project-project-1",
        snapshot_id,
        query = "Research target.py",
        durable_research_run_id = run["id"],
        durable_owner_subject = "alice",
    )
    assert frozen is not None
    assert "Keep the original research rule" in frozen.addition
    assert "A changed rule" not in frozen.addition
    authorized_messages = inference._with_project_context_messages(
        [{"role": "user", "content": "Research target.py"}],
        "project-project-1",
        snapshot_id,
        durable_research_run_id = run["id"],
        durable_owner_subject = "alice",
    )
    assert "Keep the original research rule" in authorized_messages[0]["content"]
    assert research_worker._project_context_transport(run) == {
        "session_id": "project-project-1",
        "project_context_snapshot_id": snapshot_id,
    }


def test_durable_snapshot_binding_requires_the_deep_research_workflow(
    monkeypatch: pytest.MonkeyPatch,
):
    request = SimpleNamespace(
        headers = {"authorization": f"Bearer {inference.API_KEY_PREFIX}deadbeefdeadbeef"}
    )
    monkeypatch.setattr(inference.auth_storage, "is_internal_api_key", lambda _token: False)
    assert inference._durable_research_snapshot_run_id(request, "research:run-1") is None

    monkeypatch.setattr(inference.auth_storage, "is_internal_api_key", lambda _token: True)
    monkeypatch.setattr(
        inference.auth_storage,
        "internal_api_key_name",
        lambda _token: "data-recipe workflow",
    )
    assert inference._durable_research_snapshot_run_id(request, "research:run-1") is None

    monkeypatch.setattr(
        inference.auth_storage,
        "internal_api_key_name",
        lambda _token: inference.auth_storage.DEEP_RESEARCH_WORKFLOW_KEY_NAME,
    )
    assert inference._durable_research_snapshot_run_id(request, "thread-1") is None
    assert inference._durable_research_snapshot_run_id(request, "research:../run-1") is None
    assert inference._durable_research_snapshot_run_id(request, "research:run-1") == "run-1"


def test_whitespace_handoff_question_uses_the_message_for_context_selection(
    project_research_home: Path, monkeypatch: pytest.MonkeyPatch
):
    captured_queries: list[str] = []
    original = research_routes._project_context_snapshot_for_thread

    def capture(thread: dict, *, query: str):
        captured_queries.append(query)
        return original(thread, query = query)

    monkeypatch.setattr(research_routes, "_project_context_snapshot_for_thread", capture)
    create_research_run(
        CreateResearchRun(
            threadId = "thread-1",
            userMessageId = "user-1",
            inferenceRequest = {"model": "local-model"},
            question = " \n\t ",
        ),
        SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace())),
        current_subject = "alice",
    )

    assert captured_queries == ["Research target.py"]


def test_project_create_rejects_thread_move_after_snapshot(
    project_research_home: Path, monkeypatch: pytest.MonkeyPatch
):
    original = research_routes._project_context_snapshot_for_thread

    def move_after_snapshot(thread: dict, *, query: str):
        snapshot = original(thread, query = query)
        studio_db.update_chat_thread("thread-1", {"projectId": None})
        return snapshot

    monkeypatch.setattr(
        research_routes,
        "_project_context_snapshot_for_thread",
        move_after_snapshot,
    )
    with pytest.raises(HTTPException) as caught:
        _create_project_run()

    assert caught.value.status_code == 409
    assert research_runs_db.has_thread_claim("thread-1") is False


def test_non_project_create_rejects_concurrent_move_into_project(
    project_research_home: Path, monkeypatch: pytest.MonkeyPatch
):
    studio_db.update_chat_thread("thread-1", {"projectId": None})

    def move_before_transaction(_thread: dict, *, query: str):
        assert query == "Research target.py"
        studio_db.update_chat_thread("thread-1", {"projectId": "project-1"})
        return None

    monkeypatch.setattr(
        research_routes,
        "_project_context_snapshot_for_thread",
        move_before_transaction,
    )
    with pytest.raises(HTTPException) as caught:
        _create_project_run()

    assert caught.value.status_code == 409
    assert research_runs_db.has_thread_claim("thread-1") is False


def test_project_rebind_rejects_thread_move_after_snapshot(
    project_research_home: Path, monkeypatch: pytest.MonkeyPatch
):
    first = _create_project_run()
    first_snapshot_id = first["config"]["projectContextSnapshotId"]
    research_runs_db.request_cancel(first["id"])
    research_runs_db.claim_next("worker-1")
    research_runs_db.finish(first["id"], "worker-1", "cancelled")
    studio_db.upsert_chat_message(
        {
            "id": "user-2",
            "threadId": "thread-1",
            "role": "user",
            "content": [{"type": "text", "text": "Research target.py again"}],
            "createdAt": 3,
        }
    )
    original = research_routes._project_context_snapshot_for_thread

    def move_after_snapshot(thread: dict, *, query: str):
        snapshot = original(thread, query = query)
        studio_db.update_chat_thread("thread-1", {"projectId": None})
        return snapshot

    monkeypatch.setattr(
        research_routes,
        "_project_context_snapshot_for_thread",
        move_after_snapshot,
    )
    with pytest.raises(HTTPException) as caught:
        create_research_run(
            CreateResearchRun(
                threadId = "thread-1",
                userMessageId = "user-2",
                inferenceRequest = {"model": "local-model"},
            ),
            SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace())),
            current_subject = "alice",
        )

    assert caught.value.status_code == 409
    unchanged = research_runs_db.get_run(first["id"])
    assert unchanged is not None
    assert unchanged["status"] == "cancelled"
    assert unchanged["config"]["projectContextSnapshotId"] == first_snapshot_id
    assert research_runs_db.get_project_context_snapshot(first_snapshot_id) is not None


def test_project_stream_hop_carries_bound_session_and_snapshot(
    project_research_home: Path, monkeypatch: pytest.MonkeyPatch
):
    run = _create_project_run()
    sent: list[dict] = []
    response = httpx.Response(
        200,
        text = (
            'data: {"choices":[{"delta":{"content":"done"},'
            '"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
        ),
        request = httpx.Request("POST", "http://127.0.0.1:1/v1/chat/completions"),
    )

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_exc_info):
            return False

        def build_request(self, method, url, **kwargs):
            return {"method": method, "url": url, **kwargs}

        async def send(
            self,
            request,
            *,
            stream = False,
        ):
            assert stream is True
            sent.append(request)
            return response

    monkeypatch.setattr(research_worker.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(
        research_worker.auth_storage,
        "create_api_key",
        lambda **_kwargs: ("internal-token", {"id": 1}),
    )
    monkeypatch.setattr(
        research_worker.auth_storage,
        "revoke_internal_api_key",
        lambda _key_id: None,
    )
    supervisor = research_worker.ResearchSupervisor(
        SimpleNamespace(state = SimpleNamespace(server_port = 1))
    )

    async def active(_run_id: str):
        return None

    monkeypatch.setattr(supervisor, "_check_active", active)
    result = asyncio.run(
        supervisor._stream_completion(
            run,
            [{"role": "user", "content": "Research target.py"}],
            report_progress = False,
        )
    )

    assert result[:3] == ("done", "", "stop")
    body = sent[0]["json"]
    assert body["thread_id"] == f"research:{run['id']}"
    assert body["session_id"] == "project-project-1"
    assert body["project_context_snapshot_id"] == run["config"]["projectContextSnapshotId"]


def test_project_research_snapshot_is_bound_to_run_and_project(project_research_home: Path):
    run = _create_project_run()
    snapshot_id = run["config"]["projectContextSnapshotId"]

    assert research_runs_db.get_project_context_snapshot(snapshot_id, run_id = "another-run") is None
    assert (
        research_runs_db.get_project_context_snapshot(snapshot_id, project_id = "another-project")
        is None
    )


def test_rebound_project_research_atomically_replaces_its_snapshot(project_research_home: Path):
    first = _create_project_run()
    first_snapshot_id = first["config"]["projectContextSnapshotId"]
    research_runs_db.request_cancel(first["id"])
    research_runs_db.claim_next("worker-1")
    research_runs_db.finish(first["id"], "worker-1", "cancelled")

    project_research_home.joinpath("AGENTS.md").write_text(
        "Use the replacement research rule.\n",
        encoding = "utf-8",
    )
    studio_db.upsert_chat_message(
        {
            "id": "user-2",
            "threadId": "thread-1",
            "role": "user",
            "content": [{"type": "text", "text": "Research target.py again"}],
            "createdAt": 3,
        }
    )
    rebound = create_research_run(
        CreateResearchRun(
            threadId = "thread-1",
            userMessageId = "user-2",
            inferenceRequest = {"model": "local-model"},
        ),
        SimpleNamespace(app = SimpleNamespace(state = SimpleNamespace())),
        current_subject = "alice",
    )
    rebound_snapshot_id = rebound["config"]["projectContextSnapshotId"]

    assert rebound["id"] == first["id"]
    assert rebound_snapshot_id != first_snapshot_id
    assert research_runs_db.get_project_context_snapshot(first_snapshot_id) is None
    replacement = research_runs_db.get_project_context_snapshot(
        rebound_snapshot_id,
        run_id = rebound["id"],
    )
    assert replacement is not None
    assert "Use the replacement research rule" in replacement["context"]["addition"]
    assert "Keep the original research rule" not in replacement["context"]["addition"]


@pytest.mark.parametrize(
    "field",
    (
        "projectContextSnapshotId",
        "project_context_snapshot_id",
        "projectId",
        "sessionId",
        "session_id",
    ),
)
def test_research_creation_schema_rejects_renderer_project_authority(field: str):
    with pytest.raises(ValidationError):
        CreateResearchRun(
            threadId = "thread-1",
            userMessageId = "user-1",
            inferenceRequest = {"model": "local-model"},
            **{field: "x" * 32},
        )


def test_non_project_research_keeps_legacy_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    studio_db.upsert_chat_thread(
        {
            "id": "thread-1",
            "title": "Research",
            "modelType": "base",
            "modelId": "local-model",
            "createdAt": 1,
        }
    )
    studio_db.upsert_chat_message(
        {
            "id": "user-1",
            "threadId": "thread-1",
            "role": "user",
            "content": [{"type": "text", "text": "Research this"}],
            "createdAt": 2,
        }
    )

    run = _create_project_run()

    assert "projectContextSnapshotId" not in run["config"]
    assert research_worker._project_context_transport(run) == {}
