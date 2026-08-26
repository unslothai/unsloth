# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Focused regressions for server-owned agent workspace capabilities."""

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from models.inference import (
    AnthropicMessagesRequest,
    ChatCompletionRequest,
    ChatCountTokensRequest,
    ResponsesRequest,
    ToolConfirmRequest,
)
from routes import agent_workspace, chat_history, inference
from storage import studio_db


@pytest.fixture(autouse = True)
def _isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio"))
    monkeypatch.setattr(studio_db, "_schema_ready", False)


@pytest.mark.parametrize(
    "factory",
    [
        lambda session_id: ChatCompletionRequest(
            messages = [{"role": "user", "content": "test"}],
            session_id = session_id,
        ),
        lambda session_id: ChatCountTokensRequest(
            messages = [{"role": "user", "content": "test"}],
            session_id = session_id,
        ),
        lambda session_id: ResponsesRequest(input = "test", session_id = session_id),
        lambda session_id: AnthropicMessagesRequest(
            messages = [{"role": "user", "content": "test"}],
            session_id = session_id,
        ),
        lambda session_id: ToolConfirmRequest(session_id = session_id),
    ],
)
def test_public_inference_models_reject_server_owned_task_session(factory):
    with pytest.raises(ValidationError, match = "server-owned"):
        factory("agent-task-00000000-0000-0000-0000-000000000000")


def test_folder_project_is_not_a_public_sandbox(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    metadata = root.stat()
    studio_db.upsert_chat_project(
        {
            "id": "folder-project",
            "name": "Folder project",
            "instructions": "",
            "rootPath": str(root),
            "workspaceKind": "folder",
            "workspaceDeviceId": str(metadata.st_dev),
            "workspaceFileId": str(metadata.st_ino),
            "archived": False,
            "createdAt": 1,
            "updatedAt": 1,
        }
    )

    with pytest.raises(HTTPException) as exc:
        inference._sandbox_dir_for("project-folder-project", create = False)
    assert exc.value.status_code == 403


def test_cancel_endpoint_ignores_server_owned_task_session(monkeypatch):
    seen = []
    monkeypatch.setattr(inference, "_cancel_by_keys", lambda keys: seen.extend(keys) or len(keys))

    class Request:
        async def json(self):
            return {"session_id": "agent-task-00000000-0000-0000-0000-000000000000"}

    result = asyncio.run(inference.cancel_inference(Request(), current_subject = "user"))
    assert result == {"cancelled": 0}
    assert seen == []


def test_thread_delete_cancellation_ignores_server_owned_task_session(monkeypatch):
    from state import active_generations

    seen = []
    monkeypatch.setattr(active_generations, "cancel_thread", seen.append)
    chat_history._cancel_active_generations(
        [
            "agent-task-00000000-0000-0000-0000-000000000000",
            "ordinary-thread",
        ]
    )
    assert seen == ["ordinary-thread"]


def test_saved_mcp_inference_requires_ui_session():
    request = SimpleNamespace(headers = {"authorization": "Bearer sk-unsloth-public-caller"})
    with pytest.raises(HTTPException) as exc:
        inference._require_ui_for_installed_mcp(SimpleNamespace(mcp_enabled = True), request)
    assert exc.value.status_code == 403


def test_provider_background_and_github_handoff_require_ui_session():
    with pytest.raises(HTTPException) as provider_exc:
        agent_workspace._require_ui_for_provider_task(
            {
                "kind": "agent",
                "payload": {"runtime": {"kind": "provider"}},
            },
            True,
        )
    assert provider_exc.value.status_code == 403

    payload = agent_workspace.PullRequestHandoffRequest(
        serverId = "github",
        owner = "unslothai",
        repository = "unsloth",
        base = "main",
        head = "feature",
    )
    with pytest.raises(HTTPException) as github_exc:
        asyncio.run(
            agent_workspace.prepare_connected_pull_request(
                "project",
                payload,
                current_subject = "api-key-user",
                via_api_key = True,
            )
        )
    assert github_exc.value.status_code == 403


def test_public_background_result_does_not_disclose_task_session():
    public = agent_workspace._public_background_task(
        {
            "id": "task",
            "error": None,
            "result": {
                "sessionId": "agent-task-00000000-0000-0000-0000-000000000000",
                "output": "done",
            },
        }
    )
    assert public["result"] == {"output": "done"}
