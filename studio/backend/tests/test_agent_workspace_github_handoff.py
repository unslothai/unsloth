# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import subprocess
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import authenticated_via_api_key, get_current_subject
from core.agent_workspace.common import AgentWorkspaceError
from core.agent_workspace.github_handoff import (
    consume_pull_request_handoff,
    prepare_pull_request_handoff,
    reset_pull_request_handoffs_for_tests,
)
from storage import mcp_servers_db, studio_db
from routes import agent_workspace as agent_workspace_routes
from routes.agent_workspace import router


_TOOLS = [
    {
        "name": "create_pull_request",
        "inputSchema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string"},
                "repo": {"type": "string"},
                "title": {"type": "string"},
                "body": {"type": "string"},
                "head": {"type": "string"},
                "base": {"type": "string"},
                "draft": {"type": "boolean"},
                "maintainer_can_modify": {"type": "boolean"},
            },
            "required": ["owner", "repo", "title", "head", "base"],
        },
    }
]


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd = root,
        check = True,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
    )
    return completed.stdout.strip()


def _project(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.invalid")
    (root / "tracked.txt").write_text("base\n", encoding = "utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-qm", "base")
    metadata = root.stat()
    studio_db.upsert_chat_project(
        {
            "id": "project",
            "name": "Project",
            "instructions": "",
            "rootPath": str(root),
            "workspaceKind": "folder",
            "workspaceDeviceId": str(metadata.st_dev),
            "workspaceFileId": str(metadata.st_ino),
            "goal": "Do not expose password=hunter2",
            "goalStatus": "active",
            "archived": False,
            "createdAt": 1,
            "updatedAt": 1,
        }
    )
    mcp_servers_db.create_server(
        id = "github",
        display_name = "GitHub",
        url = "https://github.example.invalid/mcp",
        headers_json = '{"Authorization":"Bearer secret"}',
        is_enabled = True,
    )
    reset_pull_request_handoffs_for_tests()


def _prepare(root: Path, **overrides):
    values = {
        "server_id": "github",
        "owner": "unslothai",
        "repository": "unsloth",
        "base": "main",
        "head": "feature/codex-workspace",
        "body_note": f"Local path {root} and token=do-not-send",
        "tools": _TOOLS,
        "now": 100,
    }
    values.update(overrides)
    return prepare_pull_request_handoff("project", **values)


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix = "/api/agent-workspace")
    app.dependency_overrides[get_current_subject] = lambda: "test-subject"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    return TestClient(app)


def test_pull_request_handoff_is_redacted_and_requires_one_use_confirmation(tmp_path):
    _project(tmp_path)
    preview = _prepare(tmp_path)

    rendered = f"{preview['request']['title']}\n{preview['request']['body']}"
    assert str(tmp_path) not in rendered
    assert "hunter2" not in rendered
    assert "do-not-send" not in rendered
    assert preview["submitted"] is False
    assert preview["request"]["draft"] is True

    server, request = consume_pull_request_handoff(
        "project",
        preview["id"],
        server_id = "github",
        confirmation_token = preview["confirmationToken"],
        expected_request_digest = preview["requestDigest"],
        tools = _TOOLS,
        now = 101,
    )
    assert server["id"] == "github"
    assert request == preview["request"]
    with pytest.raises(AgentWorkspaceError, match = "already used"):
        consume_pull_request_handoff(
            "project",
            preview["id"],
            server_id = "github",
            confirmation_token = preview["confirmationToken"],
            expected_request_digest = preview["requestDigest"],
            tools = _TOOLS,
            now = 102,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("owner", "bad/owner", "owner"),
        ("repository", ".hidden", "repository"),
        ("base", "../main", "base"),
        ("head", "feature//unsafe", "head"),
    ],
)
def test_pull_request_handoff_rejects_ambiguous_targets(tmp_path, field, value, message):
    _project(tmp_path)
    with pytest.raises(AgentWorkspaceError, match = message):
        _prepare(tmp_path, **{field: value})


def test_pull_request_handoff_expires_and_cannot_be_replayed(tmp_path):
    _project(tmp_path)
    preview = _prepare(tmp_path)

    with pytest.raises(AgentWorkspaceError, match = "expired"):
        consume_pull_request_handoff(
            "project",
            preview["id"],
            server_id = "github",
            confirmation_token = preview["confirmationToken"],
            expected_request_digest = preview["requestDigest"],
            tools = _TOOLS,
            now = 701,
        )
    with pytest.raises(AgentWorkspaceError, match = "already used"):
        consume_pull_request_handoff(
            "project",
            preview["id"],
            server_id = "github",
            confirmation_token = preview["confirmationToken"],
            expected_request_digest = preview["requestDigest"],
            tools = _TOOLS,
            now = 101,
        )


def test_pull_request_handoff_rejects_changed_connector_and_preview(tmp_path):
    _project(tmp_path)
    changed_preview = _prepare(tmp_path)
    mcp_servers_db.update_server("github", {"url": "https://changed.invalid/mcp"})
    with pytest.raises(AgentWorkspaceError, match = "connector changed"):
        consume_pull_request_handoff(
            "project",
            changed_preview["id"],
            server_id = "github",
            confirmation_token = changed_preview["confirmationToken"],
            expected_request_digest = changed_preview["requestDigest"],
            tools = _TOOLS,
            now = 101,
        )

    current = mcp_servers_db.get_server("github")
    mcp_servers_db.update_server("github", {"url": "https://github.example.invalid/mcp"})
    assert current is not None
    digest_preview = _prepare(tmp_path)
    with pytest.raises(AgentWorkspaceError, match = "preview changed"):
        consume_pull_request_handoff(
            "project",
            digest_preview["id"],
            server_id = "github",
            confirmation_token = digest_preview["confirmationToken"],
            expected_request_digest = "0" * 64,
            tools = _TOOLS,
            now = 101,
        )


def test_pull_request_handoff_rejects_repository_change_after_preview(tmp_path):
    _project(tmp_path)
    preview = _prepare(tmp_path)
    assert preview["reviewBinding"]["head"] == _git(tmp_path, "rev-parse", "HEAD")

    (tmp_path / "tracked.txt").write_text("changed after review\n", encoding = "utf-8")
    with pytest.raises(AgentWorkspaceError, match = "repository changed"):
        consume_pull_request_handoff(
            "project",
            preview["id"],
            server_id = "github",
            confirmation_token = preview["confirmationToken"],
            expected_request_digest = preview["requestDigest"],
            tools = _TOOLS,
            now = 101,
        )


def test_pull_request_handoff_rejects_new_head_after_preview(tmp_path):
    _project(tmp_path)
    preview = _prepare(tmp_path)
    (tmp_path / "tracked.txt").write_text("new head\n", encoding = "utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "-qm", "new head")

    with pytest.raises(AgentWorkspaceError, match = "repository changed"):
        consume_pull_request_handoff(
            "project",
            preview["id"],
            server_id = "github",
            confirmation_token = preview["confirmationToken"],
            expected_request_digest = preview["requestDigest"],
            tools = _TOOLS,
            now = 101,
        )


def test_pull_request_handoff_requires_an_exact_compatible_mutation_tool(tmp_path):
    _project(tmp_path)
    incompatible = [
        {
            "name": "create_pull_request",
            "inputSchema": {
                "type": "object",
                "properties": {"owner": {"type": "string"}},
            },
        }
    ]
    with pytest.raises(AgentWorkspaceError, match = "compatible"):
        _prepare(tmp_path, tools = incompatible)


def test_connected_pull_request_route_previews_then_calls_connector_once(tmp_path, monkeypatch):
    _project(tmp_path)
    calls = []

    async def list_tools_async(**_kwargs):
        return _TOOLS

    def call_tool_sync(**kwargs):
        calls.append(kwargs)
        assert kwargs["config_check"]() is True
        return "https://github.example.invalid/unslothai/unsloth/pull/1"

    monkeypatch.setattr(agent_workspace_routes, "list_tools_async", list_tools_async)
    monkeypatch.setattr(agent_workspace_routes, "call_tool_sync", call_tool_sync)
    client = _client()
    prepared = client.post(
        "/api/agent-workspace/projects/project/review/pull-request-handoff/prepare",
        json = {
            "serverId": "github",
            "owner": "unslothai",
            "repository": "unsloth",
            "base": "main",
            "head": "feature/codex-workspace",
            "bodyNote": f"local={tmp_path} password=hunter2",
            "draft": True,
        },
    )

    assert prepared.status_code == 200
    preview = prepared.json()
    assert preview["submitted"] is False
    assert calls == []
    assert str(tmp_path) not in prepared.text
    assert "hunter2" not in prepared.text

    confirmed = client.post(
        "/api/agent-workspace/projects/project/review/"
        f"pull-request-handoff/{preview['id']}/confirm",
        json = {
            "serverId": "github",
            "confirmationToken": preview["confirmationToken"],
            "expectedRequestDigest": preview["requestDigest"],
        },
    )
    replay = client.post(
        "/api/agent-workspace/projects/project/review/"
        f"pull-request-handoff/{preview['id']}/confirm",
        json = {
            "serverId": "github",
            "confirmationToken": preview["confirmationToken"],
            "expectedRequestDigest": preview["requestDigest"],
        },
    )

    assert confirmed.status_code == 200
    assert confirmed.json()["submitted"] is True
    assert len(calls) == 1
    assert calls[0]["name"] == "create_pull_request"
    assert calls[0]["args"]["draft"] is True
    assert str(tmp_path) not in str(calls[0]["args"])
    assert "hunter2" not in str(calls[0]["args"])
    assert replay.status_code == 409
    assert len(calls) == 1


def test_connected_pull_request_route_treats_connector_error_as_unknown_outcome(
    tmp_path, monkeypatch
):
    _project(tmp_path)

    async def list_tools_async(**_kwargs):
        return _TOOLS

    monkeypatch.setattr(agent_workspace_routes, "list_tools_async", list_tools_async)
    monkeypatch.setattr(
        agent_workspace_routes,
        "call_tool_sync",
        lambda **_kwargs: "Error: connector timed out after submission",
    )
    client = _client()
    prepared = client.post(
        "/api/agent-workspace/projects/project/review/pull-request-handoff/prepare",
        json = {
            "serverId": "github",
            "owner": "unslothai",
            "repository": "unsloth",
            "base": "main",
            "head": "feature/codex-workspace",
        },
    ).json()

    confirmed = client.post(
        "/api/agent-workspace/projects/project/review/"
        f"pull-request-handoff/{prepared['id']}/confirm",
        json = {
            "serverId": "github",
            "confirmationToken": prepared["confirmationToken"],
            "expectedRequestDigest": prepared["requestDigest"],
        },
    )

    assert confirmed.status_code == 502
    assert "Check GitHub" in confirmed.json()["detail"]
    assert "timed out" not in confirmed.text
