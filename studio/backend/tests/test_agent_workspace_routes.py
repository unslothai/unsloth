# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject
from core.agent_workspace.execution import ExecutionBoundaryStatus
from core.agent_workspace.common import AgentWorkspaceError
from routes import agent_workspace as agent_workspace_routes
from routes.agent_workspace import router
from storage import studio_db


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix = "/api/agent-workspace")
    app.dependency_overrides[get_current_subject] = lambda: "test-subject"
    return TestClient(app)


def _folder_project(root: Path) -> None:
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
            "archived": False,
            "createdAt": 1,
            "updatedAt": 1,
        }
    )


def test_workspace_capability_response_never_exposes_root(tmp_path):
    _folder_project(tmp_path)

    response = _client().get("/api/agent-workspace/projects/project/workspace")

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["workspaceKind"] == "folder"
    assert body["capabilities"]["instructions"] is True
    assert "rootPath" not in body
    assert str(tmp_path) not in response.text


def test_context_snapshot_endpoint_returns_only_an_opaque_expiring_id(tmp_path):
    _folder_project(tmp_path)
    response = _client().post("/api/agent-workspace/projects/project/context-snapshots")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"id", "expiresAt"}
    assert isinstance(body["id"], str) and len(body["id"]) >= 32
    assert "project" not in body["id"]
    assert isinstance(body["expiresAt"], int)
    assert str(tmp_path) not in response.text


def test_context_snapshot_endpoint_binds_the_compare_query(tmp_path, monkeypatch):
    _folder_project(tmp_path)
    observed = {}
    marker = object()

    def capture(project_id, query):
        observed.update(projectId = project_id, query = query)
        return marker

    monkeypatch.setattr(agent_workspace_routes, "create_project_context_snapshot", capture)
    monkeypatch.setattr(
        agent_workspace_routes,
        "project_context_snapshot_response",
        lambda value: {"id": "opaque-snapshot-id", "expiresAt": 123} if value is marker else None,
    )

    response = _client().post(
        "/api/agent-workspace/projects/project/context-snapshots",
        json = {"query": "Update src/service.py"},
    )

    assert response.status_code == 200
    assert observed == {"projectId": "project", "query": "Update src/service.py"}
    assert response.json() == {"id": "opaque-snapshot-id", "expiresAt": 123}


def test_unavailable_workspace_has_stable_false_capabilities(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    _folder_project(root)
    root.rmdir()

    response = _client().get("/api/agent-workspace/projects/project/workspace")

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["isGitRepository"] is False
    assert body["capabilities"] == {
        "instructions": False,
        "repositoryMap": False,
        "verification": False,
        "plans": False,
        "background": False,
        "git": False,
        "worktrees": False,
        "review": False,
    }


def test_unavailable_workspace_error_redacts_local_paths(tmp_path, monkeypatch):
    _folder_project(tmp_path)

    def unavailable(_project_id):
        raise AgentWorkspaceError(f"Workspace missing at {tmp_path / 'private'}")

    monkeypatch.setattr(agent_workspace_routes, "project_workspace", unavailable)
    response = _client().get("/api/agent-workspace/projects/project/workspace")

    assert response.status_code == 200
    assert response.json()["available"] is False
    assert str(tmp_path) not in response.json()["error"]
    assert "<local_path>" in response.json()["error"]


def test_verification_route_persists_config_and_documents_shell_contract(tmp_path):
    _folder_project(tmp_path)
    client = _client()

    saved = client.put(
        "/api/agent-workspace/projects/project/verification",
        json = {
            "requireForGoalCompletion": True,
            "expectedRevision": 0,
            "checks": [
                {
                    "name": "test",
                    "kind": "test",
                    "command": "python -m pytest",
                    "required": True,
                    "timeoutSeconds": 60,
                    "logLimitBytes": 4096,
                }
            ],
        },
    )

    assert saved.status_code == 200
    contract = saved.json()["shellContract"]
    assert "Writes are limited to the project" in contract
    assert "network access is disabled" in contract
    assert saved.json()["requireForGoalCompletion"] is True
    assert saved.json()["revision"] == 1
    loaded = client.get("/api/agent-workspace/projects/project/verification")
    assert loaded.status_code == 200
    assert loaded.json()["checks"][0]["name"] == "test"
    assert loaded.json()["requireForGoalCompletion"] is True
    assert loaded.json()["revision"] == 1

    resaved = client.put(
        "/api/agent-workspace/projects/project/verification",
        json = {
            "requireForGoalCompletion": True,
            "expectedRevision": 1,
            "checks": loaded.json()["checks"],
        },
    )
    assert resaved.status_code == 200
    assert resaved.json()["revision"] == 2

    stale = client.put(
        "/api/agent-workspace/projects/project/verification",
        json = {
            "requireForGoalCompletion": False,
            "expectedRevision": 1,
            "checks": loaded.json()["checks"],
        },
    )
    assert stale.status_code == 409
    assert "another session" in stale.text


def test_unsupported_execution_host_disables_and_rejects_command_routes(tmp_path, monkeypatch):
    _folder_project(tmp_path)
    monkeypatch.setattr(
        agent_workspace_routes,
        "execution_boundary_status",
        lambda: ExecutionBoundaryStatus(False, None, "No secure command boundary."),
    )
    client = _client()

    workspace = client.get("/api/agent-workspace/projects/project/workspace")
    verification = client.post(
        "/api/agent-workspace/projects/project/verify",
        json = {"configRevision": 0},
    )
    background = client.post(
        "/api/agent-workspace/projects/project/background/verification",
        json = {"configRevision": 0, "start": False},
    )

    assert workspace.status_code == 200
    assert workspace.json()["executionBoundary"] == {
        "available": False,
        "backend": None,
        "reason": "No secure command boundary.",
    }
    assert workspace.json()["capabilities"]["verification"] is False
    assert workspace.json()["capabilities"]["background"] is False
    assert verification.status_code == 409
    assert background.status_code == 409
    assert "No secure command boundary" in verification.text


def test_workspace_capabilities_gate_secure_traversal_support(tmp_path, monkeypatch):
    _folder_project(tmp_path)
    monkeypatch.setattr(
        agent_workspace_routes, "secure_instruction_traversal_supported", lambda: False
    )
    monkeypatch.setattr(
        agent_workspace_routes,
        "secure_repository_traversal_supported",
        lambda: False,
    )

    response = _client().get("/api/agent-workspace/projects/project/workspace")

    assert response.status_code == 200
    capabilities = response.json()["capabilities"]
    assert capabilities["instructions"] is False
    assert capabilities["repositoryMap"] is False


def test_native_git_and_worktree_ownership_fields_are_not_serialized(tmp_path, monkeypatch):
    _folder_project(tmp_path)
    internal_worktree = {
        "id": "worktree",
        "projectId": "project",
        "gitRoot": str(tmp_path),
        "path": str(tmp_path / "workspace"),
        "branch": "unsloth-studio/task",
        "baseRef": "HEAD",
        "markerPath": str(tmp_path / "owner.json"),
        "markerTokenHash": "secret-proof",
        "backgroundTaskId": None,
        "status": "active",
        "createdAt": 1,
        "updatedAt": 1,
    }
    monkeypatch.setattr(
        agent_workspace_routes, "list_worktrees", lambda project_id: [internal_worktree]
    )
    monkeypatch.setattr(
        agent_workspace_routes,
        "git_status",
        lambda project_id: {
            "repositoryRoot": str(tmp_path),
            "projectPrefix": ".",
            "head": "a" * 40,
            "branch": "main",
            "detached": False,
            "clean": True,
            "counts": {"staged": 0, "unstaged": 0, "untracked": 0, "conflicted": 0},
            "files": [],
            "truncated": False,
        },
    )
    client = _client()

    status = client.get("/api/agent-workspace/projects/project/git/status")
    worktrees = client.get("/api/agent-workspace/projects/project/worktrees")

    assert status.status_code == 200
    assert "repositoryRoot" not in status.json()
    assert "projectPrefix" not in status.json()
    assert worktrees.status_code == 200
    serialized = worktrees.json()["worktrees"][0]
    assert set(serialized) == {
        "id",
        "projectId",
        "branch",
        "baseRef",
        "backgroundTaskId",
        "status",
        "createdAt",
        "updatedAt",
    }
    assert str(tmp_path) not in worktrees.text
    assert "secret-proof" not in worktrees.text


def test_agent_workspace_errors_redact_local_paths(tmp_path, monkeypatch):
    _folder_project(tmp_path)
    hidden = tmp_path / ".agent-worktrees" / "secret" / "workspace"

    def fail_cleanup(project_id, worktree_id):
        raise AgentWorkspaceError(f"git refused to clean {hidden}")

    monkeypatch.setattr(agent_workspace_routes, "cleanup_worktree", fail_cleanup)

    response = _client().delete("/api/agent-workspace/projects/project/worktrees/worktree")

    assert response.status_code == 409
    assert str(hidden) not in response.text
    assert "<local_path>" in response.json()["detail"]


def test_prepared_commit_routes_enforce_two_phase_payload_and_hide_internal_state(
    tmp_path, monkeypatch
):
    _folder_project(tmp_path)
    token = "t" * 43
    internal = {
        "id": "preparation",
        "projectId": "project",
        "status": "awaiting_confirmation",
        "branch": "main",
        "baseHead": "a" * 40,
        "message": "Selected files",
        "ownedPaths": ["owned.txt"],
        "sourceFingerprint": "b" * 64,
        "createdAt": 1,
        "expiresAt": 2,
        "confirmationToken": token,
        "files": [{"code": " M", "path": "owned.txt"}],
        "diff": "reviewed diff",
        "diffTruncated": False,
        "gitRoot": str(tmp_path),
        "tokenDigest": "secret-digest",
        "payloadDigest": "secret-payload",
    }
    monkeypatch.setattr(
        agent_workspace_routes,
        "prepare_commit",
        lambda project_id, owned_paths, message: internal,
    )
    monkeypatch.setattr(
        agent_workspace_routes,
        "confirm_prepared_commit",
        lambda project_id, preparation_id, confirmation_token: {
            **internal,
            "status": "confirmed",
            "commitSha": "c" * 40,
            "refName": "refs/unsloth-studio/prepared-commits/preparation",
            "confirmedAt": 3,
        },
    )
    client = _client()

    prepared = client.post(
        "/api/agent-workspace/projects/project/git/commits/prepare",
        json = {"ownedPaths": ["owned.txt"], "message": "Selected files"},
    )
    confirmed = client.post(
        "/api/agent-workspace/projects/project/git/commits/preparations/preparation/confirm",
        json = {"confirmationToken": token},
    )

    assert prepared.status_code == 200
    assert prepared.json()["confirmationToken"] == token
    assert confirmed.status_code == 200
    assert confirmed.json()["commitSha"] == "c" * 40
    for response in (prepared, confirmed):
        assert "gitRoot" not in response.json()
        assert "tokenDigest" not in response.json()
        assert "payloadDigest" not in response.json()
        assert str(tmp_path) not in response.text
