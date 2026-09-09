# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Authenticated route contracts for project verification."""

from __future__ import annotations

import contextlib
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from core.agent_workspace import verification
from core.agent_workspace.verification_context import AgentWorkspaceError, ProjectWorkspace
from routes import chat_history, project_verification


def _workspace(
    tmp_path: Path,
    project_id: str = "project",
    revision: int = 7,
):
    root = tmp_path / project_id
    root.mkdir(exist_ok = True)
    metadata = root.stat(follow_symlinks = False)
    return ProjectWorkspace(
        project_id = project_id,
        root = root.resolve(strict = True),
        kind = "folder",
        device_id = int(metadata.st_dev),
        file_id = int(metadata.st_ino),
        revision = revision,
    )


def _check(command: str = "python -m pytest"):
    return {
        "name": "tests",
        "kind": "custom",
        "command": command,
        "required": True,
        "timeoutSeconds": 30,
        "logLimitBytes": 4096,
    }


def _profile(
    workspace: ProjectWorkspace,
    *,
    revision: int = 0,
    checks = None,
):
    return {
        "projectId": workspace.project_id,
        "checks": list(checks or []),
        "revision": revision,
        "configHash": "b" * 64,
        "workspaceDeviceId": int(workspace.device_id) if revision else None,
        "workspaceFileId": int(workspace.file_id) if revision else None,
        "workspaceRevision": int(workspace.revision) if revision else None,
        "updatedAt": 10 if revision else None,
    }


def _run(
    project_id: str,
    run_id: str = "run-1",
    status: str = "running",
):
    return {
        "id": run_id,
        "projectId": project_id,
        "status": status,
        "configRevision": 1,
        "configHash": "c" * 64,
        "ownerId": "private-owner",
        "heartbeatAt": 1,
        "leaseExpiresAt": 2,
        "workspaceDeviceId": 11,
        "workspaceFileId": 22,
        "workspaceRevision": 7,
        "evidenceRevision": 1,
        "checks": [_check()],
        "results": [],
        "cancelRequested": False,
        "error": None,
        "startedAt": 1,
        "updatedAt": 1,
        "completedAt": None,
        "historySequence": None,
    }


class _RouteState:
    def __init__(self, workspace: ProjectWorkspace):
        self.workspace = workspace
        self.profile = _profile(workspace)
        self.runs: dict[tuple[str, str], dict] = {}
        self.set_calls = []

    def get_config(self, project_id: str):
        assert project_id == self.workspace.project_id
        return dict(self.profile) | {"checks": [dict(check) for check in self.profile["checks"]]}

    def set_config(
        self,
        project_id: str,
        checks: list[dict],
        *,
        workspace_identity,
        workspace_revision,
        expected_revision,
    ):
        self.set_calls.append(
            {
                "projectId": project_id,
                "checks": checks,
                "workspaceIdentity": workspace_identity,
                "workspaceRevision": workspace_revision,
                "expectedRevision": expected_revision,
            }
        )
        if expected_revision != self.profile["revision"]:
            raise AgentWorkspaceError(
                "Verification settings changed in another session. Refresh and retry."
            )
        self.profile = _profile(
            self.workspace,
            revision = self.profile["revision"] + 1,
            checks = checks,
        )
        return self.get_config(project_id)

    def active(self, project_id: str):
        return next(
            (
                dict(run)
                for (owned_project, _run_id), run in self.runs.items()
                if owned_project == project_id and run["status"] == "running"
            ),
            None,
        )

    def list_run_summaries(self, project_id: str, *, limit: int):
        return [
            {key: value for key, value in run.items() if key not in {"checks", "results"}}
            | {
                "status": "running" if run["status"] == "running" else "unverified",
                "evidenceStatus": "not_loaded",
            }
            for (owned_project, _run_id), run in self.runs.items()
            if owned_project == project_id
        ][:limit]

    def get_run(self, project_id: str, run_id: str):
        run = self.runs.get((project_id, run_id))
        return dict(run) if run is not None else None

    def get_run_evidence_revision(self, project_id: str, run_id: str):
        run = self.runs.get((project_id, run_id))
        return int(run["evidenceRevision"]) if run is not None else None


def _bind_route_state(
    monkeypatch,
    state: _RouteState,
    project_ids = ("project",),
):
    projects = set(project_ids)
    monkeypatch.setattr(
        project_verification,
        "get_chat_project",
        lambda project_id: (
            {
                "id": project_id,
                "archived": False,
                "workspaceAvailable": True,
                "workspaceRevision": state.workspace.revision,
            }
            if project_id in projects
            else None
        ),
    )

    @contextlib.contextmanager
    def access(project_id, **_options):
        assert project_id == state.workspace.project_id
        yield state.workspace

    monkeypatch.setattr(project_verification.common, "project_workspace_access", access)
    monkeypatch.setattr(
        project_verification.verification_state,
        "get_verification_config",
        state.get_config,
    )
    monkeypatch.setattr(
        project_verification.verification_state,
        "set_verification_config",
        state.set_config,
    )
    monkeypatch.setattr(
        project_verification.verification_state,
        "active_verification_run",
        state.active,
    )
    monkeypatch.setattr(
        project_verification.verification_state,
        "list_verification_run_summaries",
        state.list_run_summaries,
    )
    monkeypatch.setattr(
        project_verification.verification_state,
        "get_verification_run",
        state.get_run,
    )
    monkeypatch.setattr(
        project_verification.verification_state,
        "get_verification_run_evidence_revision",
        state.get_run_evidence_revision,
    )
    monkeypatch.setattr(
        project_verification.verification,
        "execution_status",
        lambda: {"available": True, "backend": "test", "reason": None},
    )


def _client(*, via_api_key: bool = False, authenticated: bool = True):
    app = FastAPI()
    app.include_router(project_verification.router, prefix = "/api/agent")

    def subject():
        if not authenticated:
            raise HTTPException(status_code = 401, detail = "Authentication required.")
        return "api-key" if via_api_key else "ui-session"

    app.dependency_overrides[project_verification.get_current_subject] = subject
    app.dependency_overrides[project_verification.authenticated_via_api_key] = lambda: via_api_key
    return TestClient(app)


def _request(client: TestClient, method: str, path: str, body):
    request = getattr(client, method)
    if body is None:
        return request(path)
    return request(path, json = body)


@pytest.mark.parametrize(
    "method, path, body",
    [
        ("get", "/api/agent/projects/project/verification", None),
        (
            "put",
            "/api/agent/projects/project/verification",
            {"checks": [], "expectedRevision": 0, "workspaceRevision": 0},
        ),
        (
            "post",
            "/api/agent/projects/project/verifications",
            {"configRevision": 1, "workspaceRevision": 0},
        ),
        ("get", "/api/agent/projects/project/verifications", None),
        ("get", "/api/agent/projects/project/verifications/run-1", None),
        ("post", "/api/agent/projects/project/verifications/run-1/cancel", None),
    ],
)
def test_every_verification_route_requires_authentication(monkeypatch, method, path, body):
    monkeypatch.setattr(
        project_verification,
        "get_chat_project",
        lambda _project_id: pytest.fail("unauthenticated request reached project storage"),
    )
    response = _request(_client(authenticated = False), method, path, body)
    assert response.status_code == 401


@pytest.mark.parametrize(
    "method, path, body",
    [
        (
            "put",
            "/api/agent/projects/project/verification",
            {"checks": [], "expectedRevision": 0, "workspaceRevision": 0},
        ),
        (
            "post",
            "/api/agent/projects/project/verifications",
            {"configRevision": 1, "workspaceRevision": 0},
        ),
        ("post", "/api/agent/projects/project/verifications/run-1/cancel", None),
    ],
)
def test_api_keys_cannot_save_run_or_cancel_verification(monkeypatch, method, path, body):
    monkeypatch.setattr(
        project_verification,
        "get_chat_project",
        lambda _project_id: pytest.fail("API-key mutation reached project storage"),
    )
    response = _request(_client(via_api_key = True), method, path, body)
    assert response.status_code == 403
    assert "Unsloth UI" in response.text


def test_config_save_binds_workspace_revision_and_uses_cas(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    state = _RouteState(workspace)
    _bind_route_state(monkeypatch, state)
    client = _client()
    body = {
        "checks": [_check()],
        "expectedRevision": 0,
        "workspaceRevision": workspace.revision,
    }

    stale_workspace = client.put(
        f"/api/agent/projects/{workspace.project_id}/verification",
        json = body | {"workspaceRevision": workspace.revision + 1},
    )
    assert stale_workspace.status_code == 409
    assert state.set_calls == []

    saved = client.put(
        f"/api/agent/projects/{workspace.project_id}/verification",
        json = body,
    )
    assert saved.status_code == 200
    assert saved.json()["revision"] == 1
    assert saved.json()["active"] is True
    assert saved.json()["workspaceRevision"] == workspace.revision
    assert saved.json()["sourceFreshness"] == "unverified"
    assert saved.json()["execution"] == {"available": True, "backend": "test", "reason": None}
    assert state.set_calls[-1] == {
        "projectId": workspace.project_id,
        "checks": [_check()],
        "workspaceIdentity": (int(workspace.device_id), int(workspace.file_id)),
        "workspaceRevision": workspace.revision,
        "expectedRevision": 0,
    }

    stale_config = client.put(
        f"/api/agent/projects/{workspace.project_id}/verification",
        json = body,
    )
    assert stale_config.status_code == 409
    assert "another session" in stale_config.text


@pytest.mark.parametrize(
    "checks",
    [
        [_check(command = "é" * 8193)],
        [_check(command = "printf safe\u202e")],
        [
            {**_check(), "name": "Straße"},
            {**_check(command = "python -m pytest -q"), "name": "STRASSE"},
        ],
    ],
)
def test_profile_content_validation_is_422_not_workspace_conflict(tmp_path, monkeypatch, checks):
    workspace = _workspace(tmp_path)
    state = _RouteState(workspace)
    _bind_route_state(monkeypatch, state)

    response = _client().put(
        f"/api/agent/projects/{workspace.project_id}/verification",
        json = {
            "checks": checks,
            "expectedRevision": 0,
            "workspaceRevision": workspace.revision,
        },
    )

    assert response.status_code == 422
    assert state.set_calls == []


def test_start_returns_running_without_waiting_for_completion_and_hides_authority(
    tmp_path, monkeypatch
):
    workspace = _workspace(tmp_path)
    state = _RouteState(workspace)
    state.profile = _profile(workspace, revision = 1, checks = [_check()])
    _bind_route_state(monkeypatch, state)
    observed = []

    def start(project_id, *, config_revision, workspace_revision):
        observed.append((project_id, config_revision, workspace_revision))
        record = _run(project_id, status = "running")
        state.runs[(project_id, record["id"])] = record
        return record

    monkeypatch.setattr(
        project_verification.verification,
        "start_project_verification",
        start,
    )
    response = _client().post(
        f"/api/agent/projects/{workspace.project_id}/verifications",
        json = {"configRevision": 1, "workspaceRevision": workspace.revision},
    )

    assert response.status_code == 200
    assert observed == [(workspace.project_id, 1, workspace.revision)]
    assert response.json()["status"] == "running"
    assert response.json()["historySequence"] is None
    assert response.json()["sourceFreshness"] == "unverified"
    assert "configHash" not in response.json()
    assert "workspaceDeviceId" not in response.json()
    assert "workspaceFileId" not in response.json()
    assert "ownerId" not in response.json()
    assert "heartbeatAt" not in response.json()
    assert "leaseExpiresAt" not in response.json()


def test_config_and_run_reads_expose_live_progress_but_stay_project_scoped(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    state = _RouteState(workspace)
    state.profile = _profile(workspace, revision = 1, checks = [_check()])
    running = _run(workspace.project_id)
    running["evidenceRevision"] = 2
    running["results"] = [
        {
            **_check(),
            "status": "running",
            "exitCode": None,
            "output": "partial output",
            "outputBytes": 14,
            "outputTruncated": False,
            "startedAt": 2,
            "completedAt": None,
            "durationMs": None,
        }
    ]
    state.runs[(workspace.project_id, running["id"])] = running
    _bind_route_state(
        monkeypatch,
        state,
        project_ids = (workspace.project_id, "other-project"),
    )
    client = _client()

    config = client.get(f"/api/agent/projects/{workspace.project_id}/verification")
    assert config.status_code == 200
    assert config.json()["activeRun"]["id"] == running["id"]
    assert config.json()["activeRun"]["results"][0]["output"] == "partial output"

    exact = client.get(f"/api/agent/projects/{workspace.project_id}/verifications/{running['id']}")
    assert exact.status_code == 200
    assert exact.json()["results"][0]["status"] == "running"

    unchanged = client.get(
        f"/api/agent/projects/{workspace.project_id}/verifications/{running['id']}",
        params = {"afterEvidenceRevision": running["evidenceRevision"]},
    )
    assert unchanged.status_code == 204
    assert unchanged.content == b""

    changed = client.get(
        f"/api/agent/projects/{workspace.project_id}/verifications/{running['id']}",
        params = {"afterEvidenceRevision": running["evidenceRevision"] - 1},
    )
    assert changed.status_code == 200
    assert changed.json()["id"] == running["id"]

    regressed = client.get(
        f"/api/agent/projects/{workspace.project_id}/verifications/{running['id']}",
        params = {"afterEvidenceRevision": running["evidenceRevision"] + 1},
    )
    assert regressed.status_code == 409
    assert "revision regressed" in regressed.text

    history = client.get(f"/api/agent/projects/{workspace.project_id}/verifications")
    assert history.status_code == 200
    [summary] = history.json()["runs"]
    assert summary["evidenceStatus"] == "not_loaded"
    assert "checks" not in summary
    assert "results" not in summary
    assert "output" not in repr(summary)

    wrong_project = client.get(f"/api/agent/projects/other-project/verifications/{running['id']}")
    assert wrong_project.status_code == 404
    other_runs = client.get("/api/agent/projects/other-project/verifications")
    assert other_runs.status_code == 200
    assert other_runs.json() == {"runs": []}


def test_cancel_route_is_project_scoped_and_returns_durable_request_state(tmp_path, monkeypatch):
    workspace = _workspace(tmp_path)
    state = _RouteState(workspace)
    running = _run(workspace.project_id)
    state.runs[(workspace.project_id, running["id"])] = running
    _bind_route_state(
        monkeypatch,
        state,
        project_ids = (workspace.project_id, "other-project"),
    )
    cancelled = []

    def cancel(project_id, run_id):
        record = state.get_run(project_id, run_id)
        if record is None:
            raise AgentWorkspaceError("Verification run not found.")
        record["cancelRequested"] = True
        cancelled.append((project_id, run_id))
        return record, True

    monkeypatch.setattr(project_verification.verification, "cancel_verification", cancel)
    client = _client()

    cross_project = client.post(
        f"/api/agent/projects/other-project/verifications/{running['id']}/cancel"
    )
    assert cross_project.status_code == 409
    assert cancelled == []

    response = client.post(
        f"/api/agent/projects/{workspace.project_id}/verifications/{running['id']}/cancel"
    )
    assert response.status_code == 200
    assert response.json()["cancelRequested"] is True
    assert response.json()["run"]["cancelRequested"] is True
    assert cancelled == [(workspace.project_id, running["id"])]


def test_archive_durably_retires_verification_before_project_mutation(monkeypatch):
    project_id = "archive-project"
    order = []

    monkeypatch.setattr(
        chat_history,
        "get_chat_project",
        lambda requested_project: (
            {"id": requested_project, "archived": False}
            if requested_project == project_id
            else None
        ),
    )
    monkeypatch.setattr(
        verification,
        "begin_project_deletion",
        lambda requested_project: order.append(("durable-fence", requested_project)),
    )
    monkeypatch.setattr(
        verification,
        "cancel_project_verifications_and_wait",
        lambda requested_project: order.append(("cancel-and-fence-wait", requested_project)),
    )
    monkeypatch.setattr(
        verification,
        "finish_project_deletion",
        lambda requested_project: order.append(("release-fence", requested_project)),
    )

    def update(requested_project, patch, **_kwargs):
        assert patch == {"archived": True}
        assert order[-1] == ("cancel-and-fence-wait", project_id)
        order.append(("archive-row", requested_project))
        return {
            "id": requested_project,
            "archived": True,
            "name": "Archive Project",
            "createdAt": 1,
            "updatedAt": 1,
        }

    monkeypatch.setattr(chat_history, "update_chat_project", update)
    monkeypatch.setattr(
        chat_history,
        "ensure_chat_project_workspace",
        lambda _project_id: {
            "id": project_id,
            "name": "Archive Project",
            "archived": True,
            "createdAt": 1,
            "updatedAt": 1,
        },
    )

    result = chat_history.patch_project(
        project_id,
        chat_history.ChatProjectPatch(archived = True),
        current_subject = "ui-session",
    )

    assert result.archived is True
    assert order == [
        ("durable-fence", project_id),
        ("cancel-and-fence-wait", project_id),
        ("archive-row", project_id),
        ("release-fence", project_id),
    ]
