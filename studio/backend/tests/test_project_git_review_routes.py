# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import pytest
from fastapi import HTTPException

from core.agent_workspace.git_context import AgentWorkspaceError
from routes import project_git_review


def _project(revision: int = 4) -> dict:
    return {
        "id": "project-one",
        "archived": False,
        "workspaceRevision": revision,
    }


def test_status_and_diff_routes_forward_only_read_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(project_git_review, "get_chat_project", lambda project_id: _project())
    calls: list[tuple] = []

    def status(project_id: str) -> dict:
        calls.append(("status", project_id))
        return {"projectId": project_id, "workspaceRevision": 4}

    def diff(project_id: str, *, mode: str, max_bytes: int) -> dict:
        calls.append(("diff", project_id, mode, max_bytes))
        return {"projectId": project_id, "workspaceRevision": 4}

    monkeypatch.setattr(project_git_review, "git_status", status)
    monkeypatch.setattr(project_git_review, "build_diff_manifest", diff)

    assert (
        project_git_review.project_git_status("project-one", 4, "subject")["workspaceRevision"] == 4
    )
    assert (
        project_git_review.project_git_diff(
            "project-one",
            4,
            "staged",
            65_536,
            "subject",
        )["workspaceRevision"]
        == 4
    )
    assert calls == [("status", "project-one"), ("diff", "project-one", "staged", 65_536)]
    route_paths = {route.path for route in project_git_review.router.routes}
    assert route_paths == {
        "/projects/{project_id}/git/status",
        "/projects/{project_id}/git/diff",
    }
    assert all(set(route.methods or ()) == {"GET"} for route in project_git_review.router.routes)


@pytest.mark.parametrize("project", [None, {"archived": True}])
def test_routes_hide_missing_and_archived_projects(
    monkeypatch: pytest.MonkeyPatch, project: dict | None
) -> None:
    monkeypatch.setattr(project_git_review, "get_chat_project", lambda project_id: project)
    with pytest.raises(HTTPException) as error:
        project_git_review.project_git_status("project-one", 0, "subject")
    assert error.value.status_code == 404


def test_workspace_revision_is_fenced_before_and_after_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(project_git_review, "get_chat_project", lambda project_id: _project(5))
    called = False

    def status(project_id: str) -> dict:
        nonlocal called
        called = True
        return {"projectId": project_id, "workspaceRevision": 6}

    monkeypatch.setattr(project_git_review, "git_status", status)
    with pytest.raises(HTTPException) as stale_before:
        project_git_review.project_git_status("project-one", 4, "subject")
    assert stale_before.value.status_code == 409
    assert called is False

    with pytest.raises(HTTPException) as stale_after:
        project_git_review.project_git_status("project-one", 5, "subject")
    assert stale_after.value.status_code == 409
    assert called is True


def test_workspace_failures_are_safe_conflicts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(project_git_review, "get_chat_project", lambda project_id: _project())

    def fail(project_id: str) -> dict:
        raise AgentWorkspaceError("The Git repository identity changed.")

    monkeypatch.setattr(project_git_review, "git_status", fail)
    with pytest.raises(HTTPException) as error:
        project_git_review.project_git_status("project-one", 4, "subject")
    assert error.value.status_code == 409
    assert error.value.detail == "The Git repository identity changed."
