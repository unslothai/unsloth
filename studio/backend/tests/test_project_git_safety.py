# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject, authenticated_via_api_key
from core.agent_workspace import (
    checkpoints,
    git_service,
    git_retirement,
    github_handoff,
    prepared_commits,
    prepared_commit_state,
    worktrees,
)
from core.agent_workspace.git_context import AgentWorkspaceError
from routes import project_worktrees
from .test_agent_workspace_worktrees_focused import _setup, _git, managed_workspace_records  # noqa: F401


def _client(api_key = False):
    app = FastAPI()
    app.include_router(project_worktrees.router)
    app.dependency_overrides[get_current_subject] = lambda: "test"
    app.dependency_overrides[authenticated_via_api_key] = lambda: api_key
    return TestClient(app)


def test_writes_require_ui_auth_and_workspace_revision(tmp_path):
    _setup(tmp_path)
    with _client(api_key = True) as client:
        response = client.post("/projects/project/worktrees?workspaceRevision=0", json = {})
        assert response.status_code == 403
    with _client() as client:
        assert client.post("/projects/project/worktrees", json = {}).status_code == 422
        assert (
            client.post("/projects/project/worktrees?workspaceRevision=3", json = {}).status_code
            == 409
        )


def test_windows_mutation_boundary_rejects_before_git(monkeypatch):
    monkeypatch.setattr(git_service, "os", type("Windows", (), {"name": "nt"}))
    with pytest.raises(AgentWorkspaceError, match = "Windows"):
        git_service._run(Path("."), ["status"])


@pytest.mark.skipif(os.name == "nt", reason = "Native POSIX mutation contract")
def test_prepared_ref_is_one_use_and_preserves_head_index_and_unselected_paths(tmp_path):
    root = _setup(tmp_path)
    (root / "tracked.txt").write_text("selected\n")
    (root / "unselected.txt").write_text("staged separately\n")
    _git(root, "add", "unselected.txt")
    head = _git(root, "rev-parse", "HEAD")
    index = (root / ".git/index").read_bytes()
    preview = prepared_commits.prepare_commit("project", ["tracked.txt"], "Reviewed change")
    assert _git(root, "for-each-ref", "refs/unsloth-studio/prepared-commits/") == ""
    result = prepared_commits.confirm_prepared_commit(
        "project", preview["id"], preview["confirmationToken"]
    )
    assert _git(root, "rev-parse", "HEAD") == head
    assert (root / ".git/index").read_bytes() == index
    assert _git(root, "show", f"{result['commitSha']}:tracked.txt") == "selected"
    assert _git(root, "ls-tree", result["commitSha"], "unselected.txt") == ""
    with pytest.raises(AgentWorkspaceError, match = "already used"):
        prepared_commits.confirm_prepared_commit(
            "project", preview["id"], preview["confirmationToken"]
        )
    prepared_commits.remove_prepared_commit("project", preview["id"])
    assert prepared_commit_state.get_preparation(preview["id"]) is None


@pytest.mark.skipif(os.name == "nt", reason = "Native POSIX mutation contract")
def test_prepared_commit_rejects_content_change_without_publishing(tmp_path):
    root = _setup(tmp_path)
    (root / "tracked.txt").write_text("reviewed\n")
    preview = prepared_commits.prepare_commit("project", ["tracked.txt"], "Review")
    (root / "tracked.txt").write_text("changed\n")
    with pytest.raises(AgentWorkspaceError, match = "changed"):
        prepared_commits.confirm_prepared_commit(
            "project", preview["id"], preview["confirmationToken"]
        )
    assert _git(root, "for-each-ref", "refs/unsloth-studio/prepared-commits/") == ""


@pytest.mark.skipif(os.name == "nt", reason = "Native POSIX mutation contract")
def test_uncertain_ref_publication_remains_durable_and_blocks_deletion(tmp_path, monkeypatch):
    root = _setup(tmp_path)
    (root / "tracked.txt").write_text("reviewed\n")
    preview = prepared_commits.prepare_commit("project", ["tracked.txt"], "Review")
    original = prepared_commits.repository_command

    def dispatched(root, args, **kwargs):
        result = original(root, args, **kwargs)
        if args[0] == "update-ref":
            raise RuntimeError("lost acknowledgement")
        return result

    monkeypatch.setattr(prepared_commits, "repository_command", dispatched)
    with pytest.raises(RuntimeError, match = "lost acknowledgement"):
        prepared_commits.confirm_prepared_commit(
            "project", preview["id"], preview["confirmationToken"]
        )
    record = prepared_commit_state.get_preparation(preview["id"])
    assert record["status"] == "confirming"
    assert _git(root, "rev-parse", record["refName"]) == record["commitSha"]
    with pytest.raises(AgentWorkspaceError, match = "prepared commit refs"):
        git_retirement.begin_git_retirement("project", deleting = True)


@pytest.mark.skipif(os.name == "nt", reason = "Native POSIX mutation contract")
def test_ignored_files_prevent_cleanup_and_recovery_branch_survives(tmp_path):
    root = _setup(tmp_path)
    (root / ".gitignore").write_text("cache/\n")
    _git(root, "add", ".gitignore")
    _git(root, "commit", "-qm", "ignore cache")
    owned = worktrees.create_worktree("project")
    target = Path(owned["path"])
    (target / "cache").mkdir()
    (target / "cache/model.bin").write_bytes(b"valuable")
    with pytest.raises(AgentWorkspaceError, match = "ignored files"):
        worktrees.cleanup_worktree("project", owned["id"])
    assert (target / "cache/model.bin").read_bytes() == b"valuable"
    assert _git(root, "rev-parse", owned["branch"])


@pytest.mark.skipif(os.name == "nt", reason = "Native POSIX mutation contract")
def test_included_git_filters_and_hooks_cannot_execute_during_checkpoint(tmp_path):
    root = _setup(tmp_path)
    marker = tmp_path / "executed"
    included = root / ".git/extra-config"
    included.write_text(
        f'[filter "danger"]\n clean = touch {marker}\n smudge = touch {marker}\n required = true\n'
    )
    _git(root, "config", "include.path", str(included))
    (root / ".gitattributes").write_text("*.txt filter=danger\n")
    (root / "tracked.txt").write_text("safe contents\n")
    hook = root / ".git/hooks/pre-commit"
    hook.write_text(f"#!/bin/sh\ntouch {marker}\n")
    hook.chmod(0o755)
    result = checkpoints.create_checkpoint("project", ["tracked.txt"])
    assert not marker.exists()
    assert _git(root, "show", f"{result['commitSha']}:tracked.txt") == "safe contents"


@pytest.mark.skipif(os.name == "nt", reason = "Native POSIX mutation contract")
def test_project_deletion_cannot_cascade_checkpoint_ownership(tmp_path):
    root = _setup(tmp_path)
    (root / "tracked.txt").write_text("snapshot\n")
    record = checkpoints.create_checkpoint("project", ["tracked.txt"])
    with pytest.raises(AgentWorkspaceError, match = "checkpoint"):
        git_retirement.begin_git_retirement("project", deleting = True)
    assert checkpoints.list_checkpoints("project")[0]["id"] == record["id"]
    assert _git(root, "rev-parse", record["refName"]) == record["commitSha"]


@pytest.mark.skipif(os.name == "nt", reason = "Native POSIX process fence")
def test_retirement_holds_shared_execution_fence_until_finished(tmp_path):
    from core.agent_workspace.process_fence import (
        _acquire_project_execution_fence,
        _release_project_execution_fence,
    )

    _setup(tmp_path)
    retirement = git_retirement.begin_git_retirement("project", deleting = True)
    try:
        with pytest.raises(TimeoutError):
            _acquire_project_execution_fence("project:project", None, time.monotonic() + 0.1)
    finally:
        git_retirement.finish_git_retirement("project", retirement)
    descriptor = _acquire_project_execution_fence("project:project", None, time.monotonic() + 1)
    _release_project_execution_fence(descriptor)


def test_verification_stop_failure_releases_retirement_admission(tmp_path, monkeypatch):
    from core.agent_workspace.git_state import require_git_admission

    _setup(tmp_path)
    active = set()

    def cannot_stop(project_id):
        assert project_id in active
        raise AgentWorkspaceError("Process tree is still running")

    verification = SimpleNamespace(
        begin_project_deletion = active.add,
        cancel_project_verifications_and_wait = cannot_stop,
        finish_project_deletion = active.remove,
    )
    monkeypatch.setattr(
        git_retirement, "importlib", SimpleNamespace(import_module = lambda *args: verification)
    )
    with pytest.raises(AgentWorkspaceError, match = "still running"):
        git_retirement.begin_git_retirement("project", deleting = True)
    assert not active
    require_git_admission("project")
    worktrees.begin_project_deletion("project")
    worktrees.finish_project_deletion("project")


def test_remote_head_must_match_reviewed_commit():
    expected = "a" * 40
    github_handoff.require_remote_head(json.dumps({"sha": expected}), expected)
    for value in (json.dumps({"sha": "b" * 40}), "Error: disconnected", "{}", "[]"):
        with pytest.raises(AgentWorkspaceError, match = "No pull request was submitted"):
            github_handoff.require_remote_head(value, expected)


def test_remote_probe_requires_read_contract_and_exact_destination():
    request = {"owner": "owner", "repo": "repo", "head": "feature"}
    with pytest.raises(AgentWorkspaceError, match = "get_commit"):
        github_handoff.remote_head_probe([], request)
    tools = [
        {
            "name": "get_commit",
            "inputSchema": {
                "properties": {"owner": {}, "repo": {}, "sha": {}},
                "required": ["owner", "repo", "sha"],
            },
        }
    ]
    assert github_handoff.remote_head_probe(tools, request) == {
        "owner": "owner",
        "repo": "repo",
        "sha": "feature",
    }
