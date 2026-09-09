# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
from pathlib import Path

import pytest

from core.agent_workspace import git_review, prepared_commits, prepared_commit_state, worktrees
from routes import project_worktrees
from .test_project_git_safety import _client
from .test_agent_workspace_worktrees_focused import _setup, _git, managed_workspace_records  # noqa: F401
from .test_project_git_review import _repository, _commit_all, _bind_workspace, _workspace, _file


@pytest.mark.skipif(os.name == "nt", reason = "Native POSIX mutation contract")
def test_prepared_commit_preserves_reviewed_staged_rename(tmp_path):
    root = _setup(tmp_path)
    _git(root, "mv", "tracked.txt", "renamed.txt")
    head = _git(root, "rev-parse", "HEAD")
    index = (root / ".git/index").read_bytes()
    preview = prepared_commits.prepare_commit("project", ["renamed.txt"], "Move file")
    assert preview["ownedPaths"] == ["renamed.txt", "tracked.txt"]
    assert preview["previewFiles"][0]["oldPath"] == "tracked.txt"
    result = prepared_commits.confirm_prepared_commit(
        "project", preview["id"], preview["confirmationToken"]
    )
    assert _git(root, "ls-tree", "--name-only", result["commitSha"]) == "renamed.txt"
    assert _git(root, "show", f"{result['commitSha']}:renamed.txt") == "base"
    assert _git(root, "rev-parse", "HEAD") == head
    assert (root / ".git/index").read_bytes() == index


def test_expired_confirmations_without_candidates_do_not_exhaust_pending_limit(
    tmp_path, monkeypatch
):
    root = _setup(tmp_path)
    monkeypatch.setattr(prepared_commit_state, "_MAX_PENDING_PREPARATIONS", 2)
    token = "x" * 32

    def record(identifier, expiry = 10):
        return {
            "id": identifier,
            "projectId": "project",
            "operation": "prepare_commit",
            "branchRef": "refs/heads/main",
            "headSha": "a" * 40,
            "gitRoot": str(root),
            "message": "review",
            "ownedPaths": ["tracked.txt"],
            "sourceFingerprint": "c0dec0de" + "0" * 56,
            "payloadDigest": "digest",
            "refName": "refs/unsloth-studio/prepared-commits/" + identifier,
            "createdAt": 0,
            "expiresAt": expiry,
        }

    for identifier in ("abandoned", "recoverable"):
        prepared_commit_state.save_preparation(record(identifier), token, now = 0)
        prepared_commit_state.reserve_confirmation(identifier, "project", token, now = 1)
    prepared_commit_state.save_candidate_commit("recoverable", "a" * 40)
    prepared_commit_state.save_preparation(record("next", expiry = 30), token, now = 11)
    assert prepared_commit_state.get_preparation("abandoned") is None
    assert prepared_commit_state.get_preparation("recoverable")["commitSha"] == "a" * 40
    assert prepared_commit_state.get_preparation("next")["status"] == "awaiting_confirmation"


@pytest.mark.skipif(os.name == "nt", reason = "Native POSIX mutation contract")
@pytest.mark.parametrize("retry", ["startup", "explicit"])
def test_removed_worktree_marker_is_reconciled_after_interrupted_cleanup(
    tmp_path, monkeypatch, retry
):
    _setup(tmp_path)
    owned = worktrees.create_worktree("project")
    marker = Path(owned["markerPath"])
    original = Path.unlink
    with monkeypatch.context() as scoped:

        def fail_marker(path, *args, **kwargs):
            if path == marker:
                raise OSError("interrupted marker cleanup")
            return original(path, *args, **kwargs)

        scoped.setattr(Path, "unlink", fail_marker)
        assert worktrees.cleanup_worktree("project", owned["id"])["status"] == "removed"
    assert marker.exists()
    if retry == "startup":
        assert worktrees.reconcile_worktrees_on_startup()["removed"] == 1
    else:
        assert worktrees.cleanup_worktree("project", owned["id"])["status"] == "removed"
    assert not marker.exists()
    assert not marker.parent.exists()


def test_management_returns_the_validated_nonzero_workspace_revision(tmp_path, monkeypatch):
    _setup(tmp_path)
    original = project_worktrees.get_chat_project

    def revisioned(project_id):
        record = original(project_id)
        return record | {"workspaceRevision": 7} if record is not None else None

    monkeypatch.setattr(project_worktrees, "get_chat_project", revisioned)
    with _client() as client:
        response = client.get("/projects/project/git/manage?workspaceRevision=7")
        assert response.status_code == 200
        assert response.json()["workspaceRevision"] == 7
        assert client.get("/projects/project/git/manage?workspaceRevision=0").status_code == 409


def test_crlf_only_diff_keeps_the_carriage_return_visible(tmp_path, monkeypatch):
    repository = _repository(tmp_path / "repository")
    (repository / "line.txt").write_bytes(b"same text\r\n")
    _commit_all(repository)
    (repository / "line.txt").write_bytes(b"same text\n")
    _bind_workspace(monkeypatch, _workspace(repository))
    manifest = git_review.build_diff_manifest("project-one")
    lines = _file(manifest, "line.txt")["hunks"][0]["lines"]
    assert [(line["kind"], line["text"]) for line in lines] == [
        ("delete", "same text\\r"),
        ("add", "same text"),
    ]
