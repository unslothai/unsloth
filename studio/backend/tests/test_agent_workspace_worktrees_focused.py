# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Focused lifecycle coverage for the worktree dependency lane."""

import json
import os
import subprocess
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.authentication import get_current_subject, authenticated_via_api_key
from core.agent_workspace.git_context import AgentWorkspaceError
from core.agent_workspace import checkpoints
from core.agent_workspace import git_service
from core.agent_workspace import github_handoff
from core.agent_workspace import git_state as state
from core.agent_workspace import worktrees
from core.agent_workspace.git_service import workspace_fingerprint
from storage import studio_db
from storage import mcp_servers_db
from routes.project_worktrees import router


pytestmark = pytest.mark.skipif(os.name == "nt", reason = "Git mutations require a POSIX boundary")


def _git(
    root: Path,
    *args: str,
    check: bool = True,
) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd = root,
        check = check,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
    )
    return result.stdout.strip()


def _repository(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.invalid")
    (root / "tracked.txt").write_text("base\n", encoding = "utf-8")
    _git(root, "add", "tracked.txt")
    _git(root, "commit", "-qm", "base")


def _project(root: Path, project_id: str = "project") -> None:
    studio_db.upsert_chat_project(
        {
            "id": project_id,
            "name": "Project",
            "instructions": "",
            "archived": False,
            "createdAt": 1,
            "updatedAt": 1,
            "sandboxPath": str(root),
            "rootPath": str(root),
        }
    )
    conn = studio_db.get_connection()
    try:
        conn.execute("UPDATE chat_projects SET root_path = ? WHERE id = ?", (str(root), project_id))
        conn.commit()
    finally:
        conn.close()


@pytest.fixture(autouse = True)
def managed_workspace_records(monkeypatch, tmp_path):
    # Test repositories live in tmp_path; production resolves Studio-managed paths.
    from core.agent_workspace import git_context

    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "managed-projects"))

    def persisted(project_id):
        value = studio_db.get_chat_project(project_id)
        if value is not None:
            value["sandboxPath"] = value["rootPath"]
        return value

    monkeypatch.setattr(git_context, "get_chat_project", persisted)
    monkeypatch.setattr(git_context, "ensure_chat_project_workspace", persisted)


def _setup(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    repository.mkdir()
    _repository(repository)
    _project(repository)
    return repository


def test_create_records_owned_checkout_and_cleanup_is_non_destructive(tmp_path, monkeypatch):
    repository = _setup(tmp_path)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-projects"))

    created = worktrees.create_worktree("project")
    target = Path(created["path"])
    assert created["status"] == "active"
    assert target.is_dir()
    assert (target / "tracked.txt").exists()
    assert Path(created["markerPath"]).exists()
    assert state.get_worktree(created["id"])["status"] == "active"
    assert worktrees.owned_worktree_path("project", created["id"]) == target.resolve()
    assert f"refs/heads/{created['branch']}" in _git(
        repository, "for-each-ref", "--format=%(refname)"
    )

    (target / "tracked.txt").write_text("agent change\n", encoding = "utf-8")
    with pytest.raises(AgentWorkspaceError, match = "uncommitted"):
        worktrees.cleanup_worktree("project", created["id"])
    assert state.get_worktree(created["id"])["status"] == "active"

    _git(target, "add", "tracked.txt")
    _git(target, "commit", "-qm", "agent change")
    removed = worktrees.cleanup_worktree("project", created["id"])
    assert removed["status"] == "removed"
    assert not target.exists()
    assert state.get_worktree(created["id"])["status"] == "removed"


def test_branch_and_base_ref_validation_happens_before_durable_reservation(tmp_path, monkeypatch):
    _setup(tmp_path)
    with pytest.raises(AgentWorkspaceError, match = "unsloth-studio"):
        worktrees.create_worktree("project", branch = "user-owned")
    with pytest.raises(AgentWorkspaceError, match = "base reference"):
        worktrees.create_worktree("project", base_ref = "../main")
    assert state.list_worktrees("project") == []


@pytest.mark.parametrize("owned_path", [".git/config", ".git", "../outside", "/absolute"])
def test_checkpoint_rejects_repository_metadata_and_escape_paths(tmp_path, owned_path):
    _setup(tmp_path)
    with pytest.raises(AgentWorkspaceError, match = "Checkpoint paths"):
        checkpoints.create_checkpoint("project", [owned_path])


def test_nested_project_git_reads_and_checkpoints_stay_inside_project_scope(tmp_path):
    repository = _setup(tmp_path)
    with pytest.raises(AgentWorkspaceError, match = "Git metadata"):
        git_service._safe_scope_root(repository, repository / ".git")
    nested = repository / "nested"
    nested.mkdir()
    (nested / "tracked.txt").write_text("nested base\n", encoding = "utf-8")
    _git(repository, "add", "nested/tracked.txt")
    _git(repository, "commit", "-qm", "nested base")
    (repository / "tracked.txt").write_text("outside change\n", encoding = "utf-8")
    (nested / "tracked.txt").write_text("nested change\n", encoding = "utf-8")
    raw = git_service.repository_status(repository, scope_root = nested)
    files, counts = git_service._status_records(
        raw,
        repository = repository,
        scope_root = nested,
    )
    assert [entry["path"] for entry in files] == ["tracked.txt"]
    assert counts["unstaged"] == 1
    assert git_service._repository_paths(repository, nested, ["tracked.txt"]) == [
        "nested/tracked.txt"
    ]


def test_tampered_marker_is_retained_for_attention_not_deleted(tmp_path, monkeypatch):
    _setup(tmp_path)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-projects"))
    created = worktrees.create_worktree("project")
    marker = Path(created["markerPath"])
    payload = json.loads(marker.read_text(encoding = "utf-8"))
    payload["token"] = "tampered-token"
    marker.write_text(json.dumps(payload), encoding = "utf-8")

    result = worktrees.reconcile_worktrees_on_startup()
    assert result["attention"] == 1
    assert state.get_worktree(created["id"])["status"] == "needs_attention"
    assert Path(created["path"]).is_dir()
    with pytest.raises(AgentWorkspaceError, match = "cannot prove"):
        worktrees.cleanup_worktree("project", created["id"])


def test_failed_git_creation_settles_only_proven_empty_checkout(tmp_path, monkeypatch):
    _setup(tmp_path)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-projects"))

    def fail(*_args, **_kwargs):
        raise AgentWorkspaceError("injected add failure")

    monkeypatch.setattr(worktrees, "add_worktree", fail)
    with pytest.raises(AgentWorkspaceError, match = "injected add failure"):
        worktrees.create_worktree("project")
    records = state.list_worktrees("project")
    assert len(records) == 1
    assert records[0]["status"] == "removed"


def test_project_deletion_fence_blocks_new_operations(tmp_path, monkeypatch):
    _setup(tmp_path)
    worktrees.begin_project_deletion("project")
    try:
        with pytest.raises(AgentWorkspaceError, match = "being deleted"):
            worktrees.create_worktree("project")
    finally:
        worktrees.finish_project_deletion("project")


def test_merge_requires_clean_target_and_expected_head(tmp_path, monkeypatch):
    repository = _setup(tmp_path)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-projects"))
    created = worktrees.create_worktree("project")
    target = Path(created["path"])
    (target / "branch.txt").write_text("from agent\n", encoding = "utf-8")
    _git(target, "add", "branch.txt")
    _git(target, "commit", "-qm", "agent branch")
    expected = _git(repository, "rev-parse", "HEAD")

    merged = worktrees.merge_owned_worktree("project", created["id"], expected)
    assert merged["merge"]["status"] == "merged"
    assert (repository / "branch.txt").read_text(encoding = "utf-8") == "from agent\n"


def test_merge_rechecks_live_worktree_registration(tmp_path, monkeypatch):
    _setup(tmp_path)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-projects"))
    created = worktrees.create_worktree("project")
    target = Path(created["path"])
    _git(target, "commit", "--allow-empty", "-qm", "agent branch")
    expected = _git(tmp_path / "repository", "rev-parse", "HEAD")
    monkeypatch.setattr(
        worktrees,
        "worktree_entries",
        lambda _repository: {},
    )

    with pytest.raises(AgentWorkspaceError, match = "registration no longer matches"):
        worktrees.merge_owned_worktree("project", created["id"], expected)


def test_merge_conflict_preserves_primary_checkout_for_manual_recovery(tmp_path, monkeypatch):
    repository = _setup(tmp_path)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-projects"))
    created = worktrees.create_worktree("project")
    target = Path(created["path"])
    (target / "tracked.txt").write_text("agent\n", encoding = "utf-8")
    _git(target, "add", "tracked.txt")
    _git(target, "commit", "-qm", "agent conflict")
    (repository / "tracked.txt").write_text("primary\n", encoding = "utf-8")
    _git(repository, "add", "tracked.txt")
    _git(repository, "commit", "-qm", "primary conflict")
    expected = _git(repository, "rev-parse", "HEAD")

    # Force the real merge command down the conflict path after a successful
    # preflight, which models a target change between the two operations.
    monkeypatch.setattr(worktrees, "preflight_merge", lambda *_args, **_kwargs: (0, "", False))
    result = worktrees.merge_owned_worktree("project", created["id"], expected)

    assert result["merge"]["status"] == "conflict"
    assert result["merge"]["primaryWorkspaceChanged"] is True
    assert "UU tracked.txt" in _git(repository, "status", "--porcelain")
    assert "<<<<<<<" in (repository / "tracked.txt").read_text(encoding = "utf-8")


def test_checkpoint_captures_selected_paths_without_touching_index(tmp_path, monkeypatch):
    repository = _setup(tmp_path)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-projects"))
    (repository / "tracked.txt").write_text("checkpoint\n", encoding = "utf-8")
    (repository / "unrelated.txt").write_text("unrelated\n", encoding = "utf-8")
    _git(repository, "add", "unrelated.txt")
    index_before = (repository / ".git" / "index").read_bytes()
    status_before = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")

    record = checkpoints.create_checkpoint("project", ["tracked.txt"])

    assert record["sourceFingerprint"].startswith("c0dec0de")
    assert (repository / ".git" / "index").read_bytes() == index_before
    assert _git(repository, "status", "--porcelain=v1", "--untracked-files=all") == status_before
    assert _git(repository, "show", f"{record['commitSha']}:tracked.txt") == "checkpoint"
    assert (
        subprocess.run(
            ["git", "cat-file", "-e", f"{record['commitSha']}:unrelated.txt"],
            cwd = repository,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            check = False,
        ).returncode
        != 0
    )
    assert _git(repository, "show-ref", "--verify", record["refName"])


def test_checkpoint_rollback_is_fenced_by_current_fingerprint(tmp_path, monkeypatch):
    repository = _setup(tmp_path)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-projects"))
    (repository / "tracked.txt").write_text("checkpoint\n", encoding = "utf-8")
    record = checkpoints.create_checkpoint("project", ["tracked.txt"])
    current = checkpoints.workspace_fingerprint(repository)
    (repository / "tracked.txt").write_text("after\n", encoding = "utf-8")
    with pytest.raises(AgentWorkspaceError, match = "changed"):
        checkpoints.rollback_checkpoint("project", record["id"], current)

    fresh = checkpoints.workspace_fingerprint(repository)
    restored = checkpoints.rollback_checkpoint("project", record["id"], fresh)
    assert (repository / "tracked.txt").read_text(encoding = "utf-8") == "checkpoint\n"
    assert restored["checkpoint"]["id"] == record["id"]


def test_checkpoint_mutations_observe_project_deletion_fence(tmp_path, monkeypatch):
    repository = _setup(tmp_path)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-projects"))
    (repository / "tracked.txt").write_text("checkpoint\n", encoding = "utf-8")
    record = checkpoints.create_checkpoint("project", ["tracked.txt"])

    worktrees.begin_project_deletion("project")
    try:
        with pytest.raises(AgentWorkspaceError, match = "being deleted"):
            checkpoints.rollback_checkpoint(
                "project", record["id"], checkpoints.workspace_fingerprint(repository)
            )
        assert checkpoints.remove_project_checkpoints("project") == 1
    finally:
        worktrees.finish_project_deletion("project")
    assert checkpoints.list_checkpoints("project") == []
    assert _git(repository, "show-ref", "--verify", record["refName"], check = False) == ""


def test_project_checkpoint_cleanup_preserves_a_ref_changed_outside_studio(tmp_path, monkeypatch):
    repository = _setup(tmp_path)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-projects"))
    (repository / "tracked.txt").write_text("checkpoint\n", encoding = "utf-8")
    record = checkpoints.create_checkpoint("project", ["tracked.txt"])
    head = _git(repository, "rev-parse", "HEAD")
    _git(repository, "update-ref", record["refName"], head)

    with pytest.raises(AgentWorkspaceError, match = "changed outside Studio"):
        checkpoints.remove_project_checkpoints("project")
    assert checkpoints.list_checkpoints("project")[0]["id"] == record["id"]
    assert _git(repository, "show-ref", "--verify", "--hash", record["refName"]) == head


def test_checkpoint_ref_delete_uses_expected_old_value(tmp_path, monkeypatch):
    repository = _setup(tmp_path)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-projects"))
    (repository / "tracked.txt").write_text("checkpoint\n", encoding = "utf-8")
    record = checkpoints.create_checkpoint("project", ["tracked.txt"])
    head = _git(repository, "rev-parse", "HEAD")
    _git(repository, "update-ref", record["refName"], head)

    with pytest.raises(AgentWorkspaceError, match = "Git operation failed"):
        git_service.delete_ref(repository, record["refName"], record["commitSha"])
    assert _git(repository, "show-ref", "--verify", "--hash", record["refName"]) == head


def test_github_handoff_is_redacted_bound_and_one_use(tmp_path, monkeypatch):
    repository = _setup(tmp_path)
    _git(repository, "checkout", "-qb", "feature/codex")
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-projects"))
    mcp_servers_db.create_server(
        id = "github",
        display_name = "GitHub",
        url = "https://github.example.invalid/mcp",
        headers_json = '{"Authorization":"Bearer secret"}',
    )
    tools = [
        {
            "name": "create_pull_request",
            "inputSchema": {
                "properties": {
                    "owner": {},
                    "repo": {},
                    "title": {},
                    "body": {},
                    "head": {},
                    "base": {},
                },
                "required": ["owner", "repo", "title", "head", "base"],
            },
        }
    ]
    preview = github_handoff.prepare_pull_request_handoff(
        "project",
        server_id = "github",
        owner = "unslothai",
        repository = "unsloth",
        base = "main",
        head = "feature/codex",
        body_note = f"local={repository} password=hunter2",
        tools = tools,
        draft = False,
        now = 100,
    )
    assert preview["submitted"] is False
    assert str(repository) not in preview["request"]["body"]
    assert "hunter2" not in preview["request"]["body"]
    assert preview["reviewBinding"]["branch"] == "feature/codex"
    assert "draft" not in preview["request"]
    assert "maintainer_can_modify" not in preview["request"]
    server, request = github_handoff.consume_pull_request_handoff(
        "project",
        preview["id"],
        server_id = "github",
        confirmation_token = preview["confirmationToken"],
        expected_request_digest = preview["requestDigest"],
        tools = tools,
        now = 101,
    )
    assert server["id"] == "github"
    assert request == preview["request"]
    with pytest.raises(AgentWorkspaceError, match = "already used"):
        github_handoff.consume_pull_request_handoff(
            "project",
            preview["id"],
            server_id = "github",
            confirmation_token = preview["confirmationToken"],
            expected_request_digest = preview["requestDigest"],
            tools = tools,
            now = 102,
        )


def test_github_handoff_rejects_a_head_different_from_the_reviewed_branch(tmp_path, monkeypatch):
    _setup(tmp_path)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-projects"))
    mcp_servers_db.create_server(
        id = "github",
        display_name = "GitHub",
        url = "https://github.example.invalid/mcp",
        headers_json = '{"Authorization":"Bearer secret"}',
    )
    tools = [
        {
            "name": "create_pull_request",
            "inputSchema": {
                "properties": {
                    key: {} for key in ("owner", "repo", "title", "body", "head", "base")
                },
                "required": ["owner", "repo", "title", "head", "base"],
            },
        }
    ]
    with pytest.raises(AgentWorkspaceError, match = "head must match"):
        github_handoff.prepare_pull_request_handoff(
            "project",
            server_id = "github",
            owner = "unslothai",
            repository = "unsloth",
            base = "main",
            head = "other-branch",
            draft = False,
            tools = tools,
            now = 100,
        )


def test_project_worktree_routes_expose_only_owned_metadata(tmp_path, monkeypatch):
    repository = _setup(tmp_path)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-projects"))
    app = FastAPI()
    app.include_router(router, prefix = "/api/agent")
    app.dependency_overrides[get_current_subject] = lambda: "test-subject"
    app.dependency_overrides[authenticated_via_api_key] = lambda: False
    client = TestClient(app)
    client.params = {"workspaceRevision": "0"}

    assert client.get("/api/agent/projects/project/git/status").status_code == 404
    assert client.get("/api/agent/projects/project/git/diff").status_code == 404

    (repository / "tracked.txt").write_text("password=hunter2\n", encoding = "utf-8")
    checkpoint = client.post(
        "/api/agent/projects/project/git/checkpoints",
        json = {"ownedPaths": ["tracked.txt"]},
    )
    assert checkpoint.status_code == 200
    assert "gitRoot" not in checkpoint.json()
    assert checkpoint.json()["ownedPaths"] == ["tracked.txt"]

    rollback = client.post(
        f"/api/agent/projects/project/git/checkpoints/{checkpoint.json()['id']}/rollback",
        json = {"expectedCurrentFingerprint": workspace_fingerprint(repository)},
    )
    assert rollback.status_code == 200
    assert "gitRoot" not in rollback.json()["checkpoint"]
    assert "refName" not in rollback.json()["checkpoint"]

    review = client.get("/api/agent/projects/project/review")
    assert review.status_code == 200
    assert str(repository) not in review.text
    assert "root" not in review.json()["status"]


def test_git_status_preserves_new_and_original_rename_paths():
    files, counts = git_service._status_records("R  new.txt\0old.txt\0")
    assert files == [{"code": "R ", "path": "new.txt", "oldPath": "old.txt"}]
    assert counts["staged"] == 1


def test_git_processes_fail_closed_on_windows(monkeypatch, tmp_path):
    fake_os = type("WindowsOnlyOS", (), {"name": "nt"})()
    monkeypatch.setattr(git_service, "os", fake_os)
    with pytest.raises(AgentWorkspaceError, match = "Windows"):
        git_service._run(tmp_path, ["status"])


def test_git_process_ignores_caller_path(monkeypatch, tmp_path):
    attacker = tmp_path / "git"
    attacker.write_text("#!/bin/sh\nexit 0\n", encoding = "utf-8")
    attacker.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert git_service._trusted_git_executable() != attacker
