# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import shlex
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from core.agent_workspace.common import AgentWorkspaceError, workspace_fingerprint
from core.agent_workspace.background import BackgroundTaskManager
from core.agent_workspace import git_service as git_service_module
from core.agent_workspace import verification as verification_module
from core.agent_workspace.execution import (
    acquire_workspace_execution_slot,
    release_workspace_execution_slot,
)
from core.agent_workspace.git_service import (
    begin_project_deletion,
    confirm_prepared_commit,
    create_checkpoint,
    finish_project_deletion,
    git_diff,
    git_status,
    prepare_commit,
    reconcile_project_checkpoints_for_deletion,
    rollback_checkpoint,
)
from core.agent_workspace.prepared_commit_state import get_preparation
from core.agent_workspace.review import build_pull_request_draft, build_review_summary
from core.agent_workspace.state import (
    get_background_task,
    get_checkpoint,
    set_verification_config,
)
from core.agent_workspace.worktrees import cleanup_worktree, create_worktree, merge_worktree
from storage import studio_db


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd = root,
        check = True,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
    )
    return result.stdout.strip()


def _repository(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "config", "user.name", "Test")
    _git(root, "config", "user.email", "test@example.invalid")
    (root / "owned.txt").write_text("base\n", encoding = "utf-8")
    (root / "unrelated.txt").write_text("base\n", encoding = "utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "base")


def _folder_project(root: Path, project_id: str = "project") -> None:
    metadata = root.stat()
    studio_db.upsert_chat_project(
        {
            "id": project_id,
            "name": "Project",
            "instructions": "",
            "rootPath": str(root),
            "workspaceKind": "folder",
            "workspaceDeviceId": str(metadata.st_dev),
            "workspaceFileId": str(metadata.st_ino),
            "goal": "Keep token=super-secret out of the draft",
            "goalStatus": "active",
            "archived": False,
            "createdAt": 1,
            "updatedAt": 1,
        }
    )


def _executable_marker_command(
    root: Path,
    name: str,
    *,
    passthrough: bool = True,
) -> tuple[str, Path]:
    marker = root / f"{name}.executed"
    script = root / f"{name}.py"
    script.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed\\n', encoding='utf-8')\n"
        + ("sys.stdout.buffer.write(sys.stdin.buffer.read())\n" if passthrough else ""),
        encoding = "utf-8",
    )
    script.chmod(0o700)
    return shlex.quote(str(script)), marker


def _assert_commands_not_executed(markers: list[Path]) -> None:
    assert [marker.name for marker in markers if marker.exists()] == []


def test_git_status_diff_checkpoint_and_owned_rollback(tmp_path):
    _repository(tmp_path)
    _folder_project(tmp_path)
    (tmp_path / "owned.txt").write_text("checkpoint\n", encoding = "utf-8")
    checkpoint = create_checkpoint("project", ["owned.txt"])
    (tmp_path / "owned.txt").write_text("after\n", encoding = "utf-8")
    (tmp_path / "unrelated.txt").write_text("unrelated change\n", encoding = "utf-8")

    status = git_status("project")
    assert status["clean"] is False
    assert status["counts"]["unstaged"] == 2
    assert "after" in git_diff("project")["diff"]

    expected = workspace_fingerprint(tmp_path)
    rollback_checkpoint("project", checkpoint["id"], expected)

    assert (tmp_path / "owned.txt").read_text(encoding = "utf-8") == "checkpoint\n"
    assert (tmp_path / "unrelated.txt").read_text(encoding = "utf-8") == "unrelated change\n"


def test_git_operations_do_not_execute_repository_configured_commands(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    commands = tmp_path / "configured-commands"
    repository.mkdir()
    commands.mkdir()
    _repository(repository)
    _folder_project(repository)
    (repository / ".gitattributes").write_text(
        "*.txt filter=hostile diff=hostile\n", encoding = "utf-8"
    )
    (repository / "binary.bin").write_bytes(b"\x00base-binary\xff")
    _git(repository, "add", ".gitattributes", "binary.bin")
    _git(repository, "commit", "-qm", "add attributes and binary fixture")

    clean_command, clean_marker = _executable_marker_command(commands, "clean-filter")
    smudge_command, smudge_marker = _executable_marker_command(commands, "smudge-filter")
    process_command, process_marker = _executable_marker_command(
        commands, "process-filter", passthrough = False
    )
    fsmonitor_command, fsmonitor_marker = _executable_marker_command(
        commands, "fsmonitor", passthrough = False
    )
    diff_command, diff_marker = _executable_marker_command(
        commands, "external-diff", passthrough = False
    )
    generic_command, generic_marker = _executable_marker_command(
        commands, "generic-config-command", passthrough = False
    )
    _hook_command, hook_marker = _executable_marker_command(commands, "git-hook", passthrough = False)
    markers = [
        clean_marker,
        smudge_marker,
        process_marker,
        fsmonitor_marker,
        diff_marker,
        generic_marker,
        hook_marker,
    ]
    _git(repository, "config", "filter.hostile.clean", clean_command)
    _git(repository, "config", "filter.hostile.smudge", smudge_command)
    _git(repository, "config", "filter.hostile.process", process_command)
    _git(repository, "config", "filter.hostile.required", "true")
    _git(repository, "config", "core.fsmonitor", fsmonitor_command)
    _git(repository, "config", "diff.external", diff_command)
    _git(repository, "config", "diff.hostile.command", diff_command)
    _git(repository, "config", "diff.hostile.textconv", diff_command)
    _git(repository, "config", "core.pager", generic_command)
    _git(repository, "config", "core.editor", generic_command)
    _git(repository, "config", "sequence.editor", generic_command)
    _git(repository, "config", "credential.helper", f"!{generic_command}")
    _git(repository, "config", "alias.status", f"!{generic_command}")
    hooks = commands / "hooks"
    hooks.mkdir()
    for hook_name in ("post-index-change", "reference-transaction"):
        hook = hooks / hook_name
        hook.write_text(
            (commands / "git-hook.py").read_text(encoding = "utf-8"),
            encoding = "utf-8",
        )
        hook.chmod(0o700)
    _git(repository, "config", "core.hooksPath", str(hooks))

    global_config = commands / "hostile-global-config"
    global_config.write_text(
        "[core]\n"
        f"\tfsmonitor = {fsmonitor_command}\n"
        f"\tpager = {generic_command}\n"
        "[diff]\n"
        f"\texternal = {diff_command}\n"
        "[credential]\n"
        f"\thelper = !{generic_command}\n",
        encoding = "utf-8",
    )
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))

    (repository / "owned.txt").write_text("checkpoint bytes\n", encoding = "utf-8")
    binary_checkpoint = b"\x00checkpoint-binary-payload\xff"
    (repository / "binary.bin").write_bytes(binary_checkpoint)

    status = git_status("project")
    assert status["clean"] is False
    _assert_commands_not_executed(markers)

    diff = git_diff("project")["diff"]
    assert "checkpoint bytes" in diff
    assert "checkpoint-binary-payload" not in diff
    assert "Binary files" in diff
    _assert_commands_not_executed(markers)

    checkpoint = create_checkpoint("project", ["owned.txt", "binary.bin"])
    _assert_commands_not_executed(markers)

    (repository / "owned.txt").write_text("after checkpoint\n", encoding = "utf-8")
    (repository / "binary.bin").write_bytes(b"\x00after-checkpoint-binary\xff")
    expected = git_service_module.workspace_fingerprint(repository)
    rollback_checkpoint("project", checkpoint["id"], expected)

    assert (repository / "owned.txt").read_text(encoding = "utf-8") == ("checkpoint bytes\n")
    assert (repository / "binary.bin").read_bytes() == binary_checkpoint
    _assert_commands_not_executed(markers)


def test_verification_fingerprint_neutralizes_repository_executable_config(tmp_path):
    repository = tmp_path / "repo"
    commands = tmp_path / "configured-commands"
    repository.mkdir()
    commands.mkdir()
    _repository(repository)
    (repository / ".gitattributes").write_text(
        "*.txt filter=hostile-process\n*.md filter=hostile-clean\n",
        encoding = "utf-8",
    )
    (repository / "process.txt").write_text("base\n", encoding = "utf-8")
    (repository / "clean.md").write_text("base\n", encoding = "utf-8")
    _git(repository, "add", ".gitattributes", "process.txt", "clean.md")
    _git(repository, "commit", "-qm", "add hostile filter fixtures")

    fsmonitor_command, fsmonitor_marker = _executable_marker_command(
        commands, "verification-fsmonitor", passthrough = False
    )
    process_command, process_marker = _executable_marker_command(
        commands, "verification-process", passthrough = False
    )
    clean_command, clean_marker = _executable_marker_command(commands, "verification-clean")
    _git(repository, "config", "core.fsmonitor", fsmonitor_command)
    _git(repository, "config", "filter.hostile-process.process", process_command)
    _git(repository, "config", "filter.hostile-process.required", "true")
    _git(repository, "config", "filter.hostile-clean.clean", clean_command)
    _git(repository, "config", "filter.hostile-clean.required", "true")
    (repository / "process.txt").write_text("changed\n", encoding = "utf-8")
    (repository / "clean.md").write_text("changed\n", encoding = "utf-8")

    fingerprint = verification_module.workspace_fingerprint(repository)

    assert len(fingerprint) == 64
    _assert_commands_not_executed([fsmonitor_marker, process_marker, clean_marker])


def test_worktree_checkout_does_not_execute_repository_content_filters(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    commands = tmp_path / "configured-commands"
    repository.mkdir()
    commands.mkdir()
    _repository(repository)
    _folder_project(repository)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-projects"))
    (repository / ".gitattributes").write_text("*.txt filter=hostile\n", encoding = "utf-8")
    (repository / "filtered.txt").write_text("tracked bytes\n", encoding = "utf-8")
    _git(repository, "add", ".gitattributes", "filtered.txt")
    _git(repository, "commit", "-qm", "add filtered checkout fixture")

    clean_command, clean_marker = _executable_marker_command(commands, "worktree-clean")
    smudge_command, smudge_marker = _executable_marker_command(commands, "worktree-smudge")
    process_command, process_marker = _executable_marker_command(
        commands, "worktree-process", passthrough = False
    )
    _git(repository, "config", "filter.hostile.clean", clean_command)
    _git(repository, "config", "filter.hostile.smudge", smudge_command)
    _git(repository, "config", "filter.hostile.process", process_command)
    _git(repository, "config", "filter.hostile.required", "true")

    worktree = create_worktree("project")

    assert (Path(worktree["path"]) / "filtered.txt").read_text(
        encoding = "utf-8"
    ) == "tracked bytes\n"
    _assert_commands_not_executed([clean_marker, smudge_marker, process_marker])
    cleanup_worktree("project", worktree["id"])
    _assert_commands_not_executed([clean_marker, smudge_marker, process_marker])


def test_worktree_merge_replaces_repository_custom_merge_driver(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    commands = tmp_path / "configured-commands"
    repository.mkdir()
    commands.mkdir()
    _repository(repository)
    _folder_project(repository)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-projects"))
    (repository / ".gitattributes").write_text("merge.txt merge=hostile\n", encoding = "utf-8")
    (repository / "merge.txt").write_text("first\nsecond\nthird\n", encoding = "utf-8")
    _git(repository, "add", ".gitattributes", "merge.txt")
    _git(repository, "commit", "-qm", "add merge driver fixture")
    worktree = create_worktree("project")
    worktree_path = Path(worktree["path"])

    driver_command, driver_marker = _executable_marker_command(
        commands, "custom-merge-driver", passthrough = False
    )
    _git(repository, "config", "merge.hostile.driver", driver_command)
    (worktree_path / "merge.txt").write_text("agent\nsecond\nthird\n", encoding = "utf-8")
    _git(worktree_path, "add", "merge.txt")
    _git(worktree_path, "commit", "-qm", "agent merge change")
    (repository / "merge.txt").write_text("first\nsecond\nprimary\n", encoding = "utf-8")
    _git(repository, "add", "merge.txt")
    _git(repository, "commit", "-qm", "primary merge change")
    expected_head = _git(repository, "rev-parse", "HEAD")

    merged = merge_worktree("project", worktree["id"], expected_head)

    assert merged["merge"]["status"] == "merged"
    assert (repository / "merge.txt").read_text(encoding = "utf-8") == ("agent\nsecond\nprimary\n")
    assert not driver_marker.exists()
    cleanup_worktree("project", worktree["id"])


def test_checkpoint_captures_only_owned_changes_and_preserves_user_state(tmp_path):
    _repository(tmp_path)
    _folder_project(tmp_path)
    (tmp_path / "owned.txt").write_text("checkpoint\n", encoding = "utf-8")
    (tmp_path / "unrelated.txt").write_text("staged unrelated\n", encoding = "utf-8")
    _git(tmp_path, "add", "unrelated.txt")
    (tmp_path / "unrelated.txt").write_text("unstaged unrelated\n", encoding = "utf-8")
    (tmp_path / "untracked-secret.txt").write_text(
        "token=must-not-enter-checkpoint\n", encoding = "utf-8"
    )
    index_before = _git(tmp_path, "diff", "--cached", "--binary")
    status_before = _git(tmp_path, "status", "--porcelain=v1", "--untracked-files=all")

    checkpoint = create_checkpoint("project", ["owned.txt"])

    assert _git(tmp_path, "diff", "--cached", "--binary") == index_before
    assert _git(tmp_path, "status", "--porcelain=v1", "--untracked-files=all") == status_before
    assert (tmp_path / "unrelated.txt").read_text(encoding = "utf-8") == "unstaged unrelated\n"
    assert (tmp_path / "untracked-secret.txt").read_text(encoding = "utf-8") == (
        "token=must-not-enter-checkpoint\n"
    )
    assert _git(tmp_path, "show", f"{checkpoint['commitSha']}:owned.txt") == "checkpoint"
    assert _git(tmp_path, "show", f"{checkpoint['commitSha']}:unrelated.txt") == "base"
    assert (
        _git(
            tmp_path,
            "diff-tree",
            "--no-commit-id",
            "--name-only",
            "-r",
            checkpoint["commitSha"],
        )
        == "owned.txt"
    )
    missing_secret = subprocess.run(
        ["git", "cat-file", "-e", f"{checkpoint['commitSha']}:untracked-secret.txt"],
        cwd = tmp_path,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        check = False,
    )
    assert missing_secret.returncode != 0


def test_prepared_commit_confirmation_preserves_head_index_and_worktree(tmp_path):
    _repository(tmp_path)
    _folder_project(tmp_path)
    (tmp_path / "owned.txt").write_text("prepared owned\n", encoding = "utf-8")
    (tmp_path / "new.txt").write_text("prepared new\n", encoding = "utf-8")
    (tmp_path / "unrelated.txt").write_text("staged unrelated\n", encoding = "utf-8")
    _git(tmp_path, "add", "new.txt", "unrelated.txt")
    head_before = _git(tmp_path, "rev-parse", "HEAD")
    index_before = (tmp_path / ".git" / "index").read_bytes()
    status_before = _git(tmp_path, "status", "--porcelain=v1", "--untracked-files=all")

    prepared = prepare_commit("project", ["owned.txt", "new.txt"], "Prepare selected files")

    assert prepared["status"] == "awaiting_confirmation"
    assert prepared["baseHead"] == head_before
    assert prepared["branch"] in {"main", "master"}
    assert {item["path"] for item in prepared["files"]} == {"new.txt", "owned.txt"}
    assert "prepared owned" in prepared["diff"]
    assert "prepared new" in prepared["diff"]
    assert _git(tmp_path, "rev-parse", "HEAD") == head_before
    assert (tmp_path / ".git" / "index").read_bytes() == index_before
    assert _git(tmp_path, "status", "--porcelain=v1", "--untracked-files=all") == (status_before)
    missing_ref = subprocess.run(
        ["git", "show-ref", "--verify", prepared["refName"]],
        cwd = tmp_path,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        check = False,
    )
    assert missing_ref.returncode != 0

    with pytest.raises(AgentWorkspaceError, match = "token is invalid"):
        confirm_prepared_commit("project", prepared["id"], "x" * 43)

    confirmed = confirm_prepared_commit("project", prepared["id"], prepared["confirmationToken"])

    assert confirmed["status"] == "confirmed"
    assert confirmed["refName"] == prepared["refName"]
    assert _git(tmp_path, "rev-parse", "HEAD") == head_before
    assert _git(tmp_path, "rev-parse", f"{confirmed['commitSha']}^") == head_before
    assert _git(tmp_path, "show", f"{confirmed['commitSha']}:owned.txt") == ("prepared owned")
    assert _git(tmp_path, "show", f"{confirmed['commitSha']}:new.txt") == ("prepared new")
    assert _git(tmp_path, "show", f"{confirmed['commitSha']}:unrelated.txt") == "base"
    assert _git(tmp_path, "show", "-s", "--format=%B", confirmed["commitSha"]) == (
        "Prepare selected files"
    )
    assert (tmp_path / ".git" / "index").read_bytes() == index_before
    assert _git(tmp_path, "status", "--porcelain=v1", "--untracked-files=all") == (status_before)
    assert (tmp_path / "owned.txt").read_text(encoding = "utf-8") == "prepared owned\n"
    assert (tmp_path / "new.txt").read_text(encoding = "utf-8") == "prepared new\n"
    persisted = get_preparation(prepared["id"])
    assert persisted["status"] == "confirmed"
    assert persisted["tokenDigest"] is None
    with pytest.raises(AgentWorkspaceError, match = "already used"):
        confirm_prepared_commit("project", prepared["id"], prepared["confirmationToken"])


def test_prepared_commit_confirmation_rejects_stale_and_expired_state(tmp_path, monkeypatch):
    _repository(tmp_path)
    _folder_project(tmp_path)
    (tmp_path / "owned.txt").write_text("prepared\n", encoding = "utf-8")
    prepared = prepare_commit("project", ["owned.txt"], "Prepare stale fixture")
    (tmp_path / "owned.txt").write_text("changed after preview\n", encoding = "utf-8")

    with pytest.raises(AgentWorkspaceError, match = "changed after commit preparation"):
        confirm_prepared_commit("project", prepared["id"], prepared["confirmationToken"])
    assert get_preparation(prepared["id"])["status"] == "failed"
    with pytest.raises(AgentWorkspaceError, match = "already used"):
        confirm_prepared_commit("project", prepared["id"], prepared["confirmationToken"])

    (tmp_path / "owned.txt").write_text("expiry fixture\n", encoding = "utf-8")
    monkeypatch.setattr(git_service_module, "now_ms", lambda: 1_000)
    expiring = prepare_commit("project", ["owned.txt"], "Prepare expiry fixture")
    monkeypatch.setattr(
        git_service_module,
        "now_ms",
        lambda: 1_000 + git_service_module._PREPARED_COMMIT_TTL_MS + 1,
    )
    with pytest.raises(AgentWorkspaceError, match = "expired"):
        confirm_prepared_commit("project", expiring["id"], expiring["confirmationToken"])
    assert get_preparation(expiring["id"])["status"] == "expired"


def test_prepared_commit_rejects_detached_head_and_no_selected_changes(tmp_path):
    _repository(tmp_path)
    _folder_project(tmp_path)

    with pytest.raises(AgentWorkspaceError, match = "no changes"):
        prepare_commit("project", ["owned.txt"], "No changes")

    (tmp_path / "owned.txt").write_text("prepared\n", encoding = "utf-8")
    _git(tmp_path, "checkout", "--detach", "-q")
    with pytest.raises(AgentWorkspaceError, match = "attached local branch"):
        prepare_commit("project", ["owned.txt"], "Detached")


def test_prepared_commit_rejects_untracked_content_until_it_is_reviewable(tmp_path):
    _repository(tmp_path)
    _folder_project(tmp_path)
    unseen = "must appear in the exact preview before confirmation"
    (tmp_path / "new.txt").write_text(unseen + "\n", encoding = "utf-8")

    with pytest.raises(AgentWorkspaceError, match = "Untracked files must be added"):
        prepare_commit("project", ["new.txt"], "Reject unseen content")

    _git(tmp_path, "add", "new.txt")
    prepared = prepare_commit("project", ["new.txt"], "Review staged content")
    assert unseen in prepared["diff"]
    confirmed = confirm_prepared_commit("project", prepared["id"], prepared["confirmationToken"])
    assert _git(tmp_path, "show", f"{confirmed['commitSha']}:new.txt") == unseen


def test_checkpoint_and_rollback_treat_pathspec_metacharacters_literally(tmp_path):
    _repository(tmp_path)
    _folder_project(tmp_path)
    literal = tmp_path / "owned[1].txt"
    pattern_match = tmp_path / "owned1.txt"
    literal.write_text("literal checkpoint\n", encoding = "utf-8")
    pattern_match.write_text("must stay untracked\n", encoding = "utf-8")

    checkpoint = create_checkpoint("project", [literal.name])

    assert _git(tmp_path, "show", f"{checkpoint['commitSha']}:{literal.name}") == (
        "literal checkpoint"
    )
    missing_pattern_match = subprocess.run(
        ["git", "cat-file", "-e", f"{checkpoint['commitSha']}:{pattern_match.name}"],
        cwd = tmp_path,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        check = False,
    )
    assert missing_pattern_match.returncode != 0
    literal.write_text("literal after\n", encoding = "utf-8")
    pattern_match.write_text("pattern after\n", encoding = "utf-8")

    rollback_checkpoint("project", checkpoint["id"], workspace_fingerprint(tmp_path))

    assert literal.read_text(encoding = "utf-8") == "literal checkpoint\n"
    assert pattern_match.read_text(encoding = "utf-8") == "pattern after\n"


@pytest.mark.parametrize("owned_path", [".", "/absolute.txt", "C:/absolute.txt", ".git/config"])
def test_checkpoint_rejects_repository_wide_metadata_and_absolute_paths(tmp_path, owned_path):
    _repository(tmp_path)
    _folder_project(tmp_path)

    with pytest.raises(AgentWorkspaceError, match = "inside the repository"):
        create_checkpoint("project", [owned_path])


def test_project_deletion_reconciles_only_recorded_checkpoint_refs(tmp_path):
    _repository(tmp_path)
    _folder_project(tmp_path)
    (tmp_path / "owned.txt").write_text("checkpoint\n", encoding = "utf-8")
    checkpoint = create_checkpoint("project", ["owned.txt"])
    foreign_ref = "refs/unsloth-studio/checkpoints/foreign-user-ref"
    _git(tmp_path, "update-ref", foreign_ref, "HEAD")

    begin_project_deletion("project")
    try:
        with pytest.raises(AgentWorkspaceError, match = "being deleted"):
            create_checkpoint("project", ["owned.txt"])
        result = reconcile_project_checkpoints_for_deletion("project")
    finally:
        finish_project_deletion("project")

    assert result == {"projectId": "project", "removed": 1, "alreadyMissing": 0}
    assert get_checkpoint(checkpoint["id"]) is None
    checkpoint_ref = subprocess.run(
        ["git", "show-ref", "--verify", checkpoint["refName"]],
        cwd = tmp_path,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        check = False,
    )
    assert checkpoint_ref.returncode != 0
    assert _git(tmp_path, "show-ref", "--verify", "--hash", foreign_ref) == _git(
        tmp_path, "rev-parse", "HEAD"
    )


def test_project_deletion_refuses_a_checkpoint_ref_changed_outside_studio(tmp_path):
    _repository(tmp_path)
    _folder_project(tmp_path)
    (tmp_path / "owned.txt").write_text("checkpoint\n", encoding = "utf-8")
    checkpoint = create_checkpoint("project", ["owned.txt"])
    head = _git(tmp_path, "rev-parse", "HEAD")
    assert checkpoint["commitSha"] != head
    _git(tmp_path, "update-ref", checkpoint["refName"], head)

    begin_project_deletion("project")
    try:
        with pytest.raises(AgentWorkspaceError, match = "changed after creation"):
            reconcile_project_checkpoints_for_deletion("project")
    finally:
        finish_project_deletion("project")

    assert get_checkpoint(checkpoint["id"]) is not None
    assert _git(tmp_path, "show-ref", "--verify", "--hash", checkpoint["refName"]) == head


def test_project_deletion_reconciles_confirmed_prepared_commit_ref(tmp_path):
    _repository(tmp_path)
    _folder_project(tmp_path)
    (tmp_path / "owned.txt").write_text("prepared\n", encoding = "utf-8")
    prepared = prepare_commit("project", ["owned.txt"], "Prepared for review")
    confirmed = confirm_prepared_commit("project", prepared["id"], prepared["confirmationToken"])

    begin_project_deletion("project")
    try:
        result = reconcile_project_checkpoints_for_deletion("project")
    finally:
        finish_project_deletion("project")

    assert result == {"projectId": "project", "removed": 1, "alreadyMissing": 0}
    assert get_preparation(prepared["id"]) is None
    missing_ref = subprocess.run(
        ["git", "show-ref", "--verify", confirmed["refName"]],
        cwd = tmp_path,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        check = False,
    )
    assert missing_ref.returncode != 0


def test_checkpoint_and_rollback_are_serialized_per_repository(tmp_path, monkeypatch):
    _repository(tmp_path)
    _folder_project(tmp_path)
    (tmp_path / "owned.txt").write_text("first checkpoint\n", encoding = "utf-8")
    first = create_checkpoint("project", ["owned.txt"])
    (tmp_path / "owned.txt").write_text("active worktree\n", encoding = "utf-8")
    expected = "c0dec0de" + "0" * 56
    monkeypatch.setattr(
        git_service_module,
        "workspace_fingerprint",
        lambda _root: expected,
    )

    create_inside_lock = threading.Event()
    release_create = threading.Event()
    rollback_at_lock_boundary = threading.Event()
    rollback_restore_called = threading.Event()
    original_git = git_service_module._git
    original_project_git = git_service_module._project_git

    def observed_project_git(project_id, *, mutation = False):
        result = original_project_git(project_id, mutation = mutation)
        if threading.current_thread().name == "rollback-worker":
            rollback_at_lock_boundary.set()
        return result

    def controlled_git(root, args, **kwargs):
        worker = threading.current_thread().name
        if worker == "checkpoint-worker" and args[0] == "read-tree":
            create_inside_lock.set()
            if not release_create.wait(timeout = 5):
                raise AssertionError("checkpoint test did not release the mutation lock")
        if worker == "rollback-worker" and args[0] == "restore":
            rollback_restore_called.set()
        return original_git(root, args, **kwargs)

    monkeypatch.setattr(git_service_module, "_project_git", observed_project_git)
    monkeypatch.setattr(git_service_module, "_git", controlled_git)
    failures = []

    def checkpoint_worker():
        try:
            create_checkpoint("project", ["owned.txt"])
        except Exception as exc:
            failures.append(exc)

    def rollback_worker():
        try:
            rollback_checkpoint("project", first["id"], expected)
        except Exception as exc:
            failures.append(exc)

    checkpoint_thread = threading.Thread(target = checkpoint_worker, name = "checkpoint-worker")
    rollback_thread = threading.Thread(target = rollback_worker, name = "rollback-worker")
    checkpoint_thread.start()
    assert create_inside_lock.wait(timeout = 5)
    rollback_thread.start()
    assert rollback_at_lock_boundary.wait(timeout = 5)
    assert not rollback_restore_called.wait(timeout = 0.25)

    release_create.set()
    checkpoint_thread.join(timeout = 10)
    rollback_thread.join(timeout = 10)

    assert not checkpoint_thread.is_alive()
    assert not rollback_thread.is_alive()
    assert failures == []
    assert rollback_restore_called.is_set()
    assert (tmp_path / "owned.txt").read_text(encoding = "utf-8") == "first checkpoint\n"


def test_checkpoint_waits_for_managed_workspace_writer_slot(tmp_path, monkeypatch):
    _repository(tmp_path)
    _folder_project(tmp_path)
    (tmp_path / "owned.txt").write_text("checkpoint\n", encoding = "utf-8")
    metadata = tmp_path.stat()
    identity = (int(metadata.st_dev), int(metadata.st_ino))
    attempted = threading.Event()
    original_acquire = git_service_module.acquire_workspace_execution_slot

    assert acquire_workspace_execution_slot(identity)

    def observed_acquire(requested, cancel_event = None):
        if requested == identity:
            attempted.set()
        return original_acquire(requested, cancel_event)

    monkeypatch.setattr(git_service_module, "acquire_workspace_execution_slot", observed_acquire)
    with ThreadPoolExecutor(max_workers = 1) as executor:
        future = executor.submit(create_checkpoint, "project", ["owned.txt"])
        try:
            assert attempted.wait(timeout = 2)
            assert not future.done()
        finally:
            release_workspace_execution_slot(identity)
        checkpoint = future.result(timeout = 10)

    assert checkpoint["projectId"] == "project"


def test_checkpoint_rejects_stale_confirmation(tmp_path):
    _repository(tmp_path)
    _folder_project(tmp_path)
    checkpoint = create_checkpoint("project", ["owned.txt"])

    with pytest.raises(AgentWorkspaceError, match = "changed"):
        rollback_checkpoint("project", checkpoint["id"], "0" * 64)


def test_checkpoint_rollback_rejects_incomplete_bounded_evidence(tmp_path, monkeypatch):
    _repository(tmp_path)
    _folder_project(tmp_path)
    (tmp_path / "owned.txt").write_text("checkpoint\n", encoding = "utf-8")
    checkpoint = create_checkpoint("project", ["owned.txt"])
    (tmp_path / "owned.txt").write_text("active work\n", encoding = "utf-8")

    budget = 64
    monkeypatch.setattr(
        git_service_module.workspace_common,
        "_GIT_DIRTY_CONTENT_BUDGET",
        budget,
    )
    (tmp_path / "a-padding.bin").write_bytes(b"x" * budget)
    outside_bound = tmp_path / "z-unrelated.txt"
    outside_bound.write_text("outside-one\n", encoding = "utf-8")
    prepared = git_service_module.workspace_fingerprint(tmp_path)
    assert not git_service_module.workspace_common.workspace_fingerprint_complete(prepared)

    # This same-size edit is beyond the content budget. It proves why an
    # incomplete digest cannot authorize a destructive restore: the digest is
    # unchanged even though unrelated repository content changed.
    outside_bound.write_text("outside-two\n", encoding = "utf-8")
    assert git_service_module.workspace_fingerprint(tmp_path) == prepared

    restore_calls = []
    original_git = git_service_module._git

    def observed_git(root, args, **kwargs):
        if args and args[0] == "restore":
            restore_calls.append(list(args))
        return original_git(root, args, **kwargs)

    monkeypatch.setattr(git_service_module, "_git", observed_git)
    with pytest.raises(AgentWorkspaceError, match = "evidence is incomplete"):
        rollback_checkpoint("project", checkpoint["id"], prepared)

    assert restore_calls == []
    assert (tmp_path / "owned.txt").read_text(encoding = "utf-8") == "active work\n"
    assert outside_bound.read_text(encoding = "utf-8") == "outside-two\n"


def test_git_fingerprint_changes_when_dirty_contents_change(tmp_path):
    _repository(tmp_path)

    (tmp_path / "owned.txt").write_text("dirty-one\n", encoding = "utf-8")
    tracked_before = workspace_fingerprint(tmp_path)
    (tmp_path / "owned.txt").write_text("dirty-two\n", encoding = "utf-8")
    assert workspace_fingerprint(tmp_path) != tracked_before

    (tmp_path / "untracked.txt").write_text("same-one\n", encoding = "utf-8")
    untracked_before = workspace_fingerprint(tmp_path)
    (tmp_path / "untracked.txt").write_text("same-two\n", encoding = "utf-8")
    assert workspace_fingerprint(tmp_path) != untracked_before


def test_git_status_parses_rename_as_one_record(tmp_path):
    _repository(tmp_path)
    _folder_project(tmp_path)
    _git(tmp_path, "mv", "owned.txt", "renamed.txt")

    status = git_status("project")

    assert status["counts"]["staged"] == 1
    assert status["files"] == [{"code": "R ", "path": "renamed.txt", "oldPath": "owned.txt"}]


def test_studio_owned_worktree_lifecycle_and_marker_proof(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    repository.mkdir()
    _repository(repository)
    _folder_project(repository)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-projects"))

    record = create_worktree("project")
    worktree_path = Path(record["path"])
    marker_path = Path(record["markerPath"])
    assert worktree_path.is_dir()
    assert marker_path.is_file()

    marker = json.loads(marker_path.read_text(encoding = "utf-8"))
    original_token = marker["token"]
    marker["token"] = "forged"
    marker_path.write_text(json.dumps(marker), encoding = "utf-8")
    with pytest.raises(AgentWorkspaceError, match = "ownership proof"):
        cleanup_worktree("project", record["id"])
    assert worktree_path.exists()

    marker["token"] = original_token
    marker_path.write_text(json.dumps(marker), encoding = "utf-8")
    cleaned = cleanup_worktree("project", record["id"])
    assert cleaned["status"] == "removed"
    assert not worktree_path.exists()


def test_worktree_hashes_client_project_id_and_rejects_option_base_ref(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    repository.mkdir()
    _repository(repository)
    project_id = "../../client-controlled"
    _folder_project(repository, project_id)
    storage = tmp_path / "studio-projects"
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(storage))

    with pytest.raises(AgentWorkspaceError, match = "base reference"):
        create_worktree(project_id, base_ref = "--detach")

    record = create_worktree(project_id)
    worktree_path = Path(record["path"])
    assert worktree_path.is_relative_to((storage / ".agent-worktrees").resolve())
    assert "client-controlled" not in str(worktree_path)
    cleanup_worktree(project_id, record["id"])


def test_local_pull_request_draft_redacts_secrets_and_paths(tmp_path):
    _repository(tmp_path)
    _folder_project(tmp_path)
    (tmp_path / "owned.txt").write_text("changed\n", encoding = "utf-8")

    draft = build_pull_request_draft("project", body_note = f"path={tmp_path} password=hunter2")

    assert draft["localOnly"] is True
    assert draft["submitted"] is False
    assert "super-secret" not in draft["body"]
    assert "hunter2" not in draft["body"]
    assert str(tmp_path) not in draft["body"]
    review = build_review_summary("project")
    assert "repositoryRoot" not in review["git"]
    assert "projectPrefix" not in review["git"]
    assert str(tmp_path) not in json.dumps(review["git"])


def test_review_summary_redacts_git_error_paths(tmp_path, monkeypatch):
    _repository(tmp_path)
    _folder_project(tmp_path)

    def fail_status(_project_id):
        raise AgentWorkspaceError(f"Git failed under {tmp_path}/.git/worktrees/private")

    monkeypatch.setattr("core.agent_workspace.review.git_status", fail_status)

    review = build_review_summary("project")

    assert str(tmp_path) not in review["gitError"]
    assert "<project_root>" in review["gitError"]


def test_review_redaction_masks_token_shapes_bearer_values_and_local_paths(tmp_path):
    _repository(tmp_path)
    _folder_project(tmp_path)
    github_token = "ghp_" + "a" * 36
    bearer_token = "opaque-review-credential-123"
    body_note = "\n".join(
        (
            f"GitHub: {github_token}",
            f"Authorization: Bearer {bearer_token}",
            "SSH file: /Users/alice/.ssh/id_ed25519",
            r"Windows file: C:\Users\alice\.ssh\id_rsa",
            "Relative file: .ssh/id_rsa",
            "Environment file: config/.env.local",
            "Credential file: config/credentials.json",
            "Keep this URL: https://example.invalid/docs/setup",
        )
    )

    draft = build_pull_request_draft("project", body_note = body_note)
    rendered = f"{draft['title']}\n{draft['body']}"

    assert github_token not in rendered
    assert bearer_token not in rendered
    assert "/Users/alice/.ssh/id_ed25519" not in rendered
    assert r"C:\Users\alice\.ssh\id_rsa" not in rendered
    assert ".ssh/id_rsa" not in rendered
    assert ".env.local" not in rendered
    assert "credentials.json" not in rendered
    assert "https://example.invalid/docs/setup" in rendered
    assert "<redacted>" in rendered
    assert "<local_path>" in rendered
    assert "<sensitive_path>" in rendered


def test_background_verification_runs_in_owned_worktree(
    tmp_path, monkeypatch, local_verification_execution_boundary
):
    repository = tmp_path / "repo"
    repository.mkdir()
    _repository(repository)
    _folder_project(repository)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-projects"))
    worktree = create_worktree("project")
    command = (
        f"{shlex.quote(str(Path(sys.executable).resolve()))} -c "
        f"{shlex.quote('import os; print(os.getcwd())')}"
    )
    set_verification_config(
        "project",
        [
            {
                "name": "cwd",
                "kind": "custom",
                "command": command,
                "required": True,
                "timeoutSeconds": 10,
                "logLimitBytes": 4096,
            }
        ],
    )
    manager = BackgroundTaskManager(max_workers = 1)
    task = manager.enqueue_verification("project", worktree_id = worktree["id"], start = True)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        current = get_background_task(task["id"])
        if current["status"] in {"completed", "failed", "cancelled"}:
            break
        time.sleep(0.02)

    assert current["status"] == "completed", current
    assert current["payload"]["worktreeId"] == worktree["id"]
    assert current["result"]["worktreeId"] == worktree["id"]
    assert current["result"]["results"][0]["output"].strip() == worktree["path"]
    cleanup_worktree("project", worktree["id"])
