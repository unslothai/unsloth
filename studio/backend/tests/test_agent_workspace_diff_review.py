# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from core.agent_workspace import diff_review as diff_review_module
from core.agent_workspace.diff_review import build_diff_manifest
from core.agent_workspace.worktrees import cleanup_worktree, create_worktree
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
    (root / "review.txt").write_text(
        "".join(f"line {number:02d}\n" for number in range(1, 41)),
        encoding = "utf-8",
    )
    (root / "other.txt").write_text("base\n", encoding = "utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "base")


def _folder_project(root: Path) -> str:
    project_id = f"diff-review-{uuid.uuid4()}"
    metadata = root.stat()
    studio_db.upsert_chat_project(
        {
            "id": project_id,
            "name": "Diff review",
            "instructions": "",
            "rootPath": str(root),
            "workspaceKind": "folder",
            "workspaceDeviceId": str(metadata.st_dev),
            "workspaceFileId": str(metadata.st_ino),
            "goal": "Review selected hunks",
            "goalStatus": "active",
            "archived": False,
            "createdAt": 1,
            "updatedAt": 1,
        }
    )
    return project_id


def _file(manifest: dict, path: str) -> dict:
    return next(item for item in manifest["files"] if item["path"] == path)


def _hunk_ids(manifest: dict, path: str) -> list[str]:
    return [hunk["id"] for hunk in _file(manifest, path)["hunks"]]


def test_manifest_ids_are_stable_and_bound_to_head_and_complete_source(tmp_path):
    _repository(tmp_path)
    project_id = _folder_project(tmp_path)
    lines = (tmp_path / "review.txt").read_text(encoding = "utf-8").splitlines()
    lines[1] = "first worktree change"
    (tmp_path / "review.txt").write_text("\n".join(lines) + "\n", encoding = "utf-8")

    first = build_diff_manifest(project_id)
    repeated = build_diff_manifest(project_id)

    assert first["selectable"] is True
    assert first["head"] == repeated["head"]
    assert first["sourceFingerprint"].startswith("c0dec0de")
    assert first["sourceFingerprint"] == repeated["sourceFingerprint"]
    assert _hunk_ids(first, "review.txt") == _hunk_ids(repeated, "review.txt")
    assert _file(first, "review.txt")["selectionId"] == _file(repeated, "review.txt")["selectionId"]
    assert all(len(value) == 64 for value in _hunk_ids(first, "review.txt"))

    (tmp_path / "other.txt").write_text("dirty unrelated source\n", encoding = "utf-8")
    changed_source = build_diff_manifest(project_id)

    assert changed_source["sourceFingerprint"] != first["sourceFingerprint"]
    assert _hunk_ids(changed_source, "review.txt") != _hunk_ids(first, "review.txt")

    _git(tmp_path, "add", "other.txt")
    _git(tmp_path, "commit", "-qm", "unrelated head change")
    changed_head = build_diff_manifest(project_id)

    assert changed_head["head"] != first["head"]
    assert _hunk_ids(changed_head, "review.txt") != _hunk_ids(changed_source, "review.txt")

    _git(tmp_path, "add", "review.txt")
    lines[30] = "second worktree change"
    (tmp_path / "review.txt").write_text("\n".join(lines) + "\n", encoding = "utf-8")

    staged = build_diff_manifest(project_id, mode = "staged")
    unstaged = build_diff_manifest(project_id, mode = "unstaged")
    head = build_diff_manifest(project_id, mode = "head")

    assert staged["selectable"] is True
    assert unstaged["selectable"] is True
    assert head["selectable"] is True
    assert len(_hunk_ids(staged, "review.txt")) == 1
    assert len(_hunk_ids(unstaged, "review.txt")) == 1
    assert len(_hunk_ids(head, "review.txt")) == 2
    assert head["hunkCount"] == sum(len(item["hunks"]) for item in head["files"])
    assert set(_hunk_ids(staged, "review.txt")).isdisjoint(_hunk_ids(unstaged, "review.txt"))


def test_binary_rename_mode_and_untracked_entries_are_whole_file_only(tmp_path):
    _repository(tmp_path)
    project_id = _folder_project(tmp_path)
    (tmp_path / "binary.bin").write_bytes(b"\x00base-binary\xff")
    (tmp_path / "rename-source.txt").write_text("rename me\n", encoding = "utf-8")
    (tmp_path / "mode.sh").write_text("#!/bin/sh\nexit 0\n", encoding = "utf-8")
    (tmp_path / "mode.sh").chmod(0o644)
    _git(tmp_path, "add", "binary.bin", "rename-source.txt", "mode.sh")
    _git(tmp_path, "commit", "-qm", "whole file fixtures")

    (tmp_path / "binary.bin").write_bytes(b"\x00changed-binary-payload\xff")
    _git(tmp_path, "mv", "rename-source.txt", "renamed.txt")
    (tmp_path / "mode.sh").chmod(0o755)
    (tmp_path / "untracked.txt").write_text("new file\n", encoding = "utf-8")

    manifest = build_diff_manifest(project_id, mode = "head")

    assert manifest["selectable"] is True
    binary = _file(manifest, "binary.bin")
    assert binary["binary"] is True
    assert binary["wholeFileOnly"] is True
    assert binary["hunks"] == []
    rename = _file(manifest, "renamed.txt")
    assert rename["code"].startswith("R")
    assert rename["oldPath"] == "rename-source.txt"
    assert rename["wholeFileOnly"] is True
    assert rename["hunks"] == []
    mode = _file(manifest, "mode.sh")
    if mode["oldMode"] == mode["newMode"]:
        pytest.skip("Git did not report executable-bit changes on this filesystem")
    assert mode["wholeFileOnly"] is True
    assert mode["hunks"] == []
    untracked = _file(manifest, "untracked.txt")
    assert untracked["code"] == "??"
    assert untracked["wholeFileOnly"] is True
    assert untracked["hunks"] == []


def test_manifest_can_target_only_a_proven_studio_owned_worktree(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    repository.mkdir()
    _repository(repository)
    project_id = _folder_project(repository)
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "studio-projects"))
    record = create_worktree(project_id)
    worktree = Path(record["path"])
    lines = (worktree / "review.txt").read_text(encoding = "utf-8").splitlines()
    lines[5] = "owned worktree change"
    (worktree / "review.txt").write_text("\n".join(lines) + "\n", encoding = "utf-8")

    manifest = build_diff_manifest(project_id, worktree_id = record["id"])
    primary = build_diff_manifest(project_id)

    assert manifest["selectable"] is True
    assert manifest["target"] == {"kind": "worktree", "worktreeId": record["id"]}
    assert _hunk_ids(manifest, "review.txt")
    assert primary["files"] == []

    _git(worktree, "add", "review.txt")
    _git(worktree, "commit", "-qm", "finish worktree fixture")
    cleanup_worktree(project_id, record["id"])


def test_conflicts_are_reported_and_block_hunk_selection(tmp_path):
    _repository(tmp_path)
    project_id = _folder_project(tmp_path)
    branch = _git(tmp_path, "branch", "--show-current")
    _git(tmp_path, "switch", "-qc", "conflicting-branch")
    (tmp_path / "review.txt").write_text("branch version\n", encoding = "utf-8")
    _git(tmp_path, "add", "review.txt")
    _git(tmp_path, "commit", "-qm", "branch version")
    _git(tmp_path, "switch", "-q", branch)
    (tmp_path / "review.txt").write_text("primary version\n", encoding = "utf-8")
    _git(tmp_path, "add", "review.txt")
    _git(tmp_path, "commit", "-qm", "primary version")
    merge = subprocess.run(
        ["git", "merge", "conflicting-branch"],
        cwd = tmp_path,
        check = False,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
    )
    assert merge.returncode != 0

    manifest = build_diff_manifest(project_id)

    assert manifest["selectable"] is False
    assert manifest["blockedReasons"] == ["repository-conflicts"]
    assert manifest["conflictedPaths"] == ["review.txt"]
    assert manifest["files"] == []


def test_truncated_patch_and_incomplete_fingerprint_fail_closed(tmp_path, monkeypatch):
    _repository(tmp_path)
    project_id = _folder_project(tmp_path)
    (tmp_path / "review.txt").write_text(
        "".join(f"changed line {number:06d}\n" for number in range(30_000)),
        encoding = "utf-8",
    )

    truncated = build_diff_manifest(project_id, max_bytes = 4_096)

    assert truncated["selectable"] is False
    assert truncated["blockedReasons"] == ["diff-truncated-or-undecodable"]
    assert truncated["truncated"] is True
    assert truncated["files"] == []

    lines = [f"line {number:02d}" for number in range(1, 41)]
    lines[10] = "bounded parser change"
    (tmp_path / "review.txt").write_text("\n".join(lines) + "\n", encoding = "utf-8")
    monkeypatch.setattr(diff_review_module, "MAX_LINES", 5)
    bounded = build_diff_manifest(project_id)

    assert bounded["selectable"] is False
    assert bounded["blockedReasons"] == ["line-limit"]
    assert bounded["truncated"] is True

    monkeypatch.setattr(
        diff_review_module,
        "workspace_fingerprint",
        lambda _root: "badc0ffe" + ("0" * 56),
    )
    incomplete = build_diff_manifest(project_id)

    assert incomplete["selectable"] is False
    assert incomplete["blockedReasons"] == ["incomplete-source-fingerprint"]
    assert incomplete["files"] == []


def test_manifest_neutralizes_repository_configured_diff_and_filter_commands(tmp_path):
    repository = tmp_path / "repo"
    repository.mkdir()
    _repository(repository)
    project_id = _folder_project(repository)
    (repository / ".gitattributes").write_text(
        "*.txt filter=hostile diff=hostile\n", encoding = "utf-8"
    )
    _git(repository, "add", ".gitattributes")
    _git(repository, "commit", "-qm", "configure attributes")
    marker = tmp_path / "configured-command.executed"
    command = tmp_path / "configured-command.py"
    command.write_text(
        f"#!{sys.executable}\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed\\n', encoding='utf-8')\n"
        "sys.stdout.buffer.write(sys.stdin.buffer.read())\n",
        encoding = "utf-8",
    )
    command.chmod(0o700)
    _git(repository, "config", "filter.hostile.clean", str(command))
    _git(repository, "config", "filter.hostile.smudge", str(command))
    _git(repository, "config", "filter.hostile.required", "true")
    _git(repository, "config", "diff.external", str(command))
    _git(repository, "config", "diff.hostile.command", str(command))
    _git(repository, "config", "diff.hostile.textconv", str(command))
    lines = (repository / "review.txt").read_text(encoding = "utf-8").splitlines()
    lines[3] = "safe diff change"
    (repository / "review.txt").write_text("\n".join(lines) + "\n", encoding = "utf-8")

    manifest = build_diff_manifest(project_id)

    assert manifest["selectable"] is True
    assert _hunk_ids(manifest, "review.txt")
    assert not marker.exists()
