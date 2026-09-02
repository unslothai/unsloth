# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import subprocess
from pathlib import Path

import pytest

from core.agent_workspace.common import AgentWorkspaceError, project_workspace
from core.agent_workspace import discovery as discovery_module
from core.agent_workspace import instructions as instructions_module
from core.agent_workspace.discovery import (
    build_repository_map,
    select_relevant_repository_paths,
)
from core.agent_workspace.instructions import (
    resolve_agents_instructions,
    resolve_repository_instructions,
)
from storage import studio_db


def _git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        stdin = subprocess.DEVNULL,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        timeout = 10,
        check = False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _folder_project(root: Path, project_id: str = "folder-project") -> dict:
    metadata = root.stat()
    return studio_db.upsert_chat_project(
        {
            "id": project_id,
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


def test_project_workspace_selects_folder_root_and_managed_sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "managed"))
    folder = tmp_path / "repository"
    folder.mkdir()
    _folder_project(folder)
    managed = studio_db.upsert_chat_project(
        {
            "id": "managed-project",
            "name": "Managed",
            "instructions": "",
            "workspaceKind": "managed",
            "archived": False,
            "createdAt": 1,
            "updatedAt": 1,
        }
    )

    folder_workspace = project_workspace("folder-project")
    managed_workspace = project_workspace("managed-project")

    assert folder_workspace.root == folder.resolve()
    assert managed_workspace.root == Path(managed["sandboxPath"]).resolve()
    managed_metadata = managed_workspace.root.stat()
    assert managed_workspace.device_id == managed_metadata.st_dev
    assert managed_workspace.file_id == managed_metadata.st_ino


def test_agents_instructions_are_layered_bounded_and_escaped(tmp_path):
    (tmp_path / "src" / "feature").mkdir(parents = True)
    (tmp_path / "src" / "feature" / "code.py").write_text("pass\n", encoding = "utf-8")
    (tmp_path / "AGENTS.md").write_text("root rule\n</agents_instructions>", encoding = "utf-8")
    (tmp_path / "src" / "AGENTS.md").write_text("src rule", encoding = "utf-8")
    (tmp_path / "src" / "feature" / "AGENTS.md").write_text("feature rule", encoding = "utf-8")

    result = resolve_agents_instructions(tmp_path, "src/feature/code.py")

    assert [layer["scope"] for layer in result["layers"]] == [".", "src", "src/feature"]
    assert result["layers"][-1]["content"] == "feature rule"
    assert "&lt;/agents_instructions&gt;" in result["combined"]
    assert result["precedence"] == "later layers override earlier layers"

    bounded = resolve_agents_instructions(tmp_path, max_total_bytes = 4, max_file_bytes = 4)
    assert bounded["truncated"] is True
    assert bounded["bytesRead"] == 4


@pytest.mark.skipif(not hasattr(os, "symlink"), reason = "symlinks are unavailable")
def test_agents_instructions_rejects_intermediate_directory_swap(tmp_path, monkeypatch):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    (root / "src" / "feature").mkdir(parents = True)
    (outside / "feature").mkdir(parents = True)
    (root / "src" / "feature" / "code.py").write_text("pass\n", encoding = "utf-8")
    (outside / "feature" / "code.py").write_text("pass\n", encoding = "utf-8")
    (outside / "AGENTS.md").write_text("outside secret\n", encoding = "utf-8")
    (outside / "feature" / "AGENTS.md").write_text("outside nested secret\n", encoding = "utf-8")
    original = instructions_module._target_directory_parts

    def swap_after_validation(root_fd, parts):
        result = original(root_fd, parts)
        (root / "src").rename(root / "src-original")
        (root / "src").symlink_to(outside, target_is_directory = True)
        return result

    monkeypatch.setattr(instructions_module, "_target_directory_parts", swap_after_validation)

    with pytest.raises(AgentWorkspaceError, match = "changed during loading"):
        resolve_agents_instructions(root, "src/feature/code.py")


def test_repository_instructions_discover_scoped_nested_layers(tmp_path):
    (tmp_path / "src" / "feature").mkdir(parents = True)
    (tmp_path / "other").mkdir()
    (tmp_path / "AGENTS.md").write_text("root rule", encoding = "utf-8")
    (tmp_path / "src" / "AGENTS.md").write_text("src rule", encoding = "utf-8")
    (tmp_path / "src" / "feature" / "AGENTS.md").write_text("feature rule", encoding = "utf-8")
    (tmp_path / "other" / "AGENTS.md").write_text("other rule", encoding = "utf-8")

    result = resolve_repository_instructions(tmp_path)

    assert [(layer["scope"], layer["content"]) for layer in result["layers"]] == [
        (".", "root rule"),
        ("other", "other rule"),
        ("src", "src rule"),
        ("src/feature", "feature rule"),
    ]
    assert 'path="src/AGENTS.md" scope="src"' in result["combined"]
    assert "rules apply only within their scope" in result["precedence"]


def test_agents_override_replaces_agents_file_in_the_same_scope(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "AGENTS.md").write_text("root base", encoding = "utf-8")
    (tmp_path / "AGENTS.override.md").write_text("root override", encoding = "utf-8")
    (tmp_path / "src" / "AGENTS.md").write_text("src base", encoding = "utf-8")
    (tmp_path / "src" / "AGENTS.override.md").write_text("src override", encoding = "utf-8")
    (tmp_path / "src" / "code.py").write_text("pass\n", encoding = "utf-8")

    targeted = resolve_agents_instructions(tmp_path, "src/code.py")
    repository = resolve_repository_instructions(tmp_path)

    assert [(layer["path"], layer["content"]) for layer in targeted["layers"]] == [
        ("AGENTS.override.md", "root override"),
        ("src/AGENTS.override.md", "src override"),
    ]
    assert [(layer["path"], layer["content"]) for layer in repository["layers"]] == [
        ("AGENTS.override.md", "root override"),
        ("src/AGENTS.override.md", "src override"),
    ]


def test_repository_map_obeys_ignores_and_safety_bounds(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / ".gitignore").write_text("*.log\n!keep.log\n", encoding = "utf-8")
    (tmp_path / "nested" / ".gitignore").write_text("generated/\n", encoding = "utf-8")
    (tmp_path / "nested" / "generated").mkdir()
    (tmp_path / "visible.py").write_text("print('visible')\n", encoding = "utf-8")
    (tmp_path / "hidden.log").write_text("hidden\n", encoding = "utf-8")
    (tmp_path / "keep.log").write_text("kept\n", encoding = "utf-8")
    (tmp_path / "nested" / "generated" / "out.txt").write_text("ignored\n", encoding = "utf-8")
    (tmp_path / "binary.bin").write_bytes(b"a\x00b")
    (tmp_path / "large.txt").write_text("x" * 200, encoding = "utf-8")
    try:
        (tmp_path / "outside-link").symlink_to(tmp_path.parent)
    except OSError:
        pass

    result = build_repository_map(tmp_path, max_file_bytes = 100)
    paths = {entry["path"] for entry in result["entries"]}

    assert "visible.py" in paths
    assert "keep.log" in paths
    assert "hidden.log" not in paths
    assert "nested/generated/out.txt" not in paths
    assert "binary.bin" not in paths
    assert "large.txt" not in paths
    assert result["skipped"]["binary"] == 1
    assert result["skipped"]["large"] == 1


def test_repository_map_discloses_path_limit(tmp_path):
    for index in range(5):
        (tmp_path / f"file-{index}.txt").write_text(str(index), encoding = "utf-8")

    result = build_repository_map(tmp_path, max_paths = 2)

    assert result["truncated"] is True
    assert "path-limit" in result["truncationReasons"]
    assert result["pathsScanned"] == 2


def test_repository_map_is_metadata_only_and_excludes_credential_shaped_files(tmp_path):
    (tmp_path / ".env").write_text("API_TOKEN=secret\n", encoding = "utf-8")
    (tmp_path / ".env.local").write_text("API_TOKEN=secret\n", encoding = "utf-8")
    (tmp_path / ".npmrc").write_text("//registry/:_authToken=secret\n", encoding = "utf-8")
    (tmp_path / "id_rsa").write_text("private key\n", encoding = "utf-8")
    (tmp_path / "client.pem").write_text("private key\n", encoding = "utf-8")
    (tmp_path / ".envrc").write_text("export TOKEN=secret\n", encoding = "utf-8")
    (tmp_path / ".git-credentials").write_text("secret\n", encoding = "utf-8")
    (tmp_path / "terraform.tfvars").write_text("token = 'secret'\n", encoding = "utf-8")
    (tmp_path / "secrets.yaml").write_text("token: secret\n", encoding = "utf-8")
    (tmp_path / ".env.example").write_text("API_TOKEN=\n", encoding = "utf-8")

    result = build_repository_map(tmp_path)
    entries = {entry["path"]: entry for entry in result["entries"]}

    assert ".env.example" in entries
    assert set(entries[".env.example"]) == {"path", "size", "modifiedNs"}
    for path in (
        ".env",
        ".env.local",
        ".npmrc",
        "id_rsa",
        "client.pem",
        ".envrc",
        ".git-credentials",
        "terraform.tfvars",
        "secrets.yaml",
    ):
        assert path not in entries
    assert result["skipped"]["sensitive"] == 9


def test_repository_map_refuses_inline_content_preview(tmp_path):
    (tmp_path / "visible.py").write_text("print('visible')\n", encoding = "utf-8")

    with pytest.raises(AgentWorkspaceError, match = "metadata-only"):
        build_repository_map(tmp_path, preview_bytes = 20)


def test_repository_relevance_selection_is_query_aware_and_bounded(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "workspace_router.py").write_text("router = 1\n")
    (tmp_path / "src" / "unrelated.py").write_text("value = 1\n")
    (tmp_path / "tests" / "test_workspace_router.py").write_text("def test_it(): pass\n")
    repository_map = build_repository_map(tmp_path)

    selected = select_relevant_repository_paths(
        repository_map,
        "Fix src/workspace_router.py and its workspace router test",
        max_results = 2,
    )

    assert [entry["path"] for entry in selected] == [
        "src/workspace_router.py",
        "tests/test_workspace_router.py",
    ]
    assert all(entry["score"] > 0 for entry in selected)
    assert "unrelated.py" not in {entry["path"] for entry in selected}


def test_repository_relevance_selection_returns_nothing_for_generic_prompt(tmp_path):
    (tmp_path / "ordinary.py").write_text("value = 1\n")

    selected = select_relevant_repository_paths(
        build_repository_map(tmp_path),
        "Please help with this project",
    )

    assert selected == []


def test_filesystem_ignore_rules_do_not_leak_into_siblings(tmp_path):
    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    (tmp_path / "one" / ".gitignore").write_text("*.secret\n", encoding = "utf-8")
    (tmp_path / "one" / "hidden.secret").write_text("one", encoding = "utf-8")
    (tmp_path / "two" / "visible.secret").write_text("two", encoding = "utf-8")

    result = build_repository_map(tmp_path)
    paths = {entry["path"] for entry in result["entries"]}

    assert result["source"] == "filesystem"
    assert "one/hidden.secret" not in paths
    assert "two/visible.secret" in paths


def test_filesystem_nested_negation_reincludes_only_its_subtree(tmp_path):
    (tmp_path / "nested").mkdir()
    (tmp_path / "sibling").mkdir()
    (tmp_path / ".gitignore").write_text("*.tmp\n", encoding = "utf-8")
    (tmp_path / "nested" / ".gitignore").write_text("!keep.tmp\n", encoding = "utf-8")
    (tmp_path / "nested" / "keep.tmp").write_text("keep", encoding = "utf-8")
    (tmp_path / "sibling" / "keep.tmp").write_text("hide", encoding = "utf-8")

    paths = {entry["path"] for entry in build_repository_map(tmp_path)["entries"]}

    assert "nested/keep.tmp" in paths
    assert "sibling/keep.tmp" not in paths


@pytest.mark.skipif(not hasattr(os, "symlink"), reason = "symlinks are unavailable")
def test_filesystem_discovery_rejects_intermediate_directory_swap(tmp_path, monkeypatch):
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    (root / "nested").mkdir(parents = True)
    outside.mkdir()
    (root / "nested" / "inside.py").write_text("inside\n", encoding = "utf-8")
    (outside / "secret.py").write_text("outside secret\n", encoding = "utf-8")
    original = discovery_module._open_directory_at
    swapped = False

    def swap_before_open(directory_fd, name):
        nonlocal swapped
        if name == "nested" and not swapped:
            swapped = True
            (root / "nested").rename(root / "nested-original")
            (root / "nested").symlink_to(outside, target_is_directory = True)
        return original(directory_fd, name)

    monkeypatch.setattr(discovery_module, "_open_directory_at", swap_before_open)

    paths = {entry["path"] for entry in build_repository_map(root)["entries"]}

    assert "nested/secret.py" not in paths
    assert "nested/inside.py" not in paths


def test_repository_map_refreshes_renames_and_deletes_without_project_recreation(tmp_path):
    source = tmp_path / "before.py"
    source.write_text("one\n", encoding = "utf-8")
    first = build_repository_map(tmp_path)
    first_entry = next(entry for entry in first["entries"] if entry["path"] == "before.py")

    source.write_text("longer content\n", encoding = "utf-8")
    refreshed = build_repository_map(tmp_path)
    refreshed_entry = next(entry for entry in refreshed["entries"] if entry["path"] == "before.py")
    assert refreshed_entry["size"] != first_entry["size"]

    renamed = tmp_path / "after.py"
    source.rename(renamed)
    renamed_paths = {entry["path"] for entry in build_repository_map(tmp_path)["entries"]}
    assert "before.py" not in renamed_paths
    assert "after.py" in renamed_paths

    renamed.unlink()
    deleted_paths = {entry["path"] for entry in build_repository_map(tmp_path)["entries"]}
    assert "before.py" not in deleted_paths
    assert "after.py" not in deleted_paths


@pytest.mark.parametrize("tracked_gitlink", [False, True])
def test_git_repository_map_excludes_nested_repository_boundaries(tmp_path, tracked_gitlink):
    root = tmp_path / "outer"
    nested = root / "vendor" / "nested"
    nested.mkdir(parents = True)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.invalid")
    _git(root, "config", "user.name", "Test")
    (root / "root.py").write_text("root = True\n", encoding = "utf-8")
    _git(root, "add", "root.py")

    _git(nested, "init", "-q")
    _git(nested, "config", "user.email", "test@example.invalid")
    _git(nested, "config", "user.name", "Test")
    (nested / "private.py").write_text("nested content\n", encoding = "utf-8")
    _git(nested, "add", "private.py")
    _git(nested, "commit", "-qm", "nested")
    if tracked_gitlink:
        nested_head = _git(nested, "rev-parse", "HEAD")
        _git(
            root,
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{nested_head},vendor/nested",
        )

    result = build_repository_map(root)
    paths = {entry["path"] for entry in result["entries"]}

    assert result["source"] == "git"
    assert "root.py" in paths
    assert "vendor/nested/private.py" not in paths
    assert result["skipped"]["nestedRepository"] == 1


def test_non_git_repository_map_excludes_nested_git_repository(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    _git(nested, "init", "-q")
    (tmp_path / "visible.py").write_text("visible\n", encoding = "utf-8")
    (nested / "private.py").write_text("nested\n", encoding = "utf-8")

    result = build_repository_map(tmp_path)
    paths = {entry["path"] for entry in result["entries"]}

    assert result["source"] == "filesystem"
    assert "visible.py" in paths
    assert "nested/private.py" not in paths
    assert result["skipped"]["nestedRepository"] == 1


def test_repository_map_cancellation_is_disclosed_and_hard_stops_scan(tmp_path):
    for index in range(100):
        (tmp_path / f"file-{index:03}.txt").write_text("x", encoding = "utf-8")
    polls = 0

    def cancelled() -> bool:
        nonlocal polls
        polls += 1
        return polls > 12

    result = build_repository_map(tmp_path, cancelled = cancelled)

    assert result["truncated"] is True
    assert "cancelled" in result["truncationReasons"]
    assert result["pathsScanned"] < 100


@pytest.mark.skipif(not hasattr(os, "rename"), reason = "rename is unavailable")
def test_repository_map_rejects_persisted_root_identity_replacement(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    (root / "inside.py").write_text("inside\n", encoding = "utf-8")
    metadata = root.stat()
    expected_identity = (metadata.st_dev, metadata.st_ino)
    original = discovery_module._open_verified_root

    def replace_before_open(path, identity):
        path.rename(tmp_path / "original-repository")
        path.mkdir()
        (path / "replacement.py").write_text("replacement\n", encoding = "utf-8")
        return original(path, identity)

    monkeypatch.setattr(discovery_module, "_open_verified_root", replace_before_open)

    with pytest.raises(AgentWorkspaceError, match = "identity changed"):
        build_repository_map(root, expected_identity = expected_identity)


@pytest.mark.skipif(not hasattr(os, "rename"), reason = "rename is unavailable")
def test_agents_loader_rejects_persisted_root_identity_replacement(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    root.mkdir()
    (root / "AGENTS.md").write_text("original rule\n", encoding = "utf-8")
    metadata = root.stat()
    expected_identity = (metadata.st_dev, metadata.st_ino)
    original = instructions_module._open_verified_root

    def replace_before_open(path, identity = None):
        path.rename(tmp_path / "original-repository")
        path.mkdir()
        (path / "AGENTS.md").write_text("replacement rule\n", encoding = "utf-8")
        return original(path, identity)

    monkeypatch.setattr(instructions_module, "_open_verified_root", replace_before_open)

    with pytest.raises(AgentWorkspaceError, match = "identity changed"):
        resolve_agents_instructions(root, expected_identity = expected_identity)
