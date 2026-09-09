# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Explicit, selected-path Git checkpoints for project review."""

from __future__ import annotations

import hmac
import os
import re
import time
import uuid
from pathlib import Path

from .git_context import AgentWorkspaceError
from .git_service import (
    build_selected_commit,
    delete_ref,
    git_root,
    project_git,
    _repository_paths,
    repository_command,
    repository_fence,
    repository_ref_optional,
    restore_selected_paths,
    update_ref,
    workspace_fingerprint,
)
from .git_state import (
    delete_checkpoint as delete_checkpoint_record,
    get_checkpoint,
    list_all_checkpoints,
    list_checkpoints,
    save_checkpoint,
)


_OBJECT = re.compile(r"^[0-9a-fA-F]{40,64}$")
_FINGERPRINT = re.compile(r"^(?:c0dec0de|badc0ffe)[0-9a-f]{56}$")


def _owned_paths(root: Path, paths: list[str]) -> list[str]:
    if not isinstance(paths, list) or not paths or len(paths) > 5000:
        raise AgentWorkspaceError("A checkpoint requires at least one selected path.")
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in paths:
        if (
            not isinstance(raw, str)
            or not raw
            or any(c in raw for c in "\x00\r\n\\")
            or len(raw.encode("utf-8")) > 4096
        ):
            raise AgentWorkspaceError("Checkpoint paths are invalid.")
        if os.path.isabs(raw) or raw.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", raw):
            raise AgentWorkspaceError("Checkpoint paths must be relative to the project root.")
        value = raw.replace("\\", "/")
        parts = value.split("/")
        if any(part in {"", ".", ".."} or part.lower() == ".git" for part in parts):
            raise AgentWorkspaceError("Checkpoint paths cannot escape the project root.")
        candidate = (root / Path(*parts)).resolve(strict = False)
        try:
            candidate.relative_to(root.resolve(strict = True))
        except (OSError, RuntimeError, ValueError) as exc:
            raise AgentWorkspaceError("Checkpoint paths cannot escape the project root.") from exc
        if candidate.is_dir():
            raise AgentWorkspaceError("Select individual files for checkpoints and commits.")
        if (root / Path(*parts)).is_symlink():
            raise AgentWorkspaceError("Checkpoint paths cannot be symbolic links.")
        if value not in seen:
            seen.add(value)
            normalized.append(value)
    if sum(len(path.encode("utf-8")) for path in normalized) > 256000:
        raise AgentWorkspaceError("Selected checkpoint paths are too large.")
    return sorted(normalized)


def create_checkpoint(project_id: str, owned_paths: list[str]) -> dict:
    if os.name == "nt":
        raise AgentWorkspaceError(
            "Secure Git checkpoints are disabled on Windows until a boundary test passes."
        )
    from .worktrees import project_operation
    with project_operation(project_id):
        _workspace, guarded_repository = project_git(project_id)
        with repository_fence(guarded_repository):
            workspace_root, repository = project_git(project_id)
            paths = _owned_paths(workspace_root, owned_paths)
            checkpoint_id = str(uuid.uuid4())
            ref_name = f"refs/unsloth-studio/checkpoints/{checkpoint_id}"
            fingerprint = workspace_fingerprint(workspace_root)
            if not fingerprint.startswith("c0dec0de"):
                raise AgentWorkspaceError(
                    "The repository review evidence is incomplete. Reduce the change set and retry."
                )
            repository_paths = _repository_paths(repository, workspace_root, paths)
            commit_sha = build_selected_commit(
                repository, repository_paths, "Unsloth Studio checkpoint"
            )
            if workspace_fingerprint(workspace_root) != fingerprint:
                raise AgentWorkspaceError("The project changed while the checkpoint was prepared.")
            update_ref(repository, ref_name, commit_sha)
            record = {
                "id": checkpoint_id,
                "projectId": project_id,
                "gitRoot": str(repository),
                "refName": ref_name,
                "commitSha": commit_sha,
                "ownedPaths": paths,
                "sourceFingerprint": fingerprint,
                "createdAt": int(time.time() * 1000),
            }
            try:
                save_checkpoint(record)
            except Exception:
                try:
                    delete_ref(repository, ref_name, commit_sha)
                except AgentWorkspaceError:
                    pass
                raise
            return record


def rollback_checkpoint(
    project_id: str, checkpoint_id: str, expected_current_fingerprint: str
) -> dict:
    if os.name == "nt":
        raise AgentWorkspaceError(
            "Secure Git checkpoints are disabled on Windows until a boundary test passes."
        )
    from .worktrees import project_operation
    with project_operation(project_id):
        _workspace, guarded_repository = project_git(project_id)
        with repository_fence(guarded_repository):
            if _FINGERPRINT.fullmatch(expected_current_fingerprint) is None:
                raise AgentWorkspaceError("Current workspace fingerprint is invalid.")
            if not expected_current_fingerprint.startswith("c0dec0de"):
                raise AgentWorkspaceError(
                    "The repository review evidence is incomplete. Reduce the change set and retry."
                )
            record = get_checkpoint(checkpoint_id)
            if record is None or record["projectId"] != project_id:
                raise AgentWorkspaceError("Git checkpoint not found.")
            workspace_root, repository = project_git(project_id)
            if os.path.normcase(os.path.abspath(record["gitRoot"])) != os.path.normcase(
                str(repository)
            ):
                raise AgentWorkspaceError("Git checkpoint repository identity changed.")
            if repository_ref_optional(repository, record["refName"]) != record["commitSha"]:
                raise AgentWorkspaceError("Git checkpoint reference changed outside Studio.")
            current = workspace_fingerprint(workspace_root)
            if not hmac.compare_digest(current, expected_current_fingerprint):
                raise AgentWorkspaceError(
                    "The project changed after the checkpoint preview. Refresh and retry."
                )
            if _OBJECT.fullmatch(record["commitSha"]) is None:
                raise AgentWorkspaceError("Git checkpoint identity is invalid.")
            repository_paths = _repository_paths(repository, workspace_root, record["ownedPaths"])
            restore_selected_paths(repository, record["commitSha"], repository_paths)
            return {
                "checkpoint": record,
                "currentFingerprint": workspace_fingerprint(workspace_root),
            }


def delete_checkpoint(project_id: str, checkpoint_id: str) -> bool:
    from .worktrees import project_operation
    with project_operation(project_id):
        _workspace, guarded_repository = project_git(project_id)
        with repository_fence(guarded_repository):
            record = get_checkpoint(checkpoint_id)
            if record is None or record["projectId"] != project_id:
                raise AgentWorkspaceError("Git checkpoint not found.")
            repository = git_root(Path(record["gitRoot"]))
            current_ref = repository_ref_optional(repository, record["refName"])
            if current_ref is not None and current_ref.lower() != str(record["commitSha"]).lower():
                raise AgentWorkspaceError("The checkpoint reference changed outside Studio.")
            delete_ref(repository, record["refName"], record["commitSha"])
            return delete_checkpoint_record(checkpoint_id, project_id)


def remove_project_checkpoints(project_id: str) -> int:
    """Remove all recorded checkpoint refs while the deletion fence is held."""
    removed = 0
    for record in list_checkpoints(project_id):
        repository = git_root(Path(record["gitRoot"]))
        current_ref = repository_ref_optional(repository, record["refName"])
        if current_ref is not None and current_ref.lower() != str(record["commitSha"]).lower():
            raise AgentWorkspaceError("The checkpoint reference changed outside Studio.")
        delete_ref(repository, record["refName"], record["commitSha"])
        if delete_checkpoint_record(record["id"], project_id):
            removed += 1
    return removed


def reconcile_checkpoints_on_startup() -> dict[str, int]:
    """Audit checkpoint refs without deleting ambiguous durable state.

    A failed ref probe can mean a missing ref, a damaged repository, or a
    transient Git failure.  Until the recovery lane has a tri-state Git probe,
    startup retains the row and reports the uncertainty instead of deleting a
    user's recoverable checkpoint.
    """
    result = {"removed": 0, "retained": 0, "errors": 0}
    for record in list_all_checkpoints():
        try:
            repository = git_root(Path(record["gitRoot"]))
            try:
                repository_command(
                    repository,
                    ["show-ref", "--verify", "--quiet", record["refName"]],
                    timeout = 10,
                    output_limit = 1024,
                )
                result["retained"] += 1
            except AgentWorkspaceError:
                # A missing ref is not enough evidence to remove the durable
                # row. Retain it for an explicit repair until Git exposes a
                # tri-state probe that distinguishes a missing ref from I/O
                # failure.
                result["errors"] += 1
        except Exception:
            result["errors"] += 1
    return result


__all__ = [
    "create_checkpoint",
    "delete_checkpoint",
    "list_checkpoints",
    "remove_project_checkpoints",
    "reconcile_checkpoints_on_startup",
    "rollback_checkpoint",
]
