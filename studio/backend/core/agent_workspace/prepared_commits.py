# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One-use, reviewed commit objects that leave the active branch and index intact."""

import hashlib
import json
import secrets
import time
import uuid

from .git_context import AgentWorkspaceError
from . import prepared_commit_state as state
from .checkpoints import _owned_paths
from .git_service import (
    _repository_paths,
    build_selected_commit,
    project_git,
    repository_branch,
    repository_command,
    repository_fence,
    repository_head,
    repository_status,
    workspace_fingerprint,
)
from .worktrees import project_operation


def _digest(record):
    return hashlib.sha256(
        json.dumps(
            {
                key: record[key]
                for key in (
                    "id",
                    "projectId",
                    "branchRef",
                    "headSha",
                    "gitRoot",
                    "message",
                    "ownedPaths",
                    "sourceFingerprint",
                    "refName",
                    "createdAt",
                    "expiresAt",
                )
            },
            sort_keys = True,
            separators = (",", ":"),
        ).encode()
    ).hexdigest()


def _public(record):
    return {
        key: record[key]
        for key in (
            "id",
            "projectId",
            "status",
            "message",
            "ownedPaths",
            "sourceFingerprint",
            "refName",
            "createdAt",
            "expiresAt",
            "commitSha",
            "confirmedAt",
        )
        if record.get(key) is not None
    } | {
        "branch": record["branchRef"].removeprefix("refs/heads/"),
        "baseHead": record["headSha"],
    }


def prepare_commit(project_id, paths, message):
    message = message.strip()
    if not message or "\x00" in message or len(message.encode()) > 32000:
        raise AgentWorkspaceError("Commit messages require 1 to 32,000 UTF-8 bytes.")
    with project_operation(project_id):
        root, repository = project_git(project_id)
        with repository_fence(repository):
            paths = _owned_paths(root, paths)
            from .git_service import _status_records

            files, counts = _status_records(repository_status(repository, scope_root = root))
            changed = {item["path"] for item in files}
            if counts["conflicts"] or not set(_repository_paths(repository, root, paths)).issubset(
                changed
            ):
                raise AgentWorkspaceError(
                    "Select changed files and resolve conflicts before preparing a commit."
                )
            fingerprint = workspace_fingerprint(root)
            if not fingerprint.startswith("c0dec0de"):
                raise AgentWorkspaceError(
                    "Repository evidence is incomplete. Reduce the change set."
                )
            from .git_review import build_diff_manifest

            manifest = build_diff_manifest(project_id)
            preview_files = [item for item in manifest["files"] if item["path"] in paths]
            if not manifest["selectable"] or len(preview_files) != len(paths):
                raise AgentWorkspaceError(
                    "Selected file contents could not be fully reviewed. Reduce the change set."
                )
            rename_sources = []
            for item in preview_files:
                if not item["code"].startswith("R"):
                    continue
                source = item.get("oldPath")
                if not source or item.get("oldPathEncoding") != "utf-8":
                    raise AgentWorkspaceError("The rename source cannot be safely selected.")
                if source not in paths and (root / source).exists():
                    raise AgentWorkspaceError(
                        "Select the recreated rename source as well before preparing this commit."
                    )
                rename_sources.append(source)
            # A temporary index begins at HEAD: both halves of a rename must be
            # staged, or the reviewed move would become an added copy.
            paths = _owned_paths(root, paths + rename_sources)
            if workspace_fingerprint(root) != fingerprint:
                raise AgentWorkspaceError("Repository changed while the commit preview was built.")
            identifier = str(uuid.uuid4())
            now = int(time.time() * 1000)
            record = {
                "id": identifier,
                "projectId": project_id,
                "operation": "prepare_commit",
                "status": "awaiting_confirmation",
                "branchRef": "refs/heads/" + repository_branch(repository),
                "headSha": repository_head(repository),
                "gitRoot": str(repository),
                "message": message,
                "ownedPaths": paths,
                "sourceFingerprint": fingerprint,
                "refName": "refs/unsloth-studio/prepared-commits/" + identifier,
                "createdAt": now,
                "expiresAt": now + 300000,
            }
            record["payloadDigest"] = _digest(record)
            token = secrets.token_urlsafe(32)
            state.save_preparation(record, token, now = now)
            return _public(record) | {"confirmationToken": token, "previewFiles": preview_files}


def confirm_prepared_commit(project_id, preparation_id, token):
    record = state.reserve_confirmation(
        preparation_id, project_id, token, now = int(time.time() * 1000)
    )
    dispatched = False
    try:
        if record["payloadDigest"] != _digest(record):
            raise AgentWorkspaceError("Commit preparation integrity check failed.")
        with project_operation(project_id):
            root, repository = project_git(project_id)
            with repository_fence(repository):

                def require_current():
                    if (
                        str(repository) != record["gitRoot"]
                        or repository_head(repository) != record["headSha"]
                        or "refs/heads/" + repository_branch(repository) != record["branchRef"]
                        or workspace_fingerprint(root) != record["sourceFingerprint"]
                    ):
                        raise AgentWorkspaceError(
                            "Repository changed after preview. Prepare the commit again."
                        )

                require_current()
                paths = _owned_paths(root, record["ownedPaths"])
                commit = build_selected_commit(
                    repository, _repository_paths(repository, root, paths), record["message"]
                )
                state.save_candidate_commit(preparation_id, commit)
                require_current()
                dispatched = True
                repository_command(
                    repository,
                    ["update-ref", "--no-deref", record["refName"], commit, "0" * len(commit)],
                )
                now = int(time.time() * 1000)
                state.mark_confirmed(preparation_id, commit, now = now)
                return _public(
                    record | {"status": "confirmed", "commitSha": commit, "confirmedAt": now}
                )
    except Exception:
        if not dispatched:
            state.mark_failed(preparation_id)
        # A possibly published ref remains in durable state for recovery; never retry blindly.
        raise


def remove_prepared_commit(project_id, preparation_id):
    from .git_service import repository_ref_optional

    record = state.get_preparation(preparation_id)
    if record is None or record["projectId"] != project_id:
        raise AgentWorkspaceError("Prepared commit not found.")
    ref = "refs/unsloth-studio/prepared-commits/" + preparation_id
    if record["refName"] != ref:
        raise AgentWorkspaceError("Prepared commit ownership changed.")
    with project_operation(project_id):
        _, repository = project_git(project_id)
        with repository_fence(repository):
            if str(repository) != record["gitRoot"]:
                raise AgentWorkspaceError("Prepared commit repository changed.")
            current = repository_ref_optional(repository, ref)
            if current is not None:
                if current != record["commitSha"]:
                    raise AgentWorkspaceError("Prepared commit ref changed outside Studio.")
                repository_command(repository, ["update-ref", "--no-deref", "-d", ref, current])
            state.delete_preparation(preparation_id, project_id)
    return {"removed": True}
