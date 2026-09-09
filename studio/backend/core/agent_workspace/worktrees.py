# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Crash-safe lifecycle for Studio-owned Git worktrees.

The filesystem marker and the SQLite row are two halves of the ownership
proof. A checkout is never removed when either half is ambiguous, and startup
reconciliation repairs only states that can be proven from both halves.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import stat
import threading
import time
import uuid
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Iterator, Optional

from utils.paths import ensure_dir, project_workspaces_root

from .git_context import AgentWorkspaceError, project_workspace
from .git_service import (
    add_worktree,
    git_root,
    preflight_merge,
    remove_worktree,
    repository_branch,
    repository_command,
    repository_head,
    repository_ref,
    repository_status,
    repository_fence,
    worktree_entries,
)
from .git_service import _status_records
from .git_state import (
    get_worktree,
    list_all_worktrees,
    list_worktrees,
    record_worktree_merge,
    save_worktree,
    transition_worktree_status,
)


_BRANCH = re.compile(r"^unsloth-studio/[A-Za-z0-9][A-Za-z0-9._/-]{0,120}$")
_BASE_REF = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@-]{0,255}$")
_MARKER_NAME = "owner.json"
_MARKER_LIMIT = 16 * 1024
_RECONCILE_LIMIT = 1024
_SCAN_LIMIT = 8192
_ACTIVE_STATUSES = frozenset({"creating", "active", "removing", "needs_attention"})

TASK_WORKTREE_GUARD_PROTOCOL = 1


def _task_worktree_guard(project_id: str, worktree_id: str):
    try:
        from .task_workspaces import worktree_idle_guard
    except ModuleNotFoundError as exc:
        if exc.name != __package__ + ".task_workspaces":
            raise
        return nullcontext()
    return worktree_idle_guard(project_id, worktree_id)


_OPERATION_CONDITION = threading.Condition()
_PROJECT_ACTIVE: dict[str, int] = {}
_PROJECT_LOCKS: dict[str, threading.RLock] = {}
_PROJECT_DELETING: set[str] = set()
_OPERATION_LOCAL = threading.local()


def _now_ms() -> int:
    return int(time.time() * 1000)


@contextmanager
def _project_operation(project_id: str) -> Iterator[None]:
    with _OPERATION_CONDITION:
        if project_id in _PROJECT_DELETING:
            raise AgentWorkspaceError(
                "Worktree operations are unavailable while the project is being deleted."
            )
        _PROJECT_ACTIVE[project_id] = _PROJECT_ACTIVE.get(project_id, 0) + 1
        project_lock = _PROJECT_LOCKS.setdefault(project_id, threading.RLock())
    project_lock.acquire()
    try:
        depth = getattr(_OPERATION_LOCAL, "depth", {})
        if depth.get(project_id, 0):
            yield
        else:
            from .git_guard import project_git_guard
            _OPERATION_LOCAL.depth = {**depth, project_id: 1}
            try:
                with project_git_guard(project_id):
                    yield
            finally:
                _OPERATION_LOCAL.depth = depth
    finally:
        project_lock.release()
        with _OPERATION_CONDITION:
            remaining = _PROJECT_ACTIVE.get(project_id, 1) - 1
            if remaining > 0:
                _PROJECT_ACTIVE[project_id] = remaining
            else:
                _PROJECT_ACTIVE.pop(project_id, None)
                _PROJECT_LOCKS.pop(project_id, None)
            _OPERATION_CONDITION.notify_all()


# Readable public spelling for sibling dependency lanes that mutate Git-owned
# project state under the same durable deletion fence.
project_operation = _project_operation


def begin_project_deletion(project_id: str, timeout_seconds: float = 15) -> None:
    deadline = time.monotonic() + max(0.1, float(timeout_seconds))
    with _OPERATION_CONDITION:
        if project_id in _PROJECT_DELETING:
            raise AgentWorkspaceError("Project worktree deletion is already in progress.")
        _PROJECT_DELETING.add(project_id)
        while _PROJECT_ACTIVE.get(project_id, 0):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _PROJECT_DELETING.discard(project_id)
                _OPERATION_CONDITION.notify_all()
                raise AgentWorkspaceError(
                    "A worktree operation is still running. Try deleting the project again."
                )
            _OPERATION_CONDITION.wait(timeout = remaining)


def finish_project_deletion(project_id: str) -> None:
    with _OPERATION_CONDITION:
        _PROJECT_DELETING.discard(project_id)
        _OPERATION_CONDITION.notify_all()


def _lstat(path: Path) -> Optional[os.stat_result]:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _plain_directory(path: Path) -> bool:
    metadata = _lstat(path)
    return metadata is not None and stat.S_ISDIR(metadata.st_mode)


def _worktree_root(*, create: bool = True) -> Path:
    base = Path(project_workspaces_root()).expanduser()
    if create:
        try:
            ensure_dir(base)
        except OSError as exc:
            raise AgentWorkspaceError("Studio worktree storage is unavailable.") from exc
    try:
        base = base.resolve(strict = True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise AgentWorkspaceError("Studio worktree storage is unavailable.") from exc
    candidate = base / ".agent-worktrees"
    metadata = _lstat(candidate)
    if metadata is None:
        if not create:
            return candidate
        try:
            candidate.mkdir(mode = 0o700)
        except OSError as exc:
            raise AgentWorkspaceError("Studio worktree storage is unavailable.") from exc
        metadata = _lstat(candidate)
    if metadata is None or not stat.S_ISDIR(metadata.st_mode) or candidate.is_symlink():
        raise AgentWorkspaceError("Studio worktree storage is not a safe directory.")
    try:
        resolved = candidate.resolve(strict = True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise AgentWorkspaceError("Studio worktree storage is unavailable.") from exc
    if resolved.parent != base:
        raise AgentWorkspaceError("Studio worktree storage cannot be a symbolic link.")
    return resolved


def _storage_key(project_id: str) -> str:
    return hashlib.sha256(project_id.encode("utf-8")).hexdigest()[:32]


def _expected_paths(
    project_id: str,
    worktree_id: str,
    *,
    create_root: bool = True,
):
    root = _worktree_root(create = create_root)
    container = root / _storage_key(project_id) / worktree_id
    return container, container / "workspace", container / _MARKER_NAME


def _same_path(left: str | Path, right: str | Path) -> bool:
    try:
        return os.path.normcase(os.path.abspath(os.fspath(left))) == os.path.normcase(
            os.path.abspath(os.fspath(right))
        )
    except (TypeError, ValueError):
        return False


def _ensure_container(container: Path) -> None:
    root = _worktree_root()
    try:
        relative = container.relative_to(root)
    except ValueError as exc:
        raise AgentWorkspaceError("Worktree destination is outside Studio storage.") from exc
    if len(relative.parts) != 2 or not re.fullmatch(r"[0-9a-f]{32}", relative.parts[0]):
        raise AgentWorkspaceError("Worktree destination is invalid.")
    bucket = container.parent
    metadata = _lstat(bucket)
    if metadata is None:
        try:
            bucket.mkdir(mode = 0o700)
        except FileExistsError:
            pass
        except OSError as exc:
            raise AgentWorkspaceError("Worktree storage could not be prepared.") from exc
        metadata = _lstat(bucket)
    if metadata is None or not stat.S_ISDIR(metadata.st_mode) or bucket.is_symlink():
        raise AgentWorkspaceError("Worktree storage contains an unsafe path.")
    try:
        container.mkdir(mode = 0o700)
    except FileExistsError as exc:
        raise AgentWorkspaceError("Worktree destination already exists.") from exc
    except OSError as exc:
        raise AgentWorkspaceError("Worktree destination could not be prepared.") from exc


def _remove_empty_container(container: Path) -> None:
    try:
        container.rmdir()
    except OSError:
        return
    try:
        container.parent.rmdir()
    except OSError:
        pass


def _write_marker(path: Path, payload: dict) -> None:
    data = json.dumps(payload, ensure_ascii = True, sort_keys = True, separators = (",", ":"))
    encoded = data.encode("utf-8")
    if len(encoded) > _MARKER_LIMIT:
        raise AgentWorkspaceError("Worktree ownership marker is too large.")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise AgentWorkspaceError("Worktree ownership marker could not be published.")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.link(temporary, path, follow_symlinks = False)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as exc:
        raise AgentWorkspaceError("Worktree ownership marker already exists.") from exc
    except OSError as exc:
        raise AgentWorkspaceError("Worktree ownership marker could not be published.") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_marker(path: Path) -> dict:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise AgentWorkspaceError("Worktree ownership marker is invalid.") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MARKER_LIMIT:
            raise AgentWorkspaceError("Worktree ownership marker is invalid.")
        chunks: list[bytes] = []
        remaining = _MARKER_LIMIT + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(raw) > _MARKER_LIMIT:
        raise AgentWorkspaceError("Worktree ownership marker is invalid.")
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook = _reject_duplicate_fields)
    except (UnicodeError, ValueError, AgentWorkspaceError) as exc:
        raise AgentWorkspaceError("Worktree ownership marker is invalid.") from exc
    if not isinstance(payload, dict):
        raise AgentWorkspaceError("Worktree ownership marker is invalid.")
    return payload


def _reject_duplicate_fields(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise AgentWorkspaceError("Worktree ownership marker has duplicate fields.")
        result[key] = value
    return result


def _valid_branch(value: str) -> bool:
    return bool(
        _BRANCH.fullmatch(value)
        and ".." not in value
        and "//" not in value
        and not value.endswith(("/", ".", ".lock"))
    )


def _valid_base_ref(value: str) -> bool:
    return bool(
        _BASE_REF.fullmatch(value)
        and ".." not in value
        and "//" not in value
        and "@{" not in value
        and not value.endswith(("/", ".", ".lock"))
    )


def _marker_payload(record: dict, token: str) -> dict:
    return {
        "version": 1,
        "id": record["id"],
        "projectId": record["projectId"],
        "gitRoot": record["gitRoot"],
        "path": record["path"],
        "branch": record["branch"],
        "baseRef": record["baseRef"],
        "createdAt": record["createdAt"],
        "token": token,
    }


def _verify_owned_marker(
    record: dict, *, require_workspace: bool = True
) -> tuple[Path, Path, dict]:
    container, target, marker = _expected_paths(
        record["projectId"], record["id"], create_root = False
    )
    if not _same_path(record["path"], target) or not _same_path(record["markerPath"], marker):
        raise AgentWorkspaceError("Worktree is outside Studio-owned storage.")
    if not _plain_directory(container.parent) or not _plain_directory(container):
        raise AgentWorkspaceError("Worktree ownership storage is invalid.")
    if _lstat(marker) is None:
        raise AgentWorkspaceError("Worktree ownership marker is invalid.")
    payload = _read_marker(marker)
    if payload.get("version") != 1:
        raise AgentWorkspaceError("Worktree ownership marker version is unsupported.")
    required = {
        "id": record["id"],
        "projectId": record["projectId"],
        "gitRoot": record["gitRoot"],
        "path": record["path"],
        "branch": record["branch"],
        "baseRef": record["baseRef"],
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise AgentWorkspaceError("Worktree ownership marker no longer matches.")
    token = payload.get("token")
    if (
        not isinstance(token, str)
        or len(token) < 32
        or not secrets.compare_digest(
            hashlib.sha256(token.encode("utf-8")).hexdigest(), str(record["markerTokenHash"])
        )
    ):
        raise AgentWorkspaceError("Worktree ownership proof is invalid.")
    if require_workspace and not _plain_directory(target):
        raise AgentWorkspaceError("Studio worktree path is unavailable or unsafe.")
    return target, marker, payload


def _registered_entry(entries: dict[str, dict], path: Path) -> Optional[dict]:
    return entries.get(os.path.normcase(os.path.normpath(str(path))))


def _registration_matches(record: dict, entry: Optional[dict]) -> bool:
    return bool(
        entry is not None
        and entry.get("branch") == f"refs/heads/{record['branch']}"
        and not entry.get("detached")
    )


def _mark_attention(record: dict) -> None:
    if record["status"] in {"removed", "needs_attention"}:
        return
    try:
        transition_worktree_status(
            record["id"], {"creating", "active", "removing"}, "needs_attention"
        )
    except AgentWorkspaceError:
        pass


def _settle_creation(record: dict, repository: Path) -> None:
    target = Path(record["path"])
    try:
        entry = _registered_entry(worktree_entries(repository), target)
    except AgentWorkspaceError:
        _mark_attention(record)
        return
    if (
        entry is not None
        or _lstat(target) is not None
        or _lstat(Path(record["markerPath"])) is not None
    ):
        _mark_attention(record)
        return
    try:
        transition_worktree_status(record["id"], {"creating"}, "removed")
    except Exception:
        return
    _remove_empty_container(target.parent)


def create_worktree(
    project_id: str,
    *,
    branch: Optional[str] = None,
    base_ref: str = "HEAD",
) -> dict:
    if os.name == "nt":
        raise AgentWorkspaceError(
            "Secure worktree operations are disabled on Windows until a boundary test passes."
        )
    with _project_operation(project_id):
        workspace = project_workspace(project_id)
        repository = git_root(workspace.root)
        if repository != workspace.root:
            raise AgentWorkspaceError(
                "Worktree creation requires the project to own the repository root."
            )
        branch_name = branch or f"unsloth-studio/task-{uuid.uuid4().hex[:12]}"
        if not _valid_branch(branch_name):
            raise AgentWorkspaceError("Worktree branches must use the unsloth-studio/ namespace.")
        if not _valid_base_ref(base_ref):
            raise AgentWorkspaceError("Invalid worktree base reference.")
        base_ref = repository_ref(repository, base_ref)
        worktree_id = str(uuid.uuid4())
        container, target, marker = _expected_paths(project_id, worktree_id)
        token = secrets.token_urlsafe(32)
        now = _now_ms()
        record = {
            "id": worktree_id,
            "projectId": project_id,
            "gitRoot": str(repository),
            "path": str(target),
            "branch": branch_name,
            "baseRef": base_ref,
            "markerPath": str(marker),
            "markerTokenHash": hashlib.sha256(token.encode("utf-8")).hexdigest(),
            "status": "creating",
            "createdAt": now,
            "updatedAt": now,
        }
        save_worktree(record)
        try:
            _ensure_container(container)
            add_worktree(repository, target, branch_name, base_ref)
            _write_marker(marker, _marker_payload(record, token))
        except Exception as exc:
            try:
                _settle_creation(record, repository)
            except Exception:
                # Preserve the original failure and leave any ambiguous
                # checkout for startup/manual recovery.
                pass
            if isinstance(exc, AgentWorkspaceError):
                raise
            raise AgentWorkspaceError(
                "Worktree creation stopped before ownership was finalized. "
                "Studio preserved any checkout for recovery."
            ) from exc
        try:
            active = transition_worktree_status(worktree_id, {"creating"}, "active")
        except Exception as exc:
            raise AgentWorkspaceError(
                "Worktree creation finished, but durable state recovery is pending."
            ) from exc
        if active is None:
            raise AgentWorkspaceError("Worktree creation finished without durable ownership.")
        return active


def _reconcile_record(record: dict) -> str:
    if record["status"] == "removed" and _lstat(Path(record["markerPath"])) is None:
        return "unchanged"
    try:
        repository = git_root(Path(record["gitRoot"]))
        if not _same_path(record["gitRoot"], repository):
            raise AgentWorkspaceError("Worktree repository identity no longer matches.")
        entries = worktree_entries(repository)
        target = Path(record["path"])
        entry = _registered_entry(entries, target)
        marker_exists = _lstat(Path(record["markerPath"])) is not None
        if not marker_exists:
            if entry is None and _lstat(target) is None:
                changed = transition_worktree_status(
                    record["id"], {"creating", "active", "removing", "needs_attention"}, "removed"
                )
                _remove_empty_container(target.parent)
                return "removed" if changed else "error"
            _mark_attention(record)
            return "attention"
        _verify_owned_marker(record, require_workspace = _lstat(target) is not None)
        if record["status"] == "removed":
            if entry is not None or _lstat(target) is not None:
                return "attention"
            try:
                Path(record["markerPath"]).unlink()
            except OSError:
                return "error"
            _remove_empty_container(target.parent)
            return "removed"
        if entry is not None and _registration_matches(record, entry) and _plain_directory(target):
            if record["status"] in {"creating", "removing"}:
                changed = transition_worktree_status(
                    record["id"], {"creating", "removing"}, "active"
                )
                return "activated" if changed else "error"
            return "unchanged"
        if entry is None and _lstat(target) is None:
            changed = transition_worktree_status(
                record["id"], {"creating", "active", "removing", "needs_attention"}, "removed"
            )
            if changed:
                try:
                    Path(record["markerPath"]).unlink()
                except OSError:
                    pass
                _remove_empty_container(target.parent)
                return "removed"
        _mark_attention(record)
        return "attention"
    except AgentWorkspaceError:
        _mark_attention(record)
        return "attention"


def reconcile_worktrees_on_startup() -> dict[str, int]:
    deadline = time.monotonic() + 10
    result = {"activated": 0, "removed": 0, "attention": 0, "error": 0}
    try:
        records = list_all_worktrees(_RECONCILE_LIMIT)
    except Exception:
        result["error"] += 1
        return result
    for record in records:
        if time.monotonic() >= deadline:
            result["error"] += 1
            break
        try:
            action = _reconcile_record(record)
        except Exception:
            action = "error"
        if action in result:
            result[action] += 1
    return result


def list_project_worktrees(project_id: str) -> list[dict]:
    return list_worktrees(project_id)


def list_worktrees_for_project(project_id: str) -> list[dict]:
    """Compatibility spelling for callers that keep the project in the name."""
    return list_project_worktrees(project_id)


def owned_worktree_path(project_id: str, worktree_id: str) -> Path:
    with _project_operation(project_id):
        record = get_worktree(worktree_id)
        if record is None or record["projectId"] != project_id:
            raise AgentWorkspaceError("Studio worktree not found.")
        if record["status"] != "active":
            raise AgentWorkspaceError("Studio worktree is not active.")
        path, _, _ = _verify_owned_marker(record)
        repository = git_root(Path(record["gitRoot"]))
        entry = _registered_entry(worktree_entries(repository), path)
        if not _registration_matches(record, entry):
            raise AgentWorkspaceError("Studio worktree registration no longer matches.")
        return path


def cleanup_worktree(project_id: str, worktree_id: str) -> dict:
    if os.name == "nt":
        raise AgentWorkspaceError(
            "Secure worktree operations are disabled on Windows until a boundary test passes."
        )
    with _project_operation(project_id), _task_worktree_guard(project_id, worktree_id):
        record = get_worktree(worktree_id)
        if record is None or record["projectId"] != project_id:
            raise AgentWorkspaceError("Studio worktree not found.")
        if record["status"] == "removed":
            if _reconcile_record(record) in {"attention", "error"}:
                raise AgentWorkspaceError("Worktree cleanup needs manual recovery.")
            return record
        if record["status"] != "active":
            _reconcile_record(record)
            record = get_worktree(worktree_id) or record
            if record["status"] != "active":
                raise AgentWorkspaceError(
                    "Studio cannot prove this worktree is safe to remove. Inspect it manually."
                )
        path, marker, _ = _verify_owned_marker(record)
        repository = git_root(Path(record["gitRoot"]))
        if not _registration_matches(record, _registered_entry(worktree_entries(repository), path)):
            raise AgentWorkspaceError("Worktree registration no longer matches Studio ownership.")
        if repository_status(path):
            raise AgentWorkspaceError("The agent worktree has uncommitted changes.")
        ignored, ignored_truncated = repository_command(
            path,
            ["ls-files", "--others", "--ignored", "--exclude-standard", "-z"],
            output_limit = 4096,
        )
        if ignored or ignored_truncated:
            raise AgentWorkspaceError(
                "The worktree contains ignored files; preserve or move them before cleanup."
            )
        try:
            removing = transition_worktree_status(worktree_id, {"active"}, "removing")
        except AgentWorkspaceError:
            raise
        except Exception as exc:
            raise AgentWorkspaceError(
                "Studio could not reserve durable cleanup state. Worktree was left untouched."
            ) from exc
        if removing is None:
            raise AgentWorkspaceError("Studio worktree not found.")
        try:
            remove_worktree(repository, path)
        except Exception:
            try:
                transition_worktree_status(worktree_id, {"removing"}, "active")
            except AgentWorkspaceError:
                pass
            raise
        try:
            removed = transition_worktree_status(worktree_id, {"removing"}, "removed")
        except AgentWorkspaceError:
            raise
        except Exception as exc:
            raise AgentWorkspaceError(
                "Git removed the worktree, but durable cleanup is pending startup recovery."
            ) from exc
        if removed is None:
            raise AgentWorkspaceError(
                "Git removed the worktree, but durable cleanup is pending startup recovery."
            )
        try:
            marker.unlink()
        except OSError:
            pass
        _remove_empty_container(path.parent)
        return removed


def merge_owned_worktree(
    project_id: str,
    worktree_id: str,
    expected_target_head: str,
    *,
    _fenced: bool = False,
) -> dict:
    if os.name == "nt":
        raise AgentWorkspaceError(
            "Secure worktree operations are disabled on Windows until a boundary test passes."
        )
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", expected_target_head):
        raise AgentWorkspaceError("Expected target head is invalid.")
    with (
        _project_operation(project_id),
        nullcontext() if _fenced else _task_worktree_guard(project_id, worktree_id),
    ):
        record = get_worktree(worktree_id)
        if record is None or record["projectId"] != project_id:
            raise AgentWorkspaceError("Studio worktree not found.")
        if record["status"] != "active":
            raise AgentWorkspaceError("Studio worktree is not active.")
        path, _, _ = _verify_owned_marker(record)
        repository = git_root(Path(record["gitRoot"]))
        if not _registration_matches(record, _registered_entry(worktree_entries(repository), path)):
            raise AgentWorkspaceError("Worktree registration no longer matches Studio ownership.")
        if not _fenced:
            # Expected-head validation and the merge itself must share one
            # repository fence. Otherwise another writer can move HEAD after
            # the final check but before Git starts the merge.
            with repository_fence(repository):
                return merge_owned_worktree(
                    project_id,
                    worktree_id,
                    expected_target_head,
                    _fenced = True,
                )
        if repository_head(repository).lower() != expected_target_head.lower():
            raise AgentWorkspaceError("The target branch changed. Refresh and retry.")
        if repository_status(repository):
            raise AgentWorkspaceError("The primary workspace has uncommitted changes.")
        if repository_status(path):
            raise AgentWorkspaceError("The agent worktree has uncommitted changes.")
        target_branch = repository_branch(repository)
        if target_branch == record["branch"]:
            raise AgentWorkspaceError("The primary repository branch is unavailable.")
        source_head = repository_ref(repository, f"refs/heads/{record['branch']}")
        started = _now_ms()
        record_worktree_merge(
            worktree_id,
            {
                "status": "checking",
                "targetBranch": target_branch,
                "expectedTargetHead": expected_target_head,
                "sourceHead": source_head,
                "startedAt": started,
                "completedAt": None,
                "conflicts": [],
            },
        )
        code, output, truncated = preflight_merge(repository, expected_target_head, source_head)
        if code == 1:
            return record_worktree_merge(
                worktree_id,
                {
                    "status": "conflict",
                    "targetBranch": target_branch,
                    "expectedTargetHead": expected_target_head,
                    "sourceHead": source_head,
                    "startedAt": started,
                    "completedAt": _now_ms(),
                    "conflicts": [line for line in output.replace("\0", "\n").splitlines() if line][
                        :200
                    ],
                    "conflictsTruncated": truncated,
                },
            )
        if code != 0 or truncated:
            record_worktree_merge(
                worktree_id,
                {
                    "status": "failed",
                    "targetBranch": target_branch,
                    "expectedTargetHead": expected_target_head,
                    "sourceHead": source_head,
                    "startedAt": started,
                    "completedAt": _now_ms(),
                    "conflicts": [],
                },
            )
            raise AgentWorkspaceError("Git could not evaluate the worktree merge.")
        if repository_head(repository).lower() != expected_target_head.lower() or repository_status(
            repository
        ):
            raise AgentWorkspaceError("The primary workspace changed during merge preflight.")
        try:
            repository_command(
                repository,
                ["merge", "--no-ff", "--no-edit", source_head],
                timeout = 120,
                output_limit = 128 * 1024,
                neutralize_filters = True,
                neutralize_merge_drivers = True,
            )
        except AgentWorkspaceError:
            try:
                conflict_status = repository_status(repository)
                conflict_files, conflict_counts = _status_records(conflict_status)
                conflicts = [
                    item["path"]
                    for item in conflict_files
                    if item["code"] in {"DD", "AU", "UD", "UA", "DU", "AA", "UU"}
                ]
            except AgentWorkspaceError:
                conflicts, conflict_counts = [], {"conflicts": 0}
            merge_aborted = False
            if conflicts or conflict_counts.get("conflicts", 0):
                return record_worktree_merge(
                    worktree_id,
                    {
                        "status": "conflict",
                        "targetBranch": target_branch,
                        "expectedTargetHead": expected_target_head,
                        "sourceHead": source_head,
                        "startedAt": started,
                        "completedAt": _now_ms(),
                        "primaryWorkspaceChanged": not merge_aborted,
                        "conflicts": conflicts[:200],
                    },
                )
            record_worktree_merge(
                worktree_id,
                {
                    "status": "failed",
                    "targetBranch": target_branch,
                    "expectedTargetHead": expected_target_head,
                    "sourceHead": source_head,
                    "startedAt": started,
                    "completedAt": _now_ms(),
                    "primaryWorkspaceChanged": not merge_aborted,
                    "conflicts": [],
                },
            )
            raise
        result_head = repository_head(repository)
        return record_worktree_merge(
            worktree_id,
            {
                "status": "merged",
                "targetBranch": target_branch,
                "expectedTargetHead": expected_target_head,
                "sourceHead": source_head,
                "resultHead": result_head,
                "startedAt": started,
                "completedAt": _now_ms(),
                "conflicts": [],
            },
        )


def merge_worktree(project_id: str, worktree_id: str, expected_target_head: str) -> dict:
    """Public service spelling used by the review route."""
    return merge_owned_worktree(project_id, worktree_id, expected_target_head)


__all__ = [
    "begin_project_deletion",
    "cleanup_worktree",
    "create_worktree",
    "finish_project_deletion",
    "list_project_worktrees",
    "list_worktrees_for_project",
    "merge_worktree",
    "merge_owned_worktree",
    "owned_worktree_path",
    "project_operation",
    "reconcile_worktrees_on_startup",
]
