# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Crash-safe lifecycle for worktrees owned by Unsloth Studio."""

import hashlib
import json
import os
import re
import secrets
import stat
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from utils.paths import ensure_dir, project_workspaces_root

from .common import AgentWorkspaceError, now_ms
from .execution import (
    acquire_workspace_execution_slot,
    release_workspace_execution_slot,
)
from .git_service import (
    _git,
    _neutral_filter_overrides,
    _neutral_merge_driver_overrides,
    _project_git,
    _run_git,
    _serialized_repository_mutation,
)
from .state import (
    bind_background_task_worktree,
    get_background_task,
    get_worktree,
    list_active_background_tasks,
    list_all_worktrees,
    list_worktrees,
    record_worktree_merge,
    release_failed_worktree_task_reservation,
    save_worktree,
    transition_worktree_status,
)


_BRANCH = re.compile(r"^unsloth-studio/[A-Za-z0-9][A-Za-z0-9._/-]{0,120}$")
_STORAGE_KEY = re.compile(r"^[0-9a-f]{32}$")
_MARKER_NAME = "owner.json"
_MARKER_LIMIT = 16 * 1024
_RECONCILE_LIMIT = 1024
_SCAN_ENTRY_LIMIT = 8192
_RECONCILE_BUDGET_SECONDS = 10.0
_LIVE_BACKGROUND_TASK_STATUSES = frozenset({"queued", "running", "cancelling"})

_WORKTREE_CONDITION = threading.Condition()
_PROJECT_WORKTREE_ACTIVE: dict[str, int] = {}
_PROJECTS_DELETING: set[str] = set()


@contextmanager
def _project_worktree_operation(project_id: str) -> Iterator[None]:
    with _WORKTREE_CONDITION:
        if project_id in _PROJECTS_DELETING:
            raise AgentWorkspaceError(
                "Worktree operations are unavailable while the project is being deleted."
            )
        _PROJECT_WORKTREE_ACTIVE[project_id] = _PROJECT_WORKTREE_ACTIVE.get(project_id, 0) + 1
    try:
        yield
    finally:
        with _WORKTREE_CONDITION:
            remaining = _PROJECT_WORKTREE_ACTIVE.get(project_id, 1) - 1
            if remaining > 0:
                _PROJECT_WORKTREE_ACTIVE[project_id] = remaining
            else:
                _PROJECT_WORKTREE_ACTIVE.pop(project_id, None)
            _WORKTREE_CONDITION.notify_all()


def begin_project_deletion(project_id: str, timeout_seconds: float = 15) -> None:
    """Fence new worktree operations and wait for existing operations to settle."""
    deadline = time.monotonic() + max(0.1, timeout_seconds)
    with _WORKTREE_CONDITION:
        if project_id in _PROJECTS_DELETING:
            raise AgentWorkspaceError("Project worktree deletion is already in progress.")
        _PROJECTS_DELETING.add(project_id)
        while _PROJECT_WORKTREE_ACTIVE.get(project_id, 0) > 0:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _PROJECTS_DELETING.discard(project_id)
                _WORKTREE_CONDITION.notify_all()
                raise AgentWorkspaceError(
                    "A worktree operation is still running. Try deleting the project again."
                )
            _WORKTREE_CONDITION.wait(timeout = remaining)


def finish_project_deletion(project_id: str) -> None:
    with _WORKTREE_CONDITION:
        _PROJECTS_DELETING.discard(project_id)
        _WORKTREE_CONDITION.notify_all()


@contextmanager
def _workspace_writer_slots(*roots: Path) -> Iterator[None]:
    """Serialize worktree mutations with managed project command writers."""
    roots_by_identity: dict[tuple[int, int], Path] = {}
    for root in roots:
        try:
            metadata = root.stat()
        except OSError as exc:
            raise AgentWorkspaceError("A worktree workspace is unavailable.") from exc
        if not stat.S_ISDIR(metadata.st_mode):
            raise AgentWorkspaceError("A worktree workspace is not a directory.")
        identity = (int(metadata.st_dev), int(metadata.st_ino))
        roots_by_identity.setdefault(identity, root)
    identities = sorted(roots_by_identity)
    acquired: list[tuple[int, int]] = []
    try:
        for identity in identities:
            if not acquire_workspace_execution_slot(identity):
                raise AgentWorkspaceError("The worktree operation was cancelled before it started.")
            acquired.append(identity)
        for identity, root in roots_by_identity.items():
            try:
                metadata = root.stat()
            except OSError as exc:
                raise AgentWorkspaceError("A worktree workspace changed before use.") from exc
            if (int(metadata.st_dev), int(metadata.st_ino)) != identity:
                raise AgentWorkspaceError("A worktree workspace changed before use.")
        yield
    finally:
        for identity in reversed(acquired):
            release_workspace_execution_slot(identity)


def _lstat(path: Path) -> Optional[os.stat_result]:
    try:
        return path.lstat()
    except FileNotFoundError:
        return None


def _plain_directory(path: Path) -> bool:
    metadata = _lstat(path)
    return metadata is not None and stat.S_ISDIR(metadata.st_mode)


def _worktree_root(*, create: bool = True) -> Path:
    base = Path(project_workspaces_root())
    if create:
        ensure_dir(base)
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
    if metadata is None or not stat.S_ISDIR(metadata.st_mode):
        raise AgentWorkspaceError("Studio worktree storage is not a safe directory.")
    try:
        resolved = candidate.resolve(strict = True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise AgentWorkspaceError("Studio worktree storage is unavailable.") from exc
    if resolved.parent != base:
        raise AgentWorkspaceError("Studio worktree storage cannot be a symbolic link.")
    return resolved


def _project_storage_key(project_id: str) -> str:
    return hashlib.sha256(project_id.encode("utf-8")).hexdigest()[:32]


def _expected_paths(
    project_id: str,
    worktree_id: str,
    *,
    create_root: bool = True,
) -> tuple[Path, Path, Path]:
    root = _worktree_root(create = create_root)
    container = root / _project_storage_key(project_id) / worktree_id
    return container, container / "workspace", container / _MARKER_NAME


def _same_path(left: str | Path, right: str | Path) -> bool:
    return os.path.normcase(os.path.abspath(os.fspath(left))) == os.path.normcase(
        os.path.abspath(os.fspath(right))
    )


def _ensure_container(container: Path) -> None:
    root = _worktree_root()
    try:
        relative = container.relative_to(root)
    except ValueError as exc:
        raise AgentWorkspaceError("Worktree destination is outside Studio storage.") from exc
    if len(relative.parts) != 2 or not _STORAGE_KEY.fullmatch(relative.parts[0]):
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
    if metadata is None or not stat.S_ISDIR(metadata.st_mode):
        raise AgentWorkspaceError("Worktree storage contains an unsafe path.")
    try:
        container.mkdir(mode = 0o700)
    except FileExistsError as exc:
        raise AgentWorkspaceError("Worktree destination already exists.") from exc
    except OSError as exc:
        raise AgentWorkspaceError("Worktree destination could not be prepared.") from exc


def _remove_empty_container(container: Path) -> None:
    """Remove only empty generated directories. Never recurse through unknown content."""
    try:
        container.rmdir()
    except OSError:
        return
    try:
        container.parent.rmdir()
    except OSError:
        pass


def _write_marker(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    data = json.dumps(payload, sort_keys = True, separators = (",", ":"))
    if len(data.encode("utf-8")) > _MARKER_LIMIT:
        raise AgentWorkspaceError("Worktree ownership marker is too large.")
    try:
        with temporary.open("x", encoding = "utf-8") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path, follow_symlinks = False)
        try:
            descriptor = os.open(path.parent, os.O_RDONLY)
        except OSError:
            descriptor = None
        if descriptor is not None:
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _replace_marker(path: Path, payload: dict) -> None:
    """Atomically update a proven marker inside Studio-owned storage."""
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    data = json.dumps(payload, sort_keys = True, separators = (",", ":"))
    if len(data.encode("utf-8")) > _MARKER_LIMIT:
        raise AgentWorkspaceError("Worktree ownership marker is too large.")
    try:
        with temporary.open("x", encoding = "utf-8") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        try:
            descriptor = os.open(path.parent, os.O_RDONLY)
        except OSError:
            descriptor = None
        if descriptor is not None:
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_marker(path: Path) -> dict:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AgentWorkspaceError("Worktree ownership marker is invalid.") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > _MARKER_LIMIT:
            raise AgentWorkspaceError("Worktree ownership marker is invalid.")
        raw = bytearray()
        while len(raw) <= _MARKER_LIMIT:
            chunk = os.read(descriptor, min(4096, _MARKER_LIMIT + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
    finally:
        os.close(descriptor)
    if len(raw) > _MARKER_LIMIT:
        raise AgentWorkspaceError("Worktree ownership marker is invalid.")
    try:
        payload = json.loads(bytes(raw).decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise AgentWorkspaceError("Worktree ownership marker is invalid.") from exc
    if not isinstance(payload, dict):
        raise AgentWorkspaceError("Worktree ownership marker is invalid.")
    return payload


def _valid_branch(branch: str) -> bool:
    return bool(
        _BRANCH.fullmatch(branch)
        and ".." not in branch
        and "//" not in branch
        and not branch.endswith("/")
    )


def _valid_base_ref(base_ref: str) -> bool:
    return bool(
        base_ref
        and len(base_ref) <= 256
        and not base_ref.startswith("-")
        and not any(character in base_ref for character in "\x00\r\n")
    )


def _marker_payload(record: dict, token: str) -> dict:
    return {
        "version": 2,
        "id": record["id"],
        "projectId": record["projectId"],
        "gitRoot": record["gitRoot"],
        "path": record["path"],
        "branch": record["branch"],
        "baseRef": record["baseRef"],
        "backgroundTaskId": record.get("backgroundTaskId"),
        "createdAt": record["createdAt"],
        "token": token,
    }


def _record_from_marker(marker: Path, payload: dict) -> dict:
    if payload.get("version") not in {1, 2}:
        raise AgentWorkspaceError("Worktree ownership marker version is unsupported.")
    worktree_id = str(payload.get("id") or "")
    project_id = str(payload.get("projectId") or "")
    try:
        if str(uuid.UUID(worktree_id)) != worktree_id:
            raise ValueError
    except ValueError as exc:
        raise AgentWorkspaceError("Worktree ownership marker is invalid.") from exc
    if (
        not project_id
        or len(project_id) > 512
        or any(character in project_id for character in "\x00\r\n")
    ):
        raise AgentWorkspaceError("Worktree ownership marker is invalid.")
    container, target, expected_marker = _expected_paths(project_id, worktree_id, create_root = False)
    if marker != expected_marker or marker.parent != container:
        raise AgentWorkspaceError("Worktree ownership marker is misplaced.")
    branch = str(payload.get("branch") or "")
    base_ref = str(payload.get("baseRef") or "HEAD")
    git_root_value = str(payload.get("gitRoot") or "")
    path_value = str(payload.get("path") or "")
    token = str(payload.get("token") or "")
    if (
        not _valid_branch(branch)
        or not _valid_base_ref(base_ref)
        or not git_root_value
        or not _same_path(path_value, target)
        or len(token) < 32
        or len(token) > 256
    ):
        raise AgentWorkspaceError("Worktree ownership marker is invalid.")
    background_task_id = payload.get("backgroundTaskId")
    if background_task_id is not None:
        background_task_id = str(background_task_id)
        task = get_background_task(background_task_id)
        if task is None or task["projectId"] != project_id:
            background_task_id = None
    created_at = payload.get("createdAt")
    if not isinstance(created_at, int) or created_at <= 0:
        try:
            created_at = int(marker.stat().st_mtime * 1000)
        except OSError:
            created_at = now_ms()
    return {
        "id": worktree_id,
        "projectId": project_id,
        "gitRoot": git_root_value,
        "path": str(target),
        "branch": branch,
        "baseRef": base_ref,
        "markerPath": str(marker),
        "markerTokenHash": hashlib.sha256(token.encode("utf-8")).hexdigest(),
        "backgroundTaskId": background_task_id,
        "status": "creating",
        "createdAt": created_at,
        "updatedAt": now_ms(),
    }


def _verify_owned_marker(
    record: dict, *, require_workspace: bool = True
) -> tuple[Path, Path, dict]:
    container, target, marker = _expected_paths(
        record["projectId"], record["id"], create_root = False
    )
    if not _same_path(record["path"], target) or not _same_path(record["markerPath"], marker):
        raise AgentWorkspaceError("Worktree is outside Studio-owned storage.")
    for directory in (container.parent, container):
        if not _plain_directory(directory):
            raise AgentWorkspaceError("Worktree ownership storage is invalid.")
    marker_metadata = _lstat(marker)
    if marker_metadata is None or not stat.S_ISREG(marker_metadata.st_mode):
        raise AgentWorkspaceError("Worktree ownership marker is invalid.")
    payload = _read_marker(marker)
    required = {
        "id": record["id"],
        "projectId": record["projectId"],
        "gitRoot": record["gitRoot"],
        "path": record["path"],
        "branch": record["branch"],
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise AgentWorkspaceError("Worktree ownership marker no longer matches.")
    if "baseRef" in payload and payload.get("baseRef") != record["baseRef"]:
        raise AgentWorkspaceError("Worktree ownership marker no longer matches.")
    token = str(payload.get("token") or "")
    if hashlib.sha256(token.encode("utf-8")).hexdigest() != record["markerTokenHash"]:
        raise AgentWorkspaceError("Worktree ownership proof is invalid.")
    if require_workspace and not _plain_directory(target):
        raise AgentWorkspaceError("Studio worktree path is unavailable or unsafe.")
    return target, marker, payload


def _reconcile_background_task_link(record: dict, marker_payload: dict) -> bool:
    """Complete a crash-interrupted task link only while the task is still queued."""
    task_id = record.get("backgroundTaskId")
    if marker_payload.get("backgroundTaskId") != task_id:
        return False
    if task_id is None:
        return True
    task = get_background_task(str(task_id))
    if task is None or task["projectId"] != record["projectId"]:
        return False
    if task.get("worktreeId") == record["id"]:
        return True
    if task.get("worktreeId") is not None or task["status"] != "queued":
        return False
    try:
        bind_background_task_worktree(task["id"], record["id"])
    except AgentWorkspaceError:
        return False
    return True


def _require_linked_task_stopped(record: dict, marker_payload: dict) -> None:
    """Fail closed before mutating a worktree still owned by a live task."""
    task_id = record.get("backgroundTaskId")
    if marker_payload.get("backgroundTaskId") != task_id:
        raise AgentWorkspaceError("Worktree task ownership metadata no longer matches.")
    if task_id is not None:
        task = get_background_task(str(task_id))
        if (
            task is None
            or task["projectId"] != record["projectId"]
            or task.get("worktreeId") != record["id"]
        ):
            raise AgentWorkspaceError("Worktree task ownership metadata is invalid.")
        if task["status"] in _LIVE_BACKGROUND_TASK_STATUSES:
            raise AgentWorkspaceError(
                "Stop the linked background task before merging or removing its worktree."
            )

    # Verification jobs can target an owned worktree without occupying its
    # single-agent marker. Check the task-side link as well so cleanup and merge
    # cannot race either kind of managed writer.
    for active_task in list_active_background_tasks(record["projectId"]):
        if active_task.get("worktreeId") == record["id"]:
            raise AgentWorkspaceError(
                "Stop the linked background task before merging or removing its worktree."
            )


def _worktree_entries(repository: Path, *, timeout_seconds: float = 15) -> dict[str, dict]:
    listing, _ = _git(
        repository,
        ["worktree", "list", "--porcelain", "-z"],
        output_limit = 512_000,
        timeout_seconds = timeout_seconds,
    )
    entries: dict[str, dict] = {}
    current: dict[str, str | bool] = {}
    for field in listing.split("\0"):
        if not field:
            path = current.get("path")
            if isinstance(path, str):
                entries[os.path.normcase(os.path.normpath(path))] = dict(current)
            current = {}
            continue
        if field.startswith("worktree "):
            current["path"] = field[len("worktree ") :]
        elif field.startswith("branch "):
            current["branch"] = field[len("branch ") :]
        elif field == "detached":
            current["detached"] = True
    return entries


def _registered_entry(entries: dict[str, dict], path: Path) -> Optional[dict]:
    return entries.get(os.path.normcase(os.path.normpath(str(path))))


def _registration_matches(record: dict, entry: Optional[dict]) -> bool:
    return bool(
        entry is not None
        and entry.get("branch") == f"refs/heads/{record['branch']}"
        and not entry.get("detached")
    )


def _safe_transition(worktree_id: str, expected: set[str], status: str) -> Optional[dict]:
    try:
        return transition_worktree_status(worktree_id, expected, status)
    except Exception:  # noqa: BLE001 - recovery must preserve the filesystem first
        return None


def _mark_attention(record: dict) -> None:
    if record["status"] not in {"removed", "needs_attention"}:
        _safe_transition(
            record["id"],
            {"creating", "active", "removing"},
            "needs_attention",
        )


def _settle_failed_creation(record: dict, repository: Path) -> None:
    target = Path(record["path"])
    container = target.parent
    try:
        entry = _registered_entry(_worktree_entries(repository), target)
    except Exception:  # noqa: BLE001 - uncertainty must retain the worktree
        _mark_attention(record)
        return
    if entry is not None or _lstat(target) is not None or _lstat(Path(record["markerPath"])):
        _mark_attention(record)
        return
    removed = _safe_transition(record["id"], {"creating"}, "removed")
    if removed is not None and record.get("backgroundTaskId") is not None:
        try:
            release_failed_worktree_task_reservation(str(record["backgroundTaskId"]), record["id"])
        except AgentWorkspaceError:
            _mark_attention(removed)
            return
    _remove_empty_container(container)


def create_worktree(
    project_id: str,
    *,
    branch: Optional[str] = None,
    base_ref: str = "HEAD",
    background_task_id: Optional[str] = None,
) -> dict:
    with _project_worktree_operation(project_id):
        _, repository = _project_git(project_id, mutation = True)
        with _workspace_writer_slots(repository):
            return _create_worktree(
                project_id,
                branch = branch,
                base_ref = base_ref,
                background_task_id = background_task_id,
            )


def _create_worktree(
    project_id: str, *, branch: Optional[str], base_ref: str, background_task_id: Optional[str]
) -> dict:
    _, repository = _project_git(project_id, mutation = True)
    worktree_id = str(uuid.uuid4())
    branch_name = branch or f"unsloth-studio/task-{worktree_id[:12]}"
    if not _valid_branch(branch_name):
        raise AgentWorkspaceError("Worktree branches must use the unsloth-studio/ namespace.")
    if not _valid_base_ref(base_ref):
        raise AgentWorkspaceError("Invalid worktree base reference.")
    if background_task_id:
        task = get_background_task(background_task_id)
        if task is None or task["projectId"] != project_id:
            raise AgentWorkspaceError("Background task does not belong to this project.")
        if task["status"] != "queued":
            raise AgentWorkspaceError("Only a queued background task can be linked to a worktree.")
        if task.get("worktreeId") is not None:
            raise AgentWorkspaceError("Background task is already linked to another worktree.")

    container, target, marker = _expected_paths(project_id, worktree_id)
    token = secrets.token_urlsafe(32)
    current = now_ms()
    record = {
        "id": worktree_id,
        "projectId": project_id,
        "gitRoot": str(repository),
        "path": str(target),
        "branch": branch_name,
        "baseRef": base_ref,
        "markerPath": str(marker),
        "markerTokenHash": hashlib.sha256(token.encode("utf-8")).hexdigest(),
        "backgroundTaskId": background_task_id,
        "status": "creating",
        "createdAt": current,
        "updatedAt": current,
    }
    try:
        save_worktree(record)
    except Exception as exc:
        raise AgentWorkspaceError("Worktree creation could not reserve durable state.") from exc

    with _serialized_repository_mutation(repository):
        try:
            _git(
                repository,
                ["rev-parse", "--verify", f"{base_ref}^{{commit}}"],
                output_limit = 256,
            )
            code, _, _ = _run_git(
                repository,
                ["show-ref", "--verify", "--quiet", f"refs/heads/{branch_name}"],
                timeout_seconds = 5,
                output_limit = 1024,
            )
            if code == 0:
                raise AgentWorkspaceError("The requested worktree branch already exists.")
            _ensure_container(container)
            _git(
                repository,
                ["worktree", "add", "-b", branch_name, str(target), base_ref],
                timeout_seconds = 120,
                output_limit = 128_000,
                neutralize_filters = True,
            )
            _write_marker(marker, _marker_payload(record, token))
        except Exception as exc:
            _settle_failed_creation(record, repository)
            if isinstance(exc, AgentWorkspaceError):
                raise
            raise AgentWorkspaceError(
                "Worktree creation stopped before ownership could be finalized. "
                "Studio preserved any checkout for recovery."
            ) from exc
        try:
            active = transition_worktree_status(worktree_id, {"creating"}, "active")
        except Exception as exc:
            raise AgentWorkspaceError(
                "Worktree creation finished, but durable state could not be finalized. "
                "Studio preserved the checkout for startup recovery."
            ) from exc
        if active is None:
            raise AgentWorkspaceError(
                "Worktree creation finished without a durable ownership record. "
                "Studio preserved the checkout for startup recovery."
            )
        if background_task_id:
            task = get_background_task(background_task_id)
            if (
                task is None
                or task.get("worktreeId") != worktree_id
                or active.get("backgroundTaskId") != background_task_id
            ):
                _safe_transition(worktree_id, {"active"}, "needs_attention")
                raise AgentWorkspaceError(
                    "Worktree creation finished without its durable task reservation. "
                    "Studio preserved the checkout for startup recovery."
                )
        return active


def _resolve_record_repository(record: dict) -> Path:
    _, repository = _project_git(record["projectId"], mutation = True)
    if not _same_path(record["gitRoot"], repository):
        raise AgentWorkspaceError("Worktree repository identity no longer matches.")
    return repository


def _cached_record_repository(
    record: dict, repository_cache: Optional[dict[str, Optional[Path]]]
) -> Path:
    if repository_cache is None:
        return _resolve_record_repository(record)
    project_id = record["projectId"]
    if project_id in repository_cache:
        repository = repository_cache[project_id]
        if repository is None:
            raise AgentWorkspaceError("Worktree repository is unavailable.")
        if not _same_path(record["gitRoot"], repository):
            raise AgentWorkspaceError("Worktree repository identity no longer matches.")
        return repository
    try:
        repository = _resolve_record_repository(record)
    except Exception:
        repository_cache[project_id] = None
        raise
    repository_cache[project_id] = repository
    return repository


def _retire_marker(record: dict) -> None:
    marker_path = Path(record["markerPath"])
    if _lstat(marker_path) is None:
        return
    target, marker, _ = _verify_owned_marker(record, require_workspace = False)
    if _lstat(target) is not None:
        return
    try:
        marker.unlink()
    except OSError:
        return
    _remove_empty_container(marker.parent)


def _reconcile_record(
    record: dict,
    entries_cache: Optional[dict[str, dict[str, dict]]] = None,
    repository_cache: Optional[dict[str, Optional[Path]]] = None,
) -> str:
    if record["status"] == "removed" and _lstat(Path(record["markerPath"])) is None:
        return "unchanged"
    try:
        repository = _cached_record_repository(record, repository_cache)
    except Exception:  # noqa: BLE001 - unavailable projects remain blocked, not deleted
        _mark_attention(record)
        return "attention"
    target = Path(record["path"])
    try:
        cache_key = os.path.normcase(str(repository))
        if entries_cache is not None and cache_key in entries_cache:
            entries = entries_cache[cache_key]
        else:
            entries = _worktree_entries(
                repository, timeout_seconds = 3 if entries_cache is not None else 15
            )
            if entries_cache is not None:
                entries_cache[cache_key] = entries
    except Exception:  # noqa: BLE001
        _mark_attention(record)
        return "attention"
    entry = _registered_entry(entries, target)
    marker_exists = _lstat(Path(record["markerPath"])) is not None
    if not marker_exists:
        if entry is None and _lstat(target) is None:
            settled = _safe_transition(
                record["id"],
                {"creating", "active", "removing", "needs_attention"},
                "removed",
            )
            if settled is not None:
                task_id = settled.get("backgroundTaskId")
                if task_id is not None:
                    try:
                        release_failed_worktree_task_reservation(str(task_id), settled["id"])
                    except AgentWorkspaceError:
                        return "attention"
                _remove_empty_container(target.parent)
            return "removed" if settled is not None else "error"
        _mark_attention(record)
        return "attention"
    try:
        _, _, marker_payload = _verify_owned_marker(
            record, require_workspace = _lstat(target) is not None
        )
    except AgentWorkspaceError:
        _mark_attention(record)
        return "attention"
    if entry is not None:
        if not _plain_directory(target) or not _registration_matches(record, entry):
            _mark_attention(record)
            return "attention"
        if record["status"] == "removed":
            _safe_transition(record["id"], {"removed"}, "needs_attention")
            return "attention"
        action = "unchanged"
        if record["status"] != "active":
            active = _safe_transition(
                record["id"],
                {"creating", "removing", "needs_attention"},
                "active",
            )
            if active is None:
                return "error"
            record = active
            action = "activated"
        if not _reconcile_background_task_link(record, marker_payload):
            _mark_attention(record)
            return "attention"
        return action
    if _lstat(target) is not None:
        _mark_attention(record)
        return "attention"
    removed = _safe_transition(
        record["id"],
        {"creating", "active", "removing", "needs_attention", "removed"},
        "removed",
    )
    if removed is None:
        return "error"
    _retire_marker(removed)
    return "removed"


def _candidate_markers(root: Path) -> tuple[list[Path], bool]:
    if _lstat(root) is None:
        return [], False
    markers: list[Path] = []
    scanned = 0
    try:
        buckets = root.iterdir()
    except OSError:
        return [], False
    for bucket in buckets:
        scanned += 1
        if scanned > _SCAN_ENTRY_LIMIT:
            return markers, True
        if not _STORAGE_KEY.fullmatch(bucket.name) or not _plain_directory(bucket):
            continue
        try:
            containers = bucket.iterdir()
        except OSError:
            continue
        for container in containers:
            scanned += 1
            if scanned > _SCAN_ENTRY_LIMIT:
                return markers, True
            try:
                if str(uuid.UUID(container.name)) != container.name:
                    continue
            except ValueError:
                continue
            if not _plain_directory(container):
                continue
            marker = container / _MARKER_NAME
            metadata = _lstat(marker)
            if metadata is not None and stat.S_ISREG(metadata.st_mode):
                markers.append(marker)
                if len(markers) >= _RECONCILE_LIMIT:
                    return markers, True
    return markers, False


def _import_marker(
    marker: Path,
    entries_cache: Optional[dict[str, dict[str, dict]]] = None,
    repository_cache: Optional[dict[str, Optional[Path]]] = None,
) -> str:
    payload = _read_marker(marker)
    record = _record_from_marker(marker, payload)
    existing = get_worktree(record["id"])
    if existing is not None:
        return _reconcile_record(existing, entries_cache, repository_cache)
    # The marker contains its own token, so it is not independent proof of
    # installation ownership. Without the matching SQLite row and token hash,
    # startup leaves the checkout untouched for explicit manual recovery.
    return "attention"


def reconcile_worktrees_on_startup() -> dict:
    """Settle DB-proven worktrees and retain orphan markers without mutation."""
    result = {
        "scanned": 0,
        "imported": 0,
        "activated": 0,
        "removed": 0,
        "attention": 0,
        "errors": 0,
        "truncated": False,
    }
    try:
        deadline = time.monotonic() + _RECONCILE_BUDGET_SECONDS
        root = _worktree_root(create = False)
        markers, truncated = _candidate_markers(root)
        result["truncated"] = truncated
        marker_ids: set[str] = set()
        entries_cache: dict[str, dict[str, dict]] = {}
        repository_cache: dict[str, Optional[Path]] = {}
        for marker in markers:
            if time.monotonic() >= deadline:
                result["truncated"] = True
                break
            result["scanned"] += 1
            try:
                payload = _read_marker(marker)
                _record_from_marker(marker, payload)
                marker_id = str(payload.get("id") or "")
                if marker_id in marker_ids:
                    result["errors"] += 1
                    continue
                action = _import_marker(marker, entries_cache, repository_cache)
            except Exception:  # noqa: BLE001 - foreign or corrupt markers stay untouched
                action = "error"
            if action != "error":
                marker_ids.add(marker_id)
            if action in result:
                result[action] += 1
            elif action != "unchanged":
                result["errors"] += 1
        rows = list_all_worktrees(_RECONCILE_LIMIT + 1)
        if len(rows) > _RECONCILE_LIMIT:
            result["truncated"] = True
        for record in rows[:_RECONCILE_LIMIT]:
            if time.monotonic() >= deadline:
                result["truncated"] = True
                break
            if record["id"] in marker_ids:
                continue
            try:
                action = _reconcile_record(record, entries_cache, repository_cache)
            except Exception:  # noqa: BLE001 - no startup failure may destroy a checkout
                action = "error"
            if action in result:
                result[action] += 1
            elif action != "unchanged":
                result["errors"] += 1
    except Exception:  # noqa: BLE001 - startup remains available with worktrees blocked
        result["errors"] += 1
    return result


def owned_worktree_path(
    project_id: str,
    worktree_id: str,
    *,
    background_task_id: Optional[str] = None,
) -> Path:
    record = get_worktree(worktree_id)
    if record is None or record["projectId"] != project_id:
        raise AgentWorkspaceError("Studio worktree not found.")
    if record["status"] != "active":
        raise AgentWorkspaceError("Studio worktree is not active.")
    if background_task_id is not None:
        task = get_background_task(background_task_id)
        if (
            task is None
            or task["kind"] != "agent"
            or task["projectId"] != project_id
            or task.get("worktreeId") != worktree_id
            or record.get("backgroundTaskId") != background_task_id
        ):
            raise AgentWorkspaceError("Task and worktree linkage is invalid.")
    path, _, marker_payload = _verify_owned_marker(record)
    if (
        background_task_id is not None
        and marker_payload.get("backgroundTaskId") != background_task_id
    ):
        raise AgentWorkspaceError("Worktree marker belongs to another task.")
    repository = _resolve_record_repository(record)
    entry = _registered_entry(_worktree_entries(repository), path)
    if not _registration_matches(record, entry):
        raise AgentWorkspaceError("Studio worktree registration no longer matches.")
    return path


def sync_worktree_background_task_marker(
    project_id: str, worktree_id: str, background_task_id: str
) -> dict:
    """Persist a database task link into the owned recovery marker."""
    with _project_worktree_operation(project_id):
        record = get_worktree(worktree_id)
        task = get_background_task(background_task_id)
        if (
            record is None
            or record["projectId"] != project_id
            or task is None
            or task["projectId"] != project_id
        ):
            raise AgentWorkspaceError("Task and worktree linkage is invalid.")
        if record["status"] != "active" or record.get("backgroundTaskId") != background_task_id:
            raise AgentWorkspaceError("Task and worktree linkage is not active.")
        _, marker, payload = _verify_owned_marker(record)
        existing = payload.get("backgroundTaskId")
        if existing not in {None, background_task_id, task.get("parentTaskId")}:
            previous = get_background_task(str(existing))
            if (
                previous is None
                or previous["projectId"] != project_id
                or previous.get("worktreeId") != worktree_id
            ):
                raise AgentWorkspaceError("Worktree marker belongs to another task.")
        payload["backgroundTaskId"] = background_task_id
        _replace_marker(marker, payload)
        return get_worktree(worktree_id) or record


def _merge_conflict_paths(output: str) -> list[str]:
    paths: list[str] = []
    for line in output.replace("\x00", "\n").splitlines():
        value = line.strip()
        if (
            not value
            or re.fullmatch(r"[0-9a-fA-F]{40,64}", value)
            or value.startswith(("Auto-merging ", "CONFLICT ", "hint:"))
            or value.startswith(("warning:", "fatal:", "error:"))
        ):
            continue
        if os.path.isabs(value) or "\x00" in value or len(value) > 4096:
            continue
        paths.append(value)
        if len(paths) >= 200:
            break
    return paths


def merge_worktree(project_id: str, worktree_id: str, expected_target_head: str) -> dict:
    """Serialize merge inspection with managed writers for both workspace roots."""
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", expected_target_head):
        raise AgentWorkspaceError("Expected target head is invalid.")
    record = get_worktree(worktree_id)
    if record is None or record["projectId"] != project_id:
        raise AgentWorkspaceError("Studio worktree not found.")
    if record["status"] != "active":
        raise AgentWorkspaceError("Studio worktree is not active.")
    repository = _resolve_record_repository(record)
    path, _, marker_payload = _verify_owned_marker(record)
    _require_linked_task_stopped(record, marker_payload)
    with _workspace_writer_slots(repository, path):
        return _merge_worktree_after_writer_slots(project_id, worktree_id, expected_target_head)


def _merge_worktree_after_writer_slots(
    project_id: str, worktree_id: str, expected_target_head: str
) -> dict:
    """Merge a clean owned branch without reset, clean, force, or hidden fallback."""
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", expected_target_head):
        raise AgentWorkspaceError("Expected target head is invalid.")
    with _project_worktree_operation(project_id):
        record = get_worktree(worktree_id)
        if record is None or record["projectId"] != project_id:
            raise AgentWorkspaceError("Studio worktree not found.")
        if record["status"] != "active":
            raise AgentWorkspaceError("Studio worktree is not active.")
        repository = _resolve_record_repository(record)
        with _serialized_repository_mutation(repository):
            path, _, marker_payload = _verify_owned_marker(record)
            _require_linked_task_stopped(record, marker_payload)
            entry = _registered_entry(_worktree_entries(repository), path)
            if not _registration_matches(record, entry):
                raise AgentWorkspaceError(
                    "Worktree registration or branch no longer matches Studio ownership."
                )

            target_branch, _ = _git(
                repository,
                ["symbolic-ref", "--quiet", "--short", "HEAD"],
                output_limit = 1024,
            )
            target_branch = target_branch.strip()
            if not target_branch or target_branch == record["branch"]:
                raise AgentWorkspaceError("The primary repository branch is unavailable.")
            target_head, _ = _git(repository, ["rev-parse", "HEAD"], output_limit = 256)
            target_head = target_head.strip()
            if target_head.lower() != expected_target_head.lower():
                raise AgentWorkspaceError(
                    "The target branch changed. Refresh the expected head and retry."
                )
            source_head, _ = _git(
                repository,
                ["rev-parse", f"refs/heads/{record['branch']}"],
                output_limit = 256,
            )
            source_head = source_head.strip()
            primary_status, _ = _git(
                repository,
                ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
                output_limit = 512_000,
                neutralize_filters = True,
            )
            worktree_status, _ = _git(
                path,
                ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
                output_limit = 512_000,
                neutralize_filters = True,
            )
            if primary_status:
                raise AgentWorkspaceError(
                    "The primary workspace has uncommitted changes. Merge was not started."
                )
            if worktree_status:
                raise AgentWorkspaceError(
                    "The agent worktree has uncommitted changes. Commit them before merging."
                )

            started = now_ms()
            record_worktree_merge(
                worktree_id,
                {
                    "status": "checking",
                    "targetBranch": target_branch,
                    "expectedTargetHead": target_head,
                    "sourceHead": source_head,
                    "startedAt": started,
                    "completedAt": None,
                    "primaryWorkspaceChanged": False,
                    "conflicts": [],
                },
            )
            merge_overrides = _neutral_filter_overrides(repository, None)
            merge_overrides.extend(_neutral_merge_driver_overrides(repository, None))
            code, preflight, truncated = _run_git(
                repository,
                [
                    "merge-tree",
                    "--write-tree",
                    "--name-only",
                    target_head,
                    source_head,
                ],
                timeout_seconds = 60,
                output_limit = 128_000,
                config_overrides = merge_overrides,
            )
            if code == 1:
                return record_worktree_merge(
                    worktree_id,
                    {
                        "status": "conflict",
                        "targetBranch": target_branch,
                        "expectedTargetHead": target_head,
                        "sourceHead": source_head,
                        "resultHead": None,
                        "startedAt": started,
                        "completedAt": now_ms(),
                        "primaryWorkspaceChanged": False,
                        "conflicts": _merge_conflict_paths(preflight),
                        "conflictsTruncated": truncated,
                    },
                )
            if code != 0:
                record_worktree_merge(
                    worktree_id,
                    {
                        "status": "failed",
                        "targetBranch": target_branch,
                        "expectedTargetHead": target_head,
                        "sourceHead": source_head,
                        "resultHead": None,
                        "startedAt": started,
                        "completedAt": now_ms(),
                        "primaryWorkspaceChanged": False,
                        "conflicts": [],
                    },
                )
                raise AgentWorkspaceError("Git could not evaluate the worktree merge.")

            # Recheck both mutable inputs immediately before Git touches the primary
            # workspace. An unexpected real conflict is retained for explicit
            # resolution. Studio never invokes reset, clean, abort, or force.
            current_head, _ = _git(repository, ["rev-parse", "HEAD"], output_limit = 256)
            current_status, _ = _git(
                repository,
                ["status", "--porcelain=v1", "-z", "--untracked-files=all"],
                output_limit = 512_000,
                neutralize_filters = True,
            )
            if current_head.strip() != target_head or current_status:
                raise AgentWorkspaceError(
                    "The primary workspace changed during merge preflight. Merge was not started."
                )
            try:
                _git(
                    repository,
                    ["merge", "--no-ff", "--no-edit", source_head],
                    timeout_seconds = 120,
                    output_limit = 128_000,
                    neutralize_filters = True,
                    neutralize_merge_drivers = True,
                )
            except AgentWorkspaceError as exc:
                conflicts, conflicts_truncated = _git(
                    repository,
                    ["diff", "--name-only", "--diff-filter=U", "-z"],
                    output_limit = 128_000,
                    neutralize_filters = True,
                )
                conflict_paths = [
                    value for value in conflicts.split("\x00") if value and not os.path.isabs(value)
                ][:200]
                if conflict_paths:
                    return record_worktree_merge(
                        worktree_id,
                        {
                            "status": "conflict",
                            "targetBranch": target_branch,
                            "expectedTargetHead": target_head,
                            "sourceHead": source_head,
                            "resultHead": None,
                            "startedAt": started,
                            "completedAt": now_ms(),
                            "primaryWorkspaceChanged": True,
                            "conflicts": conflict_paths,
                            "conflictsTruncated": conflicts_truncated,
                        },
                    )
                record_worktree_merge(
                    worktree_id,
                    {
                        "status": "failed",
                        "targetBranch": target_branch,
                        "expectedTargetHead": target_head,
                        "sourceHead": source_head,
                        "resultHead": None,
                        "startedAt": started,
                        "completedAt": now_ms(),
                        "primaryWorkspaceChanged": False,
                        "conflicts": [],
                    },
                )
                raise AgentWorkspaceError("Git could not merge the owned worktree.") from exc

            result_head, _ = _git(repository, ["rev-parse", "HEAD"], output_limit = 256)
            return record_worktree_merge(
                worktree_id,
                {
                    "status": "merged",
                    "targetBranch": target_branch,
                    "expectedTargetHead": target_head,
                    "sourceHead": source_head,
                    "resultHead": result_head.strip(),
                    "startedAt": started,
                    "completedAt": now_ms(),
                    "primaryWorkspaceChanged": True,
                    "conflicts": [],
                },
            )


def cleanup_worktree(project_id: str, worktree_id: str) -> dict:
    with _project_worktree_operation(project_id):
        record = get_worktree(worktree_id)
        if record is None or record["projectId"] != project_id:
            raise AgentWorkspaceError("Studio worktree not found.")
        if record["status"] == "removed":
            return _cleanup_worktree(project_id, worktree_id)
        if record["status"] != "active":
            _reconcile_record(record)
            record = get_worktree(worktree_id) or record
            if record["status"] == "removed":
                return _cleanup_worktree(project_id, worktree_id)
            if record["status"] != "active":
                raise AgentWorkspaceError(
                    "Studio cannot prove this worktree is safe to remove. Inspect it manually."
                )
        repository = _resolve_record_repository(record)
        path, _, marker_payload = _verify_owned_marker(record)
        _require_linked_task_stopped(record, marker_payload)
    with _workspace_writer_slots(repository, path):
        with _project_worktree_operation(project_id):
            return _cleanup_worktree(project_id, worktree_id)


def _cleanup_worktree(project_id: str, worktree_id: str) -> dict:
    record = get_worktree(worktree_id)
    if record is None or record["projectId"] != project_id:
        raise AgentWorkspaceError("Studio worktree not found.")
    if record["status"] == "removed":
        _retire_marker(record)
        return record
    if record["status"] != "active":
        _reconcile_record(record)
        record = get_worktree(worktree_id) or record
        if record["status"] == "removed":
            _retire_marker(record)
            return record
        if record["status"] != "active":
            raise AgentWorkspaceError(
                "Studio cannot prove this worktree is safe to remove. Inspect it manually."
            )
    repository = _resolve_record_repository(record)
    with _serialized_repository_mutation(repository):
        path, _, marker_payload = _verify_owned_marker(record)
        _require_linked_task_stopped(record, marker_payload)
        entry = _registered_entry(_worktree_entries(repository), path)
        if not _registration_matches(record, entry):
            raise AgentWorkspaceError(
                "Worktree registration or branch no longer matches Studio ownership."
            )
        try:
            removing = transition_worktree_status(worktree_id, {"active"}, "removing")
        except Exception as exc:
            raise AgentWorkspaceError("Worktree cleanup could not reserve durable state.") from exc
        if removing is None:
            raise AgentWorkspaceError("Studio worktree not found.")
        try:
            _git(
                repository,
                ["worktree", "remove", str(path)],
                timeout_seconds = 120,
                output_limit = 128_000,
                neutralize_filters = True,
            )
        except Exception:
            _safe_transition(worktree_id, {"removing"}, "active")
            raise
        try:
            removed = transition_worktree_status(worktree_id, {"removing"}, "removed")
        except Exception as exc:
            raise AgentWorkspaceError(
                "Git removed the worktree, but durable cleanup is pending startup recovery."
            ) from exc
        if removed is None:
            raise AgentWorkspaceError(
                "Git removed the worktree, but durable cleanup is pending startup recovery."
            )
        _retire_marker(removed)
        return removed


__all__ = [
    "begin_project_deletion",
    "cleanup_worktree",
    "create_worktree",
    "finish_project_deletion",
    "list_worktrees",
    "merge_worktree",
    "owned_worktree_path",
    "reconcile_worktrees_on_startup",
    "sync_worktree_background_task_marker",
]
