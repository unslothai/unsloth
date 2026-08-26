# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Shared project-root and bounded-output primitives."""

import hashlib
import os
import signal
import stat
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from storage.studio_db import ensure_chat_project_workspace, get_chat_project
from utils.paths import ensure_dir, tmp_root
from utils.prebuilt.child_env import CRED_LOCATION_ENV_NAMES, scrub_env
from utils.process_lifetime import (
    adopt_pid,
    child_popen_kwargs,
    forget_pid,
    initialize_parent_lifetime,
    spawn_on_lifetime_thread,
)


class AgentWorkspaceError(RuntimeError):
    """A safe, user-readable workspace operation failure."""


@dataclass(frozen = True)
class ProjectWorkspace:
    project_id: str
    root: Path
    kind: str
    device_id: Optional[int] = None
    file_id: Optional[int] = None


def now_ms() -> int:
    return int(time.time() * 1000)


def project_workspace(project_id: str) -> ProjectWorkspace:
    """Resolve the persisted project root, without accepting a path from the renderer."""
    project = get_chat_project(project_id)
    if project is None:
        raise AgentWorkspaceError("Project not found.")
    try:
        project = ensure_chat_project_workspace(project_id) or project
    except OSError as exc:
        raise AgentWorkspaceError(
            "The project folder is unavailable. Reconnect it and reopen the project."
        ) from exc
    kind = str(project.get("workspaceKind") or "managed")
    raw_root = project.get("rootPath") if kind == "folder" else project.get("sandboxPath")
    if not raw_root:
        raise AgentWorkspaceError("The project has no workspace folder.")
    root = Path(str(raw_root)).expanduser()
    try:
        if root.is_symlink():
            raise AgentWorkspaceError("Symbolic-link project roots are not supported.")
        resolved = root.resolve(strict = True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise AgentWorkspaceError("The project folder is unavailable.") from exc
    if not resolved.is_dir():
        raise AgentWorkspaceError("The project workspace is not a directory.")
    device_id = project.get("workspaceDeviceId")
    file_id = project.get("workspaceFileId")
    try:
        expected_device = int(device_id) if device_id is not None else None
        expected_file = int(file_id) if file_id is not None else None
    except (TypeError, ValueError) as exc:
        raise AgentWorkspaceError("The project folder identity is invalid.") from exc
    if expected_device is None or expected_file is None:
        raise AgentWorkspaceError("The project folder identity is missing. Reopen it.")
    return ProjectWorkspace(
        project_id = project_id,
        root = resolved,
        kind = kind,
        device_id = expected_device,
        file_id = expected_file,
    )


def contained_path(
    root: Path,
    relative: Optional[str],
    *,
    must_exist: bool = False,
) -> Path:
    """Resolve a project-relative path and reject traversal and symlink escapes."""
    value = relative or "."
    candidate = Path(value)
    if candidate.is_absolute():
        raise AgentWorkspaceError("Workspace paths must be relative to the project root.")
    try:
        resolved = (root / candidate).resolve(strict = must_exist)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise AgentWorkspaceError("Workspace path escapes the project root.") from exc
    return resolved


def bounded_text(raw: bytes, limit: int) -> tuple[str, bool]:
    """Decode bounded process/file output without splitting a UTF-8 sequence."""
    truncated = len(raw) > limit
    return raw[:limit].decode("utf-8", errors = "replace"), truncated


def agent_child_env(root: Optional[Path] = None, extra: Optional[dict] = None) -> dict:
    """Build a credential-free child environment with an isolated home and temp root."""
    env = scrub_env(os.environ)
    for name in CRED_LOCATION_ENV_NAMES:
        env.pop(name, None)
    for name in (
        "BASH_ENV",
        "CDPATH",
        "ENV",
        "OLDPWD",
        "PYTHONSTARTUP",
        "ZDOTDIR",
    ):
        env.pop(name, None)
    scratch = ensure_dir(tmp_root() / "agent-child").resolve()
    try:
        scratch.chmod(0o700)
    except OSError:
        pass
    if root is not None:
        try:
            scratch.relative_to(root.resolve(strict = True))
        except ValueError:
            pass
        else:
            raise AgentWorkspaceError(
                "The isolated child environment overlaps the project workspace."
            )
    for name in ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA"):
        env[name] = str(scratch)
    for name in ("TMP", "TEMP", "TMPDIR"):
        env[name] = str(scratch)
    if extra:
        env.update(extra)
    return scrub_env(env)


def run_bounded(
    argv: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: float = 10.0,
    output_limit: int = 256_000,
    env: Optional[dict] = None,
) -> tuple[int, str, bool]:
    """Run an argv-only child with a timeout and bounded combined output."""
    options = {
        "cwd": str(cwd),
        "env": env if env is not None else agent_child_env(cwd),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
    }
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
        options.update(child_popen_kwargs())
    initialize_parent_lifetime()
    process = spawn_on_lifetime_thread(lambda: subprocess.Popen(list(argv), **options))
    adopt_pid(process.pid)
    group_id = process.pid if os.name != "nt" else None
    captured = bytearray()
    total = [0]

    def drain() -> None:
        while True:
            chunk = process.stdout.read(64 * 1024)
            if not chunk:
                return
            total[0] += len(chunk)
            remaining = output_limit - len(captured)
            if remaining > 0:
                captured.extend(chunk[:remaining])

    reader = threading.Thread(target = drain, daemon = True)
    reader.start()
    try:
        try:
            code = process.wait(timeout = timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            _terminate_bounded_process(process, group_id)
            reader.join(timeout = 2)
            text, _ = bounded_text(bytes(captured), output_limit)
            raise AgentWorkspaceError("Command timed out." + (f"\n{text}" if text else "")) from exc
        reader.join(timeout = 0.2)
        if reader.is_alive():
            _terminate_bounded_process(process, group_id)
            reader.join(timeout = 2)
        text, _ = bounded_text(bytes(captured), output_limit)
        return code, text, total[0] > len(captured)
    finally:
        forget_pid(process.pid)


def _terminate_bounded_process(process: subprocess.Popen, group_id: Optional[int] = None) -> None:
    if os.name == "nt":
        if process.pid <= 1:
            return
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin = subprocess.DEVNULL,
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
                timeout = 5,
                check = False,
            )
        except (OSError, subprocess.SubprocessError):
            if process.poll() is None:
                process.kill()
        return
    signalled_group = False
    try:
        if group_id is not None and group_id > 1:
            os.killpg(group_id, signal.SIGTERM)
            signalled_group = True
        elif process.poll() is None:
            process.terminate()
        else:
            return
    except OSError:
        pass
    try:
        process.wait(timeout = 0.5)
    except subprocess.TimeoutExpired:
        pass
    if signalled_group:
        deadline = time.monotonic() + 0.5
        while time.monotonic() < deadline:
            try:
                os.killpg(group_id, 0)
            except OSError:
                return
            time.sleep(0.02)
        try:
            os.killpg(group_id, signal.SIGKILL)
        except OSError:
            pass
    elif process.poll() is None:
        try:
            process.kill()
        except OSError:
            pass


def git_root(root: Path) -> Path:
    code, output, _ = run_bounded(
        ["git", "rev-parse", "--show-toplevel"],
        cwd = root,
        timeout_seconds = 5,
        output_limit = 16_384,
    )
    if code != 0:
        raise AgentWorkspaceError("The project folder is not a Git repository.")
    try:
        resolved = Path(output.strip()).resolve(strict = True)
        root.relative_to(resolved)
    except (OSError, RuntimeError, ValueError) as exc:
        raise AgentWorkspaceError("Git returned an invalid repository root.") from exc
    return resolved


_COMPLETE_FINGERPRINT_PREFIX = "c0dec0de"
_INCOMPLETE_FINGERPRINT_PREFIX = "badc0ffe"
_GIT_DIRTY_CONTENT_BUDGET = 4 * 1024 * 1024
_FILESYSTEM_CONTENT_BUDGET = 16 * 1024 * 1024
_FINGERPRINT_FILE_LIMIT = 100_000


def _finish_fingerprint(digest: "hashlib._Hash", complete: bool) -> str:
    value = digest.hexdigest()
    prefix = _COMPLETE_FINGERPRINT_PREFIX if complete else _INCOMPLETE_FINGERPRINT_PREFIX
    return prefix + value[len(prefix) :]


def workspace_fingerprint_complete(value: Optional[str]) -> bool:
    """Whether a persisted fingerprint represents complete bounded evidence."""
    return bool(value and value.startswith(_COMPLETE_FINGERPRINT_PREFIX))


def _hash_workspace_entry(
    digest: "hashlib._Hash", path: Path, label: str, remaining: int
) -> tuple[int, bool]:
    """Hash one entry through a no-follow descriptor, returning budget and completeness."""
    digest.update(label.encode("utf-8", errors = "surrogateescape"))
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            digest.update(b"\0SYMLINK\0")
            digest.update(os.readlink(path).encode("utf-8", errors = "surrogateescape"))
            return remaining, True
        descriptor = os.open(path, flags)
    except OSError:
        digest.update(b"\0UNREADABLE")
        return remaining, False
    complete = True
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            digest.update(b"\0UNSUPPORTED-TYPE")
            return remaining, False
        digest.update(f"\0{before.st_mode}\0{before.st_size}\0".encode("ascii"))
        with os.fdopen(descriptor, "rb", closefd = False) as source:
            while True:
                chunk = source.read(min(64 * 1024, max(0, remaining)) + 1)
                if not chunk:
                    break
                if remaining <= 0:
                    digest.update(b"TRUNCATED-CONTENT-BUDGET")
                    complete = False
                    break
                accepted = chunk[:remaining]
                digest.update(accepted)
                remaining -= len(accepted)
                if len(chunk) > len(accepted):
                    digest.update(b"TRUNCATED-CONTENT-BUDGET")
                    complete = False
                    break
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity:
            digest.update(b"CHANGED-DURING-SCAN")
            complete = False
    except OSError:
        digest.update(b"\0UNREADABLE-CONTENT")
        complete = False
    finally:
        os.close(descriptor)
    return remaining, complete


def _status_paths(output: str) -> tuple[list[str], bool]:
    """Extract every current path from porcelain v1 -z output."""
    if "\ufffd" in output:
        return [], False
    entries = output.split("\0")
    names: list[str] = []
    index = 0
    complete = True
    while index < len(entries):
        entry = entries[index]
        index += 1
        if not entry:
            continue
        if len(entry) < 4 or entry[2] != " ":
            complete = False
            continue
        names.append(entry[3:])
        if "R" in entry[:2] or "C" in entry[:2]:
            if index >= len(entries) or not entries[index]:
                complete = False
                continue
            names.append(entries[index])
            index += 1
    return sorted(set(names)), complete


def _git_workspace_fingerprint(root: Path) -> str:
    """Delegate Git evidence to the repository-config-neutralized runner.

    ``git_service`` imports this module for shared bounded hashing primitives, so
    importing it at module load would create a cycle. The call-time import keeps
    one hardened Git implementation while preserving the common non-Git fallback.
    """
    from .git_service import workspace_fingerprint as safe_git_fingerprint
    return safe_git_fingerprint(root)


def _filesystem_workspace_fingerprint(root: Path) -> str:
    digest = hashlib.sha256()
    complete = True
    seen = 0
    remaining = _FILESYSTEM_CONTENT_BUDGET
    walk_errors: list[OSError] = []
    for current, dirs, files in os.walk(root, followlinks = False, onerror = walk_errors.append):
        symlink_dirs = sorted(d for d in dirs if (Path(current) / d).is_symlink())
        dirs[:] = sorted(d for d in dirs if d not in symlink_dirs)
        for name in [*symlink_dirs, *sorted(files)]:
            path = Path(current) / name
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                complete = False
                continue
            seen += 1
            if seen > _FINGERPRINT_FILE_LIMIT:
                digest.update(b"TRUNCATED-FILE-COUNT")
                return _finish_fingerprint(digest, False)
            remaining, entry_complete = _hash_workspace_entry(digest, path, relative, remaining)
            complete = complete and entry_complete
    if walk_errors:
        digest.update(b"WALK-ERROR")
        complete = False
    return _finish_fingerprint(digest, complete)


def workspace_fingerprint(root: Path) -> str:
    """Hash complete bounded content evidence, marking any incomplete scan."""
    try:
        repository = git_root(root)
    except AgentWorkspaceError:
        return _filesystem_workspace_fingerprint(root)
    return _git_workspace_fingerprint(root)
