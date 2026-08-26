# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Path-scoped Git reads and explicitly owned Studio checkpoints."""

import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import stat
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from utils.paths import ensure_dir, studio_root

from . import common as workspace_common
from .common import (
    AgentWorkspaceError,
    agent_child_env,
    now_ms,
    project_workspace,
    run_bounded,
)
from .execution import (
    acquire_workspace_execution_slot,
    release_workspace_execution_slot,
)
from .state import (
    delete_checkpoint,
    get_checkpoint,
    list_checkpoints,
    save_checkpoint,
)
from .prepared_commit_state import (
    delete_preparation,
    list_ref_bearing_preparations,
    mark_confirmed,
    mark_failed,
    reserve_confirmation,
    save_candidate_commit,
    save_preparation,
)


_SAFE_PATH = re.compile(r"^[^\x00\r\n]+$")
_SAFE_OBJECT_ID = re.compile(r"^[0-9a-fA-F]{40,64}$")
_REPOSITORY_LOCKS_GUARD = threading.Lock()
_REPOSITORY_LOCKS: dict[str, threading.RLock] = {}
_PROJECT_CHECKPOINT_CONDITION = threading.Condition()
_PROJECT_CHECKPOINT_ACTIVE: dict[str, int] = {}
_PROJECTS_DELETING: set[str] = set()
_DISABLED_HOOKS_LOCK = threading.Lock()
_EMPTY_CONFIG_LOCK = threading.Lock()
_FILTER_CONFIG_KEY = re.compile(r"^filter\.(?P<driver>.+)\.(?:clean|smudge|process|required)$")
_MERGE_DRIVER_CONFIG_KEY = re.compile(r"^merge\.(?P<driver>.+)\.driver$")
_SAFE_FILTER_DRIVER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ALLOWED_GIT_ENV = frozenset(
    {
        "GIT_AUTHOR_DATE",
        "GIT_AUTHOR_EMAIL",
        "GIT_AUTHOR_NAME",
        "GIT_COMMITTER_DATE",
        "GIT_COMMITTER_EMAIL",
        "GIT_COMMITTER_NAME",
        "GIT_INDEX_FILE",
    }
)
_PREPARED_COMMIT_TTL_MS = 5 * 60 * 1000
_PREPARED_COMMIT_OPERATION = "prepare_commit"


def _disabled_hooks_path() -> Path:
    """Return an empty Studio-owned directory that makes Git hooks inert."""
    try:
        root = ensure_dir(studio_root()).resolve(strict = True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise AgentWorkspaceError("Studio Git safety storage is unavailable.") from exc
    hooks = root / ".agent-git-hooks-disabled"
    with _DISABLED_HOOKS_LOCK:
        try:
            metadata = hooks.lstat()
        except FileNotFoundError:
            try:
                hooks.mkdir(mode = 0o700)
                metadata = hooks.lstat()
            except OSError as exc:
                raise AgentWorkspaceError("Studio Git safety storage is unavailable.") from exc
        if not stat.S_ISDIR(metadata.st_mode):
            raise AgentWorkspaceError("Studio Git safety storage is not a safe directory.")
        try:
            if any(hooks.iterdir()):
                raise AgentWorkspaceError(
                    "Studio Git safety storage is not empty. Git operations were blocked."
                )
            hooks.chmod(0o700)
        except AgentWorkspaceError:
            raise
        except OSError as exc:
            raise AgentWorkspaceError("Studio Git safety storage is unavailable.") from exc
    return hooks


def _empty_git_config_path() -> Path:
    """Return an empty Studio-owned config used in place of user Git config."""
    try:
        root = ensure_dir(studio_root()).resolve(strict = True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise AgentWorkspaceError("Studio Git safety storage is unavailable.") from exc
    config = root / ".agent-git-config-empty"
    with _EMPTY_CONFIG_LOCK:
        try:
            metadata = config.lstat()
        except FileNotFoundError:
            descriptor = None
            try:
                descriptor = os.open(
                    config,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                )
                metadata = config.lstat()
            except OSError as exc:
                raise AgentWorkspaceError("Studio Git safety storage is unavailable.") from exc
            finally:
                if descriptor is not None:
                    os.close(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != 0:
            raise AgentWorkspaceError("Studio Git safety storage is not a safe file.")
        try:
            config.chmod(0o600)
        except OSError as exc:
            raise AgentWorkspaceError("Studio Git safety storage is unavailable.") from exc
    return config


def _safe_git_environment(root: Path, env: Optional[dict] = None) -> dict:
    """Build a Git environment that cannot inherit executable user settings."""
    safe = agent_child_env(root)
    for name in tuple(safe):
        if name.startswith("GIT_") or name in {
            "PAGER",
            "EDITOR",
            "VISUAL",
            "SSH_ASKPASS",
            "GCM_INTERACTIVE",
        }:
            safe.pop(name, None)
    if env:
        for name in _ALLOWED_GIT_ENV:
            value = env.get(name)
            if value is not None:
                safe[name] = str(value)
    empty_config = str(_empty_git_config_path())
    safe.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": empty_config,
            "GIT_CONFIG_GLOBAL": empty_config,
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_PAGER": "cat",
            "GIT_EDITOR": "false",
            "GIT_SEQUENCE_EDITOR": "false",
            "GIT_MERGE_AUTOEDIT": "no",
            "GCM_INTERACTIVE": "Never",
            "PAGER": "cat",
            "LC_ALL": "C",
        }
    )
    return safe


def _base_git_arguments() -> list[str]:
    hooks = _disabled_hooks_path()
    return [
        "git",
        "--no-pager",
        "-c",
        f"core.hooksPath={hooks}",
        "-c",
        "core.fsmonitor=false",
        "-c",
        "core.untrackedCache=false",
        "-c",
        "credential.helper=",
        "-c",
        "commit.gpgSign=false",
        "-c",
        "tag.gpgSign=false",
        "-c",
        "merge.verifySignatures=false",
        "-c",
        "log.showSignature=false",
        "-c",
        "gpg.program=false",
        "-c",
        "gpg.ssh.program=false",
        "-c",
        "diff.external=",
        "-c",
        "core.editor=false",
        "-c",
        "sequence.editor=false",
    ]


def _run_git(
    root: Path,
    args: list[str],
    *,
    output_limit: int = 512_000,
    timeout_seconds: float = 15,
    env: Optional[dict] = None,
    config_overrides: Optional[list[str]] = None,
) -> tuple[int, str, bool]:
    command = _base_git_arguments()
    command.extend(config_overrides or [])
    command.extend(args)
    return run_bounded(
        command,
        cwd = root,
        timeout_seconds = timeout_seconds,
        output_limit = output_limit,
        env = _safe_git_environment(root, env),
    )


def _neutral_filter_overrides(root: Path, env: Optional[dict]) -> list[str]:
    """Replace every configured content filter with a trusted pass-through."""
    code, output, truncated = _run_git(
        root,
        [
            "config",
            "--includes",
            "--name-only",
            "--get-regexp",
            r"^filter\..*\.(clean|smudge|process|required)$",
        ],
        output_limit = 256_000,
        timeout_seconds = 5,
        env = env,
    )
    if code == 1 and not output:
        return []
    if code != 0 or truncated:
        raise AgentWorkspaceError("Git content filters could not be inspected safely.")
    drivers = set()
    for key in output.splitlines():
        match = _FILTER_CONFIG_KEY.fullmatch(key.strip())
        if match is None or not _SAFE_FILTER_DRIVER.fullmatch(match.group("driver")):
            raise AgentWorkspaceError("Git content filter configuration is invalid.")
        drivers.add(match.group("driver"))
    if not drivers:
        return []
    passthrough = _trusted_passthrough_filter_command()
    overrides = []
    for driver in sorted(drivers):
        overrides.extend(
            [
                "-c",
                f"filter.{driver}.process=",
                "-c",
                f"filter.{driver}.clean={passthrough}",
                "-c",
                f"filter.{driver}.smudge={passthrough}",
                "-c",
                f"filter.{driver}.required=false",
            ]
        )
    return overrides


def _trusted_passthrough_filter_command() -> str:
    candidates = [shutil.which("cat", path = os.defpath)]
    git_program = shutil.which("git")
    if os.name == "nt" and git_program:
        install = Path(git_program).resolve().parent.parent
        candidates.extend(
            [
                str(install / "usr" / "bin" / "cat.exe"),
                str(install / "mingw64" / "bin" / "cat.exe"),
            ]
        )
    for raw in candidates:
        if not raw:
            continue
        try:
            candidate = Path(raw).resolve(strict = True)
            metadata = candidate.stat()
        except (OSError, RuntimeError, ValueError):
            continue
        if stat.S_ISREG(metadata.st_mode) and os.access(candidate, os.X_OK):
            command_path = str(candidate)
            if os.name == "nt":
                command_path = command_path.replace("\\", "/")
            return shlex.quote(command_path)
    raise AgentWorkspaceError(
        "Studio cannot safely bypass repository content filters on this system."
    )


def _trusted_merge_driver_command() -> str:
    """Return a trusted three-way file merge command for custom drivers."""
    candidates = [shutil.which("git", path = os.defpath)]
    if os.name == "nt":
        candidates.append(shutil.which("git"))
    for raw in candidates:
        if not raw:
            continue
        try:
            candidate = Path(raw).resolve(strict = True)
            metadata = candidate.stat()
        except (OSError, RuntimeError, ValueError):
            continue
        if stat.S_ISREG(metadata.st_mode) and os.access(candidate, os.X_OK):
            command_path = str(candidate)
            if os.name == "nt":
                command_path = command_path.replace("\\", "/")
            return f"{shlex.quote(command_path)} merge-file %A %O %B"
    raise AgentWorkspaceError(
        "Studio cannot safely bypass repository merge drivers on this system."
    )


def _neutral_merge_driver_overrides(root: Path, env: Optional[dict]) -> list[str]:
    """Replace repository-configured merge drivers with trusted Git behavior."""
    code, output, truncated = _run_git(
        root,
        [
            "config",
            "--includes",
            "--name-only",
            "--get-regexp",
            r"^merge\..*\.driver$",
        ],
        output_limit = 256_000,
        timeout_seconds = 5,
        env = env,
    )
    if code == 1 and not output:
        return []
    if code != 0 or truncated:
        raise AgentWorkspaceError("Git merge drivers could not be inspected safely.")
    drivers = set()
    for key in output.splitlines():
        match = _MERGE_DRIVER_CONFIG_KEY.fullmatch(key.strip())
        if match is None or not _SAFE_FILTER_DRIVER.fullmatch(match.group("driver")):
            raise AgentWorkspaceError("Git merge driver configuration is invalid.")
        drivers.add(match.group("driver"))
    if not drivers:
        return []
    command = _trusted_merge_driver_command()
    overrides = []
    for driver in sorted(drivers):
        overrides.extend(["-c", f"merge.{driver}.driver={command}"])
    return overrides


@contextmanager
def _project_checkpoint_operation(project_id: str) -> Iterator[None]:
    with _PROJECT_CHECKPOINT_CONDITION:
        if project_id in _PROJECTS_DELETING:
            raise AgentWorkspaceError(
                "Checkpoint operations are unavailable while the project is being deleted."
            )
        _PROJECT_CHECKPOINT_ACTIVE[project_id] = _PROJECT_CHECKPOINT_ACTIVE.get(project_id, 0) + 1
    try:
        yield
    finally:
        with _PROJECT_CHECKPOINT_CONDITION:
            remaining = _PROJECT_CHECKPOINT_ACTIVE.get(project_id, 1) - 1
            if remaining > 0:
                _PROJECT_CHECKPOINT_ACTIVE[project_id] = remaining
            else:
                _PROJECT_CHECKPOINT_ACTIVE.pop(project_id, None)
            _PROJECT_CHECKPOINT_CONDITION.notify_all()


def begin_project_deletion(project_id: str, timeout_seconds: float = 15) -> None:
    """Fence new checkpoint work and wait for existing operations to finish."""
    deadline = time.monotonic() + max(0.1, timeout_seconds)
    with _PROJECT_CHECKPOINT_CONDITION:
        if project_id in _PROJECTS_DELETING:
            raise AgentWorkspaceError("Project checkpoint deletion is already in progress.")
        _PROJECTS_DELETING.add(project_id)
        while _PROJECT_CHECKPOINT_ACTIVE.get(project_id, 0) > 0:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _PROJECTS_DELETING.discard(project_id)
                _PROJECT_CHECKPOINT_CONDITION.notify_all()
                raise AgentWorkspaceError(
                    "A checkpoint operation is still running. Try deleting the project again."
                )
            _PROJECT_CHECKPOINT_CONDITION.wait(timeout = remaining)


def finish_project_deletion(project_id: str) -> None:
    with _PROJECT_CHECKPOINT_CONDITION:
        _PROJECTS_DELETING.discard(project_id)
        _PROJECT_CHECKPOINT_CONDITION.notify_all()


@contextmanager
def _serialized_repository_mutation(repository: Path) -> Iterator[None]:
    """Serialize Studio checkpoint mutations for one repository.

    The desktop backend is a single process. A repository-scoped lock keeps a
    checkpoint and rollback from observing each other's intermediate state
    between identity, fingerprint, tree, and restore checks.
    """
    key = os.path.normcase(str(repository.resolve(strict = True)))
    with _REPOSITORY_LOCKS_GUARD:
        lock = _REPOSITORY_LOCKS.setdefault(key, threading.RLock())
    with lock:
        yield


@contextmanager
def _workspace_writer_slot(root: Path) -> Iterator[None]:
    """Serialize Git mutations with project commands and edit-file writes."""
    try:
        metadata = root.stat()
    except OSError as exc:
        raise AgentWorkspaceError("The project workspace is unavailable.") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise AgentWorkspaceError("The project workspace is not a directory.")
    identity = (int(metadata.st_dev), int(metadata.st_ino))
    if not acquire_workspace_execution_slot(identity):
        raise AgentWorkspaceError("The Git operation was cancelled before it started.")
    try:
        try:
            current = root.stat()
        except OSError as exc:
            raise AgentWorkspaceError("The project workspace changed before use.") from exc
        if (int(current.st_dev), int(current.st_ino)) != identity:
            raise AgentWorkspaceError("The project workspace changed before use.")
        yield
    finally:
        release_workspace_execution_slot(identity)


def _git(
    root: Path,
    args: list[str],
    *,
    output_limit: int = 512_000,
    timeout_seconds: float = 15,
    env: Optional[dict] = None,
    neutralize_filters: bool = False,
    neutralize_merge_drivers: bool = False,
) -> tuple[str, bool]:
    overrides = []
    if neutralize_filters:
        overrides.extend(_neutral_filter_overrides(root, env))
    if neutralize_merge_drivers:
        overrides.extend(_neutral_merge_driver_overrides(root, env))
    code, output, truncated = _run_git(
        root,
        args,
        timeout_seconds = timeout_seconds,
        output_limit = output_limit,
        env = env,
        config_overrides = overrides or None,
    )
    if code != 0:
        detail = output.strip()[:1000] or "Git command failed."
        raise AgentWorkspaceError(detail)
    return output, truncated


def _safe_git_root(root: Path) -> Path:
    code, output, _ = _run_git(
        root,
        ["rev-parse", "--show-toplevel"],
        timeout_seconds = 5,
        output_limit = 16_384,
    )
    if code != 0:
        raise AgentWorkspaceError("The project folder is not a Git repository.")
    try:
        selected = root.resolve(strict = True)
        repository = Path(output.strip()).resolve(strict = True)
        selected.relative_to(repository)
    except (OSError, RuntimeError, ValueError) as exc:
        raise AgentWorkspaceError("Git returned an invalid repository root.") from exc
    return repository


def workspace_fingerprint(root: Path) -> str:
    """Hash bounded repository evidence without executing repository Git config."""
    root = root.resolve(strict = True)
    repository = _safe_git_root(root)
    digest = hashlib.sha256()
    complete = True
    relative = root.relative_to(repository).as_posix()
    pathspec = "." if relative == "." else relative
    filter_overrides = _neutral_filter_overrides(repository, None)
    commands = (
        ("head", ["rev-parse", "--verify", "--quiet", "HEAD"], 16_384),
        (
            "status",
            [
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--",
                pathspec,
            ],
            2_000_000,
        ),
        (
            "unstaged-diff",
            [
                "diff",
                "--no-ext-diff",
                "--no-textconv",
                "--full-index",
                "--",
                pathspec,
            ],
            2_000_000,
        ),
        (
            "staged-diff",
            [
                "diff",
                "--cached",
                "--no-ext-diff",
                "--no-textconv",
                "--full-index",
                "--",
                pathspec,
            ],
            2_000_000,
        ),
    )
    outputs: dict[str, str] = {}
    for label, args, limit in commands:
        code, output, truncated = _run_git(
            repository,
            args,
            timeout_seconds = 12,
            output_limit = limit,
            config_overrides = filter_overrides,
        )
        outputs[label] = output
        digest.update(label.encode("ascii"))
        digest.update(str(code).encode("ascii"))
        digest.update(output.encode("utf-8", errors = "replace"))
        unborn_head = label == "head" and code == 1 and not output
        if (code != 0 and not unborn_head) or truncated or "\ufffd" in output:
            complete = False
        if truncated:
            digest.update(b"TRUNCATED")

    names, status_complete = workspace_common._status_paths(outputs.get("status", ""))
    complete = complete and status_complete
    if len(names) > 20_000:
        digest.update(b"TRUNCATED-FILE-COUNT")
        names = names[:20_000]
        complete = False
    remaining = workspace_common._GIT_DIRTY_CONTENT_BUDGET
    for name in names:
        candidate = repository / name
        try:
            candidate.relative_to(root)
        except ValueError:
            digest.update(name.encode("utf-8", errors = "surrogateescape"))
            digest.update(b"OUTSIDE-PROJECT")
            complete = False
            continue
        if not candidate.exists() and not candidate.is_symlink():
            digest.update(name.encode("utf-8", errors = "surrogateescape"))
            digest.update(b"\0ABSENT")
            continue
        remaining, entry_complete = workspace_common._hash_workspace_entry(
            digest, candidate, name, remaining
        )
        complete = complete and entry_complete
    return workspace_common._finish_fingerprint(digest, complete)


def _project_git(project_id: str, *, mutation: bool = False) -> tuple[Path, Path]:
    workspace = project_workspace(project_id)
    repository = _safe_git_root(workspace.root)
    if mutation and repository != workspace.root:
        raise AgentWorkspaceError(
            "Git mutations require the selected project folder to be the repository root."
        )
    return workspace.root, repository


def git_status(project_id: str) -> dict:
    workspace, repository = _project_git(project_id)
    head, _ = _git(repository, ["rev-parse", "HEAD"], output_limit = 256)
    branch_code, branch_output, _ = _run_git(
        repository,
        ["symbolic-ref", "--quiet", "--short", "HEAD"],
        timeout_seconds = 5,
        output_limit = 4096,
    )
    relative = workspace.relative_to(repository).as_posix()
    pathspec = "." if relative == "." else relative
    porcelain, truncated = _git(
        repository,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--", pathspec],
        output_limit = 1_000_000,
        neutralize_filters = True,
    )
    records = [record for record in porcelain.split("\0") if record]
    counts = {"staged": 0, "unstaged": 0, "untracked": 0, "conflicted": 0}
    files = []
    index = 0
    parsed_count = 0
    while index < len(records):
        record = records[index]
        code = record[:2]
        path = record[3:] if len(record) > 3 else ""
        old_path = None
        if len(code) == 2 and (code[0] in {"R", "C"} or code[1] in {"R", "C"}):
            if index + 1 < len(records):
                old_path = records[index + 1]
                index += 1
        if code == "??":
            counts["untracked"] += 1
        else:
            if code[0] not in {" ", "?"}:
                counts["staged"] += 1
            if code[1] not in {" ", "?"}:
                counts["unstaged"] += 1
            if code in {"DD", "AU", "UD", "UA", "DU", "AA", "UU"}:
                counts["conflicted"] += 1
        if len(files) < 5_000:
            item = {"code": code, "path": path}
            if old_path is not None:
                item["oldPath"] = old_path
            files.append(item)
        parsed_count += 1
        index += 1
    return {
        "repositoryRoot": str(repository),
        "projectPrefix": relative,
        "head": head.strip(),
        "branch": branch_output.strip() if branch_code == 0 else None,
        "detached": branch_code != 0,
        "clean": parsed_count == 0,
        "counts": counts,
        "files": files,
        "truncated": truncated or parsed_count > len(files),
    }


def git_diff(
    project_id: str,
    *,
    staged: bool = False,
    max_bytes: int = 512_000,
) -> dict:
    workspace, repository = _project_git(project_id)
    relative = workspace.relative_to(repository).as_posix()
    pathspec = "." if relative == "." else relative
    args = ["diff", "--no-ext-diff", "--no-textconv", "--no-color"]
    if staged:
        args.append("--cached")
    args.extend(["--", pathspec])
    diff, truncated = _git(
        repository,
        args,
        output_limit = max(4096, min(max_bytes, 2_000_000)),
        timeout_seconds = 20,
        neutralize_filters = True,
    )
    return {"staged": staged, "diff": diff, "truncated": truncated}


def _owned_paths(root: Path, paths: list[str]) -> list[str]:
    if not paths:
        raise AgentWorkspaceError("A checkpoint requires at least one explicitly owned path.")
    if len(paths) > 5_000:
        raise AgentWorkspaceError("A checkpoint can own at most 5,000 paths.")
    resolved = []
    for raw in paths:
        normalized = str(raw).replace("\\", "/")
        if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
            raise AgentWorkspaceError("Checkpoint paths must stay inside the repository.")
        value = normalized.strip("/")
        candidate = Path(value)
        if (
            not value
            or candidate == Path(".")
            or candidate.is_absolute()
            or ".." in candidate.parts
            or candidate.parts[0].lower() == ".git"
        ):
            raise AgentWorkspaceError("Checkpoint paths must stay inside the repository.")
        if not _SAFE_PATH.match(value):
            raise AgentWorkspaceError("Checkpoint path contains invalid characters.")
        try:
            (root / candidate).resolve(strict = False).relative_to(root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise AgentWorkspaceError("Checkpoint path escapes the repository.") from exc
        resolved.append(candidate.as_posix())
    return sorted(set(resolved))


def _literal_pathspecs(paths: list[str]) -> list[str]:
    """Make root-relative paths literal even when a filename contains pathspec syntax."""
    return [f":(top,literal){path}" for path in paths]


def _build_selected_commit_object(
    root: Path,
    repository: Path,
    paths: list[str],
    message: str,
    *,
    parent_sha: Optional[str] = None,
) -> tuple[str, str, Optional[str]]:
    descriptor, index_name = tempfile.mkstemp(prefix = "unsloth-studio-index-")
    os.close(descriptor)
    os.unlink(index_name)
    env = agent_child_env(
        root,
        {
            "GIT_INDEX_FILE": index_name,
            "GIT_AUTHOR_NAME": "Unsloth Studio",
            "GIT_AUTHOR_EMAIL": "studio@localhost",
            "GIT_COMMITTER_NAME": "Unsloth Studio",
            "GIT_COMMITTER_EMAIL": "studio@localhost",
        },
    )
    try:
        if parent_sha is None:
            head_code, head_output, _ = _run_git(
                repository,
                ["rev-parse", "--verify", "HEAD"],
                timeout_seconds = 5,
                output_limit = 256,
                env = env,
            )
            parent_sha = head_output.strip() if head_code == 0 else None
        if parent_sha is not None:
            if not _SAFE_OBJECT_ID.fullmatch(parent_sha):
                raise AgentWorkspaceError("Git parent identity is invalid.")
            _git(
                repository,
                ["read-tree", parent_sha],
                env = env,
                neutralize_filters = True,
            )
        else:
            _git(
                repository,
                ["read-tree", "--empty"],
                env = env,
                neutralize_filters = True,
            )
        _git(
            repository,
            ["add", "-A", "--", *_literal_pathspecs(paths)],
            timeout_seconds = 60,
            env = env,
            neutralize_filters = True,
        )
        tree, _ = _git(
            repository,
            ["write-tree"],
            output_limit = 256,
            env = env,
            neutralize_filters = True,
        )
        tree_sha = tree.strip()
        if not _SAFE_OBJECT_ID.fullmatch(tree_sha):
            raise AgentWorkspaceError("Git returned an invalid tree identity.")
        commit_args = ["commit-tree", tree_sha, "-m", message]
        if parent_sha is not None:
            commit_args.extend(["-p", parent_sha])
        commit, _ = _git(
            repository,
            commit_args,
            output_limit = 256,
            env = env,
            neutralize_filters = True,
        )
        commit_sha = commit.strip()
        if not _SAFE_OBJECT_ID.fullmatch(commit_sha):
            raise AgentWorkspaceError("Git returned an invalid commit identity.")
        return commit_sha, tree_sha, parent_sha
    finally:
        try:
            os.unlink(index_name)
        except FileNotFoundError:
            pass


def create_checkpoint(project_id: str, owned_paths: list[str]) -> dict:
    with _project_checkpoint_operation(project_id):
        root, _ = _project_git(project_id, mutation = True)
        with _workspace_writer_slot(root):
            return _create_checkpoint(project_id, owned_paths)


def _create_checkpoint(project_id: str, owned_paths: list[str]) -> dict:
    root, repository = _project_git(project_id, mutation = True)
    with _serialized_repository_mutation(repository):
        paths = _owned_paths(root, owned_paths)
        checkpoint_id = str(uuid.uuid4())
        ref_name = f"refs/unsloth-studio/checkpoints/{checkpoint_id}"
        commit_sha, _, _ = _build_selected_commit_object(
            root,
            repository,
            paths,
            "Unsloth Studio checkpoint",
        )
        record = {
            "id": checkpoint_id,
            "projectId": project_id,
            "gitRoot": str(repository),
            "refName": ref_name,
            "commitSha": commit_sha,
            "ownedPaths": paths,
            "sourceFingerprint": workspace_fingerprint(root),
            "createdAt": now_ms(),
        }
        save_checkpoint(record)
        try:
            _git(repository, ["update-ref", ref_name, commit_sha], output_limit = 256)
        except Exception:
            delete_checkpoint(checkpoint_id, project_id)
            raise
        return record


def _attached_head(repository: Path) -> tuple[str, str]:
    branch_code, branch_output, _ = _run_git(
        repository,
        ["symbolic-ref", "--quiet", "HEAD"],
        timeout_seconds = 5,
        output_limit = 4096,
    )
    if branch_code != 0:
        raise AgentWorkspaceError("Prepared commits require an attached local branch.")
    branch_ref = branch_output.strip()
    if not branch_ref.startswith("refs/heads/") or not _SAFE_PATH.fullmatch(branch_ref):
        raise AgentWorkspaceError("Git returned an invalid branch identity.")
    head, _ = _git(
        repository,
        ["rev-parse", "--verify", "HEAD^{commit}"],
        output_limit = 256,
        timeout_seconds = 5,
    )
    head_sha = head.strip()
    if not _SAFE_OBJECT_ID.fullmatch(head_sha):
        raise AgentWorkspaceError("Git returned an invalid HEAD identity.")
    return branch_ref, head_sha


def _selected_change_preview(repository: Path, paths: list[str]) -> dict:
    pathspecs = _literal_pathspecs(paths)
    porcelain, status_truncated = _git(
        repository,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--", *pathspecs],
        output_limit = 1_000_000,
        timeout_seconds = 20,
        neutralize_filters = True,
    )
    if status_truncated or "\ufffd" in porcelain:
        raise AgentWorkspaceError(
            "Selected-file status is too large or could not be decoded safely."
        )
    raw_records = [record for record in porcelain.split("\0") if record]
    files = []
    index = 0
    while index < len(raw_records):
        record = raw_records[index]
        if len(record) < 3 or record[2] != " ":
            raise AgentWorkspaceError("Git returned invalid selected-file status.")
        code = record[:2]
        item = {"code": code, "path": record[3:]}
        if code == "??":
            raise AgentWorkspaceError(
                "Untracked files must be added to Git before commit preparation so their content is reviewable."
            )
        if code in {"DD", "AU", "UD", "UA", "DU", "AA", "UU"}:
            raise AgentWorkspaceError("Resolve repository conflicts before preparing a commit.")
        if code[0] in {"R", "C"} or code[1] in {"R", "C"}:
            index += 1
            if index >= len(raw_records):
                raise AgentWorkspaceError("Git returned invalid selected-file status.")
            item["oldPath"] = raw_records[index]
        files.append(item)
        if len(files) > 5_000:
            raise AgentWorkspaceError("A prepared commit can contain at most 5,000 changed paths.")
        index += 1
    if not files:
        raise AgentWorkspaceError("Selected paths have no changes to prepare.")
    diff, diff_truncated = _git(
        repository,
        [
            "diff",
            "HEAD",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
            "--",
            *pathspecs,
        ],
        output_limit = 256_000,
        timeout_seconds = 20,
        neutralize_filters = True,
    )
    if diff_truncated:
        raise AgentWorkspaceError(
            "The selected diff is too large to review exactly. Select fewer paths."
        )
    return {"files": files, "diff": diff, "diffTruncated": diff_truncated}


def _prepared_payload_digest(record: dict) -> str:
    payload = {
        "id": record["id"],
        "projectId": record["projectId"],
        "operation": record["operation"],
        "branchRef": record["branchRef"],
        "headSha": record["headSha"],
        "gitRoot": record["gitRoot"],
        "message": record["message"],
        "ownedPaths": record["ownedPaths"],
        "sourceFingerprint": record["sourceFingerprint"],
        "refName": record["refName"],
        "createdAt": record["createdAt"],
        "expiresAt": record["expiresAt"],
    }
    encoded = json.dumps(payload, ensure_ascii = False, separators = (",", ":"), sort_keys = True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _prepared_public_record(record: dict) -> dict:
    public = {
        "id": record["id"],
        "projectId": record["projectId"],
        "status": record["status"],
        "branch": record["branchRef"].removeprefix("refs/heads/"),
        "baseHead": record["headSha"],
        "message": record["message"],
        "ownedPaths": record["ownedPaths"],
        "sourceFingerprint": record["sourceFingerprint"],
        "createdAt": record["createdAt"],
        "expiresAt": record["expiresAt"],
    }
    for key in (
        "confirmationToken",
        "files",
        "diff",
        "diffTruncated",
        "commitSha",
        "refName",
        "confirmedAt",
    ):
        if record.get(key) is not None:
            public[key] = record[key]
    return public


def prepare_commit(project_id: str, owned_paths: list[str], message: str) -> dict:
    normalized_message = str(message).strip()
    if (
        not normalized_message
        or "\x00" in normalized_message
        or len(normalized_message.encode("utf-8")) > 32_000
    ):
        raise AgentWorkspaceError("Prepared commit messages must contain 1 to 32,000 UTF-8 bytes.")
    with _project_checkpoint_operation(project_id):
        root, repository = _project_git(project_id, mutation = True)
        with _workspace_writer_slot(root), _serialized_repository_mutation(repository):
            paths = _owned_paths(root, owned_paths)
            branch_ref, head_sha = _attached_head(repository)
            status = git_status(project_id)
            if status["truncated"]:
                raise AgentWorkspaceError(
                    "Repository status is too large to prepare a commit safely."
                )
            if status["counts"]["conflicted"]:
                raise AgentWorkspaceError("Resolve repository conflicts before preparing a commit.")
            preview = _selected_change_preview(repository, paths)
            source_fingerprint = workspace_fingerprint(root)
            if not workspace_common.workspace_fingerprint_complete(source_fingerprint):
                raise AgentWorkspaceError(
                    "Repository evidence is incomplete. Reduce the change set and retry."
                )
            created_at = now_ms()
            preparation_id = str(uuid.uuid4())
            record = {
                "id": preparation_id,
                "projectId": project_id,
                "operation": _PREPARED_COMMIT_OPERATION,
                "status": "awaiting_confirmation",
                "branchRef": branch_ref,
                "headSha": head_sha,
                "gitRoot": str(repository),
                "message": normalized_message,
                "ownedPaths": paths,
                "sourceFingerprint": source_fingerprint,
                "refName": (f"refs/unsloth-studio/prepared-commits/{preparation_id}"),
                "createdAt": created_at,
                "expiresAt": created_at + _PREPARED_COMMIT_TTL_MS,
                **preview,
            }
            record["payloadDigest"] = _prepared_payload_digest(record)
            confirmation_token = secrets.token_urlsafe(32)
            save_preparation(record, confirmation_token, now = created_at)
            record["confirmationToken"] = confirmation_token
            return _prepared_public_record(record)


def confirm_prepared_commit(project_id: str, preparation_id: str, confirmation_token: str) -> dict:
    record = reserve_confirmation(
        preparation_id,
        project_id,
        confirmation_token,
        now = now_ms(),
    )
    ref_created = False
    try:
        if record["payloadDigest"] != _prepared_payload_digest(record):
            raise AgentWorkspaceError("Prepared commit state failed its integrity check.")
        expected_ref = f"refs/unsloth-studio/prepared-commits/{preparation_id}"
        if record["refName"] != expected_ref:
            raise AgentWorkspaceError("Prepared commit ownership metadata is invalid.")
        with _project_checkpoint_operation(project_id):
            root, repository = _project_git(project_id, mutation = True)
            with _workspace_writer_slot(root), _serialized_repository_mutation(repository):
                try:
                    recorded_root = Path(record["gitRoot"]).resolve(strict = True)
                except (OSError, RuntimeError, ValueError) as exc:
                    raise AgentWorkspaceError(
                        "Prepared commit repository identity is unavailable."
                    ) from exc
                if recorded_root != repository:
                    raise AgentWorkspaceError(
                        "Prepared commit repository identity no longer matches."
                    )
                branch_ref, head_sha = _attached_head(repository)
                if branch_ref != record["branchRef"] or head_sha != record["headSha"]:
                    raise AgentWorkspaceError(
                        "The branch or HEAD changed after commit preparation. Prepare it again."
                    )
                paths = _owned_paths(root, list(record["ownedPaths"]))
                _selected_change_preview(repository, paths)
                current = workspace_fingerprint(root)
                if (
                    not workspace_common.workspace_fingerprint_complete(current)
                    or current != record["sourceFingerprint"]
                ):
                    raise AgentWorkspaceError(
                        "The repository changed after commit preparation. Prepare it again."
                    )
                commit_sha, tree_sha, _ = _build_selected_commit_object(
                    root,
                    repository,
                    paths,
                    record["message"],
                    parent_sha = record["headSha"],
                )
                head_tree, _ = _git(
                    repository,
                    ["rev-parse", f"{record['headSha']}^{{tree}}"],
                    output_limit = 256,
                    timeout_seconds = 5,
                )
                if tree_sha == head_tree.strip():
                    raise AgentWorkspaceError("Selected paths have no changes to prepare.")
                if workspace_fingerprint(root) != record["sourceFingerprint"]:
                    raise AgentWorkspaceError(
                        "The repository changed while the commit was being prepared."
                    )
                final_branch, final_head = _attached_head(repository)
                if final_branch != record["branchRef"] or final_head != record["headSha"]:
                    raise AgentWorkspaceError(
                        "The branch or HEAD changed while the commit was being prepared."
                    )
                record["commitSha"] = commit_sha
                save_candidate_commit(preparation_id, commit_sha)
                zero_object = "0" * len(commit_sha)
                _git(
                    repository,
                    [
                        "update-ref",
                        "--no-deref",
                        record["refName"],
                        commit_sha,
                        zero_object,
                    ],
                    output_limit = 1024,
                )
                ref_created = True
                confirmed_at = now_ms()
                mark_confirmed(preparation_id, commit_sha, now = confirmed_at)
                record.update(
                    {
                        "status": "confirmed",
                        "commitSha": commit_sha,
                        "confirmedAt": confirmed_at,
                    }
                )
                return _prepared_public_record(record)
    except Exception:
        ref_removed = not ref_created
        if ref_created:
            try:
                cleanup_root = Path(record["gitRoot"]).resolve(strict = True)
                with (
                    _workspace_writer_slot(cleanup_root),
                    _serialized_repository_mutation(cleanup_root),
                ):
                    _git(
                        cleanup_root,
                        [
                            "update-ref",
                            "--no-deref",
                            "-d",
                            record["refName"],
                            str(record.get("commitSha") or ""),
                        ],
                        output_limit = 1024,
                    )
                ref_removed = True
            except Exception:
                ref_removed = False
        if ref_removed:
            try:
                mark_failed(preparation_id)
            except Exception:
                pass
        raise


def reconcile_project_checkpoints_for_deletion(project_id: str) -> dict:
    """Remove only verified Studio refs before the project row can cascade."""
    with _PROJECT_CHECKPOINT_CONDITION:
        if project_id not in _PROJECTS_DELETING:
            raise AgentWorkspaceError(
                "Checkpoint cleanup requires an active project deletion fence."
            )
    checkpoints = list_checkpoints(project_id)
    preparations = list_ref_bearing_preparations(project_id)
    if not checkpoints and not preparations:
        return {"projectId": project_id, "removed": 0, "alreadyMissing": 0}
    root, repository = _project_git(project_id, mutation = True)
    with _workspace_writer_slot(root), _serialized_repository_mutation(repository):
        inspected = []
        for checkpoint in checkpoints:
            checkpoint_id = str(checkpoint["id"])
            expected_ref = f"refs/unsloth-studio/checkpoints/{checkpoint_id}"
            ref_name = str(checkpoint["refName"])
            commit_sha = str(checkpoint["commitSha"])
            if ref_name != expected_ref or not _SAFE_OBJECT_ID.fullmatch(commit_sha):
                raise AgentWorkspaceError(
                    "Checkpoint ownership metadata is invalid; no Git refs were removed."
                )
            try:
                recorded_root = Path(checkpoint["gitRoot"]).resolve(strict = True)
            except (OSError, RuntimeError, ValueError) as exc:
                raise AgentWorkspaceError(
                    "Checkpoint repository identity is unavailable; project deletion stopped."
                ) from exc
            if recorded_root != repository or root != repository:
                raise AgentWorkspaceError(
                    "Checkpoint repository identity no longer matches; project deletion stopped."
                )
            code, _, _ = _run_git(
                repository,
                ["show-ref", "--verify", "--quiet", ref_name],
                timeout_seconds = 5,
                output_limit = 1024,
            )
            if code == 1:
                inspected.append(("checkpoint", checkpoint, False))
                continue
            if code != 0:
                raise AgentWorkspaceError(
                    "Git could not verify a Studio checkpoint ref; project deletion stopped."
                )
            symbolic_code, _, _ = _run_git(
                repository,
                ["symbolic-ref", "--quiet", ref_name],
                timeout_seconds = 5,
                output_limit = 1024,
            )
            if symbolic_code == 0:
                raise AgentWorkspaceError(
                    "A Studio checkpoint ref changed after creation; project deletion stopped."
                )
            if symbolic_code != 1:
                raise AgentWorkspaceError(
                    "Git could not verify a Studio checkpoint ref; project deletion stopped."
                )
            code, output, _ = _run_git(
                repository,
                ["rev-parse", "--verify", "--quiet", ref_name],
                timeout_seconds = 5,
                output_limit = 1024,
            )
            if code != 0:
                raise AgentWorkspaceError(
                    "A Studio checkpoint ref changed during cleanup; project deletion stopped."
                )
            if output.strip().lower() != commit_sha.lower():
                raise AgentWorkspaceError(
                    "A Studio checkpoint ref changed after creation; project deletion stopped."
                )
            inspected.append(("checkpoint", checkpoint, True))

        for preparation in preparations:
            preparation_id = str(preparation["id"])
            expected_ref = f"refs/unsloth-studio/prepared-commits/{preparation_id}"
            ref_name = str(preparation["refName"])
            commit_sha = str(preparation["commitSha"])
            if ref_name != expected_ref or not _SAFE_OBJECT_ID.fullmatch(commit_sha):
                raise AgentWorkspaceError(
                    "Prepared commit ownership metadata is invalid; no Git refs were removed."
                )
            try:
                recorded_root = Path(preparation["gitRoot"]).resolve(strict = True)
            except (OSError, RuntimeError, ValueError) as exc:
                raise AgentWorkspaceError(
                    "Prepared commit repository identity is unavailable; project deletion stopped."
                ) from exc
            if recorded_root != repository or root != repository:
                raise AgentWorkspaceError(
                    "Prepared commit repository identity no longer matches; project deletion stopped."
                )
            code, _, _ = _run_git(
                repository,
                ["show-ref", "--verify", "--quiet", ref_name],
                timeout_seconds = 5,
                output_limit = 1024,
            )
            if code == 1:
                inspected.append(("preparation", preparation, False))
                continue
            if code != 0:
                raise AgentWorkspaceError(
                    "Git could not verify a prepared commit ref; project deletion stopped."
                )
            symbolic_code, _, _ = _run_git(
                repository,
                ["symbolic-ref", "--quiet", ref_name],
                timeout_seconds = 5,
                output_limit = 1024,
            )
            if symbolic_code == 0:
                raise AgentWorkspaceError(
                    "A prepared commit ref changed after creation; project deletion stopped."
                )
            if symbolic_code != 1:
                raise AgentWorkspaceError(
                    "Git could not verify a prepared commit ref; project deletion stopped."
                )
            code, output, _ = _run_git(
                repository,
                ["rev-parse", "--verify", "--quiet", ref_name],
                timeout_seconds = 5,
                output_limit = 1024,
            )
            if code != 0 or output.strip().lower() != commit_sha.lower():
                raise AgentWorkspaceError(
                    "A prepared commit ref changed after creation; project deletion stopped."
                )
            inspected.append(("preparation", preparation, True))

        removed = 0
        already_missing = 0
        for record_kind, record, ref_exists in inspected:
            if ref_exists:
                _git(
                    repository,
                    [
                        "update-ref",
                        "--no-deref",
                        "-d",
                        record["refName"],
                        record["commitSha"],
                    ],
                    output_limit = 1024,
                )
                removed += 1
            else:
                already_missing += 1
            if record_kind == "checkpoint":
                delete_checkpoint(str(record["id"]), project_id)
            else:
                delete_preparation(str(record["id"]), project_id)
        return {
            "projectId": project_id,
            "removed": removed,
            "alreadyMissing": already_missing,
        }


def rollback_checkpoint(
    project_id: str, checkpoint_id: str, expected_current_fingerprint: str
) -> dict:
    with _project_checkpoint_operation(project_id):
        root, _ = _project_git(project_id, mutation = True)
        with _workspace_writer_slot(root):
            return _rollback_checkpoint(project_id, checkpoint_id, expected_current_fingerprint)


def _rollback_checkpoint(
    project_id: str, checkpoint_id: str, expected_current_fingerprint: str
) -> dict:
    root, repository = _project_git(project_id, mutation = True)
    with _serialized_repository_mutation(repository):
        checkpoint = get_checkpoint(checkpoint_id)
        if checkpoint is None or checkpoint["projectId"] != project_id:
            raise AgentWorkspaceError("Studio checkpoint not found.")
        if Path(checkpoint["gitRoot"]).resolve() != repository:
            raise AgentWorkspaceError("Checkpoint repository identity no longer matches.")
        current = workspace_fingerprint(root)
        if not expected_current_fingerprint or current != expected_current_fingerprint:
            raise AgentWorkspaceError(
                "The repository changed after rollback was prepared. Refresh and review it again."
            )
        if not workspace_common.workspace_fingerprint_complete(
            current
        ) or not workspace_common.workspace_fingerprint_complete(expected_current_fingerprint):
            raise AgentWorkspaceError(
                "Repository evidence is incomplete. Reduce the change set before rollback."
            )
        paths = _owned_paths(root, list(checkpoint["ownedPaths"]))
        for path in paths:
            code, _, _ = _run_git(
                repository,
                ["cat-file", "-e", f"{checkpoint['commitSha']}:{path}"],
                timeout_seconds = 5,
                output_limit = 1024,
            )
            if code != 0:
                raise AgentWorkspaceError(
                    f"Checkpoint does not contain {path}; rollback will not delete it implicitly."
                )
        prepared_current = workspace_fingerprint(root)
        if prepared_current != expected_current_fingerprint:
            raise AgentWorkspaceError(
                "The repository changed while rollback was being prepared. "
                "Refresh and review it again."
            )
        if not workspace_common.workspace_fingerprint_complete(prepared_current):
            raise AgentWorkspaceError(
                "Repository evidence became incomplete while rollback was being prepared."
            )
        _git(
            repository,
            [
                "restore",
                f"--source={checkpoint['commitSha']}",
                "--worktree",
                "--",
                *_literal_pathspecs(paths),
            ],
            timeout_seconds = 60,
            neutralize_filters = True,
        )
        return {
            "checkpointId": checkpoint_id,
            "restoredPaths": paths,
            "fingerprint": workspace_fingerprint(root),
        }
