# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bounded Git primitives used by the worktree lifecycle.

Git reads and worktree mutations run with repository hooks, credential helpers,
filters, and merge drivers neutralized. The caller still owns higher-level
review and commit policy; this module only establishes a predictable process
boundary and validates the output it consumes.
"""

from __future__ import annotations

import os
import hashlib
import re
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Sequence

from .git_context import AgentWorkspaceError, project_workspace


_REF = re.compile(r"^[0-9a-fA-F]{40,64}$")
_MAX_OUTPUT = 512 * 1024
_GIT_TIMEOUT = 120
_REPOSITORY_FENCES = threading.local()


def _require_secure_git_platform() -> None:
    if os.name == "nt":
        raise AgentWorkspaceError(
            "Secure Git operations are disabled on Windows until a boundary test passes."
        )


def _safe_scope_root(repository: Path, scope_root: Optional[Path]) -> Path:
    """Resolve a project scope and prove it is inside the Git repository."""
    repository = Path(repository).resolve(strict = True)
    candidate = repository if scope_root is None else Path(scope_root).expanduser()
    try:
        candidate = candidate.resolve(strict = True)
        candidate.relative_to(repository)
    except (OSError, RuntimeError, ValueError) as exc:
        raise AgentWorkspaceError("The project folder is outside its Git repository.") from exc
    if not candidate.is_dir():
        raise AgentWorkspaceError("The project folder is not a directory.")
    relative = candidate.relative_to(repository)
    if ".git" in relative.parts:
        raise AgentWorkspaceError("The project folder cannot be inside Git metadata.")
    return candidate


def _scope_prefix(repository: Path, scope_root: Path) -> str:
    relative = _safe_scope_root(repository, scope_root).relative_to(repository)
    return "" if not relative.parts else relative.as_posix()


def _repository_paths(repository: Path, scope_root: Path, paths: Sequence[str]) -> list[str]:
    """Translate project-relative paths to repository-relative Git pathspecs."""
    prefix = _scope_prefix(repository, scope_root)
    return [f"{prefix}/{path}" if prefix else path for path in paths]


@contextmanager
def repository_fence(repository: Path):
    """Serialize transactions across every linked checkout of the same Git repository."""
    from .process_fence import _acquire_project_execution_fence, _release_project_execution_fence
    import time

    _require_secure_git_platform()
    root = Path(repository).resolve(strict = True)
    code, output, truncated = _run(root, ["rev-parse", "--git-common-dir"], output_limit = 32768)
    if code or truncated:
        raise AgentWorkspaceError("Git repository ownership is unavailable.")
    common = Path(output.strip())
    common = (
        (root / common).resolve(strict = True)
        if not common.is_absolute()
        else common.resolve(strict = True)
    )
    key = str(common)
    held = getattr(_REPOSITORY_FENCES, "held", set())
    if key in held:
        yield
        return
    try:
        descriptor = _acquire_project_execution_fence(
            "repository:" + key, None, time.monotonic() + 30
        )
    except (OSError, RuntimeError, TimeoutError) as exc:
        raise AgentWorkspaceError("The Git repository is busy or unavailable.") from exc
    _REPOSITORY_FENCES.held = held | {key}
    try:
        yield
    finally:
        _REPOSITORY_FENCES.held = held
        _release_project_execution_fence(descriptor)


def _trusted_git_executable() -> Path:
    from .git_review import _trusted_git_executable as executable
    return executable()


def _run(
    root: Path,
    args: Sequence[str],
    *,
    timeout: float = _GIT_TIMEOUT,
    output_limit: int = _MAX_OUTPUT,
    overrides: Sequence[str] = (),
    extra_env: Optional[dict[str, str]] = None,
) -> tuple[int, str, bool]:
    _require_secure_git_platform()
    if not args or any("\x00" in str(argument) for argument in args):
        raise AgentWorkspaceError("Git arguments are invalid.")
    from .git_review import _GitSession, _run_bounded

    with _GitSession(_trusted_git_executable()) as session:
        if args[0] != "config":
            session.inspect_executable_config(root)
        environment = dict(session.env)
        allowed = {
            "GIT_INDEX_FILE",
            "GIT_AUTHOR_NAME",
            "GIT_AUTHOR_EMAIL",
            "GIT_COMMITTER_NAME",
            "GIT_COMMITTER_EMAIL",
        }
        if extra_env:
            if not set(extra_env).issubset(allowed):
                raise AgentWorkspaceError("Git environment override is invalid.")
            environment.update(extra_env)
        environment.setdefault("GIT_AUTHOR_NAME", "Unsloth Studio")
        environment.setdefault("GIT_AUTHOR_EMAIL", "studio@localhost")
        environment.setdefault("GIT_COMMITTER_NAME", "Unsloth Studio")
        environment.setdefault("GIT_COMMITTER_EMAIL", "studio@localhost")
        result = _run_bounded(
            [*session._base(), *overrides, *map(str, args)],
            cwd = root,
            env = environment,
            output_limit = output_limit,
            timeout_seconds = timeout,
        )
        if result.timed_out:
            raise AgentWorkspaceError("Git operation timed out.")
        return result.code, result.output.decode("utf-8", errors = "replace"), result.overflowed


def _configured_driver_overrides(root: Path) -> list[str]:
    return []


def _configured_merge_overrides(root: Path) -> list[str]:
    return []


def git_root(root: Path) -> Path:
    requested = Path(root).expanduser()
    try:
        requested = requested.resolve(strict = True)
    except (OSError, RuntimeError, ValueError) as exc:
        raise AgentWorkspaceError("The project workspace is unavailable.") from exc
    code, output, truncated = _run(
        requested, ["rev-parse", "--show-toplevel"], timeout = 10, output_limit = 16 * 1024
    )
    if code != 0 or truncated:
        raise AgentWorkspaceError("The project folder is not a Git repository.")
    try:
        repository = Path(output.strip()).resolve(strict = True)
        requested.relative_to(repository)
    except (OSError, RuntimeError, ValueError) as exc:
        raise AgentWorkspaceError("Git returned an invalid repository root.") from exc
    return repository


def project_git(project_id: str) -> tuple[Path, Path]:
    workspace = project_workspace(project_id)
    repository = git_root(workspace.root)
    return workspace.root, repository


def worktree_entries(repository: Path) -> dict[str, dict[str, str | bool]]:
    code, output, truncated = _run(
        repository, ["worktree", "list", "--porcelain", "-z"], timeout = 20
    )
    if code != 0 or truncated:
        raise AgentWorkspaceError("Git worktree state could not be inspected safely.")
    entries: dict[str, dict[str, str | bool]] = {}
    current: dict[str, str | bool] = {}
    for field in output.split("\0"):
        if not field:
            if isinstance(current.get("path"), str):
                path = os.path.normcase(os.path.normpath(str(current["path"])))
                entries[path] = dict(current)
            current = {}
            continue
        if field.startswith("worktree "):
            current["path"] = field[9:]
        elif field.startswith("branch "):
            current["branch"] = field[7:]
        elif field == "detached":
            current["detached"] = True
    return entries


def repository_command(
    repository: Path,
    args: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    timeout: float = _GIT_TIMEOUT,
    output_limit: int = _MAX_OUTPUT,
    neutralize_filters: bool = False,
    neutralize_merge_drivers: bool = False,
    extra_env: Optional[dict[str, str]] = None,
) -> tuple[str, bool]:
    run_root = _safe_scope_root(repository, cwd)
    overrides: list[str] = []
    if neutralize_filters:
        overrides.extend(_configured_driver_overrides(repository))
    if neutralize_merge_drivers:
        overrides.extend(_configured_merge_overrides(repository))
    code, output, truncated = _run(
        run_root,
        args,
        timeout = timeout,
        output_limit = output_limit,
        overrides = overrides,
        extra_env = extra_env,
    )
    if code != 0:
        raise AgentWorkspaceError("Git operation failed. Refresh repository review.")
    return output, truncated


def repository_status(repository: Path, *, scope_root: Optional[Path] = None) -> str:
    scope = _safe_scope_root(repository, scope_root)
    args = ["status", "--porcelain=v1", "-z", "--untracked-files=all"]
    if scope != Path(repository).resolve(strict = True):
        prefix = _scope_prefix(repository, scope)
        args.extend(["--", ":(top,literal)" + prefix])
    output, truncated = repository_command(
        repository,
        args,
        cwd = repository,
        timeout = 20,
        output_limit = 1_000_000,
        neutralize_filters = True,
    )
    if truncated or "\ufffd" in output:
        raise AgentWorkspaceError("Git status is too large to review safely.")
    return output


def _pathspec(path: str) -> str:
    value = str(path)
    if not value or "\x00" in value or "\r" in value or "\n" in value:
        raise AgentWorkspaceError("Git path is invalid.")
    if value.startswith("/") or value.startswith("\\") or re.match(r"^[A-Za-z]:", value):
        raise AgentWorkspaceError("Git paths must be relative to the project root.")
    parts = value.replace("\\", "/").split("/")
    if any(part in {"", ".", "..", ".git"} for part in parts):
        raise AgentWorkspaceError("Git paths cannot escape the project root.")
    return ":(literal)" + value.replace("\\", "/")


def _status_records(
    raw: str,
    *,
    repository: Optional[Path] = None,
    scope_root: Optional[Path] = None,
) -> tuple[list[dict], dict[str, int]]:
    records = [value for value in raw.split("\0") if value]
    files: list[dict] = []
    counts = {"staged": 0, "unstaged": 0, "untracked": 0, "conflicts": 0}
    index = 0
    while index < len(records):
        value = records[index]
        index += 1
        if len(value) < 3 or value[2] != " ":
            raise AgentWorkspaceError("Git returned invalid status output.")
        code = value[:2]
        path = value[3:]
        entry = {"code": code, "path": path}
        if code[0] not in {" ", "?"}:
            counts["staged"] += 1
        if code[1] not in {" ", "?"}:
            counts["unstaged"] += 1
        if code == "??":
            counts["untracked"] += 1
        if code in {"DD", "AU", "UD", "UA", "DU", "AA", "UU"}:
            counts["conflicts"] += 1
        if code[0] in {"R", "C"} or code[1] in {"R", "C"}:
            if index >= len(records):
                raise AgentWorkspaceError("Git returned an incomplete rename record.")
            # Porcelain v1 -z emits the new path in the status record followed
            # by the original path as a second NUL-delimited field.
            entry["oldPath"] = records[index]
            index += 1
        if repository is not None and scope_root is not None:
            prefix = _scope_prefix(repository, scope_root)
            prefix_with_separator = f"{prefix}/" if prefix else ""
            if prefix_with_separator:
                if not entry["path"].startswith(prefix_with_separator):
                    continue
                entry["path"] = entry["path"][len(prefix_with_separator) :]
                if "oldPath" in entry:
                    old_path = str(entry["oldPath"])
                    entry["oldPath"] = (
                        old_path[len(prefix_with_separator) :]
                        if old_path.startswith(prefix_with_separator)
                        else old_path
                    )
        files.append(entry)
    return files, counts


def git_status(project_id: str) -> dict:
    workspace, repository = project_git(project_id)
    raw = repository_status(repository, scope_root = workspace)
    files, counts = _status_records(raw, repository = repository, scope_root = workspace)
    branch = None
    detached = False
    try:
        branch = repository_branch(repository)
    except AgentWorkspaceError:
        detached = True
    return {
        "head": repository_head(repository),
        "branch": branch,
        "detached": detached,
        "clean": not files,
        "counts": counts,
        "files": files,
        "truncated": False,
        "root": str(workspace),
    }


def git_diff(
    project_id: str,
    *,
    staged: bool = False,
    max_bytes: int = 512 * 1024,
) -> dict:
    workspace, repository = project_git(project_id)
    output, truncated = git_diff_for_root(
        repository,
        staged = staged,
        scope_root = workspace,
        max_bytes = max_bytes,
    )
    return {"diff": output, "truncated": truncated}


def workspace_fingerprint(root: Path) -> str:
    """Hash two coherent captures, including index, worktree, config and untracked bytes."""
    from .git_context import ProjectWorkspace
    from .git_review import _GitSession, _discover_repository, _coherent_capture

    root = root.resolve(strict = True)
    metadata = root.stat()
    workspace = ProjectWorkspace("fingerprint", root, "managed", metadata.st_dev, metadata.st_ino)
    digest = hashlib.sha256()
    digest.update(f"{metadata.st_dev}:{metadata.st_ino}".encode())
    complete = True
    with _GitSession(_trusted_git_executable()) as session:
        repository = _discover_repository(session, workspace)
        for mode in ("head", "staged", "unstaged"):
            try:
                capture, coherent = _coherent_capture(
                    session,
                    workspace,
                    repository,
                    mode = mode,
                    max_bytes = 2_000_000,
                )
            except OverflowError:
                return "badc0ffe" + "0" * 56
            complete = (
                complete
                and coherent
                and all(value.startswith(b"content\0") for _, value in capture.untracked)
            )
            digest.update(mode.encode())
            digest.update(capture.fingerprint().encode())
            digest.update(capture.config)
    prefix = "c0dec0de" if complete else "badc0ffe"
    return prefix + digest.hexdigest()[len(prefix) :]


def git_diff_for_root(
    repository: Path,
    *,
    staged: bool,
    scope_root: Optional[Path] = None,
    max_bytes: int = 2_000_000,
) -> tuple[str, bool]:
    args = ["diff", "--no-ext-diff", "--no-textconv", "--binary"]
    if staged:
        args.append("--cached")
    if scope_root is not None:
        prefix = _scope_prefix(repository, scope_root)
        if prefix:
            args.extend(["--", f":(top,literal){prefix}"])
    return repository_command(
        repository,
        args,
        timeout = 30,
        output_limit = max(4_096, min(int(max_bytes), 2_000_000)),
        neutralize_filters = True,
    )


def build_selected_commit(
    repository: Path,
    paths: list[str],
    message: str,
    *,
    allow_unborn: bool = False,
) -> str:
    if not paths or not message.strip() or "\x00" in message:
        raise AgentWorkspaceError("A checkpoint requires selected paths and a message.")
    pathspecs = [_pathspec(path) for path in paths]
    base = repository_head(repository, allow_unborn = allow_unborn)
    with tempfile.NamedTemporaryFile(prefix = "unsloth-index-", delete = False) as handle:
        index_path = handle.name
    try:
        env = {
            "GIT_INDEX_FILE": index_path,
            "GIT_AUTHOR_NAME": "Unsloth Studio",
            "GIT_AUTHOR_EMAIL": "studio@localhost",
            "GIT_COMMITTER_NAME": "Unsloth Studio",
            "GIT_COMMITTER_EMAIL": "studio@localhost",
        }
        repository_command(
            repository,
            ["read-tree", base] if base else ["read-tree", "--empty"],
            neutralize_filters = True,
            extra_env = env,
        )
        repository_command(
            repository,
            ["add", "-A", "--", *pathspecs],
            timeout = 60,
            neutralize_filters = True,
            extra_env = env,
        )
        tree, _ = repository_command(
            repository,
            ["write-tree"],
            timeout = 10,
            output_limit = 256,
            neutralize_filters = True,
            extra_env = env,
        )
        tree_sha = tree.strip()
        if _REF.fullmatch(tree_sha) is None:
            raise AgentWorkspaceError("Git returned an invalid checkpoint tree.")
        commit, _ = repository_command(
            repository,
            [
                "commit-tree",
                tree_sha,
                "-m",
                message,
                *(["-p", base] if base else []),
            ],
            timeout = 30,
            output_limit = 256,
            neutralize_filters = True,
            extra_env = env,
        )
        commit_sha = commit.strip()
        if _REF.fullmatch(commit_sha) is None:
            raise AgentWorkspaceError("Git returned an invalid checkpoint commit.")
        return commit_sha
    finally:
        try:
            Path(index_path).unlink()
        except OSError:
            pass


def update_ref(repository: Path, ref_name: str, commit_sha: str) -> None:
    if (
        not ref_name.startswith("refs/unsloth-studio/checkpoints/")
        or _REF.fullmatch(commit_sha) is None
    ):
        raise AgentWorkspaceError("Checkpoint reference is invalid.")
    repository_command(
        repository,
        ["update-ref", "--no-deref", ref_name, commit_sha, "0" * len(commit_sha)],
        timeout = 10,
        output_limit = 1024,
    )


def delete_ref(
    repository: Path,
    ref_name: str,
    expected_sha: Optional[str] = None,
) -> None:
    if not ref_name.startswith("refs/unsloth-studio/checkpoints/"):
        raise AgentWorkspaceError("Checkpoint reference is invalid.")
    args = ["update-ref", "--no-deref", "-d", ref_name]
    if expected_sha is not None:
        if _REF.fullmatch(expected_sha) is None:
            raise AgentWorkspaceError("Checkpoint reference is invalid.")
        args.append(expected_sha)
    repository_command(repository, args, timeout = 10, output_limit = 1024)


def restore_selected_paths(repository: Path, commit_sha: str, paths: list[str]) -> None:
    if _REF.fullmatch(commit_sha) is None or not paths:
        raise AgentWorkspaceError("Checkpoint restore request is invalid.")
    repository_command(
        repository,
        [
            "restore",
            "--source",
            commit_sha,
            "--worktree",
            "--",
            *[_pathspec(path) for path in paths],
        ],
        timeout = 60,
        neutralize_filters = True,
    )


def repository_head(repository: Path, *, allow_unborn: bool = False) -> str:
    code, output, truncated = _run(
        repository,
        ["rev-parse", "--verify", "--quiet", "HEAD^{commit}"],
        timeout = 10,
        output_limit = 256,
    )
    if allow_unborn and code == 1 and not output.strip() and not truncated:
        branch, branch_truncated = repository_command(
            repository,
            ["symbolic-ref", "--quiet", "HEAD"],
            timeout = 10,
            output_limit = 4096,
        )
        branch_ref = branch.strip()
        if not branch_truncated and branch_ref.startswith("refs/heads/"):
            ref_code, ref_output, ref_truncated = _run(
                repository,
                ["show-ref", "--verify", "--quiet", branch_ref],
                timeout = 10,
                output_limit = 256,
            )
            if ref_code == 1 and not ref_output.strip() and not ref_truncated:
                # Persist an empty head in the existing non-null preparation schema.
                return ""
    value = output.strip()
    if code != 0 or truncated or _REF.fullmatch(value) is None:
        raise AgentWorkspaceError("Git returned an invalid repository head.")
    return value


def repository_branch(repository: Path) -> str:
    output, truncated = repository_command(
        repository,
        ["symbolic-ref", "--quiet", "--short", "HEAD"],
        timeout = 10,
        output_limit = 4096,
    )
    value = output.strip()
    if truncated or not value or "\x00" in value or "\n" in value:
        raise AgentWorkspaceError("Git returned an invalid branch name.")
    return value


def repository_ref(repository: Path, ref: str) -> str:
    value = repository_ref_optional(repository, ref)
    if value is None:
        raise AgentWorkspaceError("Git reference is invalid.")
    return value


def repository_ref_optional(repository: Path, ref: str) -> Optional[str]:
    if not ref or any(character in ref for character in "\x00\r\n"):
        raise AgentWorkspaceError("Git reference is invalid.")
    code, output, truncated = _run(
        repository,
        ["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        timeout = 10,
        output_limit = 256,
    )
    if code == 1 and not output.strip() and not truncated:
        return None
    if code != 0:
        raise AgentWorkspaceError("Git reference is invalid.")
    value = output.strip()
    if truncated or _REF.fullmatch(value) is None:
        raise AgentWorkspaceError("Git reference is invalid.")
    return value


def preflight_merge(repository: Path, target: str, source: str) -> tuple[int, str, bool]:
    """Run merge-tree without touching the checkout.

    Git returns 1 for a real conflict, so this deliberately exposes the exit
    code instead of routing it through ``repository_command``.
    """
    overrides = _configured_driver_overrides(repository)
    overrides.extend(_configured_merge_overrides(repository))
    return _run(
        repository,
        ["merge-tree", "--write-tree", "--name-only", target, source],
        timeout = 60,
        output_limit = 128 * 1024,
        overrides = overrides,
    )


def add_worktree(repository: Path, path: Path, branch: str, base_ref: str) -> None:
    overrides = _configured_driver_overrides(repository)
    code, output, truncated = _run(
        repository,
        ["worktree", "add", "-b", branch, str(path), base_ref],
        timeout = 120,
        output_limit = 128 * 1024,
        overrides = overrides,
    )
    if code != 0:
        detail = output.strip()[:4096]
        raise AgentWorkspaceError(
            f"Git worktree creation failed.{(' ' + detail) if detail else ''}"
        )
    if truncated:
        raise AgentWorkspaceError("Git worktree creation output was truncated.")


def remove_worktree(repository: Path, path: Path) -> None:
    repository_command(
        repository,
        ["worktree", "remove", str(path)],
        timeout = 120,
        output_limit = 128 * 1024,
        neutralize_filters = True,
    )


def merge_worktree(repository: Path, source_ref: str) -> tuple[str, str]:
    if _REF.fullmatch(source_ref) is None:
        raise AgentWorkspaceError("Worktree source identity is invalid.")
    overrides = _configured_driver_overrides(repository)
    overrides.extend(_configured_merge_overrides(repository))
    target_code, target_output, target_truncated = _run(
        repository,
        ["rev-parse", "--verify", "HEAD^{commit}"],
        timeout = 10,
        output_limit = 256,
        overrides = overrides,
    )
    if target_code != 0 or target_truncated or _REF.fullmatch(target_output.strip()) is None:
        raise AgentWorkspaceError("The target repository head is invalid.")
    target_head = target_output.strip()
    code, output, truncated = _run(
        repository,
        ["merge", "--no-ff", "--no-edit", source_ref],
        timeout = 120,
        output_limit = 128 * 1024,
        overrides = overrides,
    )
    if code != 0:
        detail = output.strip()[:4096]
        raise AgentWorkspaceError(
            f"Git could not merge the owned worktree.{(' ' + detail) if detail else ''}"
        )
    if truncated:
        raise AgentWorkspaceError("Git merge output was truncated.")
    result, result_truncated = repository_command(
        repository, ["rev-parse", "--verify", "HEAD^{commit}"], timeout = 10, output_limit = 256
    )
    if result_truncated or _REF.fullmatch(result.strip()) is None:
        raise AgentWorkspaceError("Git returned an invalid merge identity.")
    return target_head, result.strip()


__all__ = [
    "add_worktree",
    "build_selected_commit",
    "delete_ref",
    "git_diff",
    "git_status",
    "git_root",
    "merge_worktree",
    "project_git",
    "preflight_merge",
    "repository_branch",
    "remove_worktree",
    "repository_command",
    "repository_head",
    "repository_ref",
    "repository_ref_optional",
    "repository_status",
    "restore_selected_paths",
    "update_ref",
    "workspace_fingerprint",
    "worktree_entries",
]
