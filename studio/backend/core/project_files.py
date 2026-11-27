# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import heapq
import os
import re
import shutil
import stat
import subprocess
import time
from collections import deque
from typing import Optional

from utils.paths.sensitive import is_sensitive_path_component

LIST_MAX_ENTRIES = 500
LIST_MAX_VISITED = 20000
SEARCH_MAX_DIRS = 5000
SEARCH_MAX_FILES = 50000
SEARCH_MAX_ENTRIES = 200000
SEARCH_BUDGET_SECONDS = 2.0
SEARCH_MAX_LIMIT = 200
CHANGES_MAX = 500
GIT_TIMEOUT_SECONDS = 5.0
_MAX_SEGMENT = 255
_MAX_RELATIVE_PATH = 4096
_SKIP_DIRS = frozenset({"node_modules", "__pycache__"})
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_DRIVE_RE = re.compile(r"\A[A-Za-z]:")
_ENV_KEYS = ("PATH", "HOME", "LANG", "SYSTEMROOT")
_CREATE_NO_WINDOW = 0x08000000


class ProjectPathError(ValueError):
    """A requested path that is not a folder inside the root; routes answer 404 for it."""


def _within(root: str, path: str) -> bool:
    root, path = os.path.normcase(root), os.path.normcase(path)
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def _hidden_name(name: str) -> bool:
    return is_sensitive_path_component(name) or bool(_CONTROL_RE.search(name))


def relative_parts(path: Optional[str]) -> list[str]:
    if not path:
        return []
    if len(path) > _MAX_RELATIVE_PATH or _CONTROL_RE.search(path):
        raise ProjectPathError(path)
    if (
        path[0] in "/\\"
        or _DRIVE_RE.match(path)
        or os.path.isabs(path)
        or os.path.splitdrive(path)[0]
    ):
        raise ProjectPathError(path)
    parts = []
    for segment in re.split(r"[\\/]", path):
        if segment in ("", "."):
            continue
        if (
            segment == ".."
            or segment.lower() == ".git"
            or len(segment) > _MAX_SEGMENT
            or is_sensitive_path_component(segment)
        ):
            raise ProjectPathError(path)
        parts.append(segment)
    return parts


def contained_dir(root: str, parts: list[str]) -> str:
    target = os.path.realpath(os.path.join(root, *parts))
    if not _within(root, target) or not os.path.isdir(target):
        raise ProjectPathError("/".join(parts))
    return target


def list_directory(
    root: str,
    path: Optional[str] = "",
    show_hidden: bool = False,
    skip_names: frozenset = frozenset(),
) -> dict:
    """One level of root/path; skip_names applies at the root only (the managed sandbox's bookkeeping)."""
    parts = relative_parts(path)
    empty = {"path": "/".join(parts), "entries": [], "truncated": False}
    if not os.path.isdir(root):
        if parts:
            raise ProjectPathError(path)
        return empty
    root = os.path.realpath(root)
    target = contained_dir(root, parts)
    entries = []
    visited = 0
    truncated = False
    try:
        with os.scandir(target) as listing:
            for entry in listing:
                visited += 1
                if visited > LIST_MAX_VISITED:
                    truncated = True
                    break
                name = entry.name
                hidden = name.startswith(".")
                if name == ".git" or (hidden and not show_hidden) or _hidden_name(name):
                    continue
                if not parts and name in skip_names:
                    continue
                try:
                    if entry.is_symlink():
                        real = os.path.realpath(entry.path)
                        if not _within(root, real):
                            continue
                        st = os.stat(real)
                    else:
                        st = entry.stat(follow_symlinks = False)
                except OSError:
                    continue
                if stat.S_ISDIR(st.st_mode):
                    entries.append({"name": name, "kind": "dir", "hidden": hidden})
                elif stat.S_ISREG(st.st_mode):
                    entries.append(
                        {"name": name, "kind": "file", "size": st.st_size, "hidden": hidden}
                    )
    except OSError as exc:
        raise ProjectPathError(path) from exc
    entries.sort(key = lambda e: (e["kind"] != "dir", e["name"].casefold(), e["name"]))
    if len(entries) > LIST_MAX_ENTRIES:
        entries = entries[:LIST_MAX_ENTRIES]
        truncated = True
    return {"path": "/".join(parts), "entries": entries, "truncated": truncated}


def _is_subsequence(needle: str, haystack: str) -> bool:
    it = iter(haystack)
    return all(ch in it for ch in needle)


def match_rank(rel_path: str, query: str) -> Optional[int]:
    name = rel_path.rsplit("/", 1)[-1]
    subject = rel_path if "/" in query else name
    if subject == query:
        return 0
    if subject.startswith(query):
        return 1
    if "/" not in query and query in name:
        return 2
    if query in rel_path:
        return 3
    if _is_subsequence(query, rel_path):
        return 4
    return None


def _skips_dir(base: str, name: str) -> bool:
    return name in _SKIP_DIRS or os.path.isfile(os.path.join(base, name, "pyvenv.cfg"))


def search_files(
    root: str,
    query: str,
    limit: int = 50,
    *,
    clock = time.monotonic,
) -> dict:
    query = (query or "").strip().replace("\\", "/").lower()
    limit = max(1, min(int(limit), SEARCH_MAX_LIMIT))
    if not query or not os.path.isdir(root):
        return {"results": [], "truncated": False}
    root = os.path.realpath(root)
    deadline = clock() + SEARCH_BUDGET_SECONDS
    pending = deque([("", root)])
    dirs_seen = 0
    entries_seen = 0
    files_seen = 0
    truncated = False
    scored = []
    while pending and not truncated:
        if dirs_seen >= SEARCH_MAX_DIRS or clock() > deadline:
            truncated = True
            break
        rel_dir, abs_dir = pending.popleft()
        dirs_seen += 1
        try:
            listing = os.scandir(abs_dir)
        except OSError:
            continue
        with listing:
            for entry in listing:
                entries_seen += 1
                if entries_seen > SEARCH_MAX_ENTRIES or clock() > deadline:
                    truncated = True
                    break
                name = entry.name
                if name.startswith(".") or _hidden_name(name):
                    continue
                rel = f"{rel_dir}/{name}" if rel_dir else name
                try:
                    if entry.is_symlink():
                        # Never descend a link; a linked file counts only while it stays inside the root.
                        real = os.path.realpath(entry.path)
                        if not _within(root, real) or not os.path.isfile(real):
                            continue
                    elif entry.is_dir(follow_symlinks = False):
                        if dirs_seen + len(pending) >= SEARCH_MAX_DIRS:
                            truncated = True
                        elif not _skips_dir(abs_dir, name):
                            pending.append((rel, entry.path))
                        continue
                    elif not entry.is_file(follow_symlinks = False):
                        continue
                except OSError:
                    continue
                files_seen += 1
                if files_seen > SEARCH_MAX_FILES:
                    truncated = True
                    break
                rank = match_rank(rel.lower(), query)
                if rank is not None:
                    scored.append((rank, len(rel), rel.casefold(), rel, name))
    best = heapq.nsmallest(limit, scored)
    return {
        "results": [{"path": rel, "name": name} for _, _, _, rel, name in best],
        "truncated": truncated,
    }


def _git_env(root: str) -> dict:
    env = {key: os.environ[key] for key in _ENV_KEYS if key in os.environ}
    env.update(
        {
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            # A broken root/.git must not make git walk up and report an enclosing repository.
            "GIT_CEILING_DIRECTORIES": os.path.dirname(root),
        }
    )
    return env


def _account_confinement():
    """A managed account's tools write this folder, so its git runs under their confinement too;
    None for the owner (unchanged) and where the account's tools cannot run anyway."""
    if os.name == "nt":
        return None
    try:
        from core.inference.tool_confinement import account_confinement
        from core.inference.tools import _SANDBOX_SITE_DIR
        confinement = account_confinement(_SANDBOX_SITE_DIR)
    except Exception:
        return None
    return confinement if confinement is not None and confinement.confines else None


def _run_git(git: str, root: str, args: list[str], config: list[str], deadline: float) -> bytes:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise subprocess.TimeoutExpired(git, 0)
    flags = []
    for item in [
        "core.fsmonitor=false",
        "core.hooksPath=/dev/null",
        *config,
    ]:
        flags += ["-c", item]
    argv = [git, *flags, *args]
    env = _git_env(root)
    confinement = _account_confinement()
    preexec = None
    if confinement is not None:
        argv = confinement.wrap(argv)
        preexec = confinement.preexec
        # The confined child cannot read the service user's git config or ignore file; unreadable is fatal.
        env.update(
            {
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": os.devnull,
                "HOME": os.devnull,
                "XDG_CONFIG_HOME": os.devnull,
            }
        )
    result = subprocess.run(
        argv,
        cwd = root,
        env = env,
        preexec_fn = preexec,
        stdin = subprocess.DEVNULL,
        stdout = subprocess.PIPE,
        stderr = subprocess.DEVNULL,
        timeout = remaining,
        shell = False,
        check = False,
        # Desktop's backend has no console on Windows, so each call would flash a window.
        creationflags = _CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, args)
    return result.stdout


def _disabled_filters(git: str, root: str, deadline: float) -> list[str]:
    try:
        out = _run_git(git, root, ["config", "-z", "--get-regexp", r"^filter\."], [], deadline)
    except subprocess.CalledProcessError as exc:
        if exc.returncode == 1:
            return []
        raise
    names = set()
    for record in out.split(b"\0"):
        key = record.split(b"\n", 1)[0].decode("utf-8", "surrogateescape")
        if key.startswith("filter.") and key.count(".") >= 2:
            name = key[len("filter.") :].rsplit(".", 1)[0]
            if "=" in name:
                # `-c` splits at the first '=', so this driver could not be switched off.
                raise ValueError(f"unsupported git filter name {name!r}")
            names.add(name)
    config = []
    for name in sorted(names):
        for var, value in (("clean", ""), ("smudge", ""), ("process", ""), ("required", "false")):
            config.append(f"filter.{name}.{var}={value}")
    return config


def change_status(xy: str) -> Optional[str]:
    if xy == "??":
        return "untracked"
    if xy == "!!":
        return None
    if "U" in xy or xy in ("DD", "AA"):
        return "conflicted"
    if "R" in xy:
        return "renamed"
    if "D" in xy:
        return "deleted"
    if "A" in xy or "C" in xy:
        return "added"
    return "modified"


def parse_porcelain(data: bytes) -> list[dict]:
    fields = data.split(b"\0")
    changes = []
    i = 0
    while i < len(fields):
        field = fields[i]
        i += 1
        if len(field) < 4:
            continue
        xy = field[:2].decode("ascii", "replace")
        path = field[3:].decode("utf-8", "replace")
        if "R" in xy or "C" in xy:
            i += 1
        status = change_status(xy)
        if status is not None:
            changes.append({"path": path, "status": status})
    return changes


def git_changes(root: str) -> dict:
    none = {"repo": False, "branch": None, "changes": [], "truncated": False}
    if not os.path.isdir(root):
        return none
    root = os.path.realpath(root)
    if not os.path.lexists(os.path.join(root, ".git")):
        return none
    git = shutil.which("git")
    if not git:
        return none
    from utils.paths.folder_info import git_head_branch

    deadline = time.monotonic() + GIT_TIMEOUT_SECONDS
    try:
        filters = _disabled_filters(git, root, deadline)
        out = _run_git(
            git,
            root,
            [
                "status",
                "--porcelain=v1",
                "-z",
                "--untracked-files=all",
                "--ignore-submodules=dirty",
            ],
            filters,
            deadline,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        return none
    changes = [
        c
        for c in parse_porcelain(out)
        if not any(_hidden_name(part) for part in c["path"].split("/"))
    ]
    truncated = len(changes) > CHANGES_MAX
    return {
        "repo": True,
        "branch": git_head_branch(root),
        "changes": changes[:CHANGES_MAX],
        "truncated": truncated,
    }
