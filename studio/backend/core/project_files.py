# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read-only views of the folder a project's tools run in: one directory level, a file search and git changes."""

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
    """The segments of a client-supplied relative path, or ProjectPathError for anything that could leave the root."""
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
        # `.git` is hidden as an entry, so it is not a place to open either.
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
    """Real path of root/parts, which must be a directory at or under root."""
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
    """0 exact name, 1 name prefix, 2 name contains, 3 path contains, 4 path subsequence; None for no match.
    Both arguments are lower-cased with "/" separators; a query with "/" is matched against the whole path."""
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
    # A virtualenv is marked by its pyvenv.cfg, whatever it is named.
    return name in _SKIP_DIRS or os.path.isfile(os.path.join(base, name, "pyvenv.cfg"))


def search_files(
    root: str,
    query: str,
    limit: int = 50,
    *,
    clock = time.monotonic,
) -> dict:
    """Files under root ranked by match_rank, then shorter path, then name; the walk is bounded."""
    query = (query or "").strip().replace("\\", "/").lower()
    limit = max(1, min(int(limit), SEARCH_MAX_LIMIT))
    if not query or not os.path.isdir(root):
        return {"results": [], "truncated": False}
    root = os.path.realpath(root)
    deadline = clock() + SEARCH_BUDGET_SECONDS
    pending = deque([("", root)])
    dirs_seen = 0
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
            with os.scandir(abs_dir) as listing:
                children = sorted(listing, key = lambda e: e.name)
        except OSError:
            continue
        for entry in children:
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
                    if not _skips_dir(abs_dir, name):
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
    result = subprocess.run(
        [git, *flags, *args],
        cwd = root,
        env = _git_env(root),
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
    """Blank every configured filter driver: `git status` runs a clean filter to compare a stat-dirty file,
    and the repository's own config names that program."""
    try:
        out = _run_git(git, root, ["config", "-z", "--get-regexp", r"^filter\."], [], deadline)
    except subprocess.CalledProcessError as exc:
        # Exit 1 means no key matched.
        if exc.returncode == 1:
            return []
        raise
    names = set()
    for record in out.split(b"\0"):
        key = record.split(b"\n", 1)[0].decode("utf-8", "surrogateescape")
        if key.startswith("filter.") and key.count(".") >= 2:
            names.add(key[len("filter.") :].rsplit(".", 1)[0])
    config = []
    for name in sorted(names):
        for var, value in (("clean", ""), ("smudge", ""), ("process", ""), ("required", "false")):
            config.append(f"filter.{name}.{var}={value}")
    return config


def change_status(xy: str) -> Optional[str]:
    """Porcelain v1 XY code to the label the Files tab shows; None for ignored entries."""
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
    """Entries of `git status --porcelain=v1 -z`: "XY path", and for a rename or copy the source path as the next field."""
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
    """Changed files of the repository rooted at root; any failure reads as "not a repository"."""
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
    except (OSError, subprocess.SubprocessError):
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
