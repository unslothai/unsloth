# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import os
import re
import stat
from typing import Optional

_HEAD_BYTES = 512
_SHA_RE = re.compile(r"\A[0-9a-f]{40,64}\Z")


def tilde_path(path: str, home: Optional[str] = None) -> str:
    home = home if home is not None else os.path.expanduser("~")
    if not path or not home or home == os.path.dirname(home):
        return path
    base = home.rstrip("\\/")
    folded, folded_base = os.path.normcase(path), os.path.normcase(base)
    if folded == folded_base:
        return "~"
    if folded.startswith(folded_base) and path[len(base)] in ("/", "\\"):
        return "~" + path[len(base) :]
    return path


def _read_small_file(path: str) -> Optional[str]:
    # A FIFO named HEAD would block the open; only a regular file is read.
    try:
        if not stat.S_ISREG(os.stat(path).st_mode):
            return None
        with open(path, "r", encoding = "utf-8", errors = "replace") as handle:
            return handle.read(_HEAD_BYTES)
    except OSError:
        return None


def git_head_branch(folder: str) -> Optional[str]:
    git = os.path.join(folder, ".git")
    try:
        is_pointer = os.path.isfile(git)
    except OSError:
        return None
    if is_pointer:
        # A linked worktree or submodule: `.git` is a file naming the real git dir.
        pointer = _read_small_file(git) or ""
        match = re.match(r"gitdir:\s*(.+)", pointer.strip())
        if not match:
            return None
        git = os.path.normpath(os.path.join(folder, match.group(1).strip()))
    head = (_read_small_file(os.path.join(git, "HEAD")) or "").strip()
    if head.startswith("ref: refs/heads/"):
        branch = head[len("ref: refs/heads/") :].splitlines()[0].strip()
        return branch[:120] or None
    if _SHA_RE.match(head):
        return head[:7]
    return None
