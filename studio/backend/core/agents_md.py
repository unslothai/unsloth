# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AGENTS.md instructions for Studio chats: Studio's own global file, then the project folder's, as Codex reads them."""

from __future__ import annotations

import os
import stat
import threading
from pathlib import Path
from typing import Optional

from utils.account_context import is_owner_context
from utils.paths import workspace_root

# Codex's project_doc_max_bytes default, for the global and project text together.
MAX_AGENTS_MD_BYTES = 32 * 1024
TRUNCATED_NOTE = "\n[AGENTS.md truncated at 32 KiB]"
_DISABLE_ENV = "UNSLOTH_STUDIO_AGENTS_MD"

_LOCK = threading.Lock()
_CACHE: dict[str, tuple[tuple[int, int], bytes]] = {}


def agents_md_enabled() -> bool:
    return os.environ.get(_DISABLE_ENV, "").strip().lower() not in ("0", "false", "no", "off")


def _owner_home() -> Path:
    return Path.home()


def _read(path: Path, confine: Optional[Path] = None) -> bytes:
    # The model writes in the sandbox but this read is unsandboxed: no link there may pull in a host file
    # (project and global files may be links, CLAUDE.md -> AGENTS.md). Non-blocking so a FIFO cannot hang.
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    if confine is not None:
        if os.path.islink(path):
            return b""
        if os.path.dirname(os.path.realpath(path)) != os.path.realpath(confine):
            return b""
        flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return b""
    with os.fdopen(fd, "rb") as handle:
        try:
            status = os.fstat(handle.fileno())
            # A hard link is the other way to plant a host file in the sandbox.
            if not stat.S_ISREG(status.st_mode) or (confine is not None and status.st_nlink > 1):
                return b""
            key = (status.st_mtime_ns, status.st_size)
            cache_key = os.fspath(path)
            with _LOCK:
                hit = _CACHE.get(cache_key)
            if hit is not None and hit[0] == key:
                return hit[1]
            # One byte over the cap is enough to know it was cut.
            raw = handle.read(MAX_AGENTS_MD_BYTES + 1)
        except OSError:
            return b""
    with _LOCK:
        _CACHE[cache_key] = (key, raw)
    return raw


def _first(paths: tuple[Path, ...], confine: Optional[Path] = None) -> tuple[Optional[Path], bytes]:
    # A later name only stands in for a missing or empty earlier one, never adds to it.
    for path in paths:
        raw = _read(path, confine).strip()
        if raw:
            return path, raw
    return None, b""


def _label(path: Path) -> str:
    # Tells the model which file to edit without sending the account's home directory to a cloud provider.
    home = os.path.abspath(_owner_home())
    text = os.path.abspath(path)
    if text == home or text.startswith(home + os.sep):
        text = "~" + text[len(home) :]
    return text.replace(os.sep, "/")


def _sources(project: Optional[dict]) -> list[tuple[tuple[Path, ...], Optional[Path]]]:
    # Studio's own file for every account: another tool's ~/.claude or ~/.codex rules would change existing chats.
    sources = [((workspace_root() / "AGENTS.md",), None)]
    # Host project folders are single-user only (see tools._get_project_workdir).
    if (
        not project
        or project.get("archived")
        or not is_owner_context()
        or not project.get("rootPath")
    ):
        return sources
    root = Path(project["rootPath"])
    sources.append(((root / "AGENTS.md", root / "CLAUDE.md"), None))
    # Where the chat's tools run, so the model can keep notes there; after the user's file, which it refines.
    sandbox_path = project.get("sandboxPath")
    if sandbox_path and os.path.realpath(sandbox_path) != os.path.realpath(root):
        sandbox = Path(sandbox_path)
        sources.append(((sandbox / "AGENTS.md", sandbox / "CLAUDE.md"), sandbox))
    return sources


def agents_md_text(project: Optional[dict]) -> str:
    """Global then project AGENTS.md, each under its source, capped at 32 KiB; "" when there is none."""
    if not agents_md_enabled():
        return ""
    sections = []
    for candidates, confine in _sources(project):
        path, raw = _first(candidates, confine)
        if path is not None:
            sections.append(f"# Source: {_label(path)}\n\n".encode("utf-8") + raw)
    if not sections:
        return ""
    joined = b"\n\n".join(sections)
    truncated = len(joined) > MAX_AGENTS_MD_BYTES
    text = joined[:MAX_AGENTS_MD_BYTES].decode("utf-8", errors = "ignore").rstrip()
    return text + TRUNCATED_NOTE if truncated else text
