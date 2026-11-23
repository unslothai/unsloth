# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""AGENTS.md instructions for Studio chats: Studio's own global file, then the project folder's, as Codex reads them."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Optional

from utils.account_context import is_owner_context
from utils.paths import ensure_dir, workspace_root

# Codex's project_doc_max_bytes default, for the global and project text together.
MAX_AGENTS_MD_BYTES = 32 * 1024
TRUNCATED_NOTE = "\n[AGENTS.md truncated at 32 KiB]"
_DISABLE_ENV = "UNSLOTH_STUDIO_AGENTS_MD"
MEMORY_ENABLED_KEY = "chat_memory_enabled"
# The editor loads the whole file, not the chat's 32 KiB cut: saving a prefix would drop the rest.
MAX_MEMORY_EDIT_BYTES = 1024 * 1024


def agents_md_enabled() -> bool:
    return os.environ.get(_DISABLE_ENV, "").strip().lower() not in ("0", "false", "no", "off")


def _owner_home() -> Path:
    return Path.home()


def _read(
    path: Path,
    confine: Optional[Path] = None,
    limit: int = MAX_AGENTS_MD_BYTES,
) -> bytes:
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
            # One byte over the cap is enough to know it was cut.
            return handle.read(limit + 1)
        except OSError:
            return b""


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


def global_agents_md_path() -> Path:
    return workspace_root() / "AGENTS.md"


def memory_enabled() -> bool:
    try:
        from storage.studio_db import get_app_setting
        stored = get_app_setting(MEMORY_ENABLED_KEY, True)
    except Exception:
        # Fail closed: an unreadable opt-out must not send notes the user turned off.
        return False
    return stored is not False


def set_memory_enabled(value: Any) -> bool:
    if not isinstance(value, bool):
        raise ValueError("Memory must be turned on or off.")
    from storage.studio_db import upsert_app_settings

    upsert_app_settings({MEMORY_ENABLED_KEY: value}, read_back = False)
    return value


def read_memory() -> str:
    raw = _read(global_agents_md_path(), limit = MAX_MEMORY_EDIT_BYTES)[:MAX_MEMORY_EDIT_BYTES]
    return raw.decode("utf-8", errors = "ignore")


def write_memory(text: str) -> str:
    """Replace the global file with ``text``; empty text removes it. Saving the same text twice is a no-op."""
    body = text.replace("\r\n", "\n").strip()
    data = (body + "\n").encode("utf-8") if body else b""
    if len(data) > MAX_AGENTS_MD_BYTES:
        raise ValueError("Memory is limited to 32 KiB.")
    path = global_agents_md_path()
    # Replacing a link would cut it from the file the user keeps elsewhere.
    if path.is_symlink():
        raise ValueError(f"{_label(path)} is a link; edit the file it points to instead.")
    if not data:
        path.unlink(missing_ok = True)
        return ""
    # Retirement-aware: a late save must not recreate a deleted account's workspace.
    ensure_dir(path.parent)
    # Atomic, so a chat sending meanwhile reads the old notes or the new ones, never half of them.
    fd, tmp = tempfile.mkstemp(dir = path.parent, prefix = ".AGENTS.md.", suffix = ".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok = True)
        raise
    return data.decode("utf-8")


def _sources(project: Optional[dict]) -> list[tuple[tuple[Path, ...], Optional[Path]]]:
    # Studio's own file for every account: another tool's ~/.claude or ~/.codex rules would change existing chats.
    sources = [((global_agents_md_path(),), None)] if memory_enabled() else []
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
    from storage.studio_db import available_project_working_folder

    working = available_project_working_folder(project)
    if working:
        folder = Path(working)
        sources.append(((folder / "AGENTS.md", folder / "CLAUDE.md"), folder))
        return sources
    sandbox_path = project.get("sandboxPath")
    # Same containment tools._get_project_workdir requires before running anything there.
    if sandbox_path and os.path.realpath(sandbox_path).startswith(os.path.realpath(root) + os.sep):
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
