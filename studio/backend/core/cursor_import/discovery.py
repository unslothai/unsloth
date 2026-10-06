# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Discover Cursor agent transcripts under ``~/.cursor/projects``."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

CURSOR_HOME_ENV = "UNSLOTH_CURSOR_HOME"

_STATE_DIR = "projects"
_TRANSCRIPTS_DIR = "agent-transcripts"
_SUBAGENTS_DIR = "subagents"

NO_FOLDER_SLUG = "empty-window"
_NO_FOLDER_NAME = "No folder open"

_INTERNAL_SLUG_PREFIXES = (".",)

# A slug of N tokens has 2**(N-1) readings; caps directory probes per slug.
_RESOLVE_BUDGET = 4096


def cursor_home(override: Optional[Path] = None) -> Path:
    """Root of the Cursor state directory, ``~/.cursor`` unless overridden."""
    if override is not None:
        return Path(override).expanduser()
    from_env = (os.environ.get(CURSOR_HOME_ENV) or "").strip()
    if from_env:
        return Path(from_env).expanduser()
    return Path.home() / ".cursor"


def state_slug(project_path: Path) -> str:
    """The directory name ``~/.cursor/projects`` uses for a folder."""
    return re.sub(r"[\\/]+", "-", str(project_path)).strip("-")


def _resolve_roots(first_token: str) -> list[Path]:
    """Where a slug's first token starts from on this platform."""
    if os.name != "nt":
        return [Path("/")]
    # Windows slugs start with the drive, "C:" or "C".
    if re.fullmatch(r"[A-Za-z]:?", first_token):
        return [Path(f"{first_token[0]}:{os.sep}")]
    return []


def resolve_state_slug(slug: str, *, budget: int = _RESOLVE_BUDGET) -> Optional[Path]:
    """The unique existing folder a slug denotes; None if gone or ambiguous."""
    tokens = [token for token in slug.split("-") if token]
    if not tokens:
        return None

    found: list[Path] = []
    spent = 0

    def walk(base: Path, index: int) -> None:
        nonlocal spent
        if index == len(tokens):
            found.append(base)
            return
        for end in range(len(tokens), index, -1):
            if spent >= budget or len(found) > 1:
                return
            spent += 1
            child = base / "-".join(tokens[index:end])
            if child.is_dir():
                walk(child, end)

    for root in _resolve_roots(tokens[0]):
        walk(root, 0)
    return found[0] if len(found) == 1 else None


def find_transcripts(state_dir: Path) -> tuple[list[Path], int]:
    """Session transcripts plus a count of skipped subagent ones (importing those would duplicate history)."""
    root = state_dir / _TRANSCRIPTS_DIR
    if not root.is_dir():
        return [], 0

    transcripts: list[Path] = []
    subagents = 0
    for entry in sorted(root.iterdir()):
        if entry.is_file() and entry.suffix == ".jsonl":
            # Older flat layout.
            transcripts.append(entry)
            continue
        if not entry.is_dir():
            continue
        session = entry / f"{entry.name}.jsonl"
        if session.is_file():
            transcripts.append(session)
        subagent_dir = entry / _SUBAGENTS_DIR
        if subagent_dir.is_dir():
            subagents += sum(1 for path in subagent_dir.glob("*.jsonl"))
    return transcripts, subagents


@dataclass
class CursorWorkspace:
    """One folder Cursor holds conversations for."""

    slug: str
    name: str
    state_dir: Path
    project_path: Optional[Path] = None
    transcripts: list[Path] = field(default_factory = list)
    subagent_transcripts: int = 0
    last_used_ms: int = 0


def _last_used_ms(transcripts: list[Path], state_dir: Path) -> int:
    stamps = []
    for path in transcripts:
        try:
            stamps.append(path.stat().st_mtime)
        except OSError:
            continue
    if not stamps:
        try:
            stamps.append(state_dir.stat().st_mtime)
        except OSError:
            return 0
    return int(max(stamps) * 1000)


def _workspace_name(slug: str, project_path: Optional[Path]) -> str:
    if slug == NO_FOLDER_SLUG:
        return _NO_FOLDER_NAME
    if project_path is not None and project_path.name:
        return project_path.name
    # The last token is not the folder name (dashes are ambiguous); strip only the home prefix.
    try:
        home_prefix = f"{state_slug(Path.home().resolve())}-"
    except (OSError, RuntimeError):
        home_prefix = ""
    if home_prefix and slug.startswith(home_prefix):
        return slug[len(home_prefix) :] or slug
    return slug


def read_workspace(state_dir: Path, *, resolve_paths: bool = True) -> Optional[CursorWorkspace]:
    """None when it holds no conversation; ``resolve_paths=False`` skips the costly slug search."""
    slug = state_dir.name
    if slug.startswith(_INTERNAL_SLUG_PREFIXES):
        return None
    transcripts, subagents = find_transcripts(state_dir)
    if not transcripts:
        return None
    project_path = resolve_state_slug(slug) if resolve_paths and slug != NO_FOLDER_SLUG else None
    return CursorWorkspace(
        slug = slug,
        name = _workspace_name(slug, project_path),
        state_dir = state_dir,
        project_path = project_path,
        transcripts = transcripts,
        subagent_transcripts = subagents,
        last_used_ms = _last_used_ms(transcripts, state_dir),
    )


def list_cursor_workspaces(
    home: Optional[Path] = None, *, resolve_paths: bool = True
) -> list[CursorWorkspace]:
    """Every Cursor project with conversations on this machine, newest first."""
    projects_root = cursor_home(home) / _STATE_DIR
    if not projects_root.is_dir():
        return []
    workspaces = []
    for entry in sorted(projects_root.iterdir()):
        if not entry.is_dir():
            continue
        workspace = read_workspace(entry, resolve_paths = resolve_paths)
        if workspace is not None:
            workspaces.append(workspace)
    workspaces.sort(key = lambda item: (-item.last_used_ms, item.name.lower()))
    return workspaces


__all__ = [
    "CURSOR_HOME_ENV",
    "NO_FOLDER_SLUG",
    "CursorWorkspace",
    "cursor_home",
    "find_transcripts",
    "list_cursor_workspaces",
    "read_workspace",
    "resolve_state_slug",
    "state_slug",
]
