# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Find the agent transcripts Cursor keeps on disk.

Cursor files a session under ``~/.cursor/projects/<slug>/agent-transcripts``,
where the slug is the absolute path of the folder that was open, with its
separators turned into dashes. That encoding is one-way: ``/a/b-c`` and
``/a/b/c`` both dash to ``a-b-c``. Reading it back is therefore a question for
the filesystem (:func:`resolve_state_slug`), and one it cannot always answer --
the folder may have been renamed or deleted since. A slug that no longer names a
folder still lists, because its transcripts are intact and the history is the
part being imported.

Nothing here reads file contents: discovery answers "which conversations exist",
and :mod:`core.cursor_import.transcripts` reads them.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Same override the rest of Studio uses for a relocated home, so tests and
# non-default Cursor installs point both at one place.
CURSOR_HOME_ENV = "UNSLOTH_CURSOR_HOME"

_STATE_DIR = "projects"
_TRANSCRIPTS_DIR = "agent-transcripts"
_SUBAGENTS_DIR = "subagents"

# The window Cursor opens with no folder. It is not a project, but its agent
# sessions are real work, so it imports under a name that says what it is.
NO_FOLDER_SLUG = "empty-window"
_NO_FOLDER_NAME = "No folder open"

# Cursor's own bookkeeping directories, which hold no session worth importing.
_INTERNAL_SLUG_PREFIXES = (".",)

# Ceiling on the directory probes one slug may cost. A slug of N tokens has
# 2**(N-1) readings; the filesystem prunes almost all of them immediately, and
# this stops a pathological name from walking the disk.
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
    # A Windows slug begins with the drive ("C:" or, on a filesystem that cannot
    # store a colon, "C"). Anything else is unresolvable rather than guessed at.
    if re.fullmatch(r"[A-Za-z]:?", first_token):
        return [Path(f"{first_token[0]}:{os.sep}")]
    return []


def resolve_state_slug(slug: str, *, budget: int = _RESOLVE_BUDGET) -> Optional[Path]:
    """The folder a state directory name denotes, when the disk says so plainly.

    Every grouping of the dash-separated tokens is a possible path; the ones
    that exist are the candidates. A single candidate is the answer. None means
    the folder is gone or the name is genuinely ambiguous, and the caller then
    names the project from the slug rather than from a guess.
    """
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
        # Longest token group first: a folder whose own name contains a dash is
        # the likelier reading, so the common case settles in one descent.
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
    """Session transcripts, plus how many subagent transcripts were passed over.

    A session is a directory named for its id holding ``<id>.jsonl``; delegated
    runs live in a ``subagents/`` directory beside it. Those are counted and
    left alone -- they are turns inside a session that already imports, so
    replaying them as their own conversations would double the history.
    """
    root = state_dir / _TRANSCRIPTS_DIR
    if not root.is_dir():
        return [], 0

    transcripts: list[Path] = []
    subagents = 0
    for entry in sorted(root.iterdir()):
        if entry.is_file() and entry.suffix == ".jsonl":
            # Flat layout: tolerated so an older state directory still imports.
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
    """When this project was last worked in, from its newest transcript."""
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
    # No folder to read a name from. The last token is not it either: nothing in
    # the slug says which tokens were one folder's name, and taking one leaves
    # projects called "fr" or "web". What can be dropped honestly is the home
    # directory the path started with, leaving the part that identifies it.
    try:
        home_prefix = f"{state_slug(Path.home().resolve())}-"
    except (OSError, RuntimeError):
        home_prefix = ""
    if home_prefix and slug.startswith(home_prefix):
        return slug[len(home_prefix) :] or slug
    return slug


def read_workspace(state_dir: Path, *, resolve_paths: bool = True) -> Optional[CursorWorkspace]:
    """Inventory one state directory, or None when it holds no conversation.

    ``resolve_paths`` off skips the filesystem search that turns a slug back
    into a folder, which is the expensive half; the count is the same, only the
    names are less pretty. The status probe uses it, the import does not.
    """
    slug = state_dir.name
    if slug.startswith(_INTERNAL_SLUG_PREFIXES):
        return None
    transcripts, subagents = find_transcripts(state_dir)
    if not transcripts:
        return None
    project_path = (
        resolve_state_slug(slug) if resolve_paths and slug != NO_FOLDER_SLUG else None
    )
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
