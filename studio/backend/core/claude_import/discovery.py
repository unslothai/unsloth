# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Find the session transcripts Claude Code keeps on disk.

Claude Code files a session under ``~/.claude/projects/<encoded-path>/<id>.jsonl``,
where the directory name is the absolute path of the folder it was run in with
its separators turned into dashes (``/Users/me/app`` becomes
``-Users-me-app``). As with Cursor, that encoding is one-way -- ``/a/b-c`` and
``/a/b/c`` collapse to the same name -- so reading it back is a filesystem
question, and one that cannot always be answered once the folder is gone. A
project whose folder no longer resolves still imports, named from its encoded
path, because the history is the part being brought over.

Nothing here reads file contents: discovery answers "which sessions exist", and
:mod:`core.claude_import.transcripts` reads them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Same override the rest of Studio uses for a relocated home, so tests and
# non-default Claude installs point both at one place.
CLAUDE_HOME_ENV = "UNSLOTH_CLAUDE_HOME"

_PROJECTS_DIR = "projects"

# Claude Code's own bookkeeping directories hold no session worth importing.
_INTERNAL_PREFIXES = (".",)


def claude_home(override: Optional[Path] = None) -> Path:
    """Root of the Claude Code state directory, ``~/.claude`` unless overridden."""
    if override is not None:
        return Path(override).expanduser()
    from_env = (os.environ.get(CLAUDE_HOME_ENV) or "").strip()
    if from_env:
        return Path(from_env).expanduser()
    return Path.home() / ".claude"


def find_sessions(project_dir: Path) -> list[Path]:
    """Session transcripts for one project, oldest first for a stable order."""
    sessions = [
        entry for entry in project_dir.iterdir() if entry.is_file() and entry.suffix == ".jsonl"
    ]
    sessions.sort(key = lambda path: path.name)
    return sessions


def _project_name(encoded: str) -> str:
    """A readable name for a project whose folder could not be resolved.

    The encoded path's leading separators became leading dashes, and dashes
    inside a real folder name are indistinguishable from separators, so the
    exact path is unrecoverable. Dropping the empty leading tokens and rejoining
    with slashes gives back a path-shaped name (``Users/me/app``) that still
    identifies the project, which the bare last token would not.
    """
    tokens = [token for token in encoded.split("-") if token]
    return "/".join(tokens) if tokens else encoded


@dataclass
class ClaudeProject:
    """One folder Claude Code holds sessions for."""

    slug: str
    name: str
    project_dir: Path
    sessions: list[Path] = field(default_factory = list)
    last_used_ms: int = 0


def read_project(project_dir: Path) -> Optional[ClaudeProject]:
    """Inventory one project directory, or None when it holds no session."""
    slug = project_dir.name
    if slug.startswith(_INTERNAL_PREFIXES):
        return None
    sessions = find_sessions(project_dir)
    if not sessions:
        return None
    stamps = []
    for path in sessions:
        try:
            stamps.append(path.stat().st_mtime)
        except OSError:
            continue
    last_used_ms = int(max(stamps) * 1000) if stamps else 0
    return ClaudeProject(
        slug = slug,
        name = _project_name(slug),
        project_dir = project_dir,
        sessions = sessions,
        last_used_ms = last_used_ms,
    )


def list_claude_projects(home: Optional[Path] = None) -> list[ClaudeProject]:
    """Every Claude Code project with sessions on this machine, newest first."""
    projects_root = claude_home(home) / _PROJECTS_DIR
    if not projects_root.is_dir():
        return []
    projects = []
    for entry in sorted(projects_root.iterdir()):
        if not entry.is_dir():
            continue
        project = read_project(entry)
        if project is not None:
            projects.append(project)
    projects.sort(key = lambda item: (-item.last_used_ms, item.name.lower()))
    return projects


__all__ = [
    "CLAUDE_HOME_ENV",
    "ClaudeProject",
    "claude_home",
    "find_sessions",
    "list_claude_projects",
    "read_project",
]
