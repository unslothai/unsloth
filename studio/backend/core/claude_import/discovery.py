# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Discover Claude Code session transcripts under ``~/.claude/projects``."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

CLAUDE_HOME_ENV = "UNSLOTH_CLAUDE_HOME"

_PROJECTS_DIR = "projects"

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
    """Path-shaped name (``Users/me/app``); the dash encoding is lossy, so this is not the real path."""
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
