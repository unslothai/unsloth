# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bounded, redacted review material for local Git changes."""

from __future__ import annotations

import re
from pathlib import Path

from utils.log_redaction import redact_log_text

from .git_context import AgentWorkspaceError, project_workspace
from .git_service import git_diff, git_status


_SECRET = re.compile(
    r"(?i)\b(?:password|passwd|token|secret|api[_-]?key|private[_-]?key)\b\s*[:=]\s*[^\s,;]+"
)
_MAX_NOTE = 8_000


def _safe_review_path(value: object) -> str:
    """Keep unusual Git filenames from changing the review body's structure."""
    text = str(value)
    return "".join(
        character
        if 0x20 <= ord(character) and ord(character) != 0x7F
        else f"\\x{ord(character):02x}"
        for character in text
    )


def redact_review_text(value: str, project_root: str | Path = "") -> str:
    text = redact_log_text(str(value))
    if project_root:
        root = str(project_root)
        if root:
            text = text.replace(root, "<project>")
    return _SECRET.sub(
        lambda match: f"{match.group(0).split('=', 1)[0].split(':', 1)[0]}=<redacted>", text
    )


def _public_git_status(status: dict) -> dict:
    return {
        key: status[key]
        for key in (
            "head",
            "branch",
            "detached",
            "clean",
            "counts",
            "files",
            "truncated",
        )
    }


def build_review_summary(project_id: str) -> dict:
    workspace = project_workspace(project_id)
    status = git_status(project_id)
    unstaged = git_diff(project_id, staged = False)
    staged = git_diff(project_id, staged = True)
    return {
        "projectId": project_id,
        "status": _public_git_status(status),
        "unstaged": {
            "diff": redact_review_text(unstaged["diff"], workspace.root),
            "truncated": unstaged["truncated"],
        },
        "staged": {
            "diff": redact_review_text(staged["diff"], workspace.root),
            "truncated": staged["truncated"],
        },
    }


def build_pull_request_draft(
    project_id: str,
    *,
    title: str = "",
    body_note: str = "",
) -> dict:
    if len(body_note.encode("utf-8")) > _MAX_NOTE:
        raise AgentWorkspaceError("Pull request notes are too large.")
    workspace = project_workspace(project_id)
    status = git_status(project_id)
    files = status["files"]
    default_title = f"Studio changes for {project_id}"[:120]
    rendered_title = redact_review_text(title.strip() or default_title, workspace.root)[:120]
    lines = [
        "Generated from the local Studio review surface.",
        "",
        f"Changed files: {len(files)}",
    ]
    for item in files[:200]:
        lines.append(f"- {item['code']} {_safe_review_path(item['path'])}")
    if len(files) > 200:
        lines.append("- Additional files omitted from this bounded preview.")
    if body_note.strip():
        lines.extend(("", redact_review_text(body_note.strip(), workspace.root)))
    return {
        "title": rendered_title,
        "body": redact_review_text("\n".join(lines), workspace.root)[:64_000],
        "status": _public_git_status(status),
    }


__all__ = ["build_pull_request_draft", "build_review_summary", "redact_review_text"]
