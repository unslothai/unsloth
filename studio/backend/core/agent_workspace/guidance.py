# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Authoritative project guidance injected consistently across transports."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional

from storage.studio_db import get_chat_project, get_chat_thread

from .common import AgentWorkspaceError, project_workspace_access
from .instructions import resolve_agents_instructions
from .skills import discover_project_skills, render_project_skills


PROJECT_SESSION_PREFIX = "project-"
MAX_PROJECT_INSTRUCTIONS_CHARACTERS = 24_000
MAX_RENDERED_PROJECT_GUIDANCE_CHARACTERS = 128 * 1024
MAX_RENDERED_PROJECT_BLOCK_CHARACTERS = 32 * 1024
MAX_RENDERED_INSTRUCTIONS_BLOCK_CHARACTERS = 40 * 1024
MAX_RENDERED_SKILLS_BLOCK_CHARACTERS = 48 * 1024
_SERVER_GUIDANCE_BLOCK = re.compile(
    r"(?:\r?\n)*<(unsloth_project_context|unsloth_project_guidance|"
    r"unsloth_repository_instructions|"
    r"unsloth_project_skills) "
    r'version="1"[^>]*>[\s\S]*?</\1>(?:\r?\n)*'
)


class ProjectGuidanceUnavailable(AgentWorkspaceError):
    """A persisted project exists but its guidance cannot be resolved safely."""


@dataclass(frozen = True)
class ResolvedProjectGuidance:
    project_id: str
    addition: str
    instructions: dict
    skills: dict
    selected_skills: tuple[str, ...]


def _replace_controls(value: str) -> str:
    return "".join(
        "\ufffd"
        if ord(character) <= 8 or ord(character) in (11, 12, 127) or 14 <= ord(character) <= 31
        else character
        for character in value
    )


def escape_guidance(value: object) -> str:
    return (
        _replace_controls(str(value or ""))
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def strip_server_project_guidance(value: str) -> str:
    if _SERVER_GUIDANCE_BLOCK.search(value) is None:
        return value
    return _SERVER_GUIDANCE_BLOCK.sub("\n\n", value).strip()


def project_id_from_session(session_id: Optional[str]) -> Optional[str]:
    if not isinstance(session_id, str) or not session_id.startswith(PROJECT_SESSION_PREFIX):
        return None
    project_id = session_id[len(PROJECT_SESSION_PREFIX) :]
    if not project_id or get_chat_thread(session_id) is not None:
        return None
    project = get_chat_project(project_id)
    if project is None or project.get("archived"):
        return None
    return project_id


def latest_user_query(messages: Iterable[object]) -> str:
    for message in reversed(list(messages)):
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if role != "user":
            continue
        content = (
            message.get("content")
            if isinstance(message, dict)
            else getattr(message, "content", None)
        )
        if isinstance(content, str):
            return content[:16_384]
        if not isinstance(content, list):
            return ""
        fragments = []
        remaining = 16_384
        for part in content:
            kind = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
            text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
            if kind not in {"text", "input_text"} or not isinstance(text, str):
                continue
            fragments.append(text[:remaining])
            remaining -= len(fragments[-1])
            if remaining <= 0:
                break
        return "\n".join(fragments)
    return ""


def _bounded_escape(value: str, limit: int, marker: str) -> tuple[str, bool]:
    escaped = escape_guidance(value)
    if len(escaped) <= limit:
        return escaped, False
    marker_escaped = escape_guidance(marker)
    available = max(0, limit - len(marker_escaped))
    low = 0
    high = len(value)
    while low < high:
        middle = (low + high + 1) // 2
        if len(escape_guidance(value[:middle])) <= available:
            low = middle
        else:
            high = middle - 1
    return escape_guidance(value[:low]) + marker_escaped, True


def _project_block(project: dict) -> str:
    value = str(project.get("instructions") or "").strip()
    if not value:
        return ""
    bounded = value[:MAX_PROJECT_INSTRUCTIONS_CHARACTERS]
    if len(value) > MAX_PROJECT_INSTRUCTIONS_CHARACTERS:
        bounded += (
            f"\n[Project instructions truncated at "
            f"{MAX_PROJECT_INSTRUCTIONS_CHARACTERS} characters.]"
        )
    prefix = '<unsloth_project_guidance version="1">\n<project_instructions>\n'
    suffix = "\n</project_instructions>\n</unsloth_project_guidance>"
    content, _ = _bounded_escape(
        bounded,
        MAX_RENDERED_PROJECT_BLOCK_CHARACTERS - len(prefix) - len(suffix),
        "\n[Project instructions truncated at the rendered character limit.]",
    )
    return "".join(
        (
            prefix,
            content,
            suffix,
        )
    )


def _instructions_block(instructions: dict) -> str:
    layers = instructions.get("layers") or []
    if not layers:
        return ""
    prefix_lines = [
        '<unsloth_repository_instructions version="1">',
        "<scope_policy>Root guidance applies repository-wide. Deeper guidance applies only within its labeled scope and overrides ancestors.</scope_policy>",
    ]
    if instructions.get("truncated"):
        prefix_lines.append('<truncation reason="instruction-bounds" />')
    lines = list(prefix_lines)
    suffix = "</unsloth_repository_instructions>"
    for layer in layers:
        opening = (
            f'<agents_instructions path="{escape_guidance(layer["path"])}" '
            f'scope="{escape_guidance(layer["scope"])}">'
        )
        closing = "</agents_instructions>"
        fixed = len("\n".join((*lines, opening, closing, suffix))) + 2
        remaining = MAX_RENDERED_INSTRUCTIONS_BLOCK_CHARACTERS - fixed
        if remaining <= 0:
            lines.append('<truncation reason="rendered-character-limit" />')
            break
        content, content_truncated = _bounded_escape(
            str(layer["content"]),
            remaining,
            "\n[Instruction layer truncated at the rendered character limit.]",
        )
        lines.extend((opening, content, closing))
        if content_truncated:
            lines.append('<truncation reason="rendered-character-limit" />')
            break
    lines.append(suffix)
    return "\n".join(lines)


def _skills_block(skills: dict, query: str) -> tuple[str, tuple[str, ...]]:
    rendered, selected = render_project_skills(skills, query)
    if not rendered:
        return "", selected
    # Skill text is repository-controlled data carried inside a server-owned
    # envelope. Escape it so a skill cannot forge or close that envelope.
    escaped = escape_guidance(rendered)
    prefix = '<unsloth_project_skills version="1">\n'
    suffix = "\n</unsloth_project_skills>"
    budget = MAX_RENDERED_SKILLS_BLOCK_CHARACTERS - len(prefix) - len(suffix)
    if len(escaped) > budget:
        metadata_only = {
            **skills,
            "skills": [
                {key: value for key, value in skill.items() if key != "content"}
                for skill in skills.get("skills") or []
            ],
            "unavailableRequests": [
                *(skills.get("unavailableRequests") or []),
                *(
                    {
                        "name": skill["name"],
                        "path": skill["path"],
                        "selection": skill["selection"],
                        "reason": "selected skill exceeded the project guidance budget",
                    }
                    for skill in skills.get("skills") or []
                    if isinstance(skill.get("content"), str)
                ),
            ],
        }
        rendered, _ = render_project_skills(metadata_only, query)
        escaped, _ = _bounded_escape(
            rendered,
            budget,
            "\n[Project skill catalog truncated at the rendered character limit.]",
        )
        selected = ()
    return (prefix + escaped + suffix), selected


def resolve_project_guidance(
    session_id: Optional[str], *, query: str = ""
) -> Optional[ResolvedProjectGuidance]:
    project_id = project_id_from_session(session_id)
    if project_id is None:
        return None
    project = get_chat_project(project_id)
    if project is None:
        return None
    if os.name == "nt":
        instructions = {
            "layers": [],
            "combined": "",
            "truncated": False,
            "issues": [{"path": "AGENTS.md", "reason": "unsupported_platform"}],
            "precedence": "Root guidance applies first; deeper guidance overrides it.",
            "bytesRead": 0,
        }
        skills = {
            "skills": [],
            "issues": [{"path": ".agents/skills", "reason": "unsupported_platform"}],
            "truncated": False,
        }
    else:
        try:
            with project_workspace_access(project_id) as workspace:
                identity = (workspace.device_id, workspace.file_id)
                instructions = resolve_agents_instructions(
                    workspace.root,
                    expected_identity = identity,
                )
                skills = discover_project_skills(
                    workspace.root,
                    expected_identity = identity,
                    query = query,
                )
        except AgentWorkspaceError as exc:
            raise ProjectGuidanceUnavailable(
                "The project workspace is unavailable. Reconnect or reopen its folder, then retry."
            ) from exc
    skills_block, selected = _skills_block(skills, query)
    rendered = "\n\n".join(
        block
        for block in (
            _project_block(project),
            _instructions_block(instructions),
            skills_block,
        )
        if block
    )
    addition = ""
    if rendered:
        prefix = '<unsloth_project_context version="1">'
        suffix = "</unsloth_project_context>"
        addition = f"{prefix}\n{rendered}\n{suffix}"
        if len(addition) > MAX_RENDERED_PROJECT_GUIDANCE_CHARACTERS:
            addition = "\n".join(
                (
                    prefix,
                    '<guidance_unavailable reason="rendered-character-limit" />',
                    suffix,
                )
            )
            selected = ()
    return ResolvedProjectGuidance(
        project_id = project_id,
        addition = addition,
        instructions = instructions,
        skills = skills,
        selected_skills = selected,
    )


__all__ = [
    "MAX_PROJECT_INSTRUCTIONS_CHARACTERS",
    "MAX_RENDERED_INSTRUCTIONS_BLOCK_CHARACTERS",
    "MAX_RENDERED_PROJECT_GUIDANCE_CHARACTERS",
    "MAX_RENDERED_PROJECT_BLOCK_CHARACTERS",
    "MAX_RENDERED_SKILLS_BLOCK_CHARACTERS",
    "PROJECT_SESSION_PREFIX",
    "ProjectGuidanceUnavailable",
    "ResolvedProjectGuidance",
    "escape_guidance",
    "latest_user_query",
    "project_id_from_session",
    "resolve_project_guidance",
    "strip_server_project_guidance",
]
