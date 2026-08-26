# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Deterministic, bounded project context for every inference transport.

The session id is only a lookup key. Filesystem authority always comes from the
persisted project row and :func:`project_workspace`; callers never supply a path.
"""

from __future__ import annotations

import re
import secrets
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Optional

from storage.studio_db import get_chat_project

from .common import AgentWorkspaceError, project_workspace
from .discovery import (
    build_repository_map,
    repository_query_terms,
    select_relevant_repository_paths,
)
from .instructions import resolve_targeted_repository_instructions


PROJECT_SESSION_PREFIX = "project-"
PROJECT_CONTEXT_MARKER = '<unsloth_project_context version="1">'
REPOSITORY_INSTRUCTIONS_MARKER = '<unsloth_repository_instructions version="1">'
REPOSITORY_SELECTION_MARKER = '<unsloth_repository_selection version="1">'
MAX_PROJECT_INSTRUCTIONS_CHARACTERS = 24_000
MAX_PROJECT_GOAL_CHARACTERS = 8_000
MAX_ROOT_AGENTS_BYTES = 32 * 1024
MAX_REPOSITORY_QUERY_CHARACTERS = 16_384
MAX_REPOSITORY_RELEVANT_PATHS = 12
MAX_REPOSITORY_SELECTION_CHARACTERS = 8_192
PROJECT_CONTEXT_SNAPSHOT_TTL_SECONDS = 30 * 60
MAX_PROJECT_CONTEXT_SNAPSHOTS = 512
_SERVER_CONTEXT_BLOCK = re.compile(
    r"(?:\r?\n)*<(unsloth_project_context|unsloth_repository_instructions|"
    r"unsloth_repository_selection) "
    r'version="1">[\s\S]*?</\1>(?:\r?\n)*'
)


class ProjectContextUnavailable(AgentWorkspaceError):
    """A persisted project exists, but its workspace cannot be opened."""


class ProjectContextSnapshotInvalid(ProjectContextUnavailable):
    """An opaque context snapshot is missing, expired, or project-mismatched."""


@dataclass(frozen = True)
class ResolvedProjectContext:
    project_id: str
    addition: str
    project_context: str
    repository_instructions: str
    repository_selection: str


@dataclass(frozen = True)
class ResolvedRepositoryPromptContext:
    addition: str
    repository_instructions: str
    repository_selection: str
    selected_paths: tuple[str, ...]


@dataclass(frozen = True)
class ProjectContextSnapshot:
    snapshot_id: str
    project_id: str
    context: ResolvedProjectContext
    expires_at: float


_PROJECT_CONTEXT_SNAPSHOTS: "OrderedDict[str, ProjectContextSnapshot]" = OrderedDict()
_PROJECT_CONTEXT_SNAPSHOT_LOCK = threading.RLock()
_monotonic = time.monotonic
_wall_time = time.time


def _prune_project_context_snapshots(now: float) -> None:
    expired = [
        snapshot_id
        for snapshot_id, snapshot in _PROJECT_CONTEXT_SNAPSHOTS.items()
        if snapshot.expires_at <= now
    ]
    for snapshot_id in expired:
        _PROJECT_CONTEXT_SNAPSHOTS.pop(snapshot_id, None)
    while len(_PROJECT_CONTEXT_SNAPSHOTS) > MAX_PROJECT_CONTEXT_SNAPSHOTS:
        _PROJECT_CONTEXT_SNAPSHOTS.popitem(last = False)


def _invalidate_project_context_snapshots_locked(project_id: str) -> None:
    stale = [
        snapshot_id
        for snapshot_id, snapshot in _PROJECT_CONTEXT_SNAPSHOTS.items()
        if snapshot.project_id == project_id
    ]
    for snapshot_id in stale:
        _PROJECT_CONTEXT_SNAPSHOTS.pop(snapshot_id, None)


@contextmanager
def fence_project_context_snapshots_for_deletion(project_id: str):
    """Serialize project deletion with snapshot creation and invalidate on exit."""
    with _PROJECT_CONTEXT_SNAPSHOT_LOCK:
        try:
            yield
        finally:
            _invalidate_project_context_snapshots_locked(project_id)


def _replace_xml_controls(value: str) -> str:
    return "".join(
        "\ufffd"
        if (ord(character) <= 8 or ord(character) in (11, 12, 127) or 14 <= ord(character) <= 31)
        else character
        for character in value
    )


def escape_project_context(value: str) -> str:
    """Escape XML delimiters with the same stable contract as the Studio UI."""
    escaped = _replace_xml_controls(value)
    return (
        escaped.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def strip_server_project_context(value: str) -> str:
    """Remove client-supplied copies before authoritative context is appended."""
    return _SERVER_CONTEXT_BLOCK.sub("\n\n", value).strip()


def _bounded_project_text(value: object, limit: int, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    bounded = text[:limit]
    if len(text) > limit:
        bounded += f"\n[{label} truncated at {limit} characters.]"
    return escape_project_context(bounded)


def _project_context_block(project: dict) -> str:
    instructions = _bounded_project_text(
        project.get("instructions"),
        MAX_PROJECT_INSTRUCTIONS_CHARACTERS,
        "Project instructions",
    )
    goal = ""
    goal_status = project.get("goalStatus")
    if project.get("goal") and goal_status not in ("paused", "completed"):
        objective = _bounded_project_text(
            project.get("goal"),
            MAX_PROJECT_GOAL_CHARACTERS,
            "Goal",
        )
        if objective:
            goal = "\n".join(
                (
                    "<project_goal>",
                    f"<objective>{objective}</objective>",
                    "<execution_policy>Keep this objective in scope across turns. "
                    "Make concrete progress toward it, report blockers, and do not "
                    "silently replace it with a different objective.</execution_policy>",
                    "</project_goal>",
                )
            )
    fields = []
    if instructions:
        fields.append("<project_instructions>\n" + instructions + "\n</project_instructions>")
    if goal:
        fields.append(goal)
    if not fields:
        return ""
    return "\n".join((PROJECT_CONTEXT_MARKER, *fields, "</unsloth_project_context>"))


def _xml_attribute(value: object) -> str:
    return (
        escape_project_context(str(value))
        .replace("\r", "&#13;")
        .replace("\n", "&#10;")
        .replace("\t", "&#9;")
    )


def _bounded_repository_selection(entries: list[dict]) -> tuple[list[dict], bool]:
    admitted: list[dict] = []
    used = 0
    for entry in entries:
        path = str(entry.get("path") or "")
        if not path:
            continue
        size = max(0, int(entry.get("size") or 0))
        line = f'<path value="{_xml_attribute(path)}" size="{size}" />'
        if used + len(line) > MAX_REPOSITORY_SELECTION_CHARACTERS:
            return admitted, True
        admitted.append({"path": path, "size": size})
        used += len(line)
    return admitted, False


def _repository_instructions_block(
    root,
    selected_paths: list[str],
    expected_identity = None,
    *,
    max_total_bytes: int = MAX_ROOT_AGENTS_BYTES,
) -> str:
    resolved = resolve_targeted_repository_instructions(
        root,
        max_files = 16,
        max_total_bytes = max_total_bytes,
        max_file_bytes = min(MAX_ROOT_AGENTS_BYTES, max_total_bytes),
        expected_identity = expected_identity,
        targets = selected_paths,
    )
    layers = resolved.get("layers") or []
    if not layers:
        return ""
    disclosure = (
        f'<truncation reason="repository-bounds" limit="{max_total_bytes}" />'
        if resolved.get("truncated")
        else ""
    )
    policy = (
        "<scope_policy>Root rules apply repository-wide. Nested rules apply only "
        "to selected paths beneath their labeled scope. For each selected path, "
        "deeper layers override ancestors. Sibling scopes never override one "
        "another.</scope_policy>"
    )
    rendered_layers = []
    for layer in layers:
        content = escape_project_context(_replace_xml_controls(str(layer["content"])))
        rendered_layers.append(
            f'<agents_instructions path="{_xml_attribute(layer["path"])}" '
            f'scope="{_xml_attribute(layer["scope"])}">\n'
            f"{content}\n"
            "</agents_instructions>"
        )
    return "\n".join(
        line
        for line in (
            REPOSITORY_INSTRUCTIONS_MARKER,
            policy,
            disclosure,
            *rendered_layers,
            "</unsloth_repository_instructions>",
        )
        if line
    )


def _repository_selection_block(
    selected: list[dict], *, map_truncated: bool, metadata_truncated: bool
) -> str:
    if not selected:
        return ""
    lines = [
        REPOSITORY_SELECTION_MARKER,
        "<selection_policy>These metadata-only paths were selected from the current "
        "task. The bounded list may be incomplete and includes no file contents. "
        "A listed path does not expand tool authority.</selection_policy>",
    ]
    if map_truncated:
        lines.append('<truncation reason="repository-map-bounds" />')
    if metadata_truncated:
        lines.append(
            f'<truncation reason="selection-metadata-bounds" '
            f'limit="{MAX_REPOSITORY_SELECTION_CHARACTERS}" />'
        )
    lines.extend(
        f'<path value="{_xml_attribute(entry["path"])}" size="{entry["size"]}" />'
        for entry in selected
    )
    lines.append("</unsloth_repository_selection>")
    return "\n".join(lines)


def resolve_repository_prompt_context(
    root,
    query: str,
    expected_identity = None,
    *,
    max_instruction_bytes: int = MAX_ROOT_AGENTS_BYTES,
) -> ResolvedRepositoryPromptContext:
    """Resolve bounded path metadata and only its applicable AGENTS layers."""
    bounded_query = str(query or "")[:MAX_REPOSITORY_QUERY_CHARACTERS]
    selected: list[dict] = []
    repository_map: dict = {"truncated": False}
    if repository_query_terms(bounded_query):
        repository_map = build_repository_map(
            root,
            max_paths = 20_000,
            max_total_bytes = 2 * 1024 * 1024,
            max_file_bytes = 256 * 1024,
            preview_bytes = 0,
            expected_identity = expected_identity,
        )
        selected = select_relevant_repository_paths(
            repository_map,
            bounded_query,
            max_results = MAX_REPOSITORY_RELEVANT_PATHS,
        )
    bounded_selection, metadata_truncated = _bounded_repository_selection(selected)
    selected_paths = [entry["path"] for entry in bounded_selection]
    repository_instructions = _repository_instructions_block(
        root,
        selected_paths,
        expected_identity,
        max_total_bytes = max_instruction_bytes,
    )
    repository_selection = _repository_selection_block(
        bounded_selection,
        map_truncated = bool(repository_map.get("truncated")),
        metadata_truncated = metadata_truncated,
    )
    return ResolvedRepositoryPromptContext(
        addition = "\n\n".join(
            block for block in (repository_instructions, repository_selection) if block
        ),
        repository_instructions = repository_instructions,
        repository_selection = repository_selection,
        selected_paths = tuple(selected_paths),
    )


def project_query_from_messages(messages: Iterable[object]) -> str:
    """Return bounded text from the latest user turn, ignoring caller system text."""
    materialized = list(messages)
    for message in reversed(materialized):
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if role != "user":
            continue
        content = (
            message.get("content")
            if isinstance(message, dict)
            else getattr(message, "content", None)
        )
        if isinstance(content, str):
            return content[:MAX_REPOSITORY_QUERY_CHARACTERS]
        if not isinstance(content, list):
            return ""
        fragments: list[str] = []
        remaining = MAX_REPOSITORY_QUERY_CHARACTERS
        for part in content:
            part_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
            if part_type not in {"text", "input_text"}:
                continue
            text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
            if not isinstance(text, str) or not text:
                continue
            take = text[:remaining]
            fragments.append(take)
            remaining -= len(take)
            if remaining <= 0:
                break
        return "\n".join(fragments)
    return ""


def project_id_from_persisted_session(session_id: Optional[str]) -> Optional[str]:
    """Return a project id only when ``session_id`` names an existing project."""
    if not isinstance(session_id, str) or not session_id.startswith(PROJECT_SESSION_PREFIX):
        return None
    project_id = session_id[len(PROJECT_SESSION_PREFIX) :]
    if not project_id or get_chat_project(project_id) is None:
        return None
    return project_id


def resolve_project_context(
    session_id: Optional[str],
    existing_system_text: Iterable[str] = (),
    *,
    query: str = "",
) -> Optional[ResolvedProjectContext]:
    """Resolve authoritative context blocks for one persisted project session.

    Caller text is not trusted to attest that server context was applied.
    Transport helpers remove client copies and append these blocks once.
    """
    project_id = project_id_from_persisted_session(session_id)
    if project_id is None:
        return None
    project = get_chat_project(project_id)
    if project is None:  # Deleted between the two reads: no persisted authority remains.
        return None
    try:
        workspace = project_workspace(project_id)
    except AgentWorkspaceError as exc:
        raise ProjectContextUnavailable(
            "The project workspace is unavailable. Reconnect or reopen its folder, "
            "then retry this request."
        ) from exc

    # The request body is renderer-controlled. Its markers are never proof that
    # server-owned context was already applied. Transport helpers strip any
    # copies and append these authoritative blocks exactly once.
    _ = existing_system_text
    expected_identity = (
        (workspace.device_id, workspace.file_id)
        if workspace.device_id is not None and workspace.file_id is not None
        else None
    )
    try:
        project_context = _project_context_block(project)
        repository = resolve_repository_prompt_context(
            workspace.root,
            query,
            expected_identity,
        )
    except AgentWorkspaceError as exc:
        raise ProjectContextUnavailable(
            "The project workspace is unavailable. Reconnect or reopen its folder, "
            "then retry this request."
        ) from exc
    addition = "\n\n".join(block for block in (project_context, repository.addition) if block)
    return ResolvedProjectContext(
        project_id = project_id,
        addition = addition,
        project_context = project_context,
        repository_instructions = repository.repository_instructions,
        repository_selection = repository.repository_selection,
    )


def create_project_context_snapshot(project_id: str, query: str = "") -> ProjectContextSnapshot:
    """Freeze one bounded project context behind an opaque, expiring handle."""
    with _PROJECT_CONTEXT_SNAPSHOT_LOCK:
        resolved = resolve_project_context(
            f"{PROJECT_SESSION_PREFIX}{project_id}",
            query = query,
        )
        if resolved is None or resolved.project_id != project_id:
            raise ProjectContextSnapshotInvalid(
                "The project context snapshot could not be created."
            )
        now = _monotonic()
        snapshot = ProjectContextSnapshot(
            snapshot_id = secrets.token_urlsafe(32),
            project_id = project_id,
            context = resolved,
            expires_at = now + PROJECT_CONTEXT_SNAPSHOT_TTL_SECONDS,
        )
        _prune_project_context_snapshots(now)
        while len(_PROJECT_CONTEXT_SNAPSHOTS) >= MAX_PROJECT_CONTEXT_SNAPSHOTS:
            _PROJECT_CONTEXT_SNAPSHOTS.popitem(last = False)
        _PROJECT_CONTEXT_SNAPSHOTS[snapshot.snapshot_id] = snapshot
    return snapshot


def resolve_project_context_snapshot(
    session_id: Optional[str],
    snapshot_id: Optional[str],
    *,
    query: str = "",
    durable_research_run_id: Optional[str] = None,
    durable_owner_subject: Optional[str] = None,
) -> Optional[ResolvedProjectContext]:
    """Resolve a server snapshot only for the persisted project that minted it."""
    if snapshot_id is None:
        return resolve_project_context(session_id, query = query)
    project_id = project_id_from_persisted_session(session_id)
    if project_id is None:
        raise ProjectContextSnapshotInvalid(
            "The project context snapshot is not valid for this project session."
        )
    now = _monotonic()
    with _PROJECT_CONTEXT_SNAPSHOT_LOCK:
        _prune_project_context_snapshots(now)
        snapshot = _PROJECT_CONTEXT_SNAPSHOTS.get(snapshot_id)
        if snapshot is not None:
            if snapshot.project_id != project_id:
                raise ProjectContextSnapshotInvalid(
                    "The project context snapshot is invalid or expired."
                )
            _PROJECT_CONTEXT_SNAPSHOTS.move_to_end(snapshot_id)
            return snapshot.context

    # Durable research can cross a Studio restart. Unlike renderer-created compare snapshots,
    # it can be hydrated only by the authenticated workflow and exact run binding supplied by
    # the inference route. A normal UI or API request must not replay it as stale context.
    if not durable_research_run_id or not durable_owner_subject:
        raise ProjectContextSnapshotInvalid("The project context snapshot is invalid or expired.")
    from storage import research_runs_db

    durable = research_runs_db.get_project_context_snapshot(
        snapshot_id,
        run_id = durable_research_run_id,
        project_id = project_id,
        owner_subject = durable_owner_subject,
    )
    context = durable.get("context") if durable is not None else None
    if isinstance(context, dict) and isinstance(context.get("addition"), str):
        return ResolvedProjectContext(
            project_id = project_id,
            addition = context["addition"],
            project_context = str(context.get("projectContext") or ""),
            repository_instructions = str(context.get("repositoryInstructions") or ""),
            repository_selection = str(context.get("repositorySelection") or ""),
        )
    raise ProjectContextSnapshotInvalid("The project context snapshot is invalid or expired.")


def project_context_snapshot_response(snapshot: ProjectContextSnapshot) -> dict:
    remaining = max(0.0, snapshot.expires_at - _monotonic())
    return {
        "id": snapshot.snapshot_id,
        "expiresAt": int((_wall_time() + remaining) * 1000),
    }


__all__ = [
    "MAX_PROJECT_GOAL_CHARACTERS",
    "MAX_PROJECT_INSTRUCTIONS_CHARACTERS",
    "MAX_REPOSITORY_QUERY_CHARACTERS",
    "MAX_REPOSITORY_RELEVANT_PATHS",
    "MAX_REPOSITORY_SELECTION_CHARACTERS",
    "MAX_ROOT_AGENTS_BYTES",
    "PROJECT_CONTEXT_MARKER",
    "PROJECT_SESSION_PREFIX",
    "REPOSITORY_INSTRUCTIONS_MARKER",
    "REPOSITORY_SELECTION_MARKER",
    "ProjectContextUnavailable",
    "ProjectContextSnapshot",
    "ProjectContextSnapshotInvalid",
    "ResolvedProjectContext",
    "ResolvedRepositoryPromptContext",
    "create_project_context_snapshot",
    "escape_project_context",
    "fence_project_context_snapshots_for_deletion",
    "project_id_from_persisted_session",
    "project_query_from_messages",
    "resolve_project_context",
    "resolve_project_context_snapshot",
    "resolve_repository_prompt_context",
    "project_context_snapshot_response",
    "strip_server_project_context",
]
