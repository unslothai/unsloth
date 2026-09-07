# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Durable project rules, skills, lifecycle hooks, and typed schedules.

This module is deliberately independent from HTTP routes and the background
task manager.  Routes can validate user input here, prompt assembly can render
the bounded rule and skill guidance, and a scheduler can lease durable work
without making this module responsible for dispatching it.

Skills are instruction-only records.  Installing one stores already-reviewed
guidance whose SHA-256 digest must match the caller's pin; it never imports or
executes installer code.  Lifecycle hooks are Unsloth agent events, not Git
hooks, and execute through the same confined verification boundary as project
checks.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import uuid
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from storage.studio_db import get_connection

from . import verification as verification_module
from .common import AgentWorkspaceError, now_ms, project_workspace


_SCHEMA_LOCK = threading.Lock()
_READY_DATABASES: set[str] = set()
_UNSET = object()

RULE_EFFECTS = frozenset({"allow", "prompt", "deny"})
RULE_MATCH_KINDS = frozenset({"any", "exact", "prefix"})
LIFECYCLE_EVENTS = frozenset(
    {
        "before_agent",
        "after_agent",
        "before_verification",
        "after_verification",
        "before_commit",
        "after_commit",
        "before_handoff",
        "after_handoff",
        "before_scheduled_task",
        "after_scheduled_task",
    }
)
SCHEDULE_KINDS = frozenset({"once", "hourly", "daily", "weekly"})
SCHEDULE_TASK_KINDS = frozenset({"agent", "verification"})
MISFIRE_POLICIES = frozenset({"run_once", "skip"})
SCHEDULE_RUN_STATUSES = frozenset({"completed", "failed", "cancelled"})

MAX_RULE_GUIDANCE_BYTES = 16 * 1024
MAX_SKILL_GUIDANCE_BYTES = 128 * 1024
MAX_RENDER_BYTES = 64 * 1024
MAX_SCHEDULE_PAYLOAD_BYTES = 256 * 1024
MAX_HOOK_COMMAND_BYTES = 8 * 1024
MAX_HOOK_LOG_BYTES = 256 * 1024
MAX_HOOK_TIMEOUT_SECONDS = 3600
MAX_LEASE_MS = 60 * 60 * 1000
MAX_LEASE_BATCH = 100

_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _database_key(conn: sqlite3.Connection) -> str:
    row = conn.execute("PRAGMA database_list").fetchone()
    path = str(row[2])
    return path or f":memory:{id(conn)}"


def _ensure_schema(conn: sqlite3.Connection) -> None:
    key = _database_key(conn)
    if key in _READY_DATABASES:
        return
    with _SCHEMA_LOCK:
        if key in _READY_DATABASES:
            return
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS agent_project_rules (
                id TEXT NOT NULL PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES chat_projects(id) ON DELETE CASCADE,
                name TEXT NOT NULL COLLATE NOCASE,
                tool_name TEXT NOT NULL,
                match_kind TEXT NOT NULL,
                argument_pattern TEXT,
                effect TEXT NOT NULL,
                guidance TEXT NOT NULL,
                priority INTEGER NOT NULL,
                enabled INTEGER NOT NULL,
                revision INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                UNIQUE(project_id, name)
            );
            CREATE INDEX IF NOT EXISTS idx_agent_project_rules_project
                ON agent_project_rules(project_id, enabled, priority DESC, name);

            CREATE TABLE IF NOT EXISTS agent_project_skills (
                id TEXT NOT NULL PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES chat_projects(id) ON DELETE CASCADE,
                name TEXT NOT NULL COLLATE NOCASE,
                description TEXT NOT NULL,
                source TEXT NOT NULL,
                guidance TEXT NOT NULL,
                content_digest TEXT NOT NULL,
                enabled INTEGER NOT NULL,
                revision INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                UNIQUE(project_id, name)
            );
            CREATE INDEX IF NOT EXISTS idx_agent_project_skills_project
                ON agent_project_skills(project_id, enabled, name);

            CREATE TABLE IF NOT EXISTS agent_lifecycle_hooks (
                id TEXT NOT NULL PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES chat_projects(id) ON DELETE CASCADE,
                name TEXT NOT NULL COLLATE NOCASE,
                event TEXT NOT NULL,
                command TEXT NOT NULL,
                position INTEGER NOT NULL,
                required INTEGER NOT NULL,
                timeout_seconds INTEGER NOT NULL,
                log_limit_bytes INTEGER NOT NULL,
                enabled INTEGER NOT NULL,
                revision INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                UNIQUE(project_id, name)
            );
            CREATE INDEX IF NOT EXISTS idx_agent_lifecycle_hooks_project_event
                ON agent_lifecycle_hooks(project_id, event, enabled, position, name);

            CREATE TABLE IF NOT EXISTS agent_lifecycle_hook_runs (
                id TEXT NOT NULL PRIMARY KEY,
                invocation_id TEXT NOT NULL,
                hook_id TEXT REFERENCES agent_lifecycle_hooks(id) ON DELETE SET NULL,
                project_id TEXT NOT NULL REFERENCES chat_projects(id) ON DELETE CASCADE,
                hook_name TEXT NOT NULL,
                event TEXT NOT NULL,
                command TEXT NOT NULL,
                required INTEGER NOT NULL,
                status TEXT NOT NULL,
                result_json TEXT,
                error TEXT,
                started_at INTEGER NOT NULL,
                completed_at INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_agent_lifecycle_hook_runs_project
                ON agent_lifecycle_hook_runs(project_id, started_at DESC);
            CREATE INDEX IF NOT EXISTS idx_agent_lifecycle_hook_runs_invocation
                ON agent_lifecycle_hook_runs(invocation_id, started_at, id);

            CREATE TABLE IF NOT EXISTS agent_schedules (
                id TEXT NOT NULL PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES chat_projects(id) ON DELETE CASCADE,
                name TEXT NOT NULL COLLATE NOCASE,
                task_kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                cadence_json TEXT NOT NULL,
                timezone TEXT NOT NULL,
                misfire_policy TEXT NOT NULL,
                next_run_at INTEGER,
                last_run_at INTEGER,
                last_status TEXT,
                enabled INTEGER NOT NULL,
                lease_owner TEXT,
                lease_expires_at INTEGER,
                lease_run_id TEXT,
                revision INTEGER NOT NULL DEFAULT 0,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                UNIQUE(project_id, name)
            );
            CREATE INDEX IF NOT EXISTS idx_agent_schedules_due
                ON agent_schedules(enabled, next_run_at, lease_expires_at);

            CREATE TABLE IF NOT EXISTS agent_schedule_runs (
                id TEXT NOT NULL PRIMARY KEY,
                schedule_id TEXT REFERENCES agent_schedules(id) ON DELETE SET NULL,
                project_id TEXT NOT NULL REFERENCES chat_projects(id) ON DELETE CASCADE,
                schedule_name TEXT NOT NULL,
                task_kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                scheduled_for INTEGER NOT NULL,
                lease_owner TEXT,
                status TEXT NOT NULL,
                error TEXT,
                started_at INTEGER NOT NULL,
                completed_at INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_agent_schedule_runs_project
                ON agent_schedule_runs(project_id, started_at DESC);
            """
        )
        conn.commit()
        _READY_DATABASES.add(key)


def connection() -> sqlite3.Connection:
    conn = get_connection()
    _ensure_schema(conn)
    return conn


def _json(value: Any, *, limit: int, label: str) -> str:
    try:
        encoded = json.dumps(value, separators = (",", ":"), sort_keys = True)
    except (TypeError, ValueError) as exc:
        raise AgentWorkspaceError(f"{label} must be valid JSON.") from exc
    if len(encoded.encode("utf-8")) > limit:
        raise AgentWorkspaceError(f"{label} is too large.")
    return encoded


def _loads(value: Optional[str], default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _required_text(value: Any, *, label: str, maximum: int) -> str:
    text = str(value or "").strip()
    if not text or "\x00" in text or len(text.encode("utf-8")) > maximum:
        raise AgentWorkspaceError(f"{label} is invalid.")
    return text


def _optional_text(value: Any, *, label: str, maximum: int) -> str:
    text = str(value or "").strip()
    if "\x00" in text or len(text.encode("utf-8")) > maximum:
        raise AgentWorkspaceError(f"{label} is invalid.")
    return text


def _raw_text(
    value: Any,
    *,
    label: str,
    maximum: int,
    required: bool = False,
) -> str:
    text = str(value or "")
    if "\x00" in text or len(text.encode("utf-8")) > maximum or (required and not text.strip()):
        raise AgentWorkspaceError(f"{label} is invalid.")
    return text


def _validate_revision(expected_revision: int) -> int:
    if isinstance(expected_revision, bool):
        raise AgentWorkspaceError("Expected revision is invalid.")
    try:
        value = int(expected_revision)
    except (TypeError, ValueError) as exc:
        raise AgentWorkspaceError("Expected revision is invalid.") from exc
    if value < 0:
        raise AgentWorkspaceError("Expected revision is invalid.")
    return value


def _raise_conflict(label: str) -> None:
    raise AgentWorkspaceError(f"{label} changed in another session. Refresh and retry.")


def _translate_insert_error(exc: sqlite3.IntegrityError, label: str) -> None:
    detail = str(exc).casefold()
    if "foreign key" in detail:
        raise AgentWorkspaceError("Project not found.") from exc
    if "unique" in detail:
        raise AgentWorkspaceError(f"A {label.casefold()} with that name already exists.") from exc
    raise AgentWorkspaceError(f"{label} could not be saved.") from exc


def _rule(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "projectId": row["project_id"],
        "name": row["name"],
        "toolName": row["tool_name"],
        "matchKind": row["match_kind"],
        "argumentPattern": row["argument_pattern"],
        "effect": row["effect"],
        "guidance": row["guidance"],
        "priority": row["priority"],
        "enabled": bool(row["enabled"]),
        "revision": row["revision"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def _normalize_rule(
    *,
    name: Any,
    tool_name: Any,
    match_kind: Any,
    argument_pattern: Any,
    effect: Any,
    guidance: Any,
    priority: Any,
) -> dict:
    normalized_name = _required_text(name, label = "Rule name", maximum = 160)
    normalized_tool = _required_text(tool_name, label = "Rule tool name", maximum = 160)
    normalized_match = str(match_kind or "any").strip().casefold()
    normalized_effect = str(effect or "prompt").strip().casefold()
    if normalized_match not in RULE_MATCH_KINDS:
        raise AgentWorkspaceError("Rule match kind must be any, exact, or prefix.")
    if normalized_effect not in RULE_EFFECTS:
        raise AgentWorkspaceError("Rule effect must be allow, prompt, or deny.")
    pattern = _raw_text(
        argument_pattern,
        label = "Rule argument pattern",
        maximum = 4096,
    )
    if normalized_match == "any":
        pattern = ""
    elif not pattern:
        raise AgentWorkspaceError("Exact and prefix rules require an argument pattern.")
    normalized_guidance = _optional_text(
        guidance,
        label = "Rule guidance",
        maximum = MAX_RULE_GUIDANCE_BYTES,
    )
    if isinstance(priority, bool):
        raise AgentWorkspaceError("Rule priority is invalid.")
    try:
        normalized_priority = int(priority)
    except (TypeError, ValueError) as exc:
        raise AgentWorkspaceError("Rule priority is invalid.") from exc
    if normalized_priority < -1000 or normalized_priority > 1000:
        raise AgentWorkspaceError("Rule priority must be between -1000 and 1000.")
    return {
        "name": normalized_name,
        "toolName": normalized_tool,
        "matchKind": normalized_match,
        "argumentPattern": pattern or None,
        "effect": normalized_effect,
        "guidance": normalized_guidance,
        "priority": normalized_priority,
    }


def create_project_rule(
    project_id: str,
    *,
    name: str,
    tool_name: str,
    effect: str,
    match_kind: str = "any",
    argument_pattern: Optional[str] = None,
    guidance: str = "",
    priority: int = 0,
    enabled: bool = True,
) -> dict:
    values = _normalize_rule(
        name = name,
        tool_name = tool_name,
        match_kind = match_kind,
        argument_pattern = argument_pattern,
        effect = effect,
        guidance = guidance,
        priority = priority,
    )
    rule_id = str(uuid.uuid4())
    current = now_ms()
    conn = connection()
    try:
        conn.execute(
            """
            INSERT INTO agent_project_rules(
                id, project_id, name, tool_name, match_kind, argument_pattern,
                effect, guidance, priority, enabled, revision, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (
                rule_id,
                project_id,
                values["name"],
                values["toolName"],
                values["matchKind"],
                values["argumentPattern"],
                values["effect"],
                values["guidance"],
                values["priority"],
                int(bool(enabled)),
                current,
                current,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM agent_project_rules WHERE id = ?", (rule_id,)).fetchone()
        return _rule(row)
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        _translate_insert_error(exc, "Rule")
        raise AssertionError("unreachable")
    finally:
        conn.close()


def get_project_rule(rule_id: str) -> Optional[dict]:
    conn = connection()
    try:
        row = conn.execute("SELECT * FROM agent_project_rules WHERE id = ?", (rule_id,)).fetchone()
        return _rule(row) if row else None
    finally:
        conn.close()


def list_project_rules(project_id: str, *, enabled_only: bool = False) -> list[dict]:
    conn = connection()
    try:
        clause = " AND enabled = 1" if enabled_only else ""
        rows = conn.execute(
            f"""
            SELECT * FROM agent_project_rules
            WHERE project_id = ?{clause}
            ORDER BY priority DESC, name COLLATE NOCASE, id
            """,
            (project_id,),
        ).fetchall()
        return [_rule(row) for row in rows]
    finally:
        conn.close()


def resolve_project_rule(project_id: str, tool_name: str, arguments: Any) -> Optional[dict]:
    """Return the highest-priority enabled structured rule for one tool call."""
    normalized_tool = _required_text(tool_name, label = "Tool name", maximum = 160).casefold()
    if isinstance(arguments, str):
        rendered_arguments = _raw_text(
            arguments,
            label = "Tool arguments",
            maximum = MAX_SCHEDULE_PAYLOAD_BYTES,
        )
    else:
        rendered_arguments = _json(
            arguments,
            limit = MAX_SCHEDULE_PAYLOAD_BYTES,
            label = "Tool arguments",
        )
    for rule in list_project_rules(project_id, enabled_only = True):
        if rule["toolName"].casefold() not in {"*", normalized_tool}:
            continue
        pattern = rule["argumentPattern"] or ""
        if rule["matchKind"] == "exact" and rendered_arguments != pattern:
            continue
        if rule["matchKind"] == "prefix" and not rendered_arguments.startswith(pattern):
            continue
        return rule
    return None


def update_project_rule(
    rule_id: str,
    *,
    expected_revision: int,
    name: Any = _UNSET,
    tool_name: Any = _UNSET,
    effect: Any = _UNSET,
    match_kind: Any = _UNSET,
    argument_pattern: Any = _UNSET,
    guidance: Any = _UNSET,
    priority: Any = _UNSET,
    enabled: Any = _UNSET,
) -> Optional[dict]:
    expected = _validate_revision(expected_revision)
    conn = connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM agent_project_rules WHERE id = ?", (rule_id,)).fetchone()
        if row is None:
            conn.rollback()
            return None
        if row["revision"] != expected:
            _raise_conflict("Rule")
        values = _normalize_rule(
            name = row["name"] if name is _UNSET else name,
            tool_name = row["tool_name"] if tool_name is _UNSET else tool_name,
            match_kind = row["match_kind"] if match_kind is _UNSET else match_kind,
            argument_pattern = (
                row["argument_pattern"] if argument_pattern is _UNSET else argument_pattern
            ),
            effect = row["effect"] if effect is _UNSET else effect,
            guidance = row["guidance"] if guidance is _UNSET else guidance,
            priority = row["priority"] if priority is _UNSET else priority,
        )
        is_enabled = bool(row["enabled"]) if enabled is _UNSET else bool(enabled)
        try:
            cursor = conn.execute(
                """
                UPDATE agent_project_rules
                SET name = ?, tool_name = ?, match_kind = ?, argument_pattern = ?,
                    effect = ?, guidance = ?, priority = ?, enabled = ?,
                    revision = revision + 1, updated_at = ?
                WHERE id = ? AND revision = ?
                """,
                (
                    values["name"],
                    values["toolName"],
                    values["matchKind"],
                    values["argumentPattern"],
                    values["effect"],
                    values["guidance"],
                    values["priority"],
                    int(is_enabled),
                    now_ms(),
                    rule_id,
                    expected,
                ),
            )
        except sqlite3.IntegrityError as exc:
            _translate_insert_error(exc, "Rule")
            raise AssertionError("unreachable")
        if not cursor.rowcount:
            _raise_conflict("Rule")
        conn.commit()
        updated = conn.execute(
            "SELECT * FROM agent_project_rules WHERE id = ?", (rule_id,)
        ).fetchone()
        return _rule(updated)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def delete_project_rule(rule_id: str, *, expected_revision: int) -> bool:
    expected = _validate_revision(expected_revision)
    conn = connection()
    try:
        cursor = conn.execute(
            "DELETE FROM agent_project_rules WHERE id = ? AND revision = ?",
            (rule_id, expected),
        )
        if not cursor.rowcount:
            exists = conn.execute(
                "SELECT 1 FROM agent_project_rules WHERE id = ?", (rule_id,)
            ).fetchone()
            if exists:
                _raise_conflict("Rule")
            conn.rollback()
            return False
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def skill_digest(guidance: str) -> str:
    return hashlib.sha256(guidance.encode("utf-8")).hexdigest()


def _validate_digest(value: Any) -> str:
    digest = str(value or "").strip().casefold()
    if not _DIGEST_PATTERN.fullmatch(digest):
        raise AgentWorkspaceError("Skill digest must be a lowercase SHA-256 digest.")
    return digest


def _skill(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "projectId": row["project_id"],
        "name": row["name"],
        "description": row["description"],
        "source": row["source"],
        "guidance": row["guidance"],
        "contentDigest": row["content_digest"],
        "enabled": bool(row["enabled"]),
        "revision": row["revision"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def _normalize_skill(
    *, name: Any, description: Any, source: Any, guidance: Any, content_digest: Any
) -> dict:
    normalized_guidance = _raw_text(
        guidance,
        label = "Skill guidance",
        maximum = MAX_SKILL_GUIDANCE_BYTES,
        required = True,
    )
    digest = _validate_digest(content_digest)
    if skill_digest(normalized_guidance) != digest:
        raise AgentWorkspaceError("Skill guidance does not match its pinned SHA-256 digest.")
    return {
        "name": _required_text(name, label = "Skill name", maximum = 160),
        "description": _optional_text(
            description,
            label = "Skill description",
            maximum = 2000,
        ),
        "source": _required_text(source, label = "Skill source", maximum = 1000),
        "guidance": normalized_guidance,
        "contentDigest": digest,
    }


def install_project_skill(
    project_id: str,
    *,
    name: str,
    source: str,
    guidance: str,
    content_digest: str,
    description: str = "",
    enabled: bool = False,
) -> dict:
    """Store reviewed instruction text only; no source code or installer is run."""
    values = _normalize_skill(
        name = name,
        description = description,
        source = source,
        guidance = guidance,
        content_digest = content_digest,
    )
    skill_id = str(uuid.uuid4())
    current = now_ms()
    conn = connection()
    try:
        conn.execute(
            """
            INSERT INTO agent_project_skills(
                id, project_id, name, description, source, guidance,
                content_digest, enabled, revision, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (
                skill_id,
                project_id,
                values["name"],
                values["description"],
                values["source"],
                values["guidance"],
                values["contentDigest"],
                int(bool(enabled)),
                current,
                current,
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM agent_project_skills WHERE id = ?", (skill_id,)
        ).fetchone()
        return _skill(row)
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        _translate_insert_error(exc, "Skill")
        raise AssertionError("unreachable")
    finally:
        conn.close()


def get_project_skill(skill_id: str) -> Optional[dict]:
    conn = connection()
    try:
        row = conn.execute(
            "SELECT * FROM agent_project_skills WHERE id = ?", (skill_id,)
        ).fetchone()
        return _skill(row) if row else None
    finally:
        conn.close()


def get_enabled_project_skill(project_id: str, skill_id: str) -> Optional[dict]:
    """Return one enabled skill only when it belongs to the active project."""
    normalized_id = _required_text(skill_id, label = "Skill id", maximum = 128)
    conn = connection()
    try:
        row = conn.execute(
            """
            SELECT * FROM agent_project_skills
            WHERE id = ? AND project_id = ? AND enabled = 1
            """,
            (normalized_id, project_id),
        ).fetchone()
        return _skill(row) if row else None
    finally:
        conn.close()


def list_project_skills(project_id: str, *, enabled_only: bool = False) -> list[dict]:
    conn = connection()
    try:
        clause = " AND enabled = 1" if enabled_only else ""
        rows = conn.execute(
            f"""
            SELECT * FROM agent_project_skills
            WHERE project_id = ?{clause}
            ORDER BY name COLLATE NOCASE, id
            """,
            (project_id,),
        ).fetchall()
        return [_skill(row) for row in rows]
    finally:
        conn.close()


def update_project_skill(
    skill_id: str,
    *,
    expected_revision: int,
    name: Any = _UNSET,
    description: Any = _UNSET,
    source: Any = _UNSET,
    guidance: Any = _UNSET,
    content_digest: Any = _UNSET,
    enabled: Any = _UNSET,
) -> Optional[dict]:
    expected = _validate_revision(expected_revision)
    conn = connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM agent_project_skills WHERE id = ?", (skill_id,)
        ).fetchone()
        if row is None:
            conn.rollback()
            return None
        if row["revision"] != expected:
            _raise_conflict("Skill")
        next_guidance = row["guidance"] if guidance is _UNSET else guidance
        if guidance is not _UNSET and content_digest is _UNSET:
            raise AgentWorkspaceError("Changing skill guidance requires a new pinned digest.")
        values = _normalize_skill(
            name = row["name"] if name is _UNSET else name,
            description = row["description"] if description is _UNSET else description,
            source = row["source"] if source is _UNSET else source,
            guidance = next_guidance,
            content_digest = (row["content_digest"] if content_digest is _UNSET else content_digest),
        )
        try:
            cursor = conn.execute(
                """
                UPDATE agent_project_skills
                SET name = ?, description = ?, source = ?, guidance = ?,
                    content_digest = ?, enabled = ?, revision = revision + 1,
                    updated_at = ?
                WHERE id = ? AND revision = ?
                """,
                (
                    values["name"],
                    values["description"],
                    values["source"],
                    values["guidance"],
                    values["contentDigest"],
                    int(bool(row["enabled"]) if enabled is _UNSET else bool(enabled)),
                    now_ms(),
                    skill_id,
                    expected,
                ),
            )
        except sqlite3.IntegrityError as exc:
            _translate_insert_error(exc, "Skill")
            raise AssertionError("unreachable")
        if not cursor.rowcount:
            _raise_conflict("Skill")
        conn.commit()
        updated = conn.execute(
            "SELECT * FROM agent_project_skills WHERE id = ?", (skill_id,)
        ).fetchone()
        return _skill(updated)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def delete_project_skill(skill_id: str, *, expected_revision: int) -> bool:
    return _delete_revisioned("agent_project_skills", skill_id, expected_revision, "Skill")


def _bounded_json_guidance(header: str, records: list[dict], limit: int) -> str:
    if isinstance(limit, bool) or limit < 256 or limit > MAX_RENDER_BYTES:
        raise AgentWorkspaceError("Guidance render limit is invalid.")
    selected: list[dict] = []
    for record in records:
        candidate = selected + [record]
        rendered = (
            header
            + "\n"
            + json.dumps(
                candidate,
                ensure_ascii = False,
                separators = (",", ":"),
                sort_keys = True,
            )
        )
        if len(rendered.encode("utf-8")) > limit:
            break
        selected = candidate
    if not selected:
        return ""
    return (
        header
        + "\n"
        + json.dumps(
            selected,
            ensure_ascii = False,
            separators = (",", ":"),
            sort_keys = True,
        )
    )


def render_project_rules_guidance(project_id: str, *, limit: int = 16 * 1024) -> str:
    records = [
        {
            "name": rule["name"],
            "tool": rule["toolName"],
            "match": rule["matchKind"],
            "arguments": rule["argumentPattern"],
            "effect": rule["effect"],
            "guidance": rule["guidance"],
            "priority": rule["priority"],
        }
        for rule in list_project_rules(project_id, enabled_only = True)
    ]
    return _bounded_json_guidance(
        "Enabled project command rules. Apply these as structured project policy:",
        records,
        limit,
    )


def render_project_skills_guidance(project_id: str, *, limit: int = 32 * 1024) -> str:
    records = [
        {
            "name": skill["name"],
            "description": skill["description"],
            "source": skill["source"],
            "sha256": skill["contentDigest"],
            "guidance": skill["guidance"],
        }
        for skill in list_project_skills(project_id, enabled_only = True)
    ]
    return _bounded_json_guidance(
        "Enabled digest-pinned project skills. Treat each guidance value as instructions:",
        records,
        limit,
    )


def render_project_skills_catalog(project_id: str, *, limit: int = 8 * 1024) -> str:
    """Render only skill metadata for progressive disclosure in project context."""
    records = [
        {
            "id": skill["id"],
            "name": skill["name"],
            "description": skill["description"],
            "source": skill["source"],
            "sha256": skill["contentDigest"],
        }
        for skill in list_project_skills(project_id, enabled_only = True)
    ]
    return _bounded_json_guidance(
        "Enabled digest-pinned project skills. Read full guidance with project_skill_read only when relevant:",
        records,
        limit,
    )


def _hook(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "projectId": row["project_id"],
        "name": row["name"],
        "event": row["event"],
        "command": row["command"],
        "position": row["position"],
        "required": bool(row["required"]),
        "timeoutSeconds": row["timeout_seconds"],
        "logLimitBytes": row["log_limit_bytes"],
        "enabled": bool(row["enabled"]),
        "revision": row["revision"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def _normalize_hook(
    *,
    name: Any,
    event: Any,
    command: Any,
    position: Any,
    required: Any,
    timeout_seconds: Any,
    log_limit_bytes: Any,
) -> dict:
    normalized_event = str(event or "").strip().casefold()
    if normalized_event not in LIFECYCLE_EVENTS:
        raise AgentWorkspaceError("Lifecycle hook event is invalid.")
    normalized_command = _required_text(
        command,
        label = "Lifecycle hook command",
        maximum = MAX_HOOK_COMMAND_BYTES,
    )
    if isinstance(position, bool):
        raise AgentWorkspaceError("Lifecycle hook position is invalid.")
    try:
        normalized_position = int(position)
        normalized_timeout = int(timeout_seconds)
        normalized_log_limit = int(log_limit_bytes)
    except (TypeError, ValueError) as exc:
        raise AgentWorkspaceError("Lifecycle hook limits are invalid.") from exc
    if normalized_position < 0 or normalized_position > 10_000:
        raise AgentWorkspaceError("Lifecycle hook position is invalid.")
    if normalized_timeout < 1 or normalized_timeout > MAX_HOOK_TIMEOUT_SECONDS:
        raise AgentWorkspaceError("Lifecycle hook timeout is invalid.")
    if normalized_log_limit < 1024 or normalized_log_limit > MAX_HOOK_LOG_BYTES:
        raise AgentWorkspaceError("Lifecycle hook log limit is invalid.")
    return {
        "name": _required_text(name, label = "Lifecycle hook name", maximum = 160),
        "event": normalized_event,
        "command": normalized_command,
        "position": normalized_position,
        "required": bool(required),
        "timeoutSeconds": normalized_timeout,
        "logLimitBytes": normalized_log_limit,
    }


def create_lifecycle_hook(
    project_id: str,
    *,
    name: str,
    event: str,
    command: str,
    position: int = 0,
    required: bool = True,
    timeout_seconds: int = 300,
    log_limit_bytes: int = 64 * 1024,
    enabled: bool = True,
) -> dict:
    values = _normalize_hook(
        name = name,
        event = event,
        command = command,
        position = position,
        required = required,
        timeout_seconds = timeout_seconds,
        log_limit_bytes = log_limit_bytes,
    )
    hook_id = str(uuid.uuid4())
    current = now_ms()
    conn = connection()
    try:
        conn.execute(
            """
            INSERT INTO agent_lifecycle_hooks(
                id, project_id, name, event, command, position, required,
                timeout_seconds, log_limit_bytes, enabled, revision,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (
                hook_id,
                project_id,
                values["name"],
                values["event"],
                values["command"],
                values["position"],
                int(values["required"]),
                values["timeoutSeconds"],
                values["logLimitBytes"],
                int(bool(enabled)),
                current,
                current,
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM agent_lifecycle_hooks WHERE id = ?", (hook_id,)
        ).fetchone()
        return _hook(row)
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        _translate_insert_error(exc, "Lifecycle hook")
        raise AssertionError("unreachable")
    finally:
        conn.close()


def get_lifecycle_hook(hook_id: str) -> Optional[dict]:
    conn = connection()
    try:
        row = conn.execute(
            "SELECT * FROM agent_lifecycle_hooks WHERE id = ?", (hook_id,)
        ).fetchone()
        return _hook(row) if row else None
    finally:
        conn.close()


def list_lifecycle_hooks(
    project_id: str,
    *,
    event: Optional[str] = None,
    enabled_only: bool = False,
) -> list[dict]:
    if event is not None and event not in LIFECYCLE_EVENTS:
        raise AgentWorkspaceError("Lifecycle hook event is invalid.")
    conditions = ["project_id = ?"]
    values: list[Any] = [project_id]
    if event is not None:
        conditions.append("event = ?")
        values.append(event)
    if enabled_only:
        conditions.append("enabled = 1")
    conn = connection()
    try:
        rows = conn.execute(
            f"""
            SELECT * FROM agent_lifecycle_hooks
            WHERE {' AND '.join(conditions)}
            ORDER BY event, position, name COLLATE NOCASE, id
            """,
            values,
        ).fetchall()
        return [_hook(row) for row in rows]
    finally:
        conn.close()


def update_lifecycle_hook(
    hook_id: str,
    *,
    expected_revision: int,
    name: Any = _UNSET,
    event: Any = _UNSET,
    command: Any = _UNSET,
    position: Any = _UNSET,
    required: Any = _UNSET,
    timeout_seconds: Any = _UNSET,
    log_limit_bytes: Any = _UNSET,
    enabled: Any = _UNSET,
) -> Optional[dict]:
    expected = _validate_revision(expected_revision)
    conn = connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT * FROM agent_lifecycle_hooks WHERE id = ?", (hook_id,)
        ).fetchone()
        if row is None:
            conn.rollback()
            return None
        if row["revision"] != expected:
            _raise_conflict("Lifecycle hook")
        values = _normalize_hook(
            name = row["name"] if name is _UNSET else name,
            event = row["event"] if event is _UNSET else event,
            command = row["command"] if command is _UNSET else command,
            position = row["position"] if position is _UNSET else position,
            required = bool(row["required"]) if required is _UNSET else required,
            timeout_seconds = (
                row["timeout_seconds"] if timeout_seconds is _UNSET else timeout_seconds
            ),
            log_limit_bytes = (
                row["log_limit_bytes"] if log_limit_bytes is _UNSET else log_limit_bytes
            ),
        )
        try:
            cursor = conn.execute(
                """
                UPDATE agent_lifecycle_hooks
                SET name = ?, event = ?, command = ?, position = ?, required = ?,
                    timeout_seconds = ?, log_limit_bytes = ?, enabled = ?,
                    revision = revision + 1, updated_at = ?
                WHERE id = ? AND revision = ?
                """,
                (
                    values["name"],
                    values["event"],
                    values["command"],
                    values["position"],
                    int(values["required"]),
                    values["timeoutSeconds"],
                    values["logLimitBytes"],
                    int(bool(row["enabled"]) if enabled is _UNSET else bool(enabled)),
                    now_ms(),
                    hook_id,
                    expected,
                ),
            )
        except sqlite3.IntegrityError as exc:
            _translate_insert_error(exc, "Lifecycle hook")
            raise AssertionError("unreachable")
        if not cursor.rowcount:
            _raise_conflict("Lifecycle hook")
        conn.commit()
        updated = conn.execute(
            "SELECT * FROM agent_lifecycle_hooks WHERE id = ?", (hook_id,)
        ).fetchone()
        return _hook(updated)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def delete_lifecycle_hook(hook_id: str, *, expected_revision: int) -> bool:
    return _delete_revisioned("agent_lifecycle_hooks", hook_id, expected_revision, "Lifecycle hook")


def _hook_run(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "invocationId": row["invocation_id"],
        "hookId": row["hook_id"],
        "projectId": row["project_id"],
        "hookName": row["hook_name"],
        "event": row["event"],
        "command": row["command"],
        "required": bool(row["required"]),
        "status": row["status"],
        "result": _loads(row["result_json"], None),
        "error": row["error"],
        "startedAt": row["started_at"],
        "completedAt": row["completed_at"],
    }


def _begin_hook_run(invocation_id: str, hook: dict) -> dict:
    run_id = str(uuid.uuid4())
    started = now_ms()
    conn = connection()
    try:
        conn.execute(
            """
            INSERT INTO agent_lifecycle_hook_runs(
                id, invocation_id, hook_id, project_id, hook_name, event,
                command, required, status, started_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)
            """,
            (
                run_id,
                invocation_id,
                hook["id"],
                hook["projectId"],
                hook["name"],
                hook["event"],
                hook["command"],
                int(hook["required"]),
                started,
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM agent_lifecycle_hook_runs WHERE id = ?", (run_id,)
        ).fetchone()
        return _hook_run(row)
    finally:
        conn.close()


def _finish_hook_run(
    run_id: str, *, status: str, result: Optional[dict], error: Optional[str]
) -> dict:
    encoded = (
        _json(result, limit = MAX_HOOK_LOG_BYTES + 16 * 1024, label = "Hook result")
        if result is not None
        else None
    )
    conn = connection()
    try:
        conn.execute(
            """
            UPDATE agent_lifecycle_hook_runs
            SET status = ?, result_json = ?, error = ?, completed_at = ?
            WHERE id = ? AND status = 'running'
            """,
            (status, encoded, error, now_ms(), run_id),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM agent_lifecycle_hook_runs WHERE id = ?", (run_id,)
        ).fetchone()
        return _hook_run(row)
    finally:
        conn.close()


def run_lifecycle_hooks(
    project_id: str,
    event: str,
    *,
    cancel_event: Optional[threading.Event] = None,
) -> dict:
    """Run enabled Unsloth lifecycle hooks in order and persist every result."""
    if event not in LIFECYCLE_EVENTS:
        raise AgentWorkspaceError("Lifecycle hook event is invalid.")
    hooks = list_lifecycle_hooks(project_id, event = event, enabled_only = True)
    invocation_id = str(uuid.uuid4())
    if not hooks:
        return {
            "invocationId": invocation_id,
            "projectId": project_id,
            "event": event,
            "status": "passed",
            "requiredFailure": False,
            "runs": [],
        }
    workspace = project_workspace(project_id)
    expected_identity = (
        (workspace.device_id, workspace.file_id)
        if workspace.device_id is not None and workspace.file_id is not None
        else None
    )
    cancellation = cancel_event or threading.Event()
    rendered_runs: list[dict] = []
    required_failure = False
    for hook in hooks:
        pending = _begin_hook_run(invocation_id, hook)
        result: Optional[dict] = None
        error: Optional[str] = None
        try:
            result = verification_module.execute_check(
                {
                    "name": hook["name"],
                    "kind": "lifecycle_hook",
                    "command": hook["command"],
                    "required": hook["required"],
                    "timeoutSeconds": hook["timeoutSeconds"],
                    "logLimitBytes": hook["logLimitBytes"],
                    "projectId": project_id,
                },
                root = workspace.root,
                cancel_event = cancellation,
                run_id = pending["id"],
                expected_root_identity = expected_identity,
            )
            status = str(result.get("status") or "failed")
        except AgentWorkspaceError as exc:
            status = "failed"
            error = str(exc)
        except Exception:
            status = "failed"
            error = "Lifecycle hook execution failed."
        finished = _finish_hook_run(
            pending["id"],
            status = status,
            result = result,
            error = error,
        )
        rendered_runs.append(finished)
        if hook["required"] and status != "passed":
            required_failure = True
            break
    return {
        "invocationId": invocation_id,
        "projectId": project_id,
        "event": event,
        "status": "failed" if required_failure else "passed",
        "requiredFailure": required_failure,
        "runs": rendered_runs,
    }


def list_lifecycle_hook_runs(
    project_id: str,
    *,
    invocation_id: Optional[str] = None,
    limit: int = 100,
) -> list[dict]:
    if isinstance(limit, bool) or limit < 1 or limit > 1000:
        raise AgentWorkspaceError("Hook run limit is invalid.")
    conditions = ["project_id = ?"]
    values: list[Any] = [project_id]
    if invocation_id is not None:
        conditions.append("invocation_id = ?")
        values.append(invocation_id)
    values.append(limit)
    conn = connection()
    try:
        rows = conn.execute(
            f"""
            SELECT * FROM agent_lifecycle_hook_runs
            WHERE {' AND '.join(conditions)}
            ORDER BY started_at DESC, id DESC LIMIT ?
            """,
            values,
        ).fetchall()
        return [_hook_run(row) for row in rows]
    finally:
        conn.close()


def _zone(timezone_name: Any) -> tuple[str, ZoneInfo]:
    name = _required_text(timezone_name, label = "Schedule timezone", maximum = 128)
    try:
        return name, ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise AgentWorkspaceError("Schedule timezone is invalid.") from exc


def _int_field(value: Any, *, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise AgentWorkspaceError(f"{label} is invalid.")
    try:
        rendered = int(value)
    except (TypeError, ValueError) as exc:
        raise AgentWorkspaceError(f"{label} is invalid.") from exc
    if rendered < minimum or rendered > maximum:
        raise AgentWorkspaceError(f"{label} is invalid.")
    return rendered


def normalize_schedule_cadence(cadence: Any) -> dict:
    if not isinstance(cadence, dict):
        raise AgentWorkspaceError("Schedule cadence is invalid.")
    kind = str(cadence.get("kind") or "").strip().casefold()
    if kind not in SCHEDULE_KINDS:
        raise AgentWorkspaceError("Schedule cadence kind is invalid.")
    if kind == "once":
        return {
            "kind": kind,
            "at": _int_field(
                cadence.get("at"),
                label = "One-time schedule timestamp",
                minimum = 0,
                maximum = 253_402_300_799_999,
            ),
        }
    minute = _int_field(
        cadence.get("minute", 0),
        label = "Schedule minute",
        minimum = 0,
        maximum = 59,
    )
    if kind == "hourly":
        return {"kind": kind, "minute": minute}
    hour = _int_field(
        cadence.get("hour"),
        label = "Schedule hour",
        minimum = 0,
        maximum = 23,
    )
    if kind == "daily":
        return {"kind": kind, "hour": hour, "minute": minute}
    return {
        "kind": kind,
        "weekday": _int_field(
            cadence.get("weekday"),
            label = "Schedule weekday",
            minimum = 0,
            maximum = 6,
        ),
        "hour": hour,
        "minute": minute,
    }


def _valid_local_instant(
    day: date, *, hour: int, minute: int, zone: ZoneInfo
) -> Optional[datetime]:
    naive = datetime.combine(day, datetime_time(hour = hour, minute = minute))
    aware = naive.replace(tzinfo = zone, fold = 0)
    round_trip = aware.astimezone(timezone.utc).astimezone(zone).replace(tzinfo = None)
    return aware if round_trip == naive else None


def next_schedule_time(cadence: dict, timezone_name: str, after_ms: int) -> Optional[int]:
    """Return the next occurrence strictly after ``after_ms``."""
    normalized = normalize_schedule_cadence(cadence)
    _name, zone = _zone(timezone_name)
    after = datetime.fromtimestamp(after_ms / 1000, tz = timezone.utc)
    kind = normalized["kind"]
    if kind == "once":
        return normalized["at"] if normalized["at"] > after_ms else None
    local_after = after.astimezone(zone)
    if kind == "hourly":
        naive_hour = local_after.replace(minute = 0, second = 0, microsecond = 0, tzinfo = None)
        for offset in range(0, 73):
            candidate_naive = naive_hour + timedelta(hours = offset)
            candidate = _valid_local_instant(
                candidate_naive.date(),
                hour = candidate_naive.hour,
                minute = normalized["minute"],
                zone = zone,
            )
            if candidate is not None and candidate.astimezone(timezone.utc) > after:
                return int(candidate.timestamp() * 1000)
        raise AgentWorkspaceError("Could not calculate the next hourly schedule occurrence.")
    day = local_after.date()
    for offset in range(0, 740):
        candidate_day = day + timedelta(days = offset)
        if kind == "weekly" and candidate_day.weekday() != normalized["weekday"]:
            continue
        candidate = _valid_local_instant(
            candidate_day,
            hour = normalized["hour"],
            minute = normalized["minute"],
            zone = zone,
        )
        if candidate is not None and candidate.astimezone(timezone.utc) > after:
            return int(candidate.timestamp() * 1000)
    raise AgentWorkspaceError("Could not calculate the next schedule occurrence.")


def _schedule(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "projectId": row["project_id"],
        "name": row["name"],
        "taskKind": row["task_kind"],
        "payload": _loads(row["payload_json"], {}),
        "cadence": _loads(row["cadence_json"], {}),
        "timezone": row["timezone"],
        "misfirePolicy": row["misfire_policy"],
        "nextRunAt": row["next_run_at"],
        "lastRunAt": row["last_run_at"],
        "lastStatus": row["last_status"],
        "enabled": bool(row["enabled"]),
        "leased": bool(row["lease_owner"]),
        "leaseExpiresAt": row["lease_expires_at"],
        "revision": row["revision"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def _normalize_schedule(
    *,
    name: Any,
    task_kind: Any,
    payload: Any,
    cadence: Any,
    timezone_name: Any,
    misfire_policy: Any,
) -> dict:
    normalized_task_kind = str(task_kind or "").strip().casefold()
    if normalized_task_kind not in SCHEDULE_TASK_KINDS:
        raise AgentWorkspaceError("Schedule task kind must be agent or verification.")
    if not isinstance(payload, dict):
        raise AgentWorkspaceError("Schedule payload must be a JSON object.")
    encoded_payload = _json(
        payload,
        limit = MAX_SCHEDULE_PAYLOAD_BYTES,
        label = "Schedule payload",
    )
    normalized_cadence = normalize_schedule_cadence(cadence)
    zone_name, _parsed_zone = _zone(timezone_name)
    normalized_misfire = str(misfire_policy or "skip").strip().casefold()
    if normalized_misfire not in MISFIRE_POLICIES:
        raise AgentWorkspaceError("Schedule misfire policy must be run_once or skip.")
    return {
        "name": _required_text(name, label = "Schedule name", maximum = 160),
        "taskKind": normalized_task_kind,
        "payload": payload,
        "payloadJson": encoded_payload,
        "cadence": normalized_cadence,
        "cadenceJson": _json(normalized_cadence, limit = 4096, label = "Schedule cadence"),
        "timezone": zone_name,
        "misfirePolicy": normalized_misfire,
    }


def create_schedule(
    project_id: str,
    *,
    name: str,
    task_kind: str,
    payload: dict,
    cadence: dict,
    timezone_name: str,
    misfire_policy: str = "skip",
    enabled: bool = True,
    current_time_ms: Optional[int] = None,
) -> dict:
    values = _normalize_schedule(
        name = name,
        task_kind = task_kind,
        payload = payload,
        cadence = cadence,
        timezone_name = timezone_name,
        misfire_policy = misfire_policy,
    )
    current = now_ms() if current_time_ms is None else int(current_time_ms)
    next_run = next_schedule_time(values["cadence"], values["timezone"], current - 1)
    if values["cadence"]["kind"] == "once" and next_run is None:
        raise AgentWorkspaceError("One-time schedules must be set in the future.")
    schedule_id = str(uuid.uuid4())
    conn = connection()
    try:
        conn.execute(
            """
            INSERT INTO agent_schedules(
                id, project_id, name, task_kind, payload_json, cadence_json,
                timezone, misfire_policy, next_run_at, enabled, revision,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (
                schedule_id,
                project_id,
                values["name"],
                values["taskKind"],
                values["payloadJson"],
                values["cadenceJson"],
                values["timezone"],
                values["misfirePolicy"],
                next_run,
                int(bool(enabled)),
                current,
                current,
            ),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM agent_schedules WHERE id = ?", (schedule_id,)).fetchone()
        return _schedule(row)
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        _translate_insert_error(exc, "Schedule")
        raise AssertionError("unreachable")
    finally:
        conn.close()


def get_schedule(schedule_id: str) -> Optional[dict]:
    conn = connection()
    try:
        row = conn.execute("SELECT * FROM agent_schedules WHERE id = ?", (schedule_id,)).fetchone()
        return _schedule(row) if row else None
    finally:
        conn.close()


def list_schedules(project_id: str, *, enabled_only: bool = False) -> list[dict]:
    conn = connection()
    try:
        clause = " AND enabled = 1" if enabled_only else ""
        rows = conn.execute(
            f"""
            SELECT * FROM agent_schedules
            WHERE project_id = ?{clause}
            ORDER BY enabled DESC, next_run_at, name COLLATE NOCASE, id
            """,
            (project_id,),
        ).fetchall()
        return [_schedule(row) for row in rows]
    finally:
        conn.close()


def update_schedule(
    schedule_id: str,
    *,
    expected_revision: int,
    name: Any = _UNSET,
    task_kind: Any = _UNSET,
    payload: Any = _UNSET,
    cadence: Any = _UNSET,
    timezone_name: Any = _UNSET,
    misfire_policy: Any = _UNSET,
    enabled: Any = _UNSET,
    current_time_ms: Optional[int] = None,
) -> Optional[dict]:
    expected = _validate_revision(expected_revision)
    current = now_ms() if current_time_ms is None else int(current_time_ms)
    conn = connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM agent_schedules WHERE id = ?", (schedule_id,)).fetchone()
        if row is None:
            conn.rollback()
            return None
        if row["revision"] != expected:
            _raise_conflict("Schedule")
        if row["lease_owner"]:
            raise AgentWorkspaceError("A leased schedule cannot be changed until its run finishes.")
        values = _normalize_schedule(
            name = row["name"] if name is _UNSET else name,
            task_kind = row["task_kind"] if task_kind is _UNSET else task_kind,
            payload = _loads(row["payload_json"], {}) if payload is _UNSET else payload,
            cadence = _loads(row["cadence_json"], {}) if cadence is _UNSET else cadence,
            timezone_name = row["timezone"] if timezone_name is _UNSET else timezone_name,
            misfire_policy = (row["misfire_policy"] if misfire_policy is _UNSET else misfire_policy),
        )
        is_enabled = bool(row["enabled"]) if enabled is _UNSET else bool(enabled)
        timing_changed = cadence is not _UNSET or timezone_name is not _UNSET
        reenabled = enabled is not _UNSET and is_enabled and not bool(row["enabled"])
        next_run = row["next_run_at"]
        if timing_changed or reenabled:
            next_run = next_schedule_time(values["cadence"], values["timezone"], current - 1)
        if is_enabled and values["cadence"]["kind"] == "once" and next_run is None:
            raise AgentWorkspaceError("One-time schedules must be set in the future.")
        try:
            cursor = conn.execute(
                """
                UPDATE agent_schedules
                SET name = ?, task_kind = ?, payload_json = ?, cadence_json = ?,
                    timezone = ?, misfire_policy = ?, next_run_at = ?, enabled = ?,
                    revision = revision + 1, updated_at = ?
                WHERE id = ? AND revision = ?
                """,
                (
                    values["name"],
                    values["taskKind"],
                    values["payloadJson"],
                    values["cadenceJson"],
                    values["timezone"],
                    values["misfirePolicy"],
                    next_run,
                    int(is_enabled),
                    current,
                    schedule_id,
                    expected,
                ),
            )
        except sqlite3.IntegrityError as exc:
            _translate_insert_error(exc, "Schedule")
            raise AssertionError("unreachable")
        if not cursor.rowcount:
            _raise_conflict("Schedule")
        conn.commit()
        updated = conn.execute(
            "SELECT * FROM agent_schedules WHERE id = ?", (schedule_id,)
        ).fetchone()
        return _schedule(updated)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def delete_schedule(schedule_id: str, *, expected_revision: int) -> bool:
    expected = _validate_revision(expected_revision)
    conn = connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT revision, lease_owner FROM agent_schedules WHERE id = ?", (schedule_id,)
        ).fetchone()
        if row is None:
            conn.rollback()
            return False
        if row["revision"] != expected:
            _raise_conflict("Schedule")
        if row["lease_owner"]:
            raise AgentWorkspaceError("A leased schedule cannot be deleted until its run finishes.")
        conn.execute("DELETE FROM agent_schedules WHERE id = ?", (schedule_id,))
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _schedule_run(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "scheduleId": row["schedule_id"],
        "projectId": row["project_id"],
        "scheduleName": row["schedule_name"],
        "taskKind": row["task_kind"],
        "payload": _loads(row["payload_json"], {}),
        "scheduledFor": row["scheduled_for"],
        "status": row["status"],
        "error": row["error"],
        "startedAt": row["started_at"],
        "completedAt": row["completed_at"],
    }


def _next_after_reconciliation(row: sqlite3.Row, current: int) -> Optional[int]:
    cadence = _loads(row["cadence_json"], {})
    if cadence.get("kind") == "once":
        return None
    return next_schedule_time(cadence, row["timezone"], current)


def _record_skipped_schedule(conn: sqlite3.Connection, row: sqlite3.Row, current: int) -> None:
    run_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO agent_schedule_runs(
            id, schedule_id, project_id, schedule_name, task_kind,
            payload_json, scheduled_for, status, started_at, completed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'skipped', ?, ?)
        """,
        (
            run_id,
            row["id"],
            row["project_id"],
            row["name"],
            row["task_kind"],
            row["payload_json"],
            row["next_run_at"],
            current,
            current,
        ),
    )
    next_run = _next_after_reconciliation(row, current)
    conn.execute(
        """
        UPDATE agent_schedules
        SET next_run_at = ?, last_run_at = ?, last_status = 'skipped',
            enabled = CASE WHEN ? IS NULL THEN 0 ELSE enabled END,
            lease_owner = NULL, lease_expires_at = NULL, lease_run_id = NULL,
            revision = revision + 1, updated_at = ?
        WHERE id = ?
        """,
        (next_run, row["next_run_at"], next_run, current, row["id"]),
    )


def reconcile_expired_schedule_leases(*, current_time_ms: Optional[int] = None) -> int:
    current = now_ms() if current_time_ms is None else int(current_time_ms)
    conn = connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            """
            SELECT * FROM agent_schedules
            WHERE lease_owner IS NOT NULL AND lease_expires_at <= ?
            """,
            (current,),
        ).fetchall()
        for row in rows:
            if row["lease_run_id"]:
                conn.execute(
                    """
                    UPDATE agent_schedule_runs
                    SET status = 'interrupted', error = ?, completed_at = ?
                    WHERE id = ? AND status = 'leased'
                    """,
                    (
                        "The schedule lease expired before reconciliation.",
                        current,
                        row["lease_run_id"],
                    ),
                )
            conn.execute(
                """
                UPDATE agent_schedules
                SET lease_owner = NULL, lease_expires_at = NULL, lease_run_id = NULL,
                    last_status = 'interrupted', revision = revision + 1, updated_at = ?
                WHERE id = ?
                """,
                (current, row["id"]),
            )
        conn.commit()
        return len(rows)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def lease_due_schedules(
    lease_owner: str,
    *,
    current_time_ms: Optional[int] = None,
    lease_ms: int = 60_000,
    misfire_grace_ms: int = 60_000,
    limit: int = 25,
) -> list[dict]:
    owner = _required_text(lease_owner, label = "Schedule lease owner", maximum = 200)
    current = now_ms() if current_time_ms is None else int(current_time_ms)
    if isinstance(lease_ms, bool) or lease_ms < 1000 or lease_ms > MAX_LEASE_MS:
        raise AgentWorkspaceError("Schedule lease duration is invalid.")
    if isinstance(misfire_grace_ms, bool) or misfire_grace_ms < 0 or misfire_grace_ms > 86_400_000:
        raise AgentWorkspaceError("Schedule misfire grace is invalid.")
    if isinstance(limit, bool) or limit < 1 or limit > MAX_LEASE_BATCH:
        raise AgentWorkspaceError("Schedule lease batch size is invalid.")
    conn = connection()
    leased: list[dict] = []
    try:
        conn.execute("BEGIN IMMEDIATE")
        expired = conn.execute(
            """
            SELECT * FROM agent_schedules
            WHERE lease_owner IS NOT NULL AND lease_expires_at <= ?
            """,
            (current,),
        ).fetchall()
        for row in expired:
            if row["lease_run_id"]:
                conn.execute(
                    """
                    UPDATE agent_schedule_runs
                    SET status = 'interrupted', error = ?, completed_at = ?
                    WHERE id = ? AND status = 'leased'
                    """,
                    (
                        "The schedule lease expired before reconciliation.",
                        current,
                        row["lease_run_id"],
                    ),
                )
            conn.execute(
                """
                UPDATE agent_schedules
                SET lease_owner = NULL, lease_expires_at = NULL, lease_run_id = NULL,
                    last_status = 'interrupted', revision = revision + 1, updated_at = ?
                WHERE id = ?
                """,
                (current, row["id"]),
            )
        due = conn.execute(
            """
            SELECT * FROM agent_schedules
            WHERE enabled = 1 AND next_run_at IS NOT NULL AND next_run_at <= ?
                AND lease_owner IS NULL
            ORDER BY next_run_at, id
            LIMIT 1000
            """,
            (current,),
        ).fetchall()
        for original in due:
            if len(leased) >= limit:
                break
            row = conn.execute(
                "SELECT * FROM agent_schedules WHERE id = ?", (original["id"],)
            ).fetchone()
            if row is None or row["lease_owner"] is not None:
                continue
            overdue_by = current - int(row["next_run_at"])
            if row["misfire_policy"] == "skip" and overdue_by > misfire_grace_ms:
                _record_skipped_schedule(conn, row, current)
                continue
            run_id = str(uuid.uuid4())
            expires = current + lease_ms
            conn.execute(
                """
                INSERT INTO agent_schedule_runs(
                    id, schedule_id, project_id, schedule_name, task_kind,
                    payload_json, scheduled_for, lease_owner, status, started_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'leased', ?)
                """,
                (
                    run_id,
                    row["id"],
                    row["project_id"],
                    row["name"],
                    row["task_kind"],
                    row["payload_json"],
                    row["next_run_at"],
                    owner,
                    current,
                ),
            )
            conn.execute(
                """
                UPDATE agent_schedules
                SET lease_owner = ?, lease_expires_at = ?, lease_run_id = ?,
                    revision = revision + 1, updated_at = ?
                WHERE id = ? AND lease_owner IS NULL
                """,
                (owner, expires, run_id, current, row["id"]),
            )
            updated = conn.execute(
                "SELECT * FROM agent_schedules WHERE id = ?", (row["id"],)
            ).fetchone()
            lease = _schedule(updated)
            lease["runId"] = run_id
            lease["scheduledFor"] = row["next_run_at"]
            leased.append(lease)
        conn.commit()
        return leased
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def reconcile_schedule_run(
    run_id: str,
    lease_owner: str,
    *,
    status: str,
    error: Optional[str] = None,
    current_time_ms: Optional[int] = None,
) -> dict:
    owner = _required_text(lease_owner, label = "Schedule lease owner", maximum = 200)
    normalized_status = str(status or "").strip().casefold()
    if normalized_status not in SCHEDULE_RUN_STATUSES:
        raise AgentWorkspaceError("Schedule run status is invalid.")
    normalized_error = _optional_text(error, label = "Schedule run error", maximum = 16 * 1024)
    current = now_ms() if current_time_ms is None else int(current_time_ms)
    conn = connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        run = conn.execute("SELECT * FROM agent_schedule_runs WHERE id = ?", (run_id,)).fetchone()
        if run is None:
            raise AgentWorkspaceError("Schedule run not found.")
        if run["status"] != "leased" or run["lease_owner"] != owner:
            raise AgentWorkspaceError("Schedule run lease is no longer owned by this worker.")
        schedule = conn.execute(
            "SELECT * FROM agent_schedules WHERE id = ?", (run["schedule_id"],)
        ).fetchone()
        if (
            schedule is None
            or schedule["lease_owner"] != owner
            or schedule["lease_run_id"] != run_id
        ):
            raise AgentWorkspaceError("Schedule run lease is no longer current.")
        next_run = _next_after_reconciliation(schedule, current)
        conn.execute(
            """
            UPDATE agent_schedule_runs
            SET status = ?, error = ?, completed_at = ?
            WHERE id = ? AND status = 'leased'
            """,
            (normalized_status, normalized_error or None, current, run_id),
        )
        conn.execute(
            """
            UPDATE agent_schedules
            SET next_run_at = ?, last_run_at = ?, last_status = ?,
                enabled = CASE WHEN ? IS NULL THEN 0 ELSE enabled END,
                lease_owner = NULL, lease_expires_at = NULL, lease_run_id = NULL,
                revision = revision + 1, updated_at = ?
            WHERE id = ?
            """,
            (
                next_run,
                run["scheduled_for"],
                normalized_status,
                next_run,
                current,
                schedule["id"],
            ),
        )
        conn.commit()
        updated_run = conn.execute(
            "SELECT * FROM agent_schedule_runs WHERE id = ?", (run_id,)
        ).fetchone()
        updated_schedule = conn.execute(
            "SELECT * FROM agent_schedules WHERE id = ?", (schedule["id"],)
        ).fetchone()
        return {"run": _schedule_run(updated_run), "schedule": _schedule(updated_schedule)}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def list_schedule_runs(project_id: str, *, limit: int = 100) -> list[dict]:
    if isinstance(limit, bool) or limit < 1 or limit > 1000:
        raise AgentWorkspaceError("Schedule run limit is invalid.")
    conn = connection()
    try:
        rows = conn.execute(
            """
            SELECT * FROM agent_schedule_runs
            WHERE project_id = ?
            ORDER BY started_at DESC, id DESC LIMIT ?
            """,
            (project_id, limit),
        ).fetchall()
        return [_schedule_run(row) for row in rows]
    finally:
        conn.close()


def _delete_revisioned(table: str, item_id: str, expected_revision: int, label: str) -> bool:
    if table not in {"agent_project_skills", "agent_lifecycle_hooks"}:
        raise AssertionError("Unsupported revisioned table.")
    expected = _validate_revision(expected_revision)
    conn = connection()
    try:
        cursor = conn.execute(
            f"DELETE FROM {table} WHERE id = ? AND revision = ?",
            (item_id, expected),
        )
        if not cursor.rowcount:
            exists = conn.execute(f"SELECT 1 FROM {table} WHERE id = ?", (item_id,)).fetchone()
            if exists:
                _raise_conflict(label)
            conn.rollback()
            return False
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
