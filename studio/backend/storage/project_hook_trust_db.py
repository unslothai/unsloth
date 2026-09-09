# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Hash-bound trust and handler preferences for project-local hooks."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from typing import Optional

from storage import studio_db


MAX_PROJECT_ID_BYTES = 512
MAX_HANDLER_ID_BYTES = 256
MAX_DISABLED_HANDLER_IDS = 256
MAX_REVISION = (1 << 63) - 1

_CONTENT_HASH = re.compile(r"[0-9a-f]{64}")
_schema_lock = threading.Lock()
_ready_databases: set[str] = set()


class ProjectHookTrustConflictError(RuntimeError):
    """A hook trust write was based on a stale revision or source hash."""


class ProjectHookTrustStateError(RuntimeError):
    """Persisted hook trust state is invalid and cannot be applied safely."""


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS project_hook_trust (
            project_id TEXT NOT NULL PRIMARY KEY
                REFERENCES chat_projects(id) ON DELETE CASCADE,
            trusted_content_hash TEXT,
            workspace_device_id INTEGER,
            workspace_file_id INTEGER,
            workspace_revision INTEGER,
            disabled_handler_ids_json TEXT NOT NULL DEFAULT '[]',
            revision INTEGER NOT NULL DEFAULT 0,
            updated_at INTEGER NOT NULL,
            CHECK (
                trusted_content_hash IS NULL
                OR (
                    length(trusted_content_hash) = 64
                    AND trusted_content_hash NOT GLOB '*[^0-9a-f]*'
                )
            ),
            CHECK (
                (
                    trusted_content_hash IS NULL
                    AND workspace_device_id IS NULL
                    AND workspace_file_id IS NULL
                    AND workspace_revision IS NULL
                )
                OR (
                    trusted_content_hash IS NOT NULL
                    AND workspace_device_id IS NOT NULL
                    AND workspace_file_id IS NOT NULL
                    AND workspace_revision IS NOT NULL
                    AND workspace_device_id >= 0
                    AND workspace_file_id >= 0
                    AND workspace_revision >= 0
                )
            ),
            CHECK (revision >= 0)
        ) WITHOUT ROWID
        """
    )
    columns = {
        str(row[1]) for row in conn.execute("PRAGMA table_info(project_hook_trust)").fetchall()
    }
    for name in ("workspace_device_id", "workspace_file_id", "workspace_revision"):
        if name not in columns:
            conn.execute(f"ALTER TABLE project_hook_trust ADD COLUMN {name} INTEGER")


def _database_key(conn: sqlite3.Connection) -> str:
    row = conn.execute("PRAGMA database_list").fetchone()
    return str(row[2])


def _connection(busy_timeout_seconds: float = 30.0) -> sqlite3.Connection:
    conn = studio_db.get_connection(busy_timeout_seconds)
    key = _database_key(conn)
    if key not in _ready_databases:
        with _schema_lock:
            if key not in _ready_databases:
                try:
                    _ensure_schema(conn)
                    conn.commit()
                    _ready_databases.add(key)
                except Exception:
                    conn.close()
                    raise
    return conn


def _validate_project_id(project_id: str) -> str:
    if not isinstance(project_id, str):
        raise ValueError("Project id must be a string.")
    try:
        encoded = project_id.encode("utf-8", errors = "strict")
    except UnicodeEncodeError as exc:
        raise ValueError("Project id must be valid UTF-8.") from exc
    if (
        not encoded
        or len(encoded) > MAX_PROJECT_ID_BYTES
        or "\x00" in project_id
        or any(ord(character) < 32 or ord(character) == 127 for character in project_id)
    ):
        raise ValueError("Project id is invalid or exceeds its size limit.")
    return project_id


def _validate_content_hash(content_hash: str) -> str:
    if not isinstance(content_hash, str) or _CONTENT_HASH.fullmatch(content_hash) is None:
        raise ValueError("Hook content hash must be a lowercase SHA-256 digest.")
    return content_hash


def _validate_current_hash(content_hash: Optional[str]) -> Optional[str]:
    return None if content_hash is None else _validate_content_hash(content_hash)


def _validate_workspace_identity(identity: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(identity, tuple)
        or len(identity) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in identity)
    ):
        raise ValueError("Hook workspace identity is invalid.")
    normalized = identity
    if min(normalized) < 0 or max(normalized) > MAX_REVISION:
        raise ValueError("Hook workspace identity is invalid.")
    return normalized


def _validate_revision(revision: int) -> int:
    if isinstance(revision, bool) or not isinstance(revision, int):
        raise ValueError("Hook trust revision must be an integer.")
    if revision < 0 or revision > MAX_REVISION:
        raise ValueError("Hook trust revision is outside the supported range.")
    return revision


def _validate_handler_id(handler_id: str) -> str:
    if not isinstance(handler_id, str):
        raise ValueError("Hook handler id must be a string.")
    try:
        encoded = handler_id.encode("utf-8", errors = "strict")
    except UnicodeEncodeError as exc:
        raise ValueError("Hook handler id must be valid UTF-8.") from exc
    if (
        not encoded
        or len(encoded) > MAX_HANDLER_ID_BYTES
        or "\x00" in handler_id
        or any(ord(character) < 32 or ord(character) == 127 for character in handler_id)
    ):
        raise ValueError("Hook handler id is invalid or exceeds its size limit.")
    return handler_id


def _decode_disabled_handler_ids(raw: str) -> list[str]:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ProjectHookTrustStateError("Persisted hook handler preferences are invalid.") from exc
    if (
        not isinstance(value, list)
        or len(value) > MAX_DISABLED_HANDLER_IDS
        or any(not isinstance(item, str) for item in value)
    ):
        raise ProjectHookTrustStateError("Persisted hook handler preferences are invalid.")
    try:
        normalized = [_validate_handler_id(item) for item in value]
    except ValueError as exc:
        raise ProjectHookTrustStateError("Persisted hook handler preferences are invalid.") from exc
    if len(set(normalized)) != len(normalized):
        raise ProjectHookTrustStateError("Persisted hook handler preferences are invalid.")
    return sorted(normalized)


def _encode_disabled_handler_ids(handler_ids: set[str]) -> str:
    if len(handler_ids) > MAX_DISABLED_HANDLER_IDS:
        raise ValueError("Disabled hook handlers exceed the supported limit.")
    return json.dumps(sorted(handler_ids), ensure_ascii = False, separators = (",", ":"))


def _read_row(conn: sqlite3.Connection, project_id: str) -> Optional[sqlite3.Row]:
    return conn.execute(
        """
        SELECT trusted_content_hash, workspace_device_id, workspace_file_id,
               workspace_revision, disabled_handler_ids_json, revision
        FROM project_hook_trust
        WHERE project_id = ?
        """,
        (project_id,),
    ).fetchone()


def _row_values(
    row: Optional[sqlite3.Row],
) -> tuple[Optional[str], Optional[tuple[int, int]], Optional[int], list[str], int]:
    if row is None:
        return None, None, None, [], 0
    content_hash = row["trusted_content_hash"]
    if content_hash is not None:
        try:
            content_hash = _validate_content_hash(content_hash)
        except ValueError as exc:
            raise ProjectHookTrustStateError("Persisted hook trust hash is invalid.") from exc
    try:
        revision = _validate_revision(row["revision"])
    except ValueError as exc:
        raise ProjectHookTrustStateError("Persisted hook trust revision is invalid.") from exc
    disabled = _decode_disabled_handler_ids(row["disabled_handler_ids_json"])
    identity = None
    workspace_revision = None
    if content_hash is not None:
        try:
            identity = _validate_workspace_identity(
                (row["workspace_device_id"], row["workspace_file_id"])
            )
            workspace_revision = _validate_revision(row["workspace_revision"])
        except (TypeError, ValueError):
            # Earlier development builds did not bind trust to a workspace.
            # Fail those rows closed while keeping their CAS revision usable.
            content_hash = None
            disabled = []
    if content_hash is None and disabled:
        raise ProjectHookTrustStateError(
            "Revoked hook trust cannot retain disabled handler preferences."
        )
    return content_hash, identity, workspace_revision, disabled, revision


def _public_state(
    project_id: str,
    current_content_hash: Optional[str],
    current_identity: Optional[tuple[int, int]],
    current_workspace_revision: Optional[int],
    stored_content_hash: Optional[str],
    stored_identity: Optional[tuple[int, int]],
    stored_workspace_revision: Optional[int],
    disabled_handler_ids: list[str],
    revision: int,
) -> dict:
    trusted = (
        current_content_hash is not None
        and current_content_hash == stored_content_hash
        and current_identity is not None
        and current_identity == stored_identity
        and current_workspace_revision is not None
        and current_workspace_revision == stored_workspace_revision
    )
    return {
        "projectId": project_id,
        "contentHash": current_content_hash,
        "trusted": trusted,
        "disabledHandlerIds": disabled_handler_ids if trusted else [],
        "revision": revision,
    }


def get_project_hook_trust_state(
    project_id: str,
    current_content_hash: Optional[str],
    *,
    workspace_identity: Optional[tuple[int, int]],
    workspace_revision: Optional[int],
) -> dict:
    """Read trust for the rediscovered source hash without applying stale preferences."""
    project_id = _validate_project_id(project_id)
    current_content_hash = _validate_current_hash(current_content_hash)
    current_identity = (
        _validate_workspace_identity(workspace_identity) if workspace_identity is not None else None
    )
    current_workspace_revision = (
        _validate_revision(workspace_revision) if workspace_revision is not None else None
    )
    conn = _connection()
    try:
        stored_hash, stored_identity, stored_workspace_revision, disabled, revision = _row_values(
            _read_row(conn, project_id)
        )
        return _public_state(
            project_id,
            current_content_hash,
            current_identity,
            current_workspace_revision,
            stored_hash,
            stored_identity,
            stored_workspace_revision,
            disabled,
            revision,
        )
    finally:
        conn.close()


def get_project_hook_trust_record(project_id: str) -> dict:
    """Return persisted authority state without consulting an unavailable workspace."""
    project_id = _validate_project_id(project_id)
    conn = _connection()
    try:
        row = _read_row(conn, project_id)
        if row is None:
            stored_hash = None
            revision = 0
        else:
            try:
                revision = _validate_revision(row["revision"])
            except ValueError as exc:
                raise ProjectHookTrustStateError(
                    "Persisted hook trust revision is invalid."
                ) from exc
            stored_hash = row["trusted_content_hash"]
            if stored_hash is not None:
                try:
                    stored_hash = _validate_content_hash(stored_hash)
                    _validate_workspace_identity(
                        (row["workspace_device_id"], row["workspace_file_id"])
                    )
                    _validate_revision(row["workspace_revision"])
                except (TypeError, ValueError):
                    # Unbound development rows must never be presented as authority.
                    stored_hash = None
        return {
            "projectId": project_id,
            "storedContentHash": stored_hash,
            "hasStoredTrust": stored_hash is not None,
            "revision": revision,
        }
    finally:
        conn.close()


def trust_project_hooks(
    project_id: str,
    rediscovered_content_hash: str,
    *,
    workspace_identity: tuple[int, int],
    workspace_revision: int,
    expected_revision: int,
) -> dict:
    """Trust exactly the source hash rediscovered by the server."""
    project_id = _validate_project_id(project_id)
    content_hash = _validate_content_hash(rediscovered_content_hash)
    identity = _validate_workspace_identity(workspace_identity)
    workspace_revision = _validate_revision(workspace_revision)
    expected_revision = _validate_revision(expected_revision)
    conn = _connection()
    try:
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        stored_hash, stored_identity, stored_workspace_revision, disabled, revision = _row_values(
            _read_row(conn, project_id)
        )
        if revision != expected_revision:
            raise ProjectHookTrustConflictError(
                f"Hook trust revision is {revision}, not {expected_revision}."
            )
        if (
            stored_hash == content_hash
            and stored_identity == identity
            and stored_workspace_revision == workspace_revision
        ):
            conn.commit()
            return _public_state(
                project_id,
                content_hash,
                identity,
                workspace_revision,
                stored_hash,
                stored_identity,
                stored_workspace_revision,
                disabled,
                revision,
            )
        next_revision = _next_revision(revision)
        encoded_disabled = _encode_disabled_handler_ids(set())
        if revision == 0 and stored_hash is None and _read_row(conn, project_id) is None:
            conn.execute(
                """
                INSERT INTO project_hook_trust (
                    project_id, trusted_content_hash, workspace_device_id,
                    workspace_file_id, workspace_revision,
                    disabled_handler_ids_json, revision, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    content_hash,
                    identity[0],
                    identity[1],
                    workspace_revision,
                    encoded_disabled,
                    next_revision,
                    _now_ms(),
                ),
            )
        else:
            cursor = conn.execute(
                """
                UPDATE project_hook_trust
                SET trusted_content_hash = ?, workspace_device_id = ?,
                    workspace_file_id = ?, workspace_revision = ?,
                    disabled_handler_ids_json = ?,
                    revision = ?, updated_at = ?
                WHERE project_id = ? AND revision = ?
                """,
                (
                    content_hash,
                    identity[0],
                    identity[1],
                    workspace_revision,
                    encoded_disabled,
                    next_revision,
                    _now_ms(),
                    project_id,
                    revision,
                ),
            )
            if cursor.rowcount != 1:
                raise ProjectHookTrustConflictError("Hook trust changed concurrently.")
        conn.commit()
        return _public_state(
            project_id,
            content_hash,
            identity,
            workspace_revision,
            content_hash,
            identity,
            workspace_revision,
            [],
            next_revision,
        )
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        raise ValueError("Project does not exist or hook trust state is invalid.") from exc
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def revoke_project_hook_trust(project_id: str, *, expected_revision: int) -> dict:
    """Revoke even malformed trust state while retaining a monotonic CAS revision."""
    project_id = _validate_project_id(project_id)
    expected_revision = _validate_revision(expected_revision)
    conn = _connection()
    try:
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        row = _read_row(conn, project_id)
        if row is None:
            revision = 0
            has_persisted_state = False
        else:
            try:
                revision = _validate_revision(row["revision"])
            except ValueError as exc:
                raise ProjectHookTrustStateError(
                    "Persisted hook trust revision is invalid."
                ) from exc
            has_persisted_state = (
                row["trusted_content_hash"] is not None
                or row["workspace_device_id"] is not None
                or row["workspace_file_id"] is not None
                or row["workspace_revision"] is not None
                or row["disabled_handler_ids_json"] != "[]"
            )
        if revision != expected_revision:
            raise ProjectHookTrustConflictError(
                f"Hook trust revision is {revision}, not {expected_revision}."
            )
        if not has_persisted_state:
            conn.commit()
            return _public_state(project_id, None, None, None, None, None, None, [], revision)
        next_revision = _next_revision(revision)
        cursor = conn.execute(
            """
            UPDATE project_hook_trust
            SET trusted_content_hash = NULL, workspace_device_id = NULL,
                workspace_file_id = NULL, workspace_revision = NULL,
                disabled_handler_ids_json = '[]',
                revision = ?, updated_at = ?
            WHERE project_id = ? AND revision = ?
            """,
            (next_revision, _now_ms(), project_id, revision),
        )
        if cursor.rowcount != 1:
            raise ProjectHookTrustConflictError("Hook trust changed concurrently.")
        conn.commit()
        return _public_state(project_id, None, None, None, None, None, None, [], next_revision)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def set_project_hook_handler_enabled(
    project_id: str,
    rediscovered_content_hash: str,
    handler_id: str,
    *,
    workspace_identity: tuple[int, int],
    workspace_revision: int,
    enabled: bool,
    expected_revision: int,
) -> dict:
    """Set one handler preference against the exact currently trusted source."""
    project_id = _validate_project_id(project_id)
    content_hash = _validate_content_hash(rediscovered_content_hash)
    handler_id = _validate_handler_id(handler_id)
    identity = _validate_workspace_identity(workspace_identity)
    workspace_revision = _validate_revision(workspace_revision)
    if not isinstance(enabled, bool):
        raise ValueError("Hook handler enabled state must be a boolean.")
    expected_revision = _validate_revision(expected_revision)
    conn = _connection()
    try:
        conn.commit()
        conn.execute("BEGIN IMMEDIATE")
        stored_hash, stored_identity, stored_workspace_revision, disabled, revision = _row_values(
            _read_row(conn, project_id)
        )
        if revision != expected_revision:
            raise ProjectHookTrustConflictError(
                f"Hook trust revision is {revision}, not {expected_revision}."
            )
        if (
            stored_hash != content_hash
            or stored_identity != identity
            or stored_workspace_revision != workspace_revision
        ):
            raise ProjectHookTrustConflictError(
                "Hook source or workspace changed, or has not been trusted."
            )
        disabled_set = set(disabled)
        changed = False
        if enabled and handler_id in disabled_set:
            disabled_set.remove(handler_id)
            changed = True
        elif not enabled and handler_id not in disabled_set:
            if len(disabled_set) >= MAX_DISABLED_HANDLER_IDS:
                raise ValueError("Disabled hook handlers exceed the supported limit.")
            disabled_set.add(handler_id)
            changed = True
        if not changed:
            conn.commit()
            return _public_state(
                project_id,
                content_hash,
                identity,
                workspace_revision,
                stored_hash,
                stored_identity,
                stored_workspace_revision,
                disabled,
                revision,
            )
        next_revision = _next_revision(revision)
        cursor = conn.execute(
            """
            UPDATE project_hook_trust
            SET disabled_handler_ids_json = ?, revision = ?, updated_at = ?
            WHERE project_id = ? AND trusted_content_hash = ?
              AND workspace_device_id = ? AND workspace_file_id = ?
              AND workspace_revision = ? AND revision = ?
            """,
            (
                _encode_disabled_handler_ids(disabled_set),
                next_revision,
                _now_ms(),
                project_id,
                content_hash,
                identity[0],
                identity[1],
                workspace_revision,
                revision,
            ),
        )
        if cursor.rowcount != 1:
            raise ProjectHookTrustConflictError("Hook trust changed concurrently.")
        conn.commit()
        return _public_state(
            project_id,
            content_hash,
            identity,
            workspace_revision,
            content_hash,
            identity,
            workspace_revision,
            sorted(disabled_set),
            next_revision,
        )
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _next_revision(revision: int) -> int:
    if revision >= MAX_REVISION:
        raise ProjectHookTrustStateError("Hook trust revision is exhausted.")
    return revision + 1


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


__all__ = [
    "MAX_DISABLED_HANDLER_IDS",
    "MAX_HANDLER_ID_BYTES",
    "ProjectHookTrustConflictError",
    "ProjectHookTrustStateError",
    "get_project_hook_trust_record",
    "get_project_hook_trust_state",
    "revoke_project_hook_trust",
    "set_project_hook_handler_enabled",
    "trust_project_hooks",
]
