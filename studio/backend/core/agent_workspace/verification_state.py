# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Strict durable state for workspace-bound project verification.

The authority boundary covers ordinary writes to the tables owned by this
module. SQLite's internal metadata and deletion or recreation of the parent
``chat_projects`` row remain storage-owner operations outside that boundary.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import unicodedata
import uuid
from bisect import bisect_right
from typing import Any, Optional

from storage import studio_db

from .verification_context import AgentWorkspaceError


MAX_PROJECT_ID_BYTES = 512
MAX_OWNER_ID_BYTES = 256
MAX_FENCE_ID_BYTES = 256
MAX_CHECKS = 32
MAX_CHECK_NAME_CHARACTERS = 120
MAX_CHECK_KIND_BYTES = 64
MAX_CHECK_COMMAND_BYTES = 16 * 1024
MAX_CONFIG_BYTES = 128 * 1024
MAX_LOG_LIMIT_BYTES = 2 * 1024 * 1024
MAX_RUN_RESULT_BYTES = 4 * 1024 * 1024
MAX_RUN_RESULTS_JSON_BYTES = 6 * MAX_RUN_RESULT_BYTES + MAX_CONFIG_BYTES + 256 * 1024
MAX_RUN_HISTORY = 100
MAX_PROJECT_RUN_HISTORY_BYTES = 64 * 1024 * 1024
MAX_GLOBAL_RUN_HISTORY_BYTES = 512 * 1024 * 1024
MAX_TIMEOUT_SECONDS = 3600
MAX_ERROR_BYTES = 4 * 1024
MAX_TRUNCATION_DECORATION_BYTES = 512
MAX_REVISION = (1 << 63) - 1
MAX_HISTORY_SEQUENCE = (1 << 53) - 1
LEASE_DURATION_MS = 30_000
HEARTBEAT_INTERVAL_SECONDS = 5.0
DELETION_FENCE_LEASE_DURATION_MS = 30_000
DELETION_FENCE_HEARTBEAT_INTERVAL_SECONDS = 5.0

SOURCE_FRESHNESS = "unverified"

_CHECK_FIELDS = frozenset(
    {
        "name",
        "kind",
        "command",
        "required",
        "timeoutSeconds",
        "logLimitBytes",
    }
)
_RESULT_REQUIRED_FIELDS = frozenset(
    {
        "name",
        "kind",
        "command",
        "required",
        "status",
        "exitCode",
        "output",
        "outputBytes",
        "outputTruncated",
        "timeoutSeconds",
        "startedAt",
        "completedAt",
        "durationMs",
    }
)
_RESULT_FIELDS = _RESULT_REQUIRED_FIELDS | {"error"}
_RESULT_STATUSES = frozenset({"running", "passed", "failed", "cancelled", "timed_out", "blocked"})
_RUN_STATUSES = frozenset(
    {"running", "passed", "failed", "cancelled", "timed_out", "blocked", "interrupted"}
)
_TERMINAL_HINTS = frozenset({"failed", "cancelled", "timed_out", "blocked", "interrupted"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_TRUNCATION_NOTICE = re.compile(
    r"\n\[Process output was truncated\. The capture limit was ([1-9][0-9]*) bytes\.\]\n\Z"
)
_DEFAULT_IGNORABLE_CODE_POINT_RANGES = (
    (0x00AD, 0x00AD),
    (0x034F, 0x034F),
    (0x061C, 0x061C),
    (0x115F, 0x1160),
    (0x17B4, 0x17B5),
    (0x180B, 0x180F),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x206F),
    (0x3164, 0x3164),
    (0xFE00, 0xFE0F),
    (0xFEFF, 0xFEFF),
    (0xFFA0, 0xFFA0),
    (0xFFF0, 0xFFF8),
    (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A),
    (0xE0000, 0xE0FFF),
)
_DEFAULT_IGNORABLE_CODE_POINT_STARTS = tuple(
    start for start, _end in _DEFAULT_IGNORABLE_CODE_POINT_RANGES
)
_FORMAT_CONTROL_CODE_POINT_RANGES = (
    (0x00AD, 0x00AD),
    (0x0600, 0x0605),
    (0x061C, 0x061C),
    (0x06DD, 0x06DD),
    (0x070F, 0x070F),
    (0x0890, 0x0891),
    (0x08E2, 0x08E2),
    (0x180E, 0x180E),
    (0x200B, 0x200F),
    (0x202A, 0x202E),
    (0x2060, 0x2064),
    (0x2066, 0x206F),
    (0xFEFF, 0xFEFF),
    (0xFFF9, 0xFFFB),
    (0x110BD, 0x110BD),
    (0x110CD, 0x110CD),
    (0x13430, 0x1343F),
    (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A),
    (0xE0001, 0xE0001),
    (0xE0020, 0xE007F),
)
_FORMAT_CONTROL_CODE_POINT_STARTS = tuple(
    start for start, _end in _FORMAT_CONTROL_CODE_POINT_RANGES
)
_RUN_EVIDENCE_BYTES_SQL = """
    length(CAST(config_checks_json AS BLOB))
    + length(CAST(checks_json AS BLOB))
    + length(CAST(results_json AS BLOB))
    + COALESCE(length(CAST(error AS BLOB)), 0)
"""

_schema_lock = threading.RLock()
_DatabaseIdentity = tuple[int, int]
_DatabaseKey = tuple[str, int, int, int]
_ready_databases: set[_DatabaseKey] = set()


class _VerificationConnection:
    """Carry the attested file identity across the read-to-write transition."""

    def __init__(self, connection: sqlite3.Connection, authority_key: _DatabaseKey) -> None:
        self._connection = connection
        self.authority_key = authority_key

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)


class VerificationConflictError(AgentWorkspaceError):
    """A verification write lost its revision, workspace, lease, or progress authority."""


class VerificationStateError(AgentWorkspaceError):
    """Persisted verification state is invalid and cannot be trusted."""


def _is_code_point_in_ranges(
    character: str, ranges: tuple[tuple[int, int], ...], starts: tuple[int, ...]
) -> bool:
    code_point = ord(character)
    index = bisect_right(starts, code_point) - 1
    return index >= 0 and code_point <= ranges[index][1]


def _is_default_ignorable_code_point(character: str) -> bool:
    return _is_code_point_in_ranges(
        character,
        _DEFAULT_IGNORABLE_CODE_POINT_RANGES,
        _DEFAULT_IGNORABLE_CODE_POINT_STARTS,
    )


def _is_format_control_code_point(character: str) -> bool:
    return (
        _is_code_point_in_ranges(
            character,
            _FORMAT_CONTROL_CODE_POINT_RANGES,
            _FORMAT_CONTROL_CODE_POINT_STARTS,
        )
        or unicodedata.category(character) == "Cf"
    )


def _now_ms() -> int:
    return int(time.time() * 1000)


def _next_updated_at(previous: int, now: int) -> int:
    if previous >= MAX_REVISION:
        raise VerificationConflictError("Verification update revision is exhausted.")
    return max(now, previous + 1)


def _next_evidence_revision(previous: int) -> int:
    if previous >= MAX_REVISION:
        raise VerificationConflictError("Verification evidence revision is exhausted.")
    return previous + 1


def _next_deletion_fence_revision(previous: int) -> int:
    if previous >= MAX_REVISION:
        raise VerificationConflictError("Verification deletion fence revision is exhausted.")
    return previous + 1


def _next_lease_times(
    now: int, *, previous_heartbeat: int, previous_expiry: int, duration_ms: int
) -> tuple[int, int]:
    heartbeat_at = max(now, previous_heartbeat)
    extended_expiry = min(MAX_REVISION, heartbeat_at + duration_ms)
    return heartbeat_at, max(previous_expiry, extended_expiry)


def _database_file_identity(path: str) -> _DatabaseIdentity:
    try:
        metadata = os.stat(path)
    except OSError as exc:
        raise VerificationStateError(
            "Persisted verification database identity is unavailable."
        ) from exc
    return int(metadata.st_dev), int(metadata.st_ino)


def _prepare_database_identity(path: str) -> _DatabaseIdentity:
    """Bind an absent database name to an inode before SQLite opens it."""
    try:
        return _database_file_identity(path)
    except VerificationStateError as exc:
        if not isinstance(exc.__cause__, FileNotFoundError):
            raise
    try:
        os.makedirs(os.path.dirname(path), exist_ok = True)
        flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        return _database_file_identity(path)
    except OSError as exc:
        raise VerificationStateError(
            "Persisted verification database identity could not be established."
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        return int(metadata.st_dev), int(metadata.st_ino)
    finally:
        os.close(descriptor)


def _database_key(
    connection: sqlite3.Connection, *, expected_identity: Optional[_DatabaseIdentity] = None
) -> _DatabaseKey:
    row = connection.execute("SELECT file FROM pragma_database_list WHERE name = 'main'").fetchone()
    if row is None or not isinstance(row[0], str) or not row[0]:
        raise VerificationStateError("Persisted verification database path is invalid.")
    path = os.path.realpath(row[0])
    identity_before = _database_file_identity(path)
    if expected_identity is not None and identity_before != expected_identity:
        raise VerificationStateError(
            "Persisted verification database changed while it was being opened."
        )
    schema_row = connection.execute("PRAGMA schema_version").fetchone()
    schema_version = schema_row[0] if schema_row is not None else None
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise VerificationStateError("Persisted verification database schema version is invalid.")
    identity_after = _database_file_identity(path)
    if identity_after != identity_before:
        raise VerificationStateError(
            "Persisted verification database changed during schema inspection."
        )
    return path, identity_after[0], identity_after[1], schema_version


def _reject_cached_identity_change(key: _DatabaseKey) -> None:
    with _schema_lock:
        if any(ready[0] == key[0] and ready[1:3] != key[1:3] for ready in _ready_databases):
            raise VerificationStateError(
                "Persisted verification database identity changed after initialization."
            )


_CONFIG_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS agent_verification_configs (
            project_id TEXT NOT NULL PRIMARY KEY
                REFERENCES chat_projects(id) ON DELETE CASCADE,
            workspace_device_id INTEGER NOT NULL CHECK(workspace_device_id >= 0),
            workspace_file_id INTEGER NOT NULL CHECK(workspace_file_id >= 0),
            workspace_revision INTEGER NOT NULL CHECK(workspace_revision >= 0),
            checks_json TEXT NOT NULL,
            config_hash TEXT NOT NULL CHECK(
                length(config_hash) = 64
                AND config_hash NOT GLOB '*[^0-9a-f]*'
            ),
            revision INTEGER NOT NULL CHECK(revision > 0),
            updated_at INTEGER NOT NULL CHECK(updated_at >= 0)
        ) WITHOUT ROWID
"""
_RUN_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS agent_verification_runs (
            id TEXT NOT NULL PRIMARY KEY,
            project_id TEXT NOT NULL
                REFERENCES chat_projects(id) ON DELETE CASCADE,
            owner_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK(
                status IN (
                    'running', 'passed', 'failed', 'cancelled',
                    'timed_out', 'blocked', 'interrupted'
                )
            ),
            terminal_hint TEXT CHECK(
                terminal_hint IS NULL
                OR terminal_hint IN (
                    'failed', 'cancelled', 'timed_out', 'blocked', 'interrupted'
                )
            ),
            config_revision INTEGER NOT NULL CHECK(config_revision > 0),
            config_hash TEXT NOT NULL CHECK(
                length(config_hash) = 64
                AND config_hash NOT GLOB '*[^0-9a-f]*'
            ),
            workspace_device_id INTEGER NOT NULL CHECK(workspace_device_id >= 0),
            workspace_file_id INTEGER NOT NULL CHECK(workspace_file_id >= 0),
            workspace_revision INTEGER NOT NULL CHECK(workspace_revision >= 0),
            config_checks_json TEXT NOT NULL,
            checks_json TEXT NOT NULL,
            checks_hash TEXT NOT NULL CHECK(
                length(checks_hash) = 64
                AND checks_hash NOT GLOB '*[^0-9a-f]*'
            ),
            results_json TEXT NOT NULL,
            evidence_revision INTEGER NOT NULL CHECK(evidence_revision > 0),
            cancel_requested INTEGER NOT NULL DEFAULT 0
                CHECK(cancel_requested IN (0, 1)),
            recovered_after_restart INTEGER NOT NULL DEFAULT 0
                CHECK(recovered_after_restart IN (0, 1)),
            error TEXT,
            started_at INTEGER NOT NULL CHECK(started_at >= 0),
            heartbeat_at INTEGER NOT NULL CHECK(heartbeat_at >= 0),
            updated_at INTEGER NOT NULL CHECK(updated_at >= 0),
            lease_expires_at INTEGER NOT NULL CHECK(lease_expires_at >= 0),
            completed_at INTEGER,
            history_sequence INTEGER UNIQUE CHECK(
                history_sequence IS NULL
                OR (history_sequence > 0 AND history_sequence <= 9007199254740991)
            ),
            CHECK(
                (
                    status = 'running'
                    AND completed_at IS NULL
                    AND history_sequence IS NULL
                )
                OR (
                    status != 'running'
                    AND completed_at IS NOT NULL
                    AND history_sequence IS NOT NULL
                )
            )
        )
"""
_HISTORY_SEQUENCE_TABLE_SQL = f"""
        CREATE TABLE IF NOT EXISTS agent_verification_history_sequences (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT
                CHECK(sequence <= {MAX_HISTORY_SEQUENCE})
        )
"""
_PROJECT_DELETION_TABLE_SQL = """
        CREATE TABLE IF NOT EXISTS agent_verification_project_deletions (
            project_id TEXT NOT NULL PRIMARY KEY
                REFERENCES chat_projects(id) ON DELETE CASCADE,
            fence_id TEXT NOT NULL,
            active INTEGER NOT NULL CHECK(active IN (0, 1)),
            revision INTEGER NOT NULL CHECK(revision > 0),
            created_at INTEGER NOT NULL CHECK(created_at >= 0),
            heartbeat_at INTEGER NOT NULL CHECK(heartbeat_at >= 0),
            lease_expires_at INTEGER NOT NULL CHECK(lease_expires_at >= 0),
            CHECK(heartbeat_at >= created_at),
            CHECK(lease_expires_at >= heartbeat_at)
        ) WITHOUT ROWID
"""
_TABLE_DEFINITIONS = (
    ("agent_verification_configs", _CONFIG_TABLE_SQL),
    ("agent_verification_runs", _RUN_TABLE_SQL),
    ("agent_verification_history_sequences", _HISTORY_SEQUENCE_TABLE_SQL),
    ("agent_verification_project_deletions", _PROJECT_DELETION_TABLE_SQL),
)
_TRIGGER_NAMES = (
    "trg_agent_verification_configs_no_replace",
    "trg_agent_verification_configs_no_delete",
    "trg_agent_verification_configs_revision_monotonic",
    "trg_agent_verification_runs_evidence_revision",
    "trg_agent_verification_runs_terminal_lease_evidence",
    "trg_agent_verification_runs_revision_monotonic",
    "trg_agent_verification_runs_no_replace",
    "trg_agent_verification_runs_no_delete",
    "trg_agent_verification_runs_identity_immutable",
    "trg_agent_verification_runs_lifecycle_immutable",
    "trg_agent_verification_project_deletions_no_replace",
    "trg_agent_verification_project_deletions_no_delete",
    "trg_agent_verification_project_deletions_lifecycle",
)
_INDEX_NAMES = (
    "idx_agent_verification_runs_one_active_project",
    "idx_agent_verification_runs_project_history",
    "idx_agent_verification_runs_expired_lease",
)
_owned_schema_definitions: dict[tuple[str, str], tuple[str, str]] = {}
_owned_autoindexes: frozenset[tuple[str, str]] = frozenset()

_RUN_NO_DELETE_TRIGGER_SQL = """
        CREATE TRIGGER trg_agent_verification_runs_no_delete
        BEFORE DELETE ON agent_verification_runs
        WHEN EXISTS (
            SELECT 1 FROM chat_projects WHERE id = OLD.project_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'verification runs cannot be deleted');
        END
"""


def _normalize_schema_sql(value: object) -> str:
    """Collapse SQL whitespace without changing quoted literal or identifier text."""
    if not isinstance(value, str):
        return ""
    normalized: list[str] = []
    quote: Optional[str] = None
    pending_space = False
    index = 0
    while index < len(value):
        character = value[index]
        if quote is not None:
            normalized.append(character)
            if character == quote:
                if quote != "]" and index + 1 < len(value) and value[index + 1] == quote:
                    normalized.append(value[index + 1])
                    index += 2
                    continue
                quote = None
            index += 1
            continue
        # Match SQLite's tokenizer whitespace exactly. Python's isspace() also
        # accepts non-ASCII characters that SQLite treats as identifier bytes;
        # collapsing those could make a weaker declaration compare equal.
        if character in {"\t", "\n", "\f", "\r", " "}:
            pending_space = bool(normalized)
            index += 1
            continue
        if pending_space:
            normalized.append(" ")
            pending_space = False
        normalized.append(character)
        if character in {"'", '"', "`"}:
            quote = character
        elif character == "[":
            quote = "]"
        index += 1
    return "".join(normalized)


def _stored_table_sql(definition: str) -> str:
    normalized = _normalize_schema_sql(definition)
    prefix = "CREATE TABLE IF NOT EXISTS "
    if not normalized.startswith(prefix):
        raise RuntimeError("Verification table definition is not create-only.")
    return f"CREATE TABLE {normalized[len(prefix):]}"


def _verify_table_definitions(connection: sqlite3.Connection) -> None:
    for table_name, definition in _TABLE_DEFINITIONS:
        row = connection.execute(
            "SELECT type, sql FROM sqlite_master WHERE name = ?",
            (table_name,),
        ).fetchone()
        if (
            row is None
            or row[0] != "table"
            or _normalize_schema_sql(row[1]) != _stored_table_sql(definition)
        ):
            raise VerificationStateError(
                "Persisted verification schema is incompatible with this Studio version."
            )


def _owned_schema_rows(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    table_names = tuple(table_name for table_name, _definition in _TABLE_DEFINITIONS)
    return connection.execute(
        f"""
        SELECT type, name, tbl_name, sql
        FROM sqlite_master
        WHERE type IN ('trigger', 'index')
            AND tbl_name IN ({", ".join("?" for _name in table_names)})
        """,
        table_names,
    ).fetchall()


def _capture_owned_schema_definitions(connection: sqlite3.Connection) -> None:
    global _owned_autoindexes
    expected_keys = {
        *(("trigger", name) for name in _TRIGGER_NAMES),
        *(("index", name) for name in _INDEX_NAMES),
    }
    definitions: dict[tuple[str, str], tuple[str, str]] = {}
    autoindexes: set[tuple[str, str]] = set()
    for row in _owned_schema_rows(connection):
        object_type = str(row["type"])
        name = str(row["name"])
        table_name = str(row["tbl_name"])
        sql = row["sql"]
        if object_type == "index" and sql is None:
            if not name.startswith("sqlite_autoindex_"):
                raise VerificationStateError(
                    "Persisted verification schema contains an invalid automatic index."
                )
            autoindexes.add((name, table_name))
            continue
        definitions[(object_type, name)] = (table_name, _normalize_schema_sql(sql))
    if set(definitions) != expected_keys or any(not value[1] for value in definitions.values()):
        raise VerificationStateError(
            "Persisted verification authority objects are incompatible with this Studio version."
        )
    if _owned_schema_definitions and _owned_schema_definitions != definitions:
        raise RuntimeError("Verification authority definitions changed within this process.")
    if _owned_autoindexes and _owned_autoindexes != frozenset(autoindexes):
        raise RuntimeError("Verification automatic index definitions changed within this process.")
    _owned_schema_definitions.clear()
    _owned_schema_definitions.update(definitions)
    _owned_autoindexes = frozenset(autoindexes)


def _verify_schema_objects(connection: sqlite3.Connection) -> None:
    _verify_table_definitions(connection)
    if not _owned_schema_definitions:
        raise VerificationStateError("Verification authority definitions are unavailable.")
    definitions: dict[tuple[str, str], tuple[str, str]] = {}
    autoindexes: set[tuple[str, str]] = set()
    for row in _owned_schema_rows(connection):
        object_type = str(row["type"])
        name = str(row["name"])
        table_name = str(row["tbl_name"])
        sql = row["sql"]
        if object_type == "index" and sql is None:
            if not name.startswith("sqlite_autoindex_"):
                raise VerificationStateError(
                    "Persisted verification schema contains an invalid automatic index."
                )
            autoindexes.add((name, table_name))
            continue
        definitions[(object_type, name)] = (table_name, _normalize_schema_sql(sql))
    if definitions != _owned_schema_definitions or frozenset(autoindexes) != _owned_autoindexes:
        raise VerificationStateError(
            "Persisted verification authority objects are incompatible with this Studio version."
        )


def _create_run_delete_trigger(connection: sqlite3.Connection) -> None:
    connection.execute(_RUN_NO_DELETE_TRIGGER_SQL)


def _ensure_schema(connection: sqlite3.Connection) -> None:
    for _table_name, definition in _TABLE_DEFINITIONS:
        connection.execute(definition)
    _verify_table_definitions(connection)
    expected_config_columns = {
        "project_id",
        "workspace_device_id",
        "workspace_file_id",
        "workspace_revision",
        "checks_json",
        "config_hash",
        "revision",
        "updated_at",
    }
    expected_run_columns = {
        "id",
        "project_id",
        "owner_id",
        "status",
        "terminal_hint",
        "config_revision",
        "config_hash",
        "workspace_device_id",
        "workspace_file_id",
        "workspace_revision",
        "config_checks_json",
        "checks_json",
        "checks_hash",
        "results_json",
        "evidence_revision",
        "cancel_requested",
        "recovered_after_restart",
        "error",
        "started_at",
        "heartbeat_at",
        "updated_at",
        "lease_expires_at",
        "completed_at",
        "history_sequence",
    }
    config_columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(agent_verification_configs)").fetchall()
    }
    run_columns = {
        str(row[1])
        for row in connection.execute("PRAGMA table_info(agent_verification_runs)").fetchall()
    }
    expected_deletion_columns = {
        "project_id",
        "fence_id",
        "active",
        "revision",
        "created_at",
        "heartbeat_at",
        "lease_expires_at",
    }
    deletion_columns = {
        str(row[1])
        for row in connection.execute(
            "PRAGMA table_info(agent_verification_project_deletions)"
        ).fetchall()
    }
    sequence_columns = {
        str(row[1])
        for row in connection.execute(
            "PRAGMA table_info(agent_verification_history_sequences)"
        ).fetchall()
    }
    if (
        config_columns != expected_config_columns
        or run_columns != expected_run_columns
        or deletion_columns != expected_deletion_columns
        or sequence_columns != {"sequence"}
    ):
        raise VerificationStateError(
            "Persisted verification schema is incompatible with this Studio version."
        )
    # These objects enforce durable authority, not only query performance. Always
    # replace objects owned by this module so an older or weaker same-name schema
    # cannot survive an upgrade. _connection holds an immediate transaction while
    # this refresh runs, so writers never observe the gap between drop and create.
    for trigger_name in _TRIGGER_NAMES:
        connection.execute(f"DROP TRIGGER IF EXISTS {trigger_name}")
    for index_name in _INDEX_NAMES:
        connection.execute(f"DROP INDEX IF EXISTS {index_name}")
    connection.execute(
        """
        CREATE UNIQUE INDEX idx_agent_verification_runs_one_active_project
        ON agent_verification_runs(project_id)
        WHERE status = 'running'
        """
    )
    connection.execute(
        """
        CREATE INDEX idx_agent_verification_runs_project_history
        ON agent_verification_runs(project_id, history_sequence DESC)
        WHERE status != 'running'
        """
    )
    connection.execute(
        """
        CREATE INDEX idx_agent_verification_runs_expired_lease
        ON agent_verification_runs(status, lease_expires_at)
        """
    )
    connection.execute(
        """
        CREATE TRIGGER trg_agent_verification_configs_no_replace
        BEFORE INSERT ON agent_verification_configs
        WHEN EXISTS (
            SELECT 1
            FROM agent_verification_configs
            WHERE project_id = NEW.project_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'verification configs cannot be replaced');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER trg_agent_verification_configs_revision_monotonic
        BEFORE UPDATE ON agent_verification_configs
        WHEN NEW.project_id != OLD.project_id OR NEW.revision <= OLD.revision
        BEGIN
            SELECT CASE
                WHEN NEW.project_id != OLD.project_id
                THEN RAISE(ABORT, 'verification config identity is immutable')
                WHEN NEW.revision <= OLD.revision
                THEN RAISE(ABORT, 'verification config revision must increase')
            END;
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER trg_agent_verification_configs_no_delete
        BEFORE DELETE ON agent_verification_configs
        WHEN EXISTS (
            SELECT 1 FROM chat_projects WHERE id = OLD.project_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'verification configs cannot be deleted');
        END
        """
    )
    connection.execute(
        f"""
        CREATE TRIGGER trg_agent_verification_runs_evidence_revision
        AFTER UPDATE OF
            id, project_id, owner_id, status, terminal_hint,
            config_revision, config_hash,
            workspace_device_id, workspace_file_id, workspace_revision,
            config_checks_json, checks_json, checks_hash, results_json,
            cancel_requested, recovered_after_restart, error,
            started_at, updated_at, completed_at, history_sequence
        ON agent_verification_runs
        WHEN NEW.evidence_revision = OLD.evidence_revision
        BEGIN
            SELECT CASE
                WHEN OLD.evidence_revision >= {MAX_REVISION}
                THEN RAISE(ABORT, 'verification evidence revision exhausted')
            END;
            UPDATE agent_verification_runs
            SET evidence_revision = OLD.evidence_revision + 1
            WHERE id = NEW.id;
        END
        """
    )
    connection.execute(
        f"""
        CREATE TRIGGER trg_agent_verification_runs_terminal_lease_evidence
        AFTER UPDATE OF heartbeat_at, lease_expires_at
        ON agent_verification_runs
        WHEN OLD.status != 'running'
            AND NEW.evidence_revision = OLD.evidence_revision
        BEGIN
            SELECT CASE
                WHEN OLD.evidence_revision >= {MAX_REVISION}
                THEN RAISE(ABORT, 'verification evidence revision exhausted')
            END;
            UPDATE agent_verification_runs
            SET evidence_revision = OLD.evidence_revision + 1
            WHERE id = NEW.id;
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER trg_agent_verification_runs_revision_monotonic
        BEFORE UPDATE OF evidence_revision ON agent_verification_runs
        WHEN NEW.evidence_revision <= OLD.evidence_revision
        BEGIN
            SELECT RAISE(ABORT, 'verification evidence revision must increase');
        END
        """
    )
    # INSERT OR REPLACE can delete a conflicting row before inserting a fresh id,
    # so protecting only the primary key is not enough for either the active-project
    # or terminal-history unique key.
    connection.execute(
        """
        CREATE TRIGGER trg_agent_verification_runs_no_replace
        BEFORE INSERT ON agent_verification_runs
        WHEN NEW.history_sequence IS NOT NULL
            OR EXISTS (
                SELECT 1 FROM agent_verification_runs WHERE id = NEW.id
            )
            OR (
                NEW.status = 'running'
                AND EXISTS (
                    SELECT 1
                    FROM agent_verification_runs
                    WHERE project_id = NEW.project_id AND status = 'running'
                )
            )
        BEGIN
            SELECT RAISE(ABORT, 'verification runs cannot be replaced');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER trg_agent_verification_runs_identity_immutable
        BEFORE UPDATE ON agent_verification_runs
        WHEN NEW.id != OLD.id OR NEW.project_id != OLD.project_id
        BEGIN
            SELECT RAISE(ABORT, 'verification run identity is immutable');
        END
        """
    )
    _create_run_delete_trigger(connection)
    connection.execute(
        """
        CREATE TRIGGER trg_agent_verification_runs_lifecycle_immutable
        BEFORE UPDATE ON agent_verification_runs
        WHEN (
            OLD.status != 'running'
            AND (
                NEW.status != OLD.status
                OR NEW.completed_at != OLD.completed_at
                OR NEW.history_sequence != OLD.history_sequence
            )
        ) OR (
            NEW.history_sequence IS NOT NULL
            AND EXISTS (
                SELECT 1 FROM agent_verification_runs
                WHERE id != OLD.id
                    AND history_sequence = NEW.history_sequence
            )
        )
        BEGIN
            SELECT RAISE(ABORT, 'verification run lifecycle is immutable');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER trg_agent_verification_project_deletions_no_replace
        BEFORE INSERT ON agent_verification_project_deletions
        WHEN NEW.active != 1
            OR NEW.revision != 1
            OR EXISTS (
                SELECT 1
                FROM agent_verification_project_deletions
                WHERE project_id = NEW.project_id
            )
        BEGIN
            SELECT RAISE(ABORT, 'verification deletion fences cannot be replaced');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER trg_agent_verification_project_deletions_no_delete
        BEFORE DELETE ON agent_verification_project_deletions
        WHEN EXISTS (
            SELECT 1 FROM chat_projects WHERE id = OLD.project_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'verification deletion fences cannot be deleted');
        END
        """
    )
    connection.execute(
        """
        CREATE TRIGGER trg_agent_verification_project_deletions_lifecycle
        BEFORE UPDATE ON agent_verification_project_deletions
        WHEN NEW.project_id != OLD.project_id
            OR (
                OLD.active = 1
                AND (
                    NEW.revision != OLD.revision
                    OR NEW.fence_id != OLD.fence_id
                    OR NEW.created_at != OLD.created_at
                    OR NEW.heartbeat_at < OLD.heartbeat_at
                    OR NEW.lease_expires_at < OLD.lease_expires_at
                )
            )
            OR (
                OLD.active = 0
                AND (
                    NEW.active != 1
                    OR NEW.revision != OLD.revision + 1
                )
            )
        BEGIN
            SELECT RAISE(ABORT, 'verification deletion fence lifecycle is invalid');
        END
        """
    )
    _capture_owned_schema_definitions(connection)
    _verify_schema_objects(connection)


def _attest_connection(
    connection: sqlite3.Connection | _VerificationConnection,
    *,
    expected_identity: Optional[_DatabaseIdentity] = None,
) -> _DatabaseKey:
    if not connection.in_transaction:
        raise VerificationStateError(
            "Persisted verification authority was inspected without a held transaction."
        )
    before = _database_key(connection, expected_identity = expected_identity)
    expected_path = os.path.realpath(str(studio_db.studio_db_path()))
    if before[0] != expected_path:
        raise VerificationStateError(
            "Persisted verification database path changed during authority inspection."
        )
    _verify_schema_objects(connection)
    after = _database_key(connection, expected_identity = before[1:3])
    if after != before:
        raise VerificationStateError(
            "Persisted verification database changed during authority inspection."
        )
    return after


def _connection() -> _VerificationConnection:
    expected_path = os.path.realpath(str(studio_db.studio_db_path()))
    expected_identity = _prepare_database_identity(expected_path)
    connection = studio_db.get_connection(30.0)
    try:
        key = _database_key(connection, expected_identity = expected_identity)
        if key[0] != expected_path:
            raise VerificationStateError(
                "Persisted verification database path changed while it was being opened."
            )
        _reject_cached_identity_change(key)
        if key not in _ready_databases:
            with _schema_lock:
                key = _database_key(connection, expected_identity = key[1:3])
                _reject_cached_identity_change(key)
                if key not in _ready_databases:
                    connection.execute("BEGIN IMMEDIATE")
                    key = _database_key(connection, expected_identity = key[1:3])
                    if key not in _ready_databases:
                        _ensure_schema(connection)
                        key = _database_key(connection, expected_identity = key[1:3])
                    connection.commit()
                    stale_keys = {ready for ready in _ready_databases if ready[0] == key[0]}
                    _ready_databases.difference_update(stale_keys)
                    _ready_databases.add(key)
        connection.execute("BEGIN")
        key = _attest_connection(connection, expected_identity = key[1:3])
        return _VerificationConnection(connection, key)
    except Exception:
        connection.close()
        raise


def _validate_text(
    value: object,
    *,
    label: str,
    maximum_bytes: int,
    allow_empty: bool = False,
    forbid_controls: bool = False,
    forbid_format_controls: bool = False,
    forbid_nul: bool = True,
) -> str:
    if not isinstance(value, str):
        raise AgentWorkspaceError(f"{label} must be a string.")
    try:
        encoded = value.encode("utf-8", errors = "strict")
    except UnicodeEncodeError as exc:
        raise AgentWorkspaceError(f"{label} must be valid UTF-8 text.") from exc
    if (not allow_empty and not encoded) or len(encoded) > maximum_bytes:
        raise AgentWorkspaceError(f"{label} is empty or exceeds its size limit.")
    if forbid_nul and "\x00" in value:
        raise AgentWorkspaceError(f"{label} contains an invalid NUL character.")
    if forbid_controls and any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise AgentWorkspaceError(f"{label} contains invalid control characters.")
    if forbid_format_controls and any(
        _is_format_control_code_point(character) or _is_default_ignorable_code_point(character)
        for character in value
    ):
        raise AgentWorkspaceError(
            f"{label} contains invalid Unicode format controls or default-ignorable code points."
        )
    return value


def _validate_project_id(project_id: object) -> str:
    return _validate_text(
        project_id,
        label = "Project id",
        maximum_bytes = MAX_PROJECT_ID_BYTES,
        forbid_controls = True,
    )


def _validate_owner_id(owner_id: object) -> str:
    return _validate_text(
        owner_id,
        label = "Verification owner id",
        maximum_bytes = MAX_OWNER_ID_BYTES,
        forbid_controls = True,
    )


def _validate_fence_id(fence_id: object) -> str:
    return _validate_text(
        fence_id,
        label = "Verification deletion fence id",
        maximum_bytes = MAX_FENCE_ID_BYTES,
        forbid_controls = True,
    )


def _validate_integer(
    value: object,
    *,
    label: str,
    minimum: int = 0,
    maximum: int = MAX_REVISION,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AgentWorkspaceError(f"{label} must be an integer.")
    if value < minimum or value > maximum:
        raise AgentWorkspaceError(f"{label} is outside the supported range.")
    return value


def _validate_revision(value: object, *, allow_zero: bool) -> int:
    return _validate_integer(
        value,
        label = "Verification revision",
        minimum = 0 if allow_zero else 1,
    )


def _validate_workspace_identity(identity: object) -> tuple[int, int]:
    if not isinstance(identity, tuple) or len(identity) != 2:
        raise AgentWorkspaceError("Verification workspace identity is invalid.")
    return (
        _validate_integer(identity[0], label = "Workspace device id"),
        _validate_integer(identity[1], label = "Workspace file id"),
    )


def _validate_hash(value: object, *, label: str = "Verification configuration hash") -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise AgentWorkspaceError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("JSON object keys must be unique")
        result[key] = value
    return result


def _strict_json_loads(raw: object, *, label: str) -> Any:
    if not isinstance(raw, str):
        raise VerificationStateError(f"Persisted {label} is not JSON text.")
    try:
        return json.loads(
            raw,
            object_pairs_hook = _strict_object,
            parse_constant = _reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VerificationStateError(f"Persisted {label} is invalid JSON.") from exc


def _canonical_json(value: Any, *, limit: int, label: str) -> str:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii = False,
            allow_nan = False,
            sort_keys = True,
            separators = (",", ":"),
        )
        encoded = rendered.encode("utf-8", errors = "strict")
    except (TypeError, UnicodeError, ValueError) as exc:
        raise AgentWorkspaceError(f"{label} must be bounded JSON data.") from exc
    if len(encoded) > limit:
        raise AgentWorkspaceError(f"{label} exceeds the supported size limit.")
    return rendered


def _hash_json(encoded: str) -> str:
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _normalize_check_integer(
    check: dict, key: str, *, default: int, minimum: int, maximum: int
) -> int:
    value = check[key] if key in check else default
    return _validate_integer(
        value,
        label = "Verification check limit",
        minimum = minimum,
        maximum = maximum,
    )


def normalize_verification_checks(checks: list[dict]) -> list[dict]:
    """Validate and canonicalize a bounded verification profile."""
    if not isinstance(checks, list):
        raise AgentWorkspaceError("Verification checks must be a list.")
    if len(checks) > MAX_CHECKS:
        raise AgentWorkspaceError(f"At most {MAX_CHECKS} verification checks can be configured.")

    normalized: list[dict] = []
    names: set[str] = set()
    for check in checks:
        if not isinstance(check, dict):
            raise AgentWorkspaceError("Each verification check must be an object.")
        unknown = set(check) - _CHECK_FIELDS
        required_fields = {"name", "command"} - set(check)
        if unknown or required_fields:
            raise AgentWorkspaceError("A verification check has unknown or missing fields.")
        if "required" in check and not isinstance(check["required"], bool):
            raise AgentWorkspaceError("Verification check required must be a boolean.")
        name = _validate_text(
            check["name"],
            label = "Verification check name",
            maximum_bytes = MAX_CHECK_NAME_CHARACTERS * 4,
            forbid_controls = True,
            forbid_format_controls = True,
        ).strip()
        command_text = _validate_text(
            check["command"],
            label = "Verification command",
            maximum_bytes = MAX_CHECK_COMMAND_BYTES,
            forbid_format_controls = True,
        )
        if any(
            (ord(character) < 32 and character not in {"\t", "\n"}) or 127 <= ord(character) <= 159
            for character in command_text
        ):
            raise AgentWorkspaceError("Verification commands contain unsafe control characters.")
        if any(ord(character) > 127 and character.isspace() for character in command_text):
            raise AgentWorkspaceError("Verification commands contain unsafe Unicode whitespace.")
        command = command_text.strip()
        if not name or not command:
            raise AgentWorkspaceError("Verification check names and commands cannot be blank.")
        if len(name) > MAX_CHECK_NAME_CHARACTERS:
            raise AgentWorkspaceError("Verification check names are too long.")
        key = name.casefold()
        if key in names:
            raise AgentWorkspaceError("Verification check names must be unique.")
        names.add(key)

        raw_kind = check.get("kind", "custom")
        kind = _validate_text(
            raw_kind,
            label = "Verification check kind",
            maximum_bytes = MAX_CHECK_KIND_BYTES,
            forbid_controls = True,
            forbid_format_controls = True,
        ).strip()
        if not kind:
            raise AgentWorkspaceError("Verification check kind cannot be blank.")
        normalized.append(
            {
                "name": name,
                "kind": kind,
                "command": command,
                "required": check.get("required", True),
                "timeoutSeconds": _normalize_check_integer(
                    check,
                    "timeoutSeconds",
                    default = 300,
                    minimum = 1,
                    maximum = MAX_TIMEOUT_SECONDS,
                ),
                "logLimitBytes": _normalize_check_integer(
                    check,
                    "logLimitBytes",
                    default = 256 * 1024,
                    minimum = 1024,
                    maximum = MAX_LOG_LIMIT_BYTES,
                ),
            }
        )
    _canonical_json(
        normalized,
        limit = MAX_CONFIG_BYTES,
        label = "Verification configuration",
    )
    return normalized


def _encode_checks(checks: list[dict]) -> tuple[list[dict], str, str]:
    normalized = normalize_verification_checks(checks)
    encoded = _canonical_json(
        normalized,
        limit = MAX_CONFIG_BYTES,
        label = "Verification configuration",
    )
    return normalized, encoded, _hash_json(encoded)


def _decode_checks(raw: object, *, label: str) -> tuple[list[dict], str, str]:
    value = _strict_json_loads(raw, label = label)
    try:
        normalized, encoded, content_hash = _encode_checks(value)
    except AgentWorkspaceError as exc:
        raise VerificationStateError(f"Persisted {label} is invalid.") from exc
    if encoded != raw:
        raise VerificationStateError(f"Persisted {label} is not canonical JSON.")
    return normalized, encoded, content_hash


def _validate_error(error: object, *, persisted: bool = False) -> Optional[str]:
    if error is None:
        return None
    try:
        return _validate_text(
            error,
            label = "Verification error",
            maximum_bytes = MAX_ERROR_BYTES,
            allow_empty = False,
            forbid_nul = True,
        )
    except AgentWorkspaceError as exc:
        if persisted:
            raise VerificationStateError("Persisted verification error is invalid.") from exc
        raise


def _validate_result(result: object, check: dict, *, run_started_at: int) -> dict:
    if not isinstance(result, dict):
        raise AgentWorkspaceError("Each verification result must be an object.")
    if set(result) - _RESULT_FIELDS or _RESULT_REQUIRED_FIELDS - set(result):
        raise AgentWorkspaceError("A verification result has unknown or missing fields.")
    for field in ("name", "kind", "command", "required", "timeoutSeconds"):
        if result[field] != check[field] or type(result[field]) is not type(check[field]):
            raise AgentWorkspaceError(
                "Verification results must match the exact configured check prefix."
            )
    status = result["status"]
    if not isinstance(status, str) or status not in _RESULT_STATUSES:
        raise AgentWorkspaceError("Verification result status is invalid.")
    exit_code = result["exitCode"]
    if exit_code is not None:
        _validate_integer(
            exit_code,
            label = "Verification result exit code",
            minimum = -(1 << 31),
            maximum = (1 << 31) - 1,
        )
    output = _validate_text(
        result["output"],
        label = "Verification result output",
        maximum_bytes = MAX_LOG_LIMIT_BYTES + MAX_TRUNCATION_DECORATION_BYTES,
        allow_empty = True,
        forbid_nul = False,
    )
    output_size = len(output.encode("utf-8"))
    output_truncated = result["outputTruncated"]
    if not isinstance(output_truncated, bool):
        raise AgentWorkspaceError("Verification result truncation state must be a boolean.")
    allowed_output = check["logLimitBytes"] + (
        MAX_TRUNCATION_DECORATION_BYTES if output_truncated else 0
    )
    if output_size > allowed_output:
        raise AgentWorkspaceError("Verification result output exceeds its per-check limit.")
    output_bytes = _validate_integer(
        result["outputBytes"],
        label = "Verification result output byte count",
    )
    if status == "running":
        if output_truncated or output_bytes != output_size:
            raise AgentWorkspaceError("Running verification output byte metadata is inconsistent.")
    else:
        captured_output = output
        if output_truncated:
            truncation = _TRUNCATION_NOTICE.search(output)
            if truncation is None:
                raise AgentWorkspaceError(
                    "Truncated verification output lacks its exact capture notice."
                )
            capture_limit = _validate_integer(
                int(truncation.group(1)),
                label = "Verification result capture limit",
                minimum = 1,
                maximum = check["logLimitBytes"],
            )
            if capture_limit > check["logLimitBytes"] or output_bytes == 0:
                raise AgentWorkspaceError(
                    "Truncated verification output byte metadata is inconsistent."
                )
            captured_output = output[: truncation.start()]
        minimum_raw_bytes = sum(
            1 if character == "\ufffd" else len(character.encode("utf-8"))
            for character in captured_output
        )
        if (
            (output_truncated and minimum_raw_bytes > capture_limit)
            or output_bytes < minimum_raw_bytes
            or (not output_truncated and output_bytes > output_size)
        ):
            raise AgentWorkspaceError("Verification output byte metadata is inconsistent.")
    started_at = _validate_integer(
        result["startedAt"],
        label = "Verification result start time",
    )
    if started_at < run_started_at:
        raise AgentWorkspaceError("Verification result predates its run.")
    completed_at = result["completedAt"]
    duration_ms = result["durationMs"]
    item_error = _validate_error(result.get("error"))

    if status == "running":
        if exit_code is not None or completed_at is not None or duration_ms is not None:
            raise AgentWorkspaceError("A running verification result has terminal fields.")
        if item_error is not None:
            raise AgentWorkspaceError("A running verification result cannot contain an error.")
    else:
        completed_at = _validate_integer(
            completed_at,
            label = "Verification result completion time",
        )
        duration_ms = _validate_integer(
            duration_ms,
            label = "Verification result duration",
        )
        if completed_at < started_at or duration_ms != completed_at - started_at:
            raise AgentWorkspaceError("Verification result timing is inconsistent.")
        if status == "passed" and exit_code != 0:
            raise AgentWorkspaceError("A passed verification result must have exit code zero.")
        if status == "failed" and exit_code == 0:
            raise AgentWorkspaceError("A failed verification result cannot have exit code zero.")
        if status in {"cancelled", "timed_out", "blocked"} and exit_code is not None:
            raise AgentWorkspaceError(
                "Cancelled, timed out, and blocked verification results cannot have exit codes."
            )
        if item_error is not None and status not in {"failed", "blocked"}:
            raise AgentWorkspaceError("This verification result status cannot contain an error.")

    normalized = {
        "name": result["name"],
        "kind": result["kind"],
        "command": result["command"],
        "required": result["required"],
        "status": status,
        "exitCode": exit_code,
        "output": output,
        "outputBytes": output_bytes,
        "outputTruncated": output_truncated,
        "timeoutSeconds": result["timeoutSeconds"],
        "startedAt": started_at,
        "completedAt": completed_at,
        "durationMs": duration_ms,
    }
    if item_error is not None:
        normalized["error"] = item_error
    return normalized


def _validate_results(
    results: object, checks: list[dict], *, run_started_at: int, allow_running_tail: bool
) -> list[dict]:
    if not isinstance(results, list) or len(results) > len(checks):
        raise AgentWorkspaceError("Verification results are not a configured check prefix.")
    normalized: list[dict] = []
    retained_output_bytes = 0
    for index, result in enumerate(results):
        item = _validate_result(result, checks[index], run_started_at = run_started_at)
        if normalized:
            previous_completed_at = normalized[-1]["completedAt"]
            if previous_completed_at is None or item["startedAt"] < previous_completed_at:
                raise AgentWorkspaceError(
                    "Verification results must be sequential and non-overlapping."
                )
        if item["status"] == "running" and (not allow_running_tail or index != len(results) - 1):
            raise AgentWorkspaceError("Only the final progress result can be running.")
        retained_output_bytes += len(item["output"].encode("utf-8"))
        if retained_output_bytes > MAX_RUN_RESULT_BYTES:
            raise AgentWorkspaceError("Verification result output exceeds the run limit.")
        normalized.append(item)
    _canonical_json(
        normalized,
        limit = MAX_RUN_RESULTS_JSON_BYTES,
        label = "Verification results",
    )
    return normalized


def _encode_results(results: list[dict]) -> str:
    return _canonical_json(
        results,
        limit = MAX_RUN_RESULTS_JSON_BYTES,
        label = "Verification results",
    )


def _decode_results(
    raw: object, checks: list[dict], *, run_started_at: int, allow_running_tail: bool
) -> tuple[list[dict], str]:
    value = _strict_json_loads(raw, label = "verification results")
    try:
        normalized = _validate_results(
            value,
            checks,
            run_started_at = run_started_at,
            allow_running_tail = allow_running_tail,
        )
        encoded = _encode_results(normalized)
    except AgentWorkspaceError as exc:
        raise VerificationStateError("Persisted verification results are invalid.") from exc
    if encoded != raw:
        raise VerificationStateError("Persisted verification results are not canonical JSON.")
    return normalized, encoded


def _validate_progress_transition(previous: list[dict], current: list[dict]) -> None:
    stable_count = len(previous)
    previous_running = bool(previous and previous[-1]["status"] == "running")
    if previous_running:
        stable_count -= 1
    if len(current) < stable_count or current[:stable_count] != previous[:stable_count]:
        raise VerificationConflictError("Completed verification progress cannot change.")
    if not previous_running:
        return
    if len(current) != len(previous):
        raise VerificationConflictError(
            "Running verification progress must complete before the prefix can grow."
        )
    old_tail = previous[-1]
    new_tail = current[-1]
    if old_tail["startedAt"] != new_tail["startedAt"]:
        raise VerificationConflictError("Running verification progress changed its start time.")
    if not new_tail["output"].startswith(old_tail["output"]):
        raise VerificationConflictError("Running verification output is not append-only.")
    if new_tail["status"] == "running":
        if new_tail["outputBytes"] < old_tail["outputBytes"]:
            raise VerificationConflictError("Running verification output counters regressed.")
        if old_tail["outputTruncated"] and not new_tail["outputTruncated"]:
            raise VerificationConflictError("Running verification truncation state regressed.")


def _validate_selected_checks(selected: list[dict], configured: list[dict]) -> None:
    if not selected:
        raise AgentWorkspaceError("No verification checks are configured for this run.")
    configured_index = 0
    for check in selected:
        while configured_index < len(configured) and configured[configured_index] != check:
            configured_index += 1
        if configured_index >= len(configured):
            raise VerificationConflictError(
                "Verification checks changed after this run was prepared."
            )
        configured_index += 1


def _config_from_row(row: sqlite3.Row, project_id: str) -> dict:
    try:
        stored_project_id = _validate_project_id(row["project_id"])
        identity = _validate_workspace_identity(
            (row["workspace_device_id"], row["workspace_file_id"])
        )
        workspace_revision = _validate_revision(row["workspace_revision"], allow_zero = True)
        revision = _validate_revision(row["revision"], allow_zero = False)
        updated_at = _validate_integer(row["updated_at"], label = "Verification update time")
        checks, _encoded, computed_hash = _decode_checks(
            row["checks_json"],
            label = "verification configuration",
        )
        config_hash = _validate_hash(row["config_hash"])
    except (AgentWorkspaceError, TypeError) as exc:
        if isinstance(exc, VerificationStateError):
            raise
        raise VerificationStateError("Persisted verification configuration is invalid.") from exc
    if stored_project_id != project_id or computed_hash != config_hash:
        raise VerificationStateError("Persisted verification configuration hash is invalid.")
    return {
        "checks": checks,
        "configHash": config_hash,
        "workspaceDeviceId": identity[0],
        "workspaceFileId": identity[1],
        "workspaceRevision": workspace_revision,
        "revision": revision,
        "updatedAt": updated_at,
        "sourceFreshness": SOURCE_FRESHNESS,
    }


def _empty_config(project_id: str) -> dict:
    _checks, _encoded, config_hash = _encode_checks([])
    return {
        "checks": [],
        "configHash": config_hash,
        "workspaceDeviceId": None,
        "workspaceFileId": None,
        "workspaceRevision": None,
        "revision": 0,
        "updatedAt": None,
        "sourceFreshness": SOURCE_FRESHNESS,
    }


def _derive_terminal_status(
    checks: list[dict], results: list[dict], *, cancel_requested: bool, terminal_hint: Optional[str]
) -> str:
    if terminal_hint is not None and terminal_hint not in _TERMINAL_HINTS:
        raise AgentWorkspaceError("Verification terminal status hint is invalid.")
    if cancel_requested or terminal_hint == "cancelled":
        return "cancelled"
    if terminal_hint is not None:
        return terminal_hint
    for index, check in enumerate(checks):
        if not check["required"]:
            continue
        if index >= len(results):
            return "blocked"
        status = results[index]["status"]
        if status != "passed":
            if status == "running":
                return "blocked"
            return status
    return "passed"


def _run_from_row(row: sqlite3.Row) -> dict:
    try:
        run_id = str(uuid.UUID(str(row["id"])))
        if run_id != row["id"]:
            raise ValueError("non-canonical UUID")
        project_id = _validate_project_id(row["project_id"])
        owner_id = _validate_owner_id(row["owner_id"])
        status = row["status"]
        if not isinstance(status, str) or status not in _RUN_STATUSES:
            raise AgentWorkspaceError("Verification run status is invalid.")
        terminal_hint = row["terminal_hint"]
        if terminal_hint is not None and terminal_hint not in _TERMINAL_HINTS:
            raise AgentWorkspaceError("Verification terminal status hint is invalid.")
        config_revision = _validate_revision(row["config_revision"], allow_zero = False)
        config_hash = _validate_hash(row["config_hash"])
        identity = _validate_workspace_identity(
            (row["workspace_device_id"], row["workspace_file_id"])
        )
        workspace_revision = _validate_revision(row["workspace_revision"], allow_zero = True)
        config_checks, _config_json, computed_config_hash = _decode_checks(
            row["config_checks_json"],
            label = "verification run configuration",
        )
        checks, _checks_json, computed_checks_hash = _decode_checks(
            row["checks_json"],
            label = "verification run checks",
        )
        checks_hash = _validate_hash(row["checks_hash"], label = "Verification checks hash")
        evidence_revision = _validate_revision(row["evidence_revision"], allow_zero = False)
        _validate_selected_checks(checks, config_checks)
        started_at = _validate_integer(row["started_at"], label = "Verification start time")
        heartbeat_at = _validate_integer(
            row["heartbeat_at"],
            label = "Verification heartbeat time",
        )
        updated_at = _validate_integer(row["updated_at"], label = "Verification update time")
        lease_expires_at = _validate_integer(
            row["lease_expires_at"],
            label = "Verification lease expiry",
        )
        cancel_requested = row["cancel_requested"]
        recovered = row["recovered_after_restart"]
        if cancel_requested not in (0, 1) or recovered not in (0, 1):
            raise AgentWorkspaceError("Verification run flags are invalid.")
        completed_at = row["completed_at"]
        if completed_at is not None:
            completed_at = _validate_integer(
                completed_at,
                label = "Verification completion time",
            )
        history_sequence = row["history_sequence"]
        if history_sequence is not None:
            history_sequence = _validate_integer(
                history_sequence,
                label = "Verification history sequence",
                minimum = 1,
                maximum = MAX_HISTORY_SEQUENCE,
            )
        error = _validate_error(row["error"], persisted = True)
        results, _results_json = _decode_results(
            row["results_json"],
            checks,
            run_started_at = started_at,
            allow_running_tail = status == "running" or bool(recovered),
        )
    except (AgentWorkspaceError, TypeError, ValueError) as exc:
        if isinstance(exc, VerificationStateError):
            raise
        raise VerificationStateError("Persisted verification run is invalid.") from exc

    if computed_config_hash != config_hash or computed_checks_hash != checks_hash:
        raise VerificationStateError("Persisted verification run hash binding is invalid.")
    if heartbeat_at < started_at or updated_at < started_at or lease_expires_at < heartbeat_at:
        raise VerificationStateError("Persisted verification run timestamps are invalid.")
    if status == "running":
        if completed_at is not None or history_sequence is not None or recovered:
            raise VerificationStateError("Persisted running verification lifecycle is invalid.")
    else:
        if (
            completed_at is None
            or history_sequence is None
            or completed_at < started_at
            or completed_at < updated_at
        ):
            raise VerificationStateError("Persisted terminal verification timestamps are invalid.")
        derived = _derive_terminal_status(
            checks,
            results,
            cancel_requested = bool(cancel_requested),
            terminal_hint = terminal_hint,
        )
        if derived != status:
            raise VerificationStateError("Persisted verification terminal status is inconsistent.")
        if status == "passed" and (
            any(result["status"] == "running" for result in results)
            or any(
                check["required"]
                and (index >= len(results) or results[index]["status"] != "passed")
                for index, check in enumerate(checks)
            )
        ):
            raise VerificationStateError("Persisted passed verification lacks required evidence.")
        if not recovered and any(result["status"] == "running" for result in results):
            raise VerificationStateError("Persisted terminal verification has running progress.")
        if any(
            result["completedAt"] is not None and result["completedAt"] > completed_at
            for result in results
        ):
            raise VerificationStateError(
                "Persisted verification run completed before its result evidence."
            )
    if status in {"passed", "cancelled"} and error is not None:
        raise VerificationStateError("Persisted verification error conflicts with its status.")

    return {
        "id": run_id,
        "projectId": project_id,
        "ownerId": owner_id,
        "status": status,
        "configRevision": config_revision,
        "configHash": config_hash,
        "workspaceDeviceId": identity[0],
        "workspaceFileId": identity[1],
        "workspaceRevision": workspace_revision,
        "checks": checks,
        "results": results,
        "evidenceRevision": evidence_revision,
        "cancelRequested": bool(cancel_requested),
        "error": error,
        "startedAt": started_at,
        "heartbeatAt": heartbeat_at,
        "updatedAt": updated_at,
        "leaseExpiresAt": lease_expires_at,
        "completedAt": completed_at,
        "historySequence": history_sequence,
        "sourceFreshness": SOURCE_FRESHNESS,
    }


def _run_summary_from_row(row: sqlite3.Row) -> dict:
    """Validate bounded metadata without treating unloaded result bodies as evidence."""
    try:
        run_id = str(uuid.UUID(str(row["id"])))
        if run_id != row["id"]:
            raise ValueError("non-canonical UUID")
        project_id = _validate_project_id(row["project_id"])
        status = row["status"]
        if not isinstance(status, str) or status not in _RUN_STATUSES:
            raise AgentWorkspaceError("Verification run status is invalid.")
        config_revision = _validate_revision(row["config_revision"], allow_zero = False)
        workspace_revision = _validate_revision(row["workspace_revision"], allow_zero = True)
        evidence_revision = _validate_revision(row["evidence_revision"], allow_zero = False)
        cancel_requested = row["cancel_requested"]
        if cancel_requested not in (0, 1):
            raise AgentWorkspaceError("Verification cancellation state is invalid.")
        error = _validate_error(row["error"], persisted = True)
        started_at = _validate_integer(row["started_at"], label = "Verification start time")
        updated_at = _validate_integer(row["updated_at"], label = "Verification update time")
        completed_at = row["completed_at"]
        if completed_at is not None:
            completed_at = _validate_integer(
                completed_at,
                label = "Verification completion time",
            )
        history_sequence = row["history_sequence"]
        if history_sequence is not None:
            history_sequence = _validate_integer(
                history_sequence,
                label = "Verification history sequence",
                minimum = 1,
                maximum = MAX_HISTORY_SEQUENCE,
            )
    except (AgentWorkspaceError, TypeError, ValueError) as exc:
        if isinstance(exc, VerificationStateError):
            raise
        raise VerificationStateError("Persisted verification run summary is invalid.") from exc
    if updated_at < started_at:
        raise VerificationStateError("Persisted verification run summary timestamps are invalid.")
    if status == "running":
        if completed_at is not None or history_sequence is not None:
            raise VerificationStateError("Persisted running verification summary is invalid.")
    elif completed_at is None or history_sequence is None or completed_at < updated_at:
        raise VerificationStateError("Persisted terminal verification summary is invalid.")
    if status in {"passed", "cancelled"} and error is not None:
        raise VerificationStateError(
            "Persisted verification summary error conflicts with its status."
        )
    return {
        "id": run_id,
        "projectId": project_id,
        "status": "running" if status == "running" else "unverified",
        "evidenceStatus": "not_loaded",
        "configRevision": config_revision,
        "workspaceRevision": workspace_revision,
        "evidenceRevision": evidence_revision,
        "cancelRequested": bool(cancel_requested),
        "error": error,
        "startedAt": started_at,
        "updatedAt": updated_at,
        "completedAt": completed_at,
        "historySequence": history_sequence,
        "sourceFreshness": SOURCE_FRESHNESS,
    }


def _read_config_row(connection: sqlite3.Connection, project_id: str) -> Optional[sqlite3.Row]:
    return connection.execute(
        "SELECT * FROM agent_verification_configs WHERE project_id = ?",
        (project_id,),
    ).fetchone()


def _project_archived_locked(connection: sqlite3.Connection, project_id: str) -> Optional[bool]:
    row = connection.execute(
        "SELECT archived FROM chat_projects WHERE id = ?",
        (project_id,),
    ).fetchone()
    if row is None:
        return None
    archived = row["archived"]
    if archived not in (0, 1):
        raise VerificationStateError("Persisted project archive state is invalid.")
    return bool(archived)


def _read_run_row(
    connection: sqlite3.Connection, project_id: str, run_id: str
) -> Optional[sqlite3.Row]:
    return connection.execute(
        "SELECT * FROM agent_verification_runs WHERE project_id = ? AND id = ?",
        (project_id, run_id),
    ).fetchone()


def _next_history_sequence_locked(connection: sqlite3.Connection) -> int:
    # sqlite_sequence is SQLite-owned metadata. Storage-owner edits to it are
    # outside this module's ordinary owned-table write boundary.
    row = connection.execute(
        """
        SELECT seq FROM sqlite_sequence
        WHERE name = 'agent_verification_history_sequences'
        """
    ).fetchone()
    if row is not None:
        try:
            previous = _validate_integer(
                row["seq"],
                label = "Verification history sequence",
                minimum = 1,
                maximum = MAX_HISTORY_SEQUENCE,
            )
        except AgentWorkspaceError as exc:
            raise VerificationStateError(
                "Persisted verification history sequence is invalid."
            ) from exc
        if previous >= MAX_HISTORY_SEQUENCE:
            raise VerificationConflictError("Verification history sequence is exhausted.")
    cursor = connection.execute("INSERT INTO agent_verification_history_sequences DEFAULT VALUES")
    try:
        sequence = _validate_integer(
            cursor.lastrowid,
            label = "Verification history sequence",
            minimum = 1,
            maximum = MAX_HISTORY_SEQUENCE,
        )
    except AgentWorkspaceError as exc:
        raise VerificationStateError("Persisted verification history sequence is invalid.") from exc
    deleted = connection.execute(
        "DELETE FROM agent_verification_history_sequences WHERE sequence = ?",
        (sequence,),
    )
    if deleted.rowcount != 1:
        raise VerificationStateError("Verification history sequence was not reserved.")
    return sequence


def _delete_prune_candidates_locked(
    connection: sqlite3.Connection, candidate_rows: list[sqlite3.Row]
) -> None:
    candidate_ids: list[str] = []
    seen_ids: set[str] = set()
    for row in candidate_rows:
        try:
            run_id = str(uuid.UUID(str(row["id"])))
        except (TypeError, ValueError) as exc:
            raise VerificationStateError(
                "Persisted verification retention identity is invalid."
            ) from exc
        if run_id != row["id"] or run_id in seen_ids:
            raise VerificationStateError("Persisted verification retention identity is invalid.")
        candidate_ids.append(run_id)
        seen_ids.add(run_id)
    if not candidate_ids:
        return

    connection.execute("DROP TRIGGER trg_agent_verification_runs_no_delete")
    try:
        for run_id in candidate_ids:
            cursor = connection.execute(
                """
                DELETE FROM agent_verification_runs
                WHERE id = ? AND status != 'running'
                """,
                (run_id,),
            )
            if cursor.rowcount != 1:
                raise VerificationConflictError("Verification retention changed during pruning.")
        _create_run_delete_trigger(connection)
        _verify_schema_objects(connection)
    except BaseException:
        trigger = connection.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'trigger'
                AND name = 'trg_agent_verification_runs_no_delete'
            """
        ).fetchone()
        if trigger is None:
            connection.execute(_RUN_NO_DELETE_TRIGGER_SQL)
        raise


def _prune_runs_locked(connection: sqlite3.Connection, project_id: str) -> None:
    project_candidates = connection.execute(
        f"""
        WITH ranked AS (
            SELECT
                id,
                ROW_NUMBER() OVER (
                    ORDER BY history_sequence DESC
                ) AS ordinal,
                SUM({_RUN_EVIDENCE_BYTES_SQL}) OVER (
                    ORDER BY history_sequence DESC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS retained_bytes
            FROM agent_verification_runs
            WHERE project_id = ? AND status != 'running'
        )
        SELECT id FROM ranked
        WHERE ordinal > ? OR retained_bytes > ?
        """,
        (project_id, MAX_RUN_HISTORY, MAX_PROJECT_RUN_HISTORY_BYTES),
    ).fetchall()
    _delete_prune_candidates_locked(connection, project_candidates)

    global_candidates = connection.execute(
        f"""
        WITH ranked AS (
            SELECT
                id,
                SUM({_RUN_EVIDENCE_BYTES_SQL}) OVER (
                    ORDER BY history_sequence DESC
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS retained_bytes
            FROM agent_verification_runs
            WHERE status != 'running'
        )
        SELECT id FROM ranked
        WHERE retained_bytes > ?
        """,
        (MAX_GLOBAL_RUN_HISTORY_BYTES,),
    ).fetchall()
    _delete_prune_candidates_locked(connection, global_candidates)


def _expired_run_metadata_from_row(row: sqlite3.Row) -> dict:
    """Validate only the bounded lifecycle fields needed to retire an expired lease."""
    try:
        run_id = str(uuid.UUID(str(row["id"])))
        if run_id != row["id"]:
            raise ValueError("non-canonical UUID")
        project_id = _validate_project_id(row["project_id"])
        if row["status"] != "running":
            raise AgentWorkspaceError("Verification run is not active.")
        if row["history_sequence"] is not None:
            raise AgentWorkspaceError("Active verification has terminal history state.")
        cancel_requested = row["cancel_requested"]
        if cancel_requested not in (0, 1):
            raise AgentWorkspaceError("Verification cancellation state is invalid.")
        evidence_revision = _validate_revision(row["evidence_revision"], allow_zero = False)
        started_at = _validate_integer(row["started_at"], label = "Verification start time")
        heartbeat_at = _validate_integer(
            row["heartbeat_at"],
            label = "Verification heartbeat time",
        )
        updated_at = _validate_integer(row["updated_at"], label = "Verification update time")
        lease_expires_at = _validate_integer(
            row["lease_expires_at"],
            label = "Verification lease expiry",
        )
    except (AgentWorkspaceError, TypeError, ValueError) as exc:
        if isinstance(exc, VerificationStateError):
            raise
        raise VerificationStateError("Persisted verification lease metadata is invalid.") from exc
    if heartbeat_at < started_at or updated_at < started_at or lease_expires_at < heartbeat_at:
        raise VerificationStateError("Persisted verification lease timestamps are invalid.")
    return {
        "id": run_id,
        "projectId": project_id,
        "cancelRequested": bool(cancel_requested),
        "evidenceRevision": evidence_revision,
        "startedAt": started_at,
        "heartbeatAt": heartbeat_at,
        "updatedAt": updated_at,
        "leaseExpiresAt": lease_expires_at,
    }


def _owned_run_metadata_from_row(row: sqlite3.Row) -> dict:
    record = _expired_run_metadata_from_row(row)
    try:
        owner_id = _validate_owner_id(row["owner_id"])
    except (AgentWorkspaceError, TypeError) as exc:
        raise VerificationStateError(
            "Persisted verification ownership metadata is invalid."
        ) from exc
    return record | {"ownerId": owner_id, "status": "running"}


def _reconcile_expired_deletion_fences_locked(
    connection: sqlite3.Connection,
    now: int,
    *,
    project_id: Optional[str] = None,
) -> int:
    if project_id is None:
        rows = connection.execute(
            """
            SELECT project_id, fence_id, active, revision,
                   created_at, heartbeat_at, lease_expires_at
            FROM agent_verification_project_deletions
            WHERE active = 1
            """,
        ).fetchall()
    else:
        rows = connection.execute(
            """
            SELECT project_id, fence_id, active, revision,
                   created_at, heartbeat_at, lease_expires_at
            FROM agent_verification_project_deletions
            WHERE project_id = ? AND active = 1
            """,
            (project_id,),
        ).fetchall()
    expired: list[dict] = []
    for row in rows:
        record = _deletion_fence_from_row(row)
        if record["leaseExpiresAt"] <= now:
            expired.append(record)
    for record in expired:
        cursor = connection.execute(
            """
            UPDATE agent_verification_project_deletions
            SET active = 0
            WHERE project_id = ? AND fence_id = ? AND active = 1
                AND revision = ? AND lease_expires_at <= ?
            """,
            (
                record["projectId"],
                record["fenceId"],
                record["revision"],
                now,
            ),
        )
        if cursor.rowcount != 1:
            raise VerificationConflictError(
                "Verification deletion fence changed during reconciliation."
            )
    return len(expired)


def _active_deletion_fence_locked(
    connection: sqlite3.Connection, project_id: str
) -> Optional[dict]:
    row = connection.execute(
        """
        SELECT project_id, fence_id, active, revision,
               created_at, heartbeat_at, lease_expires_at
        FROM agent_verification_project_deletions
        WHERE project_id = ? AND active = 1
        """,
        (project_id,),
    ).fetchone()
    return _deletion_fence_from_row(row) if row is not None else None


def _deletion_fence_tombstone_locked(
    connection: sqlite3.Connection, project_id: str
) -> Optional[dict]:
    row = connection.execute(
        """
        SELECT project_id, fence_id, active, revision,
               created_at, heartbeat_at, lease_expires_at
        FROM agent_verification_project_deletions
        WHERE project_id = ?
        """,
        (project_id,),
    ).fetchone()
    return _deletion_fence_from_row(row) if row is not None else None


def _deletion_fence_from_row(row: sqlite3.Row) -> dict:
    try:
        project_id = _validate_project_id(row["project_id"])
        fence_id = _validate_fence_id(row["fence_id"])
        active = row["active"]
        if active not in (0, 1):
            raise AgentWorkspaceError("Verification deletion fence state is invalid.")
        revision = _validate_revision(row["revision"], allow_zero = False)
        created_at = _validate_integer(
            row["created_at"],
            label = "Verification deletion fence creation time",
        )
        heartbeat_at = _validate_integer(
            row["heartbeat_at"],
            label = "Verification deletion fence heartbeat time",
        )
        lease_expires_at = _validate_integer(
            row["lease_expires_at"],
            label = "Verification deletion fence lease expiry",
        )
    except (AgentWorkspaceError, TypeError) as exc:
        raise VerificationStateError("Persisted verification deletion fence is invalid.") from exc
    if heartbeat_at < created_at or lease_expires_at < heartbeat_at:
        raise VerificationStateError("Persisted verification deletion fence times are invalid.")
    return {
        "projectId": project_id,
        "fenceId": fence_id,
        "active": bool(active),
        "revision": revision,
        "createdAt": created_at,
        "heartbeatAt": heartbeat_at,
        "leaseExpiresAt": lease_expires_at,
    }


def _reconcile_expired_locked(
    connection: sqlite3.Connection,
    now: int,
    *,
    project_id: Optional[str] = None,
) -> int:
    if project_id is None:
        rows = connection.execute(
            """
            SELECT id, project_id, status, cancel_requested, evidence_revision,
                   started_at, heartbeat_at, updated_at, lease_expires_at,
                   history_sequence
            FROM agent_verification_runs
            WHERE status = 'running' AND lease_expires_at <= ?
            ORDER BY rowid
            """,
            (now,),
        ).fetchall()
    else:
        rows = connection.execute(
            """
            SELECT id, project_id, status, cancel_requested, evidence_revision,
                   started_at, heartbeat_at, updated_at, lease_expires_at,
                   history_sequence
            FROM agent_verification_runs
            WHERE project_id = ? AND status = 'running' AND lease_expires_at <= ?
            ORDER BY rowid
            """,
            (project_id, now),
        ).fetchall()
    affected_projects: set[str] = set()
    for row in rows:
        record = _expired_run_metadata_from_row(row)
        status = "cancelled" if record["cancelRequested"] else "interrupted"
        error = (
            None if status == "cancelled" else "Verification ownership expired before completion."
        )
        updated_at = _next_updated_at(record["updatedAt"], now)
        evidence_revision = _next_evidence_revision(record["evidenceRevision"])
        history_sequence = _next_history_sequence_locked(connection)
        cursor = connection.execute(
            """
            UPDATE agent_verification_runs
            SET status = ?, terminal_hint = 'interrupted',
                recovered_after_restart = 1, error = ?, evidence_revision = ?,
                updated_at = ?, completed_at = ?, history_sequence = ?
            WHERE id = ? AND status = 'running' AND lease_expires_at <= ?
            """,
            (
                status,
                error,
                evidence_revision,
                updated_at,
                updated_at,
                history_sequence,
                record["id"],
                now,
            ),
        )
        if cursor.rowcount != 1:
            raise VerificationConflictError("Verification lease changed during reconciliation.")
        affected_projects.add(record["projectId"])
    for affected_project in affected_projects:
        _prune_runs_locked(connection, affected_project)
    return len(rows)


def _begin_write(connection: _VerificationConnection) -> None:
    read_key = _attest_connection(
        connection,
        expected_identity = connection.authority_key[1:3],
    )
    connection.commit()
    connection.execute("BEGIN IMMEDIATE")
    connection.authority_key = _attest_connection(
        connection,
        expected_identity = read_key[1:3],
    )


def _reconcile_project_if_needed(project_id: str, now: int) -> int:
    """Keep the normal read path lock-free and take writer authority only for expired leases."""
    connection = _connection()
    try:
        expired = connection.execute(
            """
            SELECT 1 FROM agent_verification_runs
            WHERE project_id = ? AND status = 'running' AND lease_expires_at <= ?
            LIMIT 1
            """,
            (project_id, now),
        ).fetchone()
    finally:
        connection.close()
    if expired is None:
        return 0

    connection = _connection()
    try:
        _begin_write(connection)
        locked_now = _now_ms()
        changed = _reconcile_expired_locked(
            connection,
            locked_now,
            project_id = project_id,
        )
        connection.commit()
        return changed
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def get_verification_config(project_id: str) -> dict:
    project_id = _validate_project_id(project_id)
    connection = _connection()
    try:
        row = _read_config_row(connection, project_id)
        return _config_from_row(row, project_id) if row is not None else _empty_config(project_id)
    finally:
        connection.close()


def set_verification_config(
    project_id: str,
    checks: list[dict],
    *,
    workspace_identity: tuple[int, int],
    workspace_revision: int,
    expected_revision: int,
) -> dict:
    project_id = _validate_project_id(project_id)
    normalized, encoded, config_hash = _encode_checks(checks)
    identity = _validate_workspace_identity(workspace_identity)
    workspace_revision = _validate_revision(workspace_revision, allow_zero = True)
    expected_revision = _validate_revision(expected_revision, allow_zero = True)
    connection = _connection()
    try:
        _begin_write(connection)
        now = _now_ms()
        _reconcile_expired_deletion_fences_locked(
            connection,
            now,
            project_id = project_id,
        )
        _reconcile_expired_locked(connection, now, project_id = project_id)
        archived = _project_archived_locked(connection, project_id)
        if archived is None:
            raise AgentWorkspaceError("Project does not exist.")
        if archived:
            raise VerificationConflictError("Archived projects cannot change verification.")
        if _active_deletion_fence_locked(connection, project_id) is not None:
            raise VerificationConflictError("Project retirement is in progress.")
        active = connection.execute(
            """
            SELECT 1 FROM agent_verification_runs
            WHERE project_id = ? AND status = 'running'
            """,
            (project_id,),
        ).fetchone()
        if active is not None:
            raise VerificationConflictError(
                "Verification settings are locked while a run is active."
            )
        row = _read_config_row(connection, project_id)
        current = (
            _config_from_row(row, project_id) if row is not None else _empty_config(project_id)
        )
        if current["revision"] != expected_revision:
            raise VerificationConflictError(
                f"Verification configuration revision is {current['revision']}, not {expected_revision}."
            )
        if row is not None and (
            current["checks"] == normalized
            and current["workspaceDeviceId"] == identity[0]
            and current["workspaceFileId"] == identity[1]
            and current["workspaceRevision"] == workspace_revision
            and current["configHash"] == config_hash
        ):
            connection.commit()
            return current
        revision = expected_revision + 1
        if revision > MAX_REVISION:
            raise VerificationConflictError("Verification configuration revision is exhausted.")
        updated_at = _now_ms()
        if row is None:
            connection.execute(
                """
                INSERT INTO agent_verification_configs (
                    project_id, workspace_device_id, workspace_file_id,
                    workspace_revision, checks_json, config_hash, revision, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    identity[0],
                    identity[1],
                    workspace_revision,
                    encoded,
                    config_hash,
                    revision,
                    updated_at,
                ),
            )
        else:
            cursor = connection.execute(
                """
                UPDATE agent_verification_configs
                SET workspace_device_id = ?, workspace_file_id = ?, workspace_revision = ?,
                    checks_json = ?, config_hash = ?, revision = ?, updated_at = ?
                WHERE project_id = ? AND revision = ?
                """,
                (
                    identity[0],
                    identity[1],
                    workspace_revision,
                    encoded,
                    config_hash,
                    revision,
                    updated_at,
                    project_id,
                    expected_revision,
                ),
            )
            if cursor.rowcount != 1:
                raise VerificationConflictError("Verification configuration changed concurrently.")
        stored = _read_config_row(connection, project_id)
        if stored is None:
            raise VerificationStateError("Verification configuration write was not persisted.")
        record = _config_from_row(stored, project_id)
        connection.commit()
        return record
    except sqlite3.IntegrityError as exc:
        connection.rollback()
        raise AgentWorkspaceError(
            "Project does not exist or verification configuration is invalid."
        ) from exc
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def verification_config_matches(
    project_id: str,
    *,
    revision: int,
    config_hash: str,
    workspace_identity: tuple[int, int],
    workspace_revision: int,
) -> bool:
    project_id = _validate_project_id(project_id)
    revision = _validate_revision(revision, allow_zero = False)
    config_hash = _validate_hash(config_hash)
    identity = _validate_workspace_identity(workspace_identity)
    workspace_revision = _validate_revision(workspace_revision, allow_zero = True)
    connection = _connection()
    try:
        row = _read_config_row(connection, project_id)
        if row is None:
            return False
        record = _config_from_row(row, project_id)
        return (
            record["revision"] == revision
            and record["configHash"] == config_hash
            and record["workspaceDeviceId"] == identity[0]
            and record["workspaceFileId"] == identity[1]
            and record["workspaceRevision"] == workspace_revision
        )
    finally:
        connection.close()


def begin_verification_project_deletion(project_id: str, fence_id: str) -> dict:
    project_id = _validate_project_id(project_id)
    fence_id = _validate_fence_id(fence_id)
    connection = _connection()
    try:
        _begin_write(connection)
        now = _now_ms()
        lease_expires_at = min(MAX_REVISION, now + DELETION_FENCE_LEASE_DURATION_MS)
        _reconcile_expired_deletion_fences_locked(
            connection,
            now,
            project_id = project_id,
        )
        current = _active_deletion_fence_locked(connection, project_id)
        if current is not None and current["fenceId"] != fence_id:
            raise VerificationConflictError("Project deletion belongs to another owner.")
        if current is None:
            tombstone = _deletion_fence_tombstone_locked(connection, project_id)
            if tombstone is None:
                connection.execute(
                    """
                    INSERT INTO agent_verification_project_deletions (
                        project_id, fence_id, active, revision,
                        created_at, heartbeat_at, lease_expires_at
                    ) VALUES (?, ?, 1, 1, ?, ?, ?)
                    """,
                    (project_id, fence_id, now, now, lease_expires_at),
                )
            else:
                revision = _next_deletion_fence_revision(tombstone["revision"])
                cursor = connection.execute(
                    """
                    UPDATE agent_verification_project_deletions
                    SET fence_id = ?, active = 1, revision = ?,
                        created_at = ?, heartbeat_at = ?, lease_expires_at = ?
                    WHERE project_id = ? AND active = 0 AND revision = ?
                    """,
                    (
                        fence_id,
                        revision,
                        now,
                        now,
                        lease_expires_at,
                        project_id,
                        tombstone["revision"],
                    ),
                )
                if cursor.rowcount != 1:
                    raise VerificationConflictError("Project deletion fence changed concurrently.")
        else:
            heartbeat_at, lease_expires_at = _next_lease_times(
                now,
                previous_heartbeat = current["heartbeatAt"],
                previous_expiry = current["leaseExpiresAt"],
                duration_ms = DELETION_FENCE_LEASE_DURATION_MS,
            )
            cursor = connection.execute(
                """
                UPDATE agent_verification_project_deletions
                SET heartbeat_at = ?, lease_expires_at = ?
                WHERE project_id = ? AND fence_id = ? AND active = 1 AND revision = ?
                """,
                (
                    heartbeat_at,
                    lease_expires_at,
                    project_id,
                    fence_id,
                    current["revision"],
                ),
            )
            if cursor.rowcount != 1:
                raise VerificationConflictError("Project deletion fence changed concurrently.")
        stored = _active_deletion_fence_locked(connection, project_id)
        if stored is None:
            raise VerificationStateError("Project deletion fence was not persisted.")
        connection.commit()
        return stored
    except sqlite3.IntegrityError as exc:
        connection.rollback()
        raise AgentWorkspaceError("Project does not exist or deletion state is invalid.") from exc
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def heartbeat_verification_project_deletion(project_id: str, fence_id: str, revision: int) -> dict:
    project_id = _validate_project_id(project_id)
    fence_id = _validate_fence_id(fence_id)
    revision = _validate_revision(revision, allow_zero = False)
    connection = _connection()
    try:
        _begin_write(connection)
        now = _now_ms()
        _reconcile_expired_deletion_fences_locked(
            connection,
            now,
            project_id = project_id,
        )
        current = _active_deletion_fence_locked(connection, project_id)
        if current is None or current["fenceId"] != fence_id or current["revision"] != revision:
            raise VerificationConflictError("Project deletion fence ownership expired or changed.")
        heartbeat_at, lease_expires_at = _next_lease_times(
            now,
            previous_heartbeat = current["heartbeatAt"],
            previous_expiry = current["leaseExpiresAt"],
            duration_ms = DELETION_FENCE_LEASE_DURATION_MS,
        )
        cursor = connection.execute(
            """
            UPDATE agent_verification_project_deletions
            SET heartbeat_at = ?, lease_expires_at = ?
            WHERE project_id = ? AND fence_id = ? AND active = 1 AND revision = ?
            """,
            (
                heartbeat_at,
                lease_expires_at,
                project_id,
                fence_id,
                current["revision"],
            ),
        )
        if cursor.rowcount != 1:
            raise VerificationConflictError("Project deletion fence changed concurrently.")
        stored = _active_deletion_fence_locked(connection, project_id)
        if stored is None:
            raise VerificationStateError("Project deletion heartbeat was not persisted.")
        connection.commit()
        return stored
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def finish_verification_project_deletion(project_id: str, fence_id: str, revision: int) -> bool:
    """Release an exact deletion fence, succeeding if project deletion cascaded it."""
    project_id = _validate_project_id(project_id)
    fence_id = _validate_fence_id(fence_id)
    revision = _validate_revision(revision, allow_zero = False)
    connection = _connection()
    try:
        _begin_write(connection)
        current = _active_deletion_fence_locked(connection, project_id)
        if current is None:
            connection.commit()
            return False
        if current["fenceId"] != fence_id or current["revision"] != revision:
            raise VerificationConflictError("Project deletion fence belongs to another owner.")
        cursor = connection.execute(
            """
            UPDATE agent_verification_project_deletions
            SET active = 0
            WHERE project_id = ? AND fence_id = ? AND active = 1 AND revision = ?
            """,
            (project_id, fence_id, revision),
        )
        if cursor.rowcount != 1:
            raise VerificationConflictError("Project deletion fence changed concurrently.")
        connection.commit()
        return True
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def begin_verification_run(
    project_id: str,
    *,
    owner_id: str,
    config_revision: int,
    config_hash: str,
    checks: list[dict],
    workspace_identity: tuple[int, int],
    workspace_revision: int,
) -> dict:
    project_id = _validate_project_id(project_id)
    owner_id = _validate_owner_id(owner_id)
    config_revision = _validate_revision(config_revision, allow_zero = False)
    config_hash = _validate_hash(config_hash)
    selected, selected_json, checks_hash = _encode_checks(checks)
    identity = _validate_workspace_identity(workspace_identity)
    workspace_revision = _validate_revision(workspace_revision, allow_zero = True)
    run_id = str(uuid.uuid4())
    connection = _connection()
    try:
        _begin_write(connection)
        started_at = _now_ms()
        lease_expires_at = min(MAX_REVISION, started_at + LEASE_DURATION_MS)
        _reconcile_expired_deletion_fences_locked(
            connection,
            started_at,
            project_id = project_id,
        )
        _reconcile_expired_locked(connection, started_at, project_id = project_id)
        archived = _project_archived_locked(connection, project_id)
        if archived is None:
            raise AgentWorkspaceError("Project does not exist.")
        if archived:
            raise VerificationConflictError("Archived projects cannot start verification.")
        if _active_deletion_fence_locked(connection, project_id) is not None:
            raise VerificationConflictError("Project deletion is in progress.")
        config_row = _read_config_row(connection, project_id)
        if config_row is None:
            raise VerificationConflictError("Verification configuration is unavailable.")
        config = _config_from_row(config_row, project_id)
        if (
            config["revision"] != config_revision
            or config["configHash"] != config_hash
            or config["workspaceDeviceId"] != identity[0]
            or config["workspaceFileId"] != identity[1]
            or config["workspaceRevision"] != workspace_revision
        ):
            raise VerificationConflictError(
                "Verification configuration or workspace changed before the run began."
            )
        if (
            connection.execute(
                """
            SELECT 1 FROM agent_verification_runs
            WHERE project_id = ? AND status = 'running'
            """,
                (project_id,),
            ).fetchone()
            is not None
        ):
            raise VerificationConflictError("Project verification is already running.")
        _validate_selected_checks(selected, config["checks"])
        config_json = _canonical_json(
            config["checks"],
            limit = MAX_CONFIG_BYTES,
            label = "Verification configuration",
        )
        connection.execute(
            """
            INSERT INTO agent_verification_runs (
                id, project_id, owner_id, status, terminal_hint,
                config_revision, config_hash,
                workspace_device_id, workspace_file_id, workspace_revision,
                config_checks_json, checks_json, checks_hash, results_json,
                evidence_revision, cancel_requested, recovered_after_restart, error,
                started_at, heartbeat_at, updated_at, lease_expires_at, completed_at,
                history_sequence
            ) VALUES (
                ?, ?, ?, 'running', NULL, ?, ?, ?, ?, ?, ?, ?, ?, '[]',
                1, 0, 0, NULL, ?, ?, ?, ?, NULL, NULL
            )
            """,
            (
                run_id,
                project_id,
                owner_id,
                config_revision,
                config_hash,
                identity[0],
                identity[1],
                workspace_revision,
                config_json,
                selected_json,
                checks_hash,
                started_at,
                started_at,
                started_at,
                lease_expires_at,
            ),
        )
        row = _read_run_row(connection, project_id, run_id)
        if row is None:
            raise VerificationStateError("Verification run write was not persisted.")
        record = _run_from_row(row)
        connection.commit()
        return record
    except sqlite3.IntegrityError as exc:
        connection.rollback()
        if "UNIQUE constraint failed" in str(exc):
            raise VerificationConflictError("Project verification is already running.") from exc
        raise AgentWorkspaceError("Project does not exist or verification run is invalid.") from exc
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def project_execution_may_start(project_id: str) -> bool:
    """Share the durable retirement gate with reviewed project hooks."""
    project_id = _validate_project_id(project_id)
    connection = _connection()
    try:
        _begin_write(connection)
        _reconcile_expired_deletion_fences_locked(connection, _now_ms(), project_id = project_id)
        allowed = (
            _project_archived_locked(connection, project_id) is False
            and _active_deletion_fence_locked(connection, project_id) is None
        )
        connection.commit()
        return allowed
    finally:
        connection.close()


def verification_run_may_spawn(
    project_id: str,
    run_id: str,
    owner_id: str,
    *,
    revision: int,
    config_hash: str,
    workspace_identity: tuple[int, int],
    workspace_revision: int,
) -> bool:
    """Revalidate every durable run capability immediately before process creation."""
    project_id = _validate_project_id(project_id)
    owner_id = _validate_owner_id(owner_id)
    revision = _validate_revision(revision, allow_zero = False)
    config_hash = _validate_hash(config_hash)
    identity = _validate_workspace_identity(workspace_identity)
    workspace_revision = _validate_revision(workspace_revision, allow_zero = True)
    connection = _connection()
    try:
        _begin_write(connection)
        now = _now_ms()
        _reconcile_expired_deletion_fences_locked(
            connection,
            now,
            project_id = project_id,
        )
        _reconcile_expired_locked(connection, now, project_id = project_id)
        archived = _project_archived_locked(connection, project_id)
        if archived is None or archived:
            connection.commit()
            return False
        if _active_deletion_fence_locked(connection, project_id) is not None:
            connection.commit()
            return False
        row = _read_run_row(connection, project_id, run_id)
        if row is None:
            connection.commit()
            return False
        run = _run_from_row(row)
        if (
            run["status"] != "running"
            or run["ownerId"] != owner_id
            or run["cancelRequested"]
            or run["configRevision"] != revision
            or run["configHash"] != config_hash
            or run["workspaceDeviceId"] != identity[0]
            or run["workspaceFileId"] != identity[1]
            or run["workspaceRevision"] != workspace_revision
        ):
            connection.commit()
            return False
        config_row = _read_config_row(connection, project_id)
        if config_row is None:
            connection.commit()
            return False
        config = _config_from_row(config_row, project_id)
        if (
            config["revision"] != revision
            or config["configHash"] != config_hash
            or config["workspaceDeviceId"] != identity[0]
            or config["workspaceFileId"] != identity[1]
            or config["workspaceRevision"] != workspace_revision
        ):
            connection.commit()
            return False
        heartbeat_at, lease_expires_at = _next_lease_times(
            now,
            previous_heartbeat = run["heartbeatAt"],
            previous_expiry = run["leaseExpiresAt"],
            duration_ms = LEASE_DURATION_MS,
        )
        cursor = connection.execute(
            """
            UPDATE agent_verification_runs
            SET heartbeat_at = ?, lease_expires_at = ?
            WHERE id = ? AND project_id = ? AND owner_id = ?
                AND status = 'running' AND cancel_requested = 0
            """,
            (heartbeat_at, lease_expires_at, run_id, project_id, owner_id),
        )
        if cursor.rowcount != 1:
            connection.rollback()
            return False
        connection.commit()
        return True
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _owned_running_row(
    connection: sqlite3.Connection, project_id: str, run_id: str, owner_id: str, now: int
) -> sqlite3.Row:
    _reconcile_expired_locked(connection, now, project_id = project_id)
    row = _read_run_row(connection, project_id, run_id)
    if row is None:
        raise VerificationConflictError("Verification run was not found.")
    record = _run_from_row(row)
    if record["status"] != "running":
        raise VerificationConflictError("Verification run is no longer active.")
    if record["ownerId"] != owner_id:
        raise VerificationConflictError("Verification run belongs to another owner.")
    return row


def update_verification_run_progress(
    project_id: str, run_id: str, results: list[dict], *, owner_id: str
) -> dict:
    project_id = _validate_project_id(project_id)
    owner_id = _validate_owner_id(owner_id)
    connection = _connection()
    try:
        _begin_write(connection)
        now = _now_ms()
        row = _owned_running_row(connection, project_id, run_id, owner_id, now)
        current = _run_from_row(row)
        normalized = _validate_results(
            results,
            current["checks"],
            run_started_at = current["startedAt"],
            allow_running_tail = True,
        )
        _validate_progress_transition(current["results"], normalized)
        encoded = _encode_results(normalized)
        heartbeat_at, lease_expires_at = _next_lease_times(
            now,
            previous_heartbeat = current["heartbeatAt"],
            previous_expiry = current["leaseExpiresAt"],
            duration_ms = LEASE_DURATION_MS,
        )
        updated_at = _next_updated_at(current["updatedAt"], now)
        updated_at = max(
            updated_at,
            max(
                (
                    result["completedAt"]
                    for result in normalized
                    if result["completedAt"] is not None
                ),
                default = updated_at,
            ),
        )
        evidence_revision = _next_evidence_revision(current["evidenceRevision"])
        cursor = connection.execute(
            """
            UPDATE agent_verification_runs
            SET results_json = ?, evidence_revision = ?, heartbeat_at = ?,
                updated_at = ?, lease_expires_at = ?
            WHERE id = ? AND project_id = ? AND owner_id = ? AND status = 'running'
            """,
            (
                encoded,
                evidence_revision,
                heartbeat_at,
                updated_at,
                lease_expires_at,
                run_id,
                project_id,
                owner_id,
            ),
        )
        if cursor.rowcount != 1:
            raise VerificationConflictError("Verification progress ownership changed.")
        updated = _read_run_row(connection, project_id, run_id)
        if updated is None:
            raise VerificationStateError("Verification progress write was not persisted.")
        record = _run_from_row(updated)
        connection.commit()
        return record
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def heartbeat_verification_run(project_id: str, run_id: str, owner_id: str) -> bool:
    project_id = _validate_project_id(project_id)
    owner_id = _validate_owner_id(owner_id)
    connection = _connection()
    try:
        _begin_write(connection)
        now = _now_ms()
        _reconcile_expired_locked(connection, now, project_id = project_id)
        row = connection.execute(
            """
            SELECT id, project_id, owner_id, status, cancel_requested,
                   evidence_revision, started_at, heartbeat_at, updated_at,
                   lease_expires_at, history_sequence
            FROM agent_verification_runs
            WHERE project_id = ? AND id = ?
            """,
            (project_id, run_id),
        ).fetchone()
        if row is None:
            raise VerificationConflictError("Verification run was not found.")
        record = _owned_run_metadata_from_row(row)
        if record["ownerId"] != owner_id:
            raise VerificationConflictError("Verification run belongs to another owner.")
        heartbeat_at, lease_expires_at = _next_lease_times(
            now,
            previous_heartbeat = record["heartbeatAt"],
            previous_expiry = record["leaseExpiresAt"],
            duration_ms = LEASE_DURATION_MS,
        )
        cursor = connection.execute(
            """
            UPDATE agent_verification_runs
            SET heartbeat_at = ?, lease_expires_at = ?
            WHERE id = ? AND project_id = ? AND owner_id = ? AND status = 'running'
            """,
            (heartbeat_at, lease_expires_at, run_id, project_id, owner_id),
        )
        if cursor.rowcount != 1:
            raise VerificationConflictError("Verification heartbeat ownership changed.")
        connection.commit()
        return record["cancelRequested"]
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def request_verification_cancel(project_id: str, run_id: str) -> tuple[dict, bool]:
    project_id = _validate_project_id(project_id)
    connection = _connection()
    try:
        _begin_write(connection)
        now = _now_ms()
        _reconcile_expired_locked(connection, now, project_id = project_id)
        row = _read_run_row(connection, project_id, run_id)
        if row is None:
            raise VerificationConflictError("Verification run was not found.")
        record = _run_from_row(row)
        if record["status"] != "running":
            connection.commit()
            return record, False
        updated_at = _next_updated_at(record["updatedAt"], now)
        evidence_revision = _next_evidence_revision(record["evidenceRevision"])
        connection.execute(
            """
            UPDATE agent_verification_runs
            SET cancel_requested = 1, evidence_revision = ?, updated_at = ?
            WHERE id = ? AND project_id = ? AND status = 'running'
            """,
            (evidence_revision, updated_at, run_id, project_id),
        )
        updated = _read_run_row(connection, project_id, run_id)
        if updated is None:
            raise VerificationStateError("Verification cancellation was not persisted.")
        cancelled = _run_from_row(updated)
        connection.commit()
        return cancelled, True
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def request_project_verification_cancel(project_id: str) -> tuple[Optional[dict], bool]:
    project_id = _validate_project_id(project_id)
    connection = _connection()
    try:
        _begin_write(connection)
        now = _now_ms()
        _reconcile_expired_locked(connection, now, project_id = project_id)
        row = connection.execute(
            """
            SELECT id, project_id, status, cancel_requested, evidence_revision,
                   started_at, heartbeat_at, updated_at, lease_expires_at,
                   history_sequence
            FROM agent_verification_runs
            WHERE project_id = ? AND status = 'running'
            """,
            (project_id,),
        ).fetchone()
        if row is None:
            connection.commit()
            return None, False
        record = _expired_run_metadata_from_row(row)
        updated_at = _next_updated_at(record["updatedAt"], now)
        evidence_revision = _next_evidence_revision(record["evidenceRevision"])
        cursor = connection.execute(
            """
            UPDATE agent_verification_runs
            SET cancel_requested = 1, evidence_revision = ?, updated_at = ?
            WHERE id = ? AND project_id = ? AND status = 'running'
            """,
            (evidence_revision, updated_at, record["id"], project_id),
        )
        if cursor.rowcount != 1:
            raise VerificationConflictError("Project verification changed during cancellation.")
        updated = connection.execute(
            """
            SELECT id, project_id, status, cancel_requested, evidence_revision,
                   started_at, heartbeat_at, updated_at, lease_expires_at,
                   history_sequence
            FROM agent_verification_runs
            WHERE project_id = ? AND id = ? AND status = 'running'
            """,
            (project_id, record["id"]),
        ).fetchone()
        if updated is None:
            raise VerificationStateError("Project verification cancellation was not persisted.")
        cancelled = _expired_run_metadata_from_row(updated) | {"status": "running"}
        connection.commit()
        return cancelled, True
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def complete_verification_run(
    project_id: str,
    run_id: str,
    results: list[dict],
    *,
    owner_id: str,
    terminal_status: Optional[str] = None,
    error: Optional[str] = None,
) -> dict:
    project_id = _validate_project_id(project_id)
    owner_id = _validate_owner_id(owner_id)
    if terminal_status is not None and terminal_status not in _TERMINAL_HINTS:
        raise AgentWorkspaceError("Verification terminal status hint is invalid.")
    safe_error = _validate_error(error)
    connection = _connection()
    try:
        _begin_write(connection)
        now = _now_ms()
        row = _owned_running_row(connection, project_id, run_id, owner_id, now)
        current = _run_from_row(row)
        normalized = _validate_results(
            results,
            current["checks"],
            run_started_at = current["startedAt"],
            allow_running_tail = False,
        )
        _validate_progress_transition(current["results"], normalized)
        status = _derive_terminal_status(
            current["checks"],
            normalized,
            cancel_requested = current["cancelRequested"],
            terminal_hint = terminal_status,
        )
        cancel_requested = current["cancelRequested"] or status == "cancelled"
        if status in {"passed", "cancelled"}:
            safe_error = None
        encoded = _encode_results(normalized)
        updated_at = _next_updated_at(current["updatedAt"], now)
        updated_at = max(
            updated_at,
            max(
                (
                    result["completedAt"]
                    for result in normalized
                    if result["completedAt"] is not None
                ),
                default = updated_at,
            ),
        )
        evidence_revision = _next_evidence_revision(current["evidenceRevision"])
        history_sequence = _next_history_sequence_locked(connection)
        cursor = connection.execute(
            """
            UPDATE agent_verification_runs
            SET status = ?, terminal_hint = ?, results_json = ?, cancel_requested = ?,
                error = ?, evidence_revision = ?, updated_at = ?, completed_at = ?,
                history_sequence = ?
            WHERE id = ? AND project_id = ? AND owner_id = ? AND status = 'running'
            """,
            (
                status,
                terminal_status,
                encoded,
                int(cancel_requested),
                safe_error,
                evidence_revision,
                updated_at,
                updated_at,
                history_sequence,
                run_id,
                project_id,
                owner_id,
            ),
        )
        if cursor.rowcount != 1:
            raise VerificationConflictError("Verification completion ownership changed.")
        _prune_runs_locked(connection, project_id)
        updated = _read_run_row(connection, project_id, run_id)
        if updated is None:
            raise VerificationStateError("Verification completion was not persisted.")
        record = _run_from_row(updated)
        connection.commit()
        return record
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _read_after_reconciliation(
    project_id: str, query: str, parameters: tuple[Any, ...]
) -> list[dict]:
    now = _now_ms()
    _reconcile_project_if_needed(project_id, now)
    connection = _connection()
    try:
        rows = connection.execute(query, parameters).fetchall()
    finally:
        connection.close()
    return [_run_from_row(row) for row in rows]


def get_verification_run(project_id: str, run_id: str) -> Optional[dict]:
    project_id = _validate_project_id(project_id)
    records = _read_after_reconciliation(
        project_id,
        "SELECT * FROM agent_verification_runs WHERE project_id = ? AND id = ?",
        (project_id, run_id),
    )
    return records[0] if records else None


def get_verification_run_evidence_revision(project_id: str, run_id: str) -> Optional[int]:
    """Read the lightweight evidence marker without decoding result bodies."""
    project_id = _validate_project_id(project_id)
    now = _now_ms()
    _reconcile_project_if_needed(project_id, now)
    connection = _connection()
    try:
        row = connection.execute(
            """
            SELECT project_id, evidence_revision FROM agent_verification_runs
            WHERE project_id = ? AND id = ?
            """,
            (project_id, run_id),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    try:
        if _validate_project_id(row["project_id"]) != project_id:
            raise VerificationStateError("Persisted verification evidence marker is invalid.")
        return _validate_revision(row["evidence_revision"], allow_zero = False)
    except AgentWorkspaceError as exc:
        if isinstance(exc, VerificationStateError):
            raise
        raise VerificationStateError("Persisted verification evidence marker is invalid.") from exc


def list_verification_runs(project_id: str, limit: int = 20) -> list[dict]:
    project_id = _validate_project_id(project_id)
    limit = _validate_integer(limit, label = "Verification history limit", minimum = 1)
    bounded_limit = min(limit, MAX_RUN_HISTORY)
    return _read_after_reconciliation(
        project_id,
        """
        SELECT * FROM agent_verification_runs
        WHERE project_id = ?
        ORDER BY (status = 'running') DESC, history_sequence DESC
        LIMIT ?
        """,
        (project_id, bounded_limit),
    )


def list_verification_run_summaries(project_id: str, limit: int = 20) -> list[dict]:
    """Read bounded history metadata without loading checks or output evidence."""
    project_id = _validate_project_id(project_id)
    limit = _validate_integer(limit, label = "Verification history limit", minimum = 1)
    bounded_limit = min(limit, MAX_RUN_HISTORY)
    now = _now_ms()
    _reconcile_project_if_needed(project_id, now)
    connection = _connection()
    try:
        rows = connection.execute(
            """
            SELECT id, project_id, status, config_revision, workspace_revision,
                   evidence_revision, cancel_requested, error,
                   started_at, updated_at, completed_at, history_sequence
            FROM agent_verification_runs
            WHERE project_id = ?
            ORDER BY (status = 'running') DESC, history_sequence DESC
            LIMIT ?
            """,
            (project_id, bounded_limit),
        ).fetchall()
    finally:
        connection.close()
    return [_run_summary_from_row(row) for row in rows]


def active_verification_run(project_id: str) -> Optional[dict]:
    project_id = _validate_project_id(project_id)
    records = _read_after_reconciliation(
        project_id,
        """
        SELECT * FROM agent_verification_runs
        WHERE project_id = ? AND status = 'running'
        """,
        (project_id,),
    )
    return records[0] if records else None


def active_verification_run_lifecycle(project_id: str) -> Optional[dict]:
    """Read only bounded active lifecycle metadata for retirement polling."""
    project_id = _validate_project_id(project_id)
    now = _now_ms()
    _reconcile_project_if_needed(project_id, now)
    connection = _connection()
    try:
        row = connection.execute(
            """
            SELECT id, project_id, status, cancel_requested, evidence_revision,
                   started_at, heartbeat_at, updated_at, lease_expires_at,
                   history_sequence
            FROM agent_verification_runs
            WHERE project_id = ? AND status = 'running'
            """,
            (project_id,),
        ).fetchone()
    finally:
        connection.close()
    if row is None:
        return None
    record = _expired_run_metadata_from_row(row)
    return record | {"status": "running"}


def reconcile_interrupted_verification_runs() -> int:
    """Settle only expired verification leases, preserving their progress evidence."""
    connection = _connection()
    try:
        _begin_write(connection)
        now = _now_ms()
        _reconcile_expired_deletion_fences_locked(connection, now)
        changed = _reconcile_expired_locked(connection, now)
        connection.commit()
        return changed
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


__all__ = [
    "DELETION_FENCE_HEARTBEAT_INTERVAL_SECONDS",
    "DELETION_FENCE_LEASE_DURATION_MS",
    "HEARTBEAT_INTERVAL_SECONDS",
    "LEASE_DURATION_MS",
    "MAX_CHECKS",
    "MAX_CHECK_COMMAND_BYTES",
    "MAX_CONFIG_BYTES",
    "MAX_GLOBAL_RUN_HISTORY_BYTES",
    "MAX_HISTORY_SEQUENCE",
    "MAX_LOG_LIMIT_BYTES",
    "MAX_PROJECT_RUN_HISTORY_BYTES",
    "MAX_RUN_HISTORY",
    "MAX_RUN_RESULT_BYTES",
    "MAX_RUN_RESULTS_JSON_BYTES",
    "MAX_TRUNCATION_DECORATION_BYTES",
    "MAX_TIMEOUT_SECONDS",
    "SOURCE_FRESHNESS",
    "VerificationConflictError",
    "VerificationStateError",
    "active_verification_run",
    "active_verification_run_lifecycle",
    "begin_verification_project_deletion",
    "begin_verification_run",
    "complete_verification_run",
    "get_verification_config",
    "get_verification_run",
    "get_verification_run_evidence_revision",
    "heartbeat_verification_project_deletion",
    "heartbeat_verification_run",
    "list_verification_runs",
    "list_verification_run_summaries",
    "normalize_verification_checks",
    "reconcile_interrupted_verification_runs",
    "finish_verification_project_deletion",
    "request_project_verification_cancel",
    "request_verification_cancel",
    "set_verification_config",
    "update_verification_run_progress",
    "verification_config_matches",
    "verification_run_may_spawn",
]
