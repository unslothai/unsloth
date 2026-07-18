# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""SQLite storage for training run history and metrics.

Like auth/storage.py (module-level functions, raw sqlite3, per-function
connections) plus WAL mode and PRAGMA foreign_keys = ON for CASCADE deletes.
"""

import json
import logging
import os
import platform
import re
import shutil
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)
from typing import Any, Iterable, Optional


from utils.paths import (
    ensure_dir,
    project_workspaces_root,
    studio_db_path,
)
from utils.paths.external_media import is_linux_run_media_path, is_local_filesystem_root
from utils.paths.sensitive import (
    contains_sensitive_path_component as _shared_contains_sensitive_path_component,
)
from utils.training_runs import extract_project_name


def _extract_project_name_from_config_json(config_json: Optional[str]) -> Optional[str]:
    if not config_json:
        return None
    try:
        return extract_project_name(json.loads(config_json))
    except (json.JSONDecodeError, TypeError):
        return None


def _denied_path_prefixes() -> list[str]:
    """Platform-aware denylist of system directories."""
    system = platform.system()
    if system == "Linux":
        return ["/proc", "/sys", "/dev", "/etc", "/boot", "/run"]
    if system == "Darwin":
        # macOS realpath() resolves /etc -> /private/etc etc; include /private variants.
        return [
            "/System",
            "/Library",
            "/dev",
            "/etc",
            "/private/etc",
            "/tmp",
            "/private/tmp",
            "/var",
            "/private/var",
        ]
    if system == "Windows":
        win = os.environ.get("SystemRoot", r"C:\Windows")
        pf = os.environ.get("ProgramFiles", r"C:\Program Files")
        pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        return [os.path.normcase(p) for p in [win, pf, pf86]]
    return []


def is_denied_system_path(path: str) -> bool:
    """True if *path* is, or descends from, a denied system directory.

    Mirrors the denylist add_scan_folder() enforces at registration so the
    browser refuses /etc, /proc, C:\\Windows, etc. even when the allowlist holds
    a broad root (a Windows drive root C:\\ or a legacy-registered / root). The
    /run carve-out keeps Linux removable-media mounts browseable. Expects an
    already-resolved (realpath) path so symlinks cannot escape into a denied subtree.
    """
    is_win = platform.system() == "Windows"
    check = os.path.normcase(path) if is_win else path
    for prefix in _denied_path_prefixes():
        if check == prefix or check.startswith(prefix + os.sep):
            if prefix == "/run" and is_linux_run_media_path(check):
                continue
            return True
    return False


def _contains_sensitive_path_component(path: str) -> bool:
    return _shared_contains_sensitive_path_component(path)


def contains_sensitive_path_component(path: str) -> bool:
    return _contains_sensitive_path_component(path)


_schema_lock = threading.Lock()
_schema_ready = False
_SQLITE_IN_CHUNK_SIZE = 900
_PROJECT_WORKSPACE_SUBDIRS = ("sandbox",)


def _project_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip(".-_")
    return slug[:48] or "project"


def _default_project_root(project: dict) -> str:
    project_id = str(project["id"])
    suffix = re.sub(r"[^A-Za-z0-9_-]+", "-", project_id)[:8].strip("-_") or "project"
    folder_name = f"{_project_slug(str(project.get('name') or 'Project'))}-{suffix}"
    return str(project_workspaces_root() / folder_name)


def _ensure_project_workspace(root_path: str) -> str:
    root = Path(root_path).expanduser()
    root_resolved = ensure_dir(root).resolve()
    for subdir in _PROJECT_WORKSPACE_SUBDIRS:
        ensure_dir(root_resolved / subdir)
    return str(root_resolved)


def _delete_project_workspace(project: dict) -> None:
    root_path = project.get("rootPath")
    if not root_path:
        return
    root = Path(root_path).expanduser()
    try:
        root_resolved = root.resolve(strict = False)
    except (OSError, RuntimeError, ValueError):
        logger.warning("Skipping project workspace delete for invalid path %r", root_path)
        return

    project_id = str(project["id"])
    suffix = re.sub(r"[^A-Za-z0-9_-]+", "-", project_id)[:8].strip("-_") or "project"
    if not root_resolved.name.endswith(f"-{suffix}"):
        logger.warning(
            "Skipping project workspace delete for unexpected project path %s",
            root_resolved,
        )
        return
    if root_resolved.parent == root_resolved or root_resolved == Path.home().resolve():
        logger.warning(
            "Skipping project workspace delete for unsafe project path %s",
            root_resolved,
        )
        return
    check = (
        os.path.normcase(str(root_resolved))
        if platform.system() == "Windows"
        else str(root_resolved)
    )
    for prefix in _denied_path_prefixes():
        if check == prefix or check.startswith(prefix + os.sep):
            logger.warning(
                "Skipping project workspace delete under denied path %s",
                root_resolved,
            )
            return
    if not root_resolved.exists():
        return
    if root_resolved.is_symlink() or not root_resolved.is_dir():
        logger.warning(
            "Skipping project workspace delete for non-directory path %s",
            root_resolved,
        )
        return
    shutil.rmtree(root_resolved)


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables and indexes if they don't exist. Called once per process."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS training_runs (
            id TEXT NOT NULL PRIMARY KEY,
            status TEXT NOT NULL DEFAULT 'running',
            model_name TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            config_json TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            total_steps INTEGER,
            final_step INTEGER,
            final_loss REAL,
            output_dir TEXT,
            error_message TEXT,
            duration_seconds REAL,
            loss_sparkline TEXT,
            display_name TEXT
        )
        """
    )
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(training_runs)").fetchall()}
    if "display_name" not in existing_cols:
        conn.execute("ALTER TABLE training_runs ADD COLUMN display_name TEXT")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS training_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
            step INTEGER NOT NULL,
            loss REAL,
            learning_rate REAL,
            grad_norm REAL,
            eval_loss REAL,
            epoch REAL,
            num_tokens INTEGER,
            elapsed_seconds REAL,
            UNIQUE(run_id, step)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_run_id ON training_metrics(run_id)")
    # Windows: COLLATE NOCASE so C:\Models and c:\models dedup. Elsewhere keep
    # case-sensitive BINARY so /Models and /models stay distinct.
    collation = "COLLATE NOCASE" if platform.system() == "Windows" else ""
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS scan_folders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT NOT NULL UNIQUE {collation},
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_projects (
            id TEXT NOT NULL PRIMARY KEY,
            name TEXT NOT NULL,
            instructions TEXT,
            root_path TEXT,
            archived INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    chat_project_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(chat_projects)").fetchall()
    }
    if "root_path" not in chat_project_cols:
        conn.execute("ALTER TABLE chat_projects ADD COLUMN root_path TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_projects_archived_updated_at ON chat_projects(archived, updated_at)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_threads (
            id TEXT NOT NULL PRIMARY KEY,
            title TEXT NOT NULL,
            model_type TEXT NOT NULL,
            model_id TEXT,
            pair_id TEXT,
            project_id TEXT,
            archived INTEGER NOT NULL DEFAULT 0,
            created_at INTEGER NOT NULL,
            updated_at INTEGER,
            openai_code_exec_container_id TEXT,
            anthropic_code_exec_container_id TEXT,
            forked_from_thread_id TEXT,
            forked_from_message_id TEXT,
            FOREIGN KEY(project_id) REFERENCES chat_projects(id) ON DELETE CASCADE
        )
        """
    )
    chat_thread_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(chat_threads)").fetchall()
    }
    if "project_id" not in chat_thread_cols:
        conn.execute("ALTER TABLE chat_threads ADD COLUMN project_id TEXT")
    if "openai_code_exec_container_id" not in chat_thread_cols:
        conn.execute("ALTER TABLE chat_threads ADD COLUMN openai_code_exec_container_id TEXT")
    if "anthropic_code_exec_container_id" not in chat_thread_cols:
        conn.execute("ALTER TABLE chat_threads ADD COLUMN anthropic_code_exec_container_id TEXT")
    if "forked_from_thread_id" not in chat_thread_cols:
        conn.execute("ALTER TABLE chat_threads ADD COLUMN forked_from_thread_id TEXT")
    if "forked_from_message_id" not in chat_thread_cols:
        conn.execute("ALTER TABLE chat_threads ADD COLUMN forked_from_message_id TEXT")
    if "updated_at" not in chat_thread_cols:
        conn.execute("ALTER TABLE chat_threads ADD COLUMN updated_at INTEGER")
        # Floor at created_at: forked threads copy older ancestor messages,
        # so the fork's creation time must win over the branch message times.
        conn.execute(
            """
            UPDATE chat_threads SET updated_at = MAX(
                COALESCE(
                    (
                        SELECT MAX(m.created_at) FROM chat_messages m
                        WHERE m.thread_id = chat_threads.id
                    ),
                    created_at
                ),
                created_at
            )
            """
        )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_messages (
            id TEXT NOT NULL PRIMARY KEY,
            thread_id TEXT NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
            parent_id TEXT,
            role TEXT NOT NULL,
            content_json TEXT NOT NULL,
            attachments_json TEXT,
            metadata_json TEXT,
            created_at INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_threads_model_type_created_at ON chat_threads(model_type, created_at)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_threads_pair_id ON chat_threads(pair_id)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_threads_project_id ON chat_threads(project_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_chat_messages_thread_id_created_at ON chat_messages(thread_id, created_at)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_settings (
            key TEXT NOT NULL PRIMARY KEY,
            value_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT NOT NULL PRIMARY KEY,
            value_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_settings_quarantine (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT NOT NULL,
            value_json TEXT NOT NULL,
            reason TEXT NOT NULL,
            quarantined_at TEXT NOT NULL
        )
        """
    )
    # Import ledger inside studio.db (vs. a localStorage boolean) so a db wipe
    # re-triggers the legacy Dexie import instead of silently hiding threads.
    # Keyed by legacy thread id; Dexie is read-only so per-thread suffices.
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chat_legacy_imports (
            legacy_thread_id TEXT NOT NULL PRIMARY KEY,
            imported_at INTEGER NOT NULL
        ) WITHOUT ROWID
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prompt_entries (
            id TEXT NOT NULL PRIMARY KEY,
            name TEXT NOT NULL,
            text TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prompt_entries_created_at ON prompt_entries(created_at)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prompt_lists (
            id TEXT NOT NULL PRIMARY KEY,
            name TEXT NOT NULL,
            items_json TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_prompt_lists_created_at ON prompt_lists(created_at)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS research_runs (
            id TEXT NOT NULL PRIMARY KEY,
            owner_subject TEXT NOT NULL,
            thread_id TEXT NOT NULL REFERENCES chat_threads(id) ON DELETE CASCADE,
            user_message_id TEXT NOT NULL REFERENCES chat_messages(id) ON DELETE CASCADE,
            assistant_message_id TEXT REFERENCES chat_messages(id) ON DELETE SET NULL,
            status TEXT NOT NULL CHECK(status IN (
                'planning', 'awaiting_approval', 'queued', 'running', 'paused',
                'cancelling', 'cancelled', 'completed', 'failed'
            )),
            plan_json TEXT,
            plan_revision INTEGER NOT NULL DEFAULT 0,
            plan_hash TEXT,
            config_json TEXT NOT NULL,
            cancel_requested INTEGER NOT NULL DEFAULT 0,
            lease_owner TEXT,
            lease_expires_at INTEGER,
            heartbeat_at INTEGER,
            retry_count INTEGER NOT NULL DEFAULT 0,
            error_message TEXT,
            report_text TEXT,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            next_event_seq INTEGER NOT NULL DEFAULT 1
        )
        """
    )
    research_run_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(research_runs)").fetchall()
    }
    if "report_text" not in research_run_cols:
        conn.execute("ALTER TABLE research_runs ADD COLUMN report_text TEXT")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS research_thread_claims (
            owner_subject TEXT NOT NULL,
            thread_id TEXT NOT NULL PRIMARY KEY REFERENCES chat_threads(id) ON DELETE CASCADE,
            created_at INTEGER NOT NULL
        ) WITHOUT ROWID
        """
    )
    claim_pk = [
        row[1]
        for row in sorted(
            conn.execute("PRAGMA table_info(research_thread_claims)").fetchall(),
            key = lambda row: int(row[5] or 0),
        )
        if int(row[5] or 0) > 0
    ]
    if claim_pk != ["thread_id"]:
        conn.execute("ALTER TABLE research_thread_claims RENAME TO research_thread_claims_legacy")
        conn.execute(
            """
            CREATE TABLE research_thread_claims (
                owner_subject TEXT NOT NULL,
                thread_id TEXT NOT NULL PRIMARY KEY REFERENCES chat_threads(id) ON DELETE CASCADE,
                created_at INTEGER NOT NULL
            ) WITHOUT ROWID
            """
        )
        conn.execute(
            """INSERT OR IGNORE INTO research_thread_claims
               (owner_subject, thread_id, created_at)
               SELECT owner_subject, thread_id, created_at
               FROM research_thread_claims_legacy
               ORDER BY created_at, owner_subject"""
        )
        conn.execute("DROP TABLE research_thread_claims_legacy")
    conn.execute(
        """INSERT OR IGNORE INTO research_thread_claims
           (owner_subject, thread_id, created_at)
           SELECT owner_subject, thread_id, created_at
           FROM research_runs ORDER BY created_at, id"""
    )
    conn.execute(
        """UPDATE research_runs
           SET status='failed', error_message='Superseded by the global thread research claim',
               lease_owner=NULL, lease_expires_at=NULL, completed_at=COALESCE(completed_at, updated_at)
           WHERE status IN ('planning','awaiting_approval','queued','running','paused','cancelling')
             AND EXISTS (
                 SELECT 1 FROM research_thread_claims c
                 WHERE c.thread_id=research_runs.thread_id
                   AND c.owner_subject<>research_runs.owner_subject
             )"""
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS research_plan_steps (
            run_id TEXT NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
            position INTEGER NOT NULL,
            title TEXT NOT NULL,
            query TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            result_json TEXT,
            started_at INTEGER,
            completed_at INTEGER,
            PRIMARY KEY(run_id, position)
        ) WITHOUT ROWID
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS research_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
            step_position INTEGER,
            url TEXT NOT NULL,
            title TEXT,
            snippet TEXT,
            fetched_at INTEGER NOT NULL,
            UNIQUE(run_id, url)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS research_document_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
            step_position INTEGER,
            source_key TEXT NOT NULL,
            document_id TEXT,
            chunk_id TEXT,
            filename TEXT NOT NULL,
            page INTEGER,
            score REAL,
            snippet TEXT,
            fetched_at INTEGER NOT NULL,
            UNIQUE(run_id, source_key)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS research_events (
            run_id TEXT NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
            seq INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            data_json TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            PRIMARY KEY(run_id, seq)
        ) WITHOUT ROWID
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_research_runs_owner_thread_status "
        "ON research_runs(owner_subject, thread_id, status)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_research_runs_lease "
        "ON research_runs(status, lease_expires_at)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_research_sources_run ON research_sources(run_id, id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_research_document_sources_run "
        "ON research_document_sources(run_id, id)"
    )


def _prompt_entry_from_row(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "text": row["text"],
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def list_prompt_entries() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM prompt_entries ORDER BY created_at DESC").fetchall()
        return [_prompt_entry_from_row(r) for r in rows]
    finally:
        conn.close()


def upsert_prompt_entry(entry: dict) -> dict:
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO prompt_entries (id, name, text, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                text = excluded.text,
                updated_at = excluded.updated_at
            """,
            (
                entry["id"],
                entry["name"],
                entry["text"],
                entry["createdAt"],
                entry["updatedAt"],
            ),
        )
        conn.commit()
        return entry
    finally:
        conn.close()


def delete_prompt_entry(entry_id: str) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM prompt_entries WHERE id = ?", (entry_id,))
        conn.commit()
    finally:
        conn.close()


def bulk_upsert_prompt_entries(entries: list[dict]) -> int:
    if not entries:
        return 0
    conn = get_connection()
    try:
        conn.executemany(
            """
            INSERT INTO prompt_entries (id, name, text, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                text = excluded.text,
                updated_at = excluded.updated_at
            """,
            [(e["id"], e["name"], e["text"], e["createdAt"], e["updatedAt"]) for e in entries],
        )
        conn.commit()
        return len(entries)
    finally:
        conn.close()


def _prompt_list_from_row(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "items": json.loads(row["items_json"]),
        "createdAt": row["created_at"],
        "updatedAt": row["updated_at"],
    }


def list_prompt_lists_db() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT * FROM prompt_lists ORDER BY created_at DESC").fetchall()
        return [_prompt_list_from_row(r) for r in rows]
    finally:
        conn.close()


def upsert_prompt_list(lst: dict) -> dict:
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO prompt_lists (id, name, items_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                items_json = excluded.items_json,
                updated_at = excluded.updated_at
            """,
            (
                lst["id"],
                lst["name"],
                json.dumps(lst["items"]),
                lst["createdAt"],
                lst["updatedAt"],
            ),
        )
        conn.commit()
        return lst
    finally:
        conn.close()


def delete_prompt_list_db(list_id: str) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM prompt_lists WHERE id = ?", (list_id,))
        conn.commit()
    finally:
        conn.close()


def bulk_upsert_prompt_lists(lists: list[dict]) -> int:
    if not lists:
        return 0
    conn = get_connection()
    try:
        conn.executemany(
            """
            INSERT INTO prompt_lists (id, name, items_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                items_json = excluded.items_json,
                updated_at = excluded.updated_at
            """,
            [
                (
                    lst["id"],
                    lst["name"],
                    json.dumps(lst["items"]),
                    lst["createdAt"],
                    lst["updatedAt"],
                )
                for lst in lists
            ],
        )
        conn.commit()
        return len(lists)
    finally:
        conn.close()


def get_connection() -> sqlite3.Connection:
    """Open studio.db with WAL mode, create tables once per process, enable foreign keys."""
    global _schema_ready
    db_path = studio_db_path()
    ensure_dir(db_path.parent)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    # foreign_keys is session-scoped; set per connection
    conn.execute("PRAGMA foreign_keys=ON")
    if not _schema_ready:
        with _schema_lock:
            if not _schema_ready:
                try:
                    _ensure_schema(conn)
                    conn.commit()
                    _schema_ready = True
                except Exception:
                    conn.close()
                    raise
    return conn


def create_run(
    id: str,
    model_name: str,
    dataset_name: str,
    config_json: str,
    started_at: str,
    total_steps: Optional[int],
) -> None:
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO training_runs (id, model_name, dataset_name, config_json, started_at, total_steps)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (id, model_name, dataset_name, config_json, started_at, total_steps),
        )
        conn.commit()
    finally:
        conn.close()


def update_run_total_steps(id: str, total_steps: int) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE training_runs SET total_steps = ? WHERE id = ?",
            (total_steps, id),
        )
        conn.commit()
    finally:
        conn.close()


def update_run_progress(
    id: str, step: int, loss: Optional[float], duration_seconds: Optional[float]
) -> None:
    """Update current progress on a running training run (called on each metric flush)."""
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE training_runs SET final_step = ?, final_loss = ?, duration_seconds = ? WHERE id = ?",
            (step, loss, duration_seconds, id),
        )
        conn.commit()
    finally:
        conn.close()


def finish_run(
    id: str,
    status: str,
    ended_at: str,
    final_step: Optional[int],
    final_loss: Optional[float],
    duration_seconds: Optional[float],
    loss_sparkline: Optional[str] = None,
    output_dir: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    conn = get_connection()
    try:
        conn.execute(
            """
            UPDATE training_runs
            SET status = ?, ended_at = ?, final_step = ?, final_loss = ?,
                duration_seconds = ?, loss_sparkline = ?, output_dir = ?,
                error_message = ?
            WHERE id = ?
            """,
            (
                status,
                ended_at,
                final_step,
                final_loss,
                duration_seconds,
                loss_sparkline,
                output_dir,
                error_message,
                id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def insert_metrics_batch(run_id: str, metrics: list[dict]) -> None:
    if not metrics:
        return
    conn = get_connection()
    try:
        conn.executemany(
            """
            INSERT INTO training_metrics
                (run_id, step, loss, learning_rate, grad_norm, eval_loss, epoch, num_tokens, elapsed_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id, step) DO UPDATE SET
                loss = COALESCE(excluded.loss, loss),
                learning_rate = COALESCE(excluded.learning_rate, learning_rate),
                grad_norm = COALESCE(excluded.grad_norm, grad_norm),
                eval_loss = COALESCE(excluded.eval_loss, eval_loss),
                epoch = COALESCE(excluded.epoch, epoch),
                num_tokens = COALESCE(excluded.num_tokens, num_tokens),
                elapsed_seconds = COALESCE(excluded.elapsed_seconds, elapsed_seconds)
            """,
            [
                (
                    run_id,
                    m.get("step"),
                    m.get("loss"),
                    m.get("learning_rate"),
                    m.get("grad_norm"),
                    m.get("eval_loss"),
                    m.get("epoch"),
                    m.get("num_tokens"),
                    m.get("elapsed_seconds"),
                )
                for m in metrics
            ],
        )
        conn.commit()
    finally:
        conn.close()


def update_run_display_name(id: str, display_name: Optional[str]) -> None:
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE training_runs SET display_name = ? WHERE id = ?",
            (display_name, id),
        )
        conn.commit()
    finally:
        conn.close()


def list_runs(limit: int = 50, offset: int = 0) -> dict:
    conn = get_connection()
    try:
        total = conn.execute("SELECT COUNT(*) FROM training_runs").fetchone()[0]
        rows = conn.execute(
            """
            SELECT r.id, r.status, r.model_name, r.dataset_name, r.started_at,
                   r.ended_at, r.total_steps, r.final_step, r.final_loss,
                   r.output_dir, r.duration_seconds, r.error_message,
                   r.loss_sparkline, r.display_name, r.config_json,
                   CASE
                       WHEN r.status = 'stopped'
                            AND r.output_dir IS NOT NULL
                            AND EXISTS (
                                SELECT 1
                                FROM training_runs newer
                                WHERE newer.output_dir = r.output_dir
                                  AND newer.status IN ('stopped', 'completed')
                                  AND newer.started_at > r.started_at
                            )
                       THEN 1 ELSE 0
                   END AS resumed_later
            FROM training_runs r
            ORDER BY started_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
        runs = []
        for row in rows:
            run = dict(row)
            run["project_name"] = _extract_project_name_from_config_json(run.get("config_json"))
            sparkline = run.get("loss_sparkline")
            if sparkline:
                try:
                    run["loss_sparkline"] = json.loads(sparkline)
                except (json.JSONDecodeError, TypeError):
                    logger.debug("Failed to parse loss_sparkline for run %s", run.get("id"))
                    run["loss_sparkline"] = None
            runs.append(run)
        return {"runs": runs, "total": total}
    finally:
        conn.close()


def get_run(id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT r.*,
                   CASE
                       WHEN r.status = 'stopped'
                            AND r.output_dir IS NOT NULL
                            AND EXISTS (
                                SELECT 1
                                FROM training_runs newer
                                WHERE newer.output_dir = r.output_dir
                                  AND newer.status IN ('stopped', 'completed')
                                  AND newer.started_at > r.started_at
                            )
                       THEN 1 ELSE 0
                   END AS resumed_later
            FROM training_runs r
            WHERE r.id = ?
            """,
            (id,),
        ).fetchone()
        if row is None:
            return None
        run = dict(row)
        run["project_name"] = _extract_project_name_from_config_json(run.get("config_json"))
        sparkline = run.get("loss_sparkline")
        if sparkline:
            try:
                run["loss_sparkline"] = json.loads(sparkline)
            except (json.JSONDecodeError, TypeError):
                logger.debug("Failed to parse loss_sparkline for run %s", id)
                run["loss_sparkline"] = None
        return run
    finally:
        conn.close()


def get_resumable_run_by_output_dir(output_dir: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT r.*,
                   0 AS resumed_later
            FROM training_runs r
            WHERE r.output_dir = ?
              AND r.status = 'stopped'
              AND NOT EXISTS (
                  SELECT 1
                  FROM training_runs newer
                  WHERE newer.output_dir = r.output_dir
                    AND newer.status IN ('stopped', 'completed')
                    AND newer.started_at > r.started_at
              )
            ORDER BY r.started_at DESC
            LIMIT 1
            """,
            (output_dir,),
        ).fetchone()
        if row is None:
            return None
        run = dict(row)
        sparkline = run.get("loss_sparkline")
        if sparkline:
            try:
                run["loss_sparkline"] = json.loads(sparkline)
            except (json.JSONDecodeError, TypeError):
                logger.debug("Failed to parse loss_sparkline for output_dir %s", output_dir)
                run["loss_sparkline"] = None
        return run
    finally:
        conn.close()


def get_run_metrics(id: str) -> dict:
    """Return metric arrays for a run, using paired step arrays per metric."""
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT step, loss, learning_rate, grad_norm, eval_loss, epoch,
                   num_tokens, elapsed_seconds
            FROM training_metrics
            WHERE run_id = ?
            ORDER BY step
            """,
            (id,),
        ).fetchall()

        step_history: list[int] = []
        loss_history: list[float] = []
        loss_step_history: list[int] = []
        lr_history: list[float] = []
        lr_step_history: list[int] = []
        grad_norm_history: list[float] = []
        grad_norm_step_history: list[int] = []
        eval_loss_history: list[float] = []
        eval_step_history: list[int] = []
        final_epoch: float | None = None
        final_num_tokens: int | None = None

        for row in rows:
            step = row["step"]
            step_history.append(step)
            if step > 0 and row["loss"] is not None:
                loss_history.append(row["loss"])
                loss_step_history.append(step)
            if step > 0 and row["learning_rate"] is not None:
                lr_history.append(row["learning_rate"])
                lr_step_history.append(step)
            if step > 0 and row["grad_norm"] is not None:
                grad_norm_history.append(row["grad_norm"])
                grad_norm_step_history.append(step)
            if step > 0 and row["eval_loss"] is not None:
                eval_loss_history.append(row["eval_loss"])
                eval_step_history.append(step)
            if row["epoch"] is not None:
                final_epoch = row["epoch"]
            if row["num_tokens"] is not None:
                final_num_tokens = row["num_tokens"]

        return {
            "step_history": step_history,
            "loss_history": loss_history,
            "loss_step_history": loss_step_history,
            "lr_history": lr_history,
            "lr_step_history": lr_step_history,
            "grad_norm_history": grad_norm_history,
            "grad_norm_step_history": grad_norm_step_history,
            "eval_loss_history": eval_loss_history,
            "eval_step_history": eval_step_history,
            "final_epoch": final_epoch,
            "final_num_tokens": final_num_tokens,
        }
    finally:
        conn.close()


def delete_run(id: str) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM training_runs WHERE id = ?", (id,))
        conn.commit()
    finally:
        conn.close()


def cleanup_orphaned_runs() -> None:
    """Mark any 'running' rows as errored on startup (server restarted mid-training)."""
    conn = get_connection()
    try:
        conn.execute(
            """
            UPDATE training_runs
            SET status = 'error',
                error_message = 'Server restarted during training',
                ended_at = ?
            WHERE status = 'running'
            """,
            (datetime.now(timezone.utc).isoformat(),),
        )
        conn.commit()
    finally:
        conn.close()


def list_scan_folders() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT id, path, created_at FROM scan_folders ORDER BY created_at"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def add_scan_folder(path: str) -> dict:
    """Add a directory to the custom scan folder list. Returns the row."""
    if not path or not path.strip():
        raise ValueError("Path cannot be empty")
    normalized = os.path.realpath(os.path.expanduser(path.strip()))

    # Validate the path is an existing, readable directory before persisting.
    if not os.path.exists(normalized):
        raise ValueError("Path does not exist")
    if not os.path.isdir(normalized):
        raise ValueError("Path must be a directory, not a file")
    if not os.access(normalized, os.R_OK | os.X_OK):
        raise ValueError("Path is not readable")
    # Reject a local filesystem root ("/", or a bare Windows drive root "C:\\"):
    # registering one seeds the browse allowlist with a root above denied system
    # dirs. A UNC share root (\\server\share) has none under it and was
    # registerable before this guard, so it stays allowed. Mirrors scan_folders.py.
    if is_local_filesystem_root(normalized):
        raise ValueError("The filesystem root cannot be registered")
    if _contains_sensitive_path_component(normalized):
        raise ValueError("Credential or configuration directories are not allowed")

    # Windows: normcase for the denylist check but store original casing
    # so consumers see the native drive-letter casing (e.g. C:\Models).
    is_win = platform.system() == "Windows"
    check = os.path.normcase(normalized) if is_win else normalized
    for prefix in _denied_path_prefixes():
        if check == prefix or check.startswith(prefix + os.sep):
            if prefix == "/run" and is_linux_run_media_path(check):
                continue
            raise ValueError(f"Path under {prefix} is not allowed")

    conn = get_connection()
    try:
        now = datetime.now(timezone.utc).isoformat()
        # Windows: case-insensitive lookup so C:\Models and c:\models dedup.
        if is_win:
            existing = conn.execute(
                "SELECT id, path, created_at FROM scan_folders WHERE path = ? COLLATE NOCASE",
                (normalized,),
            ).fetchone()
        else:
            existing = conn.execute(
                "SELECT id, path, created_at FROM scan_folders WHERE path = ?",
                (normalized,),
            ).fetchone()
        if existing is not None:
            return dict(existing)
        try:
            conn.execute(
                "INSERT INTO scan_folders (path, created_at) VALUES (?, ?)",
                (normalized, now),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            pass  # duplicate; fall through to SELECT
        # Same collation as the pre-check to catch concurrent writes (Windows).
        fallback_sql = (
            "SELECT id, path, created_at FROM scan_folders WHERE path = ? COLLATE NOCASE"
            if is_win
            else "SELECT id, path, created_at FROM scan_folders WHERE path = ?"
        )
        row = conn.execute(fallback_sql, (normalized,)).fetchone()
        if row is None:
            raise ValueError("Folder was concurrently removed")
        return dict(row)
    finally:
        conn.close()


def remove_scan_folder(id: int) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM scan_folders WHERE id = ?", (id,))
        conn.commit()
    finally:
        conn.close()


def _json_loads(value: str | None, fallback):
    if value is None:
        return fallback
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return fallback


def _chat_thread_from_row(row: sqlite3.Row) -> dict:
    data = dict(row)
    return {
        "id": data["id"],
        "title": data["title"],
        "modelType": data["model_type"],
        "modelId": data.get("model_id") or "",
        "pairId": data.get("pair_id") or None,
        "projectId": data.get("project_id") or None,
        "archived": bool(data["archived"]),
        "createdAt": data["created_at"],
        "updatedAt": data.get("updated_at")
        if data.get("updated_at") is not None
        else data["created_at"],
        "openaiCodeExecContainerId": data.get("openai_code_exec_container_id"),
        "anthropicCodeExecContainerId": data.get("anthropic_code_exec_container_id"),
        "forkedFromThreadId": data.get("forked_from_thread_id"),
        "forkedFromMessageId": data.get("forked_from_message_id"),
    }


def _chat_project_from_row(row: sqlite3.Row) -> dict:
    data = dict(row)
    root_path = data.get("root_path")
    return {
        "id": data["id"],
        "name": data["name"],
        "instructions": data.get("instructions") or "",
        "rootPath": root_path or None,
        "sandboxPath": os.path.join(root_path, "sandbox") if root_path else None,
        "archived": bool(data["archived"]),
        "createdAt": data["created_at"],
        "updatedAt": data["updated_at"],
    }


def _chat_message_from_row(row: sqlite3.Row) -> dict:
    data = dict(row)
    message = {
        "id": data["id"],
        "threadId": data["thread_id"],
        "parentId": data.get("parent_id"),
        "role": data["role"],
        "content": _json_loads(data.get("content_json"), []),
        "createdAt": data["created_at"],
    }
    attachments = _json_loads(data.get("attachments_json"), None)
    metadata = _json_loads(data.get("metadata_json"), None)
    if attachments is not None:
        message["attachments"] = attachments
    if metadata is not None:
        message["metadata"] = metadata
    return message


def upsert_chat_thread(thread: dict) -> dict:
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO chat_threads
                (id, title, model_type, model_id, pair_id, project_id, archived, created_at, updated_at, openai_code_exec_container_id, anthropic_code_exec_container_id, forked_from_thread_id, forked_from_message_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                model_type = excluded.model_type,
                model_id = excluded.model_id,
                pair_id = excluded.pair_id,
                project_id = excluded.project_id,
                archived = excluded.archived,
                created_at = excluded.created_at,
                updated_at = COALESCE(excluded.updated_at, chat_threads.updated_at),
                openai_code_exec_container_id = excluded.openai_code_exec_container_id,
                anthropic_code_exec_container_id = excluded.anthropic_code_exec_container_id,
                forked_from_thread_id = excluded.forked_from_thread_id,
                forked_from_message_id = excluded.forked_from_message_id
            """,
            (
                thread["id"],
                thread.get("title") or "New Chat",
                thread["modelType"],
                thread.get("modelId") or "",
                thread.get("pairId"),
                thread.get("projectId"),
                1 if thread.get("archived") else 0,
                int(thread["createdAt"]),
                int(thread["updatedAt"]) if thread.get("updatedAt") is not None else None,
                thread.get("openaiCodeExecContainerId"),
                thread.get("anthropicCodeExecContainerId"),
                thread.get("forkedFromThreadId"),
                thread.get("forkedFromMessageId"),
            ),
        )
        conn.commit()
        return get_chat_thread(thread["id"]) or thread
    finally:
        conn.close()


def update_chat_thread(id: str, patch: dict) -> Optional[dict]:
    allowed = {
        "title": ("title", patch.get("title")),
        "modelType": ("model_type", patch.get("modelType")),
        "modelId": ("model_id", patch.get("modelId")),
        "pairId": ("pair_id", patch.get("pairId")),
        "projectId": ("project_id", patch.get("projectId")),
        "archived": ("archived", 1 if patch.get("archived") else 0),
        "createdAt": ("created_at", patch.get("createdAt")),
        "updatedAt": ("updated_at", patch.get("updatedAt")),
        "openaiCodeExecContainerId": (
            "openai_code_exec_container_id",
            patch.get("openaiCodeExecContainerId"),
        ),
        "anthropicCodeExecContainerId": (
            "anthropic_code_exec_container_id",
            patch.get("anthropicCodeExecContainerId"),
        ),
        "forkedFromThreadId": (
            "forked_from_thread_id",
            patch.get("forkedFromThreadId"),
        ),
        "forkedFromMessageId": (
            "forked_from_message_id",
            patch.get("forkedFromMessageId"),
        ),
    }
    assignments = []
    values = []
    for key, (column, value) in allowed.items():
        if key in patch:
            assignments.append(f"{column} = ?")
            values.append(value)
    if not assignments:
        return get_chat_thread(id)

    conn = get_connection()
    try:
        conn.execute(
            f"UPDATE chat_threads SET {', '.join(assignments)} WHERE id = ?",
            (*values, id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM chat_threads WHERE id = ?", (id,)).fetchone()
        return _chat_thread_from_row(row) if row is not None else None
    finally:
        conn.close()


def get_chat_thread(id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM chat_threads WHERE id = ?", (id,)).fetchone()
        return _chat_thread_from_row(row) if row is not None else None
    finally:
        conn.close()


def list_chat_threads(
    model_type: str | None = None,
    pair_id: str | None = None,
    project_id: str | None = None,
    include_archived: bool = True,
) -> list[dict]:
    clauses = []
    values: list[object] = []
    if model_type is not None:
        clauses.append("model_type = ?")
        values.append(model_type)
    if pair_id is not None:
        clauses.append("pair_id = ?")
        values.append(pair_id)
    if project_id is not None:
        clauses.append("project_id = ?")
        values.append(project_id)
    if not include_archived:
        clauses.append("archived = 0")
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    conn = get_connection()
    try:
        rows = conn.execute(
            f"SELECT * FROM chat_threads {where} "
            "ORDER BY COALESCE(updated_at, created_at) DESC, created_at DESC",
            values,
        ).fetchall()
        return [_chat_thread_from_row(row) for row in rows]
    finally:
        conn.close()


def delete_chat_threads(ids: list[str]) -> None:
    if not ids:
        return
    conn = get_connection()
    try:
        conn.executemany("DELETE FROM chat_threads WHERE id = ?", [(id,) for id in ids])
        conn.commit()
    finally:
        conn.close()


def clear_chat_history() -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM chat_threads")
        conn.commit()
    finally:
        conn.close()


def count_chat_threads() -> int:
    conn = get_connection()
    try:
        return int(conn.execute("SELECT COUNT(*) FROM chat_threads").fetchone()[0])
    finally:
        conn.close()


def upsert_chat_project(project: dict) -> dict:
    existing = get_chat_project(project["id"])
    root_path = existing.get("rootPath") if existing else None
    if not root_path:
        root_path = _default_project_root(project)
    root_path = _ensure_project_workspace(root_path)
    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO chat_projects
                (id, name, instructions, root_path, archived, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                instructions = excluded.instructions,
                root_path = COALESCE(chat_projects.root_path, excluded.root_path),
                archived = excluded.archived,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at
            """,
            (
                project["id"],
                project["name"],
                project.get("instructions") or "",
                root_path,
                1 if project.get("archived") else 0,
                int(project["createdAt"]),
                int(project["updatedAt"]),
            ),
        )
        conn.commit()
        return get_chat_project(project["id"]) or project
    finally:
        conn.close()


def update_chat_project(id: str, patch: dict) -> Optional[dict]:
    allowed = {
        "name": ("name", patch.get("name")),
        "instructions": ("instructions", patch.get("instructions")),
        "archived": ("archived", 1 if patch.get("archived") else 0),
        "createdAt": ("created_at", patch.get("createdAt")),
        "updatedAt": ("updated_at", patch.get("updatedAt")),
    }
    assignments = []
    values = []
    for key, (column, value) in allowed.items():
        if key in patch:
            assignments.append(f"{column} = ?")
            values.append(value)
    if not assignments:
        return get_chat_project(id)

    conn = get_connection()
    try:
        conn.execute(
            f"UPDATE chat_projects SET {', '.join(assignments)} WHERE id = ?",
            (*values, id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM chat_projects WHERE id = ?", (id,)).fetchone()
        return _chat_project_from_row(row) if row is not None else None
    finally:
        conn.close()


def ensure_chat_project_workspace(id: str) -> Optional[dict]:
    project = get_chat_project(id)
    if project is None:
        return None
    root_path = project.get("rootPath") or _default_project_root(project)
    root_path = _ensure_project_workspace(root_path)
    if project.get("rootPath") == root_path:
        return project
    conn = get_connection()
    try:
        conn.execute(
            "UPDATE chat_projects SET root_path = ? WHERE id = ?",
            (root_path, id),
        )
        conn.commit()
    finally:
        conn.close()
    return get_chat_project(id)


def get_chat_project(id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute("SELECT * FROM chat_projects WHERE id = ?", (id,)).fetchone()
        return _chat_project_from_row(row) if row is not None else None
    finally:
        conn.close()


def list_chat_projects(include_archived: bool = False) -> list[dict]:
    conn = get_connection()
    try:
        where = "" if include_archived else "WHERE archived = 0"
        rows = conn.execute(
            f"SELECT * FROM chat_projects {where} ORDER BY updated_at DESC"
        ).fetchall()
        return [_chat_project_from_row(row) for row in rows]
    finally:
        conn.close()


def delete_chat_project(id: str, delete_files: bool = False) -> Optional[dict]:
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM chat_projects WHERE id = ?", (id,)).fetchone()
        if row is None:
            conn.rollback()
            return None
        project = _chat_project_from_row(row)
        conn.execute("DELETE FROM chat_threads WHERE project_id = ?", (id,))
        conn.execute("DELETE FROM chat_projects WHERE id = ?", (id,))
        conn.commit()
        if delete_files:
            _delete_project_workspace(project)
        return project
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


class ChatMessageConflictError(RuntimeError):
    """Raised when a chat message id already belongs to another thread."""


class CorruptSettingsError(RuntimeError):
    """Raised when a partial settings patch would overwrite corrupt settings."""


def _parse_chat_setting_json(key: str, value_json: str) -> tuple[bool, Any]:
    try:
        return True, json.loads(value_json)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning(
            "Corrupt chat_settings JSON; quarantining key=%s error=%s",
            key,
            exc,
        )
        return False, None


def _load_chat_settings_for_merge(conn: sqlite3.Connection) -> tuple[dict[str, Any], set[str]]:
    rows = conn.execute("SELECT key, value_json FROM chat_settings").fetchall()
    current: dict[str, Any] = {}
    corrupt: set[str] = set()
    now = datetime.now(timezone.utc).isoformat()
    for row in rows:
        ok, value = _parse_chat_setting_json(row["key"], row["value_json"])
        if ok:
            current[row["key"]] = value
            continue
        corrupt.add(row["key"])
        conn.execute(
            """
            INSERT INTO chat_settings_quarantine
                (key, value_json, reason, quarantined_at)
            VALUES (?, ?, ?, ?)
            """,
            (row["key"], row["value_json"], "json_decode_error", now),
        )
        conn.execute(
            "DELETE FROM chat_settings WHERE key = ? AND value_json = ?",
            (row["key"], row["value_json"]),
        )
    return current, corrupt


def _raise_if_chat_message_thread_conflicts(
    conn: sqlite3.Connection, thread_id: str, message_ids: list[str]
) -> None:
    unique_ids = list(dict.fromkeys(message_ids))
    if not unique_ids:
        return
    conflicts: list[str] = []
    for start in range(0, len(unique_ids), _SQLITE_IN_CHUNK_SIZE):
        chunk = unique_ids[start : start + _SQLITE_IN_CHUNK_SIZE]
        placeholders = ",".join("?" for _ in chunk)
        rows = conn.execute(
            f"""
            SELECT id FROM chat_messages
            WHERE id IN ({placeholders}) AND thread_id != ?
            ORDER BY id
            """,
            (*chunk, thread_id),
        ).fetchall()
        conflicts.extend(row["id"] for row in rows)
    if conflicts:
        preview = ", ".join(conflicts[:5])
        suffix = "" if len(conflicts) <= 5 else f" (+{len(conflicts) - 5} more)"
        raise ChatMessageConflictError(
            f"Message id already belongs to another thread: {preview}{suffix}"
        )


def _bump_chat_thread_updated_at(
    conn: sqlite3.Connection, thread_id: str, message_created_at: int
) -> None:
    conn.execute(
        """
        UPDATE chat_threads
        SET updated_at = MAX(COALESCE(updated_at, created_at), ?)
        WHERE id = ?
        """,
        (message_created_at, thread_id),
    )


def _recompute_chat_thread_updated_at(conn: sqlite3.Connection, thread_id: str) -> None:
    """Set updated_at from the remaining messages, floored at created_at.

    Unlike the ratchet-only bump, this can lower updated_at -- needed after
    pruning, which may delete the thread's newest message.
    """
    conn.execute(
        """
        UPDATE chat_threads
        SET updated_at = MAX(
            COALESCE(
                (
                    SELECT MAX(m.created_at) FROM chat_messages m
                    WHERE m.thread_id = chat_threads.id
                ),
                created_at
            ),
            created_at
        )
        WHERE id = ?
        """,
        (thread_id,),
    )


def upsert_chat_message(message: dict) -> dict:
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        _raise_if_chat_message_thread_conflicts(
            conn,
            message["threadId"],
            [message["id"]],
        )
        conn.execute(
            """
            INSERT INTO chat_messages
                (id, thread_id, parent_id, role, content_json, attachments_json, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                parent_id = excluded.parent_id,
                role = excluded.role,
                content_json = excluded.content_json,
                attachments_json = excluded.attachments_json,
                metadata_json = excluded.metadata_json,
                created_at = excluded.created_at
            WHERE excluded.thread_id = chat_messages.thread_id
            """,
            (
                message["id"],
                message["threadId"],
                message.get("parentId"),
                message["role"],
                json.dumps(message.get("content", [])),
                json.dumps(message.get("attachments"))
                if message.get("attachments") is not None
                else None,
                json.dumps(message.get("metadata"))
                if message.get("metadata") is not None
                else None,
                int(message["createdAt"]),
            ),
        )
        _bump_chat_thread_updated_at(conn, message["threadId"], int(message["createdAt"]))
        conn.commit()
        return message
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def sync_chat_messages(
    thread_id: str,
    messages: list[dict],
    prune_missing: bool = False,
) -> list[dict]:
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        _raise_if_chat_message_thread_conflicts(
            conn,
            thread_id,
            [m["id"] for m in messages],
        )
        conn.executemany(
            """
            INSERT INTO chat_messages
                (id, thread_id, parent_id, role, content_json, attachments_json, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                parent_id = excluded.parent_id,
                role = excluded.role,
                content_json = excluded.content_json,
                attachments_json = excluded.attachments_json,
                metadata_json = excluded.metadata_json,
                created_at = excluded.created_at
            WHERE excluded.thread_id = chat_messages.thread_id
            """,
            [
                (
                    m["id"],
                    thread_id,
                    m.get("parentId"),
                    m["role"],
                    json.dumps(m.get("content", [])),
                    json.dumps(m.get("attachments")) if m.get("attachments") is not None else None,
                    json.dumps(m.get("metadata")) if m.get("metadata") is not None else None,
                    int(m["createdAt"]),
                )
                for m in messages
            ],
        )
        if prune_missing:
            survivor_ids = {str(message["id"]) for message in messages}
            existing_ids = {
                str(row["id"])
                for row in conn.execute(
                    "SELECT id FROM chat_messages WHERE thread_id = ?", (thread_id,)
                ).fetchall()
            }
            conn.executemany(
                "DELETE FROM chat_messages WHERE thread_id = ? AND id = ?",
                [(thread_id, message_id) for message_id in existing_ids - survivor_ids],
            )
            _recompute_chat_thread_updated_at(conn, thread_id)
        elif messages:
            _bump_chat_thread_updated_at(
                conn, thread_id, max(int(m["createdAt"]) for m in messages)
            )
        conn.commit()
        return list_chat_messages(thread_id)
    except ChatMessageConflictError:
        conn.rollback()
        raise
    except sqlite3.Error:
        logger.exception("Failed to sync chat messages for thread %s", thread_id)
        conn.rollback()
        raise
    finally:
        conn.close()


_RESEARCH_LINK_KEYS = {
    "researchRunId",
    "researchRun",
    "researchStatus",
    "researchPlanRevision",
    "serverManaged",
}


def _detach_research_message_json(
    content_json: str, metadata_json: str | None
) -> tuple[str, str | None]:
    content = _json_loads(content_json, [])
    metadata = _json_loads(metadata_json, None)
    custom = metadata.get("custom") if isinstance(metadata, dict) else None
    linked = (
        isinstance(metadata, dict)
        and any(key in metadata for key in _RESEARCH_LINK_KEYS)
        or isinstance(custom, dict)
        and any(key in custom for key in _RESEARCH_LINK_KEYS)
        or isinstance(content, list)
        and any(
            isinstance(part, dict) and any(key in part for key in _RESEARCH_LINK_KEYS)
            for part in content
        )
    )
    if not linked:
        return content_json, metadata_json

    if isinstance(content, list):
        content = [
            {key: value for key, value in part.items() if key not in _RESEARCH_LINK_KEYS}
            if isinstance(part, dict)
            else part
            for part in content
        ]
    if isinstance(metadata, dict):
        metadata = {key: value for key, value in metadata.items() if key not in _RESEARCH_LINK_KEYS}
        custom = metadata.get("custom")
        if isinstance(custom, dict):
            metadata["custom"] = {
                key: value for key, value in custom.items() if key not in _RESEARCH_LINK_KEYS
            }
    return (
        json.dumps(content, ensure_ascii = False),
        json.dumps(metadata, ensure_ascii = False) if metadata is not None else None,
    )


def fork_chat_thread(
    source_thread_id: str,
    branch_message_id: str,
    new_thread_id: str,
    new_title: str,
    created_at: int,
    id_factory,
) -> Optional[dict]:
    """Atomically clone thread + ancestor msgs `[root..branch_message_id]`
    into a new thread. Returns the new thread dict (with messages copied)
    or None if source missing.

    Reset both code-exec container ids -- per-provider snapshot is handled
    by the route layer (best-effort, OpenAI only).

    `id_factory()` produces fresh message uuids; injected for testability.
    """
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        src = conn.execute(
            "SELECT * FROM chat_threads WHERE id = ?", (source_thread_id,)
        ).fetchone()
        if src is None:
            conn.rollback()
            return None
        # Verify branch msg belongs to source thread.
        branch_row = conn.execute(
            "SELECT * FROM chat_messages WHERE thread_id = ? AND id = ?",
            (source_thread_id, branch_message_id),
        ).fetchone()
        if branch_row is None:
            conn.rollback()
            return None
        # Walk ancestry from branch msg back to root via parent_id chain.
        ancestry: list[sqlite3.Row] = []
        cursor_row = branch_row
        seen: set[str] = set()
        while cursor_row is not None and cursor_row["id"] not in seen:
            ancestry.append(cursor_row)
            seen.add(cursor_row["id"])
            parent = cursor_row["parent_id"]
            if not parent:
                break
            cursor_row = conn.execute(
                "SELECT * FROM chat_messages WHERE thread_id = ? AND id = ?",
                (source_thread_id, parent),
            ).fetchone()
        ancestry.reverse()  # root .. branch msg
        # Map old msg id -> new msg id for parent_id rewriting.
        id_map: dict[str, str] = {row["id"]: id_factory() for row in ancestry}
        src_dict = dict(src)
        conn.execute(
            """
            INSERT INTO chat_threads
                (id, title, model_type, model_id, pair_id, project_id, archived, created_at,
                 openai_code_exec_container_id, anthropic_code_exec_container_id,
                 forked_from_thread_id, forked_from_message_id)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?, NULL, NULL, ?, ?)
            """,
            (
                new_thread_id,
                new_title,
                src_dict["model_type"],
                src_dict.get("model_id") or "",
                None,  # pairId: forks always standalone (compare-mode disabled v1)
                src_dict.get("project_id"),
                int(created_at),
                source_thread_id,
                branch_message_id,
            ),
        )
        fork_messages = []
        for row in ancestry:
            content_json, metadata_json = _detach_research_message_json(
                row["content_json"], row["metadata_json"]
            )
            fork_messages.append(
                (
                    id_map[row["id"]],
                    new_thread_id,
                    id_map.get(row["parent_id"]) if row["parent_id"] else None,
                    row["role"],
                    content_json,
                    row["attachments_json"],
                    metadata_json,
                    int(row["created_at"]),
                )
            )
        conn.executemany(
            """
            INSERT INTO chat_messages
                (id, thread_id, parent_id, role, content_json, attachments_json,
                 metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            fork_messages,
        )
        conn.commit()
        thread_row = conn.execute(
            "SELECT * FROM chat_threads WHERE id = ?", (new_thread_id,)
        ).fetchone()
        return _chat_thread_from_row(thread_row) if thread_row is not None else None
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def count_forks_for_message(thread_id: str, message_id: str) -> int:
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) FROM chat_threads
            WHERE forked_from_thread_id = ? AND forked_from_message_id = ?
            """,
            (thread_id, message_id),
        ).fetchone()
        return int(row[0]) if row is not None else 0
    finally:
        conn.close()


def list_chat_messages(thread_id: str) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute(
            """
            SELECT * FROM chat_messages
            WHERE thread_id = ?
            ORDER BY created_at ASC, id ASC
            """,
            (thread_id,),
        ).fetchall()
        return [_chat_message_from_row(row) for row in rows]
    finally:
        conn.close()


def get_chat_message(thread_id: str, message_id: str) -> Optional[dict]:
    conn = get_connection()
    try:
        row = conn.execute(
            """
            SELECT * FROM chat_messages
            WHERE thread_id = ? AND id = ?
            """,
            (thread_id, message_id),
        ).fetchone()
        return _chat_message_from_row(row) if row is not None else None
    finally:
        conn.close()


def list_chat_messages_for_threads(thread_ids: list[str]) -> list[dict]:
    if not thread_ids:
        return []
    unique_thread_ids = list(dict.fromkeys(thread_ids))
    messages: list[dict] = []
    conn = get_connection()
    try:
        for start in range(0, len(unique_thread_ids), _SQLITE_IN_CHUNK_SIZE):
            chunk = unique_thread_ids[start : start + _SQLITE_IN_CHUNK_SIZE]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(
                f"""
                SELECT * FROM chat_messages
                WHERE thread_id IN ({placeholders})
                ORDER BY created_at ASC, id ASC
                """,
                chunk,
            ).fetchall()
            messages.extend(_chat_message_from_row(row) for row in rows)
        return sorted(
            messages,
            key = lambda message: (message["createdAt"], message["id"]),
        )
    finally:
        conn.close()


def get_app_setting(key: str, fallback = None):
    conn = get_connection()
    try:
        row = conn.execute("SELECT value_json FROM app_settings WHERE key = ?", (key,)).fetchone()
        if row is None:
            return fallback
        return _json_loads(row["value_json"], fallback)
    finally:
        conn.close()


def upsert_app_settings(settings: dict[str, Any]) -> dict[str, Any]:
    if not settings:
        return {}
    conn = get_connection()
    try:
        now = datetime.now(timezone.utc).isoformat()
        conn.executemany(
            """
            INSERT INTO app_settings (key, value_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value_json = excluded.value_json,
                updated_at = excluded.updated_at
            """,
            [(key, json.dumps(value), now) for key, value in settings.items()],
        )
        conn.commit()
        rows = conn.execute("SELECT key, value_json FROM app_settings ORDER BY key").fetchall()
        return {row["key"]: _json_loads(row["value_json"], None) for row in rows}
    finally:
        conn.close()


def upsert_app_setting_map_entry(
    key: str, entry_key: str, entry_value: dict[str, Any] | None
) -> dict[str, Any]:
    """Set (or delete, when entry_value is falsy) one sub-entry of a dict-valued
    app setting, atomically under BEGIN IMMEDIATE so concurrent writers to other
    sub-entries cannot drop each other's updates."""
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT value_json FROM app_settings WHERE key = ?", (key,)).fetchone()
        current = _json_loads(row["value_json"], {}) if row else {}
        if not isinstance(current, dict):
            current = {}
        if entry_value:
            current[entry_key] = entry_value
        else:
            current.pop(entry_key, None)
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO app_settings (key, value_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value_json = excluded.value_json,
                updated_at = excluded.updated_at
            """,
            (key, json.dumps(current), now),
        )
        conn.commit()
        return current
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def list_chat_settings() -> dict[str, Any]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT key, value_json FROM chat_settings ORDER BY key").fetchall()
        settings: dict[str, Any] = {}
        for row in rows:
            settings[row["key"]] = _json_loads(row["value_json"], None)
        return settings
    finally:
        conn.close()


def upsert_chat_settings(settings: dict[str, Any]) -> dict[str, Any]:
    if not settings:
        return list_chat_settings()
    conn = get_connection()
    try:
        now = datetime.now(timezone.utc).isoformat()
        conn.executemany(
            """
            INSERT INTO chat_settings (key, value_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value_json = excluded.value_json,
                updated_at = excluded.updated_at
            """,
            [(key, json.dumps(value), now) for key, value in settings.items()],
        )
        conn.commit()
        return list_chat_settings()
    finally:
        conn.close()


def _deep_merge_settings(current: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(current)
    for key, value in updates.items():
        current_value = merged.get(key)
        if isinstance(current_value, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_settings(current_value, value)
        else:
            merged[key] = value
    return merged


def upsert_chat_settings_merge(updates: dict[str, Any]) -> dict[str, Any]:
    """Atomic read-merge-write under BEGIN IMMEDIATE so concurrent writers
    cannot drop each other's updates."""
    if not updates:
        return list_chat_settings()
    conn = get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        current, corrupt = _load_chat_settings_for_merge(conn)
        unsafe_partial_keys = [
            key for key, value in updates.items() if key in corrupt and isinstance(value, dict)
        ]
        if unsafe_partial_keys:
            conn.commit()
            keys = ", ".join(sorted(unsafe_partial_keys))
            raise CorruptSettingsError(
                f"Cannot apply partial settings patch to corrupt key(s): {keys}"
            )
        merged = _deep_merge_settings(current, updates)
        now = datetime.now(timezone.utc).isoformat()
        conn.executemany(
            """
            INSERT INTO chat_settings (key, value_json, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value_json = excluded.value_json,
                updated_at = excluded.updated_at
            """,
            [(key, json.dumps(value), now) for key, value in merged.items()],
        )
        conn.commit()
        return merged
    except CorruptSettingsError:
        raise
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Legacy Dexie import ledger
# ---------------------------------------------------------------------------
# See the schema comment in _ensure_schema() for the recovery rationale.


def list_chat_legacy_imports() -> list[str]:
    """Return the legacy_thread_id of every thread already imported."""
    conn = get_connection()
    try:
        rows = conn.execute("SELECT legacy_thread_id FROM chat_legacy_imports").fetchall()
        return [row[0] for row in rows]
    finally:
        conn.close()


def upsert_chat_legacy_imports(legacy_thread_ids: list[str]) -> tuple[int, int]:
    """Mark each given legacy thread id as imported. Idempotent.

    Returns (accepted, inserted): count of deduped non-empty input ids, and
    count of rows actually new. RETURNING lets callers tell first-time imports
    from idempotent re-runs without an extra SELECT.
    """
    ids = list(dict.fromkeys(tid for tid in legacy_thread_ids if tid))
    if not ids:
        return 0, 0
    ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    conn = get_connection()
    try:
        inserted = 0
        for tid in ids:
            row = conn.execute(
                """
                INSERT INTO chat_legacy_imports (legacy_thread_id, imported_at)
                VALUES (?, ?)
                ON CONFLICT(legacy_thread_id) DO NOTHING
                RETURNING legacy_thread_id
                """,
                (tid, ts),
            ).fetchone()
            if row is not None:
                inserted += 1
        conn.commit()
        return len(ids), inserted
    finally:
        conn.close()
