# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
SQLite storage for training run history and metrics.

Follows the same pattern as auth/storage.py — module-level functions,
raw sqlite3, per-function connections. Enhancements over auth:
  - WAL mode for concurrent read/write access
  - PRAGMA foreign_keys = ON for CASCADE deletes
"""

import json
import logging
import os
import platform
import sqlite3
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)
from typing import Optional


from utils.paths import studio_db_path, ensure_dir


def _denied_path_prefixes() -> list[str]:
    """Platform-aware denylist of system directories."""
    system = platform.system()
    if system == "Linux":
        return ["/proc", "/sys", "/dev", "/etc", "/boot", "/run"]
    if system == "Darwin":
        # realpath() resolves /etc -> /private/etc, /tmp -> /private/tmp on macOS,
        # so include the /private variants to avoid bypasses.
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


_schema_lock = threading.Lock()
_schema_ready = False


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
    existing_cols = {
        row[1] for row in conn.execute("PRAGMA table_info(training_runs)").fetchall()
    }
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
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_metrics_run_id ON training_metrics(run_id)"
    )
    # Use COLLATE NOCASE on Windows so C:\Models and c:\models dedup via the
    # UNIQUE constraint.  On Linux/macOS (case-sensitive FS) keep the default
    # BINARY collation so /Models and /models remain distinct.
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


def get_connection() -> sqlite3.Connection:
    """Open studio.db with WAL mode, create tables once per process, enable foreign keys."""
    global _schema_ready
    db_path = studio_db_path()
    ensure_dir(db_path.parent)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    # foreign_keys is session-scoped, must be set per connection
    conn.execute("PRAGMA foreign_keys=ON")
    if not _schema_ready:
        with _schema_lock:
            if not _schema_ready:
                try:
                    _ensure_schema(conn)
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
                   r.loss_sparkline, r.display_name,
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
            sparkline = run.get("loss_sparkline")
            if sparkline:
                try:
                    run["loss_sparkline"] = json.loads(sparkline)
                except (json.JSONDecodeError, TypeError):
                    logger.debug(
                        "Failed to parse loss_sparkline for run %s", run.get("id")
                    )
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
                logger.debug(
                    "Failed to parse loss_sparkline for output_dir %s", output_dir
                )
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

    # On Windows, use normcase for denylist comparison but store the
    # original-cased path so downstream consumers see the native
    # drive-letter casing the user expects (e.g. C:\Models, not c:\models).
    is_win = platform.system() == "Windows"
    check = os.path.normcase(normalized) if is_win else normalized
    for prefix in _denied_path_prefixes():
        if check == prefix or check.startswith(prefix + os.sep):
            raise ValueError(f"Path under {prefix} is not allowed")

    conn = get_connection()
    try:
        now = datetime.now(timezone.utc).isoformat()
        # On Windows, use case-insensitive lookup so C:\Models and c:\models
        # dedup correctly while preserving the originally-stored casing.
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
            pass  # duplicate -- fall through to SELECT
        # Use the same collation as the pre-check so we find the row even
        # when a concurrent writer stored it with different casing (Windows).
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
