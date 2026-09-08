# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Durable, sequential reconciliation for linked local RAG folders."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import stat
import threading
import time
import uuid
from contextlib import closing
import weakref
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from storage import rag_db
from utils.paths import ensure_dir, rag_uploads_root

from . import config, ingestion, job_leases, store
from utils.paths.path_utils import is_appledouble_metadata

logger = logging.getLogger(__name__)

_wake = threading.Event()
_stop = threading.Event()
_thread: threading.Thread | None = None
_thread_stop: threading.Event | None = None
_thread_lock = threading.Lock()
_worker_lock = threading.Lock()
_worker_state = threading.local()
_folder_locks: weakref.WeakValueDictionary[str, threading.RLock] = weakref.WeakValueDictionary()
_scope_locks: weakref.WeakValueDictionary[str, threading.RLock] = weakref.WeakValueDictionary()
_named_locks_lock = threading.Lock()
_retirement_schema_lock = threading.Lock()
_TERMINAL = {"completed", "failed"}
_MAX_FOLDER_DEPTH = 64
_MAX_NAMED_FAILURES = 3
_MAX_WITHHELD_PATHS = 500
_JOB_EVENT_KEEPALIVE_S = 4.0
_SQLITE_INTEGER_MAX = (1 << 63) - 1


class _SyncStopped(Exception):
    pass


class _LeaseLost(Exception):
    pass


class _FolderChanged(RuntimeError):
    pass


def _folder_lock(folder_id: str) -> threading.RLock:
    with _named_locks_lock:
        return _folder_locks.setdefault(folder_id, threading.RLock())


def _scope_lock(scope: str) -> threading.RLock:
    with _named_locks_lock:
        return _scope_locks.setdefault(scope, threading.RLock())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def now_iso() -> str:
    """The timestamp format linked-folder rows are written and compared with."""
    return _now()


def _hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_key(path: str) -> str:
    return os.path.normcase(os.path.normpath(path))


def _same_file(left: str, right: str) -> bool:
    try:
        return os.path.samefile(left, right)
    except OSError:
        return False


def _paths_overlap(left: str, right: str) -> bool:
    try:
        if os.path.commonpath((left, right)) in (left, right):
            return True
    except ValueError:
        pass
    return any(_same_file(left, str(parent)) for parent in Path(right).parents) or any(
        _same_file(right, str(parent)) for parent in Path(left).parents
    )


def _error_text(exc: Exception, native_path: str | None = None) -> str:
    from utils.native_path_leases import redact_native_paths

    error = redact_native_paths(str(exc) or exc.__class__.__name__)
    if native_path:
        error = error.replace(native_path, "<native_path>")
        error = error.replace(os.path.normpath(native_path), "<native_path>")
    return error


def _is_within(root: str, path: str) -> bool:
    try:
        return os.path.normcase(os.path.commonpath([root, path])) == os.path.normcase(root)
    except ValueError:
        return False


def validate_folder_path(path: str) -> str:
    """Apply the existing model scan-folder policy without persisting there."""
    if not path or not path.strip() or "\x00" in path:
        raise ValueError("Path cannot be empty")
    expanded = os.path.abspath(os.path.expanduser(path))
    try:
        if stat.S_ISLNK(os.lstat(expanded).st_mode):
            raise ValueError("Symbolic-link folders are not allowed")
    except OSError as exc:
        raise ValueError("Path does not exist") from exc
    normalized = os.path.realpath(expanded)
    uploads_root = os.path.realpath(str(rag_uploads_root()))
    if _paths_overlap(_path_key(normalized), _path_key(uploads_root)):
        raise ValueError("The managed RAG uploads folder cannot be linked")

    from hub.storage.scan_folders import (
        contains_sensitive_path_component,
        is_denied_system_path,
    )
    from utils.paths.external_media import is_local_filesystem_root

    if not os.path.isdir(normalized):
        raise ValueError("Path must be a directory, not a file")
    if not os.access(normalized, os.R_OK | os.X_OK):
        raise ValueError("Path is not readable")
    if is_local_filesystem_root(normalized):
        raise ValueError("The filesystem root cannot be registered")
    try:
        if Path(normalized) == Path.home().resolve():
            raise ValueError("The entire home folder cannot be registered")
    except RuntimeError:
        pass
    if contains_sensitive_path_component(normalized):
        raise ValueError("Credential or configuration directories are not allowed")
    if is_denied_system_path(normalized):
        raise ValueError("System directories are not allowed")
    return normalized


def _root_identity(root: str) -> tuple[int, int]:
    try:
        root_stat = os.lstat(root)
    except OSError as exc:
        raise RuntimeError("Linked folder is unavailable") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise RuntimeError("Linked folder is no longer a regular directory")
    if os.path.normcase(os.path.realpath(root)) != os.path.normcase(root):
        raise RuntimeError("Linked folder no longer resolves to its registered path")
    return root_stat.st_dev, root_stat.st_ino


def _store_identity(identity: tuple[int, int]) -> tuple[int | str, int | str]:
    return tuple(value if value <= _SQLITE_INTEGER_MAX else f"x{value:x}" for value in identity)


def _file_identity(metadata: dict) -> tuple[int | str, int | str]:
    return _store_identity((metadata["device"], metadata["inode"]))


def _load_identity(device_id: int | str, file_id: int | str) -> tuple[int, int]:
    def load(value: int | str) -> int:
        return (
            int(value[1:], 16) if isinstance(value, str) and value.startswith("x") else int(value)
        )

    return load(device_id), load(file_id)


def _mount_points() -> frozenset[str]:
    """Return known mount boundaries without assuming directory identities are unique."""
    if os.name != "posix" or not os.path.exists("/proc/self/mountinfo"):
        return frozenset()
    escaped = {"\\040": " ", "\\011": "\t", "\\012": "\n", "\\134": "\\"}
    points = set()
    try:
        with open("/proc/self/mountinfo", encoding = "utf-8") as mountinfo:
            for line in mountinfo:
                fields = line.split()
                if len(fields) < 5:
                    continue
                path = fields[4]
                for encoded, decoded in escaped.items():
                    path = path.replace(encoded, decoded)
                points.add(_path_key(os.path.realpath(path)))
    except OSError:
        return frozenset()
    return frozenset(points)


def create_folder(
    *,
    scope_type: str,
    scope_id: str,
    path: str,
    expected_identity: tuple[int, int] | None = None,
    name: str | None = None,
    auto_sync: bool = True,
) -> dict:
    if scope_type not in {"knowledge_base", "project"}:
        raise ValueError("Linked folders support only knowledge-base and project scopes")
    normalized = validate_folder_path(path)
    try:
        root_device, root_inode = _root_identity(normalized)
    except RuntimeError as exc:
        raise ValueError(str(exc)) from exc
    if expected_identity is not None and (root_device, root_inode) != expected_identity:
        raise ValueError("Linked folder changed after it was selected")
    scope = (
        store.kb_scope(scope_id)
        if scope_type == "knowledge_base"
        else store.project_scope(scope_id)
    )
    folder_id = str(uuid.uuid4())
    now = _now()
    with _scope_lock(scope):
        conn = rag_db.get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute(
                "SELECT 1 FROM linked_folder_retired_scopes WHERE scope=?", (scope,)
            ).fetchone():
                raise ValueError("The linked-folder scope no longer exists")
            normalized_key = _path_key(normalized)
            existing = conn.execute(
                "SELECT * FROM linked_folders WHERE scope=?", (scope,)
            ).fetchall()
            for row in existing:
                existing_key = _path_key(row["path"])
                if existing_key == normalized_key or _same_file(row["path"], normalized):
                    if row["status"] == "retired" or row["delete_remove_index"] is not None:
                        raise ValueError("Linked folder is still being removed")
                    conn.rollback()
                    return _reauthorize_folder(row["id"], normalized, expected_identity)
                if _paths_overlap(existing_key, normalized_key):
                    raise ValueError("Linked folders in the same scope cannot overlap")
            try:
                current_identity = _root_identity(normalized)
            except RuntimeError as exc:
                raise ValueError(str(exc)) from exc
            if current_identity != (root_device, root_inode) or (
                expected_identity is not None and current_identity != expected_identity
            ):
                raise ValueError("Linked folder changed after it was selected")
            conn.execute(
                "INSERT INTO linked_folders(id, scope_type, scope_id, scope, path, name, "
                "root_device, root_inode, auto_sync, status, created_at, updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    folder_id,
                    scope_type,
                    scope_id,
                    scope,
                    normalized,
                    (name or Path(normalized).name or normalized).strip(),
                    *_store_identity((root_device, root_inode)),
                    int(auto_sync),
                    "pending",
                    now,
                    now,
                ),
            )
            conn.commit()
            return dict(
                conn.execute("SELECT * FROM linked_folders WHERE id=?", (folder_id,)).fetchone()
            )
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def _reauthorize_folder(
    folder_id: str, path: str, expected_identity: tuple[int, int] | None
) -> dict:
    with _folder_lock(folder_id):
        conn = rag_db.get_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            if (
                conn.execute("SELECT 1 FROM linked_folders WHERE id=?", (folder_id,)).fetchone()
                is None
            ):
                raise ValueError("Linked folder changed while it was reauthorized")
            try:
                identity = _root_identity(path)
            except RuntimeError as exc:
                raise ValueError(str(exc)) from exc
            if expected_identity is not None and identity != expected_identity:
                raise ValueError("Linked folder changed after it was selected")
            conn.execute(
                "UPDATE linked_folders SET root_device=?, root_inode=?, updated_at=? WHERE id=?",
                (*_store_identity(identity), _now(), folder_id),
            )
            conn.commit()
            return dict(
                conn.execute("SELECT * FROM linked_folders WHERE id=?", (folder_id,)).fetchone()
            )
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def create_folder_with_sync(
    *,
    scope_type: str,
    scope_id: str,
    path: str,
    expected_identity: tuple[int, int] | None = None,
    name: str | None = None,
    auto_sync: bool = True,
) -> tuple[dict, str]:
    if scope_type not in {"knowledge_base", "project"}:
        raise ValueError("Linked folders support only knowledge-base and project scopes")
    scope = (
        store.kb_scope(scope_id)
        if scope_type == "knowledge_base"
        else store.project_scope(scope_id)
    )
    with _scope_lock(scope):
        folder = create_folder(
            scope_type = scope_type,
            scope_id = scope_id,
            path = path,
            expected_identity = expected_identity,
            name = name,
            auto_sync = auto_sync,
        )
        return folder, request_sync(folder["id"])


def get_folder(folder_id: str) -> dict | None:
    with closing(rag_db.get_connection()) as conn:
        row = conn.execute("SELECT * FROM linked_folders WHERE id=?", (folder_id,)).fetchone()
        return dict(row) if row else None


def list_folders(scope: str) -> list[dict]:
    with closing(rag_db.get_connection()) as conn:
        rows = conn.execute(
            "SELECT f.*, COUNT(ff.relative_path) AS file_count, (SELECT j.id FROM "
            "linked_folder_sync_jobs j WHERE j.folder_id=f.id AND j.status IN ('pending','running') "
            "ORDER BY j.created_at LIMIT 1) AS active_job_id FROM linked_folders f "
            "LEFT JOIN linked_folder_files ff ON ff.folder_id=f.id WHERE f.scope=? "
            "GROUP BY f.id ORDER BY f.created_at",
            (scope,),
        ).fetchall()
        return [dict(row) for row in rows]


def update_folder(
    folder_id: str,
    *,
    name: str | None = None,
    auto_sync: bool | None = None,
) -> dict:
    with _folder_lock(folder_id):
        return _update_folder(folder_id, name = name, auto_sync = auto_sync)


def _update_folder(
    folder_id: str,
    *,
    name: str | None = None,
    auto_sync: bool | None = None,
) -> dict:
    with closing(rag_db.get_connection()) as conn:
        row = conn.execute(
            "SELECT status, delete_remove_index FROM linked_folders WHERE id=?", (folder_id,)
        ).fetchone()
        if row is None or row["status"] == "retired" or row["delete_remove_index"] is not None:
            raise KeyError(folder_id)
        if name is not None:
            clean_name = name.strip()
            if not clean_name:
                raise ValueError("Folder name cannot be empty")
            conn.execute(
                "UPDATE linked_folders SET name=?, updated_at=? WHERE id=?",
                (clean_name, _now(), folder_id),
            )
        if auto_sync is not None:
            conn.execute(
                "UPDATE linked_folders SET auto_sync=?, updated_at=? WHERE id=?",
                (int(auto_sync), _now(), folder_id),
            )
        conn.commit()
        return dict(
            conn.execute("SELECT * FROM linked_folders WHERE id=?", (folder_id,)).fetchone()
        )


def _remove_snapshot(path: str | None) -> None:
    if not path:
        return
    try:
        root = os.path.realpath(str(rag_uploads_root()))
        target = os.path.realpath(path)
        if _is_within(root, target) and os.path.isfile(target):
            os.remove(target)
    except Exception:
        logger.warning("failed to remove linked-folder snapshot", exc_info = True)


def _remove_retired_snapshot(path: str | None) -> None:
    """Remove a managed snapshot or raise so its retirement remains retryable."""
    if not path:
        return
    root = os.path.realpath(str(rag_uploads_root()))
    target = os.path.realpath(path)
    if not _is_within(root, target):
        return
    try:
        os.remove(target)
    except FileNotFoundError:
        pass


def _delete_retired_folder(folder_id: str) -> bool | None:
    """Finish a durable unlink once no process still owns its sync job."""
    conn = rag_db.get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        folder = conn.execute(
            "SELECT delete_remove_index FROM linked_folders "
            "WHERE id=? AND delete_remove_index IS NOT NULL",
            (folder_id,),
        ).fetchone()
        if folder is None:
            conn.rollback()
            return None
        if conn.execute(
            "SELECT 1 FROM linked_folder_sync_jobs j JOIN rag_job_leases l "
            "ON l.kind=? AND l.job_id=j.id WHERE j.folder_id=? AND l.expires_at>? LIMIT 1",
            (job_leases.FOLDER_SYNC, folder_id, _now()),
        ).fetchone():
            conn.rollback()
            return False
        docs = conn.execute(
            "SELECT d.id, d.stored_path FROM linked_folder_files ff "
            "JOIN documents d ON d.id=ff.document_id WHERE ff.folder_id=?",
            (folder_id,),
        ).fetchall()
        if folder["delete_remove_index"]:
            for snapshot in dict.fromkeys(doc["stored_path"] for doc in docs):
                _remove_retired_snapshot(snapshot)
        conn.execute("DELETE FROM linked_folder_files WHERE folder_id=?", (folder_id,))
        for doc in docs:
            if folder["delete_remove_index"]:
                store.delete_document(conn, doc["id"], commit = False)
            else:
                conn.execute(
                    "UPDATE documents SET linked_folder_id=NULL, linked_relative_path=NULL "
                    "WHERE id=?",
                    (doc["id"],),
                )
        conn.execute(
            "DELETE FROM rag_job_leases WHERE kind=? AND job_id IN "
            "(SELECT id FROM linked_folder_sync_jobs WHERE folder_id=?)",
            (job_leases.FOLDER_SYNC, folder_id),
        )
        conn.execute("DELETE FROM linked_folder_sync_jobs WHERE folder_id=?", (folder_id,))
        conn.execute("DELETE FROM linked_folders WHERE id=?", (folder_id,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return True


def _reconcile_retired_folder_deletions() -> None:
    with closing(rag_db.get_connection()) as conn:
        folder_ids = [
            row["id"]
            for row in conn.execute(
                "SELECT id FROM linked_folders WHERE delete_remove_index IS NOT NULL"
            )
        ]
    for folder_id in folder_ids:
        _delete_retired_folder(folder_id)


def delete_folder(folder_id: str, *, remove_index: bool = True) -> bool:
    conn = rag_db.get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("SELECT 1 FROM linked_folders WHERE id=?", (folder_id,)).fetchone() is None:
            conn.rollback()
            return False
        conn.execute(
            "UPDATE linked_folders SET auto_sync=0, status='retired', "
            "delete_remove_index=COALESCE(delete_remove_index, ?), updated_at=? WHERE id=?",
            (int(remove_index), _now(), folder_id),
        )
        conn.execute(
            "UPDATE linked_folder_sync_jobs SET status='failed', stage='error', "
            "error='Linked folder was removed', completed_at=? "
            "WHERE folder_id=? AND status IN ('pending','running')",
            (_now(), folder_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    while _delete_retired_folder(folder_id) is False:
        time.sleep(0.05)
    return True


def _retirement_connection():
    conn = rag_db.get_metadata_connection()
    try:
        with _retirement_schema_lock:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS linked_folder_retired_scopes ("
                "scope TEXT NOT NULL PRIMARY KEY, retired_at TEXT NOT NULL, purged_at TEXT)"
            )
            rag_db.ensure_linked_folder_columns(conn)
            conn.commit()
    except Exception:
        conn.close()
        raise
    return conn


def _metadata_table_exists(conn, table: str) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        is not None
    )


def _retire_scope_rows(
    conn,
    scope: str,
    linked_before: str | None = None,
) -> None:
    folders_exist = _metadata_table_exists(conn, "linked_folders")
    jobs_exist = folders_exist and _metadata_table_exists(conn, "linked_folder_sync_jobs")
    conn.execute(
        "INSERT OR IGNORE INTO linked_folder_retired_scopes(scope, retired_at) VALUES(?, ?)",
        (scope, _now()),
    )
    # the ownership check and this write cannot share a transaction across two databases
    bound = "" if linked_before is None else " AND created_at<=?"
    params = () if linked_before is None else (linked_before,)
    if folders_exist:
        conn.execute(
            "UPDATE linked_folders SET auto_sync=0, status='retired', "
            f"last_error='Owning scope was removed', updated_at=? WHERE scope=?{bound}",
            (_now(), scope, *params),
        )
    if jobs_exist:
        conn.execute(
            "UPDATE linked_folder_sync_jobs SET status='failed', stage='error', "
            "error='Owning scope was removed', successor_kind=NULL, completed_at=? "
            "WHERE folder_id IN (SELECT id FROM linked_folders WHERE scope=?"
            f"{bound}) AND (status IN ('pending','running') OR successor_kind IS NOT NULL)",
            (_now(), scope, *params),
        )


def retire_scope(scope: str, linked_before: str | None = None) -> None:
    """Stop all future work, even when the vector extension cannot load."""
    conn = _retirement_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        _retire_scope_rows(conn, scope, linked_before)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def retire_and_delete_kb(kb_id: str) -> bool:
    """Atomically retire a KB scope and delete its owner row.

    Document rows remain as the durable cleanup queue until
    ``delete_retired_scope`` removes their stored files and database state.
    """
    scope = store.kb_scope(kb_id)
    conn = rag_db.get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        if store.get_kb(conn, kb_id) is None:
            conn.rollback()
            return False
        _retire_scope_rows(conn, scope)
        store.delete_kb(conn, kb_id, commit = False, delete_documents = False)
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def delete_retired_scope(scope: str) -> bool:
    """Purge an ownerless scope and retain its tombstone permanently.

    Scope identifiers are treated as non-reusable lifecycle IDs. Keeping the small
    tombstone closes late cross-database upload/link races without a distributed transaction.
    File removal happens before database rows are discarded, so any failure remains
    retryable from durable metadata.
    """
    conn = rag_db.get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        if (
            conn.execute(
                "SELECT 1 FROM linked_folder_retired_scopes WHERE scope=? AND purged_at IS NULL",
                (scope,),
            ).fetchone()
            is None
        ):
            conn.rollback()
            return False
        live_ingestion = conn.execute(
            "SELECT 1 FROM ingestion_jobs j JOIN rag_job_leases l "
            "ON l.kind=? AND l.job_id=j.id WHERE j.scope=? "
            "AND j.status IN ('pending','running') AND l.expires_at>? LIMIT 1",
            (job_leases.INGESTION, scope, _now()),
        ).fetchone()
        if live_ingestion is not None:
            conn.rollback()
            return False
        live_folder_sync = conn.execute(
            "SELECT 1 FROM linked_folder_sync_jobs j JOIN linked_folders f "
            "ON f.id=j.folder_id JOIN rag_job_leases l "
            "ON l.kind=? AND l.job_id=j.id WHERE f.scope=? "
            "AND l.expires_at>? LIMIT 1",
            (job_leases.FOLDER_SYNC, scope, _now()),
        ).fetchone()
        if live_folder_sync is not None:
            conn.rollback()
            return False
        documents = conn.execute(
            "SELECT id, stored_path FROM documents WHERE scope=?", (scope,)
        ).fetchall()
        for stored_path in dict.fromkeys(document["stored_path"] for document in documents):
            _remove_retired_snapshot(stored_path)
        for document in documents:
            store.delete_document(conn, document["id"], commit = False)
        conn.execute(
            "DELETE FROM rag_job_leases WHERE kind=? AND job_id IN "
            "(SELECT id FROM ingestion_jobs WHERE scope=?)",
            (job_leases.INGESTION, scope),
        )
        conn.execute("DELETE FROM ingestion_jobs WHERE scope=?", (scope,))
        conn.execute(
            "DELETE FROM linked_folder_files WHERE folder_id IN "
            "(SELECT id FROM linked_folders WHERE scope=?)",
            (scope,),
        )
        conn.execute(
            "DELETE FROM rag_job_leases WHERE kind=? AND job_id IN "
            "(SELECT j.id FROM linked_folder_sync_jobs j JOIN linked_folders f "
            "ON f.id=j.folder_id WHERE f.scope=?)",
            (job_leases.FOLDER_SYNC, scope),
        )
        conn.execute(
            "DELETE FROM linked_folder_sync_jobs WHERE folder_id IN "
            "(SELECT id FROM linked_folders WHERE scope=?)",
            (scope,),
        )
        conn.execute("DELETE FROM linked_folders WHERE scope=?", (scope,))
        conn.execute(
            "UPDATE linked_folder_retired_scopes SET purged_at=? WHERE scope=?",
            (_now(), scope),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return True


def reconcile_retired_scopes(project_exists) -> dict[str, list[str]]:
    """Retire orphaned project scopes and reap ownerless retired scopes."""
    with closing(rag_db.get_connection()) as conn:
        retired_scopes = {
            row["scope"]
            for row in conn.execute(
                "SELECT scope FROM linked_folder_retired_scopes WHERE purged_at IS NULL"
            )
        }
        project_scopes = {
            row["scope"]
            for row in conn.execute(
                "SELECT scope FROM linked_folders WHERE scope_type='project' "
                "UNION SELECT scope FROM documents WHERE project_id IS NOT NULL"
            )
        }
    retired: list[str] = []
    deleted: list[str] = []
    restored: list[str] = []
    for scope in sorted(project_scopes - retired_scopes):
        project_id = scope.removeprefix("project_")
        if not project_id:
            continue
        try:
            # rechecked inside the lock create_folder takes: a project recreated with the same id must not have
            # its new folders retired
            with _scope_lock(scope):
                checked_at = _now()
                if project_exists(project_id):
                    continue
                retire_scope(scope, checked_at)
            retired_scopes.add(scope)
            retired.append(scope)
        except Exception:
            logger.warning("failed to retire orphaned RAG scope %s", scope, exc_info = True)
    for scope in sorted(retired_scopes):
        try:
            if scope.startswith("project_"):
                project_id = scope[len("project_") :]
                if not project_id:
                    continue
                owner_exists = project_exists(project_id)
            elif scope.startswith("kb_"):
                kb_id = scope[len("kb_") :]
                if not kb_id:
                    continue
                with closing(rag_db.get_connection()) as owner_conn:
                    owner_exists = store.get_kb(owner_conn, kb_id) is not None
            else:
                continue
            if owner_exists:
                # retirement only ever meant "ownerless", so a returned id must not keep gating its owner
                if unretire_scope(scope):
                    restored.append(scope)
            elif delete_retired_scope(scope):
                deleted.append(scope)
        except Exception:
            logger.warning("failed to reconcile retired RAG scope %s", scope, exc_info = True)
    if retired:
        logger.info("retired %s orphaned RAG scope(s)", len(retired))
    if deleted:
        logger.info("deleted %s retired RAG scope(s)", len(deleted))
    if restored:
        logger.info("restored %s recreated RAG scope(s)", len(restored))
    return {"retired": retired, "deleted": deleted, "restored": restored}


def unretire_scope(scope: str) -> bool:
    """Drop the tombstone of a scope whose owner exists again, so it can be used."""
    with _scope_lock(scope):
        conn = _retirement_connection()
        try:
            conn.execute("BEGIN IMMEDIATE")
            removed = conn.execute(
                "DELETE FROM linked_folder_retired_scopes WHERE scope=?", (scope,)
            ).rowcount
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    return bool(removed)


def scope_retired(scope: str) -> bool:
    with _scope_lock(scope):
        with closing(_retirement_connection()) as conn:
            return (
                conn.execute(
                    "SELECT 1 FROM linked_folder_retired_scopes WHERE scope=?", (scope,)
                ).fetchone()
                is not None
            )


def scope_lock(scope: str) -> threading.RLock:
    return _scope_lock(scope)


def request_sync(folder_id: str, *, rebuild: bool = False) -> str:
    with _folder_lock(folder_id):
        return _request_sync(folder_id, rebuild = rebuild)


def _fold_into_active_job(conn, active, kind: str) -> None:
    """Absorb a new request into the job already queued or running for this folder.

    A pending job has not scanned yet, so it can simply become the requested kind. A
    running job already fixed its file list, so the request becomes a successor: the
    caller's changes are only guaranteed to land in the run that follows this one.
    """
    if active["status"] == "pending":
        if kind == "rebuild":
            conn.execute(
                "UPDATE linked_folder_sync_jobs SET kind='rebuild', successor_kind=NULL WHERE id=?",
                (active["id"],),
            )
        return
    # a queued rebuild outranks a queued sync; it covers everything a sync would
    conn.execute(
        "UPDATE linked_folder_sync_jobs SET successor_kind=? WHERE id=? "
        "AND (successor_kind IS NULL OR ?='rebuild')",
        (kind, active["id"], kind),
    )


def _request_sync(folder_id: str, *, rebuild: bool = False) -> str:
    job_id = str(uuid.uuid4())
    with closing(rag_db.get_connection()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        folder = conn.execute(
            "SELECT status, delete_remove_index FROM linked_folders WHERE id=?", (folder_id,)
        ).fetchone()
        if (
            folder is None
            or folder["status"] == "retired"
            or folder["delete_remove_index"] is not None
        ):
            raise KeyError(folder_id)
        active = conn.execute(
            "SELECT id, kind, status FROM linked_folder_sync_jobs WHERE folder_id=? "
            "AND status IN ('pending','running') ORDER BY created_at LIMIT 1",
            (folder_id,),
        ).fetchone()
        if active is not None:
            _fold_into_active_job(conn, active, "rebuild" if rebuild else "sync")
            conn.commit()
            return active["id"]
        conn.execute(
            "INSERT INTO linked_folder_sync_jobs(id, folder_id, kind, status, stage, created_at) "
            "VALUES(?,?,?,'pending','queued',?)",
            (job_id, folder_id, "rebuild" if rebuild else "sync", _now()),
        )
        conn.commit()
        _prune_terminal_jobs(conn)
    _wake.set()
    return job_id


def _prune_terminal_jobs(conn) -> None:
    limit = max(0, config.FOLDER_JOB_HISTORY_LIMIT)
    conn.execute(
        "DELETE FROM linked_folder_sync_jobs WHERE id IN ("
        "SELECT id FROM linked_folder_sync_jobs WHERE status IN ('completed','failed') "
        "AND successor_kind IS NULL "
        "ORDER BY completed_at DESC, created_at DESC LIMIT -1 OFFSET ?)",
        (limit,),
    )
    conn.commit()


def get_job(job_id: str) -> dict | None:
    with closing(rag_db.get_connection()) as conn:
        row = conn.execute("SELECT * FROM linked_folder_sync_jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None


def job_events(job_id: str):
    """Poll persisted state so streams also work after a backend restart.

    Yields None as a keepalive when a pass produces no change for a while, which is what
    lets a client tell a slow job from a proxy holding the whole response back.
    """
    previous = None
    last_sent = time.monotonic()
    while True:
        row = get_job(job_id)
        if row is None:
            return
        state = tuple(row.items())
        if state != previous:
            yield row
            previous = state
            last_sent = time.monotonic()
        elif time.monotonic() - last_sent >= _JOB_EVENT_KEEPALIVE_S:
            yield None
            last_sent = time.monotonic()
        if row["status"] in _TERMINAL:
            return
        time.sleep(0.5)


def _scan(
    root: str, expected_identity: tuple[int, int] | None = None
) -> tuple[dict[str, dict], tuple[int, int]]:
    from hub.storage.scan_folders import contains_sensitive_path_component, is_denied_system_path

    identity = _root_identity(root)
    if expected_identity is not None and identity != expected_identity:
        raise RuntimeError("Linked folder root identity changed")

    found: dict[str, dict] = {}
    mount_points = _mount_points()
    root_identities = frozenset({identity}) if identity[1] not in (None, 0) else frozenset()
    pending = [(root, frozenset({_path_key(root)}), root_identities, 0)]
    while pending:
        directory, ancestor_paths, ancestor_identities, depth = pending.pop()
        with os.scandir(directory) as entries:
            for entry in entries:
                full = entry.path
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks = False):
                    resolved = os.path.realpath(full)
                    if (
                        not _is_within(root, resolved)
                        or contains_sensitive_path_component(os.path.relpath(resolved, root))
                        or is_denied_system_path(resolved)
                    ):
                        continue
                    resolved_key = _path_key(resolved)
                    if resolved_key in ancestor_paths:
                        continue
                    directory_stat = entry.stat(follow_symlinks = False)
                    directory_identity = (directory_stat.st_dev, directory_stat.st_ino)
                    identity_usable = directory_identity[1] not in (None, 0)
                    if (
                        identity_usable
                        and directory_identity in ancestor_identities
                        and (resolved_key in mount_points or os.path.ismount(full))
                    ):
                        continue
                    if depth >= _MAX_FOLDER_DEPTH:
                        raise RuntimeError(
                            f"Folder nesting exceeds the {_MAX_FOLDER_DEPTH}-level scan limit"
                        )
                    child_identities = ancestor_identities
                    if identity_usable:
                        child_identities = ancestor_identities | {directory_identity}
                    pending.append(
                        (
                            full,
                            ancestor_paths | {resolved_key},
                            child_identities,
                            depth + 1,
                        )
                    )
                    continue
                if not entry.is_file(follow_symlinks = False):
                    continue
                if os.path.splitext(entry.name)[1].lower() not in config.UPLOAD_EXTS:
                    continue
                # Finder metadata carries the document's extension, so a text parser would embed and cite it as a
                # real chunk.
                if is_appledouble_metadata(Path(full)):
                    continue
                st = entry.stat(follow_symlinks = False)
                from_path = False
                if st.st_ino in (None, 0):
                    # Windows DirEntry.stat leaves st_dev/st_ino at 0 (FindFirstFileW carries no file index); os.lstat
                    # fills both, else the stored identity is (0, 0) forever.
                    try:
                        st = os.lstat(full)
                        from_path = True
                    except OSError:
                        # A file that vanished mid-scan must reach _snapshot as a failure: a scan is never
                        # authoritative for
                        # deletion.
                        pass
                rel = os.path.relpath(full, root).replace(os.sep, "/")
                found[rel] = {
                    "path": full,
                    "size_bytes": st.st_size,
                    "mtime_ns": st.st_mtime_ns,
                    "device": st.st_dev,
                    "inode": st.st_ino,
                    # A recovered identity is comparable to the next scan's but not to os.fstat's: shared-folder and
                    # WebDAV drivers report different ids per call path.
                    "identity_from_path": from_path,
                }
                if config.FOLDER_MAX_FILES and len(found) > config.FOLDER_MAX_FILES:
                    raise RuntimeError(
                        f"Folder contains more than the {config.FOLDER_MAX_FILES} supported files limit"
                    )
    if _root_identity(root) != identity:
        raise RuntimeError("Linked folder root identity changed during scan")
    return found, identity


def _snapshot(root: str, metadata: dict) -> str:
    source = metadata["path"]
    resolved = os.path.realpath(source)
    if not _is_within(root, resolved):
        raise RuntimeError("File escaped the linked folder")
    # os.fdopen already forces this descriptor binary on Windows; O_BINARY only guards a raw os.read.
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    fd = os.open(source, flags)
    target = None
    try:
        ext = os.path.splitext(source)[1].lower()
        target = ensure_dir(rag_uploads_root()) / f"linked-{uuid.uuid4().hex}{ext}"
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError("Linked source is not a regular file")
        expected = (
            metadata["size_bytes"],
            metadata["mtime_ns"],
            metadata["device"],
            metadata["inode"],
        )
        actual = (before.st_size, before.st_mtime_ns, before.st_dev, before.st_ino)
        # Only an identity os.fstat can reproduce is comparable: os.scandir reports none on Windows, and the
        # lstat fallback disagrees with fstat on file systems without stable ids.
        usable = metadata["inode"] not in (None, 0) and not metadata.get("identity_from_path")
        compared = 4 if usable else 2
        if actual[:compared] != expected[:compared]:
            raise RuntimeError("Linked source changed during reconciliation")
        if config.MAX_UPLOAD_BYTES and before.st_size > config.MAX_UPLOAD_BYTES:
            raise RuntimeError("Linked source exceeds the RAG file size limit")
        with os.fdopen(fd, "rb", closefd = False) as src, open(target, "xb") as dst:
            _copy_exact(src, dst, before.st_size)
        after = os.fstat(fd)
        # Both sides are fstat here, so the identity is always comparable.
        if (after.st_size, after.st_mtime_ns, after.st_dev, after.st_ino) != actual:
            raise RuntimeError("Linked source changed while it was copied")
        return str(target)
    except Exception:
        if target is not None:
            _remove_snapshot(str(target))
        raise
    finally:
        os.close(fd)


def _copy_exact(source, target, size: int) -> None:
    remaining = size
    while remaining:
        block = source.read(min(1 << 20, remaining))
        if not block:
            raise RuntimeError("Linked source changed while it was copied")
        target.write(block)
        remaining -= len(block)
    if source.read(1):
        raise RuntimeError("Linked source changed while it was copied")


def _check_running() -> None:
    stop_event = getattr(_worker_state, "stop_event", _stop)
    if stop_event.is_set():
        raise _SyncStopped
    job_id = getattr(_worker_state, "job_id", None)
    if job_id is None:
        return
    with closing(rag_db.get_connection()) as conn:
        if not job_leases.renew_owned(conn, job_leases.FOLDER_SYNC, job_id):
            conn.rollback()
            raise _LeaseLost
        folder = conn.execute(
            "SELECT f.status, f.delete_remove_index FROM linked_folder_sync_jobs j "
            "JOIN linked_folders f ON f.id=j.folder_id WHERE j.id=?",
            (job_id,),
        ).fetchone()
        if (
            folder is None
            or folder["status"] == "retired"
            or folder["delete_remove_index"] is not None
        ):
            conn.commit()
            raise _LeaseLost
        conn.commit()


def _check_root_identity(root: str, expected: tuple[int, int]) -> None:
    try:
        current = _root_identity(root)
    except RuntimeError as exc:
        raise _FolderChanged(str(exc)) from exc
    if current != expected:
        raise _FolderChanged("Linked folder root identity changed during reconciliation")


def _source_reappeared(root: str, relative_path: str) -> bool:
    """Check scanner eligibility without following a replaced path component."""
    path = PurePosixPath(relative_path)
    parts = path.parts
    if (
        not parts
        or path.is_absolute()
        or "\x00" in relative_path
        or any(part == ".." for part in parts)
    ):
        raise _FolderChanged("Linked folder mapping has an invalid relative path")
    current = root
    for index, part in enumerate(parts):
        current = os.path.join(current, part)
        try:
            current_stat = os.lstat(current)
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise _FolderChanged("Linked source could not be rechecked") from exc
        if stat.S_ISLNK(current_stat.st_mode):
            return False
        if index < len(parts) - 1:
            if not stat.S_ISDIR(current_stat.st_mode):
                return False
        else:
            return stat.S_ISREG(current_stat.st_mode)
    return False


def _set_job(job_id: str, **values) -> None:
    if not values:
        return
    with closing(rag_db.get_connection()) as conn:
        columns = ", ".join(f"{key}=?" for key in values)
        conn.execute(
            f"UPDATE linked_folder_sync_jobs SET {columns} WHERE id=?",
            (*values.values(), job_id),
        )
        conn.commit()


def _discard_document(document_id: str) -> None:
    conn = rag_db.get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        doc = store.get_document(conn, document_id)
        if doc is not None:
            _remove_retired_snapshot(doc.get("stored_path"))
            store.delete_document(conn, document_id, commit = False)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _install_mapping(
    folder: dict, rel: str, metadata: dict, document_id: str, content_hash: str
) -> None:
    replaced: list[str] = []
    conn = rag_db.get_connection()
    try:
        old_rows = conn.execute(
            "SELECT ff.document_id, d.stored_path FROM linked_folder_files ff "
            "LEFT JOIN documents d ON d.id=ff.document_id "
            "WHERE ff.folder_id=? AND ff.relative_path=?",
            (folder["id"], rel),
        ).fetchall()
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO linked_folder_files(folder_id, relative_path, size_bytes, mtime_ns, "
            "device, inode, document_id, synced_at, content_hash) VALUES(?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(folder_id, relative_path) DO UPDATE SET size_bytes=excluded.size_bytes, "
            "mtime_ns=excluded.mtime_ns, device=excluded.device, inode=excluded.inode, "
            "document_id=excluded.document_id, synced_at=excluded.synced_at, "
            "content_hash=excluded.content_hash",
            (
                folder["id"],
                rel,
                metadata["size_bytes"],
                metadata["mtime_ns"],
                *_file_identity(metadata),
                document_id,
                _now(),
                content_hash,
            ),
        )
        for old in old_rows:
            if old["document_id"] != document_id:
                store.delete_document(conn, old["document_id"], commit = False)
                replaced.append(old["stored_path"])
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    # Only after the commit: a rollback here restores a live searchable document, unlike the delete
    # paths where the surviving row is the retry queue.
    for stored_path in replaced:
        _remove_snapshot(stored_path)


def _update_mapping_metadata(folder_id: str, rel: str, metadata: dict, content_hash: str) -> None:
    with closing(rag_db.get_connection()) as conn:
        conn.execute(
            "UPDATE linked_folder_files SET size_bytes=?, mtime_ns=?, device=?, inode=?, "
            "content_hash=?, synced_at=? WHERE folder_id=? AND relative_path=?",
            (
                metadata["size_bytes"],
                metadata["mtime_ns"],
                *_file_identity(metadata),
                content_hash,
                _now(),
                folder_id,
                rel,
            ),
        )
        conn.commit()


def _delete_mapping(folder_id: str, rel: str) -> None:
    conn = rag_db.get_connection()
    try:
        row = conn.execute(
            "SELECT ff.document_id, d.stored_path FROM linked_folder_files ff "
            "LEFT JOIN documents d ON d.id=ff.document_id "
            "WHERE ff.folder_id=? AND ff.relative_path=?",
            (folder_id, rel),
        ).fetchone()
        if row is None:
            return
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "DELETE FROM linked_folder_files WHERE folder_id=? AND relative_path=?",
            (folder_id, rel),
        )
        store.delete_document(conn, row["document_id"], commit = False)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    # After the commit: a rollback restores a live document, and an auto_sync=0 folder may not reconcile
    # again for a long time.
    _remove_snapshot(row["stored_path"])


def _rename_mapping(folder_id: str, old_rel: str, new_rel: str) -> None:
    with closing(rag_db.get_connection()) as conn:
        row = conn.execute(
            "SELECT document_id FROM linked_folder_files WHERE folder_id=? AND relative_path=?",
            (folder_id, old_rel),
        ).fetchone()
        if row is None:
            return
        conn.execute(
            "UPDATE linked_folder_files SET relative_path=?, synced_at=? "
            "WHERE folder_id=? AND relative_path=?",
            (new_rel, _now(), folder_id, old_rel),
        )
        conn.execute(
            "UPDATE documents SET filename=?, linked_relative_path=? WHERE id=?",
            (new_rel, new_rel, row["document_id"]),
        )
        conn.commit()


def _load_withheld(folder: dict) -> set[str]:
    try:
        stored = json.loads(folder.get("withheld_paths") or "[]")
    except ValueError:
        return set()
    return {path for path in stored if isinstance(path, str)} if isinstance(stored, list) else set()


def _store_withheld(folder_id: str, withheld: set[str]) -> None:
    # truncation only shortens a grace pass, so the cap can never block a removal
    capped = sorted(withheld)[:_MAX_WITHHELD_PATHS]
    with closing(rag_db.get_connection()) as conn:
        conn.execute(
            "UPDATE linked_folders SET withheld_paths=? WHERE id=?",
            (json.dumps(capped) if capped else None, folder_id),
        )
        conn.commit()


def _failure_summary(failures: list[str]) -> str | None:
    """Name the files that could not be indexed, so the folder error is actionable."""
    if not failures:
        return None
    named = ", ".join(sorted(failures)[:_MAX_NAMED_FAILURES])
    remainder = len(failures) - _MAX_NAMED_FAILURES
    if remainder > 0:
        named = f"{named} and {remainder} more"
    return f"{len(failures)} file(s) could not be indexed ({named})"


def _reconcile_folder(job_id: str) -> None:
    """Run one complete reconciliation; called serially by the coordinator."""
    _check_running()
    with closing(rag_db.get_connection()) as conn:
        row = conn.execute("SELECT * FROM linked_folder_sync_jobs WHERE id=?", (job_id,)).fetchone()
        if (
            row is None
            or row["status"] != "running"
            or not job_leases.owned_by_this_process(conn, job_leases.FOLDER_SYNC, job_id)
        ):
            return
        job = dict(row)
    folder = get_folder(job["folder_id"])
    if folder is None or folder["status"] == "retired" or folder["delete_remove_index"] is not None:
        _set_job(
            job_id,
            status = "failed",
            stage = "error",
            error = "Linked folder not found" if folder is None else "Owning scope was removed",
            completed_at = _now(),
        )
        return
    embedding_model = config.effective_embedding_model()
    with closing(rag_db.get_connection()) as conn:
        conn.execute(
            "UPDATE linked_folders SET status='syncing', last_error=NULL, updated_at=? WHERE id=?",
            (_now(), folder["id"]),
        )
        conn.commit()
    try:
        _check_running()
        expected_identity = _load_identity(folder["root_device"], folder["root_inode"])
        current, scanned_identity = _scan(folder["path"], expected_identity)
        _check_running()
    except (_SyncStopped, _LeaseLost):
        raise
    except Exception as exc:
        # A partial/unavailable scan is never authoritative for deletion.
        error = _error_text(exc, folder["path"])
        _set_job(job_id, status = "failed", stage = "error", error = error, completed_at = _now())
        with closing(rag_db.get_connection()) as conn:
            conn.execute(
                "UPDATE linked_folders SET status='error', last_error=?, updated_at=? WHERE id=?",
                (error, _now(), folder["id"]),
            )
            conn.commit()
            _prune_terminal_jobs(conn)
        return

    with closing(rag_db.get_connection()) as conn:
        known = {
            row["relative_path"]: dict(row)
            for row in conn.execute(
                "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchall()
        }

    rebuild = job["kind"] == "rebuild"
    missing = set(known) - set(current)
    new = set(current) - set(known)
    renamed = 0
    materially_changed = {
        rel
        for rel in set(current) & set(known)
        if current[rel]["size_bytes"] != known[rel]["size_bytes"]
        or current[rel]["mtime_ns"] != known[rel]["mtime_ns"]
        or _file_identity(current[rel]) != (known[rel]["device"], known[rel]["inode"])
    }
    changed = set(current) & set(known) if rebuild else materially_changed
    work = sorted(new | changed)
    missing_by_content: dict[tuple[str, str], list[str]] = {}
    if not rebuild:
        for rel in sorted(missing):
            content_hash = known[rel].get("content_hash")
            if content_hash:
                key = (content_hash, os.path.splitext(rel)[1].lower())
                missing_by_content.setdefault(key, []).append(rel)
    already_withheld = _load_withheld(folder)
    total = len(work) + len(missing)
    _set_job(job_id, stage = "ingesting", discovered = len(current), renamed = renamed)
    added = changed_count = 0
    failures: list[str] = []
    withheld: set[str] = set()
    for index, rel in enumerate(work):
        snapshot = None
        document_id = None
        ingestion_job = None
        try:
            _check_running()
            metadata = current[rel]
            snapshot = _snapshot(folder["path"], metadata)
            content_hash = _hash_file(snapshot)
            _check_running()
            if not rebuild and rel in changed and content_hash == known[rel].get("content_hash"):
                _check_root_identity(folder["path"], scanned_identity)
                _update_mapping_metadata(folder["id"], rel, metadata, content_hash)
                _remove_snapshot(snapshot)
                snapshot = None
                _set_job(job_id, progress = (index + 1) / max(total, 1))
                continue
            if not rebuild and rel in new:
                rename_key = (content_hash, os.path.splitext(rel)[1].lower())
                rename_candidates = missing_by_content.get(rename_key)
                if rename_candidates:
                    old_rel = rename_candidates.pop(0)
                    missing.discard(old_rel)
                    _check_root_identity(folder["path"], scanned_identity)
                    _rename_mapping(folder["id"], old_rel, rel)
                    _update_mapping_metadata(folder["id"], rel, metadata, content_hash)
                    _remove_snapshot(snapshot)
                    snapshot = None
                    renamed += 1
                    _set_job(
                        job_id,
                        renamed = renamed,
                        progress = (index + 1) / max(total, 1),
                    )
                    continue
            document_id, ingestion_job = ingestion.start_ingestion(
                folder["scope"],
                folder["scope_id"] if folder["scope_type"] == "knowledge_base" else None,
                None,
                rel,
                snapshot,
                project_id = folder["scope_id"] if folder["scope_type"] == "project" else None,
                dedupe = False,
                linked_folder_id = folder["id"],
                linked_relative_path = rel,
                model_name = embedding_model,
                background = False,
                content_hash = content_hash,
            )
            result = ingestion.get_job_status(ingestion_job)
            if result is None:
                raise RuntimeError("Ingestion job disappeared")
            if result["status"] != "completed":
                raise RuntimeError(result.get("error") or "Ingestion failed")
            _check_running()
            _check_root_identity(folder["path"], scanned_identity)
            _install_mapping(
                folder,
                rel,
                metadata,
                document_id,
                content_hash,
            )
            if rel in new:
                added += 1
            else:
                changed_count += 1
        except (_SyncStopped, _LeaseLost):
            if document_id:
                _discard_document(document_id)
            else:
                _remove_snapshot(snapshot)
            raise
        except _FolderChanged:
            if document_id:
                _discard_document(document_id)
            else:
                _remove_snapshot(snapshot)
            raise
        except Exception:
            failures.append(rel)
            # a failed file may be a rename or copy of a vanished path, so grant one pass
            withheld.update(missing - already_withheld)
            logger.warning("linked-folder ingestion failed for %s", rel, exc_info = True)
            if document_id:
                _discard_document(document_id)
            else:
                _remove_snapshot(snapshot)
        finally:
            if ingestion_job:
                try:
                    ingestion.delete_terminal_job(ingestion_job)
                except Exception:
                    logger.warning(
                        "failed to prune linked-folder ingestion job %s",
                        ingestion_job,
                        exc_info = True,
                    )
        _set_job(
            job_id,
            added = added,
            changed = changed_count,
            failed = len(failures),
            progress = (index + 1) / max(total, 1),
        )

    deleted = 0
    for rel in sorted(missing - withheld):
        _check_running()
        _check_root_identity(folder["path"], scanned_identity)
        if _source_reappeared(folder["path"], rel):
            raise _FolderChanged("Linked source reappeared during reconciliation")
        _check_root_identity(folder["path"], scanned_identity)
        _delete_mapping(folder["id"], rel)
        deleted += 1
        _set_job(
            job_id,
            deleted = deleted,
            progress = (len(work) + deleted) / max(total, 1),
        )

    _check_running()
    _store_withheld(folder["id"], withheld)
    error = _failure_summary(failures)
    _set_job(
        job_id,
        status = "completed" if error is None else "failed",
        stage = "done" if error is None else "error",
        progress = 1.0,
        error = error,
        completed_at = _now(),
    )
    with closing(rag_db.get_connection()) as conn:
        conn.execute(
            "UPDATE linked_folders SET status=?, last_error=?, last_scan_at=?, updated_at=? "
            "WHERE id=? AND delete_remove_index IS NULL",
            ("ready" if error is None else "error", error, _now(), _now(), folder["id"]),
        )
        conn.commit()
        _prune_terminal_jobs(conn)


def _fail_job(job_id: str, exc: Exception) -> None:
    job = get_job(job_id)
    folder = get_folder(job["folder_id"]) if job is not None else None
    error = _error_text(exc, folder["path"] if folder else None)
    _set_job(job_id, status = "failed", stage = "error", error = error, completed_at = _now())
    if job is None:
        return
    with closing(rag_db.get_connection()) as conn:
        conn.execute(
            "UPDATE linked_folders SET status='error', last_error=?, updated_at=? WHERE id=?",
            (error, _now(), job["folder_id"]),
        )
        conn.commit()
        _prune_terminal_jobs(conn)


def _pause_job(job_id: str) -> None:
    job = get_job(job_id)
    if job is None:
        return
    _set_job(job_id, status = "pending", stage = "queued", started_at = None)
    with closing(rag_db.get_connection()) as conn:
        conn.execute(
            "UPDATE linked_folders SET status='ready', updated_at=? WHERE id=?",
            (_now(), job["folder_id"]),
        )
        conn.commit()


def _queue_successor(job_id: str) -> None:
    """Hand a finished job's queued follow-up request to the next run."""
    conn = rag_db.get_connection()
    queued = False
    try:
        conn.execute("BEGIN IMMEDIATE")
        job = conn.execute(
            "SELECT folder_id, successor_kind, status FROM linked_folder_sync_jobs WHERE id=?",
            (job_id,),
        ).fetchone()
        if job and job["successor_kind"] and job["status"] not in ("pending", "running"):
            active = conn.execute(
                "SELECT id, kind, status FROM linked_folder_sync_jobs WHERE folder_id=? "
                "AND status IN ('pending','running') ORDER BY created_at LIMIT 1",
                (job["folder_id"],),
            ).fetchone()
            if active is None:
                conn.execute(
                    "INSERT INTO linked_folder_sync_jobs"
                    "(id, folder_id, kind, status, stage, created_at) "
                    "VALUES(?,?,?,'pending','queued',?)",
                    (str(uuid.uuid4()), job["folder_id"], job["successor_kind"], _now()),
                )
            else:
                _fold_into_active_job(conn, active, job["successor_kind"])
            conn.execute(
                "UPDATE linked_folder_sync_jobs SET successor_kind=NULL WHERE id=?", (job_id,)
            )
            queued = True
        conn.commit()
        if queued:
            _prune_terminal_jobs(conn)
    finally:
        conn.close()
    if queued:
        _wake.set()


def reconcile_folder(job_id: str) -> None:
    """Reconcile and persist a terminal folder state for every unexpected failure."""
    if _claim_job(job_id) is None:
        # Drop an already-activated claim here, or the heartbeat renews it for the process lifetime and an
        # unlink waits on a job that will never run.
        job_leases.release(job_leases.FOLDER_SYNC, job_id)
        return
    lease_lost = False
    _worker_state.job_id = job_id
    try:
        _reconcile_folder(job_id)
    except _LeaseLost:
        lease_lost = True
    except _SyncStopped:
        _pause_job(job_id)
    except Exception as exc:
        logger.exception("linked-folder job %s failed unexpectedly", job_id)
        _fail_job(job_id, exc)
    finally:
        try:
            if not lease_lost:
                _queue_successor(job_id)
        finally:
            del _worker_state.job_id
            job_leases.release(job_leases.FOLDER_SYNC, job_id)


def _enqueue_periodic() -> None:
    rag_db.reconcile_orphaned_ingestion_jobs()
    _reconcile_retired_folder_deletions()
    with closing(rag_db.get_connection()) as conn:
        now = _now()
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute(
            "SELECT id FROM linked_folders WHERE auto_sync=1 AND status!='retired' "
            "AND delete_remove_index IS NULL"
        ).fetchall()
        for row in rows:
            conn.execute(
                "INSERT OR IGNORE INTO linked_folder_sync_jobs"
                "(id, folder_id, kind, status, stage, created_at) "
                "VALUES(?,?,'sync','pending','queued',?)",
                (str(uuid.uuid4()), row["id"], now),
            )
        _reap_orphaned_documents(conn, now)
        conn.commit()
        _prune_terminal_jobs(conn)


def _claim_job(job_id: str) -> tuple[str, str] | None:
    with closing(rag_db.get_connection()) as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT id, folder_id, status FROM linked_folder_sync_jobs WHERE id=?", (job_id,)
        ).fetchone()
        if row is None or row["status"] not in ("pending", "running"):
            conn.rollback()
            return None
        if row["status"] == "running":
            if not job_leases.owned_by_this_process(
                conn, job_leases.FOLDER_SYNC, job_id
            ) and not job_leases.claim(conn, job_leases.FOLDER_SYNC, job_id):
                conn.rollback()
                return None
        else:
            if not job_leases.claim(conn, job_leases.FOLDER_SYNC, job_id):
                conn.rollback()
                return None
            conn.execute(
                "UPDATE linked_folder_sync_jobs SET status='running', stage='scanning', "
                "started_at=COALESCE(started_at, ?) WHERE id=? AND status='pending'",
                (_now(), job_id),
            )
        conn.commit()
    try:
        job_leases.activate(job_leases.FOLDER_SYNC, job_id)
    except Exception:
        with closing(rag_db.get_connection()) as conn:
            conn.execute(
                "UPDATE linked_folder_sync_jobs SET status='pending', stage='queued' WHERE id=?",
                (job_id,),
            )
            conn.commit()
        job_leases.release(job_leases.FOLDER_SYNC, job_id)
        raise
    return job_id, row["folder_id"]


def _next_job() -> tuple[str, str] | None:
    with closing(rag_db.get_connection()) as conn:
        row = conn.execute(
            "SELECT j.id FROM linked_folder_sync_jobs j WHERE j.status='pending' OR "
            "(j.status='running' AND NOT EXISTS (SELECT 1 FROM rag_job_leases l "
            "WHERE l.kind=? AND l.job_id=j.id AND l.expires_at>?)) "
            "ORDER BY CASE j.status WHEN 'running' THEN 0 ELSE 1 END, j.created_at LIMIT 1",
            (job_leases.FOLDER_SYNC, _now()),
        ).fetchone()
    return _claim_job(row["id"]) if row else None


def _worker(stop_event: threading.Event | None = None, project_exists = None) -> None:
    global _thread, _thread_stop
    stop_event = stop_event or _stop
    try:
        with _worker_lock:
            _worker_state.stop_event = stop_event
            while not stop_event.is_set():
                try:
                    _recover_startup_state()
                    if project_exists is not None:
                        reconcile_retired_scopes(project_exists)
                    _enqueue_periodic()
                    break
                except Exception:
                    logger.warning("linked-folder worker initialization failed", exc_info = True)
                    stop_event.wait(1.0)
            while not stop_event.is_set():
                try:
                    job = _next_job()
                except Exception:
                    # Writer-lock contention must not retire the only worker; the initialization and periodic paths
                    # retry.
                    logger.warning("linked-folder queue selection failed", exc_info = True)
                    stop_event.wait(1.0)
                    continue
                if job:
                    job_id, folder_id = job
                    try:
                        reconcile_folder(job_id)
                    except Exception as exc:
                        logger.exception("linked-folder job %s failed unexpectedly", job_id)
                        _fail_job(job_id, exc)
                    continue
                _wake.wait(max(1.0, config.FOLDER_SYNC_INTERVAL_S))
                _wake.clear()
                if not stop_event.is_set():
                    try:
                        if project_exists is not None:
                            reconcile_retired_scopes(project_exists)
                        _enqueue_periodic()
                    except Exception:
                        logger.warning("linked-folder periodic scheduling failed", exc_info = True)
    finally:
        _worker_state.stop_event = None
        with _thread_lock:
            if _thread is threading.current_thread():
                _thread = None
                _thread_stop = None


def _recover_startup_state() -> None:
    with closing(rag_db.get_connection()) as conn:
        now = _now()
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "UPDATE linked_folder_sync_jobs SET status='pending', stage='queued', "
            "error=NULL WHERE status='running' AND NOT EXISTS ("
            "SELECT 1 FROM rag_job_leases l WHERE l.kind=? "
            "AND l.job_id=linked_folder_sync_jobs.id AND l.expires_at>?)",
            (job_leases.FOLDER_SYNC, now),
        )
        conn.execute("DELETE FROM rag_job_leases WHERE expires_at<=?", (now,))
        successor_handoffs = conn.execute(
            "SELECT id FROM linked_folder_sync_jobs WHERE status IN ('completed','failed') "
            "AND successor_kind IS NOT NULL"
        ).fetchall()
        _reap_orphaned_documents(conn, now)
        conn.commit()
    for job in successor_handoffs:
        _queue_successor(job["id"])


def _reap_orphaned_documents(conn, now: str) -> None:
    """Drop folder-owned documents left unreferenced by a crash before mapping.

    Also runs periodically: a process that survives the crash reclaims the
    expired job and reindexes the file, so its own startup pass is long past.
    Anything still leased by a live ingestion or folder sync is left alone.
    """
    orphans = conn.execute(
        "SELECT d.id, d.stored_path FROM documents d "
        "WHERE d.linked_folder_id IS NOT NULL AND NOT EXISTS "
        "(SELECT 1 FROM linked_folder_files ff WHERE ff.document_id=d.id) AND NOT EXISTS ("
        "SELECT 1 FROM ingestion_jobs j JOIN rag_job_leases l "
        "ON l.kind=? AND l.job_id=j.id WHERE j.document_id=d.id "
        "AND j.status IN ('pending','running') AND l.expires_at>?) AND NOT EXISTS ("
        "SELECT 1 FROM linked_folder_sync_jobs sj JOIN rag_job_leases sl "
        "ON sl.kind=? AND sl.job_id=sj.id WHERE sj.folder_id=d.linked_folder_id "
        "AND sl.expires_at>?)",
        (job_leases.INGESTION, now, job_leases.FOLDER_SYNC, now),
    ).fetchall()
    for orphan in orphans:
        _remove_retired_snapshot(orphan["stored_path"])
        store.delete_document(conn, orphan["id"], commit = False)
        conn.execute(
            "DELETE FROM rag_job_leases WHERE kind=? AND job_id IN "
            "(SELECT id FROM ingestion_jobs WHERE document_id=?)",
            (job_leases.INGESTION, orphan["id"]),
        )
        conn.execute("DELETE FROM ingestion_jobs WHERE document_id=?", (orphan["id"],))


def start_auto_sync(
    *,
    admission_lock = None,
    admit = None,
    project_exists = None,
) -> bool:
    global _thread, _thread_stop
    try:
        if not rag_db.rag_available():
            return False
    except sqlite3.OperationalError:
        # The worker retries initialization, so transient database contention must not turn a startup
        # preflight into a process-lifetime outage.
        pass
    with _thread_lock:
        retired = _thread if _thread is not None and _thread.is_alive() else None
        if retired is not None and not _stop.is_set():
            return False

    def launch() -> bool:
        global _thread, _thread_stop
        with _thread_lock:
            if _thread is not None and _thread.is_alive() and not _stop.is_set():
                return False
            stop_event = threading.Event()
            _stop.clear()
            _thread_stop = stop_event
            _thread = threading.Thread(
                target = _worker,
                args = (stop_event, project_exists),
                daemon = True,
                name = "rag-folder-sync",
            )
            _thread.start()
            return True

    if admission_lock is None:
        return launch()
    with admission_lock:
        if admit is not None and not admit():
            return False
        return launch()


def stop_auto_sync(timeout: float = 2.0) -> None:
    _stop.set()
    _wake.set()
    with _thread_lock:
        thread = _thread
        stop_event = _thread_stop
    if stop_event is not None:
        stop_event.set()
    if thread is not None:
        thread.join(timeout = timeout)
