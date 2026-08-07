# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Durable, sequential reconciliation for linked local RAG folders."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import stat
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from storage import rag_db
from utils.paths import ensure_dir, rag_uploads_root

from . import config, ingestion, store

logger = logging.getLogger(__name__)

_wake = threading.Event()
_stop = threading.Event()
_thread: threading.Thread | None = None
_thread_lock = threading.Lock()
_sync_lock = threading.RLock()
_TERMINAL = {"completed", "failed"}


class _SyncStopped(Exception):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    expanded = os.path.abspath(os.path.expanduser(path.strip()))
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


def create_folder(
    *,
    scope_type: str,
    scope_id: str,
    path: str,
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
    scope = (
        store.kb_scope(scope_id)
        if scope_type == "knowledge_base"
        else store.project_scope(scope_id)
    )
    folder_id = str(uuid.uuid4())
    now = _now()
    conn = rag_db.get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        normalized_key = _path_key(normalized)
        existing = conn.execute("SELECT * FROM linked_folders WHERE scope=?", (scope,)).fetchall()
        for row in existing:
            existing_key = _path_key(row["path"])
            if existing_key == normalized_key or _same_file(row["path"], normalized):
                conn.rollback()
                return dict(row)
            if _paths_overlap(existing_key, normalized_key):
                raise ValueError("Linked folders in the same scope cannot overlap")
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
                root_device,
                root_inode,
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


def get_folder(folder_id: str) -> dict | None:
    conn = rag_db.get_connection()
    try:
        row = conn.execute("SELECT * FROM linked_folders WHERE id=?", (folder_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_folders(scope: str) -> list[dict]:
    conn = rag_db.get_connection()
    try:
        rows = conn.execute(
            "SELECT f.*, COUNT(ff.relative_path) AS file_count, (SELECT j.id FROM "
            "linked_folder_sync_jobs j WHERE j.folder_id=f.id AND j.status IN ('pending','running') "
            "ORDER BY j.created_at LIMIT 1) AS active_job_id FROM linked_folders f "
            "LEFT JOIN linked_folder_files ff ON ff.folder_id=f.id WHERE f.scope=? "
            "GROUP BY f.id ORDER BY f.created_at",
            (scope,),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def update_folder(
    folder_id: str,
    *,
    name: str | None = None,
    auto_sync: bool | None = None,
) -> dict:
    conn = rag_db.get_connection()
    try:
        if conn.execute("SELECT 1 FROM linked_folders WHERE id=?", (folder_id,)).fetchone() is None:
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
    finally:
        conn.close()


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


def delete_folder(folder_id: str, *, remove_index: bool = True) -> bool:
    with _sync_lock:
        conn = rag_db.get_connection()
        snapshots: list[str] = []
        try:
            if (
                conn.execute("SELECT 1 FROM linked_folders WHERE id=?", (folder_id,)).fetchone()
                is None
            ):
                return False
            docs = conn.execute(
                "SELECT d.id, d.stored_path FROM linked_folder_files ff "
                "JOIN documents d ON d.id=ff.document_id WHERE ff.folder_id=?",
                (folder_id,),
            ).fetchall()
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM linked_folder_files WHERE folder_id=?", (folder_id,))
            for doc in docs:
                if remove_index:
                    store.delete_document(conn, doc["id"], commit = False)
                    if doc["stored_path"]:
                        snapshots.append(doc["stored_path"])
                else:
                    conn.execute(
                        "UPDATE documents SET linked_folder_id=NULL, linked_relative_path=NULL "
                        "WHERE id=?",
                        (doc["id"],),
                    )
            conn.execute("DELETE FROM linked_folders WHERE id=?", (folder_id,))
            conn.execute(
                "UPDATE linked_folder_sync_jobs SET status='failed', stage='error', "
                "error='Linked folder was removed', completed_at=? "
                "WHERE folder_id=? AND status IN ('pending','running')",
                (_now(), folder_id),
            )
            conn.commit()
        finally:
            conn.close()
        for snapshot in snapshots:
            _remove_snapshot(snapshot)
        return True


def request_sync(folder_id: str, *, rebuild: bool = False) -> str:
    job_id = str(uuid.uuid4())
    conn = rag_db.get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        if conn.execute("SELECT 1 FROM linked_folders WHERE id=?", (folder_id,)).fetchone() is None:
            raise KeyError(folder_id)
        active = conn.execute(
            "SELECT id, kind, status FROM linked_folder_sync_jobs WHERE folder_id=? "
            "AND status IN ('pending','running') ORDER BY created_at LIMIT 1",
            (folder_id,),
        ).fetchone()
        if active is not None:
            if rebuild and active["status"] == "pending":
                conn.execute(
                    "UPDATE linked_folder_sync_jobs SET kind='rebuild', rebuild_requested=0 "
                    "WHERE id=?",
                    (active["id"],),
                )
            elif rebuild and active["kind"] != "rebuild":
                conn.execute(
                    "UPDATE linked_folder_sync_jobs SET rebuild_requested=1 WHERE id=?",
                    (active["id"],),
                )
            elif rebuild:
                conn.execute(
                    "UPDATE linked_folder_sync_jobs SET rebuild_requested=0 WHERE id=?",
                    (active["id"],),
                )
            conn.commit()
            return active["id"]
        conn.execute(
            "INSERT INTO linked_folder_sync_jobs(id, folder_id, kind, status, stage, created_at) "
            "VALUES(?,?,?,'pending','queued',?)",
            (job_id, folder_id, "rebuild" if rebuild else "sync", _now()),
        )
        conn.commit()
        _prune_terminal_jobs(conn)
    finally:
        conn.close()
    _wake.set()
    return job_id


def _prune_terminal_jobs(conn) -> None:
    limit = max(0, config.FOLDER_JOB_HISTORY_LIMIT)
    conn.execute(
        "DELETE FROM linked_folder_sync_jobs WHERE id IN ("
        "SELECT id FROM linked_folder_sync_jobs WHERE status IN ('completed','failed') "
        "AND rebuild_requested=0 "
        "ORDER BY completed_at DESC, created_at DESC LIMIT -1 OFFSET ?)",
        (limit,),
    )
    conn.commit()


def get_job(job_id: str) -> dict | None:
    conn = rag_db.get_connection()
    try:
        row = conn.execute("SELECT * FROM linked_folder_sync_jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def job_events(job_id: str):
    """Poll persisted state so streams also work after a backend restart."""
    previous = None
    while True:
        row = get_job(job_id)
        if row is None:
            return
        state = tuple(row.items())
        if state != previous:
            yield row
            previous = state
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
    pending = [root]
    while pending:
        directory = pending.pop()
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
                    pending.append(full)
                    continue
                if not entry.is_file(follow_symlinks = False):
                    continue
                if os.path.splitext(entry.name)[1].lower() not in config.UPLOAD_EXTS:
                    continue
                st = entry.stat(follow_symlinks = False)
                rel = os.path.relpath(full, root).replace(os.sep, "/")
                found[rel] = {
                    "path": full,
                    "size_bytes": st.st_size,
                    "mtime_ns": st.st_mtime_ns,
                    "device": st.st_dev,
                    "inode": st.st_ino,
                }
                if config.FOLDER_MAX_FILES and len(found) > config.FOLDER_MAX_FILES:
                    raise RuntimeError(
                        f"Folder contains more than the {config.FOLDER_MAX_FILES} supported files limit"
                    )
    if _root_identity(root) != identity:
        raise RuntimeError("Linked folder root identity changed during scan")
    return found, identity


def _establish_root_identity(folder_id: str, identity: tuple[int, int]) -> None:
    """Claim a legacy folder identity without overriding a concurrent claim."""
    conn = rag_db.get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT root_device, root_inode FROM linked_folders WHERE id=?", (folder_id,)
        ).fetchone()
        if row is None:
            raise RuntimeError("Linked folder not found")
        persisted = (row["root_device"], row["root_inode"])
        if None in persisted:
            conn.execute(
                "UPDATE linked_folders SET root_device=?, root_inode=?, updated_at=? WHERE id=?",
                (*identity, _now(), folder_id),
            )
        elif persisted != identity:
            raise RuntimeError("Linked folder root identity changed")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _snapshot(root: str, metadata: dict) -> str:
    source = metadata["path"]
    resolved = os.path.realpath(source)
    if not _is_within(root, resolved):
        raise RuntimeError("File escaped the linked folder")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(source, flags)
    ext = os.path.splitext(source)[1].lower()
    target = ensure_dir(rag_uploads_root()) / f"linked-{uuid.uuid4().hex}{ext}"
    try:
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
        if actual != expected:
            raise RuntimeError("Linked source changed during reconciliation")
        if config.MAX_UPLOAD_BYTES and before.st_size > config.MAX_UPLOAD_BYTES:
            raise RuntimeError("Linked source exceeds the RAG file size limit")
        with os.fdopen(fd, "rb", closefd = False) as src, open(target, "xb") as dst:
            shutil.copyfileobj(src, dst, length = 1 << 20)
        after = os.fstat(fd)
        if (after.st_size, after.st_mtime_ns, after.st_dev, after.st_ino) != expected:
            raise RuntimeError("Linked source changed while it was copied")
        return str(target)
    except Exception:
        _remove_snapshot(str(target))
        raise
    finally:
        os.close(fd)


def _wait_ingestion(job_id: str) -> dict:
    while True:
        try:
            row = ingestion.get_job_status(job_id)
        except Exception:
            # A transient SQLite lock must not cause us to delete a document whose
            # ingestion worker is still active.
            logger.warning("linked-folder ingestion status read failed", exc_info = True)
            time.sleep(0.1)
            continue
        if row is None:
            raise RuntimeError("Ingestion job disappeared")
        if row["status"] in _TERMINAL:
            return row
        time.sleep(0.05)


def _check_running() -> None:
    if _stop.is_set():
        raise _SyncStopped


def _set_job(job_id: str, **values) -> None:
    if not values:
        return
    conn = rag_db.get_connection()
    try:
        columns = ", ".join(f"{key}=?" for key in values)
        conn.execute(
            f"UPDATE linked_folder_sync_jobs SET {columns} WHERE id=?",
            (*values.values(), job_id),
        )
        conn.commit()
    finally:
        conn.close()


def _discard_document(document_id: str) -> None:
    conn = rag_db.get_connection()
    try:
        doc = store.get_document(conn, document_id)
        if doc is not None:
            store.delete_document(conn, document_id)
            _remove_snapshot(doc.get("stored_path"))
    finally:
        conn.close()


def _install_mapping(
    folder: dict,
    rel: str,
    metadata: dict,
    document_id: str,
    content_hash: str,
    *,
    renamed_from: str | None = None,
) -> None:
    conn = rag_db.get_connection()
    old_paths: list[str] = []
    try:
        old_rows = conn.execute(
            "SELECT ff.document_id, d.stored_path FROM linked_folder_files ff "
            "LEFT JOIN documents d ON d.id=ff.document_id "
            "WHERE ff.folder_id=? AND ff.relative_path IN (?, ?)",
            (folder["id"], rel, renamed_from or rel),
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
                metadata["device"],
                metadata["inode"],
                document_id,
                _now(),
                content_hash,
            ),
        )
        if renamed_from and renamed_from != rel:
            conn.execute(
                "DELETE FROM linked_folder_files WHERE folder_id=? AND relative_path=?",
                (folder["id"], renamed_from),
            )
        for old in old_rows:
            if old["document_id"] != document_id:
                store.delete_document(conn, old["document_id"], commit = False)
                if old["stored_path"]:
                    old_paths.append(old["stored_path"])
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    for old_path in old_paths:
        _remove_snapshot(old_path)


def _update_mapping_metadata(folder_id: str, rel: str, metadata: dict, content_hash: str) -> None:
    conn = rag_db.get_connection()
    try:
        conn.execute(
            "UPDATE linked_folder_files SET size_bytes=?, mtime_ns=?, device=?, inode=?, "
            "content_hash=?, synced_at=? WHERE folder_id=? AND relative_path=?",
            (
                metadata["size_bytes"],
                metadata["mtime_ns"],
                metadata["device"],
                metadata["inode"],
                content_hash,
                _now(),
                folder_id,
                rel,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _delete_mapping(folder_id: str, rel: str) -> None:
    conn = rag_db.get_connection()
    old_path = None
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
        old_path = row["stored_path"]
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    _remove_snapshot(old_path)


def _rename_mapping(folder_id: str, old_rel: str, new_rel: str) -> None:
    conn = rag_db.get_connection()
    try:
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
    finally:
        conn.close()


def _reconcile_folder(job_id: str) -> None:
    """Run one complete reconciliation; called serially by the coordinator."""
    _check_running()
    conn = rag_db.get_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM linked_folder_sync_jobs WHERE id=?", (job_id,)).fetchone()
        if row is None or row["status"] not in ("pending", "running"):
            conn.rollback()
            return
        job = dict(row)
        conn.execute(
            "UPDATE linked_folder_sync_jobs SET status='running', stage='scanning', "
            "started_at=COALESCE(started_at, ?) WHERE id=?",
            (_now(), job_id),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    folder = get_folder(job["folder_id"])
    if folder is None:
        _set_job(
            job_id,
            status = "failed",
            stage = "error",
            error = "Linked folder not found",
            completed_at = _now(),
        )
        return
    conn = rag_db.get_connection()
    try:
        conn.execute(
            "UPDATE linked_folders SET status='syncing', last_error=NULL, updated_at=? WHERE id=?",
            (_now(), folder["id"]),
        )
        conn.commit()
    finally:
        conn.close()
    try:
        _check_running()
        expected_identity = None
        if folder.get("root_device") is not None and folder.get("root_inode") is not None:
            expected_identity = (folder["root_device"], folder["root_inode"])
        current, scanned_identity = _scan(folder["path"], expected_identity)
        _check_running()
        _establish_root_identity(folder["id"], scanned_identity)
    except _SyncStopped:
        raise
    except Exception as exc:
        # A partial/unavailable scan is never authoritative for deletion.
        error = _error_text(exc, folder["path"])
        _set_job(job_id, status = "failed", stage = "error", error = error, completed_at = _now())
        conn = rag_db.get_connection()
        try:
            conn.execute(
                "UPDATE linked_folders SET status='error', last_error=?, updated_at=? WHERE id=?",
                (error, _now(), folder["id"]),
            )
            conn.commit()
            _prune_terminal_jobs(conn)
        finally:
            conn.close()
        return

    conn = rag_db.get_connection()
    try:
        known = {
            row["relative_path"]: dict(row)
            for row in conn.execute(
                "SELECT * FROM linked_folder_files WHERE folder_id=?", (folder["id"],)
            ).fetchall()
        }
    finally:
        conn.close()

    rebuild = job["kind"] == "rebuild"
    missing = set(known) - set(current)
    new = set(current) - set(known)
    renamed = 0
    extension_renames: dict[str, str] = {}
    by_identity: dict[tuple, list[str]] = {}
    for rel in missing:
        old = known[rel]
        key = (old["device"], old["inode"], old["size_bytes"], old["mtime_ns"])
        by_identity.setdefault(key, []).append(rel)
    for rel in sorted(list(new)):
        _check_running()
        meta = current[rel]
        key = (meta["device"], meta["inode"], meta["size_bytes"], meta["mtime_ns"])
        candidates = by_identity.get(key, [])
        if len(candidates) == 1:
            old_rel = candidates.pop()
            missing.discard(old_rel)
            new.discard(rel)
            known[rel] = {**known.pop(old_rel), "relative_path": rel}
            same_extension = (
                os.path.splitext(old_rel)[1].lower() == os.path.splitext(rel)[1].lower()
            )
            same_content = False
            if same_extension and known[rel].get("content_hash"):
                snapshot = _snapshot(folder["path"], meta)
                try:
                    same_content = _hash_file(snapshot) == known[rel]["content_hash"]
                finally:
                    _remove_snapshot(snapshot)
            _check_running()
            if same_content:
                _rename_mapping(folder["id"], old_rel, rel)
                renamed += 1
            else:
                extension_renames[rel] = old_rel

    changed = {
        rel
        for rel in set(current) & set(known)
        if rebuild
        or rel in extension_renames
        or current[rel]["size_bytes"] != known[rel]["size_bytes"]
        or current[rel]["mtime_ns"] != known[rel]["mtime_ns"]
        or current[rel]["device"] != known[rel]["device"]
        or current[rel]["inode"] != known[rel]["inode"]
    }
    work = sorted(new | changed)
    total = len(work) + len(missing)
    _set_job(job_id, stage = "ingesting", discovered = len(current), renamed = renamed)
    added = changed_count = failed = 0
    for index, rel in enumerate(work):
        snapshot = None
        document_id = None
        try:
            _check_running()
            metadata = current[rel]
            snapshot = _snapshot(folder["path"], metadata)
            content_hash = _hash_file(snapshot)
            _check_running()
            if (
                not rebuild
                and rel not in extension_renames
                and rel in changed
                and content_hash == known[rel].get("content_hash")
            ):
                _update_mapping_metadata(folder["id"], rel, metadata, content_hash)
                _remove_snapshot(snapshot)
                snapshot = None
                _set_job(job_id, progress = (index + 1) / max(total, 1))
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
            )
            result = _wait_ingestion(ingestion_job)
            if result["status"] != "completed":
                raise RuntimeError(result.get("error") or "Ingestion failed")
            _check_running()
            _install_mapping(
                folder,
                rel,
                metadata,
                document_id,
                content_hash,
                renamed_from = extension_renames.get(rel),
            )
            if rel in new:
                added += 1
            else:
                changed_count += 1
        except _SyncStopped:
            if document_id:
                _discard_document(document_id)
            else:
                _remove_snapshot(snapshot)
            raise
        except Exception:
            failed += 1
            logger.warning("linked-folder ingestion failed for %s", rel, exc_info = True)
            if document_id:
                _discard_document(document_id)
            else:
                _remove_snapshot(snapshot)
        _set_job(
            job_id,
            added = added,
            changed = changed_count,
            failed = failed,
            progress = (index + 1) / max(total, 1),
        )

    deleted = 0
    for rel in sorted(missing):
        _check_running()
        _delete_mapping(folder["id"], rel)
        deleted += 1
        _set_job(
            job_id,
            deleted = deleted,
            progress = (len(work) + deleted) / max(total, 1),
        )

    _check_running()
    status = "completed" if failed == 0 else "failed"
    error = (
        None if failed == 0 else f"{failed} file(s) could not be indexed; prior versions retained"
    )
    _set_job(
        job_id,
        status = status,
        stage = "done" if failed == 0 else "error",
        progress = 1.0,
        error = error,
        completed_at = _now(),
    )
    conn = rag_db.get_connection()
    try:
        conn.execute(
            "UPDATE linked_folders SET status=?, last_error=?, last_scan_at=?, updated_at=? WHERE id=?",
            ("ready" if failed == 0 else "error", error, _now(), _now(), folder["id"]),
        )
        conn.commit()
        _prune_terminal_jobs(conn)
    finally:
        conn.close()


def _fail_job(job_id: str, exc: Exception) -> None:
    job = get_job(job_id)
    folder = get_folder(job["folder_id"]) if job is not None else None
    error = _error_text(exc, folder["path"] if folder else None)
    _set_job(job_id, status = "failed", stage = "error", error = error, completed_at = _now())
    if job is None:
        return
    conn = rag_db.get_connection()
    try:
        conn.execute(
            "UPDATE linked_folders SET status='error', last_error=?, updated_at=? WHERE id=?",
            (error, _now(), job["folder_id"]),
        )
        conn.commit()
        _prune_terminal_jobs(conn)
    finally:
        conn.close()


def _pause_job(job_id: str) -> None:
    job = get_job(job_id)
    if job is None:
        return
    _set_job(job_id, status = "pending", stage = "queued", started_at = None)
    conn = rag_db.get_connection()
    try:
        conn.execute(
            "UPDATE linked_folders SET status='ready', updated_at=? WHERE id=?",
            (_now(), job["folder_id"]),
        )
        conn.commit()
    finally:
        conn.close()


def _queue_requested_rebuild(job_id: str) -> None:
    conn = rag_db.get_connection()
    queued = False
    try:
        conn.execute("BEGIN IMMEDIATE")
        job = conn.execute(
            "SELECT folder_id, rebuild_requested, status FROM linked_folder_sync_jobs WHERE id=?",
            (job_id,),
        ).fetchone()
        if job and job["rebuild_requested"] and job["status"] not in ("pending", "running"):
            active = conn.execute(
                "SELECT id, kind, status FROM linked_folder_sync_jobs WHERE folder_id=? "
                "AND status IN ('pending','running') ORDER BY created_at LIMIT 1",
                (job["folder_id"],),
            ).fetchone()
            if active is None:
                conn.execute(
                    "INSERT INTO linked_folder_sync_jobs"
                    "(id, folder_id, kind, status, stage, created_at) "
                    "VALUES(?,?,'rebuild','pending','queued',?)",
                    (str(uuid.uuid4()), job["folder_id"], _now()),
                )
            elif active["status"] == "pending":
                conn.execute(
                    "UPDATE linked_folder_sync_jobs SET kind='rebuild', rebuild_requested=0 "
                    "WHERE id=?",
                    (active["id"],),
                )
            elif active["kind"] != "rebuild":
                conn.execute(
                    "UPDATE linked_folder_sync_jobs SET rebuild_requested=1 WHERE id=?",
                    (active["id"],),
                )
            conn.execute(
                "UPDATE linked_folder_sync_jobs SET rebuild_requested=0 WHERE id=?", (job_id,)
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
    try:
        _reconcile_folder(job_id)
    except _SyncStopped:
        _pause_job(job_id)
    except Exception as exc:
        logger.exception("linked-folder job %s failed unexpectedly", job_id)
        _fail_job(job_id, exc)
    finally:
        _queue_requested_rebuild(job_id)


def _enqueue_periodic() -> None:
    conn = rag_db.get_connection()
    try:
        now = _now()
        conn.execute("BEGIN IMMEDIATE")
        rows = conn.execute("SELECT id FROM linked_folders WHERE auto_sync=1").fetchall()
        for row in rows:
            conn.execute(
                "INSERT OR IGNORE INTO linked_folder_sync_jobs"
                "(id, folder_id, kind, status, stage, created_at) "
                "VALUES(?,?,'sync','pending','queued',?)",
                (str(uuid.uuid4()), row["id"], now),
            )
        conn.commit()
        _prune_terminal_jobs(conn)
    finally:
        conn.close()


def _next_job() -> str | None:
    conn = rag_db.get_connection()
    try:
        row = conn.execute(
            "SELECT id FROM linked_folder_sync_jobs WHERE status='pending' ORDER BY created_at LIMIT 1"
        ).fetchone()
        return row["id"] if row else None
    finally:
        conn.close()


def _worker() -> None:
    global _thread
    try:
        while not _stop.is_set():
            job_id = _next_job()
            if job_id:
                try:
                    with _sync_lock:
                        reconcile_folder(job_id)
                except Exception as exc:
                    logger.exception("linked-folder job %s failed unexpectedly", job_id)
                    _fail_job(job_id, exc)
                continue
            _wake.wait(max(1.0, config.FOLDER_SYNC_INTERVAL_S))
            _wake.clear()
            if not _stop.is_set():
                try:
                    _enqueue_periodic()
                except Exception:
                    logger.warning("linked-folder periodic scheduling failed", exc_info = True)
    finally:
        with _thread_lock:
            if _thread is threading.current_thread():
                _thread = None


def start_auto_sync(*, admission_lock = None, admit = None) -> bool:
    global _thread
    if not rag_db.RAG_AVAILABLE:
        return False
    with _thread_lock:
        retired = _thread if _thread is not None and _thread.is_alive() else None
        if retired is not None and not _stop.is_set():
            return False
    if retired is not None:
        retired.join(timeout = 0)
        if retired.is_alive():
            return False

    def launch() -> bool:
        global _thread
        with _thread_lock:
            if _thread is not None and _thread.is_alive():
                return False
            _stop.clear()
            conn = rag_db.get_connection()
            try:
                conn.execute(
                    "UPDATE linked_folder_sync_jobs SET status='pending', stage='queued', "
                    "error=NULL WHERE status='running'"
                )
                # A crash between successful ingestion and mapping installation can leave
                # a folder-owned document unreferenced. It is safe to remove at startup.
                orphans = conn.execute(
                    "SELECT d.id, d.stored_path FROM documents d "
                    "WHERE d.linked_folder_id IS NOT NULL AND NOT EXISTS "
                    "(SELECT 1 FROM linked_folder_files ff WHERE ff.document_id=d.id)"
                ).fetchall()
                for orphan in orphans:
                    store.delete_document(conn, orphan["id"], commit = False)
                conn.commit()
            finally:
                conn.close()
            for orphan in orphans:
                _remove_snapshot(orphan["stored_path"])
            _enqueue_periodic()
            _thread = threading.Thread(target = _worker, daemon = True, name = "rag-folder-sync")
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
    if thread is not None:
        thread.join(timeout = timeout)
