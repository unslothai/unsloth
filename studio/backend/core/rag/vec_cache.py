# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""In-memory per-scope float32 matrices answering dense KNN exactly, in place of vec0's scalar scan.

SQLite stays authoritative: triggers bump ``rag_vec_epoch`` for a scope in the same transaction as any write that
can change its vectors, and an entry answers only a reader whose snapshot shows the epoch the entry was built at.
Anything else (miss, stale, over budget, no numpy, any error) returns None and the caller scans vec0."""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

GLOBAL_SCOPE = ""
_ID_OVERHEAD = 64


def _np():
    try:
        import numpy
    except ImportError:
        return None
    return numpy


def enabled() -> bool:
    return os.environ.get("RAG_VEC_CACHE", "1") != "0" and _budget() > 0


def _budget() -> int:
    try:
        return int(float(os.environ.get("RAG_VEC_CACHE_MB", "256")) * 1024 * 1024)
    except ValueError:
        return 0


@dataclass(frozen = True)
class _Entry:
    epoch: tuple
    rowids: object
    chunk_ids: object
    mat: object
    inv: object
    nbytes: int


_entries: OrderedDict[tuple, _Entry] = OrderedDict()
_too_big: dict[tuple, int] = {}
_bytes = 0
_lock = threading.Lock()
_pending: set[tuple] = set()
_cond = threading.Condition(_lock)
_worker: threading.Thread | None = None
_busy = 0


def _key(db_file: str, scope: str, dim: int) -> tuple:
    try:
        ino = os.stat(db_file).st_ino
    except OSError:
        ino = None
    return db_file, ino, scope, int(dim)


def read_epoch(conn: sqlite3.Connection, scope: str) -> tuple | None:
    """(scope epoch, global epoch, newest chunk rowid in the scope): equal tuples mean equal vectors."""
    try:
        return tuple(
            conn.execute(
                "SELECT COALESCE((SELECT n FROM rag_vec_epoch WHERE scope=?), 0), "
                "COALESCE((SELECT n FROM rag_vec_epoch WHERE scope=?), 0), "
                "COALESCE((SELECT max(rowid) FROM chunks WHERE scope=?), 0)",
                (scope, GLOBAL_SCOPE, scope),
            ).fetchone()
        )
    except sqlite3.Error:
        return None


def _make_entry(np, epoch, rowids, chunk_ids, blob: bytes, dim: int) -> _Entry:
    mat = np.frombuffer(blob, dtype = np.float32).reshape(-1, dim)
    with np.errstate(all = "ignore"):
        norms = np.sqrt(np.einsum("ij,ij->i", mat, mat, dtype = np.float64))
    # vec0 answers NULL for NaN / inf / zero rows, which _search_dense drops.
    valid = np.isfinite(norms) & (norms > 0)
    mat = np.ascontiguousarray(mat[valid])
    inv = (1.0 / norms[valid]).astype(np.float32)
    rowids = np.asarray(rowids, dtype = np.int64)[valid]
    ids = np.empty(len(chunk_ids), dtype = object)
    ids[:] = chunk_ids
    ids = ids[valid]
    for a in (mat, inv, rowids, ids):
        a.setflags(write = False)
    nbytes = mat.nbytes + inv.nbytes + rowids.nbytes + sum(len(c) + _ID_OVERHEAD for c in ids)
    return _Entry(epoch, rowids, ids, mat, inv, nbytes)


def _publish(
    key: tuple,
    entry: _Entry | None,
    expect: tuple | None = None,
) -> None:
    """Under _lock. ``expect``: replace only while the current entry is still at that epoch."""
    global _bytes
    old = _entries.get(key)
    if expect is not None and (old is None or old.epoch != expect):
        return
    if old is not None:
        _bytes -= old.nbytes
        del _entries[key]
    if entry is None:
        return
    if entry.nbytes > _budget():
        _too_big[key] = entry.epoch
        return
    _entries[key] = entry
    _bytes += entry.nbytes
    while _bytes > _budget() and _entries:
        _, evicted = _entries.popitem(last = False)
        _bytes -= evicted.nbytes


def search(conn, db_file: str, scope: str, vector, fetch: int):
    """(rows, cap) from the cache at this snapshot's epoch, else None (and a build is queued)."""
    if not enabled():
        return None
    np = _np()
    if np is None:
        return None
    try:
        epoch = read_epoch(conn, scope)
        if epoch is None:
            return None
        key = _key(db_file, scope, len(vector))
        with _lock:
            entry = _entries.get(key)
            if entry is not None and entry.epoch == epoch:
                _entries.move_to_end(key)
            else:
                entry = None
                if _too_big.get(key) != epoch:
                    _schedule(key)
        if entry is None:
            return None
        return _search_entry(np, entry, vector, fetch)
    except Exception:  # noqa: BLE001 - vec0 answers instead
        logger.debug("vec_cache.search failed", exc_info = True)
        return None


def _search_entry(np, entry: _Entry, vector, fetch: int):
    q = np.asarray(vector, dtype = np.float32)
    qn = float(np.sqrt(np.dot(q.astype(np.float64), q.astype(np.float64))))
    n = len(entry.rowids)
    if not np.isfinite(qn) or qn == 0.0 or n == 0:
        return [], n
    scores = (entry.mat @ q) * entry.inv
    scores = scores.astype(np.float64) / qn
    f = min(int(fetch), n)
    idx = np.argpartition(-scores, f - 1)[:f] if f < n else np.arange(n)
    order = idx[np.lexsort((entry.rowids[idx], -scores[idx]))]
    return [(entry.chunk_ids[i], 1.0 - float(scores[i])) for i in order], n


def note_append(
    db_file: str, scope: str, before, after, appended: int, rowids, chunk_ids, vectors
) -> None:
    """After a committed pure append: extend the entry iff this writer's rows are the only change since
    ``before``. ``appended``: chunks rows in the scope newer than before's max rowid, read under the write lock."""
    if not enabled() or before is None or after is None or not rowids:
        return
    np = _np()
    if np is None:
        return
    try:
        dim = len(vectors[0])
        key = _key(db_file, scope, dim)
        with _lock:
            entry = _entries.get(key)
            if entry is None:
                return
            if after[:2] != before[:2] or appended != len(rowids) or entry.epoch != before:
                _publish(key, None)
                return
        rows = np.asarray(vectors, dtype = np.float32).reshape(len(rowids), dim)
        add = _make_entry(np, after, rowids, list(chunk_ids), rows.tobytes(), dim)
        merged = _Entry(
            after,
            np.concatenate([entry.rowids, add.rowids]),
            np.concatenate([entry.chunk_ids, add.chunk_ids]),
            np.concatenate([entry.mat, add.mat]),
            np.concatenate([entry.inv, add.inv]),
            entry.nbytes + add.nbytes,
        )
        for a in (merged.rowids, merged.chunk_ids, merged.mat, merged.inv):
            a.setflags(write = False)
        with _lock:
            _publish(key, merged, expect = before)
    except Exception:  # noqa: BLE001
        logger.debug("vec_cache.note_append failed", exc_info = True)
        with _lock:
            _publish(_key(db_file, scope, len(vectors[0])), None)


def _schedule(key: tuple) -> None:
    """Under _lock."""
    global _worker
    if key in _pending:
        return
    _pending.add(key)
    if _worker is None or not _worker.is_alive():
        _worker = threading.Thread(target = _run, name = "rag-vec-cache", daemon = True)
        _worker.start()
    _cond.notify_all()


def _run() -> None:
    global _busy
    while True:
        with _lock:
            while not _pending:
                _cond.wait()
            key = _pending.pop()
            _busy += 1
        try:
            _build(key)
        except Exception:  # noqa: BLE001
            logger.debug("vec_cache build failed for %s", key[2], exc_info = True)
        finally:
            with _lock:
                _busy -= 1
                _cond.notify_all()


def _connect(db_file: str) -> sqlite3.Connection:
    from storage import rag_db, sqlite_gate

    conn = sqlite_gate.connect(db_file, timeout = 5)
    try:
        conn.enable_load_extension(True)
        rag_db.sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    except BaseException:
        conn.close()
        raise
    return conn


def _build(key: tuple) -> None:
    from storage import rag_db

    np = _np()
    if np is None or not enabled():
        return
    db_file, _, scope, dim = key
    conn = _connect(db_file)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")
        epoch = read_epoch(conn, scope)
        if epoch is None or rag_db.vec_table_dim(conn) != dim:
            return
        count = conn.execute("SELECT count(*) FROM chunks WHERE scope=?", (scope,)).fetchone()[0]
        if count * (dim * 4 + _ID_OVERHEAD) > _budget():
            with _lock:
                _too_big[key] = epoch
            return
        rows = conn.execute(
            "SELECT rowid, chunk_id, embedding FROM chunks_vec WHERE scope=?", (scope,)
        ).fetchall()
    finally:
        conn.close()
    entry = _make_entry(
        np,
        epoch,
        [r[0] for r in rows],
        [r[1] for r in rows],
        b"".join(r[2] for r in rows),
        dim,
    )
    with _lock:
        _too_big.pop(key, None)
        _publish(key, entry)


def drain(timeout: float = 30.0) -> bool:
    """Wait for queued builds; for tests and benchmarks."""
    with _lock:
        return _cond.wait_for(lambda: not _pending and not _busy, timeout)


def clear() -> None:
    global _bytes
    with _lock:
        _entries.clear()
        _too_big.clear()
        _pending.clear()
        _bytes = 0
