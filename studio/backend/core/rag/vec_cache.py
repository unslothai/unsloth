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


# Rows per matvec: per-query scratch stays O(block) however large the scope.
_BLOCK = 32768
_FETCH = 4096


class _Store:
    """Growable unit-row float32 matrix + rowids + chunk ids. Successive entries of one scope share it: an
    entry reads only its first ``n`` rows and appends write past every published ``n``."""

    __slots__ = ("mat", "rowids", "ids")

    def __init__(self, np, cap: int, dim: int):
        cap = max(int(cap), 1)
        self.mat = np.empty((cap, dim), dtype = np.float32)
        self.rowids = np.empty(cap, dtype = np.int64)
        self.ids = np.empty(cap, dtype = object)

    def grown(self, np, n: int, need: int) -> "_Store":
        """Self if ``need`` rows fit, else a copy of the first ``n`` rows with geometric headroom."""
        cap = len(self.rowids)
        if need <= cap:
            return self
        new = _Store(np, max(need, cap + cap // 4 + 16), self.mat.shape[1])
        new.mat[:n] = self.mat[:n]
        new.rowids[:n] = self.rowids[:n]
        new.ids[:n] = self.ids[:n]
        return new

    def write(self, np, n: int, rows, rowids, ids) -> int:
        """Normalise ``rows`` (float32, writable) in place and store the scorable ones at ``n``; returns the new n."""
        valid = _normalise(np, rows)
        keep = np.flatnonzero(valid)
        k = len(keep)
        if k == len(rows):
            self.mat[n : n + k] = rows
            self.rowids[n : n + k] = rowids
            self.ids[n : n + k] = ids
        else:
            self.mat[n : n + k] = rows[keep]
            self.rowids[n : n + k] = np.asarray(rowids, dtype = np.int64)[keep]
            self.ids[n : n + k] = [ids[i] for i in keep]
        return n + k

    def nbytes(self, avg_id: int) -> int:
        return self.mat.nbytes + self.rowids.nbytes + len(self.rowids) * (avg_id + _ID_OVERHEAD)


def _normalise(np, rows):
    """Unit rows in place; mask of the rows vec0 can score (it answers NULL for NaN / inf / zero rows)."""
    with np.errstate(all = "ignore"):
        norms = np.sqrt(np.einsum("ij,ij->i", rows, rows))
        valid = np.isfinite(norms) & (norms > 0)
        rows /= norms[:, None]
    return valid


@dataclass(frozen = True)
class _Entry:
    epoch: tuple
    n: int
    store: _Store
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


def _top(np, scores, f: int):
    """Indices of the ``f`` largest scores, unordered."""
    m = len(scores)
    return np.argpartition(scores, m - f)[m - f :] if f < m else np.arange(m)


def _search_entry(np, entry: _Entry, vector, fetch: int):
    n = entry.n
    q = np.array(vector, dtype = np.float32)
    qn = float(np.sqrt(np.dot(q, q)))
    if not np.isfinite(qn) or qn == 0.0 or n == 0:
        return [], n
    q /= qn
    store = entry.store
    f = min(int(fetch), n)
    if n <= _BLOCK:
        scores = store.mat[:n] @ q
        idx = _top(np, scores, f)
        best = scores[idx]
    else:
        buf = np.empty(_BLOCK, dtype = np.float32)
        idx = np.empty(0, dtype = np.int64)
        best = np.empty(0, dtype = np.float32)
        # Running top-f: scratch is O(block + fetch), not O(scope).
        for s0 in range(0, n, _BLOCK):
            out = buf[: min(n, s0 + _BLOCK) - s0]
            np.matmul(store.mat[s0 : s0 + len(out)], q, out = out)
            part = _top(np, out, min(f, len(out)))
            idx = np.concatenate([idx, part + s0])
            best = np.concatenate([best, out[part]])
            if len(best) > f:
                keep = _top(np, best, f)
                idx, best = idx[keep], best[keep]
    # Same order vec0 gives: distance ascending, ties by rowid.
    order = np.lexsort((store.rowids[idx], -best))
    ids = store.ids
    return [(ids[i], 1.0 - float(s)) for i, s in zip(idx[order], best[order])], n


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
        rows = np.array(vectors, dtype = np.float32).reshape(len(rowids), dim)
        with _lock:
            entry = _entries.get(key)
            if entry is None:
                return
            if after[:2] != before[:2] or appended != len(rowids) or entry.epoch != before:
                _publish(key, None)
                return
            # Under the lock with the epoch check, so no other append writes the same rows.
            store = entry.store.grown(np, entry.n, entry.n + len(rows))
            n = store.write(np, entry.n, rows, rowids, list(chunk_ids))
            _publish(key, _Entry(after, n, store, store.nbytes(_avg_id(store, n))), expect = before)
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
        if count * (dim * 4 + 8 + _ID_OVERHEAD) > _budget():
            with _lock:
                _too_big[key] = epoch
            return
        # Streamed into one preallocated matrix: peak is the matrix plus one fetch block.
        store = _Store(np, count, dim)
        n = 0
        cur = conn.execute(
            "SELECT rowid, chunk_id, embedding FROM chunks_vec WHERE scope=?", (scope,)
        )
        while rows := cur.fetchmany(_FETCH):
            blk = np.frombuffer(b"".join(r[2] for r in rows), dtype = np.float32)
            blk = blk.reshape(len(rows), dim).copy()
            store = store.grown(np, n, n + len(rows))
            n = store.write(np, n, blk, [r[0] for r in rows], [r[1] for r in rows])
    finally:
        conn.close()
    entry = _Entry(epoch, n, store, store.nbytes(_avg_id(store, n)))
    with _lock:
        _too_big.pop(key, None)
        _publish(key, entry)


def _avg_id(store: _Store, n: int) -> int:
    sample = store.ids[: min(n, 64)]
    return int(sum(len(c) for c in sample) / len(sample)) if len(sample) else 0


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
