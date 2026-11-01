# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""In-memory per-scope float32 matrices answering dense KNN exactly, in place of vec0's scalar scan; scopes of
``RAG_VEC_ANN_MIN_ROWS`` or more get a saved, memory-mapped HNSW index (``vec_ann``) instead.

SQLite stays authoritative: triggers bump ``rag_vec_epoch`` for a scope in the same transaction as any write that
can change its vectors, and an entry answers only a reader whose snapshot shows the epoch the entry was built at.
Anything else (miss, stale, over budget, no numpy / usearch, any error) returns None and the caller scans vec0."""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import threading
from collections import OrderedDict
from dataclasses import dataclass

from . import vec_ann

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


@dataclass(frozen = True)
class _AnnEntry:
    """``epoch`` = (scope epoch, global epoch, newest chunks.rowid covered by index + delta)."""

    epoch: tuple
    ann: object
    base: str | None


_entries: OrderedDict[tuple, _Entry] = OrderedDict()
_ann: OrderedDict[tuple, _AnnEntry] = OrderedDict()
_stale_files: set[str] = set()
_too_big: dict[tuple, int] = {}
_bytes = 0
_lock = threading.Lock()
_pending: set[tuple] = set()
_building: set[tuple] = set()
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
        # A scope that outgrew the exact cache through appends moves to HNSW now, not at its next write.
        if entry.n >= _ann_min_rows() and ann_enabled():
            _schedule(key)
        else:
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
            ann = _ann.get(key)
            if entry is not None and entry.epoch == epoch:
                _entries.move_to_end(key)
            else:
                entry = None
                # ANN answers only at the reader's exact epoch: rows added elsewhere are not in its delta yet.
                if ann is None or ann.epoch != epoch:
                    ann = None
                    if _too_big.get(key) != epoch:
                        _schedule(key)
                else:
                    _ann.move_to_end(key)
        if entry is not None:
            return _search_entry(np, entry, vector, fetch)
        if ann is not None:
            return _search_ann(conn, np, ann, vector, fetch)
        return None
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


_ANN_MAX_FETCH = 16384


def ann_enabled() -> bool:
    return os.environ.get("RAG_VEC_ANN", "1") != "0" and vec_ann.usearch() is not None


def _ann_min_rows() -> int:
    try:
        return int(os.environ.get("RAG_VEC_ANN_MIN_ROWS", "100000"))
    except ValueError:
        return 100000


def _ann_threads() -> int:
    # usearch 2.26 strands a few nodes (unreachable at any k) when inserting in parallel: build on one thread.
    return 1


def _search_ann(conn, np, entry: _AnnEntry, vector, fetch: int):
    """HNSW candidates (+ exact delta), keys resolved to chunk ids in the reader's own snapshot."""
    ann = entry.ann
    cap = min(ann.size, _ANN_MAX_FETCH)
    q = np.array(vector, dtype = np.float32)
    qn = float(np.sqrt(np.dot(q, q)))
    if not np.isfinite(qn) or qn == 0.0 or not ann.size:
        return [], cap
    q /= qn
    keys, dist = vec_ann.search(np, ann, q, min(int(fetch), cap))
    ids: dict[int, str] = {}
    flat = keys.tolist()
    for s0 in range(0, len(flat), _ID_BATCH):
        batch = flat[s0 : s0 + _ID_BATCH]
        for rowid, cid in conn.execute(
            f"SELECT rowid, id FROM chunks WHERE rowid IN ({','.join('?' * len(batch))})", batch
        ):
            ids[rowid] = cid
    return [(ids[k], float(d)) for k, d in zip(flat, dist.tolist()) if k in ids], cap


_ID_BATCH = 900


def _note_ann_append(np, key, before, after, appended, chunk_rowids, rows) -> None:
    """Under _lock. Rows a committed pure append added join the exact delta; anything else drops the entry."""
    entry = _ann.get(key)
    if entry is None:
        return
    if (
        chunk_rowids is None
        or after[:2] != before[:2]
        or appended != len(chunk_rowids)
        or entry.epoch != before
    ):
        _ann.pop(key, None)
        return
    keys, unit = vec_ann.unit_rows(np, chunk_rowids, rows)
    ann = vec_ann.with_delta(np, entry.ann, keys, unit)
    _ann[key] = _AnnEntry(after, ann, entry.base)
    if len(ann.delta_keys) > _delta_limit(ann.n):
        _schedule(key)


def _delta_limit(n: int) -> int:
    """Most exact delta rows kept beside an index of ``n`` rows before it is rebuilt."""
    return max(20000, n // 20)


def note_append(
    db_file: str,
    scope: str,
    before,
    after,
    appended: int,
    rowids,
    chunk_ids,
    vectors,
    chunk_rowids = None,
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
            _note_ann_append(np, key, before, after, appended, chunk_rowids, rows)
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
    if key in _pending or key in _building:
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
            _building.add(key)
            _busy += 1
        try:
            _build(key)
        except Exception:  # noqa: BLE001
            logger.debug("vec_cache build failed for %s", key[2], exc_info = True)
        finally:
            with _lock:
                _building.discard(key)
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
        if count >= _ann_min_rows() and ann_enabled():
            try:
                _build_ann(np, conn, key, epoch)
            except Exception:
                # A build takes minutes: never retry it at the same epoch.
                with _lock:
                    _too_big[key] = epoch
                raise
            return
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


def _ann_dir(db_file: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(db_file)), "rag_ann")


def _ann_prefix(key: tuple) -> str:
    db_file, ino, scope, dim = key
    return hashlib.sha1(f"{db_file}|{ino}|{scope}|{dim}".encode()).hexdigest()[:20]


def _saved_generations(key: tuple) -> list[tuple[tuple, str]]:
    """[((scope epoch, global epoch, max rowid), base)] of this scope's generations on disk."""
    d, prefix = _ann_dir(key[0]), _ann_prefix(key)
    out = []
    try:
        names = os.listdir(d)
    except OSError:
        return out
    for name in names:
        if name.startswith(prefix + "-") and name.endswith(".json"):
            try:
                epoch = tuple(int(p) for p in name[len(prefix) + 1 : -5].split("-"))
            except ValueError:
                continue
            if len(epoch) == 3:
                out.append((epoch, os.path.join(d, name[:-5])))
    return out


def _delta_since(np, conn, scope: str, dim: int, after_rowid: int):
    """(keys, unit rows) of the scope's chunks newer than ``after_rowid``: one vec0 partition scan."""
    wanted = dict(
        conn.execute(
            "SELECT id, rowid FROM chunks WHERE scope=? AND rowid > ?", (scope, after_rowid)
        ).fetchall()
    )
    keys, rows = [], []
    if wanted:
        for cid, emb in conn.execute(
            "SELECT chunk_id, embedding FROM chunks_vec WHERE scope=?", (scope,)
        ):
            if cid in wanted:
                keys.append(wanted[cid])
                rows.append(np.frombuffer(emb, dtype = np.float32))
    if not rows:
        return np.empty(0, np.int64), np.empty((0, dim), np.float32)
    return vec_ann.unit_rows(np, keys, np.stack(rows))


def _build_ann(np, conn, key: tuple, epoch: tuple) -> None:
    """In ``conn``'s read snapshot: reuse a saved generation at this epoch (catching up appends) or build one."""
    _, _, scope, dim = key
    saved = [(e, b) for e, b in _saved_generations(key) if e[:2] == epoch[:2] and e[2] <= epoch[2]]
    loaded = None
    for e, base in sorted(saved, reverse = True):
        loaded = vec_ann.load(base, dim)
        if loaded is not None:
            break
    if loaded is not None:
        index, meta = loaded
        # Counted before any vector is read: a large catch-up goes straight to the streamed rebuild.
        newer = conn.execute(
            "SELECT count(*) FROM chunks WHERE scope=? AND rowid > ?", (scope, e[2])
        ).fetchone()[0]
        if newer <= _delta_limit(len(index)):
            keys, unit = _delta_since(np, conn, scope, dim, e[2])
            ann = vec_ann.Ann(index, len(index), base + ".usearch", keys, unit, meta)
            _publish_ann(key, _AnnEntry(epoch, ann, base))
            return
        loaded = index = None
    index = vec_ann.new_index(dim)
    cur = conn.execute(
        # CROSS JOIN keeps vec0 as the outer scan; chunks is probed by its primary key.
        "SELECT c.rowid, v.embedding FROM chunks_vec v CROSS JOIN chunks c ON c.id = v.chunk_id "
        "WHERE v.scope=? AND c.rowid <= ?",
        (scope, epoch[2]),
    )
    while rows := cur.fetchmany(_FETCH):
        blk = np.frombuffer(b"".join(r[1] for r in rows), dtype = np.float32).reshape(len(rows), dim)
        vec_ann.add(np, index, [r[0] for r in rows], blk, _ann_threads())
    conn.rollback()
    os.makedirs(_ann_dir(key[0]), exist_ok = True)
    base = os.path.join(_ann_dir(key[0]), f"{_ann_prefix(key)}-{epoch[0]}-{epoch[1]}-{epoch[2]}")
    meta = {"dim": dim, "n": len(index), "epoch": list(epoch), "scope": scope}
    vec_ann.save(index, base, meta)
    del index
    loaded = vec_ann.load(base, dim)
    if loaded is None:
        return
    view, meta = loaded
    empty = np.empty(0, np.int64), np.empty((0, dim), np.float32)
    _publish_ann(
        key, _AnnEntry(epoch, vec_ann.Ann(view, len(view), base + ".usearch", *empty, meta), base)
    )


def _publish_ann(key: tuple, entry: _AnnEntry) -> None:
    with _lock:
        _ann[key] = entry
        _ann.move_to_end(key)
    for _, base in _saved_generations(key):
        if base != entry.base:
            _stale_files.add(base)
    _stale_files.update(_enforce_disk_cap(_ann_dir(key[0]), key))
    with _lock:
        keep = {e.base for e in _ann.values()}
    # Windows refuses to delete a file still mapped by a reader: retried at the next publish.
    for base in list(_stale_files):
        if base not in keep and vec_ann.remove_files(base):
            _stale_files.discard(base)


def _enforce_disk_cap(d: str, newest: tuple) -> list[str]:
    """Bases to delete so ``d`` fits ``RAG_VEC_ANN_DISK_MB`` (default 4096): unpublished files oldest first, then
    least recently used published scopes (evicted from ``_ann``), never ``newest``."""
    try:
        cap = float(os.environ.get("RAG_VEC_ANN_DISK_MB", "4096")) * 2**20
        sizes = {
            os.path.join(d, n[: -len(".usearch")]): os.path.getsize(os.path.join(d, n))
            for n in os.listdir(d)
            if n.endswith(".usearch")
        }
    except (OSError, ValueError):
        return []
    total = sum(sizes.values())
    out = []
    with _lock:
        published = [(k, e.base) for k, e in _ann.items()]
    for base in sorted(
        set(sizes) - {b for _, b in published}, key = lambda b: os.path.getmtime(b + ".usearch")
    ):
        if total <= cap:
            return out
        total -= sizes[base]
        out.append(base)
    for k, base in published:
        if total <= cap:
            break
        if k == newest:
            continue
        with _lock:
            _ann.pop(k, None)
        total -= sizes.get(base, 0)
        out.append(base)
    return out


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
        _ann.clear()
        _too_big.clear()
        _pending.clear()
        _stale_files.clear()
        _bytes = 0
