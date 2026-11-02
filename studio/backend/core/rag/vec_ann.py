# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HNSW (usearch) over a scope's unit rows, keyed by ``chunks.rowid``, plus an exact delta for rows added after the
build. Pure numpy + usearch: no SQLite here; ``vec_cache`` decides when an index may answer."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

DTYPE = "f16"
CONNECTIVITY = 16
EXPANSION_ADD = 128
# Fixed at load, not per query: the index is shared across threads. usearch still returns k > expansion.
EXPANSION_SEARCH = 256
FORMAT = 1


def usearch():
    """The usearch.index module, or None when the wheel is missing or does not load on this machine."""
    try:
        from usearch import index
    except Exception:  # noqa: BLE001 - any load failure leaves the exact / vec0 paths
        return None
    return index


@dataclass(frozen = True)
class Ann:
    """One published generation: a (usually memory-mapped) index plus the exact rows newer than it."""

    index: object
    n: int
    path: str | None
    delta_keys: object
    delta_mat: object
    meta: dict = field(default_factory = dict)

    @property
    def size(self) -> int:
        return self.n + len(self.delta_keys)


def new_index(dim: int):
    return usearch().Index(
        ndim = int(dim),
        metric = "cos",
        dtype = DTYPE,
        connectivity = CONNECTIVITY,
        expansion_add = EXPANSION_ADD,
    )


def add(np, index, keys, rows, threads: int) -> int:
    """Add float32 rows; NaN / inf / zero rows (vec0 scores them NULL) are skipped. Returns rows added."""
    rows = np.asarray(rows, dtype = np.float32)
    with np.errstate(all = "ignore"):
        norms = np.sqrt(np.einsum("ij,ij->i", rows, rows))
    valid = np.isfinite(norms) & (norms > 0)
    if not valid.all():
        rows, keys = rows[valid], np.asarray(keys)[valid]
    if len(rows):
        index.add(np.asarray(keys, dtype = np.uint64), rows, threads = threads)
    return len(rows)


def unit_rows(np, keys, rows):
    """(keys, unit float32 rows) with unscorable rows dropped: the exact delta's form."""
    rows = np.array(rows, dtype = np.float32)
    with np.errstate(all = "ignore"):
        norms = np.sqrt(np.einsum("ij,ij->i", rows, rows))
        valid = np.isfinite(norms) & (norms > 0)
        rows /= norms[:, None]
    return np.asarray(keys, dtype = np.int64)[valid], rows[valid]


def with_delta(np, ann: Ann, keys, rows) -> Ann:
    """A new generation sharing the index, with ``rows`` (already unit) appended to the delta."""
    return Ann(
        ann.index,
        ann.n,
        ann.path,
        np.concatenate([ann.delta_keys, keys]),
        np.concatenate([ann.delta_mat, rows]) if len(ann.delta_mat) else rows,
        ann.meta,
    )


def search(np, ann: Ann, q, fetch: int):
    """(keys int64, cosine distances float64) of the ``fetch`` nearest, ascending, ties by key. ``q`` is unit."""
    f = min(int(fetch), ann.size)
    keys, dist = [], []
    if ann.n and f:
        # Near-full fetches (widening) brute-force the stored rows: no graph reachability limit.
        m = ann.index.search(q, min(f, ann.n), exact = 2 * f >= ann.n)
        keys.append(np.asarray(m.keys, dtype = np.int64))
        dist.append(np.asarray(m.distances, dtype = np.float64))
    if len(ann.delta_keys) and f:
        d = 1.0 - (ann.delta_mat @ q).astype(np.float64)
        keys.append(ann.delta_keys)
        dist.append(d)
    if not keys:
        return np.empty(0, np.int64), np.empty(0, np.float64)
    keys, dist = np.concatenate(keys), np.concatenate(dist)
    order = np.lexsort((keys, dist))[:f]
    return keys[order], dist[order]


def save(index, base: str, meta: dict) -> str:
    """Write ``base``.usearch + ``base``.json via temp names and ``os.replace``; returns the index path."""
    path, meta_path = base + ".usearch", base + ".json"
    index.save(path + ".tmp")
    os.replace(path + ".tmp", path)
    with open(meta_path + ".tmp", "w", encoding = "utf-8") as fh:
        json.dump({**meta, "format": FORMAT}, fh)
    os.replace(meta_path + ".tmp", meta_path)
    return path


def load(base: str, dim: int):
    """(view index, meta) for a saved generation, or None if missing, mismatched or unreadable."""
    mod = usearch()
    if mod is None:
        return None
    try:
        with open(base + ".json", encoding = "utf-8") as fh:
            meta = json.load(fh)
        if meta.get("format") != FORMAT or int(meta.get("dim", -1)) != int(dim):
            return None
        index = mod.Index.restore(base + ".usearch", view = True)
        if index is None or index.ndim != int(dim) or len(index) != int(meta.get("n", -1)):
            return None
        index.expansion_search = EXPANSION_SEARCH
        return index, meta
    except Exception:  # noqa: BLE001 - a corrupt or foreign file is rebuilt
        logger.debug("vec_ann.load failed for %s", base, exc_info = True)
        return None


def remove_files(base: str) -> bool:
    """Delete a generation's files; False when one is still mapped (Windows) and must be retried later."""
    ok = True
    for suffix in (".usearch", ".json", ".usearch.tmp", ".json.tmp"):
        try:
            os.remove(base + suffix)
        except FileNotFoundError:
            pass
        except OSError:
            ok = False
    return ok
