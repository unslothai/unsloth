# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-scope BM25 index (rebuild on change; bm25s has no cheap incremental insert).

Each scope dir holds the bm25s files + ids.json mapping row index → chunk_id.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path
from typing import Any

from loggers import get_logger
from utils.paths.storage_roots import ensure_dir, rag_bm25_root

logger = get_logger(__name__)

_load_lock = threading.Lock()
_cache: dict[str, tuple[Any, list[str]]] = {}


def _fast() -> bool:
    """Incremental SQLite FTS5 backend instead of the rebuild-on-change bm25s one."""
    return os.environ.get("UNSLOTH_RAG_FAST") == "1"


def add_chunks(scope: str, chunks: list[dict]) -> None:
    """Incremental insert of one document's chunks (FTS5 fast path only)."""
    if _fast():
        from core.rag import bm25_fts

        bm25_fts.add_chunks(scope, chunks)
        return
    # bm25s has no incremental insert; callers on the slow path use rebuild_index.
    raise RuntimeError("add_chunks requires UNSLOTH_RAG_FAST=1")


def _scope_dir(scope: str) -> Path:
    return rag_bm25_root() / scope


def _ids_path(scope: str) -> Path:
    return _scope_dir(scope) / "ids.json"


def _has_index(scope: str) -> bool:
    return _ids_path(scope).is_file()


def _evict(scope: str) -> None:
    _cache.pop(scope, None)


def rebuild_index(scope: str, chunks: list[dict]) -> None:
    """Rebuild scope's BM25 from full chunk list. Empty list deletes the index."""
    if _fast():
        from core.rag import bm25_fts

        bm25_fts.rebuild_index(scope, chunks)
        return
    import bm25s

    base = _scope_dir(scope)
    if not chunks:
        delete_scope(scope)
        return
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    tokens = bm25s.tokenize(texts, show_progress = False)
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress = False)
    # bm25s.BM25.save does not unlink stale files; clear the dir first.
    delete_scope(scope)
    ensure_dir(base)
    retriever.save(str(base))
    _ids_path(scope).write_text(json.dumps(ids))
    with _load_lock:
        _cache[scope] = (retriever, ids)


def _load(scope: str) -> tuple[Any, list[str]] | None:
    if not _has_index(scope):
        return None
    with _load_lock:
        if scope in _cache:
            return _cache[scope]
        import bm25s

        try:
            retriever = bm25s.BM25.load(str(_scope_dir(scope)), load_corpus = False)
            ids = json.loads(_ids_path(scope).read_text())
        except (FileNotFoundError, OSError, json.JSONDecodeError, ValueError) as exc:
            # Corrupt/partial index: treat as missing so re-ingest rebuilds cleanly.
            logger.warning(
                "bm25 index unreadable for scope %s (%s: %s); treating as missing",
                scope,
                type(exc).__name__,
                exc,
            )
            return None
        _cache[scope] = (retriever, ids)
        return _cache[scope]


def search(scope: str, query: str, k: int) -> list[tuple[str, float]]:
    if _fast():
        from core.rag import bm25_fts

        return bm25_fts.search(scope, query, k)
    import bm25s

    loaded = _load(scope)
    if loaded is None:
        return []
    retriever, ids = loaded
    if not ids:
        return []
    k_actual = min(k, len(ids))
    q_tokens = bm25s.tokenize([query], show_progress = False)
    indices, scores = retriever.retrieve(q_tokens, k = k_actual, show_progress = False)
    out: list[tuple[str, float]] = []
    for pos in range(indices.shape[1]):
        idx = int(indices[0][pos])
        out.append((ids[idx], float(scores[0][pos])))
    return out


def delete_scope(scope: str) -> None:
    if _fast():
        from core.rag import bm25_fts

        bm25_fts.delete_scope(scope)
        return
    base = _scope_dir(scope)
    if base.exists():
        shutil.rmtree(base, ignore_errors = True)
    _evict(scope)
