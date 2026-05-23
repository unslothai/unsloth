# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-scope BM25 lexical index using the ``bm25s`` library.

bm25s does not support incremental insertion cheaply, so we rebuild the
full per-scope index whenever its document set changes. At studio
scale (a few hundred to a few tens of thousands of chunks per KB) this
is fast enough; the upside is that deletes are trivial.

A scope is ``kb_<uuid>`` or ``thread_<uuid>``. Each scope stores:
  - ``<scope>/`` directory holding the bm25s index files
  - ``<scope>/ids.json`` mapping the index's positional ids back to
    chunk-id strings (bm25s returns row indices, not our ids).
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
from pathlib import Path
from typing import Any

from utils.paths.storage_roots import ensure_dir, rag_bm25_root

logger = logging.getLogger(__name__)

_load_lock = threading.Lock()
_cache: dict[str, tuple[Any, list[str]]] = {}


def _scope_dir(scope: str) -> Path:
    return rag_bm25_root() / scope


def _ids_path(scope: str) -> Path:
    return _scope_dir(scope) / "ids.json"


def _has_index(scope: str) -> bool:
    return _ids_path(scope).is_file()


def _evict(scope: str) -> None:
    _cache.pop(scope, None)


def rebuild_index(scope: str, chunks: list[dict]) -> None:
    """Rebuild the BM25 index for ``scope`` from the full chunk list.

    Each chunk dict must contain ``id`` and ``text``. Passing an empty
    list deletes the scope's index files.
    """
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

        retriever = bm25s.BM25.load(str(_scope_dir(scope)), load_corpus = False)
        ids = json.loads(_ids_path(scope).read_text())
        _cache[scope] = (retriever, ids)
        return _cache[scope]


def search(scope: str, query: str, k: int) -> list[tuple[str, float]]:
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
    base = _scope_dir(scope)
    if base.exists():
        shutil.rmtree(base, ignore_errors = True)
    _evict(scope)
