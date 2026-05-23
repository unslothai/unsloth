# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Qdrant local-mode vector store.

Qdrant's local mode (``QdrantClient(path=...)``) acquires a file lock on
the storage directory, so only one process at a time can hold the
client. That means *all* vector reads/writes funnel through this module
in the FastAPI parent process. Ingestion subprocesses do not open
Qdrant directly — they compute vectors and send them back over a queue
for the parent to persist.

A "scope" is the collection name: ``kb_<uuid>`` for standalone
knowledge bases and ``thread_<uuid>`` for per-thread document sets.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Iterable

from utils.paths.storage_roots import ensure_dir, rag_vectordb_root

logger = logging.getLogger(__name__)

_client: Any | None = None
_client_lock = threading.Lock()


def get_qdrant() -> Any:
    """Lazy singleton; parent process only."""
    global _client
    with _client_lock:
        if _client is None:
            from qdrant_client import QdrantClient

            path = ensure_dir(rag_vectordb_root())
            _client = QdrantClient(path = str(path))
        return _client


def kb_scope(kb_id: str) -> str:
    return f"kb_{kb_id}"


def thread_scope(thread_id: str) -> str:
    return f"thread_{thread_id}"


def collection_exists(scope: str) -> bool:
    client = get_qdrant()
    try:
        client.get_collection(collection_name = scope)
        return True
    except Exception:
        return False


def ensure_collection(scope: str, dim: int) -> None:
    client = get_qdrant()
    if collection_exists(scope):
        return
    from qdrant_client.models import Distance, VectorParams

    client.create_collection(
        collection_name = scope,
        vectors_config = VectorParams(size = dim, distance = Distance.COSINE),
    )


def upsert_chunks(
    scope: str,
    points: Iterable[dict],
) -> None:
    """Insert/update chunk vectors.

    Each point must have keys ``id`` (str), ``vector`` (list[float]) and
    ``payload`` (dict with at least ``document_id`` and ``chunk_index``).
    """
    from qdrant_client.models import PointStruct

    client = get_qdrant()
    structured = [
        PointStruct(id = p["id"], vector = p["vector"], payload = p["payload"])
        for p in points
    ]
    if not structured:
        return
    client.upsert(collection_name = scope, points = structured)


def search(
    scope: str,
    query_vector: list[float],
    *,
    top_k: int,
    document_ids: list[str] | None = None,
) -> list[dict]:
    from qdrant_client.models import FieldCondition, Filter, MatchAny

    client = get_qdrant()
    query_filter = None
    if document_ids:
        query_filter = Filter(
            must = [
                FieldCondition(
                    key = "document_id",
                    match = MatchAny(any = document_ids),
                )
            ]
        )
    if not collection_exists(scope):
        return []
    results = client.search(
        collection_name = scope,
        query_vector = query_vector,
        limit = top_k,
        query_filter = query_filter,
    )
    return [
        {
            "chunk_id": str(r.id),
            "score": float(r.score),
            "payload": dict(r.payload or {}),
        }
        for r in results
    ]


def delete_scope(scope: str) -> None:
    client = get_qdrant()
    if not collection_exists(scope):
        return
    client.delete_collection(collection_name = scope)


def delete_document(scope: str, document_id: str) -> None:
    from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

    if not collection_exists(scope):
        return
    client = get_qdrant()
    client.delete(
        collection_name = scope,
        points_selector = FilterSelector(
            filter = Filter(
                must = [
                    FieldCondition(
                        key = "document_id",
                        match = MatchValue(value = document_id),
                    )
                ]
            )
        ),
    )
