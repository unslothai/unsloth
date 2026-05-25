# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Scope identifiers + per-scope embedder resolution.

A "scope" is the namespace key carried by vector rows and BM25
indexes — ``kb_<uuid>`` for stand-alone Knowledge Bases or
``thread_<uuid>`` for per-thread document sets.

`resolve_scope_embedder` looks up which embedder populated a scope's
vectors so the query side can re-use the same model. Lives in
``core/rag`` rather than ``routes`` so the inference-side tool
handler can call it without a route-→-core import cycle.
"""

from __future__ import annotations

from storage.studio_db import get_connection, list_chat_settings
from utils.rag.config import resolve_embedder

# Persisted chat-settings keys. Defined here so the resolver (core)
# and the route handlers (routes/rag.py) share one source of truth.
RAG_DEFAULTS_KEY = "rag.defaults"


def thread_settings_key(thread_id: str) -> str:
    return f"thread:{thread_id}:rag"


def resolve_scope_embedder(scope: str) -> str | None:
    """Return the embedder used to populate ``scope``'s vector rows.

    Resolution order:
      - ``kb_<id>``  → ``rag_knowledge_bases.embedding_model`` column.
      - ``thread_<id>`` → per-thread override → app-level defaults
        override → ``RAG_EMBEDDER_MATRIX[(mode, chunking)]``.

    Returns ``None`` for unrecognised scope strings; callers treat
    ``None`` as "fall back to the configured default embedder".
    """
    if scope.startswith("kb_"):
        kb_id = scope[len("kb_") :]
        with get_connection() as conn:
            row = conn.execute(
                "SELECT embedding_model FROM rag_knowledge_bases WHERE id = ?",
                (kb_id,),
            ).fetchone()
        return row["embedding_model"] if row else None

    if scope.startswith("thread_"):
        thread_id = scope[len("thread_") :]
        all_settings = list_chat_settings()
        defaults = all_settings.get(RAG_DEFAULTS_KEY) or {}
        if not isinstance(defaults, dict):
            defaults = {}
        per_thread = all_settings.get(thread_settings_key(thread_id)) or {}
        if not isinstance(per_thread, dict):
            per_thread = {}
        explicit = per_thread.get("embedding_model") or defaults.get("embedding_model")
        if explicit:
            return explicit
        mode = per_thread.get("mode") or defaults.get("mode") or "text"
        chunking_strategy = (
            per_thread.get("chunking_strategy")
            or defaults.get("chunking_strategy")
            or "standard"
        )
        return resolve_embedder(mode, chunking_strategy)

    return None
