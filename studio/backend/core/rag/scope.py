# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Scope identifiers (kb_<id> / thread_<id>) + per-scope embedder resolver."""

from __future__ import annotations

from storage.studio_db import closing_connection, list_chat_settings
from utils.rag.config import resolve_embedder

RAG_DEFAULTS_KEY = "rag.defaults"


def thread_settings_key(thread_id: str) -> str:
    return f"thread:{thread_id}:rag"


def resolve_scope_embedder(scope: str) -> str | None:
    """KB → kb.embedding_model; thread → per-thread/defaults/matrix. None = use default."""
    if scope.startswith("kb_"):
        kb_id = scope[len("kb_") :]
        with closing_connection() as conn:
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
        return resolve_embedder()

    return None
