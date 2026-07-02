# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted RAG embedding-model override (Settings -> General).

The stored value takes precedence over the ``RAG_EMBEDDING_MODEL`` env default in
``core.rag.config``. Vectors from different models live in different spaces, so
documents already indexed under the old model must be re-uploaded after a change
(the UI warns about this).
"""

from __future__ import annotations

import time
from typing import Any

EMBEDDING_MODEL_SETTING_KEY = "rag_embedding_model"
MAX_EMBEDDING_MODEL_LENGTH = 512

# The effective model is consulted on the embedder hot path (once per embed /
# tokenize call during ingestion), so the stored value is cached briefly instead
# of hitting sqlite each time. Writes invalidate immediately in-process; other
# readers converge within the TTL.
_CACHE_TTL_S = 2.0
_cached: tuple[float, str | None] | None = None


def _invalidate_cache() -> None:
    global _cached
    _cached = None


def default_embedding_model() -> str:
    """The env/default model from rag config (``RAG_EMBEDDING_MODEL`` or bge)."""
    from core.rag import config
    return config.EMBEDDING_MODEL


def _coerce_embedding_model(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if not cleaned or len(cleaned) > MAX_EMBEDDING_MODEL_LENGTH:
        return None
    # Newlines/control chars are never valid in a repo id or path.
    if any(ord(ch) < 32 for ch in cleaned):
        return None
    return cleaned


def validate_embedding_model(value: Any) -> str:
    cleaned = _coerce_embedding_model(value)
    if cleaned is None:
        raise ValueError(
            "Embedding model must be a Hugging Face repo id (e.g. "
            "'unsloth/bge-small-en-v1.5') or a local model path, up to "
            f"{MAX_EMBEDDING_MODEL_LENGTH} characters."
        )
    return cleaned


def get_stored_embedding_model() -> str | None:
    """The persisted override, or None when unset/invalid."""
    global _cached
    now = time.monotonic()
    cached = _cached
    if cached is not None and now - cached[0] < _CACHE_TTL_S:
        return cached[1]
    try:
        from storage.studio_db import get_app_setting
        stored = get_app_setting(EMBEDDING_MODEL_SETTING_KEY, None)
    except Exception:
        stored = None
    value = _coerce_embedding_model(stored)
    _cached = (now, value)
    return value


def get_rag_embedding_model() -> str:
    """Effective embedding model: persisted override, else env/default."""
    return get_stored_embedding_model() or default_embedding_model()


def set_rag_embedding_model(value: Any) -> str:
    parsed = validate_embedding_model(value)
    from storage.studio_db import upsert_app_settings

    # Saving the default is not an override; keeps is_custom (and the UI's
    # reset affordance) honest.
    stored = parsed if parsed != default_embedding_model() else None
    upsert_app_settings({EMBEDDING_MODEL_SETTING_KEY: stored})
    _invalidate_cache()
    return parsed


def reset_rag_embedding_model() -> str:
    """Clear the override; returns the (env/default) model now in effect."""
    from storage.studio_db import upsert_app_settings

    upsert_app_settings({EMBEDDING_MODEL_SETTING_KEY: None})
    _invalidate_cache()
    return default_embedding_model()
