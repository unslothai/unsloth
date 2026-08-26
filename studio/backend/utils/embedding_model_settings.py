# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Persisted RAG embedding-model override (Settings -> General).

The stored value takes precedence over the ``RAG_EMBEDDING_MODEL`` env default in
``core.rag.config``. Vectors from different models live in different spaces, so
documents already indexed under the old model must be re-uploaded after a change
(the UI warns about this).
"""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

EMBEDDING_MODEL_SETTING_KEY = "rag_embedding_model"
# The GGUF repo the picker resolved for that model. Stored so the loader opens
# what was actually downloaded instead of re-deriving a name that may not exist.
EMBEDDING_GGUF_SETTING_KEY = "rag_embedding_gguf_repo"
# Which backend that model needs. An embedder with no GGUF still runs fine on
# sentence-transformers (safetensors), it just costs about 1 GB more memory.
EMBEDDING_BACKEND_SETTING_KEY = "rag_embedding_backend"
# Atomic association between the selected model and the artifacts/backend the
# resolver validated for it. Unlike the override key, this may name the env
# default: an off-convention GGUF still has to remain attached to that model.
EMBEDDING_RESOLUTION_SETTING_KEY = "rag_embedding_resolution"
MAX_EMBEDDING_MODEL_LENGTH = 512

# The effective model is consulted on the embedder hot path (once per embed /
# tokenize call during ingestion), so the stored value is cached briefly instead
# of hitting sqlite each time. Writes invalidate immediately in-process; other
# readers converge within the TTL.
_CACHE_TTL_S = 2.0
# typing.Optional, not `str | None`: `from __future__ import annotations` defers
# annotations, but a type ALIAS is evaluated at import, and PEP 604 unions need
# 3.10 while the packaged floor is 3.9.
_StoredState = tuple[Optional[str], Optional[str], Optional[str], Optional[str], bool]
# (override model, resolved model, GGUF repo, backend, download pending)
_cached: tuple[float, _StoredState] | None = None
# Bumped on every write/invalidate. A reader captures it before the DB read and
# only fills the cache if it is unchanged afterward, so a read that overlapped a
# save cannot repopulate the cache with the pre-save value for the whole TTL.
_generation = 0
_lock = threading.Lock()
# Per-model, process-local: the last resolved GGUF repo this process saw for each
# model. There is only one stored resolution record, so it belongs to whichever
# model was saved last; see remembered_gguf_repo.
_resolved_gguf_memo: dict[str, str] = {}


def _invalidate_cache() -> None:
    global _cached, _generation
    with _lock:
        _cached = None
        _generation += 1


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


def get_stored_gguf_repo(model: str) -> str | None:
    """The GGUF repo stored alongside ``model``, or None when it was stored for a
    different model (a stale pair must not point the loader at the wrong weights)."""
    stored = _get_stored_state()
    if stored[1] != model:
        return None
    if stored[2]:
        _remember_resolved_gguf_repo(model, stored[2])
    return stored[2]


def _remember_resolved_gguf_repo(model: str, repo: str) -> None:
    """Keep this process's last resolved repo for ``model``."""
    with _lock:
        _resolved_gguf_memo[model] = repo


def remembered_gguf_repo(model: str) -> str | None:
    """The repo this process last saw resolved for ``model``, if any.

    There is one stored resolution record, so saving model B immediately makes
    ``get_stored_gguf_repo(A)`` None. A background ingestion pinned to A is still
    running against A's off-convention mirror, and its identity would silently
    move to the derived ``A-GGUF`` name mid-job, splitting one document set across
    two vector-space tags. This memo is process-local and per model, so it lasts
    exactly as long as the job that needs it; a later save for A refreshes it
    through ``get_stored_gguf_repo``, and a reset drops it.
    """
    with _lock:
        return _resolved_gguf_memo.get(model)


def get_stored_backend(model: str) -> str | None:
    """The backend stored for ``model``, or None. Same staleness rule as above."""
    stored = _get_stored_state()
    return stored[3] if stored[1] == model else None


def get_stored_download_pending(model: str) -> bool:
    """Whether ``model`` was activated before its required transfer finished.

    Loaders use this marker to remain cache-only instead of recreating the
    invisible first-index download the picker is designed to prevent.
    """
    stored = _get_stored_state()
    return stored[4] if stored[1] == model else False


def clear_stored_download_pending(model: str) -> bool:
    """Retire the pending marker for ``model`` once its weights are on disk.

    Nothing else clears it: the picker re-resolves after a transfer but does not
    save again, so without this the marker outlives the download it described and
    pins that model cache-only forever. A later eviction then reads as "not
    downloaded yet" and only re-picking the model would recover. Callers are the
    loaders, at the point where they have just proven the cache is complete.
    """
    stored = _get_stored_state()
    if stored[1] != model or not stored[4]:
        return False
    from storage.studio_db import compare_and_set_app_setting

    expected = {
        "model": stored[1],
        "gguf_repo": stored[2],
        "backend": stored[3],
        "download_pending": True,
    }
    # Conditional, not a plain upsert: a save for another model committing between
    # the read above and this write would otherwise be reverted, putting this
    # model's resolution back beside the other one's override and leaving that
    # model to re-derive a backend and a companion repo it never resolved.
    if not compare_and_set_app_setting(
        EMBEDDING_RESOLUTION_SETTING_KEY, expected, {**expected, "download_pending": False}
    ):
        return False
    _invalidate_cache()
    return True


def get_stored_embedding_model() -> str | None:
    """The persisted override, or None when unset/invalid."""
    return _get_stored_state()[0]


def _get_stored_state() -> _StoredState:
    """Read the override and its resolved artifact association as one snapshot.

    The resolution is one JSON value so its model/repo/backend can never be
    torn. The legacy individual fields are read in the same SQL statement for
    compatibility with builds from before the atomic record existed.
    """
    global _cached
    now = time.monotonic()
    with _lock:
        cached = _cached
        if cached is not None and now - cached[0] < _CACHE_TTL_S:
            return cached[1]
        gen = _generation
    try:
        from storage.studio_db import get_app_settings
        settings = get_app_settings(
            [
                EMBEDDING_MODEL_SETTING_KEY,
                EMBEDDING_RESOLUTION_SETTING_KEY,
                EMBEDDING_GGUF_SETTING_KEY,
                EMBEDDING_BACKEND_SETTING_KEY,
            ]
        )
    except Exception:
        # Transient store failure: keep the last known value instead of
        # silently reverting the embed/search hot path to the default model,
        # which would mix vector spaces mid-ingestion.
        with _lock:
            if _cached is not None:
                _cached = (time.monotonic(), _cached[1])
                return _cached[1]
        return (None, None, None, None, False)
    override = _coerce_embedding_model(settings.get(EMBEDDING_MODEL_SETTING_KEY))
    resolution = settings.get(EMBEDDING_RESOLUTION_SETTING_KEY)
    resolved_model = repo = backend = None
    download_pending = False
    if isinstance(resolution, dict):
        resolved_model = _coerce_embedding_model(resolution.get("model"))
        repo = _coerce_embedding_model(resolution.get("gguf_repo"))
        backend = _coerce_embedding_model(resolution.get("backend"))
        download_pending = resolution.get("download_pending") is True
    elif override:
        # Legacy PR builds stored the association in separate keys. The one-shot
        # read above still gives this compatibility path a consistent snapshot.
        resolved_model = override
        repo = _coerce_embedding_model(settings.get(EMBEDDING_GGUF_SETTING_KEY))
        backend = _coerce_embedding_model(settings.get(EMBEDDING_BACKEND_SETTING_KEY))
    value: _StoredState = (override, resolved_model, repo, backend, download_pending)
    with _lock:
        # Only cache when no save landed while we were reading; otherwise this
        # value may be pre-save, and caching it would mask the new one for the
        # TTL. The next reader re-reads the committed value.
        if _generation == gen:
            _cached = (time.monotonic(), value)
    return value


def get_rag_embedding_model() -> str:
    """Effective embedding model: persisted override, else env/default."""
    return get_stored_embedding_model() or default_embedding_model()


def set_rag_embedding_model(
    value: Any,
    gguf_repo: Any = None,
    backend: Any = None,
    download_pending: bool = False,
) -> str:
    parsed = validate_embedding_model(value)
    from storage.studio_db import upsert_app_settings

    # Saving the default is not an override; keeps is_custom (and the UI's
    # reset affordance) honest.
    stored = parsed if parsed != default_embedding_model() else None
    repo = _coerce_embedding_model(gguf_repo)
    chosen = _coerce_embedding_model(backend)
    resolution = (
        {
            "model": parsed,
            "gguf_repo": repo,
            "backend": chosen,
            "download_pending": download_pending is True,
        }
        if repo or chosen or download_pending
        else None
    )
    upsert_app_settings(
        {
            EMBEDDING_MODEL_SETTING_KEY: stored,
            EMBEDDING_RESOLUTION_SETTING_KEY: resolution,
            # Retire the pre-atomic spelling on the same commit.
            EMBEDDING_GGUF_SETTING_KEY: None,
            EMBEDDING_BACKEND_SETTING_KEY: None,
        }
    )
    _invalidate_cache()
    return parsed


def reset_rag_embedding_model() -> str:
    """Clear the override; returns the (env/default) model now in effect."""
    from storage.studio_db import upsert_app_settings

    upsert_app_settings(
        {
            EMBEDDING_MODEL_SETTING_KEY: None,
            EMBEDDING_RESOLUTION_SETTING_KEY: None,
            EMBEDDING_GGUF_SETTING_KEY: None,
            EMBEDDING_BACKEND_SETTING_KEY: None,
        }
    )
    # A reset means "forget what was resolved", so the memo goes with the record.
    with _lock:
        _resolved_gguf_memo.clear()
    _invalidate_cache()
    return default_embedding_model()
