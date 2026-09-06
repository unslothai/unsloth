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

# Consulted on the embedder hot path once per embed/tokenize call during ingestion, so the stored value is cached
# briefly; writes invalidate in-process, other readers converge within the TTL.
_CACHE_TTL_S = 2.0
# typing.Optional, not `str | None`: the future import defers annotations, but a
# type ALIAS is evaluated at import, and PEP 604 needs 3.10 over a 3.9 floor.
_StoredState = tuple[
    Optional[str], Optional[str], Optional[str], Optional[str], bool, Optional[dict]
]
# (override model, resolved model, GGUF repo, backend, download pending, raw record).
# The raw record is carried so a conditional write compares against exactly what is stored: a reconstruction never
# matches a record written by a build with one field fewer.
_cached: tuple[float, _StoredState] | None = None
# Bumped on every write/invalidate. A reader captures it before the DB read and
# only fills the cache if it is unchanged afterward, so a read that overlapped a
# save cannot repopulate the cache with the pre-save value for the whole TTL.
_generation = 0
_lock = threading.Lock()
# Per-model, process-local: the last (gguf_repo, backend, download_pending, files)
# seen for each model. The one stored record belongs to whichever model was saved
# last; see remembered_gguf_repo.
_resolved_gguf_memo: dict[str, tuple[Optional[str], Optional[str], bool, Optional[list]]] = {}


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


def _coerce_gguf_files(value: Any) -> Optional[list]:
    """Repo-relative GGUF names from ``value``, or None when it names no family.

    Same length/control-character rules as every other stored string: this record
    is read back to steer a loader, so it must not carry anything a path join
    would misread."""
    if not isinstance(value, (list, tuple)):
        return None
    named = [f for f in (_coerce_embedding_model(v) for v in value) if f]
    return named or None


def get_stored_gguf_repo(model: str) -> str | None:
    """The GGUF repo stored alongside ``model``, or None when it was stored for a
    different model (a stale pair must not point the loader at the wrong weights)."""
    stored = _get_stored_state()
    if stored[1] != model:
        return None
    _remember_resolution(model, stored)
    return stored[2]


def _remember_resolution(model: str, stored: _StoredState) -> None:
    """Keep this process's last resolved repo/backend/pending/files for ``model``."""
    with _lock:
        _resolved_gguf_memo[model] = (stored[2], stored[3], stored[4], _files_of(stored[5]))


def _remembered(model: str) -> tuple[str | None, str | None, bool, list | None] | None:
    with _lock:
        return _resolved_gguf_memo.get(model)


def _files_of(resolution: Optional[dict]) -> Optional[list]:
    """The planned GGUF file family recorded in ``resolution``, if it holds one."""
    if not isinstance(resolution, dict):
        return None
    files = resolution.get("gguf_files")
    if not isinstance(files, list):
        return None
    named = [f for f in files if isinstance(f, str) and f.strip()]
    return named or None


def remembered_gguf_repo(model: str) -> str | None:
    """The repo this process last saw resolved for ``model``, if any.

    One stored record, so saving B makes ``get_stored_gguf_repo(A)`` None while a
    job pinned to A is still ingesting, moving its identity to the derived
    ``A-GGUF`` mid-job and splitting one document set across two tags. The memo is
    process-local and per model, so it lasts as long as the job; a later save for
    A refreshes it, and a reset drops it.
    """
    remembered = _remembered(model)
    return remembered[0] if remembered else None


def get_stored_backend(model: str) -> str | None:
    """The backend stored for ``model``, or the one this process last saw for it.

    Same staleness rule as the repo, and the same reason to survive it: on an auto
    CPU install a model with no GGUF resolves to sentence-transformers, so losing
    it drops a still-running job onto the hardware default, which has no GGUF.
    """
    stored = _get_stored_state()
    if stored[1] == model:
        _remember_resolution(model, stored)
        return stored[3]
    remembered = _remembered(model)
    return remembered[1] if remembered else None


def get_stored_gguf_files(model: str) -> list | None:
    """The GGUF file family the picker planned for ``model``, if one was recorded.

    Same staleness-plus-memo rule as the backend. Loaders use it to tell the quant
    the advertised transfer actually delivered from an unrelated one left in the
    same repo by an earlier setting. None on records written before it was stored,
    which is why every consumer has to keep working without it.
    """
    stored = _get_stored_state()
    if stored[1] == model:
        _remember_resolution(model, stored)
        return _files_of(stored[5])
    remembered = _remembered(model)
    return remembered[3] if remembered else None


def get_stored_download_pending(model: str) -> bool:
    """Whether ``model`` was activated before its required transfer finished.

    Loaders stay cache-only on this marker instead of recreating the invisible
    first-index download. It outlives another model's save for the same reason the
    backend does: forgetting it re-enables that download for a pinned job.
    """
    stored = _get_stored_state()
    if stored[1] == model:
        _remember_resolution(model, stored)
        return stored[4]
    remembered = _remembered(model)
    return remembered[2] if remembered else False


def clear_stored_download_pending(model: str) -> bool:
    """Retire the pending marker for ``model`` once its weights are on disk.

    Nothing else clears it: the picker re-resolves after a transfer but does not
    save again, so the marker would outlive the download and pin the model
    cache-only forever. Callers are the loaders, once the cache is proven complete.
    """
    stored = _get_stored_state()
    if stored[1] != model or not stored[4]:
        return False
    expected = stored[5]
    if not isinstance(expected, dict):
        # Pre-atomic layout: the flag lives nowhere this can clear.
        return False
    from storage.studio_db import compare_and_set_app_setting

    # Conditional, not a plain upsert, or a save for another model committing between the read and
    # this write is reverted. Compared as read, not rebuilt, so the guard survives fields this
    # build does not know about.
    if not compare_and_set_app_setting(
        EMBEDDING_RESOLUTION_SETTING_KEY, expected, {**expected, "download_pending": False}
    ):
        return False
    # Retire the memo with the record, or a pinned job keeps reading pending=True
    # and stays cache-only after the download landed.
    _remember_resolution(model, (stored[0], stored[1], stored[2], stored[3], False, stored[5]))
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
        return (None, None, None, None, False, None)
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
    raw = resolution if isinstance(resolution, dict) else None
    value: _StoredState = (override, resolved_model, repo, backend, download_pending, raw)
    with _lock:
        # Only cache when no save landed while reading: a pre-save value would mask the new one for the
        # whole TTL.
        if _generation == gen:
            _cached = (time.monotonic(), value)
    return value


def get_rag_embedding_model() -> str:
    """Effective embedding model: persisted override, else env/default."""
    stored = _get_stored_state()
    model = stored[0] or default_embedding_model()
    # Reading this is how a job pins its model, so record the resolution here: the memo protects a pinned job only if it
    # was populated before another model's save takes the stored record.
    if stored[1] == model:
        _remember_resolution(model, stored)
    return model


def set_rag_embedding_model(
    value: Any,
    gguf_repo: Any = None,
    backend: Any = None,
    download_pending: bool = False,
    gguf_files: Any = None,
) -> str:
    parsed = validate_embedding_model(value)
    from storage.studio_db import upsert_app_settings

    # Saving the default is not an override; keeps is_custom (and the UI's
    # reset affordance) honest.
    stored = parsed if parsed != default_embedding_model() else None
    repo = _coerce_embedding_model(gguf_repo)
    chosen = _coerce_embedding_model(backend)
    files = _coerce_gguf_files(gguf_files)
    resolution = (
        {
            "model": parsed,
            "gguf_repo": repo,
            "backend": chosen,
            "download_pending": download_pending is True,
            "gguf_files": files,
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

    restored = default_embedding_model()
    # The memo survives a reset, but the restored default is not a running job, so write any remembered resolution back
    # durably rather than leave a process-only answer that changes on restart.
    remembered = _remembered(restored)
    resolution = None
    # The pending flag counts as much as a repo or a backend: a default saved over
    # a failed resolution legitimately remembers (None, None, True), and that flag
    # is what keeps the first index from starting the implicit download.
    if remembered and (remembered[0] or remembered[1] or remembered[2]):
        resolution = {
            "model": restored,
            "gguf_repo": remembered[0],
            "backend": remembered[1],
            "download_pending": remembered[2],
            "gguf_files": remembered[3],
        }
    upsert_app_settings(
        {
            EMBEDDING_MODEL_SETTING_KEY: None,
            EMBEDDING_RESOLUTION_SETTING_KEY: resolution,
            EMBEDDING_GGUF_SETTING_KEY: None,
            EMBEDDING_BACKEND_SETTING_KEY: None,
        }
    )
    _invalidate_cache()
    return restored
