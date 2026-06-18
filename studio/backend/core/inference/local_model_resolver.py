# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resolve an OpenAI-request ``model`` string to a downloaded local GGUF.

Used by the opt-in auto-switch path. The match is conservative: only names
that map to an already-downloaded local model are eligible, so an arbitrary
OpenAI model string still falls through to the loaded model (drop-in compat)
and no surprise multi-GB download is ever triggered. The local-model scan is
cached for a few seconds since auto-switch consults it per request.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

_CACHE_TTL_S = 5.0
_lock = threading.Lock()
_scan: tuple[float, dict[str, str]] = (0.0, {})


def _has_local_gguf(info) -> bool:
    """True only when the model is, or contains, a real .gguf on disk.

    The scanners also surface Transformers/safetensors models; auto-switch must
    never load those through the GGUF llama-server path. Covers a direct .gguf
    file, a models-dir folder, and the HF-cache snapshots layout.
    """
    from pathlib import Path

    path = getattr(info, "path", None)
    if not isinstance(path, str):
        return False
    p = Path(path)
    try:
        if p.is_file():
            return p.suffix.lower() == ".gguf"
        return (
            next(p.glob("*.gguf"), None) is not None
            or next(p.glob("snapshots/*/*.gguf"), None) is not None
        )
    except OSError:
        return False


def _build_index() -> dict[str, str]:
    """Map normalized id/model_id/display_name -> canonical loader identifier."""
    # Lazy import: routes.models imports core.inference, so import at call time.
    from pathlib import Path
    from routes.models import _scan_models_dir, _scan_hf_cache, _resolve_hf_cache_dir

    index: dict[str, str] = {}
    try:
        found = _scan_models_dir(Path("./models").resolve()) + _scan_hf_cache(
            _resolve_hf_cache_dir()
        )
    except Exception:
        return index
    for info in found:
        loader_id = getattr(info, "id", None)
        if not loader_id or not _has_local_gguf(info):
            continue
        for key in (info.id, getattr(info, "model_id", None), getattr(info, "display_name", None)):
            if key:
                index.setdefault(key.strip().lower(), loader_id)
    return index


def _index() -> dict[str, str]:
    global _scan
    now = time.monotonic()
    with _lock:
        ts, cached = _scan
        if now - ts < _CACHE_TTL_S:
            return cached
    fresh = _build_index()
    with _lock:
        _scan = (now, fresh)
    return fresh


def resolve_local_gguf(requested: str) -> Optional[tuple[str, Optional[str]]]:
    """Return ``(loader_id, gguf_variant)`` for a downloaded local match, else None.

    ``requested`` may be ``repo`` or ``repo:VARIANT``. An exact id match wins
    first so ids that themselves contain a colon (e.g. a Windows path) still
    resolve; only then is a trailing ``:VARIANT`` split off the last colon.
    """
    if not requested or not requested.strip():
        return None
    requested = requested.strip()
    index = _index()
    loader_id = index.get(requested.lower())
    if loader_id is not None:
        return loader_id, None
    base, sep, variant = requested.rpartition(":")
    if not sep:
        return None
    loader_id = index.get(base.strip().lower())
    if loader_id is None:
        return None
    return loader_id, (variant.strip() or None)
