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
        if not loader_id:
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

    ``requested`` may be ``repo`` or ``repo:VARIANT``; the variant is split off
    and matched on the base name so it can be forwarded to the loader.
    """
    if not requested or not requested.strip():
        return None
    base, _, variant = requested.strip().partition(":")
    loader_id = _index().get(base.strip().lower())
    if loader_id is None:
        return None
    return loader_id, (variant.strip() or None)
