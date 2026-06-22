# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resolve an OpenAI-request ``model`` string to a downloaded local GGUF.

Used by the opt-in auto-switch path. The match is conservative: only names
that map to an already-downloaded local GGUF (and a quant that is actually on
disk) are eligible, so an arbitrary OpenAI model string still falls through to
the loaded model (drop-in compat) and no surprise multi-GB download is ever
triggered. The local-model scan is cached for a few seconds since auto-switch
consults it per request.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen = True)
class _LocalGgufEntry:
    loader_id: str
    variants: tuple[str, ...]  # local quant labels; () for a standalone .gguf


_CACHE_TTL_S = 5.0
_lock = threading.Lock()
_scan: tuple[float, dict[str, _LocalGgufEntry]] = (0.0, {})


def _local_gguf_entry(loader_id: str, info) -> Optional[_LocalGgufEntry]:
    """Build an entry only when GGUF quants are on disk (not Transformers/
    safetensors), listing only on-disk quants so /load never fetches a remote one."""
    from pathlib import Path
    from utils.models.model_config import list_local_gguf_variants

    path = getattr(info, "path", None)
    if not isinstance(path, str):
        return None
    p = Path(path)
    try:
        if p.is_file():
            # A standalone .gguf loads by path; no quant sub-selection.
            return _LocalGgufEntry(loader_id, ()) if p.suffix.lower() == ".gguf" else None
        variants, _ = list_local_gguf_variants(path)
        quants = tuple(v.quant for v in variants if getattr(v, "quant", None))
        return _LocalGgufEntry(loader_id, quants) if quants else None
    except Exception:
        return None


def _build_index() -> dict[str, _LocalGgufEntry]:
    """Map normalized id/model_id/display_name -> local GGUF entry."""
    # Lazy import: routes.models imports core.inference, so import at call time.
    from pathlib import Path
    from routes.models import (
        _scan_models_dir,
        _scan_hf_cache,
        _resolve_hf_cache_dir,
        _is_hidden_model,
    )

    index: dict[str, _LocalGgufEntry] = {}
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
        # Skip what Studio hides from its pickers (validation probe, RAG embed
        # weights): not chat models, so never an auto-switch target.
        if _is_hidden_model(loader_id, getattr(info, "path", None)):
            continue
        entry = _local_gguf_entry(loader_id, info)
        if entry is None:
            continue
        for key in (info.id, getattr(info, "model_id", None), getattr(info, "display_name", None)):
            if key:
                index.setdefault(key.strip().lower(), entry)
    return index


def _index() -> dict[str, _LocalGgufEntry]:
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

    ``requested`` is ``repo`` or ``repo:VARIANT``. An exact id match wins first
    (so ids containing a colon still resolve); else the last ``:VARIANT`` is
    split off and resolves only when that quant is on disk. A bare id picks a
    concrete local quant so /load never fetches a remote one.
    """
    if not isinstance(requested, str) or not requested.strip():
        return None
    requested = requested.strip()
    try:
        index = _index()
        entry = index.get(requested.lower())
        if entry is not None:
            return entry.loader_id, (entry.variants[0] if entry.variants else None)

        base, sep, variant = requested.rpartition(":")
        if not sep:
            return None
        entry = index.get(base.strip().lower())
        if entry is None:
            return None
        wanted = variant.strip().lower()
        for v in entry.variants:
            if v.lower() == wanted:
                return entry.loader_id, v
        return None
    except Exception:
        # Best-effort: any resolver failure falls through to the loaded model,
        # so a malformed name can never turn a servable request into a 500.
        return None


def list_switch_eligible_ids() -> list[str]:
    """Distinct loader ids for every downloaded GGUF auto-switch can serve.

    Advertised in ``/v1/models`` so a client can discover what to swap to. Each
    is a name ``resolve_local_gguf`` accepts.
    """
    try:
        return sorted({entry.loader_id for entry in _index().values()})
    except Exception:
        return []
