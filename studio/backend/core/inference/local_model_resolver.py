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
    """Build an entry only when real GGUF weights exist locally.

    Excludes Transformers/safetensors models, and lists only the quants that
    are on disk so auto-switch never asks /load to fetch a remote one. Recurses
    snapshots and quant subdirs (e.g. ``snapshots/<sha>/BF16/model.gguf``).
    """
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
    from routes.models import _scan_models_dir, _scan_hf_cache, _resolve_hf_cache_dir

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

    ``requested`` may be ``repo`` or ``repo:VARIANT``. An exact id match wins
    first (so ids that themselves contain a colon, e.g. a Windows path, still
    resolve); otherwise a trailing ``:VARIANT`` is split off the last colon. A
    bare id resolves to a concrete local quant so /load never fetches a remote
    one, and a requested variant resolves only when that quant is on disk.
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

    Advertised in ``/v1/models`` so a client can discover what it can swap to,
    mirroring llama-swap. Each is a name ``resolve_local_gguf`` accepts. Hidden
    helper/probe weights (RAG embeddings, the llama.cpp validation probe) are
    excluded, matching the model pickers elsewhere in Studio.
    """
    try:
        from routes.models import _is_hidden_model

        return sorted(
            {
                entry.loader_id
                for entry in _index().values()
                if not _is_hidden_model(entry.loader_id)
            }
        )
    except Exception:
        return []
