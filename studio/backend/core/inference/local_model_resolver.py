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

from loggers import get_logger

logger = get_logger(__name__)


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
    """Map normalized id/model_id/display_name -> local GGUF entry.

    Scans the same roots Studio's model picker lists (./models, the active plus
    legacy/default HF caches, LM Studio dirs, and user scan folders) so a named
    local model is never missed and silently served as the loaded one. Ollama's
    scanner is skipped: it creates symlinks as a side effect and this runs on the
    request path.
    """
    # Lazy import: routes.models imports core.inference, so import at call time.
    from pathlib import Path
    from routes.models import (
        _scan_models_dir,
        _scan_hf_cache,
        _scan_lmstudio_dir,
        _resolve_hf_cache_dir,
        _is_hidden_model,
    )
    from utils.paths import legacy_hf_cache_dir, hf_default_cache_dir, lmstudio_model_dirs

    index: dict[str, _LocalGgufEntry] = {}
    seen_hf: set[str] = set()

    def _scan_hf_once(directory) -> list:
        if directory is None:
            return []
        try:
            d = Path(directory)
            if not d.is_dir():
                return []
            rp = str(d.resolve())
        except Exception as exc:  # a missing/None root must skip, never crash the index
            logger.debug("auto-switch: skipping HF cache dir %r: %s", directory, exc)
            return []
        if rp in seen_hf:
            return []
        seen_hf.add(rp)
        return _scan_hf_cache(directory)

    try:
        found = _scan_models_dir(Path("./models").resolve())
        for hf_dir in (_resolve_hf_cache_dir(), legacy_hf_cache_dir(), hf_default_cache_dir()):
            found += _scan_hf_once(hf_dir)
        for lm_dir in lmstudio_model_dirs():
            found += _scan_lmstudio_dir(lm_dir)
        try:
            from storage.studio_db import list_scan_folders
            for folder in list_scan_folders():
                fp = Path(folder["path"])
                found += (
                    _scan_models_dir(fp, limit = 200) + _scan_hf_once(fp) + _scan_lmstudio_dir(fp)
                )
        except Exception:
            pass
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
    # Build under the lock so concurrent callers with an expired cache don't all
    # run the (multi-dir) scan at once; the rest wait and reuse the fresh result.
    with _lock:
        now = time.monotonic()
        ts, cached = _scan
        if now - ts < _CACHE_TTL_S:
            return cached
        fresh = _build_index()
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
