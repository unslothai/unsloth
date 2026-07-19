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

from core.inference.model_ids import public_model_id
from loggers import get_logger

logger = get_logger(__name__)


@dataclass(frozen = True)
class _LocalGgufEntry:
    loader_id: str  # advertised id (repo id / folder name), also the override key
    load_path: str  # concrete on-disk dir/file passed to /load so it never downloads
    variants: tuple[str, ...]  # local quant labels; () for a standalone .gguf


_CACHE_TTL_S = 5.0
_lock = threading.Lock()
_scan: tuple[float, dict[str, _LocalGgufEntry]] = (0.0, {})


def _is_abs_path_id(value: str) -> bool:
    """True when an id is an absolute filesystem path (the ./models and LM Studio
    scanners use the on-disk path as the id) rather than a repo id like org/name."""
    from pathlib import Path
    try:
        return Path(value).is_absolute()
    except Exception:
        return False


def _advertised_loader_id(info) -> Optional[str]:
    """The id to advertise for a scanned model: prefer a client-facing alias over
    an absolute filesystem path so /v1/models and the override key never expose a
    host path (the ./models and LM Studio scanners report the path as info.id)."""
    raw_id = getattr(info, "id", None)
    if not raw_id or not _is_abs_path_id(raw_id):
        return raw_id
    for alt in (getattr(info, "model_id", None), getattr(info, "display_name", None)):
        if alt and not _is_abs_path_id(alt):
            return alt
    # No clean alias: strip to a path-free public id so a host path is never advertised.
    return public_model_id(raw_id) or raw_id


def _resolve_load_dir(p):
    """The concrete dir holding the GGUFs. For an HF cache repo (``models--*``
    with ``snapshots/``) this is the latest snapshot dir, so /load takes the
    local branch instead of the download-capable repo-id branch."""
    from pathlib import Path

    try:
        if (p / "snapshots").is_dir():
            from routes.models import _resolve_hf_cache_realpath
            real = _resolve_hf_cache_realpath(p)
            if real:
                return Path(real)
    except Exception:
        pass
    return p


def _local_gguf_entry(loader_id: str, info) -> Optional[_LocalGgufEntry]:
    """Build an entry only when GGUF quants are on disk (not Transformers/
    safetensors), listing only on-disk quants. ``load_path`` is a concrete local
    path so /load resolves the variant locally and never fetches a remote one."""
    from pathlib import Path
    from utils.models.model_config import _is_mmproj, list_local_gguf_variants

    path = getattr(info, "path", None)
    if not isinstance(path, str):
        return None
    p = Path(path)
    try:
        if p.is_file():
            # A standalone .gguf loads by its own path; no quant sub-selection. An
            # mmproj companion (vision/audio projector) is not a servable model on
            # its own: _scan_models_dir's standalone-file pass does not filter it
            # the way the directory scan does, so reject it here or /v1/models would
            # advertise a projector and a switch could load it instead of the weights,
            # evicting the loaded model. The directory branch below is already mmproj
            # free (list_local_gguf_variants drops mmproj quants).
            if p.suffix.lower() != ".gguf" or _is_mmproj(p.name):
                return None
            return _LocalGgufEntry(loader_id, str(p), ())
        load_dir = _resolve_load_dir(p)
        variants, _ = list_local_gguf_variants(str(load_dir))
        quants = tuple(v.quant for v in variants if getattr(v, "quant", None))
        return _LocalGgufEntry(loader_id, str(load_dir), quants) if quants else None
    except Exception:
        return None


def info_has_local_gguf(info) -> bool:
    """True when *info* (a LocalModelInfo) points to on-disk GGUF weights the
    auto-switch path can load. Read from the files, not ``info.model_format``: the
    HF-cache scanner leaves model_format unset for GGUF snapshots, so a
    model_format filter would drop every cached GGUF. Lets /v1/models advertise
    exactly what /v1 can serve."""
    from pathlib import Path

    path = getattr(info, "path", None)
    # Ollama-link entries come from a scanner _build_index intentionally skips (it
    # creates symlinks on the request path), so their advertised ids never resolve.
    # Don't report them as servable, or /v1/models would list unswitchable models.
    if isinstance(path, str) and any(
        seg in (".studio_links", "ollama_links") for seg in Path(path).parts
    ):
        return False
    return _local_gguf_entry(getattr(info, "id", "") or "", info) is not None


def _build_index() -> dict[str, _LocalGgufEntry]:
    """Map normalized id/model_id/display_name -> local GGUF entry.

    Scans the same roots Unsloth's model picker lists (./models, the active plus
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
            if rp in seen_hf:
                return []
            seen_hf.add(rp)
            return _scan_hf_cache(directory)
        except Exception as exc:  # a missing/malformed root must skip, never crash the index
            logger.debug("auto-switch: skipping HF cache dir %r: %s", directory, exc)
            return []

    # Each source is guarded on its own so one bad root (a permission error, a
    # malformed cache) drops only that source, not the whole index.
    found: list = []
    try:
        found += _scan_models_dir(Path("./models").resolve())
    except Exception as exc:
        logger.debug("auto-switch: ./models scan failed: %s", exc)
    try:
        for hf_dir in (_resolve_hf_cache_dir(), legacy_hf_cache_dir(), hf_default_cache_dir()):
            found += _scan_hf_once(hf_dir)
    except Exception as exc:
        logger.debug("auto-switch: HF cache scan failed: %s", exc)
    try:
        for lm_dir in lmstudio_model_dirs():
            found += _scan_lmstudio_dir(lm_dir)
    except Exception as exc:
        logger.debug("auto-switch: LM Studio scan failed: %s", exc)
    try:
        from storage.studio_db import list_scan_folders
        for folder in list_scan_folders():
            try:
                fp = Path(folder["path"])
                found += (
                    _scan_models_dir(fp, limit = 200) + _scan_hf_once(fp) + _scan_lmstudio_dir(fp)
                )
            except Exception as exc:
                logger.debug("auto-switch: scan folder %r failed: %s", folder, exc)
    except Exception as exc:
        logger.debug("auto-switch: scan folders enumerate failed: %s", exc)
    for info in found:
        raw_id = getattr(info, "id", None)
        if not raw_id:
            continue
        # Skip what Unsloth hides from its pickers (validation probe, RAG embed
        # weights): not chat models, so never an auto-switch target.
        if _is_hidden_model(raw_id, getattr(info, "path", None)):
            continue
        # Advertise a client-facing alias, not an absolute filesystem path.
        loader_id = _advertised_loader_id(info)
        entry = _local_gguf_entry(loader_id, info)
        if entry is None:
            continue
        # Index every alias (including the path) so a client can resolve by any of
        # them, even though only the non-path loader_id is advertised.
        for key in (raw_id, getattr(info, "model_id", None), getattr(info, "display_name", None)):
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
        # Stamp AFTER the scan, not with the pre-scan ``now``: a multi-root scan on
        # an install with many local models can itself exceed the TTL, which would
        # store the cache already expired and make every request rebuild the index.
        _scan = (time.monotonic(), fresh)
        return fresh


def resolve_local_gguf(requested: str) -> Optional[tuple[str, Optional[str], str]]:
    """Return ``(load_path, gguf_variant, loader_id)`` for a local match, else None.

    ``load_path`` is the concrete on-disk path to hand /load (so it never fetches
    a remote), ``loader_id`` is the advertised id used as the launch-override key.
    ``requested`` is ``repo`` or ``repo:VARIANT``. An exact id match wins first
    (so ids containing a colon still resolve); else the last ``:VARIANT`` is split
    off and resolves only when that quant is on disk.
    """
    if not isinstance(requested, str) or not requested.strip():
        return None
    requested = requested.strip()
    try:
        index = _index()
        entry = index.get(requested.lower())
        if entry is not None:
            variant = entry.variants[0] if entry.variants else None
            return entry.load_path, variant, entry.loader_id

        base, sep, variant = requested.rpartition(":")
        if not sep:
            return None
        entry = index.get(base.strip().lower())
        if entry is None:
            return None
        wanted = variant.strip().lower()
        for v in entry.variants:
            if v.lower() == wanted:
                return entry.load_path, v, entry.loader_id
        return None
    except Exception:
        # Best-effort: any resolver failure falls through to the loaded model,
        # so a malformed name can never turn a servable request into a 500.
        return None
