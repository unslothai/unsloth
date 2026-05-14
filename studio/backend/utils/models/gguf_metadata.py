# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Lightweight ``general.*`` metadata reader for GGUF files.

The Studio backend already has a hand-rolled GGUF header parser inside
``LlamaCppBackend._read_gguf_metadata`` for context length, attention
heads, chat template, etc. That parser is bound to a backend instance
and mutates ``self``, so it cannot be reused from the discovery path
in ``model_config.py``.

This module exposes a small free function that returns just the string
``general.*`` fields and the ``general.architecture``. It is used by
``detect_mmproj_file`` to pair model GGUFs with their projector via
``general.base_model.0.repo_url`` (and basename / organization
fallbacks), which is far more accurate than parsing the filename.

Reads cost about 30 ms per file even on multi-GB GGUFs because we walk
KV pairs sequentially and skip values we do not need. Results are
cached by (path, mtime, size) so repeated discovery calls on a stable
directory are free after the first scan.
"""

from __future__ import annotations

import os
import struct
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

from loggers import get_logger

logger = get_logger(__name__)


_GGUF_MAGIC = 0x46554747  # b"GGUF" little-endian u32

# String fields we care about. Anything else is skipped without
# decoding to keep header walks fast.
_WANTED_GENERAL_KEYS: frozenset[str] = frozenset(
    {
        "general.architecture",
        "general.type",
        "general.name",
        "general.basename",
        "general.organization",
        "general.size_label",
        "general.finetune",
        "general.base_model.0.name",
        "general.base_model.0.organization",
        "general.base_model.0.repo_url",
        "general.repo_url",
        "general.source.url",
        "general.source.repo_url",
        "general.source.huggingface.repository",
    }
)


# Cache key: (resolved_path, mtime_ns, size). Cache value is the
# extracted dict, or None when the file could not be parsed (so we do
# not re-pay the failed read on every discovery scan).
_CacheKey = Tuple[str, int, int]
_METADATA_CACHE: Dict[_CacheKey, Optional[Dict[str, str]]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_MAX_ENTRIES = 4096


def _cache_key(path: str) -> Optional[_CacheKey]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    try:
        resolved = str(Path(path).resolve())
    except OSError:
        resolved = str(path)
    return (resolved, st.st_mtime_ns, st.st_size)


def read_gguf_general_metadata(path: str) -> Optional[Dict[str, str]]:
    """Return the ``general.*`` string fields from a GGUF header.

    Returns ``None`` when the file is missing, unreadable, or not a
    GGUF. Returns ``{}`` for valid GGUFs that simply contain none of
    the keys in :data:`_WANTED_GENERAL_KEYS`.

    Cached by (resolved path, mtime_ns, size). Callers may invoke
    this on every discovery without paying repeated I/O after the
    first read.
    """
    key = _cache_key(path)
    if key is None:
        return None
    with _CACHE_LOCK:
        if key in _METADATA_CACHE:
            return _METADATA_CACHE[key]
    result = _parse_gguf_header(path)
    with _CACHE_LOCK:
        if len(_METADATA_CACHE) >= _CACHE_MAX_ENTRIES:
            # Drop an arbitrary entry. Header reads are cheap; we do
            # not need a true LRU here, just an upper bound on memory.
            try:
                _METADATA_CACHE.pop(next(iter(_METADATA_CACHE)))
            except StopIteration:
                pass
        _METADATA_CACHE[key] = result
    return result


def _parse_gguf_header(path: str) -> Optional[Dict[str, str]]:
    out: Dict[str, str] = {}
    try:
        with open(path, "rb") as f:
            head = f.read(24)
            if len(head) < 24:
                return None
            magic, _version, _tcount, kv_count = struct.unpack("<IIQQ", head)
            if magic != _GGUF_MAGIC:
                return None

            for _ in range(kv_count):
                try:
                    klen_bytes = f.read(8)
                    if len(klen_bytes) < 8:
                        break
                    klen = struct.unpack("<Q", klen_bytes)[0]
                    if klen > 1 << 20:  # 1 MB key length sanity bound
                        break
                    kbytes = f.read(klen)
                    if len(kbytes) < klen:
                        break
                    key = kbytes.decode("utf-8", "replace")
                    vt_bytes = f.read(4)
                    if len(vt_bytes) < 4:
                        break
                    vtype = struct.unpack("<I", vt_bytes)[0]

                    if vtype == 8 and key in _WANTED_GENERAL_KEYS:
                        slen_bytes = f.read(8)
                        if len(slen_bytes) < 8:
                            break
                        slen = struct.unpack("<Q", slen_bytes)[0]
                        if slen > 1 << 22:  # 4 MB string length sanity bound
                            break
                        sbytes = f.read(slen)
                        if len(sbytes) < slen:
                            break
                        out[key] = sbytes.decode("utf-8", "replace")
                    else:
                        if not _skip_gguf_value(f, vtype):
                            break
                except (struct.error, UnicodeDecodeError):
                    break
    except OSError as e:
        logger.debug(f"read_gguf_general_metadata: cannot open {path}: {e}")
        return None
    except Exception as e:
        logger.debug(f"read_gguf_general_metadata: parse failure on {path}: {e}")
        return None
    return out


# Fixed-size GGUF value types. Values not in this map (strings, arrays)
# are handled inline.
_FIXED_VTYPE_SIZES: Dict[int, int] = {
    0: 1,  # uint8
    1: 1,  # int8
    2: 2,  # uint16
    3: 2,  # int16
    4: 4,  # uint32
    5: 4,  # int32
    6: 4,  # float32
    7: 1,  # bool
    10: 8,  # uint64
    11: 8,  # int64
    12: 8,  # float64
}


def _skip_gguf_value(f, vtype: int) -> bool:
    """Advance ``f`` past one GGUF value. Returns False on truncation
    or unknown type so the caller can stop walking the header."""
    if vtype == 8:  # STRING
        slen_bytes = f.read(8)
        if len(slen_bytes) < 8:
            return False
        slen = struct.unpack("<Q", slen_bytes)[0]
        if slen > 1 << 30:  # 1 GB sanity bound
            return False
        return len(f.read(slen)) == slen
    if vtype == 9:  # ARRAY
        head = f.read(12)
        if len(head) < 12:
            return False
        atype, alen = struct.unpack("<IQ", head)
        if alen > 1 << 30:
            return False
        if atype == 8:
            for _ in range(alen):
                slen_bytes = f.read(8)
                if len(slen_bytes) < 8:
                    return False
                slen = struct.unpack("<Q", slen_bytes)[0]
                if slen > 1 << 30:
                    return False
                if len(f.read(slen)) != slen:
                    return False
            return True
        sz = _FIXED_VTYPE_SIZES.get(atype)
        if sz is None:
            return False
        total = sz * alen
        return len(f.read(total)) == total
    sz = _FIXED_VTYPE_SIZES.get(vtype)
    if sz is None:
        return False
    return len(f.read(sz)) == sz


def is_mmproj_by_metadata(meta: Optional[Dict[str, str]]) -> Optional[bool]:
    """Return True when ``general.type == 'mmproj'``, False when it is
    a non-mmproj value, and None when ``meta`` is missing or has no
    ``general.type`` field. Callers should fall back to a filename
    check on a None result."""
    if not meta:
        return None
    t = meta.get("general.type")
    if t is None:
        return None
    return t.lower() == "mmproj"


def pairing_score(
    weight_meta: Optional[Dict[str, str]],
    mmproj_meta: Optional[Dict[str, str]],
) -> int:
    """Return a numeric score for how confidently ``mmproj_meta``
    belongs to the model described by ``weight_meta``.

    Convention:
      * ``100`` -- ``general.base_model.0.repo_url`` matches exactly.
      * ``80``  -- ``general.basename`` and ``general.base_model.0.organization``
                   both match.
      * ``60``  -- ``general.basename`` matches.
      * ``-1``  -- both files have metadata that disagrees on every
                   strong signal we tested (definitive mismatch).
      * ``0``   -- neither side carries enough metadata to decide;
                   the caller should fall back to a filename-based
                   family-token check.
    """
    if not weight_meta or not mmproj_meta:
        return 0

    w_url = weight_meta.get("general.base_model.0.repo_url")
    p_url = mmproj_meta.get("general.base_model.0.repo_url")
    if w_url and p_url:
        return 100 if w_url.strip().rstrip("/") == p_url.strip().rstrip("/") else -1

    w_base = weight_meta.get("general.basename")
    p_base = mmproj_meta.get("general.basename")
    w_org = weight_meta.get("general.base_model.0.organization") or weight_meta.get(
        "general.organization"
    )
    p_org = mmproj_meta.get("general.base_model.0.organization") or mmproj_meta.get(
        "general.organization"
    )
    if w_base and p_base and w_org and p_org:
        if w_base.lower() == p_base.lower() and w_org.lower() == p_org.lower():
            return 80
        return -1

    if w_base and p_base:
        return 60 if w_base.lower() == p_base.lower() else -1

    return 0
