# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``general.*`` reader for GGUF headers, used by ``detect_mmproj_file`` to
pair weights and projectors via ``general.base_model.0.repo_url``. ~30 ms
per file, cached by (path, mtime, size)."""

from __future__ import annotations

import os
import struct
import threading
from pathlib import Path
from typing import Dict, Optional, Tuple

from loggers import get_logger

logger = get_logger(__name__)


_GGUF_MAGIC = 0x46554747  # b"GGUF" LE u32

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


# Cache failed parses too so a broken file is not retried each scan.
_CacheKey = Tuple[str, int, int]
_METADATA_CACHE: Dict[_CacheKey, Optional[Dict[str, str]]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_MAX_ENTRIES = 4096

# Separate cache for single bool capability keys (e.g. clip.has_audio_encoder),
# keyed by (file cache key, wanted key). None = key absent / file unreadable.
_BOOL_CACHE: Dict[Tuple[_CacheKey, str], Optional[bool]] = {}

_STRING_CACHE: Dict[Tuple[_CacheKey, str], Optional[str]] = {}

# Native training context length (``{arch}.context_length``). None = absent /
# unreadable. Lets the UI show the real context ceiling before a model loads.
_CONTEXT_CACHE: Dict[_CacheKey, Optional[int]] = {}


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
    """Return ``general.*`` strings from a GGUF header, or ``None`` if the
    file is missing, unreadable, or not a GGUF. ``{}`` means valid but
    carrying none of the wanted keys."""
    key = _cache_key(path)
    if key is None:
        return None
    with _CACHE_LOCK:
        if key in _METADATA_CACHE:
            return _METADATA_CACHE[key]
    result = _parse_gguf_header(path)
    with _CACHE_LOCK:
        # Arbitrary eviction; header reads are cheap so true LRU is overkill.
        while len(_METADATA_CACHE) >= _CACHE_MAX_ENTRIES:
            try:
                _METADATA_CACHE.pop(next(iter(_METADATA_CACHE)))
            except StopIteration:
                break
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
                    if klen > 1 << 20:  # 1 MB sanity bound
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
                        if slen > 1 << 22:  # 4 MB sanity bound
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


def read_gguf_context_length(path: str) -> Optional[int]:
    """Return the GGUF's native training context length (``{arch}.context_length``),
    or ``None`` if missing/unreadable/not a GGUF. Cached by (path, mtime, size).
    Lets the UI populate the context slider before the model is loaded."""
    key = _cache_key(path)
    if key is None:
        return None
    with _CACHE_LOCK:
        if key in _CONTEXT_CACHE:
            return _CONTEXT_CACHE[key]
    result = _parse_gguf_context_length(path)
    with _CACHE_LOCK:
        while len(_CONTEXT_CACHE) >= _CACHE_MAX_ENTRIES:
            try:
                _CONTEXT_CACHE.pop(next(iter(_CONTEXT_CACHE)))
            except StopIteration:
                break
        _CONTEXT_CACHE[key] = result
    return result


def _parse_gguf_context_length(path: str) -> Optional[int]:
    # The context key is architecture-namespaced (``llama.context_length`` etc.),
    # so we learn the key only after reading ``general.architecture``. GGUF writes
    # general.* before arch.* keys, matching the loader's own parser.
    ctx_key: Optional[str] = None
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
                    if klen > 1 << 20:  # 1 MB sanity bound
                        break
                    kbytes = f.read(klen)
                    if len(kbytes) < klen:
                        break
                    key = kbytes.decode("utf-8", "replace")
                    vt_bytes = f.read(4)
                    if len(vt_bytes) < 4:
                        break
                    vtype = struct.unpack("<I", vt_bytes)[0]

                    if vtype == 8 and key == "general.architecture":
                        slen_bytes = f.read(8)
                        if len(slen_bytes) < 8:
                            break
                        slen = struct.unpack("<Q", slen_bytes)[0]
                        if slen > 1 << 22:  # 4 MB sanity bound
                            break
                        sbytes = f.read(slen)
                        if len(sbytes) < slen:
                            break
                        ctx_key = f"{sbytes.decode('utf-8', 'replace')}.context_length"
                    elif ctx_key is not None and key == ctx_key and vtype in (4, 10):
                        width = 4 if vtype == 4 else 8
                        n_bytes = f.read(width)
                        if len(n_bytes) < width:
                            break
                        value = struct.unpack("<I" if vtype == 4 else "<Q", n_bytes)[0]
                        # A real context length is positive; treat 0/garbage as
                        # absent so the UI never builds a slider with max < min.
                        return value if value > 0 else None
                    else:
                        if not _skip_gguf_value(f, vtype):
                            break
                except (struct.error, UnicodeDecodeError):
                    break
    except OSError as e:
        logger.debug(f"read_gguf_context_length: cannot open {path}: {e}")
        return None
    except Exception as e:
        logger.debug(f"read_gguf_context_length: parse failure on {path}: {e}")
        return None
    return None


# Strings (8) and arrays (9) are handled inline.
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
    """Advance past one GGUF value. ``f.seek(.., 1)`` past EOF is legal on a
    regular file, so truncation is caught on the next read; return False only
    for unknown types or sanity-bound overflow."""
    if vtype == 8:  # STRING
        slen_bytes = f.read(8)
        if len(slen_bytes) < 8:
            return False
        slen = struct.unpack("<Q", slen_bytes)[0]
        if slen > 1 << 30:  # 1 GB sanity bound
            return False
        f.seek(slen, 1)
        return True
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
                f.seek(slen, 1)
            return True
        sz = _FIXED_VTYPE_SIZES.get(atype)
        if sz is None:
            return False
        f.seek(sz * alen, 1)
        return True
    sz = _FIXED_VTYPE_SIZES.get(vtype)
    if sz is None:
        return False
    f.seek(sz, 1)
    return True


def _parse_gguf_bool(path: str, wanted_key: str) -> Optional[bool]:
    """Bool value of ``wanted_key`` (GGUF vtype 7), or ``None`` if absent /
    unreadable. Mirrors ``_parse_gguf_header`` for a single bool key."""
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
                    if klen > 1 << 20:  # 1 MB sanity bound
                        break
                    kbytes = f.read(klen)
                    if len(kbytes) < klen:
                        break
                    key = kbytes.decode("utf-8", "replace")
                    vt_bytes = f.read(4)
                    if len(vt_bytes) < 4:
                        break
                    vtype = struct.unpack("<I", vt_bytes)[0]

                    if key == wanted_key and vtype == 7:  # BOOL (1 byte)
                        bbyte = f.read(1)
                        if len(bbyte) < 1:
                            break
                        return bbyte[0] != 0
                    if not _skip_gguf_value(f, vtype):
                        break
                except (struct.error, UnicodeDecodeError):
                    break
    except OSError as e:
        logger.debug(f"_parse_gguf_bool: cannot open {path}: {e}")
        return None
    except Exception as e:
        logger.debug(f"_parse_gguf_bool: parse failure on {path}: {e}")
        return None
    return None


def _read_gguf_bool(path: str, wanted_key: str) -> Optional[bool]:
    """Cached single-bool-key read, keyed by (path, mtime, size, wanted_key)."""
    fkey = _cache_key(path)
    if fkey is None:
        return None
    ckey = (fkey, wanted_key)
    with _CACHE_LOCK:
        if ckey in _BOOL_CACHE:
            return _BOOL_CACHE[ckey]
    result = _parse_gguf_bool(path, wanted_key)
    with _CACHE_LOCK:
        while len(_BOOL_CACHE) >= _CACHE_MAX_ENTRIES:
            try:
                _BOOL_CACHE.pop(next(iter(_BOOL_CACHE)))
            except StopIteration:
                break
        _BOOL_CACHE[ckey] = result
    return result


def _parse_gguf_string(path: str, wanted_key: str) -> Optional[str]:
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
                    if klen > 1 << 20:
                        break
                    kbytes = f.read(klen)
                    if len(kbytes) < klen:
                        break
                    key = kbytes.decode("utf-8", "replace")
                    vt_bytes = f.read(4)
                    if len(vt_bytes) < 4:
                        break
                    vtype = struct.unpack("<I", vt_bytes)[0]

                    if key == wanted_key and vtype == 8:
                        slen_bytes = f.read(8)
                        if len(slen_bytes) < 8:
                            break
                        slen = struct.unpack("<Q", slen_bytes)[0]
                        if slen > 1 << 22:
                            break
                        sbytes = f.read(slen)
                        if len(sbytes) < slen:
                            break
                        return sbytes.decode("utf-8", "replace")
                    if not _skip_gguf_value(f, vtype):
                        break
                except (struct.error, UnicodeDecodeError):
                    break
    except OSError as e:
        logger.debug(f"_parse_gguf_string: cannot open {path}: {e}")
        return None
    except Exception as e:
        logger.debug(f"_parse_gguf_string: parse failure on {path}: {e}")
        return None
    return None


def _read_gguf_string(path: str, wanted_key: str) -> Optional[str]:
    fkey = _cache_key(path)
    if fkey is None:
        return None
    ckey = (fkey, wanted_key)
    with _CACHE_LOCK:
        if ckey in _STRING_CACHE:
            return _STRING_CACHE[ckey]
    result = _parse_gguf_string(path, wanted_key)
    with _CACHE_LOCK:
        while len(_STRING_CACHE) >= _CACHE_MAX_ENTRIES:
            try:
                _STRING_CACHE.pop(next(iter(_STRING_CACHE)))
            except StopIteration:
                break
        _STRING_CACHE[ckey] = result
    return result


def read_gguf_chat_template(path: str) -> Optional[str]:
    template = _read_gguf_string(path, "tokenizer.chat_template")
    if isinstance(template, str) and template.strip():
        return template
    return None


def read_mmproj_audio_capability(path: str) -> Optional[bool]:
    """``clip.has_audio_encoder`` from an mmproj GGUF (e.g. Gemma 4's
    gemma4ua): ``True``/``False`` if present, ``None`` if absent/unreadable.
    Flags audio-input models independently of tokenizer token names."""
    return _read_gguf_bool(path, "clip.has_audio_encoder")


def is_mmproj_by_metadata(meta: Optional[Dict[str, str]]) -> Optional[bool]:
    """True/False from ``general.type``; None means fall back to filename."""
    if not meta:
        return None
    t = meta.get("general.type")
    if t is None:
        return None
    return t.lower() == "mmproj"


def pairing_score(
    weight_meta: Optional[Dict[str, str]], mmproj_meta: Optional[Dict[str, str]]
) -> int:
    """Pairing confidence: 100 = base_model URL match, 80 = basename + org,
    60 = basename, -1 = definitive mismatch, 0 = decide from filename."""
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
