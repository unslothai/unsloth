# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``general.*`` reader for GGUF headers, used by ``detect_mmproj_file`` to
pair weights and projectors via ``general.base_model.0.repo_url``. ~30 ms
per file, cached by resolved path and platform file identity."""

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
_CacheKey = Tuple[str, int, int, int, int, int]
_METADATA_CACHE: Dict[_CacheKey, Optional[Dict[str, str]]] = {}
_CACHE_LOCK = threading.Lock()
_CACHE_MAX_ENTRIES = 4096

# Separate cache for single bool capability keys (e.g. clip.has_audio_encoder),
# keyed by (file cache key, wanted key). None = key absent / file unreadable.
_BOOL_CACHE: Dict[Tuple[_CacheKey, str], Optional[bool]] = {}

_STRING_CACHE: Dict[Tuple[_CacheKey, str], Optional[str]] = {}

_TTS_AUDIO_TYPE_CACHE: Dict[_CacheKey, Optional[str]] = {}

# Whether the GGUF tensor table contains a sequence-classification head. None
# means the file could not be read or parsed, so callers can fail closed.
_CLASSIFIER_HEAD_CACHE: Dict[_CacheKey, Optional[bool]] = {}

# GGUF header dims for the staged UI in one cached pass (context_length, layer_count, moe_layer_count) so the staged
# sheet can size every slider before the model loads.
# None = unreadable / not a GGUF, and the native ``{arch}.context_length`` the UI shows before a load is read from here
# via read_gguf_context_length.
_DIMS_CACHE: Dict[_CacheKey, Optional[Dict[str, Optional[int]]]] = {}


# Cache the embedded speculative-head count separately for discovery, launch, and sizing.
_NEXTN_CACHE: Dict[_CacheKey, Optional[int]] = {}


def _cache_key(path: str) -> Optional[_CacheKey]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    try:
        resolved = str(Path(path).resolve())
    except OSError:
        resolved = str(path)
    return (
        resolved,
        st.st_mtime_ns,
        st.st_size,
        int(getattr(st, "st_ctime_ns", 0)),
        int(getattr(st, "st_dev", 0)),
        int(getattr(st, "st_ino", 0)),
    )


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

                    if vtype == 8 and key in _WANTED_GENERAL_KEYS:
                        slen_bytes = f.read(8)
                        if len(slen_bytes) < 8:
                            break
                        slen = struct.unpack("<Q", slen_bytes)[0]
                        if slen > 1 << 22:
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


def read_gguf_staged_dims(path: str) -> Optional[Dict[str, Optional[int]]]:
    """GGUF header dims for the staged-load UI in one cached pass:
    ``{"context_length", "layer_count", "moe_layer_count"}``. Each may be None
    when absent (moe_layer_count is 0 for a dense model). Returns ``None`` if not
    a GGUF / unreadable. Cached by (path, mtime, size). Lets the staged sheet size
    the context, GPU-layers and MoE sliders before the model loads."""
    key = _cache_key(path)
    if key is None:
        return None
    with _CACHE_LOCK:
        if key in _DIMS_CACHE:
            return _DIMS_CACHE[key]
    result = _parse_gguf_staged_dims(path)
    with _CACHE_LOCK:
        while len(_DIMS_CACHE) >= _CACHE_MAX_ENTRIES:
            try:
                _DIMS_CACHE.pop(next(iter(_DIMS_CACHE)))
            except StopIteration:
                break
        _DIMS_CACHE[key] = result
    return result


def read_gguf_context_length(path: str) -> Optional[int]:
    """Native training context length (``{arch}.context_length``), or ``None``.
    Thin accessor over read_gguf_staged_dims."""
    dims = read_gguf_staged_dims(path)
    return dims["context_length"] if dims else None


def _parse_gguf_arch_uints(path: str, wanted_suffixes: frozenset[str]) -> Optional[Dict[str, int]]:
    """Walk a GGUF header once and return requested architecture-namespaced
    uint (vtype 4/10) keys, e.g. ``{"block_count": 32}``.

    GGUF does not guarantee KV order, so matching uints are buffered until
    ``general.architecture`` identifies the active namespace. Returns ``None``
    if the file is unreadable/not GGUF, otherwise a possibly partial dict.
    """
    arch: Optional[str] = None
    buffered: Dict[str, int] = {}
    found: Dict[str, int] = {}
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

                    if vtype == 8 and key == "general.architecture":
                        slen_bytes = f.read(8)
                        if len(slen_bytes) < 8:
                            break
                        slen = struct.unpack("<Q", slen_bytes)[0]
                        if slen > 1 << 22:
                            break
                        sbytes = f.read(slen)
                        if len(sbytes) < slen:
                            break
                        arch = sbytes.decode("utf-8", "replace")
                        for suffix in wanted_suffixes:
                            full_key = f"{arch}.{suffix}"
                            if full_key in buffered:
                                found[suffix] = buffered[full_key]
                    elif vtype in (4, 10):
                        suffix = next(
                            (
                                candidate
                                for candidate in wanted_suffixes
                                if key.endswith(f".{candidate}")
                            ),
                            None,
                        )
                        if suffix is None:
                            if not _skip_gguf_value(f, vtype):
                                break
                            continue
                        width = 4 if vtype == 4 else 8
                        n_bytes = f.read(width)
                        if len(n_bytes) < width:
                            break
                        value = struct.unpack("<I" if vtype == 4 else "<Q", n_bytes)[0]
                        buffered[key] = value
                        if arch is not None and key == f"{arch}.{suffix}":
                            found[suffix] = value
                    else:
                        if not _skip_gguf_value(f, vtype):
                            break
                    if arch is not None and len(found) == len(wanted_suffixes):
                        break
                except (struct.error, UnicodeDecodeError):
                    break
    except OSError as e:
        logger.debug(f"_parse_gguf_arch_uints: cannot open {path}: {e}")
        return None
    except Exception as e:
        logger.debug(f"_parse_gguf_arch_uints: parse failure on {path}: {e}")
        return None
    return found


def read_gguf_nextn_predict_layers(path: str) -> Optional[int]:
    """Return the selected architecture's embedded NextN/MTP layer count.

    ``0`` is a real headless verdict. ``None`` means the key is absent or the
    header is unreadable, so callers that suppress a separate drafter can do so
    only on a positive value.
    """
    key = _cache_key(path)
    if key is None:
        return None
    with _CACHE_LOCK:
        if key in _NEXTN_CACHE:
            return _NEXTN_CACHE[key]
    values = _parse_gguf_arch_uints(path, frozenset({"nextn_predict_layers"}))
    result = values.get("nextn_predict_layers") if values is not None else None
    with _CACHE_LOCK:
        while len(_NEXTN_CACHE) >= _CACHE_MAX_ENTRIES:
            try:
                _NEXTN_CACHE.pop(next(iter(_NEXTN_CACHE)))
            except StopIteration:
                break
        _NEXTN_CACHE[key] = result
    return result


def _parse_gguf_staged_dims(path: str) -> Optional[Dict[str, Optional[int]]]:
    vals = _parse_gguf_arch_uints(
        path,
        frozenset(
            {
                "context_length",
                "block_count",
                "expert_count",
                "leading_dense_block_count",
            }
        ),
    )
    if vals is None:
        return None
    ctx = vals.get("context_length")
    block = vals.get("block_count")
    # A real context/layer count is positive; treat 0/garbage as absent
    context_length = ctx if ctx and ctx > 0 else None
    layer_count = block if block and block > 0 else None
    # MoE layer count = block_count - leading dense layers, only when experts
    # exist; else 0 (dense -> slider hidden). Mirrors n_moe_layers in
    # core/inference/llama_cpp.py.
    if not vals.get("expert_count") or not block:
        moe_layer_count: Optional[int] = 0
    else:
        moe_layer_count = max(0, block - (vals.get("leading_dense_block_count") or 0))
    return {
        "context_length": context_length,
        "layer_count": layer_count,
        "moe_layer_count": moe_layer_count,
    }


# Strings (8) and arrays (9) are handled inline.
_FIXED_VTYPE_SIZES: Dict[int, int] = {
    0: 1,
    1: 1,
    2: 2,
    3: 2,
    4: 4,
    5: 4,
    6: 4,
    7: 1,
    10: 8,
    11: 8,
    12: 8,
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
        if slen > 1 << 30:
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


def _parse_gguf_has_classifier_head(path: str) -> Optional[bool]:
    """Whether the GGUF tensor table contains llama.cpp's ``cls.*`` head."""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
            if len(head) < 24:
                return None
            magic, _version, tensor_count, kv_count = struct.unpack("<IIQQ", head)
            if magic != _GGUF_MAGIC or tensor_count > 1 << 20 or kv_count > 1 << 20:
                return None

            for _ in range(kv_count):
                klen_bytes = f.read(8)
                if len(klen_bytes) < 8:
                    return None
                klen = struct.unpack("<Q", klen_bytes)[0]
                if klen > 1 << 20 or len(f.read(klen)) < klen:
                    return None
                vtype_bytes = f.read(4)
                if len(vtype_bytes) < 4 or not _skip_gguf_value(
                    f, struct.unpack("<I", vtype_bytes)[0]
                ):
                    return None

            for _ in range(tensor_count):
                nlen_bytes = f.read(8)
                if len(nlen_bytes) < 8:
                    return None
                nlen = struct.unpack("<Q", nlen_bytes)[0]
                if nlen > 1 << 20:
                    return None
                name_bytes = f.read(nlen)
                ndim_bytes = f.read(4)
                if len(name_bytes) < nlen or len(ndim_bytes) < 4:
                    return None
                n_dimensions = struct.unpack("<I", ndim_bytes)[0]
                if n_dimensions > 16:
                    return None
                # dimensions (u64 each), ggml type (u32), and data offset (u64)
                trailer_size = n_dimensions * 8 + 4 + 8
                if len(f.read(trailer_size)) < trailer_size:
                    return None
                if name_bytes.decode("utf-8", "replace").startswith("cls."):
                    return True
    except OSError as e:
        logger.debug(f"_parse_gguf_has_classifier_head: cannot open {path}: {e}")
        return None
    except Exception as e:
        logger.debug(f"_parse_gguf_has_classifier_head: parse failure on {path}: {e}")
        return None
    return False


def _gguf_shard_has_classifier_head(path: str) -> Optional[bool]:
    key = _cache_key(path)
    if key is None:
        return None
    with _CACHE_LOCK:
        if key in _CLASSIFIER_HEAD_CACHE:
            return _CLASSIFIER_HEAD_CACHE[key]
    result = _parse_gguf_has_classifier_head(path)
    with _CACHE_LOCK:
        while len(_CLASSIFIER_HEAD_CACHE) >= _CACHE_MAX_ENTRIES:
            try:
                _CLASSIFIER_HEAD_CACHE.pop(next(iter(_CLASSIFIER_HEAD_CACHE)))
            except StopIteration:
                break
        _CLASSIFIER_HEAD_CACHE[key] = result
    return result


def _gguf_has_classifier_head(path: str) -> Optional[bool]:
    try:
        from utils.models.model_config import colocated_split_shards
        shards, complete = colocated_split_shards(Path(path))
    except Exception:
        return None
    if not complete:
        return None
    results = [_gguf_shard_has_classifier_head(str(shard)) for shard in shards]
    if any(result is True for result in results):
        return True
    return False if results and all(result is False for result in results) else None


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


_MAX_GGUF_VOCAB_ENTRIES = 2_000_000


def _parse_gguf_marker_tokens(path: str) -> Optional[Tuple[list[str], bool]]:
    """(marker tokens, whether the ids the SNAC probe detokenizes are codec codes)."""
    from utils.audio_tokens import GGUF_AUDIO_CLASSIFIER_TOKENS, SNAC_PROBE_TOKEN_IDS

    marker_bytes = {token.encode("utf-8"): token for token in GGUF_AUDIO_CLASSIFIER_TOKENS}

    try:
        with open(path, "rb") as f:
            head = f.read(24)
            if len(head) < 24:
                return None
            magic, _version, _tcount, kv_count = struct.unpack("<IIQQ", head)
            if magic != _GGUF_MAGIC:
                return None
            marker_tokens: dict[str, int] = {}
            token_types: Optional[bytes] = None
            snac_probe: Optional[dict[int, bool]] = None
            for _ in range(kv_count):
                klen_bytes = f.read(8)
                if len(klen_bytes) < 8:
                    return None
                klen = struct.unpack("<Q", klen_bytes)[0]
                if klen > 1 << 20:
                    return None
                key = f.read(klen).decode("utf-8", "replace")
                vt_bytes = f.read(4)
                if len(vt_bytes) < 4:
                    return None
                vtype = struct.unpack("<I", vt_bytes)[0]
                if key == "tokenizer.ggml.token_type" and vtype == 9:
                    raw_header = f.read(12)
                    if len(raw_header) != 12:
                        return None
                    atype, alen = struct.unpack("<IQ", raw_header)
                    if atype != 5 or alen > _MAX_GGUF_VOCAB_ENTRIES:
                        return None
                    raw_types = f.read(4 * alen)
                    if len(raw_types) != 4 * alen:
                        return None
                    token_types = raw_types
                    continue
                if key != "tokenizer.ggml.tokens" or vtype != 9:
                    if not _skip_gguf_value(f, vtype):
                        return None
                    continue
                raw_header = f.read(12)
                if len(raw_header) != 12:
                    return None
                atype, alen = struct.unpack("<IQ", raw_header)
                if atype != 8 or alen > _MAX_GGUF_VOCAB_ENTRIES:
                    return None
                # The serving detector asks what these two ids detokenize to, so the
                # vocabulary has to be read positionally, not just as a set of markers.
                snac_probe = dict.fromkeys(SNAC_PROBE_TOKEN_IDS, False)
                for index in range(alen):
                    raw_length = f.read(8)
                    if len(raw_length) != 8:
                        return None
                    slen = struct.unpack("<Q", raw_length)[0]
                    if slen > 1 << 20:
                        return None
                    raw = f.read(slen)
                    if len(raw) != slen:
                        return None
                    if index in snac_probe:
                        # Substring, not prefix: the detector asks what the id decodes
                        # to, and a tokenizer decoration would sit in front of the marker.
                        snac_probe[index] = b"<custom_token_" in raw
                    marker = marker_bytes.get(raw)
                    if marker is not None:
                        if marker in marker_tokens:
                            return None
                        marker_tokens[marker] = index
            if snac_probe is None:
                return None
            # llama.cpp's parse-special path does not treat plain NORMAL
            # vocabulary membership as a one-token capability marker.
            markers = [
                token
                for token, index in marker_tokens.items()
                if token_types is not None
                and 4 * (index + 1) <= len(token_types)
                and struct.unpack_from("<i", token_types, 4 * index)[0] in {2, 3, 4}
            ]
            return markers, all(snac_probe.values())
    except (OSError, struct.error) as e:
        logger.debug(f"_parse_gguf_marker_tokens: cannot read {path}: {e}")
    return None


def read_gguf_tts_audio_type(path: str) -> Optional[str]:
    from utils.audio_tokens import classify_gguf_vocab_audio_type, is_tts_audio_type

    fkey = _cache_key(path)
    if fkey is None:
        return None
    with _CACHE_LOCK:
        if fkey in _TTS_AUDIO_TYPE_CACHE:
            return _TTS_AUDIO_TYPE_CACHE[fkey]
    parsed = _parse_gguf_marker_tokens(path)
    audio_type = classify_gguf_vocab_audio_type(set(parsed[0]), parsed[1]) if parsed else None
    result = audio_type if is_tts_audio_type(audio_type) else None
    with _CACHE_LOCK:
        while len(_TTS_AUDIO_TYPE_CACHE) >= _CACHE_MAX_ENTRIES:
            try:
                _TTS_AUDIO_TYPE_CACHE.pop(next(iter(_TTS_AUDIO_TYPE_CACHE)))
            except StopIteration:
                break
        _TTS_AUDIO_TYPE_CACHE[fkey] = result
    return result


def read_gguf_chat_template(path: str) -> Optional[str]:
    template = _read_gguf_string(path, "tokenizer.chat_template")
    if isinstance(template, str) and template.strip():
        return template
    return None


def read_gguf_architecture(path: str) -> Optional[str]:
    """``general.architecture``, or ``None`` when absent / unreadable / not a GGUF.

    Reads only the requested key instead of walking the rest of the header."""
    architecture = _read_gguf_string(path, "general.architecture")
    if isinstance(architecture, str) and architecture.strip():
        return architecture.strip()
    return None


def read_mmproj_audio_capability(path: str) -> Optional[bool]:
    """``clip.has_audio_encoder`` from an mmproj GGUF (e.g. Gemma 4's
    gemma4ua): ``True``/``False`` if present, ``None`` if absent/unreadable.
    Flags audio-input models independently of tokenizer token names."""
    return _read_gguf_bool(path, "clip.has_audio_encoder")


def read_mmproj_projector_type(path: str) -> Optional[str]:
    """``clip.projector_type`` from an mmproj GGUF, or None if absent/unreadable.

    The family name llama.cpp keys its per-projector image-token limits on
    (``qwen3vl_merger``, ``gemma3``, ``pixtral``, ...), so a caller sizing the KV an
    image will occupy can look the ceiling up instead of assuming one.
    """
    return _read_gguf_string(path, "clip.projector_type")


def read_mmproj_vision_capability(path: str) -> Optional[bool]:
    """``clip.has_vision_encoder`` from an mmproj GGUF: ``True``/``False`` if
    present, ``None`` if absent/unreadable."""
    return _read_gguf_bool(path, "clip.has_vision_encoder")


def mmproj_capabilities(path: str) -> Tuple[bool, bool]:
    """``(declares_audio_encoder, accepts_image)`` for the projector at *path*.

    A projector serving both modalities declares both (Qwen2.5-Omni), so an audio-only
    declaration (ultravox, Voxtral, Qwen3-ASR) is evidence of no vision tower. One
    declaring neither -- an older convert, or a file this reader could not open -- is
    unknown rather than audio-only and stays image-capable.
    """
    vision = read_mmproj_vision_capability(path)
    audio = read_mmproj_audio_capability(path)
    return audio is True, (vision is True or audio is not True)


def mmproj_accepts_image(path: str) -> bool:
    """Whether images may be sent to the model this projector serves; see
    :func:`mmproj_capabilities`."""
    return mmproj_capabilities(path)[1]


def is_mmproj_by_metadata(meta: Optional[Dict[str, str]]) -> Optional[bool]:
    """True/False from ``general.type``; None means fall back to filename."""
    if not meta:
        return None
    t = meta.get("general.type")
    if t is None:
        return None
    return t.lower() == "mmproj"


def _normalize_url(url: str) -> Optional[str]:
    value = (url or "").strip().rstrip("/")
    if not value:
        return None
    if value.lower().endswith(".git"):
        value = value[:-4]
    lower = value.lower()
    has_url_host = False
    for scheme in ("https://", "http://"):
        if lower.startswith(scheme):
            value = value[len(scheme) :]
            has_url_host = True
            break
    if not has_url_host:
        return value
    host, separator, path = value.partition("/")
    return host.lower() + (separator + path if separator else "")


def _repo_path_from_url(url: str) -> Optional[str]:
    value = _normalize_url(url)
    if not value:
        return None
    lower = (url or "").strip().lower()
    if lower.startswith(("https://", "http://")):
        _, separator, path = value.partition("/")
        return path if separator and path else None
    return value


def _same_repo_reference(left: str, right: str) -> bool:
    left_normalized = _normalize_url(left)
    right_normalized = _normalize_url(right)
    if left_normalized == right_normalized:
        return True
    left_is_url = (left or "").strip().lower().startswith(("https://", "http://"))
    right_is_url = (right or "").strip().lower().startswith(("https://", "http://"))
    if left_is_url == right_is_url:
        return False
    hosted = left_normalized if left_is_url else right_normalized
    host, _, _ = hosted.partition("/")
    return host == "huggingface.co" and _repo_path_from_url(left) == _repo_path_from_url(right)


def _hf_repo_slug_from_url(url: str) -> Optional[str]:
    value = _repo_path_from_url(url)
    if not value:
        return None
    parts = [part for part in value.split("/") if part]
    if len(parts) < 2:
        return None
    return parts[-1]


def _slug_extends_base(derived: str, base: str) -> bool:
    if derived == base or not derived.startswith(base):
        return False
    if derived[len(base)] not in "-_.":
        return False
    suffix = derived[len(base) :].lstrip("-_.").lower()
    if not suffix:
        return False
    qualifier = suffix.split("-", 1)[0].split("_", 1)[0].split(".", 1)[0]
    return qualifier in {
        "gguf",
        "quant",
        "quantized",
        "qat",
        "awq",
        "gptq",
        "mlx",
        "unsloth",
        "bnb",
        "4bit",
        "8bit",
    }


def _weight_url_looks_like_derivative_of_projector(weight_url: str, projector_url: str) -> bool:
    weight_slug = _hf_repo_slug_from_url(weight_url)
    projector_slug = _hf_repo_slug_from_url(projector_url)
    if not weight_slug or not projector_slug:
        return False
    return _slug_extends_base(weight_slug, projector_slug)


def pairing_score(
    weight_meta: Optional[Dict[str, str]], mmproj_meta: Optional[Dict[str, str]]
) -> int:
    """Pairing confidence: 100 = base_model URL match, 90 = derivative URL,
    80 = basename + org, 60 = basename, -1 = definitive mismatch,
    0 = decide from filename."""
    if not weight_meta or not mmproj_meta:
        return 0

    w_url = weight_meta.get("general.base_model.0.repo_url")
    p_url = mmproj_meta.get("general.base_model.0.repo_url")
    w_base = weight_meta.get("general.basename")
    p_base = mmproj_meta.get("general.basename")
    if w_url and p_url:
        if _same_repo_reference(w_url, p_url):
            return 100
        if _weight_url_looks_like_derivative_of_projector(w_url, p_url):
            if not (w_base and p_base):
                return -1
            if w_base.lower() != p_base.lower():
                return -1
            return 90
        return -1

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


# GGUF architectures that intrinsically identify embedding models. Generic ``bert`` is
# deliberately absent: without pooling_type its required CLS/MEAN pooling cannot be recovered. A
# ``cls.*`` tensor makes an encoder a reranker instead, so matches are gated on the tensor table.
# The values are GGUF ``general.architecture`` strings, as llama.cpp defines them.
GGUF_EMBEDDING_ARCHITECTURES: frozenset[str] = frozenset(
    {
        "modern-bert",
        "nomic-bert",
        "nomic-bert-moe",
        "neo-bert",
        "jina-bert-v2",
        "jina-bert-v3",
        "eurobert",
        "gemma-embedding",
        "pangu-embedded",
        "llama-embed",
    }
)

# Name hints for model, file and intrinsic GGUF names whose architecture is not yet above.
_EMBEDDING_NAME_HINTS: tuple[str, ...] = (
    "nomic-embed",
    "llama-embed",
    "embed-text",
    "embedding",
    "bge-",
    "gte-",
    "e5-",
    "minilm",
)
_RERANKER_NAME_HINTS: tuple[str, ...] = ("reranker", "rerank")


def is_gguf_embedding_architecture(architecture: Optional[str]) -> bool:
    """True when ``architecture`` is a dedicated llama.cpp embedding arch."""
    return bool(architecture and architecture.strip().lower() in GGUF_EMBEDDING_ARCHITECTURES)


def _has_embedding_name_hint(value: Optional[str]) -> bool:
    return bool(value and any(needle in value.strip().lower() for needle in _EMBEDDING_NAME_HINTS))


def _has_reranker_name_hint(value: Optional[str]) -> bool:
    return bool(value and any(needle in value.strip().lower() for needle in _RERANKER_NAME_HINTS))


def is_gguf_embedding_model(
    gguf_path: str,
    model_identifier: Optional[str] = None,
    architecture: Optional[str] = None,
) -> bool:
    """Whether a GGUF should be launched with ``--embedding`` for /v1/embeddings."""
    meta = read_gguf_general_metadata(gguf_path) or {}
    identifier_basename = None
    if model_identifier:
        identifier_basename = model_identifier.strip().replace("\\", "/").rsplit("/", 1)[-1]
    try:
        file_basename: Optional[str] = Path(gguf_path).name
    except Exception:
        file_basename = None
    name_candidates = (
        identifier_basename,
        file_basename,
        meta.get("general.name"),
        meta.get("general.basename"),
        meta.get("general.base_model.0.name"),
    )
    if any(_has_reranker_name_hint(value) for value in name_candidates):
        return False

    arch = (architecture or meta.get("general.architecture") or "").strip().lower()
    if arch == "bert":
        # A classifier head can prove that generic BERT is a reranker
        # llama-server otherwise defaults to NONE and /v1/embeddings returns HTTP 400.
        return False
    if is_gguf_embedding_architecture(arch):
        # Generic BERT-family architectures also back cross-encoder rerankers.
        # Their standardized cls.* tensors are intrinsic evidence of that role;
        # an unreadable tensor table stays unclassified rather than guessing.
        return _gguf_has_classifier_head(gguf_path) is False
    return any(_has_embedding_name_hint(value) for value in name_candidates)


# Deliberately not re-exported: importing anything from THIS package runs utils.models.__init__,
# which pulls in model_config and therefore PyYAML, while core.inference.llama_cpp needs the
# verdict at import time. Import it from utils.gguf_archs.
