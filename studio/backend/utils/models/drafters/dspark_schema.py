# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Which Inkling 2 DSpark drafter layout a file has, and which layouts a llama-server build loads.

llama.cpp's Inkling 2 DSpark loader (ggml-org/llama.cpp#32218) changed its GGUF schema between revisions:

* ``fused`` (0a9105de4, Unsloth prebuilts b12588 and b12631): a ``dspark_hidden_out`` head and a ``dspark_fuse``
  layer mix, ``markov_w1`` rows padded to ``dflash.vocab_size``.
* ``markov2`` (e4b1cb531 and later): ``markov_ctx_mix`` / ``markov_ctx_norm`` context mixing,
  ``markov_w1`` rows equal to the real vocabulary, ``dflash.has_markov_context``.

A loader for one schema rejects the other at its tensor-shape check, which costs failed llama-server launches before
Studio falls back to no drafting. Every other DSpark drafter (``generic``: Qwen3, MiniMax, DeepSeek-V4 ...) also
carries ``markov_w2``, loads on every DSpark build and is never filtered here.
"""

from __future__ import annotations

import logging
import os
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

from utils.models.gguf_metadata import read_gguf_header_summary

logger = logging.getLogger(__name__)

DSPARK_SCHEMA_FUSED = "fused"
DSPARK_SCHEMA_MARKOV2 = "markov2"
DSPARK_SCHEMA_GENERIC = "generic"
INKLING_DSPARK_SCHEMAS = frozenset({DSPARK_SCHEMA_FUSED, DSPARK_SCHEMA_MARKOV2})

# Which Inkling 2 layout wins when a build loads both and both are on offer (local candidates). Fused: at equal
# acceptance it decodes only the drafted rows while the markov2 head runs its full 8-token block every round; on B200
# fused at depth 2 is 1.24x / 1.15x (greedy / temperature 1.0) against markov2's best (depth 3) 1.16x / 1.03x.
PREFERRED_INKLING_DSPARK_SCHEMA = DSPARK_SCHEMA_FUSED

# Tensor names that only one Inkling 2 layout has. llama.cpp keys the same way: 0a9105de4 picks its Inkling head
# on dspark_hidden_out, e4b1cb531 turns on the context-mixing head on markov_ctx_norm.
_FUSED_TENSORS = ("dspark_hidden_out.weight", "dspark_fuse.weight")
_MARKOV2_TENSORS = ("markov_ctx_mix.weight", "markov_ctx_norm.weight")

# The same names as they sit in llama.cpp's tensor-name table: only a build with that loader has the string.
_FUSED_LIB_MARKER = b"dspark_hidden_out"
_MARKOV2_LIB_MARKER = b"markov_ctx_mix"

# Published Unsloth prebuilts that load an Inkling 2 drafter, from UNSLOTH_PREBUILT_INFO.json's release_tag. Both pin
# ggml-org#32218 at a fused revision (b12631 via unslothai/llama.cpp#405). Earlier tags have no Inkling 2 DSpark head;
# later tags are recognised by the library scan, so a pin move needs no Studio release.
_RELEASE_TAG_SCHEMAS: dict[str, frozenset[str]] = {
    "b12588-mix-c0c3de8": frozenset({DSPARK_SCHEMA_FUSED}),
    "b12631-mix-acc0843": frozenset({DSPARK_SCHEMA_FUSED}),
}

# A custom LLAMA_SERVER_PATH build the release table and the scan cannot read: comma-separated schema names
# ("fused,markov2", or "none"). Ignored, with a warning, when it contradicts what the launch binary shows.
DSPARK_SCHEMAS_ENV = "UNSLOTH_DSPARK_DRAFTER_SCHEMAS"

_SCAN_CHUNK = 8 << 20
_scan_lock = threading.Lock()
# (file path, mtime_ns, size) -> the loader markers the file holds.
_scan_cache: dict[tuple[str, int, int], Optional[frozenset[bytes]]] = {}


def dspark_schema_from_tensors(tensor_names: Iterable[str], kvs: Optional[dict] = None) -> str:
    names = set(tensor_names)
    if any(t in names for t in _FUSED_TENSORS):
        return DSPARK_SCHEMA_FUSED
    if (
        any(t in names for t in _MARKOV2_TENSORS)
        or (kvs or {}).get("dflash.has_markov_context") is True
    ):
        return DSPARK_SCHEMA_MARKOV2
    return DSPARK_SCHEMA_GENERIC


def dspark_drafter_schema(path: str) -> Optional[str]:
    """``fused`` / ``markov2`` / ``generic`` from the header, or None when it cannot be read."""
    summary = read_gguf_header_summary(path)
    if summary is None:
        return None
    kvs, tensors = summary
    return dspark_schema_from_tensors(tensors, kvs)


def _env_schemas() -> Optional[frozenset[str]]:
    raw = os.environ.get(DSPARK_SCHEMAS_ENV, "")
    if not raw.strip():
        return None
    chosen = frozenset(s.strip().lower() for s in raw.split(",") if s.strip())
    if "none" in chosen:
        return frozenset()
    chosen &= INKLING_DSPARK_SCHEMAS
    return chosen or None


# Libraries that can hold llama.cpp's tensor-name table: libllama itself, the split libllama-common, the server's own
# impl library (one bundle layout links the server statically into it), on Linux, macOS and Windows.
_LLAMA_LIBRARY_STEMS = frozenset({"llama", "llama-common", "llama-server-impl"})


def _is_llama_library(name: str) -> bool:
    lower = name.lower()
    if not (".so" in lower or lower.endswith(".dylib") or lower.endswith(".dll")):
        return False
    stem = lower.split(".", 1)[0]
    return stem.removeprefix("lib") in _LLAMA_LIBRARY_STEMS


def _library_candidates(binary: str) -> list[Path]:
    """Every llama library beside llama-server (and in ../lib), plus the binary itself for a static build."""
    bin_path = Path(binary)
    out: list[Path] = []
    for directory in (bin_path.parent, bin_path.parent.parent / "lib"):
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not _is_llama_library(entry.name):
                continue
            try:
                if entry.is_file():
                    out.append(entry.resolve())
            except OSError:
                continue
    out.append(bin_path)
    # A versioned soname and its symlinks resolve to one file.
    return list(dict.fromkeys(out))


def _scan_file(path: Path) -> Optional[frozenset[bytes]]:
    """The loader markers ``path`` holds, or None when it cannot be read."""
    try:
        st = path.stat()
    except OSError:
        return None
    key = (str(path), st.st_mtime_ns, st.st_size)
    with _scan_lock:
        if key in _scan_cache:
            return _scan_cache[key]
    found: set[bytes] = set()
    markers = (_FUSED_LIB_MARKER, _MARKOV2_LIB_MARKER)
    overlap = max(len(m) for m in markers) - 1
    try:
        with open(path, "rb") as f:
            tail = b""
            while len(found) < len(markers):
                chunk = f.read(_SCAN_CHUNK)
                if not chunk:
                    break
                window = tail + chunk
                for marker in markers:
                    if marker not in found and marker in window:
                        found.add(marker)
                tail = window[-overlap:]
    except OSError:
        return None
    result = frozenset(found)
    with _scan_lock:
        _scan_cache[key] = result
    return result


def _scan_build(binary: str) -> Optional[frozenset[str]]:
    """Schemas named anywhere in the build (union over every llama library and the binary). A build where no file
    names either Inkling 2 head stays inconclusive, never "loads none": a split or renamed library may hold them."""
    found: set[bytes] = set()
    for candidate in _library_candidates(binary):
        markers = _scan_file(candidate)
        if markers:
            found |= markers
    schemas = set()
    if _FUSED_LIB_MARKER in found:
        schemas.add(DSPARK_SCHEMA_FUSED)
    if _MARKOV2_LIB_MARKER in found:
        schemas.add(DSPARK_SCHEMA_MARKOV2)
    return frozenset(schemas) or None


def _release_tag_of(binary: str) -> Optional[str]:
    try:
        from utils.llama_cpp_freshness import read_install_marker
        marker = read_install_marker(binary) or {}
    except Exception:
        return None
    if str(marker.get("published_repo") or "unslothai/llama.cpp") != "unslothai/llama.cpp":
        return None
    tag = marker.get("release_tag")
    return str(tag) if tag else None


_env_conflict_warned: set[tuple] = set()


def binary_dspark_schemas(binary: Optional[str]) -> Optional[frozenset[str]]:
    """The Inkling 2 drafter schemas ``binary`` loads, or None when that cannot be told (then nothing is filtered).

    Order: the published release table, then a scan of every llama library and the binary for the loaders' tensor
    names, then ``UNSLOTH_DSPARK_DRAFTER_SCHEMAS``. The variable is an escape hatch for a build neither can read (a
    custom ``LLAMA_SERVER_PATH`` build, a stripped library); it never overrides what the launch binary itself shows,
    which would hand llama-server a drafter it refuses. Never raises.
    """
    env = _env_schemas()
    conclusive: Optional[frozenset[str]] = None
    if binary:
        try:
            tag = _release_tag_of(binary)
            conclusive = _RELEASE_TAG_SCHEMAS.get(tag) if tag else None
            if conclusive is None:
                conclusive = _scan_build(binary)
        except Exception:
            conclusive = None
    if conclusive is None:
        return env
    if env is not None and env != conclusive:
        key = (binary, env, conclusive)
        if key not in _env_conflict_warned:
            _env_conflict_warned.add(key)
            logger.warning(
                "%s=%s ignored: %s loads %s (release table or library scan)",
                DSPARK_SCHEMAS_ENV,
                ",".join(sorted(env)) or "none",
                binary,
                ",".join(sorted(conclusive)) or "no Inkling 2 DSpark drafter",
            )
    return conclusive


_default_binary_lock = threading.Lock()
_default_binary: tuple[float, Optional[str]] = (0.0, None)


def _default_llama_server_binary() -> Optional[str]:
    """The llama-server Studio would launch; looked up at most every 30 s (local scans run per listing)."""
    global _default_binary
    import time

    now = time.monotonic()
    with _default_binary_lock:
        stamp, cached = _default_binary
        if stamp and now - stamp < 30.0:
            return cached
    found: Optional[str] = None
    core = sys.modules.get("core.inference.llama_cpp")
    try:
        if core is None:
            from core.inference import llama_cpp as core
        found = core.LlamaCppBackend._find_llama_server_binary()
    except Exception:
        found = None
    with _default_binary_lock:
        _default_binary = (now, found)
    return found


_UNSET = object()


def supported_dspark_schemas(binary: object = _UNSET) -> Optional[frozenset[str]]:
    """``binary_dspark_schemas`` of ``binary``, or of the llama-server Studio would launch when not given."""
    if binary is _UNSET:
        binary = _default_llama_server_binary()
    return binary_dspark_schemas(binary if isinstance(binary, str) else None)


def _rows(shape: Optional[tuple]) -> Optional[int]:
    # ggml order: ne0 is the inner dimension, so a [rank, vocab] tensor has vocab rows.
    return int(shape[1]) if shape is not None and len(shape) >= 2 else None


def _target_vocab(target_path: str) -> tuple[Optional[int], Optional[int]]:
    """(n_vocab from the embedding / output rows, ``{arch}.vocab_size_unpadded``) of the target."""
    summary = read_gguf_header_summary(target_path)
    if summary is None:
        return None, None
    kvs, tensors = summary
    n_vocab = _rows(tensors.get("token_embd.weight")) or _rows(tensors.get("output.weight"))
    arch = kvs.get("general.architecture")
    unpadded = kvs.get(f"{arch}.vocab_size_unpadded") if isinstance(arch, str) else None
    return n_vocab, unpadded if isinstance(unpadded, int) else None


def schema_incompatibility(
    schema: Optional[str], supported: Optional[frozenset[str]]
) -> Optional[str]:
    """Why a build that loads ``supported`` rejects a drafter of ``schema``, else None."""
    if schema not in INKLING_DSPARK_SCHEMAS or supported is None or schema in supported:
        return None
    markers = (
        "markov_ctx_mix context head, unpadded markov_w1"
        if schema == DSPARK_SCHEMA_MARKOV2
        else "dspark_hidden_out / dspark_fuse head, padded markov_w1"
    )
    loads = ", ".join(sorted(supported)) if supported else "no Inkling 2 DSpark drafter"
    return f"{schema} Inkling 2 DSpark layout ({markers}); this llama-server loads {loads}"


def dspark_drafter_incompatibility(
    path: str,
    target_path: Optional[str] = None,
    *,
    supported: object = _UNSET,
) -> Optional[str]:
    """Why the llama-server Studio launches (or one loading ``supported``) cannot load this drafter, else None.

    Fails open: an unreadable header, a generic DSpark drafter or a build whose support cannot be told return None,
    so the launch and its existing fallback decide, as before this check existed.
    """
    schema = dspark_drafter_schema(path)
    if schema not in INKLING_DSPARK_SCHEMAS:
        return None
    if supported is _UNSET:
        supported = supported_dspark_schemas()
    reason = schema_incompatibility(schema, supported)  # type: ignore[arg-type]
    if reason is not None or not target_path:
        return reason
    summary = read_gguf_header_summary(path)
    rows = _rows(summary[1].get("markov_w1.weight")) if summary else None
    n_vocab, unpadded = _target_vocab(target_path)
    expected = n_vocab if schema == DSPARK_SCHEMA_FUSED else unpadded
    if rows is not None and expected is not None and rows != expected:
        return f"markov_w1 has {rows} vocabulary rows, the target model needs {expected}"
    return None


def dspark_schema_rank(schema: Optional[str]) -> int:
    """Sort key among drafters a build loads: the preferred Inkling 2 layout first, anything else unordered."""
    return 0 if schema == PREFERRED_INKLING_DSPARK_SCHEMA else 1


# (repo, commit, filename) -> schema read from a ranged header fetch. Only a probe pinned to a commit is memoized:
# a name on a moving ref (main) can be pointed at another layout by a repo move, so those are read fresh each time.
_REMOTE_TTL_S = 600.0
_remote_cache: dict[tuple[str, str, str], tuple[float, str]] = {}
_remote_lock = threading.Lock()
_REMOTE_FIRST_RANGE = 16 << 20
_REMOTE_CAP = 64 << 20


_MARKOV_CONTEXT_KV = "dflash.has_markov_context"
_GGUF_BOOL = 7


def _header_bool_kv(header: bytes, key: str) -> Optional[bool]:
    """A boolean KV from raw GGUF header bytes (u64 key length, key, u32 type, value), or None if absent."""
    needle = struct.pack("<Q", len(key)) + key.encode()
    at = header.find(needle)
    if at < 0:
        return None
    pos = at + len(needle)
    if pos + 5 > len(header) or struct.unpack_from("<I", header, pos)[0] != _GGUF_BOOL:
        return None
    return header[pos + 4] != 0


def remote_dspark_drafter_schema(
    repo_id: str,
    filename: str,
    token: Optional[str] = None,
    *,
    size: Optional[int] = None,
    revision: Optional[str] = None,
) -> Optional[str]:
    """The schema of a Hub drafter from its header alone (Range requests, ~12 MB for Inkling 2), or None on any
    failure. Only called when the build loads one Inkling 2 layout, so a download it could not load is skipped."""
    key = (repo_id, revision, filename) if revision else None
    if key is not None:
        with _remote_lock:
            hit = _remote_cache.get(key)
            if hit is not None and time.monotonic() - hit[0] < _REMOTE_TTL_S:
                return hit[1]
    schema: Optional[str] = None
    try:
        from hub.utils.gguf_header_delta import _layout_of_remote, hub_range_fetcher

        fetch = hub_range_fetcher(repo_id, filename, token, revision = revision)
        total = int(size) if size else _REMOTE_CAP
        layout, header = _layout_of_remote(fetch, total, _REMOTE_CAP, _REMOTE_FIRST_RANGE)
        names = [
            (t[0].decode("utf-8", "replace") if isinstance(t[0], bytes) else str(t[0]))
            for t in layout.tensors
        ]
        kvs = {}
        markov_context = _header_bool_kv(header, _MARKOV_CONTEXT_KV)
        if markov_context is not None:
            kvs[_MARKOV_CONTEXT_KV] = markov_context
        schema = dspark_schema_from_tensors(names, kvs)
    except Exception:
        schema = None
    if schema is not None and key is not None:
        with _remote_lock:
            _remote_cache[key] = (time.monotonic(), schema)
    return schema
