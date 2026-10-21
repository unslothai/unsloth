# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bytes of the lookup-only tables (per-layer embeddings / Engram) llama.cpp streams from disk.

Read off GGUF headers only, local or over HTTP Range, never the tensor data. The size is
what the model listing and the Load Model panel price separately: under llama.cpp's
default ``--lazy-mode auto`` a flagged table over 4 GiB stays mmap'd from disk and pages
in only the rows a token touches, so it is neither VRAM nor resident RAM.

Nothing here raises: an unreadable or unreachable header answers 0, which is the
behaviour before this existed (the whole file priced as weights).
"""

from __future__ import annotations

import os
import re
import struct
import threading
import time
from collections import OrderedDict
from typing import Callable, Optional

from core.inference.lookup_tables import LAZY_READ_TENSORS, lazy_read_bytes

_SPLIT_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})(\.gguf)$", re.IGNORECASE)
_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_STRING, _ARRAY, _UINT32 = 8, 9, 4
_FIRST_READ = 8 << 20
# A tokenizer-heavy header is a few tens of MB at most; past this the answer is 0.
MAX_HEADER_BYTES = 64 << 20

_WANTED = {name.encode() for names in LAZY_READ_TENSORS.values() for name in names}


class _NeedMore(Exception):
    pass


def lazy_architecture(arch: Optional[str]) -> bool:
    """Whether ``arch`` flags a lookup table llama.cpp may read lazily."""
    return bool(arch) and arch in LAZY_READ_TENSORS


def parse_header(buf: bytes) -> tuple[Optional[str], dict[str, int]]:
    """(architecture, {lazy-candidate tensor name: bytes}) from a GGUF v2/v3 header held in
    ``buf``. Raises ``_NeedMore`` when ``buf`` ends inside the header, ``ValueError`` when it
    is not a GGUF this can read. A tensor of a quant type ``gguf`` does not know is skipped."""
    from gguf.constants import GGML_QUANT_SIZES

    pos = 0

    def take(n: int) -> bytes:
        nonlocal pos
        if pos + n > len(buf):
            raise _NeedMore()
        out = buf[pos : pos + n]
        pos += n
        return out

    def u32() -> int:
        return struct.unpack("<I", take(4))[0]

    def u64() -> int:
        return struct.unpack("<Q", take(8))[0]

    def string() -> bytes:
        n = u64()
        if n > (1 << 30):
            raise ValueError("implausible string length")
        return take(n)

    def skip_value(vtype: int) -> None:
        if vtype in _SCALAR_SIZES:
            take(_SCALAR_SIZES[vtype])
        elif vtype == _STRING:
            string()
        elif vtype == _ARRAY:
            item = u32()
            count = u64()
            if item in _SCALAR_SIZES:
                take(_SCALAR_SIZES[item] * count)
            elif item == _STRING:
                for _ in range(count):
                    string()
            else:
                raise ValueError(f"unsupported array item type {item}")
        else:
            raise ValueError(f"unknown value type {vtype}")

    if take(4) != b"GGUF":
        raise ValueError("not a GGUF")
    if u32() not in (2, 3):
        raise ValueError("unsupported GGUF version")
    n_tensors = u64()
    n_kv = u64()
    if n_tensors > 10_000_000 or n_kv > 10_000_000:
        raise ValueError("implausible header counts")
    arch: Optional[str] = None
    for _ in range(n_kv):
        key = string()
        vtype = u32()
        if key == b"general.architecture" and vtype == _STRING:
            arch = string().decode("utf-8", "replace")
        else:
            skip_value(vtype)
    named: dict[str, int] = {}
    for _ in range(n_tensors):
        name = string()
        n_dims = u32()
        if n_dims > 16:
            raise ValueError("implausible tensor rank")
        dims = [u64() for _ in range(n_dims)]
        ggml_type = u32()
        u64()  # offset
        if name not in _WANTED:
            continue
        block = GGML_QUANT_SIZES.get(ggml_type)
        if block is None:
            continue
        n = 1
        for d in dims:
            n *= int(d)
        named[name.decode("utf-8", "replace")] = n // block[0] * block[1]
    return arch, named


def _read_growing(read: Callable[[int], bytes], cap: int = MAX_HEADER_BYTES):
    want = _FIRST_READ
    while True:
        buf = read(want)
        try:
            return parse_header(buf)
        except _NeedMore:
            if len(buf) < want or want >= cap:
                raise ValueError("GGUF header unreadable within the cap")
            want = min(want * 2, cap)


def header_of_local(path: str) -> tuple[Optional[str], dict[str, int]]:
    with open(path, "rb") as f:

        def read(n: int) -> bytes:
            f.seek(0)
            return f.read(n)

        return _read_growing(read)


def shard_names(filename: str) -> list[str]:
    """Every shard of ``filename`` in order (itself alone when it is not a split)."""
    head, sep, base = str(filename).replace("\\", "/").rpartition("/")
    match = _SPLIT_RE.match(base)
    if not match:
        return [filename]
    # The extension as written: a case-sensitive filesystem or Hub has no .gguf for .GGUF.
    prefix, _index, total, ext = match.groups()
    return [
        f"{head}{sep}{prefix}-{i:05d}-of-{int(total):05d}{ext}" for i in range(1, int(total) + 1)
    ]


# Local answers, keyed on every shard's (path, size, mtime_ns).
_LOCAL_CACHE: "OrderedDict[tuple, tuple[Optional[str], int]]" = OrderedDict()
# Remote answers, keyed on (repo, filename, revision, identity), each with an expiry: the
# listed size is not content identity, so a same-size re-upload is read again after it.
_REMOTE_CACHE: "OrderedDict[tuple, tuple[tuple[Optional[str], int], float]]" = OrderedDict()
_REMOTE_TTL_S = 6 * 3600.0
_CACHE_LIMIT = 512
_lock = threading.Lock()


def _remember(cache: OrderedDict, key, value) -> None:
    with _lock:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > _CACHE_LIMIT:
            cache.popitem(last = False)


# Shards read per variant at most. llama.cpp's split keeps file order and the table sits
# near the front, so a table not found by then is left unpriced (0, today's pricing)
# rather than fanning out header reads over a long split.
MAX_SHARDS_READ = 8


def _scan_shards(
    read_header: Callable[[str], tuple],
    names: list[str],
    mode: str = "auto",
):
    """(architecture, lazy table bytes under ``mode``) over ``names``, stopping at the first
    shard that carries the table: llama.cpp's split keeps file order and the table sits near
    the front."""
    arch: Optional[str] = None
    for name in names[:MAX_SHARDS_READ]:
        part_arch, named = read_header(name)
        arch = arch or part_arch
        if not lazy_architecture(arch):
            # Shard 1 carries the metadata; a non-lazy arch has nothing to find further on.
            return arch, 0
        found = lazy_read_bytes(arch, named, mode)
        if found or named:
            return arch, found
    return arch, 0


def local_lookup_info(path: Optional[str], mode: str = "auto") -> tuple[Optional[str], int]:
    """(architecture, table bytes llama.cpp reads lazily under ``mode``: ``on`` drops the
    size threshold) for a local GGUF and its sibling shards. ``(None, 0)`` when unreadable
    or a shard is missing."""
    if not path:
        return None, 0
    try:
        names = shard_names(path)
        key = (mode,) + tuple(
            (p, st.st_size, st.st_mtime_ns) for p in names for st in (os.stat(p),)
        )
    except OSError:
        return None, 0
    with _lock:
        hit = _LOCAL_CACHE.get(key)
    if hit is not None:
        return hit
    try:
        answer = _scan_shards(header_of_local, names, mode)
    except Exception:  # noqa: BLE001 -- unreadable header: price the whole file, as before
        answer = (None, 0)
    _remember(_LOCAL_CACHE, key, answer)
    return answer


def local_lookup_table_bytes(path: Optional[str]) -> int:
    return local_lookup_info(path)[1]


def remote_cached(
    repo_id: str,
    filename: str,
    revision: Optional[str],
    identity = None,
):
    """The cached remote answer, keyed on the file's identity (its listed size by default),
    so a re-upload under the same name is read again."""
    with _lock:
        hit = _REMOTE_CACHE.get((repo_id, filename, revision or "", identity))
    if hit is None or hit[1] <= time.monotonic():
        return None
    return hit[0]


def remote_lookup_info(
    repo_id: str,
    filename: str,
    token = None,
    *,
    revision: Optional[str] = None,
    identity = None,
    fetcher_fn: Optional[Callable] = None,
) -> tuple[Optional[str], int]:
    """(architecture, lazily read table bytes) from Range reads of a Hub GGUF's header(s).
    Blocking; call off the event loop. Failures answer ``(None, 0)`` and are not cached,
    so a transient error is retried on the next listing."""
    key = (repo_id, filename, revision or "", identity)
    hit = remote_cached(repo_id, filename, revision, identity)
    if hit is not None:
        return hit
    if fetcher_fn is None:
        from hub.utils.gguf_header_delta import hub_range_fetcher as fetcher_fn

    def read_header(name: str):
        fetch = fetcher_fn(repo_id, name, token, revision = revision)
        buf = b""

        def read(n: int) -> bytes:
            nonlocal buf
            if n > len(buf):
                buf += fetch(len(buf), n - 1)
            return buf[:n]

        return _read_growing(read)

    try:
        answer = _scan_shards(read_header, shard_names(filename))
    except Exception:  # noqa: BLE001 -- unreachable or unreadable: today's behaviour
        return None, 0
    _remember(_REMOTE_CACHE, key, (answer, time.monotonic() + _REMOTE_TTL_S))
    return answer
