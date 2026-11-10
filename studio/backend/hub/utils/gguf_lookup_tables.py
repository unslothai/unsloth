# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Bytes of the lookup-only tables (per-layer embeddings / Engram) llama.cpp streams from disk."""

from __future__ import annotations

import bisect
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
# A later split shard carries only the split keys: its header is a few KiB.
_SPLIT_SHARD_FIRST_READ = 64 << 10
MAX_HEADER_BYTES = 64 << 20

_WANTED = {name.encode() for names in LAZY_READ_TENSORS.values() for name in names}


class _NeedMore(Exception):
    pass


def lazy_architecture(arch: Optional[str]) -> bool:
    """Whether ``arch`` flags a lookup table llama.cpp may read lazily."""
    return bool(arch) and arch in LAZY_READ_TENSORS


def parse_header(
    buf: bytes,
    file_size: Optional[int] = None,
    arch_only: bool = False,
) -> tuple[Optional[str], dict[str, int]]:
    """(architecture, {lazy-candidate tensor name: bytes}) from a GGUF v2/v3 header. A quant
    type gguf predates is sized from the next tensor's offset (or ``file_size``), as the
    launch scanner does. ``arch_only`` stops at the architecture key."""
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

    def skip_value(vtype: int, depth: int = 0) -> None:
        if depth > 8:
            raise ValueError("implausible array nesting")
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
                for _ in range(count):
                    skip_value(item, depth + 1)
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
    alignment = 32  # GGUF's default when general.alignment is absent
    for _ in range(n_kv):
        key = string()
        vtype = u32()
        if key == b"general.architecture" and vtype == _STRING:
            arch = string().decode("utf-8", "replace")
            if arch_only:
                return arch, {}
        elif key == b"general.alignment" and vtype == _UINT32:
            alignment = max(1, u32())
        else:
            skip_value(vtype)
    if arch_only:
        return arch, {}
    named: dict[str, int] = {}
    offsets: list[int] = []
    unsized: list[tuple[str, int]] = []
    for _ in range(n_tensors):
        name = string()
        n_dims = u32()
        if n_dims > 16:
            raise ValueError("implausible tensor rank")
        dims = [u64() for _ in range(n_dims)]
        ggml_type = u32()
        offset = u64()
        offsets.append(offset)
        if name not in _WANTED:
            continue
        block = GGML_QUANT_SIZES.get(ggml_type)
        if block is None:
            unsized.append((name.decode("utf-8", "replace"), offset))
            continue
        n = 1
        for d in dims:
            n *= int(d)
        named[name.decode("utf-8", "replace")] = n // block[0] * block[1]
    if unsized:
        data_start = -(-pos // alignment) * alignment
        ordered = sorted(offsets)
        for name, offset in unsized:
            later = ordered[bisect.bisect_right(ordered, offset) :]
            if later:
                named[name] = later[0] - offset
            elif file_size is not None:
                named[name] = max(0, file_size - data_start - offset)
    return arch, named


def _read_growing(
    read: Callable[[int], bytes],
    cap: int = MAX_HEADER_BYTES,
    *,
    first_read: int = _FIRST_READ,
    file_size: Optional[Callable[[], Optional[int]]] = None,
    arch_only: bool = False,
):
    want = min(first_read, cap)
    while True:
        buf = read(want)
        try:
            return parse_header(
                buf, file_size() if file_size is not None else None, arch_only = arch_only
            )
        except _NeedMore:
            if len(buf) < want or want >= cap:
                raise ValueError("GGUF header unreadable within the cap")
            want = min(want * 2, cap)


def header_of_local(path: str) -> tuple[Optional[str], dict[str, int]]:
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size

        def read(n: int) -> bytes:
            f.seek(0)
            return f.read(n)

        return _read_growing(read, file_size = lambda: size)


def shard_names(filename: str) -> list[str]:
    """Every shard of ``filename`` in order (itself alone when it is not a split)."""
    head, sep, base = str(filename).replace("\\", "/").rpartition("/")
    match = _SPLIT_RE.match(base)
    if not match:
        return [filename]
    prefix, _index, total, ext = match.groups()
    return [
        f"{head}{sep}{prefix}-{i:05d}-of-{int(total):05d}{ext}" for i in range(1, int(total) + 1)
    ]


_LOCAL_CACHE: "OrderedDict[tuple, tuple[Optional[str], int]]" = OrderedDict()
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


MAX_SHARDS_READ = 8


def _scan_shards(
    read_header: Callable[[str], tuple],
    names: list[str],
    mode: str = "auto",
    max_shards: Optional[int] = MAX_SHARDS_READ,
    known_arch: Optional[str] = None,
):
    """Returns (architecture, table bytes); stops at the first shard that carries the table."""
    arch: Optional[str] = None
    order = names[:max_shards]
    hinted = lazy_architecture(known_arch) and len(names) > 1
    if hinted:
        order = names[1:max_shards] + names[:1]
    for name in order:
        part_arch, named = read_header(name)
        if hinted and not arch and not part_arch:
            if not named:
                continue
            arch = read_header(names[0], True)[0]
        arch = part_arch or arch
        if not lazy_architecture(arch):
            return arch, 0
        found = lazy_read_bytes(arch, named, mode)
        if found or named:
            return arch, found
    return arch, 0


def local_lookup_info(path: Optional[str], mode: str = "auto") -> tuple[Optional[str], int]:
    """(architecture, table bytes llama.cpp reads lazily under ``mode``) for a local GGUF."""
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
        # Local headers are cheap: every shard, as the launch scanner reads them.
        answer = _scan_shards(header_of_local, names, mode, max_shards = None)
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
    *,
    mode: str = "auto",
):
    """The cached remote answer under lazy ``mode``, keyed on the file's identity (its listed size by default), so a re-upload under the same name is read again."""
    with _lock:
        hit = _REMOTE_CACHE.get((repo_id.lower(), filename, revision or "", identity, mode))
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
    mode: str = "auto",
    known_arch: Optional[str] = None,
) -> tuple[Optional[str], int]:
    """(architecture, lazily read table bytes) from Range reads of a Hub GGUF's header(s)."""
    # Hub repo ids are case-insensitive.
    key = (repo_id.lower(), filename, revision or "", identity, mode)
    hit = remote_cached(repo_id, filename, revision, identity, mode = mode)
    if hit is not None:
        return hit
    if fetcher_fn is None:
        from hub.utils.gguf_header_delta import hub_range_fetcher as fetcher_fn

    names = shard_names(filename)

    def read_header(name: str, arch_only: bool = False):
        fetch = fetcher_fn(repo_id, name, token, revision = revision)
        buf = b""

        def read(n: int) -> bytes:
            nonlocal buf
            if n > len(buf):
                buf += fetch(len(buf), n - 1)
            return buf[:n]

        return _read_growing(
            read,
            first_read = (
                _FIRST_READ if name == names[0] and not arch_only else _SPLIT_SHARD_FIRST_READ
            ),
            file_size = lambda: getattr(fetch, "file_size", None),
            arch_only = arch_only,
        )

    try:
        answer = _scan_shards(read_header, names, mode, known_arch = known_arch)
    except Exception:  # noqa: BLE001 -- unreachable or unreadable: today's behaviour
        return None, 0
    _remember(_REMOTE_CACHE, key, (answer, time.monotonic() + _REMOTE_TTL_S))
    return answer
