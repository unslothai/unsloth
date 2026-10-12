# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read a pre-quant checkpoint's ``format`` + ``metadata`` without reading its weights.

Safe: ``data.pkl`` is unpickled with EVERY global replaced by an inert stand-in and persistent ids
answered with None, so nothing in the file is constructed or executed. Readers fail open (None).
"""

from __future__ import annotations

import io
import json
import pickle
import pickletools
import struct
import threading
import time
import zipfile
from typing import Any, Callable, Optional

# Above this many bytes read, the probe gives up (answers None) rather than turn into a download.
MAX_PROBE_BYTES = 64 * 1024 * 1024
_BLOCK = 1024 * 1024
TIMEOUT_SECONDS = 20.0

ReadRange = Callable[[int, int], bytes]


class _ProbeBudgetExceeded(Exception):
    pass


class _RangeFile(io.RawIOBase):
    """A seekable read-only file over ``read_range``, block-cached and byte-budgeted."""

    def __init__(
        self,
        read_range: ReadRange,
        size: int,
        budget: int = MAX_PROBE_BYTES,
    ):
        super().__init__()
        self._read_range = read_range
        self._size = int(size)
        self._pos = 0
        self._blocks: dict = {}
        self._budget = budget
        self.fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(
        self,
        offset: int,
        whence: int = 0,
    ) -> int:
        if whence == 0:
            pos = offset
        elif whence == 1:
            pos = self._pos + offset
        else:
            pos = self._size + offset
        self._pos = max(0, pos)
        return self._pos

    def _block(self, index: int) -> bytes:
        if index not in self._blocks:
            start = index * _BLOCK
            end = min(self._size, start + _BLOCK) - 1
            if self.fetched + (end - start + 1) > self._budget:
                raise _ProbeBudgetExceeded(f"probe would read more than {self._budget} bytes")
            data = self._read_range(start, end)
            if len(data) != end - start + 1:
                raise OSError(f"short range read {len(data)} != {end - start + 1}")
            self.fetched += len(data)
            self._blocks[index] = data
        return self._blocks[index]

    def readinto(self, buffer: Any) -> int:
        want = min(len(buffer), max(0, self._size - self._pos))
        out = 0
        while out < want:
            index, offset = divmod(self._pos, _BLOCK)
            chunk = self._block(index)[offset : offset + (want - out)]
            buffer[out : out + len(chunk)] = chunk
            out += len(chunk)
            self._pos += len(chunk)
        return out


class _Inert:
    """Stand-in for every global ``data.pkl`` names: accepts any construction, holds nothing."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> "_Inert":
        return _Inert()


class _AttrDict(dict):
    """``OrderedDict`` stand-in: torch pickles ``state_dict._metadata`` onto it as an attribute."""


_PLAIN_GLOBALS = {
    ("collections", "OrderedDict"): _AttrDict,
    ("builtins", "set"): set,
    ("builtins", "frozenset"): frozenset,
    ("builtins", "list"): list,
    ("builtins", "dict"): dict,
    ("builtins", "tuple"): tuple,
    # ``torch.__version__`` is a str subclass, recorded as ``metadata["torch_version"]``.
    ("torch.torch_version", "TorchVersion"): str,
}


# Opcodes ``data.pkl`` may hold before the probe answers "unknown". A real torch.save checkpoint spends
# about 0.3 opcodes per byte on its state_dict skeleton: the hosted Z-Image-Turbo FP8 and INT8 files are
# 40k and 60k opcodes. 4M is ~65x the larger, yet bounds what the unpickler can build (each opcode makes
# at most one object) to a few hundred MB even for a hostile file, where the 64 MiB byte cap alone admits
# tens of millions of objects before ``_plain``'s node cap ever runs.
MAX_PICKLE_OPS = 4_000_000


# Memo indices a pickle may use beyond the writes it has made. A pickler numbers its memo sequentially
# (index = writes so far), so a real pickle never needs any slack; this is only tolerance.
_MEMO_SLACK = 1024
_MEMO_PUTS = frozenset(("PUT", "BINPUT", "LONG_BINPUT"))
_MEMO_GETS = frozenset(("GET", "BINGET", "LONG_BINGET"))


# The only opcodes the inert unpickle accepts: what torch.save writes for a metadata+state_dict checkpoint
# (the cached hosted Z-Image-Turbo FP8/INT8 headers use 30 of these), plus the plain data opcodes any
# protocol could use for the same values. Anything else is "unknown". This shuts out EXT1/EXT2/EXT4
# above all: they fetch a callable from ``copyreg``'s extension cache WITHOUT ``find_class``, so a REDUCE
# after one would call a real function. Also INST/OBJ and the out-of-band buffer opcodes.
_ALLOWED_OPS = frozenset(
    (
        "PROTO", "FRAME", "STOP", "MARK", "POP", "POP_MARK", "DUP",
        "NONE", "NEWTRUE", "NEWFALSE",
        "INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4", "FLOAT", "BINFLOAT",
        "STRING", "BINSTRING", "SHORT_BINSTRING", "UNICODE", "BINUNICODE", "SHORT_BINUNICODE",
        "BINUNICODE8", "BINBYTES", "SHORT_BINBYTES", "BINBYTES8",
        "EMPTY_TUPLE", "TUPLE", "TUPLE1", "TUPLE2", "TUPLE3",
        "EMPTY_LIST", "LIST", "APPEND", "APPENDS",
        "EMPTY_DICT", "DICT", "SETITEM", "SETITEMS",
        "EMPTY_SET", "ADDITEMS", "FROZENSET",
        "PUT", "BINPUT", "LONG_BINPUT", "MEMOIZE", "GET", "BINGET", "LONG_BINGET",
        "GLOBAL", "STACK_GLOBAL", "REDUCE", "BUILD", "NEWOBJ", "NEWOBJ_EX",
        "PERSID", "BINPERSID",
    )
)  # fmt: skip


def _pickle_ops_within(raw: bytes, limit: Optional[int] = None) -> bool:
    """Whether ``raw`` holds at most ``limit`` (default ``MAX_PICKLE_OPS``) opcodes and only memo indices
    a real pickler would write. Only walks the bytes (nothing is built), and stops at the first breach,
    so a hostile pickle costs a bounded scan rather than an unbounded unpickle.

    The memo check matters because ONE opcode can be enough: the C unpickler grows its memo table to
    the index a ``LONG_BINPUT`` names (2**31 is tens of GB) before any opcode budget could apply. Any
    opcode outside ``_ALLOWED_OPS`` fails the scan too."""
    limit = MAX_PICKLE_OPS if limit is None else limit
    count = 0
    writes = 0
    for opcode, arg, _ in pickletools.genops(raw):
        count += 1
        if count > limit:
            return False
        name = opcode.name
        if name not in _ALLOWED_OPS:
            return False
        if name == "MEMOIZE":
            writes += 1
        elif name in _MEMO_PUTS or name in _MEMO_GETS:
            try:
                index = int(arg)
            except (TypeError, ValueError):
                return False
            if index < 0 or index > writes + _MEMO_SLACK:
                return False
            if name in _MEMO_PUTS:
                writes += 1
    return True


class _InertUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        plain = _PLAIN_GLOBALS.get((module, name))
        if plain is not None:
            return plain
        # ONE shared stand-in: a fresh class per GLOBAL opcode let a hostile pickle mint millions of types.
        return _Inert

    def persistent_load(self, pid: Any) -> Any:
        return None


_MAX_PLAIN_NODES = 1_000_000


def _plain(value: Any, budget: Optional[list] = None) -> Any:
    """``value`` as JSON-like data; a stand-in anywhere inside becomes None (read as "absent").

    Pickle memo references can share one container many times over, so the walk is capped by node count."""
    if budget is None:
        budget = [_MAX_PLAIN_NODES]
    budget[0] -= 1
    if budget[0] < 0:
        raise _ProbeBudgetExceeded("metadata too large")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(v, budget) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v, budget) for k, v in value.items() if isinstance(k, str)}
    return None


# ``zipfile.ZipFile`` builds a ZipInfo per central-directory entry before anything can be filtered, so the
# directory is bounded from its end record first. A torch archive holds one entry per storage plus a few:
# the cached hosted Z-Image-Turbo FP8/INT8 files have 766 entries in a ~0.1 MB directory, and the largest
# hosted denoisers a few thousand. 50k entries / 8 MiB is >= 10x that; over either is "unknown".
MAX_ZIP_ENTRIES = 50_000
MAX_ZIP_DIRECTORY_BYTES = 8 * 1024 * 1024
_EOCD = b"PK\x05\x06"
_ZIP64_LOCATOR = b"PK\x06\x07"
_ZIP64_EOCD = b"PK\x06\x06"


def _zip_directory_within(handle: Any) -> bool:
    """Whether the archive's central directory (from its end record, ZIP64 included) is small enough to
    hand to ``zipfile``. Reads only the tail and, for ZIP64, one 56-byte record. Unparseable is False."""
    handle.seek(0, 2)
    size = handle.tell()
    tail_len = min(size, 22 + 65535)
    handle.seek(size - tail_len)
    tail = handle.read(tail_len)
    at = tail.rfind(_EOCD)
    if at < 0 or len(tail) - at < 22:
        return False
    entries, directory = struct.unpack("<HI", tail[at + 10 : at + 16])
    locator = at - 20
    if locator >= 0 and tail[locator : locator + 4] == _ZIP64_LOCATOR:
        (offset,) = struct.unpack("<Q", tail[locator + 8 : locator + 16])
        if offset >= size:
            return False
        handle.seek(offset)
        record = handle.read(56)
        if len(record) < 56 or record[:4] != _ZIP64_EOCD:
            return False
        entries64, directory64 = struct.unpack("<QQ", record[32:48])
        entries, directory = max(entries, entries64), max(directory, directory64)
    handle.seek(0)
    return entries <= MAX_ZIP_ENTRIES and directory <= MAX_ZIP_DIRECTORY_BYTES


def _from_pickle_zip(handle: Any) -> Optional[dict]:
    if not _zip_directory_within(handle):
        return None
    with zipfile.ZipFile(handle) as archive:
        names = [n for n in archive.namelist() if n == "data.pkl" or n.endswith("/data.pkl")]
        if len(names) != 1:
            return None
        info = archive.getinfo(names[0])
        if info.file_size > MAX_PROBE_BYTES:
            return None
        raw = archive.read(names[0])
    # Over budget is "unknown", never a refusal: the caller falls back to today's behaviour.
    if not _pickle_ops_within(raw):
        return None
    ckpt = _InertUnpickler(io.BytesIO(raw)).load()
    if not isinstance(ckpt, dict):
        return None
    fmt, meta = ckpt.get("format"), ckpt.get("metadata")
    if not isinstance(fmt, str) or not isinstance(meta, dict):
        return None
    return {"format": str(fmt), "metadata": _plain(meta), "container": "pt"}


# A safetensors header is parsed whole, so it is bounded by bytes AND by structure before ``json.loads``
# builds anything: 64 MiB of ``[0,0,...]`` is tens of millions of objects. Real hosted pre-quant headers
# are 0.2 to 2.9 MB with ~0.09 structural bytes per byte (cached Qwen-Image-2.1 INT8 / FP8: 0.23 to 0.33
# MB, 20k to 23k); 16 MiB and 1M structural bytes leave >= 4x on the largest, and over either is "unknown".
MAX_SAFETENSORS_HEADER_BYTES = 16 * 1024 * 1024
MAX_JSON_STRUCTURE = 1_000_000


def _json_within(raw: Any) -> bool:
    """Whether ``raw`` (bytes or str) is small enough in bytes and structure to hand to ``json.loads``.
    Counted on the raw text, so a nested JSON string inside it is counted too, before either parse."""
    if isinstance(raw, str):
        raw = raw.encode("utf-8", "surrogatepass")
    if len(raw) > MAX_SAFETENSORS_HEADER_BYTES:
        return False
    total = 0
    for mark in (b"{", b"[", b",", b":"):
        total += raw.count(mark)
        if total > MAX_JSON_STRUCTURE:
            return False
    return True


def _from_safetensors(read_range: ReadRange, size: int) -> Optional[dict]:
    if size < 8:
        return None
    (length,) = struct.unpack("<Q", read_range(0, 7))
    if length <= 0 or length > MAX_SAFETENSORS_HEADER_BYTES or 8 + length > size:
        return None
    raw = read_range(8, 8 + length - 1)
    if not _json_within(raw):
        return None
    header = json.loads(raw.decode("utf-8"))
    extra = header.get("__metadata__") if isinstance(header, dict) else None
    if not isinstance(extra, dict):
        return None
    from .prequant_safetensors import UNSLOTH_FORMAT_KEY, UNSLOTH_METADATA_KEY

    fmt = extra.get(UNSLOTH_FORMAT_KEY)
    inner = extra.get(UNSLOTH_METADATA_KEY) or "null"
    if not isinstance(inner, str) or not _json_within(inner):
        return None
    meta = json.loads(inner)
    if not isinstance(fmt, str) or not isinstance(meta, dict):
        return None
    return {"format": fmt, "metadata": meta, "container": "safetensors"}


def read_checkpoint_header(name: str, read_range: ReadRange, size: int) -> Optional[dict]:
    """``{"format", "metadata", "container", "bytes_read"}`` of a checkpoint named ``name``, or None.

    Never raises; reads at most ``MAX_PROBE_BYTES``."""
    counted = [0]

    def _counted(start: int, end: int) -> bytes:
        if counted[0] + (end - start + 1) > MAX_PROBE_BYTES:
            raise _ProbeBudgetExceeded("probe budget exceeded")
        data = read_range(start, end)
        counted[0] += len(data)
        return data

    try:
        if str(name).lower().endswith(".safetensors"):
            out = _from_safetensors(_counted, size)
        else:
            raw = _RangeFile(_counted, size)
            out = _from_pickle_zip(io.BufferedReader(raw, buffer_size = _BLOCK))
    except Exception:  # noqa: BLE001 - unreadable is "unknown", never "incompatible"
        return None
    if out is not None:
        out["bytes_read"] = counted[0]
    return out


def read_local_checkpoint_header(path: str) -> Optional[dict]:
    import os
    try:
        size = os.path.getsize(path)
        handle = open(path, "rb")
    except (OSError, ValueError):
        return None
    try:

        def _read(start: int, end: int) -> bytes:
            handle.seek(start)
            return handle.read(end - start + 1)

        return read_checkpoint_header(path, _read, size)
    finally:
        handle.close()


def _hub_range_reader(url: str, headers: dict, deadline: float) -> ReadRange:
    """Ranged GETs against ``url``; anything but a 206 of exactly the asked length raises."""
    from huggingface_hub.utils import get_session

    from .diffusion_compat import _interrupt_read, _iter_body, _ranged_stream

    def _read(start: int, end: int) -> bytes:
        want = end - start + 1
        buffer = bytearray()
        holder: list = [None]
        failure: list = []

        def _fetch() -> None:
            try:
                ranged = dict(headers)
                ranged["Range"] = f"bytes={start}-{end}"
                with _ranged_stream(get_session(), url, ranged) as response:
                    holder[0] = response
                    # A server ignoring Range answers 200 with the WHOLE checkpoint: never drain it.
                    if response.status_code != 206:
                        failure.append(f"HTTP {response.status_code}")
                        return
                    for chunk in _iter_body(response, 65536):
                        buffer.extend(chunk)
                        if len(buffer) >= want or time.monotonic() > deadline:
                            break
            except Exception as exc:  # noqa: BLE001
                failure.append(repr(exc))

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("header probe deadline")
        worker = threading.Thread(target = _fetch, name = "prequant-header-read", daemon = True)
        worker.start()
        worker.join(remaining)
        if worker.is_alive():
            _interrupt_read(holder[0])
            raise TimeoutError("header probe deadline")
        if failure or len(buffer) < want:
            raise OSError(failure[0] if failure else "short read")
        return bytes(buffer[:want])

    return _read


def hub_file_revision(
    repo_id: str,
    filename: str,
    *,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    timeout_seconds: float = TIMEOUT_SECONDS,
) -> Optional[tuple]:
    """``(etag, commit)`` the Hub serves for ``filename`` now (one HEAD, no body), or None if unanswerable."""
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        meta = get_hf_file_metadata(
            hf_hub_url(repo_id, filename, revision = revision),
            token = token,
            timeout = timeout_seconds,
        )
    except Exception:  # noqa: BLE001 - offline, refused or missing: the caller keeps what it has
        return None
    return getattr(meta, "etag", None), getattr(meta, "commit_hash", None)


def read_hub_checkpoint_header(
    repo_id: str,
    filename: str,
    *,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    timeout_seconds: float = TIMEOUT_SECONDS,
) -> tuple:
    """``(header or None, exists)`` via HTTP range requests; ``exists`` is False only on a definite 404."""
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        from huggingface_hub.utils import build_hf_headers
    except Exception:  # noqa: BLE001
        return None, None
    deadline = time.monotonic() + timeout_seconds
    try:
        meta = get_hf_file_metadata(
            hf_hub_url(repo_id, filename, revision = revision),
            token = token,
            timeout = timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 404 or type(exc).__name__ in (
            "EntryNotFoundError",
            "RemoteEntryNotFoundError",
        ):
            return None, False
        return None, None
    size = getattr(meta, "size", None)
    if not size:
        return None, True
    url = hf_hub_url(repo_id, filename, revision = getattr(meta, "commit_hash", None) or revision)
    try:
        headers = dict(build_hf_headers(token = token))
    except Exception:  # noqa: BLE001
        return None, True
    out = read_checkpoint_header(filename, _hub_range_reader(url, headers, deadline), int(size))
    if out is not None:
        out["size"] = int(size)
        out["commit"] = getattr(meta, "commit_hash", None)
    return out, True
