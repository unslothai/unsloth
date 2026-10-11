# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Read a pre-quant checkpoint's ``format`` + ``metadata`` without reading its weights.

The loader's compatibility checks (scheme, base, exclusion set, filters, rotation) only look at the
metadata, but a hosted checkpoint used to be downloaded in full (tens of GB) before they ran, so a
stale artifact cost its whole download and THEN the dense fallback's. Both containers keep the
metadata in a small region:

- ``.safetensors``: the JSON header at the front (``unsloth_format`` / ``unsloth_metadata``).
- ``.pt`` (``torch.save``): a zip whose ``data.pkl`` member holds the whole dict structure, while
  every tensor's bytes live in separate members. The zip central directory sits at the end.

Both are read through ``read_range(start, end)`` (inclusive), so the same code serves a local file
and a Hub file over HTTP range requests. ``data.pkl`` is unpickled with EVERY global replaced by an
inert stand-in and persistent ids (tensor storages) answered with None: nothing in the file is ever
constructed or executed, and only the plain ``format`` / ``metadata`` values are kept.

Every reader fails open (returns None): an unreadable header is "unknown", never "incompatible".
"""

from __future__ import annotations

import io
import json
import pickle
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

    def __init__(self, read_range: ReadRange, size: int, budget: int = MAX_PROBE_BYTES):
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

    def seek(self, offset: int, whence: int = 0) -> int:
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


class _InertUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        plain = _PLAIN_GLOBALS.get((module, name))
        if plain is not None:
            return plain
        return type(f"_Inert_{name}", (_Inert,), {})

    def persistent_load(self, pid: Any) -> Any:
        return None


def _plain(value: Any) -> Any:
    """``value`` as JSON-like data; a stand-in anywhere inside becomes None (read as "absent")."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items() if isinstance(k, str)}
    return None


def _from_pickle_zip(handle: Any) -> Optional[dict]:
    with zipfile.ZipFile(handle) as archive:
        names = [n for n in archive.namelist() if n == "data.pkl" or n.endswith("/data.pkl")]
        if len(names) != 1:
            return None
        info = archive.getinfo(names[0])
        if info.file_size > MAX_PROBE_BYTES:
            return None
        raw = archive.read(names[0])
    ckpt = _InertUnpickler(io.BytesIO(raw)).load()
    if not isinstance(ckpt, dict):
        return None
    fmt, meta = ckpt.get("format"), ckpt.get("metadata")
    if not isinstance(fmt, str) or not isinstance(meta, dict):
        return None
    return {"format": str(fmt), "metadata": _plain(meta), "container": "pt"}


def _from_safetensors(read_range: ReadRange, size: int) -> Optional[dict]:
    if size < 8:
        return None
    (length,) = struct.unpack("<Q", read_range(0, 7))
    if length <= 0 or length > MAX_PROBE_BYTES or 8 + length > size:
        return None
    header = json.loads(read_range(8, 8 + length - 1).decode("utf-8"))
    extra = header.get("__metadata__") if isinstance(header, dict) else None
    if not isinstance(extra, dict):
        return None
    from .prequant_safetensors import UNSLOTH_FORMAT_KEY, UNSLOTH_METADATA_KEY

    fmt = extra.get(UNSLOTH_FORMAT_KEY)
    meta = json.loads(extra.get(UNSLOTH_METADATA_KEY) or "null")
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
    """``read_checkpoint_header`` for a file on disk (a cached Hub file or a local override)."""
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


def read_hub_checkpoint_header(
    repo_id: str,
    filename: str,
    *,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    timeout_seconds: float = TIMEOUT_SECONDS,
) -> tuple:
    """``(header or None, exists)`` for a Hub file, read by HTTP range requests (no download).

    ``exists`` is False only on a definite 404 for this name (the resolver's cue to try the next
    name); True when the file is there; None when the Hub could not be asked."""
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
        if status == 404 or type(exc).__name__ in ("EntryNotFoundError", "RemoteEntryNotFoundError"):
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
