# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reads a safetensors checkpoint's format and metadata, never its weights; never opens .pt; never
raises, fails open with None if unknown"""

from __future__ import annotations

import json
import struct
import threading
import time
from typing import Any, Callable, Optional

# Above this many bytes read, the probe gives up (answers None) rather than turn into a download.
MAX_PROBE_BYTES = 64 * 1024 * 1024
TIMEOUT_SECONDS = 20.0

ReadRange = Callable[[int, int], bytes]


class _ProbeBudgetExceeded(Exception):
    pass


# Bounded by bytes AND structure before ``json.loads``; real headers are <= 2.9 MB, so >= 4x headroom.
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
    if not isinstance(fmt, str):
        return None
    # An Unsloth header whose metadata is not an object is malformed, not unreadable: the loader refuses it.
    return {"format": fmt, "metadata": meta, "container": "safetensors"}


def read_checkpoint_header(name: str, read_range: ReadRange, size: int) -> Optional[dict]:
    """``{"format", "metadata", "container", "bytes_read"}`` of a checkpoint named ``name``, or None.

    Only a ``.safetensors`` name is read; any other is None with nothing read. Never raises; reads at
    most ``MAX_PROBE_BYTES``."""
    counted = [0]

    def _counted(start: int, end: int) -> bytes:
        if counted[0] + (end - start + 1) > MAX_PROBE_BYTES:
            raise _ProbeBudgetExceeded("probe budget exceeded")
        data = read_range(start, end)
        counted[0] += len(data)
        return data

    if not str(name).lower().endswith(".safetensors"):
        # A pickle checkpoint is never read here: unknown, without a single byte fetched.
        return None
    try:
        out = _from_safetensors(_counted, size)
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


MISSING = "missing"


def _is_not_found(exc: BaseException) -> bool:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status == 404 or type(exc).__name__ in ("EntryNotFoundError", "RemoteEntryNotFoundError")


def hub_file_revision(
    repo_id: str,
    filename: str,
    *,
    token: Optional[str] = None,
    revision: Optional[str] = None,
    timeout_seconds: float = TIMEOUT_SECONDS,
) -> Optional[tuple]:
    """``(etag, commit)`` the Hub serves for ``filename`` now (one HEAD, no body), ``MISSING`` on a definite
    404, or None if unanswerable."""
    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url
        meta = get_hf_file_metadata(
            hf_hub_url(repo_id, filename, revision = revision),
            token = token,
            timeout = timeout_seconds,
        )
    except Exception as exc:  # noqa: BLE001 - offline or refused: the caller keeps what it has
        return MISSING if _is_not_found(exc) else None
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
        return None, (False if _is_not_found(exc) else None)
    size = getattr(meta, "size", None)
    if not size:
        return None, True
    etag = getattr(meta, "etag", None)
    cached = _header_cache_get(etag, filename, int(size))
    if cached is not None:
        cached["size"] = int(size)
        cached["commit"] = getattr(meta, "commit_hash", None)
        return cached, True
    url = hf_hub_url(repo_id, filename, revision = getattr(meta, "commit_hash", None) or revision)
    try:
        headers = dict(build_hf_headers(token = token))
    except Exception:  # noqa: BLE001
        return None, True
    out = read_checkpoint_header(filename, _hub_range_reader(url, headers, deadline), int(size))
    if out is not None:
        _header_cache_put(etag, filename, int(size), out)
        out["size"] = int(size)
        out["commit"] = getattr(meta, "commit_hash", None)
    return out, True


# "0": every fresh process reads a hosted checkpoint's header over the network again (two ranged GETs, ~1-2 s).
HEADER_CACHE_ENV = "UNSLOTH_DIFFUSION_HEADER_CACHE"
_HEADER_CACHE_NAME = "prequant_header_cache.json"
_HEADER_CACHE_MAX = 64
_HEADER_CACHE_ENTRY_MAX = 512 * 1024
_HEADER_CACHE_KEYS = ("format", "metadata", "container")
_HEADER_CACHE_LOCK = threading.Lock()


def _header_cache_file() -> Optional[Any]:
    import os
    if (os.environ.get(HEADER_CACHE_ENV) or "").strip().lower() in ("0", "off", "false", "no"):
        return None
    try:
        from pathlib import Path

        from utils.paths.storage_roots import cache_root
        return Path(cache_root()) / _HEADER_CACHE_NAME
    except Exception:  # noqa: BLE001 - no resolvable root: read over the network as before
        return None


def _header_cache_key(etag: Any, filename: str, size: int) -> Optional[str]:
    """The file's content digest (the Hub ETag of an LFS / Xet file is its sha256) plus name and size: one answer
    per content, whatever repo or revision serves it. Read only after this request's own HEAD returned it."""
    if not isinstance(etag, str) or not etag.strip('"'):
        return None
    return f"{etag.strip(chr(34))}|{filename.rsplit('/', 1)[-1]}|{int(size)}"


def _header_cache_get(etag: Any, filename: str, size: int) -> Optional[dict]:
    key = _header_cache_key(etag, filename, size)
    path = _header_cache_file() if key else None
    if path is None:
        return None
    try:
        with _HEADER_CACHE_LOCK:
            table = json.loads(path.read_text(encoding = "utf-8")) if path.is_file() else {}
        hit = table.get(key) if isinstance(table, dict) else None
    except Exception:  # noqa: BLE001 - unreadable or corrupt: a miss
        return None
    if not isinstance(hit, dict) or not isinstance(hit.get("format"), str):
        return None
    if not all(k in hit for k in _HEADER_CACHE_KEYS):
        return None
    # nothing read this time: the skip log's byte count stays true
    return {**{k: hit[k] for k in _HEADER_CACHE_KEYS}, "bytes_read": 0}


def _header_cache_put(etag: Any, filename: str, size: int, header: dict) -> None:
    import os

    key = _header_cache_key(etag, filename, size)
    path = _header_cache_file() if key else None
    if path is None:
        return
    try:
        entry = {k: header.get(k) for k in _HEADER_CACHE_KEYS}
        if len(json.dumps(entry, default = str)) > _HEADER_CACHE_ENTRY_MAX:
            return
        with _HEADER_CACHE_LOCK:
            try:
                table = json.loads(path.read_text(encoding = "utf-8")) if path.is_file() else {}
            except Exception:  # noqa: BLE001
                table = {}
            if not isinstance(table, dict):
                table = {}
            table.pop(key, None)
            table = dict(list(table.items())[-(_HEADER_CACHE_MAX - 1) :])
            table[key] = entry
            path.parent.mkdir(parents = True, exist_ok = True)
            tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_text(json.dumps(table, default = str), encoding = "utf-8")
            os.replace(tmp, path)
    except Exception:  # noqa: BLE001 - persistence is best-effort
        pass
