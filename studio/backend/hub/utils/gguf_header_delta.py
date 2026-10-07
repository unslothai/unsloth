# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Rebuild a republished GGUF from the cached older copy when only its header changed.

A metadata fix (chat template, architecture label, a new KV) rewrites the GGUF header and leaves every tensor byte
alone, but the file's sha256 changes, so huggingface_hub downloads the whole file again. Here the new header is fetched
with HTTP Range requests, the tensor table is checked against the cached copy's, and the new file is written as the new
header plus the cached copy's tensor data. It is kept only when its sha256 equals the Hub's; anything that cannot be
proven is left to the normal download, and nothing here raises."""

from __future__ import annotations

import glob
import hashlib
import os
import re
import shutil
import struct
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from loggers import get_logger

logger = get_logger(__name__)

DELTA_ENV = "UNSLOTH_GGUF_HEADER_DELTA"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GGUF_MAGIC = b"GGUF"
_FIRST_RANGE = 2 << 20
# LLM headers carry the tokenizer (tens of MB for large vocabularies); past this a full download is the better deal.
_MAX_HEADER_BYTES = 256 << 20
_COPY_CHUNK = 16 << 20
_DISK_MARGIN = 256 << 20
_METADATA_TIMEOUT = 10.0
_RANGE_TIMEOUT = 60.0
# A file found up to date is not asked about again for this long (each check costs one HEAD).
_UP_TO_DATE_TTL = 600.0

_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_STRING, _ARRAY, _UINT32 = 8, 9, 4

FetchRange = Callable[[int, int], bytes]

_up_to_date: dict[tuple, float] = {}
_up_to_date_lock = threading.Lock()


class _NeedMore(Exception):
    pass


@dataclass(frozen = True)
class GgufLayout:
    data_offset: int
    # (name, dims, ggml type, offset relative to the data section), in file order.
    tensors: tuple


@dataclass(frozen = True)
class DeltaResult:
    placed: bool = False
    fetched_bytes: int = 0
    size: int = 0
    reason: str = ""


def delta_enabled() -> bool:
    return (os.environ.get(DELTA_ENV) or "").strip().lower() not in ("0", "off", "false", "no")


def parse_gguf_layout(buf: bytes) -> GgufLayout:
    """Tensor table and data offset of a GGUF v2/v3 header held in ``buf``. Raises ``_NeedMore`` when ``buf`` ends
    inside the header (the data offset itself may lie past it), ``ValueError`` when it is not a GGUF this can read."""
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
                # Nested arrays are legal but no published GGUF writes them; not worth a reader here.
                raise ValueError(f"unsupported array item type {item}")
        else:
            raise ValueError(f"unknown value type {vtype}")

    if take(4) != _GGUF_MAGIC:
        raise ValueError("not a GGUF")
    version = u32()
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version {version}")
    n_tensors = u64()
    n_kv = u64()
    if n_tensors > 10_000_000 or n_kv > 10_000_000:
        raise ValueError("implausible header counts")
    alignment = 32
    for _ in range(n_kv):
        key = string()
        vtype = u32()
        if key == b"general.alignment" and vtype == _UINT32:
            alignment = u32()
        else:
            skip_value(vtype)
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError(f"bad alignment {alignment}")
    tensors = []
    for _ in range(n_tensors):
        name = string()
        n_dims = u32()
        if n_dims > 16:
            raise ValueError("implausible tensor rank")
        dims = tuple(u64() for _ in range(n_dims))
        ggml_type = u32()
        offset = u64()
        tensors.append((name, dims, ggml_type, offset))
    data_offset = -(-pos // alignment) * alignment
    return GgufLayout(data_offset, tuple(tensors))


def _layout_of_local(path: Path, cap: int) -> GgufLayout:
    want = _FIRST_RANGE
    with open(path, "rb") as f:
        while True:
            f.seek(0)
            buf = f.read(want)
            try:
                return parse_gguf_layout(buf)
            except _NeedMore:
                if len(buf) < want or want >= cap:
                    raise ValueError("cached GGUF header unreadable within the cap")
                want = min(want * 2, cap)


def _layout_of_remote(fetch: FetchRange, size: int, cap: int) -> tuple[GgufLayout, bytes]:
    """Layout plus the new file's bytes up to its data offset, fetched in growing ranges."""
    buf = b""
    want = min(_FIRST_RANGE, size)
    while True:
        if want > len(buf):
            part = fetch(len(buf), want - 1)
            if len(part) != want - len(buf):
                raise ValueError("short range response")
            buf += part
        try:
            layout = parse_gguf_layout(buf)
        except _NeedMore:
            if want >= min(cap, size):
                raise ValueError("new GGUF header larger than the fetch cap")
            want = min(want * 2, cap, size)
            continue
        if layout.data_offset > len(buf):
            if layout.data_offset > min(cap, size):
                raise ValueError("new GGUF header larger than the fetch cap")
            part = fetch(len(buf), layout.data_offset - 1)
            if len(part) != layout.data_offset - len(buf):
                raise ValueError("short range response")
            buf += part
        return layout, buf[: layout.data_offset]


def _other_snapshot_copies(repo_dir: Path, commit: str, rel_path: str) -> list[Path]:
    from hub.utils.hf_cache_state import snapshot_selection_key

    try:
        snaps = [p for p in (repo_dir / "snapshots").iterdir() if p.is_dir() and p.name != commit]
    except OSError:
        return []
    out = []
    for snap in sorted(snaps, key = snapshot_selection_key, reverse = True):
        candidate = snap / rel_path
        if os.path.isfile(candidate):
            out.append(candidate)
    return out


def _lock_path(repo_dir: Path, digest: str) -> Path:
    return repo_dir.parent / ".locks" / repo_dir.name / f"{digest}.lock"


def _build(
    old_path: Path,
    old_data_offset: int,
    prefix: bytes,
    tmp: Path,
    size: int,
    digest: str,
    cancel_event: Optional[threading.Event],
) -> bool:
    h = hashlib.sha256(prefix)
    written = len(prefix)
    with open(tmp, "wb") as out, open(old_path, "rb") as src:
        out.write(prefix)
        src.seek(old_data_offset)
        while True:
            if cancel_event is not None and cancel_event.is_set():
                return False
            chunk = src.read(_COPY_CHUNK)
            if not chunk:
                break
            out.write(chunk)
            h.update(chunk)
            written += len(chunk)
            if written > size:
                return False
    return written == size and h.hexdigest() == digest


def rebuild_from_older_snapshot(
    repo_dir: Path,
    commit: str,
    rel_path: str,
    size: int,
    digest: str,
    fetch: FetchRange,
    *,
    protected_blob_hashes: frozenset = frozenset(),
    cancel_event: Optional[threading.Event] = None,
) -> DeltaResult:
    """Place ``rel_path`` at ``commit`` (blob + pointer, in the cache's own layout) from an older snapshot's copy when
    only the header differs. Never raises; ``placed`` False means use the normal download."""
    tmp: Optional[Path] = None
    fetched = 0
    try:
        if not delta_enabled():
            return DeltaResult(reason = "disabled")
        if not rel_path.lower().endswith(".gguf") or not _SHA256.match(digest or "") or size <= 0:
            return DeltaResult(reason = "not a sha256-addressed GGUF")
        if digest in protected_blob_hashes:
            return DeltaResult(reason = "blob protected")
        pointer = repo_dir / "snapshots" / commit / rel_path
        blob = repo_dir / "blobs" / digest
        if os.path.lexists(pointer) or os.path.exists(blob):
            return DeltaResult(reason = "already cached")
        olds = _other_snapshot_copies(repo_dir, commit, rel_path)
        if not olds:
            return DeltaResult(reason = "no older copy")
        cap = min(_MAX_HEADER_BYTES, max(size // 4, _FIRST_RANGE))

        counted = {"n": 0}

        def counted_fetch(start: int, end: int) -> bytes:
            data = fetch(start, end)
            counted["n"] += len(data)
            return data

        layout, prefix = _layout_of_remote(counted_fetch, size, cap)
        fetched = counted["n"]
        data_len = size - layout.data_offset
        old = None
        for candidate in olds:
            try:
                old_size = os.path.getsize(candidate)
                old_layout = _layout_of_local(candidate, cap)
            except (OSError, ValueError):
                continue
            if old_layout.tensors == layout.tensors and old_size - old_layout.data_offset == data_len:
                old = (candidate, old_layout)
                break
        if old is None:
            return DeltaResult(fetched_bytes = fetched, size = size, reason = "tensor data differs")
        old_path, old_layout = old
        symlinked = os.path.islink(old_path)
        stage_dir = (repo_dir / "blobs") if symlinked else pointer.parent
        stage_dir.mkdir(parents = True, exist_ok = True)
        if shutil.disk_usage(stage_dir).free < size + _DISK_MARGIN:
            return DeltaResult(fetched_bytes = fetched, size = size, reason = "not enough disk")
        from filelock import FileLock, Timeout

        lock = _lock_path(repo_dir, digest)
        lock.parent.mkdir(parents = True, exist_ok = True)
        try:
            with FileLock(str(lock), timeout = 0):
                for stale in stage_dir.glob(f".{glob.escape(digest)}.delta-*"):
                    try:
                        stale.unlink()
                    except OSError:
                        pass
                tmp = stage_dir / f".{digest}.delta-{os.getpid()}-{uuid.uuid4().hex[:8]}"
                if not _build(
                    old_path, old_layout.data_offset, prefix, tmp, size, digest, cancel_event
                ):
                    return DeltaResult(fetched_bytes = fetched, size = size, reason = "sha256 mismatch")
                if os.path.lexists(pointer) or os.path.exists(blob):
                    return DeltaResult(fetched_bytes = fetched, size = size, reason = "raced")
                pointer.parent.mkdir(parents = True, exist_ok = True)
                if symlinked:
                    os.replace(tmp, blob)
                    tmp = None
                    from huggingface_hub.file_download import _create_symlink

                    _create_symlink(str(blob), str(pointer), new_blob = False)
                else:
                    # No-symlink cache: huggingface_hub keeps the file under the snapshot and blobs/ empty.
                    os.replace(tmp, pointer)
                    tmp = None
        except Timeout:
            return DeltaResult(fetched_bytes = fetched, size = size, reason = "blob locked")
        logger.info(
            "gguf header delta: rebuilt %s from the cached copy, fetched %.2f MB of %.2f GB",
            rel_path,
            fetched / 1e6,
            size / 1e9,
        )
        return DeltaResult(True, fetched, size, "rebuilt")
    except Exception as exc:  # noqa: BLE001 - an optimisation only: the normal download follows
        logger.info("gguf header delta skipped for %s: %s", rel_path, exc)
        return DeltaResult(fetched_bytes = fetched, size = size, reason = f"error: {exc}")
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def hub_range_fetcher(
    repo_id: str,
    filename: str,
    token,
    *,
    repo_type: str = "model",
    revision: Optional[str] = None,
) -> FetchRange:
    """Range GETs on the file's resolve URL, authenticated like huggingface_hub's own downloads. A server that ignores
    Range (status 200) is refused before its body is read."""
    from huggingface_hub import hf_hub_url
    from huggingface_hub.utils import build_hf_headers, get_session

    url = hf_hub_url(repo_id, filename, repo_type = repo_type, revision = revision)
    base_headers = build_hf_headers(token = token)

    def fetch(start: int, end: int) -> bytes:
        headers = dict(base_headers)
        headers["Range"] = f"bytes={start}-{end}"
        session = get_session()
        if type(session).__module__.split(".")[0] == "httpx":
            with session.stream(
                "GET", url, headers = headers, follow_redirects = True, timeout = _RANGE_TIMEOUT
            ) as resp:
                if resp.status_code != 206:
                    raise ValueError(f"range request answered {resp.status_code}")
                return resp.read()
        resp = session.get(
            url, headers = headers, allow_redirects = True, stream = True, timeout = _RANGE_TIMEOUT
        )
        try:
            if resp.status_code != 206:
                raise ValueError(f"range request answered {resp.status_code}")
            return resp.content
        finally:
            resp.close()

    return fetch


def reuse_for_hub_download(
    repo_id: str,
    filename: str,
    token,
    *,
    repo_type: str = "model",
    revision: Optional[str] = None,
    cache_dir: Optional[str] = None,
    cancel_event: Optional[threading.Event] = None,
    metadata_fn: Optional[Callable] = None,
    fetcher_fn: Optional[Callable] = None,
) -> DeltaResult:
    """Single-file entry (``hf_hub_download`` callers): resolve the file's current commit, size and sha256 with one
    HEAD, then try the rebuild. Asks the Hub only when an older snapshot of this repo holds the same path."""
    try:
        if not delta_enabled() or not str(filename).lower().endswith(".gguf"):
            return DeltaResult(reason = "not applicable")
        from huggingface_hub import constants
        from huggingface_hub.file_download import repo_folder_name

        if constants.HF_HUB_OFFLINE:
            return DeltaResult(reason = "offline")
        root = Path(cache_dir) if cache_dir else Path(constants.HF_HUB_CACHE)
        repo_dir = root / repo_folder_name(repo_id = repo_id, repo_type = repo_type)
        if not _other_snapshot_copies(repo_dir, "", filename):
            return DeltaResult(reason = "no older copy")
        key = (str(repo_dir), filename, revision or "main")
        now = time.monotonic()
        with _up_to_date_lock:
            seen = _up_to_date.get(key)
        if seen is not None and now - seen < _UP_TO_DATE_TTL:
            return DeltaResult(reason = "recently up to date")
        if metadata_fn is None:
            from huggingface_hub import get_hf_file_metadata, hf_hub_url

            def metadata_fn():
                return get_hf_file_metadata(
                    hf_hub_url(repo_id, filename, repo_type = repo_type, revision = revision),
                    token = token,
                    timeout = _METADATA_TIMEOUT,
                )

        meta = metadata_fn()
        commit = getattr(meta, "commit_hash", None)
        digest = (getattr(meta, "etag", None) or "").strip('"').lower()
        size = int(getattr(meta, "size", 0) or 0)
        if not commit or not _SHA256.match(digest) or size <= 0:
            return DeltaResult(reason = "no sha256 for this file")
        if os.path.lexists(repo_dir / "snapshots" / commit / filename) or os.path.exists(
            repo_dir / "blobs" / digest
        ):
            with _up_to_date_lock:
                _up_to_date[key] = now
            return DeltaResult(reason = "already cached")
        fetch = (fetcher_fn or hub_range_fetcher)(
            repo_id, filename, token, repo_type = repo_type, revision = commit
        )
        return rebuild_from_older_snapshot(
            repo_dir, commit, filename, size, digest, fetch, cancel_event = cancel_event
        )
    except Exception as exc:  # noqa: BLE001 - an optimisation only
        logger.info("gguf header delta skipped for %s/%s: %s", repo_id, filename, exc)
        return DeltaResult(reason = f"error: {exc}")


def reset_for_tests() -> None:
    with _up_to_date_lock:
        _up_to_date.clear()
