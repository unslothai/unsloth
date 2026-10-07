# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Rebuild a republished image / video GGUF from the cached older copy when only its header changed.

A metadata fix (an architecture label, a new KV) rewrites the GGUF header and leaves every tensor byte alone, but the
file's sha256 changes, so huggingface_hub downloads the whole file again. Here the new header is fetched with a Range
request, its tensor table is checked against the cached copy's, and the new file is written as the new header plus the
cached copy's tensor data, hashed in the same pass. It is kept only when its sha256 equals the Hub's.

Opt-in only: the Images and Video load paths pass ``gguf_header_delta=True`` to ``hf_hub_download_with_xet_fallback``,
and the download manager uses it only for GGUFs Studio classifies as image / video models. Nothing touches the network
unless the file for the target commit is missing AND an older snapshot holds the same path. Anything that cannot be
proven is left to the normal download, and nothing here raises."""

from __future__ import annotations

import glob
import hashlib
import json
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
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_GGUF_MAGIC = b"GGUF"
# The first Range asks for the cached header's size plus this, so a header that only gained a few KVs takes one request.
_HEADER_SLACK = 64 << 10
_FIRST_RANGE = 64 << 10
# Text encoder GGUFs carry a tokenizer (tens of MB); past this a full download is the better deal.
_MAX_HEADER_BYTES = 256 << 20
_COPY_CHUNK = 16 << 20
_DISK_MARGIN = 256 << 20
_METADATA_TIMEOUT = 10.0
_RANGE_TIMEOUT = 60.0
# The catalog's image / video GGUF tasks. "image-diffusion-unsupported" is still an image / video model, one this install
# cannot run yet (a diffusers or sd.cpp too old for the family), so it is rebuilt like the rest.
_MEDIA_TASKS = frozenset({"text-to-image", "text-to-video", "image-diffusion-unsupported"})

_SCALAR_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_STRING, _ARRAY, _UINT32 = 8, 9, 4

FetchRange = Callable[[int, int], bytes]

# Refused rebuilds, keyed by (repo cache dir, path, new sha256): not retried on later loads. Persisted under Studio's
# state dir so a restart does not try again either; in memory only when there is no state dir.
_REFUSED_FILE = "gguf_header_delta_refused.json"
_REFUSED_LIMIT = 512
_refused: Optional[list] = None
_refused_lock = threading.Lock()


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
    # Seconds until the rebuild was decided (or refused), and spent writing + hashing the new file.
    decide_s: float = 0.0
    build_s: float = 0.0


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


def _layout_of_remote(
    fetch: FetchRange, size: int, cap: int, first: int
) -> tuple[GgufLayout, bytes]:
    """Layout plus the new file's bytes up to its data offset. One Range of ``first`` bytes, grown only if the header
    turns out longer."""
    buf = b""
    want = min(max(first, 1), size, cap)
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
    """Copies of ``rel_path`` in snapshots other than ``commit``, newest first. Local only."""
    try:
        snaps = [p for p in (repo_dir / "snapshots").iterdir() if p.is_dir() and p.name != commit]
    except OSError:
        return []
    out = [snap / rel_path for snap in snaps if os.path.isfile(snap / rel_path)]
    if len(out) > 1:
        from hub.utils.hf_cache_state import snapshot_selection_key
        out.sort(key = lambda p: snapshot_selection_key(p.parent), reverse = True)
    return out


def _refused_path() -> Optional[Path]:
    try:
        from hub.utils.state_dir import state_root
        root = state_root(create = True)
    except Exception:  # noqa: BLE001 - no state dir: the memo stays in memory
        return None
    return None if root is None else root / _REFUSED_FILE


def _refused_keys() -> list:
    global _refused
    if _refused is None:
        _refused = []
        path = _refused_path()
        if path is not None:
            try:
                data = json.loads(path.read_text(encoding = "utf-8"))
                if isinstance(data, list):
                    _refused = [k for k in data if isinstance(k, str)][-_REFUSED_LIMIT:]
            except (OSError, ValueError):
                pass
    return _refused


def _refused_key(repo_dir: Path, rel_path: str, digest: str) -> str:
    return "|".join((os.path.normcase(os.path.abspath(str(repo_dir))), rel_path, digest))


def _refuse(repo_dir: Path, rel_path: str, digest: str) -> None:
    key = _refused_key(repo_dir, rel_path, digest)
    with _refused_lock:
        keys = _refused_keys()
        if key in keys:
            return
        keys.append(key)
        del keys[:-_REFUSED_LIMIT]
        path = _refused_path()
        if path is None:
            return
        tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
        try:
            tmp.write_text(json.dumps(keys), encoding = "utf-8")
            os.replace(tmp, path)
        except OSError as exc:
            logger.debug("could not persist the gguf header delta memo: %s", exc)
            try:
                tmp.unlink()
            except OSError:
                pass


def _was_refused(repo_dir: Path, rel_path: str, digest: str) -> bool:
    with _refused_lock:
        return _refused_key(repo_dir, rel_path, digest) in _refused_keys()


def _lock_path(repo_dir: Path, digest: str) -> Path:
    return repo_dir.parent / ".locks" / repo_dir.name / f"{digest}.lock"


def _is_media_gguf(path: Path, repo_id: str, rel_path: str) -> bool:
    """Whether Studio lists this GGUF for the Images or Video page (header + name, as the catalog does)."""
    try:
        from hub.services.models.catalog_classification import _gguf_file_task
        return _gguf_file_task(path, (repo_id, rel_path)) in _MEDIA_TASKS
    except Exception:  # noqa: BLE001 - unclassifiable is not a media model
        return False


def _build(
    old_path: Path,
    old_data_offset: int,
    prefix: bytes,
    tmp: Path,
    size: int,
    digest: str,
    cancel_event: Optional[threading.Event],
) -> bool:
    """Write prefix + the old tensor data to ``tmp`` and check the sha256 in the same pass."""
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
    media_gate: Optional[Callable[[Path], bool]] = None,
) -> DeltaResult:
    """Place ``rel_path`` at ``commit`` (blob + pointer, in the cache's own layout) from an older snapshot's copy when
    only the header differs. ``media_gate`` (old copy -> bool) restricts it to some files. Never raises; ``placed``
    False means use the normal download."""
    t0 = time.perf_counter()
    tmp: Optional[Path] = None
    fetched = 0
    decided = 0.0

    def done(
        reason: str,
        placed: bool = False,
        build_s: float = 0.0,
    ) -> DeltaResult:
        return DeltaResult(
            placed,
            fetched,
            size,
            reason,
            decided or time.perf_counter() - t0,
            build_s,
        )

    try:
        if not delta_enabled():
            return done("disabled")
        if not rel_path.lower().endswith(".gguf") or not _SHA256.match(digest or "") or size <= 0:
            return done("not a sha256-addressed GGUF")
        if digest in protected_blob_hashes:
            return done("blob protected")
        pointer = repo_dir / "snapshots" / commit / rel_path
        blob = repo_dir / "blobs" / digest
        if os.path.lexists(pointer) or os.path.exists(blob):
            return done("already cached")
        olds = _other_snapshot_copies(repo_dir, commit, rel_path)
        if not olds:
            return done("no older copy")
        if _was_refused(repo_dir, rel_path, digest):
            return done("refused before")
        if media_gate is not None and not media_gate(olds[0]):
            return done("not an image / video GGUF")
        cap = min(_MAX_HEADER_BYTES, max(size // 4, _FIRST_RANGE))
        local = []
        for candidate in olds:
            try:
                local.append(
                    (candidate, os.path.getsize(candidate), _layout_of_local(candidate, cap))
                )
            except (OSError, ValueError):
                continue
        if not local:
            _refuse(repo_dir, rel_path, digest)
            return done("cached copy unreadable")

        def counted_fetch(start: int, end: int) -> bytes:
            nonlocal fetched
            data = fetch(start, end)
            fetched += len(data)
            return data

        layout, prefix = _layout_of_remote(
            counted_fetch, size, cap, local[0][2].data_offset + _HEADER_SLACK
        )
        data_len = size - layout.data_offset
        old = next(
            (
                (path, old_layout)
                for path, old_size, old_layout in local
                if old_size - old_layout.data_offset == data_len
                and old_layout.tensors == layout.tensors
            ),
            None,
        )
        if old is None:
            _refuse(repo_dir, rel_path, digest)
            return done("tensor table differs")
        decided = time.perf_counter() - t0
        old_path, old_layout = old
        symlinked = os.path.islink(old_path)
        stage_dir = (repo_dir / "blobs") if symlinked else pointer.parent
        stage_dir.mkdir(parents = True, exist_ok = True)
        if shutil.disk_usage(stage_dir).free < size + _DISK_MARGIN:
            return done("not enough disk")
        from filelock import FileLock, Timeout

        lock = _lock_path(repo_dir, digest)
        lock.parent.mkdir(parents = True, exist_ok = True)
        t_build = time.perf_counter()
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
                    if cancel_event is None or not cancel_event.is_set():
                        _refuse(repo_dir, rel_path, digest)
                    return done("sha256 mismatch", build_s = time.perf_counter() - t_build)
                if os.path.lexists(pointer) or os.path.exists(blob):
                    return done("raced")
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
            return done("blob locked")
        build_s = time.perf_counter() - t_build
        logger.info(
            "gguf header delta: rebuilt %s from the cached copy, fetched %.2f MB of %.2f GB "
            "(decided in %.2f s, written in %.1f s)",
            rel_path,
            fetched / 1e6,
            size / 1e9,
            decided,
            build_s,
        )
        return done("rebuilt", True, build_s)
    except Exception as exc:  # noqa: BLE001 - an optimisation only: the normal download follows
        logger.info("gguf header delta skipped for %s: %s", rel_path, exc)
        return done(f"error: {exc}")
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
        # httpx (huggingface_hub 1.x) and httpx2 (2.x) share this API.
        with get_session().stream(
            "GET", url, headers = headers, follow_redirects = True, timeout = _RANGE_TIMEOUT
        ) as resp:
            if resp.status_code != 206:
                raise ValueError(f"range request answered {resp.status_code}")
            return resp.read()

    return fetch


def prepare_media_gguf(
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
    """Single-file entry for the Images / Video load paths, run before ``hf_hub_download``.

    Local checks first, no network: the file for the locally known target commit (``refs/<revision>``, or the commit
    itself) must be missing and an older snapshot must hold the same path. That is the only case where a download
    follows anyway. Then one HEAD for the current commit, size and sha256, and the rebuild. On success ``refs/<revision>``
    names the new commit, so the caller's cache probe finds the rebuilt file."""
    t0 = time.perf_counter()
    try:
        if not delta_enabled() or not str(filename).lower().endswith(".gguf"):
            return DeltaResult(reason = "not applicable", decide_s = time.perf_counter() - t0)
        from huggingface_hub import constants
        from huggingface_hub.file_download import repo_folder_name

        root = Path(cache_dir) if cache_dir else Path(constants.HF_HUB_CACHE)
        repo_dir = root / repo_folder_name(repo_id = repo_id, repo_type = repo_type)
        ref = revision or "main"
        if _COMMIT.match(ref):
            local_commit = ref
        else:
            try:
                local_commit = (repo_dir / "refs" / ref).read_text(encoding = "utf-8").strip()
            except OSError:
                local_commit = ""
        if local_commit and os.path.lexists(repo_dir / "snapshots" / local_commit / filename):
            return DeltaResult(reason = "already cached", decide_s = time.perf_counter() - t0)
        if not _other_snapshot_copies(repo_dir, local_commit, filename):
            return DeltaResult(reason = "no older copy", decide_s = time.perf_counter() - t0)
        if constants.HF_HUB_OFFLINE:
            return DeltaResult(reason = "offline", decide_s = time.perf_counter() - t0)
        if metadata_fn is None:
            from huggingface_hub import get_hf_file_metadata, hf_hub_url
            def metadata_fn():
                return get_hf_file_metadata(
                    hf_hub_url(repo_id, filename, repo_type = repo_type, revision = revision),
                    token = token,
                    timeout = _METADATA_TIMEOUT,
                )

        meta = metadata_fn()
        commit = getattr(meta, "commit_hash", None) or ""
        digest = (getattr(meta, "etag", None) or "").strip('"').lower()
        size = int(getattr(meta, "size", 0) or 0)
        if not _COMMIT.match(commit) or not _SHA256.match(digest) or size <= 0:
            return DeltaResult(reason = "no sha256 for this file", decide_s = time.perf_counter() - t0)
        fetch = (fetcher_fn or hub_range_fetcher)(
            repo_id, filename, token, repo_type = repo_type, revision = commit
        )
        result = rebuild_from_older_snapshot(
            repo_dir, commit, filename, size, digest, fetch, cancel_event = cancel_event
        )
        if result.placed and ref != commit:
            from huggingface_hub.file_download import _cache_commit_hash_for_specific_revision
            _cache_commit_hash_for_specific_revision(str(repo_dir), ref, commit)
        return result
    except Exception as exc:  # noqa: BLE001 - an optimisation only
        logger.info("gguf header delta skipped for %s/%s: %s", repo_id, filename, exc)
        return DeltaResult(reason = f"error: {exc}", decide_s = time.perf_counter() - t0)


def reset_for_tests() -> None:
    global _refused
    with _refused_lock:
        _refused = None
