# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Verify a cached ``.pt`` / ``.bin`` is byte-identical to the Hub's copy (LFS sha256) so a load can reuse it.

Digests are memoised per (real path, size, mtime) in memory and on disk. Any unanswerable case (offline, no LFS
record, error) is "not verified", so the caller keeps safetensors first. Never raises."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

ENV_DISABLE = "UNSLOTH_REUSE_CACHED_PICKLE_WEIGHTS"
_CHUNK = 8 * 1024 * 1024
# Hub misses retry after _MISS_TTL_S, branch answers after _MUTABLE_TTL_S; commit-id answers never expire.
_MISS_TTL_S = 60.0
_MUTABLE_TTL_S = 300.0
_FILE_NAME = "weight_sha256.json"
_MAX_ENTRIES = 512

_lock = threading.Lock()
_hub: dict = {}
_local: dict = {}
_persisted: Optional[dict] = None


def reuse_enabled() -> bool:
    return (os.environ.get(ENV_DISABLE) or "").strip().lower() not in ("0", "false", "no", "off")


def _offline() -> bool:
    try:
        from huggingface_hub import constants
        return bool(constants.HF_HUB_OFFLINE)
    except Exception:  # noqa: BLE001 - no hub library: nothing to compare against
        return True


def hub_sha256s(
    repo_id: Optional[str],
    filenames: Iterable[str],
    *,
    revision: Optional[str] = None,
    token: Optional[str] = None,
) -> dict:
    """``{filename: sha256}`` for the LFS files of ``repo_id`` among ``filenames``, one Hub call for the uncached
    ones. A file the Hub stores without an LFS record, or any failure, is simply absent from the answer."""
    names = [str(n) for n in dict.fromkeys(filenames) if n]
    if not repo_id or not names or _offline():
        return {}
    now = time.monotonic()
    pinned = (
        bool(revision)
        and len(str(revision)) == 40
        and all(c in "0123456789abcdef" for c in str(revision))
    )
    out: dict = {}
    wanted: list = []
    with _lock:
        for name in names:
            hit = _hub.get((repo_id, revision, name))
            if (
                hit is None
                or (hit[0] is None and now - hit[1] > _MISS_TTL_S)
                or (not pinned and now - hit[1] > _MUTABLE_TTL_S)
            ):
                wanted.append(name)
            elif hit[0]:
                out[name] = hit[0]
    if not wanted:
        return out
    answered: dict = {}
    try:
        from huggingface_hub import HfApi
        for entry in HfApi().get_paths_info(
            repo_id, wanted, revision = revision, token = token or None
        ):
            lfs = getattr(entry, "lfs", None)
            sha = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
            path = getattr(entry, "path", None)
            if path and isinstance(sha, str) and len(sha) == 64:
                answered[path] = sha.lower()
    except Exception:  # noqa: BLE001 - offline, gated, rate limited: unverified, retried later
        with _lock:
            for name in wanted:
                _hub[(repo_id, revision, name)] = (None, now)
        return out
    with _lock:
        for name in wanted:
            sha = answered.get(name)
            # A listed non-LFS file is answered for good (""), a missing one retried after the TTL.
            _hub[(repo_id, revision, name)] = (sha or None, now)
            if sha:
                out[name] = sha
    return out


def _store_path() -> Optional[Path]:
    try:
        from utils.paths.storage_roots import cache_root
        return cache_root() / _FILE_NAME
    except Exception:  # noqa: BLE001 - outside the Studio backend: memory only
        return None


def _load_persisted() -> dict:
    global _persisted
    if _persisted is None:
        _persisted = {}
        path = _store_path()
        try:
            if path is not None and path.is_file():
                data = json.loads(path.read_text(encoding = "utf-8"))
                if isinstance(data, dict):
                    _persisted = {
                        k: v for k, v in data.items() if isinstance(v, list) and len(v) == 3
                    }
        except Exception:  # noqa: BLE001 - a bad record file is an empty one
            _persisted = {}
    return _persisted


def _save_persisted(data: dict) -> None:
    path = _store_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents = True, exist_ok = True)
        keep = dict(list(data.items())[-_MAX_ENTRIES:])
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(keep), encoding = "utf-8")
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001 - the digest is still remembered in memory
        pass


def file_sha256(path: str) -> Optional[str]:
    """The sha256 of ``path``'s bytes (symlinks followed), from the record when the size and mtime still match."""
    try:
        real = os.path.realpath(path)
        st = os.stat(real)
    except OSError:
        return None
    key = (real, int(st.st_size), int(st.st_mtime_ns))
    with _lock:
        hit = _local.get(key)
        if hit is not None:
            return hit
        rec = _load_persisted().get(real)
        if rec and int(rec[0]) == key[1] and int(rec[1]) == key[2]:
            _local[key] = str(rec[2])
            return str(rec[2])
    digest = hashlib.sha256()
    try:
        with open(real, "rb") as handle:
            for chunk in iter(lambda: handle.read(_CHUNK), b""):
                digest.update(chunk)
    except OSError:
        return None
    sha = digest.hexdigest()
    with _lock:
        _local[key] = sha
        data = _load_persisted()
        data.pop(real, None)
        data[real] = [key[1], key[2], sha]
        _save_persisted(data)
    return sha


def cached_copy_matches_hub(
    repo_id: Optional[str],
    filename: Optional[str],
    path: Optional[str],
    *,
    revision: Optional[str] = None,
    token: Optional[str] = None,
    expected: Optional[str] = None,
) -> bool:
    """True only when the Hub records an sha256 for ``repo_id/filename`` and ``path`` hashes to it."""
    if not reuse_enabled() or not repo_id or not filename or not path:
        return False
    try:
        sha = expected or hub_sha256s(repo_id, [filename], revision = revision, token = token).get(
            filename
        )
        if not sha:
            return False
        # Always the bytes, never the cache blob's name: newer huggingface_hub caches name blobs by a Xet / etag
        # hash that is not the file's sha256.
        return file_sha256(path) == sha
    except Exception:  # noqa: BLE001 - unverified
        return False
