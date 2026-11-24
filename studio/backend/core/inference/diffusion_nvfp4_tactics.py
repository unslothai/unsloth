# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""mm_fp4 tactics per (GEMM shape, M bucket): each bucket profiled once at its own M, reused by every M in it,
persisted per GPU, CUDA and FlashInfer build (a tactic indexes a static, shape-independent config table)."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# auto (default): bucketed + on disk; memory: bucketed, never written; 0: the old per-length tuning.
NVFP4_TACTIC_CACHE_ENV = "UNSLOTH_NVFP4_TACTIC_CACHE"
NVFP4_TACTIC_CACHE_DIR_ENV = "UNSLOTH_NVFP4_TACTIC_CACHE_DIR"
MODE_AUTO = "auto"
MODE_MEMORY = "memory"
MODE_OFF = "0"

SCHEMA = 1
_OP = "mm_fp4"

_LOCK = threading.Lock()
# env hash -> {entry key -> tactic}
_MEMORY: dict[str, dict[str, int]] = {}
_LOADED: set = set()
_IDENTITY: dict[tuple, tuple] = {}
_STATS = {"tuned": 0, "disk_entries_loaded": 0, "memory_hits": 0, "persist_failures": 0}


def mode() -> str:
    raw = (os.environ.get(NVFP4_TACTIC_CACHE_ENV) or "").strip().lower()
    if raw in ("0", "off", "false", "no", "legacy"):
        return MODE_OFF
    if raw == MODE_MEMORY:
        return MODE_MEMORY
    return MODE_AUTO


def bucketed_enabled() -> bool:
    return mode() != MODE_OFF


def persistent_enabled() -> bool:
    return mode() == MODE_AUTO


def m_bucket(m: int) -> int:
    """Largest bucket <= ``m``: powers of two, plus 1.5x each power from 768 up (spacing <= 1.5x)."""
    m = int(m)
    if m <= 1:
        return 1
    p = 1 << (m.bit_length() - 1)
    if p >= 512 and m >= p + (p >> 1):
        return p + (p >> 1)
    return p


def cache_dir() -> Optional[Path]:
    raw = os.environ.get(NVFP4_TACTIC_CACHE_DIR_ENV)
    if raw:
        return Path(raw)
    try:
        from utils.paths.storage_roots import cache_root
    except ImportError:
        return None
    return cache_root() / "nvfp4_tactics"


def environment(
    *,
    flashinfer_version: str,
    capability: tuple,
    device_name: str,
    sm_count: Any,
    cuda: Any,
    tactic_count: Any,
    backend: str = "cutlass",
) -> tuple[str, dict]:
    """``(hash, meta)``: everything a stored tactic index depends on. Any change is a different file."""
    meta = {
        "schema": SCHEMA,
        "op": _OP,
        "backend": str(backend),
        "sf_layout": "128x4",
        "out_dtype": "bfloat16",
        "flashinfer": str(flashinfer_version),
        "capability": [int(c) for c in capability],
        "device": str(device_name),
        "sm_count": None if sm_count is None else int(sm_count),
        "cuda": None if cuda is None else str(cuda),
        "tactic_count": None if tactic_count is None else int(tactic_count),
    }
    digest = hashlib.sha256(json.dumps(meta, sort_keys = True).encode("utf-8")).hexdigest()[:16]
    return digest, meta


def device_environment(
    device: Any,
    tactic_count: Any,
    backend: str = "cutlass",
) -> tuple[str, dict]:
    """``environment`` for a CUDA device, memoised per (device index, tactic count, backend)."""
    from .diffusion_nvfp4_ops import _device_index

    index = _device_index(device)
    key = (index, tactic_count, backend)
    got = _IDENTITY.get(key)
    if got is not None:
        return got
    import torch

    import flashinfer

    props = torch.cuda.get_device_properties(index)
    got = environment(
        flashinfer_version = str(getattr(flashinfer, "__version__", "unknown")),
        capability = tuple(torch.cuda.get_device_capability(index)),
        device_name = str(getattr(props, "name", "")) or torch.cuda.get_device_name(index),
        sm_count = getattr(props, "multi_processor_count", None),
        cuda = getattr(torch.version, "cuda", None),
        tactic_count = tactic_count,
        backend = backend,
    )
    with _LOCK:
        return _IDENTITY.setdefault(key, got)


def _entry_key(k: int, n: int, bucket: int) -> str:
    return f"{int(k)}x{int(n)}@{int(bucket)}"


def _path_for(env_hash: str) -> Optional[Path]:
    root = cache_dir()
    return None if root is None else root / f"{_OP}_{env_hash}.json"


def _valid_tactic(value: Any, meta: dict) -> bool:
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    count = meta.get("tactic_count")
    if count is None:
        return value >= -1
    return -1 <= value < int(count)


def _read_file(path: Path, meta: dict) -> dict[str, int]:
    """The valid entries of ``path``; a missing, foreign or corrupt file reads as empty, never raises."""
    try:
        with open(path, "r", encoding = "utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {}
    except Exception as exc:  # noqa: BLE001 - a corrupt file costs a retune, never a load
        logger.debug(
            "nvfp4.tactics: ignoring unreadable %s (%s: %s)", path, type(exc).__name__, exc
        )
        return {}
    if not isinstance(data, dict) or data.get("schema") != SCHEMA or data.get("meta") != meta:
        return {}
    entries = data.get("entries")
    if not isinstance(entries, dict):
        return {}
    return {str(k): int(v) for k, v in entries.items() if _valid_tactic(v, meta)}


class _FileLock:
    """Advisory cross-process lock where the OS has one (FlashInfer itself is Linux only)."""

    def __init__(self, path: Path):
        self.path = path.with_name(path.name + ".lock")
        self.fh = None

    def __enter__(self):
        try:
            import fcntl
            self.fh = open(self.path, "a+", encoding = "utf-8")
            fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX)
        except Exception:  # noqa: BLE001 - no lock means last writer wins, still atomic per file
            if self.fh is not None:
                self.fh.close()
            self.fh = None
        return self

    def __exit__(self, *exc):
        if self.fh is not None:
            try:
                import fcntl
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_UN)
            except Exception:  # noqa: BLE001
                pass
            self.fh.close()
            self.fh = None
        return False


def _persist(path: Path, meta: dict, update: dict[str, int]) -> bool:
    """Merge ``update`` into the file and publish it with an atomic rename."""
    try:
        path.parent.mkdir(parents = True, exist_ok = True)
        with _FileLock(path):
            entries = _read_file(path, meta)
            entries.update(update)
            tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            with open(tmp, "w", encoding = "utf-8") as fh:
                json.dump(
                    {"schema": SCHEMA, "meta": meta, "entries": dict(sorted(entries.items()))},
                    fh,
                    indent = 1,
                    sort_keys = True,
                )
            os.replace(tmp, path)
        return True
    except Exception as exc:  # noqa: BLE001 - persistence is an optimisation
        _STATS["persist_failures"] += 1
        logger.debug("nvfp4.tactics: could not write %s (%s: %s)", path, type(exc).__name__, exc)
        return False


def _disk_ok(meta: dict) -> bool:
    """On disk only when the table size is known: an index is only checkable against it."""
    return persistent_enabled() and meta.get("tactic_count") is not None


def _ensure_loaded(env_hash: str, meta: dict) -> dict[str, int]:
    table = _MEMORY.get(env_hash)
    if table is not None and env_hash in _LOADED:
        return table
    with _LOCK:
        table = _MEMORY.setdefault(env_hash, {})
        if env_hash in _LOADED:
            return table
        if _disk_ok(meta):
            path = _path_for(env_hash)
            if path is not None:
                on_disk = _read_file(path, meta)
                for key, value in on_disk.items():
                    table.setdefault(key, value)
                _STATS["disk_entries_loaded"] += len(on_disk)
        _LOADED.add(env_hash)
        return table


def lookup(env: tuple[str, dict], k: int, n: int, bucket: int) -> Optional[int]:
    """The stored tactic for this GEMM shape and bucket (memory, then disk once per environment), or None."""
    env_hash, meta = env
    table = _ensure_loaded(env_hash, meta)
    got = table.get(_entry_key(k, n, bucket))
    if got is not None:
        _STATS["memory_hits"] += 1
    return got


def record(
    env: tuple[str, dict],
    k: int,
    n: int,
    bucket: int,
    tactic: int,
    *,
    persist: bool = True,
) -> int:
    """Keep a tuned tactic (first writer wins in this process) and publish it; returns the kept one."""
    env_hash, meta = env
    if not _valid_tactic(tactic, meta):
        raise ValueError(
            f"tactic {tactic!r} is outside this build's table ({meta.get('tactic_count')})"
        )
    table = _ensure_loaded(env_hash, meta)
    key = _entry_key(k, n, bucket)
    with _LOCK:
        fresh = key not in table
        kept = table.setdefault(key, int(tactic))
        if fresh:
            _STATS["tuned"] += 1
    if fresh and persist and _disk_ok(meta):
        path = _path_for(env_hash)
        if path is not None:
            _persist(path, meta, {key: kept})
    return kept


def forget_memory() -> None:
    """Drop the in-process tables (tests; the disk is the source of truth across processes)."""
    with _LOCK:
        _MEMORY.clear()
        _LOADED.clear()
        _IDENTITY.clear()
        for key in _STATS:
            _STATS[key] = 0


def describe() -> dict:
    return {
        "mode": mode(),
        "dir": None if cache_dir() is None else str(cache_dir()),
        "entries": sum(len(t) for t in _MEMORY.values()),
        **_STATS,
    }
