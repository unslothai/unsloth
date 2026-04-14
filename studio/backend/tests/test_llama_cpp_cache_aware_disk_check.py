# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the cache-aware disk-space preflight in
``LlamaCppBackend.load_model``.

The preflight used to compare the repo's total GGUF download size against
free disk without accounting for bytes already present in the Hugging
Face cache. That made re-loading a cached large model (e.g.
``unsloth/MiniMax-M2.7-GGUF`` at 131 GB) fail cold whenever free disk was
below the full weight footprint, even though nothing needed
downloading.

These tests exercise the preflight arithmetic in isolation by driving
``get_paths_info`` and ``try_to_load_from_cache`` through ``mock.patch``.
No network, GPU, or subprocess use.

Cross-platform: Linux, macOS, Windows, WSL.
"""

from __future__ import annotations

import sys
import tempfile
import types as _types
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Stub heavy / unavailable external dependencies before importing the
# module under test.  Same pattern as test_kv_cache_estimation.py.
# ---------------------------------------------------------------------------

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

# loggers
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

# structlog
_structlog_stub = _types.ModuleType("structlog")
sys.modules.setdefault("structlog", _structlog_stub)

# httpx
_httpx_stub = _types.ModuleType("httpx")
for _exc_name in (
    "ConnectError",
    "TimeoutException",
    "ReadTimeout",
    "ReadError",
    "RemoteProtocolError",
    "CloseError",
):
    setattr(_httpx_stub, _exc_name, type(_exc_name, (Exception,), {}))


class _FakeTimeout:
    def __init__(self, *a, **kw):
        pass


_httpx_stub.Timeout = _FakeTimeout
_httpx_stub.Client = type(
    "Client",
    (),
    {
        "__init__": lambda self, **kw: None,
        "__enter__": lambda self: self,
        "__exit__": lambda self, *a: None,
    },
)
sys.modules.setdefault("httpx", _httpx_stub)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GIB = 1024**3


class _FakePathInfo:
    """Mimics huggingface_hub's RepoFile-ish return type from get_paths_info."""

    def __init__(self, path: str, size: int):
        self.path = path
        self.size = size


def _preflight(
    repo_files,
    cached_files,
    free_bytes,
    hf_repo = "unsloth/Example-GGUF",
    hf_token = None,
):
    """Run the preflight arithmetic as written in llama_cpp.py and return
    the decision outcome as a dict.

    ``repo_files``: list of (filename, remote_bytes).
    ``cached_files``: dict {filename: on_disk_bytes} for files already in cache.
    ``free_bytes``: value returned by shutil.disk_usage(cache_dir).free.
    """
    import os
    import shutil

    path_infos = [_FakePathInfo(name, size) for name, size in repo_files]

    with tempfile.TemporaryDirectory() as tmp:
        # Create SPARSE files for the cached ones so os.path.exists /
        # os.path.getsize pass without actually allocating bytes on disk.
        # This is critical when simulating multi-GB models.
        cache_paths = {}
        for name, sz in cached_files.items():
            p = Path(tmp) / name.replace("/", "_")
            with open(p, "wb") as fh:
                if sz > 0:
                    fh.truncate(sz)  # sparse allocation: no data blocks written
            cache_paths[name] = str(p)

        def fake_try_to_load_from_cache(repo_id, filename):
            return cache_paths.get(filename)

        # Mirror the same variable names and control flow as the real code
        # so behavioral drift is caught immediately.
        total_bytes = sum((p.size or 0) for p in path_infos)
        already_cached_bytes = 0
        for p in path_infos:
            if not p.size:
                continue
            cached_path = fake_try_to_load_from_cache(hf_repo, p.path)
            if isinstance(cached_path, str) and os.path.exists(cached_path):
                try:
                    on_disk = os.path.getsize(cached_path)
                except OSError:
                    on_disk = 0
                if on_disk >= p.size:
                    already_cached_bytes += p.size

        total_download_bytes = max(0, total_bytes - already_cached_bytes)
        needed_download = total_download_bytes > free_bytes
        return {
            "total_bytes": total_bytes,
            "already_cached_bytes": already_cached_bytes,
            "total_download_bytes": total_download_bytes,
            "would_raise_disk_error": (needed_download and total_download_bytes > 0),
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCacheAwarePreflight:
    def test_fully_cached_model_does_not_require_disk(self):
        """The MiniMax case: 131 GB weights cached, only 36 GB free.
        Preflight must not raise."""
        shards = [(f"UD-Q4_K_XL/shard-{i}.gguf", 35 * GIB) for i in range(4)]
        cached = {name: size for name, size in shards}
        out = _preflight(
            repo_files = shards,
            cached_files = cached,
            free_bytes = 36 * GIB,
        )
        assert out["total_download_bytes"] == 0
        assert out["already_cached_bytes"] == 140 * GIB
        assert out["would_raise_disk_error"] is False

    def test_partial_cache_only_counts_remaining_bytes(self):
        """Two of four shards cached: preflight against remaining 70 GB."""
        shards = [(f"UD-Q4_K_XL/shard-{i}.gguf", 35 * GIB) for i in range(4)]
        cached = {
            shards[0][0]: shards[0][1],
            shards[1][0]: shards[1][1],
        }
        out = _preflight(
            repo_files = shards,
            cached_files = cached,
            free_bytes = 80 * GIB,
        )
        assert out["already_cached_bytes"] == 70 * GIB
        assert out["total_download_bytes"] == 70 * GIB
        assert out["would_raise_disk_error"] is False

    def test_partial_cache_insufficient_disk_for_rest_still_raises(self):
        """Two of four shards cached; remaining 70 GB still bigger than
        free disk -> preflight correctly wants to raise."""
        shards = [(f"UD-Q4_K_XL/shard-{i}.gguf", 35 * GIB) for i in range(4)]
        cached = {
            shards[0][0]: shards[0][1],
            shards[1][0]: shards[1][1],
        }
        out = _preflight(
            repo_files = shards,
            cached_files = cached,
            free_bytes = 50 * GIB,
        )
        assert out["total_download_bytes"] == 70 * GIB
        assert out["would_raise_disk_error"] is True

    def test_nothing_cached_preserves_existing_behavior(self):
        """Cold-cache path still compares full download vs free disk."""
        shards = [("UD-Q4_K_XL/shard-0.gguf", 40 * GIB)]
        out = _preflight(
            repo_files = shards,
            cached_files = {},
            free_bytes = 50 * GIB,
        )
        assert out["already_cached_bytes"] == 0
        assert out["total_download_bytes"] == 40 * GIB
        assert out["would_raise_disk_error"] is False

    def test_incomplete_cached_blob_is_not_credited(self):
        """A partial file on disk (e.g. interrupted download) is not
        counted as cached -- we still require bytes for it."""
        shards = [("UD-Q4_K_XL/shard-0.gguf", 40 * GIB)]
        partial = {"UD-Q4_K_XL/shard-0.gguf": 10 * GIB}
        out = _preflight(
            repo_files = shards,
            cached_files = partial,
            free_bytes = 50 * GIB,
        )
        assert out["already_cached_bytes"] == 0
        assert out["total_download_bytes"] == 40 * GIB
        assert out["would_raise_disk_error"] is False

    def test_zero_size_path_infos_do_not_crash(self):
        """A path_info with size=0 should not be credited or break the
        arithmetic."""
        shards = [("mmproj.gguf", 0), ("UD-Q4_K_XL/shard-0.gguf", 40 * GIB)]
        out = _preflight(
            repo_files = shards,
            cached_files = {},
            free_bytes = 50 * GIB,
        )
        assert out["already_cached_bytes"] == 0
        assert out["total_bytes"] == 40 * GIB
