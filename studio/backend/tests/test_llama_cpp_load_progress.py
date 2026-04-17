# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for ``LlamaCppBackend.load_progress()``.

The chat settings flow and the training overlay both show a generic
"Starting model..." spinner during the window after a GGUF download
finishes and before llama-server reports healthy. For small models
that window is a second or two and nobody notices. For large MoE GGUFs
(MiniMax-M2.7, Qwen3.5-397B-A17B, etc.) the llama-server process spends
minutes in kernel state D, paging tens or hundreds of GB of shards
into the page cache. The UI has no way to show a real progress bar,
rate, or ETA during that window.

``load_progress()`` samples ``/proc/<pid>/status VmRSS`` (what the
kernel has actually paged in) against the total shard file size on
disk, so the frontend can render a real bar plus rate/ETA. This
module pins that contract:

  * returns ``None`` when no load is in flight
  * returns ``{"phase": "mmap", ...}`` while the subprocess is alive
    but ``_healthy`` is False
  * returns ``{"phase": "ready", ...}`` once ``_healthy`` flips
  * ``bytes_total`` is derived from the resolved on-disk path
    (which the paired fix assigns to ``self._gguf_path`` on both the
    local-GGUF and HF-download code paths)
  * ``bytes_loaded`` is VmRSS in bytes, capped by total, rounded
  * ``fraction`` is clamped to 0..1 and rounded to 4 decimal places

Linux-only via ``/proc``. On platforms without ``/proc`` the method
returns ``None`` instead of raising.
Cross-platform test: skips cleanly on macOS / Windows if ``/proc`` is
not available.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types as _types
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Stub heavy / unavailable external dependencies before importing the
# module under test. Same pattern as test_kv_cache_estimation.py.
# ---------------------------------------------------------------------------

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
sys.modules.setdefault("structlog", _structlog_stub)

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

from core.inference.llama_cpp import LlamaCppBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_instance():
    inst = LlamaCppBackend.__new__(LlamaCppBackend)
    inst._process = None
    inst._gguf_path = None
    inst._healthy = False
    return inst


class _FakeProc:
    """Minimal stand-in for subprocess.Popen that just carries a pid."""

    def __init__(self, pid: int):
        self.pid = pid


def _write_sparse_file(path: Path, size_bytes: int) -> None:
    """Create a sparse file of the given size without allocating blocks."""
    with open(path, "wb") as fh:
        if size_bytes > 0:
            fh.truncate(size_bytes)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadProgressEmptyStates:
    def test_returns_none_when_no_process(self):
        inst = _make_instance()
        assert inst.load_progress() is None

    def test_returns_none_when_process_has_no_pid(self):
        inst = _make_instance()
        inst._process = _FakeProc(pid = None)  # type: ignore[arg-type]
        assert inst.load_progress() is None


class TestLoadProgressSingleShard:
    def test_mmap_phase_for_alive_but_unhealthy(self, tmp_path):
        """VmRSS below total -> phase='mmap', fraction reflects progress."""
        gguf = tmp_path / "model.gguf"
        _write_sparse_file(gguf, 40 * 1024**3)  # 40 GB

        inst = _make_instance()
        inst._process = _FakeProc(pid = os.getpid())  # use our own pid
        inst._gguf_path = str(gguf)
        inst._healthy = False

        # Patch /proc read to claim 10 GB RSS.
        def fake_open(path, *args, **kwargs):
            if str(path).startswith("/proc/"):
                import io

                return io.StringIO(f"Name:\ttest\nVmRSS:\t{10 * 1024 ** 2}\tkB\n")
            return open(path, *args, **kwargs)  # fall through

        with patch("builtins.open", side_effect = fake_open):
            out = inst.load_progress()

        assert out is not None
        assert out["phase"] == "mmap"
        assert out["bytes_total"] == 40 * 1024**3
        assert out["bytes_loaded"] == 10 * 1024**3
        assert 0.24 < out["fraction"] < 0.26  # ~25%

    def test_ready_phase_when_healthy(self, tmp_path):
        gguf = tmp_path / "model.gguf"
        _write_sparse_file(gguf, 8 * 1024**3)

        inst = _make_instance()
        inst._process = _FakeProc(pid = os.getpid())
        inst._gguf_path = str(gguf)
        inst._healthy = True

        def fake_open(path, *args, **kwargs):
            if str(path).startswith("/proc/"):
                import io

                return io.StringIO(f"VmRSS:\t{8 * 1024 ** 2}\tkB\n")
            return open(path, *args, **kwargs)

        with patch("builtins.open", side_effect = fake_open):
            out = inst.load_progress()

        assert out is not None
        assert out["phase"] == "ready"
        assert out["bytes_total"] == 8 * 1024**3
        assert out["bytes_loaded"] == 8 * 1024**3
        assert out["fraction"] == 1.0


class TestLoadProgressMultiShard:
    """Shard-aware total: for ``*-00001-of-00004.gguf`` primaries the
    method sums sibling files with the same prefix."""

    def test_sharded_total_aggregates_siblings(self, tmp_path):
        for i in range(1, 5):
            _write_sparse_file(
                tmp_path / f"model-{i:05d}-of-00004.gguf",
                size_bytes = 20 * 1024**3,
            )
        # Drop an unrelated .gguf in the same folder -- must not be counted.
        _write_sparse_file(tmp_path / "mmproj-BF16.gguf", 2 * 1024**3)

        inst = _make_instance()
        inst._process = _FakeProc(pid = os.getpid())
        inst._gguf_path = str(tmp_path / "model-00001-of-00004.gguf")
        inst._healthy = False

        def fake_open(path, *args, **kwargs):
            if str(path).startswith("/proc/"):
                import io

                return io.StringIO("VmRSS:\t0\tkB\n")
            return open(path, *args, **kwargs)

        with patch("builtins.open", side_effect = fake_open):
            out = inst.load_progress()

        assert out is not None
        assert out["bytes_total"] == 80 * 1024**3  # 4 x 20 GB, no mmproj


class TestLoadProgressDegradation:
    """Broken / unusual inputs never raise; they produce best-effort output."""

    def test_missing_gguf_path_still_reports_rss(self, tmp_path):
        inst = _make_instance()
        inst._process = _FakeProc(pid = os.getpid())
        inst._gguf_path = None
        inst._healthy = False

        def fake_open(path, *args, **kwargs):
            if str(path).startswith("/proc/"):
                import io

                return io.StringIO("VmRSS:\t1024\tkB\n")
            return open(path, *args, **kwargs)

        with patch("builtins.open", side_effect = fake_open):
            out = inst.load_progress()

        assert out is not None
        assert out["phase"] == "mmap"
        assert out["bytes_total"] == 0
        assert out["bytes_loaded"] == 1024 * 1024
        assert out["fraction"] == 0.0

    def test_unreadable_proc_returns_none(self, tmp_path):
        inst = _make_instance()
        # Pid that doesn't exist -> /proc read fails.
        inst._process = _FakeProc(pid = 999_999_999)
        inst._gguf_path = str(tmp_path / "model.gguf")  # doesn't need to exist
        inst._healthy = False

        out = inst.load_progress()
        # FileNotFoundError on /proc path -> load_progress returns None.
        assert out is None
