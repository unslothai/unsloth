# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Extended test matrix for ``LlamaCppBackend.load_progress()``.

Companion to ``test_llama_cpp_load_progress.py`` (which pins the basic
contract). This file widens coverage to the edge cases that bit users
or were hypothesized to bite them on cross-platform installs:

  * Platform matrix — macOS/Windows simulation via ``/proc`` absence.
  * ``VmRSS`` parsing — tab vs space delimiter, missing line, malformed
    integer.
  * Filesystem edges — HF-cache symlinks, broken symlinks, nonexistent
    paths, relative paths.
  * Shard aggregation — partial multi-shard downloads where some shards
    are still ``.incomplete``, two shard series in the same dir,
    ``mmproj-*.gguf`` sibling exclusion for non-sharded primaries,
    single-file models.
  * Lifecycle races — process set before ``_gguf_path`` is assigned,
    process dead mid-sample, ``_healthy`` flipped to True.
  * Concurrent sampling — 10 threads × 50 iterations against a single
    backend, hitting real ``/proc`` (no mocks — see the note in
    ``TestConcurrentSampling`` for why).
  * Fraction bounds — capped at 1.0 when RSS exceeds total; 0.0 when
    total is zero.

All tests are Linux-only in practice (we stub ``/proc`` where needed).
The stable subset runs in well under a second.
"""

from __future__ import annotations

import io
import os
import sys
import threading
import types as _types
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Stub heavy / unavailable external dependencies before importing the
# module under test. Same pattern as test_llama_cpp_load_progress.py.
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


def _make():
    inst = LlamaCppBackend.__new__(LlamaCppBackend)
    inst._process = None
    inst._gguf_path = None
    inst._healthy = False
    return inst


class _Proc:
    def __init__(self, pid):
        self.pid = pid


def _sparse(path, size):
    with open(path, "wb") as f:
        if size > 0:
            f.truncate(size)


def _fake_proc_reader(rss_kb):
    """Return an ``open()`` replacement that fakes /proc reads with a VmRSS line."""

    def fake_open(path, *args, **kwargs):
        if str(path).startswith("/proc/"):
            return io.StringIO(f"VmRSS:\t{rss_kb}\tkB\n")
        return open(path, *args, **kwargs)

    return fake_open


# ---------------------------------------------------------------------------
# A. Platform matrix
# ---------------------------------------------------------------------------


class TestPlatformMatrix:
    """The method is Linux-first via /proc. On macOS/Windows it must
    degrade to None rather than crash."""

    def test_linux_live_proc_is_self_pid(self, tmp_path):
        """Self-pid /proc read uses the real kernel interface."""
        gguf = tmp_path / "m.gguf"
        _sparse(gguf, 1 * 1024**3)
        inst = _make()
        inst._process = _Proc(os.getpid())
        inst._gguf_path = str(gguf)
        inst._healthy = False
        out = inst.load_progress()
        assert out is not None
        assert out["phase"] == "mmap"
        assert out["bytes_total"] == 1 * 1024**3
        # Our Python process has some RSS -- just sanity-check positive.
        assert out["bytes_loaded"] > 0

    def test_macos_no_proc_returns_none(self, tmp_path):
        """Simulate macOS: /proc open fails with FileNotFoundError."""
        gguf = tmp_path / "m.gguf"
        _sparse(gguf, 1 * 1024**3)
        inst = _make()
        inst._process = _Proc(pid = 12345)
        inst._gguf_path = str(gguf)

        def fake_open(path, *args, **kwargs):
            if str(path).startswith("/proc/"):
                raise FileNotFoundError(f"No such file: {path}")
            return open(path, *args, **kwargs)

        with patch("builtins.open", side_effect = fake_open):
            out = inst.load_progress()
        assert out is None

    def test_windows_no_proc_returns_none(self, tmp_path):
        """Simulate Windows: opening /proc raises PermissionError or OSError."""
        gguf = tmp_path / "m.gguf"
        _sparse(gguf, 1 * 1024**3)
        inst = _make()
        inst._process = _Proc(pid = 4567)
        inst._gguf_path = str(gguf)

        def fake_open(path, *args, **kwargs):
            if str(path).startswith("/proc/"):
                raise PermissionError("access denied")
            return open(path, *args, **kwargs)

        with patch("builtins.open", side_effect = fake_open):
            out = inst.load_progress()
        assert out is None


# ---------------------------------------------------------------------------
# B. VmRSS parsing edge cases
# ---------------------------------------------------------------------------


class TestVmRSSParsing:
    def test_standard_tab_delimited(self, tmp_path):
        gguf = tmp_path / "m.gguf"
        _sparse(gguf, 4 * 1024**3)
        inst = _make()
        inst._process = _Proc(os.getpid())
        inst._gguf_path = str(gguf)
        with patch("builtins.open", side_effect = _fake_proc_reader(2 * 1024**2)):
            out = inst.load_progress()
        assert out["bytes_loaded"] == 2 * 1024**3

    def test_space_separated_fallback(self, tmp_path):
        """Some kernels emit single-space rather than tab."""
        gguf = tmp_path / "m.gguf"
        _sparse(gguf, 4 * 1024**3)
        inst = _make()
        inst._process = _Proc(os.getpid())
        inst._gguf_path = str(gguf)

        def fake_open(path, *a, **kw):
            if str(path).startswith("/proc/"):
                return io.StringIO("VmRSS: 4194304 kB\n")
            return open(path, *a, **kw)

        with patch("builtins.open", side_effect = fake_open):
            out = inst.load_progress()
        assert out["bytes_loaded"] == 4 * 1024**3

    def test_missing_vmrss_line(self, tmp_path):
        """Kernel with VmRSS stripped (zombie / kthread) -> 0."""
        gguf = tmp_path / "m.gguf"
        _sparse(gguf, 1 * 1024**3)
        inst = _make()
        inst._process = _Proc(os.getpid())
        inst._gguf_path = str(gguf)

        def fake_open(path, *a, **kw):
            if str(path).startswith("/proc/"):
                return io.StringIO("Name:\ttest\nState:\tZ (zombie)\n")
            return open(path, *a, **kw)

        with patch("builtins.open", side_effect = fake_open):
            out = inst.load_progress()
        assert out is not None
        assert out["bytes_loaded"] == 0
        assert out["fraction"] == 0.0

    def test_malformed_vmrss_value(self, tmp_path):
        """Non-integer VmRSS value should be treated as if the line were
        absent (early ValueError caught)."""
        gguf = tmp_path / "m.gguf"
        _sparse(gguf, 1 * 1024**3)
        inst = _make()
        inst._process = _Proc(os.getpid())
        inst._gguf_path = str(gguf)

        def fake_open(path, *a, **kw):
            if str(path).startswith("/proc/"):
                return io.StringIO("VmRSS:\tXXXX\tkB\n")
            return open(path, *a, **kw)

        with patch("builtins.open", side_effect = fake_open):
            out = inst.load_progress()
        # The implementation catches ValueError on int() and returns None.
        assert out is None


# ---------------------------------------------------------------------------
# C. Filesystem edge cases
# ---------------------------------------------------------------------------


class TestFilesystemEdges:
    def test_symlink_primary_follows_to_blob(self, tmp_path):
        """HF cache stores blobs under blobs/ and symlinks them from
        snapshots/. The method must follow the symlink."""
        blob = tmp_path / "blob"
        _sparse(blob, 12 * 1024**3)
        snap = tmp_path / "snap"
        snap.mkdir()
        link = snap / "m.gguf"
        link.symlink_to(blob)

        inst = _make()
        inst._process = _Proc(os.getpid())
        inst._gguf_path = str(link)
        with patch("builtins.open", side_effect = _fake_proc_reader(6 * 1024**2)):
            out = inst.load_progress()
        assert out["bytes_total"] == 12 * 1024**3

    def test_broken_symlink_skipped(self, tmp_path):
        snap = tmp_path / "snap"
        snap.mkdir()
        link = snap / "m.gguf"
        link.symlink_to(tmp_path / "missing-blob")
        inst = _make()
        inst._process = _Proc(os.getpid())
        inst._gguf_path = str(link)
        with patch("builtins.open", side_effect = _fake_proc_reader(1024)):
            out = inst.load_progress()
        assert out["bytes_total"] == 0
        assert out["bytes_loaded"] == 1024 * 1024

    def test_nonexistent_path_skipped(self, tmp_path):
        inst = _make()
        inst._process = _Proc(os.getpid())
        inst._gguf_path = str(tmp_path / "ghost.gguf")
        with patch("builtins.open", side_effect = _fake_proc_reader(1024)):
            out = inst.load_progress()
        assert out["bytes_total"] == 0

    def test_relative_gguf_path(self, tmp_path):
        """Relative paths shouldn't crash; behaviour depends on CWD but
        the method must not raise."""
        cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            _sparse(Path("rel.gguf"), 8 * 1024**3)
            inst = _make()
            inst._process = _Proc(os.getpid())
            inst._gguf_path = "rel.gguf"
            with patch("builtins.open", side_effect = _fake_proc_reader(0)):
                out = inst.load_progress()
            assert out is not None
            assert out["bytes_total"] == 8 * 1024**3
        finally:
            os.chdir(cwd)


# ---------------------------------------------------------------------------
# D. Shard aggregation
# ---------------------------------------------------------------------------


class TestShardAggregation:
    def test_partial_multi_shard_download(self, tmp_path):
        """Primary present but shards 2..N still downloading as
        ``.incomplete``. Sums only the fully-arrived ``.gguf`` files."""
        _sparse(tmp_path / "m-00001-of-00004.gguf", 30 * 1024**3)
        _sparse(tmp_path / "m-00002-of-00004.gguf", 30 * 1024**3)
        # 3 and 4 still downloading as .incomplete
        _sparse(tmp_path / "m-00003-of-00004.gguf.incomplete", 5 * 1024**3)
        inst = _make()
        inst._process = _Proc(os.getpid())
        inst._gguf_path = str(tmp_path / "m-00001-of-00004.gguf")
        with patch("builtins.open", side_effect = _fake_proc_reader(0)):
            out = inst.load_progress()
        assert out["bytes_total"] == 60 * 1024**3  # only the .gguf siblings

    def test_two_shard_series_in_same_dir(self, tmp_path):
        """Defensive: if two quant series share a dir, prefix filter
        only sums siblings of the chosen primary."""
        for i in range(1, 3):
            _sparse(tmp_path / f"m_q4-{i:05d}-of-00002.gguf", 10 * 1024**3)
            _sparse(tmp_path / f"m_q8-{i:05d}-of-00002.gguf", 20 * 1024**3)
        inst = _make()
        inst._process = _Proc(os.getpid())
        inst._gguf_path = str(tmp_path / "m_q8-00001-of-00002.gguf")
        with patch("builtins.open", side_effect = _fake_proc_reader(0)):
            out = inst.load_progress()
        assert out["bytes_total"] == 40 * 1024**3  # just q8 series

    def test_mmproj_sibling_not_counted(self, tmp_path):
        """Vision models drop an ``mmproj-*.gguf`` alongside. For a
        single-file (non-sharded) primary we only count the primary."""
        _sparse(tmp_path / "m.gguf", 8 * 1024**3)
        _sparse(tmp_path / "mmproj-BF16.gguf", 2 * 1024**3)
        inst = _make()
        inst._process = _Proc(os.getpid())
        inst._gguf_path = str(tmp_path / "m.gguf")
        with patch("builtins.open", side_effect = _fake_proc_reader(0)):
            out = inst.load_progress()
        # Non-sharded primary: only the primary is counted.
        assert out["bytes_total"] == 8 * 1024**3

    def test_single_file_model(self, tmp_path):
        """Non-sharded model: primary only."""
        _sparse(tmp_path / "small.gguf", 4 * 1024**3)
        inst = _make()
        inst._process = _Proc(os.getpid())
        inst._gguf_path = str(tmp_path / "small.gguf")
        with patch("builtins.open", side_effect = _fake_proc_reader(2 * 1024**2)):
            out = inst.load_progress()
        assert out["bytes_total"] == 4 * 1024**3
        assert out["bytes_loaded"] == 2 * 1024**3


# ---------------------------------------------------------------------------
# E. Lifecycle races
# ---------------------------------------------------------------------------


class TestLifecycleRaces:
    def test_process_set_but_gguf_path_not_yet(self, tmp_path):
        """Moment between Popen and self._gguf_path=model_path."""
        inst = _make()
        inst._process = _Proc(os.getpid())
        inst._gguf_path = None
        with patch("builtins.open", side_effect = _fake_proc_reader(1024)):
            out = inst.load_progress()
        assert out is not None
        assert out["phase"] == "mmap"
        assert out["bytes_total"] == 0
        assert out["bytes_loaded"] == 1024 * 1024

    def test_process_died_mid_sample(self, tmp_path):
        """/proc/<pid> disappears -> None."""
        _sparse(tmp_path / "m.gguf", 1 * 1024**3)
        inst = _make()
        inst._process = _Proc(pid = 999_999_999)
        inst._gguf_path = str(tmp_path / "m.gguf")
        assert inst.load_progress() is None

    def test_healthy_true_ready_phase(self, tmp_path):
        _sparse(tmp_path / "m.gguf", 1 * 1024**3)
        inst = _make()
        inst._process = _Proc(os.getpid())
        inst._gguf_path = str(tmp_path / "m.gguf")
        inst._healthy = True
        with patch("builtins.open", side_effect = _fake_proc_reader(1024)):
            out = inst.load_progress()
        assert out["phase"] == "ready"


# ---------------------------------------------------------------------------
# F. Concurrent sampling  (simulates multiple browser tabs polling)
# ---------------------------------------------------------------------------


class TestConcurrentSampling:
    def test_parallel_invocations_never_raise(self, tmp_path):
        """Many concurrent samplers hitting the same backend must not raise.

        We intentionally do NOT patch ``builtins.open`` here because
        ``unittest.mock.patch`` is not thread-safe: interleaved
        enter/exit across threads can leak a Mock into ``builtins.open``
        and poison every subsequent test in the session. Instead, we
        let each thread hit the real ``/proc/self/status`` of the test
        process, which is exactly the code path that matters in prod.
        """
        _sparse(tmp_path / "m.gguf", 1 * 1024**3)
        inst = _make()
        inst._process = _Proc(os.getpid())
        inst._gguf_path = str(tmp_path / "m.gguf")
        errors = []

        def run():
            try:
                for _ in range(50):
                    inst.load_progress()
            except Exception as e:  # pragma: no cover
                errors.append(e)

        threads = [threading.Thread(target = run) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors


# ---------------------------------------------------------------------------
# G. Fraction bounds
# ---------------------------------------------------------------------------


class TestFractionBounds:
    def test_fraction_capped_at_one(self, tmp_path):
        _sparse(tmp_path / "m.gguf", 1 * 1024**3)
        inst = _make()
        inst._process = _Proc(os.getpid())
        inst._gguf_path = str(tmp_path / "m.gguf")
        # RSS > total (post-paged-in + extra structures)
        with patch("builtins.open", side_effect = _fake_proc_reader(2 * 1024**2)):
            out = inst.load_progress()
        assert 0.0 <= out["fraction"] <= 1.0

    def test_fraction_zero_when_total_zero(self):
        inst = _make()
        inst._process = _Proc(os.getpid())
        inst._gguf_path = None
        with patch("builtins.open", side_effect = _fake_proc_reader(1024**2)):
            out = inst.load_progress()
        assert out["fraction"] == 0.0
