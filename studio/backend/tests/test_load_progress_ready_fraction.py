# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""load_progress() must report a complete load once llama-server is healthy.

With layers offloaded to VRAM (-ngl) the server releases the mmap'd weight pages
after upload, so its VmRSS sinks back well below the shard total. The raw RSS
fraction would then sit at a partial (~8%) value forever and freeze a
fraction-driven progress bar even though the model is ready -- the "stuck around
8% on the second pass" symptom in #5740. In the ready phase the fraction must be
1.0 regardless of resident set size.
"""

from __future__ import annotations

import io
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

# Stub heavy/unavailable deps before importing the module under test, so a
# targeted run in the lightweight backend env (no structlog/httpx) still
# collects. setdefault keeps the real modules when they are installed. Mirrors
# test_llama_cpp_load_progress_matrix.py.
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

sys.modules.setdefault("structlog", types.ModuleType("structlog"))

_httpx_stub = types.ModuleType("httpx")
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

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402


def _backend(
    gguf_path,
    *,
    healthy,
    pid = 4321,
):
    # Bare instance: exercise load_progress() without the heavy real __init__.
    be = object.__new__(LlamaCppBackend)
    be._process = types.SimpleNamespace(pid = pid)
    be._gguf_path = str(gguf_path)
    be._healthy = healthy
    return be


def _gguf(tmp_path, size_bytes):
    f = tmp_path / "model-Q4_K_M.gguf"
    f.write_bytes(b"\0" * size_bytes)
    return f


def test_ready_reports_complete_despite_low_rss(tmp_path, monkeypatch):
    # Healthy, but VmRSS has dropped to ~8% of the shard total after VRAM upload.
    monkeypatch.setattr(
        LlamaCppBackend, "_read_rss_bytes", staticmethod(lambda pid: 800)
    )
    be = _backend(_gguf(tmp_path, 10000), healthy = True)
    p = be.load_progress()
    assert p["phase"] == "ready"
    assert p["fraction"] == 1.0  # not 0.08
    assert p["bytes_loaded"] == p["bytes_total"] == 10000


def test_mmap_phase_reports_raw_rss_fraction(tmp_path, monkeypatch):
    # Still loading: the bar should track real residency, not jump to 1.0.
    monkeypatch.setattr(
        LlamaCppBackend, "_read_rss_bytes", staticmethod(lambda pid: 800)
    )
    be = _backend(_gguf(tmp_path, 10000), healthy = False)
    p = be.load_progress()
    assert p["phase"] == "mmap"
    assert p["fraction"] == 0.08
    assert p["bytes_loaded"] == 800
    assert p["bytes_total"] == 10000


def test_progress_fraction_is_monotonic(tmp_path, monkeypatch):
    # RSS peaks during page-in, then drops after -ngl offload; the bar must hold
    # its high-water mark instead of collapsing back to ~8% (#5740).
    be = _backend(_gguf(tmp_path, 10000), healthy = False)
    monkeypatch.setattr(
        LlamaCppBackend, "_read_rss_bytes", staticmethod(lambda pid: 9000)
    )
    assert be.load_progress()["fraction"] == 0.9
    monkeypatch.setattr(
        LlamaCppBackend, "_read_rss_bytes", staticmethod(lambda pid: 800)
    )
    p = be.load_progress()
    assert p["fraction"] == 0.9
    assert p["bytes_loaded"] == 9000


def test_ready_without_shard_size_still_completes(tmp_path, monkeypatch):
    # bytes_total unknown (file unstattable): fraction must still read complete.
    monkeypatch.setattr(
        LlamaCppBackend, "_read_rss_bytes", staticmethod(lambda pid: 800)
    )
    be = _backend(tmp_path / "missing.gguf", healthy = True)
    p = be.load_progress()
    assert p["phase"] == "ready"
    assert p["fraction"] == 1.0
    assert p["bytes_total"] == 0


def test_none_when_no_process(tmp_path):
    be = _backend(_gguf(tmp_path, 10000), healthy = True)
    be._process = None
    assert be.load_progress() is None


def test_none_when_rss_unreadable(tmp_path, monkeypatch):
    # /proc unavailable (macOS/Windows) or unreadable -> no progress payload.
    monkeypatch.setattr(
        LlamaCppBackend, "_read_rss_bytes", staticmethod(lambda pid: None)
    )
    be = _backend(_gguf(tmp_path, 10000), healthy = False)
    assert be.load_progress() is None


def test_read_rss_bytes_absent_pid_is_none():
    # A pid with no readable /proc entry (or no /proc at all) yields None, never
    # raises.
    assert LlamaCppBackend._read_rss_bytes(2**31 - 1) is None


def test_read_rss_bytes_valueless_line_is_none():
    # A "VmRSS:" line with no value column must not raise (IndexError) -> None.
    def fake_open(path, *a, **kw):
        if str(path).startswith("/proc/"):
            return io.StringIO("Name:\ttest\nVmRSS:\n")
        return open(path, *a, **kw)

    with patch("builtins.open", side_effect = fake_open):
        assert LlamaCppBackend._read_rss_bytes(4321) is None


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "/proc is Linux-only")
def test_read_rss_bytes_reads_self_on_linux():
    rss = LlamaCppBackend._read_rss_bytes(__import__("os").getpid())
    assert isinstance(rss, int) and rss > 0
