# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Uses real subprocesses and /proc, not mocks, so kernel /proc layout drift is caught; Linux only."""

from __future__ import annotations

import os
import subprocess
import sys
import time
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
_structlog_stub = _types.ModuleType("structlog")
sys.modules.setdefault("structlog", _structlog_stub)
_httpx_stub = _types.ModuleType("httpx")
for _exc in (
    "ConnectError",
    "TimeoutException",
    "ReadTimeout",
    "ReadError",
    "RemoteProtocolError",
    "CloseError",
):
    setattr(_httpx_stub, _exc, type(_exc, (Exception,), {}))
_httpx_stub.Timeout = type("Timeout", (), {"__init__": lambda self, *a, **k: None})
_httpx_stub.Client = type(
    "Client",
    (),
    {
        "__init__": lambda self, **kw: None,
        "__enter__": lambda self: self,
        "__exit__": lambda self, *a: None,
    },
)
# Stub httpx only if missing: one without Response breaks later starlette.testclient imports.
try:
    import httpx  # noqa: F401
except ImportError:
    sys.modules.setdefault("httpx", _httpx_stub)

from core.inference.llama_cpp import LlamaCppBackend


pytestmark = pytest.mark.skipif(
    not Path("/proc").exists(),
    reason = "live /proc test is Linux-only",
)


def _make_backend(
    pid: int,
    gguf_path: str,
    healthy: bool = False,
):
    inst = LlamaCppBackend.__new__(LlamaCppBackend)
    inst._process = type("P", (), {"pid": pid})()
    inst._gguf_path = gguf_path
    inst._healthy = healthy
    return inst


def test_live_mapped_pages_count_and_allocations_do_not(tmp_path):
    """Spawn a real child that pages in a mapped file and fills an allocation: ``bytes_loaded``
    follows the mapped pages, not the allocation (a pinned upload buffer is one)."""
    data = tmp_path / "weights.bin"
    data.write_bytes(b"\1" * (100 * 1024 * 1024))
    script = tmp_path / "burn.py"
    script.write_text(
        "import mmap, sys, time\n"
        "f = open(sys.argv[1], 'rb')\n"
        "m = mmap.mmap(f.fileno(), 0, access = mmap.ACCESS_READ)\n"
        "# touch every page of the mapping and of the allocation so both are resident\n"
        "seen = sum(m[i] for i in range(0, len(m), 4096))\n"
        "buf = bytearray(200 * 1024 * 1024)\n"
        "for i in range(0, len(buf), 4096):\n"
        "    buf[i] = 1\n"
        "sys.stdout.write('ready\\n')\n"
        "sys.stdout.flush()\n"
        "time.sleep(10)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, str(script), str(data)],
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
    )
    try:
        ready = proc.stdout.readline()
        assert ready.strip() == b"ready"

        gguf = tmp_path / "model.gguf"
        with open(gguf, "wb") as f:
            f.truncate(400 * 1024 * 1024)

        inst = _make_backend(proc.pid, str(gguf), healthy = False)
        out = inst.load_progress()

        assert out is not None, "load_progress returned None for live pid"
        assert out["phase"] == "mmap"
        assert out["bytes_total"] == 400 * 1024 * 1024
        # The 100 MB mapping plus the interpreter's own files; the 200 MB allocation is left out.
        assert (
            out["bytes_loaded"] >= 90 * 1024 * 1024
        ), f"bytes_loaded unexpectedly low: {out['bytes_loaded']}"
        assert (
            out["bytes_loaded"] < 200 * 1024 * 1024
        ), f"the allocation counted: {out['bytes_loaded']}"
        assert 0.0 < out["fraction"] < 0.5
    finally:
        proc.terminate()
        try:
            proc.wait(timeout = 5)
        except subprocess.TimeoutExpired:
            proc.kill()


def test_live_ready_phase_when_healthy(tmp_path):
    gguf = tmp_path / "m.gguf"
    with open(gguf, "wb") as f:
        f.truncate(1 * 1024 * 1024)

    inst = _make_backend(os.getpid(), str(gguf), healthy = True)
    out = inst.load_progress()
    assert out is not None
    assert out["phase"] == "ready"
    assert out["bytes_total"] == 1 * 1024 * 1024
    # Self-pid RSS is well above 1 MiB for CPython; fraction caps at 1.
    assert out["fraction"] == 1.0


def test_live_dead_pid_returns_none(tmp_path):
    """Dead pids can linger in /proc, so use a clearly invalid id that reliably fails."""
    gguf = tmp_path / "m.gguf"
    gguf.touch()

    inst = _make_backend(9_999_999_999, str(gguf), healthy = False)
    out = inst.load_progress()
    assert out is None


def test_live_shard_aggregation_counts_real_files(tmp_path):
    """4 real sibling shards on disk: ``bytes_total`` equals their summed size to the byte."""
    shard_size = 7 * 1024 * 1024
    for i in range(1, 5):
        f = tmp_path / f"model-{i:05d}-of-00004.gguf"
        with open(f, "wb") as fh:
            fh.truncate(shard_size)
    with open(tmp_path / "config.json", "wb") as fh:
        fh.truncate(123)

    inst = _make_backend(
        os.getpid(),
        str(tmp_path / "model-00001-of-00004.gguf"),
        healthy = False,
    )
    out = inst.load_progress()
    assert out is not None
    assert out["bytes_total"] == 4 * shard_size


def test_live_repeated_polling_stays_sane(tmp_path):
    """Repeated sampling must not raise or give non-numeric output despite kernel RSS jitter."""
    gguf = tmp_path / "m.gguf"
    with open(gguf, "wb") as f:
        f.truncate(500 * 1024 * 1024)

    inst = _make_backend(os.getpid(), str(gguf), healthy = False)
    seen = []
    for _ in range(20):
        out = inst.load_progress()
        assert out is not None
        assert isinstance(out["bytes_loaded"], int)
        assert isinstance(out["bytes_total"], int)
        assert 0.0 <= out["fraction"] <= 1.0
        seen.append(out["bytes_loaded"])
        time.sleep(0.01)
    assert min(seen) > 1 * 1024 * 1024
