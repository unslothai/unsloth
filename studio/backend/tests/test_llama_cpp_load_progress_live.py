# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Live, no-mock integration test for ``LlamaCppBackend.load_progress()``.

The companion files (``test_llama_cpp_load_progress.py`` and
``test_llama_cpp_load_progress_matrix.py``) patch ``builtins.open`` to
feed synthetic VmRSS values. This file is the opposite: it uses **real**
subprocesses, **real** file sizes, and the **real** ``/proc``
interface. It is the sanity check that the contract we keep in the
mocked tests still maps to what the kernel actually returns on a live
Linux system.

Why both: the mocked tests can be fooled by a buggy implementation that
parses ``/proc`` output in a format the kernel no longer uses, or that
makes assumptions about ``Path.stat()`` vs ``os.path.getsize``. This
file hits the real APIs so any format drift gets caught.

Skipped cleanly on non-Linux (no ``/proc``).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import types as _types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Same stubs as the matrix file (keep self-contained so the file can be
# run standalone as well as via the full suite).
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
sys.modules.setdefault("httpx", _httpx_stub)

from core.inference.llama_cpp import LlamaCppBackend


pytestmark = pytest.mark.skipif(
    not Path("/proc").exists(),
    reason = "live /proc test is Linux-only",
)


def _make_backend(pid: int, gguf_path: str, healthy: bool = False):
    inst = LlamaCppBackend.__new__(LlamaCppBackend)
    inst._process = type("P", (), {"pid": pid})()
    inst._gguf_path = gguf_path
    inst._healthy = healthy
    return inst


def test_live_rss_matches_kernel_vmrss(tmp_path):
    """Spawn a real child, let it allocate real bytes, confirm
    ``bytes_loaded`` tracks the kernel's VmRSS within a sane tolerance."""
    # Child that allocates ~100 MB of zero'd bytes and then idles.
    script = tmp_path / "burn.py"
    script.write_text(
        "import time, sys\n"
        "buf = bytearray(100 * 1024 * 1024)\n"  # 100 MB
        "# touch every page so RSS actually grows\n"
        "for i in range(0, len(buf), 4096):\n"
        "    buf[i] = 1\n"
        "sys.stdout.write('ready\\n')\n"
        "sys.stdout.flush()\n"
        "time.sleep(10)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
    )
    try:
        # Wait for the child to finish touching pages.
        ready = proc.stdout.readline()
        assert ready.strip() == b"ready"

        # Create a fake 200 MB sparse gguf so bytes_total is concrete.
        gguf = tmp_path / "model.gguf"
        with open(gguf, "wb") as f:
            f.truncate(200 * 1024 * 1024)

        inst = _make_backend(proc.pid, str(gguf), healthy = False)
        out = inst.load_progress()

        assert out is not None, "load_progress returned None for live pid"
        assert out["phase"] == "mmap"
        assert out["bytes_total"] == 200 * 1024 * 1024
        # VmRSS for the Python child includes the interpreter + the 100MB
        # buffer, so a realistic floor is 50 MB and ceiling is 200 MB.
        assert (
            out["bytes_loaded"] >= 50 * 1024 * 1024
        ), f"bytes_loaded unexpectedly low: {out['bytes_loaded']}"
        assert out["bytes_loaded"] <= 200 * 1024 * 1024
        assert 0.0 < out["fraction"] <= 1.0
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
    """A recently-dead pid may linger in /proc for ms; use a clearly
    invalid id so the read reliably fails."""
    gguf = tmp_path / "m.gguf"
    gguf.touch()

    inst = _make_backend(9_999_999_999, str(gguf), healthy = False)
    out = inst.load_progress()
    assert out is None


def test_live_shard_aggregation_counts_real_files(tmp_path):
    """With 4 real sibling shards on disk, ``bytes_total`` equals their
    summed size to the byte."""
    shard_size = 7 * 1024 * 1024  # 7 MB each
    for i in range(1, 5):
        f = tmp_path / f"model-{i:05d}-of-00004.gguf"
        with open(f, "wb") as fh:
            fh.truncate(shard_size)
    # Unrelated file in same dir -- must not be counted.
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
    """Sampling the same backend 20 times should not raise or produce
    non-numeric output, even under normal kernel RSS jitter."""
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
    # RSS of a healthy Python process doesn't go below ~5 MB.
    assert min(seen) > 1 * 1024 * 1024
