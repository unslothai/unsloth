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

import sys
import types

import pytest

from core.inference.llama_cpp import LlamaCppBackend


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
    monkeypatch.setattr(LlamaCppBackend, "_read_rss_bytes", staticmethod(lambda pid: 800))
    be = _backend(_gguf(tmp_path, 10000), healthy = True)
    p = be.load_progress()
    assert p["phase"] == "ready"
    assert p["fraction"] == 1.0  # not 0.08
    assert p["bytes_loaded"] == p["bytes_total"] == 10000


def test_mmap_phase_reports_raw_rss_fraction(tmp_path, monkeypatch):
    # Still loading: the bar should track real residency, not jump to 1.0.
    monkeypatch.setattr(LlamaCppBackend, "_read_rss_bytes", staticmethod(lambda pid: 800))
    be = _backend(_gguf(tmp_path, 10000), healthy = False)
    p = be.load_progress()
    assert p["phase"] == "mmap"
    assert p["fraction"] == 0.08
    assert p["bytes_loaded"] == 800
    assert p["bytes_total"] == 10000


def test_ready_without_shard_size_still_completes(tmp_path, monkeypatch):
    # bytes_total unknown (file unstattable): fraction must still read complete.
    monkeypatch.setattr(LlamaCppBackend, "_read_rss_bytes", staticmethod(lambda pid: 800))
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
    monkeypatch.setattr(LlamaCppBackend, "_read_rss_bytes", staticmethod(lambda pid: None))
    be = _backend(_gguf(tmp_path, 10000), healthy = False)
    assert be.load_progress() is None


def test_read_rss_bytes_absent_pid_is_none():
    # A pid with no readable /proc entry (or no /proc at all) yields None, never
    # raises.
    assert LlamaCppBackend._read_rss_bytes(2**31 - 1) is None


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason = "/proc is Linux-only")
def test_read_rss_bytes_reads_self_on_linux():
    rss = LlamaCppBackend._read_rss_bytes(__import__("os").getpid())
    assert isinstance(rss, int) and rss > 0
