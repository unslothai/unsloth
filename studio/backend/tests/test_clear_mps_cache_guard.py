# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Clearing the MPS cache must not crash an Apple Silicon host that has no usable Metal device."""

from __future__ import annotations

import platform
import subprocess
import sys
import types
from pathlib import Path

import pytest

from utils.hardware import hardware as hw

BACKEND_ROOT = Path(__file__).resolve().parents[1]


def _fake_torch(monkeypatch, available):
    calls = []
    torch = types.ModuleType("torch")
    torch.mps = types.SimpleNamespace(empty_cache = lambda: calls.append("empty_cache"))
    torch.backends = types.SimpleNamespace(
        mps = types.SimpleNamespace(is_available = lambda: available)
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return calls


def test_the_mps_cache_is_left_alone_without_a_metal_device(monkeypatch):
    calls = _fake_torch(monkeypatch, available = False)
    hw._clear_mps_cache()
    assert calls == []


def test_the_mps_cache_is_still_cleared_with_a_metal_device(monkeypatch):
    calls = _fake_torch(monkeypatch, available = True)
    hw._clear_mps_cache()
    assert calls == ["empty_cache"]


@pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64", reason = "Apple Silicon only"
)
def test_clearing_the_mps_cache_does_not_crash_this_host():
    """A subprocess, so a segfault in torch.mps reads as a failure rather than killing the run."""
    pytest.importorskip("torch")
    code = (
        "import sys, torch\n"
        f"sys.path.insert(0, {str(BACKEND_ROOT)!r})\n"
        "print('mps available', torch.backends.mps.is_available(), 'built', torch.backends.mps.is_built(), flush = True)\n"
        "from utils.hardware import hardware\n"
        "hardware._clear_mps_cache()\n"
        "print('cleared', flush = True)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output = True, text = True, timeout = 300)
    assert (
        proc.returncode == 0 and "cleared" in proc.stdout
    ), f"exit {proc.returncode}\nstdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
