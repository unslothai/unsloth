# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the shared NVIDIA host-capability probe.

Pins the CUDA_VISIBLE_DEVICES handling ported from install_llama_prebuilt.py: a
GPU explicitly hidden by an index/UUID selector must report has_usable_nvidia
False (not usable), while a non-mappable selector on a physical NVIDIA host leaves
the GPU usable. This is the correctness fix that stops a hidden-GPU host from
being served a CUDA bundle it can't run.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_studio = Path(__file__).resolve().parent.parent.parent
if str(_studio) not in sys.path:
    sys.path.insert(0, str(_studio))

from backend.utils.prebuilt import hosts as H  # noqa: E402


# ── token helpers ──
def test_supports_explicit_visible_device_matching():
    assert H.supports_explicit_visible_device_matching(["0", "1"]) is True
    assert H.supports_explicit_visible_device_matching(["GPU-abc", "gpu-def"]) is True
    assert H.supports_explicit_visible_device_matching(["0", "MIG-xyz"]) is False
    assert H.supports_explicit_visible_device_matching([]) is False
    assert H.supports_explicit_visible_device_matching(None) is False


def test_select_visible_rows_matches_index_uuid_and_gpu_prefix():
    rows = [("0", "GPU-aaa", "8.9"), ("1", "GPU-bbb", "9.0")]
    assert H._select_visible_rows(rows, None) == rows  # all visible
    assert H._select_visible_rows(rows, []) == []  # none visible
    assert H._select_visible_rows(rows, ["1"]) == [("1", "GPU-bbb", "9.0")]
    # UUID with and without the 'gpu-' prefix both map.
    assert H._select_visible_rows(rows, ["GPU-aaa"]) == [("0", "GPU-aaa", "8.9")]
    assert H._select_visible_rows(rows, ["aaa"]) == [("0", "GPU-aaa", "8.9")]
    # A selector that maps to no row is skipped (not "keep all").
    assert H._select_visible_rows(rows, ["7"]) == []
    # De-dupe by index.
    assert H._select_visible_rows(rows, ["0", "GPU-aaa"]) == [("0", "GPU-aaa", "8.9")]


def _fake_nvidia(
    monkeypatch,
    *,
    rows,
    driver = "13.0",
    has_L = True,
):
    """Patch hosts.shutil.which + hosts._run to simulate an nvidia-smi host."""
    monkeypatch.setattr(H.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    csv = "\n".join(f"{i}, {u}, {c}" for i, u, c in rows)

    def fake_run(args, timeout = 20):
        if args[-1] == "-L":
            if not has_L:
                return None
            body = "\n".join(f"GPU {i}: NVIDIA ({u})" for i, u, _ in rows)
            return types.SimpleNamespace(stdout = body, stderr = "")
        if args == ["/usr/bin/nvidia-smi"]:
            return types.SimpleNamespace(stdout = f"CUDA Version: {driver}", stderr = "")
        if "--query-gpu=index,uuid,compute_cap" in args:
            return types.SimpleNamespace(stdout = csv, stderr = "")
        return None

    monkeypatch.setattr(H, "_run", fake_run)


def test_detect_nvidia_caps_all_visible(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    _fake_nvidia(monkeypatch, rows = [("0", "GPU-aaa", "10.0")])
    caps = H.detect_nvidia_caps(is_linux = True)
    assert caps.has_usable_nvidia is True
    assert caps.has_physical_nvidia is True
    assert caps.compute_caps == ["100"]
    assert caps.driver_cuda_version == (13, 0)


def test_detect_nvidia_caps_explicitly_hidden_index(monkeypatch):
    # Single GPU at index 0, but CUDA_VISIBLE_DEVICES=7 hides it: not usable.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    _fake_nvidia(monkeypatch, rows = [("0", "GPU-aaa", "8.9")])
    caps = H.detect_nvidia_caps(is_linux = True)
    assert caps.has_usable_nvidia is False
    assert caps.has_physical_nvidia is True  # the GPU physically exists
    assert caps.compute_caps == []  # nothing visible -> no caps


def test_detect_nvidia_caps_empty_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    _fake_nvidia(monkeypatch, rows = [("0", "GPU-aaa", "8.9")])
    caps = H.detect_nvidia_caps(is_linux = True)
    assert caps.has_usable_nvidia is False
    assert caps.compute_caps == []


def test_detect_nvidia_caps_non_mappable_selector_stays_usable(monkeypatch):
    # A MIG-style selector we can't enumerate: don't rule the physical GPU out.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "MIG-12345")
    _fake_nvidia(monkeypatch, rows = [("0", "GPU-aaa", "9.0")])
    caps = H.detect_nvidia_caps(is_linux = True)
    assert caps.has_usable_nvidia is True
    assert caps.has_physical_nvidia is True
    assert caps.compute_caps == []  # can't map the selector -> no caps recorded


def test_detect_nvidia_caps_selects_only_visible_gpu(monkeypatch):
    # Two GPUs, only index 1 visible: caps come from that GPU alone.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    _fake_nvidia(monkeypatch, rows = [("0", "GPU-aaa", "8.0"), ("1", "GPU-bbb", "9.0")])
    caps = H.detect_nvidia_caps(is_linux = True)
    assert caps.has_usable_nvidia is True
    assert caps.compute_caps == ["90"]


def test_detect_nvidia_caps_no_nvidia_smi(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    monkeypatch.setattr(H.shutil, "which", lambda name: None)
    # is_linux=False so the /proc fallback doesn't fire on a real NVIDIA CI host.
    caps = H.detect_nvidia_caps(is_linux = False)
    assert caps.has_usable_nvidia is False
    assert caps.has_physical_nvidia is False
    assert caps.compute_caps == []
    assert caps.driver_cuda_version is None
