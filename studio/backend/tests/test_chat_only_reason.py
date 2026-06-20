# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""detect_hardware() records WHY a host is chat-only so the UI can explain the
greyed-out Train/Export instead of disabling them silently.

The key case is Apple Silicon without an importable MLX -> "mlx_unavailable",
which is the usual cause of "Train and Export greyed out" on Macs after a
reinstall/update dropped MLX.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.hardware.hardware as hw  # noqa: E402


@pytest.fixture(autouse = True)
def _no_torch(monkeypatch):
    # Force the non-CUDA/XPU path regardless of the test host's real GPUs.
    monkeypatch.setattr(hw, "_has_torch", lambda: False)


def test_apple_silicon_without_mlx_is_chat_only_with_reason(monkeypatch):
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(hw, "_has_mlx", lambda: False)
    hw.detect_hardware()
    assert hw.CHAT_ONLY is True
    assert hw.CHAT_ONLY_REASON == "mlx_unavailable"


def test_apple_silicon_with_mlx_enables_training(monkeypatch):
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(hw, "_has_mlx", lambda: True)
    hw.detect_hardware()
    assert hw.CHAT_ONLY is False
    assert hw.CHAT_ONLY_REASON is None


def test_intel_mac_reason(monkeypatch):
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(hw, "_has_mlx", lambda: False)
    monkeypatch.setattr(hw.platform, "system", lambda: "Darwin")
    hw.detect_hardware()
    assert hw.CHAT_ONLY is True
    assert hw.CHAT_ONLY_REASON == "intel_mac"


def test_cpu_only_non_mac_reason(monkeypatch):
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(hw, "_has_mlx", lambda: False)
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    hw.detect_hardware()
    assert hw.CHAT_ONLY is True
    assert hw.CHAT_ONLY_REASON == "no_gpu"
