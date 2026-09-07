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
    # The no_torch verdict reads the venv's install manifest; pin it so the answer does
    # not depend on which venv runs these tests.
    monkeypatch.setattr(hw, "_installed_without_torch", lambda: False)
    # detect_hardware() assigns these module globals directly (not via monkeypatch),
    # so save and restore them; otherwise a chat-only verdict here leaks into other
    # backend tests (e.g. test_utils.py) when they share a process on a GPU host.
    saved = (hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON, hw.IS_ROCM)
    try:
        yield
    finally:
        hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON, hw.IS_ROCM = saved


def test_apple_silicon_without_mlx_is_chat_only_with_reason(monkeypatch):
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(hw, "_has_usable_mlx_stack", lambda: False)
    hw.detect_hardware()
    assert hw.CHAT_ONLY is True
    assert hw.CHAT_ONLY_REASON == "mlx_unavailable"


def test_apple_silicon_with_mlx_enables_training(monkeypatch):
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(hw, "_has_usable_mlx_stack", lambda: True)
    hw.detect_hardware()
    assert hw.CHAT_ONLY is False
    assert hw.CHAT_ONLY_REASON is None


def test_apple_silicon_with_incomplete_mlx_stack_stays_chat_only(monkeypatch):
    # Bare `import mlx.core` works but the full mlx/mlx-lm/mlx-vlm stack does not
    # (e.g. a backtracked/old mlx-vlm). The training gate must match the self-heal
    # validator and stay chat-only so the UI does not enable a broken Train/Export.
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(hw, "_has_mlx", lambda: True)
    monkeypatch.setattr(hw, "_has_usable_mlx_stack", lambda: False)
    assert hw.detect_hardware() == hw.DeviceType.CPU
    assert hw.CHAT_ONLY is True
    assert hw.CHAT_ONLY_REASON == "mlx_unavailable"


def test_apple_silicon_no_torch_install_without_usable_mlx_is_off_by_request(monkeypatch):
    # GGUF-only by request: not a broken stack, so the UI must not send the user to
    # `unsloth studio update`, which keeps no-torch and cannot enable Train. The same
    # verdict whether mlx is absent or present but partial: the self-heal declines both.
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(hw, "_installed_without_torch", lambda: True)
    monkeypatch.setattr(hw, "_has_usable_mlx_stack", lambda: False)
    for mlx_on_disk in (False, True):
        monkeypatch.setattr(hw, "_has_mlx", lambda v = mlx_on_disk: v)
        assert hw.detect_hardware() == hw.DeviceType.CPU
        assert hw.CHAT_ONLY is True
        assert hw.CHAT_ONLY_REASON == "no_torch"
        assert hw.CHAT_ONLY_DETAIL is None


def test_a_no_torch_verdict_is_still_overturned_by_a_usable_stack(monkeypatch):
    # The warm's first stage can lose the import race on a healthy hand-installed stack
    # (#9120); the post-warm probe must be able to overturn no_torch like mlx_unavailable.
    hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON = hw.DeviceType.CPU, True, "no_torch"
    assert hw.verdict_blames_the_mlx_stack() is True

    def _usable_after_the_warm():
        hw.DEVICE, hw.CHAT_ONLY, hw.CHAT_ONLY_REASON = hw.DeviceType.MLX, False, None
        return hw.DEVICE

    monkeypatch.setattr(hw, "_detect_hardware_locked", _usable_after_the_warm)
    was_complete = hw.DETECTION_COMPLETE.is_set()
    hw.DETECTION_COMPLETE.set()
    try:
        assert hw.overturn_the_mlx_verdict(hw.current_detection_epoch()) is True
    finally:
        hw.DETECTION_COMPLETE.set() if was_complete else hw.DETECTION_COMPLETE.clear()
    assert hw.CHAT_ONLY is False


def test_apple_silicon_no_torch_install_with_usable_mlx_enables_training(monkeypatch):
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(hw, "_installed_without_torch", lambda: True)
    monkeypatch.setattr(hw, "_has_usable_mlx_stack", lambda: True)
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
