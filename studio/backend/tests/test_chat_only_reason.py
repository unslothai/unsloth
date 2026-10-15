# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""detect_hardware() records why a host is chat-only; Apple Silicon without MLX gives mlx_unavailable."""

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
    monkeypatch.setattr(hw, "_NO_TORCH_SETTLED_EPOCH", None)
    # detect_hardware sets these globals directly; restore them so they do not leak.
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
    # mlx imports but the full mlx/mlx-lm/mlx-vlm stack does not; stay chat-only.
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(hw, "_has_mlx", lambda: True)
    monkeypatch.setattr(hw, "_has_usable_mlx_stack", lambda: False)
    assert hw.detect_hardware() == hw.DeviceType.CPU
    assert hw.CHAT_ONLY is True
    assert hw.CHAT_ONLY_REASON == "mlx_unavailable"


def _no_torch_apple_silicon(monkeypatch, *, mlx_on_disk: bool):
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(hw, "_installed_without_torch", lambda: True)
    monkeypatch.setattr(hw, "_mlx_distribution_installed", lambda: mlx_on_disk)
    monkeypatch.setattr(hw, "_has_usable_mlx_stack", lambda: False)


def test_apple_silicon_no_torch_install_without_mlx_is_off_by_request(monkeypatch):
    _no_torch_apple_silicon(monkeypatch, mlx_on_disk = False)
    assert hw.detect_hardware() == hw.DeviceType.CPU
    assert hw.CHAT_ONLY is True
    assert hw.CHAT_ONLY_REASON == "no_torch"
    assert hw.CHAT_ONLY_DETAIL is None


def test_apple_silicon_no_torch_install_with_mlx_on_disk_waits_for_the_probe(monkeypatch):
    # A hand-installed stack can lose the warm's import race and recover; keep polling.
    _no_torch_apple_silicon(monkeypatch, mlx_on_disk = True)
    hw.detect_hardware()
    assert hw.CHAT_ONLY is True
    assert hw.CHAT_ONLY_REASON == "mlx_unavailable"
    assert hw.verdict_blames_the_mlx_stack() is True


def test_the_probe_settles_a_no_torch_host_once_the_stack_measures_unusable(monkeypatch):
    _no_torch_apple_silicon(monkeypatch, mlx_on_disk = True)
    hw.detect_hardware()
    assert hw.settle_the_no_torch_verdict(hw.current_detection_epoch()) is True
    assert hw.CHAT_ONLY_REASON == "no_torch"
    assert hw.CHAT_ONLY_DETAIL is None
    assert hw.verdict_blames_the_mlx_stack() is False
    hw.detect_hardware()
    assert hw.CHAT_ONLY_REASON == "no_torch"


def test_a_probe_retired_by_a_shutdown_settles_nothing(monkeypatch):
    _no_torch_apple_silicon(monkeypatch, mlx_on_disk = True)
    hw.detect_hardware()
    assert hw.settle_the_no_torch_verdict(hw.current_detection_epoch() - 1) is False
    assert hw.CHAT_ONLY_REASON == "mlx_unavailable"
    assert hw._NO_TORCH_SETTLED_EPOCH is None


def test_a_settled_flag_does_not_outlive_its_lifespan(monkeypatch):
    _no_torch_apple_silicon(monkeypatch, mlx_on_disk = True)
    monkeypatch.setattr(hw, "_NO_TORCH_SETTLED_EPOCH", hw.current_detection_epoch() - 1)
    hw.detect_hardware()
    assert hw.CHAT_ONLY_REASON == "mlx_unavailable"


def test_settling_leaves_any_other_verdict_alone(monkeypatch):
    for chat_only, reason in ((False, None), (True, "intel_mac"), (True, "no_torch")):
        hw.CHAT_ONLY, hw.CHAT_ONLY_REASON = chat_only, reason
        assert hw.settle_the_no_torch_verdict(hw.current_detection_epoch()) is False
        assert (hw.CHAT_ONLY, hw.CHAT_ONLY_REASON) == (chat_only, reason)


def test_a_declined_settle_does_not_mark_the_epoch(monkeypatch):
    # A declined settle must not record the epoch, or a later failure reads as settled.
    _no_torch_apple_silicon(monkeypatch, mlx_on_disk = True)
    hw.CHAT_ONLY, hw.CHAT_ONLY_REASON = False, None
    assert hw.settle_the_no_torch_verdict(hw.current_detection_epoch()) is False
    assert hw._NO_TORCH_SETTLED_EPOCH != hw.current_detection_epoch()

    hw.detect_hardware()
    assert hw.CHAT_ONLY_REASON == "mlx_unavailable"


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
