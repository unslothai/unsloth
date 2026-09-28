# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An NVIDIA + AMD host launched with CUDA_VISIBLE_DEVICES="" must say why ROCm torch sees nothing.

That mask is how the installer is steered to ROCm torch on a mixed host, and HIP reads it too,
so the AMD card vanishes at runtime and the log said only "CPU training backend". The mask
stays deliberate for the System page (#9858): on an AMD-only host it is, and "repair the
installation" is the wrong advice on either. Inventory and torch are faked; nothing here
reads the runner's hardware.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

import utils.hardware.hardware as hw

_NVIDIA = {"vendor": "nvidia", "name": "NVIDIA GeForce RTX 3080"}
_AMD = {"vendor": "amd", "name": "AMD Radeon AI PRO R9700", "gfx_candidates": ["gfx1201"]}
_HINT = "hides the AMD GPU from ROCm torch"


def _fake_torch(flavor: str):
    torch = types.ModuleType("torch")
    if flavor == "rocm":
        torch.version = SimpleNamespace(hip = "7.1.52802", cuda = None)
        torch.__version__ = "2.9.1+rocm7.1"
    else:
        torch.version = SimpleNamespace(hip = None, cuda = "12.8")
        torch.__version__ = "2.9.1+cu128"
    torch.cuda = SimpleNamespace(is_available = lambda: False, device_count = lambda: 0)
    return torch


class _Recorder:
    def __init__(self):
        self.warnings: list[str] = []

    def warning(self, msg, *args, **_kw):
        self.warnings.append(msg % args if args else msg)

    def info(self, *_a, **_kw):
        pass

    debug = error = exception = info


@pytest.fixture(autouse = True)
def _pinned_host(monkeypatch, tmp_path):
    for var in (
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "ZE_AFFINITY_MASK",
        "UNSLOTH_FORCE_XPU",
        "UNSLOTH_TORCH_INDEX_URL",
        "UNSLOTH_TORCH_INDEX_FAMILY",
    ):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setattr(hw, "TORCH_IMPORT_ERROR", None)
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    # No install manifest to read a recorded flavor from.
    monkeypatch.setattr(hw.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(hw, "_torch_build_snapshot_cache", None)
    monkeypatch.setattr(hw, "_physical_gpu_inventory_cache", None)
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("rocm"))


def _inventory(monkeypatch, devices):
    monkeypatch.setattr(
        hw,
        "get_physical_gpu_inventory",
        lambda **_kw: {"available": bool(devices), "devices": devices, "unknown": False},
    )


def _detect(monkeypatch) -> _Recorder:
    recorder = _Recorder()
    monkeypatch.setattr(hw, "logger", recorder)
    for name, value in (
        ("DEVICE", None),
        ("CHAT_ONLY", True),
        ("CHAT_ONLY_REASON", None),
        ("CHAT_ONLY_DETAIL", None),
    ):
        monkeypatch.setattr(hw, name, value)
    with hw._DETECT_LOCK:
        assert hw._detect_hardware_locked() == hw.DeviceType.CPU
    return recorder


def _hinted(recorder: _Recorder) -> bool:
    return any(_HINT in line for line in recorder.warnings)


@pytest.mark.parametrize("mask", ["", " ", "-1"])
def test_the_reporters_mixed_host_is_told_why(monkeypatch, mask):
    _inventory(monkeypatch, [_NVIDIA, _AMD])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)

    recorder = _detect(monkeypatch)

    assert _hinted(recorder)
    line = next(entry for entry in recorder.warnings if _HINT in entry)
    assert "HIP_VISIBLE_DEVICES" in line
    assert "UNSLOTH_FORCE_ROCM_TORCH=1" in line
    # Still a deliberate mask for #9858: no repair offered, no mismatch published.
    assert hw.CHAT_ONLY_REASON == "no_gpu"
    assert hw._torch_gpu_mismatch_report() == {}


def test_an_amd_only_host_with_the_same_mask_stays_deliberate(monkeypatch):
    _inventory(monkeypatch, [_AMD])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    recorder = _detect(monkeypatch)

    assert not _hinted(recorder)
    assert hw.CHAT_ONLY_REASON == "no_gpu"


def test_hip_visible_devices_overrides_the_mask_so_no_hint(monkeypatch):
    _inventory(monkeypatch, [_NVIDIA, _AMD])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")

    assert not _hinted(_detect(monkeypatch))
    assert hw._emptied_cuda_mask_hides_amd_on_a_mixed_host() is False


def test_a_cuda_wheel_on_the_mixed_host_gets_no_rocm_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch("cuda"))
    _inventory(monkeypatch, [_NVIDIA, _AMD])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    assert hw._emptied_cuda_mask_hides_amd_on_a_mixed_host() is False


@pytest.mark.parametrize("mask", [None, "0"])
def test_no_hint_without_an_emptied_mask(monkeypatch, mask):
    _inventory(monkeypatch, [_NVIDIA, _AMD])
    if mask is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)

    assert hw._emptied_cuda_mask_hides_amd_on_a_mixed_host() is False


def test_an_inventory_that_raises_gives_no_hint(monkeypatch):
    def _boom(**_kw):
        raise RuntimeError("probe failed")

    monkeypatch.setattr(hw, "get_physical_gpu_inventory", _boom)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    assert hw._emptied_cuda_mask_hides_amd_on_a_mixed_host() is False
