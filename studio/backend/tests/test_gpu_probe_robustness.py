# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A working GPU survives a failing probe or a missing name (XPU, Linux Arc, WSL nvidia-smi); all faked."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

import utils.hardware.hardware as hw
from utils.hardware import nvidia


def _boom(*_a, **_k):
    raise RuntimeError("probe failed")


def _fake_torch(
    *,
    cuda = False,
    xpu = False,
    cuda_raises = False,
    xpu_raises = False,
    xpu_name_raises = False,
):
    torch = types.ModuleType("torch")
    torch.__version__ = "2.11.0+xpu" if xpu else "2.11.0+cu128"
    torch.version = SimpleNamespace(hip = None, cuda = "12.8", xpu = None)
    torch.cuda = SimpleNamespace(
        is_available = _boom if cuda_raises else (lambda: cuda),
        device_count = lambda: 1 if cuda else 0,
        get_device_properties = lambda _i: SimpleNamespace(name = "NVIDIA RTX A4000"),
    )
    torch.xpu = SimpleNamespace(
        is_available = _boom if xpu_raises else (lambda: xpu),
        get_device_name = _boom if xpu_name_raises else (lambda _i: "Intel(R) Arc(TM) B580"),
    )
    return torch


def _detect(
    monkeypatch,
    torch,
    *,
    force_xpu = False,
):
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(hw, "TORCH_IMPORT_ERROR", None)
    monkeypatch.setattr(hw, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(hw, "_mismatch_verdict_for_this_host", lambda *_a: (None, None))
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    for var in ("UNSLOTH_FORCE_XPU", "ZE_AFFINITY_MASK", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising = False)
    if force_xpu:
        monkeypatch.setenv("UNSLOTH_FORCE_XPU", "1")
    for name, value in (
        ("DEVICE", None),
        ("CHAT_ONLY", True),
        ("CHAT_ONLY_REASON", None),
        ("CHAT_ONLY_DETAIL", None),
    ):
        monkeypatch.setattr(hw, name, value)
    with hw._DETECT_LOCK:
        return hw._detect_hardware_locked()


def test_a_raising_cuda_probe_falls_through_to_a_working_xpu(monkeypatch):
    assert _detect(monkeypatch, _fake_torch(cuda_raises = True, xpu = True)) == hw.DeviceType.XPU
    assert hw.CHAT_ONLY is False


def test_an_xpu_whose_name_cannot_be_read_is_still_an_xpu(monkeypatch):
    assert _detect(monkeypatch, _fake_torch(xpu = True, xpu_name_raises = True)) == hw.DeviceType.XPU
    assert hw.CHAT_ONLY is False


def test_a_forced_xpu_whose_name_cannot_be_read_is_still_an_xpu(monkeypatch):
    torch = _fake_torch(xpu = True, xpu_name_raises = True)
    assert _detect(monkeypatch, torch, force_xpu = True) == hw.DeviceType.XPU


def test_a_raising_xpu_probe_is_a_measured_cpu_host_not_a_detection_failure(monkeypatch):
    assert _detect(monkeypatch, _fake_torch(xpu_raises = True)) == hw.DeviceType.CPU
    assert hw.CHAT_ONLY_REASON == "no_gpu"


def test_the_winner_is_unchanged_when_every_probe_succeeds(monkeypatch):
    assert _detect(monkeypatch, _fake_torch(cuda = True, xpu = True)) == hw.DeviceType.CUDA
    assert _detect(monkeypatch, _fake_torch(xpu = True)) == hw.DeviceType.XPU


@pytest.mark.parametrize(
    ("device_id", "xpu_class"),
    [
        ("0x56a0", True),  # Arc A770 (DG2)
        ("0x56c0", True),  # Data Center GPU Flex 170 (ATS-M)
        ("0x0bd5", True),  # Data Center GPU Max 1550 (PVC)
        ("0xe20b", True),  # Arc B580 (BMG)
        ("0x4905", False),  # Iris Xe MAX (DG1): discrete, but no XPU wheel supports it
        ("0x46a6", False),  # Alder Lake iGPU
        ("garbage", None),
    ],
)
def test_the_pci_device_id_tells_an_xpu_card_from_other_intel_graphics(tmp_path, device_id, xpu_class):
    (tmp_path / "device").write_text(device_id + "\n", encoding = "utf-8")
    assert hw._intel_pci_device_is_xpu_class(str(tmp_path)) is xpu_class


def test_an_unreadable_pci_device_id_is_unknown(tmp_path):
    assert hw._intel_pci_device_is_xpu_class(str(tmp_path / "missing")) is None


def test_an_xpu_class_intel_record_establishes_a_mismatch(monkeypatch):
    monkeypatch.setattr(hw, "_expected_xpu_flavor_was_chosen", lambda: False)
    monkeypatch.setattr(hw, "_torch_reports_an_xpu_runtime", lambda: False)
    monkeypatch.setattr(hw, "_vendors_masked_off", lambda: set())
    arc = [{"vendor": "intel", "name": None, "index": 0, "xpu_class": True}]
    assert hw._devices_that_can_establish_a_mismatch(arc) == arc
    # Controls: an iGPU or DG1 host's correct CPU install must not be flagged.
    for xpu_class in (False, None):
        igpu = [{"vendor": "intel", "name": None, "index": 0, "xpu_class": xpu_class}]
        assert hw._devices_that_can_establish_a_mismatch(igpu) == []


@pytest.mark.parametrize(
    ("system", "vendors", "pinned"),
    [
        ("Linux", {"intel"}, True),
        ("Windows", {"intel"}, False),  # install.ps1 autodetects Arc, so Repair works there
        ("Linux", {"nvidia"}, False),
        ("Linux", {"intel", "nvidia"}, False),
    ],
)
def test_a_linux_intel_host_is_told_the_pin_instead_of_a_repair_that_reinstalls_cpu(
    monkeypatch, system, vendors, pinned
):
    monkeypatch.setattr(hw.platform, "system", lambda: system)
    monkeypatch.setattr(hw, "CHAT_ONLY_MISMATCH_VENDORS", frozenset(vendors))
    message = hw._gpu_present_but_unusable_message("training", ("torch_cpu_build", "2.11.0+cpu"))
    assert ("UNSLOTH_TORCH_INDEX_FAMILY=xpu" in message) is pinned
    assert ("Repair installation" in message) is not pinned


@pytest.mark.parametrize("present", [True, False])
def test_wsl_nvidia_smi_is_found_off_path(monkeypatch, present):
    monkeypatch.setattr(nvidia.platform, "system", lambda: "Linux")
    monkeypatch.setattr(nvidia.shutil, "which", lambda _name: None)
    monkeypatch.setattr(nvidia.os.path, "isfile", lambda p: present and p == nvidia._WSL_NVIDIA_SMI)
    expected = nvidia._WSL_NVIDIA_SMI if present else "nvidia-smi"
    assert nvidia._nvidia_smi_executable() == expected


def test_path_still_wins_over_the_wsl_location(monkeypatch):
    monkeypatch.setattr(nvidia.platform, "system", lambda: "Linux")
    monkeypatch.setattr(nvidia.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(nvidia.os.path, "isfile", lambda _p: True)
    assert nvidia._nvidia_smi_executable() == "/usr/bin/nvidia-smi"
