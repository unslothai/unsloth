# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A GPU that works must not be lost to a probe that fails, or to a name the kernel omits.

Three shapes, each measured against main before the fix:
  * one vendor's probe raising turned the whole pass into CPU + detection_failed, taking a
    working device of another vendor (or the device whose NAME could not be read) with it;
  * a Linux Intel Arc on a CPU wheel read "no_gpu", because sysfs publishes no Intel name
    and the mismatch check keyed on the name, so no repair was ever offered;
  * a WSL NVIDIA host whose nvidia-smi sits in /usr/lib/wsl/lib, off PATH, read "no_gpu"
    for the same reason: WSL has no /proc/driver/nvidia to fall back on.

No Arc, XPU or WSL hardware exists here, so torch, sysfs and the filesystem are faked in
the shapes those hosts present.
"""

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


# -- one probe failing must not take a working device with it ----------------------------


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
    """The control: CUDA still outranks XPU on a host where both answer."""
    assert _detect(monkeypatch, _fake_torch(cuda = True, xpu = True)) == hw.DeviceType.CUDA
    assert _detect(monkeypatch, _fake_torch(xpu = True)) == hw.DeviceType.XPU


# -- a nameless Linux Intel record: discrete by PCI address, not by name -----------------


@pytest.mark.parametrize(
    ("address", "discrete"),
    [
        ("0000:03:00.0", True),  # an Arc behind a PCIe root port
        ("0000:00:02.0", False),  # Intel integrated graphics, a root-complex endpoint
        ("not-a-pci-address", None),
    ],
)
def test_the_pci_address_tells_a_discrete_card_from_an_igpu(tmp_path, address, discrete):
    target = tmp_path / "pci" / address
    target.mkdir(parents = True)
    link = tmp_path / "card0-device"
    link.symlink_to(target)
    assert hw._pci_function_is_behind_a_port(str(link)) is discrete


def test_a_discrete_intel_record_establishes_a_mismatch(monkeypatch):
    monkeypatch.setattr(hw, "_expected_xpu_flavor_was_chosen", lambda: False)
    monkeypatch.setattr(hw, "_torch_reports_an_xpu_runtime", lambda: False)
    monkeypatch.setattr(hw, "_vendors_masked_off", lambda: set())
    arc = [{"vendor": "intel", "name": None, "index": 0, "discrete": True}]
    assert hw._devices_that_can_establish_a_mismatch(arc) == arc
    # The controls: an integrated or unreadable one still does not, which is what keeps an
    # ordinary UHD laptop from being told its correct CPU install is broken.
    for discrete in (False, None):
        igpu = [{"vendor": "intel", "name": None, "index": 0, "discrete": discrete}]
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
    # Repair and a plain re-run reinstall the CPU build there, so they must not be offered.
    assert ("Repair installation" in message) is not pinned


# -- WSL keeps nvidia-smi off PATH -------------------------------------------------------


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
