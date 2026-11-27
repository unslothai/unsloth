# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Linux NPU detection over a fake /sys tree, with real amdxdna 2.25 and firmware 1.1.2.65 values."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from utils.hardware import npu
from utils.npu_runtime_config import FLM_BINARY_ENV, LEMONADE_ARCHIVE_ENV


def _device(
    root: Path,
    name: str,
    vendor: str,
    device: str,
    *,
    driver: str | None = None,
) -> Path:
    # Windows forbids ":" in names; the probe never reads the directory name.
    path = root / "devices" / (name.replace(":", "_") if os.name == "nt" else name)
    path.mkdir(parents = True)
    (path / "vendor").write_text(f"0x{vendor}\n")
    (path / "device").write_text(f"0x{device}\n")
    if driver:
        target = root / "drivers" / driver
        target.mkdir(parents = True, exist_ok = True)
        (path / "driver").symlink_to(target)
    return path


def test_strix_halo_npu_with_driver(tmp_path):
    _device(tmp_path, "0000:c3:00.0", "1002", "1586", driver = "amdgpu")
    npu_dir = _device(tmp_path, "0000:c4:00.1", "1022", "17f0", driver = "amdxdna")
    (npu_dir / "fw_version").write_text("1.1.2.65\n")
    (npu_dir / "vbnv").write_text("NPU Strix Halo\n")
    (npu_dir / "accel" / "accel0").mkdir(parents = True)
    (tmp_path / "module" / "amdxdna").mkdir(parents = True)
    (tmp_path / "module" / "amdxdna" / "version").write_text("2.25.260102\n")

    assert npu._linux_probe(tmp_path / "devices", tmp_path / "module") == {
        "present": True,
        "family": "XDNA2",
        "name": "NPU Strix Halo",
        "driver": "amdxdna",
        "driver_version": "2.25.260102",
        "firmware_version": "1.1.2.65",
        "device_node": "/dev/accel/accel0",
    }


def test_an_npu_without_a_driver_is_still_found(tmp_path):
    _device(tmp_path, "0000:c4:00.1", "1022", "17f0")
    info = npu._linux_probe(tmp_path / "devices", tmp_path / "module")
    assert info["present"] is True
    assert info["driver"] is None
    assert info["device_node"] is None


def test_no_npu(tmp_path):
    _device(tmp_path, "0000:c3:00.0", "1002", "1586", driver = "amdgpu")
    assert npu._linux_probe(tmp_path / "devices") == {"present": False}
    assert npu._linux_probe(tmp_path / "missing") == {"present": False}


@pytest.mark.parametrize("device, supported", [("17f0", True), ("1502", False)])
def test_only_xdna2_is_supported(tmp_path, monkeypatch, device, supported):
    _device(tmp_path, "0000:c4:00.1", "1022", device, driver = "amdxdna")
    probe = npu._linux_probe
    monkeypatch.setattr(npu.sys, "platform", "linux")
    monkeypatch.setattr(
        npu, "_linux_probe", lambda: probe(tmp_path / "devices", tmp_path / "module")
    )
    monkeypatch.setenv(npu.ENABLE_ENV, "1")
    info = npu.detect_amd_npu()
    assert info["present"] is True
    assert info["supported"] is supported


@pytest.mark.parametrize(
    "value, enabled",
    [(None, True), ("", True), ("1", True), ("0", False), ("false", False), (" Off ", False)],
)
def test_an_xdna2_npu_is_on_unless_the_flag_turns_it_off(tmp_path, monkeypatch, value, enabled):
    _device(tmp_path, "0000:c4:00.1", "1022", "17f0", driver = "amdxdna")
    probe = npu._linux_probe
    monkeypatch.setattr(npu.sys, "platform", "linux")
    monkeypatch.setattr(
        npu, "_linux_probe", lambda: probe(tmp_path / "devices", tmp_path / "module")
    )
    if value is None:
        monkeypatch.delenv(npu.ENABLE_ENV, raising = False)
    else:
        monkeypatch.setenv(npu.ENABLE_ENV, value)
    info = npu.detect_amd_npu()
    assert info["present"] is True and info["family"] == "XDNA2"
    assert info["enabled"] is enabled and info["supported"] is enabled


def test_detection_never_raises(monkeypatch):
    def broken():
        raise PermissionError("sysfs")

    monkeypatch.setattr(npu.sys, "platform", "linux")
    monkeypatch.setattr(npu, "_linux_probe", broken)
    monkeypatch.delenv(npu.ENABLE_ENV, raising = False)
    assert npu.detect_amd_npu() == {
        "present": False,
        "error": "sysfs",
        "enabled": True,
        "supported": False,
    }


def test_custom_runtime_allows_probe_and_respects_disable_flag(monkeypatch):
    monkeypatch.setattr(npu.sys, "platform", "win32")
    monkeypatch.setattr(npu, "_windows_probe", lambda: {"present": False})
    monkeypatch.setenv(FLM_BINARY_ENV, "custom-flm.exe")
    monkeypatch.setenv(npu.ENABLE_ENV, "1")

    assert npu.detect_amd_npu() == {"present": False, "enabled": True, "supported": True}

    monkeypatch.setenv(npu.ENABLE_ENV, "0")
    assert npu.detect_amd_npu() == {"present": False, "enabled": False, "supported": False}


def test_supported_windows_npu_needs_no_runtime_override(monkeypatch):
    monkeypatch.setattr(npu.sys, "platform", "win32")
    monkeypatch.setattr(npu, "_windows_probe", lambda: {"present": True, "family": "XDNA2"})
    monkeypatch.delenv(FLM_BINARY_ENV, raising = False)
    monkeypatch.delenv(LEMONADE_ARCHIVE_ENV, raising = False)
    monkeypatch.setenv(npu.ENABLE_ENV, "1")

    assert npu.detect_amd_npu()["supported"]
