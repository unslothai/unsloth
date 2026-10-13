# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Linux NPU detection over a fake /sys tree.

The values are the ones the AMD DevLab Strix Halo runner reported (amdxdna 2.25 DKMS,
firmware 1.1.2.65)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from utils.hardware import npu


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
    info = npu.detect_amd_npu()
    assert info["present"] is True
    assert info["supported"] is supported


def test_detection_never_raises(monkeypatch):
    def broken():
        raise PermissionError("sysfs")

    monkeypatch.setattr(npu.sys, "platform", "linux")
    monkeypatch.setattr(npu, "_linux_probe", broken)
    assert npu.detect_amd_npu() == {"present": False, "error": "sysfs", "supported": False}
