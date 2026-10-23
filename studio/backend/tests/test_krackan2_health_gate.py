# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A lone Linux Krackan Point 2 (gfx1153) gets CPU torch on purpose, so the health gate must not
read it as a gfx1152 the installers would have served and prompt for a repair."""

from __future__ import annotations

import pytest

import utils.hardware.hardware as hw
import utils.hardware.nvidia as nvidia


def _linux_inventory(
    monkeypatch,
    tmp_path,
    *,
    name = None,
    kfd = ("110503",),
    rocm = (),
):
    """One AMD sysfs record through the real inventory refresh, KFD nodes from a fixture."""
    nodes = tmp_path / "nodes"
    for i, version in enumerate(("0", *kfd)):
        node = nodes / str(i)
        node.mkdir(parents = True)
        props = f"simd_count {0 if i == 0 else 16}\nvendor_id {0 if i == 0 else 4098}\n"
        (node / "properties").write_text(props + f"gfx_target_version {version}\n")
    import utils.hardware.amd as amd

    real = amd.amd_kfd_gpu_gfx_targets
    monkeypatch.setattr(amd, "amd_kfd_gpu_gfx_targets", lambda: real(str(nodes)))
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        nvidia, "get_physical_gpu_inventory", lambda: {"devices": [], "absent": True}
    )
    record = {"vendor": "amd", "index": 0, "name": name, "memory_total_gb": None}
    monkeypatch.setattr(hw, "_linux_drm_sysfs_records", lambda **_: [dict(record)])
    monkeypatch.setattr(hw, "_linux_amd_gfx_candidates", lambda: list(rocm))
    monkeypatch.setattr(hw, "_linux_kfd_reports_an_amd_gpu", lambda: True)
    devices = hw._probe_physical_gpu_inventory()["devices"]
    return [d for d in devices if d.get("vendor") == "amd"]


@pytest.mark.parametrize(
    "name,expected",
    [
        ("AMD Radeon 820M", False),
        ("AMD Ryzen AI 5 330 w/ Radeon Graphics", False),
        ("AMD Radeon 860M", True),
    ],
)
def test_a_named_linux_krackan_point_2_cannot_establish_a_mismatch(
    monkeypatch, tmp_path, name, expected
):
    (device,) = _linux_inventory(monkeypatch, tmp_path, name = name, kfd = ())
    assert hw._amd_device_can_establish_a_mismatch(device) is expected


def test_windows_krackan_point_2_still_establishes_a_mismatch(monkeypatch):
    # Windows routing is unchanged: the multi-arch route serves gfx1153 there.
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hw, "_linux_kfd_reports_an_amd_gpu", lambda: True)
    assert hw._amd_device_can_establish_a_mismatch({"name": "AMD Radeon 820M"}) is True


def test_a_nameless_linux_gfx1153_is_resolved_from_kfd(monkeypatch, tmp_path):
    # sysfs publishes no name and no ROCm tool answers: only KFD can name the gfx1153.
    (device,) = _linux_inventory(monkeypatch, tmp_path)
    assert device["gfx_candidates"] == ["gfx1153"]
    assert hw._amd_device_can_establish_a_mismatch(device) is False


@pytest.mark.parametrize("kfd", [("110000",), ("110501",)])
def test_other_kfd_arches_keep_the_no_candidates_answer(monkeypatch, tmp_path, kfd):
    (device,) = _linux_inventory(monkeypatch, tmp_path, kfd = kfd)
    assert "gfx_candidates" not in device
    assert hw._amd_device_can_establish_a_mismatch(device) is True


def test_gfx1153_beside_a_covered_card_still_establishes_a_mismatch(monkeypatch, tmp_path):
    (device,) = _linux_inventory(monkeypatch, tmp_path, kfd = ("110503", "110000"))
    assert device["gfx_candidates"] == ["gfx1100", "gfx1153"]
    assert hw._amd_device_can_establish_a_mismatch(device) is True


def test_rocm_tools_answer_first(monkeypatch, tmp_path):
    (device,) = _linux_inventory(monkeypatch, tmp_path, rocm = ("gfx1100",))
    assert device["gfx_candidates"] == ["gfx1100"]


def test_a_health_poll_does_no_platform_or_name_work(monkeypatch):
    # The Krackan classification lives in the inventory refresh; a poll on a cached
    # candidate-less record must not re-run it.
    calls = []
    monkeypatch.setattr(hw.platform, "system", lambda: calls.append("system") or "Linux")
    monkeypatch.setattr(hw, "_linux_kfd_reports_an_amd_gpu", lambda: True)
    for _ in range(3):
        assert hw._amd_device_can_establish_a_mismatch({"vendor": "amd", "name": None})
    assert calls == []
