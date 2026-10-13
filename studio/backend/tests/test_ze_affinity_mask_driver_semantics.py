# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""ZE_AFFINITY_MASK read the way Level Zero reads it. Expected values come from the real
intel/compute-runtime 26.35 driver with three spoofed COMPOSITE root devices."""

from __future__ import annotations

import types

import pytest

import utils.hardware.hardware as hw

_INTEL = {"vendor": "intel", "source": "sysfs-drm"}


@pytest.fixture
def xpu_host(monkeypatch):
    def _set(
        mask,
        *,
        roots = 3,
        sysfs = True,
        torch_count = None,
    ):
        monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.XPU)
        monkeypatch.setattr(hw, "get_physical_gpu_count", lambda: roots)
        monkeypatch.setattr(hw.sys, "platform", "linux")
        monkeypatch.setattr(
            hw,
            "_linux_drm_sysfs_records",
            lambda **_kw: [dict(_INTEL, index = i) for i in range(roots)] if sysfs else None,
        )
        monkeypatch.setenv("ZE_FLAT_DEVICE_HIERARCHY", "COMPOSITE")
        if mask is None:
            monkeypatch.delenv("ZE_AFFINITY_MASK", raising = False)
        else:
            monkeypatch.setenv("ZE_AFFINITY_MASK", mask)
        monkeypatch.setattr(hw, "_visible_gpu_count", None)

        def _count():
            if torch_count is None:
                raise RuntimeError("no torch")
            return torch_count

        monkeypatch.setitem(
            __import__("sys").modules,
            "torch",
            types.SimpleNamespace(xpu = types.SimpleNamespace(device_count = _count)),
        )

    return _set


@pytest.mark.parametrize(
    ("mask", "roots"),
    [
        (None, [0, 1, 2]),
        ("", [0, 1, 2]),
        ("default", [0, 1, 2]),
        ("0", [0]),
        ("2", [2]),
        ("1,0", [0, 1]),
        ("2,0", [0, 2]),
        ("0,0", [0]),
        ("0,5", [0]),
        ("0x1", [1]),
        ("01", [1]),
        ("+1", [1]),
        ("+0x1", [1]),
        ("-0", [0]),
        ("+2,0", [0, 2]),
        ("-1,1", [1]),
    ],
)
def test_composite_roots_match_the_driver(xpu_host, mask, roots):
    xpu_host(mask)
    spec = hw._get_parent_visible_gpu_spec()
    assert spec["numeric_ids"] == roots
    assert spec["supports_explicit_gpu_ids"] is True
    assert hw.get_visible_gpu_count() == len(roots)


@pytest.mark.parametrize("mask", ["", "default"])
def test_an_empty_or_default_mask_is_unset(xpu_host, mask):
    xpu_host(mask)
    assert hw._ze_affinity_mask() is None
    assert hw._mask_is_emptied("ZE_AFFINITY_MASK") is False
    assert hw._get_parent_visible_gpu_spec()["raw"] is None
    assert hw._backend_visible_devices_env() is None


@pytest.mark.parametrize("mask", ["-1", " ", "Default"])
def test_masks_the_driver_does_not_read_as_unset_keep_their_meaning(xpu_host, mask):
    """ "-1" is out of range and selects nothing; whitespace and "Default" make stoul abort."""
    xpu_host(mask)
    assert hw._ze_affinity_mask() == mask
    assert hw._mask_is_emptied("ZE_AFFINITY_MASK") is (mask.strip() in ("", "-1"))


def test_a_cuda_mask_still_reads_empty_as_none(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert hw._mask_is_emptied("CUDA_VISIBLE_DEVICES") is True


def test_a_high_root_is_kept_when_it_is_the_only_one_visible(xpu_host):
    """torch.xpu.device_count() is filtered by the mask, so it cannot bound root ids."""
    xpu_host("2", torch_count = 1)
    assert hw._get_parent_visible_gpu_spec()["numeric_ids"] == [2]
    assert hw.get_visible_gpu_count() == 1


def test_without_a_sysfs_bound_out_of_range_ids_are_kept(xpu_host):
    xpu_host("0,5", sysfs = False)
    assert hw._get_parent_visible_gpu_spec()["numeric_ids"] == [0, 5]


def test_flat_tile_ids_are_not_bounded_by_cards(xpu_host, monkeypatch):
    """One two-tile card exposes ids 0 and 1 under FLAT."""
    xpu_host("0,1", roots = 1)
    monkeypatch.setenv("ZE_FLAT_DEVICE_HIERARCHY", "FLAT")
    assert hw.get_visible_gpu_count() == 2


def test_torch_still_wins_the_visible_count(xpu_host):
    xpu_host("", torch_count = 3)
    assert hw.get_visible_gpu_count() == 3


@pytest.mark.parametrize(
    ("mask", "roots"),
    [
        ("2.0,0.1,0.2", [0, 2]),
        ("08", [0]),
        ("1abc", [1]),
        ("abc,1", [1]),
        ("-1", []),
        ("1, 0", [0, 1]),
        ("-18446744073709551615", [1]),
    ],
)
def test_parse_ze_mask_roots(mask, roots):
    assert hw._parse_ze_mask_roots(mask) == roots


def test_non_linux_hosts_have_no_root_bound(monkeypatch):
    monkeypatch.setattr(hw.sys, "platform", "win32")
    assert hw._intel_root_device_bound() is None
