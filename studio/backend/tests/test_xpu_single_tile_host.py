# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Explicit XPU GPU ids at the driver's default hierarchy: Arc / Flex cards and Intel iGPUs have one
tile per root, so FLAT and COMPOSITE enumerate the same devices. Fake sysfs trees only."""

from __future__ import annotations

import os
import sys

import pytest

import utils.hardware.hardware as hw


def _card(drm, pci, n, vendor, device):
    dev = pci / f"0000:{n}"
    dev.mkdir(parents = True)
    if vendor is not None:
        (dev / "vendor").write_text(vendor + "\n")
        (dev / "device").write_text(device + "\n")
    card = drm / f"card{len(os.listdir(drm))}"
    card.mkdir()
    (card / "device").symlink_to(dev)


@pytest.fixture
def sysfs(tmp_path, monkeypatch):
    monkeypatch.setattr(hw.sys, "platform", "linux")
    drm, pci = tmp_path / "drm", tmp_path / "pci"
    drm.mkdir()
    pci.mkdir()

    def _add(
        n,
        vendor,
        device = "0x0000",
    ):
        _card(drm, pci, n, vendor, device)
        return str(drm)

    return _add, str(drm)


@pytest.mark.parametrize(
    ("cards", "single_tile"),
    [
        ([("03:00.0", "0x8086", "0xe20b"), ("04:00.0", "0x8086", "0xe20b")], True),  # 2x B580
        (
            [("00:02.0", "0x8086", "0xa780"), ("03:00.0", "0x8086", "0xe211")],
            True,
        ),  # UHD 770 + Pro B60
        ([("03:00.0", "0x8086", "0x56a0"), ("01:00.0", "0x10de", "0x2684")], True),  # A770 + NVIDIA
        (
            [("00:03.0", None, None), ("03:00.0", "0x8086", "0x56c0")],
            True,
        ),  # framebuffer + Flex 170
        ([("03:00.0", "0x8086", "0x0bd5"), ("04:00.0", "0x8086", "0x0bd5")], False),  # Max 1550
        ([("03:00.0", "0x8086", "0x4905")], False),  # DG1: not an allowlisted part
        ([("01:00.0", "0x10de", "0x2684")], False),  # no Intel card
        ([], False),
    ],
)
@pytest.mark.skipif(sys.platform == "win32", reason = "fake sysfs needs ':' in paths and symlinks")
def test_single_tile_host(sysfs, cards, single_tile):
    add, drm = sysfs
    for pci, vendor, device in cards:
        add(pci, vendor, device)
    assert hw._intel_xpu_single_tile_host(drm) is single_tile


def test_non_linux_hosts_keep_the_flat_assumption(monkeypatch):
    monkeypatch.setattr(hw.sys, "platform", "win32")
    assert hw._intel_xpu_single_tile_host() is False


@pytest.mark.parametrize(
    ("hierarchy", "single_tile", "composite"),
    [
        (None, True, True),
        (None, False, False),
        ("COMPOSITE", False, True),
        ("FLAT", True, False),
        ("COMBINED", True, False),
    ],
)
def test_hierarchy(monkeypatch, hierarchy, single_tile, composite):
    for var in hw._SYCL_DEVICE_FILTER_VARS:
        monkeypatch.delenv(var, raising = False)
    if hierarchy is None:
        monkeypatch.delenv("ZE_FLAT_DEVICE_HIERARCHY", raising = False)
    else:
        monkeypatch.setenv("ZE_FLAT_DEVICE_HIERARCHY", hierarchy)
    monkeypatch.setattr(hw, "_intel_xpu_single_tile_host", lambda: single_tile)
    assert hw._xpu_hierarchy_is_composite() is composite


def test_multi_tile_ids_are_allowlisted_pvc():
    assert hw._INTEL_MULTI_TILE_PCI_IDS <= hw._INTEL_XPU_PCI_IDS


@pytest.mark.parametrize(
    "var", ["ONEAPI_DEVICE_SELECTOR", "SYCL_DEVICE_FILTER", "SYCL_DEVICE_ALLOWLIST"]
)
def test_a_sycl_device_filter_keeps_ordinals_relative(monkeypatch, var):
    """The filter re-indexes torch's devices, so ordinal 0 need not be root 0."""
    monkeypatch.delenv("ZE_FLAT_DEVICE_HIERARCHY", raising = False)
    monkeypatch.setattr(hw, "_intel_xpu_single_tile_host", lambda: True)
    monkeypatch.setenv(var, "level_zero:1")
    assert hw._xpu_hierarchy_is_composite() is False
    monkeypatch.setenv("ZE_FLAT_DEVICE_HIERARCHY", "COMPOSITE")
    assert hw._xpu_hierarchy_is_composite() is True
