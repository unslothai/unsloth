# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""rocm_linux_kernel_free_mib: free VRAM as the kernel counts it, for the card torch is on. No GPU needed: the DRM
cards and torch's device properties are staged."""

from __future__ import annotations

import types

import pytest

from utils.hardware import hardware

_GIB = 1024**3


def _card(tmp_path, name, used, total):
    d = tmp_path / name
    d.mkdir()
    (d / "mem_info_vram_used").write_text(f"{used}\n")
    (d / "mem_info_vram_total").write_text(f"{total}\n")
    return str(d)


def _module(
    bus,
    total,
    domain = 0,
    dev = 0,
):
    props = types.SimpleNamespace(
        pci_domain_id = domain, pci_bus_id = bus, pci_device_id = dev, total_memory = total
    )
    return types.SimpleNamespace(current_device = lambda: 0, get_device_properties = lambda i: props)


@pytest.fixture
def rocm_linux(monkeypatch):
    import torch
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(torch.version, "hip", "7.2.26015", raising = False)


def test_reads_the_card_torch_is_on(tmp_path, monkeypatch, rocm_linux):
    """A Vulkan process holding VRAM is in the kernel's count, which the HIP figure misses."""
    cards = [
        ("0000:03:00.0", 0, _card(tmp_path, "a", 1 * _GIB, 8 * _GIB)),
        ("0000:0b:00.0", 1, _card(tmp_path, "b", 6 * _GIB, 12 * _GIB)),
    ]
    monkeypatch.setattr(hardware, "_rocm_linux_amdgpu_cards", lambda: cards)
    assert hardware.rocm_linux_kernel_free_mib(module = _module(0x0B, 12 * _GIB)) == 6 * 1024
    assert hardware.rocm_linux_kernel_free_mib(module = _module(0x03, 8 * _GIB)) == 7 * 1024


def test_unknown_when_it_cannot_be_told(tmp_path, monkeypatch, rocm_linux):
    """No matching card, or a sysfs total that is another memory scope (APU carve-out, partitioned card)."""
    cards = [("0000:0b:00.0", 1, _card(tmp_path, "b", 1 * _GIB, 12 * _GIB))]
    monkeypatch.setattr(hardware, "_rocm_linux_amdgpu_cards", lambda: cards)
    assert hardware.rocm_linux_kernel_free_mib(module = _module(0x0C, 12 * _GIB)) is None
    assert hardware.rocm_linux_kernel_free_mib(module = _module(0x0B, 64 * _GIB)) is None
    broken = types.SimpleNamespace(
        current_device = lambda: 0, get_device_properties = lambda i: object()
    )
    assert hardware.rocm_linux_kernel_free_mib(module = broken) is None


def test_only_on_linux_rocm(tmp_path, monkeypatch):
    import torch

    cards = [("0000:0b:00.0", 1, _card(tmp_path, "b", 1 * _GIB, 12 * _GIB))]
    monkeypatch.setattr(hardware, "_rocm_linux_amdgpu_cards", lambda: cards)
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    assert hardware.rocm_linux_kernel_free_mib(module = _module(0x0B, 12 * _GIB)) is None
    monkeypatch.setattr(torch.version, "hip", "7.2.26015", raising = False)
    monkeypatch.setattr(hardware.platform, "system", lambda: "Windows")
    assert hardware.rocm_linux_kernel_free_mib(module = _module(0x0B, 12 * _GIB)) is None
