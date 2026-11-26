# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""rocm_linux_kernel_free_mib against staged DRM cards and torch device properties; no GPU needed."""

from __future__ import annotations

import types

import pytest

from utils.hardware import hardware

_GIB = 1024**3


def _card(
    tmp_path,
    name,
    used,
    total,
    gtt_used = None,
    gtt_total = None,
):
    d = tmp_path / name
    d.mkdir()
    (d / "mem_info_vram_used").write_text(f"{used}\n")
    (d / "mem_info_vram_total").write_text(f"{total}\n")
    if gtt_total is not None:
        (d / "mem_info_gtt_used").write_text(f"{gtt_used}\n")
        (d / "mem_info_gtt_total").write_text(f"{gtt_total}\n")
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
    return types.SimpleNamespace(
        current_device = lambda: 0,
        get_device_properties = lambda i: props,
        mem_get_info = lambda *a: (total, total),
    )


@pytest.fixture
def rocm_linux(monkeypatch):
    import torch
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(torch.version, "hip", "7.2.26015", raising = False)


def test_reads_the_card_torch_is_on(tmp_path, monkeypatch, rocm_linux):
    cards = [
        ("0000:03:00.0", 0, _card(tmp_path, "a", 1 * _GIB, 8 * _GIB)),
        ("0000:0b:00.0", 1, _card(tmp_path, "b", 6 * _GIB, 12 * _GIB)),
    ]
    monkeypatch.setattr(hardware, "_rocm_linux_amdgpu_cards", lambda: cards)
    assert hardware.rocm_linux_kernel_free_mib(module = _module(0x0B, 12 * _GIB)) == 6 * 1024
    assert hardware.rocm_linux_kernel_free_mib(module = _module(0x03, 8 * _GIB)) == 7 * 1024


def test_unknown_when_it_cannot_be_told(tmp_path, monkeypatch, rocm_linux):
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
    monkeypatch.setattr(torch, "__version__", "2.11.0+cu130")
    assert hardware.rocm_linux_kernel_free_mib(module = _module(0x0B, 12 * _GIB)) is None
    # Radeon wheels: no version.hip, ROCm tag in __version__.
    monkeypatch.setattr(torch, "__version__", "2.9.1+rocm7.2.1.lw.gitff65f5bc")
    assert hardware.rocm_linux_kernel_free_mib(module = _module(0x0B, 12 * _GIB)) == 11 * 1024
    monkeypatch.setattr(torch.version, "hip", "7.2.26015", raising = False)
    monkeypatch.setattr(hardware.platform, "system", lambda: "Windows")
    assert hardware.rocm_linux_kernel_free_mib(module = _module(0x0B, 12 * _GIB)) is None


def _hip(
    bus,
    total,
    free,
    driver_total = None,
):
    mod = _module(bus, total)
    mod.mem_get_info = lambda *a: (free, total if driver_total is None else driver_total)
    return mod


def test_trusted_mem_get_info_caps_hip_free_at_the_kernel_figure(tmp_path, monkeypatch, rocm_linux):
    cards = [("0000:0b:00.0", 1, _card(tmp_path, "b", 6 * _GIB, 12 * _GIB))]
    monkeypatch.setattr(hardware, "_rocm_linux_amdgpu_cards", lambda: cards)
    hip = _hip(0x0B, 12 * _GIB, 11 * _GIB + _GIB // 2)
    assert hardware.trusted_mem_get_info(module = hip) == (6 * _GIB, 12 * _GIB)
    assert hardware.trusted_mem_get_info(0, module = hip) == (6 * _GIB, 12 * _GIB)
    assert hardware.trusted_mem_get_info(module = _hip(0x0B, 12 * _GIB, 4 * _GIB)) == (
        4 * _GIB,
        12 * _GIB,
    )
    # A driver total spanning GTT (APU) is not the scope sysfs VRAM describes.
    apu = _hip(0x0B, 12 * _GIB, 40 * _GIB, driver_total = 64 * _GIB)
    assert hardware.trusted_mem_get_info(module = apu) == (40 * _GIB, 64 * _GIB)


def test_trusted_mem_get_info_passes_cuda_through(tmp_path, monkeypatch):
    import torch

    cards = [("0000:0b:00.0", 1, _card(tmp_path, "b", 6 * _GIB, 12 * _GIB))]
    monkeypatch.setattr(hardware, "_rocm_linux_amdgpu_cards", lambda: cards)
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hardware.sys, "platform", "linux")
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch, "__version__", "2.11.0+cu130")
    hip = _hip(0x0B, 12 * _GIB, 11 * _GIB)
    assert hardware.trusted_mem_get_info(module = hip) == (11 * _GIB, 12 * _GIB)


def test_segmenter_and_diffusion_planner_see_the_kernel_figure(tmp_path, monkeypatch, rocm_linux):
    import torch

    from core.inference import diffusion_memory, diffusion_segment

    cards = [("0000:0b:00.0", 1, _card(tmp_path, "b", 6 * _GIB, 12 * _GIB))]
    monkeypatch.setattr(hardware, "_rocm_linux_amdgpu_cards", lambda: cards)
    hip = _hip(0x0B, 12 * _GIB, 11 * _GIB + _GIB // 2)
    for name in ("current_device", "get_device_properties", "mem_get_info"):
        monkeypatch.setattr(torch.cuda, name, getattr(hip, name))
    monkeypatch.setattr(diffusion_memory, "cuda_reclaimable_bytes", lambda *a, **k: 0)
    monkeypatch.setattr(diffusion_memory, "user_budget_reserve_bytes", lambda *a, **k: 0)
    assert diffusion_segment._cuda_free_mib(0) == (6 * 1024, 12 * 1024)
    assert diffusion_memory._cuda_memory("cuda")[:2] == (6 * 1024, 12 * 1024)


@pytest.fixture
def apu(monkeypatch):
    from core.inference import video_minimax_h3

    state = types.SimpleNamespace(gtt_free = None)
    monkeypatch.setattr(hardware, "_rocm_props_unified_status", lambda props: True)
    monkeypatch.setattr(video_minimax_h3, "amdgpu_gtt_reading", lambda slot: (state.gtt_free, True))
    return state


def test_an_apu_whose_driver_total_is_the_gtt_pool_reads_the_gtt_pool(
    tmp_path, monkeypatch, rocm_linux, apu
):
    # Strix Halo (gfx1151) as measured: 512 MiB carve-out, 100 GiB GTT, a Vulkan llama-server holding 4.8 GiB.
    cards = [
        ("0000:c3:00.0", 0, _card(tmp_path, "apu", 189 << 20, 512 << 20, 4864 << 20, 100 * _GIB))
    ]
    monkeypatch.setattr(hardware, "_rocm_linux_amdgpu_cards", lambda: cards)
    apu.gtt_free = 100 * _GIB - (4864 << 20)
    assert (
        hardware.rocm_linux_kernel_free_mib(module = _module(0xC3, 100 * _GIB)) == 100 * 1024 - 4864
    )
    assert hardware.rocm_linux_kernel_free_mib(module = _module(0xC3, 40 * _GIB)) is None


def test_the_pool_closest_to_the_driver_total_wins(tmp_path, monkeypatch, rocm_linux, apu):
    cards = [("0000:c3:00.0", 0, _card(tmp_path, "apu", 1 * _GIB, 32 * _GIB, 24 * _GIB, 34 * _GIB))]
    monkeypatch.setattr(hardware, "_rocm_linux_amdgpu_cards", lambda: cards)
    apu.gtt_free = 10 * _GIB
    assert hardware.rocm_linux_kernel_free_mib(module = _module(0xC3, 34 * _GIB)) == 10 * 1024


def test_gtt_is_only_read_on_a_known_apu(tmp_path, monkeypatch, rocm_linux):
    # A partitioned discrete card whose total happens to match the GTT pool keeps HIP's figure.
    cards = [
        ("0000:0b:00.0", 0, _card(tmp_path, "dgpu", 1 * _GIB, 48 * _GIB, 38 * _GIB, 40 * _GIB))
    ]
    monkeypatch.setattr(hardware, "_rocm_linux_amdgpu_cards", lambda: cards)
    monkeypatch.setattr(hardware, "_rocm_props_unified_status", lambda props: False)
    assert hardware.rocm_linux_kernel_free_mib(module = _module(0x0B, 40 * _GIB)) is None


def test_vram_wins_a_size_tie_with_gtt(tmp_path, monkeypatch, rocm_linux, apu):
    cards = [("0000:c3:00.0", 0, _card(tmp_path, "apu", 2 * _GIB, 32 * _GIB, 30 * _GIB, 32 * _GIB))]
    monkeypatch.setattr(hardware, "_rocm_linux_amdgpu_cards", lambda: cards)
    apu.gtt_free = 2 * _GIB
    assert hardware.rocm_linux_kernel_free_mib(module = _module(0xC3, 32 * _GIB)) == 30 * 1024


def test_an_unreadable_counter_in_the_other_pool_does_not_hide_the_chosen_one(
    tmp_path, monkeypatch, rocm_linux, apu
):
    card = _card(tmp_path, "apu", 189 << 20, 512 << 20, 24 * _GIB, 100 * _GIB)
    (tmp_path / "apu" / "mem_info_vram_used").unlink()
    monkeypatch.setattr(hardware, "_rocm_linux_amdgpu_cards", lambda: [("0000:c3:00.0", 0, card)])
    apu.gtt_free = 76 * _GIB
    assert hardware.rocm_linux_kernel_free_mib(module = _module(0xC3, 100 * _GIB)) == 76 * 1024
