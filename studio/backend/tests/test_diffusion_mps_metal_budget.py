# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Apple Silicon: the unified-memory refusal sizes against Metal's working set, not RAM minus a 20% reserve.

The machine is a 24 GiB MacBook Air with every app closed: 16.8 GiB available and a 16 GiB Metal working set (two
thirds of RAM). Weights are the Hub sizes of the unsloth GGUFs and of the bf16 Qwen3-4B text encoder and VAE every
Z-Image and FLUX.2 Klein 4B load adds. Under the RAM reading Z-Image Q4_K_M was refused with "needs about 15 GB, only
about 12 GB usable", the numbers in the report.
"""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_memory as dm
from core.inference.diffusion_memory import (
    plan_diffusion_memory,
    snapshot_device_memory,
    unified_memory_shortfall_message,
)

RAM_MIB = 24 * 1024
AVAILABLE_MIB = 17_203  # 16.8 GiB
WORKING_SET_MIB = 16 * 1024

TEXT_ENCODER_MIB = 7_672  # Qwen3-4B, bf16, shared by both families
ZIMAGE_VAE_MIB = 159
KLEIN_VAE_MIB = 160
GGUF_MIB = {
    "z-image Q2_K": 3_471,
    "z-image Q3_K_S": 3_768,
    "z-image Q4_K_M": 4_785,
    "klein Q4_K_M": 2_483,
    "klein Q6_K": 3_251,
    "klein Q8_0": 4_101,
}

MPS = types.SimpleNamespace(device = "mps", backend = "mps", supports_model_cpu_offload = False)


@pytest.fixture
def mac(monkeypatch):
    state = {"available": AVAILABLE_MIB, "working_set": (WORKING_SET_MIB, 0)}
    monkeypatch.setattr(dm, "_system_memory_mib", lambda: (RAM_MIB, state["available"]))
    monkeypatch.setattr(dm, "_metal_working_set_mib", lambda: state["working_set"])
    return state


def _verdict(name):
    vae = ZIMAGE_VAE_MIB if name.startswith("z-image") else KLEIN_VAE_MIB
    plan = plan_diffusion_memory(
        target = MPS,
        device_memory = snapshot_device_memory(MPS),
        model_dense_mib = dm.estimate_gguf_resident_mib(GGUF_MIB[name]) + TEXT_ENCODER_MIB + vae,
        companion_dense_mib = TEXT_ENCODER_MIB + vae,
        text_encoder_dense_mib = TEXT_ENCODER_MIB,
        runtime_headroom_mib = dm.estimate_image_runtime_mib(width = None, height = None),
    )
    return plan, unified_memory_shortfall_message(plan, family = name.split()[0])


def test_the_working_set_bounds_the_mps_reading(mac):
    memory = snapshot_device_memory(MPS)
    assert (memory.free_mib, memory.total_mib) == (WORKING_SET_MIB, WORKING_SET_MIB)
    assert memory.metal_working_set_mib == WORKING_SET_MIB
    # 15% of the working set held back: 13.6 GiB, not 12.
    assert dm._safe_device_budget_mib(memory) == WORKING_SET_MIB - int(WORKING_SET_MIB * 0.15)


def test_available_ram_and_this_process_metal_still_bound_it(mac):
    mac["available"] = 10_000
    assert snapshot_device_memory(MPS).free_mib == 10_000
    mac["available"] = AVAILABLE_MIB
    mac["working_set"] = (WORKING_SET_MIB, 1_024)
    assert snapshot_device_memory(MPS).free_mib == WORKING_SET_MIB - 1_024


def test_the_working_set_never_budgets_below_the_ram_reading(mac):
    # 2 GiB already in Metal: 14,336 - 2,457 = 11,879 would undercut RAM minus 20% (12,288), so RAM stands.
    mac["working_set"] = (WORKING_SET_MIB, 2_048)
    memory = snapshot_device_memory(MPS)
    assert (memory.free_mib, memory.metal_working_set_mib) == (AVAILABLE_MIB, None)
    assert dm._safe_device_budget_mib(memory) == AVAILABLE_MIB - int(RAM_MIB * 0.20)


# Idle Macs with more RAM available than the working set (2/3 of RAM up to 32 GB, 3/4 above).
IDLE_MACS = [
    (24, 19_456, 16_384),
    (32, 27_648, 21_845),
    (64, 57_671, 49_152),
    (128, 115_343, 98_304),
    (192, 173_015, 147_456),
]


@pytest.mark.parametrize("ram_gb, available, working_set", IDLE_MACS)
def test_an_idle_mac_keeps_every_load_the_ram_reading_admitted(mac, monkeypatch, ram_gb, available, working_set):
    ram = ram_gb * 1024
    monkeypatch.setattr(dm, "_system_memory_mib", lambda: (ram, available))
    mac["working_set"] = (working_set, 0)
    ram_budget = available - int(ram * 0.20)
    assert dm._safe_device_budget_mib(snapshot_device_memory(MPS)) == ram_budget
    # A model needing exactly the RAM budget: admitted before the working set reading existed, so still admitted.
    plan = plan_diffusion_memory(
        target = MPS,
        device_memory = snapshot_device_memory(MPS),
        model_dense_mib = ram_budget - dm.DEFAULT_BASE_OVERHEAD_MIB,
        runtime_headroom_mib = dm.estimate_image_runtime_mib(width = None, height = None),
    )
    assert unified_memory_shortfall_message(plan) is None


def test_without_a_working_set_reading_the_ram_reading_stands(mac):
    mac["working_set"] = (None, 0)
    memory = snapshot_device_memory(MPS)
    assert (memory.free_mib, memory.total_mib, memory.metal_working_set_mib) == (
        AVAILABLE_MIB,
        RAM_MIB,
        None,
    )
    assert dm._safe_device_budget_mib(memory) == AVAILABLE_MIB - int(RAM_MIB * 0.20)


def test_the_ram_reading_reproduces_the_report(mac):
    mac["working_set"] = (None, 0)
    _, message = _verdict("z-image Q4_K_M")
    assert "needs about 15 GB" in message
    assert "only about 12 GB is usable" in message


@pytest.mark.parametrize("name", ["z-image Q2_K", "z-image Q3_K_S", "klein Q4_K_M", "klein Q6_K"])
def test_models_that_fit_the_working_set_now_load(mac, name):
    mac["working_set"] = (None, 0)
    assert _verdict(name)[1] is not None  # refused under the RAM reading
    mac["working_set"] = (WORKING_SET_MIB, 0)
    assert _verdict(name)[1] is None


@pytest.mark.parametrize("name", ["z-image Q4_K_M", "klein Q8_0"])
def test_weights_past_the_working_set_budget_are_still_refused(mac, name):
    plan, message = _verdict(name)
    assert message is not None
    assert "the GPU can use right now" in message
    # A lower quant does fit here (z-image Q2_K above), so the quant advice is the right one.
    assert "lower GGUF quant" in message


def test_close_figures_get_a_decimal_instead_of_reading_as_equal(mac):
    # 14,186 vs 13,927 MiB would both print as "about 14 GB".
    assert "needs about 13.9 GB" in _verdict("klein Q8_0")[1]
    assert "only about 13.6 GB is usable" in _verdict("klein Q8_0")[1]
    assert "needs about 15 GB" in _verdict("z-image Q4_K_M")[1]
    assert "only about 14 GB is usable" in _verdict("z-image Q4_K_M")[1]


def test_the_encoder_floor_replaces_the_quant_advice_when_no_quant_can_fit(mac):
    # A busy Mac: 9,000 - 2,457 leaves less than the 7.8 GiB of companions + overhead.
    mac["available"] = 9_000
    _, message = _verdict("z-image Q2_K")
    assert "text encoders and VAE alone need about 10 GB" in message
    assert "lower GGUF quant" not in message


def test_the_working_set_reading_comes_from_torch_mps(monkeypatch):
    mps = types.SimpleNamespace(
        recommended_max_memory = lambda: WORKING_SET_MIB * 1024 * 1024,
        driver_allocated_memory = lambda: 512 * 1024 * 1024,
    )
    monkeypatch.setitem(dm.sys.modules, "torch", types.SimpleNamespace(mps = mps))
    assert dm._metal_working_set_mib() == (WORKING_SET_MIB, 512)

    def unsupported():
        raise AttributeError("recommended_max_memory")

    mps.recommended_max_memory = unsupported
    assert dm._metal_working_set_mib() == (None, 0)


def test_other_unified_devices_keep_their_reserve():
    integrated = dm.DeviceMemory("cuda", "cuda", "unified_memory", AVAILABLE_MIB, RAM_MIB)
    assert dm._safe_device_budget_mib(integrated) == AVAILABLE_MIB - int(RAM_MIB * 0.20)
