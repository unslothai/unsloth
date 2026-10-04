# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Linux ROCm APU (Strix Halo): the unified OS reserve is not taken twice. Numbers are the gfx1151 runner's at the
refusal."""

from __future__ import annotations

import types

import pytest

import core.inference.diffusion_memory as dm
from core.inference.diffusion_memory import (
    DeviceMemory,
    plan_diffusion_memory,
    unified_memory_shortfall_message,
)

POOL_TOTAL = 64 * 1024
POOL_FREE = 31 * 1024
HOST_AVAILABLE = 86 * 1024
ZIMAGE_WEIGHTS = 20 * 1024  # + the 1 GiB default base overhead = the "about 21 GB" of the refusal


def _rocm_torch():
    return types.SimpleNamespace(
        __version__ = "2.11.0+rocm7.13.0", version = types.SimpleNamespace(hip = "7.13")
    )


def _cuda_torch():
    return types.SimpleNamespace(
        __version__ = "2.11.0+cu130", version = types.SimpleNamespace(hip = None)
    )


@pytest.fixture
def host(monkeypatch):
    state = {"torch": _rocm_torch(), "platform": "linux", "available": HOST_AVAILABLE}
    monkeypatch.setattr(dm.sys, "platform", "linux")

    def apply():
        monkeypatch.setitem(dm.sys.modules, "torch", state["torch"])
        monkeypatch.setattr(dm.sys, "platform", state["platform"])
        monkeypatch.setattr(dm, "_available_system_memory_mib", lambda: state["available"])

    state["apply"] = apply
    apply()
    return state


def _plan(
    free = POOL_FREE,
    total = POOL_TOTAL,
    kind = "unified_memory",
):
    return plan_diffusion_memory(
        target = types.SimpleNamespace(
            device = "cuda", backend = "cuda", supports_model_cpu_offload = True
        ),
        device_memory = DeviceMemory("cuda", "cuda:0", kind, free, total),
        model_dense_mib = ZIMAGE_WEIGHTS,
        runtime_headroom_mib = 3072,
    )


def test_linux_rocm_apu_with_host_room_loads_what_comfyui_loads(host):
    plan = _plan()
    assert plan.estimates["safe_device_budget_mib"] == POOL_FREE - int(POOL_TOTAL * 0.10)
    assert unified_memory_shortfall_message(plan, family = "z-image") is None


def test_still_refuses_what_cannot_fit_the_pool(host):
    plan = _plan(free = 18 * 1024)
    assert unified_memory_shortfall_message(plan, family = "z-image") is not None


def test_host_ram_too_tight_keeps_the_unified_reserve(host):
    host["available"] = (
        POOL_FREE + 4 * 1024
    )  # 4 GiB outside the pool: the OS reserve is not covered
    host["apply"]()
    plan = _plan()
    assert plan.estimates["safe_device_budget_mib"] == POOL_FREE - int(POOL_TOTAL * 0.20)
    assert unified_memory_shortfall_message(plan, family = "z-image") is not None


def test_unknown_host_reading_keeps_the_unified_reserve(host):
    host["available"] = None
    host["apply"]()
    assert _plan().estimates["safe_device_budget_mib"] == POOL_FREE - int(POOL_TOTAL * 0.20)


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_other_platforms_unchanged(host, platform):
    host["platform"] = platform
    host["apply"]()
    assert _plan().estimates["safe_device_budget_mib"] == POOL_FREE - int(POOL_TOTAL * 0.20)


def test_nvidia_unified_memory_unchanged(host):
    host["torch"] = _cuda_torch()
    host["apply"]()
    assert _plan().estimates["safe_device_budget_mib"] == POOL_FREE - int(POOL_TOTAL * 0.20)


def test_discrete_vram_unchanged(host):
    assert _plan(kind = "discrete_vram").estimates["safe_device_budget_mib"] == POOL_FREE - int(
        POOL_TOTAL * 0.10
    )


def test_fast_budget_follows_the_same_reserve(host):
    memory = DeviceMemory("cuda", "cuda:0", "unified_memory", POOL_FREE, POOL_TOTAL)
    assert dm._fast_device_budget_mib(memory) == POOL_FREE - max(2048, int(POOL_TOTAL * 0.10) // 2)
    host["torch"] = _cuda_torch()
    host["apply"]()
    assert dm._fast_device_budget_mib(memory) == POOL_FREE - max(2048, int(POOL_TOTAL * 0.20) // 2)


def test_total_capacity_gates_follow_the_same_reserve(host):
    # 47 GiB fits 0.85 * (64 - 6.4), not 0.85 * (64 - 12.8): a disagreeing prefetch gate drops the load to GGUF.
    memory = DeviceMemory("cuda", "cuda:0", "unified_memory", POOL_FREE, POOL_TOTAL)
    plan = types.SimpleNamespace(
        estimates = {"resident_required_mib": 47 * 1024}, device_memory = memory
    )
    assert dm.total_capacity_budget_mib(memory) == int((POOL_TOTAL - int(POOL_TOTAL * 0.10)) * 0.85)
    assert dm.plan_fits_total_capacity(plan)
    host["torch"] = _cuda_torch()
    host["apply"]()
    assert dm.total_capacity_budget_mib(memory) == int((POOL_TOTAL - int(POOL_TOTAL * 0.20)) * 0.85)
    assert not dm.plan_fits_total_capacity(plan)


def test_dense_prefetch_gate_uses_the_shared_capacity_budget():
    import ast
    import inspect

    import core.inference.diffusion as diffusion

    src = inspect.getsource(diffusion)
    assert "total_capacity_budget_mib(snapshot_device_memory(target))" in src
    names = {n.id for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Name)}
    assert "_reserve_mib" not in names
