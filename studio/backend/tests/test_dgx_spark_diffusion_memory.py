# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Integrated SoC cudaMemGetInfo free counts page cache as used, so a download shrinks the budget."""

from __future__ import annotations

import sys
import types

import core.inference.diffusion_memory as diffusion_memory
import pytest

GIB = 1 << 30
MIB = 1 << 20
SPARK_TOTAL_BYTES = 124609 * MIB
SPARK_TOTAL_GB = round(SPARK_TOTAL_BYTES / GIB, 2)


class _SparkProps:
    """cudaDeviceProp as torch surfaces it for a GB10."""

    name = "NVIDIA GB10"
    total_memory = SPARK_TOTAL_BYTES
    is_integrated = 1
    gcnArchName = ""


class _DiscreteProps:
    name = "NVIDIA GB200"
    total_memory = 183 * GIB
    is_integrated = 0
    gcnArchName = ""


@pytest.mark.parametrize(
    ("driver_free_mib", "available_mib", "expected_mib"),
    [
        (3 * 1024, 115 * 1024, 115 * 1024),
        (3 * 1024, 3 * 1024, 3 * 1024),
        (3 * 1024, 900 * 1024, 121 * 1024),
        (100 * 1024, 40 * 1024, 100 * 1024),
    ],
)
def test_unified_free_credits_reclaimable_page_cache(
    monkeypatch, driver_free_mib, available_mib, expected_mib
):
    monkeypatch.setattr(diffusion_memory, "_available_system_memory_mib", lambda: available_mib)
    monkeypatch.setattr(diffusion_memory, "_cgroup_available_memory_mib", lambda: None)
    monkeypatch.setattr(diffusion_memory, "_cgroup_memory_limit_mib", lambda: None)

    assert (
        diffusion_memory._unified_reclaimable_memory_mib(driver_free_mib, 121 * 1024)[0]
        == expected_mib
    )


def test_unified_free_is_unchanged_when_system_memory_is_unreadable(monkeypatch):
    monkeypatch.setattr(diffusion_memory, "_available_system_memory_mib", lambda: None)
    monkeypatch.setattr(diffusion_memory, "_cgroup_available_memory_mib", lambda: None)
    monkeypatch.setattr(diffusion_memory, "_cgroup_memory_limit_mib", lambda: None)

    assert diffusion_memory._unified_reclaimable_memory_mib(3 * 1024, 121 * 1024) == (
        3 * 1024,
        121 * 1024,
    )


def test_spark_snapshot_is_unified_and_credits_the_cache(monkeypatch):
    """End to end through the snapshot the diffusion refusal is measured against."""
    torch_stub = types.SimpleNamespace(
        version = types.SimpleNamespace(hip = None),
        cuda = types.SimpleNamespace(
            current_device = lambda: 0,
            get_device_properties = lambda ordinal: _SparkProps(),
        ),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", torch_stub)
    monkeypatch.setattr(diffusion_memory, "_available_system_memory_mib", lambda: 115 * 1024)
    monkeypatch.setattr(diffusion_memory, "_cgroup_available_memory_mib", lambda: None)
    monkeypatch.setattr(diffusion_memory, "_cgroup_memory_limit_mib", lambda: None)
    hardware_stub = types.ModuleType("utils.hardware")
    hardware_stub.trusted_mem_get_info = lambda: (3 * 1024 * MIB, 121 * 1024 * MIB)
    monkeypatch.setitem(__import__("sys").modules, "utils.hardware", hardware_stub)

    memory = diffusion_memory.snapshot_device_memory(
        types.SimpleNamespace(device = "cuda", backend = "cuda")
    )

    assert memory.memory_kind == "unified_memory"
    assert memory.total_mib == 121 * 1024
    assert memory.free_mib == 115 * 1024


def test_a_rocm_apu_snapshot_is_not_credited(monkeypatch):
    """No host-memory credit for ROCm APUs: Windows HIP already over-reports free memory as total."""
    import sys as _sys

    class _ApuProps:
        name = "AMD Radeon 8060S Graphics"
        total_memory = 96 * GIB
        is_integrated = 1

    torch_stub = types.SimpleNamespace(
        version = types.SimpleNamespace(hip = "6.2.0"),
        cuda = types.SimpleNamespace(
            current_device = lambda: 0,
            get_device_properties = lambda ordinal: _ApuProps(),
        ),
    )
    monkeypatch.setitem(_sys.modules, "torch", torch_stub)
    monkeypatch.setattr(diffusion_memory, "_available_system_memory_mib", lambda: 115 * 1024)
    monkeypatch.setattr(diffusion_memory, "_cgroup_available_memory_mib", lambda: None)
    monkeypatch.setattr(diffusion_memory, "_cgroup_memory_limit_mib", lambda: None)
    hardware_stub = types.ModuleType("utils.hardware")
    hardware_stub.trusted_mem_get_info = lambda: (90 * 1024 * MIB, 96 * 1024 * MIB)
    monkeypatch.setitem(_sys.modules, "utils.hardware", hardware_stub)

    memory = diffusion_memory.snapshot_device_memory(
        types.SimpleNamespace(device = "cuda", backend = "cuda")
    )

    assert memory.memory_kind == "unified_memory"
    assert memory.free_mib == 90 * 1024


@pytest.mark.parametrize(
    ("driver_free_mib", "available_mib", "cgroup_mib", "expected_mib"),
    [
        (102400, 16384, 16384, 16384),
        (4096, 16384, 16384, 16384),
        (29509, 118451, None, 118451),
    ],
)
def test_unified_free_is_bounded_by_an_enforcing_cgroup(
    monkeypatch, driver_free_mib, available_mib, cgroup_mib, expected_mib
):
    monkeypatch.setattr(diffusion_memory, "_available_system_memory_mib", lambda: available_mib)
    monkeypatch.setattr(diffusion_memory, "_cgroup_available_memory_mib", lambda: cgroup_mib)
    # Every host read (incl. the capacity probe) is stubbed, or the runner's memory.max decides.
    monkeypatch.setattr(diffusion_memory, "_cgroup_memory_limit_mib", lambda: cgroup_mib)

    assert (
        diffusion_memory._unified_reclaimable_memory_mib(driver_free_mib, 124609)[0] == expected_mib
    )


def test_a_bound_cgroup_prices_the_reserve_against_the_container(monkeypatch):
    """Reserve is 20% of capacity, so the device total must be the container's limit, not the host's
    pool."""
    monkeypatch.setattr(diffusion_memory, "_available_system_memory_mib", lambda: 32 * 1024)
    monkeypatch.setattr(diffusion_memory, "_cgroup_available_memory_mib", lambda: 32 * 1024)
    monkeypatch.setattr(diffusion_memory, "_cgroup_memory_limit_mib", lambda: 32 * 1024)

    free_mib, total_mib = diffusion_memory._unified_reclaimable_memory_mib(102400, 124609)

    assert (free_mib, total_mib) == (32 * 1024, 32 * 1024)
    budget = diffusion_memory._safe_device_budget_mib(
        diffusion_memory.DeviceMemory(
            backend = "cuda",
            device = "cuda",
            memory_kind = "unified_memory",
            free_mib = free_mib,
            total_mib = total_mib,
        )
    )
    assert budget == 32 * 1024 - int(32 * 1024 * 0.20)


def test_a_slack_cgroup_leaves_the_device_total_alone(monkeypatch):
    """Only a binding cgroup limit lowers the device total; a slack limit must not shrink it."""
    monkeypatch.setattr(diffusion_memory, "_available_system_memory_mib", lambda: 115 * 1024)
    monkeypatch.setattr(diffusion_memory, "_cgroup_available_memory_mib", lambda: 200 * 1024)
    monkeypatch.setattr(diffusion_memory, "_cgroup_memory_limit_mib", lambda: 200 * 1024)

    assert diffusion_memory._unified_reclaimable_memory_mib(29509, 124609) == (
        115 * 1024,
        124609,
    )


def test_capacity_is_the_cgroup_limit_not_what_is_left_of_it(monkeypatch):
    """Capacity is the cgroup limit, not the remainder, which shrinks as the container fills."""
    monkeypatch.setattr(diffusion_memory, "_available_system_memory_mib", lambda: 34 * 1024)
    monkeypatch.setattr(diffusion_memory, "_cgroup_available_memory_mib", lambda: 34 * 1024)
    monkeypatch.setattr(diffusion_memory, "_cgroup_memory_limit_mib", lambda: 64 * 1024)

    free_mib, total_mib = diffusion_memory._unified_reclaimable_memory_mib(102400, 124609)

    assert (free_mib, total_mib) == (34 * 1024, 64 * 1024)


def test_an_equal_remainder_still_binds(monkeypatch):
    """``_available_system_memory_mib`` is itself cgroup-capped, so credited == remainder
    is the ordinary result in a container, not a sign that the limit does not bind."""
    monkeypatch.setattr(diffusion_memory, "_available_system_memory_mib", lambda: 32 * 1024)
    monkeypatch.setattr(diffusion_memory, "_cgroup_available_memory_mib", lambda: 32 * 1024)
    monkeypatch.setattr(diffusion_memory, "_cgroup_memory_limit_mib", lambda: 32 * 1024)

    assert diffusion_memory._unified_reclaimable_memory_mib(3 * 1024, 124609) == (
        32 * 1024,
        32 * 1024,
    )


def test_an_unreadable_limit_leaves_the_device_total(monkeypatch):
    """A remainder without a readable limit is not evidence of a smaller pool."""
    monkeypatch.setattr(diffusion_memory, "_available_system_memory_mib", lambda: 32 * 1024)
    monkeypatch.setattr(diffusion_memory, "_cgroup_available_memory_mib", lambda: 32 * 1024)
    monkeypatch.setattr(diffusion_memory, "_cgroup_memory_limit_mib", lambda: None)

    assert diffusion_memory._unified_reclaimable_memory_mib(3 * 1024, 124609) == (
        32 * 1024,
        124609,
    )


def test_the_two_cgroup_readings_are_not_the_same_number(monkeypatch):
    """The remainder and the limit come from the same walk but answer different
    questions, so a container that is already holding something must report them
    differently."""
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(
        LlamaCppBackend,
        "_cgroup_memory_budgets",
        staticmethod(lambda: [(34 * 1024 * 1024 * 1024, 64 * 1024 * 1024 * 1024)]),
    )

    assert LlamaCppBackend._cgroup_available_memory_mib() == 34 * 1024
    assert LlamaCppBackend._cgroup_memory_limit_mib() == 64 * 1024


def test_a_finite_limit_caps_capacity_even_when_the_remainder_is_slack():
    """A finite cgroup limit caps capacity even when the remainder is slack; the host total is wrong."""
    from core.inference import diffusion_memory as dm

    saved = (
        dm._available_system_memory_mib,
        dm._cgroup_available_memory_mib,
        dm._cgroup_memory_limit_mib,
    )
    dm._available_system_memory_mib = lambda: 16 * 1024
    dm._cgroup_available_memory_mib = lambda: 44 * 1024
    dm._cgroup_memory_limit_mib = lambda: 64 * 1024
    try:
        free_mib, total_mib = dm._unified_reclaimable_memory_mib(10 * 1024, 124609)
    finally:
        (
            dm._available_system_memory_mib,
            dm._cgroup_available_memory_mib,
            dm._cgroup_memory_limit_mib,
        ) = saved

    assert (free_mib, total_mib) == (16 * 1024, 64 * 1024)
    budget = diffusion_memory._safe_device_budget_mib(
        diffusion_memory.DeviceMemory(
            backend = "cuda",
            device = "cuda",
            memory_kind = "unified_memory",
            free_mib = free_mib,
            total_mib = total_mib,
        )
    )
    assert budget == 16 * 1024 - int(64 * 1024 * 0.20)
    assert budget > 0


def test_free_memory_above_a_finite_limit_is_not_free():
    """The driver's host-wide MemFree can exceed what the container may charge."""
    from core.inference import diffusion_memory as dm

    saved = (
        dm._available_system_memory_mib,
        dm._cgroup_available_memory_mib,
        dm._cgroup_memory_limit_mib,
    )
    dm._available_system_memory_mib = lambda: 100 * 1024
    dm._cgroup_available_memory_mib = lambda: 90 * 1024
    dm._cgroup_memory_limit_mib = lambda: 32 * 1024
    try:
        answer = dm._unified_reclaimable_memory_mib(80 * 1024, 124609)
    finally:
        (
            dm._available_system_memory_mib,
            dm._cgroup_available_memory_mib,
            dm._cgroup_memory_limit_mib,
        ) = saved

    assert answer == (32 * 1024, 32 * 1024)


def test_a_rocm_wheel_without_version_hip_is_still_rocm(monkeypatch):
    """AMD and Radeon wheels leave version.hip unset and tag only __version__; read them as ROCm."""
    import sys as _sys

    class _ApuProps:
        name = "AMD Radeon 8060S Graphics"
        total_memory = 96 * GIB
        is_integrated = 1

    torch_stub = types.SimpleNamespace(
        __version__ = "2.9.0+rocm6.4",
        version = types.SimpleNamespace(),
        cuda = types.SimpleNamespace(
            current_device = lambda: 0,
            get_device_properties = lambda ordinal: _ApuProps(),
        ),
    )
    monkeypatch.setitem(_sys.modules, "torch", torch_stub)
    monkeypatch.setattr(diffusion_memory, "_available_system_memory_mib", lambda: 115 * 1024)
    monkeypatch.setattr(diffusion_memory, "_cgroup_available_memory_mib", lambda: None)
    monkeypatch.setattr(diffusion_memory, "_cgroup_memory_limit_mib", lambda: None)
    hardware_stub = types.ModuleType("utils.hardware")
    hardware_stub.trusted_mem_get_info = lambda: (90 * 1024 * MIB, 96 * 1024 * MIB)
    monkeypatch.setitem(_sys.modules, "utils.hardware", hardware_stub)

    memory = diffusion_memory.snapshot_device_memory(
        types.SimpleNamespace(device = "cuda", backend = "cuda")
    )

    assert memory.memory_kind == "unified_memory"
    assert memory.free_mib == 90 * 1024
