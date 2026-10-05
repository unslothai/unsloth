# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A diffusion load on a DGX Spark must not be refused against its own download.

``cudaMemGetInfo``'s free half on an integrated SoC is the kernel's ``MemFree``, which
counts the page cache as used. Measured on a GB10: writing a 60 GiB file took it from
103.25 GiB to 41.52 GiB while ``MemAvailable`` never moved off 115.6 GiB. Downloading a
diffusion model is that same write, so the load that follows is budgeted against a pool
the download appears to have consumed, and ``flux.2-klein`` was refused at "about 0 GB
usable (of the 3 GB currently free)" on a 121 GiB machine (#9919).

Hermetic: torch and host memory are stubbed, so these run anywhere.
"""

from __future__ import annotations

import sys
import types

import core.inference.diffusion_memory as diffusion_memory
import pytest

GIB = 1 << 30
MIB = 1 << 20
# What a DGX Spark actually reports.
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


# ── the diffusion free-memory reading ────────────────────────────────────────


@pytest.mark.parametrize(
    ("driver_free_mib", "available_mib", "expected_mib"),
    [
        # The measured Spark case: a fresh 60 GiB download sits in reclaimable cache.
        (3 * 1024, 115 * 1024, 115 * 1024),
        # A genuinely full machine is untouched: MemAvailable agrees with the driver.
        (3 * 1024, 3 * 1024, 3 * 1024),
        # Never above the device total, however much host memory is advertised.
        (3 * 1024, 900 * 1024, 121 * 1024),
        # Never below the driver's own figure.
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
    # The cgroup probe is a second, independent read of the host: left live, a runner
    # capped below 115 GiB lowers the snapshot and this case asserts the machine it
    # happens to run on rather than the change.
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
    # Was 3 GiB, which the shortfall guard turned into "about 0 GB is usable".
    assert memory.free_mib == 115 * 1024


def test_a_rocm_apu_snapshot_is_not_credited(monkeypatch):
    """A ROCm APU sets the same integrated flag and reaches `unified_memory` too.

    Its free reading is wrong in the OTHER direction (Windows HIP reports free == total,
    #7072), so crediting host memory on top would enlarge an over-report. Caught by the
    scenario matrix, not by reading the code.
    """
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
        # A container's limit is a CEILING. Reading it only as a lower bound threw it
        # away whenever the driver's host-wide MemFree was larger, which is the normal
        # case in a container: the load is then sized above memory.max and killed.
        (102400, 16384, 16384, 16384),
        # ... including when the credit itself would have stopped lower.
        (4096, 16384, 16384, 16384),
        # No enforcing limit: the credited pool stands.
        (29509, 118451, None, 118451),
    ],
)
def test_unified_free_is_bounded_by_an_enforcing_cgroup(
    monkeypatch, driver_free_mib, available_mib, cgroup_mib, expected_mib
):
    monkeypatch.setattr(diffusion_memory, "_available_system_memory_mib", lambda: available_mib)
    monkeypatch.setattr(diffusion_memory, "_cgroup_available_memory_mib", lambda: cgroup_mib)
    # The capacity probe is a THIRD read of the host and has to be stubbed with the
    # other two, or a runner with a real memory.max decides the "no enforcing limit"
    # case and this file stops being hermetic.
    monkeypatch.setattr(diffusion_memory, "_cgroup_memory_limit_mib", lambda: cgroup_mib)

    assert (
        diffusion_memory._unified_reclaimable_memory_mib(driver_free_mib, 124609)[0] == expected_mib
    )


def test_a_bound_cgroup_prices_the_reserve_against_the_container(monkeypatch):
    """The reserve is 20% of capacity, so capacity has to be the pool that exists.

    Capping the free reading alone left the device total at the host's 121 GiB, and
    ``_safe_device_budget_mib`` then took 24 GiB of reserve out of a 32 GiB container:
    about 8 GiB usable on a machine that could serve 25, refusing models that fit.
    """
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
    # 32 GiB less its own 20%, not less 20% of a host total the container cannot reach.
    assert budget == 32 * 1024 - int(32 * 1024 * 0.20)


def test_a_slack_cgroup_leaves_the_device_total_alone(monkeypatch):
    """A readable limit that does not bind says nothing about capacity.

    Shrinking the total whenever a limit is merely present would report an idle Spark's
    121 GiB pool as whatever happened to be free at snapshot time.
    """
    monkeypatch.setattr(diffusion_memory, "_available_system_memory_mib", lambda: 115 * 1024)
    monkeypatch.setattr(diffusion_memory, "_cgroup_available_memory_mib", lambda: 200 * 1024)
    monkeypatch.setattr(diffusion_memory, "_cgroup_memory_limit_mib", lambda: 200 * 1024)

    assert diffusion_memory._unified_reclaimable_memory_mib(29509, 124609) == (
        115 * 1024,
        124609,
    )


def test_capacity_is_the_cgroup_limit_not_what_is_left_of_it(monkeypatch):
    """The remainder shrinks as the container fills; the capacity does not.

    A 64 GiB container holding a 30 GiB model has about 34 GiB left, and reporting that
    as total capacity prices the reserve against a pool that gets smaller the more of it
    is in use, and refuses a 40 GiB replacement that only has to fit once the resident
    model is evicted.
    """
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

    # 64 GiB limit, 30 GiB of it in use.
    monkeypatch.setattr(
        LlamaCppBackend,
        "_cgroup_memory_budgets",
        staticmethod(lambda: [(34 * 1024 * 1024 * 1024, 64 * 1024 * 1024 * 1024)]),
    )

    assert LlamaCppBackend._cgroup_available_memory_mib() == 34 * 1024
    assert LlamaCppBackend._cgroup_memory_limit_mib() == 64 * 1024


def test_a_finite_limit_caps_capacity_even_when_the_remainder_is_slack():
    """A tighter host reading does not make a 64 GiB container a 121 GiB device.

    With 44 GiB of cgroup remainder but only 16 GiB of host availability, the remainder
    is not what caps the free reading, yet the limit still bounds the pool. Leaving the
    device total in place there reserved 24 GiB against a pool that cannot exceed 64 and
    produced a zero budget for a model that fits.
    """
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
    # 20% of 64 GiB, not of 121 GiB, so there is a budget at all.
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
    """AMD SDK and Radeon wheels leave version.hip unset and only tag __version__.

    Reading one as CUDA credits host memory onto an APU's free reading, which is
    already an over-report on Windows HIP (free == total, #7072), so the guard would
    approve a load the OS then kills.
    """
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
    # Untouched: 90 GiB as the driver reported it, not talked up to 96.
    assert memory.free_mib == 90 * 1024
