# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An integrated CUDA SoC (Jetson, DGX Spark) publishes no capacity through nvidia-smi.

``nvidia-smi --query-gpu=memory.total`` answers ``[N/A]`` on a DGX Spark, which NVIDIA
documents as a known issue. The row is kept so the card stays in the inventory, but with
no capacity the frontend maps it to zero, the GGUF fit classifier returns ``ram``, and
the picker warns "No GPU detected. Runs on system RAM and CPU" on a 121 GiB Blackwell,
while the live monitor names the GB10 and prints "Unknown / 0.00 GiB" beside it (#10691).

Hermetic: torch, nvidia-smi and host memory are stubbed, so these run anywhere.
"""

from __future__ import annotations

import sys
import types

import psutil
import utils.hardware.hardware as hw
from utils.hardware import nvidia

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


def _torch_module(props) -> types.SimpleNamespace:
    return types.SimpleNamespace(get_device_properties = lambda ordinal: props)


def _not_hip(monkeypatch) -> None:
    """Stub the torch the CUDA classifier asks for HIP.

    Without this the suite reads the HOST's torch, so every assertion about an integrated
    CUDA part inverts on a ROCm machine, where ``torch.version.hip`` is set and the
    classifier correctly declines. Caught on a real gfx1151 runner, not by reading it.
    """
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(version = types.SimpleNamespace(hip = None))
    )


def _cuda_host(monkeypatch, props) -> None:
    _not_hip(monkeypatch)
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: [0])
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": "0", "numeric_ids": [0], "supports_explicit_gpu_ids": True},
    )
    monkeypatch.setattr(hw, "_torch_get_device_module", lambda: (_torch_module(props), "cuda"))


def _smi_rows(monkeypatch, memory_total_gb) -> None:
    """Stand in for nvidia.py's parsed inventory, whose [N/A] total is already a None."""
    monkeypatch.setattr(
        nvidia,
        "_query_gpu_inventory",
        lambda caller: [{"index": 0, "name": "NVIDIA GB10", "memory_total_gb": memory_total_gb}],
    )


def test_spark_recovers_the_capacity_nvidia_smi_will_not_report(monkeypatch):
    _cuda_host(monkeypatch, _SparkProps())
    _smi_rows(monkeypatch, None)

    result = hw.get_backend_visible_gpu_info()
    device = result["devices"][0]

    assert result["available"] is True
    assert device["name"] == "NVIDIA GB10"
    # The regression: this was None, which the frontend reads as a 0 GiB card.
    assert device["memory_total_gb"] == SPARK_TOTAL_GB


def test_spark_is_reported_as_one_pool_not_two(monkeypatch):
    """The recovered total IS system memory, so it must not be added to it."""
    _cuda_host(monkeypatch, _SparkProps())
    _smi_rows(monkeypatch, None)

    device = hw.get_backend_visible_gpu_info()["devices"][0]

    assert device["unified_memory"] is True
    assert device["shared_memory_host_backed_gb"] == SPARK_TOTAL_GB


def test_readable_smi_capacity_is_kept_as_is(monkeypatch):
    """A discrete card answers memory.total, and nothing here may second-guess it."""
    _cuda_host(monkeypatch, _DiscreteProps())
    _smi_rows(monkeypatch, 23.99)

    device = hw.get_backend_visible_gpu_info()["devices"][0]

    assert device["memory_total_gb"] == 23.99
    assert device.get("unified_memory") is not True


def test_torch_fallback_also_flags_the_integrated_pool(monkeypatch):
    """With nvidia-smi missing entirely the inventory comes from torch; same verdict."""
    _cuda_host(monkeypatch, _SparkProps())
    monkeypatch.setattr(nvidia, "_query_gpu_inventory", lambda caller: None)

    device = hw.get_backend_visible_gpu_info()["devices"][0]

    assert device["memory_total_gb"] == SPARK_TOTAL_GB
    assert device["unified_memory"] is True
    assert device["shared_memory_host_backed_gb"] == SPARK_TOTAL_GB


def test_rocm_is_not_classified_by_the_cuda_integrated_flag(monkeypatch):
    """HIP left that field unassigned before 6.2, so reading it there calls a card unified."""
    props = _SparkProps()
    monkeypatch.setattr(hw, "IS_ROCM", True)

    assert hw._cuda_props_are_integrated(props) is False


# ── nothing above may reach a host that is not one of these parts ────────────


def test_a_readable_smi_host_never_consults_torch(monkeypatch):
    """The repair is scoped to hosts with a missing capacity.

    /api/system is polled every 5s by the floating monitor and every 3s from Settings.
    A discrete card answers memory.total, so it must leave that poll exactly as it was:
    no torch import, no property query, no new failure mode on a machine that was fine.
    """
    _cuda_host(monkeypatch, _DiscreteProps())
    _smi_rows(monkeypatch, 23.99)

    def _forbidden(*args, **kwargs):
        raise AssertionError("torch inventory consulted on a host with nothing to repair")

    monkeypatch.setattr(hw, "_torch_get_device_inventory", _forbidden)

    assert hw.get_backend_visible_gpu_info()["devices"][0]["memory_total_gb"] == 23.99


def test_multi_gpu_smi_host_is_untouched(monkeypatch):
    """Two discrete cards, both readable: same rows out as in."""
    _cuda_host(monkeypatch, _DiscreteProps())
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": "0,1", "numeric_ids": [0, 1], "supports_explicit_gpu_ids": True},
    )
    monkeypatch.setattr(
        nvidia,
        "_query_gpu_inventory",
        lambda caller: [
            {"index": 0, "name": "NVIDIA H100", "memory_total_gb": 79.65},
            {"index": 1, "name": "NVIDIA H100", "memory_total_gb": 79.65},
        ],
    )

    devices = hw.get_backend_visible_gpu_info()["devices"]

    assert [d["memory_total_gb"] for d in devices] == [79.65, 79.65]
    assert not any(d.get("unified_memory") for d in devices)
    assert not any(d.get("shared_memory_host_backed_gb") for d in devices)


def test_xpu_is_not_classified_by_the_cuda_integrated_flag(monkeypatch):
    """A same-named field on a future Intel wheel must not rewrite an iGPU's capacity."""
    _not_hip(monkeypatch)
    monkeypatch.setattr(hw, "IS_ROCM", False)
    assert hw._cuda_props_are_integrated(_SparkProps(), "xpu") is False
    assert hw._cuda_props_are_integrated(_SparkProps(), None) is False
    assert hw._cuda_props_are_integrated(_SparkProps(), "cuda") is True


def test_an_unsizeable_card_is_still_reported_when_torch_cannot_answer(monkeypatch):
    """Both sources blank: keep the row nvidia-smi found rather than claim no GPU."""
    _cuda_host(monkeypatch, _SparkProps())
    _smi_rows(monkeypatch, None)
    monkeypatch.setattr(hw, "_torch_get_device_module", lambda: (None, None))

    result = hw.get_backend_visible_gpu_info()

    assert result["available"] is True
    assert result["devices"][0]["name"] == "NVIDIA GB10"
    assert result["devices"][0]["memory_total_gb"] is None


# ── the live monitor ─────────────────────────────────────────────────────────


def _smi_utilization(
    monkeypatch,
    vram_total_gb,
    vram_used_gb = None,
) -> dict:
    """nvidia-smi's utilization rows, whose [N/A] memory columns are already None."""
    payload = {
        "available": True,
        "devices": [
            {
                "index": 0,
                "index_kind": "physical",
                "visible_ordinal": 0,
                "gpu_utilization_pct": 0.0,
                "temperature_c": 46.0,
                "vram_used_gb": vram_used_gb,
                "vram_total_gb": vram_total_gb,
                "vram_utilization_pct": None,
                "power_draw_w": 12.09,
                "power_limit_w": None,
                "power_utilization_pct": None,
            }
        ],
        "backend_cuda_visible_devices": "0",
        "parent_visible_gpu_ids": [0],
        "index_kind": "physical",
    }
    monkeypatch.setattr(hw, "_smi_query", lambda *a, **k: payload)
    return payload


def test_monitor_sizes_the_spark_instead_of_showing_unknown(monkeypatch):
    _cuda_host(monkeypatch, _SparkProps())
    _smi_utilization(monkeypatch, vram_total_gb = None)
    import psutil

    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: types.SimpleNamespace(total = 121 * GIB, available = 100 * GIB),
    )

    device = hw.get_gpu_utilization()["devices"][0]

    assert device["vram_total_gb"] == SPARK_TOTAL_GB
    assert device["vram_used_gb"] == 21.0
    assert device["vram_utilization_pct"] == round(21.0 / SPARK_TOTAL_GB * 100, 1)
    # Columns nvidia-smi DID answer are the CLI's, untouched.
    assert device["gpu_utilization_pct"] == 0.0
    assert device["temperature_c"] == 46.0
    assert device["power_draw_w"] == 12.09


def test_monitor_leaves_a_readable_card_alone(monkeypatch):
    """A discrete card answers both memory columns, so nothing here may run."""
    _cuda_host(monkeypatch, _DiscreteProps())
    _smi_utilization(monkeypatch, vram_total_gb = 79.65, vram_used_gb = 12.0)

    def _forbidden(*args, **kwargs):
        raise AssertionError("torch inventory consulted for a card nvidia-smi could size")

    monkeypatch.setattr(hw, "_torch_get_device_inventory", _forbidden)

    device = hw.get_gpu_utilization()["devices"][0]

    assert (device["vram_total_gb"], device["vram_used_gb"]) == (79.65, 12.0)


def test_monitor_does_not_size_a_discrete_card_smi_could_not_read(monkeypatch):
    """An unreadable DISCRETE card keeps its unknown: host RAM is not its VRAM."""
    _cuda_host(monkeypatch, _DiscreteProps())
    _smi_utilization(monkeypatch, vram_total_gb = None)

    device = hw.get_gpu_utilization()["devices"][0]

    assert device["vram_total_gb"] is None
    assert device["vram_used_gb"] is None


def test_monitor_reconciliation_creates_no_driver_context(monkeypatch):
    """The poll runs every 3-5s; mem_get_info would pin ~612 MiB for the process life."""
    _cuda_host(monkeypatch, _SparkProps())
    _smi_utilization(monkeypatch, vram_total_gb = None)

    def _forbidden(*args, **kwargs):
        raise AssertionError("the monitor poll reached the occupancy probe")

    monkeypatch.setattr(hw, "_torch_get_per_device_info", _forbidden)

    assert hw.get_gpu_utilization()["devices"][0]["vram_total_gb"] == SPARK_TOTAL_GB


def test_a_hip_torch_is_never_read_with_the_cuda_rule(monkeypatch):
    """HIP reuses this namespace and a real APU sets the same flag.

    IS_ROCM is a global that detection publishes, so the classifier also asks torch
    directly: this covers the window before detection has settled, and the ROCm CI runner
    where the previous revision of these tests inverted.
    """
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(version = types.SimpleNamespace(hip = "6.2.0"))
    )

    assert hw._cuda_props_are_integrated(_SparkProps(), "cuda") is False


def _smi_visible_utilization(monkeypatch, rows) -> dict:
    payload = {
        "available": True,
        "devices": rows,
        "backend_cuda_visible_devices": "0",
        "parent_visible_gpu_ids": [0],
        "index_kind": "physical",
    }
    monkeypatch.setattr(hw, "_smi_query", lambda *a, **k: payload)
    return payload


def _unsized_row(
    index = 0,
    ordinal = 0,
    index_kind = "physical",
) -> dict:
    return {
        "index": index,
        "index_kind": index_kind,
        "visible_ordinal": ordinal,
        "gpu_utilization_pct": 0.0,
        "temperature_c": 46.0,
        "vram_used_gb": None,
        "vram_total_gb": None,
        "vram_utilization_pct": None,
        "power_draw_w": 12.09,
        "power_limit_w": None,
        "power_utilization_pct": None,
    }


def test_the_system_poll_function_is_the_one_that_gets_repaired(monkeypatch):
    """/api/system reads get_visible_gpu_utilization, NOT get_gpu_utilization.

    main.py::_get_cached_system_gpu_info calls the former, so repairing only the latter
    left the floating monitor showing Unknown / 0.00 GiB, which is the screen #10691 is
    actually about.
    """
    _cuda_host(monkeypatch, _SparkProps())
    _smi_visible_utilization(monkeypatch, [_unsized_row()])
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: types.SimpleNamespace(total = 121 * GIB, available = 100 * GIB),
    )

    device = hw.get_visible_gpu_utilization()["devices"][0]

    assert device["vram_total_gb"] == SPARK_TOTAL_GB
    assert device["vram_used_gb"] == 21.0


def test_a_uuid_mask_still_reaches_the_reconciliation(monkeypatch):
    """A UUID or MIG mask resolves to numeric_ids=None, and nvidia.py answers anyway.

    Its rows are relative-indexed and ordered by the mask, which is the order torch
    enumerates too, so the join is on visible_ordinal rather than a physical id.
    """
    _cuda_host(monkeypatch, _SparkProps())
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": "GPU-9254a6cb", "numeric_ids": None, "supports_explicit_gpu_ids": False},
    )
    monkeypatch.setattr(hw, "_torch_get_physical_gpu_count", lambda: 1)
    _smi_visible_utilization(monkeypatch, [_unsized_row(index = 0, index_kind = "relative")])

    device = hw.get_visible_gpu_utilization()["devices"][0]

    assert device["vram_total_gb"] == SPARK_TOTAL_GB


def test_a_mismatched_device_order_refuses_the_join(monkeypatch):
    """FASTEST_FIRST puts torch ordinals in a different space from nvidia-smi rows.

    Joining them anyway attaches another card's capacity to a row. The module already
    rejects this exact cross-source mapping for its SMI VRAM query; the repair uses the
    same gate rather than guessing.
    """
    _cuda_host(monkeypatch, _DiscreteProps())
    monkeypatch.setattr(hw, "_cuda_order_matches_smi", lambda: False)
    _smi_rows(monkeypatch, None)

    # The resolver refuses outright rather than handing back a mapping to guess with.
    assert hw._integrated_cuda_inventory([0]) == ({}, "index")

    # ...and the endpoint keeps the nvidia-smi rows rather than reaching the torch
    # fallback. That fallback labels its rows with physical ids taken from the mask, so
    # here it would publish one card's name and capacity under another card's index,
    # which is the same wrong join by another route and is what GPU selection reads.
    # An SMI row missing a total is the lesser answer, and the one this host had before
    # the repair existed.
    device = hw.get_backend_visible_gpu_info()["devices"][0]
    assert device["name"] == "NVIDIA GB10"
    assert device["memory_total_gb"] is None
    assert device.get("unified_memory") is not True


def test_a_partial_torch_inventory_never_drops_an_smi_card(monkeypatch):
    """_torch_get_device_inventory skips a device whose probe failed, so its list can
    be SHORT rather than empty. A short list must not replace the cards nvidia-smi saw.
    """
    _cuda_host(monkeypatch, _SparkProps())
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": "0,1", "numeric_ids": [0, 1], "supports_explicit_gpu_ids": True},
    )
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: [0, 1])
    monkeypatch.setattr(
        nvidia,
        "_query_gpu_inventory",
        lambda caller: [
            {"index": 0, "name": "NVIDIA GB10", "memory_total_gb": None},
            {"index": 1, "name": "NVIDIA GB10", "memory_total_gb": None},
        ],
    )
    # Only ordinal 0 answers, the shape a fallen-off-the-bus card produces.
    monkeypatch.setattr(
        hw,
        "_torch_get_device_inventory",
        lambda indices: [
            {
                "index": 0,
                "visible_ordinal": 0,
                "name": "NVIDIA GB10",
                "total_gb": SPARK_TOTAL_GB,
                "used_gb": None,
                "shared_memory": False,
                "shared_memory_host_backed_gb": None,
                "_rocm_known_unified": False,
                "_rocm_gfx": "",
                "_cuda_integrated": True,
            }
        ],
    )

    result = hw.get_backend_visible_gpu_info()

    assert len(result["devices"]) == 2
    assert [d["name"] for d in result["devices"]] == ["NVIDIA GB10", "NVIDIA GB10"]


def test_a_repaired_spark_row_is_marked_shared(monkeypatch):
    """gpu-vram.ts splits the dedicated and shared pools on `shared_memory` alone.

    The host-backed figure is read only after that split, so a repaired row that carries
    the figure without the flag lands in the dedicated total and is then counted a
    second time beside the same system RAM, which is the double count this repair
    exists to prevent.
    """
    _cuda_host(monkeypatch, _SparkProps())
    _smi_rows(monkeypatch, None)

    device = hw.get_backend_visible_gpu_info()["devices"][0]

    assert device["shared_memory"] is True
    assert device["unified_memory"] is True
    assert device["shared_memory_host_backed_gb"] == SPARK_TOTAL_GB


def test_a_discrete_card_is_never_marked_shared(monkeypatch):
    """The flag follows the integrated property, not the repair."""
    _cuda_host(monkeypatch, _DiscreteProps())
    _smi_rows(monkeypatch, None)

    device = hw.get_backend_visible_gpu_info()["devices"][0]

    assert device.get("shared_memory") is not True
    assert device.get("unified_memory") is not True


def test_a_known_usage_still_gets_its_percentage(monkeypatch):
    """memory.used can be readable on a row whose memory.total is [N/A].

    Filling the total is what makes the percentage computable, so a device that already
    carries a usage must not be skipped: it was the one case where both operands existed
    and the monitor still showed an unknown.
    """
    _cuda_host(monkeypatch, _SparkProps())
    utilization = {
        "devices": [
            {
                "index": 0,
                "visible_ordinal": 0,
                "vram_total_gb": None,
                "vram_used_gb": 30.0,
                "vram_utilization_pct": None,
            }
        ]
    }

    hw._reconcile_cuda_integrated_memory(utilization, [0])
    device = utilization["devices"][0]

    assert device["vram_total_gb"] == SPARK_TOTAL_GB
    # The CLI's own figure, not the host counter that stands in when it is absent.
    assert device["vram_used_gb"] == 30.0
    assert device["vram_utilization_pct"] == round((30.0 / SPARK_TOTAL_GB) * 100, 1)
