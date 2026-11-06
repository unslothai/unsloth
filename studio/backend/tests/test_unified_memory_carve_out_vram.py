# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""On integrated CUDA SoCs nvidia-smi reports only the carve-out; torch reports the real pool."""

from __future__ import annotations

import sys
import types

import psutil
import pytest
import utils.hardware.hardware as hw
from core.inference.llama_cpp import LlamaCppBackend
from utils.hardware import nvidia

GIB = 1 << 30
MIB = 1 << 20

# Measured on the RTX Spark N1X.
N1X_POOL_MIB = 46477
N1X_POOL_BYTES = 48735117312
N1X_POOL_GB = round(N1X_POOL_BYTES / GIB, 2)  # 45.39
N1X_CARVE_OUT_MIB = 8128
N1X_CARVE_OUT_GB = round(N1X_CARVE_OUT_MIB * MIB / GIB, 2)  # 7.94
N1X_USED_GB = 5.73
HOST_TOTAL_GB = 54.21
HOST_AVAILABLE_GB = 42.55


class _N1XProps:
    """cudaDeviceProp as torch surfaces it for an RTX Spark N1X."""

    name = "NVIDIA RTX Spark N1X (5120-core Blackwell RTX GPU)"
    total_memory = N1X_POOL_BYTES
    is_integrated = 1
    gcnArchName = ""


class _DiscreteProps:
    name = "NVIDIA GeForce RTX 4090"
    total_memory = 24 * GIB
    is_integrated = 0
    gcnArchName = ""


def _torch_module(props_by_ordinal) -> types.SimpleNamespace:
    def _get(ordinal):
        try:
            return props_by_ordinal[ordinal]
        except (IndexError, KeyError):
            raise RuntimeError("Invalid device id")

    return types.SimpleNamespace(get_device_properties = _get)


def _not_hip(monkeypatch) -> None:
    """Stub the torch the CUDA classifier asks for HIP, so a ROCm runner does not invert
    every assertion here (the same trap test_dgx_spark_gpu_inventory.py documents)."""
    monkeypatch.setitem(
        sys.modules, "torch", types.SimpleNamespace(version = types.SimpleNamespace(hip = None))
    )


def _cuda_host(
    monkeypatch,
    *props,
    numeric_ids = None,
) -> None:
    ids = [0] if numeric_ids is None else numeric_ids
    _not_hip(monkeypatch)
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: list(ids))
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {
            "raw": ",".join(str(i) for i in ids),
            "numeric_ids": list(ids),
            "supports_explicit_gpu_ids": True,
        },
    )
    monkeypatch.setattr(hw, "_cuda_order_matches_smi", lambda: True)
    monkeypatch.setattr(hw, "_torch_get_device_module", lambda: (_torch_module(props), "cuda"))
    monkeypatch.setattr(hw, "_torch_get_physical_gpu_count", lambda: len(props))


def _host_memory(
    monkeypatch,
    total_gb = HOST_TOTAL_GB,
    available_gb = HOST_AVAILABLE_GB,
) -> None:
    monkeypatch.setattr(
        psutil,
        "virtual_memory",
        lambda: types.SimpleNamespace(total = int(total_gb * GIB), available = int(available_gb * GIB)),
    )


def _util_row(
    index = 0,
    ordinal = 0,
    total_gb = N1X_CARVE_OUT_GB,
    used_gb = N1X_USED_GB,
) -> dict:
    """A row exactly as nvidia.py::_build_gpu_metrics emits it."""
    return {
        "index": index,
        "index_kind": "physical",
        "visible_ordinal": ordinal,
        "gpu_utilization_pct": 0.0,
        "temperature_c": 36.0,
        "vram_used_gb": used_gb,
        "vram_total_gb": total_gb,
        "vram_utilization_pct": (
            round(used_gb / total_gb * 100, 1)
            if used_gb is not None and total_gb not in (None, 0)
            else None
        ),
        "power_draw_w": 0.34,
        "power_limit_w": None,
        "power_utilization_pct": None,
    }


def _smi_utilization(
    monkeypatch,
    rows,
    numeric_ids = None,
) -> None:
    ids = [0] if numeric_ids is None else numeric_ids
    monkeypatch.setattr(
        hw,
        "_smi_query",
        lambda *a, **k: {
            "available": True,
            "devices": rows,
            "backend_cuda_visible_devices": None,
            "parent_visible_gpu_ids": list(ids),
            "index_kind": "physical",
        },
    )


def _smi_inventory(monkeypatch, rows) -> None:
    monkeypatch.setattr(nvidia, "_query_gpu_inventory", lambda caller: rows)


def test_a_readable_carve_out_total_is_widened_to_the_pool(monkeypatch):
    """The regression: 7.94 GiB published for a device that can allocate 45.39 GiB."""
    _cuda_host(monkeypatch, _N1XProps())
    _host_memory(monkeypatch)
    _smi_utilization(monkeypatch, [_util_row()])

    device = hw.get_visible_gpu_utilization()["devices"][0]

    assert device["vram_total_gb"] == N1X_POOL_GB


def test_the_free_half_grows_with_the_total(monkeypatch):
    """Free memory must take used from the host counter: memory.used covers only the carve-out."""
    _cuda_host(monkeypatch, _N1XProps())
    _host_memory(monkeypatch)
    _smi_utilization(monkeypatch, [_util_row()])

    device = hw.get_visible_gpu_utilization()["devices"][0]
    free_gb = device["vram_total_gb"] - device["vram_used_gb"]

    # Host used is 54.21 - 42.55 = 11.66, which is larger than the carve-out's 5.73.
    assert device["vram_used_gb"] == pytest.approx(HOST_TOTAL_GB - HOST_AVAILABLE_GB, abs = 0.02)
    # The number the training gate reads. It was 7.94 - 5.73 = 2.21.
    assert free_gb > 30
    assert device["vram_utilization_pct"] == pytest.approx(25.7, abs = 0.5)


def test_a_270m_model_now_fits(monkeypatch):
    """The reported symptom: a 270M model judged unfit at usable_gb=0.3 on a 45 GiB unified pool."""
    from routes.training_vram import _free_vram_by_index

    _cuda_host(monkeypatch, _N1XProps())
    _host_memory(monkeypatch)
    _smi_utilization(monkeypatch, [_util_row()])

    free = _free_vram_by_index(hw.get_visible_gpu_utilization()["devices"])

    assert free[0] > 2.251


def test_the_system_inventory_is_widened_too(monkeypatch):
    """The system inventory must widen too: its probe showed 7.94 GiB where torch reported 45.39 GiB."""
    _cuda_host(monkeypatch, _N1XProps())
    _smi_inventory(
        monkeypatch,
        [{"index": 0, "name": _N1XProps.name, "memory_total_gb": N1X_CARVE_OUT_GB}],
    )

    device = hw.get_backend_visible_gpu_info()["devices"][0]

    assert device["memory_total_gb"] == N1X_POOL_GB
    assert device["unified_memory"] is True
    # The pool IS system memory, so the frontend must count it once, not twice.
    assert device["shared_memory_host_backed_gb"] == N1X_POOL_GB


def test_llama_cpp_prices_the_gguf_fit_against_the_pool(monkeypatch):
    """llama.cpp's nvidia-smi arm wins on this host, so its GGUF fit must be priced against the pool."""
    _integrated_llama_host(monkeypatch)
    _smi_free_total(monkeypatch, [(0, 2256, N1X_CARVE_OUT_MIB)])

    gpus = LlamaCppBackend._get_gpu_memory()

    assert len(gpus) == 1
    idx, free_mib, total_mib = gpus[0]
    assert idx == 0
    assert total_mib == N1X_POOL_MIB
    assert free_mib > 8 * 1024


def test_a_discrete_card_is_byte_identical(monkeypatch):
    """A 4090 answers memory.total correctly and nothing here may second-guess it."""
    _cuda_host(monkeypatch, _DiscreteProps())
    _host_memory(monkeypatch)
    rows = [_util_row(total_gb = 23.99, used_gb = 1.5)]
    before = [dict(row) for row in rows]
    _smi_utilization(monkeypatch, rows)

    devices = hw.get_visible_gpu_utilization()["devices"]

    assert devices == before


def test_a_discrete_inventory_is_byte_identical(monkeypatch):
    _cuda_host(monkeypatch, _DiscreteProps())
    _smi_inventory(
        monkeypatch,
        [{"index": 0, "name": _DiscreteProps.name, "memory_total_gb": 23.99}],
    )

    device = hw.get_backend_visible_gpu_info()["devices"][0]

    assert device["memory_total_gb"] == 23.99
    assert device.get("unified_memory") is not True
    assert device.get("shared_memory") is not True


def test_llama_cpp_leaves_discrete_rows_alone(monkeypatch):
    _discrete_llama_host(monkeypatch)
    _smi_free_total(monkeypatch, [(0, 20000, 24564), (1, 24000, 24564)])

    assert LlamaCppBackend._get_gpu_memory() == [(0, 20000, 24564), (1, 24000, 24564)]


def test_a_larger_cli_total_is_never_shrunk(monkeypatch):
    """Adopt only a larger widened total: a too-small total hides models the device could hold."""

    class _UnderReportingProps(_N1XProps):
        total_memory = 4 * GIB

    _cuda_host(monkeypatch, _UnderReportingProps())
    _host_memory(monkeypatch)
    _smi_utilization(monkeypatch, [_util_row(total_gb = 16.0, used_gb = 2.0)])

    device = hw.get_visible_gpu_utilization()["devices"][0]

    assert device["vram_total_gb"] == 16.0
    assert device["vram_used_gb"] == 2.0


def test_totals_that_agree_within_rounding_are_left_alone(monkeypatch):
    """Totals within MiB rounding stay untouched: nvidia-smi rounds, props.total_memory is exact bytes."""
    _cuda_host(monkeypatch, _N1XProps())
    _host_memory(monkeypatch)
    _smi_utilization(monkeypatch, [_util_row(total_gb = N1X_POOL_GB - 0.01, used_gb = 3.0)])

    device = hw.get_visible_gpu_utilization()["devices"][0]

    assert device["vram_total_gb"] == N1X_POOL_GB - 0.01
    assert device["vram_used_gb"] == 3.0


def test_free_bytes_never_shrink_when_the_host_is_nearly_full(monkeypatch):
    """Widening never lowers the free bytes the driver reported, even when host RAM is nearly full."""
    _cuda_host(monkeypatch, _N1XProps())
    _host_memory(monkeypatch, total_gb = HOST_TOTAL_GB, available_gb = 2.0)
    _smi_utilization(monkeypatch, [_util_row(total_gb = N1X_CARVE_OUT_GB, used_gb = 1.0)])

    device = hw.get_visible_gpu_utilization()["devices"][0]
    free_gb = device["vram_total_gb"] - device["vram_used_gb"]

    assert device["vram_total_gb"] == N1X_POOL_GB
    # The carve-out promised 7.94 - 1.0 = 6.94 GiB and that promise is kept.
    assert free_gb >= N1X_CARVE_OUT_GB - 1.0 - 0.02


def test_llama_cpp_free_never_shrinks(monkeypatch):
    _integrated_llama_host(monkeypatch, avail_mib = 512)
    _smi_free_total(monkeypatch, [(0, 6000, N1X_CARVE_OUT_MIB)])

    _idx, free_mib, total_mib = LlamaCppBackend._get_gpu_memory()[0]

    assert total_mib == N1X_POOL_MIB
    assert free_mib >= 6000


def test_the_npu_row_is_left_unknown(monkeypatch):
    """The NPU row reports N/A memory.total under MCDM, so it stays unknown rather than invented."""
    _cuda_host(monkeypatch, _N1XProps(), numeric_ids = [0, 1])
    _host_memory(monkeypatch)
    _smi_utilization(
        monkeypatch,
        [_util_row(), _util_row(index = 1, ordinal = 1, total_gb = None, used_gb = None)],
        numeric_ids = [0, 1],
    )

    devices = hw.get_visible_gpu_utilization()["devices"]

    assert devices[0]["vram_total_gb"] == N1X_POOL_GB
    assert devices[1]["vram_total_gb"] is None
    assert devices[1]["vram_used_gb"] is None
    assert devices[1]["vram_utilization_pct"] is None


def test_the_npu_cannot_drag_down_the_training_budget(monkeypatch):
    """An unmeasurable row is absent from the free-VRAM map, since a zero would rank as a real device."""
    from routes.training_vram import _free_vram_by_index

    _cuda_host(monkeypatch, _N1XProps(), numeric_ids = [0, 1])
    _host_memory(monkeypatch)
    _smi_utilization(
        monkeypatch,
        [_util_row(), _util_row(index = 1, ordinal = 1, total_gb = None, used_gb = None)],
        numeric_ids = [0, 1],
    )

    free = _free_vram_by_index(hw.get_visible_gpu_utilization()["devices"])

    assert set(free) == {0}


def test_only_the_integrated_device_is_widened_on_a_mixed_host(monkeypatch):
    """A discrete card beside an integrated one keeps its own, correct capacity."""
    _cuda_host(monkeypatch, _N1XProps(), _DiscreteProps(), numeric_ids = [0, 1])
    _host_memory(monkeypatch)
    _smi_utilization(
        monkeypatch,
        [
            _util_row(),
            _util_row(index = 1, ordinal = 1, total_gb = 23.99, used_gb = 1.5),
        ],
        numeric_ids = [0, 1],
    )

    devices = hw.get_visible_gpu_utilization()["devices"]

    assert devices[0]["vram_total_gb"] == N1X_POOL_GB
    assert devices[1]["vram_total_gb"] == 23.99
    assert devices[1]["vram_used_gb"] == 1.5


def test_a_mismatched_device_order_refuses_the_join(monkeypatch):
    """CUDA's FASTEST_FIRST order differs from nvidia-smi's PCI order, so the join is refused."""
    _cuda_host(monkeypatch, _N1XProps(), numeric_ids = [0, 1])
    _host_memory(monkeypatch)
    monkeypatch.setattr(hw, "_cuda_order_matches_smi", lambda: False)
    _smi_utilization(monkeypatch, [_util_row()], numeric_ids = [0, 1])

    device = hw.get_visible_gpu_utilization()["devices"][0]

    assert device["vram_total_gb"] == N1X_CARVE_OUT_GB


def test_a_uuid_mask_joins_on_the_visible_ordinal(monkeypatch):
    """A UUID or MIG mask has no physical ids, so visible_ordinal is the join key to nvidia-smi rows."""
    _cuda_host(monkeypatch, _N1XProps())
    _host_memory(monkeypatch)
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": "GPU-f9df3c40", "numeric_ids": None, "supports_explicit_gpu_ids": False},
    )
    _smi_utilization(monkeypatch, [_util_row()])

    device = hw.get_visible_gpu_utilization()["devices"][0]

    assert device["vram_total_gb"] == N1X_POOL_GB


def test_the_poll_never_pins_a_cuda_context(monkeypatch):
    """The poll must avoid mem_get_info, which pins a CUDA primary context that is never released."""
    _cuda_host(monkeypatch, _N1XProps())
    _host_memory(monkeypatch)
    _smi_utilization(monkeypatch, [_util_row()])

    def _forbidden(*args, **kwargs):
        raise AssertionError("mem_get_info pins a primary context on the /api/system poll")

    monkeypatch.setattr(hw, "trusted_mem_get_info", _forbidden)
    monkeypatch.setattr(hw, "_torch_get_per_device_info", _forbidden)

    assert hw.get_visible_gpu_utilization()["devices"][0]["vram_total_gb"] == N1X_POOL_GB


def test_a_torch_that_cannot_answer_keeps_the_cli_rows(monkeypatch):
    """A CPU-only build, or no torch: the CLI reading stands, unchanged."""
    _cuda_host(monkeypatch, _N1XProps())
    _host_memory(monkeypatch)
    monkeypatch.setattr(hw, "_torch_get_device_module", lambda: (None, None))
    _smi_utilization(monkeypatch, [_util_row()])

    device = hw.get_visible_gpu_utilization()["devices"][0]

    assert device["vram_total_gb"] == N1X_CARVE_OUT_GB
    assert device["vram_used_gb"] == N1X_USED_GB


def test_a_host_memory_probe_failure_still_widens_the_total(monkeypatch):
    """A psutil failure must still widen the total: psutil only feeds the used numerator."""
    _cuda_host(monkeypatch, _N1XProps())

    def _boom():
        raise RuntimeError("no host counters here")

    monkeypatch.setattr(psutil, "virtual_memory", _boom)
    _smi_utilization(monkeypatch, [_util_row()])

    device = hw.get_visible_gpu_utilization()["devices"][0]

    assert device["vram_total_gb"] == N1X_POOL_GB
    # memory.used is carve-out scoped; pairing it with the POOL total would invent free space.
    cli_free_gb = round(N1X_CARVE_OUT_GB - N1X_USED_GB, 2)
    assert device["vram_total_gb"] - device["vram_used_gb"] == pytest.approx(cli_free_gb, abs = 0.01)


def test_the_predicate_is_the_whole_rule():
    """Direct table for the one function every site above shares."""
    # A blank CLI total: the DGX Spark shape, always widened.
    assert hw._integrated_total_is_understated(None, 121.0) is True
    # A carve-out: the N1X shape.
    assert hw._integrated_total_is_understated(N1X_CARVE_OUT_GB, N1X_POOL_GB) is True
    assert hw._integrated_total_is_understated(45.39, 45.39) is False
    assert hw._integrated_total_is_understated(45.39, 45.40) is False
    assert hw._integrated_total_is_understated(45.39, 8.0) is False
    assert hw._integrated_total_is_understated(8.0, None) is False
    assert hw._integrated_total_is_understated(8.0, 0) is False


def _llama_common(monkeypatch, avail_mib):
    monkeypatch.setattr(LlamaCppBackend, "_INTEGRATED_CUDA_IDS", {})
    monkeypatch.setattr(LlamaCppBackend, "_INTEGRATED_CUDA_POOL_MIB", {})
    monkeypatch.setattr(LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: None))
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b: False))
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: avail_mib)
    )
    monkeypatch.setattr(LlamaCppBackend, "_cgroup_available_memory_mib", staticmethod(lambda: None))
    monkeypatch.setattr(LlamaCppBackend, "_visible_devices_mask", staticmethod(lambda name: None))
    monkeypatch.setattr(
        LlamaCppBackend, "_resolve_visible_physical_ids", staticmethod(lambda: None)
    )
    # None means NO MASK, so clear the env too; ordering mirrors main.py (PCI_BUS_ID).
    for _var in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(_var, raising = False)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def _integrated_llama_host(monkeypatch, avail_mib = int(HOST_AVAILABLE_GB * 1024)):
    _llama_common(monkeypatch, avail_mib)
    monkeypatch.setattr(LlamaCppBackend, "_integrated_cuda_gpu_ids", staticmethod(lambda: {0}))
    monkeypatch.setattr(
        LlamaCppBackend,
        "_integrated_cuda_pool_total_mib",
        staticmethod(lambda: {0: N1X_POOL_MIB}),
    )


def _discrete_llama_host(monkeypatch, avail_mib = 32000):
    _llama_common(monkeypatch, avail_mib)
    monkeypatch.setattr(LlamaCppBackend, "_integrated_cuda_gpu_ids", staticmethod(lambda: set()))
    monkeypatch.setattr(
        LlamaCppBackend, "_integrated_cuda_pool_total_mib", staticmethod(lambda: {})
    )


def _smi_free_total(monkeypatch, rows):
    """Stand in for `nvidia-smi --query-gpu=index,memory.free,memory.total`."""
    stdout = "\n".join(f"{idx}, {free}, {total}" for idx, free, total in rows)
    monkeypatch.setattr(
        "core.inference.llama_cpp.subprocess.run",
        lambda *a, **k: types.SimpleNamespace(returncode = 0, stdout = stdout, stderr = ""),
    )


def test_a_cgroup_ceiling_survives_the_never_shrink_floor(monkeypatch):
    """Inside a cgroup the never-shrink floor must respect memory.max, not republish the carve-out."""
    _integrated_llama_host(monkeypatch, avail_mib = 40000)
    monkeypatch.setattr(LlamaCppBackend, "_cgroup_available_memory_mib", staticmethod(lambda: 2048))

    rows = LlamaCppBackend._widen_integrated_cuda_rows([(0, 6000, N1X_CARVE_OUT_MIB)])

    assert rows[0][1] <= 2048, rows
    # A cgroup-bound row publishes no total, which is the shared-pool marker.
    assert rows[0][2] == 0


def test_an_unmappable_mask_refuses_the_join_under_any_ordering(monkeypatch):
    """A UUID or MIG mask makes ordinals unjoinable, so the widening is refused rather than guessed."""
    _llama_common(monkeypatch, avail_mib = 43000)
    # A UUID mask cannot be parsed, so CLI rows are never filtered to match it.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-deadbeef-0000-0000-0000-000000000003")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setattr(LlamaCppBackend, "_integrated_cuda_gpu_ids", staticmethod(lambda: {1}))
    monkeypatch.setattr(
        LlamaCppBackend,
        "_integrated_cuda_pool_total_mib",
        staticmethod(lambda: {1: N1X_POOL_MIB}),
    )
    smi_rows = [(0, 2256, N1X_CARVE_OUT_MIB), (1, 20000, 24564)]

    assert LlamaCppBackend._widen_integrated_cuda_rows(list(smi_rows)) == smi_rows


def test_a_blank_total_is_still_filled_on_a_discrete_card(monkeypatch):
    """Blank totals on discrete cards are still filled, since a blank row forces the torch fallback."""
    _cuda_host(monkeypatch, _DiscreteProps())
    devices = [{"index": 0, "visible_ordinal": 0, "memory_total_gb": None}]

    complete = hw._repair_smi_visible_devices(devices, [0])

    assert complete is True
    assert devices[0]["memory_total_gb"] == 24.0
    assert devices[0].get("unified_memory") is not True
    assert devices[0].get("shared_memory") is not True


def test_a_numeric_mask_under_fastest_first_refuses_the_join(monkeypatch):
    """Numeric masks use CUDA indices, not PCI ones; under FASTEST_FIRST the join is refused."""
    _llama_common(monkeypatch, avail_mib = 43000)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    monkeypatch.setattr(
        LlamaCppBackend, "_resolve_visible_physical_ids", staticmethod(lambda: [0, 1])
    )
    monkeypatch.setattr(LlamaCppBackend, "_integrated_cuda_gpu_ids", staticmethod(lambda: {1}))
    monkeypatch.setattr(
        LlamaCppBackend,
        "_integrated_cuda_pool_total_mib",
        staticmethod(lambda: {1: N1X_POOL_MIB}),
    )
    smi_rows = [(0, 2256, N1X_CARVE_OUT_MIB), (1, 20000, 24564)]

    assert LlamaCppBackend._widen_integrated_cuda_rows(list(smi_rows)) == smi_rows

    # PCI_BUS_ID makes the same join provable, so the integrated row widens.
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    widened = LlamaCppBackend._widen_integrated_cuda_rows(list(smi_rows))
    assert widened[0] == smi_rows[0]
    assert widened[1][2] == N1X_POOL_MIB


def test_the_widened_utilization_is_capped_by_the_cgroup(monkeypatch):
    """psutil reads host-wide counters in most containers, and unified-memory
    allocations are charged to memory.max, so the published free bytes must not exceed
    what this process can still charge."""
    _cuda_host(monkeypatch, _N1XProps())
    _host_memory(monkeypatch)
    monkeypatch.setattr(LlamaCppBackend, "_cgroup_available_memory_mib", staticmethod(lambda: 2048))
    utilization = {
        "devices": [
            {
                "index": 0,
                "visible_ordinal": 0,
                "vram_total_gb": N1X_CARVE_OUT_GB,
                "vram_used_gb": N1X_USED_GB,
                "vram_utilization_pct": 72.2,
            }
        ]
    }

    hw._reconcile_cuda_integrated_memory(utilization, [0])
    device = utilization["devices"][0]

    assert device["vram_total_gb"] == N1X_POOL_GB
    free_gb = device["vram_total_gb"] - device["vram_used_gb"]
    assert free_gb == pytest.approx(2.0, abs = 0.01), device


def test_one_surviving_row_is_not_proof_of_one_gpu(monkeypatch):
    """One surviving nvidia-smi row does not prove one GPU: dropped rows may hide discrete cards."""
    _llama_common(monkeypatch, avail_mib = 43000)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    monkeypatch.setattr(LlamaCppBackend, "_integrated_cuda_gpu_ids", staticmethod(lambda: {1}))
    monkeypatch.setattr(
        LlamaCppBackend,
        "_integrated_cuda_pool_total_mib",
        staticmethod(lambda: {1: N1X_POOL_MIB}),
    )
    # Index 1 here is the DISCRETE card; the integrated part is the row that dropped.
    smi_rows = [(1, 20000, 24564)]

    assert LlamaCppBackend._widen_integrated_cuda_rows(list(smi_rows)) == smi_rows


def test_the_nvml_fallback_is_widened_too(monkeypatch):
    """The nvidia-smi fallback reports the same carve-out as NVML, so it must be widened too."""
    _integrated_llama_host(monkeypatch)
    monkeypatch.setattr(
        "core.inference.llama_cpp.subprocess.run",
        lambda *a, **k: types.SimpleNamespace(returncode = 1, stdout = "", stderr = "no smi"),
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "_get_gpu_memory_nvml",
        staticmethod(lambda: [(0, 2256, N1X_CARVE_OUT_MIB)]),
    )

    rows = LlamaCppBackend._get_gpu_memory()

    assert rows[0][2] == N1X_POOL_MIB, rows
    assert rows[0][1] > N1X_CARVE_OUT_MIB, rows
