# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An integrated CUDA SoC whose nvidia-smi ANSWERS, with the wrong number.

test_dgx_spark_gpu_inventory.py covers the shape where nvidia-smi answers ``[N/A]`` for
memory.total and the capacity had to be filled in. This file covers the other shape of
the same fault, measured on an NVIDIA RTX Spark N1X (Blackwell, compute capability 12.1,
unified memory) running Windows on ARM64:

    nvidia-smi memory.total       8128 MiB     the dedicated carve-out ONLY
    torch props.total_memory     46477 MiB     what CUDA can actually allocate
    torch mem_get_info()[1]      46477 MiB     agrees with props
    llama.cpp --list-devices     46477 MiB     agrees with props
    props.is_integrated              1

An under-report of about 5.7x, and a readable number rather than a blank, so every
"fill in what the CLI could not answer" repair skipped it. A 270M model was judged not
to fit ("usable_gb=0.3 required_gb=2.251"), and speculative decoding was refused for
wanting "7.9 GB of a 6.9 GB budget" on a machine with 45 GiB.

Hermetic: torch, nvidia-smi and host memory are stubbed, so these run anywhere. The
figures above are real, and are used as the stub values.
"""

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


# ── the fault itself ─────────────────────────────────────────────────────────


def test_a_readable_carve_out_total_is_widened_to_the_pool(monkeypatch):
    """The regression: 7.94 GiB published for a device that can allocate 45.39 GiB."""
    _cuda_host(monkeypatch, _N1XProps())
    _host_memory(monkeypatch)
    _smi_utilization(monkeypatch, [_util_row()])

    device = hw.get_visible_gpu_utilization()["devices"][0]

    assert device["vram_total_gb"] == N1X_POOL_GB


def test_the_free_half_grows_with_the_total(monkeypatch):
    """A widened total paired with the carve-out's own used figure is still wrong.

    memory.used is scoped to the carve-out, so it is a floor on pool occupancy rather
    than a measure of it. The host counter spans the same ground the widened total does,
    which is the rule _rocm_windows_unified_used_bytes states for the AMD twin.
    """
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
    """The reported symptom, end to end through the helper the training gate uses.

    "Falling back to all visible GPUs; model may not fit:
     model=unsloth/gemma-3-270m-it usable_gb=0.3 required_gb=2.251"
    """
    from routes.training_vram import _free_vram_by_index

    _cuda_host(monkeypatch, _N1XProps())
    _host_memory(monkeypatch)
    _smi_utilization(monkeypatch, [_util_row()])

    free = _free_vram_by_index(hw.get_visible_gpu_utilization()["devices"])

    assert free[0] > 2.251


def test_the_system_inventory_is_widened_too(monkeypatch):
    """Settings > System reads get_backend_visible_gpu_info, a different probe.

    It showed "7.94 GiB total" for the card while Settings > About, which reads torch,
    showed 45.39 GiB for the same machine in the same session.
    """
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
    """llama.cpp has its own probe, and its nvidia-smi arm wins on this host.

    "Speculative decoding disabled for this load: the model fits in VRAM at context
     83968 but its drafter does not (needs 7.9 GB of a 6.9 GB budget, on any GPU subset)"
    """
    _integrated_llama_host(monkeypatch)
    _smi_free_total(monkeypatch, [(0, 2256, N1X_CARVE_OUT_MIB)])

    gpus = LlamaCppBackend._get_gpu_memory()

    assert len(gpus) == 1
    idx, free_mib, total_mib = gpus[0]
    assert idx == 0
    assert total_mib == N1X_POOL_MIB
    # It was 2256 MiB. The drafter needed 7.9 GB.
    assert free_mib > 8 * 1024


# ── the constraint: never shrink, never touch a discrete card ────────────────


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
    """The rule the ROCm path already encodes: adopt only a LARGER total.

    A driver that under-reports props.total_memory while the CLI is right must not cost
    the device its capacity, because a too-small total hides models the device can hold.
    """

    class _UnderReportingProps(_N1XProps):
        total_memory = 4 * GIB

    _cuda_host(monkeypatch, _UnderReportingProps())
    _host_memory(monkeypatch)
    _smi_utilization(monkeypatch, [_util_row(total_gb = 16.0, used_gb = 2.0)])

    device = hw.get_visible_gpu_utilization()["devices"][0]

    assert device["vram_total_gb"] == 16.0
    assert device["vram_used_gb"] == 2.0


def test_totals_that_agree_within_rounding_are_left_alone(monkeypatch):
    """props.total_memory is exact bytes; nvidia-smi rounds to whole MiB.

    An integrated part whose CLI total is already pool-scoped must not churn its own
    figures over the difference.
    """
    _cuda_host(monkeypatch, _N1XProps())
    _host_memory(monkeypatch)
    _smi_utilization(monkeypatch, [_util_row(total_gb = N1X_POOL_GB - 0.01, used_gb = 3.0)])

    device = hw.get_visible_gpu_utilization()["devices"][0]

    assert device["vram_total_gb"] == N1X_POOL_GB - 0.01
    assert device["vram_used_gb"] == 3.0


def test_free_bytes_never_shrink_when_the_host_is_nearly_full(monkeypatch):
    """The widening must not cost a device free bytes the driver already vouched for.

    A host with 2 GiB of 54 GiB available would otherwise publish a 45 GiB pool with
    less free memory than the 8 GiB carve-out it replaced.
    """
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


# ── the NPU, which must stay honestly unknown ────────────────────────────────


def test_the_npu_row_is_left_unknown(monkeypatch):
    """nvidia-smi enumerates an "NVIDIA NPU" as GPU 1 on this machine.

    It runs under MCDM rather than WDDM and ``nvidia-smi -q -i 1`` genuinely answers
    ``FB Memory Usage: Total: N/A``. torch does not enumerate it at all. There is no
    number to publish for it and inventing one would be worse than a blank.
    """
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
    """An unmeasurable row must be absent from the free-VRAM map, not present as zero.

    Present as a zero it would rank as a real device with no memory, and a multi-GPU
    split would be sized against a card that cannot hold anything.
    """
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


# ── mixed hosts and index spaces ─────────────────────────────────────────────


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
    """CUDA enumerates FASTEST_FIRST while nvidia-smi reports PCI order.

    Joining them anyway attaches one card's capacity to another card's row. The widening
    uses the same gate the rest of the module already applies, so it declines instead.
    """
    _cuda_host(monkeypatch, _N1XProps(), numeric_ids = [0, 1])
    _host_memory(monkeypatch)
    monkeypatch.setattr(hw, "_cuda_order_matches_smi", lambda: False)
    _smi_utilization(monkeypatch, [_util_row()], numeric_ids = [0, 1])

    device = hw.get_visible_gpu_utilization()["devices"][0]

    assert device["vram_total_gb"] == N1X_CARVE_OUT_GB


def test_a_uuid_mask_joins_on_the_visible_ordinal(monkeypatch):
    """A UUID or MIG mask resolves to numeric_ids=None and has no physical ids.

    nvidia.py resolves the mask itself and returns rows in its order, which is the order
    torch enumerates, so visible_ordinal is the join.
    """
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


# ── cost: no primary context on a polling path ───────────────────────────────


def test_the_poll_never_pins_a_cuda_context(monkeypatch):
    """mem_get_info attaches a primary context the process never gives back.

    Measured at 116 MiB on the N1X and ~612 MiB elsewhere. get_device_properties
    attaches none (measured at 0 MiB), and this whole repair is built on that
    difference, so nothing here may reach for the other call.
    """
    _cuda_host(monkeypatch, _N1XProps())
    _host_memory(monkeypatch)
    _smi_utilization(monkeypatch, [_util_row()])

    def _forbidden(*args, **kwargs):
        raise AssertionError("mem_get_info pins a primary context on the /api/system poll")

    monkeypatch.setattr(hw, "trusted_mem_get_info", _forbidden)
    monkeypatch.setattr(hw, "_torch_get_per_device_info", _forbidden)

    assert hw.get_visible_gpu_utilization()["devices"][0]["vram_total_gb"] == N1X_POOL_GB


# ── hosts that cannot answer at all ──────────────────────────────────────────


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
    """psutil is the numerator's source, not the total's.

    Losing it must not cost the device the capacity, which is the half that decides
    whether a model is offered at all.
    """
    _cuda_host(monkeypatch, _N1XProps())

    def _boom():
        raise RuntimeError("no host counters here")

    monkeypatch.setattr(psutil, "virtual_memory", _boom)
    _smi_utilization(monkeypatch, [_util_row()])

    device = hw.get_visible_gpu_utilization()["devices"][0]

    assert device["vram_total_gb"] == N1X_POOL_GB
    # memory.used is scoped to the carve-out, so pairing it with the POOL total would
    # advertise the whole difference as free on no pool-scoped evidence. The capacity
    # still widens, which is the half that decides whether a model is offered at all;
    # the budget stays the one the CLI vouched for.
    cli_free_gb = round(N1X_CARVE_OUT_GB - N1X_USED_GB, 2)
    assert device["vram_total_gb"] - device["vram_used_gb"] == pytest.approx(cli_free_gb, abs = 0.01)


def test_the_predicate_is_the_whole_rule():
    """Direct table for the one function every site above shares."""
    # A blank CLI total: the DGX Spark shape, always widened.
    assert hw._integrated_total_is_understated(None, 121.0) is True
    # A carve-out: the N1X shape.
    assert hw._integrated_total_is_understated(N1X_CARVE_OUT_GB, N1X_POOL_GB) is True
    # Equal, and within rounding of equal: nothing to do.
    assert hw._integrated_total_is_understated(45.39, 45.39) is False
    assert hw._integrated_total_is_understated(45.39, 45.40) is False
    # Smaller: never adopted.
    assert hw._integrated_total_is_understated(45.39, 8.0) is False
    # Nothing to adopt.
    assert hw._integrated_total_is_understated(8.0, None) is False
    assert hw._integrated_total_is_understated(8.0, 0) is False


# ── llama.cpp host stubs ─────────────────────────────────────────────────────


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
    # `_resolve_visible_physical_ids` returning None means NO MASK here, so the env has
    # to say the same or the runner's own CUDA_VISIBLE_DEVICES is read as one. The
    # ordering mirrors main.py:19, which sets PCI_BUS_ID on import.
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
    """Inside a container the floor must not hand back memory `memory.max` forbids.

    The free half is floored at the reading nvidia-smi already vouched for, which is
    right on bare metal and wrong inside a cgroup: allocations on a unified part are
    charged to it, so republishing the larger carve-out figure prices a fit against
    memory the kernel will not give and the child is killed rather than offloaded.
    """
    _integrated_llama_host(monkeypatch, avail_mib = 40000)
    monkeypatch.setattr(LlamaCppBackend, "_cgroup_available_memory_mib", staticmethod(lambda: 2048))

    rows = LlamaCppBackend._widen_integrated_cuda_rows([(0, 6000, N1X_CARVE_OUT_MIB)])

    assert rows[0][1] <= 2048, rows
    # A cgroup-bound row publishes no total, which is the shared-pool marker.
    assert rows[0][2] == 0


def test_an_unmappable_mask_refuses_the_join_under_any_ordering(monkeypatch):
    """A UUID or MIG mask leaves torch ordinals and nvidia-smi indices unjoinable.

    `_resolve_visible_physical_ids()` returns None there, so `_integrated_cuda_gpu_ids`
    falls back to the ordinal, while `_visible_devices_mask` also returns None and the
    CLI rows are NOT filtered to match. Re-pricing row 1 because ordinal 1 is integrated
    would advertise a discrete card with a system-RAM-sized pool.
    """
    _llama_common(monkeypatch, avail_mib = 43000)
    # A real UUID mask: `_resolve_visible_physical_ids` cannot parse it, and neither can
    # `_visible_devices_mask`, so the CLI rows are never filtered to match. PCI_BUS_ID
    # ordering does not help, and main.py sets it by default, so the refusal cannot be
    # left to the ordering alone.
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
    """MIG and vGPU rows read [N/A] for memory.total on a card that is not integrated.

    Filling those was this function's original job and is not part of the widening; a
    row left blank sends the whole response down get_backend_visible_gpu_info's torch
    fallback instead of publishing the nvidia-smi rows it already has.
    """
    _cuda_host(monkeypatch, _DiscreteProps())
    devices = [{"index": 0, "visible_ordinal": 0, "memory_total_gb": None}]

    complete = hw._repair_smi_visible_devices(devices, [0])

    assert complete is True
    assert devices[0]["memory_total_gb"] == 24.0
    assert devices[0].get("unified_memory") is not True
    assert devices[0].get("shared_memory") is not True


def test_a_numeric_mask_under_fastest_first_refuses_the_join(monkeypatch):
    """A numeric CUDA_VISIBLE_DEVICES carries CUDA's indices, not PCI ones.

    CUDA_DEVICE_ORDER defaults to FASTEST_FIRST, which pins only device 0 and leaves the
    rest unspecified, while nvidia-smi numbers by the kernel's NVML enumeration. Joining
    the two spaces can hand a discrete card the shared pool, so the widening is refused
    unless the ordering is provably PCI_BUS_ID.
    """
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
