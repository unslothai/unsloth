# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The /api/system poll must not pin a CUDA/HIP primary context.

The frontend polls GET /api/system every 5s from the root-mounted floating
monitor and every 3s from Settings -> Resources. Both land on
get_backend_visible_gpu_info and get_visible_gpu_utilization. Where nvidia-smi is
absent (ROCm, or any host without it on PATH) those used to reach
torch.cuda.mem_get_info, which attaches a primary context worth ~612 MiB on this
class of GPU and is never released while the process lives. An idle Studio would
therefore lose that memory to telemetry alone.

get_device_properties answers name and total capacity with no context, and it
returns the same total mem_get_info does, so the inventory half of the poll is
free. These tests pin that: the honest check is a FRESH process, because a
context, once created, is never given back and any earlier test in the session
would mask the regression.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import textwrap
import types
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))


def _maybe_stub(name: str, builder):
    # Stub only if the real module is missing, so we never shadow it for later tests.
    try:
        importlib.import_module(name)
    except ImportError:
        sys.modules[name] = builder()


def _build_loggers_stub():
    m = types.ModuleType("loggers")
    m.get_logger = lambda name: __import__("logging").getLogger(name)
    return m


def _build_structlog_stub():
    m = types.ModuleType("structlog")
    m.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
    return m


_maybe_stub("loggers", _build_loggers_stub)
_maybe_stub("structlog", _build_structlog_stub)

import pytest

import utils.hardware.hardware as hw  # noqa: E402


def _has_cuda() -> bool:
    try:
        import torch
    except Exception:
        return False
    try:
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


needs_cuda = pytest.mark.skipif(
    not _has_cuda(), reason = "needs a real CUDA/HIP device to observe context creation"
)


# ========== Fresh-process context assertions ==========

# Runs in a child interpreter: memory_reserved(i) needs no context itself (it reads
# the allocator's stats, which are empty before one exists), so it is a safe probe.
_CHILD = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {backend!r})
    import torch

    assert sum(torch.cuda.memory_reserved(i) for i in range(torch.cuda.device_count())) == 0

    import utils.hardware.hardware as hw
    {call}

    leaked = [i for i in range(torch.cuda.device_count()) if torch.cuda.memory_reserved(i)]
    print("LEAKED:" + repr(leaked))
    print("CONTEXT:" + repr(_probe_context()))
    """
)

# Per-PID nvidia-smi is the ground truth: a primary context shows up as this
# process holding memory even with nothing allocated. Absent nvidia-smi we fall
# back to reporting None, and the test then relies on the allocator probe alone.
_PROBE = textwrap.dedent(
    """
    import os, subprocess
    def _probe_context():
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                 "--format=csv,noheader,nounits"],
                capture_output = True, text = True, timeout = 60,
            )
        except Exception:
            return None
        if out.returncode != 0:
            return None
        me = str(os.getpid())
        for line in out.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2 and parts[0] == me:
                return int(parts[1])
        return 0
    """
)


def _run_child(call: str):
    """Run ``call`` in a fresh interpreter; return (leaked_devices, mib_or_None)."""
    src = _PROBE + _CHILD.format(backend = str(_BACKEND_DIR), call = call)
    proc = subprocess.run([sys.executable, "-c", src], capture_output = True, text = True, timeout = 600)
    assert proc.returncode == 0, f"child failed:\n{proc.stdout}\n{proc.stderr}"
    leaked, mib = None, None
    for line in proc.stdout.splitlines():
        if line.startswith("LEAKED:"):
            leaked = eval(line[len("LEAKED:") :])
        elif line.startswith("CONTEXT:"):
            mib = eval(line[len("CONTEXT:") :])
    assert leaked is not None, proc.stdout
    return leaked, mib


@needs_cuda
def test_backend_visible_gpu_info_creates_no_context():
    leaked, mib = _run_child("hw.get_backend_visible_gpu_info()")
    assert leaked == []
    if mib is not None:
        assert mib == 0, f"the visibility poll pinned {mib} MiB of primary context"


@needs_cuda
def test_clear_gpu_cache_creates_no_context_when_nothing_reserved():
    leaked, mib = _run_child("hw.clear_gpu_cache()")
    assert leaked == []
    if mib is not None:
        assert mib == 0, f"clear_gpu_cache pinned {mib} MiB of primary context"


@needs_cuda
def test_the_probe_would_catch_a_regression():
    # Guards the two tests above: prove mem_get_info really does pin a context on
    # this host, so a green run means the poll avoided it rather than that nothing
    # could have created one.
    leaked, mib = _run_child("__import__('torch').cuda.mem_get_info(0)")
    if mib is None:
        pytest.skip("no per-PID nvidia-smi accounting available to observe the context")
    assert mib > 0, "mem_get_info created no context here, so the assertions above prove nothing"


def _summary_torch(*, properties_total, driver_free, driver_total):
    props = types.SimpleNamespace(total_memory = properties_total, name = "Test GPU")

    def _mem_get_info(_ordinal):
        if driver_free is None:
            raise AssertionError("the native probe answered; mem_get_info would pin a context")
        return driver_free, driver_total

    cuda = types.SimpleNamespace(
        current_device = lambda: 0,
        get_device_properties = lambda _ordinal: props,
        memory_allocated = lambda _ordinal: 1 << 30,
        memory_reserved = lambda _ordinal: 2 << 30,
        mem_get_info = _mem_get_info,
    )
    return types.SimpleNamespace(cuda = cuda), props


def test_gpu_summary_prefers_context_free_driver_memory(monkeypatch):
    gib = 1 << 30
    torch_stub, _props = _summary_torch(
        properties_total = 16 * gib,
        driver_free = None,
        driver_total = 16 * gib,
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": None, "numeric_ids": [0], "supports_explicit_gpu_ids": True},
    )
    monkeypatch.setattr(
        hw,
        "_smi_query",
        lambda *a, **k: {
            "available": True,
            "devices": [{"visible_ordinal": 0, "vram_used_gb": 7.0, "vram_total_gb": 16.0}],
        },
    )

    summary = hw.get_gpu_summary()

    assert summary == {"gpu_name": "Test GPU", "vram_total_gb": 16.0, "vram_free_gb": 9.0}


def test_gpu_summary_pairs_rocm_apu_free_with_driver_total(monkeypatch):
    gib = 1 << 30
    torch_stub, _props = _summary_torch(
        properties_total = 8 * gib,
        driver_free = 98 * gib,
        driver_total = 100 * gib,
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw, "_rocm_props_total_is_carve_out", lambda _props: True)
    monkeypatch.setattr(
        hw,
        "_smi_query",
        lambda *a, **k: pytest.fail("an APU needs hipMemGetInfo's matching GTT total"),
    )
    # No WDDM counter reading available: free falls back to hipMemGetInfo's.
    monkeypatch.setattr(hw, "_rocm_windows_unified_used_bytes", lambda: None)

    summary = hw.get_gpu_summary()

    assert summary == {"gpu_name": "Test GPU", "vram_total_gb": 100.0, "vram_free_gb": 98.0}


def test_gpu_summary_apu_prefers_wddm_counters_over_process_local_hip(monkeypatch):
    # hipMemGetInfo is process-local on Windows WDDM, so an APU takes its GTT
    # *total* from HIP but its *used* from the counters. Driver says 98 free of
    # 100; the counters say 40 GiB is in use across all processes -> 60 free.
    gib = 1 << 30
    torch_stub, _props = _summary_torch(
        properties_total = 8 * gib,
        driver_free = 98 * gib,
        driver_total = 100 * gib,
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw, "_rocm_props_total_is_carve_out", lambda _props: True)
    monkeypatch.setattr(hw, "_rocm_props_are_positively_unified", lambda _props: True)
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        hw,
        "_smi_query",
        lambda *a, **k: pytest.fail("an APU must not take its total from the vendor CLI"),
    )
    monkeypatch.setattr(hw, "_rocm_windows_unified_used_bytes", lambda: 40.0 * gib)

    summary = hw.get_gpu_summary()

    # Total still HIP's, free now derived from the counters against that total.
    assert summary == {"gpu_name": "Test GPU", "vram_total_gb": 100.0, "vram_free_gb": 60.0}


def test_context_free_rocm_smi_translates_hip_ordinals(monkeypatch):
    from utils.hardware import amd

    gib = 1 << 30
    monkeypatch.setattr(hw, "IS_ROCM", True)
    # The visibility vars are read from the real environment on the way through,
    # so a runner that exports one (a CI job pinning a GPU beside this test) can
    # change the outcome. Its siblings already isolate; this one did not.
    for name in (
        "GPU_DEVICE_ORDINAL",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": None, "numeric_ids": [0, 1], "supports_explicit_gpu_ids": True},
    )
    monkeypatch.setattr(amd, "get_hip_id_by_gpu_index", lambda: {0: 1, 1: 0})
    monkeypatch.setattr(hw, "_torch_get_physical_gpu_count", lambda: 2)

    # amd-smi GPU 1 is HIP 0 and owns the 96 GiB pool. Passing HIP ids through
    # unchanged would instead publish GPU 0's 1 GiB free for torch device 0.
    readings = {0: (15.0, 16.0), 1: (2.0, 96.0)}

    def _smi(_name, visible_ids, **_kwargs):
        return {
            "available": True,
            "devices": [
                {
                    "index": smi_id,
                    "visible_ordinal": ordinal,
                    "vram_used_gb": readings[smi_id][0],
                    "vram_total_gb": readings[smi_id][1],
                }
                for ordinal, smi_id in enumerate(visible_ids)
            ],
        }

    monkeypatch.setattr(hw, "_smi_query", _smi)

    assert hw._context_free_cuda_memory_info(0, 96 * gib) == 94 * gib


def test_context_free_rocm_smi_declines_without_an_id_mapping(monkeypatch):
    from utils.hardware import amd

    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": None, "numeric_ids": [0, 1], "supports_explicit_gpu_ids": True},
    )
    monkeypatch.setattr(amd, "get_hip_id_by_gpu_index", lambda: None)
    monkeypatch.setattr(amd, "get_physical_gpu_count", lambda: 2)
    monkeypatch.setattr(
        hw,
        "_smi_query",
        lambda *a, **k: pytest.fail("an unproven amd-smi/HIP mapping must be declined"),
    )

    assert hw._context_free_cuda_memory_info(0, 16 << 30) is None


def test_context_free_rocm_smi_declines_stacked_visibility_masks(monkeypatch):
    from utils.hardware import amd

    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw.sys, "platform", "linux")
    monkeypatch.setattr(hw.platform, "system", lambda: "Darwin")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1,2")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": "0", "numeric_ids": [0], "supports_explicit_gpu_ids": True},
    )
    monkeypatch.setattr(
        amd,
        "get_hip_id_by_gpu_index",
        lambda: pytest.fail("stacked masks have no global HIP mapping"),
    )
    monkeypatch.setattr(
        hw,
        "_smi_query",
        lambda *a, **k: pytest.fail("stacked masks must decline amd-smi telemetry"),
    )

    assert hw._context_free_cuda_memory_info(0, 24 << 30) is None


def test_context_free_rocm_smi_declines_gpu_device_ordinal(monkeypatch):
    from utils.hardware import amd

    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw.platform, "system", lambda: "Darwin")
    monkeypatch.setenv("GPU_DEVICE_ORDINAL", "1")
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": None, "numeric_ids": [0, 1], "supports_explicit_gpu_ids": True},
    )
    monkeypatch.setattr(
        amd,
        "get_hip_id_by_gpu_index",
        lambda: pytest.fail("GPU_DEVICE_ORDINAL makes the global mapping unusable"),
    )
    monkeypatch.setattr(
        hw,
        "_smi_query",
        lambda *a, **k: pytest.fail("an external GPU ordinal filter must decline amd-smi"),
    )

    assert hw._context_free_cuda_memory_info(0, 24 << 30) is None


def test_context_free_rocm_smi_declines_inventory_count_mismatch(monkeypatch):
    from utils.hardware import amd

    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw.platform, "system", lambda: "Darwin")
    for name in (
        "GPU_DEVICE_ORDINAL",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": None, "numeric_ids": [0, 1], "supports_explicit_gpu_ids": True},
    )
    monkeypatch.setattr(amd, "get_hip_id_by_gpu_index", lambda: {0: 0, 1: 1})
    monkeypatch.setattr(hw, "_torch_get_physical_gpu_count", lambda: 1)
    monkeypatch.setattr(
        hw,
        "_smi_query",
        lambda *a, **k: pytest.fail("amd-smi and HIP inventory counts must agree"),
    )

    assert hw._context_free_cuda_memory_info(0, 24 << 30) is None


def test_context_free_nvidia_smi_resolves_a_uuid_mask(monkeypatch):
    from utils.hardware import nvidia

    gib = 1 << 30
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {
            "raw": "GPU-bbbbbbbb",
            "numeric_ids": None,
            "supports_explicit_gpu_ids": False,
        },
    )

    def _run(command, **_kwargs):
        assert "--query-gpu=index,uuid,utilization.gpu,temperature.gpu," in command[1]
        return types.SimpleNamespace(
            returncode = 0,
            stdout = "\n".join(
                [
                    "0, GPU-aaaaaaaa-1111-2222-3333-444444444444, 5, 30, 15360, 16384, 100, 300",
                    "1, GPU-bbbbbbbb-1111-2222-3333-444444444444, 7, 31, 2048, 98304, 120, 350",
                ]
            ),
        )

    monkeypatch.setattr(nvidia.subprocess, "run", _run)

    assert hw._context_free_cuda_memory_info(0, 96 * gib) == 94 * gib


def test_nvidia_uuid_mask_keeps_public_indices_relative(monkeypatch):
    from utils.hardware import nvidia

    def _run(_command, **_kwargs):
        return types.SimpleNamespace(
            returncode = 0,
            stdout = "\n".join(
                [
                    "0, GPU-aaaaaaaa-1111-2222-3333-444444444444, 5, 30, 15360, 16384, 100, 300",
                    "1, GPU-bbbbbbbb-1111-2222-3333-444444444444, 7, 31, 2048, 98304, 120, 350",
                ]
            ),
        )

    monkeypatch.setattr(nvidia.subprocess, "run", _run)

    result = nvidia.get_visible_gpu_utilization(
        None,
        "GPU-bbbbbbbb,GPU-aaaaaaaa",
    )

    assert result["index_kind"] == "relative"
    assert result["parent_visible_gpu_ids"] == []
    assert {device["visible_ordinal"]: device["index"] for device in result["devices"]} == {
        0: 0,
        1: 1,
    }
    # The mask asks for the b-card first, so ordinal 0 must carry the b-card's
    # 2048 MiB, not the a-card's 15360. List position has to agree with it too:
    # nvidia-smi emitted these rows the other way round.
    assert [device["vram_used_gb"] for device in result["devices"]] == [2.0, 15.0]
    assert [device["visible_ordinal"] for device in result["devices"]] == [0, 1]


def _nvidia_smi_reading(monkeypatch, used_gb, driver_total_gb):
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": None, "numeric_ids": [0], "supports_explicit_gpu_ids": True},
    )
    monkeypatch.setattr(
        hw,
        "_smi_query",
        lambda *a, **k: {
            "available": True,
            "devices": [
                {"visible_ordinal": 0, "vram_used_gb": used_gb, "vram_total_gb": driver_total_gb}
            ],
        },
    )


def test_context_free_free_excludes_the_reserved_framebuffer(monkeypatch):
    # nvidia-smi reports reserved separately from used, and torch's total already
    # excludes it. Subtracting used from the DRIVER total would hand back that
    # reservation as free: 100 - 2 = 98 where only 97.5 is allocatable.
    gib = 1 << 30
    _nvidia_smi_reading(monkeypatch, used_gb = 2.0, driver_total_gb = 100.0)
    assert hw._context_free_cuda_memory_info(0, round(99.5 * gib)) == round(97.5 * gib)


def test_context_free_free_is_zero_on_a_full_card(monkeypatch):
    # The reservation is the whole margin here: a card with every allocatable byte
    # taken must not be advertised as having the reserved 0.5 GiB free.
    gib = 1 << 30
    _nvidia_smi_reading(monkeypatch, used_gb = 99.5, driver_total_gb = 100.0)
    assert hw._context_free_cuda_memory_info(0, round(99.5 * gib)) == 0


def test_context_free_free_never_goes_negative(monkeypatch):
    # A stale or over-subscribed reading can exceed torch's total.
    gib = 1 << 30
    _nvidia_smi_reading(monkeypatch, used_gb = 120.0, driver_total_gb = 100.0)
    assert hw._context_free_cuda_memory_info(0, round(99.5 * gib)) == 0


def test_context_free_declines_when_cuda_order_is_not_pci_bus_id(monkeypatch):
    # CUDA enumerates FASTEST_FIRST while nvidia-smi reports PCI order, so torch
    # ordinal 0 need not be row 0. Two equal-sized cards (a 3090 beside a 4090)
    # slip past the total-scope check, so only the order gate catches it.
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    _nvidia_smi_reading(monkeypatch, used_gb = 2.0, driver_total_gb = 24.0)
    monkeypatch.setattr(hw, "get_physical_gpu_count", lambda: 2)
    monkeypatch.setattr(
        hw,
        "_smi_query",
        lambda *a, **k: pytest.fail("an unshared index space must decline SMI telemetry"),
    )

    assert hw._context_free_cuda_memory_info(0, 24 << 30) is None


def test_context_free_keeps_smi_on_a_single_gpu_under_fastest_first(monkeypatch):
    # One card makes every ordering the identity, so the gate must not cost this
    # host a permanent primary context.
    gib = 1 << 30
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    _nvidia_smi_reading(monkeypatch, used_gb = 2.0, driver_total_gb = 24.0)
    monkeypatch.setattr(hw, "get_physical_gpu_count", lambda: 1)
    monkeypatch.setattr(hw, "_physical_gpu_count_from_smi", True)

    assert hw._context_free_cuda_memory_info(0, 24 * gib) == 22 * gib


def test_context_free_declines_when_the_gpu_count_is_not_smi_confirmed(monkeypatch):
    # `nvidia-smi -L` failing falls back to torch, which counts VISIBLE devices,
    # so a one-device mask on a multi-GPU host reads as single-GPU. Granting the
    # exemption there reads the mask in nvidia-smi's index space, which under
    # FASTEST_FIRST can be a different card. The count is cached, so one
    # transient timeout would disable the gate for the life of the process.
    gib = 1 << 30
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    _nvidia_smi_reading(monkeypatch, used_gb = 2.0, driver_total_gb = 24.0)
    monkeypatch.setattr(hw, "get_physical_gpu_count", lambda: 1)
    monkeypatch.setattr(hw, "_physical_gpu_count_from_smi", False)

    assert hw._context_free_cuda_memory_info(0, 24 * gib) is None


def test_physical_gpu_count_records_whether_the_smi_answered(monkeypatch):
    from utils.hardware import nvidia

    # The flag is what the order gate reads, so it must track the SOURCE of the
    # count, not merely that one was produced.
    monkeypatch.setattr(hw, "_physical_gpu_count", None)
    monkeypatch.setattr(hw, "_physical_gpu_count_from_smi", False)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(nvidia, "get_physical_gpu_count", lambda: 4)
    assert hw.get_physical_gpu_count() == 4
    assert hw._physical_gpu_count_from_smi is True

    monkeypatch.setattr(hw, "_physical_gpu_count", None)
    monkeypatch.setattr(hw, "_physical_gpu_count_from_smi", False)
    monkeypatch.setattr(nvidia, "get_physical_gpu_count", lambda: None)
    monkeypatch.setattr(hw, "_torch_get_physical_gpu_count", lambda: 1)
    assert hw.get_physical_gpu_count() == 1
    assert hw._physical_gpu_count_from_smi is False


def test_free_in_torch_scope_never_exceeds_the_total(monkeypatch):
    # Both SMI parsers accept signed values, so a transient negative used counter
    # would otherwise advertise more free than the card holds. utilization_pct
    # and the training-method policy both read this.
    gib = 1 << 30
    assert hw._free_in_torch_scope(24 * gib, -1.0) == 24 * gib
    assert hw._free_in_torch_scope(24 * gib, 0.0) == 24 * gib
    assert hw._free_in_torch_scope(24 * gib, 2.0) == 22 * gib
    assert hw._free_in_torch_scope(24 * gib, 99.0) == 0


def test_context_free_uuid_mask_skips_the_cuda_order_gate(monkeypatch):
    from utils.hardware import nvidia

    # A UUID mask names its devices absolutely and CUDA enumerates them in the
    # order listed, so FASTEST_FIRST changes nothing. Declining here would spend
    # the ~612 MiB primary context this whole path exists to avoid.
    gib = 1 << 30
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw, "get_physical_gpu_count", lambda: 2)
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {
            "raw": "GPU-bbbbbbbb",
            "numeric_ids": None,
            "supports_explicit_gpu_ids": False,
        },
    )

    def _run(_command, **_kwargs):
        return types.SimpleNamespace(
            returncode = 0,
            stdout = "\n".join(
                [
                    "0, GPU-aaaaaaaa-1111-2222-3333-444444444444, 5, 30, 15360, 16384, 100, 300",
                    "1, GPU-bbbbbbbb-1111-2222-3333-444444444444, 7, 31, 2048, 98304, 120, 350",
                ]
            ),
        )

    monkeypatch.setattr(nvidia.subprocess, "run", _run)

    assert hw._context_free_cuda_memory_info(0, 96 * gib) == 94 * gib


def test_parent_visible_spec_ignores_a_rocr_mask_on_windows(monkeypatch):
    # Windows HIP has no ROCr layer, so a stray ROCR var masks nothing there and
    # must not be read as the ordinal to physical mapping.
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw, "get_physical_gpu_count", lambda: 2)
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "1,0")

    monkeypatch.setattr(hw.sys, "platform", "win32")
    assert hw._get_parent_visible_gpu_spec()["numeric_ids"] == [0, 1]

    # Linux does honour it, so the reordering must survive there.
    monkeypatch.setattr(hw.sys, "platform", "linux")
    assert hw._get_parent_visible_gpu_spec()["numeric_ids"] == [1, 0]


def test_context_free_rocm_windows_declines_gpu_device_ordinal(monkeypatch):
    # GPU_DEVICE_ORDINAL=1 makes physical GPU 1 torch ordinal 0, but it is a
    # ROCclr-layer variable that _get_parent_visible_gpu_spec never reads, so the
    # adapter list is still [0, 1] and ordinal 0 would publish GPU 0's VRAM.
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setenv("GPU_DEVICE_ORDINAL", "1")
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": None, "numeric_ids": [0, 1], "supports_explicit_gpu_ids": True},
    )
    monkeypatch.setattr(
        hw,
        "_rocm_windows_per_device_vram",
        lambda ids, adapters = None: pytest.fail(
            "an external GPU ordinal filter must decline the adapter path"
        ),
    )

    assert hw._context_free_cuda_memory_info(0, 24 << 30) is None


def test_context_free_nvidia_smi_declines_whole_gpu_metrics_for_mig(monkeypatch):
    gib = 1 << 30
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": None, "numeric_ids": [0], "supports_explicit_gpu_ids": True},
    )
    monkeypatch.setattr(
        hw,
        "_smi_query",
        lambda *a, **k: {
            "available": True,
            "devices": [{"visible_ordinal": 0, "vram_used_gb": 70.0, "vram_total_gb": 80.0}],
        },
    )

    # torch reports the 10 GiB MIG allocation, while --query-gpu reports its
    # 80 GiB parent. Capping the parent's 10 GiB free would falsely report the
    # instance as completely empty even when its own allocation is full.
    assert hw._context_free_cuda_memory_info(0, 10 * gib) is None


def test_context_free_never_spawns_amd_smi_on_windows_without_a_hip_sdk(monkeypatch):
    """The probe must stay behind amd.py's elevation guard.

    amd-smi elevates a child on Windows without a HIP runtime, and the resulting
    UAC/DiskPart prompt cannot be suppressed. This path reaches amd-smi through a
    helper, so a later refactor calling it directly would reintroduce the prompt
    with nothing failing. Assert on the spawn itself, not on the return value.
    """
    from utils.hardware import amd

    spawned = []

    def _spy(command, *args, **kwargs):
        spawned.append(command)
        raise FileNotFoundError("no amd-smi should be spawned here")

    monkeypatch.setattr(amd, "platform", types.SimpleNamespace(system = lambda: "Windows"))
    monkeypatch.setattr(amd, "_hip_sdk_present", lambda: False)
    monkeypatch.setattr(amd.shutil, "which", lambda _n: r"C:\rocm\bin\amd-smi.exe")
    monkeypatch.setattr(amd.subprocess, "run", _spy)
    monkeypatch.delenv("UNSLOTH_ENABLE_AMD_SMI", raising = False)
    monkeypatch.setattr(amd, "_amd_smi_disabled", False)
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": None, "numeric_ids": [0, 1], "supports_explicit_gpu_ids": True},
    )
    # Windows reads per-adapter counters, so declining amd-smi still answers.
    # One instance per visible device, so the identity check upstream holds.
    monkeypatch.setattr(
        hw,
        "_rocm_windows_perf_counter_vram_by_adapter",
        lambda counter = "Dedicated Usage": [("a", 8.0 * (1 << 30)), ("b", 0.0)],
    )
    monkeypatch.setattr(
        hw,
        "_rocm_windows_per_device_vram",
        lambda ids, adapters = None: (
            [{"index": 0, "visible_ordinal": 0, "name": "RX", "used_gb": 8.0, "total_gb": 24.0}],
            8.0,
        ),
    )

    assert hw._context_free_cuda_memory_info(0, 24 << 30) == 16 << 30
    assert spawned == []


@pytest.mark.skipif(sys.platform != "win32", reason = "the guard is Windows-only")
def test_amd_smi_is_refused_on_a_real_windows_host_without_a_hip_sdk():
    # The test above fakes the platform. This one runs the same predicate against
    # the real OS, PATH and environment, so the guard is pinned where it applies.
    from utils.hardware import amd
    if amd._hip_sdk_present() or os.environ.get("UNSLOTH_ENABLE_AMD_SMI"):
        pytest.skip("this host may legitimately run amd-smi un-elevated")
    assert amd._amd_smi_allowed() is False


def test_context_free_rocm_smi_declines_whole_gpu_metrics_for_a_partition(monkeypatch):
    from utils.hardware import amd

    gib = 1 << 30
    monkeypatch.setattr(hw, "IS_ROCM", True)
    for name in (
        "GPU_DEVICE_ORDINAL",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
    ):
        monkeypatch.delenv(name, raising = False)
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": None, "numeric_ids": [0], "supports_explicit_gpu_ids": True},
    )
    monkeypatch.setattr(amd, "get_hip_id_by_gpu_index", lambda: {0: 0})
    monkeypatch.setattr(hw, "_torch_get_physical_gpu_count", lambda: 1)
    monkeypatch.setattr(
        hw,
        "_smi_query",
        lambda *a, **k: {
            "available": True,
            "devices": [{"visible_ordinal": 0, "vram_used_gb": 70.0, "vram_total_gb": 80.0}],
        },
    )
    # amd-smi is only the first probe. Leaving the Linux sysfs branch to the real
    # host made this vacuous exactly where it matters: where sysfs is unreachable
    # the branch never runs, and where it IS reachable the inventory was built
    # from the host's own card rather than the partition, so the totals matched
    # and a figure came back. Pin the whole path, sysfs included.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        hw,
        "_torch_get_device_module",
        lambda: (
            types.SimpleNamespace(
                get_device_properties = lambda _o: _FakeProps("MI300X partition", 10 * gib)
            ),
            "cuda",
        ),
    )
    monkeypatch.setattr(hw, "_rocm_kfd_gpu_pci_ids", lambda: {0: "0000:03:00.0"})
    monkeypatch.setattr(hw, "_rocm_visibility_mask_active", lambda: False)
    # sysfs speaks for the whole 80 GiB card, which is not this partition's pool.
    monkeypatch.setattr(
        hw, "_rocm_linux_sysfs_vram_by_pci_gb", lambda: {"0000:03:00.0": (70.0, 80.0)}
    )

    assert hw._context_free_cuda_memory_info(0, 10 * gib) is None


# ========== Inventory parity ==========


class _FakeProps:
    def __init__(self, name, total):
        self.name = name
        self.total_memory = total


def _fake_mod(
    totals,
    names = None,
    *,
    with_mem_get_info = True,
):
    mod = types.SimpleNamespace()
    names = names or [f"GPU{i}" for i in range(len(totals))]
    mod.get_device_properties = lambda o: _FakeProps(names[o], totals[o])
    if with_mem_get_info:
        # free == total - used; mirrors a driver reporting the same total both ways.
        mod.mem_get_info = lambda o: (totals[o] - (1 << 30), totals[o])
    mod.memory_allocated = lambda o: 1 << 30
    return mod


@pytest.mark.parametrize("totals", [[191505498112], [25757220864, 8589934592]])
def test_inventory_totals_match_the_occupancy_path(monkeypatch, totals):
    # Parity: switching the visibility endpoint to the inventory helper must not
    # move a single displayed number. Only used_gb differs, and that field is
    # discarded by every caller of the inventory helper.
    mod = _fake_mod(totals)
    monkeypatch.setattr(hw, "_torch_get_device_module", lambda: (mod, "cuda"))
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "rocm_windows_free_is_untrusted", lambda: False)
    indices = list(range(len(totals)))

    inventory = hw._torch_get_device_inventory(indices)
    occupancy = hw._torch_get_per_device_info(indices)

    assert [d["total_gb"] for d in inventory] == [d["total_gb"] for d in occupancy]
    assert [d["name"] for d in inventory] == [d["name"] for d in occupancy]
    assert [d["index"] for d in inventory] == [d["index"] for d in occupancy]
    assert [d["visible_ordinal"] for d in inventory] == [d["visible_ordinal"] for d in occupancy]
    assert all(d["used_gb"] is None for d in inventory)


def test_inventory_never_touches_mem_get_info(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("mem_get_info would create a primary context")

    mod = _fake_mod([8589934592])
    mod.mem_get_info = _boom
    mod.memory_allocated = _boom
    monkeypatch.setattr(hw, "_torch_get_device_module", lambda: (mod, "cuda"))
    assert hw._torch_get_device_inventory([0])[0]["total_gb"] == 8.0


def test_inventory_skips_a_device_that_fails_to_enumerate(monkeypatch):
    def _props(o):
        if o == 1:
            raise RuntimeError("device lost")
        return _FakeProps("GPU0", 8589934592)

    mod = types.SimpleNamespace(get_device_properties = _props)
    monkeypatch.setattr(hw, "_torch_get_device_module", lambda: (mod, "cuda"))
    assert [d["index"] for d in hw._torch_get_device_inventory([0, 1])] == [0]


def test_inventory_is_empty_without_torch(monkeypatch):
    monkeypatch.setattr(hw, "_torch_get_device_module", lambda: (None, None))
    assert hw._torch_get_device_inventory([0]) == []


def test_visibility_endpoint_uses_inventory_not_occupancy(monkeypatch):
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "IS_ROCM", True)  # skips the nvidia-smi shortcut
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: [0, 1])
    monkeypatch.setattr(
        hw,
        "_torch_get_per_device_info",
        lambda ids, adapters = None: pytest.fail(
            "the visibility poll must not ask torch for occupancy"
        ),
    )
    monkeypatch.setattr(
        hw,
        "_torch_get_device_inventory",
        lambda ids, adapters = None: [
            {
                "index": 0,
                "visible_ordinal": 0,
                "name": "MI300X",
                "total_gb": 192.0,
                "used_gb": None,
            },
            {
                "index": 1,
                "visible_ordinal": 1,
                "name": "MI300X",
                "total_gb": 192.0,
                "used_gb": None,
            },
        ],
    )
    result = hw.get_backend_visible_gpu_info()
    assert result["available"] is True
    assert [d["memory_total_gb"] for d in result["devices"]] == [192.0, 192.0]
    assert [d["name"] for d in result["devices"]] == ["MI300X", "MI300X"]
    assert result["index_kind"] == "physical"


# ========== ROCm APU totals ==========

# props.total_memory and the hipMemGetInfo total are the same number everywhere
# except a unified-memory APU, where props reports the dedicated carve-out and
# hipMemGetInfo the GTT-spanning pool. This module already treats the larger GTT
# figure as authoritative (_apply_unified_memory_correction adopts it over
# amd-smi's), so the inventory must not quietly hand back the carve-out and
# budget a 128 GiB Strix Halo as ~8 GiB.

_APU_CARVE_OUT = 8 * 1024**3
_APU_GTT_TOTAL = 100 * 1024**3


class _FakeApuProps(_FakeProps):
    def __init__(self, total = _APU_CARVE_OUT):
        super().__init__("AMD Radeon 8060S Graphics", total)
        self.gcnArchName = "gfx1151"  # Strix Halo
        self.is_integrated = 1


def _apu_mod(gtt_total = _APU_GTT_TOTAL, *, carve_out = _APU_CARVE_OUT):
    mod = types.SimpleNamespace()
    mod.get_device_properties = lambda o: _FakeApuProps(carve_out)
    mod.mem_get_info = lambda o: (gtt_total - (2 << 30), gtt_total)
    mod.memory_allocated = lambda o: 1 << 30
    return mod


def test_rocm_apu_inventory_keeps_the_gtt_total(monkeypatch):
    mod = _apu_mod()
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw, "_torch_get_device_module", lambda: (mod, "cuda"))
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "rocm_windows_free_is_untrusted", lambda: False)

    inventory = hw._torch_get_device_inventory([0])
    occupancy = hw._torch_get_per_device_info([0])

    assert inventory[0]["total_gb"] == occupancy[0]["total_gb"] == 100.0
    assert inventory[0]["total_gb"] != round(_APU_CARVE_OUT / (1024**3), 2)


class _FakeRocmProps(_FakeProps):
    """Discrete card as a current ROCm runtime describes it: the flag is filled in."""

    def __init__(
        self,
        name = "MI300X",
        total = 191505498112,
        arch = "gfx942",
        integrated = 0,
    ):
        super().__init__(name, total)
        self.gcnArchName = arch
        self.is_integrated = integrated


def _rocm_mod(props, gtt_total = _APU_GTT_TOTAL):
    mod = types.SimpleNamespace()
    mod.get_device_properties = lambda o: props
    mod.mem_get_info = lambda o: (gtt_total - (2 << 30), gtt_total)
    return mod


def test_rocm_discrete_inventory_stays_context_free(monkeypatch):
    # Only APUs pay for mem_get_info. An MI300X must keep the free path.
    mod = _rocm_mod(_FakeRocmProps())
    mod.mem_get_info = lambda o: pytest.fail("mem_get_info would create a primary context")
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw, "_hip_runtime_version", lambda: (6, 4))
    monkeypatch.setattr(hw, "_torch_get_device_module", lambda: (mod, "cuda"))
    assert hw._torch_get_device_inventory([0])[0]["total_gb"] == 178.35


def test_an_apu_shaped_name_on_cuda_stays_context_free(monkeypatch):
    # The carve-out only exists on ROCm, so IS_ROCM gates the whole probe.
    mod = _apu_mod()
    mod.mem_get_info = lambda o: pytest.fail("no ROCm here, so no reason to pay for a context")
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw, "_torch_get_device_module", lambda: (mod, "cuda"))
    assert hw._torch_get_device_inventory([0])[0]["total_gb"] == 8.0


def test_rocm_apu_keeps_the_carve_out_when_the_driver_total_fails(monkeypatch):
    # Degraded, not dropped: the device still has to appear in the inventory.
    mod = _apu_mod()

    def _fail(o):
        raise RuntimeError("hipMemGetInfo unavailable")

    mod.mem_get_info = _fail
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw, "_torch_get_device_module", lambda: (mod, "cuda"))
    assert [d["total_gb"] for d in hw._torch_get_device_inventory([0])] == [8.0]


def test_rocm_apu_sysfs_overlay_still_declines_against_the_gtt_total(monkeypatch):
    # The sysfs-first path compares the overlay's total against the inventory's.
    # With the carve-out there the two would agree and the overlay would adopt
    # sysfs, shrinking the APU. Against the GTT total it declines, as documented.
    mod = _apu_mod()
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw, "_torch_get_device_module", lambda: (mod, "cuda"))
    monkeypatch.setattr(hw, "platform", types.SimpleNamespace(system = lambda: "Linux"))
    monkeypatch.setattr(hw, "_rocm_kfd_gpu_pci_ids", lambda: {0: "0000:03:00.0"})
    monkeypatch.setattr(hw, "_rocm_visibility_mask_active", lambda: False)
    monkeypatch.setattr(
        hw, "_rocm_linux_sysfs_vram_by_pci_gb", lambda: {"0000:03:00.0": (1.0, 8.0)}
    )

    inventory = hw._torch_get_device_inventory([0])
    probe = [{"index": d["index"], "vram_total_gb": d["total_gb"]} for d in inventory]
    assert hw._rocm_system_wide_vram_by_index(probe) == {}


# ========== APUs the classifier cannot place ==========

# The classifier answers unified or not-unified, never "cannot tell", and it knows an
# APU by the driver's integrated flag or by a hardcoded gfx1150/1151/1152 set. Every
# other iGPU therefore reads as not-unified, and taking that for discrete would publish
# an 8 GiB-class carve-out as the whole device and hide models. clr only fills that flag
# in from ROCm 6.1.2, so only a runtime new enough to answer gets to settle it.


class _FakeUnclassifiedApuProps(_FakeProps):
    """gfx1103 Phoenix: a real APU that sits outside the classifier's arch set."""

    def __init__(
        self,
        total = _APU_CARVE_OUT,
        *,
        integrated = 0,
    ):
        super().__init__("AMD Radeon 780M Graphics", total)
        self.gcnArchName = "gfx1103"
        if integrated is not None:
            self.is_integrated = integrated


def test_rocm_unclassified_apu_keeps_the_driver_total_on_an_older_runtime(monkeypatch):
    mod = _rocm_mod(_FakeUnclassifiedApuProps())
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw, "_hip_runtime_version", lambda: (6, 1))
    monkeypatch.setattr(hw, "_torch_get_device_module", lambda: (mod, "cuda"))
    assert hw._torch_get_device_inventory([0])[0]["total_gb"] == 100.0


def test_rocm_unclassified_apu_keeps_the_driver_total_without_the_flag(monkeypatch):
    # A wheel that omits the field reads as 0 through getattr, exactly like a runtime
    # that never set it, so the arch set is again the only signal left.
    mod = _rocm_mod(_FakeUnclassifiedApuProps(integrated = None))
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw, "_hip_runtime_version", lambda: (6, 4))
    monkeypatch.setattr(hw, "_torch_get_device_module", lambda: (mod, "cuda"))
    assert hw._torch_get_device_inventory([0])[0]["total_gb"] == 100.0


def test_rocm_props_that_cannot_be_classified_keep_the_driver_total(monkeypatch):
    class _HostileProps(_FakeProps):
        def __init__(self):
            super().__init__("AMD Radeon Graphics", _APU_CARVE_OUT)

        @property
        def gcnArchName(self):
            raise RuntimeError("properties unreadable")

    mod = _rocm_mod(_HostileProps())
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw, "_hip_runtime_version", lambda: (6, 4))
    monkeypatch.setattr(hw, "_torch_get_device_module", lambda: (mod, "cuda"))
    assert hw._torch_get_device_inventory([0])[0]["total_gb"] == 100.0


def test_hip_runtime_version_reads_both_wheel_spellings(monkeypatch):
    import torch

    monkeypatch.setattr(torch.version, "hip", "6.4.43483-a187df25c", raising = False)
    assert hw._hip_runtime_version() == (6, 4)
    # AMD SDK / Radeon wheels leave version.hip unset; the tag is in __version__.
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.setattr(torch, "__version__", "2.9.0+rocm6.2", raising = False)
    assert hw._hip_runtime_version() == (6, 2)
    monkeypatch.setattr(torch, "__version__", "2.9.0+cu128", raising = False)
    assert hw._hip_runtime_version() is None


# ========== clear_gpu_cache ==========


def _torch_stub(reserved_by_device, calls):
    cuda = types.SimpleNamespace(
        is_available = lambda: True,
        device_count = lambda: len(reserved_by_device),
        memory_reserved = lambda i = 0: reserved_by_device[i],
        synchronize = lambda: calls.append("synchronize"),
        empty_cache = lambda: calls.append("empty_cache"),
        ipc_collect = lambda: calls.append("ipc_collect"),
    )
    return types.SimpleNamespace(cuda = cuda)


def test_clear_gpu_cache_skips_synchronize_with_nothing_reserved(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setitem(sys.modules, "torch", _torch_stub([0, 0], calls))
    hw.clear_gpu_cache()
    # Only the synchronize is skipped. empty_cache/ipc_collect need no context and
    # are no-ops without an allocator, so they stay exactly where they were.
    assert calls == ["empty_cache", "ipc_collect"]


def test_clear_gpu_cache_still_drains_a_live_allocator(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setitem(sys.modules, "torch", _torch_stub([0, 4 << 30], calls))
    hw.clear_gpu_cache()
    # Reserved on a NON-current device still counts: summing over the visible set
    # is what keeps a multi-GPU process from being skipped.
    assert calls == ["synchronize", "empty_cache", "ipc_collect"]


def test_clear_gpu_cache_still_propagates_a_sticky_cuda_fault(monkeypatch):
    # The diffusion and video unload paths wrap this in try/finally precisely
    # because a sticky fault has to surface, so the guard must not swallow one.
    calls: list[str] = []
    stub = _torch_stub([4 << 30], calls)

    def _boom():
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    stub.cuda.synchronize = _boom
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setitem(sys.modules, "torch", stub)
    with pytest.raises(RuntimeError, match = "illegal memory access"):
        hw.clear_gpu_cache()


def test_clear_gpu_cache_no_ops_without_cuda(monkeypatch):
    calls: list[str] = []
    stub = _torch_stub([0], calls)
    stub.cuda.is_available = lambda: False
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setitem(sys.modules, "torch", stub)
    hw.clear_gpu_cache()
    assert calls == ["empty_cache", "ipc_collect"]


# ========== Linux ROCm sysfs-first ==========


def _rocm_linux(monkeypatch, resolved):
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "_smi_query", lambda *a, **k: None)  # amd-smi unavailable
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw, "_rocm_windows_per_device_vram", lambda ids, adapters = None: ([], None))
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": None, "numeric_ids": [0, 1], "supports_explicit_gpu_ids": True},
    )
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: [0, 1])
    monkeypatch.setattr(
        hw,
        "_torch_get_device_inventory",
        lambda ids, adapters = None: [
            {"index": i, "visible_ordinal": i, "name": "MI210", "total_gb": 64.0, "used_gb": None}
            for i in ids
        ],
    )
    monkeypatch.setattr(hw, "_rocm_system_wide_vram_by_index", lambda devs: resolved)


def test_rocm_linux_reads_sysfs_without_asking_torch_for_occupancy(monkeypatch):
    _rocm_linux(monkeypatch, {0: (12.0, 64.0), 1: (3.0, 64.0)})
    monkeypatch.setattr(
        hw,
        "_torch_get_per_device_info",
        lambda ids, adapters = None: pytest.fail(
            "sysfs answered in full; torch occupancy is dead weight"
        ),
    )
    result = hw.get_visible_gpu_utilization()
    assert result["available"] is True
    assert result["index_kind"] == "physical"
    assert [(d["vram_used_gb"], d["vram_total_gb"]) for d in result["devices"]] == [
        (12.0, 64.0),
        (3.0, 64.0),
    ]
    assert [d["vram_utilization_pct"] for d in result["devices"]] == [18.8, 4.7]


def test_rocm_linux_sysfs_first_matches_the_old_torch_then_overlay_result(monkeypatch):
    # Parity: the overlay overwrote torch's used AND total, so every field the old
    # path produced for a fully covered set is reproduced without the context.
    resolved = {0: (12.0, 64.0), 1: (3.0, 64.0)}
    _rocm_linux(monkeypatch, resolved)
    monkeypatch.setattr(
        hw,
        "_torch_get_per_device_info",
        lambda ids, adapters = None: [
            {"index": i, "visible_ordinal": i, "name": "MI210", "total_gb": 64.0, "used_gb": 0.02}
            for i in ids
        ],
    )
    new = hw.get_visible_gpu_utilization()

    # Recreate the old shape: torch figures, then the overlay on top.
    old_devices = [
        {
            "index": i,
            "index_kind": "physical",
            "visible_ordinal": i,
            "gpu_utilization_pct": None,
            "temperature_c": None,
            "vram_used_gb": 0.02,
            "vram_total_gb": 64.0,
            "vram_utilization_pct": 0.0,
            "power_draw_w": None,
            "power_limit_w": None,
            "power_utilization_pct": None,
        }
        for i in (0, 1)
    ]
    hw._apply_system_wide_vram(old_devices, resolved)
    assert new["devices"] == old_devices


def test_rocm_linux_falls_back_to_torch_when_sysfs_covers_only_some(monkeypatch):
    # One device uncovered (a unified-memory APU or an MI300 partition): the whole
    # set must go back to torch rather than ship a device with no number at all.
    _rocm_linux(monkeypatch, {0: (12.0, 64.0)})
    used = []
    monkeypatch.setattr(
        hw,
        "_torch_get_per_device_info",
        lambda ids, adapters = None: (
            used.append(ids)
            or [
                {
                    "index": i,
                    "visible_ordinal": i,
                    "name": "MI210",
                    "total_gb": 64.0,
                    "used_gb": 1.0,
                }
                for i in ids
            ]
        ),
    )
    result = hw.get_visible_gpu_utilization()
    assert used == [[0, 1]]
    # The overlay still applies to the device sysfs does know about.
    assert result["devices"][0]["vram_used_gb"] == 12.0
    assert result["devices"][1]["vram_used_gb"] == 1.0


def test_rocm_linux_falls_back_when_sysfs_knows_nothing(monkeypatch):
    _rocm_linux(monkeypatch, {})
    used = []
    monkeypatch.setattr(
        hw,
        "_torch_get_per_device_info",
        lambda ids, adapters = None: (
            used.append(ids)
            or [
                {
                    "index": i,
                    "visible_ordinal": i,
                    "name": "MI210",
                    "total_gb": 64.0,
                    "used_gb": 1.0,
                }
                for i in ids
            ]
        ),
    )
    result = hw.get_visible_gpu_utilization()
    assert used == [[0, 1]]
    assert [d["vram_used_gb"] for d in result["devices"]] == [1.0, 1.0]


def test_nvidia_torch_fallback_keeps_using_occupancy(monkeypatch):
    # No sysfs shortcut off ROCm: an NVIDIA host without nvidia-smi genuinely needs
    # torch for live usage, so it must keep asking for it.
    monkeypatch.setattr(hw, "IS_ROCM", False)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "_smi_query", lambda *a, **k: None)
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": None, "numeric_ids": [0], "supports_explicit_gpu_ids": True},
    )
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: [0])
    monkeypatch.setattr(
        hw,
        "_torch_get_per_device_info",
        lambda ids, adapters = None: [
            {"index": 0, "visible_ordinal": 0, "name": "L40S", "total_gb": 44.0, "used_gb": 4.0}
        ],
    )
    result = hw.get_visible_gpu_utilization()
    assert result["devices"][0]["vram_used_gb"] == 4.0


def test_rocm_relative_index_skips_the_sysfs_shortcut(monkeypatch):
    # A UUID/MIG mask makes index relative, so it is not a host ordinal sysfs can
    # be keyed on. Same gate the overlay itself applies.
    _rocm_linux(monkeypatch, {0: (12.0, 64.0), 1: (3.0, 64.0)})
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: [])
    monkeypatch.setattr(hw, "_torch_get_physical_gpu_count", lambda: 2)
    seen = []
    monkeypatch.setattr(
        hw,
        "_torch_get_per_device_info",
        lambda ids, adapters = None: (
            seen.append(ids)
            or [
                {
                    "index": i,
                    "visible_ordinal": i,
                    "name": "MI210",
                    "total_gb": 64.0,
                    "used_gb": 1.0,
                }
                for i in ids
            ]
        ),
    )
    result = hw.get_visible_gpu_utilization()
    assert result["index_kind"] == "relative"
    assert seen == [[0, 1]]


def test_rocm_windows_does_not_take_the_linux_sysfs_path(monkeypatch):
    _rocm_linux(monkeypatch, {0: (12.0, 64.0), 1: (3.0, 64.0)})
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        hw,
        "_rocm_windows_per_device_vram",
        lambda ids, adapters = None: (
            [
                {"index": i, "visible_ordinal": i, "name": "RX", "used_gb": 2.0, "total_gb": 24.0}
                for i in ids
            ],
            4.0,
        ),
    )
    result = hw.get_visible_gpu_utilization()
    assert result["vram_used_gb_aggregate"] == 4.0
    assert [d["vram_used_gb"] for d in result["devices"]] == [2.0, 2.0]


# ========== The refactored overlay keeps its old contract ==========


def test_overlay_wrapper_still_mutates_in_place(monkeypatch):
    devices = [
        {"index": 0, "vram_used_gb": 0.02, "vram_total_gb": 64.0, "vram_utilization_pct": 0.0}
    ]
    monkeypatch.setattr(hw, "_rocm_system_wide_vram_by_index", lambda d: {0: (32.0, 64.0)})
    assert hw._overlay_system_wide_vram(devices) is None
    assert devices[0] == {
        "index": 0,
        "vram_used_gb": 32.0,
        "vram_total_gb": 64.0,
        "vram_utilization_pct": 50.0,
    }


def test_decision_helper_does_not_mutate(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw, "_rocm_kfd_gpu_pci_ids", lambda: ["0000:00:02.0"])
    monkeypatch.setattr(hw, "_rocm_visibility_mask_active", lambda: False)
    monkeypatch.setattr(
        hw, "_rocm_linux_sysfs_vram_by_pci_gb", lambda: {"0000:00:02.0": (32.0, 64.0)}
    )
    devices = [{"index": 0, "vram_total_gb": 64.0}]
    snapshot = dict(devices[0])
    assert hw._rocm_system_wide_vram_by_index(devices) == {0: (32.0, 64.0)}
    assert devices[0] == snapshot


def test_context_free_windows_rocm_declines_when_counters_outnumber_devices(monkeypatch):
    # A hidden adapter makes the counter set larger than the visible set. The
    # capacity ranking can then hand its reading to a visible card: over 89/89/8
    # GiB a card really holding 86 GiB reports 10, overstating free by 76 GiB.
    # The System tab tolerates that mapping; free_gb feeds training-method
    # selection, where overstating free is what OOMs. So decline instead.
    gib = 1 << 30
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw, "_rocm_device_ordinal_active", lambda: False)
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": "0,1", "numeric_ids": [0, 1], "supports_explicit_gpu_ids": True},
    )
    monkeypatch.setattr(hw, "_smi_query", lambda *a, **k: {"available": False, "devices": []})
    monkeypatch.setattr(
        hw,
        "_rocm_windows_per_device_vram",
        lambda _ids: (_ for _ in ()).throw(
            AssertionError("must decline before consulting the capacity mapping")
        ),
    )
    # Three instances, two visible devices: not an identity mapping.
    monkeypatch.setattr(
        hw,
        "_rocm_windows_perf_counter_vram_by_adapter",
        lambda counter = "Dedicated Usage": [("a", 0.0), ("b", 1.0 * gib), ("hidden", 20.0 * gib)],
    )

    assert hw._context_free_cuda_memory_info(0, 48 * gib) is None


def test_gpu_summary_apu_counters_need_positively_identified_uma(monkeypatch):
    # _rocm_props_total_is_carve_out answers True for an UNCERTAIN device too (old
    # HIP, unreadable integrated flag), because a too-small total hides models.
    # That justifies the driver total, not adding Shared Usage: shared bytes are
    # not part of a discrete card's props.total_memory, so a discrete GPU misread
    # as uncertain would have its free understated.
    gib = 1 << 30
    torch_stub, _props = _summary_torch(
        properties_total = 8 * gib,
        driver_free = 98 * gib,
        driver_total = 100 * gib,
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw, "_rocm_props_total_is_carve_out", lambda _props: True)
    monkeypatch.setattr(hw, "_rocm_props_are_positively_unified", lambda _props: False)
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        hw,
        "_rocm_windows_unified_used_bytes",
        lambda: (_ for _ in ()).throw(AssertionError("uncertain is not positively unified")),
    )

    summary = hw.get_gpu_summary()

    # Driver total kept, free still hipMemGetInfo's.
    assert summary == {"gpu_name": "Test GPU", "vram_total_gb": 100.0, "vram_free_gb": 98.0}
