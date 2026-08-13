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
        lambda ids: pytest.fail("the visibility poll must not ask torch for occupancy"),
    )
    monkeypatch.setattr(
        hw,
        "_torch_get_device_inventory",
        lambda ids: [
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
    monkeypatch.setattr(hw, "_rocm_windows_per_device_vram", lambda ids: ([], None))
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": None, "numeric_ids": [0, 1], "supports_explicit_gpu_ids": True},
    )
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: [0, 1])
    monkeypatch.setattr(
        hw,
        "_torch_get_device_inventory",
        lambda ids: [
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
        lambda ids: pytest.fail("sysfs answered in full; torch occupancy is dead weight"),
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
        lambda ids: [
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
        lambda ids: used.append(ids)
        or [
            {"index": i, "visible_ordinal": i, "name": "MI210", "total_gb": 64.0, "used_gb": 1.0}
            for i in ids
        ],
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
        lambda ids: used.append(ids)
        or [
            {"index": i, "visible_ordinal": i, "name": "MI210", "total_gb": 64.0, "used_gb": 1.0}
            for i in ids
        ],
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
        lambda ids: [
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
        lambda ids: seen.append(ids)
        or [
            {"index": i, "visible_ordinal": i, "name": "MI210", "total_gb": 64.0, "used_gb": 1.0}
            for i in ids
        ],
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
        lambda ids: (
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
