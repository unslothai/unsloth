# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The System tab's multi-GPU view must show system-wide VRAM on ROCm (#7072).

When amd-smi is unavailable, get_visible_gpu_utilization fell back to torch,
whose readings are process-local: a model held by the separate llama-server
process read as ~0 VRAM used even with the GPU full. These tests cover the
per-GPU system-wide overlay the multi-device endpoint now applies, matched by
physical device identity.
"""

from __future__ import annotations

import importlib
import sys
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

import utils.hardware.hardware as hw  # noqa: E402


def _device(
    index,
    used,
    total,
    *,
    ordinal = None,
):
    return {
        "index": index,
        "index_kind": "physical",
        "visible_ordinal": index if ordinal is None else ordinal,
        "gpu_utilization_pct": None,
        "temperature_c": None,
        "vram_used_gb": used,
        "vram_total_gb": total,
        "vram_utilization_pct": round((used / total) * 100, 1) if total > 0 else None,
        "power_draw_w": None,
        "power_limit_w": None,
        "power_utilization_pct": None,
    }


# ── Linux per-card sysfs ──


def _fake_drm(tmp_path, monkeypatch, cards):
    """Fake /sys/class/drm tree; glob returns cards REVERSED so the PCI sort must order them.

    ``cards``: (card_no, pci_bdf, driver, vram) tuples; vram is (used_gb, total_gb)
    or None for a device with no mem_info_vram_* files.
    """
    drivers = tmp_path / "drivers"
    card_paths = []
    for card_no, bdf, driver, vram in cards:
        pci_dir = tmp_path / "pci" / bdf
        pci_dir.mkdir(parents = True, exist_ok = True)
        drv_dir = drivers / driver
        drv_dir.mkdir(parents = True, exist_ok = True)
        (pci_dir / "driver").symlink_to(drv_dir)
        if vram is not None:
            used, total = vram
            (pci_dir / "mem_info_vram_used").write_text(str(int(used * 1024**3)))
            (pci_dir / "mem_info_vram_total").write_text(str(int(total * 1024**3)))
        card_dir = tmp_path / "drm" / f"card{card_no}"
        card_dir.mkdir(parents = True, exist_ok = True)
        (card_dir / "device").symlink_to(pci_dir)
        card_paths.append(str(card_dir))
    monkeypatch.setattr(hw.glob, "glob", lambda pattern: list(reversed(card_paths)))
    return card_paths


def test_linux_vram_keyed_by_pci_excludes_foreign_adapters(monkeypatch, tmp_path):
    # Foreign (non-amdgpu) adapters contribute no entry, so they cannot shift ordinals.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    _fake_drm(
        tmp_path,
        monkeypatch,
        [
            (0, "0000:00:02.0", "i915", (0.5, 2.0)),  # foreign adapter: excluded
            (1, "0000:03:00.0", "amdgpu", (40, 48)),  # AMD device 0
            (2, "0000:41:00.0", "amdgpu", (1, 8)),  # AMD device 1
        ],
    )
    assert hw._rocm_linux_sysfs_vram_by_pci_gb() == {
        "0000:03:00.0": (40.0, 48.0),
        "0000:41:00.0": (1.0, 8.0),
    }


def test_linux_vram_omits_bad_cards_without_shifting(monkeypatch, tmp_path):
    # A zero-total card has no entry; identity keying means its absence renumbers nothing.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    _fake_drm(
        tmp_path,
        monkeypatch,
        [
            (0, "0000:03:00.0", "amdgpu", (0, 0)),  # zero total -> no entry
            (1, "0000:41:00.0", "amdgpu", (2, 16)),
        ],
    )
    assert hw._rocm_linux_sysfs_vram_by_pci_gb() == {"0000:41:00.0": (2.0, 16.0)}


def test_linux_vram_omits_amd_card_without_vram_files(monkeypatch, tmp_path):
    # An APU with no mem_info_vram_* files has no entry; the discrete card keeps its address.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    _fake_drm(
        tmp_path,
        monkeypatch,
        [
            (0, "0000:03:00.0", "amdgpu", None),  # APU: no VRAM sysfs files
            (1, "0000:41:00.0", "amdgpu", (2, 16)),
        ],
    )
    assert hw._rocm_linux_sysfs_vram_by_pci_gb() == {"0000:41:00.0": (2.0, 16.0)}


# ── KFD topology: the authoritative ROCm device order ──


_AMD = 4098  # 0x1002
_NVIDIA = 4318  # 0x10DE -- the open kernel module also registers KFD nodes


def _fake_kfd(tmp_path, monkeypatch, nodes):
    """Fake KFD topology nodes tree, returned out of node order so the sort must order it.

    ``nodes``: (node_id, simd_count, location_id, domain, vendor_id); simd_count 0
    marks a CPU node, location_id None omits the property.
    """
    node_paths = []
    for node_id, simd_count, location_id, domain, vendor_id in nodes:
        d = tmp_path / "kfd" / str(node_id)
        d.mkdir(parents = True, exist_ok = True)
        lines = [f"cpu_cores_count {0 if simd_count else 8}", f"simd_count {simd_count}"]
        if location_id is not None:
            lines.append(f"location_id {location_id}")
        lines.append(f"domain {domain}")
        if vendor_id is not None:
            lines.append(f"vendor_id {vendor_id}")
        (d / "properties").write_text("\n".join(lines) + "\n")
        node_paths.append(str(d))
    monkeypatch.setattr(hw.glob, "glob", lambda pattern: list(reversed(node_paths)))
    return node_paths


def test_kfd_lists_gpu_nodes_in_device_order(monkeypatch, tmp_path):
    # The CPU node (simd_count 0) takes no ordinal; GPU nodes in node-id order are HIP's order.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    _fake_kfd(
        tmp_path,
        monkeypatch,
        [
            (0, 0, None, 0, None),  # CPU node
            (1, 304, (0x03 << 8) | (0x00 << 3) | 0, 0, _AMD),  # 0000:03:00.0 -> dev 0
            (2, 304, (0x41 << 8) | (0x00 << 3) | 0, 0, _AMD),  # 0000:41:00.0 -> dev 1
        ],
    )
    assert hw._rocm_kfd_gpu_pci_ids() == ["0000:03:00.0", "0000:41:00.0"]


def test_kfd_decodes_domain_device_and_function(monkeypatch, tmp_path):
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    _fake_kfd(tmp_path, monkeypatch, [(1, 64, (0xC1 << 8) | (0x1F << 3) | 5, 0x1234, _AMD)])
    assert hw._rocm_kfd_gpu_pci_ids() == ["1234:c1:1f.5"]


def test_kfd_skips_non_amd_gpu_nodes(monkeypatch, tmp_path):
    # An NVIDIA KFD node is not a HIP device: it must take no ordinal, else it
    # shifts every AMD GPU and ROCm device 1 resolves to AMD GPU 0.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    _fake_kfd(
        tmp_path,
        monkeypatch,
        [
            (0, 0, None, 0, None),  # CPU
            (1, 128, (0x01 << 8) | 0, 0, _NVIDIA),  # NVIDIA: no ordinal
            (2, 304, (0x03 << 8) | 0, 0, _AMD),  # AMD device 0
            (3, 304, (0x41 << 8) | 0, 0, _AMD),  # AMD device 1
        ],
    )
    assert hw._rocm_kfd_gpu_pci_ids() == ["0000:03:00.0", "0000:41:00.0"]


def test_kfd_fails_closed_when_a_gpu_has_no_location(monkeypatch, tmp_path):
    # Dropping an unplaceable AMD GPU shifts later ordinals; fail closed for the whole map.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    _fake_kfd(
        tmp_path,
        monkeypatch,
        [
            (1, 304, None, 0, _AMD),  # AMD GPU with no location_id
            (2, 304, (0x41 << 8) | 0, 0, _AMD),
        ],
    )
    assert hw._rocm_kfd_gpu_pci_ids() == []


def test_kfd_fails_closed_when_a_node_is_unreadable(monkeypatch, tmp_path):
    # An unreadable node could be a GPU; assuming otherwise would shift ordinals.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    paths = _fake_kfd(
        tmp_path,
        monkeypatch,
        [
            (1, 304, (0x03 << 8) | 0, 0, _AMD),
            (2, 304, (0x41 << 8) | 0, 0, _AMD),
        ],
    )
    (Path(paths[0]) / "properties").unlink()
    assert hw._rocm_kfd_gpu_pci_ids() == []


def test_kfd_absent_yields_no_device_order(monkeypatch):
    monkeypatch.setattr(hw.glob, "glob", lambda pattern: [])
    assert hw._rocm_kfd_gpu_pci_ids() == []


# ── overlay ──


def _patch_pci_map(monkeypatch, bdfs):
    """Declare the ROCm device order by PCI address (index N is device N) and clear
    the visibility masks the overlay requires unset.
    """
    for var in (
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setattr(hw, "_rocm_kfd_gpu_pci_ids", lambda: list(bdfs))


def _pci(n):
    """A distinct, well-formed PCI address for card n."""
    return f"0000:{n:02x}:00.0"


def test_overlay_windows_is_noop_keeps_torch(monkeypatch):
    # Windows is intentionally not overlaid (perf counters can't map to ROCm ordinals): keep torch.
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        hw,
        "_rocm_linux_sysfs_vram_by_pci_gb",
        lambda: (_ for _ in ()).throw(AssertionError("sysfs must not run on Windows")),
    )
    devices = [_device(0, used = 0.02, total = 8.0)]
    _patch_pci_map(monkeypatch, [_pci(0)])
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 0.02  # untouched


def test_overlay_linux_matches_by_device_ordinal(monkeypatch):
    # Devices arriving as [index 1, index 0] each get their own GPU's figures by ordinal.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        hw,
        "_rocm_linux_sysfs_vram_by_pci_gb",
        lambda: {_pci(0): (30.0, 45.0), _pci(1): (0.5, 8.0)},  # dev 0 big, dev 1 small
    )
    devices = [_device(1, used = 0.01, total = 8.0), _device(0, used = 0.02, total = 45.0)]
    _patch_pci_map(monkeypatch, [_pci(0), _pci(1)])
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 0.5  # index 1 -> device 1 (small)
    assert devices[0]["vram_total_gb"] == 8.0
    assert devices[1]["vram_used_gb"] == 30.0  # index 0 -> device 0 (big)
    assert devices[1]["vram_total_gb"] == 45.0


def test_overlay_linux_ordinal_hole_does_not_shift(monkeypatch):
    # Device 0's card dropped: index 0 keeps torch, index 1 still maps to ordinal 1 (no compaction).
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw, "_rocm_linux_sysfs_vram_by_pci_gb", lambda: {_pci(1): (0.5, 8.0)})
    devices = [_device(0, used = 0.02, total = 45.0), _device(1, used = 0.01, total = 8.0)]
    _patch_pci_map(monkeypatch, [_pci(0), _pci(1)])
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 0.02  # no ordinal 0 -> torch kept
    assert devices[1]["vram_used_gb"] == 0.5  # ordinal 1 -> device 1, not device 0


def test_overlay_linux_skips_unified_memory_card(monkeypatch):
    # Unified-memory APU: the smaller sysfs total must not shrink torch's GTT-backed pool.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw, "_rocm_linux_sysfs_vram_by_pci_gb", lambda: {_pci(0): (0.4, 1.0)})
    devices = [_device(0, used = 12.0, total = 96.0)]  # torch's unified pool
    _patch_pci_map(monkeypatch, [_pci(0)])
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 12.0
    assert devices[0]["vram_total_gb"] == 96.0


def test_overlay_linux_skips_partitioned_device(monkeypatch):
    # Partitioned MI300: the whole-card sysfs total dwarfs the partition, so the overlay must not overwrite it.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw, "_rocm_linux_sysfs_vram_by_pci_gb", lambda: {_pci(0): (40.0, 192.0)})
    devices = [_device(0, used = 1.0, total = 24.0)]  # torch partition
    _patch_pci_map(monkeypatch, [_pci(0)])
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 1.0  # partition figures kept
    assert devices[0]["vram_total_gb"] == 24.0


def test_overlay_linux_out_of_range_index_untouched(monkeypatch):
    # A masked host exposing physical index 5 with no card 5: keep torch data.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        hw, "_rocm_linux_sysfs_vram_by_pci_gb", lambda: {_pci(0): (30.0, 45.0), _pci(1): (0.5, 8.0)}
    )
    devices = [_device(5, used = 0.02, total = 45.0)]
    _patch_pci_map(monkeypatch, [_pci(0), _pci(1)])
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 0.02


def test_overlay_ignores_adapters_rocm_cannot_enumerate(monkeypatch):
    # A HIP-unenumerable amdgpu adapter has no KFD node, so device 0 resolves to
    # the supported GPU's own address, never the display card's.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        hw,
        "_rocm_linux_sysfs_vram_by_pci_gb",
        # Both in DRM sysfs with similar capacity -- what the total-size guard can't separate.
        lambda: {_pci(9): (30.0, 45.0), _pci(3): (12.0, 45.0)},
    )
    _patch_pci_map(monkeypatch, [_pci(3)])  # KFD lists only the supported GPU
    devices = [_device(0, used = 0.02, total = 45.0)]  # torch sees that one GPU
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 12.0  # the supported GPU's own figures


def test_overlay_skips_masked_subsets(monkeypatch):
    # Under a mask the index is not verifiably a host ordinal, so keep torch's figures.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    _patch_pci_map(monkeypatch, [_pci(0), _pci(1), _pci(2), _pci(3)])
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1,3")
    monkeypatch.setattr(
        hw,
        "_rocm_linux_sysfs_vram_by_pci_gb",
        lambda: {_pci(1): (30.0, 48.0), _pci(3): (12.0, 48.0)},
    )
    devices = [_device(1, used = 0.02, total = 48.0), _device(3, used = 0.01, total = 48.0)]
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 0.02  # torch kept
    assert devices[1]["vram_used_gb"] == 0.01


def test_overlay_skips_device_cgroup_filtered_container(monkeypatch):
    # A device-cgroup container sets no env var yet compacts torch's indices from
    # zero while KFD/DRM list every GPU, so the count mismatch must disable the overlay.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    _patch_pci_map(monkeypatch, [_pci(0), _pci(1), _pci(2), _pci(3)])  # host has 4
    monkeypatch.setattr(
        hw,
        "_rocm_linux_sysfs_vram_by_pci_gb",
        lambda: {_pci(0): (30.0, 48.0), _pci(2): (12.0, 48.0)},
    )
    devices = [_device(0, used = 0.02, total = 48.0)]  # container sees 1, as index 0
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 0.02  # torch kept, not host GPU 0's 30.0


def test_overlay_skips_without_kfd_topology(monkeypatch):
    # No KFD means no identity to join on; fall back to torch rather than guess.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw, "_rocm_kfd_gpu_pci_ids", lambda: [])
    monkeypatch.setattr(
        hw,
        "_rocm_linux_sysfs_vram_by_pci_gb",
        lambda: (_ for _ in ()).throw(AssertionError("must not read sysfs without KFD")),
    )
    devices = [_device(0, used = 0.02, total = 45.0)]
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 0.02


def test_overlay_empty_devices_is_noop(monkeypatch):
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    hw._overlay_system_wide_vram([])  # must not raise


# ── integration: the ROCm torch fallback applies the overlay ──


def test_visible_utilization_rocm_fallback_overlays(monkeypatch):
    for _var in (
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        monkeypatch.delenv(_var, raising = False)
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "_smi_query", lambda *a, **k: None)  # amd-smi unavailable
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": None, "numeric_ids": [0, 1], "supports_explicit_gpu_ids": True},
    )
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: [0, 1])
    monkeypatch.setattr(
        hw,
        "_torch_get_per_device_info",
        lambda ids: [
            {"index": 0, "visible_ordinal": 0, "used_gb": 0.02, "total_gb": 45.0},
            {"index": 1, "visible_ordinal": 1, "used_gb": 0.01, "total_gb": 8.0},
        ],
    )
    overlaid = []
    monkeypatch.setattr(
        hw, "_overlay_system_wide_vram", lambda devices: overlaid.append(len(devices))
    )
    result = hw.get_visible_gpu_utilization()
    assert result["available"] is True
    assert overlaid == [2]


def test_visible_utilization_relative_index_skips_overlay(monkeypatch):
    # UUID/MIG mask gives relative indices; the overlay matches physical index, so it must not run.
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "_smi_query", lambda *a, **k: None)
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": "GPU-uuid-a", "numeric_ids": None, "supports_explicit_gpu_ids": False},
    )
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: [])  # UUID mask
    monkeypatch.setattr(hw, "_torch_get_physical_gpu_count", lambda: 1)
    monkeypatch.setattr(
        hw,
        "_torch_get_per_device_info",
        lambda ids: [{"index": 0, "visible_ordinal": 0, "used_gb": 0.02, "total_gb": 8.0}],
    )
    called = []
    monkeypatch.setattr(hw, "_overlay_system_wide_vram", lambda devices: called.append(1))
    result = hw.get_visible_gpu_utilization()
    assert result["index_kind"] == "relative"
    assert called == []


def test_visible_utilization_nvidia_fallback_skips_overlay(monkeypatch):
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
        lambda ids: [{"index": 0, "visible_ordinal": 0, "used_gb": 1.0, "total_gb": 24.0}],
    )
    called = []
    monkeypatch.setattr(hw, "_overlay_system_wide_vram", lambda devices: called.append(1))
    result = hw.get_visible_gpu_utilization()
    assert result["available"] is True
    assert called == []


def test_any_visibility_mask_is_detected(monkeypatch):
    # Any of these makes the index not a host-physical ordinal, so each must disable the overlay.
    for var in (
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        monkeypatch.delenv(var, raising = False)
    assert hw._rocm_visibility_mask_active() is False
    for var in (
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        monkeypatch.setenv(var, "1")
        assert hw._rocm_visibility_mask_active() is True, var
        monkeypatch.setenv(var, "  ")  # empty is not an active filter
        assert hw._rocm_visibility_mask_active() is False, var
        monkeypatch.delenv(var, raising = False)


def test_overlay_skips_under_gpu_device_ordinal(monkeypatch):
    # GPU_DEVICE_ORDINAL=1 surfaces GPU 1 as torch ordinal 0, so index 0 is not GPU 0; overlay must not run.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    _patch_pci_map(monkeypatch, [_pci(0)])
    monkeypatch.setenv("GPU_DEVICE_ORDINAL", "1")
    monkeypatch.setattr(hw, "_rocm_linux_sysfs_vram_by_pci_gb", lambda: {_pci(0): (30.0, 45.0)})
    devices = [_device(0, used = 0.02, total = 45.0)]
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 0.02


def test_visible_utilization_delegates_gating_to_the_overlay(monkeypatch):
    # The call site no longer pre-checks masks; the overlay gates itself, so a physical payload always reaches it.
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(hw, "_smi_query", lambda *a, **k: None)
    monkeypatch.setattr(
        hw,
        "_get_parent_visible_gpu_spec",
        lambda: {"raw": "1", "numeric_ids": [1], "supports_explicit_gpu_ids": True},
    )
    monkeypatch.setattr(hw, "get_parent_visible_gpu_ids", lambda: [1])
    monkeypatch.setattr(
        hw,
        "_torch_get_per_device_info",
        lambda ids: [{"index": 1, "visible_ordinal": 0, "used_gb": 0.02, "total_gb": 8.0}],
    )
    # Real overlay + gating: the layered mask must leave torch's figures.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw, "_rocm_kfd_gpu_pci_ids", lambda: [_pci(0), _pci(1)])
    monkeypatch.setattr(hw, "_rocm_linux_sysfs_vram_by_pci_gb", lambda: {_pci(1): (30.0, 8.0)})
    result = hw.get_visible_gpu_utilization()
    assert result["index_kind"] == "physical"
    assert result["devices"][0]["vram_used_gb"] == 0.02  # untouched
