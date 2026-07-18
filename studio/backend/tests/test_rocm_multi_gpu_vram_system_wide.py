# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The System tab's multi-GPU view must show system-wide VRAM on ROCm (#7072).

When amd-smi is unavailable, get_visible_gpu_utilization fell back to torch,
whose readings are process-local: on Windows WDDM hands each process its own
budget, so a model held by the separate llama-server process read as ~0 VRAM
used even with the GPU full. The primary-GPU endpoint already overlays
system-wide sources (Windows Performance Counters, Linux DRM sysfs); the
multi-device endpoint now applies the same per-GPU overlay, matched by
physical device index so reordering visibility masks stay correct.
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
    # Stub only if the real module is unavailable, so this file never shadows
    # real packages for later tests in the same pytest process.
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
    """Build a fake /sys/class/drm tree and point the helper's glob at it.

    ``cards``: (card_no, pci_bdf, driver, vram) tuples, where ``vram`` is a
    (used_gb, total_gb) pair or None for a device exposing no mem_info_vram_*
    files at all. Mirrors real sysfs: ``card<N>/device`` symlinks to the PCI dir
    and ``device/driver`` symlinks to the bound driver. The glob returns the card
    dirs in REVERSED order so the PCI sort has to establish the ordinals.
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
    # Keyed by PCI address, so the caller can join it to ROCm's device order by
    # identity rather than position. A non-amdgpu adapter (Intel) contributes no
    # entry at all, so it cannot occupy or shift anything.
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
    # A zero-total card (transient / headless) simply has no entry. Because the
    # map is keyed by identity, its absence cannot renumber anything.
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
    # An AMD device with incomplete sysfs support (an APU exposing no
    # mem_info_vram_* files) contributes no entry. Identity keying means the
    # discrete card keeps its own address regardless.
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
    """Fake /sys/class/kfd/kfd/topology/nodes tree.

    ``nodes``: (node_id, simd_count, location_id, domain, vendor_id) tuples.
    simd_count 0 marks a CPU node; location_id None omits the property. Returned
    out of node order so the sort has to establish it.
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
    # Node 0 is the CPU (simd_count 0) and must not take a device ordinal; the
    # GPU nodes in node-id order are HIP's device order. location_id is the
    # kernel's (bus << 8) | devfn, with domain carried separately.
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
    # The NVIDIA open kernel module registers KFD nodes with a positive SIMD
    # count and vendor_id 0x10DE. It is not a HIP device, so it must take no
    # ordinal -- otherwise an earlier NVIDIA node shifts every AMD GPU and ROCm
    # device 1 would resolve to AMD GPU 0. install.sh filters on the same id.
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
    # Dropping an unplaceable AMD GPU would shift every later ordinal, and a
    # similar-capacity GPU would then pass the total guard with another card's
    # usage. Fail closed for the whole map instead.
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
    """Declare the ROCm device order by PCI address (KFD topology stand-in).

    ``bdfs``: PCI address per ROCm physical ordinal, so index N is device N.
    Also clears the visibility masks -- the overlay only runs when host visibility
    is verified (no mask, and device count equal to the host GPU count).
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
    # Windows multi-GPU is intentionally not overlaid (perf counters can't be
    # mapped to ROCm ordinals and miss WDDM shared memory): keep torch figures.
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
    # HIP_VISIBLE_DEVICES=1,0: devices arrive as [index 1, index 0]. Each must
    # get ITS OWN GPU's figures by ROCm device ordinal, not the other's.
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
    # Device 0's card is unreadable (dropped): device index 0 has no entry and
    # keeps torch; device index 1 still gets ordinal 1, not ordinal-1-compacted-to-0.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw, "_rocm_linux_sysfs_vram_by_pci_gb", lambda: {_pci(1): (0.5, 8.0)})
    devices = [_device(0, used = 0.02, total = 45.0), _device(1, used = 0.01, total = 8.0)]
    _patch_pci_map(monkeypatch, [_pci(0), _pci(1)])
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 0.02  # no ordinal 0 -> torch kept
    assert devices[1]["vram_used_gb"] == 0.5  # ordinal 1 -> device 1, not device 0


def test_overlay_linux_skips_unified_memory_card(monkeypatch):
    # Strix Halo: sysfs reports only the small dedicated slice while torch sees
    # the GTT-backed pool; the smaller sysfs total must NOT shrink the device
    # (mirrors _apply_unified_memory_correction's larger-total-wins rule).
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw, "_rocm_linux_sysfs_vram_by_pci_gb", lambda: {_pci(0): (0.4, 1.0)})
    devices = [_device(0, used = 12.0, total = 96.0)]  # torch's unified pool
    _patch_pci_map(monkeypatch, [_pci(0)])
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 12.0
    assert devices[0]["vram_total_gb"] == 96.0


def test_overlay_linux_skips_partitioned_device(monkeypatch):
    # MI300 CPX mode: HIP exposes several logical devices for one physical card,
    # but sysfs reports the WHOLE card's aggregate. The card total (192) dwarfs
    # the partition total (24), so the overlay must NOT overwrite the partition
    # with whole-card usage/capacity -- else downstream selection would treat the
    # partition as having the entire card's free VRAM.
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
    # An amdgpu-bound adapter HIP cannot enumerate (an unsupported older AMD GPU
    # beside a supported one) has NO KFD GPU node, so it never appears in the
    # device order and cannot take an ordinal. Device 0 resolves to the supported
    # GPU's own address, never the display card's -- no count heuristic needed.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        hw,
        "_rocm_linux_sysfs_vram_by_pci_gb",
        # Both cards are in DRM sysfs with SIMILAR capacity, which is exactly the
        # case the total-size guard cannot separate.
        lambda: {_pci(9): (30.0, 45.0), _pci(3): (12.0, 45.0)},
    )
    _patch_pci_map(monkeypatch, [_pci(3)])  # KFD lists only the supported GPU
    devices = [_device(0, used = 0.02, total = 45.0)]  # torch sees that one GPU
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 12.0  # the supported GPU's own figures


def test_overlay_skips_masked_subsets(monkeypatch):
    # Under a mask the reported index is not verifiably a HOST-physical ordinal:
    # with device-cgroup filtering underneath, a mask indexes the container's set
    # instead. torch exposes no PCI id to check against, so the safe answer is to
    # keep its figures rather than risk another card's usage.
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
    # A container exposing only some render devices through device cgroups sets
    # NO visibility env var, yet torch compacts what it can see to ordinals from
    # zero while the host-mounted KFD/DRM trees still list every GPU. Treating
    # those compacted indices as host-physical would overlay host GPU 0 onto a
    # container device that is really another card, so the count mismatch must
    # disable the overlay.
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
    assert overlaid == [2]  # overlay ran over both devices


def test_visible_utilization_relative_index_skips_overlay(monkeypatch):
    # UUID/MIG mask -> no numeric ids -> torch ordinals with index_kind
    # "relative". The overlay matches by PHYSICAL index, so it must not run here
    # (relative 0 is not physical card/adapter 0).
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
    assert called == []  # overlay skipped for relative indices


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
    assert called == []  # NVIDIA keeps the plain torch fallback


def test_any_visibility_mask_is_detected(monkeypatch):
    # Each of these makes the reported index something other than a verifiable
    # host-physical ordinal, so any one of them must disable the overlay: a
    # layered HIP-over-ROCR value is ROCR-relative, GPU_DEVICE_ORDINAL is never
    # consulted when the index is assigned, and with device-cgroup filtering
    # underneath even a plain mask indexes the container's set.
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
    # GPU_DEVICE_ORDINAL=1 surfaces physical GPU 1 as torch ordinal 0, which the
    # parent spec never corrects, so index 0 is not GPU 0. Overlay must not run.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    _patch_pci_map(monkeypatch, [_pci(0)])
    monkeypatch.setenv("GPU_DEVICE_ORDINAL", "1")
    monkeypatch.setattr(hw, "_rocm_linux_sysfs_vram_by_pci_gb", lambda: {_pci(0): (30.0, 45.0)})
    devices = [_device(0, used = 0.02, total = 45.0)]
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 0.02


def test_visible_utilization_delegates_gating_to_the_overlay(monkeypatch):
    # The call site no longer pre-checks masks: the overlay verifies host
    # visibility itself (no mask, and device count equal to the host GPU count),
    # so a physical-index payload always reaches it and it decides.
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
    # Real overlay, real gating: the layered mask must leave torch's figures.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw, "_rocm_kfd_gpu_pci_ids", lambda: [_pci(0), _pci(1)])
    monkeypatch.setattr(hw, "_rocm_linux_sysfs_vram_by_pci_gb", lambda: {_pci(1): (30.0, 8.0)})
    result = hw.get_visible_gpu_utilization()
    assert result["index_kind"] == "physical"
    assert result["devices"][0]["vram_used_gb"] == 0.02  # untouched
