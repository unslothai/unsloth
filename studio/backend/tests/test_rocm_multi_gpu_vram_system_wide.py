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


def test_linux_per_card_keyed_by_rocm_ordinal_via_pci(monkeypatch, tmp_path):
    # A non-amdgpu adapter (Intel) owns card0; the AMD GPUs are card1/card2, so
    # ROCm sees them as devices 0/1. The helper must key by ROCm ordinal (amdgpu
    # cards in PCI order), NOT the DRM card number -- else device 1 would get
    # card1 (AMD device 0) and device 0 nothing. The Intel card takes no ordinal.
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
    assert hw._rocm_linux_sysfs_vram_per_card_gb() == {0: (40.0, 48.0), 1: (1.0, 8.0)}


def test_linux_per_card_keeps_holes_from_bad_cards(monkeypatch, tmp_path):
    # An amdgpu card with a zero total (transient / headless) yields no entry but
    # must still CONSUME its ROCm ordinal: the good card keeps ordinal 1, never
    # compacted to 0 -- else its usage would be misattributed to device 0.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    _fake_drm(
        tmp_path,
        monkeypatch,
        [
            (0, "0000:03:00.0", "amdgpu", (0, 0)),  # zero total -> no entry
            (1, "0000:41:00.0", "amdgpu", (2, 16)),
        ],
    )
    assert hw._rocm_linux_sysfs_vram_per_card_gb() == {1: (2.0, 16.0)}


def test_linux_per_card_reserves_ordinal_for_amd_without_vram_files(monkeypatch, tmp_path):
    # An AMD device with incomplete sysfs support (an APU exposing no
    # mem_info_vram_* files at all) is still a ROCm device, so it must consume
    # ordinal 0 -- enumerating by bound driver rather than by the VRAM-file glob.
    # Otherwise the discrete card shifts down to ordinal 0 and a similar-capacity
    # GPU would receive another device's system-wide usage.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    _fake_drm(
        tmp_path,
        monkeypatch,
        [
            (0, "0000:03:00.0", "amdgpu", None),  # APU: no VRAM sysfs files
            (1, "0000:41:00.0", "amdgpu", (2, 16)),
        ],
    )
    assert hw._rocm_linux_sysfs_vram_per_card_gb() == {1: (2.0, 16.0)}


# ── overlay ──


def _patch_card_count(monkeypatch, n):
    """Declare how many amdgpu cards the host has.

    Also clears the visibility masks, so these stay UNMASKED-host cases -- the
    only ones where the overlay requires the card count to equal the device count.
    """
    for var in (
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setattr(hw, "_rocm_linux_amdgpu_cards", lambda: [("", i, "") for i in range(n)])


def test_overlay_windows_is_noop_keeps_torch(monkeypatch):
    # Windows multi-GPU is intentionally not overlaid (perf counters can't be
    # mapped to ROCm ordinals and miss WDDM shared memory): keep torch figures.
    monkeypatch.setattr(hw.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        hw,
        "_rocm_linux_sysfs_vram_per_card_gb",
        lambda: (_ for _ in ()).throw(AssertionError("sysfs must not run on Windows")),
    )
    devices = [_device(0, used = 0.02, total = 8.0)]
    _patch_card_count(monkeypatch, 1)
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 0.02  # untouched


def test_overlay_linux_matches_by_device_ordinal(monkeypatch):
    # HIP_VISIBLE_DEVICES=1,0: devices arrive as [index 1, index 0]. Each must
    # get ITS OWN GPU's figures by ROCm device ordinal, not the other's.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        hw,
        "_rocm_linux_sysfs_vram_per_card_gb",
        lambda: {0: (30.0, 45.0), 1: (0.5, 8.0)},  # device 0 big, device 1 small
    )
    devices = [_device(1, used = 0.01, total = 8.0), _device(0, used = 0.02, total = 45.0)]
    _patch_card_count(monkeypatch, 2)
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 0.5  # index 1 -> device 1 (small)
    assert devices[0]["vram_total_gb"] == 8.0
    assert devices[1]["vram_used_gb"] == 30.0  # index 0 -> device 0 (big)
    assert devices[1]["vram_total_gb"] == 45.0


def test_overlay_linux_ordinal_hole_does_not_shift(monkeypatch):
    # Device 0's card is unreadable (dropped): device index 0 has no entry and
    # keeps torch; device index 1 still gets ordinal 1, not ordinal-1-compacted-to-0.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw, "_rocm_linux_sysfs_vram_per_card_gb", lambda: {1: (0.5, 8.0)})
    devices = [_device(0, used = 0.02, total = 45.0), _device(1, used = 0.01, total = 8.0)]
    _patch_card_count(monkeypatch, 2)
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 0.02  # no ordinal 0 -> torch kept
    assert devices[1]["vram_used_gb"] == 0.5  # ordinal 1 -> device 1, not device 0


def test_overlay_linux_skips_unified_memory_card(monkeypatch):
    # Strix Halo: sysfs reports only the small dedicated slice while torch sees
    # the GTT-backed pool; the smaller sysfs total must NOT shrink the device
    # (mirrors _apply_unified_memory_correction's larger-total-wins rule).
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw, "_rocm_linux_sysfs_vram_per_card_gb", lambda: {0: (0.4, 1.0)})
    devices = [_device(0, used = 12.0, total = 96.0)]  # torch's unified pool
    _patch_card_count(monkeypatch, 1)
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
    monkeypatch.setattr(hw, "_rocm_linux_sysfs_vram_per_card_gb", lambda: {0: (40.0, 192.0)})
    devices = [_device(0, used = 1.0, total = 24.0)]  # torch partition
    _patch_card_count(monkeypatch, 1)
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 1.0  # partition figures kept
    assert devices[0]["vram_total_gb"] == 24.0


def test_overlay_linux_out_of_range_index_untouched(monkeypatch):
    # A masked host exposing physical index 5 with no card 5: keep torch data.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(
        hw, "_rocm_linux_sysfs_vram_per_card_gb", lambda: {0: (30.0, 45.0), 1: (0.5, 8.0)}
    )
    devices = [_device(5, used = 0.02, total = 45.0)]
    _patch_card_count(monkeypatch, 1)
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 0.02


def test_overlay_skips_when_amdgpu_card_count_exceeds_devices(monkeypatch):
    # An amdgpu-bound adapter HIP cannot enumerate (an unsupported older AMD GPU
    # beside a supported one) still appears in the DRM card list, so it would take
    # ordinal 0 and shift the real compute device onto the wrong card. With no
    # torch-side PCI identity to match on, a count disagreement is the signal that
    # position-in-PCI-order is NOT a sound 1:1 mapping -- keep torch's figures.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setattr(hw, "_rocm_linux_sysfs_vram_per_card_gb", lambda: {0: (30.0, 45.0)})
    _patch_card_count(monkeypatch, 2)  # 2 amdgpu cards, but ROCm exposes 1 device
    devices = [_device(0, used = 0.02, total = 45.0)]
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 0.02  # untouched


def test_overlay_applies_to_a_masked_gpu_subset(monkeypatch):
    # HIP_VISIBLE_DEVICES=1,3 on a 4-GPU host: 2 devices against 4 amdgpu cards.
    # Requiring count equality would disable the overlay for exactly the masked
    # GPUs that need it, leaving placement checks to overestimate free VRAM on
    # memory llama-server holds. Each physical index maps into the card list.
    monkeypatch.setattr(hw.platform, "system", lambda: "Linux")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1,3")
    monkeypatch.setattr(hw, "_rocm_linux_amdgpu_cards", lambda: [("", i, "") for i in range(4)])
    monkeypatch.setattr(
        hw,
        "_rocm_linux_sysfs_vram_per_card_gb",
        lambda: {0: (5.0, 48.0), 1: (30.0, 48.0), 2: (7.0, 48.0), 3: (12.0, 48.0)},
    )
    devices = [_device(1, used = 0.02, total = 48.0), _device(3, used = 0.01, total = 48.0)]
    hw._overlay_system_wide_vram(devices)
    assert devices[0]["vram_used_gb"] == 30.0  # physical 1 -> card 1
    assert devices[1]["vram_used_gb"] == 12.0  # physical 3 -> card 3


def test_gpu_device_ordinal_makes_index_unreliable(monkeypatch):
    # GPU_DEVICE_ORDINAL is a supported ROCm visibility variable that the parent
    # spec never consults, so GPU_DEVICE_ORDINAL=1 surfaces physical GPU 1 as
    # torch ordinal 0 and it gets mislabeled index 0. Any value must disable the
    # overlay, else card 0's usage lands on GPU 1.
    for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.delenv("GPU_DEVICE_ORDINAL", raising = False)
    assert hw._rocm_device_index_unreliable() is False
    monkeypatch.setenv("GPU_DEVICE_ORDINAL", "1")
    assert hw._rocm_device_index_unreliable() is True
    monkeypatch.setenv("GPU_DEVICE_ORDINAL", "  ")  # empty is not an active mask
    assert hw._rocm_device_index_unreliable() is False


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


def test_masks_layered_only_when_both_active(monkeypatch):
    for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising = False)
    assert hw._rocm_device_index_unreliable() is False
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
    assert hw._rocm_device_index_unreliable() is False  # HIP alone -> physical
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2,3")
    assert hw._rocm_device_index_unreliable() is True  # HIP over ROCR -> layered
    monkeypatch.delenv("HIP_VISIBLE_DEVICES", raising = False)
    assert hw._rocm_device_index_unreliable() is False  # ROCR alone -> physical
    # An empty value is not an active filter, so it does not layer.
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "  ")
    assert hw._rocm_device_index_unreliable() is False


def test_cuda_over_rocr_counts_as_layered(monkeypatch):
    # On ROCm the HIP layer honors CUDA_VISIBLE_DEVICES too, so a CUDA mask
    # composed over ROCR layers exactly like a HIP one: ROCR=2,3 + CUDA=1 means
    # physical GPU 3, while the spec reports the ROCR value [2,3]. Must be layered.
    for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising = False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    assert hw._rocm_device_index_unreliable() is False  # CUDA alone -> physical
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2,3")
    assert hw._rocm_device_index_unreliable() is True  # CUDA over ROCR -> layered


def test_visible_utilization_layered_masks_skip_overlay(monkeypatch):
    # ROCR_VISIBLE_DEVICES exposes physical 2,3; HIP_VISIBLE_DEVICES=1 selects
    # WITHIN that set (physical GPU 3). The reported index is then a ROCR-relative
    # ordinal, not a physical id, so the overlay must be skipped -- else it would
    # pull sysfs data from physical GPU 1.
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
    called = []
    monkeypatch.setattr(hw, "_overlay_system_wide_vram", lambda devices: called.append(1))
    result = hw.get_visible_gpu_utilization()
    assert result["index_kind"] == "physical"
    assert called == []  # layered masks -> overlay skipped
