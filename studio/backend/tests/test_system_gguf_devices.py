# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""/api/system GPU payload: llama-server (Vulkan) device inventory.

On a Vulkan llama.cpp build, llama-server can drive cards the torch backend
can't see (e.g. a pre-ROCm RX 480 next to a ROCm-only RX 9070 XT). The system
payload must surface that inventory as ``gguf_devices`` -- in ggml's Vulkan
ordinal space, the one /load pins with ``--device Vulkan<i>`` -- so GGUF fit
labels budget against the VRAM llama-server actually uses and the GPU picker
can offer both cards, while ``devices`` stays the torch view training relies on.
"""

import logging
import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


@pytest.fixture(scope = "module")
def main_module():
    import main as _main
    return _main


_TORCH_VIEW = {
    "available": True,
    "backend": "rocm",
    "devices": [
        {
            "index": 0,
            "index_kind": "physical",
            "visible_ordinal": 0,
            "name": "AMD Radeon RX 9070 XT",
            "memory_total_gb": 15.92,
        }
    ],
    "index_kind": "physical",
}

_VULKAN_INVENTORY = [
    {
        "index": 0,
        "name": "AMD Radeon RX 9070 XT",
        "free_mib": 15 * 1024,
        "total_mib": 16 * 1024,
        "is_igpu": False,
    },
    {
        "index": 1,
        "name": "Radeon (TM) RX 480 Graphics",
        "free_mib": 7 * 1024,
        "total_mib": 8 * 1024,
        "is_igpu": False,
    },
]


def _gpu_info(main_module, monkeypatch, *, is_vulkan, inventory):
    import utils.hardware as hw
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(hw, "get_backend_visible_gpu_info", lambda: dict(_TORCH_VIEW))
    monkeypatch.setattr(hw, "get_visible_gpu_utilization", lambda: {"devices": []})
    # Never let the real get_device() run: it would trigger detect_hardware(),
    # whose module globals (IS_ROCM, DEVICE) leak into later tests in the same
    # process and flip their probe paths.
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(
        LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda binary = None: is_vulkan)
    )
    monkeypatch.setattr(
        LlamaCppBackend, "vulkan_device_inventory", staticmethod(lambda binary = None: inventory)
    )
    monkeypatch.setattr(main_module, "_system_gpu_cache", None)
    monkeypatch.setattr(main_module, "_gguf_devices_cache", None)
    return main_module._get_cached_system_gpu_info(logging.getLogger(__name__))


def test_vulkan_build_surfaces_llama_server_devices(main_module, monkeypatch):
    info = _gpu_info(main_module, monkeypatch, is_vulkan = True, inventory = _VULKAN_INVENTORY)

    # Torch view untouched: training budgets keep the ROCm-only card set.
    assert [d["name"] for d in info["devices"]] == ["AMD Radeon RX 9070 XT"]

    # The llama-server inventory carries both cards with real totals, in ggml
    # Vulkan ordinal space.
    assert [(d["index"], d["index_kind"], d["name"]) for d in info["gguf_devices"]] == [
        (0, "vulkan", "AMD Radeon RX 9070 XT"),
        (1, "vulkan", "Radeon (TM) RX 480 Graphics"),
    ]
    assert [d["memory_total_gb"] for d in info["gguf_devices"]] == [16.0, 8.0]
    assert [d["vram_free_gb"] for d in info["gguf_devices"]] == [15.0, 7.0]

    # Picks are valid Vulkan ordinals now, so the picker may offer them.
    assert info["gguf_gpu_ids_supported"] is True
    assert info["gguf_backend_is_vulkan"] is True


def test_vulkan_build_with_failed_probe_keeps_picks_unsupported(main_module, monkeypatch):
    info = _gpu_info(main_module, monkeypatch, is_vulkan = True, inventory = [])
    assert info["gguf_devices"] == []
    # No enumerable ordinal space -> the frontend has no valid picks to offer.
    assert info["gguf_gpu_ids_supported"] is False
    # ...but the frontend must still know this is a Vulkan build so it treats the
    # empty inventory as an unknown GGUF budget (0) rather than reusing the torch
    # VRAM total, which on a mixed host would overclaim VRAM /load can't place.
    assert info["gguf_backend_is_vulkan"] is True


def test_vulkan_inventory_survives_gpu_cache_refreshes_without_reprobing(main_module, monkeypatch):
    # /api/system pollers (floating monitor, Resources tab) expire the 10s GPU
    # cache continuously; the Vulkan inventory spawns a probe subprocess, so it
    # rides its own longer TTL instead of re-probing on every refresh.
    import utils.hardware as hw
    from core.inference.llama_cpp import LlamaCppBackend

    calls = {"n": 0}

    def counting_inventory(binary = None):
        calls["n"] += 1
        return list(_VULKAN_INVENTORY)

    monkeypatch.setattr(hw, "get_backend_visible_gpu_info", lambda: dict(_TORCH_VIEW))
    monkeypatch.setattr(hw, "get_visible_gpu_utilization", lambda: {"devices": []})
    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.CUDA)
    monkeypatch.setattr(
        LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda binary = None: True)
    )
    monkeypatch.setattr(
        LlamaCppBackend, "vulkan_device_inventory", staticmethod(counting_inventory)
    )
    monkeypatch.setattr(main_module, "_system_gpu_cache", None)
    monkeypatch.setattr(main_module, "_gguf_devices_cache", None)

    logger = logging.getLogger(__name__)
    first = main_module._get_cached_system_gpu_info(logger)
    main_module._system_gpu_cache = None  # simulate the 10s GPU cache expiring
    second = main_module._get_cached_system_gpu_info(logger)

    assert calls["n"] == 1
    assert second["gguf_devices"] == first["gguf_devices"]


def test_non_vulkan_build_reports_no_gguf_inventory(main_module, monkeypatch):
    info = _gpu_info(main_module, monkeypatch, is_vulkan = False, inventory = _VULKAN_INVENTORY)
    # CUDA/ROCm llama builds see the same devices torch does; no separate list.
    assert info["gguf_devices"] == []
    assert info["gguf_gpu_ids_supported"] is True
    # Not a Vulkan build: the frontend keeps budgeting GGUF against the torch
    # total, since llama-server runs on those same devices.
    assert info["gguf_backend_is_vulkan"] is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
