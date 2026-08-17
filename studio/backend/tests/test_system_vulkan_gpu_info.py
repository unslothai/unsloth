# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from types import SimpleNamespace

import main


def test_system_gpu_info_preserves_vulkan_visibility_metrics(monkeypatch):
    import utils.hardware as hardware

    vulkan_device = {
        "index": 0,
        "index_kind": "relative",
        "visible_ordinal": 0,
        "name": "Vulkan0",
        "memory_total_gb": 8.0,
        "vram_used_gb": 0.77,
        "vram_free_gb": 7.23,
        "vram_utilization_pct": 9.6,
        "shared_memory": False,
    }
    monkeypatch.setattr(
        hardware,
        "get_backend_visible_gpu_info",
        lambda: {
            "available": False,
            "backend": "cpu",
            "devices": [],
            "index_kind": "relative",
        },
    )
    monkeypatch.setattr(
        hardware,
        "get_visible_gpu_utilization",
        lambda: {"available": False, "backend": "cpu", "devices": []},
    )
    monkeypatch.setattr(
        hardware,
        "get_vulkan_inference_gpu_info",
        lambda: {
            "available": True,
            "backend": "vulkan",
            "devices": [vulkan_device],
            "index_kind": "relative",
        },
    )

    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda: True))
    monkeypatch.setattr(main, "_system_gpu_cache", None)

    gpu, inference_gpu = main._get_cached_system_gpu_info(SimpleNamespace(debug = lambda *args: None))

    assert gpu["available"] is False
    assert gpu["backend"] == "cpu"
    assert gpu["index_kind"] == "relative"
    # The training inventory must not advertise physical pins for a Vulkan
    # llama.cpp build. Its ordinals live in inference_gpu below.
    assert gpu["gguf_gpu_ids_supported"] is False
    # Torch's view stays empty; the ggml ordinals stay in inference_gpu.
    assert gpu["devices"] == []
    assert inference_gpu["backend"] == "vulkan"
    assert inference_gpu["devices"] == [vulkan_device]


def test_system_gpu_info_withholds_gguf_pin_when_the_vulkan_probe_enumerates_nothing(monkeypatch):
    """A Vulkan build whose probe returns no ordinals has nothing valid to pin,
    so the picker must be told pins are unsupported rather than offered an empty
    namespace it would 400 on."""
    import utils.hardware as hardware

    monkeypatch.setattr(
        hardware,
        "get_backend_visible_gpu_info",
        lambda: {"available": False, "backend": "cpu", "devices": [], "index_kind": "relative"},
    )
    monkeypatch.setattr(
        hardware,
        "get_visible_gpu_utilization",
        lambda: {"available": False, "backend": "cpu", "devices": []},
    )
    monkeypatch.setattr(hardware, "get_vulkan_inference_gpu_info", lambda: None)

    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda: True))
    monkeypatch.setattr(main, "_system_gpu_cache", None)

    gpu, _ = main._get_cached_system_gpu_info(SimpleNamespace(debug = lambda *args: None))

    assert gpu["gguf_gpu_ids_supported"] is False


def test_system_gpu_info_keeps_forced_vulkan_separate_from_training_metrics(monkeypatch):
    import utils.hardware as hardware

    monkeypatch.setattr(
        hardware,
        "get_backend_visible_gpu_info",
        lambda: {
            "available": True,
            "backend": "cuda",
            "devices": [{"index": 0, "name": "CUDA0", "memory_total_gb": 24.0}],
        },
    )
    monkeypatch.setattr(
        hardware,
        "get_visible_gpu_utilization",
        lambda: {
            "available": True,
            "backend": "cuda",
            "devices": [
                {
                    "index": 0,
                    "vram_total_gb": 24.0,
                    "vram_used_gb": 6.0,
                    "vram_utilization_pct": 25.0,
                }
            ],
        },
    )
    monkeypatch.setattr(
        hardware,
        "get_vulkan_inference_gpu_info",
        lambda: {
            "available": True,
            "backend": "vulkan",
            "devices": [
                {
                    "index": 0,
                    "name": "Vulkan0",
                    "memory_total_gb": 8.0,
                    "vram_used_gb": 1.0,
                    "vram_free_gb": 7.0,
                    "vram_utilization_pct": 12.5,
                    "shared_memory": False,
                }
            ],
            "index_kind": "relative",
        },
    )

    from core.inference.llama_cpp import LlamaCppBackend
    from utils.hardware import DeviceType

    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda: True))
    monkeypatch.setattr(hardware, "get_device", lambda: DeviceType.CUDA)
    monkeypatch.setattr(main, "_system_gpu_cache", None)

    gpu, inference_gpu = main._get_cached_system_gpu_info(SimpleNamespace(debug = lambda *args: None))

    assert gpu["backend"] == "cuda"
    assert gpu["devices"][0]["vram_used_gb"] == 6.0
    assert inference_gpu["backend"] == "vulkan"
    assert inference_gpu["devices"][0]["vram_used_gb"] == 1.0
    # Probed devices exist, so the ordinals are known and picks are offered.
    assert inference_gpu["gguf_gpu_ids_supported"] is True


def test_system_gpu_info_does_not_merge_metrics_across_backend_index_spaces(monkeypatch):
    import utils.hardware as hardware

    vulkan_device = {
        "index": 0,
        "name": "Vulkan0",
        "memory_total_gb": 8.0,
        "vram_used_gb": 1.0,
        "vram_free_gb": 7.0,
        "vram_utilization_pct": 12.5,
    }
    monkeypatch.setattr(
        hardware,
        "get_backend_visible_gpu_info",
        lambda: {"available": True, "backend": "vulkan", "devices": [vulkan_device]},
    )
    monkeypatch.setattr(
        hardware,
        "get_visible_gpu_utilization",
        lambda: {
            "available": True,
            "backend": "cuda",
            "devices": [
                {
                    "index": 0,
                    "vram_total_gb": 24.0,
                    "vram_used_gb": 20.0,
                    "vram_utilization_pct": 83.3,
                }
            ],
        },
    )

    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda: True))
    monkeypatch.setattr(main, "_system_gpu_cache", None)

    gpu, inference_gpu = main._get_cached_system_gpu_info(SimpleNamespace(debug = lambda *args: None))

    assert gpu["devices"] == [vulkan_device]
    assert inference_gpu == gpu


def test_vulkan_inference_gpu_uses_real_device_names_and_igpu_flag(monkeypatch):
    """The picker and the GPU labels need ggml's real device description, not a
    Vulkan<i> placeholder, and an explicit iGPU flag rather than inferring one
    from a zero total. Memory still comes from _get_gpu_memory so the iGPU host
    reserve is applied; budgeting off the raw shared total would hand out the
    whole machine's RAM with no OS headroom.
    """
    from core.inference import llama_cpp
    from core.inference.llama_cpp import LlamaCppBackend
    from utils.hardware.hardware import get_vulkan_inference_gpu_info

    monkeypatch.setattr(
        LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda binary = None: True)
    )
    monkeypatch.setattr(
        llama_cpp,
        "_apply_igpu_host_reserve_mib",
        lambda free_mib, is_igpu: 12 * 1024 if is_igpu else free_mib,
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "vulkan_device_inventory",
        staticmethod(
            lambda binary = None: [
                {
                    "index": 0,
                    "name": "AMD Radeon RX 9070 XT",
                    "free_mib": 15 * 1024,
                    "total_mib": 16 * 1024,
                    "is_igpu": False,
                },
                {
                    "index": 1,
                    "name": "AMD Radeon(TM) 8060S Graphics",
                    "free_mib": 89 * 1024,
                    "total_mib": 91 * 1024,
                    "is_igpu": True,
                },
            ]
        ),
    )

    info = get_vulkan_inference_gpu_info()
    assert info is not None and info["index_kind"] == "vulkan"
    dgpu, igpu = info["devices"]

    assert dgpu["name"] == "AMD Radeon RX 9070 XT"
    assert dgpu["index_kind"] == "vulkan"
    assert dgpu["shared_memory"] is False
    assert dgpu["memory_total_gb"] == 16.0

    assert igpu["name"] == "AMD Radeon(TM) 8060S Graphics"
    assert igpu["shared_memory"] is True
    # The capped free budget from _get_gpu_memory, NOT the 91 GiB raw total.
    assert igpu["memory_total_gb"] == 12.0


def test_vulkan_inference_gpu_uses_inventory_fallback_names(monkeypatch):
    """The inventory's fallback name must flow through unchanged."""
    from core.inference.llama_cpp import LlamaCppBackend
    from utils.hardware.hardware import get_vulkan_inference_gpu_info

    monkeypatch.setattr(
        LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda binary = None: True)
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "vulkan_device_inventory",
        staticmethod(
            lambda binary = None: [
                {
                    "index": 0,
                    "name": "Vulkan0",
                    "free_mib": 15 * 1024,
                    "total_mib": 16 * 1024,
                    "is_igpu": False,
                }
            ]
        ),
    )

    info = get_vulkan_inference_gpu_info()
    assert info["devices"][0]["name"] == "Vulkan0"
    assert info["devices"][0]["memory_total_gb"] == 16.0
