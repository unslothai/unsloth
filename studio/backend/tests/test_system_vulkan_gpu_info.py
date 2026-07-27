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
    assert gpu["gguf_gpu_ids_supported"] is False
    assert gpu["devices"] == []
    assert inference_gpu["backend"] == "vulkan"
    assert inference_gpu["devices"] == [vulkan_device]


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
    assert inference_gpu["gguf_gpu_ids_supported"] is False


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
