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
    }
    monkeypatch.setattr(
        hardware,
        "get_backend_visible_gpu_info",
        lambda: {
            "available": True,
            "backend": "vulkan",
            "devices": [vulkan_device],
            "index_kind": "relative",
        },
    )
    monkeypatch.setattr(
        hardware,
        "get_visible_gpu_utilization",
        lambda: {"available": False, "backend": "cpu", "devices": []},
    )

    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda: True))
    monkeypatch.setattr(main, "_system_gpu_cache", None)

    result = main._get_cached_system_gpu_info(SimpleNamespace(debug = lambda *args: None))

    assert result["available"] is True
    assert result["backend"] == "vulkan"
    assert result["index_kind"] == "relative"
    assert result["gguf_gpu_ids_supported"] is False
    assert result["devices"] == [vulkan_device]
