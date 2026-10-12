# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from types import SimpleNamespace

import main

_LOGGER = SimpleNamespace(debug = lambda *args: None)


def _patch(
    monkeypatch,
    util_device,
    util_backend = "cuda",
):
    import utils.hardware as hardware
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(main, "_system_gpu_cache", None)
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda: False))
    monkeypatch.setattr(LlamaCppBackend, "_backend_lacks_gpu_lib", staticmethod(lambda: False))
    monkeypatch.setattr(hardware, "get_vulkan_inference_gpu_info", lambda: None)
    monkeypatch.setattr(hardware, "get_cross_vendor_inference_gpu_info", lambda: None)
    monkeypatch.setattr(
        hardware,
        "get_backend_visible_gpu_info",
        lambda: {
            "available": True,
            "backend": "cuda",
            "index_kind": "physical",
            "devices": [{"index": 0, "name": "NVIDIA B200", "memory_total_gb": 179.0}],
        },
    )
    calls = []

    def utilization():
        calls.append(1)
        return {"available": True, "backend": util_backend, "devices": [util_device]}

    monkeypatch.setattr(hardware, "get_visible_gpu_utilization", utilization)
    return calls


def test_system_gpu_devices_carry_the_live_telemetry_the_probe_already_read(monkeypatch):
    calls = _patch(
        monkeypatch,
        {
            "index": 0,
            "vram_used_gb": 10.0,
            "vram_total_gb": 179.0,
            "vram_utilization_pct": 5.6,
            "gpu_utilization_pct": 42.0,
            "temperature_c": 61.0,
            "power_draw_w": 312.5,
            "power_limit_w": 1000.0,
            "power_utilization_pct": 31.2,
        },
    )

    gpu, _ = main._get_cached_system_gpu_info(_LOGGER)

    (device,) = gpu["devices"]
    assert device["gpu_utilization_pct"] == 42.0
    assert device["temperature_c"] == 61.0
    assert device["power_draw_w"] == 312.5
    assert device["power_limit_w"] == 1000.0
    assert device["vram_used_gb"] == 10.0
    # No extra probe: one utilization read per cache window, as before.
    main._get_cached_system_gpu_info(_LOGGER)
    assert len(calls) == 1


def test_system_gpu_devices_omit_telemetry_the_probe_did_not_report(monkeypatch):
    _patch(monkeypatch, {"index": 0, "vram_used_gb": 1.0, "vram_total_gb": 179.0})

    gpu, _ = main._get_cached_system_gpu_info(_LOGGER)

    (device,) = gpu["devices"]
    for key in ("gpu_utilization_pct", "temperature_c", "power_draw_w", "power_limit_w"):
        assert key not in device


def test_system_gpu_devices_never_take_telemetry_from_another_backend(monkeypatch):
    _patch(
        monkeypatch,
        {"index": 0, "temperature_c": 70.0, "power_draw_w": 100.0},
        util_backend = "vulkan",
    )

    gpu, _ = main._get_cached_system_gpu_info(_LOGGER)

    (device,) = gpu["devices"]
    assert "temperature_c" not in device
    assert "power_draw_w" not in device
