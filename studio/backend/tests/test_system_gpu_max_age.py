# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Live monitor polls /api/system every 1-10 s; GPU readings follow it, never faster than 1 s."""

import types


def test_requested_gpu_age_is_clamped_to_one_to_ten_seconds():
    import main

    assert main._system_gpu_max_age_s(None) == main._SYSTEM_GPU_CACHE_TTL_SECONDS
    assert main._system_gpu_max_age_s(1) == 1.0
    assert main._system_gpu_max_age_s(0.1) == 1.0
    assert main._system_gpu_max_age_s(60) == main._SYSTEM_GPU_CACHE_TTL_SECONDS
    assert main._system_gpu_max_age_s(float("nan")) == main._SYSTEM_GPU_CACHE_TTL_SECONDS


def test_route_passes_the_age_only_when_asked(monkeypatch):
    import main

    seen = []

    def fake(logger, **kwargs):
        seen.append(kwargs)
        return {"available": False}, {"available": False}

    monkeypatch.setattr(main, "_get_cached_system_gpu_info", fake)
    main.get_system_info(current_subject = "test")
    main.get_system_info(current_subject = "test", gpu_max_age = 1)
    assert "max_age_s" not in seen[0]
    assert seen[1]["max_age_s"] == 1.0


def test_dense_quant_reprobe_is_throttled(monkeypatch):
    import sys

    import main

    calls = []
    monkeypatch.setitem(sys.modules, "torchao", types.ModuleType("torchao"))
    monkeypatch.setattr(main, "_refresh_dense_quant_capability", lambda: calls.append(1) or True)
    monkeypatch.setattr(main, "_dense_quant_capability", True)
    monkeypatch.setattr(main, "_dense_quant_probed_at", None)
    clock = iter([100.0, 101.0, 104.9, 105.1])
    monkeypatch.setattr(main.time, "monotonic", lambda: next(clock))
    for _ in range(4):
        main._dense_quant_supported()
    # Probed at 100, skipped at 101 and 104.9, probed again at 105.1.
    assert len(calls) == 2


def test_process_tree_children_are_reused_within_the_ttl(monkeypatch):
    import psutil

    import main

    scans = []
    me = psutil.Process()
    monkeypatch.setattr(main, "_descendants_cache", {})
    monkeypatch.setattr(type(me), "children", lambda self, recursive = False: scans.append(1) or [])
    main._cached_descendants(me)
    main._cached_descendants(me)
    assert len(scans) == 1


def test_llama_pin_support_is_not_rescanned_on_every_fast_read(monkeypatch):
    import logging

    import main
    from core.inference.llama_cpp import LlamaCppBackend

    scans = []
    monkeypatch.setattr(main, "_gguf_pin_support_cache", None)
    monkeypatch.setattr(
        LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda: scans.append(1) or False)
    )
    monkeypatch.setattr(LlamaCppBackend, "_backend_lacks_gpu_lib", staticmethod(lambda: False))
    log = logging.getLogger("t")
    main._gguf_pin_support(log, refresh = True)
    main._gguf_pin_support(log)
    main._gguf_pin_support(log)
    assert len(scans) == 1
    # A full (past-the-window) read re-resolves, so a reinstall shows within 10 s.
    main._gguf_pin_support(log, refresh = True)
    assert len(scans) == 2


def test_a_slow_gpu_probe_spaces_out_fast_polls(monkeypatch):
    import logging

    import main

    cached = ({"available": True, "devices": []}, {"available": True, "devices": []})
    now = main.time.monotonic()
    monkeypatch.setattr(main, "_system_gpu_cache", (now - 2.0, cached))
    log = logging.getLogger("t")
    # A 1.5 s probe (Windows ROCm) keeps a 2 s old reading for a 1 s poll.
    monkeypatch.setattr(main, "_system_gpu_probe_s", 1.5)
    assert main._get_cached_system_gpu_info(log, max_age_s = 1.0) is cached
    # A fast probe lets the same poll through to a fresh read.
    monkeypatch.setattr(main, "_system_gpu_probe_s", 0.05)
    monkeypatch.setattr(main, "_system_gpu_cache", (now - 2.0, cached))
    assert main._get_cached_system_gpu_info(log, max_age_s = 1.0) is not cached
