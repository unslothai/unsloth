# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Windows CUDA sysmem-fallback detector: warn-only, once, and only on the full
fingerprint (VRAM pegged, no OOM, generation an order of magnitude slow).

Every sample here is fabricated. The policy cannot be queried per process, so the only
thing there is to test is the decision made from the samples.
"""

import sys

import pytest

from core.inference.llama_stats import LlamaServerStatsLogger, SysmemFallbackWatch


class _Capture:
    def __init__(self):
        self.events = []

    def info(self, event, **kw):
        self.events.append((event, dict(kw)))

    def warning(self, event, **kw):
        self.events.append((event, dict(kw)))

    def debug(self, *a, **k):
        pass


def _watch(
    samples,
    cap,
    *,
    floor = 10.0,
    consecutive = 3,
):
    """A watch whose VRAM probe replays `samples`, one per observe()."""
    it = iter(samples)

    def probe():
        return next(it, samples[-1] if samples else None)

    return SysmemFallbackWatch(probe, floor, cap, consecutive = consecutive)


# 16 GiB card with 160 MiB free: 1 percent, the ollama#16725 reading.
_PEGGED = [(0, 160, 16303)]
# The same card with 6 GiB free.
_ROOMY = [(0, 6144, 16303)]


def test_warns_once_on_the_full_fingerprint():
    cap = _Capture()
    w = _watch([_PEGGED] * 6, cap)
    fired = [w.observe(running = 1, gen_tok_s = 0.8) for _ in range(6)]
    # Third sustained sample, not the first: one slow tick is a cold prompt.
    assert fired == [False, False, True, False, False, False]
    events = [e for e, _ in cap.events]
    assert events == ["cuda_sysmem_fallback_suspected"], events
    detail = cap.events[0][1]["detail"]
    assert "CUDA - Sysmem Fallback Policy" in detail
    assert "NVIDIA Control Panel" in detail
    # It must say the allocation SUCCEEDED and is paging, not that it failed.
    assert "pages it to system RAM" in detail
    assert "out-of-memory" in detail


def test_silent_when_vram_is_not_pegged():
    cap = _Capture()
    w = _watch([_ROOMY] * 6, cap)
    assert not any(w.observe(running = 1, gen_tok_s = 0.8) for _ in range(6))
    assert cap.events == []


def test_silent_when_generation_is_healthy():
    cap = _Capture()
    w = _watch([_PEGGED] * 6, cap)
    # A resident placement fills VRAM too; only the throughput separates the two.
    assert not any(w.observe(running = 1, gen_tok_s = 45.0) for _ in range(6))
    assert cap.events == []


def test_silent_while_idle():
    cap = _Capture()
    w = _watch([_PEGGED] * 6, cap)
    assert not any(w.observe(running = 0, gen_tok_s = 0.0) for _ in range(6))
    assert cap.events == []


def test_streak_resets_on_a_healthy_tick():
    cap = _Capture()
    w = _watch([_PEGGED] * 8, cap)
    w.observe(running = 1, gen_tok_s = 0.8)
    w.observe(running = 1, gen_tok_s = 0.8)
    w.observe(running = 1, gen_tok_s = 45.0)
    w.observe(running = 1, gen_tok_s = 0.8)
    w.observe(running = 1, gen_tok_s = 0.8)
    assert cap.events == []
    assert w.observe(running = 1, gen_tok_s = 0.8) is True


def test_streak_resets_when_the_engine_goes_idle():
    """An idle tick is not a slow tick: two slow samples either side of it are not a run."""
    cap = _Capture()
    w = _watch([_PEGGED] * 8, cap)
    w.observe(running = 1, gen_tok_s = 0.8)
    w.observe(running = 1, gen_tok_s = 0.8)
    w.observe(running = 0, gen_tok_s = 0.0)
    w.observe(running = 1, gen_tok_s = 0.8)
    w.observe(running = 1, gen_tok_s = 0.8)
    assert cap.events == []


def test_unreadable_probe_never_manufactures_the_fingerprint():
    cap = _Capture()
    w = SysmemFallbackWatch(lambda: [], 10.0, cap)
    assert not any(w.observe(running = 1, gen_tok_s = 0.8) for _ in range(6))
    w2 = SysmemFallbackWatch(lambda: (_ for _ in ()).throw(OSError("no nvidia-smi")), 10.0, cap)
    assert not any(w2.observe(running = 1, gen_tok_s = 0.8) for _ in range(6))
    assert cap.events == []


def test_stats_logger_without_a_watch_logs_exactly_what_it_did_before():
    """The flag-off path: no watch, so the loop emits engine_stats and nothing else."""
    cap = _Capture()
    lg = LlamaServerStatsLogger("http://127.0.0.1:0", cap)
    assert lg._sysmem_watch is None
    lg._interval = 0.001
    snaps = [
        {"predicted_tokens_seconds": 0.8, "requests_processing": 1.0},
        {"predicted_tokens_seconds": 0.8, "requests_processing": 1.0},
        {"predicted_tokens_seconds": 0.8, "requests_processing": 1.0},
    ]
    state = {"i": 0}

    def fake_scrape():
        i = state["i"]
        state["i"] += 1
        if i >= len(snaps):
            lg.stop()
            return None
        return snaps[i]

    lg._scrape = fake_scrape
    lg._run()
    assert {e for e, _ in cap.events} == {"engine_stats"}


def test_stats_logger_feeds_the_watch_from_the_metrics_tick():
    cap = _Capture()
    w = _watch([_PEGGED] * 6, cap)
    lg = LlamaServerStatsLogger("http://127.0.0.1:0", cap, sysmem_watch = w)
    lg._interval = 0.001
    snaps = [{"predicted_tokens_seconds": 0.8, "requests_processing": 1.0}] * 4
    state = {"i": 0}

    def fake_scrape():
        i = state["i"]
        state["i"] += 1
        if i >= len(snaps):
            lg.stop()
            return None
        return snaps[i]

    lg._scrape = fake_scrape
    lg._run()
    assert [e for e, _ in cap.events].count("cuda_sysmem_fallback_suspected") == 1


def _backend_stub():
    from core.inference.llama_cpp import LlamaCppBackend
    return LlamaCppBackend.__new__(LlamaCppBackend)


def test_factory_returns_none_off_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    assert (
        _backend_stub()._windows_sysmem_fallback_watch(
            gpu_indices = [0],
            model_bytes = 8 * 1024**3,
            is_vulkan_backend = False,
            fully_gpu_offloaded = True,
        )
        is None
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"is_vulkan_backend": True},
        {"fully_gpu_offloaded": False},
        {"model_bytes": 0},
    ],
)
def test_factory_declines_the_cases_it_cannot_speak_about(monkeypatch, kwargs):
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        LlamaCppBackend, "_nvidia_smi_free_total_mib", staticmethod(lambda *a, **k: _PEGGED)
    )
    base = {
        "gpu_indices": [0],
        "model_bytes": 8 * 1024**3,
        "is_vulkan_backend": False,
        "fully_gpu_offloaded": True,
    }
    base.update(kwargs)
    assert _backend_stub()._windows_sysmem_fallback_watch(**base) is None


def test_factory_builds_a_watch_on_windows_cuda(monkeypatch):
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(
        LlamaCppBackend, "_nvidia_smi_free_total_mib", staticmethod(lambda *a, **k: _PEGGED)
    )
    monkeypatch.setattr(LlamaCppBackend, "_nvidia_link_gib_s", staticmethod(lambda *a, **k: 25.0))
    w = _backend_stub()._windows_sysmem_fallback_watch(
        gpu_indices = [0],
        model_bytes = 8 * 1024**3,
        is_vulkan_backend = False,
        fully_gpu_offloaded = True,
    )
    assert w is not None
    # 25 GiB/s over an 8 GiB read per token: nothing paging can beat 3.125 tok/s.
    assert w._floor == pytest.approx(25.0 / 8.0)


def test_nvidia_smi_probe_gives_no_sample_when_the_mask_is_in_cuda_ordinals(monkeypatch):
    """A numeric CUDA_VISIBLE_DEVICES without CUDA_DEVICE_ORDER=PCI_BUS_ID names CUDA
    ordinals, and nvidia-smi rows are PCI indices; filtering one by the other can drop the
    launch's own card and the watch then never starts. No sample instead, the rule
    _cuda_compute_caps already applies; with the shared order the filter is exact."""
    from core.inference.llama_cpp import LlamaCppBackend

    out = "0, 160, 16303\n1, 200, 24564\n"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising = False)
    assert LlamaCppBackend._nvidia_smi_free_total_mib(runner = lambda: out) == []
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    assert LlamaCppBackend._nvidia_smi_free_total_mib(runner = lambda: out) == [(1, 200, 24564)]


def test_nvidia_smi_probe_drops_unreadable_totals(monkeypatch):
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    # "N/A" on MIG/vGPU, and a literal 0 total, are both unusable to a fraction-of-total test.
    out = "0, 160, 16303\n1, 900, N/A\n3, 100, 0\n2, 512, 24564\n"
    rows = LlamaCppBackend._nvidia_smi_free_total_mib(runner = lambda: out)
    assert rows == [(0, 160, 16303), (2, 512, 24564)]
    assert LlamaCppBackend._nvidia_smi_free_total_mib([2], runner = lambda: out) == [(2, 512, 24564)]
    assert LlamaCppBackend._nvidia_smi_free_total_mib(runner = lambda: None) == []
