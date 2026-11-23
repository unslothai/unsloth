# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What /api/system and the training poll spawn per request, and what they give back.

Display polls reuse recent Vulkan / amd-smi / ioreg answers; fit and refresh paths still read
fresh. Also: CPU percent sampling, process-tree memory, embedder release, upload read bounds,
and a broken torch import retried once per detection epoch.
"""

import asyncio
import builtins
import subprocess
import sys
import types

import pytest

from utils import gpu_memory_events
from utils.hardware import gpu_query


@pytest.fixture
def vulkan(monkeypatch):
    from core.inference import llama_cpp

    monkeypatch.setattr(llama_cpp, "_vulkan_inventory_cache", {})
    calls = []
    answer = {
        "rows": [{"index": 0, "free_mib": 1024, "total_mib": 2048, "is_igpu": False, "name": ""}]
    }

    def probe(binary = None):
        calls.append(binary)
        return [dict(row) for row in answer["rows"]]

    monkeypatch.setattr(llama_cpp.LlamaCppBackend, "_run_vulkan_probe", staticmethod(probe))
    return llama_cpp, calls, answer


def test_vulkan_inventory_is_reused_by_display_polls(vulkan):
    llama_cpp, calls, _ = vulkan
    with gpu_query.display_reads(max_stale = 0):
        first = llama_cpp.LlamaCppBackend.vulkan_device_inventory("/bin/llama-server")
        first[0]["name"] = "mutated"
        second = llama_cpp.LlamaCppBackend.vulkan_device_inventory("/bin/llama-server")
    assert len(calls) == 1
    assert second[0]["name"] == "Vulkan0"


def test_vulkan_inventory_reprobes_outside_display_and_after_memory_events(vulkan):
    llama_cpp, calls, _ = vulkan
    inventory = llama_cpp.LlamaCppBackend.vulkan_device_inventory
    inventory("/bin/llama-server")
    inventory("/bin/llama-server")
    assert len(calls) == 2  # refresh_memory path: no display_reads, always live
    with gpu_query.display_reads(max_stale = 0):
        inventory("/bin/llama-server")  # served from the live read above
        assert len(calls) == 2
        gpu_memory_events.invalidate_gpu_memory("model loaded")
        inventory("/bin/llama-server")
        with gpu_query.fresh_reads():
            inventory("/bin/llama-server")
    assert len(calls) == 4


def test_vulkan_inventory_never_serves_an_empty_answer_or_a_load_memo(vulkan, monkeypatch):
    llama_cpp, calls, answer = vulkan
    inventory = llama_cpp.LlamaCppBackend.vulkan_device_inventory
    answer["rows"] = []
    with gpu_query.display_reads(max_stale = 0):
        inventory("/bin/llama-server")
        inventory("/bin/llama-server")
    assert len(calls) == 2
    monkeypatch.setattr(llama_cpp, "_load_probe_memo_armed", lambda: True)
    with gpu_query.display_reads(max_stale = 0):
        inventory("/bin/llama-server")
    assert len(calls) == 3


def test_system_route_reads_vulkan_inside_display_reads_unless_refreshing():
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding = "utf-8")
    block = src[src.index("vulkan_info = get_vulkan_inference_gpu_info()") - 200 :]
    assert "nullcontext() if refresh_memory else gpu_query.display_reads" in block[:220]


@pytest.fixture
def amd(monkeypatch):
    from utils.hardware import amd as amd_mod

    monkeypatch.setattr(amd_mod, "_display_cache", {})
    # An empty mask (CPU-only CI) would make the primary GPU id None and skip amd-smi.
    for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising = False)
    calls = []
    answer = {"data": [{"gpu": 0, "usage": {"gfx_activity": {"value": 50, "unit": "%"}}}]}

    def run(*args, **kwargs):
        calls.append(args)
        return answer["data"]

    monkeypatch.setattr(amd_mod, "_run_amd_smi", run)
    return amd_mod, calls, answer


def test_amd_smi_metric_is_coalesced_for_display_polls(amd):
    amd_mod, calls, _ = amd
    with gpu_query.display_reads():
        amd_mod.get_visible_gpu_utilization([0])
        amd_mod.get_visible_gpu_utilization([0])
        amd_mod.get_primary_gpu_utilization()
        amd_mod.get_primary_gpu_utilization()
    assert calls == [("metric",), ("metric", "-g", "0")]


def test_amd_smi_metric_stays_live_for_fit_and_after_failures(amd):
    amd_mod, calls, answer = amd
    amd_mod.get_visible_gpu_utilization([0])
    amd_mod.get_visible_gpu_utilization([0])
    amd_mod.get_gpu_vram_report()
    assert len(calls) == 3
    answer["data"] = None
    with gpu_query.display_reads():
        amd_mod.get_visible_gpu_utilization([0])
        amd_mod.get_visible_gpu_utilization([0])
    assert len(calls) == 5


def test_cpu_percent_is_sampled_at_most_once_per_window(monkeypatch):
    psutil = pytest.importorskip("psutil")

    import main

    clock = [100.0]
    readings = iter([12.5, 40.0])
    calls = []

    def cpu_percent(interval = None):
        calls.append(interval)
        return next(readings)

    monkeypatch.setattr(psutil, "cpu_percent", cpu_percent)
    monkeypatch.setattr(main, "time", types.SimpleNamespace(monotonic = lambda: clock[0]))
    monkeypatch.setattr(main, "_cpu_percent_reading", (99.5, None))

    assert main._cpu_usage_percent() == 12.5
    clock[0] += 0.4
    assert main._cpu_usage_percent() == 12.5
    clock[0] += 1.0
    assert main._cpu_usage_percent() == 40.0
    assert calls == [None, None]


def test_cpu_percent_is_primed_at_import():
    pytest.importorskip("psutil")
    import main
    assert main._cpu_percent_reading is not None


def test_process_tree_memory_adds_children_and_skips_vanished_ones():
    psutil = pytest.importorskip("psutil")

    import main

    mib = 1024**2

    class Proc:
        def __init__(
            self,
            rss,
            exc = None,
            children = (),
        ):
            self._rss, self._exc, self._children = rss, exc, list(children)

        def memory_info(self):
            if self._exc:
                raise self._exc
            return types.SimpleNamespace(rss = self._rss)

        def children(self, recursive = False):
            assert recursive
            return self._children

    root = Proc(
        100 * mib,
        children = [
            Proc(900 * mib),
            Proc(0, psutil.NoSuchProcess(1)),
            Proc(0, psutil.AccessDenied(2)),
        ],
    )
    assert main._process_tree_rss_mb(root) == 1000

    class Locked(Proc):
        def children(self, recursive = False):
            raise psutil.AccessDenied(3)

    assert main._process_tree_rss_mb(Locked(100 * mib)) == 100


def test_mlx_visible_utilization_carries_apple_telemetry(monkeypatch):
    from utils.hardware import apple
    from utils.hardware import hardware as hw

    monkeypatch.setattr(hw, "get_device", lambda: hw.DeviceType.MLX)
    monkeypatch.setattr(
        hw,
        "get_gpu_memory_info",
        lambda: {
            "available": True,
            "allocated_gb": 4.0,
            "total_gb": 32.0,
            "free_gb": 20.0,
            "utilization_pct": 12.5,
        },
    )
    monkeypatch.setattr(hw, "_read_apple_gpu_stats", lambda: {"utilization_pct": 37})
    monkeypatch.setattr(apple, "read_gpu_temperature_c", lambda: 51.5)
    monkeypatch.setattr(apple, "read_gpu_power_w", lambda: 6.2)

    (device,) = hw.get_visible_gpu_utilization()["devices"]
    assert (device["gpu_utilization_pct"], device["temperature_c"], device["power_draw_w"]) == (
        37,
        51.5,
        6.2,
    )
    assert device["vram_total_gb"] == 32.0 and device["vram_free_gb"] == 20.0

    def broken():
        raise OSError("no SMC")

    monkeypatch.setattr(apple, "read_gpu_temperature_c", broken)
    (device,) = hw.get_visible_gpu_utilization()["devices"]
    assert device["temperature_c"] is None and device["vram_used_gb"] == 4.0


def test_agx_ioreg_is_read_once_per_display_refresh(monkeypatch):
    from utils.hardware import hardware as hw

    monkeypatch.setattr(hw, "_apple_gpu_stats_cache", None)
    calls = []
    block = '"PerformanceStatistics" = {"Device Utilization %"=21,"In use system memory"=1024}'

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return types.SimpleNamespace(stdout = block.encode())

    monkeypatch.setattr(subprocess, "run", fake_run)
    # Visibility reads live and seeds the reading; the display read right after reuses it.
    assert hw._read_apple_gpu_stats() == {"utilization_pct": 21, "vram_used_bytes": 1024}
    with gpu_query.display_reads(max_stale = 0):
        assert hw._read_apple_gpu_stats()["utilization_pct"] == 21
    assert len(calls) == 1
    # Fit and placement never take the cached reading.
    hw._read_apple_gpu_stats()
    assert len(calls) == 2


def test_releasing_the_st_embedder_frees_the_allocator_cache(monkeypatch):
    from core.rag import embeddings
    import utils.hardware as hardware

    cleared = []
    monkeypatch.setattr(hardware, "clear_gpu_cache", lambda: cleared.append(True))
    monkeypatch.setattr(embeddings, "_model", object())
    monkeypatch.setattr(embeddings, "_name", "m")

    # Before detection publishes, a release must not trigger it through clear_gpu_cache().
    monkeypatch.setattr(hardware.hardware, "DEVICE", None)
    assert embeddings._release_st_model() is True
    assert cleared == []

    monkeypatch.setattr(hardware.hardware, "DEVICE", hardware.DeviceType.CUDA)
    monkeypatch.setattr(embeddings, "_model", object())
    assert embeddings._release_st_model() is True
    assert cleared == [True]
    assert embeddings._release_st_model() is False
    assert cleared == [True]

    def boom():
        raise RuntimeError("CUDA error")

    monkeypatch.setattr(hardware, "clear_gpu_cache", boom)
    monkeypatch.setattr(embeddings, "_model", object())
    assert embeddings._release_st_model() is True


class _Upload:
    def __init__(self, data: bytes, content_type: str):
        self._data, self.content_type, self.sizes = data, content_type, []

    async def read(self, size: int = -1) -> bytes:
        self.sizes.append(size)
        return self._data if size < 0 else self._data[:size]


def test_video_reference_uploads_read_at_most_one_byte_past_the_cap(monkeypatch):
    import routes.video as video_routes

    monkeypatch.setattr(video_routes, "_VIDEO_REFERENCE_AUDIO_MAX_BYTES", 8)
    monkeypatch.setattr(video_routes, "VIDEO_INPUT_REFERENCE_MAX_BYTES", 16)

    audio = _Upload(b"x" * 1000, "audio/wav")
    with pytest.raises(Exception) as exc:
        asyncio.run(video_routes._audio_upload_to_data_url(audio, "reference_audios[0]"))
    assert audio.sizes == [9]
    assert "is too large; the limit is 24 MB." in str(getattr(exc.value, "detail", exc.value))

    image = _Upload(b"x" * 1000, "image/png")
    monkeypatch.setattr(video_routes, "UploadFile", _Upload)
    with pytest.raises(Exception) as exc:
        asyncio.run(video_routes._reference_to_data_url(image))
    assert image.sizes == [17]
    assert "is too large; the limit is 0 MB." in str(getattr(exc.value, "detail", exc.value))


def test_a_broken_torch_import_is_retried_once_per_detection_epoch(monkeypatch):
    from utils import torch_warmup
    from utils.hardware import hardware as hw

    monkeypatch.setattr(hw, "TORCH_IMPORT_ERROR", hw.TORCH_IMPORT_ERROR)
    monkeypatch.setattr(hw, "_TORCH_IMPORT_FAILED_EPOCH", None)
    monkeypatch.setattr(torch_warmup, "purge_partial_import", lambda name: None)
    for name in [n for n in sys.modules if n == "torch" or n.startswith("torch.")]:
        monkeypatch.delitem(sys.modules, name)
    errors = []
    monkeypatch.setattr(hw.logger, "error", lambda *a, **k: errors.append(a))
    real_import = builtins.__import__
    attempts = []

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            attempts.append(name)
            raise OSError("libcudart.so.12: cannot open shared object file")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert hw._has_torch() is False
    assert hw._has_torch() is False
    assert len(attempts) == 1 and len(errors) == 1
    assert hw.TORCH_IMPORT_ERROR is not None

    monkeypatch.setattr(hw, "DETECTION_EPOCH", hw.DETECTION_EPOCH + 1)
    assert hw._has_torch() is False
    assert len(attempts) == 2
