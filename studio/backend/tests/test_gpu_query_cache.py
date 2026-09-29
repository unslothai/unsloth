# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Cached, coalesced nvidia-smi reads, driven by a real fake ``nvidia-smi`` on PATH."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from core.inference.llama_cpp import LlamaCppBackend
from utils import gpu_memory_events
from utils.hardware import gpu_query, nvidia

_FAKE = r"""#!{python}
import json, os, sys, time
state = json.load(open({state!r}))
with open({calls!r}, "a") as f:
    f.write(" ".join(sys.argv[1:]) + "\n")
time.sleep(state.get("delay", 0))
if state.get("exit"):
    sys.stderr.write("NVIDIA-SMI has failed\n")
    sys.exit(state["exit"])
args = sys.argv[1:]
gpus = state["gpus"]
if args[:1] == ["-L"]:
    for i, g in enumerate(gpus):
        print(f"GPU {{i}}: {{g['name']}} (UUID: {{g['uuid']}})")
    sys.exit(0)
if args[:1] == ["topo"]:
    print("\tGPU0\tGPU1\nGPU0\t X \tNV18\nGPU1\tNV18\t X \n\nLegend:\n")
    sys.exit(0)
fields = [a.split("=", 1)[1].split(",") for a in args if a.startswith("--query-gpu=")][0]
nounits = any("nounits" in a for a in args if a.startswith("--format="))
for i, g in enumerate(gpus):
    total = g["total"]
    free = g["free"]
    vals = {{
        "index": str(i), "uuid": g["uuid"], "name": g["name"], "compute_cap": "10.0",
        "memory.total": str(total), "memory.free": str(free), "memory.used": str(total - free),
        "utilization.gpu": str(g.get("util", 0)), "temperature.gpu": "40",
        "power.draw": "100.0", "power.limit": "1000.0",
    }}
    out = []
    for name in fields:
        v = vals[name]
        if not nounits and name.startswith("memory."):
            v += " MiB"
        out.append(v)
    print(", ".join(out))
"""


class FakeSmi:
    def __init__(self, root: Path):
        self.root = root
        self.state_path = root / "state.json"
        self.calls_path = root / "calls.log"
        self.calls_path.write_text("")
        self.state = {
            "delay": 0,
            "exit": 0,
            "gpus": [
                {"name": "NVIDIA B200", "uuid": "GPU-aaaa", "total": 183359, "free": 180000},
                {"name": "NVIDIA B200", "uuid": "GPU-bbbb", "total": 183359, "free": 170000},
            ],
        }
        self.save()
        exe = root / "nvidia-smi"
        exe.write_text(
            _FAKE.format(
                python = sys.executable, state = str(self.state_path), calls = str(self.calls_path)
            )
        )
        exe.chmod(0o755)

    def save(self):
        self.state_path.write_text(json.dumps(self.state))

    def set(self, **kw):
        self.state.update(kw)
        self.save()

    def set_free(self, *free_mib):
        for gpu, free in zip(self.state["gpus"], free_mib):
            gpu["free"] = free
        self.save()

    def calls(self, needle: str = "") -> int:
        return sum(1 for line in self.calls_path.read_text().splitlines() if needle in line)

    def wait_for_call(
        self,
        needle: str,
        count: int,
        timeout: float = 30.0,
    ) -> None:
        """Block until ``count`` children matching ``needle`` have started (each reads its state first).

        A fixed sleep after ``Thread.start()`` raced on loaded CI: a child spawned late read the state the
        test set for the NEXT reading."""
        deadline = time.monotonic() + timeout
        while self.calls(needle) < count:
            assert time.monotonic() < deadline, f"no {needle!r} child started within {timeout}s"
            time.sleep(0.01)


@pytest.fixture
def smi(tmp_path, monkeypatch):
    fake = FakeSmi(tmp_path)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.delenv("UNSLOTH_GPU_QUERY_CACHE", raising = False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    # No NVML child unless a test asks for one: it would read this host's real driver.
    monkeypatch.setenv("UNSLOTH_NVIDIA_LIBRARY_PROBE", "0")
    monkeypatch.setattr(nvidia, "_nvidia_smi_executable", lambda: "nvidia-smi")
    gpu_query.reset()
    yield fake
    gpu_query.reset()


@pytest.fixture
def llama_probe(monkeypatch):
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b = None: False))
    monkeypatch.setattr(
        LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: "/opt/llama-server")
    )
    monkeypatch.setattr(LlamaCppBackend, "_get_gpu_memory_nvml", staticmethod(lambda: []))
    monkeypatch.setattr(
        LlamaCppBackend, "_get_gpu_memory_amd_smi", staticmethod(lambda *a, **k: [])
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_widen_integrated_cuda_rows", staticmethod(lambda rows: rows)
    )
    monkeypatch.setitem(sys.modules, "torch", None)
    return LlamaCppBackend._get_gpu_memory


def _free_by_index(result):
    return {d["index"]: round(d["vram_total_gb"] - d["vram_used_gb"], 2) for d in result["devices"]}


def test_classification():
    assert gpu_query.classify(["nvidia-smi", "-L"]) == gpu_query.STATIC
    assert gpu_query.classify(["nvidia-smi", "topo", "-m"]) == gpu_query.STATIC
    assert (
        gpu_query.classify(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv"])
        == gpu_query.STATIC
    )
    assert (
        gpu_query.classify(["nvidia-smi", "--query-gpu=index,compute_cap", "--format=csv"])
        == gpu_query.STATIC
    )
    live = ["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv"]
    assert gpu_query.classify(live) == gpu_query.CRITICAL
    with gpu_query.display_reads():
        assert gpu_query.classify(live) == gpu_query.DISPLAY
    assert gpu_query.classify(["nvidia-smi", "-q"]) == gpu_query.CRITICAL


def test_concurrent_identical_display_queries_start_one_child(smi):
    smi.set(delay = 0.6)
    results, errors = [], []

    def worker():
        try:
            with gpu_query.display_reads():
                results.append(nvidia.get_visible_gpu_utilization([0, 1]))
        except Exception as e:  # pragma: no cover - surfaced below
            errors.append(e)

    threads = [threading.Thread(target = worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert not errors
    assert len(results) == 16
    assert all(r == results[0] for r in results)
    assert results[0]["available"] is True
    assert smi.calls("--query-gpu") == 1
    assert gpu_query.stats()["coalesced"] >= 1


def test_concurrent_fit_checks_each_run_the_cli(smi, llama_probe):
    smi.set(delay = 0.3)
    threads = [threading.Thread(target = llama_probe) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert smi.calls("memory.free") == 4


def test_concurrent_fit_checks_never_answer_from_before_they_started(smi, llama_probe):
    lock = threading.Lock()
    current = [1000]
    stop = threading.Event()

    def writer():
        while not stop.is_set():
            with lock:
                current[0] += 1
                for gpu in smi.state["gpus"]:
                    gpu["free"] = current[0]
                tmp = smi.state_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(smi.state))
                os.replace(tmp, smi.state_path)
            time.sleep(0.01)

    violations, errors = [], []

    def reader():
        for _ in range(5):
            with lock:
                floor = current[0]
            try:
                rows = llama_probe()
            except Exception as e:  # pragma: no cover - surfaced below
                errors.append(e)
                continue
            if not rows or rows[0][1] < floor:
                violations.append((floor, rows))

    w = threading.Thread(target = writer)
    w.start()
    readers = [threading.Thread(target = reader) for _ in range(16)]
    for t in readers:
        t.start()
    for t in readers:
        t.join(60)
    stop.set()
    w.join(5)
    assert not errors
    assert violations == []
    assert smi.calls("memory.free") == 80


def test_different_queries_are_not_merged(smi):
    nvidia.get_visible_gpu_utilization([0, 1])
    nvidia.get_primary_gpu_utilization()
    assert smi.calls("--query-gpu") == 2


def test_fit_checks_always_read_afresh(smi, llama_probe):
    """Another process's allocation raises no Studio event, so the next fit check must see it."""
    assert llama_probe() == [(0, 180000, 183359), (1, 170000, 183359)]
    smi.set_free(1000, 1000)  # another process took nearly all of both GPUs, no invalidation
    assert llama_probe() == [(0, 1000, 183359), (1, 1000, 183359)]
    assert smi.calls("memory.free") == 2


def test_display_reads_are_served_stale_while_one_child_refreshes(smi, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GPU_QUERY_DISPLAY_TTL", "0.2")
    with gpu_query.display_reads():
        first = _free_by_index(nvidia.get_visible_gpu_utilization([0, 1]))
    smi.set(delay = 0.5)
    smi.set_free(2048, 2048)
    time.sleep(0.3)
    t0 = time.monotonic()
    with gpu_query.display_reads():
        stale = _free_by_index(nvidia.get_visible_gpu_utilization([0, 1]))
    assert time.monotonic() - t0 < 0.3
    assert stale == first
    deadline = time.monotonic() + 5
    while gpu_query.stats()["inflight"] and time.monotonic() < deadline:
        time.sleep(0.05)
    with gpu_query.display_reads():
        refreshed = _free_by_index(nvidia.get_visible_gpu_utilization([0, 1]))
    assert refreshed == {0: 2.0, 1: 2.0}


def test_display_ttl_does_not_leak_into_fit_checks(smi, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GPU_QUERY_DISPLAY_TTL", "60")
    with gpu_query.display_reads():
        nvidia.get_visible_gpu_utilization([0, 1])
    smi.set_free(512, 512)
    fresh = _free_by_index(nvidia.get_visible_gpu_utilization([0, 1]))
    assert fresh == {0: 0.5, 1: 0.5}


def test_static_inventory_is_cached_until_redetection(smi):
    first = nvidia.get_physical_gpu_inventory()
    assert [d["name"] for d in first["devices"]] == ["NVIDIA B200", "NVIDIA B200"]
    assert nvidia.get_physical_gpu_count() == 2
    for _ in range(5):
        nvidia.get_physical_gpu_inventory()
        nvidia.get_physical_gpu_count()
        LlamaCppBackend._cuda_compute_caps()
        LlamaCppBackend._probe_nvlink_topology()
    assert smi.calls("memory.total") == 1
    assert smi.calls("-L") == 1
    assert smi.calls("compute_cap") == 1
    assert smi.calls("topo") == 5
    gpu_memory_events.invalidate_gpu_memory("load")
    nvidia.get_physical_gpu_inventory()
    assert smi.calls("memory.total") == 1
    gpu_query.invalidate_static("redetect")
    nvidia.get_physical_gpu_inventory()
    assert smi.calls("memory.total") == 2


def test_an_answered_empty_inventory_replaces_the_cached_one(smi, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GPU_QUERY_STATIC_TTL", "0")
    assert nvidia.get_physical_gpu_count() == 2
    smi.set(gpus = [])
    counts = []
    for _ in range(20):
        counts.append(nvidia.get_physical_gpu_count())
        if counts[-1] in (0, None):
            break
        time.sleep(0.1)
    assert counts[-1] in (0, None), counts


def test_a_static_read_after_redetection_does_not_join_an_older_child(smi):
    smi.set(delay = 0.6)
    first = threading.Thread(target = nvidia.get_physical_gpu_count)
    started = smi.calls("-L")
    first.start()
    smi.wait_for_call("-L", started + 1)
    smi.state["gpus"] = smi.state["gpus"][:1]  # a GPU went away; re-detection follows
    smi.save()
    gpu_query.invalidate_static("redetect")
    assert nvidia.get_physical_gpu_count() == 1
    first.join(5)
    assert smi.calls("-L") == 2


def test_display_after_a_load_reads_afresh_inside_the_ttl(smi, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GPU_QUERY_DISPLAY_TTL", "3600")
    with gpu_query.display_reads():
        nvidia.get_visible_gpu_utilization([0, 1])

    @gpu_memory_events.invalidates_gpu_memory("test load")
    def load_model():
        smi.set_free(1024, 1024)  # the model now occupies the cards

    load_model()
    with gpu_query.display_reads():
        after = _free_by_index(nvidia.get_visible_gpu_utilization([0, 1]))
    assert after == {0: 1.0, 1: 1.0}
    assert smi.calls("--query-gpu") == 2


def test_studio_load_unload_and_training_methods_invalidate():
    from core.inference import diffusion, inference, orchestrator, video
    from core.inference import native_audio, sd_cpp_backend
    from core.rag import embed_llama_server
    from core.inference import stt_registry
    from core.training import diffusion_training_service, training

    methods = [
        LlamaCppBackend.load_model,
        LlamaCppBackend.unload_model,
        LlamaCppBackend._kill_process,
        orchestrator.InferenceOrchestrator.load_model,
        orchestrator.InferenceOrchestrator.unload_model,
        training.TrainingBackend.start_training,
        training.TrainingBackend.stop_training,
        diffusion.DiffusionBackend.load_pipeline,
        diffusion.DiffusionBackend.unload,
        video.VideoBackend.load_pipeline,
        video.VideoBackend.unload,
        sd_cpp_backend.SdCppDiffusionBackend._run_load,
        sd_cpp_backend.SdCppDiffusionBackend.unload,
        embed_llama_server.LlamaServerBackend._spawn,
        embed_llama_server.LlamaServerBackend._kill_process,
        inference.InferenceBackend.load_model,
        inference.InferenceBackend.unload_model,
        native_audio.NativeAudioBackend.load_model,
        native_audio.NativeAudioBackend.unload_model,
        stt_registry.load,
        stt_registry.unload,
        diffusion_training_service.DiffusionTrainingService.start,
        diffusion_training_service.DiffusionTrainingService.stop,
    ]
    for method in methods:
        code = method.__code__
        assert (
            code is gpu_memory_events.invalidates_gpu_memory("x")(lambda: None).__code__
        ), method.__qualname__


def test_invalidation_happens_even_when_the_load_fails():
    before = gpu_memory_events.generation()

    @gpu_memory_events.invalidates_gpu_memory("failing load")
    def load():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        load()
    assert gpu_memory_events.generation() == before + 2


def test_a_child_started_before_an_invalidation_is_not_joined(smi, llama_probe):
    """A display read arriving after a load must not share a CLI call that began before it."""
    smi.set(delay = 0.5)
    first_result = []

    def display_probe():
        with gpu_query.display_reads():
            return llama_probe()

    t = threading.Thread(target = lambda: first_result.append(display_probe()))
    started = smi.calls("memory.free")
    t.start()
    smi.wait_for_call("memory.free", started + 1)
    smi.set_free(64, 64)
    gpu_memory_events.invalidate_gpu_memory("load")
    second = display_probe()
    t.join(5)
    assert first_result == [[(0, 180000, 183359), (1, 170000, 183359)]]
    assert second == [(0, 64, 183359), (1, 64, 183359)]
    assert smi.calls("memory.free") == 2


def test_fresh_reads_always_run_the_cli(smi):
    for _ in range(3):
        with gpu_query.fresh_reads():
            nvidia.get_visible_gpu_utilization([0, 1])
    assert smi.calls("--query-gpu") == 3


def test_a_fit_check_waits_its_whole_timeout_on_a_slow_driver(smi):
    smi.set(delay = 1.5)
    argv = ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]
    with pytest.raises(subprocess.TimeoutExpired):
        gpu_query.run_nvidia_smi(argv, timeout = 0.5, capture_output = True, text = True)
    assert gpu_query.driver_slow()
    out = gpu_query.run_nvidia_smi(argv, timeout = 5, capture_output = True, text = True)
    assert out.stdout.split() == ["0,", "180000", "1,", "170000"]


def test_a_slow_cli_holds_the_caller_only_for_its_timeout(smi, monkeypatch):
    monkeypatch.setattr(gpu_query, "_background_timeout", lambda: 3.0)
    smi.set(delay = 2.0)
    t0 = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        gpu_query.run_nvidia_smi(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            timeout = 0.3,
            capture_output = True,
            text = True,
        )
    assert time.monotonic() - t0 < 1.0
    assert gpu_query.driver_slow()


def test_a_child_the_kernel_cannot_reap_does_not_hold_the_caller(monkeypatch):
    gpu_query.reset()
    monkeypatch.delenv("UNSLOTH_GPU_QUERY_CACHE", raising = False)
    monkeypatch.setenv("UNSLOTH_NVIDIA_LIBRARY_PROBE", "0")
    calls = []

    def unreapable(argv, **kwargs):
        calls.append(argv)
        time.sleep(12.0)
        raise subprocess.TimeoutExpired(argv, kwargs.get("timeout"))

    monkeypatch.setattr(subprocess, "run", unreapable)
    for _ in range(2):
        t0 = time.monotonic()
        out = nvidia.get_visible_gpu_utilization([0, 1])
        assert out["available"] is False
        assert time.monotonic() - t0 < 7.0
    assert len(calls) == 2
    gpu_query.reset()


def test_slow_cli_answers_a_display_read_from_the_last_good_value(smi, monkeypatch):
    with gpu_query.display_reads():
        good = nvidia.get_visible_gpu_utilization([0, 1])
    monkeypatch.setenv("UNSLOTH_GPU_QUERY_DISPLAY_TTL", "0")
    monkeypatch.setattr(gpu_query, "_background_timeout", lambda: 4.0)
    smi.set(delay = 12.0)
    monkeypatch.setattr(gpu_query, "run_nvidia_smi", _with_timeout(gpu_query.run_nvidia_smi, 0.5))
    t0 = time.monotonic()
    with gpu_query.display_reads():
        out = nvidia.get_visible_gpu_utilization([0, 1])
    assert time.monotonic() - t0 < 2.0
    assert out == good


def test_slow_cli_never_shows_a_pre_load_reading_after_a_load(smi, monkeypatch):
    with gpu_query.display_reads():
        nvidia.get_visible_gpu_utilization([0, 1])
    monkeypatch.setenv("UNSLOTH_GPU_QUERY_DISPLAY_TTL", "0")
    gpu_memory_events.invalidate_gpu_memory("load")
    monkeypatch.setattr(gpu_query, "_background_timeout", lambda: 4.0)
    smi.set(delay = 12.0)
    monkeypatch.setattr(gpu_query, "run_nvidia_smi", _with_timeout(gpu_query.run_nvidia_smi, 0.5))
    t0 = time.monotonic()
    with gpu_query.display_reads():
        out = nvidia.get_visible_gpu_utilization([0, 1])
    assert time.monotonic() - t0 < 2.0
    assert out["available"] is False


def test_a_failed_answer_replaces_the_cached_inventory(smi, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GPU_QUERY_STATIC_TTL", "0")
    assert nvidia.get_physical_gpu_count() == 2
    smi.set(exit = 6)  # real nvidia-smi on a host with no GPU: "No devices were found", exit 6
    counts = []
    for _ in range(20):
        counts.append(nvidia.get_physical_gpu_count())
        if counts[-1] is None:
            break
        time.sleep(0.1)
    assert counts[-1] is None, counts


def test_an_expired_inventory_is_not_served_after_a_card_goes_away(smi, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GPU_QUERY_STATIC_TTL", "0.2")
    assert nvidia.get_physical_gpu_count() == 2
    time.sleep(0.5)
    smi.state["gpus"] = smi.state["gpus"][:1]  # an eGPU detached
    smi.save()
    assert nvidia.get_physical_gpu_count() == 1


def test_a_check_true_failure_replaces_the_cached_answer(smi, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GPU_QUERY_DISPLAY_TTL", "0")
    argv = ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"]

    def run():
        with gpu_query.display_reads():
            return gpu_query.run_nvidia_smi(
                argv, capture_output = True, text = True, timeout = 5, check = True
            )

    run()
    smi.set(exit = 6)
    for _ in range(20):
        try:
            run()
        except subprocess.CalledProcessError:
            break
        time.sleep(0.1)
    with pytest.raises(subprocess.CalledProcessError):
        run()


@pytest.mark.parametrize("code, failed", [(6, False), (9, True)])
def test_no_devices_found_is_an_answer_not_a_failure(smi, code, failed):
    assert nvidia.get_backend_visible_gpu_info([0, 1], "0,1")["available"]
    smi.set(exit = code)  # 6: "No devices were found"; 9: driver not loaded
    gpu_query.invalidate_static("redetect")
    out = nvidia.get_backend_visible_gpu_info([0, 1], "0,1")
    assert out["available"] is False
    assert bool(out.get("probe_failed")) is failed


def test_a_read_after_a_hung_child_does_not_wait_on_it(smi, monkeypatch):
    argv = ["nvidia-smi", "--query-gpu=index,compute_cap", "--format=csv,noheader"]

    def run(timeout):
        return gpu_query.run_nvidia_smi(argv, capture_output = True, text = True, timeout = timeout)

    monkeypatch.setattr(gpu_query, "_background_timeout", lambda: 10.0)
    smi.set(delay = 8.0)
    with pytest.raises(subprocess.TimeoutExpired):
        run(0.5)
    smi.set(delay = 0)  # the driver recovered; the first child is still running
    t0 = time.monotonic()
    assert run(0.5).stdout.split()[0] == "0,"
    assert time.monotonic() - t0 < 1.0


def test_display_reads_without_stale_wait_for_a_new_reading(smi, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GPU_QUERY_DISPLAY_TTL", "0.2")
    with gpu_query.display_reads(max_stale = 0):
        first = _free_by_index(nvidia.get_visible_gpu_utilization([0, 1]))
    time.sleep(0.4)
    smi.set_free(1000, 1000)  # another process allocated; no Studio event
    with gpu_query.display_reads(max_stale = 0):
        after = _free_by_index(nvidia.get_visible_gpu_utilization([0, 1]))
    assert after != first
    assert smi.calls("--query-gpu") == 2


def test_display_reads_without_stale_still_survive_a_hung_cli(smi, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GPU_QUERY_DISPLAY_TTL", "0")
    with gpu_query.display_reads(max_stale = 0):
        good = nvidia.get_visible_gpu_utilization([0, 1])
    monkeypatch.setattr(gpu_query, "_background_timeout", lambda: 4.0)
    smi.set(delay = 12.0)
    monkeypatch.setattr(gpu_query, "run_nvidia_smi", _with_timeout(gpu_query.run_nvidia_smi, 0.5))
    t0 = time.monotonic()
    with gpu_query.display_reads(max_stale = 0):
        out = nvidia.get_visible_gpu_utilization([0, 1])
    assert time.monotonic() - t0 < 2.0
    assert out == good


def test_an_older_answer_does_not_overwrite_a_newer_empty_one(smi, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GPU_QUERY_DISPLAY_TTL", "3600")
    argv = ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"]

    def run():
        return gpu_query.run_nvidia_smi(argv, capture_output = True, text = True, timeout = 5)

    smi.set(delay = 1.0)
    older = threading.Thread(target = run)  # fit checks never share a child
    started = smi.calls("--query-gpu")
    older.start()
    smi.wait_for_call("--query-gpu", started + 1)
    smi.set(delay = 0, gpus = [])
    assert run().stdout == ""
    older.join(5)
    with gpu_query.display_reads():
        assert run().stdout == ""


def test_slow_cli_never_answers_a_fit_check_from_an_old_reading(smi, llama_probe, monkeypatch):
    """Another process can allocate between readings, so no earlier sample may stand in."""
    monkeypatch.setattr(gpu_query, "_background_timeout", lambda: 4.0)
    assert llama_probe() == [(0, 180000, 183359), (1, 170000, 183359)]
    smi.set_free(1000, 1000)  # another process took nearly all of both GPUs
    smi.set(delay = 12.0)
    monkeypatch.setattr(gpu_query, "run_nvidia_smi", _with_timeout(gpu_query.run_nvidia_smi, 0.5))
    assert llama_probe() == []
    assert gpu_query.stats()["stale_served"] == 0


def test_a_fit_check_does_not_join_an_older_child(smi, monkeypatch):
    monkeypatch.setattr(gpu_query, "_background_timeout", lambda: 4.0)
    smi.set(delay = 3.0)
    argv = ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]
    kw = dict(timeout = 5, capture_output = True, text = True)
    first = threading.Thread(target = lambda: gpu_query.run_nvidia_smi(argv, **kw))
    started = smi.calls("--query-gpu=index,memory.free")
    first.start()
    smi.wait_for_call("--query-gpu=index,memory.free", started + 1)
    smi.set(delay = 0)
    smi.set_free(1000, 1000)
    out = gpu_query.run_nvidia_smi(argv, **kw)
    assert out.stdout.split() == ["0,", "1000", "1,", "1000"]
    first.join()
    assert smi.calls("--query-gpu=index,memory.free") == 2


def test_no_stale_fit_answer_across_a_load(smi, llama_probe, monkeypatch):
    monkeypatch.setattr(gpu_query, "_background_timeout", lambda: 4.0)
    llama_probe()
    gpu_memory_events.invalidate_gpu_memory("load")
    smi.set(delay = 12.0)
    monkeypatch.setattr(gpu_query, "run_nvidia_smi", _with_timeout(gpu_query.run_nvidia_smi, 0.5))
    assert llama_probe() == []


def test_a_hung_cli_leaves_llama_cpp_its_own_mig_aware_fallback(smi, llama_probe, monkeypatch):
    monkeypatch.setattr(gpu_query, "_background_timeout", lambda: 4.0)
    slice_rows = [(0, 5000, 10240)]
    monkeypatch.setattr(LlamaCppBackend, "_get_gpu_memory_nvml", staticmethod(lambda: slice_rows))
    monkeypatch.setenv("UNSLOTH_NVIDIA_LIBRARY_PROBE", "1")
    whole_gpu = [{"index": 0, "memory_total_mib": 81920, "memory_free_mib": 70000}]
    monkeypatch.setattr(gpu_query, "_nvml_rows", lambda timeout: whole_gpu, raising = False)
    smi.set(delay = 12.0)
    monkeypatch.setattr(gpu_query, "run_nvidia_smi", _with_timeout(gpu_query.run_nvidia_smi, 0.5))
    assert llama_probe() == slice_rows


def _with_timeout(fn, timeout):
    def wrapped(argv, **kwargs):
        kwargs["timeout"] = timeout
        return fn(argv, **kwargs)

    return wrapped


def test_failing_cli_is_not_cached(smi):
    smi.set(exit = 9)
    for _ in range(3):
        assert nvidia.get_visible_gpu_utilization([0, 1])["available"] is False
    assert smi.calls("--query-gpu") == 3
    smi.set(exit = 0)
    assert nvidia.get_visible_gpu_utilization([0, 1])["available"] is True


def test_missing_cli_is_reported_absent(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr(nvidia, "_nvidia_smi_executable", lambda: "nvidia-smi")
    gpu_query.reset()
    assert nvidia._query_gpu_inventory("test") is nvidia.NVIDIA_SMI_ABSENT
    assert nvidia.get_physical_gpu_count() is None


def test_cache_can_be_turned_off(smi, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GPU_QUERY_CACHE", "0")
    for _ in range(3):
        nvidia.get_visible_gpu_utilization([0, 1])
        nvidia.get_physical_gpu_inventory()
    assert smi.calls("--query-gpu") == 6


def test_results_are_copies(smi):
    argv = ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]
    a = gpu_query.run_nvidia_smi(argv, timeout = 5, capture_output = True, text = True)
    a.stdout = "mutated"
    b = gpu_query.run_nvidia_smi(argv, timeout = 5, capture_output = True, text = True)
    assert b.stdout.startswith("0, 180000")
