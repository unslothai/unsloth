# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Cached, coalesced nvidia-smi reads (utils/hardware/gpu_query.py).

Driven by a real executable named ``nvidia-smi`` on PATH whose answers, latency and exit
status each test controls, so the child-process, timeout and parsing paths all run for real.
The properties pinned here:

* concurrent identical static or display queries start one child;
* static and display reads keep their own TTL; decision-critical reads have none;
* a Studio load/unload/train invalidation forces the next display read to fetch afresh;
* a slow CLI holds a caller for its own timeout, not for the child's lifetime;
* a fit check never gets a cached or shared reading, only its own new one;
* failures and a missing CLI are passed through uncached, as before.
"""

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
    """LlamaCppBackend._get_gpu_memory with only the nvidia-smi branch reachable."""
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


# ── Classification ────────────────────────────────────────────────────────────


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
    # Anything unrecognised is treated as the strictest category.
    assert gpu_query.classify(["nvidia-smi", "-q"]) == gpu_query.CRITICAL


# ── Coalescing ────────────────────────────────────────────────────────────────


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
    """A fit check never shares a reading: each one's sample is taken after it began."""
    smi.set(delay = 0.3)
    threads = [threading.Thread(target = llama_probe) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert smi.calls("memory.free") == 4


def test_concurrent_fit_checks_never_answer_from_before_they_started(smi, llama_probe):
    """Memory keeps changing under 16 threads of fit checks; each answer must be a reading
    taken after its own call began (free memory here only grows, so it is at least the value
    current at the call)."""
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


# ── TTL per category ──────────────────────────────────────────────────────────


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
    # Answered from the cache immediately, not after the 0.5 s child.
    assert time.monotonic() - t0 < 0.3
    assert stale == first
    deadline = time.monotonic() + 5
    while gpu_query.stats()["inflight"] and time.monotonic() < deadline:
        time.sleep(0.05)
    with gpu_query.display_reads():
        refreshed = _free_by_index(nvidia.get_visible_gpu_utilization([0, 1]))
    assert refreshed == {0: 2.0, 1: 2.0}


def test_display_ttl_does_not_leak_into_fit_checks(smi, monkeypatch):
    """A cached display reading must never answer a critical read."""
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
    # The topology keeps its own process-lifetime cache with an explicit refresh, so the
    # helper does not cache it a second time (only the bounded wait applies).
    assert smi.calls("topo") == 5
    # A model load does not touch the static inventory...
    gpu_memory_events.invalidate_gpu_memory("load")
    nvidia.get_physical_gpu_inventory()
    assert smi.calls("memory.total") == 1
    # ...a hardware re-detection does.
    gpu_query.invalidate_static("redetect")
    nvidia.get_physical_gpu_inventory()
    assert smi.calls("memory.total") == 2


def test_a_static_read_after_redetection_does_not_join_an_older_child(smi):
    smi.set(delay = 0.6)
    first = threading.Thread(target = nvidia.get_physical_gpu_count)
    first.start()
    time.sleep(0.2)
    smi.state["gpus"] = smi.state["gpus"][:1]  # a GPU went away; re-detection follows
    smi.save()
    gpu_query.invalidate_static("redetect")
    assert nvidia.get_physical_gpu_count() == 1
    first.join(5)
    assert smi.calls("-L") == 2


# ── Invalidation on Studio's own load / unload / training ─────────────────────


def test_display_after_a_load_reads_afresh_inside_the_ttl(smi, monkeypatch):
    monkeypatch.setenv("UNSLOTH_GPU_QUERY_DISPLAY_TTL", "3600")
    with gpu_query.display_reads():
        nvidia.get_visible_gpu_utilization([0, 1])

    @gpu_memory_events.invalidates_gpu_memory("test load")
    def load_model():
        smi.set_free(1024, 1024)  # the model now occupies the cards

    load_model()
    # Well inside the TTL, but the load invalidated it: the panel shows the new low at once.
    with gpu_query.display_reads():
        after = _free_by_index(nvidia.get_visible_gpu_utilization([0, 1]))
    assert after == {0: 1.0, 1: 1.0}
    assert smi.calls("--query-gpu") == 2


def test_studio_load_unload_and_training_methods_invalidate():
    """The real entry points carry the invalidation decorator (outermost)."""
    from core.inference import diffusion, inference, orchestrator, video
    from core.inference import native_audio, sd_cpp_backend
    from core.rag import embed_llama_server
    from core.training import training

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
        sd_cpp_backend.SdCppDiffusionBackend.unload,
        embed_llama_server.LlamaServerBackend._spawn,
        embed_llama_server.LlamaServerBackend._kill_process,
        inference.InferenceBackend.load_model,
        inference.InferenceBackend.unload_model,
        native_audio.NativeAudioBackend.load_model,
        native_audio.NativeAudioBackend.unload_model,
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
    t.start()
    time.sleep(0.15)
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
    """A driver that answers slowly must still be heard, as before, even after a timeout."""
    smi.set(delay = 1.5)
    argv = ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]
    with pytest.raises(subprocess.TimeoutExpired):
        gpu_query.run_nvidia_smi(argv, timeout = 0.5, capture_output = True, text = True)
    assert gpu_query.driver_slow()
    out = gpu_query.run_nvidia_smi(argv, timeout = 5, capture_output = True, text = True)
    assert out.stdout.split() == ["0,", "180000", "1,", "170000"]


# ── Slow / hung nvidia-smi ────────────────────────────────────────────────────


def test_a_slow_cli_holds_the_caller_only_for_its_timeout(smi, monkeypatch):
    """subprocess.run(timeout=5) waits for the killed child without a deadline; here the
    caller gets its answer (or its TimeoutExpired) on time."""
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
    """What the congested 8x B200 host did: subprocess.run(timeout=5) kills nvidia-smi and
    then blocks in wait() until the driver lets the process die, 25-100 s later. Emulated
    with a runner that only returns (raising TimeoutExpired, as run() does) after 12 s."""
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
        # The caller's own "unavailable" answer, as before, after its 5 s timeout (plus the
        # 1 s reaping grace) rather than after the 12 s the stuck child takes to be reaped.
        assert out["available"] is False
        assert time.monotonic() - t0 < 7.0
    # A fit check never shares a child, so each ran its own, as before.
    assert len(calls) == 2
    gpu_query.reset()


def test_slow_cli_answers_a_display_read_from_the_last_good_value(smi, monkeypatch):
    with gpu_query.display_reads():
        good = nvidia.get_visible_gpu_utilization([0, 1])
    monkeypatch.setenv("UNSLOTH_GPU_QUERY_DISPLAY_TTL", "0")
    gpu_memory_events.invalidate_gpu_memory("x")  # force past the SWR fast path
    monkeypatch.setattr(gpu_query, "_background_timeout", lambda: 4.0)
    smi.set(delay = 12.0)
    monkeypatch.setattr(gpu_query, "run_nvidia_smi", _with_timeout(gpu_query.run_nvidia_smi, 0.5))
    t0 = time.monotonic()
    with gpu_query.display_reads():
        out = nvidia.get_visible_gpu_utilization([0, 1])
    assert time.monotonic() - t0 < 2.0
    assert out == good


def test_slow_cli_never_answers_a_fit_check_from_an_old_reading(smi, llama_probe, monkeypatch):
    """Another process can allocate between readings, so no earlier sample may stand in."""
    monkeypatch.setattr(gpu_query, "_background_timeout", lambda: 4.0)
    assert llama_probe() == [(0, 180000, 183359), (1, 170000, 183359)]
    smi.set_free(1000, 1000)  # another process took nearly all of both GPUs
    smi.set(delay = 12.0)
    monkeypatch.setattr(gpu_query, "run_nvidia_smi", _with_timeout(gpu_query.run_nvidia_smi, 0.5))
    # Nothing new to answer with: the caller's own fallback chain (here: none) decides, as before.
    assert llama_probe() == []
    assert gpu_query.stats()["stale_served"] == 0


def test_a_fit_check_does_not_join_an_older_child(smi, monkeypatch):
    """On a slow driver an in-flight child may describe memory from seconds ago."""
    monkeypatch.setattr(gpu_query, "_background_timeout", lambda: 4.0)
    smi.set(delay = 3.0)
    argv = ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"]
    kw = dict(timeout = 5, capture_output = True, text = True)
    first = threading.Thread(target = lambda: gpu_query.run_nvidia_smi(argv, **kw))
    first.start()
    time.sleep(0.5)
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
    # Nothing safe to serve: the caller's own fallback chain (here: none) decides.
    assert llama_probe() == []


def test_a_hung_cli_leaves_llama_cpp_its_own_mig_aware_fallback(smi, llama_probe, monkeypatch):
    """The CLI timing out must reach llama.cpp's NVML branch, which knows MIG slices and
    CUDA_VISIBLE_DEVICES, exactly as before; nothing may answer in its place."""
    monkeypatch.setattr(gpu_query, "_background_timeout", lambda: 4.0)
    slice_rows = [(0, 5000, 10240)]
    monkeypatch.setattr(LlamaCppBackend, "_get_gpu_memory_nvml", staticmethod(lambda: slice_rows))
    # A generic whole-GPU NVML answer (what a stand-in in the helper would give) must not win.
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


# ── Failures pass through exactly as before ───────────────────────────────────


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
