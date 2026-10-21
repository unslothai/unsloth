# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An auto-selected Vulkan bundle on a host whose driver cannot answer (#142).

Two paths reach the staged CPU runtime without a crash: the Vulkan probe saw no
usable device before launch, or the first Vulkan child stalled in startup.
"""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from .test_gpu_init_crash_message import (  # noqa: F401  (stubs loggers on import)
    _run_cpu_fallback_load,
)
from core.inference import llama_cpp
from core.inference.llama_cpp import LlamaCppBackend
from models.inference import LoadResponse


# ── probe outcome classification ──────────────────────────────────────────


def _probe_dir(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    binary = bindir / "llama-server"
    binary.write_bytes(b"binary")
    (bindir / "libggml-vulkan.so").write_bytes(b"vulkan")
    return str(binary)


def _outcome(monkeypatch, tmp_path, run):
    binary = _probe_dir(tmp_path)
    monkeypatch.setattr(llama_cpp.subprocess, "run", run)
    with llama_cpp._load_probe_memo_scope():
        llama_cpp._arm_load_probe_memo()
        rows = LlamaCppBackend._run_vulkan_probe(binary)
        outcome = LlamaCppBackend._vulkan_probe_outcome(binary)
    return rows, outcome


def _completed(returncode, stdout = ""):
    return lambda *a, **k: SimpleNamespace(returncode = returncode, stdout = stdout, stderr = "")


def test_zero_enumerated_devices_reads_none(monkeypatch, tmp_path):
    assert _outcome(monkeypatch, tmp_path, _completed(0, "")) == ([], "none")


def test_listed_device_reads_devices(monkeypatch, tmp_path):
    line = "0\t1073741824\t1\t2147483648\tIntel(R) HD Graphics 530\t1"
    rows, outcome = _outcome(monkeypatch, tmp_path, _completed(0, line))
    assert outcome == "devices"
    assert rows[0]["name"] == "Intel(R) HD Graphics 530"


def test_timeout_reads_timeout(monkeypatch, tmp_path):
    def _hang(*a, **k):
        raise subprocess.TimeoutExpired(cmd = "probe", timeout = 15)

    assert _outcome(monkeypatch, tmp_path, _hang) == ([], "timeout")


@pytest.mark.parametrize("rc", [-11, 0xC0000005])
def test_driver_fault_reads_crashed(monkeypatch, tmp_path, rc):
    assert _outcome(monkeypatch, tmp_path, _completed(rc)) == ([], "crashed")


def test_ordinary_failure_reads_error(monkeypatch, tmp_path):
    assert _outcome(monkeypatch, tmp_path, _completed(1)) == ([], "error")


def test_unparseable_output_is_not_zero_devices(monkeypatch, tmp_path):
    assert _outcome(monkeypatch, tmp_path, _completed(0, "garbage")) == ([], "error")


def test_outcome_outside_a_load_never_claims_no_device(monkeypatch, tmp_path):
    """Unarmed memo: nothing recorded, so nothing acts on it."""
    binary = _probe_dir(tmp_path)
    monkeypatch.setattr(llama_cpp.subprocess, "run", _completed(0, ""))
    assert LlamaCppBackend._run_vulkan_probe(binary) == []
    assert LlamaCppBackend._vulkan_probe_outcome(binary) == "error"


# ── pre-launch route ──────────────────────────────────────────────────────


def _route_env(monkeypatch, outcome):
    # Class-level session memo: no test may see another's stall.
    monkeypatch.setattr(LlamaCppBackend, "_auto_vulkan_cpu_routes", {})
    monkeypatch.setattr(
        LlamaCppBackend, "_binary_revision", staticmethod(lambda _binary: ("rev", 1))
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_vulkan_probe_outcome", staticmethod(lambda _binary = None: outcome)
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_cpu_isolated_binary", lambda self, _binary: "/staged/llama-server"
    )


def test_no_usable_device_starts_on_the_cpu_runtime(monkeypatch, tmp_path):
    _route_env(monkeypatch, "none")
    sink = {}
    backend, loaded, launches, fallback_sources = _run_cpu_fallback_load(
        monkeypatch, tmp_path, returncodes = [None], sink = sink
    )
    assert loaded is True
    # One launch, and it is the staged CPU runtime: the Vulkan child never starts.
    assert len(launches) == 1
    assert launches[0][0][0] == "/staged/llama-server"
    assert len(fallback_sources) == 1
    assert backend._cpu_fallback_reason == "vulkan_no_device"
    assert backend._gpu_memory_mode == "manual" and backend._gpu_layers == 0
    # The CPU child keeps the ordinary progress rule.
    assert sink["health_kwargs"][0].get("cpu_counts_as_progress", True) is True


@pytest.mark.parametrize("outcome", ["devices", "error", "timeout", "crashed"])
def test_a_listed_device_or_unproven_probe_keeps_vulkan(monkeypatch, tmp_path, outcome):
    """A timed-out probe is not proof: a healthy discrete card can be that slow, and a
    driver can fault on exit after listing a working device."""
    _route_env(monkeypatch, outcome)
    sink = {}
    backend, loaded, launches, fallback_sources = _run_cpu_fallback_load(
        monkeypatch, tmp_path, returncodes = [None], sink = sink
    )
    assert loaded is True
    assert launches[0][0][0] == "/fake/llama-server"
    assert fallback_sources == []
    assert backend._cpu_fallback_reason is None
    # Suspect only after a timeout or a fault: the shorter stall window.
    expected = (
        min(
            LlamaCppBackend._HEALTH_STALL_TIMEOUT_S,
            LlamaCppBackend._SUSPECT_VULKAN_STALL_TIMEOUT_S,
        )
        if outcome in ("timeout", "crashed")
        else LlamaCppBackend._HEALTH_STALL_TIMEOUT_S
    )
    assert sink["health_timeouts"][0] == expected
    # Every Vulkan launch of an eligible load is capped, renewals included.
    assert sink["health_kwargs"][0]["max_total_s"] >= expected


def test_a_lowered_stall_window_still_governs_a_suspect_launch(monkeypatch, tmp_path):
    _route_env(monkeypatch, "timeout")
    monkeypatch.setattr(LlamaCppBackend, "_HEALTH_STALL_TIMEOUT_S", 60.0)
    sink = {}
    _run_cpu_fallback_load(monkeypatch, tmp_path, returncodes = [None], sink = sink)
    assert sink["health_timeouts"][0] == 60.0


def test_a_hung_suspect_launch_replays_on_cpu(monkeypatch, tmp_path):
    _route_env(monkeypatch, "timeout")
    sink = {}
    backend, loaded, launches, fallback_sources = _run_cpu_fallback_load(
        monkeypatch, tmp_path, returncodes = [None, None], stall_launches = (0,), sink = sink
    )
    assert loaded is True
    assert len(launches) == 2
    assert backend._cpu_fallback_reason == "vulkan_startup_stall"
    assert sink["health_timeouts"][0] == LlamaCppBackend._SUSPECT_VULKAN_STALL_TIMEOUT_S
    # The CPU replay gets the ordinary window.
    assert sink["health_timeouts"][1] == LlamaCppBackend._HEALTH_STALL_TIMEOUT_S


def test_a_user_chosen_vulkan_install_is_not_routed(monkeypatch, tmp_path):
    """A Vulkan bundle picked in Settings is not auto-selected, so not eligible.

    The helper forces auto-selection True, so the eligibility gate it feeds is
    what this patches (the marker reading itself is covered elsewhere)."""
    _route_env(monkeypatch, "none")
    monkeypatch.setattr(
        LlamaCppBackend,
        "_auto_vulkan_cpu_fallback_eligible",
        classmethod(lambda cls, *a, **k: False),
    )
    backend, loaded, launches, fallback_sources = _run_cpu_fallback_load(
        monkeypatch, tmp_path, returncodes = [None]
    )
    assert loaded is True
    assert launches[0][0][0] == "/fake/llama-server"
    assert fallback_sources == []
    assert backend._cpu_fallback_reason is None


def test_explicit_device_placement_is_not_routed(monkeypatch, tmp_path):
    _route_env(monkeypatch, "none")
    backend, loaded, launches, fallback_sources = _run_cpu_fallback_load(
        monkeypatch, tmp_path, returncodes = [None], extra_args = ["--device", "Vulkan0"]
    )
    assert loaded is True
    assert launches[0][0][0] == "/fake/llama-server"
    assert fallback_sources == []
    assert backend._cpu_fallback_reason is None


def test_no_cpu_library_to_stage_keeps_vulkan(monkeypatch, tmp_path):
    monkeypatch.setattr(
        LlamaCppBackend, "_vulkan_probe_outcome", staticmethod(lambda _binary = None: "none")
    )
    monkeypatch.setattr(LlamaCppBackend, "_cpu_isolated_binary", lambda self, _binary: None)
    backend, loaded, launches, fallback_sources = _run_cpu_fallback_load(
        monkeypatch, tmp_path, returncodes = [None]
    )
    assert loaded is True
    assert launches[0][0][0] == "/fake/llama-server"
    assert backend._cpu_fallback_reason is None


def test_a_reload_keeps_the_no_device_reason(monkeypatch, tmp_path):
    _route_env(monkeypatch, "devices")
    backend, loaded, launches, _ = _run_cpu_fallback_load(
        monkeypatch,
        tmp_path,
        returncodes = [None],
        cpu_fallback = True,
        resident_fallback = "vulkan_no_device",
    )
    assert loaded is True
    assert launches[0][0][0] == "/staged/llama-server"
    assert backend._cpu_fallback_reason == "vulkan_no_device"


# ── stalled first launch ──────────────────────────────────────────────────


def test_a_stalled_vulkan_child_replays_once_on_cpu(monkeypatch, tmp_path):
    _route_env(monkeypatch, "devices")
    sink = {}
    backend, loaded, launches, fallback_sources = _run_cpu_fallback_load(
        monkeypatch, tmp_path, returncodes = [None, None], stall_launches = (0,), sink = sink
    )
    assert loaded is True
    assert len(launches) == 2
    assert len(fallback_sources) == 1
    assert fallback_sources[0][0] == "/fake/llama-server"
    assert backend._cpu_fallback_reason == "vulkan_startup_stall"
    # Only the Vulkan child ignores CPU-only activity; the CPU replay counts it.
    assert sink["health_kwargs"][0]["cpu_counts_as_progress"] is False
    assert sink["health_kwargs"][1].get("cpu_counts_as_progress", True) is True


def test_a_stall_sends_later_loads_of_the_same_build_straight_to_cpu(monkeypatch, tmp_path):
    _route_env(monkeypatch, "devices")
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _run_cpu_fallback_load(
        monkeypatch, tmp_path / "a", returncodes = [None, None], stall_launches = (0,)
    )
    backend, loaded, launches, _ = _run_cpu_fallback_load(
        monkeypatch, tmp_path / "b", returncodes = [None]
    )
    assert loaded is True
    assert len(launches) == 1
    assert launches[0][0][0] == "/staged/llama-server"
    assert backend._cpu_fallback_reason == "vulkan_startup_stall"


def test_a_failed_stall_replay_leaves_only_bounded_vulkan_rungs(monkeypatch, tmp_path):
    """The CPU replay is not terminal: a rung below (e.g. the speculative-off retry) may
    still bring the GPU up, but no Vulkan relaunch may renew its wait on CPU again."""
    _route_env(monkeypatch, "devices")
    sink = {}
    with pytest.raises(RuntimeError):
        _run_cpu_fallback_load(
            monkeypatch,
            tmp_path,
            returncodes = [None] * 6,
            stall_launches = tuple(range(6)),
            sink = sink,
        )
    assert len(sink["fallback_sources"]) == 1
    for (argv, _env), kwargs in zip(sink["launches"], sink["health_kwargs"]):
        if argv[0] != "/staged/llama-server":
            assert kwargs["cpu_counts_as_progress"] is False
            assert "max_total_s" in kwargs


def test_a_stall_on_a_non_vulkan_runtime_keeps_cpu_progress(monkeypatch, tmp_path):
    _route_env(monkeypatch, "devices")
    sink = {}
    with pytest.raises(RuntimeError):
        _run_cpu_fallback_load(
            monkeypatch,
            tmp_path,
            # Whatever rungs a stall reaches on another backend respawn and fail.
            returncodes = [None] + [1] * 8,
            stall_launches = (0,),
            vulkan = False,
            sink = sink,
        )
    assert sink["fallback_sources"] == []
    assert sink["health_kwargs"][0].get("cpu_counts_as_progress", True) is True


def test_a_stall_with_explicit_placement_does_not_replay(monkeypatch, tmp_path):
    _route_env(monkeypatch, "devices")
    sink = {}
    with pytest.raises(RuntimeError):
        _run_cpu_fallback_load(
            monkeypatch,
            tmp_path,
            returncodes = [None],
            stall_launches = (0,),
            extra_args = ["--device", "Vulkan0"],
            sink = sink,
        )
    assert sink["fallback_sources"] == []
    assert sink["health_kwargs"][0].get("cpu_counts_as_progress", True) is True


def test_a_cancel_during_the_stall_never_spawns_the_replay(monkeypatch, tmp_path):
    _route_env(monkeypatch, "devices")
    sink = {}
    with pytest.raises(RuntimeError):
        _run_cpu_fallback_load(
            monkeypatch,
            tmp_path,
            returncodes = [None, None],
            stall_launches = (0,),
            cancel_after = 1,
            sink = sink,
        )
    assert len(sink["launches"]) == 1
    assert sink["fallback_sources"] == []


# ── health wait: CPU-only activity ────────────────────────────────────────


class _Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now


def _wait(
    monkeypatch,
    *,
    cpu_counts,
    rss_growth = 0,
    max_total_s = None,
):
    """A live child that burns CPU (and optionally allocates); returns whether the
    wait gave up."""
    clock = _Clock()
    monkeypatch.setattr(llama_cpp.time, "monotonic", clock.monotonic)

    class _Event:
        def clear(self):
            pass

        def wait(self, interval):
            clock.now += interval

    def _no_server(*a, **k):
        raise llama_cpp.httpx.ConnectError("refused")

    monkeypatch.setattr(llama_cpp.httpx, "get", _no_server)
    cpu = {"t": 0.0}

    def _sample(cls, _process):
        # Half a core of CPU per second, RSS growing by ``rss_growth`` per second.
        cpu["t"] = clock.now * 0.5
        return (cpu["t"], (100 << 20) + int(clock.now * rss_growth), 0, 0)

    monkeypatch.setattr(LlamaCppBackend, "_startup_work_sample", classmethod(_sample))
    backend = LlamaCppBackend()
    backend._process = SimpleNamespace(poll = lambda: None, pid = 1)
    backend._health_probe_event = _Event()
    backend._stdout_lines = []
    backend._cancel_event = None
    healthy = backend._wait_for_health(
        timeout = 30.0,
        interval = 1.0,
        cancelled = lambda: clock.now > 600,
        cpu_counts_as_progress = cpu_counts,
        **({"max_total_s": max_total_s} if max_total_s is not None else {}),
    )
    return healthy, clock.now, backend


def test_cpu_only_activity_does_not_renew_a_flagged_wait(monkeypatch):
    healthy, elapsed, backend = _wait(monkeypatch, cpu_counts = False)
    assert healthy is False
    assert backend._health_wait_stalled is True
    assert elapsed < 60


def test_cpu_only_activity_still_renews_the_default_wait(monkeypatch):
    healthy, elapsed, backend = _wait(monkeypatch, cpu_counts = True)
    # Renewed until the cancel predicate ends it, never a stall.
    assert healthy is False
    assert backend._health_wait_stalled is False
    assert elapsed > 600


def test_a_driver_that_keeps_allocating_still_hits_the_ceiling(monkeypatch):
    """Shader compilation touches fresh pages, so RSS alone renews the stall window;
    only the ceiling ends it."""
    healthy, elapsed, backend = _wait(
        monkeypatch, cpu_counts = False, rss_growth = 1 << 20, max_total_s = 120.0
    )
    assert healthy is False
    assert backend._health_wait_stalled is True
    assert 120 <= elapsed < 130
    assert "not healthy after 120s" in backend._stdout_lines[-1]


def test_an_allocating_child_without_a_ceiling_keeps_renewing(monkeypatch):
    healthy, elapsed, backend = _wait(monkeypatch, cpu_counts = False, rss_growth = 1 << 20)
    assert backend._health_wait_stalled is False
    assert elapsed > 600


def test_loading_progress_renews_a_flagged_wait():
    before = (0.0, 100 << 20, 0, 0)
    assert LlamaCppBackend._made_startup_progress(
        before, (0.0, 200 << 20, 0, 0), 5.0, cpu_counts = False
    )
    assert not LlamaCppBackend._made_startup_progress(
        before, (5.0, 100 << 20, 0, 0), 5.0, cpu_counts = False
    )
    assert LlamaCppBackend._made_startup_progress(before, (5.0, 100 << 20, 0, 0), 5.0)


@pytest.mark.parametrize("reason", ["vulkan_no_device", "vulkan_startup_stall"])
def test_response_accepts_the_new_reasons(reason):
    response = LoadResponse(
        status = "loaded",
        model = "m",
        display_name = "m",
        inference = {},
        cpu_fallback_reason = reason,
    )
    assert response.model_dump()["cpu_fallback_reason"] == reason
