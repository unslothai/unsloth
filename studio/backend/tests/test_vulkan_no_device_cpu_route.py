# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An auto-selected Vulkan bundle on a host whose driver cannot answer (#142).

Two paths reach the staged CPU runtime without a crash: the Vulkan probe completed
and listed no device, or the probe could not enumerate (timed out or faulted) and the
first Vulkan child then did no loading work for the whole stall window. A stall after
a probe that listed a device fails with advice instead: it does not prove the GPU
unusable. Nothing is remembered across loads.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest

from .test_gpu_init_crash_message import (  # noqa: F401  (stubs loggers on import)
    _run_cpu_fallback_load,
)
from core.inference import llama_cpp
from core.inference.llama_cpp import LlamaCppBackend
from models.inference import LoadResponse


def _probe_dir(tmp_path):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    binary = bindir / "llama-server"
    binary.write_bytes(b"binary")
    # The name _lib_dir_has_ggml_backend looks for on this platform.
    (bindir / ("ggml-vulkan.dll" if sys.platform == "win32" else "libggml-vulkan.so")).write_bytes(
        b"vulkan"
    )
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


def _route_env(monkeypatch, outcome):
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
    # Only a suspect probe stops CPU work from renewing the stall window.
    assert sink["health_kwargs"][0]["cpu_counts_as_progress"] is (
        outcome not in ("timeout", "crashed")
    )
    if outcome not in ("timeout", "crashed"):
        # That stall only ever fails with advice: a slow disk gets three times the room.
        assert sink["health_kwargs"][0]["max_total_s"] >= (
            expected * LlamaCppBackend._LISTED_VULKAN_CEILING_FACTOR
        )


def test_a_lowered_stall_window_still_governs_a_suspect_launch(monkeypatch, tmp_path):
    _route_env(monkeypatch, "timeout")
    monkeypatch.setattr(LlamaCppBackend, "_HEALTH_STALL_TIMEOUT_S", 60.0)
    sink = {}
    _run_cpu_fallback_load(monkeypatch, tmp_path, returncodes = [None], sink = sink)
    assert sink["health_timeouts"][0] == 60.0


@pytest.mark.parametrize("outcome", ["timeout", "crashed"])
def test_a_hung_suspect_launch_replays_on_cpu(monkeypatch, tmp_path, outcome):
    _route_env(monkeypatch, outcome)
    sink = {}
    backend, loaded, launches, fallback_sources = _run_cpu_fallback_load(
        monkeypatch, tmp_path, returncodes = [None, None], stall_launches = (0,), sink = sink
    )
    assert loaded is True
    assert len(launches) == 2
    assert backend._cpu_fallback_reason == "vulkan_startup_stall"
    assert sink["health_timeouts"][0] == LlamaCppBackend._SUSPECT_VULKAN_STALL_TIMEOUT_S
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


@pytest.mark.parametrize("outcome", ["devices", "error"])
def test_a_stall_after_a_listed_device_fails_without_a_cpu_replay(monkeypatch, tmp_path, outcome):
    """The probe saw a working device, so a slow start is not proof of a broken driver:
    no CPU replay, no relaunch into the same hang, and the error points at Settings."""
    _route_env(monkeypatch, outcome)
    sink = {}
    with pytest.raises(
        RuntimeError, match = r"no loading progress\)\. .*Settings > GGUF inference engine"
    ):
        _run_cpu_fallback_load(
            monkeypatch,
            tmp_path,
            returncodes = [None, None],
            stall_launches = (0,),
            sink = sink,
        )
    assert [argv[0] for argv, _env in sink["launches"]] == ["/fake/llama-server"]
    assert sink["fallback_sources"] == []
    # A listed device counts CPU work (a first shader compile); the ceiling ends a spinning driver.
    assert sink["health_kwargs"][0]["cpu_counts_as_progress"] is True


def test_a_suspect_launch_cut_off_by_the_ceiling_fails_without_a_cpu_replay(monkeypatch, tmp_path):
    """Still loading when the ceiling hit: that is progress, not a dead driver."""
    _route_env(monkeypatch, "timeout")
    sink = {}
    with pytest.raises(
        RuntimeError, match = r"startup time limit reached\)\. .*Settings > GGUF inference engine"
    ):
        _run_cpu_fallback_load(
            monkeypatch,
            tmp_path,
            returncodes = [None, None],
            stall_launches = (0,),
            stall_kind = "ceiling",
            sink = sink,
        )
    assert len(sink["launches"]) == 1
    assert sink["fallback_sources"] == []


def test_a_recovered_stall_is_not_remembered(monkeypatch, tmp_path):
    """The next load re-probes and launches Vulkan again; one bad start never pins a
    build to CPU."""
    _route_env(monkeypatch, "timeout")
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    first, loaded, launches, _ = _run_cpu_fallback_load(
        monkeypatch, tmp_path / "a", returncodes = [None, None], stall_launches = (0,)
    )
    assert loaded is True
    assert first._cpu_fallback_reason == "vulkan_startup_stall"
    _route_env(monkeypatch, "devices")
    backend, loaded, launches, fallback_sources = _run_cpu_fallback_load(
        monkeypatch, tmp_path / "b", returncodes = [None]
    )
    assert loaded is True
    assert len(launches) == 1
    assert launches[0][0][0] == "/fake/llama-server"
    assert fallback_sources == []
    assert backend._cpu_fallback_reason is None


def test_a_stalled_cpu_replay_is_terminal(monkeypatch, tmp_path):
    _route_env(monkeypatch, "timeout")
    sink = {}
    with pytest.raises(RuntimeError):
        _run_cpu_fallback_load(
            monkeypatch,
            tmp_path,
            returncodes = [None] * 6,
            stall_launches = tuple(range(6)),
            sink = sink,
        )
    # One Vulkan launch, one CPU replay, nothing after.
    assert [argv[0] for argv, _env in sink["launches"]] == [
        "/fake/llama-server",
        "/staged/llama-server",
    ]
    assert len(sink["fallback_sources"]) == 1


def test_the_text_only_retry_is_bounded_too(monkeypatch, tmp_path):
    """The projector recovery starts its text-only child outside _spawn_and_wait."""
    _route_env(monkeypatch, "devices")
    sink = {}
    backend, loaded, launches, _ = _run_cpu_fallback_load(
        monkeypatch,
        tmp_path,
        returncodes = [1, None],
        first_output = "projector-incompatible",
        mmproj_from_argv = True,
        sink = sink,
    )
    assert loaded is True
    assert "--mmproj" not in launches[1][0]
    assert sink["health_kwargs"][1]["cpu_counts_as_progress"] is True
    assert "max_total_s" in sink["health_kwargs"][1]


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


class _ProbeClient:
    """Stands in for the one client a health wait probes through; each get goes to ``get``."""

    def __init__(self, get):
        self._get = get
        self.cookies = SimpleNamespace(clear = lambda: None)

    def get(self, url, **kw):
        return self._get(url, **kw)

    def close(self):
        pass


def _route_health_probes(monkeypatch, get):
    monkeypatch.setattr(llama_cpp, "_health_probe_client", lambda: _ProbeClient(get))


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
    healthy_at = None,
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
        if healthy_at is not None and clock.now >= healthy_at:
            return SimpleNamespace(status_code = 200)
        raise llama_cpp.httpx.ConnectError("refused")

    _route_health_probes(monkeypatch, _no_server)
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
    assert backend._health_wait_stall_kind == "ceiling"


@pytest.mark.parametrize("max_total_s", [None, 30.0, 120.0])
def test_a_child_doing_no_loading_work_reads_no_progress(monkeypatch, max_total_s):
    """Including a ceiling no longer than the window: nothing renewed, so it never cut
    anything off."""
    healthy, elapsed, backend = _wait(monkeypatch, cpu_counts = False, max_total_s = max_total_s)
    assert healthy is False
    assert backend._health_wait_stalled is True
    assert backend._health_wait_stall_kind == "no_progress"


def test_a_slow_but_loading_child_comes_up_on_its_device(monkeypatch):
    """Loading work renews the window well past it, so a slow healthy start is not a stall."""
    healthy, elapsed, backend = _wait(
        monkeypatch, cpu_counts = False, rss_growth = 1 << 20, max_total_s = 600.0, healthy_at = 200
    )
    assert healthy is True
    assert 200 <= elapsed < 210
    assert backend._health_wait_stalled is False
    assert backend._health_wait_stall_kind is None


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


def test_a_child_exiting_at_the_deadline_is_a_crash_not_a_stall(monkeypatch):
    """Exited during the last probe: the crash recovery decides, the stall branch must not."""
    clock = _Clock()
    monkeypatch.setattr(llama_cpp.time, "monotonic", clock.monotonic)
    polls = {"n": 0}

    class _Event:
        def clear(self):
            pass

        def wait(self, interval):
            clock.now += interval

    def _poll():
        # Alive for every in-loop check, gone by the deadline re-check.
        polls["n"] += 1
        return None if clock.now < 30.0 else -11

    _route_health_probes(
        monkeypatch,
        lambda *a, **k: (_ for _ in ()).throw(llama_cpp.httpx.ConnectError("refused")),
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_startup_work_sample", classmethod(lambda cls, p: (0.0, 100 << 20, 0, 0))
    )
    backend = LlamaCppBackend()
    backend._process = SimpleNamespace(poll = _poll, pid = 1, returncode = -11)
    backend._health_probe_event = _Event()
    backend._stdout_lines = []
    backend._cancel_event = None
    healthy = backend._wait_for_health(
        timeout = 30.0, interval = 30.0, cancelled = lambda: False, cpu_counts_as_progress = False
    )
    assert healthy is False
    assert backend._health_wait_stalled is False


@pytest.mark.parametrize("reason", ["vulkan_no_device", "vulkan_startup_stall"])
def test_a_reload_after_a_vulkan_cpu_route_leaves_the_decision_to_its_own_probe(
    monkeypatch, tmp_path, reason
):
    """No probe here: the load's single memoised probe decides, so a hung driver costs one
    timeout and a device that appeared since is seen."""
    from core.inference.llama_cpp import GgufLoadIntent

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")
    backend = LlamaCppBackend()
    backend._process = SimpleNamespace(poll = lambda: None)
    backend._healthy = True
    backend._model_identifier = "owner/model"
    backend._gguf_path = str(gguf)
    backend._cpu_fallback_reason = reason
    probes = []
    monkeypatch.setattr(LlamaCppBackend, "matches_load_source", lambda self, intent: True)
    monkeypatch.setattr(
        LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: "/fake/llama-server")
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "_run_vulkan_probe",
        staticmethod(lambda binary: probes.append(binary) or []),
    )
    monkeypatch.setattr(
        LlamaCppBackend,
        "_vulkan_probe_outcome",
        staticmethod(lambda _binary = None: probes.append(_binary) or "none"),
    )
    intent = GgufLoadIntent(
        model_identifier = "owner/model", gguf_path = str(gguf), n_ctx = 4096, gpu_memory_mode = "auto"
    )
    assert backend._preserve_cpu_fallback_intent(intent).cpu_fallback is False
    assert probes == []
    # A crash recovery is still sticky for the same source.
    backend._cpu_fallback_reason = "vulkan_startup_crash"
    assert backend._preserve_cpu_fallback_intent(intent).cpu_fallback is True


def test_a_stalled_text_only_retry_after_a_timed_out_probe_replays_on_cpu(monkeypatch, tmp_path):
    """The stall rule follows every bounded retry, not only the first launch."""
    _route_env(monkeypatch, "timeout")
    sink = {}
    backend, loaded, launches, fallback_sources = _run_cpu_fallback_load(
        monkeypatch,
        tmp_path,
        returncodes = [1, None, None],
        first_output = "projector-incompatible",
        mmproj_from_argv = True,
        stall_launches = (1,),
        sink = sink,
    )
    assert loaded is True
    assert launches[-1][0][0] == "/staged/llama-server"
    assert len(fallback_sources) == 1
    assert backend._cpu_fallback_reason == "vulkan_startup_stall"


def test_a_stalled_text_only_retry_after_a_listed_device_fails_with_advice(monkeypatch, tmp_path):
    _route_env(monkeypatch, "devices")
    with pytest.raises(RuntimeError, match = "Settings > GGUF inference engine"):
        _run_cpu_fallback_load(
            monkeypatch,
            tmp_path,
            returncodes = [1, None, None],
            first_output = "projector-incompatible",
            mmproj_from_argv = True,
            stall_launches = (1,),
        )


@pytest.mark.parametrize(
    "outcome, cpu_route",
    [("none", True), ("timeout", False), ("crashed", False), ("error", False), ("devices", False)],
)
def test_a_reload_after_a_stall_is_routed_by_the_loads_one_probe(
    monkeypatch, tmp_path, outcome, cpu_route
):
    """Same-source reload after a stall replay: one probe, and only a conclusive none goes to CPU."""
    _route_env(monkeypatch, outcome)
    calls = []
    real = LlamaCppBackend._vulkan_probe_outcome
    monkeypatch.setattr(
        LlamaCppBackend,
        "_vulkan_probe_outcome",
        staticmethod(lambda _binary = None: calls.append(_binary) or real(_binary)),
    )
    monkeypatch.setattr(LlamaCppBackend, "matches_load_source", lambda self, intent: True)
    real_init = LlamaCppBackend.__init__

    def _init_after_a_stall(self, *a, **k):
        real_init(self, *a, **k)
        self._cpu_fallback_reason = "vulkan_startup_stall"

    monkeypatch.setattr(LlamaCppBackend, "__init__", _init_after_a_stall)
    backend, loaded, launches, fallback_sources = _run_cpu_fallback_load(
        monkeypatch, tmp_path, returncodes = [None]
    )
    assert loaded is True
    assert len(calls) == 1
    assert (launches[0][0][0] == "/staged/llama-server") is cpu_route
    assert (len(fallback_sources) == 1) is cpu_route


def test_one_vulkan_probe_per_load(monkeypatch, tmp_path):
    """Placement and the launch's outcome read share one answer (a hung probe costs one timeout)."""
    probes = []

    def _probe(binary):
        probes.append(binary)
        llama_cpp._load_probe_memo_put("vulkan_outcome", binary, "timeout")
        llama_cpp._load_probe_memo_put("vulkan", binary, [])
        return []

    real_outcome = LlamaCppBackend._vulkan_probe_outcome
    monkeypatch.setattr(LlamaCppBackend, "_binary_revision", staticmethod(lambda _b: ("rev", 1)))
    monkeypatch.setattr(
        LlamaCppBackend, "_cpu_isolated_binary", lambda self, _binary: "/staged/llama-server"
    )

    def _memo_probe(binary):
        memo = llama_cpp._load_probe_memo_get("vulkan", binary)
        return memo if memo is not None else _probe(binary)

    monkeypatch.setattr(LlamaCppBackend, "_run_vulkan_probe", staticmethod(_memo_probe))
    monkeypatch.setattr(LlamaCppBackend, "_vulkan_probe_outcome", staticmethod(real_outcome))
    sink = {}

    def _placement_probe(
        self,
        binary = None,
        **_kw,
    ):
        LlamaCppBackend._run_vulkan_probe(binary or "/fake/llama-server")
        return []

    with llama_cpp._load_probe_memo_scope():
        backend, loaded, launches, _ = _run_cpu_fallback_load(
            monkeypatch,
            tmp_path,
            returncodes = [None],
            sink = sink,
            gpu_memory = _placement_probe,
        )
    assert loaded is True
    assert probes == ["/fake/llama-server"]


def test_a_stall_replay_with_the_vision_argv_clears_the_text_only_diagnosis(monkeypatch, tmp_path):
    """A guessed projector fault, then a stalled text-only retry after a timed-out probe: the CPU
    replay runs with the original projector, so vision is back and the text-only reason must go."""
    _route_env(monkeypatch, "timeout")
    sink = {}
    backend, loaded, launches, fallback_sources = _run_cpu_fallback_load(
        monkeypatch,
        tmp_path,
        returncodes = [-11, None, None],
        mmproj_from_argv = True,
        stall_launches = (1,),
        sink = sink,
    )
    assert loaded is True
    assert "--mmproj" in fallback_sources[0]
    assert backend._cpu_fallback_reason == "vulkan_startup_stall"
    assert backend._mmproj_fallback_reason is None
