# SPDX-License-Identifier: AGPL-3.0-only
import builtins
import concurrent.futures
import subprocess
import sys
import threading
import time

import pytest

from core.inference import os_sandbox, tools


@pytest.mark.parametrize(
    "platform,backend", [("linux", "sandbox_linux"), ("darwin", "sandbox_macos")]
)
def test_native_capability_never_imports_srt(monkeypatch, platform, backend):
    from types import SimpleNamespace

    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setitem(
        sys.modules,
        "core.inference." + backend,
        SimpleNamespace(
            BACKEND_NAME = backend,
            PROFILE_ID = "native",
            LIMITATIONS = (),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "core.inference.sandbox_probe",
        SimpleNamespace(probe = lambda *a, **k: (True, "passed")),
    )
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert "srt" not in name
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert os_sandbox.capability_snapshot().backend == backend


@pytest.mark.parametrize("kind", ["python", "terminal"])
@pytest.mark.parametrize("mode", ["auto", "required"])
@pytest.mark.parametrize("reason", ["missing", "failed", "timed out"])
def test_unavailable_mode_matrix(monkeypatch, tmp_path, kind, mode, reason):
    monkeypatch.setattr(
        os_sandbox,
        "capability_snapshot",
        lambda **k: os_sandbox.SandboxCapability("srt", False, reason),
    )
    plan = os_sandbox.ToolLaunchPlan(
        (sys.executable, "-c", "print(17)"),
        str(tmp_path),
        {},
        requested_mode = mode,
        execution_kind = kind,
    )
    starts = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: starts.append(a))
    if mode == "required":
        with pytest.raises(os_sandbox.SandboxUnavailableError):
            tools._prepare_tool_launch(plan)
        assert starts == []
    else:
        prepared = tools._prepare_tool_launch(plan)
        os_sandbox.spawn_prepared_launch(prepared)
        assert len(starts) == 1
        assert not prepared.execution_record.os_isolation


@pytest.mark.parametrize(
    "error", [OSError("startup"), RuntimeError("uncertain"), os_sandbox.SandboxBuildError("unsafe")]
)
def test_auto_preparation_failure_does_not_replay(monkeypatch, tmp_path, error):
    def fail(plan):
        raise error

    monkeypatch.setattr(os_sandbox, "prepare_tool_launch", fail)
    monkeypatch.setattr(tools, "_software_safeguards_launch", lambda *a: pytest.fail("host replay"))
    with pytest.raises(os_sandbox.SandboxBuildError):
        tools._prepare_tool_launch(os_sandbox.ToolLaunchPlan(("python",), str(tmp_path), {}))


def test_cancel_after_capability_wait_refuses(monkeypatch, tmp_path):
    cancel = threading.Event()

    def probe(**kwargs):
        cancel.set()
        return os_sandbox.SandboxCapability("srt", False, "unavailable")

    monkeypatch.setattr(os_sandbox, "capability_snapshot", probe)
    with pytest.raises(os_sandbox.SandboxBuildError, match = "cancelled"):
        os_sandbox.prepare_tool_launch(
            os_sandbox.ToolLaunchPlan((sys.executable,), str(tmp_path), {}, cancel_event = cancel)
        )


def test_windows_warm_and_concurrent_probe_cache(monkeypatch):
    from core.inference import srt_probe, srt_adapter

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(srt_adapter, "installation_identity", lambda: "runtime-a")
    monkeypatch.setattr(srt_probe, "runtime_inputs", lambda: ())
    count = []

    def probe(**kwargs):
        count.append(1)
        time.sleep(0.05)
        return True, "passed"

    monkeypatch.setattr(srt_probe, "_native_probe", probe)
    srt_probe.invalidate_cache()
    with concurrent.futures.ThreadPoolExecutor(4) as executor:
        assert all(result[0] for result in executor.map(lambda _: srt_probe.probe(), range(4)))
    assert srt_probe.probe()[0]
    assert len(count) == 1
    monkeypatch.setattr(srt_adapter, "installation_identity", lambda: "runtime-b")
    assert srt_probe.probe()[0]
    assert len(count) == 2
    srt_probe.invalidate_cache()


@pytest.mark.parametrize("mode", ["auto", "required"])
@pytest.mark.parametrize("kind", ["python", "terminal"])
def test_available_windows_launch_executes_once(monkeypatch, tmp_path, mode, kind):
    from core.inference import sandbox_windows, srt_adapter

    monkeypatch.setattr(sys, "platform", "win32")
    capability = os_sandbox.SandboxCapability("srt", True, "passed", environment = "win32")
    monkeypatch.setattr(os_sandbox, "capability_snapshot", lambda **kwargs: capability)
    monkeypatch.setattr(srt_adapter, "request_for", lambda *args: {})
    starts = []
    monkeypatch.setattr(srt_adapter, "spawn", lambda *args, **kwargs: starts.append(1) or object())
    monkeypatch.setattr(srt_adapter, "release_control", lambda proc: None)
    plan = os_sandbox.ToolLaunchPlan(
        (sys.executable,), str(tmp_path), {}, requested_mode = mode, execution_kind = kind
    )
    prepared = tools._prepare_tool_launch(plan)
    os_sandbox.spawn_prepared_launch(prepared)
    assert starts == [1]
    assert prepared.execution_record.os_isolation
    assert prepared.execution_record.requested_mode == mode
    prepared.cleanup()


def test_isolation_failure_stops_model_retries():
    from core.inference.tool_loop_controller import ToolLoopController

    controller = ToolLoopController(tools = None)
    call = {"id": "one", "function": {"name": "python", "arguments": {"code": "print(1)"}}}
    decision = controller.prepare_call(call)
    controller.record_result(
        decision, tools._sandbox_refusal(os_sandbox.SandboxBuildError("startup failed"))
    )
    assert controller.force_final_answer
    assert controller.active_tools() == []
    assert controller.prepare_call(call).action == "disabled"


@pytest.mark.parametrize("mode", ["auto", "required"])
@pytest.mark.parametrize("kind", ["python", "terminal"])
def test_failed_windows_admission_rechecks_next_request_without_replay(
    monkeypatch, tmp_path, mode, kind
):
    from core.inference import sandbox_windows, srt_adapter, srt_probe

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(srt_adapter, "installation_identity", lambda: "unchanged-runtime")
    monkeypatch.setattr(srt_probe, "runtime_inputs", lambda: ())
    monkeypatch.setattr(srt_probe, "_windows_probe_shell", lambda: sys.executable)
    available = [True]
    probes = []

    def probe(**kwargs):
        probes.append(available[0])
        return available[0], "controlled native readiness"

    monkeypatch.setattr(srt_probe, "_native_probe", probe)
    monkeypatch.setattr(srt_adapter, "request_for", lambda *args: {})
    launches = []

    def fail(*args, **kwargs):
        launches.append(1)
        raise srt_adapter.SrtError("native setup is no longer available")

    monkeypatch.setattr(srt_adapter, "spawn", fail)
    host_starts = []
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: host_starts.append(1))
    srt_probe.invalidate_cache()
    try:
        assert sandbox_windows.capability_snapshot().available
        assert sandbox_windows.capability_snapshot().available
        assert probes == [True]
        available[0] = False
        plan = os_sandbox.ToolLaunchPlan(
            (sys.executable,), str(tmp_path), {}, requested_mode = mode, execution_kind = kind
        )
        with pytest.raises(os_sandbox.SandboxBuildError, match = "launch failed"):
            os_sandbox.spawn_prepared_launch(os_sandbox.prepare_tool_launch(plan))
        assert launches == [1]
        assert host_starts == []
        assert probes == [True]
        if mode == "required":
            with pytest.raises(os_sandbox.SandboxUnavailableError):
                os_sandbox.prepare_tool_launch(plan)
            assert host_starts == []
        else:
            next_launch = os_sandbox.prepare_tool_launch(plan)
            assert not next_launch.execution_record.os_isolation
            os_sandbox.spawn_prepared_launch(next_launch)
            assert host_starts == [1]
        assert probes == [True, False]
        assert launches == [1]
    finally:
        srt_probe.invalidate_cache()
