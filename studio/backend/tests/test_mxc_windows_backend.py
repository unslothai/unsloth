# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from core.inference import os_sandbox, tools


def _plan(tmp_path, mode="auto"):
    return os_sandbox.ToolLaunchPlan(
        argv=(str(Path(__file__).resolve()), "arg"),
        workdir=str(tmp_path),
        env={"PATH": "trusted"},
        requested_mode=mode,
        timeout_seconds=10,
        execution_kind="python",
    )


def _unavailable():
    return os_sandbox.SandboxCapability(
        backend="mxc-processcontainer",
        available=False,
        reason="controlled unavailable",
        environment="win32",
        remediation="install the pinned runner",
    )


def test_auto_unavailable_preserves_original_launch(monkeypatch, tmp_path):
    plan = _plan(tmp_path)
    monkeypatch.setattr(os_sandbox, "capability_snapshot", lambda **_kwargs: _unavailable())
    prepared = os_sandbox.prepare_tool_launch(plan)
    assert prepared.argv == plan.argv
    assert prepared.workdir == plan.workdir
    assert prepared.env is plan.env
    assert prepared.preexec_fn is plan.preexec_fn
    assert prepared.backend == "software-safeguards"
    assert prepared.execution_record.effective_mode == "software_safeguards"


def test_required_unavailable_refuses_with_remediation(monkeypatch, tmp_path):
    plan = _plan(tmp_path, "required")
    monkeypatch.setattr(os_sandbox, "capability_snapshot", lambda **_kwargs: _unavailable())
    with pytest.raises(
        os_sandbox.SandboxUnavailableError, match="controlled unavailable"
    ) as raised:
        os_sandbox.prepare_tool_launch(plan)
    assert "pinned runner" in raised.value.remediation


def test_windows_terminal_uses_its_own_capability_identity(monkeypatch, tmp_path):
    from core.inference import sandbox_windows_mxc

    monkeypatch.setattr(sandbox_windows_mxc.sys, "platform", "win32")
    observed = []
    monkeypatch.setattr(
        sandbox_windows_mxc.mxc_probe,
        "probe",
        lambda executable, **kwargs: (
            observed.append((executable, kwargs["execution_kind"])) or (True, "ok")
        ),
    )
    monkeypatch.setattr(sandbox_windows_mxc.mxc_runtime, "installation_identity", lambda: "runner")
    capability = sandbox_windows_mxc.capability_snapshot(
        execution_kind="terminal",
        selected_executable=str(tmp_path / "powershell.exe"),
    )
    assert capability.available
    assert observed == [(str(tmp_path / "powershell.exe"), "terminal")]


def test_unexpected_planner_failure_never_becomes_auto_host_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(
        os_sandbox, "prepare_tool_launch", lambda _plan: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    with pytest.raises(os_sandbox.SandboxBuildError, match="without host fallback"):
        tools._prepare_tool_launch(_plan(tmp_path))


def test_selected_python_spelling_is_preserved_in_trusted_policy(monkeypatch, tmp_path):
    from core.inference import mxc_policy

    monkeypatch.setattr(mxc_policy.sys, "platform", "win32")
    selected = str(tmp_path / "venv" / "Scripts" / "python.exe")
    Path(selected).parent.mkdir(parents=True)
    Path(selected).touch()
    plan = os_sandbox.ToolLaunchPlan(
        argv=(selected, "-u", str(tmp_path / "tool.py")),
        workdir=str(tmp_path),
        env={"PATH": str(Path(selected).parent)},
        execution_kind="python",
    )
    request = mxc_policy.build_launch_request(plan)
    assert request["argv"][0] == selected
    assert request["containerId"].startswith("unsloth-")
    assert request["policyHash"].startswith("sha256:")
    assert request["readwritePaths"] == [str(tmp_path)]
    assert request["allowDaclMutation"] is False


def test_dacl_refusal_after_successful_probe_is_never_replayed(monkeypatch, tmp_path):
    from core.inference import sandbox_windows_mxc

    plan = _plan(tmp_path, "required")
    identity = "qualified-runner"
    capability = os_sandbox.SandboxCapability(
        backend="mxc-processcontainer",
        available=True,
        reason="probe passed before the host changed",
        environment="win32",
        profile_id="unsloth-mxc-windows-basecontainer-v1",
        environment_fingerprint=sandbox_windows_mxc._capability_fingerprint(
            identity, "python", plan.argv[0]
        ),
    )
    monkeypatch.setattr(os_sandbox, "capability_snapshot", lambda **_kwargs: capability)
    monkeypatch.setattr(sandbox_windows_mxc.mxc_runtime, "installation_identity", lambda: identity)
    monkeypatch.setattr(
        sandbox_windows_mxc.mxc_adapter,
        "spawn",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("fallback.allowDaclMutation=false refused Tier 3")
        ),
    )
    host_calls = []
    monkeypatch.setattr(subprocess, "Popen", lambda *_args, **_kwargs: host_calls.append(True))

    prepared = os_sandbox.prepare_tool_launch(plan)
    with pytest.raises(os_sandbox.SandboxBuildError, match="without host replay"):
        os_sandbox.spawn_prepared_launch(prepared)
    assert host_calls == []


def test_capability_probe_is_single_flight_and_keyed_by_runtime(monkeypatch, tmp_path):
    from core.inference import mxc_probe

    mxc_probe.invalidate_cache()
    identity = ["generation-a"]
    calls = []
    monkeypatch.setattr(mxc_probe.mxc_runtime, "installation_identity", lambda: identity[0])

    def live_probe(executable, execution_kind, cancel_event=None):
        calls.append((identity[0], executable, execution_kind))
        time.sleep(0.05)
        return True, "qualified"

    monkeypatch.setattr(mxc_probe, "_probe", live_probe)
    executable = str(tmp_path / "python.exe")
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda _index: mxc_probe.probe(executable, execution_kind="python"), range(4)
            )
        )
    assert results == [(True, "qualified")] * 4
    assert len(calls) == 1

    identity[0] = "generation-b"
    assert mxc_probe.probe(executable, execution_kind="python") == (True, "qualified")
    assert len(calls) == 2


def test_runner_replacement_after_probe_is_refused_before_spawn(monkeypatch, tmp_path):
    from core.inference import sandbox_windows_mxc

    plan = _plan(tmp_path, "required")
    qualified_identity = "qualified-runner"
    fingerprint = sandbox_windows_mxc._capability_fingerprint(
        qualified_identity, "python", plan.argv[0]
    )
    capability = os_sandbox.SandboxCapability(
        backend="mxc-processcontainer",
        available=True,
        reason="qualified",
        environment="win32",
        profile_id="unsloth-mxc-windows-basecontainer-v1",
        environment_fingerprint=fingerprint,
    )
    monkeypatch.setattr(os_sandbox, "capability_snapshot", lambda **_kwargs: capability)
    monkeypatch.setattr(
        sandbox_windows_mxc.mxc_runtime,
        "installation_identity",
        lambda: "replacement-runner",
    )
    spawned = []
    monkeypatch.setattr(
        sandbox_windows_mxc.mxc_adapter,
        "spawn",
        lambda *_args, **_kwargs: spawned.append(True),
    )
    prepared = os_sandbox.prepare_tool_launch(plan)
    with pytest.raises(os_sandbox.SandboxBuildError, match="changed after capability"):
        os_sandbox.spawn_prepared_launch(prepared)
    assert spawned == []


def test_terminal_policy_preserves_constructed_shell_argv(monkeypatch, tmp_path):
    from core.inference import mxc_policy

    shell = tmp_path / "Program Files" / "PowerShell" / "pwsh.exe"
    shell.parent.mkdir(parents=True)
    shell.touch()
    monkeypatch.setattr(mxc_policy.sys, "platform", "win32")
    plan = os_sandbox.ToolLaunchPlan(
        argv=(str(shell), "-NoProfile", "-Command", '"quoted"; $env:VALUE | Out-File "x y.txt"'),
        workdir=str(tmp_path),
        env={"PATH": str(shell.parent)},
        execution_kind="terminal",
    )
    request = mxc_policy.build_launch_request(plan)
    assert request["argv"] == list(plan.argv)
    assert request["executionKind"] == "terminal"
    assert request["allowDaclMutation"] is False


def test_wsl_bash_is_refused_before_execution(monkeypatch, tmp_path):
    from core.inference import mxc_policy

    shell = tmp_path / "Windows" / "System32" / "bash.exe"
    shell.parent.mkdir(parents=True)
    shell.touch()
    monkeypatch.setattr(mxc_policy.sys, "platform", "win32")
    plan = os_sandbox.ToolLaunchPlan(
        argv=(str(shell), "-c", "echo hi"),
        workdir=str(tmp_path),
        env={},
        execution_kind="terminal",
    )
    with pytest.raises(mxc_policy.MxcPolicyError, match="WSL-backed"):
        mxc_policy.build_launch_request(plan)


def test_workdir_junction_escape_is_refused(monkeypatch, tmp_path):
    from core.inference import mxc_policy

    if not hasattr(os.path, "isjunction"):
        pytest.skip("junction detection requires Python 3.12 on Windows")
    workdir = tmp_path / "会話"
    outside = tmp_path / "other-chat"
    workdir.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    junction = workdir / "junction_to_secret"
    created = subprocess.run(
        ["cmd", "/d", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        text=True,
        check=False,
    )
    if created.returncode != 0:
        pytest.skip(f"junction creation unavailable: {created.stderr or created.stdout}")
    monkeypatch.setattr(mxc_policy.sys, "platform", "win32")
    plan = os_sandbox.ToolLaunchPlan(
        argv=(sys.executable, "-c", "print('never')"),
        workdir=str(workdir),
        env={},
        execution_kind="python",
    )
    with pytest.raises(mxc_policy.MxcPolicyError, match="reparse point"):
        mxc_policy.build_launch_request(plan)


@pytest.mark.parametrize(
    ("event", "expected", "message"),
    [
        (
            {
                "v": 1,
                "event": "STARTED",
                "runId": "run",
                "token": "wrong",
                "backendTier": "base-container",
            },
            "STARTED",
            "authentication",
        ),
        (
            {
                "v": 1,
                "event": "STARTED",
                "runId": "other",
                "token": "secret",
                "backendTier": "base-container",
            },
            "STARTED",
            "authentication",
        ),
        (
            {
                "v": 1,
                "event": "STARTED",
                "runId": "run",
                "token": "secret",
                "backendTier": "base-container",
            },
            "FINISHED",
            "out-of-order",
        ),
        (
            {"v": 1, "event": "FINISHED", "runId": "run", "token": "secret"},
            "STARTED",
            "out-of-order",
        ),
    ],
)
def test_control_frames_fail_closed(event, expected, message):
    import json

    from core.inference import mxc_adapter

    with pytest.raises(mxc_adapter.MxcAdapterError, match=message):
        mxc_adapter._validated_event(
            json.dumps(event).encode(), {"runId": "run"}, "secret", expected
        )


def test_control_frame_bounds_and_malformed_input():
    from core.inference import mxc_adapter

    with pytest.raises(mxc_adapter.MxcAdapterError, match="exceeds"):
        mxc_adapter._validated_event(
            b"x" * (mxc_adapter.MAX_CONTROL + 1), {"runId": "run"}, "secret", "STARTED"
        )
    with pytest.raises(mxc_adapter.MxcAdapterError, match="malformed"):
        mxc_adapter._validated_event(b"{", {"runId": "run"}, "secret", "STARTED")


def test_named_pipe_endpoints_are_per_run_and_connect_cancel_is_prompt():
    from core.inference import mxc_pipe

    first = mxc_pipe.PrivatePipeServer(buffer_size=4096)
    second = mxc_pipe.PrivatePipeServer(buffer_size=4096)
    try:
        assert first.name != second.name
        cancel = threading.Event()
        cancel.set()
        started = time.monotonic()
        with pytest.raises(mxc_pipe.PipeError, match="cancelled"):
            first.accept(deadline=time.monotonic() + 30, cancel_event=cancel)
        assert time.monotonic() - started < 1
    finally:
        first.close()
        second.close()


def test_launch_failure_is_not_replayed(monkeypatch, tmp_path):
    calls = []
    prepared = os_sandbox.PreparedSandboxLaunch(
        argv=("model-authored.exe", "argument"),
        workdir=str(tmp_path),
        env={},
        preexec_fn=None,
        backend="mxc-processcontainer",
    )

    def fail(_prepared, _kwargs):
        calls.append("mxc")
        raise os_sandbox.SandboxBuildError("controlled launch failure")

    prepared.spawn_callback = fail
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: calls.append("host"))
    with pytest.raises(os_sandbox.SandboxBuildError, match="controlled"):
        os_sandbox.spawn_prepared_launch(prepared)
    assert calls == ["mxc"]


def test_uncertain_completion_is_terminal_and_invalidates_probe(monkeypatch, tmp_path):
    from core.inference import sandbox_windows_mxc

    prepared = os_sandbox.PreparedSandboxLaunch(
        argv=("python.exe",),
        workdir=str(tmp_path),
        env={},
        preexec_fn=None,
        backend="mxc-processcontainer",
    )
    invalidated = []
    monkeypatch.setattr(
        sandbox_windows_mxc.mxc_adapter,
        "completion_receipt",
        lambda _proc: (_ for _ in ()).throw(RuntimeError("missing FINISHED")),
    )
    monkeypatch.setattr(
        sandbox_windows_mxc.mxc_probe, "invalidate_cache", lambda: invalidated.append(True)
    )
    with pytest.raises(os_sandbox.SandboxBuildError, match="state is uncertain"):
        sandbox_windows_mxc.verify_success(prepared, object())
    assert invalidated == [True]


def test_public_full_remains_separate_from_tool_execution_mode():
    with pytest.raises(os_sandbox.SandboxUnavailableError, match="not requestable"):
        tools._requested_execution_mode("full", False)
    assert tools._requested_execution_mode("auto", True) == "full"
    assert os_sandbox.PUBLIC_TOOL_EXECUTION_MODES == ("auto", "required")


def test_native_backends_do_not_import_or_reference_mxc():
    inference = Path(os_sandbox.__file__).parent
    for name in ("sandbox_linux.py", "sandbox_macos.py"):
        source = (inference / name).read_text(encoding="utf-8").lower()
        assert "mxc" not in source


def test_non_windows_capability_does_not_import_windows_backend(monkeypatch):
    import sys

    sys.modules.pop("core.inference.sandbox_windows_mxc", None)
    monkeypatch.setattr(os_sandbox.sys, "platform", "unsupported-test-platform")
    capability = os_sandbox.capability_snapshot()
    assert not capability.available
    assert "core.inference.sandbox_windows_mxc" not in sys.modules
