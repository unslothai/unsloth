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
from dataclasses import replace

import pytest

from core.inference import mxc_runtime, os_sandbox, tools


def _plan(tmp_path, mode = "auto"):
    return os_sandbox.ToolLaunchPlan(
        argv = (str(Path(__file__).resolve()), "arg"),
        workdir = str(tmp_path),
        env = {"PATH": "trusted"},
        requested_mode = mode,
        timeout_seconds = 10,
        execution_kind = "python",
    )


def _unavailable():
    return os_sandbox.SandboxCapability(
        backend = "mxc-processcontainer",
        available = False,
        reason = "controlled unavailable",
        environment = "win32",
        remediation = "install the pinned runner",
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
        os_sandbox.SandboxUnavailableError, match = "controlled unavailable"
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
        execution_kind = "terminal",
        selected_executable = str(tmp_path / "cmd.exe"),
    )
    assert capability.available
    assert observed == [(str(tmp_path / "cmd.exe"), "terminal")]


@pytest.mark.parametrize("name", ["powershell.exe", "pwsh.exe"])
def test_terminal_probe_supports_powershell_argv(name, tmp_path):
    from core.inference import mxc_probe

    argv = mxc_probe._terminal_probe(
        str(tmp_path / name),
        tmp_path,
        tmp_path / "other's secret.txt",
        tmp_path / "outside write.txt",
    )
    assert argv[:5] == (
        str(tmp_path / name),
        "-NoLogo",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
    )
    assert "UNSLOTH_MXC_TERMINAL_PROBE_OK" in argv[-1]
    assert "other''s secret.txt" in argv[-1]


def test_terminal_probe_supports_native_bash_argv(tmp_path):
    from core.inference import mxc_probe

    argv = mxc_probe._terminal_probe(
        str(tmp_path / "bash.exe"),
        tmp_path,
        tmp_path / "secret.txt",
        tmp_path / "outside.txt",
    )
    assert argv[:2] == (str(tmp_path / "bash.exe"), "-c")
    assert "UNSLOTH_MXC_TERMINAL_PROBE_OK" in argv[-1]


def test_unexpected_planner_failure_never_becomes_auto_host_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(
        os_sandbox, "prepare_tool_launch", lambda _plan: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    with pytest.raises(os_sandbox.SandboxBuildError, match = "without host fallback"):
        tools._prepare_tool_launch(_plan(tmp_path))


def test_selected_python_spelling_is_preserved_in_trusted_policy(monkeypatch, tmp_path):
    from core.inference import mxc_policy

    monkeypatch.setattr(mxc_policy.sys, "platform", "win32")
    selected = str(tmp_path / "venv" / "Scripts" / "python.exe")
    Path(selected).parent.mkdir(parents = True)
    Path(selected).touch()
    plan = os_sandbox.ToolLaunchPlan(
        argv = (selected, "-u", str(tmp_path / "tool.py")),
        workdir = str(tmp_path),
        env = {"PATH": str(Path(selected).parent)},
        execution_kind = "python",
    )
    request = mxc_policy.build_launch_request(plan)
    assert request["config"]["containerId"].startswith("unsloth-")
    assert request["policyHash"].startswith("sha256:")
    assert request["config"]["filesystem"]["readwritePaths"] == [str(tmp_path)]
    assert request["config"]["fallback"] == {"allowDaclMutation": False}
    assert request["config"]["ui"] == {"disable": False, "clipboard": "none", "injection": False}
    assert request["config"]["processContainer"]["ui"]["isolation"] == "container"


@pytest.mark.skipif(sys.platform != "win32", reason = "requires Windows MXC launch routing")
def test_dacl_refusal_after_successful_probe_is_never_replayed(monkeypatch, tmp_path):
    from core.inference import sandbox_windows_mxc

    plan = _plan(tmp_path, "required")
    identity = "qualified-runner"
    capability = os_sandbox.SandboxCapability(
        backend = "mxc-processcontainer",
        available = True,
        reason = "probe passed before the host changed",
        environment = "win32",
        profile_id = mxc_runtime.PROFILE_ID,
        environment_fingerprint = sandbox_windows_mxc._capability_fingerprint(
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
    assert "ui_isolation" in prepared.execution_record.retained_safeguards
    with pytest.raises(os_sandbox.SandboxBuildError, match = "without host replay"):
        os_sandbox.spawn_prepared_launch(prepared)
    assert host_calls == []


def test_capability_probe_is_single_flight_and_keyed_by_runtime(monkeypatch, tmp_path):
    from core.inference import mxc_probe

    mxc_probe.invalidate_cache()
    identity = ["generation-a"]
    calls = []
    monkeypatch.setattr(mxc_probe.mxc_runtime, "installation_identity", lambda: identity[0])

    def live_probe(
        executable,
        execution_kind,
        cancel_event = None,
    ):
        calls.append((identity[0], executable, execution_kind))
        time.sleep(0.05)
        return True, "qualified"

    monkeypatch.setattr(mxc_probe, "_probe", live_probe)
    executable = str(tmp_path / "python.exe")
    with ThreadPoolExecutor(max_workers = 4) as executor:
        results = list(
            executor.map(
                lambda _index: mxc_probe.probe(executable, execution_kind = "python"), range(4)
            )
        )
    assert results == [(True, "qualified")] * 4
    assert len(calls) == 1

    identity[0] = "generation-b"
    assert mxc_probe.probe(executable, execution_kind = "python") == (True, "qualified")
    assert len(calls) == 2


def test_cancelled_probe_waiter_does_not_block_behind_the_live_probe(monkeypatch, tmp_path):
    from core.inference import mxc_probe

    mxc_probe.invalidate_cache()
    started = threading.Event()
    release = threading.Event()
    calls = []
    monkeypatch.setattr(mxc_probe.mxc_runtime, "installation_identity", lambda: "generation")

    def live_probe(*_args, **_kwargs):
        calls.append(True)
        started.set()
        assert release.wait(5)
        return True, "qualified"

    monkeypatch.setattr(mxc_probe, "_probe", live_probe)
    executable = str(tmp_path / "python.exe")
    cancel = threading.Event()
    with ThreadPoolExecutor(max_workers = 2) as executor:
        leader = executor.submit(mxc_probe.probe, executable)
        assert started.wait(2)
        waiter = executor.submit(mxc_probe.probe, executable, cancel_event = cancel)
        cancel.set()
        assert waiter.result(timeout = 1) == (False, "MXC capability probe was cancelled")
        assert not release.is_set()
        mxc_probe.invalidate_cache()
        release.set()
        assert leader.result(timeout = 2) == (True, "qualified")
    assert calls == [True]


@pytest.mark.skipif(sys.platform != "win32", reason = "requires Windows MXC launch routing")
def test_runner_replacement_after_probe_is_refused_before_spawn(monkeypatch, tmp_path):
    from core.inference import sandbox_windows_mxc

    plan = _plan(tmp_path, "required")
    qualified_identity = "qualified-runner"
    fingerprint = sandbox_windows_mxc._capability_fingerprint(
        qualified_identity, "python", plan.argv[0]
    )
    capability = os_sandbox.SandboxCapability(
        backend = "mxc-processcontainer",
        available = True,
        reason = "qualified",
        environment = "win32",
        profile_id = mxc_runtime.PROFILE_ID,
        environment_fingerprint = fingerprint,
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
    with pytest.raises(os_sandbox.SandboxBuildError, match = "changed after capability"):
        os_sandbox.spawn_prepared_launch(prepared)
    assert spawned == []


def test_terminal_policy_preserves_constructed_shell_argv(monkeypatch, tmp_path):
    from core.inference import mxc_policy

    shell = tmp_path / "Program Files" / "PowerShell" / "7" / "pwsh.exe"
    shell.parent.mkdir(parents = True)
    shell.touch()
    monkeypatch.setattr(mxc_policy.sys, "platform", "win32")
    plan = os_sandbox.ToolLaunchPlan(
        argv = (str(shell), "-NoProfile", "-Command", '"quoted"; $env:VALUE | Out-File "x y.txt"'),
        workdir = str(tmp_path),
        env = {"PATH": str(shell.parent)},
        execution_kind = "terminal",
    )
    request = mxc_policy.build_launch_request(plan)
    assert request["config"]["process"]["commandLine"] == subprocess.list2cmdline(list(plan.argv))
    assert request["config"]["fallback"]["allowDaclMutation"] is False


def test_wsl_bash_is_refused_before_execution(monkeypatch, tmp_path):
    from core.inference import mxc_policy

    shell = tmp_path / "Windows" / "System32" / "bash.exe"
    shell.parent.mkdir(parents = True)
    shell.touch()
    monkeypatch.setattr(mxc_policy.sys, "platform", "win32")
    plan = os_sandbox.ToolLaunchPlan(
        argv = (str(shell), "-c", "echo hi"),
        workdir = str(tmp_path),
        env = {},
        execution_kind = "terminal",
    )
    with pytest.raises(mxc_policy.MxcPolicyError, match = "WSL-backed"):
        mxc_policy.build_launch_request(plan)


@pytest.mark.skipif(sys.platform != "win32", reason = "requires Windows junction semantics")
def test_workdir_junction_escape_is_refused(monkeypatch, tmp_path):
    from core.inference import mxc_policy

    if not hasattr(os.path, "isjunction"):
        pytest.skip("junction detection requires Python 3.12 on Windows")
    workdir = tmp_path / "会話"
    outside = tmp_path / "other-chat"
    workdir.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding = "utf-8")
    junction = workdir / "junction_to_secret"
    created = subprocess.run(
        ["cmd", "/d", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output = True,
        text = True,
        check = False,
    )
    if created.returncode != 0:
        pytest.skip(f"junction creation unavailable: {created.stderr or created.stdout}")
    monkeypatch.setattr(mxc_policy.sys, "platform", "win32")
    plan = os_sandbox.ToolLaunchPlan(
        argv = (sys.executable, "-c", "print('never')"),
        workdir = str(workdir),
        env = {},
        execution_kind = "python",
    )
    with pytest.raises(mxc_policy.MxcPolicyError, match = "reparse point"):
        mxc_policy.build_launch_request(plan)


def test_workdir_hard_link_to_outside_object_is_refused(monkeypatch, tmp_path):
    from core.inference import mxc_policy

    workdir = tmp_path / "workdir"
    outside = tmp_path / "other-chat"
    workdir.mkdir()
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("secret", encoding = "utf-8")
    try:
        os.link(secret, workdir / "linked-secret.txt")
    except OSError as exc:
        pytest.skip(f"hard-link creation unavailable: {exc}")
    monkeypatch.setattr(mxc_policy.sys, "platform", "win32")
    plan = os_sandbox.ToolLaunchPlan(
        argv = (sys.executable, "-c", "print('never')"),
        workdir = str(workdir),
        env = {},
        execution_kind = "python",
    )
    with pytest.raises(mxc_policy.MxcPolicyError, match = "hard-linked"):
        mxc_policy.build_launch_request(plan)


def _policy_plan(workdir, mode = "auto"):
    # A plain file: a venv's python is a symlink on POSIX, which the policy refuses.
    runtime = Path(workdir).parent / "runtime" / "python.exe"
    runtime.parent.mkdir(exist_ok = True)
    runtime.touch()
    return os_sandbox.ToolLaunchPlan(
        argv = (str(runtime), "-c", "print('ok')"),
        workdir = str(workdir),
        env = {},
        requested_mode = mode,
        execution_kind = "python",
    )


def test_workdir_hard_links_within_it_are_allowed(monkeypatch, tmp_path):
    # uv and pip hard-link from their caches; refusing every nlink > 1 bricked the chat.
    from core.inference import mxc_policy

    workdir = tmp_path / "workdir"
    workdir.mkdir()
    (workdir / "a.txt").write_text("same", encoding = "utf-8")
    try:
        os.link(workdir / "a.txt", workdir / "b.txt")
    except OSError as exc:
        pytest.skip(f"hard-link creation unavailable: {exc}")
    monkeypatch.setattr(mxc_policy.sys, "platform", "win32")
    request = mxc_policy.build_launch_request(_policy_plan(workdir))
    assert request["launchLimitations"] == ()


@pytest.mark.parametrize("mode", ["auto", "required"])
def test_oversized_workdir_degrades_auto_and_refuses_required(monkeypatch, tmp_path, mode):
    from core.inference import mxc_policy

    for index in range(8):
        (tmp_path / f"f{index}.txt").write_text("x", encoding = "utf-8")
    monkeypatch.setattr(mxc_policy.sys, "platform", "win32")
    monkeypatch.setattr(os_sandbox, "WORKDIR_SCAN_ENTRIES", 3)
    if mode == "required":
        with pytest.raises(mxc_policy.MxcPolicyError, match = "too large"):
            mxc_policy.build_launch_request(_policy_plan(tmp_path, mode))
        return
    request = mxc_policy.build_launch_request(_policy_plan(tmp_path, mode))
    assert request["launchLimitations"] == (os_sandbox.WORKDIR_SCAN_INCOMPLETE,)
    mxc_policy.verify_launch_identities(request)


def test_incomplete_workdir_scan_is_named_on_the_execution_record(monkeypatch, tmp_path):
    from core.inference import sandbox_windows_mxc

    plan = _plan(tmp_path, "auto")
    capability = os_sandbox.SandboxCapability(
        backend = "mxc-processcontainer",
        available = True,
        reason = "qualified",
        environment = "win32",
        profile_id = mxc_runtime.PROFILE_ID,
        limitations = ("mxc_preview_not_a_security_boundary",),
    )
    monkeypatch.setattr(
        sandbox_windows_mxc.mxc_policy,
        "build_launch_request",
        lambda _plan: {
            "policyHash": "sha256:controlled",
            "launchLimitations": (os_sandbox.WORKDIR_SCAN_INCOMPLETE,),
        },
    )
    prepared = sandbox_windows_mxc.prepare(plan, capability)
    assert prepared.launch_limitations == (os_sandbox.WORKDIR_SCAN_INCOMPLETE,)
    assert os_sandbox.WORKDIR_SCAN_INCOMPLETE in prepared.execution_record.limitations


def test_registered_model_folders_are_granted_read_only(monkeypatch, tmp_path):
    from core.inference import mxc_policy

    workdir = tmp_path / "workdir"
    models = tmp_path / "models"
    inside = workdir / "local-models"
    for path in (workdir, models, inside):
        path.mkdir()
    missing = tmp_path / "gone"
    monkeypatch.setattr(mxc_policy.sys, "platform", "win32")
    monkeypatch.setattr(
        os_sandbox, "model_library_roots", lambda: (str(models), str(missing), str(inside))
    )
    # Only the runtime's own dir: here sys.prefix may contain tmp_path and swallow the grant.
    monkeypatch.setattr(
        mxc_policy, "_runtime_read_roots", lambda executable: [os.path.dirname(executable)]
    )
    request = mxc_policy.build_launch_request(_policy_plan(workdir))
    filesystem = request["config"]["filesystem"]
    assert str(models) in filesystem["readonlyPaths"]
    assert str(missing) not in filesystem["readonlyPaths"]
    assert str(inside) not in filesystem["readonlyPaths"]
    assert filesystem["readwritePaths"] == [str(workdir)]


@pytest.mark.skipif(sys.platform != "win32", reason = "requires Windows 8.3 short names")
def test_short_name_workdir_is_not_mistaken_for_a_reparse_point(monkeypatch, tmp_path):
    import ctypes
    from core.inference import mxc_policy

    workdir = tmp_path / "long directory name for 8dot3"
    workdir.mkdir()
    buffer = ctypes.create_unicode_buffer(1024)
    if not ctypes.windll.kernel32.GetShortPathNameW(str(workdir), buffer, 1024):
        pytest.skip("GetShortPathNameW failed")
    short = buffer.value
    if os.path.normcase(short) == os.path.normcase(str(workdir)):
        pytest.skip("8.3 name generation is disabled on this volume")
    monkeypatch.setattr(mxc_policy.sys, "platform", "win32")
    canonical = mxc_policy._safe_canonical_path(short, directory = True)
    assert os.path.normcase(canonical) == os.path.normcase(os.path.realpath(workdir))


@pytest.mark.parametrize(
    "path",
    [
        r"\\server\share\workdir",
        r"\\?\C:\workdir",
        r"\\.\C:\workdir",
        r"C:\workdir\file.txt:secret",
        "C:\\workdir\\e\u0301",
    ],
)
@pytest.mark.skipif(sys.platform != "win32", reason = "requires Windows path semantics")
def test_unsupported_windows_path_namespaces_are_refused(monkeypatch, path):
    from core.inference import mxc_policy
    monkeypatch.setattr(mxc_policy.sys, "platform", "win32")
    with pytest.raises(mxc_policy.MxcPolicyError, match = "not supported"):
        mxc_policy._safe_canonical_path(path, directory = True)


def test_direct_adapter_sends_the_exact_canonical_config(monkeypatch, tmp_path):
    import base64
    import json
    from types import SimpleNamespace

    from core.inference import mxc_adapter, mxc_policy

    config = {
        "version": mxc_runtime.MXC_SCHEMA_VERSION,
        "fallback": {"allowDaclMutation": False},
        "ui": {"disable": False},
    }
    request = {
        "config": config,
        "configBytes": mxc_policy.canonical_config_bytes(config),
        "policyHash": mxc_policy.compute_policy_hash(config),
    }
    lease = SimpleNamespace(
        info = SimpleNamespace(path = tmp_path / "wxc-exec.exe", sha256 = "digest"),
        release = lambda: None,
    )
    monkeypatch.setattr(mxc_adapter.mxc_runtime, "acquire_runtime", lambda: lease)
    monkeypatch.setattr(mxc_adapter.mxc_policy, "verify_launch_identities", lambda _request: None)
    observed = {}

    class Proc:
        returncode = None

    def popen(argv, **kwargs):
        observed["argv"] = argv
        observed["kwargs"] = kwargs
        return Proc()

    monkeypatch.setattr(mxc_adapter.subprocess, "Popen", popen)
    proc = mxc_adapter.spawn(request)
    sent = base64.b64decode(observed["argv"][2])
    assert observed["argv"][:2] == [str(lease.info.path), "--config-base64"]
    assert sent == request["configBytes"]
    assert json.loads(sent)["fallback"]["allowDaclMutation"] is False
    assert json.loads(sent)["ui"]["disable"] is False
    assert proc._mxc_backend_tier == "unknown"


def test_policy_mutation_is_refused_before_wxc_dispatch(monkeypatch):
    from core.inference import mxc_adapter, mxc_policy

    config = {
        "fallback": {"allowDaclMutation": False},
        "ui": {"disable": False},
    }
    request = {
        "config": config,
        "configBytes": mxc_policy.canonical_config_bytes(config),
        "policyHash": mxc_policy.compute_policy_hash(config),
    }
    request["config"]["fallback"]["allowDaclMutation"] = True
    monkeypatch.setattr(
        mxc_adapter.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("mutated configuration was executed"),
    )
    with pytest.raises(mxc_adapter.MxcAdapterError, match = "changed before dispatch"):
        mxc_adapter.spawn(request)


def test_launch_failure_is_not_replayed(monkeypatch, tmp_path):
    calls = []
    prepared = os_sandbox.PreparedSandboxLaunch(
        argv = ("model-authored.exe", "argument"),
        workdir = str(tmp_path),
        env = {},
        preexec_fn = None,
        backend = "mxc-processcontainer",
    )

    def fail(_prepared, _kwargs):
        calls.append("mxc")
        raise os_sandbox.SandboxBuildError("controlled launch failure")

    prepared.spawn_callback = fail
    monkeypatch.setattr(subprocess, "Popen", lambda *_a, **_k: calls.append("host"))
    with pytest.raises(os_sandbox.SandboxBuildError, match = "controlled"):
        os_sandbox.spawn_prepared_launch(prepared)
    assert calls == ["mxc"]


@pytest.mark.skipif(sys.platform != "win32", reason = "requires Windows MXC launch routing")
def test_auto_spawn_failure_before_dispatch_uses_software_safeguards(monkeypatch, tmp_path):
    from core.inference import mxc_adapter, sandbox_windows_mxc

    plan = _plan(tmp_path, "auto")
    identity = "qualified-wxc"
    capability = os_sandbox.SandboxCapability(
        backend = "mxc-processcontainer",
        available = True,
        reason = "qualified",
        environment = "win32",
        profile_id = mxc_runtime.PROFILE_ID,
        environment_fingerprint = sandbox_windows_mxc._capability_fingerprint(
            identity, "python", plan.argv[0]
        ),
    )
    monkeypatch.setattr(os_sandbox, "capability_snapshot", lambda **_kwargs: capability)
    monkeypatch.setattr(sandbox_windows_mxc.mxc_runtime, "installation_identity", lambda: identity)
    monkeypatch.setattr(
        sandbox_windows_mxc.mxc_adapter,
        "spawn",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            mxc_adapter.MxcAdapterError("CreateProcess failed", stage = "spawn")
        ),
    )
    calls = []

    class Proc:
        pass

    monkeypatch.setattr(
        sandbox_windows_mxc.subprocess,
        "Popen",
        lambda argv, **_kwargs: calls.append(tuple(argv)) or Proc(),
    )
    prepared = os_sandbox.prepare_tool_launch(plan)
    os_sandbox.spawn_prepared_launch(prepared)
    assert calls == [plan.argv]
    assert prepared.backend == "software-safeguards"
    assert prepared.execution_record.effective_mode == "software_safeguards"


def test_auto_cancellation_before_dispatch_never_replays_on_host(monkeypatch, tmp_path):
    from core.inference import sandbox_windows_mxc

    cancel = threading.Event()
    plan = replace(_plan(tmp_path, "auto"), cancel_event = cancel)
    identity = "qualified-wxc"
    capability = os_sandbox.SandboxCapability(
        backend = "mxc-processcontainer",
        available = True,
        reason = "qualified",
        environment = "win32",
        profile_id = mxc_runtime.PROFILE_ID,
        environment_fingerprint = sandbox_windows_mxc._capability_fingerprint(
            identity, "python", plan.argv[0]
        ),
    )
    monkeypatch.setattr(sandbox_windows_mxc.mxc_runtime, "installation_identity", lambda: identity)
    monkeypatch.setattr(
        sandbox_windows_mxc.mxc_policy,
        "build_launch_request",
        lambda _plan: {"policyHash": "sha256:controlled"},
    )
    monkeypatch.setattr(
        sandbox_windows_mxc.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("cancelled command was replayed on the host"),
    )

    prepared = sandbox_windows_mxc.prepare(plan, capability)
    cancel.set()
    with pytest.raises(os_sandbox.SandboxBuildError, match = "without host replay"):
        os_sandbox.spawn_prepared_launch(prepared)
    assert prepared.execution_record.execution_status == "not_started"
    assert prepared.execution_record.completion_status == "cancelled"
    assert prepared.execution_record.cleanup_status == "complete"


@pytest.mark.skipif(sys.platform != "win32", reason = "requires Windows MXC launch routing")
def test_auto_failure_after_possible_dispatch_never_replays_on_host(monkeypatch, tmp_path):
    from core.inference import mxc_adapter, sandbox_windows_mxc

    plan = _plan(tmp_path, "auto")
    identity = "qualified-wxc"
    capability = os_sandbox.SandboxCapability(
        backend = "mxc-processcontainer",
        available = True,
        reason = "qualified",
        environment = "win32",
        profile_id = mxc_runtime.PROFILE_ID,
        environment_fingerprint = sandbox_windows_mxc._capability_fingerprint(
            identity, "python", plan.argv[0]
        ),
    )
    monkeypatch.setattr(os_sandbox, "capability_snapshot", lambda **_kwargs: capability)
    monkeypatch.setattr(sandbox_windows_mxc.mxc_runtime, "installation_identity", lambda: identity)
    monkeypatch.setattr(
        sandbox_windows_mxc.mxc_adapter,
        "spawn",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            mxc_adapter.MxcAdapterError(
                "WXC process state was lost",
                stage = "dispatch",
                may_have_started = True,
            )
        ),
    )
    monkeypatch.setattr(
        sandbox_windows_mxc.subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("original command was replayed"),
    )
    prepared = os_sandbox.prepare_tool_launch(plan)
    with pytest.raises(os_sandbox.SandboxBuildError, match = "without host replay"):
        os_sandbox.spawn_prepared_launch(prepared)
    assert prepared.execution_record.execution_status == "unknown_start"


def test_uncertain_completion_is_terminal_and_invalidates_probe(monkeypatch, tmp_path):
    from core.inference import sandbox_windows_mxc

    prepared = os_sandbox.PreparedSandboxLaunch(
        argv = ("python.exe",),
        workdir = str(tmp_path),
        env = {},
        preexec_fn = None,
        backend = "mxc-processcontainer",
    )
    invalidated = []
    monkeypatch.setattr(
        sandbox_windows_mxc.mxc_adapter,
        "completion_result",
        lambda _proc: (_ for _ in ()).throw(RuntimeError("missing completion state")),
    )
    monkeypatch.setattr(
        sandbox_windows_mxc.mxc_probe, "invalidate_cache", lambda: invalidated.append(True)
    )
    with pytest.raises(os_sandbox.SandboxBuildError, match = "state is uncertain"):
        sandbox_windows_mxc.verify_success(prepared, object())
    assert invalidated == [True]


def test_public_full_remains_separate_from_tool_execution_mode():
    with pytest.raises(os_sandbox.SandboxUnavailableError, match = "not requestable"):
        tools._requested_execution_mode("full", False)
    assert tools._requested_execution_mode("auto", True) == "full"
    assert os_sandbox.PUBLIC_TOOL_EXECUTION_MODES == ("auto", "required")


def test_native_backends_do_not_import_or_reference_mxc():
    inference = Path(os_sandbox.__file__).parent
    for name in ("sandbox_linux.py", "sandbox_macos.py"):
        source = (inference / name).read_text(encoding = "utf-8").lower()
        assert "mxc" not in source


def test_non_windows_capability_does_not_import_windows_backend(monkeypatch):
    import sys

    sys.modules.pop("core.inference.sandbox_windows_mxc", None)
    monkeypatch.setattr(os_sandbox.sys, "platform", "unsupported-test-platform")
    capability = os_sandbox.capability_snapshot()
    assert not capability.available
    assert "core.inference.sandbox_windows_mxc" not in sys.modules
