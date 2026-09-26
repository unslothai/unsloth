# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.inference import mxc_probe, os_sandbox, sandbox_windows_mxc

# What Git for Windows' bash prints under an AppContainer before it reads its command (microsoft/mxc#1061).
MSYS_STARTUP_FAILURE = (
    "      0 [main] bash (8852) C:\\Program Files\\Git\\bin\\..\\usr\\bin\\bash.exe: *** fatal error - "
    "NtCreateDirectoryObject(\\BaseNamedObjects\\msys-2.0S5-1888ae32e00d56aa): 0xC0000022\n"
)
DLL_INIT_FAILED = 3221225794  # 0xC0000142


@pytest.fixture(autouse = True)
def _fresh_probe_cache():
    mxc_probe.invalidate_cache()
    yield
    mxc_probe.invalidate_cache()


def _run_probe(
    monkeypatch,
    tmp_path,
    executable,
    kind,
    output,
    exit_code,
    cleanup = "complete",
):
    class Finished:
        returncode = exit_code

        def poll(self):
            return exit_code

        def communicate(self, timeout = None):
            return output, None

    monkeypatch.setattr(mxc_probe.sys, "platform", "win32")
    monkeypatch.setattr(
        mxc_probe.sys, "getwindowsversion", lambda: SimpleNamespace(build = 26100), raising = False
    )
    monkeypatch.setattr(mxc_probe.mxc_runtime, "wxc_path", lambda: tmp_path / "wxc-exec.exe")
    monkeypatch.setattr(mxc_probe.mxc_policy, "build_launch_request", lambda _plan: {})
    monkeypatch.setattr(mxc_probe.mxc_adapter, "spawn", lambda *_a, **_k: Finished())
    monkeypatch.setattr(mxc_probe.mxc_adapter, "abort", lambda _proc: None)
    monkeypatch.setattr(mxc_probe.mxc_adapter, "release_runtime", lambda _proc: None)
    monkeypatch.setattr(
        mxc_probe.mxc_adapter,
        "completion_result",
        lambda _proc: {"exitCode": exit_code, "cleanup": cleanup},
    )
    monkeypatch.setattr(mxc_probe.subprocess, "CREATE_NO_WINDOW", 0, raising = False)
    return mxc_probe._probe(str(tmp_path / executable), kind)


def test_git_bash_namespace_failure_is_named(monkeypatch, tmp_path):
    available, reason = _run_probe(
        monkeypatch, tmp_path, "bash.exe", "terminal", MSYS_STARTUP_FAILURE, DLL_INIT_FAILED
    )
    assert available is False
    assert reason == mxc_probe.MSYS_NAMESPACE_REASON


@pytest.mark.parametrize(
    "executable, kind, output, cleanup",
    [
        # Another bash failure is not this one.
        ("bash.exe", "terminal", "bash: some other startup error\n", "complete"),
        # Uncertain cleanup outranks any diagnosis of the shell.
        ("bash.exe", "terminal", MSYS_STARTUP_FAILURE, "uncertain"),
        # The signature only means something from the shell it describes.
        ("cmd.exe", "terminal", MSYS_STARTUP_FAILURE, "complete"),
        ("pwsh.exe", "terminal", MSYS_STARTUP_FAILURE, "complete"),
        ("python.exe", "python", MSYS_STARTUP_FAILURE, "complete"),
    ],
)
def test_other_failures_keep_the_generic_reason(
    monkeypatch, tmp_path, executable, kind, output, cleanup
):
    available, reason = _run_probe(
        monkeypatch, tmp_path, executable, kind, output, DLL_INIT_FAILED, cleanup
    )
    assert available is False
    assert reason == "the live MXC probe did not complete cleanly"


def _count_live_probes(monkeypatch, reason):
    clock = [1000.0]
    calls = []
    monkeypatch.setattr(mxc_probe.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(mxc_probe.mxc_runtime, "installation_identity", lambda: "runtime-a")

    def live_probe(
        executable,
        execution_kind,
        cancel_event = None,
    ):
        calls.append(executable)
        return False, reason

    monkeypatch.setattr(mxc_probe, "_probe", live_probe)
    return clock, calls


def test_an_incompatible_shell_is_not_probed_again_each_negative_ttl(monkeypatch, tmp_path):
    clock, calls = _count_live_probes(monkeypatch, mxc_probe.MSYS_NAMESPACE_REASON)
    bash = str(tmp_path / "bash.exe")
    for _ in range(5):
        assert mxc_probe.probe(bash, execution_kind = "terminal") == (
            False,
            mxc_probe.MSYS_NAMESPACE_REASON,
        )
        clock[0] += mxc_probe.NEGATIVE_TTL + 1
    assert len(calls) == 1
    clock[0] += mxc_probe.INCOMPATIBLE_TTL
    mxc_probe.probe(bash, execution_kind = "terminal")
    assert len(calls) == 2


def test_a_generic_failure_still_expires_at_the_negative_ttl(monkeypatch, tmp_path):
    clock, calls = _count_live_probes(monkeypatch, "the live MXC probe did not complete cleanly")
    bash = str(tmp_path / "bash.exe")
    mxc_probe.probe(bash, execution_kind = "terminal")
    clock[0] += mxc_probe.NEGATIVE_TTL + 1
    mxc_probe.probe(bash, execution_kind = "terminal")
    assert len(calls) == 2


def test_a_new_runtime_or_opt_in_probes_the_shell_again(monkeypatch, tmp_path):
    _clock, calls = _count_live_probes(monkeypatch, mxc_probe.MSYS_NAMESPACE_REASON)
    bash = str(tmp_path / "bash.exe")
    monkeypatch.delenv("UNSLOTH_MXC_ALLOW_DACL_FALLBACK", raising = False)
    mxc_probe.probe(bash, execution_kind = "terminal")
    monkeypatch.setenv("UNSLOTH_MXC_ALLOW_DACL_FALLBACK", "1")
    mxc_probe.probe(bash, execution_kind = "terminal")
    monkeypatch.setattr(mxc_probe.mxc_runtime, "installation_identity", lambda: "runtime-b")
    mxc_probe.probe(bash, execution_kind = "terminal")
    assert len(calls) == 3


def _incompatible_snapshot(monkeypatch):
    monkeypatch.setattr(sandbox_windows_mxc.sys, "platform", "win32")
    monkeypatch.setenv("UNSLOTH_MXC_ALLOW_DACL_FALLBACK", "1")
    monkeypatch.setattr(
        sandbox_windows_mxc.mxc_probe,
        "probe",
        lambda *_a, **_k: (False, mxc_probe.MSYS_NAMESPACE_REASON),
    )
    monkeypatch.setattr(sandbox_windows_mxc.mxc_runtime, "installation_identity", lambda: "runner")
    monkeypatch.setattr(
        mxc_probe.mxc_runtime,
        "probe_host_prep_steps",
        lambda **_kwargs: pytest.fail("host preparation was probed for a shell it cannot help"),
    )


def test_the_remediation_names_the_shell_not_setup(monkeypatch, tmp_path):
    _incompatible_snapshot(monkeypatch)
    capability = sandbox_windows_mxc.capability_snapshot(
        execution_kind = "terminal", selected_executable = str(tmp_path / "bash.exe")
    )
    assert capability.available is False
    assert capability.reason == mxc_probe.MSYS_NAMESPACE_REASON
    assert "Install the pinned" not in capability.remediation
    assert "prepare" not in capability.remediation
    assert "Python tool" in capability.remediation


def _terminal_plan(tmp_path, mode):
    return os_sandbox.ToolLaunchPlan(
        argv = (str(tmp_path / "bash.exe"), "-c", "echo hi"),
        workdir = str(tmp_path),
        env = {"PATH": "trusted"},
        requested_mode = mode,
        timeout_seconds = 10,
        execution_kind = "terminal",
    )


def test_auto_runs_the_terminal_with_software_safeguards(monkeypatch, tmp_path):
    _incompatible_snapshot(monkeypatch)
    prepared = os_sandbox.prepare_tool_launch(_terminal_plan(tmp_path, "auto"))
    assert prepared.backend == "software-safeguards"
    assert prepared.execution_record.effective_mode == "software_safeguards"
    assert prepared.argv[0] == str(tmp_path / "bash.exe")


def test_required_refuses_with_the_real_cause(monkeypatch, tmp_path):
    _incompatible_snapshot(monkeypatch)
    with pytest.raises(os_sandbox.SandboxUnavailableError) as raised:
        os_sandbox.prepare_tool_launch(_terminal_plan(tmp_path, "required"))
    assert "Git Bash" in str(raised.value)
    assert "Install the pinned" not in raised.value.remediation


def test_an_in_place_git_update_probes_the_shell_again(monkeypatch, tmp_path):
    _clock, calls = _count_live_probes(monkeypatch, mxc_probe.MSYS_NAMESPACE_REASON)
    bin_dir = tmp_path / "Git" / "usr" / "bin"
    bin_dir.mkdir(parents = True)
    bash = bin_dir / "bash.exe"
    bash.write_bytes(b"bash")
    runtime = bin_dir / "msys-2.0.dll"
    runtime.write_bytes(b"msys-old")
    mxc_probe.probe(str(bash), execution_kind = "terminal")
    mxc_probe.probe(str(bash), execution_kind = "terminal")
    assert len(calls) == 1
    # Same path, new runtime: an upgrade that may have fixed the namespace call.
    runtime.write_bytes(b"msys-new-and-longer")
    mxc_probe.probe(str(bash), execution_kind = "terminal")
    assert len(calls) == 2
