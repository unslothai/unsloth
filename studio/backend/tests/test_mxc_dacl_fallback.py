# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import base64
import os
import shutil
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.inference import mxc_adapter, mxc_policy, mxc_probe, mxc_runtime, os_sandbox
from core.inference import sandbox_windows_mxc

OPT_IN = mxc_policy.DACL_FALLBACK_ENV


@pytest.fixture(autouse = True)
def _studio_home(monkeypatch, tmp_path):
    home = tmp_path / "studio-home"
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(home))
    monkeypatch.delenv(OPT_IN, raising = False)
    mxc_probe.invalidate_cache()
    yield home
    mxc_probe.invalidate_cache()


def _policy_plan(tmp_path):
    workdir = tmp_path / "workdir"
    workdir.mkdir(exist_ok = True)
    runtime = tmp_path / "runtime" / "python.exe"
    runtime.parent.mkdir(exist_ok = True)
    runtime.touch()
    return os_sandbox.ToolLaunchPlan(
        argv = (str(runtime), "-c", "print('ok')"),
        workdir = str(workdir),
        env = {},
        execution_kind = "python",
    )


def _build(monkeypatch, plan):
    monkeypatch.setattr(mxc_policy.sys, "platform", "win32")
    # Only the runtime's own dir: here sys.prefix may contain tmp_path.
    monkeypatch.setattr(
        mxc_policy,
        "_runtime_read_roots",
        lambda executable, extra = (): [str(Path(executable).parent)],
    )
    monkeypatch.setattr(os_sandbox, "model_library_roots", lambda: ())
    return mxc_policy.build_launch_request(plan, run_id = "fixed")


@pytest.mark.parametrize(
    ("value", "enabled"),
    [(None, False), ("", False), ("0", False), ("true", False), ("yes", False), ("1", True)],
)
def test_dacl_fallback_needs_the_explicit_opt_in(monkeypatch, value, enabled):
    if value is not None:
        monkeypatch.setenv(OPT_IN, value)
    assert mxc_policy.dacl_fallback_enabled() is enabled


def test_opt_in_is_the_only_config_change(monkeypatch, tmp_path):
    plan = _policy_plan(tmp_path)
    default = _build(monkeypatch, plan)
    assert default["config"]["fallback"] == {"allowDaclMutation": False}
    monkeypatch.setenv(OPT_IN, "1")
    opted = _build(monkeypatch, plan)
    assert opted["config"]["fallback"] == {"allowDaclMutation": True}
    opted["config"]["fallback"] = default["config"]["fallback"]
    assert opted["config"] == default["config"]
    assert opted["policyHash"] != default["policyHash"]


def test_grant_covering_the_dacl_journal_is_refused(monkeypatch, tmp_path):
    plan = _policy_plan(tmp_path)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(Path(plan.workdir) / "studio"))
    with pytest.raises(mxc_policy.MxcPolicyError, match = "DACL restore journal"):
        _build(monkeypatch, plan)


def _request(fallback):
    config = {"fallback": fallback, "ui": {"disable": False}}
    return {
        "config": config,
        "configBytes": mxc_policy.canonical_config_bytes(config),
        "policyHash": mxc_policy.compute_policy_hash(config),
    }


def _stub_dispatch(monkeypatch, tmp_path):
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
        observed["env"] = kwargs["env"]
        return Proc()

    monkeypatch.setattr(mxc_adapter.subprocess, "Popen", popen)
    return observed


def test_dacl_config_is_refused_without_the_host_opt_in(monkeypatch, tmp_path):
    _stub_dispatch(monkeypatch, tmp_path)
    monkeypatch.setattr(
        mxc_adapter.subprocess, "Popen", lambda *_a, **_k: pytest.fail("DACL config dispatched")
    )
    with pytest.raises(mxc_adapter.MxcAdapterError, match = "not enabled on this host"):
        mxc_adapter.spawn(_request({"allowDaclMutation": True}))


def test_dacl_config_dispatches_with_the_host_opt_in(monkeypatch, tmp_path):
    observed = _stub_dispatch(monkeypatch, tmp_path)
    monkeypatch.setenv(OPT_IN, "1")
    mxc_adapter.spawn(_request({"allowDaclMutation": True}))
    assert json.loads(base64.b64decode(observed["argv"][2]))["fallback"] == {
        "allowDaclMutation": True
    }


@pytest.mark.parametrize(
    "fallback",
    [None, {}, {"allowDaclMutation": 1}, {"allowDaclMutation": False, "extra": True}, [False]],
)
def test_malformed_fallback_is_refused_even_with_the_opt_in(monkeypatch, tmp_path, fallback):
    _stub_dispatch(monkeypatch, tmp_path)
    monkeypatch.setenv(OPT_IN, "1")
    monkeypatch.setattr(
        mxc_adapter.subprocess,
        "Popen",
        lambda *_a, **_k: pytest.fail("malformed config dispatched"),
    )
    with pytest.raises(mxc_adapter.MxcAdapterError, match = "malformed"):
        mxc_adapter.spawn(_request(fallback))


@pytest.mark.parametrize("opt_in", [False, True])
def test_wxc_always_gets_the_studio_dacl_journal(monkeypatch, tmp_path, _studio_home, opt_in):
    observed = _stub_dispatch(monkeypatch, tmp_path)
    if opt_in:
        monkeypatch.setenv(OPT_IN, "1")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "profile"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "profile"))
    mxc_adapter.spawn(_request({"allowDaclMutation": opt_in}))
    journal = _studio_home / "mxc-runtime" / "dacl-restore"
    assert observed["env"]["MXC_DACL_STATE_DIR"] == str(journal)
    assert journal.is_dir()
    assert "LOCALAPPDATA" not in observed["env"]
    assert "USERPROFILE" not in observed["env"]


def _capability(monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox_windows_mxc.sys, "platform", "win32")
    monkeypatch.setattr(sandbox_windows_mxc.mxc_probe, "probe", lambda *_a, **_k: (True, "ok"))
    monkeypatch.setattr(sandbox_windows_mxc.mxc_runtime, "installation_identity", lambda: "runner")
    return sandbox_windows_mxc.capability_snapshot(selected_executable = str(tmp_path / "python.exe"))


def test_capability_names_the_dacl_tier_only_when_opted_in(monkeypatch, tmp_path):
    default = _capability(monkeypatch, tmp_path)
    assert "mxc_tier3_dacl_host_permission_changes" not in default.limitations
    assert OPT_IN in default.remediation
    monkeypatch.setenv(OPT_IN, "1")
    opted = _capability(monkeypatch, tmp_path)
    assert "mxc_tier3_dacl_host_permission_changes" in opted.limitations
    assert "does not enable" not in opted.remediation
    assert "reboot" in opted.remediation
    # A qualification under one fallback setting never vouches for launches under the other.
    assert opted.environment_fingerprint != default.environment_fingerprint


def test_probe_cache_is_keyed_by_the_opt_in(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(mxc_probe.mxc_runtime, "installation_identity", lambda: "runner")
    monkeypatch.setattr(
        mxc_probe,
        "_probe",
        lambda executable, kind, cancel: calls.append(mxc_policy.dacl_fallback_enabled())
        or (False, "controlled"),
    )
    executable = str(tmp_path / "python.exe")
    mxc_probe.probe(executable)
    monkeypatch.setenv(OPT_IN, "1")
    mxc_probe.probe(executable)
    assert calls == [False, True]


class _Exited:
    _mxc_dispatched = True
    _mxc_policy_hash = "sha256:controlled"
    returncode = 1

    def __init__(self, reason, dacl):
        self._unsloth_completion_reason = reason
        self._mxc_dacl = dacl

    def poll(self):
        return self.returncode


@pytest.mark.parametrize("reason", ["timed_out", "cancelled"])
@pytest.mark.parametrize("recovered", [True, False])
def test_forced_dacl_exit_replays_the_journal_before_claiming_cleanup(
    monkeypatch, reason, recovered
):
    # Studio's kill skips wxc-exec's own ACE restore, so "complete" must be earned.
    calls = []
    monkeypatch.setattr(
        mxc_runtime, "recover_dacl_state", lambda env: calls.append(env) or recovered
    )
    result = mxc_adapter.completion_result(_Exited(reason, dacl = True))
    assert result["cleanup"] == ("complete" if recovered else "uncertain")
    assert calls and calls[0]["MXC_DACL_STATE_DIR"] == str(mxc_runtime.dacl_state_path())


@pytest.mark.parametrize(("reason", "dacl"), [("finished", True), ("timed_out", False)])
def test_clean_or_non_dacl_exits_need_no_replay(monkeypatch, reason, dacl):
    monkeypatch.setattr(
        mxc_runtime,
        "recover_dacl_state",
        lambda env: pytest.fail("replayed without a forced DACL exit"),
    )
    assert mxc_adapter.completion_result(_Exited(reason, dacl))["cleanup"] == "complete"


@pytest.mark.parametrize(
    ("returncode", "stderr", "clean"),
    [
        (0, "", True),
        (0, "DACL recovery: 1 file(s), 3 ACE(s) restored, 0 pruned (missing), 0 error(s)\n", True),
        (0, "DACL recovery: 1 file(s), 2 ACE(s) restored, 0 pruned (missing), 1 error(s)\n", False),
        (0, "DACL recovery failed: state file I/O error\n", False),
        (1, "", False),
    ],
    ids = ["nothing", "restored", "restore_error", "recovery_failed", "probe_failed"],
)
def test_journal_replay_reads_wxc_recovery_report(monkeypatch, returncode, stderr, clean):
    import subprocess
    monkeypatch.setattr(
        mxc_runtime,
        "_run_wxc_probe",
        lambda _root, _env: subprocess.CompletedProcess([], returncode, stdout = "{}", stderr = stderr),
    )
    assert mxc_runtime.recover_dacl_state({}) is clean


def test_trusted_terminal_path_dirs_are_granted_read_only(monkeypatch, tmp_path):
    # Git for Windows' usr\bin is on the terminal PATH; without a grant, ls/cat/grep fail inside MXC.
    from core.inference import tools

    trusted = tmp_path / "Program Files" / "Git" / "usr" / "bin"
    untrusted = tmp_path / "Users" / "me" / "bin"
    trusted.mkdir(parents = True)
    untrusted.mkdir(parents = True)
    monkeypatch.setattr(
        tools, "_is_trusted_windows_program_dir", lambda path: "Program Files" in path
    )
    plan = _policy_plan(tmp_path)
    plan = os_sandbox.ToolLaunchPlan(
        argv = plan.argv,
        workdir = plan.workdir,
        env = {"PATH": os.pathsep.join([str(untrusted), str(trusted)])},
        execution_kind = "terminal",
    )
    monkeypatch.setattr(mxc_policy.sys, "platform", "win32")
    monkeypatch.setattr(os_sandbox, "model_library_roots", lambda: ())
    # This host's venv is the workspace root, which would contain tmp_path and the journal.
    monkeypatch.setattr(mxc_policy.sys, "prefix", str(tmp_path / "runtime"))
    monkeypatch.setattr(mxc_policy.sys, "base_prefix", str(tmp_path / "runtime"))
    monkeypatch.setattr(mxc_policy.site, "getsitepackages", lambda: [])
    readonly = mxc_policy.build_launch_request(plan, run_id = "fixed")["config"]["filesystem"][
        "readonlyPaths"
    ]
    assert str(trusted) in readonly
    assert str(untrusted) not in readonly


@pytest.mark.skipif(not shutil.which("bash"), reason = "needs bash")
@pytest.mark.parametrize("has_cat", [True, False])
def test_bash_probe_needs_a_working_cat_to_qualify(tmp_path, has_cat):
    # A missing cat left the capture empty, which read as a denied read.
    import subprocess

    argv = mxc_probe._terminal_probe(
        shutil.which("bash"), tmp_path, tmp_path / "canary", tmp_path / "outside"
    )
    env = {"PATH": os.environ["PATH"] if has_cat else str(tmp_path / "empty")}
    out = subprocess.run(argv, cwd = tmp_path, env = env, capture_output = True, text = True).stdout
    assert ("UNSLOTH_MXC_TERMINAL_PROBE_OK" in out) is has_cat


def test_the_journal_path_is_absolute_under_a_relative_home(monkeypatch, tmp_path):
    # wxc-exec runs with the runtime dir as its cwd, so a relative path would name another journal.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", "relative-home")
    path = mxc_runtime.dacl_state_path()
    assert path.is_absolute()
    assert path == tmp_path / "relative-home" / "mxc-runtime" / "dacl-restore"


@pytest.mark.parametrize("replay", [True, False])
def test_probe_journal_is_forced_and_empty_when_not_replaying(monkeypatch, tmp_path, replay):
    import contextlib
    import subprocess

    seen = {}

    @contextlib.contextmanager
    def lease(package_root = None):
        yield SimpleNamespace(info = SimpleNamespace(path = tmp_path / "wxc-exec.exe"))

    def run(argv, **kwargs):
        journal = kwargs["env"]["MXC_DACL_STATE_DIR"]
        seen.update(journal = journal, contents = os.listdir(journal))
        return subprocess.CompletedProcess(argv, 0, stdout = "{}", stderr = "")

    monkeypatch.setattr(mxc_runtime, "acquire_runtime", lease)
    monkeypatch.setattr(mxc_runtime.subprocess, "run", run)
    # An inherited value must not win: a same-user process could point it anywhere.
    mxc_runtime._run_wxc_probe(
        None, {"MXC_DACL_STATE_DIR": str(tmp_path / "planted")}, replay_journal = replay
    )
    studio_journal = str(mxc_runtime.dacl_state_path())
    assert (seen["journal"] == studio_journal) is replay
    assert seen["journal"] != str(tmp_path / "planted")
    if not replay:
        assert seen["contents"] == [] and not os.path.exists(seen["journal"])


def test_a_policy_refusal_at_dispatch_never_replays_on_the_host(monkeypatch, tmp_path):
    # A workdir that gained a reparse point after planning refuses, as it would at planning time.
    plan = os_sandbox.ToolLaunchPlan(
        argv = (str(tmp_path / "python.exe"), "-c", "print(1)"),
        workdir = str(tmp_path),
        env = {},
        requested_mode = "auto",
        execution_kind = "python",
    )
    capability = os_sandbox.SandboxCapability(
        backend = "mxc-processcontainer",
        available = True,
        reason = "qualified",
        environment = "win32",
        profile_id = mxc_runtime.PROFILE_ID,
        environment_fingerprint = sandbox_windows_mxc._capability_fingerprint(
            "qualified", "python", plan.argv[0]
        ),
    )
    monkeypatch.setattr(
        sandbox_windows_mxc.mxc_runtime, "installation_identity", lambda: "qualified"
    )
    monkeypatch.setattr(
        sandbox_windows_mxc.mxc_policy,
        "build_launch_request",
        lambda _plan: {"policyHash": "sha256:controlled"},
    )

    def refuse(*_args, **_kwargs):
        raise mxc_adapter.MxcAdapterError(
            "the MXC workdir contains a reparse point", stage = "policy"
        )

    monkeypatch.setattr(sandbox_windows_mxc.mxc_adapter, "spawn", refuse)
    monkeypatch.setattr(
        sandbox_windows_mxc.subprocess,
        "Popen",
        lambda *_a, **_k: pytest.fail("a policy refusal was replayed on the host"),
    )
    prepared = sandbox_windows_mxc.prepare(plan, capability)
    with pytest.raises(os_sandbox.SandboxBuildError, match = "without host replay"):
        os_sandbox.spawn_prepared_launch(prepared)


def test_verify_failure_is_reported_as_a_policy_stage(monkeypatch, tmp_path):
    request = _request({"allowDaclMutation": False})
    _stub_dispatch(monkeypatch, tmp_path)

    def changed(_request):
        raise mxc_policy.MxcPolicyError("the MXC workdir changed before WXC dispatch")

    monkeypatch.setattr(mxc_policy, "verify_launch_identities", changed)
    with pytest.raises(mxc_adapter.MxcAdapterError) as raised:
        mxc_adapter.spawn(request)
    assert raised.value.stage == "policy"
    assert raised.value.may_have_started is False


def test_a_probe_error_reclaims_the_running_workload(monkeypatch, tmp_path):
    aborted = []

    class Running:
        returncode = None

        def poll(self):
            return None

        def communicate(self, timeout = None):
            raise OSError("pipe broke")

    monkeypatch.setattr(mxc_probe.sys, "platform", "win32")
    monkeypatch.setattr(
        mxc_probe.sys, "getwindowsversion", lambda: SimpleNamespace(build = 26100), raising = False
    )
    monkeypatch.setattr(mxc_probe.mxc_runtime, "wxc_path", lambda: tmp_path / "wxc-exec.exe")
    monkeypatch.setattr(mxc_probe.mxc_policy, "build_launch_request", lambda _plan: {})
    monkeypatch.setattr(mxc_probe.mxc_adapter, "spawn", lambda *_a, **_k: Running())
    monkeypatch.setattr(mxc_probe.mxc_adapter, "abort", lambda proc: aborted.append(proc))
    monkeypatch.setattr(mxc_probe.mxc_adapter, "release_runtime", lambda _proc: None)
    monkeypatch.setattr(mxc_probe.subprocess, "CREATE_NO_WINDOW", 0, raising = False)
    available, reason = mxc_probe._probe(str(tmp_path / "python.exe"), "python")
    assert available is False and "pipe broke" in reason
    assert len(aborted) == 1


def test_a_model_folder_inside_a_later_one_is_not_granted_twice(monkeypatch, tmp_path):
    models = tmp_path / "models"
    child = models / "child"
    child.mkdir(parents = True)
    monkeypatch.setattr(os_sandbox, "model_library_roots", lambda: (str(child), str(models)))
    monkeypatch.setattr(mxc_policy.sys, "platform", "win32")
    monkeypatch.setattr(
        mxc_policy,
        "_runtime_read_roots",
        lambda executable, extra = (): [str(Path(executable).parent)],
    )
    readonly = mxc_policy.build_launch_request(_policy_plan(tmp_path), run_id = "fixed")["config"][
        "filesystem"
    ]["readonlyPaths"]
    assert str(models) in readonly and str(child) not in readonly
