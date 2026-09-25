# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import base64
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
        mxc_policy, "_runtime_read_roots", lambda executable: [str(Path(executable).parent)]
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
        mxc_adapter.subprocess, "Popen", lambda *_a, **_k: pytest.fail("malformed config dispatched")
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
