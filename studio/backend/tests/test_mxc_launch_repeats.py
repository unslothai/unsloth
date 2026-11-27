# SPDX-License-Identifier: AGPL-3.0-only

"""Work an MXC launch used to repeat: policy serialization, read-grant leases, runtime root checks."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.inference import mxc_adapter, mxc_policy, mxc_read_grants, mxc_runtime, os_sandbox


@pytest.fixture(autouse = True)
def _isolated_dacl_state(monkeypatch, tmp_path):
    outside = Path(os.path.abspath(os.sep)) / "unsloth-test-dacl-journal-never-created"
    monkeypatch.setattr(mxc_runtime, "dacl_state_path", lambda: outside)
    monkeypatch.setattr(mxc_runtime, "dacl_state_dir", lambda: tmp_path)
    monkeypatch.setattr(mxc_read_grants, "ensure", lambda _roots: ())
    monkeypatch.setattr(mxc_read_grants, "revoke_recorded", lambda: ())
    # Root discovery imports tools, whose shell setup cannot run under a faked win32 on POSIX.
    monkeypatch.setattr(os_sandbox, "model_library_roots", lambda: ())
    monkeypatch.setattr(os_sandbox, "editable_source_roots", lambda: ())


def _policy_plan(tmp_path):
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    runtime = tmp_path / "runtime" / "python.exe"
    runtime.parent.mkdir()
    runtime.touch()
    return os_sandbox.ToolLaunchPlan(
        argv = (str(runtime), "-c", "print('ok')"),
        workdir = str(workdir),
        env = {},
        requested_mode = "auto",
        execution_kind = "python",
    )


def test_a_launch_serializes_its_policy_once_per_phase(monkeypatch, tmp_path):
    plan = _policy_plan(tmp_path)
    monkeypatch.setattr(mxc_policy.sys, "platform", "win32")
    serialized = []
    real = mxc_policy.canonical_config_bytes
    monkeypatch.setattr(
        mxc_policy, "canonical_config_bytes", lambda config: serialized.append(1) or real(config)
    )
    lease = SimpleNamespace(
        info = SimpleNamespace(path = tmp_path / "wxc-exec.exe"), release = lambda: None
    )
    monkeypatch.setattr(mxc_adapter.mxc_runtime, "acquire_runtime", lambda: lease)
    sent = {}

    class Proc:
        returncode = None

    monkeypatch.setattr(
        mxc_adapter.subprocess, "Popen", lambda argv, **_k: sent.setdefault("argv", argv) and Proc()
    )
    request = mxc_policy.build_launch_request(plan)
    assert request["policyHash"] == mxc_policy.compute_policy_hash(request["config"])
    serialized.clear()
    request = mxc_policy.build_launch_request(plan)
    mxc_adapter.spawn(request)
    # Build once, verify-at-dispatch once (was four: each hash re-serialized the config).
    assert len(serialized) == 2
    import base64

    assert base64.b64decode(sent["argv"][2]) == request["configBytes"]


def test_dispatch_still_refuses_a_hash_that_does_not_match_the_bytes(monkeypatch, tmp_path):
    config = {
        "version": mxc_runtime.MXC_SCHEMA_VERSION,
        "fallback": {"allowDaclMutation": False},
        "ui": {"disable": False},
    }
    request = {
        "config": config,
        "configBytes": mxc_policy.canonical_config_bytes(config),
        "policyHash": "sha256:" + "0" * 64,
    }
    monkeypatch.setattr(
        mxc_adapter.subprocess, "Popen", lambda *_a, **_k: pytest.fail("dispatched")
    )
    with pytest.raises(mxc_adapter.MxcAdapterError, match = "hash is invalid"):
        mxc_adapter.spawn(request)


def test_no_read_grant_lease_without_the_dacl_fallback(monkeypatch):
    monkeypatch.setattr(mxc_policy, "dacl_fallback_enabled", lambda: False)
    monkeypatch.setattr(mxc_read_grants, "hold", lambda: pytest.fail("leased a native-tier launch"))
    assert mxc_read_grants.hold_if_needed() is None
    request = {"config": {"fallback": {"allowDaclMutation": False}}}
    assert mxc_read_grants.hold_for_request(request, None, tried = False) is None


def test_a_request_built_after_the_switch_turned_on_is_still_leased(monkeypatch):
    lease = object()
    monkeypatch.setattr(mxc_read_grants, "hold", lambda: lease)
    monkeypatch.setattr(mxc_read_grants, "enabled", lambda: True)
    request = {"config": {"fallback": {"allowDaclMutation": True}}}
    assert mxc_read_grants.hold_for_request(request, None, tried = False) is lease
    held = object()
    assert mxc_read_grants.hold_for_request(request, held, tried = True) is held


def test_a_failed_early_lease_never_waits_on_the_record_lock_again(monkeypatch):
    attempts = []

    def refuse():
        attempts.append(1)
        raise mxc_read_grants.ReadGrantError("controlled lease failure")

    monkeypatch.setattr(mxc_policy, "dacl_fallback_enabled", lambda: True)
    monkeypatch.setattr(mxc_read_grants, "enabled", lambda: False)
    monkeypatch.setattr(mxc_read_grants, "hold", refuse)
    held, tried = mxc_read_grants.hold_before_build()
    assert (held, tried) == (None, True)
    request = {"config": {"fallback": {"allowDaclMutation": True}}}
    assert mxc_read_grants.hold_for_request(request, held, tried = tried) is None
    assert attempts == [1]


def test_a_dacl_request_whose_lease_cannot_be_recorded_is_refused(monkeypatch):
    def refuse():
        raise mxc_read_grants.ReadGrantError("controlled lease failure")

    monkeypatch.setattr(mxc_read_grants, "hold", refuse)
    monkeypatch.setattr(mxc_read_grants, "enabled", lambda: True)
    with pytest.raises(mxc_read_grants.ReadGrantError):
        mxc_read_grants.hold_for_request(
            {"config": {"fallback": {"allowDaclMutation": True}}}, None, tried = False
        )


def test_prepare_leases_when_the_switch_turns_on_during_the_build(monkeypatch, tmp_path):
    from core.inference import sandbox_windows_mxc

    events = []
    lease = SimpleNamespace(release = lambda: events.append("released"))
    state = {"dacl": False}
    monkeypatch.setattr(mxc_policy, "dacl_fallback_enabled", lambda: state["dacl"])
    monkeypatch.setattr(mxc_read_grants, "enabled", lambda: True)
    monkeypatch.setattr(mxc_read_grants, "hold", lambda: events.append("hold") or lease)

    def build(_plan, **_kw):
        state["dacl"] = True  # turned on by another request between the lease check and the build
        events.append("build")
        return {"policyHash": "sha256:x", "config": {"fallback": {"allowDaclMutation": True}}}

    monkeypatch.setattr(sandbox_windows_mxc.mxc_policy, "build_launch_request", build)
    capability = os_sandbox.SandboxCapability(
        backend = "mxc-processcontainer",
        available = True,
        reason = "qualified",
        environment = "win32",
        profile_id = mxc_runtime.PROFILE_ID,
    )
    plan = os_sandbox.ToolLaunchPlan(
        argv = ("python.exe",),
        workdir = str(tmp_path),
        env = {},
        requested_mode = "auto",
        timeout_seconds = 10,
        execution_kind = "python",
    )
    prepared = sandbox_windows_mxc.prepare(plan, capability)
    prepared.cleanup()
    assert events == ["build", "hold", "released"]


def test_prepare_takes_no_lease_for_a_native_tier_request(monkeypatch, tmp_path):
    from core.inference import sandbox_windows_mxc

    monkeypatch.setattr(mxc_policy, "dacl_fallback_enabled", lambda: False)
    monkeypatch.setattr(mxc_read_grants, "hold", lambda: pytest.fail("leased a native-tier launch"))
    monkeypatch.setattr(
        sandbox_windows_mxc.mxc_policy,
        "build_launch_request",
        lambda _plan, **_kw: {
            "policyHash": "sha256:x",
            "config": {"fallback": {"allowDaclMutation": False}},
        },
    )
    capability = os_sandbox.SandboxCapability(
        backend = "mxc-processcontainer",
        available = True,
        reason = "qualified",
        environment = "win32",
        profile_id = mxc_runtime.PROFILE_ID,
    )
    plan = os_sandbox.ToolLaunchPlan(
        argv = ("python.exe",),
        workdir = str(tmp_path),
        env = {},
        requested_mode = "auto",
        timeout_seconds = 10,
        execution_kind = "python",
    )
    sandbox_windows_mxc.prepare(plan, capability).cleanup()


def test_runtime_read_roots_check_each_spelling_once(monkeypatch, tmp_path):
    prefix = tmp_path / "venv"
    (prefix / "Scripts").mkdir(parents = True)
    executable = prefix / "Scripts" / "python.exe"
    executable.touch()
    monkeypatch.setattr(mxc_policy.sys, "prefix", str(prefix))
    monkeypatch.setattr(mxc_policy.sys, "base_prefix", str(prefix))
    # Windows: site.getsitepackages() == [sys.prefix, <prefix>\\Lib\\site-packages]
    monkeypatch.setattr(mxc_policy.site, "getsitepackages", lambda: [str(prefix), str(prefix)])
    monkeypatch.delenv("SystemRoot", raising = False)
    monkeypatch.delenv("WINDIR", raising = False)
    checked = []
    real = mxc_policy._safe_canonical_path
    monkeypatch.setattr(
        mxc_policy,
        "_safe_canonical_path",
        lambda path, **kw: checked.append(path) or real(path, **kw),
    )
    roots = mxc_policy._runtime_read_roots(str(executable))
    assert sorted(map(os.path.normcase, checked)) == sorted(set(map(os.path.normcase, checked)))
    assert os.path.realpath(prefix) in roots


def test_runtime_read_roots_keep_every_spelling_that_is_not_an_exact_repeat(monkeypatch, tmp_path):
    root = tmp_path / "Runtime"
    root.mkdir()
    checked = []
    monkeypatch.setattr(
        mxc_policy, "_safe_canonical_path", lambda path, **_k: checked.append(path) or str(path)
    )
    monkeypatch.setattr(mxc_policy.sys, "prefix", str(root))
    monkeypatch.setattr(mxc_policy.sys, "base_prefix", str(root))
    monkeypatch.setattr(mxc_policy.site, "getsitepackages", lambda: [str(root), str(root).lower()])
    mxc_policy._runtime_read_roots(str(root / "python.exe"))
    assert checked.count(str(root)) == 1
    if os.path.isdir(str(root).lower()):
        assert checked.count(str(root).lower()) == 1
