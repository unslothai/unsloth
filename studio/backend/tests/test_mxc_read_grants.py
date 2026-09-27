# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

from core.inference import mxc_read_grants, mxc_runtime, os_sandbox


class _Host:
    """Stands in for the Windows DACL: which roots already grant, and what icacls was asked to do."""

    def __init__(self):
        self.granted: set[str] = set()
        self.calls: list[tuple[str, str]] = []
        self.fail_grant = False

    def check(self, root):
        return os.path.normcase(root) in self.granted

    def grant(self, root):
        self.calls.append(("grant", root))
        if self.fail_grant:
            return False, "Access is denied."
        self.granted.add(os.path.normcase(root))
        return True, "Successfully processed 1 files"

    def revoke(self, root):
        self.calls.append(("revoke", root))
        self.granted.discard(os.path.normcase(root))
        return True, "Successfully processed 1 files"


@pytest.fixture
def host(monkeypatch, tmp_path):
    fake = _Host()
    studio_home = tmp_path / "home" / ".unsloth" / "studio"
    studio_home.mkdir(parents = True)
    monkeypatch.setattr(mxc_read_grants, "_on_windows", lambda: True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio_home))
    monkeypatch.setenv("SystemRoot", str(tmp_path / "Windows"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv(mxc_read_grants.PERSISTENT_GRANTS_ENV, raising = False)
    monkeypatch.setattr(os_sandbox, "studio_state_roots", lambda: (str(studio_home),))
    monkeypatch.setattr(
        mxc_runtime, "dacl_state_path", lambda: studio_home / "mxc-runtime" / "dacl-restore"
    )
    monkeypatch.setattr(mxc_read_grants, "_dacl_grants_read", fake.check)
    monkeypatch.setattr(mxc_read_grants, "_grant", fake.grant)
    monkeypatch.setattr(mxc_read_grants, "_revoke", fake.revoke)
    fake.studio_home = studio_home
    return fake


def _runtime(host, name = "unsloth_studio"):
    root = host.studio_home / name
    (root / "Lib" / "site-packages").mkdir(parents = True)
    return str(root)


def _record(host):
    return json.loads(mxc_read_grants.record_path().read_text(encoding = "utf-8"))["grants"]


def test_eligible_runtime_root_is_granted_once_and_recorded(host):
    venv = _runtime(host)
    assert mxc_read_grants.ensure([venv]) == (venv,)
    assert host.calls == [("grant", venv)]
    assert _record(host) == {os.path.normcase(venv): "complete"}
    # The second launch finds the ACE and never walks the tree again.
    assert mxc_read_grants.ensure([venv]) == (venv,)
    assert host.calls == [("grant", venv)]


def test_a_folder_windows_already_grants_is_left_alone(host, tmp_path):
    program_files = tmp_path / "Program Files" / "Git" / "usr" / "bin"
    program_files.mkdir(parents = True)
    host.granted.add(os.path.normcase(str(program_files)))
    assert mxc_read_grants.ensure([str(program_files)]) == (str(program_files),)
    assert host.calls == []
    assert not mxc_read_grants.record_path().exists()


@pytest.mark.parametrize(
    "make_root, why",
    [
        (lambda host, tmp: str(host.studio_home.parent), "contains"),
        (lambda host, tmp: str(tmp / "home"), "contains"),
        (lambda host, tmp: str(tmp / "Windows" / "System32"), "Windows directory"),
    ],
)
def test_roots_above_protected_state_or_under_windows_are_never_granted(
    host, tmp_path, make_root, why
):
    root = make_root(host, tmp_path)
    os.makedirs(root, exist_ok = True)
    assert why in mxc_read_grants.ineligible_reason(root)
    assert mxc_read_grants.ensure([root]) == ()
    assert host.calls == []


@pytest.mark.parametrize("name", ["pip.ini", "PIP.CONF", ".pypirc", ".netrc", ".env"])
def test_a_root_holding_a_credential_file_keeps_the_per_launch_grant(host, name):
    venv = _runtime(host)
    Path(venv, name).write_text(
        "[global]\nindex-url = https://user:secret@example.invalid/simple\n"
    )
    assert "credential file" in mxc_read_grants.ineligible_reason(venv)
    assert mxc_read_grants.ensure([venv]) == ()
    assert host.calls == []


def test_a_failed_grant_is_revoked_so_wxc_exec_never_skips_an_unreadable_tree(host):
    venv = _runtime(host)
    host.fail_grant = True
    assert mxc_read_grants.ensure([venv]) == ()
    assert host.calls == [("grant", venv), ("revoke", venv)]
    assert _record(host) == {}


def test_a_grant_interrupted_by_a_crash_is_redone(host):
    venv = _runtime(host)
    host.granted.add(os.path.normcase(venv))  # the root ACE landed, the propagation may not have
    mxc_read_grants._save_record({os.path.normcase(venv): "pending"})
    assert mxc_read_grants.ensure([venv]) == (venv,)
    assert host.calls == [("grant", venv)]
    assert _record(host) == {os.path.normcase(venv): "complete"}


def test_opting_out_revokes_every_recorded_grant(host, monkeypatch):
    venv = _runtime(host)
    mxc_read_grants.ensure([venv])
    monkeypatch.setenv(mxc_read_grants.PERSISTENT_GRANTS_ENV, "0")
    assert not mxc_read_grants.enabled()
    assert mxc_read_grants.ensure([venv]) == ()
    assert host.calls == [("grant", venv), ("revoke", os.path.normcase(venv))]
    assert _record(host) == {}


def test_off_windows_nothing_is_touched(host, monkeypatch):
    monkeypatch.setattr(mxc_read_grants, "_on_windows", lambda: False)
    assert mxc_read_grants.ensure([_runtime(host)]) == ()
    assert mxc_read_grants.revoke_recorded() == ()
    assert host.calls == []


def _python_plan(tmp_path):
    selected = tmp_path / "venv" / "Scripts" / "python.exe"
    selected.parent.mkdir(parents = True)
    selected.touch()
    workdir = tmp_path / "work"
    workdir.mkdir()
    return os_sandbox.ToolLaunchPlan(
        argv = (str(selected), "-c", "pass"),
        workdir = str(workdir),
        env = {"PATH": str(selected.parent)},
        execution_kind = "python",
    )


@pytest.mark.parametrize("dacl", [True, False])
def test_launch_policy_grants_runtime_roots_only_on_the_dacl_tier(monkeypatch, tmp_path, dacl):
    from core.inference import mxc_policy

    outside = Path(os.path.abspath(os.sep)) / "unsloth-test-dacl-journal-never-created"
    monkeypatch.setattr(mxc_runtime, "dacl_state_path", lambda: outside)
    monkeypatch.setattr(mxc_policy.sys, "platform", "win32")
    model = os.path.abspath(os.path.join(os.sep, "unsloth-test-model-library"))
    monkeypatch.setattr(mxc_policy, "_model_read_roots", lambda _workdir, _granted: [model])
    if dacl:
        monkeypatch.setenv(mxc_policy.DACL_FALLBACK_ENV, "1")
    else:
        monkeypatch.delenv(mxc_policy.DACL_FALLBACK_ENV, raising = False)
    calls = []
    monkeypatch.setattr(
        mxc_read_grants, "ensure", lambda roots: calls.append(("ensure", list(roots))) or ()
    )
    monkeypatch.setattr(
        mxc_read_grants, "revoke_recorded", lambda: calls.append(("revoke", None)) or ()
    )

    request = mxc_policy.build_launch_request(_python_plan(tmp_path))
    readonly = request["config"]["filesystem"]["readonlyPaths"]
    assert model in readonly
    if dacl:
        [(kind, roots)] = calls
        assert kind == "ensure"
        # The model library stays a per-launch grant: it is user data, not Studio's runtime.
        assert model not in roots
        assert set(roots) <= set(readonly)
        assert os.path.realpath(sys.prefix) in [
            os.path.realpath(r) for r in roots
        ] or not os.path.isdir(sys.prefix)
    else:
        assert calls == [("revoke", None)]


def test_capability_names_the_persistent_grant_only_when_it_applies(monkeypatch):
    from core.inference import mxc_policy, sandbox_windows_mxc

    monkeypatch.setattr(sandbox_windows_mxc.sys, "platform", "win32")
    monkeypatch.setattr(sandbox_windows_mxc.mxc_probe, "probe", lambda *_a, **_k: (True, "ok"))
    monkeypatch.setattr(sandbox_windows_mxc.mxc_runtime, "installation_identity", lambda: "id")
    monkeypatch.setenv(mxc_policy.DACL_FALLBACK_ENV, "1")
    monkeypatch.delenv(mxc_read_grants.PERSISTENT_GRANTS_ENV, raising = False)
    on = sandbox_windows_mxc.capability_snapshot(
        execution_kind = "python", selected_executable = sys.executable
    )
    assert "mxc_tier3_persistent_runtime_read_grants" in on.limitations
    monkeypatch.setenv(mxc_read_grants.PERSISTENT_GRANTS_ENV, "0")
    off = sandbox_windows_mxc.capability_snapshot(
        execution_kind = "python", selected_executable = sys.executable
    )
    assert "mxc_tier3_persistent_runtime_read_grants" not in off.limitations
    monkeypatch.delenv(mxc_policy.DACL_FALLBACK_ENV)
    default = sandbox_windows_mxc.capability_snapshot(
        execution_kind = "python", selected_executable = sys.executable
    )
    assert "mxc_tier3_persistent_runtime_read_grants" not in default.limitations
    assert mxc_read_grants.PERSISTENT_GRANTS_ENV in default.remediation


@pytest.mark.skipif(sys.platform != "win32", reason = "reads a real Windows DACL")
def test_real_dacl_check_sees_a_granted_folder(tmp_path):
    import subprocess

    folder = tmp_path / "runtime"
    folder.mkdir()
    assert not mxc_read_grants._dacl_grants_read(str(folder))
    subprocess.run(
        ["icacls", str(folder), "/grant", "*S-1-15-2-1:(OI)(CI)(RX)", "/Q"],
        check = True,
        capture_output = True,
    )
    assert mxc_read_grants._dacl_grants_read(str(folder))
    subprocess.run(
        ["icacls", str(folder), "/remove:g", "*S-1-15-2-1", "/Q"], check = True, capture_output = True
    )
    assert not mxc_read_grants._dacl_grants_read(str(folder))
