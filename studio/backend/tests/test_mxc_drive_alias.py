# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from core.inference import mxc_adapter, mxc_drive_alias, mxc_runtime, os_sandbox


class _Host:
    """A fake DOS-device table: letter -> stack of targets, the first one in effect."""

    def __init__(self, drives = ""):
        self.table: dict[str, list[str]] = {}
        self.drives = drives
        self.alive: dict[int, float] = {4242: 1.0}
        self.pid = 4242
        self.refuse_define: set[str] = set()
        self.redirect: dict[str, str] = {}
        self.removed: list[tuple[str, str]] = []

    def logical_drives(self):
        return sum(1 << (ord(d) - ord("A")) for d in self.drives)

    def query(self, letter):
        stack = self.table.get(letter)
        return stack[0] if stack else None

    def define(self, letter, target):
        if letter in self.refuse_define:
            return False
        self.table.setdefault(letter, []).insert(0, "\\??\\" + self.redirect.get(letter, target))
        return True

    def remove(self, letter, target):
        stack = self.table.get(letter) or []
        if "\\??\\" + target not in stack:
            return False
        stack.remove("\\??\\" + target)
        self.removed.append((letter, target))
        if not stack:
            self.table.pop(letter, None)
        return True

    def process_alive(self, pid, create_time):
        return self.alive.get(pid) == create_time

    def own_identity(self):
        return self.pid, self.alive[self.pid]


@pytest.fixture
def host(monkeypatch, tmp_path):
    fake = _Host()
    monkeypatch.setattr(mxc_drive_alias, "_on_windows", lambda: True)
    monkeypatch.setattr(mxc_drive_alias, "_host", fake)
    monkeypatch.setattr(mxc_drive_alias, "_active", {})
    monkeypatch.setattr(
        mxc_drive_alias, "record_path", lambda: tmp_path / "state" / "drive-aliases.json"
    )
    monkeypatch.delenv(mxc_drive_alias.DRIVE_ALIAS_ENV, raising = False)
    return fake


def _workdir(tmp_path, name = "session"):
    path = tmp_path / name
    path.mkdir(exist_ok = True)
    return str(path)


def _record():
    return json.loads(mxc_drive_alias.record_path().read_text(encoding = "utf-8"))["aliases"]


def test_maps_the_highest_free_letter_and_records_it(host, tmp_path):
    host.drives = "CDZ"
    host.table["Y"] = ["\\??\\C:\\someone-else"]
    workdir = _workdir(tmp_path)
    lease = mxc_drive_alias.acquire(workdir)
    assert lease.letter == "X" and lease.root == "X:\\" and lease.target == workdir
    assert host.query("X") == "\\??\\" + workdir
    assert _record()["X"] == {"target": workdir, "pid": 4242, "pid_create_time": 1.0}
    lease.release()
    assert host.query("X") is None and host.query("Y") == "\\??\\C:\\someone-else"
    assert _record() == {}


def test_a_definition_that_does_not_verify_is_removed_and_skipped(host, tmp_path):
    host.redirect["Z"] = "C:\\elsewhere"
    host.refuse_define.add("Y")
    workdir = _workdir(tmp_path)
    lease = mxc_drive_alias.acquire(workdir)
    assert lease.letter == "X"
    # Removal is exact-match on our own target, so the misdirected definition stays for its owner.
    assert host.query("Z") == "\\??\\C:\\elsewhere"
    assert ("Z", "C:\\elsewhere") not in host.removed


def test_concurrent_launches_of_one_workdir_share_a_letter(host, tmp_path):
    workdir = _workdir(tmp_path)
    first = mxc_drive_alias.acquire(workdir)
    second = mxc_drive_alias.acquire(workdir)
    other = mxc_drive_alias.acquire(_workdir(tmp_path, "other"))
    assert first.letter == second.letter == "Z" and other.letter == "Y"
    first.release()
    first.release()  # idempotent: a second release must not drop the other lease
    assert host.query("Z") == "\\??\\" + workdir
    second.release()
    assert host.query("Z") is None
    other.release()
    assert host.table == {}


def test_a_letter_redefined_by_someone_else_is_left_alone(host, tmp_path):
    workdir = _workdir(tmp_path)
    lease = mxc_drive_alias.acquire(workdir)
    host.table["Z"].insert(0, "\\??\\C:\\other-owner")
    again = mxc_drive_alias.acquire(workdir)
    assert again.letter == "Y"
    lease.release()
    assert host.query("Z") == "\\??\\C:\\other-owner"


def test_a_dead_owners_mapping_is_reclaimed_only_while_it_still_matches(host, tmp_path):
    stale, moved = _workdir(tmp_path, "stale"), _workdir(tmp_path, "moved")
    host.table["Z"] = ["\\??\\" + stale]
    host.table["Y"] = ["\\??\\C:\\now-someone-else"]
    path = mxc_drive_alias.record_path()
    path.parent.mkdir(parents = True)
    path.write_text(
        json.dumps(
            {
                "aliases": {
                    "Z": {"target": stale, "pid": 99, "pid_create_time": 5.0},
                    "Y": {"target": moved, "pid": 99, "pid_create_time": 5.0},
                }
            }
        ),
        encoding = "utf-8",
    )
    lease = mxc_drive_alias.acquire(_workdir(tmp_path))
    assert ("Z", stale) in host.removed
    assert host.query("Y") == "\\??\\C:\\now-someone-else"
    assert lease.letter == "Z"
    assert set(_record()) == {"Z"}


def test_a_live_owners_mapping_is_never_reclaimed(host, tmp_path):
    theirs = _workdir(tmp_path, "theirs")
    host.alive[77] = 3.0
    host.table["Z"] = ["\\??\\" + theirs]
    path = mxc_drive_alias.record_path()
    path.parent.mkdir(parents = True)
    path.write_text(
        json.dumps({"aliases": {"Z": {"target": theirs, "pid": 77, "pid_create_time": 3.0}}}),
        encoding = "utf-8",
    )
    lease = mxc_drive_alias.acquire(_workdir(tmp_path))
    assert lease.letter == "Y" and host.query("Z") == "\\??\\" + theirs
    assert host.removed == []


def test_a_locked_record_maps_nothing(host, tmp_path, monkeypatch):
    from contextlib import contextmanager

    @contextmanager
    def locked():
        yield None

    monkeypatch.setattr(mxc_drive_alias, "_transaction", locked)
    assert mxc_drive_alias.acquire(_workdir(tmp_path)) is None
    assert host.table == {}


def test_an_unwritable_record_undoes_the_mapping(host, tmp_path, monkeypatch):
    monkeypatch.setattr(mxc_drive_alias, "_save_record", lambda _aliases: False)
    assert mxc_drive_alias.acquire(_workdir(tmp_path)) is None
    assert host.table == {}


def test_no_free_letter_maps_nothing(host, tmp_path):
    host.drives = mxc_drive_alias.LETTERS
    assert mxc_drive_alias.acquire(_workdir(tmp_path)) is None


def test_off_windows_and_opted_out_map_nothing(host, tmp_path, monkeypatch):
    monkeypatch.setenv(mxc_drive_alias.DRIVE_ALIAS_ENV, "0")
    assert mxc_drive_alias.acquire(_workdir(tmp_path)) is None
    monkeypatch.delenv(mxc_drive_alias.DRIVE_ALIAS_ENV)
    monkeypatch.setattr(mxc_drive_alias, "_on_windows", lambda: False)
    assert mxc_drive_alias.acquire(_workdir(tmp_path)) is None
    assert host.table == {}


def test_release_runtime_drops_the_alias():
    class _Lease:
        released = 0

        def release(self):
            self.released += 1

    class _Proc:
        pass

    proc, lease = _Proc(), _Lease()
    proc._mxc_drive_alias = lease
    mxc_adapter.release_runtime(proc)
    mxc_adapter.release_runtime(proc)
    assert lease.released == 1


# prepare() wiring: only a cmd Terminal launch gets an alias, and it is released on every path.


@pytest.fixture(autouse = True)
def _dacl_journal_outside_grants(monkeypatch, tmp_path):
    outside = Path(os.path.abspath(os.sep)) / "unsloth-test-dacl-journal-never-created"
    monkeypatch.setattr(mxc_runtime, "dacl_state_path", lambda: outside)
    monkeypatch.setattr(mxc_runtime, "dacl_state_dir", lambda: tmp_path)


class _FakeLease:
    def __init__(self):
        self.root = "Z:\\"
        self.released = 0

    def release(self):
        self.released += 1


def _terminal_plan(tmp_path, argv = ("cmd", "/c", "git status")):
    return os_sandbox.ToolLaunchPlan(
        argv = argv,
        workdir = str(tmp_path),
        env = {"PATH": "trusted"},
        requested_mode = "auto",
        timeout_seconds = 10,
        execution_kind = "terminal",
    )


def _capability(sandbox_windows_mxc, identity, plan):
    return os_sandbox.SandboxCapability(
        backend = "mxc-processcontainer",
        available = True,
        reason = "qualified",
        environment = "win32",
        profile_id = mxc_runtime.PROFILE_ID,
        environment_fingerprint = sandbox_windows_mxc._capability_fingerprint(
            identity, "terminal", plan.argv[0]
        ),
    )


@pytest.fixture
def backend(monkeypatch, tmp_path):
    from core.inference import sandbox_windows_mxc

    calls, leases = [], []

    def build(plan, *, cwd_alias = None):
        calls.append(cwd_alias)
        return {"policyHash": "sha256:controlled"}

    def acquire(workdir):
        lease = _FakeLease()
        leases.append((workdir, lease))
        return lease

    monkeypatch.setattr(sandbox_windows_mxc.mxc_policy, "build_launch_request", build)
    monkeypatch.setattr(
        sandbox_windows_mxc.mxc_policy, "_safe_canonical_path", lambda path, directory: path
    )
    monkeypatch.setattr(sandbox_windows_mxc.mxc_drive_alias, "acquire", acquire)
    monkeypatch.setattr(
        sandbox_windows_mxc.mxc_runtime, "installation_identity", lambda: "qualified-wxc"
    )
    return sandbox_windows_mxc, calls, leases


def test_a_cmd_terminal_launch_runs_from_the_alias_and_releases_it(backend, tmp_path, monkeypatch):
    sandbox_windows_mxc, calls, leases = backend

    class _Proc:
        pass

    monkeypatch.setattr(sandbox_windows_mxc.mxc_adapter, "spawn", lambda *_a, **_k: _Proc())
    plan = _terminal_plan(tmp_path)
    prepared = sandbox_windows_mxc.prepare(
        plan, _capability(sandbox_windows_mxc, "qualified-wxc", plan)
    )
    assert calls == ["Z:\\"] and leases[0][0] == str(tmp_path)
    assert sandbox_windows_mxc.CMD_PROFILE_LIMITATION in prepared.execution_record.limitations
    proc = os_sandbox.spawn_prepared_launch(prepared)
    assert proc._mxc_drive_alias is leases[0][1]
    prepared.cleanup()
    assert leases[0][1].released >= 1 and proc._mxc_drive_alias is None


def test_python_and_non_cmd_terminal_launches_get_no_alias(backend, tmp_path):
    sandbox_windows_mxc, calls, leases = backend
    for plan in (
        _terminal_plan(tmp_path, argv = ("C:\\Program Files\\Git\\bin\\bash.exe", "-c", "ls")),
        os_sandbox.ToolLaunchPlan(
            argv = ("python.exe", "-c", "1"),
            workdir = str(tmp_path),
            env = {},
            requested_mode = "auto",
            timeout_seconds = 10,
            execution_kind = "python",
        ),
    ):
        prepared = sandbox_windows_mxc.prepare(
            plan, _capability(sandbox_windows_mxc, "qualified-wxc", plan)
        )
        assert mxc_drive_alias.LIMITATION_UNAVAILABLE not in prepared.execution_record.limitations
    assert calls == [None, None] and leases == []


def test_a_cmd_launch_without_a_letter_is_named_on_the_record(backend, tmp_path, monkeypatch):
    sandbox_windows_mxc, calls, _leases = backend
    monkeypatch.setattr(sandbox_windows_mxc.mxc_drive_alias, "acquire", lambda _workdir: None)
    plan = _terminal_plan(tmp_path)
    prepared = sandbox_windows_mxc.prepare(
        plan, _capability(sandbox_windows_mxc, "qualified-wxc", plan)
    )
    assert calls == [None]
    assert mxc_drive_alias.LIMITATION_UNAVAILABLE in prepared.execution_record.limitations


def test_the_alias_is_released_when_policy_or_spawn_fails(backend, tmp_path, monkeypatch):
    sandbox_windows_mxc, _calls, leases = backend
    plan = _terminal_plan(tmp_path)
    capability = _capability(sandbox_windows_mxc, "qualified-wxc", plan)

    def refuse(plan, *, cwd_alias = None):
        raise RuntimeError("controlled policy refusal")

    monkeypatch.setattr(sandbox_windows_mxc.mxc_policy, "build_launch_request", refuse)
    with pytest.raises(os_sandbox.SandboxBuildError):
        sandbox_windows_mxc.prepare(plan, capability)
    assert leases[-1][1].released == 1

    monkeypatch.setattr(
        sandbox_windows_mxc.mxc_policy,
        "build_launch_request",
        lambda plan, *, cwd_alias = None: {"policyHash": "sha256:controlled"},
    )

    def fail(*_args, **_kwargs):
        raise sandbox_windows_mxc.mxc_adapter.MxcAdapterError("controlled refusal", stage = "policy")

    monkeypatch.setattr(sandbox_windows_mxc.mxc_adapter, "spawn", fail)
    prepared = sandbox_windows_mxc.prepare(plan, capability)
    with pytest.raises(os_sandbox.SandboxBuildError):
        os_sandbox.spawn_prepared_launch(prepared)
    assert leases[-1][1].released == 1


def test_without_the_policy_kwarg_no_alias_is_taken(backend, tmp_path, monkeypatch):
    sandbox_windows_mxc, _calls, leases = backend
    monkeypatch.setattr(
        sandbox_windows_mxc.mxc_policy,
        "build_launch_request",
        lambda _plan: {"policyHash": "sha256:controlled"},
    )
    plan = _terminal_plan(tmp_path)
    sandbox_windows_mxc.prepare(plan, _capability(sandbox_windows_mxc, "qualified-wxc", plan))
    assert leases == []
