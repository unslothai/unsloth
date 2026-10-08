# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
import sys

import pytest

from core.inference import mxc_read_grants, mxc_runtime, os_sandbox


class _Host:
    """Stands in for the Windows DACL: which roots already grant, and what icacls was asked to do."""

    def __init__(self):
        self.granted: set[str] = set()
        self.explicit: set[str] = set()
        self.calls: list[tuple[str, str]] = []
        self.fail_grant = False
        self.fail_revoke = False
        # Folders this account cannot re-ACL, like a Store Python package under WindowsApps.
        self.locked: set[str] = set()

    def aces(self, root):
        key = os.path.normcase(root)
        return key in self.granted, key in self.explicit

    def grant(self, root):
        self.calls.append(("grant", root))
        if os.path.normcase(root) in self.locked:
            return False, "Successfully processed 0 files; Failed processing 1 files"
        if self.fail_grant:
            # The root entry landed and propagation below it failed.
            self.explicit.add(os.path.normcase(root))
            return False, "Access is denied."
        self.granted.add(os.path.normcase(root))
        self.explicit.add(os.path.normcase(root))
        return True, "Successfully processed 1 files"

    def revoke(self, root):
        self.calls.append(("revoke", root))
        if self.fail_revoke or os.path.normcase(root) in self.locked:
            return False, "Access is denied."
        self.granted.discard(os.path.normcase(root))
        self.explicit.discard(os.path.normcase(root))
        return True, "Successfully processed 1 files"


def _isolate(monkeypatch, tmp_path):
    studio_home = tmp_path / "home" / ".unsloth" / "studio"
    studio_home.mkdir(parents = True, exist_ok = True)
    monkeypatch.setattr(mxc_read_grants, "_on_windows", lambda: True)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(studio_home))
    monkeypatch.setenv("SystemRoot", str(tmp_path / "Windows"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv(mxc_read_grants.PERSISTENT_GRANTS_ENV, raising = False)
    monkeypatch.setattr(os_sandbox, "studio_state_roots", lambda: (str(studio_home),))
    monkeypatch.setattr(
        mxc_runtime, "dacl_state_path", lambda: studio_home / "mxc-runtime" / "dacl-restore"
    )
    mxc_read_grants._scanned.clear()
    monkeypatch.setattr(mxc_read_grants, "_refused", {}, raising = False)
    return studio_home


@pytest.fixture
def host(monkeypatch, tmp_path):
    fake = _Host()
    fake.studio_home = _isolate(monkeypatch, tmp_path)
    monkeypatch.setattr(mxc_read_grants, "_package_aces", fake.aces)
    monkeypatch.setattr(mxc_read_grants, "_grant", fake.grant)
    monkeypatch.setattr(mxc_read_grants, "_revoke", fake.revoke)
    return fake


def _runtime(host, name = "unsloth_studio"):
    root = host.studio_home / name
    (root / "Lib" / "site-packages" / "six").mkdir(parents = True)
    (root / "Lib" / "site-packages" / "six" / "__init__.py").write_text("")
    return str(root)


def _record():
    path = mxc_read_grants.record_path()
    return json.loads(path.read_text(encoding = "utf-8"))["grants"] if path.exists() else {}


def _states():
    return {key: value["state"] for key, value in _record().items()}


def test_eligible_runtime_root_is_granted_once_and_recorded(host):
    venv = _runtime(host)
    assert mxc_read_grants.ensure([venv]) == (venv,)
    assert host.calls == [("grant", venv)]
    assert _states() == {os.path.normcase(venv): "complete"}
    assert _record()[os.path.normcase(venv)]["identity"]
    # The second launch finds the ACE and never walks the tree again.
    assert mxc_read_grants.ensure([venv]) == (venv,)
    assert host.calls == [("grant", venv)]


def test_a_folder_windows_already_grants_is_left_alone(host, tmp_path):
    program_files = tmp_path / "Program Files" / "Git" / "usr" / "bin"
    program_files.mkdir(parents = True)
    host.granted.add(os.path.normcase(str(program_files)))
    assert mxc_read_grants.ensure([str(program_files)]) == (str(program_files),)
    assert host.calls == []
    assert _record() == {}


def test_an_existing_entry_for_the_group_is_never_taken_over(host):
    # Someone else's partial grant: adding ours and later /remove:g would delete theirs too.
    venv = _runtime(host)
    host.explicit.add(os.path.normcase(venv))
    assert mxc_read_grants.ensure([venv]) == ()
    assert host.calls == []


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


@pytest.mark.parametrize("where", [".", "Lib/site-packages/six"])
@pytest.mark.parametrize("name", ["pip.ini", "PIP.CONF", ".pypirc", ".netrc", ".env"])
def test_a_credential_file_anywhere_in_the_tree_keeps_the_per_launch_grant(host, where, name):
    venv = _runtime(host)
    Path(venv, where, name).write_text(
        "[global]\nindex-url = https://user:secret@example.invalid/simple\n"
    )
    assert "credential file" in mxc_read_grants.ineligible_reason(venv)
    assert mxc_read_grants.ensure([venv]) == ()
    assert host.calls == []


def test_a_link_out_of_the_tree_keeps_the_per_launch_grant(host, tmp_path):
    venv = _runtime(host)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    try:
        os.symlink(elsewhere, Path(venv, "Lib", "linked"), target_is_directory = True)
    except OSError:
        pytest.skip("symlinks are not available")
    assert "reparse point" in mxc_read_grants.ineligible_reason(venv)
    assert mxc_read_grants.ensure([venv]) == ()
    assert host.calls == []


def test_a_link_to_a_file_in_the_same_tree_is_granted(host):
    venv = _runtime(host)
    Path(venv, "python.dat").write_bytes(b"")
    try:
        os.symlink(Path(venv, "python.dat"), Path(venv, "python3.dat"))
    except OSError:
        pytest.skip("symlinks are not available")
    assert mxc_read_grants.ineligible_reason(venv) is None
    assert mxc_read_grants.ensure([venv]) == (venv,)


def test_a_credential_named_link_inside_the_tree_is_refused(host):
    venv = _runtime(host)
    try:
        os.symlink(Path(venv, "Lib", "site-packages", "six", "__init__.py"), Path(venv, ".env"))
    except OSError:
        pytest.skip("symlinks are not available")
    assert "credential file" in mxc_read_grants.ineligible_reason(venv)
    assert mxc_read_grants.ensure([venv]) == ()


def test_a_link_that_leaves_the_tree_through_an_inner_link_is_refused(host, tmp_path):
    venv = _runtime(host)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    try:
        os.symlink(elsewhere, Path(venv, "Lib", "outer"), target_is_directory = True)
        os.symlink(Path(venv, "Lib", "outer"), Path(venv, "inner"), target_is_directory = True)
    except OSError:
        pytest.skip("symlinks are not available")
    assert "reparse point" in mxc_read_grants.ineligible_reason(venv)


def test_a_credential_file_added_after_the_grant_takes_it_back(host):
    venv = _runtime(host)
    mxc_read_grants.ensure([venv])
    Path(venv, "pip.ini").write_text("[global]\n")
    assert mxc_read_grants.ensure([venv]) == ()
    assert host.calls == [("grant", venv), ("revoke", os.path.normcase(venv))]
    assert _record() == {}


def test_a_failed_grant_is_revoked_so_wxc_exec_never_skips_an_unreadable_tree(host):
    venv = _runtime(host)
    host.fail_grant = True
    assert mxc_read_grants.ensure([venv]) == ()
    assert host.calls == [("grant", venv), ("revoke", venv)]
    assert _record() == {}


def test_a_failed_rollback_refuses_the_launch_and_keeps_the_retry(host):
    venv = _runtime(host)
    host.fail_grant = host.fail_revoke = True
    with pytest.raises(mxc_read_grants.ReadGrantError, match = "could not be rolled back"):
        mxc_read_grants.ensure([venv])
    assert _states() == {os.path.normcase(venv): "pending"}
    host.fail_grant = host.fail_revoke = False
    assert mxc_read_grants.ensure([venv]) == (venv,)
    assert _states() == {os.path.normcase(venv): "complete"}


def _store_python(host, tmp_path):
    root = tmp_path / "Program Files" / "WindowsApps" / "PythonSoftwareFoundation.Python.3.12"
    (root / "Lib" / "encodings").mkdir(parents = True)
    (root / "python.exe").write_text("")
    host.locked.add(os.path.normcase(str(root)))
    return str(root)


def test_a_folder_windows_refuses_keeps_the_per_launch_grant(host, tmp_path):
    # Store Python: TrustedInstaller owns the package, so icacls is denied on the root (#12941).
    root = _store_python(host, tmp_path)
    for _ in range(3):
        assert mxc_read_grants.ensure([root]) == ()
    # One attempt per process, no rollback of a change that never happened.
    assert host.calls == [("grant", root)]
    assert _record() == {}


def test_a_stuck_pending_grant_on_a_folder_windows_refuses_is_dropped(host, tmp_path):
    root = _store_python(host, tmp_path)
    key = os.path.normcase(root)
    pending = {key: {"state": "pending", "identity": mxc_read_grants._identity(root)}}
    mxc_read_grants._save_record(pending)
    assert mxc_read_grants.ensure([root]) == ()
    assert _record() == {}
    # The opt-out cleanup path, which logged "Could not remove" on every check.
    mxc_read_grants._save_record(pending)
    assert mxc_read_grants.revoke_recorded() == ()
    assert _record() == {}
    assert host.calls == [("grant", root), ("revoke", key)]


def test_a_pending_grant_on_a_locked_folder_with_its_own_entry_still_refuses(host, tmp_path):
    # An entry for the group is there and cannot be taken back: the state is still unknown.
    root = _store_python(host, tmp_path)
    key = os.path.normcase(root)
    host.explicit.add(key)
    mxc_read_grants._save_record(
        {key: {"state": "pending", "identity": mxc_read_grants._identity(root)}}
    )
    with pytest.raises(mxc_read_grants.ReadGrantError):
        mxc_read_grants.ensure([root])
    assert _states() == {key: "pending"}


def test_a_grant_interrupted_by_a_crash_is_redone(host):
    venv = _runtime(host)
    host.granted.add(os.path.normcase(venv))  # the root ACE landed, the propagation may not have
    host.explicit.add(os.path.normcase(venv))
    identity = mxc_read_grants._identity(venv)
    mxc_read_grants._save_record(
        {os.path.normcase(venv): {"state": "pending", "identity": identity}}
    )
    assert mxc_read_grants.ensure([venv]) == (venv,)
    assert host.calls == [("grant", venv)]
    assert _states() == {os.path.normcase(venv): "complete"}


def test_an_unwritable_record_keeps_the_per_launch_grant(host, monkeypatch):
    venv = _runtime(host)

    def refuse(_grants):
        raise PermissionError("read-only")

    monkeypatch.setattr(mxc_read_grants, "_save_record", refuse)
    assert mxc_read_grants.ensure([venv]) == ()
    assert host.calls == []


def test_opting_out_revokes_every_recorded_grant(host, monkeypatch):
    venv = _runtime(host)
    mxc_read_grants.ensure([venv])
    monkeypatch.setenv(mxc_read_grants.PERSISTENT_GRANTS_ENV, "0")
    assert not mxc_read_grants.enabled()
    assert mxc_read_grants.ensure([venv]) == ()
    assert host.calls == [("grant", venv), ("revoke", os.path.normcase(venv))]
    assert _record() == {}


def test_revocation_never_follows_a_folder_that_was_replaced(host, tmp_path):
    venv = _runtime(host)
    mxc_read_grants.ensure([venv])
    os.replace(venv, tmp_path / "moved-away")
    other = tmp_path / "another-app"
    other.mkdir()
    try:
        os.symlink(other, venv, target_is_directory = True)
        linked = True
    except OSError:
        os.mkdir(venv)  # no symlinks here: a fresh folder at the same path has another identity
        linked = False
    assert mxc_read_grants.revoke_recorded() == ()
    assert ("revoke", os.path.normcase(venv)) not in host.calls
    # A link cannot be identified, so the entry waits; a different folder voids it.
    assert (_record() != {}) is linked


def test_an_unreadable_identity_keeps_the_record_for_a_later_revoke(host, monkeypatch):
    venv = _runtime(host)
    mxc_read_grants.ensure([venv])
    monkeypatch.setattr(mxc_read_grants, "_identity", lambda _root: None)
    assert mxc_read_grants.revoke_recorded() == ()
    assert _states() == {os.path.normcase(venv): "complete"}
    assert ("revoke", os.path.normcase(venv)) not in host.calls


def test_revocation_waits_for_a_running_workload_and_its_release_finishes_it(host, monkeypatch):
    # On Windows a lease held open cannot be deleted; here an existing file stands in for that.
    monkeypatch.setattr(mxc_read_grants, "_lease_is_live", lambda path: path.exists())
    venv = _runtime(host)
    mxc_read_grants.ensure([venv])
    lease = mxc_read_grants.hold()
    monkeypatch.setenv(mxc_read_grants.PERSISTENT_GRANTS_ENV, "0")
    assert mxc_read_grants.revoke_recorded() == ()
    assert mxc_read_grants.ensure([venv]) == ()
    assert ("revoke", os.path.normcase(venv)) not in host.calls
    assert _states() == {os.path.normcase(venv): "complete"}
    lease.release()
    assert host.calls[-1] == ("revoke", os.path.normcase(venv))
    assert _record() == {}
    lease.release()
    assert host.calls.count(("revoke", os.path.normcase(venv))) == 1


def test_release_keeps_the_grants_while_they_are_still_on(host, monkeypatch):
    from core.inference import mxc_policy

    monkeypatch.setattr(mxc_policy, "dacl_fallback_enabled", lambda: True)
    venv = _runtime(host)
    mxc_read_grants.ensure([venv])
    mxc_read_grants.hold().release()
    assert host.calls == [("grant", venv)]
    assert _states() == {os.path.normcase(venv): "complete"}


def test_a_lease_left_by_a_crashed_process_never_blocks_revocation(host, monkeypatch):
    venv = _runtime(host)
    mxc_read_grants.ensure([venv])
    stale = mxc_read_grants._leases_dir() / "4242-crashed"
    stale.parent.mkdir(parents = True, exist_ok = True)
    stale.write_text("")
    monkeypatch.setenv(mxc_read_grants.PERSISTENT_GRANTS_ENV, "0")
    assert mxc_read_grants.revoke_recorded() == (os.path.normcase(venv),)
    assert not stale.exists()


def test_a_lease_waits_for_a_revocation_already_in_progress(host):
    import threading

    leases = []
    with mxc_read_grants._transaction():
        worker = threading.Thread(target = lambda: leases.append(mxc_read_grants.hold()))
        worker.start()
        worker.join(0.5)
        assert worker.is_alive() and not leases
    worker.join(10)
    assert leases and leases[0] is not None
    leases[0].release()


def test_release_reads_the_switches_fresh_after_another_process_opts_out(host, monkeypatch):
    from core.inference import mxc_policy
    from storage import studio_db
    from utils import account_context, mxc_isolation_settings as saved

    store = {saved.DACL_SETTING_KEY: True, saved.GRANTS_SETTING_KEY: True}
    monkeypatch.delenv(mxc_policy.DACL_FALLBACK_ENV, raising = False)
    monkeypatch.setattr(studio_db, "get_app_settings", lambda keys: {k: store[k] for k in keys})
    monkeypatch.setattr(account_context, "run_as", lambda _who, fn, *a, **k: fn(*a, **k))
    monkeypatch.setattr(mxc_read_grants, "_lease_is_live", lambda path: path.exists())
    saved.forget_cached_setting()
    venv = _runtime(host)
    lease = mxc_read_grants.hold_if_needed()
    assert lease is not None and mxc_read_grants.ensure([venv]) == (venv,)
    assert saved.persistent_grants_setting()  # cached "on" in this process
    store[saved.GRANTS_SETTING_KEY] = False  # another Studio process turns it off, no local write
    lease.release()
    assert host.calls[-1] == ("revoke", os.path.normcase(venv))
    assert _record() == {}
    saved.forget_cached_setting()


def test_a_lease_that_cannot_be_recorded_refuses_instead_of_running_unguarded(host):
    blocker = mxc_read_grants._leases_dir()
    blocker.parent.mkdir(parents = True, exist_ok = True)
    blocker.write_text("")  # a file where the lease folder belongs
    with pytest.raises(mxc_read_grants.ReadGrantError, match = "could not record"):
        mxc_read_grants.hold()


@pytest.mark.parametrize("dacl, grants", [(False, True), (True, False)])
def test_a_lease_that_cannot_be_recorded_only_blocks_a_launch_that_needs_the_grants(
    host, monkeypatch, dacl, grants
):
    from core.inference import mxc_policy

    def refuse():
        raise mxc_read_grants.ReadGrantError("could not record")

    monkeypatch.setattr(mxc_read_grants, "hold", refuse)
    monkeypatch.setattr(mxc_policy, "dacl_fallback_enabled", lambda: dacl)
    monkeypatch.setattr(mxc_read_grants, "enabled", lambda: grants)
    assert mxc_read_grants.hold_if_needed() is None
    monkeypatch.setattr(mxc_policy, "dacl_fallback_enabled", lambda: True)
    monkeypatch.setattr(mxc_read_grants, "enabled", lambda: True)
    with pytest.raises(mxc_read_grants.ReadGrantError):
        mxc_read_grants.hold_if_needed()


def test_a_launch_during_a_deferred_revocation_keeps_the_grants_until_it_exits(host, monkeypatch):
    from core.inference import mxc_policy

    switch = {"on": True}
    monkeypatch.setattr(mxc_policy, "dacl_fallback_enabled", lambda: True)
    monkeypatch.setattr(mxc_read_grants, "enabled", lambda: switch["on"])
    monkeypatch.setattr(mxc_read_grants, "_lease_is_live", lambda path: path.exists())
    venv = _runtime(host)
    first = mxc_read_grants.hold_if_needed()
    mxc_read_grants.ensure([venv])
    switch["on"] = False  # the owner turns the grants off while the first call runs
    assert mxc_read_grants.revoke_recorded() == ()
    second = mxc_read_grants.hold_if_needed()  # starts while the entries are still in place
    assert second is not None
    first.release()
    assert ("revoke", os.path.normcase(venv)) not in host.calls
    second.release()
    assert host.calls[-1] == ("revoke", os.path.normcase(venv))


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows refuses to delete a file held open")
def test_an_open_lease_is_live_until_released(host):
    lease = mxc_read_grants.hold()
    assert mxc_read_grants._live_leases() == 1
    lease.release()
    assert mxc_read_grants._live_leases() == 0


def test_a_replaced_folder_is_not_adopted_through_a_stale_record(host, tmp_path):
    venv = _runtime(host)
    mxc_read_grants.ensure([venv])
    os.replace(venv, tmp_path / "moved-away")
    _runtime(host)  # a new folder at the same path, carrying its owner's own entry for the group
    host.explicit.add(os.path.normcase(venv))
    host.granted.discard(os.path.normcase(venv))
    assert mxc_read_grants.ensure([venv]) == ()
    assert host.calls == [("grant", venv)]
    assert _record() == {}


def test_a_folder_replaced_at_a_scanned_path_is_scanned_again(host, tmp_path):
    venv = _runtime(host)
    assert mxc_read_grants.ensure([venv]) == (venv,)
    os.replace(venv, tmp_path / "moved-away")
    _runtime(host)
    Path(venv, "Lib", "site-packages", "six", ".env").write_text("TOKEN=secret\n")
    host.granted.discard(os.path.normcase(venv))
    host.explicit.discard(os.path.normcase(venv))
    assert mxc_read_grants.ensure([venv]) == ()
    assert host.calls == [("grant", venv)]


@pytest.mark.parametrize("unknown", ["aces", "identity"])
def test_an_unfinished_grant_that_cannot_be_checked_refuses_the_launch(host, monkeypatch, unknown):
    venv = _runtime(host)
    identity = mxc_read_grants._identity(venv)
    mxc_read_grants._save_record(
        {os.path.normcase(venv): {"state": "pending", "identity": identity}}
    )
    if unknown == "aces":

        def unreadable(_root):
            raise OSError(5, "Access is denied")

        monkeypatch.setattr(mxc_read_grants, "_package_aces", unreadable)
    else:
        monkeypatch.setattr(mxc_read_grants, "_identity", lambda _root: None)
    with pytest.raises(mxc_read_grants.ReadGrantError, match = "cannot be checked"):
        mxc_read_grants.ensure([venv])
    assert _states() == {os.path.normcase(venv): "pending"}


def test_a_record_write_failure_after_a_clean_rollback_does_not_block_the_launch(host, monkeypatch):
    venv = _runtime(host)
    host.fail_grant = True
    real_save = mxc_read_grants._save_record
    writes = []

    def save_then_fail(grants):
        writes.append(dict(grants))
        if len(writes) > 1:
            raise PermissionError("disk went read-only")
        real_save(grants)

    monkeypatch.setattr(mxc_read_grants, "_save_record", save_then_fail)
    assert mxc_read_grants.ensure([venv]) == ()
    assert host.calls == [("grant", venv), ("revoke", venv)]


def test_another_process_mid_grant_refuses_the_launch_but_not_the_cleanup(host, monkeypatch):
    from filelock import FileLock

    venv = _runtime(host)
    mxc_read_grants.ensure([venv])
    monkeypatch.setattr(mxc_read_grants, "LOCK_TIMEOUT_SECONDS", 0.2)
    holder = FileLock(str(mxc_read_grants.record_path()) + ".lock")
    holder.acquire()
    try:
        # Held by "another process": this launch cannot know whether that grant finished.
        mxc_read_grants._scanned.clear()
        host.granted.discard(os.path.normcase(venv))
        monkeypatch.setattr(mxc_read_grants, "_lock", __import__("threading").Lock())
        with pytest.raises(mxc_read_grants.ReadGrantError, match = "another Studio process"):
            _in_thread(lambda: mxc_read_grants.ensure([venv]))
        assert _in_thread(mxc_read_grants.revoke_recorded) == ()
    finally:
        holder.release()


def _in_thread(function):
    """filelock is reentrant within a thread, so a second holder must run on another thread."""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        return pool.submit(function).result()


def test_off_windows_nothing_is_touched(host, monkeypatch):
    monkeypatch.setattr(mxc_read_grants, "_on_windows", lambda: False)
    assert mxc_read_grants.ensure([_runtime(host)]) == ()
    assert mxc_read_grants.revoke_recorded() == ()
    assert host.calls == []


def _grant_in_another_process(tmp_path, name, queue):
    monkeypatch = pytest.MonkeyPatch()
    studio_home = _isolate(monkeypatch, Path(tmp_path))
    monkeypatch.setattr(mxc_read_grants, "_package_aces", lambda _root: (False, False))
    monkeypatch.setattr(mxc_read_grants, "_grant", lambda _root: (True, "ok"))
    root = studio_home / name
    root.mkdir()
    queue.put(mxc_read_grants.ensure([str(root)]))


def test_concurrent_studio_processes_keep_every_record(tmp_path):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    names = [f"runtime-{index}" for index in range(4)]
    workers = [
        context.Process(target = _grant_in_another_process, args = (str(tmp_path), name, queue))
        for name in names
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(120)
    assert [worker.exitcode for worker in workers] == [0] * len(names)
    assert all(queue.get(timeout = 5) for _ in names)
    record = (
        tmp_path / "home" / ".unsloth" / "studio" / "mxc-runtime" / "persistent-read-grants.json"
    )
    assert sorted(Path(key).name for key in json.loads(record.read_text())["grants"]) == names


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


def _policy_on_linux(monkeypatch):
    from core.inference import mxc_policy

    outside = Path(os.path.abspath(os.sep)) / "unsloth-test-dacl-journal-never-created"
    monkeypatch.setattr(mxc_runtime, "dacl_state_path", lambda: outside)
    monkeypatch.setattr(mxc_policy.sys, "platform", "win32")
    return mxc_policy


@pytest.mark.parametrize("dacl", [True, False])
def test_launch_policy_grants_runtime_roots_only_on_the_dacl_tier(monkeypatch, tmp_path, dacl):
    mxc_policy = _policy_on_linux(monkeypatch)
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
    else:
        assert calls == [("revoke", None)]


def test_a_grant_left_in_an_unknown_state_refuses_the_launch(monkeypatch, tmp_path):
    mxc_policy = _policy_on_linux(monkeypatch)
    monkeypatch.setenv(mxc_policy.DACL_FALLBACK_ENV, "1")

    def unknown(_roots):
        raise mxc_read_grants.ReadGrantError("rollback failed")

    monkeypatch.setattr(mxc_read_grants, "ensure", unknown)
    with pytest.raises(mxc_read_grants.ReadGrantError):
        mxc_policy.build_launch_request(_python_plan(tmp_path))


def test_capability_names_the_persistent_grant_and_cleans_up_when_off(monkeypatch):
    from core.inference import mxc_policy, sandbox_windows_mxc

    monkeypatch.setattr(sandbox_windows_mxc.sys, "platform", "win32")
    monkeypatch.setattr(
        sandbox_windows_mxc.mxc_probe, "probe", lambda *_a, **_k: (False, "unavailable")
    )
    monkeypatch.setattr(sandbox_windows_mxc.mxc_probe, "host_prep_remediation", lambda: None)
    monkeypatch.setattr(sandbox_windows_mxc.mxc_runtime, "installation_identity", lambda: "id")
    revoked = []
    monkeypatch.setattr(mxc_read_grants, "revoke_recorded", lambda: revoked.append(True) or ())
    monkeypatch.setenv(mxc_policy.DACL_FALLBACK_ENV, "1")
    monkeypatch.delenv(mxc_read_grants.PERSISTENT_GRANTS_ENV, raising = False)
    snapshot = sandbox_windows_mxc.capability_snapshot
    on = snapshot(execution_kind = "python", selected_executable = sys.executable)
    assert "mxc_tier3_persistent_runtime_read_grants" in on.limitations
    assert revoked == []
    monkeypatch.setenv(mxc_read_grants.PERSISTENT_GRANTS_ENV, "0")
    off = snapshot(execution_kind = "python", selected_executable = sys.executable)
    assert "mxc_tier3_persistent_runtime_read_grants" not in off.limitations
    assert revoked == [True]
    monkeypatch.delenv(mxc_read_grants.PERSISTENT_GRANTS_ENV)
    monkeypatch.delenv(mxc_policy.DACL_FALLBACK_ENV)
    # Leaving the DACL tier on a host with nothing else: the capability is unavailable, no launch is
    # ever built, and the grants still come back.
    default = snapshot(execution_kind = "python", selected_executable = sys.executable)
    assert "mxc_tier3_persistent_runtime_read_grants" not in default.limitations
    assert mxc_read_grants.PERSISTENT_GRANTS_ENV in default.remediation
    assert revoked == [True, True]


@pytest.mark.skipif(sys.platform != "win32", reason = "reads a real Windows DACL")
def test_real_dacl_check_sees_a_granted_folder(tmp_path):
    import subprocess

    folder = tmp_path / "runtime"
    folder.mkdir()
    grant = ["icacls", str(folder), "/grant", "*S-1-15-2-1:(OI)(CI)(RX)", "/Q"]
    assert mxc_read_grants._package_aces(str(folder)) == (False, False)
    subprocess.run(grant, check = True, capture_output = True)
    assert mxc_read_grants._package_aces(str(folder)) == (True, True)
    child = folder / "child"
    child.mkdir()
    # Inherited, not explicit: a child of a granted folder is covered but was never set by Studio.
    assert mxc_read_grants._package_aces(str(child)) == (True, False)
    subprocess.run(
        ["icacls", str(folder), "/remove:g", "*S-1-15-2-1", "/Q"], check = True, capture_output = True
    )
    assert mxc_read_grants._package_aces(str(folder)) == (False, False)


@pytest.mark.parametrize(
    "aces, expected",
    [
        ([(0x1200A9, 0), (0xA0000000, 0x0B)], True),
        ([(0x1200A9, 0x03)], True),
        ([(0x10000000, 0x03)], True),
        ([(0x1200A9, 0)], False),
        ([(0xA0000000, 0x0B)], False),
        ([(0x1200A9, 0), (0x80000000, 0x0B)], False),
        ([(0x1200A9, 0), (0xA0000000, 0x09)], False),
        ([(0x1200A9, 0), (0xA0000000, 0x0A)], False),
        ([(0x1200A9, 0), (0xA0000000, 0x0F)], False),
    ],
)
def test_split_inherited_package_read_access(aces, expected):
    assert mxc_read_grants._read_execute_covered(aces) is expected


@pytest.mark.skipif(sys.platform != "win32", reason = "reads a real Windows DACL")
def test_real_dacl_split_folder_and_child_grants(tmp_path):
    import subprocess

    folder = tmp_path / "split-runtime"
    folder.mkdir()
    subprocess.run(
        [
            "icacls",
            str(folder),
            "/grant",
            "*S-1-15-2-1:(RX)",
            "*S-1-15-2-1:(OI)(CI)(IO)(GR,GE)",
            "/Q",
        ],
        check = True,
        capture_output = True,
    )
    assert mxc_read_grants._package_aces(str(folder)) == (True, True)
    child = folder / "child"
    child.mkdir()
    assert mxc_read_grants._package_aces(str(child)) == (True, False)


def test_failed_grant_with_only_inherited_access_recovers_without_acl_edits(host):
    root = _runtime(host)
    key = os.path.normcase(root)
    host.granted.add(key)
    mxc_read_grants._save_record(
        {
            key: {
                "state": "pending",
                "identity": mxc_read_grants._identity(root),
            }
        }
    )
    assert mxc_read_grants.ensure([root]) == (root,)
    assert _record() == {}
    assert host.calls == []


def test_opt_out_drops_proven_empty_pending_grant_without_revoking_windows_access(host):
    root = _runtime(host)
    key = os.path.normcase(root)
    host.granted.add(key)
    mxc_read_grants._save_record(
        {
            key: {
                "state": "pending",
                "identity": mxc_read_grants._identity(root),
            }
        }
    )
    mxc_read_grants.revoke_recorded()
    assert _record() == {}
    assert key in host.granted
    assert host.calls == []


@pytest.mark.parametrize("obstacle", ["explicit_child", "unreadable_child", "changed_identity"])
def test_pending_recovery_retains_uncertain_or_partial_grants(host, monkeypatch, obstacle):
    root = _runtime(host)
    key = os.path.normcase(root)
    host.granted.add(key)
    identity = mxc_read_grants._identity(root)
    mxc_read_grants._save_record({key: {"state": "pending", "identity": identity}})
    child = os.path.normcase(str(Path(root) / "Lib"))
    if obstacle == "explicit_child":
        host.explicit.add(child)
    elif obstacle == "unreadable_child":

        def aces(path):
            if os.path.normcase(path) == child:
                raise OSError("cannot inspect child ACL")
            return host.aces(path)

        monkeypatch.setattr(mxc_read_grants, "_package_aces", aces)
    else:
        monkeypatch.setattr(mxc_read_grants, "_identity", lambda _path: {"fileId": -1})
    assert not mxc_read_grants._pending_grant_has_no_explicit_aces(root, identity)
    assert _states() == {key: "pending"}
