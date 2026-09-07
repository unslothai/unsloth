# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Actual fixed-worker profile cleanup, not full Windows qualification."""

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from test_preparation import BACKEND, lpac
from test_reader_cleanup import observe_worker
from core.inference.windows_sandbox import identity, preparation
from core.inference.windows_sandbox.profiles import WindowsRuntimeError

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows profile cleanup")


@pytest.fixture
def profile_owner(tmp_path, monkeypatch):
    local = tmp_path / "LocalAppData"
    local.mkdir()
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    work = tmp_path / "work"
    work.mkdir()
    owner = identity.InvocationReservation(identity.InvocationRecipe.new())
    try:
        actual = owner.create(work)
        lpac._grant_modify(str(work), actual.sid)
        lpac._grant_modify(actual.private_temp, actual.sid)
        yield owner, work
    finally:
        owner.cleanup()


def test_worker_cleans_exact_profile_without_broker_filesystem_calls(
    profile_owner, monkeypatch, observe_worker
):
    owner, work = profile_owner
    other = identity.InvocationReservation(identity.InvocationRecipe.new())
    two = other.create(work)
    lpac._grant_modify(str(work), two.sid)
    actual, path = owner.identity, owner.path
    try:
        with monkeypatch.context() as context:

            def forbidden(*args, **kwargs):
                pytest.fail("Profile filesystem cleanup ran in the broker")

            context.setattr(identity, "_journal_root", forbidden)
            context.setattr(lpac._InvocationIdentity, "cleanup", forbidden)
            owner.cleanup(in_worker = True)
        assert owner.closed and actual.cleaned and not actual.sid
        assert not path.exists() and identity._profile_path(actual.sid_string) is None
        assert other.path.exists() and identity._profile_path(two.sid_string) is not None
        assert identity._read(other.path)["sid"] == two.sid_string
        owner.cleanup(in_worker = True)
    finally:
        other.cleanup()


@pytest.mark.parametrize("phase", ["acl", "temp", "delete_before", "delete_after", "unlink"])
def test_profile_cleanup_deadline_and_retry(
    profile_owner, tmp_path, monkeypatch, observe_worker, phase
):
    owner, _ = profile_owner
    original = preparation._run_worker
    wrapper = tmp_path / "fixed-profile-stall.py"
    wrapper.write_text(
        f"""
import sys, os
sys.path.insert(0, {str(BACKEND)!r})
from core.inference.windows_sandbox import native_compat as lpac
from core.inference.windows_sandbox import identity
from core.inference.windows_sandbox.profile_cleanup_worker import main
def pause(*args, **kwargs):
    r,w = os.pipe()
    os.read(r,1)
if {phase!r} == 'acl':
    lpac._revoke_sid = pause
elif {phase!r} == 'temp':
    lpac.shutil.rmtree = pause
elif {phase!r} in ('delete_before', 'delete_after'):
    original = lpac._api().userenv.DeleteAppContainerProfile
    def deleting(*args):
        if {phase!r} == 'delete_after':
            assert original(*args) == 0
        pause()
    lpac._api().userenv.DeleteAppContainerProfile = deleting
else:
    original = lpac.os.unlink
    def unlink(path, *args, **kwargs):
        if str(path).endswith('.json'):
            pause()
        return original(path, *args, **kwargs)
    lpac.os.unlink = unlink
raise SystemExit(main())
""",
        encoding = "utf-8",
    )

    def worker(argv, *args, **kwargs):
        assert argv[1:4] == ["-I", "-S", "-B"]
        assert argv[4].endswith("profile_cleanup_worker.py")
        return original([*argv[:4], str(wrapper), *argv[5:]], *args, **kwargs)

    actual = owner.identity
    with monkeypatch.context() as context:
        context.setattr(preparation, "_run_worker", worker)
        started = time.monotonic()
        with pytest.raises(WindowsRuntimeError, match = "TIMEOUT"):
            preparation.cleanup_invocation_profile(owner.recipe, str(owner.path), timeout = 0.75)
        assert time.monotonic() - started < 8
    assert owner.path.exists() and not actual.cleaned and actual.sid
    owner.cleanup(in_worker = True)
    assert owner.closed and actual.cleaned and not owner.path.exists()
    assert identity._profile_path(actual.sid_string) is None


@pytest.mark.parametrize("failed_tombstone", [False, True])
def test_bounded_collision_cleanup_never_deletes_existing_profile(
    tmp_path, monkeypatch, observe_worker, failed_tombstone
):
    local = tmp_path / "LocalAppData"
    local.mkdir()
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    work = tmp_path / "work"
    work.mkdir()
    owner = identity.InvocationReservation(identity.InvocationRecipe.new())
    api = lpac._api()
    original_create, original_write = api.userenv.CreateAppContainerProfile, identity._write
    conflicting = ctypes.c_void_p()

    def create(*args):
        assert original_create(*args[:-1], ctypes.byref(conflicting)) == 0
        return ctypes.c_int32(0x800700B7).value

    def write(path, value):
        if failed_tombstone and value["state"] == "collision":
            raise OSError("Injected collision tombstone failure")
        return original_write(path, value)

    try:
        with monkeypatch.context() as context:
            context.setattr(api.userenv, "CreateAppContainerProfile", create)
            context.setattr(identity, "_write", write)
            with pytest.raises((WindowsRuntimeError, OSError), match = "collision"):
                owner.create(work)
        sid = lpac._sid_string(api, conflicting)
        owner.cleanup(in_worker = True)
        assert owner.closed and not owner.path.exists()
        assert identity._profile_path(sid) is not None
    finally:
        owner.cleanup()
        if conflicting:
            assert api.userenv.DeleteAppContainerProfile(owner.recipe.moniker) == 0
            api.advapi32.FreeSid(conflicting)


@pytest.mark.parametrize("field", ["nonce", "pid", "error"])
def test_unconfirmed_profile_response_retains_local_identity(profile_owner, monkeypatch, field):
    owner, _ = profile_owner
    run = preparation._run_worker

    def altered(*args, **kwargs):
        output, pid = run(*args, **kwargs)
        result = json.loads(output)
        result[field] = {"nonce": "0" * 64, "pid": pid + 1, "error": []}[field]
        return json.dumps(result).encode(), pid

    actual = owner.identity
    with monkeypatch.context() as context:
        context.setattr(preparation, "_run_worker", altered)
        with pytest.raises(WindowsRuntimeError):
            owner.cleanup(in_worker = True)
    assert not owner.closed and not actual.cleaned and actual.sid
    assert not owner.path.exists()  # worker completed, but its response was rejected
    owner.cleanup(in_worker = True)
    assert owner.closed and actual.cleaned and not actual.sid


def test_missing_journal_cannot_report_existing_profile_cleaned(profile_owner, observe_worker):
    owner, _ = profile_owner
    contents = owner.path.read_bytes()
    owner.path.unlink()
    try:
        with pytest.raises(WindowsRuntimeError, match = "no durable ownership journal"):
            owner.cleanup(in_worker = True)
        assert not owner.closed and not owner.identity.cleaned
        assert identity._profile_path(owner.identity.sid_string) is not None
    finally:
        identity.native_files().create(owner.path, contents)
    owner.cleanup(in_worker = True)
    assert owner.closed and not owner.path.exists()


def test_different_cleanup_namespace_cannot_touch_profile(profile_owner, observe_worker):
    owner, _ = profile_owner
    wrong = owner.path.parent.parent / owner.path.name
    with pytest.raises(WindowsRuntimeError, match = "directory changed"):
        preparation.cleanup_invocation_profile(owner.recipe, str(wrong))
    assert owner.path.exists() and not owner.closed
    assert identity._profile_path(owner.identity.sid_string) is not None


def test_broker_sid_release_failure_retains_exact_allocation(
    profile_owner, monkeypatch, observe_worker
):
    owner, _ = profile_owner
    actual = owner.identity
    pointer = actual.sid.value
    api = lpac._api().advapi32
    original = api.FreeSid

    def fail_owned_sid(sid):
        return pointer if sid.value == pointer else original(sid)

    with monkeypatch.context() as context:
        context.setattr(api, "FreeSid", fail_owned_sid)
        with pytest.raises(WindowsRuntimeError, match = "SID allocation"):
            owner.cleanup(in_worker = True)
    assert not owner.closed and not actual.cleaned and actual.sid.value == pointer
    assert not owner.path.exists()
    owner.cleanup(in_worker = True)
    assert owner.closed and actual.cleaned and not actual.sid


def test_creating_cleanup_accepts_confirmed_already_deleted_profile(
    tmp_path, monkeypatch, observe_worker, request
):
    local = tmp_path / "LocalAppData"
    local.mkdir()
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    work = tmp_path / "work"
    work.mkdir()
    owner = identity.InvocationReservation(identity.InvocationRecipe.new())
    request.addfinalizer(owner.cleanup)
    api, create = lpac._api(), lpac._api().userenv.CreateAppContainerProfile

    def interrupted_create(*args):
        assert create(*args) == 0
        raise OSError("Injected interruption after profile creation")

    with monkeypatch.context() as context:
        context.setattr(api.userenv, "CreateAppContainerProfile", interrupted_create)
        with pytest.raises(OSError, match = "interruption"):
            owner.create(work)
    sid = lpac._sid_string(api, owner.created_sid)
    assert identity._read(owner.path)["state"] == "creating"
    assert api.userenv.DeleteAppContainerProfile(owner.recipe.moniker) == 0
    assert identity._profile_path(sid) is None
    wrapper = tmp_path / "fixed-already-deleted-profile.py"
    wrapper.write_text(
        f"""
import sys
sys.path.insert(0, {str(BACKEND)!r})
from core.inference.windows_sandbox import native_compat as lpac
from core.inference.windows_sandbox.profile_cleanup_worker import main
lpac._api().userenv.DeleteAppContainerProfile = lambda name: -2147024894
raise SystemExit(main())
""",
        encoding = "utf-8",
    )
    run = preparation._run_worker

    def worker(argv, *args, **kwargs):
        assert argv[4].endswith("profile_cleanup_worker.py")
        return run([*argv[:4], str(wrapper), *argv[5:]], *args, **kwargs)

    try:
        with monkeypatch.context() as context:
            context.setattr(preparation, "_run_worker", worker)
            owner.cleanup(in_worker = True)
        assert owner.closed and not owner.path.exists() and not owner.created_sid
    finally:
        owner.cleanup()


def test_parent_death_kills_profile_worker_and_leaves_recoverable_journal(tmp_path, monkeypatch):
    local = tmp_path / "LocalAppData"
    local.mkdir()
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    work = tmp_path / "work"
    work.mkdir()
    marker, record = tmp_path / "worker.json", tmp_path / "recipe.json"
    worker = tmp_path / "fixed-paused-profile.py"
    worker.write_text(
        f"""
import sys, os, json
from pathlib import Path
sys.path.insert(0, {str(BACKEND)!r})
from core.inference.windows_sandbox import native_compat as lpac
from core.inference.windows_sandbox.profile_cleanup_worker import main
def pause(*args):
    Path({str(marker)!r}).write_text(json.dumps({{'pid':os.getpid()}}), encoding='utf-8')
    r,w = os.pipe()
    os.read(r,1)
lpac._api().userenv.DeleteAppContainerProfile = pause
raise SystemExit(main())
""",
        encoding = "utf-8",
    )
    source = f"""
import sys, json
from pathlib import Path
sys.path.insert(0, {str(BACKEND)!r})
from core.inference.windows_sandbox import identity, preparation
owner = identity.InvocationReservation(identity.InvocationRecipe.new())
actual = owner.create({str(work)!r})
Path({str(record)!r}).write_text(json.dumps({{'recipe':vars(owner.recipe),'path':str(owner.path),'sid':actual.sid_string}}), encoding='utf-8')
run = preparation._run_worker
def fixed_worker(argv, *args, **kwargs):
    assert argv[4].endswith('profile_cleanup_worker.py')
    return run([*argv[:4], {str(worker)!r}, *argv[5:]], *args, **kwargs)
preparation._run_worker = fixed_worker
owner.cleanup(in_worker=True)
"""
    parent = subprocess.Popen(
        [sys._base_executable, "-I", "-S", "-B", "-c", source],
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
    )
    handle = None
    kernel = lpac._api().kernel32
    try:
        deadline = time.monotonic() + 15
        while not marker.exists():
            assert parent.poll() is None, parent.communicate(timeout = 5)
            assert time.monotonic() < deadline, "Profile worker did not enter the controlled stall"
            time.sleep(0.01)
        pid = json.loads(marker.read_text(encoding = "utf-8"))["pid"]
        handle = kernel.OpenProcess(0x100001, False, pid)
        assert handle and kernel.WaitForSingleObject(handle, 0) == 258
        parent.kill()
        parent.communicate(timeout = 5)
        assert (
            kernel.WaitForSingleObject(handle, 5000) == 0
        ), "Parent death left the cleanup worker alive"
        value = json.loads(record.read_text(encoding = "utf-8"))
        assert Path(value["path"]).exists()
        assert identity._profile_path(value["sid"]) is not None
        identity.recover_identities()
        assert not Path(value["path"]).exists()
        assert identity._profile_path(value["sid"]) is None
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.communicate(timeout = 5)
        if handle:
            try:
                if kernel.WaitForSingleObject(handle, 0) != 0:
                    assert kernel.TerminateProcess(handle, 1)
                    assert kernel.WaitForSingleObject(handle, 5000) == 0
            finally:
                assert kernel.CloseHandle(handle)
        identity.recover_identities()
