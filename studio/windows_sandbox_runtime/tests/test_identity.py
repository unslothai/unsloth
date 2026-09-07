# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Real profile/journal recovery, not complete backend qualification."""

import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import time

import pytest

from test_preparation import BACKEND, lpac
from test_launch import LAUNCH, run_harness, installed_runtime, runtime_wheel
from core.inference.windows_sandbox import identity
from core.inference.windows_sandbox.content_files import PathLease, native_files

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows profile recovery")


@pytest.fixture
def journal(tmp_path, monkeypatch):
    base = tmp_path / "journals"
    work = tmp_path / "work"
    base.mkdir()
    work.mkdir()
    monkeypatch.setattr(lpac, "_manifest_root", lambda: str(base))
    return base, work


@pytest.fixture
def reservation(journal):
    owner = identity.InvocationReservation(identity.InvocationRecipe.new())
    yield owner
    owner.cleanup()


def test_private_intent_precedes_profile_creation_and_ready_precedes_grants(
    journal, reservation, monkeypatch
):
    base, work = journal
    api, original = lpac._api(), lpac._api().userenv.CreateAppContainerProfile
    records = []

    def create(*args):
        record = identity._read(reservation.path)
        assert record["state"] == "creating"
        assert record["moniker"] == args[0] == reservation.recipe.moniker
        assert record["owner_pid"] == os.getpid()
        assert record["owner_created"] == lpac._process_identity()[1]
        assert identity._profile_path(record["sid"]) is None
        with PathLease() as pins:
            native_files().require_private(pins.file(reservation.path))
            native_files().require_private(pins.directory(reservation.path.parent))
        records.append(record)
        return original(*args)

    monkeypatch.setattr(api.userenv, "CreateAppContainerProfile", create)
    actual = reservation.create(work)
    ready = identity._read(reservation.path)
    assert len(records) == 1 and ready["state"] == "ready"
    assert ready["sid"] == actual.sid_string
    assert (actual.owner_pid, actual.owner_created) == lpac._process_identity()
    assert identity._profile_path(actual.sid_string) == actual.profile_folder
    assert not Path(str(reservation.path) + ".tmp").exists()
    identity.recover_identities()  # alive broker must not be considered stale
    assert reservation.path.exists() and not actual.cleaned
    reservation.cleanup()
    assert actual.cleaned and not reservation.path.exists()
    assert identity._profile_path(actual.sid_string) is None
    assert not list((base / "python-bootstrap-v2").iterdir())


def test_preexisting_profile_is_never_adopted_or_deleted(journal, reservation, monkeypatch):
    _, work = journal
    existing = reservation.create(work)
    contender = identity.InvocationReservation(reservation.recipe)
    api, original = lpac._api(), lpac._api().userenv.DeleteAppContainerProfile
    removed = []

    def deleting(name):
        removed.append(name)
        return original(name)

    monkeypatch.setattr(api.userenv, "DeleteAppContainerProfile", deleting)
    with pytest.raises(identity.WindowsRuntimeError, match = "already exists"):
        contender.create(work)
    contender.cleanup()
    assert not removed
    assert identity._profile_path(existing.sid_string) == existing.profile_folder
    assert reservation.path.exists()


@pytest.mark.parametrize(
    "phase", ["before_create", "after_create", "ready", "grant", "partial_intent"]
)
def test_parent_death_recovers_exact_profile_and_partial_journal(journal, tmp_path, phase):
    base, work = journal
    marker = tmp_path / "paused"
    source = f"""
import sys, os, json
from pathlib import Path
sys.path.insert(0, {str(BACKEND)!r})
sys.path.append({sysconfig.get_path("purelib")!r})
from core.inference.windows_sandbox import native_compat as lpac
from core.inference.windows_sandbox import identity
lpac._manifest_root = lambda: {str(base)!r}
owner = identity.InvocationReservation(identity.InvocationRecipe.new())
original_create = lpac._api().userenv.CreateAppContainerProfile
original_write = identity._write

def pause():
    Path({str(marker)!r}).write_text(json.dumps({{
        'recipe': vars(owner.recipe), 'path': str(owner.path), 'pid': os.getpid()
    }}), encoding='utf-8')
    r, w = os.pipe()
    os.read(r, 1)

def create(*args):
    if {phase!r} == 'before_create':
        pause()
    result = original_create(*args)
    assert result == 0
    if {phase!r} == 'after_create':
        pause()
    return result

def write(path, value):
    if {phase!r} == 'partial_intent':
        identity.native_files().create(Path(str(path)+'.tmp'), b'{{"version":')
        pause()
    return original_write(path, value)

lpac._api().userenv.CreateAppContainerProfile = create
identity._write = write
actual = owner.create({str(work)!r})
if {phase!r} == 'grant':
    lpac._grant_modify({str(work)!r}, actual.sid)
pause()
raise AssertionError('Paused preparation unexpectedly resumed')
"""
    parent = subprocess.Popen(
        [sys._base_executable, "-I", "-S", "-c", source],
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
    )
    observed = None
    recipe = None
    try:
        deadline = time.monotonic() + 15
        while not marker.exists() and parent.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists(), parent.communicate(timeout = 5)
        value = json.loads(marker.read_text())
        recipe = identity.InvocationRecipe(**value["recipe"])
        assert value["pid"] == parent.pid == recipe.owner_pid
        path = Path(value["path"])
        observed = lpac._api().kernel32.OpenProcess(0x100000, False, parent.pid)
        assert observed and lpac._api().kernel32.WaitForSingleObject(observed, 0) == 258
        identity.recover_identities()
        assert path.exists() or Path(str(path) + ".tmp").exists()
        parent.kill()
        parent.communicate(timeout = 5)
        assert lpac._api().kernel32.WaitForSingleObject(observed, 5000) == 0
        with identity._derived_sid(recipe.moniker) as (_, sid):
            assert (identity._profile_path(sid) is not None) == (
                phase in ("after_create", "ready", "grant")
            )
            with PathLease() as pins:
                before = native_files().security_text(pins.directory(work))
            assert (sid in before) == (phase == "grant")
            identity.recover_identities()
            assert identity._profile_path(sid) is None
            with PathLease() as pins:
                after = native_files().security_text(pins.directory(work))
            assert sid not in after
        assert not list((base / "python-bootstrap-v2").iterdir())
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.communicate(timeout = 5)
        if observed:
            assert lpac._api().kernel32.CloseHandle(observed)
        if recipe is not None:
            # Retry only this test's private journal after the exact owner died.
            identity.recover_identities()


def test_profile_delete_failure_retains_creating_journal(journal, reservation, monkeypatch):
    _, work = journal
    original_write = identity._write

    def fail_ready(path, value):
        if value["state"] == "ready":
            raise OSError("injected ready journal failure")
        return original_write(path, value)

    monkeypatch.setattr(identity, "_write", fail_ready)
    with pytest.raises(OSError, match = "ready journal"):
        reservation.create(work)
    path = reservation.path
    assert identity._read(path)["state"] == "creating"
    original_delete = lpac._api().userenv.DeleteAppContainerProfile
    monkeypatch.setattr(lpac._api().userenv, "DeleteAppContainerProfile", lambda _: -2147024891)
    with pytest.raises(OSError, match = "DeleteAppContainerProfile"):
        reservation.cleanup()
    assert path.exists() and not reservation.closed
    monkeypatch.setattr(lpac._api().userenv, "DeleteAppContainerProfile", original_delete)
    reservation.cleanup()
    assert reservation.closed and not path.exists()


def test_completed_profile_deletion_before_journal_unlink_is_recoverable(
    journal, reservation, monkeypatch
):
    _, work = journal
    actual = reservation.create(work)
    path = reservation.path
    original_unlink = lpac.os.unlink

    def fail(pathname, *args, **kwargs):
        if str(pathname) == str(path):
            raise PermissionError("injected journal unlink failure")
        return original_unlink(pathname, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(lpac.os, "unlink", fail)
        with pytest.raises(OSError, match = "manifest"):
            reservation.cleanup()
    assert path.exists() and identity._profile_path(actual.sid_string) is None
    # Recover independently, as a later broker would, not through the old object.
    identity._recover(path, reservation.recipe)
    assert not path.exists()
    reservation.cleanup()


@pytest.mark.parametrize("change", ["sid", "pid", "created", "state", "workdir", "profile"])
def test_malformed_record_cannot_authorize_profile_cleanup(
    journal, reservation, monkeypatch, change
):
    _, work = journal
    actual = reservation.create(work)
    original = identity._read(reservation.path)
    altered = dict(original)
    key, value = {
        "sid": ("sid", "S-1-15-2-1"),
        "pid": ("owner_pid", True),
        "created": ("owner_created", original["owner_created"] + 1),
        "state": ("state", "anything"),
        "workdir": ("workdir", "C:\\"),
        "profile": ("profile_folder", str(work)),
    }[change]
    altered[key] = value
    identity._write(reservation.path, altered)
    try:
        with pytest.raises(identity.WindowsRuntimeError):
            identity._recover(reservation.path, reservation.recipe)
        assert identity._profile_path(actual.sid_string) == actual.profile_folder
    finally:
        identity._write(reservation.path, original)


def test_owner_query_failure_preserves_live_profile(journal, reservation, monkeypatch):
    _, work = journal
    actual = reservation.create(work)

    def fail(*args):
        ctypes.set_last_error(5)
        return False

    with monkeypatch.context() as context:
        context.setattr(lpac._api().kernel32, "GetProcessTimes", fail)
        with pytest.raises(OSError, match = "GetProcessTimes"):
            identity.recover_identities()
    assert reservation.path.exists()
    assert identity._profile_path(actual.sid_string) == actual.profile_folder


def test_reused_pid_with_different_creation_identity_is_not_a_live_owner(
    journal, reservation, monkeypatch
):
    _, work = journal
    actual = reservation.create(work)
    original = lpac._process_identity
    recipe = reservation.recipe

    def reused(pid = None):
        if pid == recipe.owner_pid:
            return pid, recipe.owner_created + 1
        return original(pid)

    with monkeypatch.context() as context:
        context.setattr(lpac, "_process_identity", reused)
        identity.recover_identities()
    assert not reservation.path.exists()
    assert identity._profile_path(actual.sid_string) is None


def test_worker_exit_does_not_make_its_live_broker_identity_stale(journal, reservation):
    base, work = journal
    recipe = reservation.recipe
    source = f"""
import sys, json
sys.path.insert(0, {str(BACKEND)!r})
sys.path.append({sysconfig.get_path("purelib")!r})
from core.inference.windows_sandbox import native_compat as lpac
from core.inference.windows_sandbox import identity
lpac._manifest_root = lambda: {str(base)!r}
owner = identity.InvocationReservation(identity.InvocationRecipe(
    {recipe.moniker!r}, {recipe.owner_pid}, {recipe.owner_created}))
actual = owner.create({str(work)!r})
print(json.dumps({{'path': str(owner.path), 'sid': actual.sid_string}}), flush=True)
# Exit releases native SID allocations, not the handed-off profile.
"""
    result = subprocess.run(
        [sys._base_executable, "-I", "-S", "-c", source],
        capture_output = True,
        timeout = 15,
    )
    assert result.returncode == 0, result.stderr.decode(errors = "replace")
    value = json.loads(result.stdout)
    path = Path(value["path"])
    try:
        identity.recover_identities()
        assert path.exists() and identity._profile_path(value["sid"]) is not None
        assert identity._read(path)["owner_pid"] == os.getpid()
    finally:
        identity._recover(path, recipe)
    assert not path.exists() and identity._profile_path(value["sid"]) is None


@pytest.mark.parametrize("fail_collision_write", [False, True])
def test_observed_api_collision_does_not_delete_the_conflicting_profile(
    journal, reservation, monkeypatch, fail_collision_write
):
    _, work = journal
    api = lpac._api()
    original_create, original_write = api.userenv.CreateAppContainerProfile, identity._write
    conflicting = ctypes.c_void_p()
    sid_text = None

    def collide(*args):
        nonlocal sid_text
        assert original_create(*args[:-1], ctypes.byref(conflicting)) == 0
        sid_text = lpac._sid_string(api, conflicting)
        return ctypes.c_int32(0x800700B7).value

    def write(path, value):
        if fail_collision_write and value["state"] == "collision":
            raise PermissionError("injected collision journal failure")
        return original_write(path, value)

    try:
        with monkeypatch.context() as context:
            context.setattr(api.userenv, "CreateAppContainerProfile", collide)
            context.setattr(identity, "_write", write)
            with pytest.raises((identity.WindowsRuntimeError, PermissionError), match = "collision"):
                reservation.create(work)
            if fail_collision_write:
                with pytest.raises(PermissionError, match = "collision journal"):
                    reservation.cleanup()
                assert not reservation.closed
        reservation.cleanup()
        assert not reservation.path.exists()
        assert identity._profile_path(sid_text) is not None
    finally:
        # This test created the conflicting profile, not the reservation owner.
        if conflicting:
            assert api.userenv.DeleteAppContainerProfile(reservation.recipe.moniker) == 0
            api.advapi32.FreeSid(conflicting)


@pytest.mark.parametrize("stage", ["creating", "ready", "create_result", "grant"])
def test_real_launch_identity_failure_never_returns_a_payload(installed_runtime, tmp_path, stage):
    fixed_worker = f"""
import sys,json
from pathlib import Path
sys.path.insert(0, {str(BACKEND)!r})
from core.inference.windows_sandbox import identity as identities, launch
from core.inference.windows_sandbox.preparation_worker import main
write_original = identities._write
create_original = launch.lpac._api().userenv.CreateAppContainerProfile
grant_original = launch.lpac._grant_modify
journal_path = None
def write(path,value):
    global journal_path
    journal_path = path
    with open({str(tmp_path / "observed-identities.jsonl")!r},'a',encoding='utf-8') as trace:
        trace.write(json.dumps({{'sid':value['sid'],'path':str(path)}})+'\\n')
    write_original(path,value)
    if value['state'] == {stage!r}:
        raise OSError('injected durable identity failure')
def create(*args):
    result = create_original(*args)
    assert result == 0
    if {stage!r} == 'create_result':
        raise OSError('injected durable identity failure')
    return result
def grant(path,sid):
    assert journal_path is not None and identities._read(journal_path)['state'] == 'ready'
    grant_original(path,sid)
    if {stage!r} == 'grant':
        raise OSError('injected durable identity failure')
identities._write = write
launch.lpac._api().userenv.CreateAppContainerProfile = create
launch.lpac._grant_modify = grant
raise SystemExit(main())
"""
    body = f"""
import json
from core.inference.windows_sandbox import identity as identities, launch, preparation
script.write_text("from pathlib import Path; Path('payload-ran').touch()", encoding='utf-8')
spec = ToolLaunchPlan(argv=(sys.executable,'-u',str(script)), workdir=str(work), env={{}}, execution_kind='python')
owners = []
original_init = launch._PythonLaunch.__init__
original_run = preparation._run_worker
wrapper = root/'fixed-identity-failure.py'
wrapper.write_text({fixed_worker!r},encoding='utf-8')

def capture(owner, *args):
    original_init(owner, *args)
    owners.append(owner)

def fixed(argv,*args,**kwargs):
    if argv[4].endswith('preparation_worker.py'):
        argv = [*argv[:4],str(wrapper),*argv[5:]]
    return original_run(argv,*args,**kwargs)

launch._PythonLaunch.__init__ = capture
preparation._run_worker = fixed
try:
    try:
        prepare_python_launch(spec, root/'cache')
    except launch.WindowsRuntimeError as error:
        assert 'injected durable identity failure' in str(error)
    else:
        raise AssertionError('Failed identity preparation returned a payload')
finally:
    launch._PythonLaunch.__init__ = original_init
    preparation._run_worker = original_run
    for owner in owners:
        owner.cleanup()
assert len(owners) == 1 and owners[0].closed and owners[0].reservation.closed
records = [json.loads(line) for line in (root/'observed-identities.jsonl').read_text(encoding='utf-8').splitlines()]
assert records and all(identities._profile_path(record['sid']) is None for record in records)
assert all(not Path(record['path']).exists() for record in records)
assert not list((root/'cache'/'.readers').iterdir())
assert not owners[0].pins.handles and not launch._pending_cleanup
assert not (work/'payload-ran').exists()
print('IDENTITY_PREPARATION_FAILED_CLOSED')
"""
    assert "IDENTITY_PREPARATION_FAILED_CLOSED" in run_harness(installed_runtime, tmp_path, body)


def test_payload_cannot_read_or_modify_its_ownership_journal(installed_runtime, tmp_path):
    payload = """
import ctypes
from ctypes import wintypes as W
from pathlib import Path
path = Path('journal-path').read_text(encoding='utf-8')
api = ctypes.WinDLL('kernel32', use_last_error=True)
api.CreateFileW.argtypes = [W.LPCWSTR,W.DWORD,W.DWORD,ctypes.c_void_p,W.DWORD,W.DWORD,W.HANDLE]
api.CreateFileW.restype = W.HANDLE
api.CloseHandle.argtypes = [W.HANDLE]
api.CloseHandle.restype = W.BOOL
for access in (0x80000000, 0x40000000):
    handle = api.CreateFileW(path, access, 3, None, 3, 0, None)
    error = ctypes.get_last_error()
    if handle != ctypes.c_void_p(-1).value:
        api.CloseHandle(handle)
        raise AssertionError('Payload accessed the broker identity journal')
    assert error == 5, ('Expected native ACL denial', error)
print('OWNERSHIP_JOURNAL_DENIED_BY_WINDOWS', flush=True)
"""
    body = f"""
script.write_text({payload!r}, encoding='utf-8')
{LAUNCH}
(work/'journal-path').write_text(str(manifest), encoding='utf-8')
before = manifest.read_bytes()
assert json.loads(before)['state'] == 'ready'
try:
    process = spawn_prepared_launch(prepared, **kwargs)
    assert 'OWNERSHIP_JOURNAL_DENIED_BY_WINDOWS' in process.stdout.read()
    assert process.wait(timeout=5) == 0
    assert manifest.read_bytes() == before
finally:
    prepared.cleanup()
assert not prepared.cleanup_diagnostics and owner.closed and not manifest.exists()
print('OWNERSHIP_JOURNAL_BOUNDARY_OK')
"""
    assert "OWNERSHIP_JOURNAL_BOUNDARY_OK" in run_harness(installed_runtime, tmp_path, body)
