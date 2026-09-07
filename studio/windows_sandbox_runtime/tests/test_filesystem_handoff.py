# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Real installed launch filesystem handoff, deadlines and failed transfer."""

import json
from pathlib import Path
import subprocess
import sys
import time

import pytest

from test_launch import installed_runtime, runtime_wheel, run_harness, LAUNCH
from test_preparation import BACKEND, lpac
from test_publication import harness_source
from core.inference.windows_sandbox import identity, preparation

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows filesystem handoff")


def test_real_launch_does_not_reopen_prepared_files_in_broker(installed_runtime, tmp_path):
    body = f"""
from core.inference.windows_sandbox import launch, identity, dependencies, content_files
script.write_text("print('HANDOFF_PAYLOAD_OK', flush=True)", encoding='utf-8')
originals = []
def forbidden(*args, **kwargs):
    raise AssertionError('Broker performed filesystem preparation')
for module, name in ((launch, 'checked_path'), (launch, '_validate_paths'),
        (launch.lpac, '_validate_workdir'), (launch.lpac, '_grant_modify'),
        (identity, '_journal_root'), (dependencies, 'checked_path'),
        (content_files.PathLease, 'file'), (content_files.PathLease, 'directory')):
    originals.append((module,name,getattr(module,name)))
    setattr(module,name,forbidden)
prepared = None
try:
{"".join("    " + line + chr(10) for line in LAUNCH.splitlines())}
    assert owner.reservation.worker_environment and owner.reservation.started
    assert owner.access.store is None
    assert owner.access.pins.handles and owner.file_pins.handles
    process = spawn_prepared_launch(prepared, **kwargs)
    assert process.stdout.readline().strip() == 'HANDOFF_PAYLOAD_OK'
    assert process.wait(timeout=10) == 0
finally:
    if prepared is not None:
        prepared.cleanup()
    for module,name,value in originals:
        setattr(module,name,value)
assert owner.closed and not prepared.cleanup_diagnostics
assert not manifest.exists() and not list((root/'cache'/'.readers').iterdir())
print('NO_BROKER_FILESYSTEM_OK')
"""
    assert "NO_BROKER_FILESYSTEM_OK" in run_harness(installed_runtime, tmp_path, body)


@pytest.mark.parametrize("phase", ["validate", "grant", "reader", "config"])
@pytest.mark.parametrize("mode", ["timeout", "cancel"])
def test_blocked_filesystem_preparation_is_reaped_without_reply(
    installed_runtime, tmp_path, phase, mode
):
    marker = tmp_path / "entered-phase"
    worker = f"""
import sys, os
from pathlib import Path
sys.path.insert(0, {str(BACKEND)!r})
from core.inference.windows_sandbox import launch, content_access
from core.inference.windows_sandbox.preparation_worker import main
def pause(*args, **kwargs):
    Path({str(marker)!r}).write_text('entered',encoding='utf-8')
    r,w = os.pipe()
    os.read(r,1)
if {phase!r} == 'validate':
    launch._validate_paths = pause
elif {phase!r} == 'grant':
    launch.lpac._grant_modify = pause
elif {phase!r} == 'reader':
    content_access._change = pause
else:
    create = launch.native_files().create
    def creating(path,data):
        if Path(path).name == 'startup-config':
            create(path,data)
            pause()
        return create(path,data)
    launch.native_files().create = creating
raise SystemExit(main())
"""
    body = f"""
import time
from core.inference.windows_sandbox import launch, preparation, identity
script.write_text("from pathlib import Path; Path('payload-ran').touch()", encoding='utf-8')
wrapper = root/'fixed-filesystem-stall.py'
wrapper.write_text({worker!r}, encoding='utf-8')
owners, observed = [], []
original_init = launch._PythonLaunch.__init__
original_process = launch.lpac.WindowsLpacProcess.__init__
kernel = launch.lpac._api().kernel32
def capture(owner,*args):
    original_init(owner,*args)
    owners.append(owner)
def process_capture(process,*args):
    original_process(process,*args)
    handle = kernel.OpenProcess(0x100000,False,process.pid)
    assert handle
    observed.append(handle)
launch._PythonLaunch.__init__ = capture
launch.lpac.WindowsLpacProcess.__init__ = process_capture
run, check = preparation._run_worker, preparation._check_deadline
cancel = threading.Event()
armed = []
def deadline(limit, signal):
    if signal is cancel and Path({str(marker)!r}).exists():
        if not armed:
            armed.append(time.monotonic())
        if {mode!r} == 'cancel':
            cancel.set()
        else:
            limit = min(limit,armed[0]+0.25)
    return check(limit,signal)
def fixed_worker(argv,*args,**kwargs):
    if argv[4].endswith('preparation_worker.py'):
        argv = [*argv[:4],str(wrapper),*argv[5:]]
    return run(argv,*args,**kwargs)
preparation._run_worker, preparation._check_deadline = fixed_worker, deadline
try:
    spec = ToolLaunchPlan(argv=(sys.executable,'-u',str(script)),workdir=str(work),env={{}},execution_kind='python')
    try:
        prepare_python_launch(spec,root/'cache',cancel=cancel)
    except launch.WindowsRuntimeError as error:
        assert ('CANCELLED' if {mode!r} == 'cancel' else 'TIMEOUT') in error.code, str(error)
    else:
        raise AssertionError('Blocked filesystem preparation returned a launch')
    assert armed and time.monotonic()-armed[0] < 10
    owner = owners[0]
    assert owner.closed and owner.reservation.closed
    assert owner.identity is None and owner.reservation.path is None
    assert not owner.pins.handles and not owner.file_pins.handles
    assert not launch._pending_cleanup and not (work/'payload-ran').exists()
    with identity._derived_sid(owner.reservation.recipe.moniker) as (_,sid):
        assert identity._profile_path(sid) is None
    with identity._journal_root() as journal:
        assert not (journal/owner.reservation.recipe.filename()).exists()
    readers = root/'cache'/'.readers'
    assert not readers.exists() or not list(readers.iterdir())
finally:
    preparation._run_worker, preparation._check_deadline = run,check
    launch._PythonLaunch.__init__ = original_init
    launch.lpac.WindowsLpacProcess.__init__ = original_process
    for owner in owners:
        owner.cleanup()
    for handle in observed:
        try:
            assert kernel.WaitForSingleObject(handle,0) == 0
        finally:
            assert kernel.CloseHandle(handle)
print('FILESYSTEM_STALL_REAPED')
"""
    assert "FILESYSTEM_STALL_REAPED" in run_harness(installed_runtime, tmp_path, body)


@pytest.mark.parametrize("phase", ["reader_duplicate", "file_duplicate", "ack"])
def test_failed_launch_pin_handoff_never_returns_payload(installed_runtime, tmp_path, phase):
    body = f"""
from core.inference.windows_sandbox import launch, preparation
script.write_text("from pathlib import Path; Path('payload-ran').touch()",encoding='utf-8')
owners = []
original_init, original_duplicate = launch._PythonLaunch.__init__, launch.PathLease.duplicate_from
from core.inference.windows_sandbox.protocol import _pipe_api
pipe_api = _pipe_api()
original_write = pipe_api.WriteFile
def capture(owner,*args):
    original_init(owner,*args)
    owners.append(owner)
def duplicate(pins,process,entries,check):
    if ({phase!r} == 'reader_duplicate' and any(p.endswith('.lock') for p,h in entries)) or (
        {phase!r} == 'file_duplicate' and any(p.endswith('startup-config') for p,h in entries)):
        original_duplicate(pins,process,entries[:2],check)
        assert pins.handles
        raise OSError('injected partial handoff')
    return original_duplicate(pins,process,entries,check)
def write(handle,data,size,written,overlapped):
    if {phase!r} == 'ack' and size == 32:
        raise OSError('injected private ACK failure')
    return original_write(handle,data,size,written,overlapped)
launch._PythonLaunch.__init__, launch.PathLease.duplicate_from = capture,duplicate
pipe_api.WriteFile = write
try:
    spec = ToolLaunchPlan(argv=(sys.executable,'-u',str(script)),workdir=str(work),env={{}},execution_kind='python')
    try:
        prepare_python_launch(spec,root/'cache')
    except launch.WindowsRuntimeError as error:
        assert 'injected' in str(error), str(error)
    else:
        raise AssertionError('Incomplete handoff returned a payload')
finally:
    launch._PythonLaunch.__init__, launch.PathLease.duplicate_from = original_init,original_duplicate
    pipe_api.WriteFile = original_write
    for owner in owners:
        owner.cleanup()
assert owners and all(owner.closed and not owner.pins.handles and not owner.file_pins.handles for owner in owners)
assert not (work/'payload-ran').exists() and not launch._pending_cleanup
assert not list((root/'cache'/'.readers').iterdir())
assert not owners[0].reservation.path.exists()
print('FAILED_HANDOFF_CLEANED')
"""
    assert "FAILED_HANDOFF_CLEANED" in run_harness(installed_runtime, tmp_path, body)


@pytest.mark.parametrize("field", ["sid", "temp", "reader_pins", "file_pins", "env"])
def test_corrupt_launch_metadata_never_acknowledges_or_spawns(installed_runtime, tmp_path, field):
    body = f"""
import json
from core.inference.windows_sandbox import launch, preparation, identity
script.write_text("from pathlib import Path; Path('payload-ran').touch()",encoding='utf-8')
owners, calls = [], []
original_init, run = launch._PythonLaunch.__init__, preparation._run_worker
def capture(owner,*args):
    original_init(owner,*args)
    owners.append(owner)
def worker(argv,*args,**kwargs):
    if 'transfer' in kwargs:
        transfer = kwargs['transfer']
        def altered(process,data):
            calls.append(process.pid)
            value = json.loads(data)
            record = value['launch']
            if {field!r} == 'sid':
                record['sid'] = 'S-1-15-2-1-2-3-4-5-6-7'
            elif {field!r} == 'temp':
                record['temp'] = str(root/'foreign-temp')
            elif {field!r} == 'reader_pins':
                record['reader_pins'].pop()
            elif {field!r} == 'file_pins':
                record['file_pins'][0][1] = value['pins'][0][1]
            else:
                record['env']['malformed'] = []
            return transfer(process,json.dumps(value).encode())
        kwargs['transfer'] = altered
    return run(argv,*args,**kwargs)
launch._PythonLaunch.__init__, preparation._run_worker = capture,worker
try:
    spec = ToolLaunchPlan(argv=(sys.executable,'-u',str(script)),workdir=str(work),env={{}},execution_kind='python')
    try:
        prepare_python_launch(spec,root/'cache')
    except launch.WindowsRuntimeError as error:
        assert error.code == 'WINDOWS_SANDBOX_PREPARATION_FAILED', str(error)
    else:
        raise AssertionError('Corrupt handoff returned a launch')
finally:
    launch._PythonLaunch.__init__, preparation._run_worker = original_init,run
    for owner in owners:
        owner.cleanup()
assert len(calls) == len(owners) == 1
owner = owners[0]
assert owner.closed and not owner.started and owner.identity is None
assert not owner.pins.handles and not owner.file_pins.handles
assert not owner.reservation.created_sid and not launch._pending_cleanup
assert not (work/'payload-ran').exists() and not list((root/'cache'/'.readers').iterdir())
with identity._derived_sid(owner.reservation.recipe.moniker) as (_,sid):
    assert identity._profile_path(sid) is None
print('CORRUPT_HANDOFF_REJECTED')
"""
    assert "CORRUPT_HANDOFF_REJECTED" in run_harness(installed_runtime, tmp_path, body)


@pytest.mark.parametrize("failed_tombstone", [False, True])
def test_worker_collision_handoff_preserves_the_existing_profile(
    installed_runtime, tmp_path, failed_tombstone
):
    wrapper_source = f"""
import sys,ctypes
sys.path.insert(0,{str(BACKEND)!r})
from core.inference.windows_sandbox import native_compat as lpac
from core.inference.windows_sandbox import identity
from core.inference.windows_sandbox.preparation_worker import main
create,write = lpac._api().userenv.CreateAppContainerProfile,identity._write
conflicting = ctypes.c_void_p()
def collision(*args):
    assert create(*args[:-1],ctypes.byref(conflicting)) == 0
    return ctypes.c_int32(0x800700B7).value
def tombstone(path,value):
    if {failed_tombstone!r} and value['state'] == 'collision':
        raise OSError('injected collision tombstone failure')
    return write(path,value)
lpac._api().userenv.CreateAppContainerProfile,identity._write = collision,tombstone
try:
    raise SystemExit(main())
finally:
    if conflicting:
        assert not lpac._api().advapi32.FreeSid(conflicting)
"""
    body = f"""
from core.inference.windows_sandbox import launch,preparation,identity
script.write_text("from pathlib import Path; Path('payload-ran').touch()",encoding='utf-8')
wrapper = root/'fixed-worker-collision.py'
wrapper.write_text({wrapper_source!r},encoding='utf-8')
owners = []
original_init,run = launch._PythonLaunch.__init__,preparation._run_worker
def capture(owner,*args):
    original_init(owner,*args)
    owners.append(owner)
def worker(argv,*args,**kwargs):
    if argv[4].endswith('preparation_worker.py'):
        argv = [*argv[:4],str(wrapper),*argv[5:]]
    return run(argv,*args,**kwargs)
launch._PythonLaunch.__init__,preparation._run_worker = capture,worker
try:
    spec = ToolLaunchPlan(argv=(sys.executable,'-u',str(script)),workdir=str(work),env={{}},execution_kind='python')
    try:
        prepare_python_launch(spec,root/'cache')
    except launch.WindowsRuntimeError as error:
        assert 'collision' in str(error) and error.code != 'WINDOWS_SANDBOX_CLEANUP_FAILED', str(error)
    else:
        raise AssertionError('A collided profile produced a launch')
    assert len(owners) == 1
    owner = owners[0]
    assert owner.closed and owner.reservation.closed and not owner.started
    assert owner.reservation.collision_record['state'] == 'collision'
    with identity._derived_sid(owner.reservation.recipe.moniker) as (_,sid):
        assert identity._profile_path(sid) is not None, 'Recovery deleted the conflicting profile'
    with identity._journal_root() as journal:
        assert not (journal/owner.reservation.recipe.filename()).exists()
    assert not (work/'payload-ran').exists() and not list((root/'cache'/'.readers').iterdir())
finally:
    launch._PythonLaunch.__init__,preparation._run_worker = original_init,run
    for owner in owners:
        try:
            owner.cleanup()
        finally:
            # Only this fixture created the conflicting profile. Production
            # recovery is required to leave it alone, including on a retry.
            result = launch.lpac._api().userenv.DeleteAppContainerProfile(owner.reservation.recipe.moniker)
            assert result == 0, result
assert not launch._pending_cleanup
print('COLLIDED_PROFILE_PRESERVED')
"""
    assert "COLLIDED_PROFILE_PRESERVED" in run_harness(installed_runtime, tmp_path, body)


def test_broker_death_before_reply_releases_reader_pins_and_recovers_identity(
    installed_runtime, tmp_path, monkeypatch
):
    local = tmp_path / "LocalAppData"
    local.mkdir()
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    marker = tmp_path / "reader-ready.json"
    work = tmp_path / "work"
    work.mkdir()
    script = work / "tool.py"
    script.write_text("from pathlib import Path; Path('payload-ran').touch()", encoding = "utf-8")
    wrapper = tmp_path / "fixed-held-reader.py"
    wrapper.write_text(
        f"""
import sys,os,json
from pathlib import Path
sys.path.insert(0,{str(BACKEND)!r})
from core.inference.windows_sandbox import content_access
from core.inference.windows_sandbox.preparation_worker import main
original = content_access.RuntimeReadLease.__enter__
def entered(lease):
    result = original(lease)
    request = json.loads(sys.argv[-1])
    Path({str(marker)!r}).write_text(json.dumps({{'recipe':request['launch']['recipe'],
        'sid':lease.sid,'pid':os.getpid(),'reader':lease.name,'store':str(lease.store_root)}}),encoding='utf-8')
    r,w = os.pipe()
    os.read(r,1)
    return result
content_access.RuntimeReadLease.__enter__ = entered
raise SystemExit(main())
""",
        encoding = "utf-8",
    )
    source = harness_source(f"""
from core.inference.os_sandbox import ToolLaunchPlan
from core.inference.windows_sandbox import preparation,launch
run = preparation._run_worker
def fixed(argv,*args,**kwargs):
    if argv[4].endswith('preparation_worker.py'):
        argv = [*argv[:4],{str(wrapper)!r},*argv[5:]]
    return run(argv,*args,**kwargs)
preparation._run_worker = fixed
spec = ToolLaunchPlan(argv=(sys.executable,'-u',{str(script)!r}),workdir={str(work)!r},env={{}},execution_kind='python')
launch.prepare_python_launch(spec,{str(tmp_path / "cache")!r})
raise AssertionError('Held preparation returned a launch')
""")
    parent = subprocess.Popen(
        [str(installed_runtime / "Scripts/python.exe"), "-I", "-c", source],
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
    )
    value = handle = None
    kernel = lpac._api().kernel32
    try:
        deadline = time.monotonic() + 90
        while not marker.exists():
            assert parent.poll() is None, parent.communicate(timeout = 5)
            assert time.monotonic() < deadline, "Worker never reached its owned read lease"
            time.sleep(0.01)
        # A small file can be visible before its writer closes; wait for complete
        # JSON within the same deadline, not by restarting the owned processes.
        while value is None:
            try:
                value = json.loads(marker.read_text(encoding = "utf-8"))
            except json.JSONDecodeError:
                assert time.monotonic() < deadline
                time.sleep(0.01)
        handle = kernel.OpenProcess(0x100001, False, value["pid"])
        assert handle and kernel.WaitForSingleObject(handle, 0) == 258
        parent.kill()
        parent.communicate(timeout = 5)
        assert kernel.WaitForSingleObject(handle, 5000) == 0, "Broker death left preparation alive"
        recipe = identity.InvocationRecipe(**value["recipe"])
        assert identity._profile_path(value["sid"]) is not None
        readers = Path(value["store"]) / ".readers"
        assert (readers / (value["reader"] + ".json")).exists()
        preparation.cleanup_invocation_profile(recipe, None)
        assert identity._profile_path(value["sid"]) is None
        assert not list(readers.iterdir()) and not (work / "payload-ran").exists()
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
        if value is not None:
            preparation.cleanup_invocation_profile(
                identity.InvocationRecipe(**value["recipe"]), None
            )
        else:
            identity.recover_identities()
