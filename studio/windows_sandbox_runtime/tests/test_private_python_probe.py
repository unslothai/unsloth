# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Live fixed private core probe; full backend qualification remains separate."""

import sys

import pytest

from test_launch import run_harness, installed_runtime, runtime_wheel
from test_preparation import BACKEND

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows private core probe")


def test_fixed_probe_runs_privately_and_cleans_up(installed_runtime, tmp_path):
    output = run_harness(
        installed_runtime,
        tmp_path,
        """
from core.inference.windows_sandbox.probe import prepare_python_probe, PROBE_FILENAME, probe_source
from core.inference.windows_sandbox import identity, launch
prepared = prepare_python_probe(sys.executable,root/'cache')
owner = prepared.spawn_callback.__self__
profile = owner.identity
assert prepared.execution_record is None and owner.workdir == Path(profile.private_temp)
assert owner.script == owner.workdir/PROBE_FILENAME
assert owner.script.read_text(encoding='utf-8') == probe_source(owner.nonce)
assert owner.script in owner.file_pins.handles
record = identity._read(owner.reservation.path)
assert record['version'] == 4 and record['purpose'] == 'qualification'
kwargs = dict(stdout=subprocess.PIPE,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
    text=True,encoding='utf-8',errors='replace',cwd=prepared.workdir,env=prepared.env,
    close_fds=True,creationflags=subprocess.CREATE_NO_WINDOW)
try:
    process = spawn_prepared_launch(prepared,**kwargs)
    text = process.stdout.read()
    assert process.wait(timeout=10) == 0,text
    value = json.loads(text)
    assert value['probe_nonce'] == owner.nonce.hex()
    assert value['version'] == list(sys.version_info[:3])
    assert value['checks'] == ['private_workdir','stdlib_native_asyncio','threads_private_pipe',
        'aap_sentinel_denied','runtime_write_denied','host_read_denied',
        'host_write_denied','process_policy_diagnostic',
        'host_registry_read_denied','host_registry_write_denied']
    assert 'qualified' not in value and 'available' not in value
    from core.inference.windows_sandbox.probe import _parse_dns_context
    context = _parse_dns_context(value['dns_context'])
    assert not hasattr(context,'qualified') and not hasattr(context,'available')
    # Record the actual post-drop observations without declaring a platform
    # qualified merely because this partial core probe completed.
    print('POST_DROP_DNS_CONTEXT='+json.dumps(value['dns_context']))
    assert prepared.execution_record is None
finally:
    owner.cleanup()
assert owner.closed and not owner.handles and not owner.pins.handles and not owner.file_pins.handles
assert not Path(profile.profile_folder).exists() and not owner.reservation.path.exists()
assert not list((root/'cache'/'.readers').iterdir()) and not launch._pending_cleanup
assert not list(work.iterdir()), 'Probe touched the contributor workdir'
print('PRIVATE_CORE_PROBE_OK')
""",
    )
    assert "PRIVATE_CORE_PROBE_OK" in output
    print(next(line for line in output.splitlines() if line.startswith("POST_DROP_DNS_CONTEXT=")))


@pytest.mark.parametrize("field", ["workdir", "script", "profile_location", "probe_pin"])
def test_private_probe_rejects_changed_handoff(installed_runtime, tmp_path, field):
    output = run_harness(
        installed_runtime,
        tmp_path,
        f"""
from core.inference.windows_sandbox import launch, launch_transfer, preparation
from core.inference.windows_sandbox.probe import prepare_python_probe, PROBE_FILENAME
original_init,original_adopt = launch._PythonLaunch.__init__,launch_transfer.adopt_launch
owners = []
def capture(owner,*args):
    original_init(owner,*args)
    owners.append(owner)
def corrupt(owner,value,*args):
    value = dict(value)
    if {field!r} in ('workdir','script'):
        value[{field!r}] = str(root/'external')
    elif {field!r} == 'probe_pin':
        value['file_pins'] = [row for row in value['file_pins'] if not row[0].endswith(PROBE_FILENAME)]
    else:
        # Preserve all the simple layout and inventory predicates, but point
        # the response at a different host location than Windows assigned.
        drive = Path(value['profile']).drive
        other = 'E:' if drive.upper() != 'E:' else 'D:'
        for key in ('profile','temp','workdir','script'):
            value[key] = value[key].replace(drive,other,1)
        value['traverse'] = [p.replace(drive,other,1) for p in value['traverse']]
        value['file_pins'] = [[p.replace(drive,other,1),h] for p,h in value['file_pins']]
    return original_adopt(owner,value,*args)
launch._PythonLaunch.__init__,launch_transfer.adopt_launch = capture,corrupt
try:
    try:
        prepare_python_probe(sys.executable,root/'cache')
    except launch.WindowsRuntimeError as error:
        assert error.code == 'WINDOWS_SANDBOX_PREPARATION_FAILED',str(error)
        if {field!r} == 'profile_location':
            assert 'actual profile location' in str(error),str(error)
    else:
        raise AssertionError('Changed private handoff returned a launch')
finally:
    launch._PythonLaunch.__init__,launch_transfer.adopt_launch = original_init,original_adopt
    for owner in owners:
        owner.cleanup()
assert owners and all(owner.closed and not owner.pins.handles and not owner.file_pins.handles for owner in owners)
assert not list((root/'cache'/'.readers').iterdir()) and not launch._pending_cleanup
assert not owners[0].reservation.path or not owners[0].reservation.path.exists()
print('PRIVATE_PROBE_HANDOFF_REJECTED')
""",
    )
    assert "PRIVATE_PROBE_HANDOFF_REJECTED" in output


def test_private_probe_request_cannot_accept_tool_arguments(monkeypatch):
    from types import SimpleNamespace
    from core.inference.windows_sandbox import launch_transfer, identity
    from core.inference.windows_sandbox.profiles import WindowsRuntimeError

    recipe = identity.InvocationRecipe.new()
    request = dict(
        probe = sys.executable, network_ports = [], recipe = vars(recipe), reader = "a" * 32, nonce = "b" * 64
    )
    broker = SimpleNamespace(executable = sys.executable, pid = recipe.owner_pid)
    for key, value in [
        ("argv", ["ignored"]),
        ("workdir", "C:\\external"),
        ("env", {}),
        ("code", "print(1)"),
    ]:
        with pytest.raises(WindowsRuntimeError, match = "Invalid fixed Python"):
            launch_transfer.worker_owner({**request, key: value}, broker, "C:\\unused")


@pytest.mark.parametrize("termination", ["worker_exit", "cancel", "timeout"])
def test_private_probe_recovers_granted_reader_without_handoff(
    installed_runtime, tmp_path, termination
):
    worker = f"""
import sys,os,json,time
from pathlib import Path
sys.path.insert(0,{str(BACKEND)!r})
from core.inference.windows_sandbox import launch,identity
from core.inference.windows_sandbox.preparation_worker import main
original = launch._PythonLaunch.prepare_files
def stop(owner):
    original(owner)
    record = identity._read(owner.reservation.path)
    assert record['version'] == 4 and record['purpose'] == 'qualification'
    assert owner.access is not None and record['reader']['name'] == owner.reader_name
    Path({str(tmp_path / "private-worker-ready.json")!r}).write_text(
        json.dumps({{'record':record,'path':str(owner.reservation.path),'pid':os.getpid()}}),
        encoding='utf-8')
    if {termination!r} == 'worker_exit':
        os._exit(42)
    time.sleep(60)
    raise AssertionError('Cancelled preparation worker survived')
launch._PythonLaunch.prepare_files = stop
raise SystemExit(main())
"""
    output = run_harness(
        installed_runtime,
        tmp_path,
        f"""
from core.inference.windows_sandbox import launch,preparation,identity
from core.inference.windows_sandbox.probe import prepare_python_probe
from core.inference.windows_sandbox.content import RuntimeContentStore
from core.inference.windows_sandbox.content_files import native_files
wrapper = root/'fixed-private-worker.py'
wrapper.write_text({worker!r},encoding='utf-8')
marker = root/'private-worker-ready.json'
original_run,original_init = preparation._run_worker,launch._PythonLaunch.__init__
original_check = preparation._check_deadline
owners = []
armed = []
cancel,finished,observed = threading.Event(),threading.Event(),threading.Event()
def capture(owner,*args):
    original_init(owner,*args)
    owners.append(owner)
def intercept(argv,*args,**kwargs):
    if argv[4].endswith('preparation_worker.py'):
        argv = [*argv[:4],str(wrapper),*argv[5:]]
    return original_run(argv,*args,**kwargs)
def check_deadline(limit,signal):
    if signal is cancel and {termination!r} == 'timeout' and marker.exists():
        if not armed:
            armed.append(time.monotonic())
        # Shorten only the existing deadline after the real lease/grants exist.
        limit = min(limit,armed[0]+.25)
    return original_check(limit,signal)
def watcher():
    while not finished.wait(.01):
        if marker.exists():
            observed.set()
            if {termination!r} == 'cancel':
                cancel.set()
            return
thread = threading.Thread(target=watcher)
launch._PythonLaunch.__init__,preparation._run_worker = capture,intercept
preparation._check_deadline = check_deadline
thread.start()
try:
    try:
        prepare_python_probe(sys.executable,root/'cache',cancel=cancel)
    except launch.WindowsRuntimeError as error:
        expected = {{'cancel':'WINDOWS_SANDBOX_CANCELLED','timeout':'WINDOWS_SANDBOX_PREPARATION_TIMEOUT',
            'worker_exit':'WINDOWS_SANDBOX_PREPARATION_FAILED'}}[{termination!r}]
        assert error.code == expected,str(error)
    else:
        raise AssertionError('Failed private preparation returned a launch')
finally:
    finished.set()
    thread.join(5)
    launch._PythonLaunch.__init__,preparation._run_worker = original_init,original_run
    preparation._check_deadline = original_check
    for owner in owners:
        owner.cleanup()
assert not thread.is_alive() and marker.is_file()
if {termination!r} == 'cancel':
    assert observed.is_set()
if {termination!r} == 'timeout':
    assert armed and time.monotonic()-armed[0] < 10
trace = json.loads(marker.read_text(encoding='utf-8'))
record = trace['record']
assert all(owner.closed and not owner.started and owner.process is None for owner in owners)
assert all(not owner.handles and not owner.pins.handles and not owner.file_pins.handles for owner in owners)
assert identity._profile_path(record['sid']) is None
assert not Path(trace['path']).exists() and not Path(record['profile_folder']).exists()
assert not list((root/'cache'/'.readers').iterdir()) and not launch._pending_cleanup
assert not list(work.iterdir()),'Probe touched the contributor workdir'
api = native_files()
store = RuntimeContentStore(root/'cache')
with store.lease(record['reader']['digest']) as generation:
    for path in generation.files:
        handle = api.open(path)
        try:
            assert record['sid'] not in api.security_text(handle),'Leaked private reader ACE'
        finally:
            assert api.kernel.CloseHandle(handle)
import ctypes
api = launch.lpac._api().kernel32
handle = api.OpenProcess(0x100000,False,trace['pid'])
if handle:
    try:
        assert api.WaitForSingleObject(handle,0) == 0,'Preparation worker is still alive'
    finally:
        assert api.CloseHandle(handle)
else:
    assert ctypes.get_last_error() == 87,ctypes.get_last_error()
print('PRIVATE_PROBE_WORKER_RECOVERED')
""",
    )
    assert "PRIVATE_PROBE_WORKER_RECOVERED" in output
