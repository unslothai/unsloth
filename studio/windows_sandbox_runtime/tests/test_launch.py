# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Installed-runtime launch assembly through the real shared spawn seam."""

from pathlib import Path
import subprocess
import sys

import pytest

import test_artifacts as artifacts
from test_publication import harness_source

runtime_wheel = artifacts.runtime_wheel
installed_runtime = artifacts.installed_runtime
pytestmark = pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows launch lane")


def run_harness(
    prefix,
    directory,
    body,
    *,
    timeout = 120,
):
    source = harness_source(f"""
from pathlib import Path
import subprocess, threading
from core.inference.os_sandbox import ToolLaunchPlan, spawn_prepared_launch
from core.inference.windows_sandbox.launch import prepare_python_launch
root = Path({str(directory)!r})
work = root / 'work'
work.mkdir()
script = work / 'tool.py'
{body}
""")
    result = subprocess.run(
        [str(prefix / "Scripts/python.exe"), "-I", "-c", source],
        capture_output = True,
        timeout = timeout,
    )
    assert result.returncode == 0, result.stderr.decode(
        "utf-8", errors = "replace"
    ) + result.stdout.decode("utf-8", errors = "replace")
    return result.stdout.decode("utf-8", errors = "replace")


LAUNCH = """
spec = ToolLaunchPlan(argv=(sys.executable, '-u', str(script)), workdir=str(work), env={'PYTHONIOENCODING':'utf-8'}, execution_kind='python')
prepared = prepare_python_launch(spec, root / 'cache')
owner = prepared.spawn_callback.__self__
identity = owner.identity
manifest = Path(identity.manifest_path)
assert prepared.execution_record is None
kwargs = dict(stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
    text=True, encoding='utf-8', errors='replace', cwd=prepared.workdir, env=prepared.env,
    close_fds=True, creationflags=subprocess.CREATE_NO_WINDOW)
"""


PROFILE_FAILURE_SOURCE = """
import sys
sys.path.insert(0, sys.argv.pop(1))
from core.inference.windows_sandbox import native_compat as lpac
from core.inference.windows_sandbox.profile_cleanup_worker import main
def failed_cleanup(identity):
    raise OSError('injected identity ACL cleanup failure')
lpac._InvocationIdentity.cleanup = failed_cleanup
raise SystemExit(main())
"""

PROFILE_FAILURE_SETUP = f"""
import json
from core.inference.windows_sandbox import preparation
failure_worker = root/'fixed-profile-failure.py'
failure_worker.write_text({PROFILE_FAILURE_SOURCE!r}, encoding='utf-8')
original_worker = preparation._run_worker
def failed_profile_cleanup(argv, *args, **kwargs):
    if argv[4].endswith('profile_cleanup_worker.py'):
        request = json.loads(argv[-1])
        if only_moniker is None or request['recipe']['moniker'] == only_moniker:
            assert argv[1:4] == ['-I','-S','-B']
            return original_worker([*argv[:4], str(failure_worker), str(Path(launch.__file__).parents[3]), *argv[5:]], *args, **kwargs)
    return original_worker(argv, *args, **kwargs)
preparation._run_worker = failed_profile_cleanup
"""


def test_installed_launch_streams_before_completion_and_cleans_identity(
    installed_runtime, tmp_path
):
    payload = "import asyncio, sqlite3, time; print('ASSEMBLY_STARTED', flush=True); time.sleep(1); assert asyncio.run(asyncio.sleep(0, result=7)) == 7; print(sqlite3.connect(':memory:').execute('select 17').fetchone())"
    output = run_harness(
        installed_runtime,
        tmp_path,
        f"""
script.write_text({payload!r}, encoding='utf-8')
{LAUNCH}
try:
    process = spawn_prepared_launch(prepared, **kwargs)
    assert prepared.execution_record is None
    assert process.stdout.readline().strip() == 'ASSEMBLY_STARTED'
    assert process.poll() is None, 'Streaming waited until completion'
    assert '(17,)' in process.stdout.read()
    assert process.wait(timeout=10) == 0
    try:
        spawn_prepared_launch(prepared, **kwargs)
    except Exception as error:
        assert 'reused or replayed' in str(error)
    else:
        raise AssertionError('Launch replayed')
finally:
    prepared.cleanup()
assert not prepared.cleanup_diagnostics, prepared.cleanup_diagnostics
assert owner.closed and identity.cleaned and not manifest.exists()
assert not list((root/'cache'/'.readers').iterdir())
print('ASSEMBLY_OK')
""",
    )
    assert "ASSEMBLY_OK" in output


def test_installed_shim_keeps_windows_alias_and_native_workdir_paths(installed_runtime, tmp_path):
    payload = r"""
from pathlib import Path
alias = Path(r'\mnt\data\nested\value.txt')
alias.parent.mkdir(parents=True, exist_ok=True)
alias.write_text('alias content', encoding='utf-8')
assert alias.read_text(encoding='utf-8') == 'alias content'
assert (Path.cwd() / 'nested/value.txt').read_text(encoding='utf-8') == 'alias content'
native = Path.cwd() / 'native.txt'
native.write_text('native content', encoding='utf-8')
assert native.read_text(encoding='utf-8') == 'native content'
print('INSTALLED_SHIM_PATHS_OK', flush=True)
"""
    body = f"""
script.write_text({payload!r}, encoding='utf-8')
{LAUNCH}
try:
    process = spawn_prepared_launch(prepared, **kwargs)
    assert 'INSTALLED_SHIM_PATHS_OK' in process.stdout.read()
    assert process.wait(timeout=10) == 0
finally:
    prepared.cleanup()
assert not prepared.cleanup_diagnostics and owner.closed
assert identity.cleaned and not manifest.exists()
assert (work/'nested/value.txt').read_text(encoding='utf-8') == 'alias content'
assert (work/'native.txt').read_text(encoding='utf-8') == 'native content'
assert not list((root/'cache'/'.readers').iterdir())
print('INSTALLED_SHIM_CLEAN')
"""
    assert "INSTALLED_SHIM_CLEAN" in run_harness(installed_runtime, tmp_path, body)


def test_private_cache_parent_does_not_require_broad_read_grants(installed_runtime, tmp_path):
    body = f"""
from core.inference.windows_sandbox.content_files import native_files
private = root / 'private-cache-parent'
native_files().mkdir(private)
secret = private / 'unrelated-secret'
native_files().create(secret, b'private parent secret')
script.write_text("from pathlib import Path\\ntry:\\n Path(" + repr(str(secret)) + ").read_bytes()\\nexcept PermissionError:\\n print('PRIVATE_PARENT_DENIED')\\nelse:\\n raise AssertionError('Private parent secret exposed')", encoding='utf-8')
{LAUNCH.replace("root / 'cache'", "private / 'cache'")}
try:
    process = spawn_prepared_launch(prepared, **kwargs)
    assert 'PRIVATE_PARENT_DENIED' in process.stdout.read()
    assert process.wait(timeout=10) == 0
finally:
    prepared.cleanup()
assert not prepared.cleanup_diagnostics, prepared.cleanup_diagnostics
assert owner.closed and identity.cleaned
print('PRIVATE_CACHE_OK')
"""
    assert "PRIVATE_CACHE_OK" in run_harness(installed_runtime, tmp_path, body)


@pytest.mark.parametrize("raw_handles", [False, True])
def test_creation_failure_retains_live_owner_until_reaped(installed_runtime, tmp_path, raw_handles):
    body = f"""
from core.inference.windows_sandbox import launch
script.write_text("from pathlib import Path; Path('payload-ran').touch()", encoding='utf-8')
{LAUNCH}
original_create = launch.create_suspended_host
retained = []
def fail_after_creation(*args, **kwargs):
    process = original_create(*args, **kwargs)
    retained.append((process, process._unsloth_job.terminate))
    process._unsloth_job.terminate = lambda: False
    error = launch.WindowsRuntimeError('WINDOWS_SANDBOX_CLEANUP_FAILED', 'injected creation failure')
    if {raw_handles!r}:
        error.retained_job = process._unsloth_job
        error.retained_native_handles = (process._handle, process._thread_handle)
        error.retained_token_handles = (process._startup_token,)
        assert process._startup_token is not None
        process._startup_token = None  # Raw error now owns every native handle.
    else:
        error.retained_processes = (process,)
    raise error
launch.create_suspended_host = fail_after_creation
try:
    try:
        spawn_prepared_launch(prepared, **kwargs)
    except launch.WindowsRuntimeError as error:
        assert error.code == 'WINDOWS_SANDBOX_CLEANUP_FAILED'
        assert error.retained_launch is owner
    else:
        raise AssertionError('Creation failure returned a process')
    assert len(retained) == 1 and retained[0][0].poll() is None
    assert not owner.closed and not owner.access.closed
    assert not identity.cleaned and manifest.exists()
    assert owner.handles and list((root/'cache'/'.readers').iterdir())
    assert prepared.execution_record is None and not (work/'payload-ran').exists()
finally:
    launch.create_suspended_host = original_create
    for process, terminate in retained:
        process._unsloth_job.terminate = terminate
    prepared.cleanup()
assert not prepared.cleanup_diagnostics, prepared.cleanup_diagnostics
assert owner.closed and identity.cleaned and not manifest.exists()
assert not owner.retained_processes and not owner.retained_raw and not owner.handles
assert not list((root/'cache'/'.readers').iterdir())
assert not (work/'payload-ran').exists()
print('RETAINED_OWNER_REAPED')
"""
    assert "RETAINED_OWNER_REAPED" in run_harness(installed_runtime, tmp_path, body)


@pytest.mark.parametrize("failure", ["cancel", "gate", "stdio"])
def test_failed_launch_never_runs_or_replays_payload(installed_runtime, tmp_path, failure):
    body = f"""
from core.inference.windows_sandbox import launch
script.write_text("from pathlib import Path; Path('payload-ran').touch()", encoding='utf-8')
{LAUNCH}
original_gate = launch.authorize_startup
if {failure!r} == 'cancel':
    owner.cancel = threading.Event()
    owner.cancel.set()
elif {failure!r} == 'stdio':
    kwargs['close_fds'] = False
else:
    # The protocol owns these handles on entry. Preserve that ownership when
    # injecting failure before ACK; a resumed helper must not execute payload.
    def failed_gate(process, status, ack, *args, **kwargs):
        error = launch.WindowsRuntimeError('WINDOWS_SANDBOX_PROTOCOL_MISMATCH', 'injected gate failure')
        error.retained_control_handles = (status, ack)
        raise error
    launch.authorize_startup = failed_gate
try:
    try:
        spawn_prepared_launch(prepared, **kwargs)
    except launch.WindowsRuntimeError:
        pass
    else:
        raise AssertionError('Invalid launch returned a process')
    assert prepared.execution_record is None
    assert owner.closed and identity.cleaned and not manifest.exists()
    assert not (work/'payload-ran').exists()
    try:
        spawn_prepared_launch(prepared, **kwargs)
    except launch.WindowsRuntimeError as error:
        assert 'reused or replayed' in str(error)
    else:
        raise AssertionError('Failed launch replayed')
finally:
    launch.authorize_startup = original_gate
    prepared.cleanup()
assert not prepared.cleanup_diagnostics, prepared.cleanup_diagnostics
assert not list((root/'cache'/'.readers').iterdir())
print('FAILED_LAUNCH_CLEAN')
"""
    assert "FAILED_LAUNCH_CLEAN" in run_harness(installed_runtime, tmp_path, body)


@pytest.mark.parametrize("raw_handles", [False, True])
def test_discarded_creation_error_keeps_native_owner_and_blocks_new_launch(
    installed_runtime, tmp_path, raw_handles
):
    body = f"""
import gc, weakref
from core.inference.windows_sandbox import launch
script.write_text("from pathlib import Path; Path('payload-ran').touch()", encoding='utf-8')
{LAUNCH}
original_create = launch.create_suspended_host
original_snapshot = launch.prepare_runtime_snapshot
retained = []
def fail_after_creation(*args, **kwargs):
    process = original_create(*args, **kwargs)
    retained.append((process, process._unsloth_job.terminate))
    process._unsloth_job.terminate = lambda: False
    error = launch.WindowsRuntimeError('WINDOWS_SANDBOX_CLEANUP_FAILED', 'injected creation failure')
    if {raw_handles!r}:
        error.retained_job = process._unsloth_job
        error.retained_native_handles = (process._handle, process._thread_handle)
        error.retained_token_handles = (process._startup_token,)
        assert process._startup_token is not None
        process._startup_token = None  # Raw error now owns every native handle.
    else:
        error.retained_processes = (process,)
    raise error
launch.create_suspended_host = fail_after_creation
reference = weakref.ref(owner)
def tool_like_call(prepared):
    try:
        spawn_prepared_launch(prepared, **kwargs)
    except Exception as error:
        return 'Execution error: ' + str(error)
    finally:
        prepared.cleanup()
try:
    assert 'Execution error:' in tool_like_call(prepared)
    assert prepared.cleanup_diagnostics
    assert not prepared.cleanup_callbacks
    del owner, prepared, identity
    gc.collect()
    assert reference() is not None and reference() in launch._pending_cleanup
    assert retained[0][0].poll() is None
    assert manifest.exists() and not (work/'payload-ran').exists()
    def forbidden_snapshot(*args, **kwargs):
        raise AssertionError('Preparation worker ran despite an unresolved owner')
    launch.prepare_runtime_snapshot = forbidden_snapshot
    try:
        prepare_python_launch(spec, root/'different-cache')
    except launch.WindowsRuntimeError as error:
        assert error.code == 'WINDOWS_SANDBOX_CLEANUP_FAILED'
    else:
        raise AssertionError('Pending cleanup did not block the next launch')
finally:
    launch.create_suspended_host = original_create
    launch.prepare_runtime_snapshot = original_snapshot
    for process, terminate in retained:
        process._unsloth_job.terminate = terminate
    # Production preparation retries cleanup only, never the original command.
    retained_launch = reference()
    recovered = prepare_python_launch(spec, root/'cache')
    recovered.cleanup()
    assert retained_launch.closed and not retained_launch.handles
    assert not retained_launch.retained_processes and not retained_launch.retained_raw
    del retained_launch
assert not recovered.cleanup_diagnostics
assert not launch._pending_cleanup
gc.collect()
assert reference() is None
assert not manifest.exists() and not (work/'payload-ran').exists()
assert not list((root/'cache'/'.readers').iterdir())
print('DISCARDED_ERROR_OWNER_REAPED')
"""
    assert "DISCARDED_ERROR_OWNER_REAPED" in run_harness(installed_runtime, tmp_path, body)


@pytest.mark.parametrize("phase", ["preparation", "normal_exit"])
def test_discarded_acl_cleanup_failure_retains_identity(installed_runtime, tmp_path, phase):
    body = f"""
import gc, weakref
from core.inference.windows_sandbox import launch
script.write_text("from pathlib import Path; Path('payload-ran').touch()", encoding='utf-8')
spec = ToolLaunchPlan(argv=(sys.executable, '-u', str(script)), workdir=str(work), env={{}}, execution_kind='python')
original_prepare = launch._PythonLaunch.prepare
only_moniker = None
{PROFILE_FAILURE_SETUP}
references = []
def prepare_then_fail(owner, prepared):
    original_prepare(owner, prepared)
    references.append((weakref.ref(owner), Path(owner.identity.manifest_path)))
    if {phase!r} == 'preparation':
        raise launch.WindowsRuntimeError('WINDOWS_SANDBOX_LAUNCH_FAILED', 'injected preparation failure')
launch._PythonLaunch.prepare = prepare_then_fail
def tool_like_call():
    prepared = None
    try:
        prepared = prepare_python_launch(spec, root/'cache')
        process = spawn_prepared_launch(prepared, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, cwd=prepared.workdir, env=prepared.env, close_fds=True)
        assert process.wait(timeout=10) == 0
        return 'completed'
    except launch.WindowsRuntimeError as error:
        return error.code
    finally:
        if prepared is not None:
            prepared.cleanup()
            assert prepared.cleanup_diagnostics
try:
    result = tool_like_call()
    assert result == ('WINDOWS_SANDBOX_CLEANUP_FAILED' if {phase!r} == 'preparation' else 'completed')
    gc.collect()
    reference, manifest = references[0]
    assert reference() is not None and reference() in launch._pending_cleanup
    assert not reference().identity.cleaned and manifest.exists()
    assert (work/'payload-ran').exists() == ({phase!r} == 'normal_exit')
finally:
    launch._PythonLaunch.prepare = original_prepare
    preparation._run_worker = original_worker
    recovered = prepare_python_launch(spec, root/'cache')
    recovered.cleanup()
assert not recovered.cleanup_diagnostics and not launch._pending_cleanup
gc.collect()
assert reference() is None and not manifest.exists()
assert not list((root/'cache'/'.readers').iterdir())
assert (work/'payload-ran').exists() == ({phase!r} == 'normal_exit')
print('DISCARDED_ACL_OWNER_CLEANED')
"""
    assert "DISCARDED_ACL_OWNER_CLEANED" in run_harness(installed_runtime, tmp_path, body)


def test_concurrent_cleanup_retries_release_identity_once(installed_runtime, tmp_path):
    body = f"""
from concurrent.futures import ThreadPoolExecutor
from core.inference.windows_sandbox import launch, preparation
script.write_text('raise AssertionError("Payload must not run")', encoding='utf-8')
{LAUNCH}
original_cleanup = preparation.cleanup_invocation_profile
entered, release = threading.Event(), threading.Event()
calls = []
def held_cleanup(*args, **kwargs):
    calls.append(1)
    entered.set()
    assert release.wait(10), 'Test did not release identity cleanup'
    original_cleanup(*args, **kwargs)
preparation.cleanup_invocation_profile = held_cleanup
try:
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(owner.cleanup)
        try:
            assert entered.wait(10)
            second = pool.submit(owner.cleanup)
        finally:
            release.set()
        first.result(timeout=10)
        second.result(timeout=10)
finally:
    preparation.cleanup_invocation_profile = original_cleanup
    release.set()
    prepared.cleanup()
assert calls == [1]
assert owner.closed and identity.cleaned and not manifest.exists()
assert not prepared.cleanup_diagnostics and not launch._pending_cleanup
assert not list((root/'cache'/'.readers').iterdir())
print('SERIAL_CLEANUP_OK')
"""
    assert "SERIAL_CLEANUP_OK" in run_harness(installed_runtime, tmp_path, body)


def test_pending_cleanup_is_rechecked_before_native_spawn(installed_runtime, tmp_path):
    body = f"""
from core.inference.windows_sandbox import launch
script.write_text("from pathlib import Path; Path('payload-ran').touch()", encoding='utf-8')
{LAUNCH}
second = prepare_python_launch(spec, root/'cache')
second_owner = second.spawn_callback.__self__
only_moniker = identity.moniker
{PROFILE_FAILURE_SETUP}
original_create = launch.create_suspended_host
def forbidden_create(*args, **kwargs):
    raise AssertionError('Native creation passed pending cleanup')
try:
    prepared.cleanup()
    assert prepared.cleanup_diagnostics and owner in launch._pending_cleanup
    launch.create_suspended_host = forbidden_create
    second_kwargs = dict(kwargs, env=second.env, cwd=second.workdir)
    try:
        spawn_prepared_launch(second, **second_kwargs)
    except launch.WindowsRuntimeError as error:
        assert error.code == 'WINDOWS_SANDBOX_CLEANUP_FAILED'
    else:
        raise AssertionError('Pending cleanup allowed another spawn')
    assert second_owner.closed and second_owner.process is None
    assert second.execution_record is None and not (work/'payload-ran').exists()
finally:
    launch.create_suspended_host = original_create
    preparation._run_worker = original_worker
    owner.cleanup()
    second.cleanup()
assert not launch._pending_cleanup and not second.cleanup_diagnostics
assert owner.closed and identity.cleaned and not manifest.exists()
assert not list((root/'cache'/'.readers').iterdir())
print('SPAWN_RECHECK_OK')
"""
    assert "SPAWN_RECHECK_OK" in run_harness(installed_runtime, tmp_path, body)


@pytest.mark.parametrize("failure_stage", ["terminate", "wait"])
def test_prepublication_worker_owner_survives_discarded_error(
    installed_runtime, tmp_path, failure_stage
):
    body = """
import gc, weakref
from core.inference.windows_sandbox import launch, preparation
script.write_text("raise AssertionError('Payload must not run')", encoding='utf-8')
spec = ToolLaunchPlan(argv=(sys.executable, '-u', str(script)), workdir=str(work), env={}, execution_kind='python')
original_snapshot = launch.prepare_runtime_snapshot
original_init = launch.lpac.WindowsLpacProcess.__init__
original_wait = launch.lpac.WindowsLpacProcess.wait
original_terminate = launch.lpac._WindowsJob.terminate
workers, references, waits = [], [], []

def capture_worker(process, *args):
    original_init(process, *args)
    workers.append(process)
def failed_wait(process, timeout=None):
    waits.append(process)
    raise subprocess.TimeoutExpired(process.args, timeout)
def blocked_publication(*args, **kwargs):
    return preparation._run_worker(
        [sys._base_executable, '-I', '-S', '-c', 'import time; time.sleep(60)'],
        {}, str(root), deadline=time.monotonic()+0.5, cancel=None)
launch.prepare_runtime_snapshot = blocked_publication
launch.lpac.WindowsLpacProcess.__init__ = capture_worker
launch.lpac.WindowsLpacProcess.wait = failed_wait
if FAILURE_STAGE == 'terminate':
    launch.lpac._WindowsJob.terminate = lambda self: False
def tool_like_call():
    try:
        prepare_python_launch(spec, root/'cache')
    except launch.WindowsRuntimeError as error:
        owner = getattr(error, 'retained_launch', None)
        references.append(weakref.ref(owner) if owner is not None else None)
        return error.code
try:
    assert tool_like_call() == 'WINDOWS_SANDBOX_CLEANUP_FAILED'
    gc.collect()
    assert len(workers) == 1
    if FAILURE_STAGE == 'terminate':
        assert not waits and workers[0].poll() is None
    else:
        assert waits and all(process is workers[0] for process in waits)
    assert references[0] is not None, 'Pre-publication owner was lost'
    assert references[0]() in launch._pending_cleanup
    assert references[0]().identity is None and references[0]().access is None
    def forbidden_snapshot(*args, **kwargs):
        raise AssertionError('Unreaped publication worker allowed another worker')
    launch.prepare_runtime_snapshot = forbidden_snapshot
    try:
        prepare_python_launch(spec, root/'cache')
    except launch.WindowsRuntimeError as error:
        assert error.code == 'WINDOWS_SANDBOX_CLEANUP_FAILED'
    else:
        raise AssertionError('Another launch passed unreaped worker')
finally:
    launch.prepare_runtime_snapshot = original_snapshot
    launch.lpac.WindowsLpacProcess.__init__ = original_init
    launch.lpac.WindowsLpacProcess.wait = original_wait
    launch.lpac._WindowsJob.terminate = original_terminate
    for process in workers:
        if process._handle:
            process.terminate()
            process.wait(timeout=5)
    # Keep the failed assertion visible, but never strand its real worker.
    def cleanup_retained():
        for owner in tuple(launch._pending_cleanup):
            owner.cleanup()
    cleanup_retained()
    for process in workers:
        process.close()
assert not launch._pending_cleanup
gc.collect()
assert references[0]() is None
assert not (root/'cache').exists(), 'A failed publication was replayed'
print('PUBLICATION_OWNER_REAPED')
"""
    body = f"FAILURE_STAGE = {failure_stage!r}\n" + body
    assert "PUBLICATION_OWNER_REAPED" in run_harness(installed_runtime, tmp_path, body)


@pytest.mark.parametrize("running", [False, True])
def test_read_lease_cleanup_retry_does_not_reterminate_closed_process(
    installed_runtime, tmp_path, running
):
    worker_source = """
import sys
sys.path.insert(0, sys.argv.pop(1))
from core.inference.windows_sandbox import content_access
from core.inference.windows_sandbox.reader_cleanup_worker import main
original = content_access._change
def failed_revoke(*args, **kwargs):
    if kwargs.get('remove'):
        raise OSError('injected read-grant removal failure after process close')
    return original(*args, **kwargs)
content_access._change = failed_revoke
raise SystemExit(main())
"""
    payload = "from pathlib import Path; p=Path('payload-count'); p.write_text(str(int(p.read_text())+1) if p.exists() else '1'); print('READY', flush=True)"
    if running:
        payload += "; import time; time.sleep(60)"
    body = f"""
from core.inference.windows_sandbox import launch, preparation
script.write_text({payload!r}, encoding='utf-8')
{LAUNCH}
process = spawn_prepared_launch(prepared, **kwargs)
assert process.stdout.readline().strip() == 'READY'
if not {running!r}:
    assert process.wait(timeout=10) == 0
else:
    assert process.poll() is None
worker = root / 'fixed-reader-failure.py'
worker.write_text({worker_source!r}, encoding='utf-8')
original_worker = preparation._run_worker
def failed_worker(argv, *args, **kwargs):
    assert argv[1:4] == ['-I', '-S', '-B']
    assert argv[4].endswith('reader_cleanup_worker.py')
    return original_worker([*argv[:4], str(worker), str(Path(launch.__file__).parents[3]), *argv[5:]], *args, **kwargs)
preparation._run_worker = failed_worker
try:
    prepared.cleanup()
    assert prepared.cleanup_diagnostics
    assert 'injected read-grant removal failure' in str(prepared.cleanup_diagnostics)
    assert process.returncode is not None and process._handle is None
    assert owner in launch._pending_cleanup and not owner.closed
    # Recovery needs the reader lock released, but keeps its exact journal and
    # snapshot pins until the bounded worker confirms that revocation finished.
    assert owner.access.process is None and not owner.access.pins.handles
    assert owner.pins.handles
    assert not identity.cleaned and manifest.exists()
    assert list((root/'cache'/'.readers').iterdir())
    preparation._run_worker = original_worker
    owner.cleanup()
    assert owner.closed and identity.cleaned and not manifest.exists()
finally:
    preparation._run_worker = original_worker
    # A red regression must still release the test's real ACLs and identity.
    if owner.process is not None and owner.process._handle is None:
        owner.process = None
    owner.cleanup()
assert not launch._pending_cleanup
assert not list((root/'cache'/'.readers').iterdir())
assert (work/'payload-count').read_text() == '1', 'Cleanup replayed payload'
print('PARTIAL_READ_CLEANUP_OK')
"""
    assert "PARTIAL_READ_CLEANUP_OK" in run_harness(installed_runtime, tmp_path, body)


@pytest.mark.parametrize("failure_stage", ["terminate", "wait"])
@pytest.mark.parametrize("kind", ["reader", "profile"])
def test_unreaped_cleanup_worker_stays_owned_and_blocks_launch(
    installed_runtime, tmp_path, failure_stage, kind
):
    body = f"""
from core.inference.windows_sandbox import launch, preparation
script.write_text("from pathlib import Path; p=Path('payload-count'); p.write_text(str(int(p.read_text())+1) if p.exists() else '1'); print('READY', flush=True)", encoding='utf-8')
{LAUNCH}
process = spawn_prepared_launch(prepared, **kwargs)
assert process.stdout.readline().strip() == 'READY'
assert process.wait(timeout=10) == 0
# Close the actual payload first; injected failures below concern only the
# newly created cleanup worker, not the already-reaped sandbox process.
owner.access.process.close()
owner.access.process = owner.process = None
if {kind!r} == 'profile':
    owner.access.close()
    owner.access = None
original_worker = preparation._run_worker
original_init = launch.lpac.WindowsLpacProcess.__init__
original_wait = launch.lpac.WindowsLpacProcess.wait
original_terminate = launch.lpac._WindowsJob.terminate
workers, watched = [], []
kernel = launch.lpac._api().kernel32
def capture_worker(worker, *args):
    original_init(worker, *args)
    workers.append(worker)
    handle = kernel.OpenProcess(0x100000, False, worker.pid)
    assert handle
    watched.append(handle)
def stalled_worker(argv, *args, **kwargs):
    assert argv[4].endswith({kind!r} + '_cleanup_worker.py')
    return original_worker([*argv[:4], '-c', 'import os; r,w=os.pipe(); os.read(r,1)'],
        {{}}, str(root), deadline=time.monotonic()+0.5, cancel=None)
def failed_wait(worker, timeout=None):
    raise subprocess.TimeoutExpired(worker.args, timeout)
preparation._run_worker = stalled_worker
launch.lpac.WindowsLpacProcess.__init__ = capture_worker
launch.lpac.WindowsLpacProcess.wait = failed_wait
if {failure_stage!r} == 'terminate':
    launch.lpac._WindowsJob.terminate = lambda self: False
try:
    prepared.cleanup()
    assert prepared.cleanup_diagnostics
    assert len(workers) == 1 and owner.retained_processes == workers
    assert owner in launch._pending_cleanup and not owner.closed
    assert manifest.exists() and not identity.cleaned
    if {kind!r} == 'reader':
        assert owner.pins.handles and not owner.access.pins.handles
    else:
        assert not owner.pins.handles and owner.access is None
    if {failure_stage!r} == 'terminate':
        assert workers[0].poll() is None
    def forbidden_worker(*args, **kwargs):
        raise AssertionError('An unreaped cleanup worker permitted another worker')
    preparation._run_worker = forbidden_worker
    try:
        prepare_python_launch(spec, root/'cache')
    except launch.WindowsRuntimeError as error:
        assert error.code == 'WINDOWS_SANDBOX_CLEANUP_FAILED'
    else:
        raise AssertionError('Pending worker did not block launch')
    assert owner.retained_processes == workers and len(workers) == 1
finally:
    preparation._run_worker = original_worker
    launch.lpac.WindowsLpacProcess.__init__ = original_init
    launch.lpac.WindowsLpacProcess.wait = original_wait
    launch.lpac._WindowsJob.terminate = original_terminate
    try:
        # Retained ownership, not the test's observation handle, must reap it.
        owner.cleanup()
        assert watched and all(kernel.WaitForSingleObject(h,0) == 0 for h in watched)
    finally:
        for worker in workers:
            worker.reap(timeout=5)
            worker.close()
        for handle in watched:
            assert kernel.CloseHandle(handle)
assert owner.closed and identity.cleaned and not manifest.exists()
assert not owner.retained_processes and not launch._pending_cleanup
assert not list((root/'cache'/'.readers').iterdir())
assert (work/'payload-count').read_text() == '1', 'Cleanup replayed payload'
print('READER_CLEANUP_WORKER_REAPED')
"""
    assert "READER_CLEANUP_WORKER_REAPED" in run_harness(installed_runtime, tmp_path, body)


@pytest.mark.parametrize("stage", ["before", "publication", "native"])
def test_explicit_cancellation_reaches_preparation_and_native_startup(
    installed_runtime, tmp_path, stage
):
    body = f"""
import ctypes
from core.inference.windows_sandbox import launch
script.write_text("from pathlib import Path; Path('payload-ran').touch()", encoding='utf-8')
cancel = threading.Event()
spec = ToolLaunchPlan(argv=(sys.executable, '-u', str(script)), workdir=str(work), env={{}}, execution_kind='python')
api = launch.lpac._api().kernel32
original_create = api.CreateProcessW
original_native = launch.create_suspended_host
original_snapshot = launch.prepare_runtime_snapshot
watched, owners = [], []
def observe_and_cancel(pid):
    handle = api.OpenProcess(0x100000, False, pid)
    assert handle, 'Could not retain an observation handle'
    watched.append(handle)
    cancel.set()
def cancel_worker_creation(*args):
    result = original_create(*args)
    if result:
        info = ctypes.cast(args[-1], ctypes.POINTER(launch.lpac._PROCESS_INFORMATION)).contents
        observe_and_cancel(int(info.dwProcessId))
    return result
def cancel_native_creation(*args, **kwargs):
    process = original_native(*args, **kwargs)
    observe_and_cancel(process.pid)
    return process
def forbidden_snapshot(*args, **kwargs):
    raise AssertionError('Cancelled plan attempted publication')
if {stage!r} == 'before':
    cancel.set()
    launch.prepare_runtime_snapshot = forbidden_snapshot
elif {stage!r} == 'publication':
    api.CreateProcessW = cancel_worker_creation
else:
    launch.create_suspended_host = cancel_native_creation
prepared = None
try:
    try:
        prepared = prepare_python_launch(spec, root/'cache', cancel=cancel)
        owner = prepared.spawn_callback.__self__
        owners.append(owner)
        assert owner.cancel is cancel
        spawn_prepared_launch(prepared, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, cwd=prepared.workdir, env=prepared.env, close_fds=True)
    except launch.WindowsRuntimeError as error:
        expected = 'WINDOWS_SANDBOX_CANCELLED'
        assert error.code == expected, str(error)
    else:
        raise AssertionError('Cancelled plan returned a process')
    # Cancelling preparation also runs its fixed recovery worker, even when no
    # preparation reply arrived. Both exact process handles must be signalled.
    assert len(watched) == (2 if {stage!r} == 'publication' else 1 if {stage!r} == 'native' else 0)
    assert all(api.WaitForSingleObject(handle, 0) == 0 for handle in watched)
    assert not (work/'payload-ran').exists()
    assert prepared is None or prepared.execution_record is None
finally:
    api.CreateProcessW = original_create
    launch.create_suspended_host = original_native
    launch.prepare_runtime_snapshot = original_snapshot
    if prepared is not None:
        prepared.cleanup()
        assert not prepared.cleanup_diagnostics
    for handle in watched:
        api.CloseHandle(handle)
assert all(owner.closed for owner in owners)
assert not launch._pending_cleanup
if (root/'cache'/'.readers').exists():
    assert not list((root/'cache'/'.readers').iterdir())
print('PLAN_CANCELLED_WITHOUT_PAYLOAD')
"""
    assert "PLAN_CANCELLED_WITHOUT_PAYLOAD" in run_harness(installed_runtime, tmp_path, body)


@pytest.mark.parametrize("channel", ["status_read", "ack_write"])
def test_gate_failed_close_transfers_handle_to_launch_cleanup(installed_runtime, tmp_path, channel):
    body = f"""
import ctypes
from core.inference.windows_sandbox import launch, protocol
script.write_text("from pathlib import Path; Path('payload-ran').touch()", encoding='utf-8')
{LAUNCH}
failed_handle = getattr(owner, {channel!r})
api = launch.lpac._api().kernel32
original_pipe_api = protocol._pipe_api
original_close = owner._close_handle
adopted = []
class CloseFailure:
    def __getattr__(self, name):
        return getattr(api, name)
    def CloseHandle(self, handle):
        if handle == failed_handle:
            ctypes.set_last_error(5)
            return False
        return api.CloseHandle(handle)
def tracked_close(handle):
    if handle == failed_handle:
        assert handle in owner.handles, 'The launch did not adopt its failed control close'
        adopted.append(handle)
    return original_close(handle)
protocol._pipe_api = CloseFailure
owner._close_handle = tracked_close
try:
    try:
        spawn_prepared_launch(prepared, **kwargs)
    except launch.WindowsRuntimeError as error:
        assert error.code == 'WINDOWS_SANDBOX_CLEANUP_FAILED'
    else:
        raise AssertionError('Failed control close returned a launch')
    assert adopted == [failed_handle]
    assert owner.closed and not owner.handles
    assert prepared.execution_record is None and not (work/'payload-ran').exists()
finally:
    protocol._pipe_api = original_pipe_api
    owner._close_handle = original_close
    prepared.cleanup()
assert identity.cleaned and not manifest.exists()
assert not launch._pending_cleanup and not prepared.cleanup_diagnostics
assert not list((root/'cache'/'.readers').iterdir())
print('GATE_CLOSE_OWNER_CLEANED')
"""
    assert "GATE_CLOSE_OWNER_CLEANED" in run_harness(installed_runtime, tmp_path, body)


def test_nested_creation_cleanup_failure_preserves_both_processes(installed_runtime, tmp_path):
    body = f"""
from core.inference.windows_sandbox import launch
script.write_text("from pathlib import Path; Path('payload-ran').touch()", encoding='utf-8')
{LAUNCH}
original_adapter = launch.lpac.WindowsLpacProcess
original_terminate = launch.lpac._WindowsJob.terminate
allocations = []
def fail_target_adapter(*args):
    allocations.append(args[3])
    if len(allocations) == 2:
        raise RuntimeError('injected target adapter failure')
    return original_adapter(*args)
launch.lpac.WindowsLpacProcess = fail_target_adapter
launch.lpac._WindowsJob.terminate = lambda job: False
try:
    try:
        spawn_prepared_launch(prepared, **kwargs)
    except launch.WindowsRuntimeError as error:
        assert error.retained_launch is owner
    else:
        raise AssertionError('Broken process allocation succeeded')
    assert len(allocations) == 2
    assert len(owner.retained_processes) == len(owner.retained_raw) == 1
    assert owner.retained_processes[0].poll() is None
    assert launch.lpac._api().kernel32.WaitForSingleObject(owner.retained_raw[0][1][0], 0) == 258
    assert not owner.access.closed and manifest.exists()
    assert prepared.execution_record is None and not (work/'payload-ran').exists()
finally:
    launch.lpac.WindowsLpacProcess = original_adapter
    launch.lpac._WindowsJob.terminate = original_terminate
    prepared.cleanup()
assert not prepared.cleanup_diagnostics, prepared.cleanup_diagnostics
assert owner.closed and identity.cleaned and not manifest.exists()
assert not owner.retained_processes and not owner.retained_raw
assert not list((root/'cache'/'.readers').iterdir())
assert not (work/'payload-ran').exists()
print('BOTH_CREATION_OWNERS_REAPED')
"""
    assert "BOTH_CREATION_OWNERS_REAPED" in run_harness(installed_runtime, tmp_path, body)


def test_running_process_timeout_terminates_before_cleanup(installed_runtime, tmp_path):
    payload = "import time; from pathlib import Path; print('RUNNING', flush=True); time.sleep(60); Path('late-write').touch()"
    body = f"""
script.write_text({payload!r}, encoding='utf-8')
{LAUNCH}
try:
    process = spawn_prepared_launch(prepared, **kwargs)
    assert process.stdout.readline().strip() == 'RUNNING'
    try:
        process.wait(timeout=0.05)
    except subprocess.TimeoutExpired:
        pass
    else:
        raise AssertionError('Running process unexpectedly exited')
    process.terminate()
    assert process.wait(timeout=5) != 0
finally:
    prepared.cleanup()
assert not prepared.cleanup_diagnostics, prepared.cleanup_diagnostics
assert owner.closed and identity.cleaned and not manifest.exists()
assert not list((root/'cache'/'.readers').iterdir())
assert not (work/'late-write').exists()
print('RUNNING_TIMEOUT_REAPED')
"""
    assert "RUNNING_TIMEOUT_REAPED" in run_harness(installed_runtime, tmp_path, body)


@pytest.mark.parametrize("invalid", ["terminal", "limited", "full", "overlap", "cancel"])
def test_invalid_preparation_does_not_create_identity_or_scan(installed_runtime, tmp_path, invalid):
    body = f"""
from dataclasses import replace
from core.inference.windows_sandbox import launch
script.write_text("raise AssertionError('payload ran')", encoding='utf-8')
spec = ToolLaunchPlan(argv=(sys.executable, '-u', str(script)), workdir=str(work), env={{}}, execution_kind='python')
store = root / 'cache'
cancel = threading.Event()
if {invalid!r} == 'terminal':
    spec = replace(spec, execution_kind='terminal')
elif {invalid!r} in ('limited', 'full'):
    spec = replace(spec, requested_mode={invalid!r})
elif {invalid!r} == 'overlap':
    store = work / 'cache'
else:
    cancel.set()
def unexpected(*args, **kwargs):
    raise AssertionError('Invalid preparation reached resource allocation')
if {invalid!r} != 'overlap':
    launch.prepare_runtime_snapshot = unexpected
launch.lpac._create_identity = unexpected
try:
    launch.prepare_python_launch(spec, store, cancel=cancel)
except launch.WindowsRuntimeError as error:
    if {invalid!r} == 'overlap':
        assert 'overlaps runtime storage' in str(error)
else:
    raise AssertionError('Invalid preparation succeeded')
assert not store.exists()
print('INVALID_PREPARATION_BLOCKED')
"""
    assert "INVALID_PREPARATION_BLOCKED" in run_harness(installed_runtime, tmp_path, body)


def test_successive_launches_have_private_temp_and_read_only_runtime(installed_runtime, tmp_path):
    payload = """import os, sqlite3
from pathlib import Path
temporary = Path(os.environ['TEMP'])
assert not (temporary / 'previous-call').exists()
(temporary / 'previous-call').write_text('private')
Path('work-write').write_text('persistent')
try:
    (Path(sqlite3.__file__).parent / 'runtime-write').write_text('forbidden')
except PermissionError:
    pass
else:
    raise AssertionError('Runtime was writable')
print(str(temporary), flush=True)
"""
    body = f"""
script.write_text({payload!r}, encoding='utf-8')
temps = []
for iteration in range(2):
{"".join("    " + line + chr(10) for line in LAUNCH.splitlines())}
    try:
        process = spawn_prepared_launch(prepared, **kwargs)
        output = process.stdout.read().strip()
        assert process.wait(timeout=10) == 0, output
        temps.append(output)
    finally:
        prepared.cleanup()
    assert not prepared.cleanup_diagnostics, prepared.cleanup_diagnostics
    assert owner.closed and identity.cleaned and not manifest.exists()
    assert not Path(temps[-1]).exists()
assert len(set(temps)) == 2
assert (work/'work-write').read_text() == 'persistent'
assert not list((root/'cache'/'.readers').iterdir())
print('FRESH_TEMP_READ_ONLY_RUNTIME')
"""
    assert "FRESH_TEMP_READ_ONLY_RUNTIME" in run_harness(installed_runtime, tmp_path, body)
