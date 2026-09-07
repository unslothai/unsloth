# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native close failures retain ownership; these are not qualification probes."""

import sys

import pytest

from test_launch import LAUNCH, run_harness, installed_runtime, runtime_wheel

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows cleanup lane")


@pytest.mark.parametrize("kind", ["job", "thread", "process", "stdout"])
def test_live_close_failure_retains_exact_owner_until_retry(installed_runtime, tmp_path, kind):
    body = f"""
import ctypes
from ctypes import wintypes as W
from types import SimpleNamespace
from core.inference.windows_sandbox import launch
script.write_text("print('READY', flush=True); import time; time.sleep(60)", encoding='utf-8')
{LAUNCH}
process = spawn_prepared_launch(prepared, **kwargs)
assert process.stdout.readline().strip() == 'READY'
assert process.poll() is None
api = launch.lpac._api().kernel32
original_close = api.CloseHandle
stream = process.stdout
handles = dict(job=process._unsloth_job._handle, thread=process._thread_handle, process=process._handle)
target = handles.get({kind!r})
emergency_close = False
def failed_close(handle):
    if handle == target:
        ctypes.set_last_error(5)
        return False
    return original_close(handle)
def failed_stream_close():
    raise OSError('injected stream close failure')
if target is None:
    process.stdout = SimpleNamespace(close=failed_stream_close)
else:
    api.CloseHandle = failed_close
try:
    prepared.cleanup()
    emergency_close = target is not None and not prepared.cleanup_diagnostics
    assert prepared.cleanup_diagnostics, 'The native close failure was silently discarded'
    assert owner in launch._pending_cleanup and not owner.closed
    assert owner.access.process is process and not owner.access.closed
    assert process.returncode is not None
    if {kind!r} == 'job':
        assert process._unsloth_job._handle == target
    elif {kind!r} == 'thread':
        assert process._unsloth_job._handle is None and process._thread_handle == target
    elif {kind!r} == 'process':
        assert process._unsloth_job._handle is None and process._thread_handle is None
        assert process._handle == target
    else:
        assert process._unsloth_job._handle is None and process.stdout is not None
    assert owner.access.pins.handles and manifest.exists() and not identity.cleaned
finally:
    api.CloseHandle = original_close
    process.stdout = stream
    owner.cleanup()
    if emergency_close:
        # The old broken implementation forgets this still-open native handle.
        assert original_close(target)
assert owner.closed and identity.cleaned and not manifest.exists()
assert not launch._pending_cleanup
assert process._unsloth_job._handle is None and process._handle is None and process._thread_handle is None
assert process.stdout is None and stream.closed
assert not list((root/'cache'/'.readers').iterdir())
print('CHECKED_CLOSE_RETRIED')
"""
    assert "CHECKED_CLOSE_RETRIED" in run_harness(installed_runtime, tmp_path, body)


@pytest.mark.parametrize("kind", ["runtime", "sentinel"])
def test_failed_pin_close_retains_lease_until_retry(installed_runtime, tmp_path, kind):
    body = f"""
import ctypes
from core.inference.windows_sandbox import launch
script.write_text("print('PINNED', flush=True)", encoding='utf-8')
{LAUNCH}
process = spawn_prepared_launch(prepared, **kwargs)
assert process.stdout.readline().strip() == 'PINNED' and process.wait(timeout=5) == 0
pins = owner.access.pins if {kind!r} == 'runtime' else owner.file_pins
path, target = tuple(pins.handles.items())[-1]
original_close = pins.api.kernel.CloseHandle
failures = []
def failed_close(handle):
    if handle == target:
        failures.append(handle)
        ctypes.set_last_error(5)
        return False
    return original_close(handle)
pins.api.kernel.CloseHandle = failed_close
try:
    prepared.cleanup()
    assert failures, 'The targeted native pin close was not exercised'
    assert prepared.cleanup_diagnostics and owner in launch._pending_cleanup
    assert pins.handles.get(path) == target, 'Failed pin ownership was discarded'
    assert not owner.closed and not identity.cleaned
finally:
    pins.api.kernel.CloseHandle = original_close
    if target not in pins.handles.values():
        # Red-test recovery for the old implementation's forgotten live pin.
        assert original_close(target)
    owner.cleanup()
assert owner.closed and identity.cleaned and not manifest.exists()
assert not pins.handles and not launch._pending_cleanup
assert not list((root/'cache'/'.readers').iterdir())
print('PIN_CLOSE_RETRIED')
"""
    assert "PIN_CLOSE_RETRIED" in run_harness(installed_runtime, tmp_path, body)


def test_missing_job_cannot_reap_a_running_process(installed_runtime, tmp_path):
    body = f"""
script.write_text("print('READY', flush=True); import time; time.sleep(60)", encoding='utf-8')
{LAUNCH}
process = spawn_prepared_launch(prepared, **kwargs)
assert process.stdout.readline().strip() == 'READY' and process.poll() is None
job = process._unsloth_job
handle = job._handle
try:
    job._handle = None
    try:
        process.reap(timeout=0)
    except Exception as error:
        assert 'without its owned Job' in str(error)
    else:
        raise AssertionError('Missing Job ownership was treated as reaped')
    assert not process._reaped and process.returncode is None
    assert process.poll() is None
finally:
    job._handle = handle
    prepared.cleanup()
assert not prepared.cleanup_diagnostics and owner.closed
print('MISSING_JOB_REJECTED')
"""
    assert "MISSING_JOB_REJECTED" in run_harness(installed_runtime, tmp_path, body)


def test_signalled_process_exit_259_is_cached_after_native_close(installed_runtime, tmp_path):
    body = f"""
script.write_text("import ctypes; ctypes.WinDLL('kernel32').ExitProcess(259)", encoding='utf-8')
{LAUNCH}
try:
    process = spawn_prepared_launch(prepared, **kwargs)
    assert process.wait(timeout=5) == 259
    assert process.returncode == 259, 'Signalled process exit was mistaken for STILL_ACTIVE'
finally:
    prepared.cleanup()
assert not prepared.cleanup_diagnostics and owner.closed
assert process.poll() == 259 and process.wait(timeout=0) == 259
print('EXIT_259_REAPED')
"""
    assert "EXIT_259_REAPED" in run_harness(installed_runtime, tmp_path, body)


def test_failed_native_poll_does_not_query_or_report_exit(installed_runtime, tmp_path):
    body = f"""
import ctypes
from core.inference.windows_sandbox import launch
script.write_text("print('READY', flush=True); import time; time.sleep(60)", encoding='utf-8')
{LAUNCH}
process = spawn_prepared_launch(prepared, **kwargs)
assert process.stdout.readline().strip() == 'READY' and process.poll() is None
api = launch.lpac._api().kernel32
original_wait, original_query = api.WaitForSingleObject, api.GetExitCodeProcess
queries = []
def failed_wait(*args):
    ctypes.set_last_error(5)
    return 0xffffffff
def query(*args):
    queries.append(1)
    return original_query(*args)
api.WaitForSingleObject, api.GetExitCodeProcess = failed_wait, query
try:
    try:
        process.poll()
    except Exception as error:
        assert 'WaitForSingleObject' in str(error)
    else:
        raise AssertionError('A failed wait was treated as process exit')
    assert not queries and process.returncode is None
finally:
    api.WaitForSingleObject, api.GetExitCodeProcess = original_wait, original_query
    prepared.cleanup()
assert not prepared.cleanup_diagnostics and owner.closed
print('FAILED_WAIT_IS_NOT_EXIT')
"""
    assert "FAILED_WAIT_IS_NOT_EXIT" in run_harness(installed_runtime, tmp_path, body)


@pytest.mark.parametrize("phase", ["publication", "raw_publication", "raw_native"])
@pytest.mark.parametrize("kind", ["job", "thread", "process"])
def test_creation_close_failure_transfers_partial_owner(installed_runtime, tmp_path, phase, kind):
    initial = (
        LAUNCH
        if phase == "raw_native"
        else """
spec = ToolLaunchPlan(argv=(sys.executable, '-u', str(script)), workdir=str(work), env={}, execution_kind='python')
prepared = None
"""
    )
    body = f"""
import ctypes
from core.inference.windows_sandbox import launch
script.write_text("from pathlib import Path; Path('payload-ran').touch()", encoding='utf-8')
{initial}
original_adapter = launch.lpac.WindowsLpacProcess
api = launch.lpac._api().kernel32
original_close = api.CloseHandle
created, failed_handles = [], set()
retained_owner = None
def failed_close(handle):
    if handle in failed_handles:
        ctypes.set_last_error(5)
        return False
    return original_close(handle)
def adapter(*args):
    created.append(args)
    selected = len(created) == (2 if {phase!r} == 'raw_native' else 1)
    if selected:
        failed_handles.add({{'job':args[5]._handle, 'thread':args[2], 'process':args[1]}}[{kind!r}])
        # Also fail donor cleanup when the target retains only its Job. This
        # must not discard an earlier error with an empty native-handle list.
        if {phase!r} == 'raw_native' and {kind!r} == 'job':
            failed_handles.add(created[0][5]._handle)
        if {phase!r} != 'publication':
            raise MemoryError('injected adapter allocation failure')
    return original_adapter(*args)
launch.lpac.WindowsLpacProcess = adapter
api.CloseHandle = failed_close
try:
    try:
        if {phase!r} == 'raw_native':
            spawn_prepared_launch(prepared, **kwargs)
        else:
            prepared = prepare_python_launch(spec, root/'cache')
    except launch.WindowsRuntimeError as error:
        assert error.code == 'WINDOWS_SANDBOX_CLEANUP_FAILED'
        retained_owner = error.retained_launch
    else:
        raise AssertionError('Creation close failure was discarded')
    assert retained_owner in launch._pending_cleanup and not retained_owner.closed
    assert not (work/'payload-ran').exists()
    if {phase!r} == 'publication':
        assert len(retained_owner.retained_processes) == 1
        process = retained_owner.retained_processes[0]
        assert process.returncode == 0
        assert not retained_owner.retained_raw
    else:
        assert len(retained_owner.retained_raw) == 1
        job, pending = retained_owner.retained_raw[0]
        assert job._handle in failed_handles if {kind!r} == 'job' else job._handle is not None
        assert len(pending) == {{'job':0, 'thread':2, 'process':1}}[{kind!r}]
        assert len(retained_owner.retained_processes) == (1 if {phase!r} == 'raw_native' and {kind!r} == 'job' else 0)
finally:
    api.CloseHandle = original_close
    launch.lpac.WindowsLpacProcess = original_adapter
    if retained_owner is not None:
        retained_owner.cleanup()
    if prepared is not None:
        prepared.cleanup()
assert retained_owner.closed and not launch._pending_cleanup
assert not retained_owner.retained_processes and not retained_owner.retained_raw
assert not (work/'payload-ran').exists()
if (root/'cache'/'.readers').exists():
    assert not list((root/'cache'/'.readers').iterdir())
print('PARTIAL_CREATION_CLOSE_RETRIED')
"""
    assert "PARTIAL_CREATION_CLOSE_RETRIED" in run_harness(installed_runtime, tmp_path, body)
