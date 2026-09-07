# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Real native handles with deterministic scheduling at the wait/close boundary."""

import sys

import pytest

from test_launch import LAUNCH, run_harness, installed_runtime, runtime_wheel

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows lifecycle lane")


@pytest.mark.parametrize("phase", ["wait", "poll", "query"])
def test_close_never_invalidates_a_concurrent_native_wait(installed_runtime, tmp_path, phase):
    body = f"""
import ctypes
from core.inference.windows_sandbox import launch
script.write_text("print('READY', flush=True); import time; time.sleep(60)", encoding='utf-8')
{LAUNCH}
process = spawn_prepared_launch(prepared, **kwargs)
assert process.stdout.readline().strip() == 'READY'
api = launch.lpac._api().kernel32
target = process._handle
original_wait, original_query, original_close = api.WaitForSingleObject, api.GetExitCodeProcess, api.CloseHandle
entered, release, closing_done = threading.Event(), threading.Event(), threading.Event()
unsafe, errors, results = [], [], []
waiter_id = None

def pause():
    entered.set()
    assert release.wait(10), 'Scheduling fixture was not released'

def waiting(handle, timeout):
    if handle == target and threading.get_ident() == waiter_id and {phase!r} != 'query':
        pause()
    return original_wait(handle, timeout)

def querying(handle, output):
    if handle == target and threading.get_ident() == waiter_id and {phase!r} == 'query':
        pause()
    return original_query(handle, output)

def closing(handle):
    if handle == target and entered.is_set() and not release.is_set():
        # Observe the bug without actually invoking undefined Win32 behavior.
        unsafe.append(handle)
        ctypes.set_last_error(5)
        return False
    return original_close(handle)

def wait_thread():
    global waiter_id
    waiter_id = threading.get_ident()
    try:
        results.append(process.poll() if {phase!r} == 'poll' else process.wait())
    except BaseException as error:
        errors.append(error)

def close_thread():
    try:
        process.close()
    except BaseException as error:
        errors.append(error)
    finally:
        closing_done.set()

api.WaitForSingleObject, api.GetExitCodeProcess, api.CloseHandle = waiting, querying, closing
waiter = threading.Thread(target=wait_thread, daemon=True)
closer = threading.Thread(target=close_thread, daemon=True)
try:
    if {phase!r} == 'query':
        process.terminate()
        assert original_wait(target, 5000) == 0
    waiter.start()
    assert entered.wait(5)
    closer.start()
    # Job termination must progress even though the waiter has not returned.
    assert original_wait(target, 5000) == 0
    assert not closing_done.wait(0.1), 'Close returned while a process operation still used its handle'
    assert not unsafe, 'CloseHandle reached a handle used by another thread'
finally:
    release.set()
    waiter.join(5)
    if closer.ident is not None:
        closer.join(5)
    api.WaitForSingleObject, api.GetExitCodeProcess, api.CloseHandle = original_wait, original_query, original_close
    prepared.cleanup()
assert not waiter.is_alive() and not closer.is_alive()
assert not errors, errors
assert not unsafe and len(results) == 1 and results[0] is not None
assert process.poll() == results[0] and process.wait(timeout=0) == results[0]
assert process._handle is None and process._thread_handle is None
assert not prepared.cleanup_diagnostics and owner.closed
print('CONCURRENT_WAIT_CLOSE_OK')
"""
    assert "CONCURRENT_WAIT_CLOSE_OK" in run_harness(installed_runtime, tmp_path, body)


def test_stalled_wait_retains_handles_for_bounded_close_retry(installed_runtime, tmp_path):
    body = f"""
from core.inference.windows_sandbox import launch
script.write_text("print('READY', flush=True); import time; time.sleep(60)", encoding='utf-8')
{LAUNCH}
process = spawn_prepared_launch(prepared, **kwargs)
assert process.stdout.readline().strip() == 'READY'
api = launch.lpac._api().kernel32
target, thread = process._handle, process._thread_handle
original_wait, original_bound = api.WaitForSingleObject, launch.lpac._PROCESS_CLOSE_TIMEOUT
entered, release = threading.Event(), threading.Event()
errors, results = [], []
waiter_id = None

def waiting(handle, timeout):
    if handle == target and threading.get_ident() == waiter_id:
        entered.set()
        assert release.wait(10)
    return original_wait(handle, timeout)

def wait_thread():
    global waiter_id
    waiter_id = threading.get_ident()
    try:
        results.append(process.wait())
    except BaseException as error:
        errors.append(error)

api.WaitForSingleObject = waiting
launch.lpac._PROCESS_CLOSE_TIMEOUT = 0.1
waiter = threading.Thread(target=wait_thread, daemon=True)
try:
    waiter.start()
    assert entered.wait(5)
    started = time.monotonic()
    try:
        process.close()
    except launch.lpac.SandboxUnavailableError as error:
        assert 'active native wait' in str(error)
    else:
        raise AssertionError('Stalled waiter ownership was discarded')
    assert time.monotonic() - started < 2
    assert process._handle == target and process._thread_handle == thread
    assert process._unsloth_job._handle is None
    assert original_wait(target, 5000) == 0, 'Bounded close failed to stop the Job'
    assert process.stdout is not None and not process.stdout.closed
    # Admission is restored after a failed close, without closing its handle.
    assert process.poll() is not None
finally:
    release.set()
    waiter.join(5)
    api.WaitForSingleObject = original_wait
    launch.lpac._PROCESS_CLOSE_TIMEOUT = original_bound
    prepared.cleanup()
assert not waiter.is_alive() and not errors and len(results) == 1
assert process._handle is None and process._thread_handle is None
assert not prepared.cleanup_diagnostics and owner.closed
print('BOUNDED_CONCURRENT_CLOSE_RETRY_OK')
"""
    assert "BOUNDED_CONCURRENT_CLOSE_RETRY_OK" in run_harness(installed_runtime, tmp_path, body)


def test_job_close_waits_for_concurrent_terminate_without_closing_its_handle(monkeypatch):
    import ctypes
    import threading
    from core.inference.windows_sandbox import native_compat as lpac

    api = lpac._api().kernel32
    job = lpac._create_job(None, active_process_limit = 1)
    target = job._handle
    original_terminate, original_close = api.TerminateJobObject, api.CloseHandle
    entered, release, close_done = threading.Event(), threading.Event(), threading.Event()
    unsafe, errors, results = [], [], []

    def terminate(handle, code):
        assert handle == target
        entered.set()
        assert release.wait(5)
        return original_terminate(handle, code)

    def close(handle):
        if handle == target and entered.is_set() and not release.is_set():
            unsafe.append(handle)
            ctypes.set_last_error(5)
            return False
        return original_close(handle)

    def terminating():
        try:
            results.append(job.terminate())
        except BaseException as error:
            errors.append(error)

    def closing():
        try:
            job.close()
        except BaseException as error:
            errors.append(error)
        finally:
            close_done.set()

    monkeypatch.setattr(api, "TerminateJobObject", terminate)
    monkeypatch.setattr(api, "CloseHandle", close)
    terminator = threading.Thread(target = terminating, daemon = True)
    closer = threading.Thread(target = closing, daemon = True)
    try:
        terminator.start()
        assert entered.wait(2)
        closer.start()
        assert not close_done.wait(0.1)
        assert not unsafe
    finally:
        release.set()
        terminator.join(5)
        if closer.ident is not None:
            closer.join(5)
        monkeypatch.setattr(api, "TerminateJobObject", original_terminate)
        monkeypatch.setattr(api, "CloseHandle", original_close)
        job.close()
    assert not errors and not unsafe and results == [True]
    assert not terminator.is_alive() and not closer.is_alive()
    assert job._handle is None and not job.terminate()
