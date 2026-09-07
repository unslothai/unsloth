# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native worker channel ownership, including simultaneous cleanup failures."""

import ctypes
import os
import subprocess
import sys

import pytest

from test_preparation import lpac, preparation, run_fixed, blocked_worker
from test_launch import run_harness, installed_runtime, runtime_wheel

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows worker channels")


def test_crt_close_failure_cannot_mask_retained_worker(tmp_path, monkeypatch):
    original_init = lpac.WindowsLpacProcess.__init__
    original_pipe, original_open, original_close = os.pipe, os.open, os.close
    owners, descriptors = [], set()

    def capture(owner, *args):
        original_init(owner, *args)
        owners.append(owner)

    def pipe():
        result = original_pipe()
        descriptors.update(result)
        return result

    def opened(*args, **kwargs):
        result = original_open(*args, **kwargs)
        descriptors.add(result)
        return result

    def close(fd):
        raise OSError("injected CRT close failure")

    def wait(owner, timeout = None):
        raise subprocess.TimeoutExpired(owner.args, timeout)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(lpac.WindowsLpacProcess, "__init__", capture)
            patch.setattr(lpac.WindowsLpacProcess, "wait", wait)
            patch.setattr(os, "pipe", pipe)
            patch.setattr(os, "open", opened)
            patch.setattr(os, "close", close)
            with pytest.raises(preparation.WindowsRuntimeError) as caught:
                run_fixed(blocked_worker(tmp_path), tmp_path, timeout = 0.3)
            assert caught.value.code == "WINDOWS_SANDBOX_CLEANUP_FAILED"
            assert caught.value.retained_process is owners[0]
            assert owners[0]._handle and owners[0]._unsloth_job._handle
            assert not descriptors, "Worker transport must not own CRT descriptors"
    finally:
        for owner in owners:
            owner.reap(timeout = 5)
            owner.close()
        for fd in descriptors:
            original_close(fd)
    assert not (tmp_path / "unexpected-completion").exists()


@pytest.mark.parametrize("failed_reap", [False, True])
def test_native_close_failure_retains_channels_and_process(tmp_path, monkeypatch, failed_reap):
    api = lpac._api().kernel32
    original_close, original_pipe = api.CloseHandle, preparation._WorkerChannels.pipe
    original_init = lpac.WindowsLpacProcess.__init__
    channels, readers, processes = [], [], []
    attempts = []

    def pipe(owner):
        read, write = original_pipe(owner)
        channels.append(owner)
        readers.append(read)
        return read, write

    def capture(owner, *args):
        original_init(owner, *args)
        processes.append(owner)

    def close(handle):
        if handle in readers:
            attempts.append(handle)
            ctypes.set_last_error(5)
            return False
        return original_close(handle)

    def wait(owner, timeout = None):
        raise subprocess.TimeoutExpired(owner.args, timeout)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(preparation._WorkerChannels, "pipe", pipe)
            patch.setattr(lpac.WindowsLpacProcess, "__init__", capture)
            patch.setattr(api, "CloseHandle", close)
            if failed_reap:
                patch.setattr(lpac.WindowsLpacProcess, "wait", wait)
            source = blocked_worker(tmp_path) if failed_reap else "print('completed')"
            with pytest.raises(preparation.WindowsRuntimeError) as caught:
                run_fixed(source, tmp_path, timeout = 0.5 if failed_reap else 5)
            error = caught.value
            assert error.code == "WINDOWS_SANDBOX_CLEANUP_FAILED"
            assert set(error.retained_control_handles) == set(readers)
            assert attempts == readers
            assert channels[0].handles == set(readers)
            assert not os.get_handle_inheritable(readers[0])
            if failed_reap:
                assert error.retained_process is processes[0]
                assert error.retained_process._handle
            else:
                assert not hasattr(error, "retained_process")
                assert not processes[0]._handle
            assert any("CloseHandle" in note for note in error.__notes__)
    finally:
        for process in processes:
            if process._handle:
                process.reap(timeout = 5)
                process.close()
        for owner in channels:
            for handle in tuple(owner.handles):
                owner.close(handle)
    assert not (tmp_path / "unexpected-completion").exists()


def test_failed_channel_close_before_resume_never_runs_worker(tmp_path, monkeypatch):
    original = preparation._WorkerChannels.close
    seen = []

    def close(owner, handle):
        if not seen:
            seen.append(handle)
            raise OSError("injected pre-resume close failure")
        return original(owner, handle)

    monkeypatch.setattr(preparation._WorkerChannels, "close", close)
    with pytest.raises(OSError, match = "pre-resume close"):
        run_fixed("from pathlib import Path; Path('payload-ran').touch()", tmp_path)
    assert seen and not (tmp_path / "payload-ran").exists()


def test_channel_failure_preserves_unreturned_raw_job_owner(tmp_path, monkeypatch):
    from core.inference.windows_sandbox.native_process import close_unreturned_process

    api = lpac._api().kernel32
    original_close, original_pipe = api.CloseHandle, preparation._WorkerChannels.pipe
    channels, readers, raw = [], [], []

    def pipe(owner):
        read, write = original_pipe(owner)
        channels.append(owner)
        readers.append(read)
        return read, write

    def failed_adapter(owner, argv, process_handle, thread_handle, pid, stdout, job):
        raw.append((job, process_handle, thread_handle))
        raise OSError("injected adapter construction failure")

    def close(handle):
        if handle in readers:
            ctypes.set_last_error(5)
            return False
        return original_close(handle)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(preparation._WorkerChannels, "pipe", pipe)
            patch.setattr(lpac.WindowsLpacProcess, "__init__", failed_adapter)
            patch.setattr(lpac._WindowsJob, "terminate", lambda owner: False)
            patch.setattr(api, "CloseHandle", close)
            with pytest.raises(preparation.WindowsRuntimeError) as caught:
                run_fixed("from pathlib import Path; Path('payload-ran').touch()", tmp_path)
            error = caught.value
            assert error.retained_job is raw[0][0]
            assert error.retained_native_handles == raw[0][1:]
            assert set(error.retained_control_handles) == set(readers)
            assert channels[0].handles == set(readers)
    finally:
        for job, process, thread in raw:
            close_unreturned_process(job, process, thread)
        for owner in channels:
            for handle in tuple(owner.handles):
                owner.close(handle)
    assert not (tmp_path / "payload-ran").exists()


def test_launch_retains_failed_worker_channels_until_cleanup(installed_runtime, tmp_path):
    output = run_harness(
        installed_runtime,
        tmp_path,
        """
import ctypes
from core.inference.windows_sandbox import native_compat as lpac
from core.inference.windows_sandbox import launch, preparation
script.write_text("from pathlib import Path; Path('payload-ran').touch()",encoding='utf-8')
api = lpac._api().kernel32
original_close, original_pipe = api.CloseHandle, preparation._WorkerChannels.pipe
original_init = launch._PythonLaunch.__init__
owners, targets = [], set()
def capture(owner,*args):
    original_init(owner,*args)
    owners.append(owner)
def pipe(owner):
    read,write = original_pipe(owner)
    if not targets:
        targets.add(read)
    return read,write
def close(handle):
    if handle in targets:
        ctypes.set_last_error(5)
        return False
    return original_close(handle)
launch._PythonLaunch.__init__ = capture
preparation._WorkerChannels.pipe = pipe
api.CloseHandle = close
try:
    spec = ToolLaunchPlan(argv=(sys.executable,'-u',str(script)),workdir=str(work),env={},execution_kind='python')
    try:
        prepare_python_launch(spec,root/'cache')
    except launch.WindowsRuntimeError as error:
        assert error.code == 'WINDOWS_SANDBOX_CLEANUP_FAILED', str(error)
        assert error.retained_launch is owners[0]
        assert targets <= owners[0].handles
        assert owners[0] in launch._pending_cleanup
        assert not owners[0].closed
    else:
        raise AssertionError('Failed worker channel cleanup returned a launch')
finally:
    api.CloseHandle = original_close
    preparation._WorkerChannels.pipe = original_pipe
    launch._PythonLaunch.__init__ = original_init
    for owner in owners:
        owner.cleanup()
assert owners and all(owner.closed and not owner.handles for owner in owners)
assert not launch._pending_cleanup and not (work/'payload-ran').exists()
assert not list((root/'cache'/'.readers').iterdir())
assert not owners[0].reservation.path.exists()
print('WORKER_CHANNELS_RECOVERED')
""",
    )
    assert "WORKER_CHANNELS_RECOVERED" in output
