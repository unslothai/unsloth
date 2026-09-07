# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native stdout ownership and compatibility with Studio's text-stream reader."""

import ctypes
from ctypes import wintypes as W
import io
import sys

import pytest

from test_launch import LAUNCH, run_harness, installed_runtime, runtime_wheel
from core.inference.windows_sandbox import launch, preparation

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows stdout")


def test_stdout_wrapping_cannot_lose_a_crt_descriptor(installed_runtime, tmp_path):
    output = run_harness(
        installed_runtime,
        tmp_path,
        """
import os, msvcrt
from core.inference.windows_sandbox import launch
script.write_text("from pathlib import Path; Path('payload-ran').touch()",encoding='utf-8')
original_init = launch._PythonLaunch.__init__
original_adopt, original_wrap, original_close = msvcrt.open_osfhandle, os.fdopen, os.close
owners, descriptors = [], []
prepared = None
def capture(owner,*args):
    original_init(owner,*args)
    owners.append(owner)
def adopt(handle,flags):
    fd = original_adopt(handle,flags)
    descriptors.append(fd)
    return fd
def wrap(*args,**kwargs):
    raise OSError('injected stdout wrapper failure')
def close(fd):
    if fd in descriptors:
        raise OSError('injected stdout CRT close failure')
    return original_close(fd)
launch._PythonLaunch.__init__ = capture
msvcrt.open_osfhandle, os.fdopen, os.close = adopt,wrap,close
try:
    spec = ToolLaunchPlan(argv=(sys.executable,'-u',str(script)),workdir=str(work),env={},execution_kind='python')
    try:
        prepared = prepare_python_launch(spec,root/'cache')
    except OSError as error:
        assert 'injected stdout' in str(error), str(error)
    assert not descriptors, 'Stdout ownership escaped to a CRT descriptor that cleanup lost'
    assert prepared is not None and prepared.execution_record is None
finally:
    msvcrt.open_osfhandle, os.fdopen, os.close = original_adopt,original_wrap,original_close
    launch._PythonLaunch.__init__ = original_init
    for owner in owners:
        owner.cleanup()
    for fd in descriptors:
        original_close(fd)
assert owners and all(owner.closed and not owner.handles for owner in owners)
assert not launch._pending_cleanup and not (work/'payload-ran').exists()
assert not list((root/'cache'/'.readers').iterdir())
assert not owners[0].reservation.path.exists()
print('NO_CRT_STDOUT_OWNERSHIP')
""",
    )
    assert "NO_CRT_STDOUT_OWNERSHIP" in output


@pytest.mark.parametrize("wrapper", ["raw", "buffered", "text"])
def test_native_stdout_close_failure_is_retryable(monkeypatch, wrapper):
    channels = preparation._WorkerChannels()
    reader, writer = channels.pipe()
    raw = launch._NativePipeReader(reader, channels.close)
    stream = raw
    if wrapper != "raw":
        stream = io.BufferedReader(stream)
    if wrapper == "text":
        stream = io.TextIOWrapper(stream, encoding = "utf-8", errors = "replace")
    original_close = channels.api.CloseHandle

    def failed_close(handle):
        if handle == reader:
            ctypes.set_last_error(5)
            return False
        return original_close(handle)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(channels.api, "CloseHandle", failed_close)
            with pytest.raises(Exception, match = "CloseHandle"):
                stream.close()
            assert not stream.closed and not raw.closed
            assert reader in channels.handles
        stream.close()
        assert stream.closed and raw.closed and reader not in channels.handles
        stream.close()
    finally:
        stream.close()
        for handle in tuple(channels.handles):
            channels.close(handle)


def test_native_stdout_read_errors_are_not_eof(monkeypatch):
    channels = preparation._WorkerChannels()
    reader, writer = channels.pipe()
    raw = launch._NativePipeReader(reader, channels.close)

    def failed_read(*args):
        ctypes.set_last_error(5)
        return False

    try:
        with monkeypatch.context() as patch:
            patch.setattr(raw._api, "ReadFile", failed_read)
            with pytest.raises(Exception, match = "ReadFile"):
                raw.readinto(bytearray(8))
        assert raw.readinto(bytearray()) == 0
        with pytest.raises(TypeError, match = "writable"):
            raw.readinto(b"readonly")
        payload = b"private stdout bytes"
        count = W.DWORD()
        assert raw._api.WriteFile(writer, payload, len(payload), ctypes.byref(count), None)
        assert count.value == len(payload)
        buffer = bytearray(64)
        assert raw.readinto(buffer) == len(payload)
        assert bytes(buffer[: len(payload)]) == payload
        channels.close(writer)
        assert raw.readinto(buffer) == 0
        raw.close()
        with pytest.raises(ValueError, match = "closed"):
            raw.readinto(buffer)
    finally:
        raw.close()
        for handle in tuple(channels.handles):
            channels.close(handle)


@pytest.mark.parametrize("wrapper", ["BufferedReader", "TextIOWrapper"])
@pytest.mark.parametrize("close_fails", [False, True])
def test_failed_native_stdout_wrapper_never_returns_launch(
    installed_runtime, tmp_path, wrapper, close_fails
):
    output = run_harness(
        installed_runtime,
        tmp_path,
        f"""
import ctypes
from core.inference.windows_sandbox import launch
script.write_text("from pathlib import Path; Path('payload-ran').touch()",encoding='utf-8')
original_init = launch._PythonLaunch.__init__
original_wrapper = getattr(launch.io,{wrapper!r})
kernel = launch.lpac._api().kernel32
original_close = kernel.CloseHandle
owners,attempts = [],[]
targets = set()
def capture(owner,*args):
    original_init(owner,*args)
    owners.append(owner)
def fail(stream,*args,**kwargs):
    if isinstance(stream,launch._NativePipeReader) or isinstance(getattr(stream,'raw',None),launch._NativePipeReader):
        attempts.append(stream)
        raw = stream if isinstance(stream,launch._NativePipeReader) else stream.raw
        targets.add(raw._handle)
        raise OSError('injected native stdout wrapper failure')
    return original_wrapper(stream,*args,**kwargs)
def close(handle):
    if {close_fails!r} and handle in targets:
        ctypes.set_last_error(5)
        return False
    return original_close(handle)
launch._PythonLaunch.__init__ = capture
setattr(launch.io,{wrapper!r},fail)
kernel.CloseHandle = close
try:
    spec = ToolLaunchPlan(argv=(sys.executable,'-u',str(script)),workdir=str(work),env={{}},execution_kind='python')
    try:
        prepare_python_launch(spec,root/'cache')
    except OSError as error:
        assert not {close_fails!r}
        assert 'injected native stdout' in str(error),str(error)
    except launch.WindowsRuntimeError as error:
        assert {close_fails!r} and error.code == 'WINDOWS_SANDBOX_CLEANUP_FAILED',str(error)
        assert error.retained_launch is owners[0]
        assert owners[0] in launch._pending_cleanup and targets <= owners[0].handles
    else:
        raise AssertionError('Wrapping failure returned launch')
finally:
    kernel.CloseHandle = original_close
    setattr(launch.io,{wrapper!r},original_wrapper)
    launch._PythonLaunch.__init__ = original_init
    for owner in owners:
        owner.cleanup()
assert len(attempts)==1 and all(stream.closed for stream in attempts)
assert owners and all(owner.closed and not owner.handles for owner in owners)
assert not launch._pending_cleanup and not (work/'payload-ran').exists()
assert not list((root/'cache'/'.readers').iterdir())
assert not owners[0].reservation.path.exists()
print('WRAPPER_FAILURE_CLEANED')
""",
    )
    assert "WRAPPER_FAILURE_CLEANED" in output


def test_live_stdout_native_close_failure_retains_launch(installed_runtime, tmp_path):
    output = run_harness(
        installed_runtime,
        tmp_path,
        f"""
import ctypes
from core.inference.windows_sandbox import launch
script.write_text("print('READY',flush=True); import time; time.sleep(60)",encoding='utf-8')
{LAUNCH}
process = spawn_prepared_launch(prepared,**kwargs)
stream = process.stdout
assert stream.readline().strip() == 'READY' and process.poll() is None
raw = stream.buffer.raw
assert isinstance(raw,launch._NativePipeReader)
handle = raw._handle
assert handle in owner.handles
kernel = launch.lpac._api().kernel32
original_close = kernel.CloseHandle
def failed_close(value):
    if value == handle:
        ctypes.set_last_error(5)
        return False
    return original_close(value)
kernel.CloseHandle = failed_close
try:
    prepared.cleanup()
    assert prepared.cleanup_diagnostics and owner in launch._pending_cleanup
    assert handle in owner.handles and not raw.closed and not stream.closed
    assert process.returncode is not None and process.stdout is stream
    assert owner.access.process is process and owner.access.pins.handles
    assert manifest.exists() and not owner.closed
finally:
    kernel.CloseHandle = original_close
    owner.cleanup()
assert owner.closed and raw.closed and stream.closed and not owner.handles
assert process.stdout is None and process._handle is None
assert not launch._pending_cleanup and not manifest.exists()
assert not list((root/'cache'/'.readers').iterdir())
print('NATIVE_STDOUT_CLOSE_RETRIED')
""",
    )
    assert "NATIVE_STDOUT_CLOSE_RETRIED" in output


@pytest.mark.parametrize("mode", ["complete", "timeout", "cancel"])
def test_native_stdout_uses_real_tool_drain(installed_runtime, tmp_path, mode):
    payload = "import os,time; os.write(1,b'READY\\r\\n'); time.sleep(0.3); " + (
        "os.write(1,('Δ 😀\\r\\n'+'x'*150000+'\\n').encode()+b'bad:\\xff\\n')"
        if mode == "complete"
        else "time.sleep(60)"
    )
    output = run_harness(
        installed_runtime,
        tmp_path,
        f"""
from core.inference.tools import _drain_process_output, _cancel_watcher
script.write_text({payload!r},encoding='utf-8')
{LAUNCH}
process = spawn_prepared_launch(prepared,**kwargs)
cancel = threading.Event()
observed = []
ready_live = []
watcher = None
def observe(line):
    observed.append(line)
    if line == 'READY\\n':
        ready_live.append(process.poll() is None)
        if {mode!r} == 'cancel':
            cancel.set()
if {mode!r} == 'cancel':
    watcher = threading.Thread(target=_cancel_watcher,args=(process,cancel,0.01,None))
    watcher.start()
try:
    output,timed_out = _drain_process_output(process,1 if {mode!r} == 'timeout' else 10,observe,cancel)
    assert ''.join(observed) == output
    assert ready_live == [True], 'Output was not streamed before completion'
    assert output.startswith('READY\\n')
    assert timed_out == ({mode!r} == 'timeout')
    if {mode!r} == 'complete':
        assert output == 'READY\\nΔ 😀\\n'+'x'*150000+'\\nbad:\\ufffd\\n',repr(output[:80])
        assert process.returncode == 0
    else:
        assert process.returncode is not None
finally:
    owner.cleanup()
    if watcher is not None:
        watcher.join(5)
        assert not watcher.is_alive()
assert owner.closed and not owner.handles and not manifest.exists()
assert not list((root/'cache'/'.readers').iterdir())
print('REAL_TOOL_DRAIN_OK')
""",
    )
    assert "REAL_TOOL_DRAIN_OK" in output


def test_cleanup_unblocks_an_active_native_stdout_reader(installed_runtime, tmp_path):
    output = run_harness(
        installed_runtime,
        tmp_path,
        f"""
from core.inference.windows_sandbox import launch
script.write_text("print('READY',flush=True); import time; time.sleep(60)",encoding='utf-8')
{LAUNCH}
process = spawn_prepared_launch(prepared,**kwargs)
stream = process.stdout
assert stream.readline() == 'READY\\n'
raw = stream.buffer.raw
entered = threading.Event()
outcomes = []
original_read = raw.readinto
def blocked_read(buffer):
    entered.set()
    return original_read(buffer)
raw.readinto = blocked_read
def read():
    try:
        outcomes.append(stream.read())
    except (ValueError,OSError):
        outcomes.append('closed during cleanup')
thread = threading.Thread(target=read)
thread.start()
try:
    assert entered.wait(5)
    started = time.monotonic()
    owner.cleanup()
    thread.join(5)
    assert not thread.is_alive() and time.monotonic()-started < 10
    assert outcomes in ([''],['closed during cleanup']),outcomes
finally:
    owner.cleanup()
    thread.join(5)
assert owner.closed and stream.closed and not owner.handles and not manifest.exists()
assert not launch._pending_cleanup
print('ACTIVE_STDOUT_READER_CLEANED')
""",
    )
    assert "ACTIVE_STDOUT_READER_CLEANED" in output
