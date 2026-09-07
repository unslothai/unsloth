# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Cancellation and ownership races at the temporary-token handoff."""

import ctypes
import os
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from test_launch import LAUNCH, installed_runtime, runtime_wheel, run_harness

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from core.inference.windows_sandbox import native_process, native_compat, native_bindings, protocol
from core.inference.windows_sandbox.profiles import PYTHON_PROFILE, WindowsRuntimeError

pytestmark = pytest.mark.skipif(os.name != "nt", reason = "Windows bootstrap ownership")


def test_cancel_during_attachment_lock_never_assigns_token(monkeypatch):
    entered, cancelled = threading.Event(), threading.Event()
    lock = threading.Lock()
    lock.acquire()

    class ObservedLock:
        def acquire(self, **kwargs):
            entered.set()
            return lock.acquire(**kwargs)

        def release(self):
            lock.release()

    process = SimpleNamespace(_close_lock = ObservedLock(), _startup_token = 123)

    def forbidden():
        pytest.fail("Cancelled attachment reached the native API")

    monkeypatch.setattr(native_compat, "_api", forbidden)
    errors = []

    def attach():
        try:
            native_process.attach_delayed_startup(
                process, deadline = time.monotonic() + 5, cancel = cancelled
            )
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target = attach)
    worker.start()
    try:
        assert entered.wait(2)
        cancelled.set()
    finally:
        lock.release()
        worker.join(3)
    assert not worker.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], WindowsRuntimeError)
    assert errors[0].code == "WINDOWS_SANDBOX_CANCELLED"
    assert process._startup_token == 123  # Still owned for cleanup, never consumed.
    assert lock.acquire(blocking = False)
    lock.release()


@pytest.mark.parametrize("reason", ["cancel", "deadline", "exit"])
def test_callback_invalidates_startup_before_any_go_write(monkeypatch, reason):
    binding = protocol.LaunchBinding(
        123, b"n" * 32, bytes.fromhex(PYTHON_PROFILE.digest), b"c" * 32
    )
    status = protocol.STATUS.pack(
        protocol.STATUS_MAGIC,
        protocol.VERSION,
        protocol.CLEAN_ENTRY,
        binding.pid,
        0,
        0,
        1,
        binding.nonce,
        binding.profile_digest,
        binding.content_digest,
    )
    cancelled = threading.Event()
    clock = [10.0]
    events = []
    process = SimpleNamespace(pid = binding.pid, exited = False)
    process.poll = lambda: 0 if process.exited else None
    process.terminate = lambda: events.append("terminate")
    process.wait = lambda **kwargs: events.append("reap")

    class PipeApi:
        def PeekNamedPipe(self, handle, _buf, _size, _read, available, _left):
            available._obj.value = len(status)
            return True

        def ReadFile(self, handle, buffer, size, count, _overlap):
            ctypes.memmove(buffer, status, len(status))
            count._obj.value = len(status)
            return True

        def WriteFile(self, *_args):
            pytest.fail("Invalidated callback must not deliver GO or payload ACK")

        def CloseHandle(self, handle):
            events.append(("close", handle))
            return True

    api = PipeApi()
    monkeypatch.setattr(protocol, "_pipe_api", lambda: api)
    monkeypatch.setattr(native_bindings, "_api", lambda: SimpleNamespace(kernel32 = api))
    monkeypatch.setattr(protocol, "time", SimpleNamespace(monotonic = lambda: clock[0]))

    def grant():
        events.append("grant")
        if reason == "cancel":
            cancelled.set()
        elif reason == "deadline":
            clock[0] = 20.0
        else:
            process.exited = True
        return protocol.startup_permission(binding, r"E:\private\winsock.hiv", b"m" * 32)

    with pytest.raises(WindowsRuntimeError):
        protocol.authorize_startup(
            process, 101, 102, binding, timeout = 1, cancel = cancelled, grant_startup = grant
        )
    assert events == ["grant", "terminate", "reap", ("close", 101), ("close", 102)]


@pytest.mark.skipif(
    not os.environ.get("UNSLOTH_TEST_RUNTIME_WHEEL"),
    reason = "Explicit installed runtime wheel required",
)
def test_cleanup_waits_for_actual_suspended_process_adoption(installed_runtime, tmp_path):
    body = (
        """
from core.inference.windows_sandbox import launch
script.write_text("from pathlib import Path; Path(__file__).with_name('payload-ran').write_text('ran')", encoding='utf-8')
"""
        + LAUNCH
        + """
entered, release, cleanup_started, cleanup_done = [threading.Event() for _ in range(4)]
owner.cancel = threading.Event()
original = launch.create_suspended_host
created, errors = [], []
def paused_create(*args, **kwargs):
    process = original(*args, **kwargs)
    created.append(process)
    entered.set()
    if not release.wait(10):
        process.reap(); process.close()
        raise RuntimeError('creation release timed out')
    return process
launch.create_suspended_host = paused_create
def spawning():
    try: owner.spawn(prepared, kwargs)
    except BaseException as error: errors.append(error)
def cleaning():
    cleanup_started.set()
    try: owner.cleanup()
    except BaseException as error: errors.append(error)
    finally: cleanup_done.set()
worker = threading.Thread(target=spawning)
cleaner = threading.Thread(target=cleaning)
worker.start()
try:
    assert entered.wait(20), 'native process creation never reached pause'
    cleaner.start()
    assert cleanup_started.wait(2)
    assert not cleanup_done.wait(.15), 'cleanup released ownership during native creation'
    owner.cancel.set()
finally:
    release.set()
    worker.join(20)
    if cleaner.ident is not None: cleaner.join(20)
    launch.create_suspended_host = original
    prepared.cleanup()
assert not worker.is_alive() and not cleaner.is_alive()
assert len(errors) == 1 and 'CANCELLED' in str(errors[0]), errors
assert len(created) == 1 and created[0]._reaped
assert created[0]._handle is None and created[0]._thread_handle is None
assert getattr(created[0], '_startup_token', None) is None
assert owner.closed and not manifest.exists()
assert not (work/'payload-ran').exists()
assert not prepared.cleanup_diagnostics, prepared.cleanup_diagnostics
print('BOOTSTRAP_LIFECYCLE_CLEAN')
"""
    )
    assert "BOOTSTRAP_LIFECYCLE_CLEAN" in run_harness(installed_runtime, tmp_path, body)
