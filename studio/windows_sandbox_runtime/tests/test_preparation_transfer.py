# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Fixed worker handoff and kernel pin lifetime, not LPAC qualification."""

import ctypes
from ctypes import wintypes as W
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import sysconfig
import threading
import time

import pytest

from test_preparation import BACKEND, lpac, preparation, _worker_handle
from test_launch import run_harness, installed_runtime, runtime_wheel
from core.inference.windows_sandbox.content_files import PathLease
from core.inference.windows_sandbox.profiles import WindowsRuntimeError

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows pin transfer")


@pytest.fixture
def observed_worker(monkeypatch):
    original = lpac.WindowsLpacProcess.__init__
    handles = []

    def capture(process, *args):
        original(process, *args)
        handle = _worker_handle(process.pid)
        assert handle
        handles.append(handle)

    monkeypatch.setattr(lpac.WindowsLpacProcess, "__init__", capture)
    yield handles
    api = lpac._api().kernel32
    try:
        assert handles
        assert all(api.WaitForSingleObject(handle, 0) == 0 for handle in handles)
    finally:
        for handle in handles:
            assert api.CloseHandle(handle)


def run_transfer(
    source,
    directory,
    callback,
    *,
    timeout = 5,
    cancel = None,
):
    return preparation._run_worker(
        [sys._base_executable, "-I", "-S", "-c", source],
        {},
        str(directory),
        deadline = time.monotonic() + timeout,
        cancel = cancel,
        transfer = callback,
    )


def fixed_pin_worker(
    root,
    nonce,
    *,
    after_ack = "",
    binding = "",
    fragmented = False,
):
    return f"""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, {str(BACKEND)!r})
from core.inference.windows_sandbox.content_files import PathLease
root = Path({str(root)!r})
pins = PathLease()
try:
    pins.file(root / 'file')
    pins.directory(root / 'directory')
    value = dict(pid=os.getpid(), nonce={nonce!r}, pins=[[str(p), h] for p, h in pins.handles.items()])
    {binding}
    data = json.dumps(value).encode('utf-8')
    frame = len(data).to_bytes(4, 'little') + data
    if {fragmented!r}:
        for byte in frame:
            os.write(1, bytes([byte]))
            time.sleep(0.0001)
    else:
        os.write(1, frame)
    ack = sys.stdin.buffer.read(33)
    assert ack == bytes.fromhex({nonce!r}), 'Invalid or missing broker acknowledgement'
    (root / 'acknowledged').write_text('worker acknowledged')
    {after_ack}
finally:
    pins.close()
"""


@pytest.fixture
def transfer_case(tmp_path):
    (tmp_path / "file").write_bytes(b"pinned content")
    (tmp_path / "directory").mkdir()
    nonce = secrets.token_hex(32)
    pins = PathLease()
    calls = []
    expected = {tmp_path / "file", tmp_path / "directory", tmp_path, *tmp_path.parents}

    def receive(process, data):
        value = preparation._json(data)
        if (
            set(value) != {"pid", "nonce", "pins"}
            or value["pid"] != process.pid
            or value["nonce"] != nonce
            or {Path(row[0]) for row in value["pins"]} != expected
        ):
            raise WindowsRuntimeError(
                "WINDOWS_SANDBOX_PREPARATION_FAILED", "Invalid worker binding"
            )
        calls.append(process.pid)
        assert not (tmp_path / "acknowledged").exists()
        pins.duplicate_from(
            process._handle, tuple(tuple(row) for row in value["pins"]), lambda: None
        )
        return bytes.fromhex(nonce)

    yield nonce, pins, calls, receive
    pins.close()


@pytest.mark.parametrize("fragmented", [False, True])
def test_handoff_preserves_noninherited_file_and_directory_locks_after_worker_exit(
    tmp_path, observed_worker, transfer_case, fragmented
):
    nonce, pins, calls, receive = transfer_case
    data, pid = run_transfer(
        fixed_pin_worker(tmp_path, nonce, fragmented = fragmented), tmp_path, receive
    )
    assert json.loads(data)["pid"] == pid and calls == [pid]
    assert (tmp_path / "acknowledged").exists()
    api = lpac._api().kernel32
    api.GetHandleInformation.argtypes = [W.HANDLE, ctypes.POINTER(W.DWORD)]
    api.GetHandleInformation.restype = W.BOOL
    for handle in pins.handles.values():
        flags = W.DWORD()
        assert api.GetHandleInformation(handle, ctypes.byref(flags)) and not flags.value & 1
    for name in ("file", "directory"):
        with pytest.raises(PermissionError):
            (tmp_path / name).rename(tmp_path / (name + "-moved"))
    with pytest.raises(PermissionError):
        (tmp_path / "file").write_bytes(b"unexpected")
    assert (tmp_path / "file").read_bytes() == b"pinned content"
    pins.close()
    (tmp_path / "file").write_bytes(b"post-close control")
    for name in ("file", "directory"):
        (tmp_path / name).rename(tmp_path / (name + "-moved"))


@pytest.mark.parametrize("stage", ["callback", "cancel", "duplicate"])
def test_partial_transfer_never_acknowledges_and_retains_owned_duplicates(
    tmp_path, observed_worker, transfer_case, monkeypatch, stage
):
    nonce, pins, calls, receive = transfer_case
    cancel = threading.Event()
    if stage == "duplicate":
        original = pins.api.kernel.DuplicateHandle
        # Fail only the last copy; the file plus its ancestor pins already belong
        # to the parent and must survive the worker's forced exit.
        count = 0
        total = len({tmp_path, *tmp_path.parents}) + 2

        def fail_last(*args):
            nonlocal count
            count += 1
            if count == total:
                ctypes.set_last_error(5)
                return False
            return original(*args)

        monkeypatch.setattr(pins.api.kernel, "DuplicateHandle", fail_last)

    def abort(process, data):
        ack = receive(process, data)
        if stage == "callback":
            raise WindowsRuntimeError("WINDOWS_SANDBOX_PREPARATION_FAILED", "Transfer rejected")
        if stage == "cancel":
            cancel.set()
        return ack

    with pytest.raises(WindowsRuntimeError):
        run_transfer(fixed_pin_worker(tmp_path, nonce), tmp_path, abort, cancel = cancel)
    assert len(calls) == 1 and not (tmp_path / "acknowledged").exists()
    assert tmp_path / "file" in pins.handles
    with pytest.raises(PermissionError):
        (tmp_path / "file").write_bytes(b"still pinned after failed handoff")
    pins.close()
    (tmp_path / "file").write_bytes(b"post-close control")


@pytest.mark.parametrize("binding", ["value['pid'] += 1", "value['nonce'] = '0' * 64"])
def test_invalid_worker_binding_copies_nothing(tmp_path, observed_worker, transfer_case, binding):
    nonce, pins, calls, receive = transfer_case
    with pytest.raises(WindowsRuntimeError, match = "binding"):
        run_transfer(fixed_pin_worker(tmp_path, nonce, binding = binding), tmp_path, receive)
    assert not pins.handles and not calls and not (tmp_path / "acknowledged").exists()


@pytest.mark.parametrize(
    "frame",
    [
        b"",
        b"\x01",
        b"\x00\x00\x00\x00",
        b"\xff\xff\xff\xff",
        b"\x03\x00\x00\x00x",
        b"\x01\x00\x00\x00xy",
    ],
)
def test_invalid_frame_never_calls_transfer(tmp_path, observed_worker, frame):
    source = f"import os; os.write(1, {frame!r})"
    with pytest.raises(WindowsRuntimeError):
        run_transfer(source, tmp_path, lambda *_: pytest.fail("Invalid frame reached transfer"))


@pytest.mark.parametrize(
    "after_ack",
    [
        "raise SystemExit(7)",
        "os.write(1, b'extra')",
        "os.close(1); r, w = os.pipe(); os.read(r, 1)",
    ],
)
def test_ack_is_not_success_without_clean_eof_and_exit(
    tmp_path, observed_worker, transfer_case, after_ack
):
    nonce, pins, calls, receive = transfer_case
    started = time.monotonic()
    with pytest.raises(WindowsRuntimeError):
        run_transfer(
            fixed_pin_worker(tmp_path, nonce, after_ack = after_ack), tmp_path, receive, timeout = 1.5
        )
    assert time.monotonic() - started < 8
    assert len(calls) == 1 and pins.handles and (tmp_path / "acknowledged").exists()


@pytest.mark.parametrize(
    "entries",
    [
        (("relative", 4),),
        (("E:/a/../b", 4),),
        (("E:/a", 4), ("E:/a/.", 8)),
        (("E:/a", 4), ("E:/b", 4)),
        (("E:/a", -1),),
        (("E:/a", True),),
    ],
)
def test_invalid_pin_inventory_does_not_duplicate(entries, monkeypatch):
    with PathLease() as pins:
        monkeypatch.setattr(
            pins.api.kernel, "DuplicateHandle", lambda *_: pytest.fail("Invalid inventory copied")
        )
        with pytest.raises(WindowsRuntimeError):
            pins.duplicate_from(123, entries, lambda: None)
        assert not pins.handles


@pytest.mark.parametrize("phase", ["before_ack", "after_ack"])
def test_parent_death_releases_transferred_pins_and_kills_worker(tmp_path, phase):
    (tmp_path / "file").write_bytes(b"pinned content")
    (tmp_path / "directory").mkdir()
    nonce = secrets.token_hex(32)
    worker = fixed_pin_worker(tmp_path, nonce, after_ack = "r, w = os.pipe(); os.read(r, 1)")
    source = f"""
import os, sys, json, time
from pathlib import Path
sys.path.insert(0, {str(BACKEND)!r})
sys.path.append({sysconfig.get_path("purelib")!r})
from core.inference.windows_sandbox.content_files import PathLease
from core.inference.windows_sandbox.preparation import _run_worker
pins = PathLease()
def receive(process, data):
    value = json.loads(data)
    assert value['pid'] == process.pid and value['nonce'] == {nonce!r}
    pins.duplicate_from(process._handle, tuple(tuple(row) for row in value['pins']), lambda: None)
    Path({str(tmp_path / "ready")!r}).write_text(str(process.pid))
    if {phase!r} == 'before_ack':
        r, w = os.pipe()
        os.read(r, 1)
    return bytes.fromhex({nonce!r})
_run_worker([sys.executable, '-I', '-S', '-c', {worker!r}], {{}}, {str(tmp_path)!r},
    deadline=time.monotonic()+60, cancel=None, transfer=receive)
Path({str(tmp_path / "unexpected-success")!r}).touch()
"""
    # Own the actual interpreter, not a venv redirector's separate waiting PID.
    parent = subprocess.Popen(
        [sys._base_executable, "-I", "-S", "-c", source],
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
    )
    observation = None
    try:
        marker = tmp_path / ("ready" if phase == "before_ack" else "acknowledged")
        deadline = time.monotonic() + 10
        while not marker.exists() and parent.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        observation = _worker_handle(int((tmp_path / "ready").read_text()))
        assert observation and lpac._api().kernel32.WaitForSingleObject(observation, 0) == 258
        with pytest.raises(PermissionError):
            (tmp_path / "file").write_bytes(b"unexpected")
        parent.kill()
        parent.wait(timeout = 5)
        assert lpac._api().kernel32.WaitForSingleObject(observation, 5000) == 0
        (tmp_path / "file").write_bytes(b"after parent and worker death")
        (tmp_path / "directory").rename(tmp_path / "directory-moved")
        assert not (tmp_path / "unexpected-success").exists()
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout = 5)
        parent.stdout.close()
        parent.stderr.close()
        if observation:
            assert lpac._api().kernel32.CloseHandle(observation)


def test_actual_snapshot_worker_hands_off_all_pins_and_excludes_gc(installed_runtime, tmp_path):
    body = """
from core.inference.windows_sandbox.content_files import PathLease
from core.inference.windows_sandbox.content import RuntimeContentStore
from core.inference.windows_sandbox.profiles import WindowsRuntimeError
with PathLease() as pins:
    published = prepare_runtime_snapshot(sys.executable, root/'cache', pins=pins)
    directory = root/'cache'/published.content_digest
    assert directory/'.lease' in pins.handles
    assert directory/'manifest.json' in pins.handles
    runtime_file = directory/'files'/published.spec().files[0].relative_path
    assert runtime_file in pins.handles
    try:
        runtime_file.write_bytes(b'forbidden overwrite')
    except PermissionError:
        pass
    else:
        raise AssertionError('Transferred runtime pin permitted replacement')
    store = RuntimeContentStore(root/'cache')
    try:
        store.collect(published.content_digest)
    except WindowsRuntimeError as error:
        assert error.code == 'WINDOWS_SANDBOX_STORE_BUSY'
    else:
        raise AssertionError('GC removed a generation with transferred pins')
assert not pins.handles
store.collect(published.content_digest)
assert not directory.exists()
print('SNAPSHOT_PIN_HANDOFF_OK')
"""
    assert "SNAPSHOT_PIN_HANDOFF_OK" in run_harness(installed_runtime, tmp_path, body)


@pytest.mark.parametrize("failure", ["inventory", "binding", "partial_copy", "during_copy"])
def test_snapshot_handoff_failure_cleans_partial_pins_without_payload(
    installed_runtime, tmp_path, failure
):
    body = f"""
from core.inference.windows_sandbox import preparation, launch
from core.inference.windows_sandbox.content_files import PathLease
script.write_text("from pathlib import Path; Path('payload-ran').touch()", encoding='utf-8')
spec = ToolLaunchPlan(argv=(sys.executable, '-u', str(script)), workdir=str(work), env={{}}, execution_kind='python')
original_worker = preparation._run_worker
original_copy = PathLease.duplicate_from
copied = []
observations = []
cancel = threading.Event()
def copy(pins, process, entries, check):
    copied.append(pins)
    if {failure!r} == 'during_copy':
        checks = 0
        def cancel_during_copy():
            nonlocal checks
            checks += 1
            if checks == 2:
                assert pins.handles
                cancel.set()
            check()
        return original_copy(pins, process, entries, cancel_during_copy)
    if {failure!r} == 'partial_copy':
        original_copy(pins, process, entries[:1], check)
        assert pins.handles
        raise launch.WindowsRuntimeError('WINDOWS_SANDBOX_CONTENT_INVALID', 'Injected partial copy')
    return original_copy(pins, process, entries, check)
def worker(*args, **kwargs):
    if 'transfer' not in kwargs:
        return original_worker(*args, **kwargs)
    original_transfer = kwargs['transfer']
    def corrupt(process, data):
        observed = launch.lpac._api().kernel32.OpenProcess(0x100000 | 0x1000, False, process.pid)
        assert observed
        observations.append(observed)
        value = preparation._json(data)
        if {failure!r} == 'inventory':
            value['pins'].pop()
        elif {failure!r} == 'binding':
            value['nonce'] = '0' * 64
        return original_transfer(process, json.dumps(value).encode())
    kwargs['transfer'] = corrupt
    return original_worker(*args, **kwargs)
preparation._run_worker = worker
PathLease.duplicate_from = copy
try:
    try:
        prepare_python_launch(spec, root/'cache', cancel=cancel)
    except launch.WindowsRuntimeError as error:
        if {failure!r} == 'during_copy':
            assert error.code == 'WINDOWS_SANDBOX_CANCELLED'
    else:
        raise AssertionError('Failed handoff returned a prepared payload')
    assert len(observations) == 1
    assert all(launch.lpac._api().kernel32.WaitForSingleObject(h, 0) == 0 for h in observations)
    assert not launch._pending_cleanup and not (work/'payload-ran').exists()
    assert bool(copied) == ({failure!r} in ('partial_copy', 'during_copy'))
    assert all(not pins.handles for pins in copied)
finally:
    preparation._run_worker = original_worker
    PathLease.duplicate_from = original_copy
    for retained in tuple(launch._pending_cleanup):
        retained.cleanup()
    for handle in observations:
        assert launch.lpac._api().kernel32.CloseHandle(handle)
print('FAILED_SNAPSHOT_HANDOFF_CLEAN')
"""
    assert "FAILED_SNAPSHOT_HANDOFF_CLEAN" in run_harness(installed_runtime, tmp_path, body)


def test_post_publication_pin_error_preserves_worker_failure_code(installed_runtime, tmp_path):
    wrapper = f"""
import sys
sys.path.insert(0, {str(BACKEND)!r})
sys.path.append(sys.argv[1])
from core.inference.windows_sandbox.content import RuntimeContentStore
from core.inference.windows_sandbox.profiles import WindowsRuntimeError
from core.inference.windows_sandbox.preparation_worker import main
def fail(*args, **kwargs):
    raise WindowsRuntimeError('WINDOWS_SANDBOX_CONTENT_INVALID', 'Injected post-publication pin failure')
RuntimeContentStore._lease_marker = fail
raise SystemExit(main())
"""
    body = f"""
from core.inference.windows_sandbox import preparation, launch
script.write_text("raise AssertionError('Payload must never run')", encoding='utf-8')
spec = ToolLaunchPlan(argv=(sys.executable, '-u', str(script)), workdir=str(work), env={{}}, execution_kind='python')
wrapper = root/'fixed-worker-fixture.py'
wrapper.write_text({wrapper!r}, encoding='utf-8')
original = preparation._run_worker
def worker(argv, *args, **kwargs):
    if not argv[4].endswith('preparation_worker.py'):
        return original(argv, *args, **kwargs)
    assert argv[1:4] == ['-I', '-S', '-B']
    assert argv[4].endswith('preparation_worker.py')
    return original([*argv[:4], str(wrapper), *argv[5:]], *args, **kwargs)
preparation._run_worker = worker
try:
    try:
        prepare_python_launch(spec, root/'cache')
    except launch.WindowsRuntimeError as error:
        assert error.code == 'WINDOWS_SANDBOX_CONTENT_INVALID'
        assert 'Injected post-publication pin failure' in str(error)
    else:
        raise AssertionError('Failed worker returned a prepared payload')
finally:
    preparation._run_worker = original
    for retained in tuple(launch._pending_cleanup):
        retained.cleanup()
assert not launch._pending_cleanup
assert any(path.is_dir() and len(path.name) == 64 for path in (root/'cache').iterdir())
assert not list((root/'cache'/'.readers').iterdir())
print('PIN_FAILURE_DIAGNOSTIC_PRESERVED')
"""
    assert "PIN_FAILURE_DIAGNOSTIC_PRESERVED" in run_harness(installed_runtime, tmp_path, body)
