# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Live trusted-worker ownership, deadlines and admission; not LPAC qualification."""

from dataclasses import asdict
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import threading
import time

import pytest

BACKEND = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND))
from core.inference.windows_sandbox import native_compat as lpac
from core.inference.windows_sandbox import admission, preparation
from core.inference.windows_sandbox.native_plan import ScanBounds
from core.inference.windows_sandbox.profiles import WindowsRuntimeError

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows preparation worker")

# Fixed controls need the real process, not a venv redirector and its child.
BASE_PYTHON = str(Path(sys.base_prefix) / "python.exe")


def run_fixed(
    source,
    directory,
    *,
    timeout = 5,
    cancel = None,
):
    return preparation._run_worker(
        [BASE_PYTHON, "-I", "-S", "-c", source],
        {},
        str(directory),
        deadline = time.monotonic() + timeout,
        cancel = cancel,
    )


def test_actual_worker_retains_parent_admission():
    direct = admission.admit_studio_runtime(sys.executable)
    worker = preparation.prepare_admitted_core(sys.executable)
    assert worker == direct
    assert worker.broker_pid == os.getpid()
    assert worker.digest == direct.digest
    assert not hasattr(worker, "qualified")


def test_different_selected_interpreter_never_launches_worker(monkeypatch, tmp_path):
    monkeypatch.setattr(preparation, "_run_worker", lambda *a, **k: pytest.fail("worker launched"))
    with pytest.raises(WindowsRuntimeError, match = "running Studio interpreter"):
        preparation.prepare_admitted_core(str(tmp_path / "python.exe"))


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf"), 121, "30"])
def test_invalid_deadline_never_launches_worker(timeout, monkeypatch):
    monkeypatch.setattr(preparation, "_run_worker", lambda *a, **k: pytest.fail("worker launched"))
    with pytest.raises(WindowsRuntimeError, match = "Invalid preparation timeout"):
        preparation.prepare_admitted_core(sys.executable, timeout = timeout)


def test_cancel_before_launch_starts_nothing(monkeypatch):
    cancel = threading.Event()
    cancel.set()
    monkeypatch.setattr(preparation, "_capture_broker_runtime", lambda: pytest.fail("captured"))
    with pytest.raises(WindowsRuntimeError, match = "cancelled"):
        preparation.prepare_admitted_core(sys.executable, cancel = cancel)


def test_worker_preserves_failure_code_without_retry():
    with pytest.raises(WindowsRuntimeError) as error:
        preparation.prepare_admitted_core(sys.executable, bounds = ScanBounds(entries = 1))
    assert error.value.code == "WINDOWS_SANDBOX_SCAN_LIMIT"


def test_failed_reap_retains_process_ownership(tmp_path, monkeypatch):
    original = lpac.WindowsLpacProcess.wait

    def failed_wait(self, timeout = None):
        raise subprocess.TimeoutExpired(self.args, timeout)

    monkeypatch.setattr(lpac.WindowsLpacProcess, "wait", failed_wait)
    with pytest.raises(WindowsRuntimeError) as caught:
        run_fixed(blocked_worker(tmp_path), tmp_path, timeout = 1)
    error = caught.value
    assert error.code == "WINDOWS_SANDBOX_CLEANUP_FAILED"
    owner = error.retained_process
    try:
        assert owner._handle and owner._unsloth_job._handle
        owner.terminate()
        original(owner, timeout = 5)
    finally:
        owner.close()
    assert not (tmp_path / "unexpected-completion").exists()


@pytest.mark.parametrize("field", ["nonce", "profile", "pid"])
def test_stale_response_binding_is_not_accepted(monkeypatch, field):
    def stale(argv, *args, **kwargs):
        request = json.loads(argv[-1])
        value = {
            "nonce": request["nonce"],
            "profile": request["profile"],
            "pid": 1234,
            "core": None,
            "error": None,
            "publication": None,
        }
        value[field] = 5678 if field == "pid" else "0" * 64
        return json.dumps(value).encode(), 1234

    monkeypatch.setattr(preparation, "_run_worker", stale)
    with pytest.raises(WindowsRuntimeError, match = "different invocation"):
        preparation.prepare_admitted_core(sys.executable)


def _worker_handle(pid):
    handle = lpac._api().kernel32.OpenProcess(0x100000 | 0x1000 | 1, False, pid)
    if not handle:
        assert ctypes.get_last_error() == 87  # no such process
    return handle


def assert_dead(pid):
    api = lpac._api().kernel32
    handle = _worker_handle(pid)
    if handle:
        try:
            assert api.WaitForSingleObject(handle, 0) == 0
        finally:
            api.CloseHandle(handle)


def blocked_worker(tmp_path, *, close_output = False):
    # An indefinitely blocked OS pipe read ignores Python cooperative deadlines.
    return f"""
import os
open({str(tmp_path / "pid")!r}, 'w').write(str(os.getpid()))
{"os.close(1)" if close_output else ""}
r, w = os.pipe()
os.read(r, 1)
open({str(tmp_path / "unexpected-completion")!r}, 'w').write('bad')
"""


@pytest.mark.parametrize("close_output", [False, True])
def test_timeout_reaps_blocked_os_read_and_eof_is_not_success(tmp_path, close_output):
    start = time.monotonic()
    with pytest.raises(WindowsRuntimeError) as error:
        run_fixed(blocked_worker(tmp_path, close_output = close_output), tmp_path, timeout = 1)
    assert error.value.code == "WINDOWS_SANDBOX_PREPARATION_TIMEOUT"
    assert time.monotonic() - start < 4
    assert_dead(int((tmp_path / "pid").read_text()))
    assert not (tmp_path / "unexpected-completion").exists()


def test_cancel_reaps_stuck_worker(tmp_path):
    cancel = threading.Event()

    def request_cancel():
        deadline = time.monotonic() + 3
        while not (tmp_path / "pid").exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        cancel.set()

    observer = threading.Thread(target = request_cancel)
    observer.start()
    try:
        with pytest.raises(WindowsRuntimeError, match = "cancelled"):
            run_fixed(blocked_worker(tmp_path), tmp_path, cancel = cancel)
    finally:
        observer.join(timeout = 4)
    assert not observer.is_alive()
    assert_dead(int((tmp_path / "pid").read_text()))


def test_bounded_output_reaps_writer_instead_of_blocking(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "MAX_RESULT", 32)
    source = f"import os; open({str(tmp_path / 'pid')!r}, 'w').write(str(os.getpid())); os.write(1, b'x' * 65536)"
    with pytest.raises(WindowsRuntimeError, match = "size bound"):
        run_fixed(source, tmp_path)
    assert_dead(int((tmp_path / "pid").read_text()))


def test_nonzero_worker_exit_cannot_return_success(tmp_path):
    with pytest.raises(WindowsRuntimeError, match = "exited unsuccessfully"):
        run_fixed("print('fake success'); raise SystemExit(12)", tmp_path)


def test_worker_startup_error_preserves_bounded_diagnostic(tmp_path):
    with pytest.raises(WindowsRuntimeError, match = "trusted-worker-startup-failure") as error:
        run_fixed("raise RuntimeError('trusted-worker-startup-failure')", tmp_path)
    assert "exited unsuccessfully (1)" in str(error.value)
    assert len(str(error.value)) < 4600


def test_worker_stderr_cannot_become_a_success_record(tmp_path):
    output, _ = run_fixed("import sys; print('{}'); print('diagnostic', file=sys.stderr)", tmp_path)
    with pytest.raises(WindowsRuntimeError):
        preparation._json(output)


@pytest.mark.parametrize(
    "api_name", ["CreateProcessW", "UpdateProcThreadAttribute", "ResumeThread"]
)
def test_creation_ownership_or_resume_failure_runs_no_worker(tmp_path, monkeypatch, api_name):
    api = lpac._api().kernel32

    def fail(*args):
        ctypes.set_last_error(5)
        return 0xFFFFFFFF if api_name == "ResumeThread" else 0

    monkeypatch.setattr(api, api_name, fail)
    sentinel = tmp_path / "must-not-run"
    with pytest.raises(OSError, match = api_name):
        run_fixed(f"open({str(sentinel)!r}, 'w').write('bad')", tmp_path)
    assert not sentinel.exists()


def test_worker_job_denies_child_and_closes_unexpected_handles(tmp_path):
    import msvcrt

    outside = tmp_path / "parent-only"
    outside.write_bytes(b"positive host control")
    fd = os.open(outside, os.O_RDONLY)
    os.set_inheritable(fd, True)
    handle = msvcrt.get_osfhandle(fd)
    # Numeric slots can be reused by startup. Compare the file identity/path,
    # not merely GetHandleInformation(handle)'s return value.
    source = f"""
import ctypes, subprocess, sys
from ctypes import wintypes as W
k = ctypes.WinDLL('kernel32', use_last_error=True)
k.GetFinalPathNameByHandleW.argtypes = [W.HANDLE, W.LPWSTR, W.DWORD, W.DWORD]
k.GetFinalPathNameByHandleW.restype = W.DWORD
path = ctypes.create_unicode_buffer(32768)
size = k.GetFinalPathNameByHandleW({handle}, path, len(path), 0)
assert not size or 'parent-only' not in path.value
try:
    child = subprocess.run([sys.executable, '-I', '-S', '-c', "open({str(tmp_path / "child-ran")!r}, 'w').write('bad')"], timeout=2)
except OSError as e:
    assert e.winerror == 1816, repr(e)
else:
    assert child.returncode != 0
print('JOB_AND_HANDLES_OK')
"""
    try:
        assert os.read(fd, 100) == b"positive host control"
        output, _ = run_fixed(source, tmp_path)
        assert output.strip() == b"JOB_AND_HANDLES_OK"
        assert not (tmp_path / "child-ran").exists()
    finally:
        os.close(fd)


@pytest.mark.parametrize("phase", ["suspended", "running"])
def test_parent_death_kills_job_before_or_after_resume(tmp_path, phase):
    source = blocked_worker(tmp_path)
    hook = ""
    if phase == "suspended":
        hook = f"""
api = lpac._api().kernel32
original = api.CreateProcessW
def create(*args):
    ok = original(*args)
    if ok:
        info = ctypes.cast(args[-1], ctypes.POINTER(lpac._PROCESS_INFORMATION)).contents
        open({str(tmp_path / "pid")!r}, 'w').write(str(info.dwProcessId))
    return ok
api.CreateProcessW = create
def suspend(*args):
    time.sleep(120)
api.ResumeThread = suspend
"""
    harness = f"""
import sys, ctypes, time
sys.path.insert(0, {str(BACKEND)!r})
sys.path.append({sysconfig.get_path("purelib")!r})
from core.inference.windows_sandbox import native_compat as lpac
from core.inference.windows_sandbox.preparation import _run_worker
{hook}
_run_worker([sys.executable, '-I', '-S', '-c', {source!r}], {{}}, {str(tmp_path)!r}, deadline=time.monotonic()+60, cancel=None)
"""
    parent = subprocess.Popen(
        [BASE_PYTHON, "-I", "-S", "-c", harness],
        stdout = subprocess.DEVNULL,
        stderr = subprocess.PIPE,
    )
    worker = None
    try:
        deadline = time.monotonic() + 6
        while (
            not (tmp_path / "pid").exists()
            and time.monotonic() < deadline
            and parent.poll() is None
        ):
            time.sleep(0.01)
        assert (tmp_path / "pid").exists(), parent.communicate(timeout = 2)
        # The writer can have created its file before writing the PID bytes.
        while not (text := (tmp_path / "pid").read_text()) and time.monotonic() < deadline:
            time.sleep(0.01)
        worker = _worker_handle(int(text))
        assert worker and lpac._api().kernel32.WaitForSingleObject(worker, 0) == 258
        parent.kill()
        parent.wait(timeout = 5)
        assert lpac._api().kernel32.WaitForSingleObject(worker, 5000) == 0
        assert not (tmp_path / "unexpected-completion").exists()
    finally:
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout = 5)
        if worker:
            lpac._api().kernel32.TerminateProcess(worker, 1)
            lpac._api().kernel32.WaitForSingleObject(worker, 5000)
            lpac._api().kernel32.CloseHandle(worker)
        parent.stderr.close()


@pytest.mark.parametrize("data", [b'{"a":1,"a":2}', b"not-json", b"[" * 2000])
def test_malformed_or_duplicate_result_is_rejected(data):
    with pytest.raises(WindowsRuntimeError):
        preparation._json(data)


@pytest.mark.parametrize("change", ["extra", "bool_pid", "path_type", "origin"])
def test_data_schema_and_parent_binding_fail_closed(monkeypatch, change):
    core = asdict(admission.admit_studio_runtime(sys.executable))

    def forged(argv, *args, **kwargs):
        request = json.loads(argv[-1])
        if change == "extra":
            core["qualified"] = True
        elif change == "bool_pid":
            core["broker_pid"] = True
        elif change == "path_type":
            core["runtime"]["executable"]["file"]["path"] = []
        else:
            core["origin"] = "user-approved"
        return json.dumps(
            {
                "nonce": request["nonce"],
                "profile": request["profile"],
                "pid": 1234,
                "core": core,
                "error": None,
                "publication": None,
            }
        ).encode(), 1234

    monkeypatch.setattr(preparation, "_run_worker", forged)
    with pytest.raises(WindowsRuntimeError):
        preparation.prepare_admitted_core(sys.executable)


def test_venv_parent_prefix_and_disabled_hooks_survive_worker(tmp_path):
    import pefile

    environment = tmp_path / "worker venv λ"
    selected = os.environ.get("UNSLOTH_TEST_PYTHON_EXECUTABLE", sys.executable)
    subprocess.run(
        [selected, "-I", "-S", "-m", "venv", "--without-pip", str(environment)],
        check = True,
        timeout = 30,
        capture_output = True,
    )
    marker = tmp_path / "hook-ran"
    (environment / "Lib/site-packages/hostile.pth").write_text(
        f"import pathlib; pathlib.Path({str(marker)!r}).write_text('bad')\n", encoding = "utf-8"
    )
    control = subprocess.run(
        [str(environment / "Scripts/python.exe"), "-I", "-c", "print('HOOK_CONTROL')"],
        capture_output = True,
        timeout = 10,
    )
    assert control.returncode == 0, control.stderr
    assert marker.read_text() == "bad"
    marker.unlink()
    source = f"""
import sys
sys.path.insert(0, {str(BACKEND)!r})
sys.path.extend([{str(Path(pefile.__file__).parent)!r}, {sysconfig.get_path("purelib")!r}])
# The fixed harness supplies venv facts without executing its hostile .pth.
sys.prefix = sys.exec_prefix = {str(environment)!r}
from core.inference.windows_sandbox.preparation import prepare_admitted_core
value = prepare_admitted_core(sys.executable)
assert value.runtime.prefix == sys.prefix
assert value.runtime.base_prefix == sys.base_prefix
print('VENV_ADMISSION_OK')
"""
    result = subprocess.run(
        [str(environment / "Scripts/python.exe"), "-I", "-S", "-c", source],
        capture_output = True,
        timeout = 30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == b"VENV_ADMISSION_OK"
    assert not marker.exists()
