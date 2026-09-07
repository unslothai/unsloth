# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Native Terminal creation ownership, independent of Python qualification."""

import ctypes
from ctypes import wintypes as W
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

BACKEND = Path(__file__).resolve().parents[2] / "backend"
sys.path.insert(0, str(BACKEND))
from core.inference.windows_sandbox import terminal_native as lpac
from core.inference.os_sandbox import spawn_prepared_launch
from core.inference.windows_sandbox.prepared import PreparedPythonLaunch as PreparedSandboxLaunch
from core.inference.windows_sandbox import terminal_job as job_owner
from core.inference.windows_sandbox.profiles import WindowsRuntimeError

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason = "Native Windows Terminal Job tests")
KWARGS = dict(
    stdout = subprocess.PIPE,
    stderr = subprocess.STDOUT,
    stdin = subprocess.DEVNULL,
    close_fds = True,
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0),
)


@pytest.fixture
def terminal(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    from core.inference.windows_sandbox.identity import InvocationRecipe, InvocationReservation

    reservation = InvocationReservation(InvocationRecipe.new())
    identity = reservation.create(str(work))
    prepared = None
    try:
        lpac._grant_modify(str(work), identity.sid)
        lpac._grant_modify(identity.private_temp, identity.sid)
        for root in identity.traverse_roots:
            lpac._grant_traverse(root, identity.sid)
        argv = (
            str(Path(os.environ["SystemRoot"]) / "System32/cmd.exe"),
            "/d",
            "/c",
            "echo NATIVE_TERMINAL",
        )
        prepared = PreparedSandboxLaunch(
            argv,
            str(work),
            lpac._safe_environment({}, str(work), identity, argv),
            None,
            "windows-lpac",
            spawn_callback = lambda launch, kwargs: lpac._spawn_lpac(launch, kwargs, identity),
        )
        yield prepared, identity
    finally:
        try:
            if prepared is not None:
                prepared.cleanup()
                assert tuple(prepared.cleanup_diagnostics) == getattr(
                    prepared, "_test_expected_cleanup_diagnostics", ()
                ), prepared.cleanup_diagnostics
        finally:
            reservation.cleanup()
            assert identity.cleaned and not Path(identity.manifest_path).exists()


def test_terminal_job_membership_exists_at_creation_return(monkeypatch, terminal):
    prepared, _ = terminal
    api = lpac._api().kernel32
    original_create, original_job = api.CreateProcessW, lpac._create_job
    jobs, membership = [], []
    api.IsProcessInJob.argtypes = [W.HANDLE, W.HANDLE, ctypes.POINTER(W.BOOL)]
    api.IsProcessInJob.restype = W.BOOL

    def make_job(handle, **kwargs):
        job = original_job(handle, **kwargs)
        jobs.append(job)
        return job

    def create(*args):
        result = original_create(*args)
        if result:
            info = ctypes.cast(args[-1], ctypes.POINTER(lpac._PROCESS_INFORMATION)).contents
            member = W.BOOL()
            membership.append(
                bool(
                    jobs
                    and api.IsProcessInJob(info.hProcess, jobs[0]._handle, ctypes.byref(member))
                    and member.value
                )
            )
        return result

    monkeypatch.setattr(lpac, "_create_job", make_job)
    monkeypatch.setattr(api, "CreateProcessW", create)
    process = spawn_prepared_launch(prepared, **KWARGS)
    assert process.wait(timeout = 5) == 0
    assert process.stdout.read().strip() == "NATIVE_TERMINAL"
    assert membership == [True], "Process existed before its sandbox Job ownership"
    assert prepared.execution_record is None


def test_broker_death_at_creation_return_kills_suspended_terminal(terminal, tmp_path):
    import _winapi

    prepared, identity = terminal
    marker = tmp_path / "created-pid"
    diagnostic = tmp_path / "broker-error"
    source = f"""
import ctypes, os, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,{str(BACKEND)!r})
from core.inference.windows_sandbox import terminal_native as lpac
from core.inference.os_sandbox import PreparedSandboxLaunch
from core.inference.windows_sandbox.identity import _derived_sid
api = lpac._api().kernel32
original = api.CreateProcessW
def pause_after_create(*args):
    result = original(*args)
    if result:
        info = ctypes.cast(args[-1],ctypes.POINTER(lpac._PROCESS_INFORMATION)).contents
        marker = Path({str(marker)!r})
        marker.with_suffix('.tmp').write_text(str(info.dwProcessId),encoding='ascii')
        marker.with_suffix('.tmp').replace(marker)
        api.Sleep(30000)
    return result
api.CreateProcessW = pause_after_create
with _derived_sid({identity.moniker!r}) as (sid,_):
    identity = SimpleNamespace(sid=sid,profile_folder={identity.profile_folder!r},moniker={identity.moniker!r})
    prepared = PreparedSandboxLaunch({prepared.argv!r},{prepared.workdir!r},{prepared.env!r},None,'test')
    lpac._spawn_lpac(prepared,{KWARGS!r},identity)
"""
    source = (
        "try:\n"
        + "\n".join("    " + line for line in source.splitlines())
        + f"\nexcept BaseException:\n    import traceback\n    open({str(diagnostic)!r},'w').write(traceback.format_exc())\n    raise\n"
    )
    startup = subprocess.STARTUPINFO()
    process, thread, _, _ = _winapi.CreateProcess(
        sys.executable,
        subprocess.list2cmdline([sys.executable, "-I", "-c", source]),
        None,
        None,
        False,
        0x4 | 0x08000000,
        {"SystemRoot": os.environ["SystemRoot"], "TEMP": str(tmp_path), "TMP": str(tmp_path)},
        str(tmp_path),
        startup,
    )
    api = lpac._api().kernel32
    safety_job = child = None
    try:
        # This outer Job is kept open through the assertion; only the broker
        # process is terminated. It prevents leaked children on the red control.
        safety_job = lpac._create_job(process, active_process_limit = 10)
        assert api.ResumeThread(thread) == 1
        deadline = time.monotonic() + 10
        while not marker.exists():
            assert time.monotonic() < deadline and api.WaitForSingleObject(process, 0) == 258, (
                diagnostic.read_text() if diagnostic.exists() else "Broker exited or timed out"
            )
            time.sleep(0.01)
        child = api.OpenProcess(0x1000 | 0x100000, False, int(marker.read_text(encoding = "ascii")))
        assert child and api.WaitForSingleObject(child, 0) == 258
        assert api.TerminateProcess(process, 1)
        assert api.WaitForSingleObject(process, 5000) == 0
        assert api.WaitForSingleObject(child, 2000) == 0, "Suspended Terminal survived broker death"
    finally:
        if safety_job is not None:
            safety_job.close()
        api.TerminateProcess(process, 1)
        assert api.WaitForSingleObject(process, 5000) == 0
        if child:
            assert api.WaitForSingleObject(child, 5000) == 0
            api.CloseHandle(child)
        api.CloseHandle(thread)
        api.CloseHandle(process)


@pytest.mark.parametrize("stage", ["attribute", "create", "resume"])
def test_terminal_creation_failures_close_job_without_fallback(monkeypatch, terminal, stage):
    prepared, _ = terminal
    api = lpac._api().kernel32
    original_job, original_create = lpac._create_job, api.CreateProcessW
    original_update = api.UpdateProcThreadAttribute
    jobs, children, calls = [], [], []

    def make_job(handle, **kwargs):
        job = original_job(handle, **kwargs)
        jobs.append(job)
        return job

    def update(*args):
        if stage == "attribute" and args[2] == lpac._PROC_THREAD_ATTRIBUTE_JOB_LIST:
            ctypes.set_last_error(50)  # Unsupported host must block, not retry.
            return False
        return original_update(*args)

    def create(*args):
        calls.append("create")
        if stage == "create":
            ctypes.set_last_error(5)
            return False
        result = original_create(*args)
        if result:
            info = ctypes.cast(args[-1], ctypes.POINTER(lpac._PROCESS_INFORMATION)).contents
            child = api.OpenProcess(0x100000, False, info.dwProcessId)
            assert child
            children.append(child)
        return result

    def failed_resume(*args):
        ctypes.set_last_error(5)
        return 0xFFFFFFFF

    monkeypatch.setattr(lpac, "_create_job", make_job)
    monkeypatch.setattr(api, "UpdateProcThreadAttribute", update)
    monkeypatch.setattr(api, "CreateProcessW", create)
    if stage == "resume":
        monkeypatch.setattr(api, "ResumeThread", failed_resume)
    try:
        with pytest.raises(OSError) as failure:
            spawn_prepared_launch(prepared, **KWARGS)
        assert failure.value.errno == (50 if stage == "attribute" else 5)
        assert calls == ([] if stage == "attribute" else ["create"])
        assert len(children) == int(stage == "resume")
        assert len(jobs) == 1 and jobs[0]._handle is None
        assert prepared.execution_record is None
        assert prepared._lpac_native_owner.closed and not prepared._lpac_native_owner.handles
        prepared.cleanup()
        assert not prepared.cleanup_callbacks and not prepared.cleanup_diagnostics
        for child in children:
            assert api.WaitForSingleObject(child, 5000) == 0
    finally:
        for child in children:
            api.CloseHandle(child)


def test_terminal_job_limits_and_noninheritance_are_preserved(monkeypatch, terminal):
    prepared, _ = terminal
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_NPROC", "7")
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_AS_GB", "1")
    monkeypatch.setenv("UNSLOTH_STUDIO_SANDBOX_CPU_S", "5")
    api = lpac._api().kernel32
    query = api.QueryInformationJobObject
    query.argtypes = [W.HANDLE, ctypes.c_int, ctypes.c_void_p, W.DWORD, ctypes.c_void_p]
    query.restype = W.BOOL
    api.GetHandleInformation.argtypes = [W.HANDLE, ctypes.POINTER(W.DWORD)]
    api.GetHandleInformation.restype = W.BOOL
    process = spawn_prepared_launch(prepared, **KWARGS)
    info, flags = lpac._JOBOBJECT_EXTENDED_LIMIT_INFORMATION(), W.DWORD()
    job = process._unsloth_job._handle
    assert query(job, 9, ctypes.byref(info), ctypes.sizeof(info), None)
    assert info.BasicLimitInformation.ActiveProcessLimit == 7
    assert info.BasicLimitInformation.PerProcessUserTimeLimit == 5 * 10_000_000
    assert info.ProcessMemoryLimit == info.JobMemoryLimit == 1024**3
    assert info.BasicLimitInformation.LimitFlags & 0x2000
    assert not info.BasicLimitInformation.LimitFlags & (0x800 | 0x1000)
    assert api.GetHandleInformation(job, ctypes.byref(flags)) and not flags.value & 1
    assert process.wait(timeout = 5) == 0


@pytest.mark.parametrize("mode", ["timeout", "cancel"])
def test_terminal_job_keeps_real_tool_streaming_and_termination(terminal, mode):
    from core.inference.tools import _cancel_watcher, _drain_process_output

    prepared, _ = terminal
    prepared.argv = (*prepared.argv[:3], "echo READY & for /L %i in (1,1,1000000000) do @rem")
    process = spawn_prepared_launch(prepared, **KWARGS)
    cancelled = threading.Event()
    observed = []
    live = []

    def output(chunk):
        observed.append(chunk)
        live.append(process.poll() is None)
        if mode == "cancel":
            cancelled.set()

    watcher = threading.Thread(target = _cancel_watcher, args = (process, cancelled, 0.01, None))
    watcher.start()
    try:
        text, timed_out = _drain_process_output(
            process, 1 if mode == "timeout" else 10, output, cancelled
        )
    finally:
        process.terminate()
        watcher.join(5)
        assert not watcher.is_alive()
    assert "READY" in text and "".join(observed) == text
    assert live and all(live), "Output was delayed until process exit"
    assert timed_out is (mode == "timeout")
    assert process.poll() is not None


class _FakeJobKernel:
    def __init__(
        self,
        events,
        *,
        inherited = False,
        member = True,
    ):
        self.events = events
        self.inherited = inherited
        self.member = member
        self.terminate_results = [True]
        self.close_results = [True]

    def GetCurrentProcess(self):
        return 1

    def DuplicateHandle(self, source, job, target, duplicate, access, inherit, options):
        self.events.append(("duplicate", job, bool(inherit), options))
        duplicate._obj.value = 300
        return True

    def GetHandleInformation(self, handle, flags):
        self.events.append(("handle_flags", handle))
        flags._obj.value = int(self.inherited)
        return True

    def IsProcessInJob(self, process, job, belongs):
        self.events.append(("membership", process, job))
        belongs._obj.value = self.member
        return True

    def TerminateJobObject(self, handle, code):
        self.events.append(("terminate", handle))
        return self.terminate_results.pop(0) if self.terminate_results else True

    def CloseHandle(self, handle):
        self.events.append(("job.close", handle))
        return self.close_results.pop(0) if self.close_results else True


def _fake_native_process(events, close_results = None):
    process = object.__new__(lpac.WindowsLpacProcess)
    process._handle = 200
    process._thread_handle = 201
    process._unsloth_job = SimpleNamespace(_handle = 100)
    results = list(close_results or (None,))

    def close():
        events.append(("process.close", process._handle))
        result = results.pop(0) if results else None
        if result is not None:
            raise result
        process._handle = process._thread_handle = None
        process._unsloth_job._handle = None

    process.close = close
    return process


def _install_fake_job(
    monkeypatch,
    *,
    flags = 0x2000 | 8,
    limit = 7,
    inherited = False,
    member = True,
    active = None,
    close_results = None,
):
    events = []
    kernel = _FakeJobKernel(events, inherited = inherited, member = member)
    process = _fake_native_process(events, close_results)
    active = list((0,) if active is None else active)

    def query(api, handle, kind, value):
        if kind == 9:
            events.append(("query.limits", handle))
            return SimpleNamespace(
                BasicLimitInformation = SimpleNamespace(LimitFlags = flags, ActiveProcessLimit = limit)
            )
        events.append(("query.accounting", handle))
        assert process._handle is None and process._unsloth_job._handle is None
        count = active.pop(0) if active else 0
        return SimpleNamespace(ActiveProcesses = count)

    monkeypatch.setattr(job_owner, "_kernel", lambda: kernel)
    monkeypatch.setattr(job_owner, "_query", query)
    return kernel, process, events


def test_terminal_job_closes_leader_before_job_zero_query(monkeypatch):
    kernel, process, events = _install_fake_job(monkeypatch)
    owner = job_owner.TerminalJobOwner()
    owner.bind_process(process)
    owner.cleanup()
    names = [event[0] for event in events]
    assert names.index("process.close") < names.index("query.accounting") < names.index("job.close")
    assert owner.closed and owner.process is None and owner.job is None
    assert kernel.close_results == []


def test_live_terminal_descendants_prevent_job_release(monkeypatch):
    kernel, process, events = _install_fake_job(monkeypatch, active = [1])
    actual_monotonic = time.monotonic
    clock = iter((10.0, 16.0))
    monkeypatch.setattr(job_owner.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(job_owner.time, "sleep", lambda _: None)
    owner = job_owner.TerminalJobOwner()
    owner.bind_process(process)
    with pytest.raises(WindowsRuntimeError, match = "descendants"):
        owner.cleanup()
    assert owner.process is None and owner.job == 300 and not owner.closed
    assert not any(event[0] == "job.close" for event in events)
    assert kernel.terminate_results == []
    monkeypatch.setattr(job_owner.time, "monotonic", actual_monotonic)
    owner.cleanup()
    assert owner.closed


def test_terminal_job_termination_failure_preserves_owner_for_retry(monkeypatch):
    kernel, process, events = _install_fake_job(monkeypatch)
    kernel.terminate_results = [False, True]
    owner = job_owner.TerminalJobOwner()
    owner.bind_process(process)
    with pytest.raises(WindowsRuntimeError, match = "termination"):
        owner.cleanup()
    assert owner.process is process and owner.job == 300 and not owner.closed
    owner.cleanup()
    assert owner.closed and [event[0] for event in events].count("terminate") == 2


def test_terminal_process_close_failure_preserves_owner_for_retry(monkeypatch):
    failure = OSError("fixed process close failure")
    _, process, events = _install_fake_job(monkeypatch, close_results = [failure, None])
    owner = job_owner.TerminalJobOwner()
    owner.bind_process(process)
    with pytest.raises(OSError, match = "fixed process close failure"):
        owner.cleanup()
    assert owner.process is process and owner.job == 300 and not owner.closed
    owner.cleanup()
    assert owner.closed and [event[0] for event in events].count("process.close") == 2


def test_terminal_duplicate_close_failure_retains_only_duplicate(monkeypatch):
    kernel, process, events = _install_fake_job(monkeypatch)
    kernel.close_results = [False, True]
    owner = job_owner.TerminalJobOwner()
    owner.bind_process(process)
    with pytest.raises(WindowsRuntimeError, match = "could not be closed"):
        owner.cleanup()
    assert owner.process is None and owner.job == 300 and not owner.closed
    owner.cleanup()
    assert owner.closed and [event[0] for event in events].count("job.close") == 2


@pytest.mark.parametrize(
    "change",
    [
        {"flags": 8},
        {"flags": 0x2000 | 8 | 0x800},
        {"flags": 0x2000 | 8 | 0x1000},
        {"flags": 0x2000},
        {"limit": 0},
        {"member": False},
    ],
)
def test_invalid_terminal_job_policy_is_owned_and_blocks_before_resume(monkeypatch, change):
    kernel, process, events = _install_fake_job(monkeypatch, **change)
    kernel.ResumeThread = lambda *args: pytest.fail("process resumed")
    owner = job_owner.TerminalJobOwner()
    with pytest.raises(WindowsRuntimeError, match = "bounded Job"):
        owner.bind_process(process)
    assert owner.attempted and owner.process is process and owner.job == 300
    assert events[0][0] == "duplicate"
    with pytest.raises(WindowsRuntimeError, match = "rebound"):
        owner.bind_process(process)
    assert [event[0] for event in events].count("duplicate") == 1
    owner.cleanup()
    assert owner.closed


def test_inheritable_terminal_job_duplicate_is_rejected_but_owned(monkeypatch):
    _, process, events = _install_fake_job(monkeypatch, inherited = True)
    owner = job_owner.TerminalJobOwner()
    with pytest.raises(WindowsRuntimeError, match = "non-inherited"):
        owner.bind_process(process)
    assert owner.process is process and owner.job == 300 and not owner.closed
    assert events[0] == ("duplicate", 100, False, 2)
    owner.cleanup()


def test_terminal_job_owner_cannot_bind_twice(monkeypatch):
    _, process, events = _install_fake_job(monkeypatch)
    owner = job_owner.TerminalJobOwner()
    owner.bind_process(process)
    with pytest.raises(WindowsRuntimeError, match = "rebound"):
        owner.bind_process(process)
    assert [event[0] for event in events].count("duplicate") == 1
    owner.cleanup()


def test_terminal_job_rejects_non_native_process_without_kernel_call(monkeypatch):
    monkeypatch.setattr(job_owner, "_kernel", lambda: pytest.fail("kernel queried"))
    owner = job_owner.TerminalJobOwner()
    with pytest.raises(WindowsRuntimeError, match = "native Windows process"):
        owner.bind_process(SimpleNamespace())
    assert owner.attempted and owner.process is None and owner.job is None
    owner.cleanup()
    assert owner.closed


def test_terminal_job_query_rejects_partial_accounting_result():
    def partial(handle, kind, output, length, returned):
        returned._obj.value = length - 1
        return True

    api = SimpleNamespace(QueryInformationJobObject = partial)
    with pytest.raises(WindowsRuntimeError, match = "could not be verified"):
        job_owner._query(api, 300, 1, job_owner._Accounting())


def test_terminal_job_query_never_uses_the_brokers_implicit_job():
    api = SimpleNamespace(QueryInformationJobObject = lambda *_: pytest.fail("implicit broker Job"))
    with pytest.raises(WindowsRuntimeError, match = "ownership handle is unavailable"):
        job_owner._query(api, None, 1, job_owner._Accounting())
