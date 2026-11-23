# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tool processes are noticed by their exit event, with Popen.wait's own loop as the fallback; and the Landlock
scope wrapper only appears where the kernel can apply it, so older kernels keep the vfork shim path."""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.inference import tools  # noqa: E402
from core.inference.tools import (  # noqa: E402
    _capture_process_group,
    _drain_process_output,
    _open_exit_waiter,
    _wait_process,
)

posix_only = pytest.mark.skipif(sys.platform == "win32", reason = "POSIX exit notification")
stdlib_loop_only = pytest.mark.skipif(
    hasattr(subprocess.Popen, "_wait_pidfd"),
    reason = "Python 3.15+ Popen.wait already waits on the exit event",
)
linux_pidfd = pytest.mark.skipif(
    sys.platform != "linux" or not hasattr(os, "pidfd_open"), reason = "pidfd is Linux only"
)


def _spawn(script):
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        stdin = subprocess.DEVNULL,
        text = True,
        encoding = "utf-8",
        errors = "replace",
        start_new_session = sys.platform != "win32",
    )
    return proc, _capture_process_group(proc)


def _open_fds():
    return len(os.listdir("/proc/self/fd")) if os.path.isdir("/proc/self/fd") else None


@pytest.fixture
def event_waits(monkeypatch):
    """Counts exit-event waits that actually ran."""
    seen = []
    real = tools._wait_exit_event

    def spy(waiter, deadline):
        fired = real(waiter, deadline)
        seen.append((waiter[0], fired))
        return fired

    monkeypatch.setattr(tools, "_wait_exit_event", spy)
    return seen


_CASES = [
    ("print('hello')", 0),
    ("import sys; sys.stdout.write('a\\nb'); sys.exit(3)", 3),
    ("import os, signal; print('x', flush=True); os.kill(os.getpid(), signal.SIGKILL)", -9),
    ("import time; time.sleep(0.07); print(1)", 0),
    ("import sys\nfor i in range(300):\n    print(i)\n    print('e', i, file=sys.stderr)", 0),
]


@posix_only
@stdlib_loop_only
@pytest.mark.parametrize("script, code", _CASES)
def test_the_exit_event_path_matches_the_stdlib_loop(script, code, monkeypatch, event_waits):
    proc, pgid = _spawn(script)
    event = _drain_process_output(proc, 30, None, pgid = pgid)
    assert proc.returncode == code
    assert event_waits and event_waits[0][1] is True, event_waits
    monkeypatch.setattr(tools, "_open_exit_waiter", lambda proc: None)
    proc, pgid = _spawn(script)
    loop = _drain_process_output(proc, 30, None, pgid = pgid)
    assert proc.returncode == code
    assert event == loop
    assert len(event_waits) == 1


@linux_pidfd
@stdlib_loop_only
def test_linux_uses_a_pidfd_and_closes_it(event_waits):
    before = _open_fds()
    proc, pgid = _spawn("print(1)")
    _drain_process_output(proc, 30, None, pgid = pgid)
    proc.stdout.close()
    assert event_waits == [("pidfd", True)]
    assert _open_fds() == before


@posix_only
@stdlib_loop_only
def test_a_timeout_still_fires_at_the_deadline_and_kills(event_waits):
    before = _open_fds()
    proc, pgid = _spawn("import time; print('a', flush=True); time.sleep(30)")
    started = time.monotonic()
    output, timed_out, _ = _drain_process_output(proc, 0.5, None, pgid = pgid)
    elapsed = time.monotonic() - started
    proc.stdout.close()
    assert timed_out is True
    assert output == "a\n"
    assert proc.returncode == -9
    assert 0.5 <= elapsed < 5
    assert event_waits[0] == (event_waits[0][0], False)
    assert _open_fds() == before


@posix_only
def test_wait_process_raises_the_callers_timeout():
    proc, _ = _spawn("import time; time.sleep(30)")
    try:
        started = time.monotonic()
        with pytest.raises(subprocess.TimeoutExpired) as info:
            _wait_process(proc, 0.3)
        assert time.monotonic() - started >= 0.3
        assert info.value.timeout == 0.3
        assert proc.returncode is None
    finally:
        proc.kill()
        proc.wait()
        proc.stdout.close()


@posix_only
def test_no_timeout_and_non_positive_timeouts_keep_popen_wait(monkeypatch):
    calls = []
    monkeypatch.setattr(tools, "_open_exit_waiter", lambda proc: calls.append(proc))
    proc, _ = _spawn("print(1)")
    _wait_process(proc, None)
    assert proc.returncode == 0
    proc.stdout.close()
    proc, _ = _spawn("import time; time.sleep(30)")
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            _wait_process(proc, 0)
    finally:
        proc.kill()
        proc.wait()
        proc.stdout.close()
    assert calls == []


@linux_pidfd
@stdlib_loop_only
def test_pidfd_unavailable_falls_back_to_the_stdlib_loop(monkeypatch, event_waits):
    def enosys(*args):
        raise OSError(38, "Function not implemented")

    monkeypatch.setattr(os, "pidfd_open", enosys)
    proc, pgid = _spawn("import sys; print(1); sys.exit(5)")
    output, timed_out, _ = _drain_process_output(proc, 30, None, pgid = pgid)
    proc.stdout.close()
    assert (output, timed_out, proc.returncode) == ("1\n", False, 5)
    assert event_waits == []


@linux_pidfd
@stdlib_loop_only
def test_waitid_refusal_falls_back_and_closes_the_pidfd(monkeypatch, event_waits):
    def echild(*args):
        raise ChildProcessError(10, "No child processes")

    monkeypatch.setattr(os, "waitid", echild)
    before = _open_fds()
    proc, pgid = _spawn("print(1)")
    _drain_process_output(proc, 30, None, pgid = pgid)
    proc.stdout.close()
    assert proc.returncode == 0
    assert event_waits == []
    assert _open_fds() == before


@posix_only
def test_no_waiter_for_a_reaped_child_or_a_busy_lock_or_a_non_popen():
    proc, _ = _spawn("print(1)")
    proc.wait()
    proc.stdout.close()
    assert _open_exit_waiter(proc) is None  # returncode already set
    proc, _ = _spawn("import time; time.sleep(30)")
    try:
        with proc._waitpid_lock:
            assert _open_exit_waiter(proc) is None  # another thread is inside waitpid
    finally:
        proc.kill()
        proc.wait()
        proc.stdout.close()

    class NotPopen:
        pid = os.getpid()
        returncode = None
        _waitpid_lock = threading.Lock()

    assert _open_exit_waiter(NotPopen()) is None


@linux_pidfd
@stdlib_loop_only
def test_a_pid_reaped_outside_popen_is_not_watched():
    proc, _ = _spawn("print(1)")
    os.waitpid(proc.pid, 0)  # an outside waitpid: the pid is free for reuse
    try:
        assert _open_exit_waiter(proc) is None
    finally:
        proc.returncode = 0
        proc.stdout.close()


@posix_only
@stdlib_loop_only
def test_a_concurrent_poll_does_not_break_the_wait(event_waits):
    proc, pgid = _spawn("import time; time.sleep(0.3); print('done')")
    stop = threading.Event()

    def poller():
        while not stop.is_set() and proc.poll() is None:
            time.sleep(0.001)

    thread = threading.Thread(target = poller, daemon = True)
    thread.start()
    try:
        output, timed_out, _ = _drain_process_output(proc, 30, None, pgid = pgid)
    finally:
        stop.set()
        thread.join(5)
        proc.stdout.close()
    assert (output, timed_out, proc.returncode) == ("done\n", False, 0)


@posix_only
def test_a_redirected_background_job_still_gets_its_grace():
    proc, pgid = _spawn(
        "import subprocess, sys\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],"
        " stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)\n"
        "print('leader')"
    )
    output, timed_out, _ = _drain_process_output(proc, 30, None, pgid = pgid)
    proc.stdout.close()
    assert (output, timed_out) == ("leader\n", False)


@pytest.mark.skipif(sys.platform != "darwin", reason = "kqueue NOTE_EXIT is the macOS path")
@stdlib_loop_only
def test_macos_uses_kqueue(event_waits):
    proc, pgid = _spawn("import sys; print(1); sys.exit(4)")
    output, timed_out, _ = _drain_process_output(proc, 30, None, pgid = pgid)
    proc.stdout.close()
    assert (output, timed_out, proc.returncode) == ("1\n", False, 4)
    assert event_waits == [("kqueue", True)]


@pytest.mark.skipif(sys.platform != "linux", reason = "Landlock is Linux only")
class TestScopeWrapperOnlyWhereItApplies:
    def _plan(self, tmp_path, preexec_fn):
        from core.inference.os_sandbox import ToolLaunchPlan
        return ToolLaunchPlan(
            argv = ("/bin/true",),
            workdir = str(tmp_path),
            env = {"PATH": "/usr/bin"},
            preexec_fn = preexec_fn,
        )

    def test_scoped_preexec_follows_the_live_abi(self, monkeypatch):
        from core.inference import sandbox_landlock

        def plan_preexec():
            pass

        monkeypatch.setattr(sandbox_landlock, "_abi_reports_scope", lambda: False)
        assert sandbox_landlock.scoped_preexec(plan_preexec) is plan_preexec
        assert sandbox_landlock.scoped_preexec(None) is None
        monkeypatch.setattr(sandbox_landlock, "_abi_reports_scope", lambda: True)
        wrapped = sandbox_landlock.scoped_preexec(plan_preexec)
        assert wrapped is not plan_preexec and callable(wrapped)

    def test_a_probe_timeout_never_drops_the_scope(self, monkeypatch):
        from core.inference import sandbox_landlock

        # The cached probe saying no (a stalled child) must not remove the wrapper on a kernel that has the ABI.
        monkeypatch.setattr(sandbox_landlock, "abstract_scope_supported", lambda: False)
        monkeypatch.setattr(sandbox_landlock, "_abi_reports_scope", lambda: True)
        assert sandbox_landlock.scoped_preexec(tools._sandbox_preexec) is not tools._sandbox_preexec

    @pytest.mark.parametrize("abi_has_scope", [False, True])
    def test_the_shim_is_chosen_only_below_abi_6(self, tmp_path, monkeypatch, abi_has_scope):
        from core.inference import sandbox_landlock, sandbox_linux

        if sandbox_linux.shutil.which("bwrap") is None:
            pytest.skip(reason = "bubblewrap is not installed on this host")
        monkeypatch.setattr(sandbox_landlock, "_abi_reports_scope", lambda: abi_has_scope)
        monkeypatch.setattr(tools, "_sandbox_shims", ("/usr/bin/setpriv", "/usr/bin/prlimit"))
        monkeypatch.setattr(tools, "_sandbox_rlimit_args", lambda: ["--cpu=600:600"])
        prepared = sandbox_linux.prepare(self._plan(tmp_path, tools._sandbox_preexec))
        try:
            assert (prepared.preexec_fn is tools._sandbox_preexec) is not abi_has_scope
            kwargs = tools._apply_prepared_launch(prepared, {"preexec_fn": tools._sandbox_preexec})
            if abi_has_scope:
                assert prepared.spawn_callback is None
                assert kwargs["preexec_fn"] is prepared.preexec_fn
            else:
                assert prepared.spawn_callback is tools._spawn_behind_sandbox_shims
                assert "preexec_fn" not in kwargs
                assert kwargs["start_new_session"] is True and kwargs["umask"] == 0o077
                assert prepared.argv[:5] == (
                    "/usr/bin/setpriv",
                    "--no-new-privs",
                    "--pdeathsig",
                    "KILL",
                    "--",
                )
        finally:
            prepared.cleanup()

    def test_below_abi_6_the_scope_was_already_a_no_op(self, monkeypatch):
        """The wrapper it drops did nothing there: create_ruleset fails, so restrict_self is never reached."""
        from core.inference import sandbox_landlock

        if sandbox_landlock._libc is None:
            pytest.skip(reason = "ctypes cannot load libc on this platform")
        calls = []

        class FakeLibc:
            def syscall(self, nr, *args):
                calls.append(nr)
                return (
                    -1
                )  # E2BIG for the scoped attr on ABI < 6, EOPNOTSUPP/ENOSYS without Landlock

            def prctl(self, *args):
                calls.append(("prctl", args))
                return 0

        monkeypatch.setattr(sandbox_landlock, "_libc", FakeLibc())
        assert sandbox_landlock._abi_reports_scope() is False
        sandbox_landlock.apply_abstract_scope()
        assert calls == [sandbox_landlock._NR_LANDLOCK_CREATE_RULESET] * 2


@linux_pidfd
@stdlib_loop_only
def test_slices_halve_so_poll_slack_never_reaches_the_deadline(monkeypatch):
    """Linux poll() may wake 0.1% of its timeout late: every slice stays under half the lead, and the tail is Popen's."""
    import select as real_select
    import types

    slices = []

    class Recorder:
        def __init__(self):
            self._poller = real_select.poll()

        def register(self, *args):
            self._poller.register(*args)

        def poll(self, ms):
            slices.append((ms / 1000, deadline - time.monotonic()))
            return self._poller.poll(ms)

    monkeypatch.setattr(
        tools, "select", types.SimpleNamespace(poll = Recorder, POLLIN = real_select.POLLIN)
    )
    proc, _ = _spawn("import time; time.sleep(30)")
    try:
        waiter = _open_exit_waiter(proc)
        assert waiter is not None and waiter[0] == "pidfd"
        deadline = time.monotonic() + 1.0
        assert tools._wait_exit_event(waiter, deadline) is False
        assert deadline - time.monotonic() <= tools._EXIT_WAIT_TAIL_S
    finally:
        proc.kill()
        proc.wait()
        proc.stdout.close()
    assert len(slices) >= 3
    for wait_s, left in slices:
        lead = left - tools._EXIT_WAIT_TAIL_S
        # The recorder reads the clock a moment after the code under test did.
        assert wait_s <= max(lead / 2, tools._EXIT_WAIT_TAIL_S) + 1e-3
        assert wait_s <= lead + 1e-3
