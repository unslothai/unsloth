# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the parent-lifetime reaper (utils/process_lifetime).

The Linux PDEATHSIG cases spawn real processes and assert actual liveness; the
Windows Job Object path is exercised with a mocked kernel32 so it runs on CI.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.process_lifetime as pl  # noqa: E402

IS_POSIX = os.name == "posix"
IS_LINUX = sys.platform.startswith("linux")


@pytest.fixture(autouse = True)
def _reset_module_state():
    pl._tracked_pids.clear()
    pl._win_job_handle = None
    pl._initialized = False
    yield
    pl._tracked_pids.clear()
    pl._win_job_handle = None
    pl._initialized = False


def _is_zombie(pid: int) -> bool:
    """True when *pid* has exited but has not been reaped yet (Linux ``Z`` state).

    Always False where /proc is unavailable (macOS/Windows), which leaves the
    plain existence probe as the answer there.
    """
    try:
        with open(f"/proc/{pid}/stat") as f:
            data = f.read()
    except OSError:
        return False
    # Format is "pid (comm) state ...", and comm may itself contain spaces and
    # parentheses, so the state field is the first token after the LAST ')'.
    try:
        return data[data.rindex(")") + 1 :].split()[0] == "Z"
    except (ValueError, IndexError):
        return False


def _alive(pid: int) -> bool:
    if sys.platform == "win32":
        return _win_alive(pid)
    try:
        os.kill(pid, 0)  # POSIX existence probe (on Windows this would terminate it)
    except OSError:
        return False
    # os.kill(pid, 0) still succeeds for a zombie, which has already exited and
    # only lingers holding an exit status. In a container whose PID 1 does not
    # reap adopted children it lingers indefinitely, so counting it as alive
    # would fail these "did the watcher kill it" assertions spuriously.
    return not _is_zombie(pid)


def _win_alive(pid: int) -> bool:
    import ctypes

    PROCESS_QUERY_LIMITED_INFORMATION, STILL_ACTIVE = 0x1000, 259
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return False
    code = ctypes.c_ulong()
    kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
    kernel32.CloseHandle(handle)
    return code.value == STILL_ACTIVE


def _wait_dead(pid: int, timeout: float) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


def _killpg(proc) -> None:
    # Reap the process and its whole session/group (forkserver + workers) so a
    # test never leaves long-lived Python processes behind. With
    # start_new_session=True the leader's pid IS the pgid, and the group lives
    # as long as any member does -- so kill proc.pid AS the pgid directly rather
    # than os.getpgid(proc.pid), which raises ESRCH once proc.wait() has already
    # reaped the leader and would silently skip the group kill.
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        proc.kill()
    except Exception:
        pass


# ── No-op safety / composition ──


def test_initialize_idempotent_and_noop_on_posix():
    pl.initialize_parent_lifetime()
    pl.initialize_parent_lifetime()  # second call short-circuits
    if IS_POSIX:
        assert pl._win_job_handle is None  # POSIX installs no job


def test_adopt_pid_tolerates_none_and_dead_pid():
    pl.adopt_pid(None)  # ignored
    pl.adopt_pid(2**31 - 1)  # almost-certainly-dead pid: recorded, never raises
    assert None not in pl._tracked_pids


def test_child_popen_kwargs_linux_vs_other(monkeypatch):
    monkeypatch.setattr(pl, "_is_linux", lambda: True)
    assert "preexec_fn" in pl.child_popen_kwargs()
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    assert pl.child_popen_kwargs() == {}  # Windows/macOS add nothing here


def test_compose_preexec_runs_pdeathsig_then_existing(monkeypatch):
    calls = []
    monkeypatch.setattr(pl, "_is_linux", lambda: True)
    monkeypatch.setattr(pl, "_pdeathsig_preexec", lambda _ppid: calls.append("death"))
    pl.compose_preexec(lambda: calls.append("existing"))()
    assert calls == ["death", "existing"]  # ordering matters for sandbox hooks


def test_compose_preexec_passthrough_off_linux(monkeypatch):
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    sentinel = lambda: None  # noqa: E731
    assert pl.compose_preexec(sentinel) is sentinel
    assert pl.compose_preexec(None) is None


def test_compose_preexec_passes_real_parent_pid(monkeypatch):
    # The pid handed to the child must be the spawner's own pid (captured before
    # the fork), not the literal 1 -- otherwise a PID-1 parent is indistinguishable
    # from a dead one.
    monkeypatch.setattr(pl, "_is_linux", lambda: True)
    monkeypatch.setattr(pl.os, "getpid", lambda: 4321)
    seen = []
    monkeypatch.setattr(pl, "_pdeathsig_preexec", lambda ppid: seen.append(ppid))
    pl.compose_preexec(None)()
    assert seen == [4321]


def test_spawn_parent_pid_prefers_multiprocessing(monkeypatch):
    # spawn/fork: multiprocessing captures the spawning parent's pid (== our OS
    # parent) at spawn; it stays put even after that parent dies, so an orphaned
    # worker (getppid() now 1) still compares against the real original parent
    # rather than looking like a healthy PID-1 child.
    import multiprocessing

    monkeypatch.setattr(multiprocessing, "get_start_method", lambda allow_none = True: "spawn")
    monkeypatch.setattr(multiprocessing, "parent_process", lambda: types.SimpleNamespace(pid = 4242))
    monkeypatch.setattr(pl.os, "getppid", lambda: 1)  # pretend we were reparented
    assert pl._spawn_parent_pid() == 4242


def test_spawn_parent_pid_forkserver_uses_getppid(monkeypatch):
    # forkserver: parent_process().pid is the LOGICAL requester, but our OS
    # parent is the forkserver. Using the logical pid would make the guard kill a
    # healthy worker, so the real (forkserver) getppid() must win.
    import multiprocessing

    monkeypatch.setattr(multiprocessing, "get_start_method", lambda allow_none = True: "forkserver")
    monkeypatch.setattr(multiprocessing, "parent_process", lambda: types.SimpleNamespace(pid = 4242))
    monkeypatch.setattr(pl.os, "getppid", lambda: 6080)  # the forkserver
    assert pl._spawn_parent_pid() == 6080


def test_spawn_parent_pid_falls_back_to_getppid(monkeypatch):
    # Not started via multiprocessing (main process / plain fork): no recorded
    # parent, so use the live getppid().
    import multiprocessing

    monkeypatch.setattr(multiprocessing, "get_start_method", lambda allow_none = True: "fork")
    monkeypatch.setattr(multiprocessing, "parent_process", lambda: None)
    monkeypatch.setattr(pl.os, "getppid", lambda: 777)
    assert pl._spawn_parent_pid() == 777


# ── reparent guard: fire on a real orphan, never on a healthy PID-1 child ──


class _Exited(BaseException):  # not an Exception, so it escapes the guard's try
    pass


def test_pdeathsig_bails_when_reparented(monkeypatch):
    # Parent died between fork and here: our parent is no longer the one that
    # forked us, so the guard must fire.
    monkeypatch.setattr(pl, "_arm_parent_death_signal", lambda: None)
    monkeypatch.setattr(pl.os, "getppid", lambda: 999)
    monkeypatch.setattr(pl.os, "_exit", lambda _code: (_ for _ in ()).throw(_Exited()))
    with pytest.raises(_Exited):
        pl._pdeathsig_preexec(42)  # expected parent 42, but reparented to 999


def test_pdeathsig_survives_healthy_child_of_pid1(monkeypatch):
    # Studio running as PID 1 (container with no init): a healthy child has
    # getppid() == 1 from birth, which must NOT be read as a dead parent
    # (regression for the guard killing every child when Studio is PID 1).
    monkeypatch.setattr(pl, "_arm_parent_death_signal", lambda: None)
    monkeypatch.setattr(pl.os, "getppid", lambda: 1)
    exits = []
    monkeypatch.setattr(pl.os, "_exit", lambda code: exits.append(code))
    pl._pdeathsig_preexec(1)  # expected parent == getppid() == 1
    assert exits == []  # left alone


# ── Real Linux PDEATHSIG: child dies when the parent dies abnormally ──


@pytest.mark.skipif(not IS_LINUX, reason = "PR_SET_PDEATHSIG is Linux-only")
def test_pdeathsig_child_dies_when_parent_sigkilled(tmp_path):
    mid = tmp_path / "mid.py"
    mid.write_text(
        "import sys, subprocess, time\n"
        f"sys.path.insert(0, {str(_BACKEND)!r})\n"
        "from utils.process_lifetime import child_popen_kwargs\n"
        "p = subprocess.Popen(['sleep', '300'], **child_popen_kwargs())\n"
        "print(p.pid, flush = True)\n"
        "time.sleep(300)\n"
    )
    proc = subprocess.Popen([sys.executable, str(mid)], stdout = subprocess.PIPE, text = True)
    try:
        sleeper_pid = int(proc.stdout.readline().strip())
        assert _alive(sleeper_pid)
        proc.kill()  # hard-kill the parent (no graceful shutdown runs)
        proc.wait(timeout = 5)
        assert _wait_dead(sleeper_pid, 5.0), "child orphaned after parent SIGKILL"
    finally:
        proc.kill()


@pytest.mark.skipif(sys.platform != "win32", reason = "Windows Job Object")
def test_windows_job_kills_child_when_parent_dies(tmp_path):
    # Real kill-on-job-close: the parent installs the job and assigns itself, a
    # child inherits it automatically, and terminating the parent must reap the
    # child (the orphaned-cloudflared.exe scenario).
    mid = tmp_path / "mid.py"
    mid.write_text(
        "import sys, subprocess, time\n"
        f"sys.path.insert(0, {str(_BACKEND)!r})\n"
        "import utils.process_lifetime as pl\n"
        "pl.initialize_parent_lifetime()\n"
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
        "print(p.pid, int(pl._win_job_handle is not None), flush = True)\n"
        "time.sleep(300)\n"
    )
    proc = subprocess.Popen([sys.executable, str(mid)], stdout = subprocess.PIPE, text = True)
    try:
        first = proc.stdout.readline().split()
        child_pid, installed = int(first[0]), first[1] == "1"
        assert installed, "Windows Job Object was not installed"
        assert _alive(child_pid)
        proc.kill()  # TerminateProcess the parent -> last job handle closes
        proc.wait(timeout = 5)
        assert _wait_dead(child_pid, 5.0), "child orphaned after parent killed"
    finally:
        proc.kill()


# ── terminate_all backstop sweep ──


@pytest.mark.skipif(not IS_POSIX, reason = "POSIX process sweep")
def test_terminate_all_signals_tracked_and_is_idempotent():
    p = subprocess.Popen(["sleep", "300"])
    pl.adopt_pid(p.pid)
    pl.terminate_all()
    assert p.wait(timeout = 5) is not None  # reap + confirm it died
    pl.terminate_all()  # registry now empty; must not raise


@pytest.mark.skipif(not IS_POSIX, reason = "POSIX process sweep")
def test_terminate_all_escalates_to_sigkill():
    # A child that ignores SIGTERM must still be reaped via SIGKILL after timeout.
    p = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)",
        ]
    )
    time.sleep(0.5)  # let the handler install
    pl.adopt_pid(p.pid)
    pl.terminate_all(timeout = 0.3)
    assert p.wait(timeout = 5) == -signal.SIGKILL  # SIGTERM ignored, SIGKILL wins


@pytest.mark.skipif(not IS_POSIX, reason = "POSIX process sweep")
def test_terminate_all_lets_cooperative_child_exit_cleanly(tmp_path):
    # A child that handles SIGTERM gets `timeout` to exit cleanly (not -SIGKILL).
    marker = tmp_path / "clean.txt"
    p = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, sys, time\n"
            f"def h(*a): open({str(marker)!r}, 'w').write('clean'); sys.exit(0)\n"
            "signal.signal(signal.SIGTERM, h)\n"
            "time.sleep(300)\n",
        ]
    )
    time.sleep(0.5)
    pl.adopt_pid(p.pid)
    pl.terminate_all(timeout = 3.0)
    assert p.wait(timeout = 3) == 0  # exited via its own handler, not SIGKILL
    assert marker.read_text() == "clean"


def test_forget_pid_unregisters():
    pl.adopt_pid(4242)
    assert 4242 in pl._tracked_pids
    pl.forget_pid(4242)
    assert 4242 not in pl._tracked_pids


@pytest.mark.skipif(not IS_POSIX, reason = "POSIX process sweep")
def test_terminate_all_skips_recycled_pid(monkeypatch):
    # A tracked pid whose identity changed (recycled) must not be signalled.
    p = subprocess.Popen(["sleep", "300"])
    pl.adopt_pid(p.pid)  # records the real identity
    monkeypatch.setattr(pl, "_pid_identity", lambda _pid: "DIFFERENT")
    pl.terminate_all()
    assert _alive(p.pid)  # left untouched: identity mismatch
    p.kill()
    p.wait(timeout = 5)


@pytest.mark.skipif(not IS_LINUX, reason = "PR_SET_PDEATHSIG is Linux-only")
def test_bind_kills_multiprocessing_child_on_parent_death(tmp_path):
    # multiprocessing workers can't take a preexec_fn, so the child binds itself
    # via bind_current_process_to_parent_lifetime(). Killing the parent must reap
    # it (the gap reviewers found in adopt_pid alone).
    mid = tmp_path / "mid_mp.py"
    mid.write_text(
        "import sys, time, multiprocessing as mp\n"
        f"sys.path.insert(0, {str(_BACKEND)!r})\n"
        "from utils.process_lifetime import bind_current_process_to_parent_lifetime\n"
        "def _child():\n"
        "    bind_current_process_to_parent_lifetime()\n"
        "    time.sleep(300)\n"
        "if __name__ == '__main__':\n"
        "    p = mp.get_context('spawn').Process(target = _child, daemon = True)\n"
        "    p.start()\n"
        "    print(p.pid, flush = True)\n"
        "    time.sleep(300)\n"
    )
    proc = subprocess.Popen([sys.executable, str(mid)], stdout = subprocess.PIPE, text = True)
    try:
        child_pid = int(proc.stdout.readline().strip())
        assert _alive(child_pid)
        proc.kill()
        proc.wait(timeout = 5)
        assert _wait_dead(child_pid, 5.0), "mp child orphaned after parent SIGKILL"
    finally:
        proc.kill()


@pytest.mark.skipif(not IS_LINUX, reason = "forkserver PDEATHSIG is Linux-only")
def test_bind_does_not_kill_forkserver_worker(tmp_path):
    # A forkserver worker's OS parent is the forkserver, not the logical process
    # that requested it; binding must not mistake that for a reparent and exit(1)
    # before the target runs. The worker writes a marker then exits cleanly, and
    # the requester joins and exits normally, so nothing is left running.
    marker = tmp_path / "ran.txt"
    mid = tmp_path / "mid_fs.py"
    mid.write_text(
        "import sys, multiprocessing as mp\n"
        f"sys.path.insert(0, {str(_BACKEND)!r})\n"
        "from utils.process_lifetime import bind_current_process_to_parent_lifetime\n"
        "def _child(path):\n"
        "    bind_current_process_to_parent_lifetime()\n"
        f"    open(path, 'w').write('ran')\n"  # never reached if bind exited(1)
        "if __name__ == '__main__':\n"
        "    p = mp.get_context('forkserver').Process(target = _child, "
        f"args = ({str(marker)!r},))\n"
        "    p.start()\n"
        "    p.join(timeout = 10)\n"
        "    print('exit', p.exitcode, flush = True)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, str(mid)], stdout = subprocess.PIPE, text = True, start_new_session = True
    )
    try:
        line = proc.stdout.readline().strip()
        proc.wait(timeout = 10)
        assert marker.exists(), "forkserver worker exited before running its target"
        assert line == "exit 0", f"worker did not exit cleanly: {line!r}"
    finally:
        _killpg(proc)


@pytest.mark.skipif(not IS_LINUX, reason = "forkserver PDEATHSIG is Linux-only")
def test_bind_kills_forkserver_worker_when_requester_dies(tmp_path):
    # The requester ("Studio") is the LOGICAL parent; the worker's OS parent is
    # the forkserver. PDEATHSIG alone would miss this, so the parent-sentinel
    # watcher must reap the worker when the requester is SIGKILLed.
    mid = tmp_path / "mid_fs_kill.py"
    mid.write_text(
        "import sys, time, multiprocessing as mp\n"
        f"sys.path.insert(0, {str(_BACKEND)!r})\n"
        "from utils.process_lifetime import bind_current_process_to_parent_lifetime\n"
        "def _child():\n"
        "    bind_current_process_to_parent_lifetime()\n"
        "    time.sleep(300)\n"
        "if __name__ == '__main__':\n"
        "    p = mp.get_context('forkserver').Process(target = _child, daemon = False)\n"
        "    p.start()\n"
        "    print(p.pid, flush = True)\n"
        "    time.sleep(300)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, str(mid)], stdout = subprocess.PIPE, text = True, start_new_session = True
    )
    try:
        child_pid = int(proc.stdout.readline().strip())
        assert _alive(child_pid)
        proc.kill()  # SIGKILL the requester
        proc.wait(timeout = 5)
        assert _wait_dead(child_pid, 8.0), "forkserver worker survived requester death"
    finally:
        _killpg(proc)


# ── Windows Job Object path (mocked kernel32, runs on Linux CI) ──


class _Call:
    def __init__(self, name, log, ret):
        self.name, self.log, self.ret = name, log, ret
        self.restype = self.argtypes = None

    def __call__(self, *a, **k):
        self.log.append(self.name)
        return self.ret


class _FakeKernel32:
    def __init__(
        self,
        log,
        create_ret = 4321,
        set_ret = 1,
        assign_ret = 1,
    ):
        self.CreateJobObjectW = _Call("create", log, create_ret)
        self.SetInformationJobObject = _Call("set", log, set_ret)
        self.AssignProcessToJobObject = _Call("assign", log, assign_ret)
        self.GetCurrentProcess = _Call("getcur", log, -1)
        self.CloseHandle = _Call("close", log, 1)


def _patch_windows(monkeypatch, fake):
    import ctypes
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **k: fake, raising = False)


def test_windows_job_install_order(monkeypatch):
    log: list[str] = []
    _patch_windows(monkeypatch, _FakeKernel32(log))
    pl._install_windows_job()
    assert log.index("create") < log.index("set") < log.index("assign")
    assert pl._win_job_handle == 4321  # handle retained


def test_windows_job_install_degrades_on_create_failure(monkeypatch):
    log: list[str] = []
    _patch_windows(monkeypatch, _FakeKernel32(log, create_ret = 0))
    pl._install_windows_job()  # must not raise
    assert pl._win_job_handle is None
    assert "set" not in log  # short-circuited after the failed create


def test_windows_job_install_degrades_on_assign_failure(monkeypatch):
    log: list[str] = []
    _patch_windows(monkeypatch, _FakeKernel32(log, assign_ret = 0))
    pl._install_windows_job()
    assert pl._win_job_handle is None  # not retained when assignment fails
    assert "close" in log  # the orphaned job handle is closed


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX zombie semantics")
def test_zombie_child_counts_as_dead():
    """An exited-but-unreaped child must read as dead.

    os.kill(pid, 0) succeeds for a zombie, so without the /proc state check a
    container whose PID 1 does not reap adopted children would report a killed
    worker as "survived" and fail the lifetime assertions spuriously.
    """
    pid = os.fork()
    if pid == 0:  # child: exit immediately, parent deliberately does not reap yet
        os._exit(0)
    try:
        end = time.time() + 5.0
        while time.time() < end and not _is_zombie(pid):
            time.sleep(0.01)
        if not _is_zombie(pid):
            pytest.skip("no /proc zombie state available on this platform")
        assert not _alive(pid)
        assert _wait_dead(pid, 1.0)
    finally:
        try:
            os.waitpid(pid, 0)
        except OSError:
            pass
