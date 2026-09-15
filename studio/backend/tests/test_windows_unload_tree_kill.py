# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""An unload has to reap the whole server tree on Windows, not just its leader (#9790).

`collect_descendants` and `terminate_descendants` returned early and did nothing at all
on Windows, so an unload terminated llama-server and left whatever it had started
holding the model's mapping, and the pidfile was dropped either way so the next launch
could not reap it. The platform arms are covered twice: once against a real process
tree on this host, with only the one seam that cannot exist here faked, and once for
real on a Windows runner with nothing faked at all.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
import types as _types
from pathlib import Path
from unittest import mock

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# Mirror the sibling llama_cpp tests so the module imports without fastapi/structlog.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
sys.modules.setdefault("structlog", _types.ModuleType("structlog"))

import utils.process_lifetime as pl  # noqa: E402

IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")


@pytest.fixture(autouse = True)
def _warm_the_owner_identity():
    """Read this process's own identity before anything here fakes `_pid_identity`.

    `_own_identity` caches into a module global on first call and keeps it for the life
    of the interpreter, deliberately, since it cannot change. A test that fakes
    `_pid_identity` and then reaches any code path writing a breadcrumb would cache the
    fake, and every later test in the session would read this process as a dead owner
    and reap its children.
    """
    pl._own_identity()


# ── a real two-level process tree ──

_TREE = textwrap.dedent(
    """
    import pathlib, subprocess, sys, time
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    pathlib.Path(sys.argv[1]).write_text(str(child.pid))
    time.sleep(300)
    """
)


def _alive(pid: int) -> bool:
    if IS_WINDOWS:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output = True, text = True
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # A child nobody has waited on answers signal 0 exactly like a live one.
    return not pl._pid_is_zombie(pid)


def _wait_dead(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return not _alive(pid)


def _hard_kill(pid: int) -> None:
    try:
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output = True)
        else:
            os.kill(pid, signal.SIGKILL)
    except Exception:
        pass


@pytest.fixture
def tree(tmp_path):
    """(leader Popen, child pid) for a leader that spawned one child and sleeps."""
    marker = tmp_path / "child.pid"
    leader = subprocess.Popen([sys.executable, "-c", _TREE, str(marker)])
    child_pid = None
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            child_pid = int(marker.read_text())
            break
        except (OSError, ValueError):
            time.sleep(0.05)
    if child_pid is None:
        leader.kill()
        pytest.fail("the spawned child never recorded its pid")
    assert _alive(child_pid)
    try:
        yield leader, child_pid
    finally:
        _hard_kill(child_pid)
        try:
            leader.kill()
            leader.wait(timeout = 10)
        except Exception:
            pass


# ── the collect side ──


def _windows_shaped_identity(pid: int):
    """A real Linux start time dressed as a Windows creation FILETIME.

    Windows identities are `high:low` of a 64-bit FILETIME and the walk ORDERS two of
    them to reject a stranger, so a constant would not exercise the floor. Splitting the
    real start time the same way keeps the real ordering between leader and child.
    """
    try:
        with open(f"/proc/{pid}/stat", encoding = "utf-8") as fh:
            stat = fh.read()
        ticks = int(stat[stat.rfind(")") + 2 :].split()[19])
    except Exception:
        return None
    return f"{ticks >> 32}:{ticks & 0xFFFF_FFFF}"


@pytest.mark.skipif(not IS_LINUX, reason = "reads real start times out of /proc")
def test_the_windows_walk_claims_a_real_child(monkeypatch, tree):
    """The guard this replaces returned [] for every Windows caller, which is the
    whole of #9790: the unload had nothing to terminate but the leader."""
    leader, child_pid = tree
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_pid_identity", _windows_shaped_identity)
    collected = pl.collect_descendants(leader.pid)
    assert child_pid in [
        pid for pid, _ in collected
    ], f"collect_descendants({leader.pid}) -> {collected}, missing the spawned child"


@pytest.mark.skipif(not IS_LINUX, reason = "reads real start times out of /proc")
def test_the_windows_walk_claims_nothing_without_a_readable_root(monkeypatch, tree):
    """Windows keeps a creating pid on a process forever, so with no creation time for
    the root there is no way to tell its children from a stranger left behind by an
    earlier holder of that number. Claiming nothing is the only safe answer."""
    leader, _child_pid = tree
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: None)
    assert pl.collect_descendants(leader.pid) == []


def test_the_windows_walk_rejects_a_stranger_that_predates_the_root(monkeypatch):
    """A recycled pid whose old holder created this process. Its recorded parent is our
    root's number, it is not our child, and taskkilling it would take down an unrelated
    tree, which is worse than the leak being fixed."""
    root, real_child, stranger, grandchild = 500, 600, 700, 800
    created = {root: 1_000, real_child: 2_000, stranger: 10, grandchild: 20}
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: f"0:{created[pid]}")
    monkeypatch.setattr(
        pl, "_child_pid_map", lambda: {root: [real_child, stranger], stranger: [grandchild]}
    )
    assert [pid for pid, _ in pl.collect_descendants(root)] == [real_child]


# ── the terminate side ──


@pytest.mark.skipif(IS_WINDOWS, reason = "the real taskkill is covered below")
def test_the_windows_terminate_reaps_every_survivor(monkeypatch, tree):
    """`_windows_terminate_collected` is the whole Windows arm of terminate_descendants.
    Only `taskkill`, which does not exist here, is stood in for; the pids, the liveness
    probe and the identity check are all real, and so is the process that has to die."""
    leader, child_pid = tree
    asked = []

    def fake_taskkill(pid):
        asked.append(pid)
        _hard_kill(pid)
        return True

    monkeypatch.setattr(pl, "_windows_terminate_tree", fake_taskkill)
    collected = [(child_pid, pl._pid_identity(child_pid))]
    leader.terminate()
    leader.wait(timeout = 10)
    pl._windows_terminate_collected(collected)
    assert asked == [child_pid]
    assert _wait_dead(child_pid), "the spawned child outlived the unload"


def test_the_windows_terminate_skips_a_recycled_pid(monkeypatch):
    """The leader's terminate runs between the snapshot and this call, so a number here
    can already belong to something else by now."""
    # `_same_identity` compares the start time alone on Linux, for records written before
    # it carried a comm, and a whole Windows FILETIME pair on Windows. Both halves of the
    # synthetic identities here have to count, so this reads as the Windows host it is about.
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: "0:999")
    asked = []
    monkeypatch.setattr(pl, "_windows_terminate_tree", lambda pid: asked.append(pid))
    pl._windows_terminate_collected([(4242, "0:1")])
    assert asked == []


def test_the_windows_terminate_works_deepest_first(monkeypatch):
    """`taskkill /T` also reaches what a survivor started after the snapshot, and the
    link it walks is gone the moment that survivor exits."""
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: None)
    order = []
    monkeypatch.setattr(pl, "_windows_terminate_tree", lambda pid: order.append(pid))
    # collect_descendants emits breadth first, so the parent comes before its child.
    pl._windows_terminate_collected([(10, "a"), (11, "b"), (12, "c")])
    assert order == [12, 11, 10]


# ── the real thing, on a real Windows runner ──


@pytest.mark.skipif(not IS_WINDOWS, reason = "the Windows arms, unfaked")
def test_windows_unload_reaps_the_spawned_child_for_real(tree):
    """The round-2 repro, as a test: snapshot the tree, terminate the leader the way
    Popen does, then sweep. On the guarded code the sweep had nothing to sweep and the
    child was still alive here."""
    leader, child_pid = tree
    collected = pl.collect_descendants(leader.pid)
    assert child_pid in [
        pid for pid, _ in collected
    ], f"collect_descendants({leader.pid}) -> {collected}"
    leader.terminate()
    leader.wait(timeout = 10)
    pl.terminate_descendants(collected, timeout = 10.0)
    assert _wait_dead(child_pid), "the spawned child outlived the unload"


@pytest.mark.skipif(not IS_WINDOWS, reason = "Toolhelp snapshot")
def test_the_toolhelp_snapshot_reads_this_process_as_its_parent(tree):
    """The table itself, before any filtering: a real Popen child has to be listed
    under this interpreter's pid."""
    leader, _child_pid = tree
    table = pl._windows_child_pid_map()
    assert table is not None, "the Toolhelp snapshot could not be read"
    assert leader.pid in table.get(
        os.getpid(), []
    ), f"pid {leader.pid} is not listed under its own parent {os.getpid()}"


def test_a_windows_identity_orders_by_creation_time():
    assert pl._windows_creation_time("2:1") == (2 << 32) | 1
    assert pl._windows_creation_time("1:4294967295") < pl._windows_creation_time("2:0")
    for bad in (None, 5, "", "12345", "a:b", "1:2:3"):
        assert pl._windows_creation_time(bad) is None


# ── the unload path in the backend ──


def _make_backend():
    from core.inference.llama_cpp import LlamaCppBackend

    b = LlamaCppBackend.__new__(LlamaCppBackend)
    b._port = 12345
    b._stdout_thread = None
    b._stdout_lines = []
    b._process = mock.Mock()
    b._stats_logger = None
    b._llama_log_fh = None
    b._stop_mtp_crash_watchdog = lambda *a, **kw: None
    b._reset_effective_parallel_slots = lambda *a, **kw: None
    b._leading_process_group = lambda *a, **kw: None
    b._collect_descendants = lambda *a, **kw: []
    b._kill_process_group = lambda *a, **kw: None
    b._terminate_descendants = lambda *a, **kw: None
    return b


def _instrument(monkeypatch, *, gone: bool):
    from core.inference.llama_cpp import LlamaCppBackend

    cleared, killed = [], []
    monkeypatch.setattr(
        LlamaCppBackend, "_clear_server_pid", classmethod(lambda cls: cleared.append(1))
    )

    def tree_kill(pid):
        killed.append(pid)
        return gone

    monkeypatch.setattr(LlamaCppBackend, "_tree_kill_surviving_process", staticmethod(tree_kill))
    return cleared, killed


def test_an_unload_tree_kills_a_server_that_ignored_the_terminate(monkeypatch):
    """Popen.terminate is TerminateProcess on the leader alone on Windows. Nothing on
    this path used to reach for taskkill, and the pidfile went either way, so the next
    launch could not reap what was left."""
    b = _make_backend()
    b._process.pid = 4242
    b._process.poll.return_value = None  # still running after terminate and kill
    cleared, killed = _instrument(monkeypatch, gone = False)
    b._kill_process()
    assert killed == [4242]
    assert cleared == [], "dropped the pidfile while the server was still running"


def test_an_unload_clears_the_pidfile_once_the_tree_kill_worked(monkeypatch):
    b = _make_backend()
    b._process.pid = 4242
    b._process.poll.return_value = None
    cleared, killed = _instrument(monkeypatch, gone = True)
    b._kill_process()
    assert killed == [4242]
    assert cleared == [1]


def test_an_unload_that_already_worked_does_not_tree_kill(monkeypatch):
    """The common path. A server that exited on the terminate must not pay for a
    taskkill, and must still have its pidfile removed."""
    b = _make_backend()
    b._process.pid = 4242
    b._process.poll.return_value = 0
    cleared, killed = _instrument(monkeypatch, gone = False)
    b._kill_process()
    assert killed == []
    assert cleared == [1]


def test_an_unload_with_nothing_to_kill_still_clears_the_pidfile(monkeypatch):
    """A stand-in _process with no pid, as the tests that mean "a server is loaded"
    without spawning one use. There is nothing to prove gone, so a stale pidfile from a
    previous run must not be kept forever."""
    b = _make_backend()
    b._process = mock.Mock(spec = [])  # no pid, no terminate, no poll
    cleared, killed = _instrument(monkeypatch, gone = False)
    b._kill_process()
    assert killed == []
    assert cleared == [1]


def test_the_tree_kill_tells_the_reaper_the_owner_is_known(monkeypatch):
    """The unload holds the Popen that spawned the pid, so terminate_pid must not refuse
    on "cannot prove this is still our child" when the start time behind the lifetime
    record can no longer be read."""
    from core.inference.llama_cpp import LlamaCppBackend

    calls = []
    monkeypatch.setattr(pl, "terminate_pid", lambda pid, **kw: calls.append((pid, kw)))
    monkeypatch.setattr(pl, "pid_is_running", lambda pid: False)
    assert LlamaCppBackend._tree_kill_surviving_process(4242) is True
    assert calls == [(4242, {"timeout": 5.0, "owner_verified": True})]


def test_the_tree_kill_reports_a_server_it_could_not_stop(monkeypatch):
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(pl, "terminate_pid", lambda pid, **kw: None)
    monkeypatch.setattr(pl, "pid_is_running", lambda pid: True)
    assert LlamaCppBackend._tree_kill_surviving_process(4242) is False


@pytest.mark.parametrize("pid", [None, 0, 1])
def test_the_tree_kill_never_signals_a_reserved_pid(monkeypatch, pid):
    from core.inference.llama_cpp import LlamaCppBackend

    calls = []
    monkeypatch.setattr(pl, "terminate_pid", lambda p, **kw: calls.append(p))
    assert LlamaCppBackend._tree_kill_surviving_process(pid) is False
    assert calls == []


def test_owner_verified_still_leaves_a_recycled_pid_alone(monkeypatch):
    """The waiver is for "cannot prove it is ours", not for "it provably is not"."""
    monkeypatch.setattr(pl, "_pid_alive", lambda p: True)
    monkeypatch.setattr(pl, "_pid_identity", lambda p: "NOW-SOMEONE-ELSE")
    signalled = []
    monkeypatch.setattr(pl, "_windows_terminate_tree", lambda p: signalled.append(p))
    monkeypatch.setattr(pl, "_posix_terminate", lambda p, t: signalled.append(p))
    # A registry of its own, and no breadcrumb: the real ones belong to the process
    # running the suite, and `forget_pid` on the mismatch path writes one out.
    monkeypatch.setattr(pl, "_tracked_pids", {4242: "WHEN-WE-SPAWNED-IT"})
    monkeypatch.setattr(pl, "_tracked_pgids", {})
    monkeypatch.setattr(pl, "_write_breadcrumb", lambda: None)
    pl.terminate_pid(4242, timeout = 0.01, owner_verified = True)
    assert signalled == []
    assert 4242 not in pl._tracked_pids, "a provably recycled pid must not stay recorded"


def test_a_stand_in_process_with_a_pid_is_never_signalled(monkeypatch):
    """The waiver is spent only on a pid we hold the spawning handle for.

    `_process` is a real Popen in the server, but several sibling tests stand in a plain
    object carrying a pid to mean "a server is loaded", and one of them uses this very
    process's pid. Such an object has no `poll`, so the unload reads it as "still
    running" and used to hand the number to a reaper that had been told the owner was
    verified, which waives the "cannot prove this is ours" refusal: the number's current
    holder was terminated. Nothing is signalled here, and the pidfile still goes, which
    is what the unload did before the tree kill existed.
    """
    b = _make_backend()
    b._process = type("P", (), {"pid": 4242})()
    cleared, killed = _instrument(monkeypatch, gone = False)
    b._kill_process()
    assert killed == [], "signalled a pid this backend never spawned"
    assert cleared == [1], "kept the pidfile for a stand-in that can never confirm an exit"


def test_a_stand_in_with_a_pid_leaves_a_real_bystander_running():
    """The same rule against a live process, with nothing about the kill faked.

    A stand-in's pid belongs to whoever holds the number now. Terminating it is the
    failure, so the assertion is on a process that has to survive the unload.
    """
    from core.inference.llama_cpp import LlamaCppBackend

    bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        for _ in range(50):
            if bystander.poll() is None:
                break
            time.sleep(0.05)
        assert bystander.poll() is None, "the bystander never started"
        b = _make_backend()
        b._process = type("P", (), {"pid": bystander.pid})()
        b._clear_server_pid = lambda: None
        b._kill_process()
        time.sleep(1.0)
        assert bystander.poll() is None, (
            f"the unload terminated an unrelated process (rc {bystander.poll()})"
        )
    finally:
        bystander.kill()
        bystander.wait(timeout = 10)
