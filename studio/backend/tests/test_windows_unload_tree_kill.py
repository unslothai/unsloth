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


def test_the_windows_walk_rejects_a_stranger_under_a_reused_intermediate_pid(monkeypatch):
    """The root's own creation time is NOT a sufficient floor.

    The machine this is about, with times as Windows would report them:

      * the Unsloth root starts at t=100;
      * an unrelated process U starts at t=200, created by some pid P;
      * that original P exits;
      * at t=300 Windows hands P's number to a genuine Unsloth child;
      * U still records P as its creator and nothing ever clears that.

    U is later than the ROOT, so a root-only floor admits it, and the survivor sweep
    then passes it to `taskkill /PID <U> /T /F`, which takes down U and everything
    under it: someone's training run, editor or server. Ordering U against its own
    parent's CURRENT creation time rejects it, because a process cannot predate the
    process that created it. Nothing real is signalled here: the table, the identities
    and the kill are all fabricated.
    """
    root, reused_parent, stranger, stranger_child, real_grandchild = 500, 600, 700, 701, 601
    created = {
        root: 100,
        reused_parent: 300,  # P after the reuse: a genuine Unsloth child
        stranger: 200,  # U, created by the OLD holder of P
        stranger_child: 250,  # U's own work, hanging off a link we do not have
        real_grandchild: 400,  # a true descendant of the reused pid
    }
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: f"0:{created[pid]}")
    monkeypatch.setattr(
        pl,
        "_child_pid_map",
        lambda: {
            root: [reused_parent],
            reused_parent: [stranger, real_grandchild],
            stranger: [stranger_child],
        },
    )
    found = [pid for pid, _ in pl.collect_descendants(root)]
    assert stranger not in found, "a stranger predating its own parent was claimed"
    assert stranger_child not in found, "the stranger's subtree was walked"
    assert sorted(found) == [reused_parent, real_grandchild]


def test_the_kill_does_not_re_expand_a_rejected_stranger(monkeypatch):
    """The collector's filter must survive the kill that acts on it.

    `taskkill /T` terminates the named process AND its child processes, and it finds
    them by walking the live parent-pid links -- exactly the links the collector above
    refuses to trust. So U is correctly kept OUT of the collected list, but U still
    records the reused number P as its creator, and a `/T` on P, which IS collected and
    IS identity-verified, rediscovers U through Windows' own walk. The filter is defeated
    by the kill it protects, and someone's training run dies anyway.

    Same fixture as the collector test above, driven one step further.
    """
    root, reused_parent, stranger, stranger_child, real_grandchild = 500, 600, 700, 701, 601
    created = {
        root: 100,
        reused_parent: 300,
        stranger: 200,
        stranger_child: 250,
        real_grandchild: 400,
    }
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: f"0:{created[pid]}")
    monkeypatch.setattr(
        pl,
        "_child_pid_map",
        lambda: {
            root: [reused_parent],
            reused_parent: [stranger, real_grandchild],
            stranger: [stranger_child],
        },
    )
    killed = []
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid: killed.append(pid))

    # Reaching for /T at all is the regression: the expansion happens inside Windows, so
    # no assertion on the resulting pid set can see it once the call has been made.
    def _no_tree_kill(pid):
        raise AssertionError(f"taskkill /T was used on {pid}; it re-expands rejected pids")

    monkeypatch.setattr(pl, "_windows_terminate_tree", _no_tree_kill)

    collected = pl.collect_descendants(root)
    pl._windows_terminate_collected(collected)

    assert stranger not in killed, "the rejected stranger was killed by tree expansion"
    assert stranger_child not in killed, "the stranger's own work was killed with it"
    # Everything the collector DID claim still dies, deepest first.
    assert set(killed) == {reused_parent, real_grandchild}
    assert killed.index(real_grandchild) < killed.index(reused_parent)


def test_the_windows_walk_skips_a_candidate_whose_identity_cannot_be_read(monkeypatch):
    """An unreadable creation time proves nothing, and the action taken on the result
    is a forced tree kill, so the candidate and its subtree are dropped."""
    root, child, grandchild = 500, 600, 700
    created = {root: 100, child: None, grandchild: 900}
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(
        pl,
        "_pid_identity",
        lambda pid: None if created[pid] is None else f"0:{created[pid]}",
    )
    monkeypatch.setattr(pl, "_child_pid_map", lambda: {root: [child], child: [grandchild]})
    assert pl.collect_descendants(root) == []


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

    monkeypatch.setattr(pl, "_windows_terminate_pid", fake_taskkill)
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
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid: asked.append(pid))
    pl._windows_terminate_collected([(4242, "0:1")])
    assert asked == []


def test_the_windows_terminate_works_deepest_first(monkeypatch):
    """A child has to go before the parent whose link named it, or the link is gone."""
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    identities = {10: "0:10", 11: "0:11", 12: "0:12"}
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: identities[pid])
    order = []
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid: order.append(pid))
    monkeypatch.setattr(pl, "_windows_collect_descendants_known", lambda pid: ([], True))
    # collect_descendants emits breadth first, so the parent comes before its child.
    pl._windows_terminate_collected([(10, "0:10"), (11, "0:11"), (12, "0:12")])
    assert order == [12, 11, 10]


def test_the_windows_terminate_leaves_an_unreadable_identity_alone(monkeypatch):
    """Fail closed, not open.

    `_still_the_same` answers "is this provably somebody else", which is right for a
    SIGTERM at a pid we still hold a record for. It is wrong here: this runs AFTER the
    leader's terminate, an unreadable identity is exactly what a recycled pid looks
    like, and the action is `taskkill /T /F`, which reaches everything the number now
    owns. So a pid whose identity cannot be read on either side is skipped.
    """
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    asked = []
    monkeypatch.setattr(pl, "_windows_terminate_tree", lambda pid: asked.append(pid))

    # the identity cannot be read NOW
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: None)
    pl._windows_terminate_collected([(4242, "0:1")])
    assert asked == [], "an unreadable current identity must not be tree killed"

    # the identity was not readable AT COLLECTION either
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: "0:1")
    pl._windows_terminate_collected([(4242, None)])
    assert asked == [], "a survivor collected without an identity must not be tree killed"


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
    b._collect_descendants = lambda *a, **kw: ([], True)
    b._kill_process_group = lambda *a, **kw: None
    b._terminate_descendants = lambda *a, **kw: []
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
        assert (
            bystander.poll() is None
        ), f"the unload terminated an unrelated process (rc {bystander.poll()})"
    finally:
        bystander.kill()
        bystander.wait(timeout = 10)


# ── the surviving-root fallback, and what it leaves behind ──


def test_the_surviving_root_fallback_does_not_re_expand_the_tree(monkeypatch):
    """`terminate_pid` is where the unload's last resort lands, and it used to mean `/T`.

    The collector refuses to trust the live parent-pid links, and `taskkill /T` finds its
    children by walking exactly those links, so routing the fallback through it hands the
    rejected stranger straight back to the kill. Windows does the expansion, so no
    assertion on the resulting pid set can observe it: the tree call is made to raise
    instead, which fails this test the moment anything reaches for it.
    """
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: pid not in dead)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    identities = {500: "0:500", 600: "0:600"}
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: identities.get(pid))
    monkeypatch.setattr(pl, "_tracked_pids", {500: "0:500"})
    monkeypatch.setattr(pl, "_tracked_pgids", {})
    monkeypatch.setattr(pl, "_write_breadcrumb", lambda: None)
    monkeypatch.setattr(pl, "_group_has_members", lambda pgid: False)

    def _no_tree(pid):
        raise AssertionError("taskkill /T re-expands through the links the filter rejects")

    monkeypatch.setattr(pl, "_windows_terminate_tree", _no_tree)
    monkeypatch.setattr(
        pl, "_windows_collect_descendants_known",
        lambda pid: ([(600, "0:600")], True) if pid == 500 else ([], True)
    )
    dead: "set[int]" = set()
    killed = []

    def _kill(pid):
        killed.append(pid)
        dead.add(pid)
        return True

    monkeypatch.setattr(pl, "_windows_terminate_pid", _kill)

    pl.terminate_pid(500, timeout = 0.01, owner_verified = True)
    assert killed == [600, 500], "deepest first, and the validated set only"
    assert 500 not in pl._tracked_pids, "a tree that went down releases its record"


def test_the_surviving_root_fallback_keeps_the_record_when_something_lives(monkeypatch):
    """The other half of the contract the `/T` call carried: False means the record stays.

    Reading it back from the pids matters more here than it did with `/T`, because the
    exit status of one `taskkill /F` says nothing about the descendant killed before it.
    """
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    # The root dies, the descendant does not: exactly the shape a per-pid kill can produce
    # and a single /T status could never report.
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: pid == 600)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    identities = {500: "0:500", 600: "0:600"}
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: identities.get(pid))
    monkeypatch.setattr(pl, "_tracked_pids", {500: "0:500"})
    monkeypatch.setattr(pl, "_tracked_pgids", {})
    monkeypatch.setattr(pl, "_write_breadcrumb", lambda: None)
    monkeypatch.setattr(pl, "_group_has_members", lambda pgid: False)
    monkeypatch.setattr(
        pl, "_windows_collect_descendants_known",
        lambda pid: ([(600, "0:600")], True) if pid == 500 else ([], True)
    )
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid: True)

    pl.terminate_pid(500, timeout = 0.01, owner_verified = True)
    assert 500 in pl._tracked_pids, "dropped the only handle on a worker that is still up"


def test_a_descendant_that_would_not_die_is_reported(monkeypatch):
    """`taskkill /F` can fail, and can report success on a process that has not gone.

    Discarding that answer is what lets the unload delete the record and the pidfile with a
    worker still holding the GPU: nothing else names it after the leader is reaped.
    """
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: f"0:{pid}")
    monkeypatch.setattr(pl, "_windows_collect_descendants_known", lambda pid: ([], True))
    # Returns True, and the process is still there. The read-back is the only thing that
    # can tell the difference.
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid: True)
    assert pl._windows_terminate_collected([(10, "0:10"), (11, "0:11")]) == [11, 10]


def test_a_descendant_that_did_die_is_not_reported(monkeypatch):
    """The ordinary path reports nothing, or every unload would keep its record forever."""
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: f"0:{pid}")
    monkeypatch.setattr(pl, "_windows_collect_descendants_known", lambda pid: ([], True))
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid: True)
    assert pl._windows_terminate_collected([(10, "0:10")]) == []


@pytest.mark.skipif(IS_WINDOWS, reason = "the POSIX arm")
def test_the_posix_sweep_reports_a_child_the_signals_did_not_reach(monkeypatch, tree):
    """Same contract on POSIX, with the real child and the real liveness probe.

    SIGKILL cannot be caught, but it is not instantaneous either, and a worker asleep in a
    driver ioctl outlives it. The signals are stubbed out rather than the process being
    made unkillable, which is the only way to reach that state deterministically.
    """
    leader, child_pid = tree
    collected = pl.collect_descendants(leader.pid)
    assert child_pid in [pid for pid, _ in collected]
    monkeypatch.setattr(pl.os, "kill", lambda pid, sig: None)
    survivors = pl.terminate_descendants(collected, timeout = 0.2)
    assert child_pid in survivors, "a child that is plainly still running was reported dead"
    assert _alive(child_pid)


def test_an_unload_keeps_the_pidfile_when_a_descendant_survives(monkeypatch):
    """The leader exited cleanly, so the unload would have dropped the record and the
    pidfile. Its worker is still up, and after the leader is reaped those are the only
    things that name anything about this server to the next launch."""
    b = _make_backend()
    b._process.pid = 4242
    b._process.poll.return_value = 0  # the leader itself went down on the terminate
    b._terminate_descendants = lambda *a, **kw: [777]
    cleared, killed = _instrument(monkeypatch, gone = True)
    b._kill_process()
    assert killed == [], "the leader exited, so there is nothing to tree kill"
    assert cleared == [], "dropped the pidfile with a worker of this server still running"


def test_an_unload_clears_the_pidfile_when_every_descendant_died(monkeypatch):
    """The common path, unchanged: nothing survived, so nothing is kept."""
    b = _make_backend()
    b._process.pid = 4242
    b._process.poll.return_value = 0
    b._terminate_descendants = lambda *a, **kw: []
    cleared, killed = _instrument(monkeypatch, gone = True)
    b._kill_process()
    assert cleared == [1]


def test_a_surviving_descendant_is_adopted_so_a_later_sweep_can_reach_it(monkeypatch):
    """The leader's record is keyed on the leader. Once it is gone the survivor needs a
    record of its own, and on Windows there is no process group to stand in for one."""
    from core.inference.llama_cpp import LlamaCppBackend

    adopted = []
    monkeypatch.setattr(pl, "terminate_descendants", lambda collected, timeout: [777])
    monkeypatch.setattr(pl, "adopt_pid", adopted.append)
    assert LlamaCppBackend._terminate_descendants([(777, "0:777")]) == [777]
    assert adopted == [777]


def test_a_sweep_that_raised_names_every_pid_it_had(monkeypatch):
    """An exception is "unknown", not "none". The pids were collected, so they can still
    be named, and naming them is the whole point of the return value."""
    from core.inference.llama_cpp import LlamaCppBackend

    def _raises(collected, timeout):
        raise RuntimeError("Toolhelp snapshot unavailable")

    monkeypatch.setattr(pl, "terminate_descendants", _raises)
    assert LlamaCppBackend._terminate_descendants([(777, "0:777"), (778, None)]) == [777, 778]


# ── an answer that could not be obtained is not an answer of "none" ──


def test_an_unreadable_process_table_is_indeterminate_not_empty(monkeypatch):
    """`[]` has two meanings and only one of them is safe to act on.

    The Toolhelp snapshot can fail outright, and the walk can fail partway. Reported as an
    empty tree, the unload terminates the leader, finds no survivors, and deletes the
    record and the pidfile -- so the workers the failed snapshot never listed are left
    running with nothing naming them, which is the leak this collector was added to close,
    reached through a failed snapshot instead of through a reparent.
    """
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: "0:500")
    monkeypatch.setattr(pl, "_child_pid_map", lambda: None)
    assert pl.collect_descendants_known(500) == ([], False)
    # The old spelling still answers with the list alone, so callers that only want
    # something to signal are unaffected.
    assert pl.collect_descendants(500) == []

    # A readable table with no children is the other meaning, and it must stay knowable.
    monkeypatch.setattr(pl, "_child_pid_map", lambda: {999: [1000]})
    assert pl.collect_descendants_known(500) == ([], True)


def test_a_root_with_no_readable_identity_is_indeterminate(monkeypatch):
    """The root's creation time is the ancestry floor, so without it the walk can prove
    nothing about anything -- which is not the same as proving there is nothing."""
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: None)
    monkeypatch.setattr(pl, "_child_pid_map", lambda: {500: [600]})
    assert pl.collect_descendants_known(500) == ([], False)


def test_an_unenumerable_tree_keeps_the_pidfile(monkeypatch):
    """The leader exited cleanly and the sweep found nothing, but the sweep ran over a list
    that was never built. Keeping the record costs a stale pidfile; dropping it costs the
    only name anything has for a process that may still hold the GPU."""
    b = _make_backend()
    b._process.pid = 4242
    b._process.poll.return_value = 0
    b._collect_descendants = lambda *a, **kw: ([], False)
    b._terminate_descendants = lambda *a, **kw: []
    cleared, _killed = _instrument(monkeypatch, gone = True)
    b._kill_process()
    assert cleared == [], "dropped the pidfile after a walk that could not be made"


def test_a_live_but_unverifiable_descendant_is_reported_not_signalled(monkeypatch):
    """Refusing to kill it is right. Saying nothing about it is not.

    An identity that cannot be read on either side is exactly what a recycled pid looks
    like, so it must not be signalled. It is also what a live worker under momentary
    handle pressure looks like, and omitting it from the survivor list told the caller the
    sweep was complete.
    """
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: None)  # unreadable NOW
    monkeypatch.setattr(pl, "_windows_collect_descendants_known", lambda pid: ([], True))
    killed = []
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid: killed.append(pid))
    assert pl._windows_terminate_collected([(11, "0:11")]) == [11]
    assert killed == [], "an unverifiable pid must never be signalled"


def test_a_pid_that_provably_moved_on_is_neither_killed_nor_adopted(monkeypatch):
    """The boundary. A number that now belongs to a stranger is not our leaked worker, so
    adopting it would put someone else's process into our lifetime record."""
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: "0:999")  # somebody else
    monkeypatch.setattr(pl, "_windows_collect_descendants_known", lambda pid: ([], True))
    killed = []
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid: killed.append(pid))
    assert pl._windows_terminate_collected([(11, "0:11")]) == []
    assert killed == []


def test_the_two_provable_questions_are_asked_separately(monkeypatch):
    """`_provably_the_same` folds "somebody else" and "cannot tell" into one False, which
    is right for deciding whether to signal and wrong for deciding whether to report."""
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: "0:11")
    assert pl._provably_the_same(11, "0:11") is True
    assert pl._provably_different(11, "0:11") is False
    assert pl._provably_the_same(11, "0:99") is False
    assert pl._provably_different(11, "0:99") is True

    monkeypatch.setattr(pl, "_pid_identity", lambda pid: None)
    # Unknown on one side: neither question can be answered yes.
    assert pl._provably_the_same(11, "0:11") is False
    assert pl._provably_different(11, "0:11") is False
    assert pl._provably_the_same(11, None) is False
    assert pl._provably_different(11, None) is False


# ── an enumeration that failed is not an enumeration that ended ──


def test_a_walk_that_failed_partway_is_not_a_table(monkeypatch):
    """`Process32NextW` returning FALSE is not "that was the last one".

    It is also every way the walk can fail partway, and only ERROR_NO_MORE_FILES says the
    list ended. Reading a termination error as the end returns a table missing whatever
    came after it, and a table is read as an ANSWER: the unload then finds no survivors and
    deletes the record and the pidfile for workers the walk never reached. The Windows
    scanner in studio/src-tauri/src/process.rs already makes this distinction.
    """
    source = Path(pl.__file__).read_text(encoding = "utf-8")
    start = source.index("def _windows_child_pid_map")
    body = source[start : source.index("\ndef ", start + 10)]
    assert "ERROR_NO_MORE_FILES = 18" in body
    assert "ctypes.get_last_error() != ERROR_NO_MORE_FILES" in body
    # And the branch returns None, which is what `known` is derived from, rather than the
    # partial table.
    error_branch = body[body.index("ctypes.get_last_error()"):]
    assert error_branch.split("\n")[1].strip() == "return None", error_branch.split("\n")[1]


def test_a_failed_late_walk_kills_the_survivor_and_reports_it(monkeypatch):
    """The second walk covers whatever the survivor started AFTER the snapshot.

    Failing it and carrying on killed the survivor and then reported nothing, so a
    grandchild only that walk could have named was left running with the record and the
    pidfile deleted. The survivor itself is still killed -- it is collected and
    identity-verified, and refusing on a walk BELOW it would leave the process this sweep
    exists for running -- but the tree is reported unresolved so the record stays.
    """
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: f"0:{pid}")
    monkeypatch.setattr(pl, "_windows_collect_descendants_known", lambda pid: ([], False))
    killed = []
    dead: "set[int]" = set()

    # The kill WORKS here, so the survivor read-back finds nothing and the unresolved
    # report is the only thing that can name this tree. Without that, a pid that merely
    # outlived its own kill would satisfy the assertion for the wrong reason.
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: pid not in dead)

    def _kill(pid):
        killed.append(pid)
        dead.add(pid)
        return True

    monkeypatch.setattr(pl, "_windows_terminate_pid", _kill)
    assert pl._windows_terminate_collected([(11, "0:11")]) == [11]
    assert killed == [11], "the collected, verified survivor must still be killed"


def test_a_successful_late_walk_still_reports_nothing(monkeypatch):
    """The ordinary path is unchanged, or every unload would keep its record."""
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: f"0:{pid}")
    monkeypatch.setattr(pl, "_windows_collect_descendants_known", lambda pid: ([], True))
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid: True)
    assert pl._windows_terminate_collected([(11, "0:11")]) == []


def test_an_unenumerable_tree_is_not_a_completed_tree_kill(monkeypatch):
    """`terminate_pid` forgets the record on True, so an unenumerable tree cannot answer
    True after a root-only kill: the workers the walk never named would be left with
    nothing pointing at them."""
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: False)  # the root itself went down
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: f"0:{pid}")
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid: True)

    monkeypatch.setattr(pl, "_windows_collect_descendants_known", lambda pid: ([], False))
    assert pl._windows_terminate_validated_tree(500) is False

    # And a walk that DID run, with nothing under the root, is still a completed kill.
    monkeypatch.setattr(pl, "_windows_collect_descendants_known", lambda pid: ([], True))
    assert pl._windows_terminate_validated_tree(500) is True
