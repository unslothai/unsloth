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

import contextlib
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


@contextlib.contextmanager
def _signals_that_never_land(pids):
    """`os.kill` swallows the stop signals aimed at *pids*, and only for this block.

    SIGKILL cannot be caught, so the way a process survives one is that the signal never
    lands: denied, or the target is wedged in an uninterruptible driver ioctl. Faking that
    is the only way to reach the state deterministically.

    Deliberately NOT `monkeypatch.setattr`. A fixture's finalizer runs BEFORE monkeypatch
    undoes anything, so a test that fakes `os.kill` for the whole test hands the fake to
    its own tree fixture's cleanup: `_hard_kill` becomes a no-op and the processes are
    still sleeping when the run ends, one set per run. The block has to end before the
    fixture tears down, and `finally` is what guarantees it even when an assert fails.

    Signal 0 passes straight through, so the liveness probe stays real.
    """
    blocked = set(pids)
    real_kill = os.kill

    def _kill(pid, sig, *rest):
        if pid in blocked and sig in (signal.SIGTERM, signal.SIGKILL):
            return None
        return real_kill(pid, sig, *rest)

    os.kill = _kill
    try:
        yield
    finally:
        os.kill = real_kill


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
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid, identity = None: killed.append(pid))

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

    def fake_taskkill(pid, identity = None):
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
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid, identity = None: asked.append(pid))
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
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid, identity = None: order.append(pid))
    monkeypatch.setattr(
        pl, "_windows_collect_descendants_known", lambda pid, identity = None: ([], True)
    )
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
        pl,
        "_windows_collect_descendants_known",
        lambda pid, identity = None: ([(600, "0:600")], True) if pid == 500 else ([], True),
    )
    dead: "set[int]" = set()
    killed = []

    def _kill(pid, identity = None):
        killed.append(pid)
        dead.add(pid)
        return True

    monkeypatch.setattr(pl, "_windows_terminate_pid", _kill)

    pl.terminate_pid(500, timeout = 0.01, owner_verified = True)
    # Root first, then the snapshot deepest-first. The root is stopped the moment the
    # snapshot exists, because working through the snapshot spends one `taskkill` per
    # descendant and a root that is still running can spawn into a window that nothing
    # afterwards examines. Its descendants are already named, with their identities, so
    # stopping it cannot lose them.
    assert killed == [500, 600], "the root is stopped first, then the validated set only"
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
        pl,
        "_windows_collect_descendants_known",
        lambda pid, identity = None: ([(600, "0:600")], True) if pid == 500 else ([], True),
    )
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid, identity = None: True)

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
    monkeypatch.setattr(
        pl, "_windows_collect_descendants_known", lambda pid, identity = None: ([], True)
    )
    # Returns True, and the process is still there. The read-back is the only thing that
    # can tell the difference.
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid, identity = None: True)
    assert pl._windows_terminate_collected([(10, "0:10"), (11, "0:11")]) == [
        (11, "0:11"),
        (10, "0:10"),
    ]


def test_a_descendant_that_did_die_is_not_reported(monkeypatch):
    """The ordinary path reports nothing, or every unload would keep its record forever."""
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: f"0:{pid}")
    monkeypatch.setattr(
        pl, "_windows_collect_descendants_known", lambda pid, identity = None: ([], True)
    )
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid, identity = None: True)
    assert pl._windows_terminate_collected([(10, "0:10")]) == []


@pytest.mark.skipif(IS_WINDOWS, reason = "the POSIX arm")
def test_the_posix_sweep_reports_a_child_the_signals_did_not_reach(tree):
    """Same contract on POSIX, with the real child and the real liveness probe.

    SIGKILL cannot be caught, but it is not instantaneous either, and a worker asleep in a
    driver ioctl outlives it. The signals are stubbed out rather than the process being
    made unkillable, which is the only way to reach that state deterministically.
    """
    leader, child_pid = tree
    collected = pl.collect_descendants(leader.pid)
    assert child_pid in [pid for pid, _ in collected]
    with _signals_that_never_land([child_pid]):
        survivors = pl.terminate_descendants(collected, timeout = 0.2)
        assert child_pid in [
            pid for pid, _ in survivors
        ], "a child that is plainly still running was reported dead"
        assert _alive(child_pid)


def test_an_unload_keeps_the_pidfile_when_a_descendant_survives(monkeypatch):
    """The leader exited cleanly, so the unload would have dropped the record and the
    pidfile. Its worker is still up, and after the leader is reaped those are the only
    things that name anything about this server to the next launch."""
    b = _make_backend()
    b._process.pid = 4242
    b._process.poll.return_value = 0  # the leader itself went down on the terminate
    b._terminate_descendants = lambda *a, **kw: [(777, "0:777")]
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
    monkeypatch.setattr(
        pl,
        "terminate_descendants",
        lambda collected, timeout: [(777, "0:777")],
    )
    monkeypatch.setattr(
        pl,
        "adopt_pid",
        lambda pid, identity = None, from_snapshot = False: adopted.append(
            (pid, identity, from_snapshot)
        ),
    )
    assert LlamaCppBackend._terminate_descendants([(777, "0:777")]) == [(777, "0:777")]
    # The identity the sweep verified travels with the pid, so the adopt cannot record a
    # process that took the number over in between -- and it is flagged as coming from a
    # snapshot, so a survivor whose identity could NOT be read is refused rather than having
    # one captured for it now, which would record whoever holds the number at this moment.
    assert adopted == [(777, "0:777", True)]


def test_a_sweep_that_raised_names_every_pid_it_had(monkeypatch):
    """An exception is "unknown", not "none". The pids were collected, so they can still
    be named, and naming them is the whole point of the return value."""
    from core.inference.llama_cpp import LlamaCppBackend

    def _raises(collected, timeout):
        raise RuntimeError("Toolhelp snapshot unavailable")

    monkeypatch.setattr(pl, "terminate_descendants", _raises)
    assert LlamaCppBackend._terminate_descendants([(777, "0:777"), (778, None)]) == [
        (777, "0:777"),
        (778, None),
    ]


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
    monkeypatch.setattr(
        pl, "_windows_collect_descendants_known", lambda pid, identity = None: ([], True)
    )
    killed = []
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid, identity = None: killed.append(pid))
    assert pl._windows_terminate_collected([(11, "0:11")]) == [(11, "0:11")]
    assert killed == [], "an unverifiable pid must never be signalled"


def test_a_pid_that_provably_moved_on_is_neither_killed_nor_adopted(monkeypatch):
    """The boundary. A number that now belongs to a stranger is not our leaked worker, so
    adopting it would put someone else's process into our lifetime record."""
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: "0:999")  # somebody else
    monkeypatch.setattr(
        pl, "_windows_collect_descendants_known", lambda pid, identity = None: ([], True)
    )
    killed = []
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid, identity = None: killed.append(pid))
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
    error_branch = body[body.index("ctypes.get_last_error()") :]
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
    monkeypatch.setattr(
        pl, "_windows_collect_descendants_known", lambda pid, identity = None: ([], False)
    )
    killed = []
    dead: "set[int]" = set()

    # The kill WORKS here, so the survivor read-back finds nothing and the unresolved
    # report is the only thing that can name this tree. Without that, a pid that merely
    # outlived its own kill would satisfy the assertion for the wrong reason.
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: pid not in dead)

    def _kill(pid, identity = None):
        killed.append(pid)
        dead.add(pid)
        return True

    monkeypatch.setattr(pl, "_windows_terminate_pid", _kill)
    assert pl._windows_terminate_collected([(11, "0:11")]) == [(11, "0:11")]
    assert killed == [11], "the collected, verified survivor must still be killed"


def test_a_successful_late_walk_still_reports_nothing(monkeypatch):
    """The ordinary path is unchanged, or every unload would keep its record."""
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: f"0:{pid}")
    monkeypatch.setattr(
        pl, "_windows_collect_descendants_known", lambda pid, identity = None: ([], True)
    )
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid, identity = None: True)
    assert pl._windows_terminate_collected([(11, "0:11")]) == []


def test_an_unenumerable_tree_is_not_a_completed_tree_kill(monkeypatch):
    """`terminate_pid` forgets the record on True, so an unenumerable tree cannot answer
    True after a root-only kill: the workers the walk never named would be left with
    nothing pointing at them."""
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: False)  # the root itself went down
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: f"0:{pid}")
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid, identity = None: True)

    monkeypatch.setattr(
        pl, "_windows_collect_descendants_known", lambda pid, identity = None: ([], False)
    )
    assert pl._windows_terminate_validated_tree(500) is False

    # And a walk that DID run, with nothing under the root, is still a completed kill.
    monkeypatch.setattr(
        pl, "_windows_collect_descendants_known", lambda pid, identity = None: ([], True)
    )
    assert pl._windows_terminate_validated_tree(500) is True


def test_a_live_unverifiable_descendant_is_an_incomplete_tree_kill(monkeypatch):
    """`terminate_pid` forgets the record on True, so this read-back has to fail closed.

    A descendant that is still alive and whose creation time cannot be read right now is
    not proof of a recycled pid: handle pressure and access denial look exactly the same as
    a number that has moved on. Ignoring it dropped the last persistent handle on a worker
    that was neither confirmed dead nor shown to be somebody else.
    """
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid, identity = None: True)
    monkeypatch.setattr(
        pl,
        "_windows_collect_descendants_known",
        lambda pid, identity = None: ([(600, "0:600")], True) if pid == 500 else ([], True),
    )
    # The root dies, the descendant survives, and its identity is unreadable NOW.
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: pid == 600)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: None)
    assert pl._windows_terminate_validated_tree(500) is False

    # And a descendant that PROVABLY belongs to someone else is not ours to wait for, so
    # the kill is still complete: this must not turn every recycled number into a kept
    # record, which would be the leak in the other direction.
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: "0:999")
    assert pl._windows_terminate_validated_tree(500) is True


def test_a_recycled_survivor_is_not_adopted(monkeypatch):
    """A survivor can exit between the sweep's last liveness check and the adopt.

    The number is then free, and adopting it records a stranger as this process's child --
    and where a job object is active, puts it in a job that kills its members when the app
    closes. The identity the sweep verified travels with the pid so the adopt can tell.
    """
    recorded = {}
    monkeypatch.setattr(pl, "_signalable", lambda pid: True)
    monkeypatch.setattr(pl, "_adopt_fork_reset", lambda: None)
    monkeypatch.setattr(pl, "_own_process_group", lambda pid: None)
    monkeypatch.setattr(pl, "_write_breadcrumb", lambda: None)
    monkeypatch.setattr(pl, "_tracked_pids", recorded)
    # The number now belongs to something that started later.
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: "9:9999")

    pl.adopt_pid(4242, "0:4242")
    assert recorded == {}, "adopted a pid that is provably somebody else"

    # Unknown adopts nobody either: an identity that cannot be read is not a match.
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: None)
    pl.adopt_pid(4242, "0:4242")
    assert recorded == {}, "adopted a pid whose identity could not be confirmed"

    # And the pid that IS still the same is adopted, with the identity that was verified.
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: "0:4242")
    pl.adopt_pid(4242, "0:4242")
    assert recorded == {4242: "0:4242"}

    # A caller with no identity to offer is unchanged: this is the spawn path.
    recorded.clear()
    monkeypatch.setattr(pl, "_identity_for_record", lambda pid: "read-now")
    pl.adopt_pid(4343)
    assert recorded == {4343: "read-now"}


class _FakeKernel32:
    """Enough of kernel32 for the adopt path: every name is a settable stub.

    `_win_signatures` assigns argtypes and restype onto each entry point, so the attributes
    have to exist and carry assignment, and the three the adopt path actually calls are
    overridden by the test.
    """

    def __init__(self, **calls):
        self._stubs = {}
        for name, fn in calls.items():
            self._stubs[name] = fn

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        stub = self._stubs.get(name)
        if stub is None:

            def stub(*args, **kwargs):  # noqa: ANN001,ANN202 -- a no-op entry point
                return 1

            self._stubs[name] = stub
        return stub


def test_the_handle_identity_is_read_through_the_handle(monkeypatch):
    """The creation time has to come from the handle, not from the number again.

    A handle pins the process it was opened on, so what it reports is what the later call on
    that same handle will act upon. Unreadable times are None, which the caller treats as
    "not proven" rather than as a match.
    """
    import ctypes

    def _times(handle, created_ptr, *rest):
        if handle != 77:
            return 0
        created_ptr._obj.dwHighDateTime = 7
        created_ptr._obj.dwLowDateTime = 4242
        return 1

    kernel32 = _FakeKernel32(GetProcessTimes = _times)
    assert pl._windows_identity_of_handle(kernel32, 77) == "7:4242"
    # A failed read is None, never a fabricated match.
    assert pl._windows_identity_of_handle(kernel32, 78) is None
    del ctypes


def test_a_pid_recycled_before_the_job_assignment_is_not_assigned(monkeypatch):
    """The pid check happens before the record write and the breadcrumb flush.

    The process can exit anywhere in that interval, its number be taken by a stranger, and
    the `OpenProcess` that follows then lands on the stranger. Assigning that handle to a
    job whose limit is kill-on-close means closing Unsloth kills an unrelated process, so
    the creation time is re-read THROUGH the handle that is about to be assigned.
    """
    import ctypes

    assigned: "list[int]" = []
    opened_rights: "list[int]" = []
    closed: "list[int]" = []
    handle_identity = {"value": "0:4242"}

    def _open(rights, inherit, pid):
        opened_rights.append(rights)
        return 77

    def _times(handle, created_ptr, *rest):
        spelling = handle_identity["value"]
        if spelling is None:
            return 0
        high, low = spelling.split(":")
        created_ptr._obj.dwHighDateTime = int(high)
        created_ptr._obj.dwLowDateTime = int(low)
        return 1

    kernel32 = _FakeKernel32(
        OpenProcess = _open,
        GetProcessTimes = _times,
        AssignProcessToJobObject = lambda job, handle: assigned.append(handle) or 1,
        CloseHandle = lambda handle: closed.append(handle) or 1,
    )
    # ctypes has no WinDLL off Windows, so it is created rather than replaced.
    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **k: kernel32, raising = False)
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_win_job_handle", 4321)
    monkeypatch.setattr(pl, "_signalable", lambda pid: True)
    monkeypatch.setattr(pl, "_adopt_fork_reset", lambda: None)
    monkeypatch.setattr(pl, "_own_process_group", lambda pid: None)
    monkeypatch.setattr(pl, "_write_breadcrumb", lambda: None)
    monkeypatch.setattr(pl, "_tracked_pids", {})
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: "0:4242")

    # The number was taken by something that started later, in the window the pid check
    # cannot cover.
    handle_identity["value"] = "9:9999"
    pl.adopt_pid(4242, "0:4242")
    assert assigned == [], "a stranger was assigned to the kill-on-close job"
    assert closed == [77], "the handle was leaked"

    # Unreadable times are not a match either.
    handle_identity["value"] = None
    pl.adopt_pid(4242, "0:4242")
    assert assigned == [], "assigned a handle whose process could not be identified"

    # The process that is still the same one is assigned, which is the whole point of the
    # belt-and-suspenders pass.
    handle_identity["value"] = "0:4242"
    pl.adopt_pid(4242, "0:4242")
    assert assigned == [77], assigned
    # And the read right is requested, or GetProcessTimes could never succeed on the handle.
    assert all(rights & 0x1000 for rights in opened_rights), opened_rights


def test_the_forced_kill_goes_through_the_handle_it_verified(monkeypatch):
    """A pid is a NAME, and `taskkill /PID` resolves it inside itself.

    That lookup happens after the caller's identity check and after this process has spent
    time getting to the call, so a number freed in between carries the kill onto whatever
    inherited it. A handle is the process rather than its name: the creation time is read
    from the same object TerminateProcess then ends.
    """
    import ctypes
    import subprocess

    terminated: "list[int]" = []
    closed: "list[int]" = []
    handle_identity = {"value": "0:4242"}
    spawned: "list[list]" = []

    def _times(handle, created_ptr, *rest):
        spelling = handle_identity["value"]
        if spelling is None:
            return 0
        high, low = spelling.split(":")
        created_ptr._obj.dwHighDateTime = int(high)
        created_ptr._obj.dwLowDateTime = int(low)
        return 1

    kernel32 = _FakeKernel32(
        OpenProcess = lambda rights, inherit, pid: 91,
        GetProcessTimes = _times,
        TerminateProcess = lambda handle, code: terminated.append(handle) or 1,
        CloseHandle = lambda handle: closed.append(handle) or 1,
    )
    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **k: kernel32, raising = False)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: spawned.append(a)
        or (_ for _ in ()).throw(
            AssertionError("taskkill was spawned for a pid a handle could answer for")
        ),
    )

    assert pl._windows_terminate_pid(4242, "0:4242") is True
    assert terminated == [91], terminated
    assert closed == [91], "the handle was leaked"

    # The number now belongs to something that started later: refused, and nothing is
    # signalled by any route.
    terminated.clear()
    handle_identity["value"] = "9:9999"
    assert pl._windows_terminate_pid(4242, "0:4242") is False
    assert terminated == []

    # Unreadable is not a match either.
    handle_identity["value"] = None
    assert pl._windows_terminate_pid(4242, "0:4242") is False
    assert terminated == []

    # A handle that cannot be opened at all is "cannot answer", not "wrong process", so the
    # old ladder still runs -- but only once ownership has been re-established on the number,
    # immediately before the spawn. `taskkill /PID` resolves that number itself, which is the
    # race the handle route exists to close.
    kernel32._stubs["OpenProcess"] = lambda rights, inherit, pid: 0
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: spawned.append(a[0]) or _types.SimpleNamespace(returncode = 0),
    )
    monkeypatch.setattr(pl, "_provably_the_same", lambda pid, identity: True)
    assert pl._windows_terminate_pid(4242, "0:4242") is True
    assert spawned and spawned[-1][0] == "taskkill", spawned

    # And when it cannot be established, nothing is spawned at all: the old code ran taskkill
    # first and only checked afterwards, so the one signal that could hit a stranger was the
    # one nothing had vouched for.
    spawned.clear()
    monkeypatch.setattr(pl, "_provably_the_same", lambda pid, identity: False)
    assert pl._windows_terminate_pid(4242, "0:4242") is False
    assert spawned == [], spawned
    # With no identity there is nothing to establish, so the same refusal applies.
    monkeypatch.setattr(pl, "_provably_the_same", lambda pid, identity: identity is not None)
    assert pl._windows_terminate_pid(4242) is False
    assert spawned == [], spawned


def test_no_windows_kill_path_calls_taskkill_slash_t_any_more():
    """The validated collector is only a filter while every forced kill goes through it.

    `taskkill /T` re-walks the live parent-pid links, so any call site that keeps it hands
    the rejected stranger back to the kill -- including the deferred one, where a record is
    reaped at the next application start.
    """
    import ast
    from pathlib import Path

    tree = ast.parse(Path(pl.__file__).read_text(encoding = "utf-8"))
    callers = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_windows_terminate_tree"
    ]
    assert callers == [], "a kill path still expands the tree through taskkill /T"


def test_an_unreadable_child_identity_makes_the_walk_incomplete(monkeypatch):
    """Toolhelp listed it and this cannot classify it: that is a candidate, not an absence.

    Reported complete, the unload terminated the leader, saw no survivors and deleted the
    record and the pidfile while the child was still running. It is still never signalled --
    unknown is not a licence to kill -- only never silently dropped.
    """
    root, child = 5000, 5001
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_child_pid_map", lambda: {root: [child]})
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(
        pl,
        "_pid_identity",
        lambda pid: "0:100" if pid == root else None,
    )
    found, known = pl.collect_descendants_known(root)
    assert found == []
    assert known is False, "a live child this walk could not classify was reported as absent"

    # A pid that has GONE says nothing about completeness: there is nothing left to name.
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: pid == root)
    found, known = pl.collect_descendants_known(root)
    assert (found, known) == ([], True)


def test_an_unverifiable_late_descendant_is_reported_not_dropped(monkeypatch):
    """The second walk, under a survivor, had two outcomes where the first has three.

    An identity that becomes unreadable between the late walk and the revalidation is not a
    recycled pid, and dropping it let `attempted` come back empty while the descendant was
    still running, so the caller deleted the only records naming it.
    """
    survivor, late_pid = 6000, 6001
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_signalable", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid, identity = None: None)
    monkeypatch.setattr(
        pl,
        "_windows_collect_descendants_known",
        lambda pid, identity = None: ([(late_pid, "0:6001")], True),
    )
    # The survivor still reads as itself; the late descendant has become unreadable.
    monkeypatch.setattr(
        pl,
        "_pid_identity",
        lambda pid: "0:6000" if pid == survivor else None,
    )
    reported = pl._windows_terminate_collected([(survivor, "0:6000")])
    assert late_pid in [pid for pid, _ in reported], reported

    # A late descendant that is PROVABLY somebody else is still ignored.
    monkeypatch.setattr(
        pl,
        "_pid_identity",
        lambda pid: "0:6000" if pid == survivor else "9:9999",
    )
    reported = pl._windows_terminate_collected([(survivor, "0:6000")])
    assert late_pid not in [pid for pid, _ in reported], reported


def test_the_root_is_stopped_before_its_snapshot_is_worked_through(monkeypatch):
    """Each `taskkill` has a 15 second ceiling, so working through a snapshot of N
    descendants leaves a root that is still running for as long as N of those.

    Anything it starts in that window is in no snapshot, the read-back examines only what
    WAS snapshotted, and the caller is told the tree is gone. The root therefore goes first,
    the moment the snapshot exists.
    """
    root, child = 700, 701
    spawned_during_teardown = []
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_signalable", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(
        pl,
        "_pid_identity",
        lambda pid: {root: "0:700", child: "0:701"}.get(pid),
    )
    monkeypatch.setattr(
        pl,
        "_windows_collect_descendants_known",
        lambda pid, identity = None: ([(child, "0:701")], True) if pid == root else ([], True),
    )
    dead: "set[int]" = set()
    order = []

    def _kill(pid, identity = None):
        order.append(pid)
        dead.add(pid)
        # A root that is still alive keeps spawning. This is what the old order allowed.
        if root not in dead:
            spawned_during_teardown.append(len(order))
        return True

    monkeypatch.setattr(pl, "_windows_terminate_pid", _kill)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: pid not in dead)

    assert pl._windows_terminate_validated_tree(root) is True
    assert order[0] == root, order
    assert (
        spawned_during_teardown == []
    ), "the root was still running while its snapshot was being killed"


def test_a_child_started_after_the_snapshot_is_still_reached(monkeypatch):
    """The snapshot is taken once, and each `taskkill` after it can take 15 seconds.

    A descendant therefore has a long window in which to start a child of its own before
    its own turn comes. That child is in no snapshot, so a read-back over the snapshot
    alone reports the tree gone, `terminate_pid` drops the record and the pidfile, and the
    late child keeps its GPU memory and its port with nothing left naming it. Each
    descendant is re-collected immediately before it is killed, through the same
    creation-time validation, so the late child is reached instead.
    """
    root, child, late = 800, 801, 802
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_signalable", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    identities = {root: "0:800", child: "0:801", late: "0:802"}
    monkeypatch.setattr(pl, "_pid_identity", identities.get)
    snapshot_taken: "list[int]" = []

    def _collect(pid, identity = None):
        if pid == root:
            snapshot_taken.append(pid)
            # The late child does not exist yet: it is started while the root is being
            # killed, which is after this walk has already returned.
            return [(child, "0:801")], True
        if pid == child:
            # The re-collect at kill time, which is the only walk that can see it.
            return [(late, "0:802")], True
        return [], True

    monkeypatch.setattr(pl, "_windows_collect_descendants_known", _collect)
    dead: "set[int]" = set()
    order: "list[int]" = []

    def _kill(pid, identity = None):
        order.append(pid)
        dead.add(pid)
        return True

    monkeypatch.setattr(pl, "_windows_terminate_pid", _kill)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: pid not in dead)

    assert pl._windows_terminate_validated_tree(root) is True
    assert late in order, "a child started after the snapshot was never signalled"
    assert order.index(late) < order.index(child), order
    assert order[0] == root, order


def test_an_unaccounted_late_child_keeps_the_record(monkeypatch):
    """And when that re-collect cannot be done, the answer is not "the tree is gone".

    True here is what makes `terminate_pid` call `forget_pid`. A walk that failed below a
    descendant has not shown there is nothing under it, so the tree stands and the record
    that names it outlives the call.
    """
    root, child = 900, 901
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_signalable", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", {root: "0:900", child: "0:901"}.get)
    monkeypatch.setattr(
        pl,
        "_windows_collect_descendants_known",
        lambda pid, identity = None: ([(child, "0:901")], True) if pid == root else ([], False),
    )
    dead: "set[int]" = set()
    monkeypatch.setattr(
        pl, "_windows_terminate_pid", lambda pid, identity = None: dead.add(pid) or True
    )
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: pid not in dead)

    assert pl._windows_terminate_validated_tree(root) is False


def test_the_fallback_signal_revalidates_the_pid(monkeypatch):
    """`taskkill` has a fifteen second ceiling, and the fallback runs when it failed.

    The caller's identity check happened before that call. The process can exit inside it --
    which is the case the fallback exists for -- and Windows hands the number on, so the
    bare `os.kill` there killed whoever holds it now. An identity that cannot be proven the
    same means no signal at all.
    """
    import subprocess as real_subprocess

    signalled: "list[int]" = []

    def _taskkill_fails(*args, **kwargs):
        raise OSError("taskkill is not on PATH")

    monkeypatch.setattr(real_subprocess, "run", _taskkill_fails)
    monkeypatch.setattr(pl.os, "kill", lambda pid, sig: signalled.append(pid))

    # Recycled while taskkill ran: the number now reads as a different process. No colon in
    # either, since on Linux `_same_identity` compares only what precedes the first one.
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: "999")
    assert pl._windows_terminate_pid(4242, "111") is False
    assert signalled == [], "a recycled pid was signalled by the fallback"

    # Still ours: the fallback is exactly what it was.
    assert pl._windows_terminate_pid(4242, "999") is False
    assert signalled == [4242]

    # And with nothing to check against, the fallback stands down rather than guessing.
    signalled.clear()
    assert pl._windows_terminate_pid(4242) is False
    assert signalled == []


def test_a_number_recycled_between_the_snapshot_and_the_identity_read_is_rejected(monkeypatch):
    """The snapshot lists pids; the identity of each is read after it.

    A child that exits in between frees its number at once, and the identity then describes
    whatever took it. That replacement is newer than the parent floor, so the ancestry test
    admits it, and every later check re-verifies the stranger right up to the forced kill.
    A process created after the snapshot was taken cannot be one the snapshot listed.
    """
    root, child = 300, 301
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_child_pid_map", lambda: {root: [child]})
    # The snapshot was taken at 1000; the number was reused by a process that started at
    # 2000, which is after the reading below and therefore after the snapshot.
    monkeypatch.setattr(pl, "_windows_filetime_now", lambda: 1000)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: {root: "0:500", child: "0:2000"}.get(pid))
    monkeypatch.setattr(
        pl,
        "_windows_creation_time",
        lambda identity: (None if identity is None else int(identity.split(":")[1])),
    )

    found, known = pl._windows_collect_descendants_known(root)
    assert found == [], found
    # The entry's own process is gone, so nothing of this tree is unaccounted for.
    assert known is True

    # A child that predates the snapshot is still collected, which is every real one.
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: {root: "0:500", child: "0:600"}.get(pid))
    found, known = pl._windows_collect_descendants_known(root)
    assert found == [(child, "0:600")], found
    assert known is True


def test_a_child_of_a_late_child_is_reached_too(monkeypatch):
    """The late walk has the same problem the first snapshot had.

    Every taskkill in the loop below it spends up to fifteen seconds, so a late descendant
    has its own window in which to start a child, and that grandchild was in neither walk:
    the sweep reported nothing about it and the caller deleted the record and the pidfile
    while it held a GPU.
    """
    survivor, late, later = 400, 401, 402
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_signalable", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    identities = {survivor: "0:400", late: "0:401", later: "0:402"}
    monkeypatch.setattr(pl, "_pid_identity", identities.get)
    dead: "set[int]" = set()
    killed: "list[int]" = []
    walks = {"n": 0}

    def _collect(pid, identity = None):
        if pid != survivor:
            return [], True
        walks["n"] += 1
        # The grandchild only exists from the second walk on: `late` started it while the
        # first round was spending its taskkill budget.
        if walks["n"] == 1:
            return [(late, "0:401")], True
        return [(late, "0:401"), (later, "0:402")], True

    monkeypatch.setattr(pl, "_windows_collect_descendants_known", _collect)
    monkeypatch.setattr(
        pl,
        "_windows_terminate_pid",
        lambda pid, identity = None: (killed.append(pid), dead.add(pid), True)[-1],
    )
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: pid not in dead)

    survivors = pl._windows_terminate_collected([(survivor, "0:400")])
    assert later in killed, killed
    assert killed[-1] == survivor, "the survivor must go after what is under it"
    assert survivors == [], survivors


def test_a_subtree_that_keeps_growing_is_reported_rather_than_looped_on(monkeypatch):
    """A process that respawns faster than it can be killed is not something a loop wins.

    The rounds are bounded, and when they run out the honest answer is that whatever is
    under this survivor has not been accounted for -- which keeps the record for the next
    sweep instead of reporting the tree gone.
    """
    survivor = 500
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_signalable", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    spawned = {"n": 0}

    def _collect(pid, identity = None):
        if pid != survivor:
            return [], True
        spawned["n"] += 1
        fresh = 600 + spawned["n"]
        return [(fresh, f"0:{fresh}")], True

    monkeypatch.setattr(pl, "_windows_collect_descendants_known", _collect)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: f"0:{pid}")
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid, identity = None: True)

    survivors = pl._windows_terminate_collected([(survivor, "0:500")])
    assert spawned["n"] == pl._LATE_WALK_ROUNDS, spawned
    assert (survivor, "0:500") in survivors, survivors


def test_a_grandchild_is_captured_before_its_parent_is_removed(monkeypatch):
    """Killing the intermediate first severs the only link to what it started.

    Windows' parent-pid field still names it, but a walk from the root has to pass THROUGH
    it, and it no longer exists -- so the grandchild is in no later snapshot either, the
    sweep reports no survivors, and the caller discards the record and the pidfile while it
    holds a GPU. The children of a process are therefore read immediately before that
    process is signalled, never after.
    """
    survivor, middle, grandchild = 1100, 1101, 1102
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_signalable", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: f"{pid}")
    dead: "set[int]" = set()
    killed: "list[int]" = []

    def _collect(pid, identity = None):
        # The root's walk names the intermediate only: the grandchild does not exist yet.
        if pid == survivor:
            return [(middle, f"{middle}")], True
        # The intermediate's own walk, taken right before it is killed, is the ONLY place
        # the grandchild can be seen -- and once the intermediate is dead the root's walk
        # cannot reach it at all.
        if pid == middle and middle not in dead:
            return [(grandchild, f"{grandchild}")], True
        return [], True

    monkeypatch.setattr(pl, "_windows_collect_descendants_known", _collect)
    monkeypatch.setattr(
        pl,
        "_windows_terminate_pid",
        lambda pid, identity = None: (killed.append(pid), dead.add(pid), True)[-1],
    )
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: pid not in dead)

    survivors = pl._windows_terminate_collected([(survivor, f"{survivor}")])
    assert grandchild in killed, killed
    assert killed.index(grandchild) < killed.index(middle), killed
    assert killed[-1] == survivor, killed
    assert survivors == [], survivors


def test_a_walk_that_fails_below_an_intermediate_keeps_the_record(monkeypatch):
    """The intermediate is still killed -- refusing would leave the process the sweep exists
    for running -- but a walk that could not say what is under it has not shown there is
    nothing, so the pid is reported rather than dropped."""
    survivor, middle = 1200, 1201
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_signalable", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: f"{pid}")
    dead: "set[int]" = set()
    killed: "list[int]" = []

    def _collect(pid, identity = None):
        if pid == survivor:
            return [(middle, f"{middle}")], True
        if pid == middle:
            return [], False  # handle pressure, access denied: cannot tell
        return [], True

    monkeypatch.setattr(pl, "_windows_collect_descendants_known", _collect)
    monkeypatch.setattr(
        pl,
        "_windows_terminate_pid",
        lambda pid, identity = None: (killed.append(pid), dead.add(pid), True)[-1],
    )
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: pid not in dead)

    survivors = pl._windows_terminate_collected([(survivor, f"{survivor}")])
    assert middle in killed, killed
    assert (middle, f"{middle}") in survivors, survivors


def test_a_child_born_while_the_snapshot_was_taken_is_reported_not_signalled(monkeypatch):
    """A pid created inside the snapshot window is unprovable in both directions.

    It is either a genuine child started while the table was being read, or the replacement
    for a number the table listed for a process that exited in the same window; the ceiling
    cannot separate them, because the replacement predates it too. Signalling it can hand a
    stranger to a forced tree kill. Dropping it silently reports a tree as fully enumerated
    when it may not be, which is the leak this function exists to prevent. So neither: the
    pid is skipped and the walk says it is incomplete, and the late rounds re-snapshot.
    """
    root, child = 1300, 1301
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    clock = {"now": 1000}

    def _table():
        # The child is created while the snapshot is being taken, so it is in the table and
        # its creation time is after the moment the walk started.
        clock["now"] = 2000
        return {root: [child]}

    monkeypatch.setattr(pl, "_child_pid_map", _table)
    monkeypatch.setattr(pl, "_windows_filetime_now", lambda: clock["now"])
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: {root: "0:500", child: "0:1500"}.get(pid))
    monkeypatch.setattr(
        pl,
        "_windows_creation_time",
        lambda identity: (None if identity is None else int(identity.split(":")[1])),
    )

    found, known = pl._windows_collect_descendants_known(root)
    assert found == [], found
    assert known is False

    # And a child that predates the floor is provably one the table listed, so it is
    # collected and the walk is complete: the bound narrows to the window, not to everything.
    clock["now"] = 1000
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: {root: "0:500", child: "0:900"}.get(pid))
    found, known = pl._windows_collect_descendants_known(root)
    assert found == [(child, "0:900")], found
    assert known is True


def test_a_survivor_whose_identity_cannot_be_read_is_still_reported(monkeypatch):
    """The final pass is accounting, not a decision to signal.

    A pid that is plainly still running and whose identity cannot be read right now is proof
    of nothing, and requiring proof there dropped it from the report, so the caller deleted
    the record and the pidfile that were the only handles on a process the kill had failed to
    stop.
    """
    survivor = 1400
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_signalable", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(
        pl, "_windows_collect_descendants_known", lambda pid, identity = None: ([], True)
    )
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid, identity = None: True)
    reads = {"n": 0}

    def _identity(pid):
        reads["n"] += 1
        # Readable for the check before the kill, unreadable at the final accounting.
        return "1400" if reads["n"] == 1 else None

    monkeypatch.setattr(pl, "_pid_identity", _identity)
    survivors = pl._windows_terminate_collected([(survivor, "1400")])
    assert survivors == [(survivor, "1400")], survivors


def test_a_tree_kill_does_not_follow_the_number_to_a_stranger(monkeypatch):
    """The leader can exit, and its number be taken, after the caller validated it.

    Re-reading the identity inside the helper blesses whatever holds the number now: the
    descendant collection walks the STRANGER's tree with the stranger as its ancestry
    floor, and the root terminate confirms a handle whose identity it derived from the
    same reading. The caller's identity has to travel in, so the mismatch is visible.
    """
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_signalable", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    # The number now belongs to a process created later.
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: "900:500")
    killed: "list[int]" = []
    walked: "list[int]" = []
    alive = {500, 600}

    def _collect(pid, identity = None):
        walked.append(pid)
        return ([(600, "0:600")], True) if pid == 500 else ([], True)

    def _kill(pid, identity = None):
        killed.append(pid)
        alive.discard(pid)
        return True

    monkeypatch.setattr(pl, "_windows_collect_descendants_known", _collect)
    monkeypatch.setattr(pl, "_windows_terminate_pid", _kill)
    # False, because nothing was signalled and the record is the only handle left on
    # whatever our leader started before it exited.
    assert pl._windows_terminate_validated_tree(500, "0:500") is False
    assert killed == [], "a stranger holding a recycled number must not be signalled"
    assert walked == [], "nor may its tree be enumerated as if it were ours"

    # The same call for the process the caller actually validated still kills the tree.
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: f"0:{pid}")
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: pid in alive)
    assert pl._windows_terminate_validated_tree(500, "0:500") is True
    assert killed == [500, 600]


def test_the_root_is_killed_through_the_identity_the_caller_validated(monkeypatch):
    """Not one derived from the pid being acted on, which is the thing in question."""
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_signalable", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: "0:500")
    monkeypatch.setattr(
        pl, "_windows_collect_descendants_known", lambda pid, identity = None: ([], True)
    )
    seen: "list[tuple[int, object]]" = []
    alive = {500}

    def _kill(pid, identity = None):
        seen.append((pid, identity))
        alive.discard(pid)
        return True

    monkeypatch.setattr(pl, "_pid_alive", lambda pid: pid in alive)
    monkeypatch.setattr(pl, "_windows_terminate_pid", _kill)
    assert pl._windows_terminate_validated_tree(500, "0:500") is True
    assert seen == [(500, "0:500")]


def test_a_subtree_deeper_than_the_capture_depth_is_unresolved(monkeypatch):
    """Exhausting the depth is an unperformed walk, not an empty one.

    A process that starts another child after each preceding snapshot produces a chain
    longer than the recursion goes. Answering False there told the caller the anchor had
    nothing below it, so the anchor was killed -- severing the only traversable link to
    the child no walk had seen -- and the sweep went on to report success, taking the
    lifetime record and the pidfile with it.
    """
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_signalable", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: f"0:{pid}")
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid, identity = None: True)
    # Every process has exactly one child, one number higher, forever.
    monkeypatch.setattr(
        pl,
        "_windows_collect_descendants_known",
        lambda pid, identity = None: ([(pid + 1, f"0:{pid + 1}")], True),
    )
    attempted: "list[tuple[int, object]]" = []
    unresolved: "list[tuple[int, object]]" = []
    result = pl._windows_kill_below(500, attempted, unresolved, set(), 3)
    assert result is True, "the levels it did reach were killed"
    # The deepest anchor it could not walk below is carried out, not reported gone.
    assert (503, "0:503") in unresolved, unresolved

    # The depth limit itself answers None rather than "nothing below".
    assert pl._windows_kill_below(500, [], [], set(), 0) is None


def test_the_snapshot_is_rooted_at_the_identity_the_caller_validated(monkeypatch):
    """The other half of the recycled-root guard.

    The root kill refuses a number that has moved on, but the walk below it used to read its
    own floor off the pid: rooted at the replacement, every one of the STRANGER's children is
    later than it, passes the ancestry test with a perfectly valid identity, and is handed to
    the sweep as one of ours. Nothing about the root kill's refusal reaches them.
    """
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_signalable", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    # The number now belongs to a process created later than ours.
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: "900:500")
    monkeypatch.setattr(
        pl, "_windows_creation_time", lambda ident: int((ident or "0:0").split(":")[0])
    )
    monkeypatch.setattr(pl, "_child_pid_map", lambda: {500: [600]})
    monkeypatch.setattr(pl, "_windows_filetime_now", lambda: 10_000)

    assert pl._windows_collect_descendants_known(500, "0:500") == ([], False)
    # And with no identity to check against, the old reading stands: this is the argument-less
    # call every other caller still makes.
    found, known = pl._windows_collect_descendants_known(500)
    assert known is True
    assert [pid for pid, _ in found] == [600]


def test_a_survivor_with_no_readable_identity_is_not_adopted(monkeypatch):
    """`(pid, None)` out of a walk is "could not be read", not "newly spawned".

    The POSIX collector returns it for a survivor it could not classify, and capturing an
    identity at the adopt records whatever holds the number at THAT moment -- the recycled
    stranger the check exists to keep out, put into a job that kills its members when the app
    closes.
    """
    recorded = {}
    monkeypatch.setattr(pl, "_signalable", lambda pid: True)
    monkeypatch.setattr(pl, "_is_windows", lambda: False)
    monkeypatch.setattr(pl, "_identity_for_record", lambda pid: "999:whoever")
    monkeypatch.setattr(pl, "_own_process_group", lambda pid: None)
    monkeypatch.setattr(pl, "_write_breadcrumb", lambda: None)
    monkeypatch.setattr(pl, "_tracked_pids", recorded)

    pl.adopt_pid(777, None, from_snapshot = True)
    assert recorded == {}, recorded

    # A child this process has just spawned is the other case, and it still captures one.
    pl.adopt_pid(777, None)
    assert recorded == {777: "999:whoever"}


# ── the survivor read-back needs a grace period ──
#
# Every kill in this module is asynchronous. `SIGKILL` returns once the signal is queued
# and the kernel tears the process down afterwards; `TerminateProcess` is documented as
# initiating termination and returning immediately. Re-reading liveness on the line after
# either one therefore measures the read's own latency, and a process that died on the
# spot is reported as a survivor -- which keeps the lifetime record and the pidfile, so the
# next launch sweeps records naming processes that died during the previous unload.


_SIGTERM_PROOF_TREE = textwrap.dedent(
    """
    import pathlib, subprocess, sys, time
    kid = (
        "import signal, time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "time.sleep(300)"
    )
    kids = [subprocess.Popen([sys.executable, "-c", kid]) for _ in range(int(sys.argv[2]))]
    pathlib.Path(sys.argv[1]).write_text(",".join(str(k.pid) for k in kids))
    time.sleep(300)
    """
)


@pytest.fixture
def stubborn_tree(tmp_path):
    """(leader, [child pids]) where every child ignores SIGTERM and dies on SIGKILL.

    The shape the sweep is written for: the escalation is what kills these, so the
    read-back runs immediately after a SIGKILL rather than after a polite exit.
    """
    marker = tmp_path / "kids.pid"
    leader = subprocess.Popen([sys.executable, "-c", _SIGTERM_PROOF_TREE, str(marker), "4"])
    kids: "list[int]" = []
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            kids = [int(x) for x in marker.read_text().split(",") if x]
        except (OSError, ValueError):
            kids = []
        if len(kids) == 4:
            break
        time.sleep(0.05)
    if len(kids) != 4:
        leader.kill()
        pytest.fail("the spawned children never recorded their pids")
    try:
        yield leader, kids
    finally:
        for pid in kids:
            _hard_kill(pid)
        try:
            leader.kill()
            leader.wait(timeout = 10)
        except Exception:
            pass


@pytest.mark.skipif(IS_WINDOWS, reason = "POSIX signals; the Windows arm is faked below")
def test_a_descendant_that_died_on_the_kill_is_not_reported_as_a_survivor(stubborn_tree):
    """The read-back used to answer "still there" for four processes it had just killed.

    Measured on this host with eight of them: eight reported, none alive half a second
    later. The cost is not the wasted adopt; it is that a reported survivor KEEPS the
    record and the pidfile, so the next launch runs a reap sweep over ghosts.
    """
    leader, kids = stubborn_tree
    collected, known = pl.collect_descendants_known(leader.pid)
    assert known, "the walk itself failed, so this test proves nothing"
    assert sorted(pid for pid, _ in collected) == sorted(kids)

    survivors = pl.terminate_descendants(collected, timeout = 1.0)

    assert survivors == [], f"reported {survivors} as still running"
    for pid in kids:
        assert not _alive(pid), f"{pid} was reported gone and is not"


@pytest.mark.skipif(IS_WINDOWS, reason = "POSIX signals; the Windows arm is faked below")
def test_a_descendant_that_refuses_to_die_is_still_reported(stubborn_tree):
    """The other half. A grace period that swallowed a real survivor would be worse than
    the immediate read it replaces: the record and the pidfile are the only handles left on
    a worker still holding a GPU, and this return value is what keeps them.

    Only the one pid's signals are blocked; the other three are killed for real, so this
    also pins that the sweep separates the two in one pass.
    """
    leader, kids = stubborn_tree
    victim = kids[0]

    collected, known = pl.collect_descendants_known(leader.pid)
    assert known
    with _signals_that_never_land([victim]):
        started = time.monotonic()
        survivors = pl.terminate_descendants(collected, timeout = 1.0)
        elapsed = time.monotonic() - started

        assert [pid for pid, _ in survivors] == [victim], survivors
        assert _alive(victim), "the fake did not hold; the victim really died"
        # Bounded, not waited out: the SIGTERM timeout plus one grace period, not a loop
        # that runs until the process gives up. This sits on the teardown path.
        assert elapsed < 1.0 + pl._KILL_SETTLE_SECONDS + 3.0, elapsed


@pytest.mark.skipif(IS_WINDOWS, reason = "kills a real process and reads it back")
def test_confirming_an_exit_waits_for_the_kill_to_land():
    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    try:
        os.kill(victim.pid, signal.SIGKILL)
        # On the next line, which is the whole point: the process is still signalable for
        # as long as the kernel takes to tear it down.
        assert pl.confirm_pid_exited(victim.pid) is True
    finally:
        _hard_kill(victim.pid)
        try:
            victim.wait(timeout = 10)
        except Exception:
            pass


def test_confirming_an_exit_gives_up_on_a_process_that_stays(monkeypatch):
    monkeypatch.setattr(pl, "pid_is_running", lambda pid: True)
    started = time.monotonic()
    assert pl.confirm_pid_exited(4242) is False
    elapsed = time.monotonic() - started
    assert pl._KILL_SETTLE_SECONDS <= elapsed < pl._KILL_SETTLE_SECONDS + 3.0, elapsed


def test_confirming_an_exit_polls_rather_than_reading_once(monkeypatch):
    """The deterministic half of the test above it. A single read, at any point inside the
    window the kill takes to land, answers "still running"."""
    reads = {"n": 0}

    def _running(pid):
        reads["n"] += 1
        return reads["n"] <= 3

    monkeypatch.setattr(pl, "pid_is_running", _running)
    assert pl.confirm_pid_exited(4242) is True
    assert reads["n"] > 1, "read once, which is the defect"


@pytest.mark.parametrize("pid", [None, 0, 1])
def test_confirming_an_exit_never_probes_a_reserved_pid(monkeypatch, pid):
    """And answers False, not True. This return value authorises deleting the record and
    the pidfile, so "this module will not touch that pid" must not read as "it is gone"."""
    probed = []
    monkeypatch.setattr(pl, "pid_is_running", lambda p: probed.append(p) or True)
    assert pl.confirm_pid_exited(pid) is False
    assert probed == []


def test_a_probe_that_lies_once_does_not_lose_a_live_worker(monkeypatch):
    """`_pid_alive` fails OPEN on Windows: out of handles, out of memory, or any ctypes
    failure all read as "gone". Polling a fail-open probe fifty times and dropping a pid on
    the first "gone" would multiply the chance of deleting the record and the pidfile for a
    process the kill did not stop, which is the one direction this module refuses to fail
    in. Agreements have to be consecutive for that reason.
    """
    reads = {"n": 0}

    def _alive_except_once(pid, identity):
        reads["n"] += 1
        return reads["n"] != 2  # the second read lies

    survivors = pl._survivors_after_settling([(700, "0:700")], _alive_except_once)
    assert survivors == [(700, "0:700")], "a live worker was dropped on one bad read"


def test_the_settled_report_keeps_the_callers_order(monkeypatch):
    """Deepest first, which is what the Windows sweep hands out and what the caller adopts
    in turn. The middle one goes; the two around it must not swap."""
    gone = {601}
    survivors = pl._survivors_after_settling(
        [(602, "0:602"), (601, "0:601"), (600, "0:600")],
        lambda pid, _identity: pid not in gone,
    )
    assert survivors == [(602, "0:602"), (600, "0:600")]


def _terminate_then_linger(monkeypatch, reads_after_the_kill = 3):
    """Fake Windows where `TerminateProcess` returns before the process is gone.

    Returns the list the kills are recorded in. Everything stays visible until the kill,
    then for `reads_after_the_kill` more liveness reads, then goes.
    """
    killed: "list[int]" = []
    reads: "dict[int, int]" = {}

    def _pid_alive(pid):
        if pid not in killed:
            return True
        reads[pid] = reads.get(pid, 0) + 1
        return reads[pid] <= reads_after_the_kill

    def _terminate(pid, identity = None):
        killed.append(pid)
        return True

    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_signalable", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: f"0:{pid}")
    monkeypatch.setattr(pl, "_pid_alive", _pid_alive)
    monkeypatch.setattr(pl, "_windows_terminate_pid", _terminate)
    return killed


def test_the_validated_tree_kill_gives_the_terminate_time_to_land(monkeypatch):
    """The ordinary Windows unload: a leader with nothing under it.

    `_windows_terminate_pid` is `TerminateProcess` through a handle, which returns in
    microseconds, so the read-back on the next line saw the leader still open and answered
    "the tree stands". False here keeps the record and the pidfile for a process that was
    already dying.
    """
    killed = _terminate_then_linger(monkeypatch)
    monkeypatch.setattr(
        pl, "_windows_collect_descendants_known", lambda pid, identity = None: ([], True)
    )
    assert pl._windows_terminate_validated_tree(500, "0:500") is True
    assert killed == [500]


def test_the_validated_tree_kill_still_reports_a_root_it_could_not_stop(monkeypatch):
    """A leader that is genuinely stuck is still a survivor, and the wait is bounded."""
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_signalable", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: f"0:{pid}")
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid, identity = None: True)
    monkeypatch.setattr(
        pl, "_windows_collect_descendants_known", lambda pid, identity = None: ([], True)
    )
    started = time.monotonic()
    assert pl._windows_terminate_validated_tree(500, "0:500") is False
    assert time.monotonic() - started < pl._KILL_SETTLE_SECONDS + 3.0


def test_the_windows_sweep_separates_a_dying_descendant_from_a_stuck_one(monkeypatch):
    """Same read-back, the descendant side of it. One of these died on the kill and the
    other did not, and an immediate read cannot tell them apart."""
    killed = _terminate_then_linger(monkeypatch)
    real_alive = pl._pid_alive
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True if pid == 601 else real_alive(pid))
    monkeypatch.setattr(
        pl, "_windows_collect_descendants_known", lambda pid, identity = None: ([], True)
    )
    survivors = pl._windows_terminate_collected([(600, "0:600"), (601, "0:601")])
    assert sorted(killed) == [600, 601]
    assert survivors == [(601, "0:601")], survivors


def test_a_windows_shutdown_does_not_report_a_leader_that_is_merely_dying(monkeypatch):
    """The windows-latest staging failure this read-back caused, reproduced.

    `test_a_confirmed_shutdown_still_clears_the_record` failed there with
    `assert [8792] == []`: `terminate_all` terminated a child that exited on the spot and
    then read the pid back one line later, so `_windows_terminate_validated_tree` answered
    False, the pid came out as a survivor and the record it names was written straight back.
    `main` cannot fail that way because `_windows_terminate_tree` there reports the tree
    gone on taskkill's exit code alone and never re-reads anything.
    """
    _terminate_then_linger(monkeypatch)
    monkeypatch.setattr(
        pl, "_windows_collect_descendants_known", lambda pid, identity = None: ([], True)
    )
    monkeypatch.setattr(pl, "_group_has_members", lambda pgid: False)
    monkeypatch.setattr(pl, "_write_breadcrumb", lambda: None)
    monkeypatch.setattr(pl, "_tracked_pids", {500: "0:500"})
    monkeypatch.setattr(pl, "_tracked_pgids", {})

    assert pl.terminate_all(timeout = 3.0) == []
    assert pl._tracked_pids == {}, "the record was written back for a process that is gone"


@pytest.mark.skipif(IS_WINDOWS, reason = "POSIX signals; the Windows arm is above")
def test_a_posix_shutdown_does_not_report_a_child_that_is_merely_dying(monkeypatch, tmp_path):
    """The same defect on the POSIX shutdown path, with a real process.

    `_posix_terminate_one` ends on killpg(SIGKILL) and returns, and `terminate_all` reads
    liveness on the line after it: measured here, it reported a survivor and wrote the
    record straight back for a child that was gone half a second later. Settling inside
    `_posix_terminate_one` covers `terminate_all`, `terminate_pid` and `_reap_one_record`
    at once, since all three read back immediately after it.

    A child that ignores SIGTERM is the point: one that exits politely returns from the
    poll loop above the SIGKILL and never reaches the line this is about.
    """
    monkeypatch.setenv("UNSLOTH_STUDIO_CHILD_RECORD", str(tmp_path / "children"))
    monkeypatch.setattr(pl, "_tracked_pids", {})
    monkeypatch.setattr(pl, "_tracked_pgids", {})
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)",
        ],
        start_new_session = True,
    )
    try:
        pl.adopt_pid(child.pid)
        assert pl._tracked_pids.get(child.pid) is not None

        survivors = pl.terminate_all(timeout = 1.0)

        assert survivors == [], f"reported {survivors} as still running"
        assert pl._tracked_pids == {}, "the record was written back for a process that is gone"
        assert child.wait(timeout = 10) is not None
    finally:
        _hard_kill(child.pid)
        try:
            child.wait(timeout = 10)
        except Exception:
            pass


def test_a_windows_startup_sweep_consumes_the_record_it_has_emptied(monkeypatch, tmp_path):
    """The second windows-latest staging failure, reproduced.

    `test_macos_style_orphans_are_recorded_and_reaped` failed there on "the record should
    be consumed". `_reap_one_record` kills a recorded child and then asks whether it is
    still there, and the answer it got was the terminate's latency rather than the
    outcome: the record was marked unresolved and left on disk, so every later launch
    re-reads it and re-sweeps a process that died during the first one.
    """
    import json

    _terminate_then_linger(monkeypatch)
    monkeypatch.setattr(
        pl, "_windows_collect_descendants_known", lambda pid, identity = None: ([], True)
    )
    monkeypatch.setattr(pl, "_group_has_members", lambda pgid: False)
    record = tmp_path / "4321.json"
    record.write_text(
        json.dumps(
            {
                "owner_pid": 4321,
                "owner_identity": "a-previous-studio-that-is-gone",
                "children": [{"pid": 500, "identity": "0:500"}],
            }
        )
    )

    killed, deferred = pl._reap_one_record(record, timeout = 3.0)

    assert killed == [500]
    assert deferred is False
    assert not record.exists(), "the record should be consumed"


# ── the teardown lock is not held across a kill killpg already did ──


@pytest.mark.skipif(IS_WINDOWS, reason = "no killpg here, so the branch below is POSIX-only")
def test_a_killed_process_group_makes_the_tree_kill_unnecessary(monkeypatch):
    """`_kill_process_group` sends killpg SIGKILL to this exact tree eleven lines earlier.

    `terminate_pid`'s POSIX arm is killpg SIGTERM, a poll loop and killpg SIGKILL over the
    same group, so on POSIX with a real group it buys no reach and costs a /proc walk plus
    up to five more seconds with `_teardown_lock` held.
    """
    from core.inference.llama_cpp import LlamaCppBackend

    b = _make_backend()
    b._process.pid = 4242
    b._process.poll.return_value = None
    b._leading_process_group = lambda pid: pid  # POSIX, and the server leads its own group
    cleared, killed = _instrument(monkeypatch, gone = False)
    confirmed = []
    monkeypatch.setattr(
        LlamaCppBackend,
        "_confirm_group_kill_landed",
        staticmethod(lambda pid: bool(confirmed.append(pid)) or True),
    )

    b._kill_process()

    assert killed == [], "killpg already covered this tree"
    assert confirmed == [4242], "the exit still has to be confirmed, just not re-killed"
    assert cleared == [1]


def test_without_a_process_group_the_tree_kill_still_runs(monkeypatch):
    """Windows, where `_leading_process_group` always answers None. taskkill is the only
    reach left there and this PR exists to use it, so nothing about that path moves."""
    from core.inference.llama_cpp import LlamaCppBackend

    b = _make_backend()  # _leading_process_group already answers None
    b._process.pid = 4242
    b._process.poll.return_value = None
    cleared, killed = _instrument(monkeypatch, gone = False)
    confirmed = []
    monkeypatch.setattr(
        LlamaCppBackend,
        "_confirm_group_kill_landed",
        staticmethod(lambda pid: bool(confirmed.append(pid)) or True),
    )

    b._kill_process()

    assert killed == [4242]
    assert confirmed == []
    assert cleared == [], "dropped the pidfile while the server was still running"


@pytest.mark.skipif(IS_WINDOWS, reason = "deletes os.killpg, which is not here to delete")
def test_a_process_group_with_no_killpg_still_tree_kills(monkeypatch):
    """The second half of the condition. `_kill_process_group` is a no-op when `os.killpg`
    is missing, so a pgid on its own is not evidence that anything was signalled, and
    skipping the tree kill on it would skip a kill that never happened."""
    import core.inference.llama_cpp as lc
    from core.inference.llama_cpp import LlamaCppBackend

    b = _make_backend()
    b._process.pid = 4242
    b._process.poll.return_value = None
    b._leading_process_group = lambda pid: pid
    monkeypatch.delattr(lc.os, "killpg", raising = False)
    cleared, killed = _instrument(monkeypatch, gone = True)
    confirmed = []
    monkeypatch.setattr(
        LlamaCppBackend,
        "_confirm_group_kill_landed",
        staticmethod(lambda pid: bool(confirmed.append(pid)) or True),
    )

    b._kill_process()

    assert killed == [4242]
    assert confirmed == []


def test_confirming_a_group_kill_signals_nothing(monkeypatch):
    """It answers a question. The kill already happened, and sending another one from here
    is the cost this replaced.

    Against the signal calls themselves, not just against `terminate_pid`: asserting that
    one helper is not called proves nothing about a helper that never referenced it.
    """
    from core.inference.llama_cpp import LlamaCppBackend

    signalled: "list[tuple[str, int, int]]" = []
    monkeypatch.setattr(pl.os, "kill", lambda pid, sig, *a: signalled.append(("kill", pid, sig)))
    if hasattr(pl.os, "killpg"):
        monkeypatch.setattr(
            pl.os, "killpg", lambda pgid, sig: signalled.append(("killpg", pgid, sig))
        )
    monkeypatch.setattr(
        pl, "terminate_pid", lambda pid, **kw: signalled.append(("terminate_pid", pid, 0))
    )
    monkeypatch.setattr(
        pl,
        "_windows_terminate_pid",
        lambda pid, identity = None: signalled.append(("taskkill", pid, 0)),
    )

    monkeypatch.setattr(pl, "pid_is_running", lambda pid: False)
    assert LlamaCppBackend._confirm_group_kill_landed(4242) is True
    assert signalled == []

    monkeypatch.setattr(pl, "pid_is_running", lambda pid: True)
    assert LlamaCppBackend._confirm_group_kill_landed(4242) is False
    assert signalled == []


@pytest.mark.parametrize("pid", [None, 0, 1])
def test_confirming_a_group_kill_never_probes_a_reserved_pid(monkeypatch, pid):
    from core.inference.llama_cpp import LlamaCppBackend

    probed = []
    monkeypatch.setattr(pl, "pid_is_running", lambda p: probed.append(p) or True)
    assert LlamaCppBackend._confirm_group_kill_landed(pid) is False
    assert probed == []
