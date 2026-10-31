# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unload must reap the whole server tree on Windows, not just the leader."""

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

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
sys.modules.setdefault("structlog", _types.ModuleType("structlog"))

import utils.process_lifetime as pl  # noqa: E402

IS_WINDOWS = sys.platform == "win32"
IS_LINUX = sys.platform.startswith("linux")


@pytest.fixture(autouse = True)
def _warm_the_owner_identity():
    """Warm _own_identity before faking _pid_identity, or a cached fake makes this process look dead."""
    pl._own_identity()


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
    """Scoped block, not monkeypatch: fixture teardown runs before undo, so the fake would reach cleanup."""
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


def _windows_shaped_identity(pid: int):
    """Real Linux start time split into high:low FILETIME form; a constant would not exercise the floor."""
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
    """A root-only creation-time floor admits strangers; compare each child to its parent's current time."""
    root, reused_parent, stranger, stranger_child, real_grandchild = 500, 600, 700, 701, 601
    created = {
        root: 100,
        reused_parent: 300,
        stranger: 200,
        stranger_child: 250,
        real_grandchild: 400,
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
    """taskkill /T re-walks parent-pid links, so it can re-expand a stranger the collector rejected."""
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

    # Calling /T at all is the regression: Windows expands the tree out of sight.
    def _no_tree_kill(pid):
        raise AssertionError(f"taskkill /T was used on {pid}; it re-expands rejected pids")

    monkeypatch.setattr(pl, "_windows_terminate_tree", _no_tree_kill)

    collected = pl.collect_descendants(root)
    pl._windows_terminate_collected(collected)

    assert stranger not in killed, "the rejected stranger was killed by tree expansion"
    assert stranger_child not in killed, "the stranger's own work was killed with it"
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
    # Make _same_identity compare both identity halves, as on Windows.
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
    pl._windows_terminate_collected([(10, "0:10"), (11, "0:11"), (12, "0:12")])
    assert order == [12, 11, 10]


def test_the_windows_terminate_leaves_an_unreadable_identity_alone(monkeypatch):
    """Skip a pid with unreadable identity on both sides: a recycled number would get taskkill /T /F."""
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    asked = []
    monkeypatch.setattr(pl, "_windows_terminate_tree", lambda pid: asked.append(pid))

    monkeypatch.setattr(pl, "_pid_identity", lambda pid: None)
    pl._windows_terminate_collected([(4242, "0:1")])
    assert asked == [], "an unreadable current identity must not be tree killed"

    monkeypatch.setattr(pl, "_pid_identity", lambda pid: "0:1")
    pl._windows_terminate_collected([(4242, None)])
    assert asked == [], "a survivor collected without an identity must not be tree killed"


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
    b._process.poll.return_value = None
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
    b._process = mock.Mock(spec = [])
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
    monkeypatch.setattr(pl, "_tracked_pids", {4242: "WHEN-WE-SPAWNED-IT"})
    monkeypatch.setattr(pl, "_tracked_pgids", {})
    monkeypatch.setattr(pl, "_write_breadcrumb", lambda: None)
    pl.terminate_pid(4242, timeout = 0.01, owner_verified = True)
    assert signalled == []
    assert 4242 not in pl._tracked_pids, "a provably recycled pid must not stay recorded"


def test_a_stand_in_process_with_a_pid_is_never_signalled(monkeypatch):
    """A stand-in object's pid is never signalled: the waiver needs the spawning Popen we hold."""
    b = _make_backend()
    b._process = type("P", (), {"pid": 4242})()
    cleared, killed = _instrument(monkeypatch, gone = False)
    b._kill_process()
    assert killed == [], "signalled a pid this backend never spawned"
    assert cleared == [1], "kept the pidfile for a stand-in that can never confirm an exit"


def test_a_stand_in_with_a_pid_leaves_a_real_bystander_running():
    """A stand-in's pid may now belong to a bystander, which must survive the unload."""
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


def test_the_surviving_root_fallback_does_not_re_expand_the_tree(monkeypatch):
    """The fallback must not use taskkill /T, which re-walks parent-pid links the collector distrusts."""
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
    # Root is stopped first: while alive it can spawn children nothing re-examines.
    assert killed == [500, 600], "the root is stopped first, then the validated set only"
    assert 500 not in pl._tracked_pids, "a tree that went down releases its record"


def test_the_surviving_root_fallback_keeps_the_record_when_something_lives(monkeypatch):
    """False keeps the record: read the pids back, as one taskkill /F's exit says nothing about the rest."""
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
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
    """taskkill /F can report success on a live process, so survivors must be read back."""
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: f"0:{pid}")
    monkeypatch.setattr(
        pl, "_windows_collect_descendants_known", lambda pid, identity = None: ([], True)
    )
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
    b._process.poll.return_value = 0
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


def test_an_unreadable_process_table_is_indeterminate_not_empty(monkeypatch):
    """A failed snapshot is indeterminate, not an empty tree; the walk must report it as unknown."""
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: "0:500")
    monkeypatch.setattr(pl, "_child_pid_map", lambda: None)
    assert pl.collect_descendants_known(500) == ([], False)
    assert pl.collect_descendants(500) == []

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
    """Unverifiable live descendants are never signalled, but must be reported as survivors, not dropped."""
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: None)
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
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: "0:999")
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
    assert pl._provably_the_same(11, "0:11") is False
    assert pl._provably_different(11, "0:11") is False
    assert pl._provably_the_same(11, None) is False
    assert pl._provably_different(11, None) is False


def test_a_walk_that_failed_partway_is_not_a_table(monkeypatch):
    """Only ERROR_NO_MORE_FILES marks the end of a Process32NextW walk; other FALSE is a failure."""
    source = Path(pl.__file__).read_text(encoding = "utf-8")
    start = source.index("def _windows_child_pid_map")
    body = source[start : source.index("\ndef ", start + 10)]
    assert "ERROR_NO_MORE_FILES = 18" in body
    assert "ctypes.get_last_error() != ERROR_NO_MORE_FILES" in body
    error_branch = body[body.index("ctypes.get_last_error()") :]
    assert error_branch.split("\n")[1].strip() == "return None", error_branch.split("\n")[1]


def test_a_failed_late_walk_kills_the_survivor_and_reports_it(monkeypatch):
    """A failed late walk still kills the survivor, but reports the tree unresolved so the record stays."""
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: f"0:{pid}")
    monkeypatch.setattr(
        pl, "_windows_collect_descendants_known", lambda pid, identity = None: ([], False)
    )
    killed = []
    dead: "set[int]" = set()

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
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: f"0:{pid}")
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid, identity = None: True)

    monkeypatch.setattr(
        pl, "_windows_collect_descendants_known", lambda pid, identity = None: ([], False)
    )
    assert pl._windows_terminate_validated_tree(500) is False

    monkeypatch.setattr(
        pl, "_windows_collect_descendants_known", lambda pid, identity = None: ([], True)
    )
    assert pl._windows_terminate_validated_tree(500) is True


def test_a_live_unverifiable_descendant_is_an_incomplete_tree_kill(monkeypatch):
    """A live descendant with unreadable creation time is an incomplete kill, not proof of pid reuse."""
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid, identity = None: True)
    monkeypatch.setattr(
        pl,
        "_windows_collect_descendants_known",
        lambda pid, identity = None: ([(600, "0:600")], True) if pid == 500 else ([], True),
    )
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: pid == 600)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: None)
    assert pl._windows_terminate_validated_tree(500) is False

    monkeypatch.setattr(pl, "_pid_identity", lambda pid: "0:999")
    assert pl._windows_terminate_validated_tree(500) is True


def test_a_recycled_survivor_is_not_adopted(monkeypatch):
    """Adopt must check the verified identity; a recycled pid could land in the kill-on-close job."""
    recorded = {}
    monkeypatch.setattr(pl, "_signalable", lambda pid: True)
    monkeypatch.setattr(pl, "_adopt_fork_reset", lambda: None)
    monkeypatch.setattr(pl, "_own_process_group", lambda pid: None)
    monkeypatch.setattr(pl, "_write_breadcrumb", lambda: None)
    monkeypatch.setattr(pl, "_tracked_pids", recorded)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: "9:9999")

    pl.adopt_pid(4242, "0:4242")
    assert recorded == {}, "adopted a pid that is provably somebody else"

    monkeypatch.setattr(pl, "_pid_identity", lambda pid: None)
    pl.adopt_pid(4242, "0:4242")
    assert recorded == {}, "adopted a pid whose identity could not be confirmed"

    monkeypatch.setattr(pl, "_pid_identity", lambda pid: "0:4242")
    pl.adopt_pid(4242, "0:4242")
    assert recorded == {4242: "0:4242"}

    recorded.clear()
    monkeypatch.setattr(pl, "_identity_for_record", lambda pid: "read-now")
    pl.adopt_pid(4343)
    assert recorded == {4343: "read-now"}


class _FakeKernel32:
    """Each kernel32 entry is a settable stub: _win_signatures assigns argtypes and restype onto it."""

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
    """Read the creation time through the handle, which pins the process the later call acts on."""
    import ctypes

    def _times(handle, created_ptr, *rest):
        if handle != 77:
            return 0
        created_ptr._obj.dwHighDateTime = 7
        created_ptr._obj.dwLowDateTime = 4242
        return 1

    kernel32 = _FakeKernel32(GetProcessTimes = _times)
    assert pl._windows_identity_of_handle(kernel32, 77) == "7:4242"
    assert pl._windows_identity_of_handle(kernel32, 78) is None
    del ctypes


def test_a_pid_recycled_before_the_job_assignment_is_not_assigned(monkeypatch):
    """Re-read creation time through the handle just before job assignment; a pid check can go stale."""
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

    handle_identity["value"] = "9:9999"
    pl.adopt_pid(4242, "0:4242")
    assert assigned == [], "a stranger was assigned to the kill-on-close job"
    assert closed == [77], "the handle was leaked"

    handle_identity["value"] = None
    pl.adopt_pid(4242, "0:4242")
    assert assigned == [], "assigned a handle whose process could not be identified"

    handle_identity["value"] = "0:4242"
    pl.adopt_pid(4242, "0:4242")
    assert assigned == [77], assigned
    assert all(rights & 0x1000 for rights in opened_rights), opened_rights


def test_the_forced_kill_goes_through_the_handle_it_verified(monkeypatch):
    """A pid is a name: taskkill /PID resolves it after the check, so kill through the verified handle."""
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

    terminated.clear()
    handle_identity["value"] = "9:9999"
    assert pl._windows_terminate_pid(4242, "0:4242") is False
    assert terminated == []

    handle_identity["value"] = None
    assert pl._windows_terminate_pid(4242, "0:4242") is False
    assert terminated == []

    kernel32._stubs["OpenProcess"] = lambda rights, inherit, pid: 0
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: spawned.append(a[0]) or _types.SimpleNamespace(returncode = 0),
    )
    monkeypatch.setattr(pl, "_provably_the_same", lambda pid, identity: True)
    assert pl._windows_terminate_pid(4242, "0:4242") is True
    assert spawned and spawned[-1][0] == "taskkill", spawned

    spawned.clear()
    monkeypatch.setattr(pl, "_provably_the_same", lambda pid, identity: False)
    assert pl._windows_terminate_pid(4242, "0:4242") is False
    assert spawned == [], spawned
    monkeypatch.setattr(pl, "_provably_the_same", lambda pid, identity: identity is not None)
    assert pl._windows_terminate_pid(4242) is False
    assert spawned == [], spawned


def test_no_windows_kill_path_calls_taskkill_slash_t_any_more():
    """The collector only filters if no forced kill uses taskkill /T, which re-walks the parent-pid
    links."""
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
    """A listed child with unreadable identity makes the walk incomplete, though it is never signalled."""
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

    monkeypatch.setattr(pl, "_pid_alive", lambda pid: pid == root)
    found, known = pl.collect_descendants_known(root)
    assert (found, known) == ([], True)


def test_an_unverifiable_late_descendant_is_reported_not_dropped(monkeypatch):
    """An unreadable late descendant is not a recycled pid: report it, never drop it from `attempted`."""
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
    monkeypatch.setattr(
        pl,
        "_pid_identity",
        lambda pid: "0:6000" if pid == survivor else None,
    )
    reported = pl._windows_terminate_collected([(survivor, "0:6000")])
    assert late_pid in [pid for pid, _ in reported], reported

    monkeypatch.setattr(
        pl,
        "_pid_identity",
        lambda pid: "0:6000" if pid == survivor else "9:9999",
    )
    reported = pl._windows_terminate_collected([(survivor, "0:6000")])
    assert late_pid not in [pid for pid, _ in reported], reported


def test_the_root_is_stopped_before_its_snapshot_is_worked_through(monkeypatch):
    """Stop the root first: each kill can take 15 seconds, and the root may start children meanwhile."""
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
    """Re-collect each descendant just before its kill, so children started since the snapshot are
    reached."""
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
            return [(child, "0:801")], True
        if pid == child:
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
    """A failed re-collect below a descendant must not report the tree gone, or the record is forgotten."""
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
    """The fallback signal revalidates identity first; an unprovable identity gets no signal at all."""
    import subprocess as real_subprocess

    signalled: "list[int]" = []

    def _taskkill_fails(*args, **kwargs):
        raise OSError("taskkill is not on PATH")

    monkeypatch.setattr(real_subprocess, "run", _taskkill_fails)
    monkeypatch.setattr(pl.os, "kill", lambda pid, sig: signalled.append(pid))

    # No colons: on Linux `_same_identity` compares only the part before the first one.
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: "999")
    assert pl._windows_terminate_pid(4242, "111") is False
    assert signalled == [], "a recycled pid was signalled by the fallback"

    assert pl._windows_terminate_pid(4242, "999") is False
    assert signalled == [4242]

    signalled.clear()
    assert pl._windows_terminate_pid(4242) is False
    assert signalled == []


def test_a_number_recycled_between_the_snapshot_and_the_identity_read_is_rejected(monkeypatch):
    """A pid created after the snapshot cannot be one it listed, so a recycled number must be rejected."""
    root, child = 300, 301
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_child_pid_map", lambda: {root: [child]})
    monkeypatch.setattr(pl, "_windows_filetime_now", lambda: 1000)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: {root: "0:500", child: "0:2000"}.get(pid))
    monkeypatch.setattr(
        pl,
        "_windows_creation_time",
        lambda identity: (None if identity is None else int(identity.split(":")[1])),
    )

    found, known = pl._windows_collect_descendants_known(root)
    assert found == [], found
    assert known is True

    monkeypatch.setattr(pl, "_pid_identity", lambda pid: {root: "0:500", child: "0:600"}.get(pid))
    found, known = pl._windows_collect_descendants_known(root)
    assert found == [(child, "0:600")], found
    assert known is True


def test_a_child_of_a_late_child_is_reached_too(monkeypatch):
    """The late walk has the same race: a late child can start a grandchild during the kills."""
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
    """Kill rounds are bounded; a subtree that keeps respawning is reported unaccounted, record kept."""
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
    """Read a process's children before signalling it; once the parent dies, the walk cannot reach them."""
    survivor, middle, grandchild = 1100, 1101, 1102
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_signalable", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: f"{pid}")
    dead: "set[int]" = set()
    killed: "list[int]" = []

    def _collect(pid, identity = None):
        if pid == survivor:
            return [(middle, f"{middle}")], True
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
            return [], False
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
    """A pid created inside the snapshot window is unprovable: skip it and mark the walk incomplete."""
    root, child = 1300, 1301
    monkeypatch.setattr(pl, "_is_windows", lambda: True)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    clock = {"now": 1000}

    def _table():
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

    clock["now"] = 1000
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: {root: "0:500", child: "0:900"}.get(pid))
    found, known = pl._windows_collect_descendants_known(root)
    assert found == [(child, "0:900")], found
    assert known is True


def test_a_survivor_whose_identity_cannot_be_read_is_still_reported(monkeypatch):
    """The final pass is accounting: an unreadable-identity survivor is reported, not signalled."""
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
        return "1400" if reads["n"] == 1 else None

    monkeypatch.setattr(pl, "_pid_identity", _identity)
    survivors = pl._windows_terminate_collected([(survivor, "1400")])
    assert survivors == [(survivor, "1400")], survivors


def test_a_tree_kill_does_not_follow_the_number_to_a_stranger(monkeypatch):
    """Take the caller's identity as an argument; re-reading it would bless a stranger holding the pid."""
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_signalable", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
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
    assert pl._windows_terminate_validated_tree(500, "0:500") is False
    assert killed == [], "a stranger holding a recycled number must not be signalled"
    assert walked == [], "nor may its tree be enumerated as if it were ours"

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
    """Hitting the capture depth is an unperformed walk, not an empty one; the tree stays unresolved."""
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_signalable", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: f"0:{pid}")
    monkeypatch.setattr(pl, "_windows_terminate_pid", lambda pid, identity = None: True)
    monkeypatch.setattr(
        pl,
        "_windows_collect_descendants_known",
        lambda pid, identity = None: ([(pid + 1, f"0:{pid + 1}")], True),
    )
    attempted: "list[tuple[int, object]]" = []
    unresolved: "list[tuple[int, object]]" = []
    result = pl._windows_kill_below(500, attempted, unresolved, set(), 3)
    assert result is True, "the levels it did reach were killed"
    assert (503, "0:503") in unresolved, unresolved

    assert pl._windows_kill_below(500, [], [], set(), 0) is None


def test_the_snapshot_is_rooted_at_the_identity_the_caller_validated(monkeypatch):
    """Root the walk at the caller's validated identity; a re-read pid lets a stranger's children pass."""
    monkeypatch.setattr(pl, "_is_linux", lambda: False)
    monkeypatch.setattr(pl, "_signalable", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_pid_identity", lambda pid: "900:500")
    monkeypatch.setattr(
        pl, "_windows_creation_time", lambda ident: int((ident or "0:0").split(":")[0])
    )
    monkeypatch.setattr(pl, "_child_pid_map", lambda: {500: [600]})
    monkeypatch.setattr(pl, "_windows_filetime_now", lambda: 10_000)

    assert pl._windows_collect_descendants_known(500, "0:500") == ([], False)
    found, known = pl._windows_collect_descendants_known(500)
    assert known is True
    assert [pid for pid, _ in found] == [600]


def test_a_survivor_with_no_readable_identity_is_not_adopted(monkeypatch):
    """A (pid, None) survivor is unreadable, not new; adopt skips it rather than capturing an identity."""
    recorded = {}
    monkeypatch.setattr(pl, "_signalable", lambda pid: True)
    monkeypatch.setattr(pl, "_is_windows", lambda: False)
    monkeypatch.setattr(pl, "_identity_for_record", lambda pid: "999:whoever")
    monkeypatch.setattr(pl, "_own_process_group", lambda pid: None)
    monkeypatch.setattr(pl, "_write_breadcrumb", lambda: None)
    monkeypatch.setattr(pl, "_tracked_pids", recorded)

    pl.adopt_pid(777, None, from_snapshot = True)
    assert recorded == {}, recorded

    pl.adopt_pid(777, None)
    assert recorded == {777: "999:whoever"}


# Kills are asynchronous, so the survivor read-back needs a grace period.


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
    """Returns (leader, child pids) where each child ignores SIGTERM and dies only to SIGKILL."""
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
        assert elapsed < 1.0 + pl._KILL_SETTLE_SECONDS + 3.0, elapsed


@pytest.mark.skipif(IS_WINDOWS, reason = "kills a real process and reads it back")
def test_confirming_an_exit_waits_for_the_kill_to_land():
    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"])
    try:
        os.kill(victim.pid, signal.SIGKILL)
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
    """_pid_alive fails open on Windows, so a pid is dropped only after consecutive 'gone' readings."""
    reads = {"n": 0}

    def _alive_except_once(pid, identity):
        reads["n"] += 1
        return reads["n"] != 2

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
    """TerminateProcess returns before the process is gone: it lingers for reads_after_the_kill reads."""
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
    """Give the read-back time: TerminateProcess returns before the process is gone."""
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
    """A dying process must not count as a survivor; a read-back right after terminate races its exit."""
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
    """The record must be consumed after reaping; a read right after terminate reflects latency."""
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
    b._leading_process_group = lambda pid: pid
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

    b = _make_backend()
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
    """Confirming a group kill must signal nothing; the test checks os.kill itself, not only a helper."""
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
