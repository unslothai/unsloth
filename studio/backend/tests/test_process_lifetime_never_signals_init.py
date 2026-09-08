# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The reaper must never signal pid 0 or pid 1.

`killpg(1, sig)` is not "process group 1". POSIX defines it as `kill(-1, sig)`:
every process the caller has permission to signal. So a single record naming
pid 1 as a child turns the startup sweep into a SIGTERM of everything the user
owns, a five second wait, then a SIGKILL of the same.

That is not hypothetical. It was observed on a shared build box, from a record
holding `{"pid": 1, "identity": "...", "pgid": 1}`: starting Unsloth killed the
user's tmux server and all twenty of their unrelated agent processes within one
second, then SIGKILLed the replacement tmux server exactly `timeout` later.

Nothing rejected it on the way in or on the way out:

  * `adopt_pid` guarded `not pid`, which rejects None and 0 but not 1.
  * The recycled-pid defence compares a recorded start time against the current
    one. init's start time never changes, so a recorded pid 1 matches forever.
    The check designed to make this safe is what guaranteed it fired.
  * `getpgid(1) == 1`, so pid 1 reads as a group leader and selects `killpg`.
  * The liveness probe between SIGTERM and SIGKILL is `killpg(1, 0)`, which can
    never fail, so the grace period always runs to completion.

These tests assert the outcome rather than the helper: no signalling call in
this module is reached with a pid below 2, on any path, and the same floor is
asserted at the sibling boundaries in `llama_cpp` and `download_registry` that
signal on a pid they read from disk rather than one they hold a handle to.
"""

from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

import utils.process_lifetime as pl  # noqa: E402

IS_POSIX = os.name == "posix"


@pytest.fixture
def recorded_signals(monkeypatch):
    """Capture every signal this module would send, and send none of them."""
    sent: "list[tuple[str, int, int]]" = []

    def _kill(pid, sig):
        sent.append(("kill", pid, sig))

    def _killpg(pgid, sig):
        sent.append(("killpg", pgid, sig))

    monkeypatch.setattr(pl.os, "kill", _kill)
    if hasattr(pl.os, "killpg"):
        monkeypatch.setattr(pl.os, "killpg", _killpg)
    return sent


# --- the guard itself ------------------------------------------------------


@pytest.mark.parametrize("pid", [None, 0, 1, -1, -12345, "1", 1.0, True, False])
def test_unsignalable_values_are_rejected(pid):
    """`True` is here for the floor's benefit, not the bool check's: `True >= 2`
    is already False, so `not isinstance(pid, bool)` is belt to the floor's
    braces and this case would still pass without it. It earns its place by
    pinning the behaviour if the floor is ever expressed a different way."""
    assert pl.is_signalable_pid(pid) is False


def test_the_public_name_is_the_internal_one():
    """Other modules import the public spelling. If the two ever come apart, the
    floor stops meaning one thing across Unsloth, which is how a site gets missed."""
    assert pl._signalable is pl.is_signalable_pid


@pytest.mark.parametrize("pid", [2, 3, 12345, 4194304])
def test_real_pids_are_accepted(pid):
    assert pl._signalable(pid) is True


# --- the write side: the bad record cannot be created ----------------------


def test_adopt_pid_refuses_init(monkeypatch):
    monkeypatch.setattr(pl, "_tracked_pids", {})
    monkeypatch.setattr(pl, "_tracked_pgids", {})
    monkeypatch.setattr(pl, "_write_breadcrumb", lambda: None)
    pl.adopt_pid(1)
    assert pl._tracked_pids == {}, "pid 1 must never enter the record"
    assert pl._tracked_pgids == {}


@pytest.mark.skipif(not IS_POSIX, reason = "process groups are POSIX only")
def test_own_process_group_refuses_group_one(monkeypatch):
    monkeypatch.setattr(pl.os, "getpgid", lambda pid: 1)
    assert pl._own_process_group(1) is None


# --- the signal side: an existing bad record cannot fire -------------------


@pytest.mark.skipif(not IS_POSIX, reason = "POSIX signalling path")
def test_posix_terminate_sends_nothing_for_init(recorded_signals):
    pl._posix_terminate(1, timeout = 0.01)
    assert recorded_signals == [], "killpg(1, sig) is kill(-1, sig): every process the user owns"


@pytest.mark.skipif(not IS_POSIX, reason = "POSIX signalling path")
@pytest.mark.parametrize("pid", [0, 1])
def test_posix_terminate_one_sends_nothing(recorded_signals, pid):
    pl._posix_terminate_one(pid, group_leader = True, timeout = 0.01)
    pl._posix_terminate_one(pid, group_leader = False, timeout = 0.01)
    assert recorded_signals == []


@pytest.mark.skipif(not IS_POSIX, reason = "POSIX signalling path")
def test_reap_orphaned_group_refuses_group_one(recorded_signals):
    assert pl._reap_orphaned_group(1, 1, timeout = 0.01) is False
    assert recorded_signals == []


@pytest.mark.skipif(not IS_POSIX, reason = "POSIX signalling path")
def test_terminate_descendants_skips_init(recorded_signals, monkeypatch):
    monkeypatch.setattr(pl, "_still_the_same", lambda pid, identity: True)
    pl.terminate_descendants([(1, "irrelevant")], timeout = 0.01)
    assert recorded_signals == []


# --- the end to end case that actually happened ----------------------------


@pytest.mark.skipif(not IS_POSIX, reason = "POSIX signalling path")
def test_poisoned_record_is_dropped_not_retried(tmp_path, monkeypatch, recorded_signals):
    """A record written by a build without the guard must fire nothing, and must
    not survive to be retried on every subsequent launch."""
    import json

    record = tmp_path / "4157196.json"
    record.write_text(
        json.dumps(
            {
                "owner_pid": 4157196,
                "owner_identity": "243506968",
                # Verbatim from the record that caused the incident. "963" is
                # init's start time in jiffies on that machine, so it is not
                # portable, and nothing here compares against it: the pid floor
                # short-circuits before identity is ever read. It stays because a
                # regression test for a specific incident should carry the bytes
                # that caused it.
                "children": [{"pid": 1, "identity": "963", "pgid": 1}],
            }
        ),
        encoding = "utf-8",
    )
    monkeypatch.setattr(pl, "_breadcrumb_dir", lambda: tmp_path)
    # The owner is long gone, which is what makes the sweep consider the record.
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: pid == 1)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)

    reaped = pl.reap_recorded_children(timeout = 0.01)

    assert recorded_signals == [], "the sweep must not signal init"
    assert reaped == [], "nothing was reaped, so nothing may be reported as reaped"
    assert (
        not record.exists()
    ), "a poisoned record must be unlinked, or every launch retries it forever"


@pytest.mark.skipif(not IS_POSIX, reason = "POSIX signalling path")
@pytest.mark.parametrize("pgid", [0, 1])
def test_poisoned_pgid_does_not_make_a_record_immortal(
    tmp_path, monkeypatch, recorded_signals, pgid
):
    """A real pid paired with a poisoned pgid.

    `killpg(1, 0)` is `kill(-1, 0)` and `killpg(0, 0)` is our own group, so both
    always succeed. Without a floor `_group_has_members` answers True for either,
    the entry is held `unresolved`, and the record is never unlinked: retried on
    every launch forever, which is the opposite of what dropping a poisoned
    record is for. The pid floor cannot catch this one, because the pid is fine.
    """
    import json

    record = tmp_path / "555555.json"
    record.write_text(
        json.dumps(
            {
                "owner_pid": 555555,
                "owner_identity": "111",
                "children": [{"pid": 424242, "identity": "222", "pgid": pgid}],
            }
        ),
        encoding = "utf-8",
    )
    monkeypatch.setattr(pl, "_breadcrumb_dir", lambda: tmp_path)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: False)  # the child is long gone
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)

    pl.reap_recorded_children(timeout = 0.01)

    assert [s for s in recorded_signals if s[1] < 2] == [], "not even a probe below pid 2"
    assert not record.exists(), "a record with a poisoned pgid must not be immortal"


@pytest.mark.skipif(not IS_POSIX, reason = "POSIX signalling path")
@pytest.mark.parametrize("pgid", [None, 0, 1, -1, True, "1"])
def test_group_has_members_refuses_unsignalable_groups(recorded_signals, pgid):
    assert pl._group_has_members(pgid) is False
    assert recorded_signals == []


@pytest.mark.skipif(not IS_POSIX, reason = "POSIX signalling path")
@pytest.mark.parametrize("pid", [0, 1, True])
def test_terminate_pid_sends_nothing(recorded_signals, monkeypatch, pid):
    """`terminate_pid` is the public single-child stop. Its old `if not pid:`
    admitted 1, and on Windows it reaches `_windows_terminate_tree` without
    passing through the POSIX helper that carries the other floor."""
    monkeypatch.setattr(pl, "_tracked_pids", {pid: "963"})
    monkeypatch.setattr(pl, "_tracked_pgids", {pid: pid})
    monkeypatch.setattr(pl, "_write_breadcrumb", lambda: None)

    pl.terminate_pid(pid, timeout = 0.01)

    assert recorded_signals == []


def test_terminate_all_never_signals_a_poisoned_tracked_pid(monkeypatch, recorded_signals):
    """The in-memory table cannot hold a 1 now, but `terminate_all` is what runs
    at shutdown and at `atexit`, so it carries its own floor. Nothing else covers
    that line: the existing suites drive it with real spawned pids only."""
    monkeypatch.setattr(pl, "_tracked_pids", {1: "963", 0: "0"})
    monkeypatch.setattr(pl, "_tracked_pgids", {1: 1, 0: 0})
    monkeypatch.setattr(pl, "_write_breadcrumb", lambda: None)

    pl.terminate_all(timeout = 0.01)

    assert recorded_signals == []


# --- the same shape elsewhere: the llama-server group killer ---------------


@pytest.mark.skipif(not IS_POSIX, reason = "POSIX signalling path")
@pytest.mark.parametrize("pid", [None, 0, 1, -1, True])
def test_leading_process_group_never_returns_init(monkeypatch, pid):
    """`getpgid(1) == 1`, so without a floor init reads as a group leader and the
    killer below broadcasts SIGKILL with no SIGTERM grace at all."""
    from core.inference.llama_cpp import LlamaCppBackend

    monkeypatch.setattr(os, "getpgid", lambda p: p)
    assert LlamaCppBackend._leading_process_group(pid) is None


@pytest.mark.skipif(not IS_POSIX, reason = "POSIX signalling path")
@pytest.mark.parametrize("pgid", [None, 0, 1, -1, True])
def test_kill_process_group_sends_nothing_for_init(monkeypatch, pgid):
    from core.inference import llama_cpp as lc

    sent: "list[tuple[int, int]]" = []
    monkeypatch.setattr(lc.os, "killpg", lambda g, s: sent.append((g, s)))
    lc.LlamaCppBackend._kill_process_group(pgid)
    assert sent == []


@pytest.mark.skipif(not IS_POSIX, reason = "POSIX signalling path")
def test_kill_process_group_still_kills_a_real_group(monkeypatch):
    """The floor must not disarm the cleanup it guards."""
    from core.inference import llama_cpp as lc

    sent: "list[tuple[int, int]]" = []
    monkeypatch.setattr(lc.os, "killpg", lambda g, s: sent.append((g, s)))
    lc.LlamaCppBackend._kill_process_group(424242)
    assert [g for g, _s in sent] == [424242]


@pytest.mark.skipif(not IS_POSIX, reason = "POSIX signalling path")
def test_valid_record_still_reaps(tmp_path, monkeypatch, recorded_signals):
    """The guard must not disarm the feature it protects: a real child is still
    signalled."""
    import json

    record = tmp_path / "999999.json"
    record.write_text(
        json.dumps(
            {
                "owner_pid": 999999,
                "owner_identity": "111",
                "children": [{"pid": 424242, "identity": "222", "pgid": 424242}],
            }
        ),
        encoding = "utf-8",
    )
    monkeypatch.setattr(pl, "_breadcrumb_dir", lambda: tmp_path)
    monkeypatch.setattr(pl, "_pid_alive", lambda pid: pid == 424242)
    monkeypatch.setattr(pl, "_pid_is_zombie", lambda pid: False)
    monkeypatch.setattr(pl, "_identity_or_none", lambda pid: "222")
    monkeypatch.setattr(pl.os, "getpgid", lambda pid: pid)

    reaped = pl.reap_recorded_children(timeout = 0.01)

    assert 424242 in reaped
    # The signal number is asserted, not just the pid. Without it a fully
    # disarmed reaper passes: `_group_has_members` emits `killpg(pid, 0)` as a
    # liveness probe, and a bare `pid == 424242` match accepts that probe as
    # proof of a kill that never happened.
    assert any(
        pid == 424242 and sig == signal.SIGTERM for _call, pid, sig in recorded_signals
    ), "a genuine orphan must still be sent a terminating signal"


# --- the sibling reapers that read a pid off disk ---------------------------


@pytest.mark.skipif(not IS_POSIX, reason = "POSIX signalling path")
def test_llama_pidfile_reaper_refuses_init(tmp_path, monkeypatch, recorded_signals):
    """The llama-server pidfile is the other place a pid arrives from disk rather
    than from a live handle, which is the precondition the incident needed.

    Every check behind the floor is stubbed to say yes, deliberately. Leave them
    real and this test passes with the floor removed, because `_pid_is_llama_server(1)`
    is False on a normal box and it is the cmdline check, not the floor, doing the
    work. #7894 established Unsloth can run as a container entrypoint, and a
    container whose entrypoint is llama-server has a process at pid 1 that answers
    yes to all of them.
    """
    from core.inference import llama_cpp as lc

    pidfile = tmp_path / "llama-server.pid"
    pidfile.write_text("1:963", encoding = "utf-8")
    monkeypatch.setattr(
        lc.LlamaCppBackend, "_server_pidfile_path", classmethod(lambda cls: pidfile)
    )
    monkeypatch.setattr(lc.LlamaCppBackend, "_pid_parent_is_alive", staticmethod(lambda pid: False))
    monkeypatch.setattr(lc.LlamaCppBackend, "_pid_start_identity", staticmethod(lambda pid: "963"))
    monkeypatch.setattr(lc.LlamaCppBackend, "_pid_is_llama_server", staticmethod(lambda pid: True))
    monkeypatch.setattr(lc.os, "kill", lambda pid, sig: recorded_signals.append(("kill", pid, sig)))

    assert lc.LlamaCppBackend._reap_recorded_pid() == 0, "init is never a reaped orphan"
    assert recorded_signals == [], "not even with every identity check saying yes"
    assert not pidfile.exists(), "a pidfile naming init is garbage, not something to retry"


@pytest.mark.skipif(sys.platform != "linux", reason = "the procfs scan only runs on Linux")
def test_llama_orphan_sweep_skips_init(tmp_path, monkeypatch):
    """The orphan sweep must not kill pid 1 even when pid 1 looks exactly like an
    owned, parentless llama-server.

    That is not a contrived shape: #7894 established Unsloth can run as a
    container entrypoint, and a container whose entrypoint is llama-server puts
    a process this sweep recognises at pid 1. Killing it takes the container
    down. The /proc scan and the psutil scan both feed one kill loop, so the
    floor sits on the loop rather than in each scanner.
    """
    from core.inference import llama_cpp as llama_cpp_module
    from core.inference.llama_cpp import LlamaCppBackend

    owned_dir = tmp_path / "unsloth-test-llama"
    owned_dir.mkdir()
    binary = owned_dir / "llama-server"
    binary.write_text("x")

    mypid = os.getpid()
    root = tmp_path / "fake-proc"
    root.mkdir()
    for pid in (1, mypid + 1):
        d = root / str(pid)
        d.mkdir()
        # Same 52-field stat shape the scanner parses; only comm and the start
        # time field are read.
        fields = " ".join(["0"] * 50)
        (d / "stat").write_bytes(f"{pid} (llama-server) S {fields}".encode())
        (d / "exe").symlink_to(binary)

    killed: "list[int]" = []
    monkeypatch.setenv("LLAMA_SERVER_PATH", str(binary))
    monkeypatch.setattr(llama_cpp_module, "_PROC_ROOT", str(root))
    monkeypatch.setattr(LlamaCppBackend, "_reap_recorded_pid", staticmethod(lambda: 0))
    monkeypatch.setattr(LlamaCppBackend, "_pid_parent_is_alive", staticmethod(lambda pid: False))
    monkeypatch.setattr(os, "kill", lambda pid, sig: killed.append(pid))

    LlamaCppBackend._kill_orphaned_servers()

    assert 1 not in killed, "the sweep must never SIGKILL init"
    assert killed == [mypid + 1], "the genuine owned orphan must still be reaped"


@pytest.mark.skipif(not IS_POSIX, reason = "POSIX signalling path")
def test_download_registry_never_signals_init(tmp_path, monkeypatch):
    """`reap_orphan_workers` had no test at all, so its floor had no test either:
    reverting it to the old `pid <= 0` passed the whole suite."""
    import json

    from hub.utils import download_registry as dr

    sent: "list[tuple[int, int]]" = []
    monkeypatch.setattr(dr.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(dr.state_dir, "workers_dir", lambda: tmp_path)
    settled: "list[object]" = []
    monkeypatch.setattr(dr, "_settle_orphaned_download", lambda *a, **k: settled.append(a))
    monkeypatch.setattr(dr, "_boot_sweep", lambda reaped: None)

    entry = tmp_path / "poisoned.json"
    entry.write_text(
        json.dumps({"pid": 1, "repo_type": "model", "repo_id": "Org/Model"}), encoding = "utf-8"
    )

    dr.reap_orphan_workers()

    assert sent == [], "the download reaper must not signal init either"
    assert not entry.exists(), "the poisoned breadcrumb is dropped"
    assert settled, "the partial must still be settled, or the user re-downloads from scratch"


@pytest.mark.skipif(not IS_POSIX, reason = "POSIX signalling path")
def test_kill_orphan_refuses_init_on_its_own(monkeypatch):
    """The helper that sends the signal carries the floor itself, so it does not
    depend on every future caller having checked first."""
    from hub.utils import download_registry as dr

    sent: "list[tuple[int, int]]" = []
    monkeypatch.setattr(dr.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    assert dr._kill_orphan(1) is False
    assert sent == []
