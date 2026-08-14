# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The reaper must never signal pid 0 or pid 1.

`killpg(1, sig)` is not "process group 1". POSIX defines it as `kill(-1, sig)`:
every process the caller has permission to signal. So a single record naming
pid 1 as a child turns the startup sweep into a SIGTERM of everything the user
owns, a five second wait, then a SIGKILL of the same.

That is not hypothetical. It was observed on a shared build box, from a record
holding `{"pid": 1, "identity": "...", "pgid": 1}`: starting Studio killed the
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
this module is reached with a pid below 2, on any path.
"""

from __future__ import annotations

import os
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
    """True is included deliberately: bool subclasses int, so an unguarded
    comparison reads it as pid 1."""
    assert pl._signalable(pid) is False


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
    assert recorded_signals == [], (
        "killpg(1, sig) is kill(-1, sig): every process the user owns"
    )


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
    assert not record.exists(), (
        "a poisoned record must be unlinked, or every launch retries it forever"
    )


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
    assert any(pid == 424242 for _call, pid, _sig in recorded_signals), (
        "a genuine orphan must still be signalled"
    )
