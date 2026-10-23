# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two simultaneous runs must not both take one output directory; the lock needs one fixed name."""

from __future__ import annotations

import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]

# Sized by measurement: spawn collides less than fork, and 40 trials reliably catch the defect.
# spawn, not fork: pytest has threads running and forking them can deadlock.
TRIALS = 40


def _contend(repo_root: str, outdir: str, index: int, start, hold, q) -> None:
    """Take the directory, report, and keep holding until every contender has tried."""
    sys.path.insert(0, repo_root)
    from tests.studio.studiobench.runtime.types import Recorder, new_session_id

    rec = None
    start.wait()
    try:
        rec = Recorder(Path(outdir) / "payload.jsonl", new_session_id())
        q.put((True, ""))
    except SystemExit as exc:
        q.put((False, str(exc)))
    except Exception as exc:  # noqa: BLE001 - reported rather than lost
        q.put((False, f"UNEXPECTED {type(exc).__name__}: {exc}"))
    finally:
        # Nobody releases until all have attempted, so an admission is genuine overlap.
        try:
            hold.wait(timeout = 60)
        except Exception:  # noqa: BLE001
            pass
        if rec is not None:
            rec.close()


def _dead_pid() -> int:
    with open("/proc/sys/kernel/pid_max", encoding = "utf-8") as fh:
        return int(fh.read().strip()) - 1


def _trial(
    tmp_path: Path,
    n: int,
    trial: int,
    stale: bool = False,
) -> list[tuple[bool, str]]:
    out = tmp_path / f"out{trial}"
    out.mkdir()
    if stale:
        (out / ".running.lock").write_text(f"{_dead_pid()} crashedsession\n", encoding = "utf-8")
    ctx = mp.get_context("spawn")
    start, hold, q = ctx.Barrier(n), ctx.Barrier(n), ctx.Queue()
    procs = [
        ctx.Process(target = _contend, args = (str(REPO_ROOT), str(out), i, start, hold, q))
        for i in range(n)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout = 120)
    return [q.get(timeout = 10) for _ in range(n)]


def _admissions(
    tmp_path: Path,
    n: int,
    stale: bool = False,
) -> list[int]:
    counts = []
    for trial in range(TRIALS):
        got = _trial(tmp_path, n, trial, stale = stale)
        for ok, why in got:
            if not ok and why.startswith("UNEXPECTED"):
                pytest.fail(f"a contender failed for the wrong reason: {why}")
        counts.append(sum(1 for ok, _ in got if ok))
    return counts


def test_two_simultaneous_runs_cannot_both_take_one_output_directory(tmp_path):
    """Exactly one of two simultaneous runs may take an output directory; the racy guard let both in."""
    counts = _admissions(tmp_path, 2)
    assert set(counts) == {1}, (
        f"admitted-per-trial counts were {counts}; every trial must admit exactly one. Two runs "
        f"in one output directory both append to one payload.jsonl, every cell id is written "
        f"twice, and a reader keyed on the cell id sees whichever was appended last. That is the "
        f"withdrawn 149.8% regression."
    )


def test_four_simultaneous_runs_admit_exactly_one(tmp_path):
    """More contenders widen the window, so this is the same property under a harder push."""
    counts = _admissions(tmp_path, 4)
    assert set(counts) == {1}, f"admitted-per-trial counts were {counts}"


def test_the_refusal_names_the_run_that_holds_the_directory(tmp_path):
    """A refusal that does not say who holds it sends the reader looking for a phantom."""
    got = _trial(tmp_path, 2, 0)
    refused = [why for ok, why in got if not ok]
    assert len(refused) == 1
    assert "still running" in refused[0]


def test_the_directory_is_free_again_once_the_holder_exits(tmp_path):
    """The refusal must not outlive the run that caused it, or one race locks the dir forever."""
    got = _trial(tmp_path, 2, 0)
    assert sum(1 for ok, _ in got if ok) == 1
    sys.path.insert(0, str(REPO_ROOT))
    from tests.studio.studiobench.runtime.types import Recorder, new_session_id

    rec = Recorder(tmp_path / "out0" / "payload.jsonl", new_session_id())
    rec.close()


def test_a_crashed_run_does_not_let_two_launchers_in_at_once(tmp_path):
    """A stale marker reclaimed by unlinking raced; a kernel-released lock leaves nothing to reclaim."""
    counts = _admissions(tmp_path, 2, stale = True)
    assert set(counts) == {1}, (
        f"admitted-per-trial counts were {counts} against a crashed run's marker. Two launchers "
        f"reclaimed the same stale lock and both took the directory."
    )


def test_four_launchers_against_a_crashed_run_still_admit_one(tmp_path):
    counts = _admissions(tmp_path, 4, stale = True)
    assert set(counts) == {1}, f"admitted-per-trial counts were {counts}"


def test_a_crashed_run_does_not_lock_the_directory_forever(tmp_path):
    """The other direction: the refusal must not outlive the process that earned it."""
    out = tmp_path / "solo"
    out.mkdir()
    (out / ".running.lock").write_text(f"{_dead_pid()} crashedsession\n", encoding = "utf-8")
    sys.path.insert(0, str(REPO_ROOT))
    from tests.studio.studiobench.runtime.types import Recorder, new_session_id

    rec = Recorder(out / "payload.jsonl", new_session_id())
    rec.close()


# The marker is never unlinked, so a reused directory holds the previous run's line; not a holder.


def _stalled_holder(
    marker: Path,
    write_after_s: float,
    session: str = "realsession",
):
    """A holder that takes the lock and only then publishes itself, which is the required order:
    the write is what makes the marker say anything, and writing before the lock would let a LOSER
    publish itself as the holder."""
    code = (
        "import os,fcntl,time,sys\n"
        "fd=os.open(sys.argv[1],os.O_CREAT|os.O_RDWR,0o644)\n"
        "fcntl.flock(fd,fcntl.LOCK_EX)\n"
        "print('locked',flush=True)\n"
        "time.sleep(float(sys.argv[2]))\n"
        "os.ftruncate(fd,0); os.lseek(fd,0,0)\n"
        "os.write(fd,('%d %s\\n'%(os.getpid(),sys.argv[3])).encode()); os.fsync(fd)\n"
        "time.sleep(30)\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code, str(marker), str(write_after_s), session],
        stdout = subprocess.PIPE,
        text = True,
    )
    assert proc.stdout is not None
    proc.stdout.readline()
    return proc


def _refusal(out: Path) -> str:
    sys.path.insert(0, str(REPO_ROOT))
    from tests.studio.studiobench.runtime.types import Recorder, new_session_id

    with pytest.raises(SystemExit) as excinfo:
        Recorder(out / "payload.jsonl", new_session_id())
    return str(excinfo.value)


def test_a_clean_close_leaves_no_identity_behind(tmp_path):
    """close() blanks the marker record but keeps the file; unlinking it reopens the reclaim race."""
    sys.path.insert(0, str(REPO_ROOT))
    from tests.studio.studiobench.runtime.types import Recorder

    out = tmp_path / "out0"
    out.mkdir()
    rec = Recorder(out / "payload.jsonl", "sessionAAAA")
    marker = out / ".running.lock"
    assert "sessionAAAA" in marker.read_text()
    rec.close()
    assert marker.exists(), "the marker must not be unlinked"
    assert marker.read_text() == "", marker.read_text()


def test_a_retained_record_is_not_named_as_the_current_holder(tmp_path):
    """A record left by a dead run must not be named as the current holder of the directory."""
    out = tmp_path / "out0"
    out.mkdir()
    marker = out / ".running.lock"
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    marker.write_text(f"{dead.pid} sessionGONE\n")

    holder = _stalled_holder(marker, write_after_s = 0.35)
    try:
        message = _refusal(out)
    finally:
        holder.kill()
    assert "sessionGONE" not in message, message
    assert str(holder.pid) in message, message
    assert "realsession is still running" in message, message


def test_the_refusal_stays_generic_when_no_live_holder_can_be_named(tmp_path):
    """When no live holder can be named within the bound, the refusal stays generic rather than wrong."""
    out = tmp_path / "out0"
    out.mkdir()
    marker = out / ".running.lock"
    holder = _stalled_holder(marker, write_after_s = 60.0)
    try:
        message = _refusal(out)
    finally:
        holder.kill()
    assert "another run is still holding it" in message, message
