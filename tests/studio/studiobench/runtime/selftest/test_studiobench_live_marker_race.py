# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two runs that start at the same moment: exactly one may have the output directory.

THE EXISTING GUARD TEST BUILDS ITS TWO RECORDERS ONE AFTER THE OTHER, which the broken code passed
trivially. A sequential test cannot see a race, and the guard was racy from the day it landed:

    the marker was named `.running.{session_id}`, so the two contenders raced on DIFFERENT paths.
    Each globbed the directory, each found no marker but its own, and each then wrote one. Scan
    and write are two syscalls and the window between them is wide enough to lose.

Measured on the shipped guard with two processes released together, holding their recorders open so
the overlap is real rather than sequential reuse: 107 of 200 trials admitted BOTH, and with four
processes 68 of 100 admitted more than one. That is the defect-9 corruption reproduced through the
guard written to prevent it -- two `run_meta` rows, one `cell_id` completed twice, the two copies
carrying 73.4 ms and 144.5 ms because the runs were contending with each other.

With the lock acquired by exclusive create on ONE fixed name, the same harness admits exactly one
in 200 trials at two processes and 100 at four.

A BARRIER, NOT SPAWNED PROCESSES ON A SHARED FLAG FILE. The window between the scan and the write
is roughly 100 to 200 microseconds, so the release has to be tighter than that or the contenders
simply arrive at different times and the broken guard looks fine. The first version of this test
used `subprocess.Popen` children spinning on a flag file and PASSED against the racy code three
times out of three, which is worth recording: a concurrency test that does not synchronise finely
enough is not a weak test, it is a green one that proves nothing.

Processes rather than threads: the liveness check is `os.kill(pid, 0)`, so two threads would be
judged against this process's own pid and the test would say nothing about two runs.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]

#: SIZED BY MEASUREMENT, not by guesswork. Under `spawn` the per-trial collision rate on the broken
#: guard is well below the 50% a barrier reaches under `fork`, because each contender pays its own
#: interpreter startup before arriving. At 10 trials the two headline tests missed the defect in one
#: run out of four; at 40 they caught it in six runs out of six, and the whole file still costs
#: about ten seconds. A concurrency test that only usually fails on the broken code is not much
#: better than one that never does.
#:
#: `spawn` rather than `fork` because pytest has already started threads by the time this runs, and
#: forking a multi-threaded process is deprecated for the good reason that the child can deadlock.
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
        # Nobody releases the marker until everyone has attempted, so an admission is genuine
        # overlap rather than sequential reuse of a directory the first run already let go.
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
        # What a crashed run leaves behind: a marker naming a pid that is gone.
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
    """The property the guard exists for, asserted against genuine concurrency.

    Against the racy guard this reports about half the trials admitting two.
    """
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
    """The reclaim path was the half the exclusive create did not fix.

    A marker naming a dead pid used to be cleared by unlinking it, and two launchers meeting the
    same stale marker both judged it dead: one unlinked and created its own, the other's unlink
    then deleted THAT, and both were admitted. Measured on the shipped code, 23 of 200 trials with
    two processes and 40 of 100 with four.

    There is no atomic "unlink if this is still the same file", so the fix is not a better check
    but a lock the kernel releases when the holder dies -- which leaves nothing to reclaim and no
    reclaim path to race.
    """
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


def _lock_then_write_later(repo_root: str, outdir: str, delay_s: float, locked, done) -> None:
    """Hold the lock across the gap a real holder has between locking and saying who it is."""
    import os
    import time

    sys.path.insert(0, repo_root)
    from tests.studio.studiobench.runtime.types import Recorder

    marker = Path(outdir) / ".running.lock"
    fd = os.open(marker, os.O_CREAT | os.O_RDWR, 0o644)
    Recorder._lock_fd_exclusive(fd)
    locked.set()
    time.sleep(delay_s)
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, f"{os.getpid()} heldsession\n".encode())
    os.fsync(fd)
    done.wait(timeout = 60)


def test_the_refusal_names_the_holder_that_has_not_written_its_marker_yet(tmp_path):
    """The refusal must survive the window between the holder's lock and its write.

    A holder takes the lock BEFORE it writes its pid and session, because it does not know it may
    write until it holds the lock. A loser arriving inside that window reads an empty marker, and
    the refusal used to fall back to "another run is still holding it" -- the phantom the named
    refusal exists to avoid. The window is microseconds on an idle machine, which is why the
    barrier tests only caught it on a loaded CI runner; here it is held open on purpose.
    """
    out = tmp_path / "held"
    out.mkdir()
    ctx = mp.get_context("spawn")
    locked, done = ctx.Event(), ctx.Event()
    holder = ctx.Process(
        target = _lock_then_write_later,
        args = (str(REPO_ROOT), str(out), 0.5, locked, done),
    )
    holder.start()
    try:
        assert locked.wait(timeout = 60), "the holder never took the lock"
        sys.path.insert(0, str(REPO_ROOT))
        from tests.studio.studiobench.runtime.types import Recorder, new_session_id

        with pytest.raises(SystemExit) as caught:
            Recorder(out / "payload.jsonl", new_session_id())
        assert "heldsession is still running" in str(caught.value), str(caught.value)
        assert f"pid {holder.pid}" in str(caught.value), str(caught.value)
    finally:
        done.set()
        holder.join(timeout = 60)
