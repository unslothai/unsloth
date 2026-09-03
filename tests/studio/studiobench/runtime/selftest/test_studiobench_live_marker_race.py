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
import subprocess
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


# ── a retained record is not a holder ────────────────────────────────
#
# The marker is deliberately never unlinked, so on a directory that has been used before it already
# contains the PREVIOUS run's `pid session` line. These two are deterministic on purpose: they stand
# in the window rather than racing for it, so they say the same thing on a loaded runner as on an
# idle one.


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
    """`close()` blanks the marker while it still holds the lock.

    The file stays, because unlinking it reintroduces the reclaim race in reverse. Its CONTENT does
    not, because a retained `pid session` line outlives the run that wrote it.
    """
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
    """THE MISATTRIBUTION. A refusal that names the wrong run is worse than one that names none.

    Seeded as an UNCLEAN exit leaves it -- a killed run never reaches the blanking above -- so the
    stale line survives. A new holder then takes the lock and has not rewritten yet, and a reader
    that stops at the first non-empty record is handed the dead run and prints it as the holder.
    """
    out = tmp_path / "out0"
    out.mkdir()
    marker = out / ".running.lock"
    # A pid that is definitely not running: spawn and reap.
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
    """The bound is not load-bearing: when it expires the wording gets less specific, never wrong.

    A holder that never publishes itself is the shape a run killed between the lock and the write
    leaves behind, and there is nothing truthful to say about who holds the directory.
    """
    out = tmp_path / "out0"
    out.mkdir()
    marker = out / ".running.lock"
    holder = _stalled_holder(marker, write_after_s = 60.0)
    try:
        message = _refusal(out)
    finally:
        holder.kill()
    assert "another run is still holding it" in message, message
