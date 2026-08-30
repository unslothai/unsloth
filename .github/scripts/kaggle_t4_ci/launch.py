# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Push the kernels to Kaggle, wait for them, and bring the evidence back.

One invocation handles every kernel of the run. All of them are pushed FIRST
and only then waited on: pushing one, waiting, then pushing the next would
serialise sessions Kaggle runs happily at once, and the control/canary pair's
whole value is that they ran at the same time on the same account. A kernel that
could not be pushed does not stop the others; the verdict compares the reports
that came back against the number expected, so a half-launched run reports
``partial``, a warning rather than a failure, since half a comparison is not
evidence of a regression.

Failure semantics are the point of this file. Two kinds of bad outcome exit
differently, because conflating them is how a flaky external service blocks
merges:

* ``exit 0`` with ``verdict=infra`` -- the test never ran, or its result never
  got back: push throttled, account at its concurrency cap, a kernel that died
  on Kaggle's side, a download that would not complete, our own wall-clock
  ceiling. Nothing was learned, so nothing turns red.
* ``exit 0`` with ``verdict=pass`` / ``verdict=fail`` -- the payload ran and
  reached a conclusion. Judging it is ``report.py``'s job; this file only
  transports it.

The only nonzero exit is a usage error.

Wall clock is bounded three times over, and the order of trust is the opposite
of what it looks like:

* **Deleting the kernel**, the control observed to work. Every kernel this
  process pushed is deleted on the way out, on every path including failures,
  and deletion stops the billing (measured: the account's used-hours figure went
  DOWN when a wedged kernel was deleted).
* **Our polling deadline** (``--max-wait``), which decides when to give up and
  therefore when to delete.
* **Kaggle's own kernel timeout**, passed at push time. A backstop, NOT
  sufficient alone: on 2026-08-11 a kernel pushed with ``-t 5400`` whose
  nbconvert crashed at t=406s sat in RUNNING for over two hours, past that
  ceiling and past this process's deadline, and stopped only on a manual delete.
  The value is still passed, but nothing rests on it.

A socket timeout is set globally for the same reason: one status call that never
returns stalls the poll loop past every deadline above, which is how that
two-hour kernel went unnoticed.

No credential is printed. The token is read from the environment by the Kaggle
client and never echoed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import select
import atexit
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path, PurePosixPath, PureWindowsPath

sys.path.insert(0, str(Path(__file__).resolve().parent))

# What Kaggle says when a kernel is not there. Defined next to the gate's
# survey, which has to tell a deleted kernel from an unreadable one for the
# same reason cleanup does. See _already_gone.
from gate import GONE_MARKERS  # noqa: E402

API_ROOT = "https://www.kaggle.com/api/v1"
RESULT_PREFIX = "T4_SMOKE_REPORT "
OUTPUT_SUFFIX = "_output.ipynb"

TERMINAL_OK = {"COMPLETE"}
TERMINAL_BAD = {"ERROR", "CANCEL_REQUESTED", "CANCEL_ACKNOWLEDGED"}

_STATUS_RE = re.compile(r"KernelWorkerStatus\.(?P<status>[A-Z_]+)")

PUSH_ATTEMPTS = 4
PUSH_BACKOFF_SEC = 45

# Ceiling on one `kaggle kernels push` subprocess. Named because the workflow's
# job timeout derives from it, and because exceeding it is a retryable,
# ambiguous outcome rather than an error -- see push().
PUSH_SUBPROCESS_TIMEOUT_SEC = 600

# What a throttled or briefly unavailable push looks like coming back. The
# JSON-decode message is the common face: Kaggle answers 429 and 503 with an
# HTML error page, the client decodes it as JSON anyway, and the throttling is
# never named. Everything else (a bad slug, a rejected accelerator, missing
# credentials) is deterministic and must not be retried.
THROTTLED_PUSH = (
    "expecting value: line 1 column 1",
    "429",
    "too many requests",
    "502",
    "503",
    "service unavailable",
    "bad gateway",
    "timed out",
    "connection reset",
    "connection aborted",
)

# Kaggle's concurrency cap, reported as a push rejection rather than a queue.
CAPACITY_MARKERS = (
    "maximum batch gpu session count",
    "session count of 2 reached",
    "toomanyassignments",
    "precondition failed",
    "412",
    "no accelerator quota",
    "no quota for",
)

# Ceiling on any single network call. urllib takes an explicit timeout and the
# Kaggle client does not, so this is the only bound on its status and quota
# calls. Without it, one call that never returns outlasts every deadline in this
# file while the kernel it was watching keeps billing.
SOCKET_TIMEOUT_SEC = 120

# Consecutive unreadable statuses before we stop waiting. One is not enough: the
# API returns transient 5xx that the client prints exactly like a permanent
# refusal, and giving up on a blip abandons a kernel still doing the work.
MAX_CONSECUTIVE_UNKNOWN = 10

# Attempts at deleting ONE kernel, and the first gap between them. Deleting is
# the budget control, so a refused delete is retried rather than written off,
# against the same transient 5xx and reset connections push() retries.
#
# Named, like the push ceiling, because the job timeout is derived from them:
# ONE delete costs DELETE_ATTEMPTS x DELETE_SUBPROCESS_TIMEOUT_SEC plus its
# backoffs, and both push() (which discards the previous attempt's slug before
# each retry) and release() (which reconciles every slug filed) pay it. The
# harness suite recomputes the total from these three and asserts the workflow's
# deadline sits above it.
DELETE_ATTEMPTS = 3
DELETE_BACKOFF_SEC = 5
DELETE_SUBPROCESS_TIMEOUT_SEC = 180

# Pages of `kernels/output` one listing walks before it stops asking, and the
# ceiling on the WHOLE evidence phase: every kernel's listing pages and every
# notebook download, together.
#
# ONE budget for all of it, started before the first collection, because that
# is the only shape the job deadline can be derived from. Per call, the bound
# was the product of things nobody multiplied out: OUTPUT_PAGE_LIMIT pages at
# the 120s socket ceiling is 2400s for ONE kernel's listing alone, and each
# executed notebook is another 300s with no cap on how many Kaggle lists. The
# workflow header budgeted 600s for two kernels and the harness suite asserted
# the same 600s as a literal, so the term in the job deadline restated the
# intention rather than measuring the code: a paginating or slow endpoint spent
# the job's remaining wall clock HERE, and the runner was then killed before
# finish() -> release() ran, leaving billable kernels up. That is the exact
# outcome the deadline exists to prevent.
#
# Enforcement is "start no new work past the deadline", each call's own timeout
# is clamped to what remains, AND the body is read in chunks against the same
# absolute deadline. The last part is not redundant: the timeout urllib takes is
# a per-socket-operation timeout ("a timeout in seconds for blocking operations
# like the connection attempt" -- docs.python.org/3/library/urllib.request.html),
# not a ceiling on the transfer, so an endpoint that keeps returning bytes
# resets it forever and a single `resp.read()` outlasts the whole budget while
# every deadline check sits before the call. Evidence is best effort by design
# -- whatever arrived is reported and the collection is marked truncated --
# because the alternative is spending the deletion window on it.
EVIDENCE_BUDGET_SEC = 600
OUTPUT_PAGE_LIMIT = 20
# Per read. Small enough that the deadline is re-checked often against a slow
# stream, large enough not to syscall per kilobyte on a fast one.
READ_CHUNK_BYTES = 1 << 16


def worst_case_seconds(max_wait: int, kernels: int) -> int:
    """Wall clock ONE invocation of this launcher can take, from its constants.

    Every phase that can keep a pushed kernel UP is in it, because a kernel
    bills from the moment Kaggle accepts it until a delete is confirmed:

    * ``push()``, per notebook: PUSH_ATTEMPTS attempts at the subprocess
      ceiling, the backoffs between them, and the ``_discard()`` of the previous
      attempt's slug that rides along with every retry.
    * the polling, which shares ONE deadline started before the first push, so
      it does not stack on top of the pushes; the longer of the two is spent.
    * the evidence phase, ONE budget for every kernel together.
    * ``release()``, which deletes every slug every push FILED, not just the
      accepted one.

    Computed here rather than restated by each consumer: the workflow's job
    deadline, the quota the gate reserves, and the pre-push guard in main() are
    all wrong by the same amount if a phase is left out, and lowering
    PUSH_ATTEMPTS or a delete ceiling has to move all three at once.
    """
    one_delete = DELETE_ATTEMPTS * DELETE_SUBPROCESS_TIMEOUT_SEC + sum(
        DELETE_BACKOFF_SEC * 2**i for i in range(DELETE_ATTEMPTS - 1)
    )
    per_push = (
        PUSH_ATTEMPTS * PUSH_SUBPROCESS_TIMEOUT_SEC
        + sum(PUSH_BACKOFF_SEC * 2**i for i in range(PUSH_ATTEMPTS - 1))
        + (PUSH_ATTEMPTS - 1) * one_delete
    )
    return (
        max(kernels * per_push, max_wait)
        + EVIDENCE_BUDGET_SEC
        + kernels * PUSH_ATTEMPTS * one_delete
    )


_STDOUT_FD = 1

# Set once, on entry to the signal handler, and never cleared: the process is dying.
# Below it, NO line written to stdout may either block or raise, and both have to be
# handled here rather than at the handler's own call sites. release() reports a refused
# delete and warns about a kernel it could not delete through the ordinary path, so those
# lines are reached transitively, and either failure strands the cleanup: a block never
# reaches the retry or the `finally`, and a raise propagates out of delete_kernel() and
# abandons the remaining delete attempts for a kernel that is still billing.
_IN_SIGNAL_HANDLER = False


def _writable(fd: int) -> bool:
    """Whether a write to ``fd`` can proceed without blocking, asked without writing.

    A regular file or a terminal always answers yes; a pipe answers no exactly when it
    has backed up, which is the case worth avoiding. Any error answers no: a closed or
    unselectable descriptor is not somewhere to risk a stall from a signal handler.
    """
    try:
        return bool(select.select([], [fd], [], 0)[1])
    except BaseException:  # noqa: BLE001
        return False


def _line_from_signal(line: str) -> None:
    """One line out, from a context where the ordinary path can raise OR block.

    It can RAISE: a handler runs on the main thread wherever that thread happened to be,
    and if it was inside a write to stdout the interpreter refuses the second one
    outright, ``RuntimeError: reentrant call inside <_io.BufferedWriter name='<stdout>'>``.
    That is not hypothetical; it was captured on a loaded CI runner, where it escaped the
    handler and left a cancelled launcher exiting 1 instead of dying of its signal.
    ``os.write`` goes straight to the descriptor and takes no lock the interrupted frame
    could already hold, which is what makes it usable as the fallback.

    It can also BLOCK, which raising does not cover and no ``except`` or ``finally``
    catches. stdout in CI is a pipe, and if the collector stops draining it both the
    buffered flush and the raw write sleep in the kernel. That backpressure is also what
    leaves the main thread parked mid-write, so the two arrive together. Asking first
    turns the stall into a dropped line: POSIX reports a pipe writable only when at least
    PIPE_BUF bytes fit, and these lines are far shorter than that.
    """
    if not _writable(_STDOUT_FD):
        return
    try:
        _emit(line)
    except BaseException:  # noqa: BLE001 -- a log line may never decide whether we die
        try:
            os.write(_STDOUT_FD, (line + "\n").encode("utf-8", "replace"))
        except BaseException:  # noqa: BLE001
            pass


def _emit(line: str) -> None:
    print(line, flush = True)


def _write_line(line: str) -> None:
    """Every line this script puts on stdout goes through here.

    Unconditional on the ordinary path, which is everything before something kills us:
    nothing dropped, nothing swallowed, no syscall added.
    """
    if not _IN_SIGNAL_HANDLER:
        _emit(line)
        return
    _line_from_signal(line)


def _log(msg: str) -> None:
    _write_line(f"[launch] {msg}")


def _log_from_signal(msg: str) -> None:
    """``_log`` for the handler's own lines. Kept as its own name because the handler
    runs before the flag it sets can matter to anything else reading this."""
    _line_from_signal(f"[launch] {msg}")


def _out(key: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding = "utf-8") as fh:
            if "\n" in value:
                delim = f"ghadelim{uuid.uuid4().hex}"
                fh.write(f"{key}<<{delim}\n{value}\n{delim}\n")
            else:
                fh.write(f"{key}={value}\n")


def _slugify(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return re.sub(r"-{2,}", "-", s)[:50].strip("-")


def _api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


# A kernel this process pushed but has not yet deleted is recorded here, so a
# LATER launch can reclaim it. This is the only cover for `kill -9`, where no
# handler of ours ever runs. Deliberately keyed on pid: an entry whose owner
# is still alive belongs to a run in progress and must not be touched, or one
# launcher would delete a concurrent launcher's kernel and report its absence
# as a failure of the code under test.
INFLIGHT = (
    Path(os.environ.get("UNSLOTH_WORKSPACE") or Path(__file__).resolve().parents[3])
    / "logs"
    / "kaggle_inflight.json"
)


def _inflight_read() -> list[dict]:
    try:
        data = json.loads(INFLIGHT.read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def _inflight_write(entries: list[dict]) -> None:
    try:
        INFLIGHT.parent.mkdir(parents = True, exist_ok = True)
        INFLIGHT.write_text(json.dumps(entries, indent = 1), encoding = "utf-8")
    except OSError:
        pass  # bookkeeping only; never fail a run over it


def _inflight_add(slug: str) -> None:
    entries = [e for e in _inflight_read() if e.get("slug") != slug]
    entries.append({"slug": slug, "pid": os.getpid(), "at": time.time()})
    _inflight_write(entries)


def _inflight_drop(slug: str) -> None:
    _inflight_write([e for e in _inflight_read() if e.get("slug") != slug])


def _inflight_mark_kept(slug: str) -> None:
    """Flag a kernel as deliberately retained, so no later sweep reclaims it.

    ``--keep-kernel`` leaves the kernel up on purpose. Its registry entry
    still names this process, and this process is about to exit, so the next
    launcher would see a dead owner, call it an orphan and delete the very
    thing the flag asked to keep.
    """
    entries = _inflight_read()
    for entry in entries:
        if entry.get("slug") == slug:
            entry["keep"] = True
    _inflight_write(entries)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    except (OSError, TypeError):
        return True  # unknown: assume alive, deleting is the costly error
    return True


def sweep_orphans() -> list[str]:
    """Delete kernels left behind by a launcher that was killed outright.

    Only entries whose owning process is gone are eligible, so a concurrent
    run is never disturbed. Returns the slugs reclaimed.
    """
    entries = _inflight_read()
    if not entries:
        return []
    keep: list[dict] = []
    reclaimed: list[str] = []
    for entry in entries:
        slug, pid = entry.get("slug"), entry.get("pid")
        if not slug:
            continue
        if entry.get("keep"):
            # --keep-kernel asked for this one to stay up.
            keep.append(entry)
            continue
        if isinstance(pid, int) and _pid_alive(pid) and pid != os.getpid():
            keep.append(entry)
            continue
        if pid == os.getpid():
            keep.append(entry)
            continue
        try:
            proc = subprocess.run(
                ["kaggle", "kernels", "delete", slug, "-y"],
                capture_output = True,
                text = True,
                timeout = 180,
            )
        except Exception:  # noqa: BLE001
            keep.append(entry)  # try again next time rather than forget it
            continue
        # A nonzero exit does NOT raise, so the return code is the only thing
        # that separates "reclaimed" from "Kaggle refused and the kernel is
        # still running and still billing". Forgetting the entry there is how
        # one bills to its ceiling unnoticed.
        if proc.returncode == 0:
            reclaimed.append(slug)
        else:
            _log(f"could not delete orphan {slug} (rc={proc.returncode}); keeping the record")
            keep.append(entry)
    _inflight_write(keep)
    return reclaimed


def _pushed(ok: bool, reason: str, out: str, attempted: list[str]) -> dict:
    """A push outcome, always carrying the slugs the call filed.

    A failed attempt may still have landed, Kaggle answering a committed push
    with a 5xx often enough to be a known issue, so the slug is reported rather
    than forgotten and the caller can reconcile it.
    """
    return {"ok": ok, "reason": reason, "detail": out.strip()[:400], "attempts": attempted}


def push(
    notebook: Path,
    user: str,
    kernel_timeout_sec: int,
    accelerator: str = "NvidiaTeslaT4",
    attempted: list[str] | None = None,
) -> dict:
    """Push as a fresh private kernel. Every attempt gets its own slug.

    A fresh slug per attempt is not cosmetic: pushing to an id that ALREADY
    exists does not replace what is there, it files a new version and starts a
    SECOND batch session while the running one keeps running. ``kernels status``
    and ``kernels/output`` send no version label, so they answer for the newest
    session only, and a reused slug means the evidence belongs to whichever
    execution was latest while the other burns a slot and its quota unseen.

    The retried failures are exactly the ambiguous ones (a reset connection, an
    aborted transfer, a 5xx) where Kaggle may have accepted the push whose
    response never arrived. So each attempt also DELETES the previous attempt's
    slug first: deletion is kernel-level, costs one call, and frees the session
    slot the retry is probably waiting on.

    Returns the accepted attempt's slug, plus ``attempts``: every slug this call
    filed, newest last.

    ``attempted`` is that list, and the caller may OWN it: pass a list and every
    slug appears in it the moment it is filed, rather than on return. Only a
    return can be lost, and this function reaches the network, so anything it
    raises other than the timeout it handles (``subprocess.run(text=True)``
    decodes with strict error handling and raises ``UnicodeDecodeError`` on a
    malformed response; the runner can also answer with ``OSError`` or
    ``MemoryError``) would otherwise take the slug of a push Kaggle may have
    ACCEPTED with it, leaving a kernel nothing will delete. See main().
    """
    base = _slugify("unsloth t4 ci")[:32]
    attempted = [] if attempted is None else attempted

    def _discard(slug: str) -> None:
        """Best effort; the attempt usually created nothing at all.

        release() reconciles whatever this leaves, so the outcome is only logged
        here, but it IS read rather than assumed.
        """
        if not delete_kernel(slug):
            _log(f"could not discard the previous push attempt {slug}")

    workdir = Path(tempfile.mkdtemp(prefix = "kaggle-t4-ci-"))
    try:
        out = ""
        for attempt in range(PUSH_ATTEMPTS):
            if attempted:
                _discard(attempted[-1])
            slug_name = f"{base}-{uuid.uuid4().hex[:8]}"
            # The slug derives from the TITLE, not the metadata id: a mismatch
            # files the kernel at an unexpected address and every later
            # status/output call 403s, so assert the round trip.
            title = slug_name.replace("-", " ")
            assert _slugify(title) == slug_name, f"title {title!r} slugifies to {_slugify(title)!r}"
            attempted.append(f"{user}/{slug_name}")

            for stale in workdir.glob("*.ipynb"):
                stale.unlink()
            code_file = workdir / f"{slug_name}.ipynb"
            shutil.copy(notebook, code_file)
            (workdir / "kernel-metadata.json").write_text(
                json.dumps(
                    {
                        "id": f"{user}/{slug_name}",
                        "title": title,
                        "code_file": code_file.name,
                        "language": "python",
                        "kernel_type": "notebook",
                        "is_private": "true",
                        "enable_gpu": "true",
                        "enable_internet": "true",
                        "machine_shape": accelerator,
                        "dataset_sources": [],
                        "competition_sources": [],
                        "kernel_sources": [],
                        "model_sources": [],
                    },
                    indent = 2,
                ),
                encoding = "utf-8",
            )

            try:
                proc = subprocess.run(
                    [
                        "kaggle",
                        "kernels",
                        "push",
                        "-p",
                        str(workdir),
                        "--accelerator",
                        accelerator,
                        "-t",
                        str(kernel_timeout_sec),
                    ],
                    capture_output = True,
                    text = True,
                    timeout = PUSH_SUBPROCESS_TIMEOUT_SEC,
                )
                out = proc.stdout + proc.stderr
            except subprocess.TimeoutExpired:
                # A push that ran out of wall clock is the MOST ambiguous
                # outcome, not the least: the client was killed mid-call, so
                # whether Kaggle accepted the kernel is unknowable from here,
                # and letting the exception out loses the slug and with it every
                # chance of deleting the session it may have started.
                #
                # So it is recorded as a failed attempt like any other: the slug
                # stays in `attempted`, the retry _discard()s it, and release()
                # reconciles the rest. "timed out" is in THROTTLED_PUSH because
                # that is what it is, Kaggle under load, so the retry applies.
                out = f"push subprocess exceeded {PUSH_SUBPROCESS_TIMEOUT_SEC}s and was killed; timed out"
                _log(f"push timed out after {PUSH_SUBPROCESS_TIMEOUT_SEC}s ({attempted[-1]})")
                # Also recorded on disk. release() covers the paths this process
                # gets to run; the registry covers the one it does not, a kill
                # between here and cleanup, by leaving the slug for the next
                # launcher's sweep.
                _inflight_add(attempted[-1])
            lowered = out.lower()
            if "successfully pushed" in lowered:
                if "does not resolve to the specified id" in lowered:
                    return _pushed(False, "slug_mismatch", out, attempted)
                # Recorded the instant it exists, before anything can go
                # wrong downstream: a kernel that is billing but unknown to
                # the registry is exactly the case the registry is for.
                _inflight_add(attempted[-1])
                return {"ok": True, "slug": attempted[-1], "attempts": attempted}
            if any(m in lowered for m in CAPACITY_MARKERS):
                return _pushed(False, "at_capacity", out, attempted)
            if attempt + 1 == PUSH_ATTEMPTS or not any(m in lowered for m in THROTTLED_PUSH):
                return _pushed(False, "push_failed", out, attempted)
            delay = PUSH_BACKOFF_SEC * (2**attempt)
            _log(
                f"push looks throttled, retrying in {delay}s "
                f"(attempt {attempt + 1}/{PUSH_ATTEMPTS})"
            )
            time.sleep(delay)
        return _pushed(False, "push_failed", out, attempted)
    finally:
        shutil.rmtree(workdir, ignore_errors = True)


def _already_gone(text: str) -> bool:
    """Is this failed delete Kaggle saying the kernel is not there?

    The gate's vocabulary, imported rather than copied: the two files ask the
    same question of the same account through the same client, and a second
    list would drift out of agreement with the first without either being wrong
    on its own.
    """
    return any(marker in text.lower() for marker in GONE_MARKERS)


def delete_kernel(slug: str) -> bool:
    """Delete one kernel, and answer whether Kaggle actually deleted it.

    ``subprocess.run`` does not raise on a nonzero exit, so the caller used to
    record every slug as released whatever came back: a refused delete, an
    expired token, or the case that made this visible, where the pinned client
    had no ``kernels delete`` subcommand at all and argparse exited 2 before any
    request was sent while the run still reported the kernel released. Cleanup
    is this workflow's budget control, so an unestablished delete has to read as
    STILL BILLING.

    The exit code is the signal Kaggle's client offers, and it means what it
    says: kaggle/cli.py exits 1 on a failed call, commented "This is so that
    scripts that pick up on error codes can tell when there was a failure", and
    0 once the kernel is gone.

    NOT FOUND IS THE OTHER CONFIRMED ANSWER, and it is a common one here rather
    than an edge case: most of the slugs release() reconciles are earlier push
    attempts, which usually created nothing at all, and one that a retry's
    _discard() did delete is asked about a second time. Kaggle answering "this
    kernel is not there" settles the only question cleanup asks -- is the slot
    still billing -- so retrying it three times and then naming it in a
    "may still be running, delete them by hand" warning spends the deletion
    window on absent kernels, ahead of the live one, and points a human at a
    slug that does not exist. The gate reads a 404 the same way and for the
    same reason, so it reads it through the same GONE_MARKERS; the client
    surfaces one as `404 Client Error: Not Found for url: ...` on stderr with
    exit 1 (requests' raise_for_status, via kagglesdk's response handler).

    Every OTHER nonzero exit keeps its retries: a 5xx, a reset connection or an
    argparse refusal says nothing about the kernel.

    Returns True only on a confirmed deletion or a confirmed absence. A slug
    this refuses needs a human, which is what the caller's warning is for.
    """
    for attempt in range(DELETE_ATTEMPTS):
        try:
            proc = subprocess.run(
                ["kaggle", "kernels", "delete", slug, "-y"],
                capture_output = True,
                text = True,
                timeout = DELETE_SUBPROCESS_TIMEOUT_SEC,
            )
        except Exception as exc:  # noqa: BLE001
            _log(f"delete {slug} did not run: {type(exc).__name__}")
        else:
            if proc.returncode == 0:
                return True
            detail = " ".join(f"{proc.stdout} {proc.stderr}".split())
            if _already_gone(detail):
                _log(f"delete {slug}: Kaggle says it is not there, so the slot is free")
                return True
            _log(f"delete {slug} exited {proc.returncode}: {detail[:200]}")
        if attempt + 1 < DELETE_ATTEMPTS:
            time.sleep(DELETE_BACKOFF_SEC * (2**attempt))
    return False


def _slugs_filed(entry: dict) -> list[str]:
    """Every slug one kernel entry's push filed, in the order it filed them.

    Cleanup reconciles ALL of them, not only the accepted one, and both routes
    to a leaked slug are ambiguous by construction:

    * The last attempt of a FAILED push. ``push()`` keeps the slug precisely
      because Kaggle answers an accepted push with a 5xx or a reset connection
      often enough to be a known issue, and that entry carries no ``slug``.
    * An EARLIER attempt of a push that later succeeded. ``_discard()`` runs
      before each retry but is best effort, so a delete Kaggle refuses even
      after ``delete_kernel``'s retries leaves the previous attempt up.

    Either keeps a session slot and bills GPU quota with nobody reading the
    result. A delete for a slug Kaggle never created is refused and costs one
    call.
    """
    filed = [*(entry.get("attempted") or []), entry.get("slug")]
    return list(dict.fromkeys(s for s in filed if s))


def poll(api, slug: str) -> str:
    try:
        raw = str(getattr(api.kernels_status(slug), "status", ""))
    except Exception as exc:  # noqa: BLE001
        _log(f"status unreadable: {type(exc).__name__}")
        return "UNKNOWN"
    match = _STATUS_RE.search(raw)
    return match.group("status") if match else (raw.strip().upper() or "UNKNOWN")


def wait(api, slug: str, poll_every: int, max_wait: int) -> str:
    """Poll to a terminal state.

    An unreadable status must NOT count as "still running" forever: with no
    status to match, sitting here for the full ceiling on a kernel that long
    since finished is an hour of wall clock spent learning nothing. So bound the
    consecutive failures and hand back UNREADABLE, never COMPLETE, since how it
    ended is genuinely unknown.
    """
    deadline = time.time() + max_wait
    unknowns = 0
    last = "UNKNOWN"
    while time.time() < deadline:
        state = poll(api, slug)
        if state != last:
            _log(f"state: {state}")
            last = state
        if state in TERMINAL_OK or state in TERMINAL_BAD:
            return state
        if state == "UNKNOWN":
            unknowns += 1
            if unknowns >= MAX_CONSECUTIVE_UNKNOWN:
                return "UNREADABLE"
        else:
            unknowns = 0
        time.sleep(poll_every)
    return "HARNESS_TIMEOUT"


def _bearer() -> str:
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        raise RuntimeError("KAGGLE_API_TOKEN unset; REST path unavailable")
    return token


def _evidence_deadline(deadline: float | None) -> float:
    """The shared budget if the caller has one, otherwise a fresh one.

    Never unbounded. ``deadline=None`` used to mean "no ceiling", which is the
    state this constant exists to end, so the default is a budget of its own:
    a caller can only make the bound TIGHTER by sharing one, never absent.
    """
    return time.time() + EVIDENCE_BUDGET_SEC if deadline is None else deadline


def _time_left(deadline: float, timeout: int) -> int | None:
    """Seconds this call may take, or None when the deadline has passed.

    The single place the evidence budget is applied, so no caller can start a
    request the phase has no time for and none can outlast it: the timeout
    handed to urllib is the SMALLER of the call's own ceiling and what is left.
    """
    remaining = int(deadline - time.time())
    if remaining <= 0:
        return None
    return min(timeout, remaining)


def _clamp_socket(resp, seconds: float) -> None:
    """Tighten the live socket's timeout to what is left of the budget.

    The timeout handed to ``urlopen`` is fixed when the response opens, so a
    read that starts just inside the deadline may still block for all of it.
    Re-clamping per chunk bounds the overshoot by one chunk rather than by one
    socket timeout. Best effort: the loop below stops at the deadline whether or
    not the socket can be reached.
    """
    sock = getattr(getattr(getattr(resp, "fp", None), "raw", None), "_sock", None)
    setter = getattr(sock, "settimeout", None)
    if setter is None:
        return
    try:
        setter(max(seconds, 1.0))
    except OSError:
        pass


def _read_within(
    resp,
    deadline: float,
    sink = None,
) -> bytes:
    """Read a response body under the ABSOLUTE deadline, not a socket timeout.

    ``urlopen(timeout=...)`` bounds each blocking socket operation, so a server
    that keeps trickling bytes renews it indefinitely and ``resp.read()`` runs
    as long as it likes -- past the evidence budget, into the wall clock
    ``release()`` needs, with billable kernels still up. Checking the deadline
    only before opening the response therefore bounds nothing about the read.

    So: one chunk at a time, deadline re-checked before each, and ``read1`` so a
    chunk is at most ONE underlying socket read (``read(n)`` loops until it has
    n bytes, which a slow trickle can stretch arbitrarily). ``sink`` streams
    straight to disk, which also keeps a response nobody sized out of memory.
    """
    reader = getattr(resp, "read1", None) or resp.read
    chunks: list[bytes] = []
    while True:
        left = deadline - time.time()
        if left <= 0:
            raise TimeoutError("the evidence budget expired while reading the response")
        _clamp_socket(resp, left)
        chunk = reader(READ_CHUNK_BYTES)
        if not chunk:
            break
        if sink is None:
            chunks.append(chunk)
        else:
            sink.write(chunk)
    return b"".join(chunks)


def list_outputs(
    slug: str,
    timeout: int = 120,
    deadline: float | None = None,
) -> dict:
    """List a kernel's outputs, one page at a time, inside the budget.

    ``truncated`` says the listing is INCOMPLETE -- the page limit or the
    evidence deadline stopped it -- so a caller can tell "Kaggle listed these
    files" from "these are the files we got to before the clock ran out".
    """
    deadline = _evidence_deadline(deadline)
    user, _, name = slug.partition("/")
    params = {"userName": user, "kernelSlug": name}
    files: list[dict] = []
    log = ""
    truncated = True
    for _ in range(OUTPUT_PAGE_LIMIT):
        call_timeout = _time_left(deadline, timeout)
        if call_timeout is None:
            _log(f"evidence budget spent while listing {slug}")
            break
        url = f"{API_ROOT}/kernels/output?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url,
            headers = {
                "Authorization": f"Bearer {_bearer()}",
                "User-Agent": "unsloth-kaggle-t4-ci/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout = call_timeout) as resp:
                data = json.loads(_read_within(resp, deadline))
        except TimeoutError:
            # The budget ran out mid-body. Same answer as running out between
            # pages: stop, and say the listing is incomplete.
            _log(f"evidence budget spent while reading a page of {slug}")
            break
        files.extend(f for f in data.get("files") or [] if f.get("fileName"))
        log = log or (data.get("log") or "")
        token = data.get("nextPageToken") or ""
        if not (data.get("hasNextPageToken") and token):
            truncated = False
            break
        params = dict(params, pageToken = token)
    return {"files": files, "log": log, "truncated": truncated}


def _dest_name(file_name: str) -> str:
    """Basename of a listed output, safe on every platform.

    Kaggle lists nested outputs with POSIX separators. Joining a listed name
    onto the output directory unexamined would let ``../`` walk out of it, and
    ``Path`` answers for the HOST rather than for the name, so peel POSIX first
    and Windows second with both pure flavours.
    """
    name = PureWindowsPath(PurePosixPath(file_name).name).name
    return name or PurePosixPath(file_name).name


def fetch_evidence(
    slug: str,
    outdir: Path,
    timeout: int = 300,
    deadline: float | None = None,
) -> dict:
    """Pull the executed notebooks and the kernel log by direct URL.

    By direct URL rather than the bulk download: the bulk call returns the WHOLE
    of /kaggle/working as one stream, and a previous incident lost two PASSING
    notebooks because a multi-GB saved model sorted alphabetically ahead of them
    and the stream broke partway through.

    ``deadline`` is the shared evidence budget (EVIDENCE_BUDGET_SEC), an
    absolute ``time.time()`` value covering every kernel of the run. Nothing
    starts after it and every call is clamped to what is left of it, so this
    phase cannot eat the wall clock release() needs to delete the kernels.
    """
    deadline = _evidence_deadline(deadline)
    outdir.mkdir(parents = True, exist_ok = True)
    listing = list_outputs(slug, timeout = min(timeout, 120), deadline = deadline)
    fetched = []
    truncated = bool(listing.get("truncated"))
    for entry in listing["files"]:
        name = _dest_name(entry["fileName"])
        if not name.endswith(OUTPUT_SUFFIX):
            continue
        url = entry.get("url") or entry.get("urlNullable")
        if not url:
            continue
        call_timeout = _time_left(deadline, timeout)
        if call_timeout is None:
            _log(f"evidence budget spent; {name} and anything after it were not downloaded")
            truncated = True
            break
        dest = outdir / name
        part = dest.with_suffix(dest.suffix + ".part")
        try:
            req = urllib.request.Request(url, headers = {"User-Agent": "unsloth-kaggle-t4-ci/1.0"})
            with urllib.request.urlopen(req, timeout = call_timeout) as resp, part.open("wb") as fh:
                _read_within(resp, deadline, sink = fh)
            # Only publish once it parses: a download killed mid-write leaves a
            # file of plausible size, which is evidence that looks present and
            # is not.
            json.loads(part.read_text(encoding = "utf-8", errors = "replace"))
            part.replace(dest)
            fetched.append(dest.name)
        except Exception as exc:  # noqa: BLE001
            # A listed notebook that did not land is missing evidence, whether
            # the budget ran out mid-body or the transfer failed, so the
            # collection is incomplete and has to say so rather than read as a
            # complete set that happens to be short.
            _log(f"could not fetch {name}: {type(exc).__name__}")
            part.unlink(missing_ok = True)
            truncated = True
    log_path = outdir / "kernel.log"
    if listing.get("log"):
        log_path.write_text(listing["log"], encoding = "utf-8")
    return {
        "notebooks": fetched,
        "log": log_path.name if log_path.exists() else None,
        "truncated": truncated,
    }


def flatten_kernel_log(raw: str) -> str:
    """A kernel log as flat text, whichever shape Kaggle returned it in.

    ``kernels/output`` returns the log as a JSON array of
    ``{stream_name, time, data}`` records, whose boundaries are not line
    boundaries. Scanning it as-is finds no line beginning with the report
    prefix, so a payload whose executed notebook never came back -- the case
    this fallback exists for -- read as no report and was downgraded to
    ``infra``.

    ``report.kernel_log_text`` does the same for the summary; both are kept
    because the two scripts are separate processes and neither imports the
    other.
    """
    try:
        records = json.loads(raw)
    except ValueError:
        return raw
    if not isinstance(records, list):
        return raw
    return "".join(r.get("data", "") for r in records if isinstance(r, dict))


def extract_reports(outdir: Path) -> list[dict]:
    """Every T4_SMOKE_REPORT payload found in the collected evidence.

    Cell outputs of the executed notebooks first, flat kernel log second. The
    notebook is the better source (one cell, unambiguous ownership), but the log
    survives cases where the notebook never got written back.
    """
    reports: list[dict] = []
    seen: set[str] = set()

    def _consume(text: str) -> None:
        for line in text.splitlines():
            if not line.startswith(RESULT_PREFIX):
                continue
            blob = line[len(RESULT_PREFIX) :].strip()
            try:
                parsed = json.loads(blob)
            except json.JSONDecodeError:
                continue
            key = f"{parsed.get('label')}|{parsed.get('model')}"
            if key in seen:
                continue
            seen.add(key)
            reports.append(parsed)

    # rglob, not glob: each kernel collects into its own subdirectory so two
    # cannot overwrite each other's kernel.log.
    for nb_path in sorted(outdir.rglob(f"*{OUTPUT_SUFFIX}")):
        try:
            nb = json.loads(nb_path.read_text(encoding = "utf-8", errors = "replace"))
        except Exception:  # noqa: BLE001
            continue
        for cell in nb.get("cells", []):
            for output in cell.get("outputs", []):
                text = output.get("text") or ""
                if isinstance(text, list):
                    text = "".join(text)
                _consume(text)
    for log_path in sorted(outdir.rglob("kernel.log")):
        raw = log_path.read_text(encoding = "utf-8", errors = "replace")
        _consume(flatten_kernel_log(raw))
    return reports


def _install_release_handlers(release: Callable[[], None]) -> None:
    """Make release survive the ways this process actually dies.

        normal return / handled error   finish() calls it directly
        unhandled exception             atexit
        Ctrl-C, kill, Actions cancel    the signal handlers here
        kill -9                         nothing in-process can; the orphan
                                        sweep at the next launch reclaims it

    Before this, only the first row worked: `finish()` is reached only on
    paths that RETURN, KeyboardInterrupt was re-raised past it, and the
    default SIGTERM disposition exits without running atexit or finally. A
    cancelled workflow therefore left its kernel running.
    """
    atexit.register(release)

    def _release_and_die(signum, _frame):
        # First, and outside the try: from here on every _log in this process drops a
        # line rather than stalling on a stdout nobody is draining. release() logs
        # through the ordinary path -- delete_kernel() reports a refused delete that way
        # -- and a stall there is a stall before the retry and before the `finally`.
        global _IN_SIGNAL_HANDLER
        _IN_SIGNAL_HANDLER = True
        # Everything before the `finally` is best effort. The death is not: a handler
        # that returns normally leaves the process exiting on whatever code main()
        # computes, and a cancelled job then reads as a completed one.
        try:
            _log_from_signal(f"received signal {signum}; deleting kernels before exiting")
            release()
        except BaseException as exc:  # noqa: BLE001
            # A raise here used to propagate into the main thread, where main()'s
            # `except BaseException` caught it and finish() called release() again.
            # A transient first failure (the observed one: an OSError from a
            # subprocess spawn on a loaded runner) therefore ended in main
            # RETURNING 0, and the cancelled job read as completed. Deleting is
            # best effort, the exit status is not. Retried once, since the kernel
            # is billing meanwhile.
            _log_from_signal(f"release() failed under signal {signum}: {type(exc).__name__}: {exc}")
            try:
                release()
            except BaseException as retry_exc:  # noqa: BLE001
                # Nothing more to try in-process: the slug stays in the registry
                # for the next launch's orphan sweep, as after a kill -9.
                _log_from_signal(
                    f"release() failed again: {type(retry_exc).__name__}: {retry_exc}. "
                    f"The kernels stay in the registry for the next launcher's sweep"
                )
        finally:
            # Die of the original signal rather than exiting 0, so the status
            # still reads "killed by signal N". A CI system treats a 0 from a
            # cancelled job as a completed one. In a `finally` because anything
            # above can raise -- including the logging, which is how this was found.
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    for _sig in (signal.SIGINT, signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if _sig is None:
            continue
        try:
            signal.signal(_sig, _release_and_die)
        except (ValueError, OSError, AttributeError):
            # No SIGHUP on Windows, and signal() only works on the main
            # thread. One handler failing must not stop the others.
            pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--notebook",
        required = True,
        action = "append",
        help = "kernel notebook to push. Repeatable: all of them "
        "are pushed before any of them is waited on",
    )
    ap.add_argument("--user", required = True)
    ap.add_argument("--outdir", required = True)
    ap.add_argument(
        "--expect", type = int, default = 2, help = "payload reports this kernel should produce"
    )
    ap.add_argument(
        "--kernel-timeout-sec",
        type = int,
        default = 3600,
        help = "hard ceiling enforced by KAGGLE on the session",
    )
    ap.add_argument(
        "--max-wait",
        type = int,
        default = 4200,
        help = "wall clock this invocation gives the kernels, measured from BEFORE the first push",
    )
    ap.add_argument("--poll-every", type = int, default = 60)
    ap.add_argument(
        "--keep-kernel", action = "store_true", help = "do not delete the kernel after collecting"
    )
    ap.add_argument(
        "--deadline-epoch",
        type = int,
        default = 0,
        help = "unix time at which the CALLER is killed. Nothing is pushed unless "
        "worst_case_seconds() still fits before it. 0 disables the check",
    )
    args = ap.parse_args()

    # Before the first network call, and globally: the Kaggle client has no
    # per-call timeout of its own. See SOCKET_TIMEOUT_SEC.
    socket.setdefaulttimeout(SOCKET_TIMEOUT_SEC)

    outdir = Path(args.outdir)
    outdir.mkdir(parents = True, exist_ok = True)
    result: dict = {
        "verdict": "infra",
        "reason": "",
        "slug": None,
        "kernel_state": None,
        "reports": [],
    }

    def release() -> None:
        """Delete every kernel this process pushed. Idempotent, and on every
        path out of main().

        The budget control, not a tidy-up: a kernel left behind bills to its own
        ceiling with nobody reading the result, and Kaggle's push-time timeout
        has been observed not to stop one that wedged. Deleting a two-hour-old
        stuck kernel took the account's used-hours figure DOWN.

        Every slug the push FILED is reconciled, not only the accepted one; see
        ``_slugs_filed`` for the two ways an unaccepted slug can still be
        running. A slug counts as released only once Kaggle CONFIRMS the delete
        (see delete_kernel); anything else may still be billing, so it is named
        in the log, in ``launch_result.json`` and in a workflow annotation
        rather than quietly counted as cleaned up.
        """
        if args.keep_kernel:
            # Flagged, not just skipped: the registry entry still names this
            # process, and the next launcher would read a dead owner as an
            # orphan and delete the very kernel the flag asked to keep.
            for entry in result.get("kernels") or []:
                for slug in _slugs_filed(entry):
                    _inflight_mark_kept(slug)
            return
        leaked: list[str] = []
        for entry in result.get("kernels") or []:
            done = set(entry.get("released_slugs") or [])
            for slug in _slugs_filed(entry):
                if slug in done:
                    continue
                if delete_kernel(slug):
                    done.add(slug)
                    _inflight_drop(slug)
                else:
                    leaked.append(slug)
                    _log(f"could not delete {slug}; it may keep billing")
            entry["released_slugs"] = sorted(done)
            entry["released"] = all(s in done for s in _slugs_filed(entry))
        result["unreleased"] = leaked
        if leaked:
            # _write_line, not _log: the [launch] prefix would stop GitHub parsing
            # this as an annotation. Not print: release() runs from the signal handler
            # too, and this line is emitted on exactly the path where a kernel is still
            # billing, so a raw write here blocks or raises before the handler can
            # re-raise its signal.
            _write_line(
                "::warning title=Kaggle kernels may still be running::"
                + ", ".join(leaked)
                + " could not be deleted, so they may keep billing accelerator "
                "quota until they hit their own ceiling. Delete them by hand."
            )

    # From here on release() is reachable from a signal and from atexit too,
    # not only from finish(): a cancelled workflow used to leave its kernel up.
    _install_release_handlers(release)

    def finish(code: int = 0) -> int:
        release()
        (outdir / "launch_result.json").write_text(json.dumps(result, indent = 2), encoding = "utf-8")
        _out("verdict", result["verdict"])
        _out("reason", result["reason"])
        _out("slug", result["slug"] or "")
        _log(f"verdict={result['verdict']} reason={result['reason']}")
        return code

    def window_fits(after: str) -> bool:
        """Does the job's remaining wall clock still cover the worst case?

        Asked TWICE, because the answer expires: everything between the two
        asks is time the guard already granted. See the two call sites.
        """
        if not args.deadline_epoch:
            return True
        need = worst_case_seconds(args.max_wait, len(args.notebook))
        left = int(args.deadline_epoch - time.time())
        if left >= need:
            _log(f"{left}s left of the job deadline, worst case {need}s ({after})")
            return True
        result["reason"] = (
            f"only {left}s of the job's deadline are left {after} and this launcher "
            f"can take {need}s, so a push now could be killed during cleanup"
        )
        print(
            "::warning title=Stood down before pushing::"
            f"{after} there were {left}s of the job's deadline left and the launcher's "
            f"worst case is {need}s. Pushing now risks the runner being killed while "
            "kernels are still up, which would bill accelerator quota to their own "
            "ceiling. Nothing was pushed and nothing was learned about this change.",
            flush = True,
        )
        return False

    # BEFORE authentication and long before the first push: the one question
    # that has to be answered while the answer can still be acted on.
    #
    # The caller (the workflow job) is killed at a fixed time, and killing it
    # takes finish() -> release() with it: GitHub sends SIGINT to the step's
    # entry process and kills the process tree about ten seconds later
    # ("Canceling a workflow", docs.github.com), which is not a window in which
    # DELETE_ATTEMPTS retries against a slow Kaggle can finish. A kernel nobody
    # deletes bills accelerator quota to its own ceiling with nobody reading the
    # result, so the only safe moment to notice that the window has gone is
    # before anything is pushed.
    #
    # Nothing bounds the steps that run BEFORE this one -- a checkout, a pip
    # install off a slow index, the harness suite -- so "the job deadline sits
    # above the worst case with room for setup" is an assumption about their
    # duration rather than a property of the run. This measures what is left
    # instead, against the same worst case the job deadline and the reserved
    # quota are derived from.
    #
    # Standing down green is the answer the workflow's FAILURE SEMANTICS give
    # every infrastructure outcome: nothing was learned about the code, and no
    # quota was spent finding that out.
    if not window_fits("after the setup steps"):
        return finish()

    try:
        api = _api()
    except BaseException as exc:  # noqa: BLE001
        if isinstance(exc, KeyboardInterrupt):
            raise
        result["reason"] = f"kaggle auth failed: {type(exc).__name__}"
        return finish()

    # Reclaim anything a previous launcher was killed outright before it
    # could delete. Done BEFORE pushing, so the freed session slots are
    # available to this run -- Kaggle allows only two GPU sessions at once,
    # and an orphan holds one until its ceiling.
    for _slug in sweep_orphans():
        _log(f"reclaimed orphaned kernel {_slug} from a killed launcher")
    # AGAIN, now that authentication is paid for and the next thing is a push.
    # The check above is not enough on its own: authenticate() reaches the
    # network -- with KAGGLE_API_TOKEN, which is the only credential this
    # workflow passes, kaggle 2.2.4 introspects the token over HTTP -- and its
    # only bound is SOCKET_TIMEOUT_SEC. A window that fitted by less than that
    # is gone by the time the first kernel is pushed, and the guard would have
    # granted the push on an answer that had expired.
    if not window_fits("after authenticating"):
        return finish()

    # ANY unforeseen exception from here on still has to delete the kernels.
    # Everything below may have pushed one already, and a kernel this process
    # does not delete bills to its own ceiling with nobody reading it (Kaggle's
    # push-time timeout has been measured not to stop a wedged one, and nothing
    # in the workflow cleans up after this script). Letting an exception out
    # costs GPU quota, not just a report.
    try:
        # ONE deadline for the whole invocation, started BEFORE the first push.
        # A kernel bills from the moment Kaggle accepts it, so the clock that
        # decides when it is deleted must include the time spent pushing the
        # others. Started after the push loop instead, a throttled second push
        # (PUSH_ATTEMPTS attempts at the 600s subprocess ceiling plus backoffs,
        # about 45 minutes) landed on top of --max-wait for the kernel already
        # accepted: 135 minutes of billing against a ceiling that reads as 90.
        # Kaggle's push-time timeout does not cover that gap either (see this
        # file's docstring for the kernel that ignored it for two hours), so the
        # deletion deadline is the only bound there is.
        deadline = time.time() + args.max_wait

        # Push everything first. See this file's docstring: waiting between
        # pushes would serialise sessions Kaggle runs happily in parallel and put
        # an hour between the control leg and the canary leg.
        kernels: list[dict] = []
        # Published BEFORE the loop, as the same list object throughout.
        # release() reads result["kernels"], so a push that dies part-way (the
        # subprocess timeout above used to) must not leave the entries already
        # filed invisible to the cleanup.
        result["kernels"] = kernels
        for notebook in args.notebook:
            _log(f"pushing {notebook} (kernel ceiling {args.kernel_timeout_sec}s)")
            entry = {
                "notebook": notebook,
                "slug": None,
                # Every slug filed, accepted or not: a push that reported an
                # error may still have landed, and this is the only record of
                # what to reconcile against the account afterwards.
                #
                # Published EMPTY, before the push, and filled by push() as it
                # files each slug. Reading the returned list instead meant the
                # entry existed only if push() RETURNED: an exception it does
                # not handle (a decode error on a malformed response, an OSError
                # from the runner) unwound past this line, so release() found no
                # entry for this notebook and the kernel Kaggle may have just
                # accepted was left billing. The per-notebook granularity below
                # was already fixed for the same reason; this is the same bug one
                # level down, and the list being the caller's own object is what
                # closes it for every raise rather than for the ones foreseen.
                "attempted": [],
                "state": None,
                "push_error": None,
            }
            kernels.append(entry)
            pushed = push(
                Path(notebook),
                args.user,
                args.kernel_timeout_sec,
                attempted = entry["attempted"],
            )
            entry["slug"] = pushed.get("slug")
            entry["push_error"] = (
                None if pushed["ok"] else f"{pushed['reason']}: {pushed.get('detail', '')}"
            )
            if pushed["ok"]:
                _log(f"pushed as {pushed['slug']}")
            else:
                _log(f"push failed for {notebook}: {entry['push_error']}")

        live = [k for k in kernels if k["slug"]]
        if not live:
            result["reason"] = "; ".join(k["push_error"] for k in kernels if k["push_error"])
            return finish()
        # Kept for the summary and for anything reading this file's previous
        # single-kernel shape.
        result["slug"] = live[0]["slug"]

        # The deadline above is shared, not one per kernel: they run
        # concurrently, so consuming the ceiling per kernel would let a
        # two-kernel run wait twice its own stated bound.
        for entry in live:
            remaining = max(0, int(deadline - time.time()))
            entry["state"] = wait(api, entry["slug"], args.poll_every, remaining)
            _log(f"{entry['slug']} terminal state: {entry['state']}")
        result["kernel_state"] = ",".join(k["state"] or "?" for k in live)

        # ONE budget for the whole phase, shared by every kernel and started
        # here, for the same reason the polling deadline is shared: the kernels
        # are still billing and release() has not run yet. Per kernel it would
        # scale with the kernel count, which is what the job deadline could not
        # then be derived from. See EVIDENCE_BUDGET_SEC.
        evidence_deadline = time.time() + EVIDENCE_BUDGET_SEC
        for entry in live:
            # One directory per kernel: Kaggle names the executed notebooks
            # after the payloads, so two kernels of a run would otherwise
            # overwrite each other's kernel.log.
            try:
                entry["evidence"] = fetch_evidence(
                    entry["slug"],
                    outdir / entry["slug"].rsplit("/", 1)[-1],
                    deadline = evidence_deadline,
                )
                _log(f"collected {entry['slug']}: {entry['evidence']}")
            except Exception as exc:  # noqa: BLE001
                entry["evidence"] = None
                entry["collect_error"] = type(exc).__name__
                _log(f"could not collect {entry['slug']}: {type(exc).__name__}")

        reports = extract_reports(outdir)
        result["reports"] = reports
        _log(f"extracted {len(reports)} payload report(s) of {args.expect} expected")

        if not reports:
            result["reason"] = (
                f"kernel(s) ended {result['kernel_state']} but produced no payload "
                f"report; nothing was learned about the code under test"
            )
            return finish()

        # A kernel that ended badly but still reported is worth reading: the
        # payload deliberately does not propagate a nonzero exit, so ERROR here
        # usually means the SESSION died (timeout, box OOM, Kaggle side), which
        # is infra unless a report says otherwise.
        failing = [r for r in reports if not r.get("passed")]
        if failing:
            result["verdict"] = "fail"
            result["reason"] = (
                f"{len(failing)} of {len(reports)} payload(s) failed their " f"assertions"
            )
        elif len(reports) < args.expect:
            result["verdict"] = "partial"
            result["reason"] = (
                f"only {len(reports)} of {args.expect} payload(s) reported back "
                f"(kernel state {result['kernel_state']}); the ones that did, "
                f"passed"
            )
        else:
            result["verdict"] = "pass"
            result["reason"] = f"all {len(reports)} payload(s) passed"

        # The kernels are released by finish(), on this path and on every other.
        return finish()
    except BaseException as exc:  # noqa: BLE001
        # An abort is infra by this file's contract: nothing was learned about
        # the code under test, so it must not colour the pull request red.
        # finish() deletes every slug filed so far and still writes
        # launch_result.json, which the summary and the artifact read.
        result["verdict"] = "infra"
        result["reason"] = (
            f"the launcher aborted: {type(exc).__name__}: {str(exc)[:300]}. "
            f"Every kernel it had pushed was deleted on the way out"
        )
        _log(result["reason"])
        code = finish()
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return code


if __name__ == "__main__":
    raise SystemExit(main())
