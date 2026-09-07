# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""One `pwsh` runner for every test that shells out to PowerShell, so an interpreter
that dies is reported as an interpreter that died.

Backend CI run 32341628757 on `1c3dde199` finished `284 failed, 8498 passed` and every
one of the 284 was a `pwsh` subprocess ending `died with <Signals.SIGABRT: 6>`, spread
over 19 files that all read as Windows-installer regressions. They were not. The three
tests in that run that assert on `returncode` instead of passing `check = True` kept
pwsh's own stderr, and it says:

    AssertionError: "cd '/tmp/.../me'" failed: 'Stack overflow.\\n'

`Stack overflow.` is the .NET runtime's failfast: the CLR cannot unwind a blown stack, so
it prints that one line and calls `abort()`, which is the SIGABRT. It is a crash *of the
interpreter at startup*, matching PowerShell/PowerShell#24461 ("Stack overflow error when
starting pwsh with -Command"), and it is independent of what we asked pwsh to run -- the
script that produced the line above is a bare `cd`, while its neighbours in the same run
are 60-line installer excerpts.

The reason this is worth a shared module rather than 19 private copies is attribution, not
tidiness. `subprocess.run(..., check = True)` renders a dead interpreter as
`CalledProcessError` carrying the whole script, which reads exactly like the script having
failed, so a runner-level crash costs a full log download and a per-file triage before
anyone can see it was never our code. The crash is also not rare enough to ignore: of the
1409 tests in those 19 files roughly 20% died, interleaved with passes throughout the
12-minute run, which is a per-process coin flip rather than one bad moment.

Two rules, both load-bearing:

  * **A signal is not a verdict.** A shell killed by a signal did not finish its script, so
    it returned no answer either way. That is what makes retrying it honest -- there is no
    failure being papered over yet -- and it is why the crash test is the signal itself
    rather than a message: it needs no per-call-site marker and cannot misread output.
  * **A normal exit is returned untouched, first time, whatever its code.** A pwsh that runs
    to completion and gives the WRONG answer is a real regression and must fail with its own
    message. Nothing here retries it, and nothing here rewrites it.

Generalised from `_run_pwsh` in tests/studio/test_install_phase_timing.py, which handles a
second, signal-free shape: pwsh printing its "The PowerShell process will exit" banner and
exiting normally with nothing on stdout. That one cannot be spotted from the exit status, so
it stays a text match, and a caller that can name a marker its script prints on success can
pass `verdict = ` to say "this run reached a conclusion" without relying on either.
"""

from __future__ import annotations

import atexit
import os
import shutil
import signal
import subprocess
import tempfile

# pwsh aborting mid-flight prints this and leaves stdout empty while still exiting through the normal path, so unlike
# the SIGABRT case there is no signal to key on.
PWSH_CRASH_BANNER = "The PowerShell process will exit"

# Resolved once. `None` on a box with no PowerShell, which is what the skipif guards read.
PWSH = shutil.which("pwsh") or shutil.which("powershell")


# --------------------------------------------------------------------------------------
# Why the crash happens, and the one-line change that stops it
# --------------------------------------------------------------------------------------
# Every `-NonInteractive` startup reads and rewrites an ~83 KB
# $XDG_CACHE_HOME/powershell/StartupProfileData-NonInteractive, and XDG_CACHE_HOME defaults
# to $HOME/.cache. Under `-n 4` all four xdist workers share one $HOME, so the whole job's
# pwsh processes race on that single file, and a startup that deserialises a half-written
# one dies before it reaches our script.
#
# Measured on this repo's suite shape, 4000 startups per arm:
#
#   shared cache dir   7/4000 died -- returncodes {-11: 3, -6: 4}, stderr 'Stack overflow.'
#                      and 'The PowerShell process will exit. Unhandled exception.
#                      System.IO.FileLoadException: The given assembly name ...'
#   private cache dirs 0/4000
#
# That reproduces BOTH crash shapes this repo has hit -- the SIGABRT that made run
# 32341628757 red and the exit-banner that `_run_pwsh` in test_install_phase_timing.py was
# written for -- and the FileLoadException names the torn cache outright. It is also the
# independent confirmation from CI itself: of the pwsh-heavy test files in that run, exactly
# one had zero failures, tests/test_windows_amd_gpu_scan_fallback.py, and it is the only one
# that hands its child a private HOME (`{"PATH": ..., "HOME": str(tmp_path)}`) and so never
# joined the race, across ~80 startups where a 20% rate predicts ~16 failures.
#
# So the fix is to stop sharing the file rather than to serialise access to it: one cache
# directory per xdist worker. Workers run their tests one at a time, so within a worker the
# startups are sequential and the cache still does its job warm; across workers the
# directories are disjoint and there is nothing left to race on. This is why the runner does
# not bound pwsh concurrency with a lock and does not ask for `-n 4` to be given up: the
# contended resource is removed, not rationed.
_CACHE_ROOT = None


def _pwsh_cache_dir() -> str:
    """A cache directory private to this xdist worker, fresh for this pytest session.

    Fresh rather than a stable path under TMPDIR: a cache torn by a previous run would
    otherwise persist and poison every later session on the same box, which is the failure
    this whole module exists to remove.
    """
    global _CACHE_ROOT
    if _CACHE_ROOT is None:
        worker = os.environ.get("PYTEST_XDIST_WORKER", "master")
        _CACHE_ROOT = tempfile.mkdtemp(prefix = f"unsloth-pwsh-cache-{worker}-")
        atexit.register(shutil.rmtree, _CACHE_ROOT, True)
    return _CACHE_ROOT


class PwshInterpreterCrash(AssertionError):
    """The interpreter died before producing a verdict. Says nothing about the script."""


def _crash_reason(proc: subprocess.CompletedProcess) -> str | None:
    """Why this run produced no verdict, or None if it produced one."""
    if proc.returncode < 0:
        # Popen reports "killed by signal N" as -N. .NET's stack-overflow failfast is SIGABRT; a SIGSEGV or a SIGKILL
        # from the OOM killer would land here too, and all three mean the same thing to us: the script did not run to
        # its end.
        try:
            name = signal.Signals(-proc.returncode).name
        except ValueError:
            name = f"signal {-proc.returncode}"
        return f"killed by {name}"
    # Only inspectable when the caller captured the streams; a call site that streams to the console gets the signal
    # check alone, which is the case that actually bit CI. The byte-level tests capture without `text = True`, so the
    # banner is searched for in whichever form the caller asked for rather than assuming str.
    captured = [stream for stream in (proc.stdout, proc.stderr) if stream]
    if any(isinstance(stream, bytes) for stream in captured):
        streams = b"".join(
            stream if isinstance(stream, bytes) else stream.encode("utf-8", errors = "replace")
            for stream in captured
        ).decode("utf-8", errors = "replace")
    else:
        streams = "".join(captured)
    if PWSH_CRASH_BANNER in streams:
        return "self-aborted with the PowerShell crash banner"
    return None


def run_pwsh(
    argv: list[str],
    *,
    attempts: int = 3,
    verdict: str | None = None,
    check: bool = False,
    **kwargs,
) -> subprocess.CompletedProcess:
    """`subprocess.run(argv)`, retrying only a run that crashed without answering.

    `argv` is the complete command the call site already built, pwsh path included, so
    migrating a test is a one-word change and no invocation flags move.

    `attempts` defaults to 3 because the observed crash is an independent per-process event
    at roughly p = 0.2: one retry leaves 4% of invocations still red, two leaves 0.8%, which
    across ~1400 tests is the difference between a red run most days and one every few
    months. Retries are consecutive and unslept -- the trigger is process startup, not a
    resource that frees up over time.

    `verdict`, when given, is a marker the script prints once it has reached a conclusion.
    Its presence in stdout ends the loop immediately even if the run also looks crashy,
    which keeps a script that legitimately mentions the banner from being retried.

    `check` is honoured after the loop, not passed down, because `subprocess.run` would
    raise `CalledProcessError` on the crashing attempt and lose the retry.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")

    # Redirect only pwsh's own startup cache, leaving every other variable as the call site meant it: `env = None`
    # still means "inherit", and a hermetic env dict still gets exactly the keys it listed plus this one.
    env = kwargs.get("env")
    env = dict(os.environ if env is None else env)
    env["XDG_CACHE_HOME"] = _pwsh_cache_dir()
    kwargs["env"] = env

    proc = None
    reason = None
    for _ in range(attempts):
        proc = subprocess.run(argv, **kwargs)
        if verdict is not None and verdict in (proc.stdout or ""):
            break
        reason = _crash_reason(proc)
        if reason is None:
            break
    else:
        raise PwshInterpreterCrash(
            f"pwsh itself {reason} on all {attempts} attempts without running the script to "
            f"completion, so this run says nothing about what the script does -- it is the "
            f"interpreter dying, not an assertion failing. A `Stack overflow.` on stderr is "
            f".NET's failfast at pwsh startup (PowerShell/PowerShell#24461) and is a property "
            f"of the runner, not of this repository.\n"
            f"argv: {argv!r}\n"
            f"returncode: {proc.returncode}\n"
            f"stdout: {proc.stdout!r}\n"
            f"stderr: {proc.stderr!r}"
        )

    if check:
        proc.check_returncode()
    return proc
