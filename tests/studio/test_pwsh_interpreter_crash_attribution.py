# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The shared pwsh runner must blame the interpreter for a crash and nothing else.

Backend CI run 32341628757 reported one runner-level pwsh failure mode as 284 independent
Windows-installer regressions across 19 files. tests/_shared/unsloth_pwsh_runner.py exists
to stop that, and its value is entirely in which of these two it does to a given run:

  * a shell killed by a signal produced no verdict  -> retry, then blame the interpreter;
  * a shell that exited normally produced a verdict -> hand it back untouched, right or wrong.

Getting the second one wrong would turn this helper into a way to retry real regressions
into green, which is strictly worse than the bug it fixes. So both directions are executed
here against a real SIGABRT rather than reviewed.
"""

import json
import os
import subprocess
import sys

import pytest

from unsloth_pwsh_runner import PWSH, PwshInterpreterCrash, run_pwsh

# A SIGABRT is forged with Python rather than pwsh: the real trigger is a .NET startup stack overflow we cannot summon
# on demand, and the helper keys on the signal, not on who sent it, so `os.abort()` reproduces exactly the condition CI
# hit.
# PR_SET_DUMPABLE = 0 first, per tests/test_deliberate_crashes_suppress_cores.py. The child still dies of SIGABRT, so
# every verdict below is unchanged; what goes away is apport reading a multi-MB core before the child is reaped, on
# each of the several aborts these tests spend.
# Linux-only, and deliberately not fatal elsewhere: Windows has no CDLL(None) and pipes no core anywhere, so failing to
# arm it there would turn a no-op into a lost test.
_ABORT = [
    sys.executable,
    "-c",
    "import ctypes, os, sys; "
    "sys.platform == 'linux' and ctypes.CDLL(None).prctl(4, 0, 0, 0, 0); "
    "os.abort()",
]
_CLEAN_WRONG_ANSWER = [sys.executable, "-c", "print('WRONG'); raise SystemExit(3)"]


def test_a_signal_death_is_attributed_to_the_interpreter_not_the_assertion():
    """The message must name pwsh dying, and must not read as the script being wrong."""
    with pytest.raises(PwshInterpreterCrash) as excinfo:
        run_pwsh(_ABORT, attempts = 2, capture_output = True, text = True)

    message = str(excinfo.value)
    assert "pwsh itself killed by SIGABRT on all 2 attempts" in message, message
    assert "interpreter dying, not an assertion failing" in message, message


def test_a_clean_run_with_the_wrong_answer_still_fails_with_its_own_message():
    """The load-bearing negative. Exit 3 is a verdict, so it is returned as-is, once."""
    proc = run_pwsh(_CLEAN_WRONG_ANSWER, attempts = 3, capture_output = True, text = True)
    assert proc.returncode == 3
    assert proc.stdout.strip() == "WRONG"

    # ...and `check` still raises the ordinary CalledProcessError a caller expects, so migrating a `subprocess.run(...,
    # check = True)` call site changes no failure text.
    with pytest.raises(subprocess.CalledProcessError):
        run_pwsh(_CLEAN_WRONG_ANSWER, check = True, capture_output = True, text = True)


def test_a_clean_run_is_not_retried():
    """A regression retried three times is a regression that can flake green."""
    calls = []
    real_run = subprocess.run

    def counting_run(argv, **kwargs):
        calls.append(argv)
        return real_run(argv, **kwargs)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(subprocess, "run", counting_run)
        run_pwsh(_CLEAN_WRONG_ANSWER, attempts = 3, capture_output = True, text = True)
    assert len(calls) == 1, f"a normally-exiting run was retried {len(calls)} times"


def test_the_startup_cache_is_redirected_without_disturbing_the_callers_env():
    """The mechanism fix, asserted from the child's own view of its environment.

    Four xdist workers sharing one $HOME share one 83 KB StartupProfileData-NonInteractive,
    and a startup that reads a half-written one dies. Measured at 7/4000 shared against
    0/4000 private, reproducing both crash shapes this repo has seen. The redirect must
    therefore actually reach the child, and must add exactly one variable: a hermetic env
    dict is how several of these tests keep a developer's exported settings from deciding
    an inference assertion, and quietly widening it would break that silently.
    """
    dump = [sys.executable, "-c", "import json,os; print(json.dumps(dict(os.environ)))"]

    marker = "UNSLOTH_PWSH_RUNNER_LEAK_PROBE"
    os.environ[marker] = "ambient"
    try:
        hermetic = {"PATH": "/usr/bin:/bin", "HOME": "/nonexistent"}
        proc = run_pwsh(dump, env = dict(hermetic), capture_output = True, text = True)
        child = json.loads(proc.stdout)
    finally:
        os.environ.pop(marker, None)

    assert "XDG_CACHE_HOME" in child, sorted(child)
    assert child["HOME"] == "/nonexistent", "the caller's hermetic HOME was overwritten"
    assert child["PATH"] == "/usr/bin:/bin"
    # The whole point of a hermetic dict: nothing ambient may reach the child.
    # (CPython's own PEP 538 locale coercion can add LC_CTYPE, which is why this probes for a named leak rather than
    # asserting an exact key set.)
    assert marker not in child, "the ambient environment leaked past a hermetic env dict"

    # env = None must still mean inherit, or every call site that relies on the ambient PATH silently loses it.
    inherited = json.loads(run_pwsh(dump, capture_output = True, text = True).stdout)
    assert inherited["XDG_CACHE_HOME"] == child["XDG_CACHE_HOME"]
    assert inherited.get("PATH") == os.environ.get("PATH")


@pytest.mark.skipif(PWSH is None, reason = "no PowerShell on this platform")
def test_a_real_pwsh_that_answers_correctly_is_untouched():
    """The helper must be transparent on the path every migrated call site takes."""
    proc = run_pwsh(
        [PWSH, "-NoProfile", "-NonInteractive", "-Command", "Write-Output 'ANSWER=7'"],
        capture_output = True,
        text = True,
    )
    assert proc.returncode == 0
    assert "ANSWER=7" in proc.stdout
