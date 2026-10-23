# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""pwsh can crash at startup with SIGABRT; retry only signal deaths, and never a normal exit."""

from __future__ import annotations

import atexit
import os
import shutil
import signal
import subprocess
import tempfile

# pwsh aborting prints this with empty stdout and a normal exit, so there is no signal to key on.
PWSH_CRASH_BANNER = "The PowerShell process will exit"

PWSH = shutil.which("pwsh") or shutil.which("powershell")


# xdist workers sharing one $HOME race on pwsh's StartupProfileData cache and crash;
# one cache dir per worker removes the race.
_CACHE_ROOT = None


def _pwsh_cache_dir() -> str:
    """Fresh per-worker cache dir for this pytest session, so a torn cache cannot poison later sessions."""
    global _CACHE_ROOT
    if _CACHE_ROOT is None:
        worker = os.environ.get("PYTEST_XDIST_WORKER", "master")
        _CACHE_ROOT = tempfile.mkdtemp(prefix = f"unsloth-pwsh-cache-{worker}-")
        atexit.register(shutil.rmtree, _CACHE_ROOT, True)
    return _CACHE_ROOT


def pwsh_env(env: dict | None = None) -> dict:
    """Copy of env (default os.environ) with XDG_CACHE_HOME set to the private pwsh cache."""
    env = dict(os.environ if env is None else env)
    env["XDG_CACHE_HOME"] = _pwsh_cache_dir()
    return env


class PwshInterpreterCrash(AssertionError):
    """The interpreter died before producing a verdict. Says nothing about the script."""


def _crash_reason(proc: subprocess.CompletedProcess) -> str | None:
    """Why this run produced no verdict, or None if it produced one."""
    if proc.returncode < 0:
        # Negative returncode means killed by signal (SIGABRT, SIGSEGV, OOM SIGKILL): the script did not finish.
        try:
            name = signal.Signals(-proc.returncode).name
        except ValueError:
            name = f"signal {-proc.returncode}"
        return f"killed by {name}"
    # Streams may be bytes or str depending on the caller.
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


# UTF8Encoding($false): [Text.Encoding]::UTF8 would emit a BOM into stdout.
_UTF8_PROLOGUE = "[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)\n"


_UTF8_ALIASES = frozenset({"utf-8", "utf8", "u8", "utf", "u-8", "cp65001"})


def _agree_on_utf8(argv: list[str], kwargs: dict) -> list[str]:
    """Make pwsh write UTF-8 and decode it as UTF-8, both halves together; the list is never mutated."""
    if not (kwargs.get("text") or kwargs.get("universal_newlines")):
        return argv
    named = kwargs.get("encoding")
    if named is not None and named.lower().replace("_", "-") not in _UTF8_ALIASES:
        return argv
    kwargs["encoding"] = "utf-8"
    # Only exact -Command; prepending to the wrong element runs the prologue as a file path.
    try:
        script = argv.index("-Command") + 1
    except ValueError:
        return argv
    if script >= len(argv) or not isinstance(argv[script], str):
        return argv
    return argv[:script] + [_UTF8_PROLOGUE + argv[script]] + argv[script + 1 :]


def run_pwsh(
    argv: list[str],
    *,
    attempts: int = 3,
    verdict: str | None = None,
    check: bool = False,
    **kwargs,
) -> subprocess.CompletedProcess:
    """Retry a pwsh run killed by a signal, up to attempts times; any normal exit is returned at once."""
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")

    argv = _agree_on_utf8(argv, kwargs)

    # Only redirect pwsh's startup cache; env=None still means inherit.
    kwargs["env"] = pwsh_env(kwargs.get("env"))

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
