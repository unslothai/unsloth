#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Run a build under a deadline and, when it passes, stop the WHOLE process tree.

The Linux legs use coreutils `timeout --signal=INT --kill-after=120`. That does not carry over to
the Windows legs: Git Bash's `timeout` can only end the one native process it started, and a
setup.py build is python -> ninja -> ccache -> nvcc -> cicc/ptxas. The orphans keep the step's
output pipe open, so the step does not end at the deadline, the job runs into GitHub's six-hour
limit, and the steps that save a warm slice's partial ccache never get to run. That cache is
the point of the deadline, so the tree is ended with `taskkill /T /F` instead.

On POSIX it mirrors the coreutils behaviour the Linux legs already rely on: SIGINT to the process
group, then SIGKILL after a grace period. Exit status 124 on a timeout, like `timeout`.

Usage: prebuilt_wheels_timeout.py <duration, e.g. 300m, 90s, 2h> -- <command> [args...]
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

GRACE_SECONDS = 120
UNITS = {"s": 1, "m": 60, "h": 3600}


def parse_duration(text: str) -> float:
    text = text.strip()
    unit = text[-1:] if text[-1:] in UNITS else "s"
    number = text[:-1] if text[-1:] in UNITS else text
    try:
        value = float(number)
    except ValueError:
        raise SystemExit(f"not a duration: {text!r}") from None
    if value <= 0:
        raise SystemExit(f"not a duration: {text!r}")
    return value * UNITS[unit]


def kill_tree(process: subprocess.Popen) -> None:
    if os.name == "nt":
        # /T takes every descendant, /F because nvcc ignores a polite close.
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(process.pid)],
            stdout = subprocess.DEVNULL,
            stderr = subprocess.DEVNULL,
            check = False,
        )
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout = GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run(seconds: float, command: list[str]) -> int:
    if os.name == "nt":
        process = subprocess.Popen(command, creationflags = subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        process = subprocess.Popen(command, start_new_session = True)
    deadline = time.monotonic() + seconds
    try:
        return process.wait(timeout = max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        print(
            f"::warning::deadline of {seconds:.0f}s reached; ending the process tree",
            flush = True,
        )
        kill_tree(process)
        try:
            process.wait(timeout = GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        return 124


def main(argv: list[str]) -> int:
    if len(argv) < 4 or argv[2] != "--":
        print(__doc__.strip().splitlines()[-1], file = sys.stderr)
        return 2
    return run(parse_duration(argv[1]), argv[3:])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
