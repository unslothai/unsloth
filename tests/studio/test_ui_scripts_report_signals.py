# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The Playwright wrapper scripts must say when a signal killed them.

Observed on windows-latest at roughly one run in twenty-five: the suite prints
"permission-only run passed" and the step then ends with "Process completed with exit
code 143". 143 is 128+SIGTERM, so the script was signalled AFTER its work succeeded, and
nothing in either the step log or the server log records what did it. The server log just
stops mid-request, which is exactly what the script's own cleanup killing it looks like,
so the two cannot be told apart.

This file pins the diagnostic rather than a fix, because the cause is not known yet. A
one-in-twenty-five failure that reports nothing costs a whole run every time it lands and
teaches nothing, and the obvious tidy-up -- deleting a signal handler that "never fires"
-- puts it straight back. Both scripts have the same shape (background server, EXIT trap,
suite as the last command), so both can lose a passing run the same way, and since #9391
they run concurrently on Windows rather than one after another.

`suite_done` is the fact worth capturing: it separates a signal that interrupted the
browser run from one that arrived during teardown, which is the first thing anyone
reading the next occurrence needs to know.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / ".github" / "scripts"
WRAPPERS = ("run-studio-permission-browser.sh", "run-studio-indicator-browser.sh")


def _body(name: str) -> str:
    """Source with comments stripped: a comment must not satisfy these assertions."""
    text = (SCRIPTS / name).read_text(encoding = "utf-8")
    return "\n".join(re.sub(r"(^|\s)#.*$", "", line) for line in text.split("\n"))


@pytest.mark.parametrize("script", WRAPPERS)
def test_the_wrapper_traps_a_terminating_signal(script: str) -> None:
    body = _body(script)
    for signal_name in ("TERM", "INT", "HUP"):
        assert re.search(rf"trap\s+'_on_signal {signal_name}\b", body), (
            f"{script} no longer traps SIG{signal_name}, so a signal that kills it mid-run "
            f"is reported only as a bare exit code with no indication of what happened"
        )


@pytest.mark.parametrize("script", WRAPPERS)
def test_the_report_distinguishes_teardown_from_a_live_run(script: str) -> None:
    """Without suite_done the report cannot answer the only question worth asking."""
    body = _body(script)
    assert "suite_done=0" in body, f"{script} does not initialise suite_done"
    assert re.search(r"^suite_done=1\s*$", body, re.M), (
        f"{script} never sets suite_done=1, so every signal report claims the suite was "
        f"still running -- including the observed case, which was signalled after it passed"
    )
    assert "suite_done" in body.split("_on_signal()")[1][:400], (
        f"{script}'s signal handler does not report suite_done, so the flag is recorded "
        f"and never printed"
    )


@pytest.mark.parametrize("script", WRAPPERS)
def test_the_handler_exits_with_the_signal_status(script: str) -> None:
    body = _body(script)
    assert re.search(r"exit\s+\$\(\(\s*128\s*\+\s*(number|\$?\{?number)", body), (
        f"{script}'s handler does not exit 128+signal, so the status it reports is "
        f"whatever bash happened to pick rather than the signal that caused it"
    )


@pytest.mark.parametrize("script", WRAPPERS)
def test_the_snapshot_does_not_print_command_lines(script: str) -> None:
    """This lands in a public CI log.

    `ps -o ...,args` would quote every running command line, and one of those can carry a
    token that ::add-mask:: never saw. Process names answer the question the snapshot is
    there for without that risk.
    """
    body = _body(script)
    assert "pid,ppid,comm" in body, f"{script}'s process snapshot is not limited to names"
    assert not re.search(
        r"ps\s+-o\s+[\w,]*args", body
    ), f"{script} snapshots full command lines into a public log"


def test_the_guard_reads_real_files() -> None:
    for name in WRAPPERS:
        assert (SCRIPTS / name).is_file(), name
        assert len(_body(name)) > 400, name
