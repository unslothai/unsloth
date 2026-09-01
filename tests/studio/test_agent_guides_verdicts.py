# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""
agent-guides-drive.sh must not blame the recipe for a failure that is not the
recipe's.

The whole value of this job is the sentence it prints when it goes red. It exists
to catch "guide drift" -- the documented flow in `unsloth start` no longer works
-- and a red run that says "drift" costs whoever reads it a trip through
start.py. So a wrong attribution is not cosmetic here, it is the entire output.

Two of those were live on 2026-08-19, in one 13-minute failure:

  ##[warning] ... judging the turn on its assertions instead of calling it guide drift.
  ##[error]  [guide drift] agent=codex: the documented launch command exited
             non-zero (rc=124) ... so the documented flow in start.py drifted.

The warning promised something the next line contradicted, because run_timed
spoke for callers that treat a cap as fatal on purpose. And the error blamed
start.py for a turn whose transcript showed the CLI launching perfectly (correct
provider, correct model) and then sitting on `ERROR: Reconnecting... 1/5` for the
full 600s. Nothing had drifted; the model server never answered.

These are static checks. The script needs five agent CLIs and a live model server
to run, so what can be pinned without them is the shape of what it says.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / ".github" / "scripts" / "agent-guides-drive.sh"


def _source() -> str:
    return SCRIPT.read_text(encoding = "utf-8")


def _block(name: str) -> str:
    """A shell function's body, by brace depth."""
    source = _source()
    start = source.index(f"{name}() {{")
    depth, i = 0, start
    while i < len(source):
        if source[i] == "{":
            depth += 1
        elif source[i] == "}":
            depth -= 1
            if depth == 0:
                return source[start : i + 1]
        i += 1
    raise AssertionError(f"{name}() never closes")


def test_run_timed_does_not_speak_for_its_callers() -> None:
    """
    `connection`, `resume` and `attribution-ab` have no assertion that can rescue
    a partial turn and treat a cap as fatal, deliberately. A blanket "judging the
    turn on its assertions" from inside run_timed is therefore false for exactly
    the callers most likely to hit it, and it is printed immediately above the
    error that contradicts it.
    """
    body = _block("run_timed")
    assert "judging the turn on its assertions" not in body, (
        "run_timed still promises the caller will judge on assertions; three "
        "callers treat a timeout as fatal instead, and the reader sees both"
    )
    # It must still say the cap was hit -- silence here is how a stall reads as an ordinary non-zero exit.
    assert "did not exit within" in body


def test_a_timed_out_connection_is_not_reported_as_recipe_drift() -> None:
    """
    A cap means the launch command was fine and the turn never came back. The
    recipe is the one thing that is NOT implicated, so `guide_fail` (which
    asserts the documented flow drifted) is the wrong reporter for it.
    """
    source = _source()
    connection = source[source.index("\n  connection)") :]
    connection = connection[: connection.index("\n  file-edit)")]

    timed_out = re.search(
        r'if \[ "\$\{TIMED_OUT:-0\}" = 1 \]; then(.*?)\n    fi', connection, re.DOTALL
    )
    assert timed_out, "the connection case no longer distinguishes a timeout from a non-zero exit"
    branch = timed_out.group(1)

    assert "guide_fail" not in branch, (
        "a timed-out connection still goes through guide_fail, which states that "
        "the documented flow drifted -- the one thing a cap does not show"
    )
    assert "not implicated" in branch, "the message does not clear the recipe it used to blame"
    # Still fatal.
    # Still fatal. Waiving a cap here would report "connection OK" for a recipe that printed a banner and then blocked
    assert "exit 1" in branch, (
        "a timed-out connection must stay fatal; assert_reply cannot tell a "
        "finished reply from a startup banner"
    )


def test_a_non_zero_exit_is_still_drift() -> None:
    """
    The narrowing must not swallow the case guide_fail is right about: a launch
    command that exits non-zero on its own really is the documented flow failing.
    """
    source = _source()
    connection = source[source.index("\n  connection)") :]
    connection = connection[: connection.index("\n  file-edit)")]
    assert re.search(
        r'\[ "\$rc" -eq 0 \] \|\| guide_fail', connection
    ), "a non-zero exit from the launch command no longer reports drift"


def test_guide_fail_still_names_the_recipe() -> None:
    """It is the right message for the case it is now reserved for."""
    body = _block("guide_fail")
    assert "guide drift" in body and "CONNECT_REF" in body


def test_opencode_v2_guide_moves_standalone_after_run() -> None:
    """Appending run to the bare V2 recipe must keep standalone a run option."""
    body = _block("invoke_via_connect")
    assert '[[ "$cmd" == *" --standalone" ]]' in body
    assert 'set -- run --standalone "${@:2}"' in body
