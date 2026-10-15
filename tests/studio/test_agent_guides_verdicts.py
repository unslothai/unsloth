# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""agent-guides-drive.sh must not blame the recipe for a failure that is not the recipe's."""

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
    """run_timed must not promise judgement on assertions, since three callers treat a cap as fatal."""
    body = _block("run_timed")
    assert "judging the turn on its assertions" not in body, (
        "run_timed still promises the caller will judge on assertions; three "
        "callers treat a timeout as fatal instead, and the reader sees both"
    )
    # It must still say the cap was hit, or a stall reads as an ordinary non-zero exit.
    assert "did not exit within" in body


def test_a_timed_out_connection_is_not_reported_as_recipe_drift() -> None:
    """A timed-out turn means the launch command was fine, so guide_fail is the wrong reporter."""
    source = _source()
    connection = source[source.index("\n  connection)") :]
    connection = connection[: connection.index("\n  file-edit)")]

    timed_out = re.search(
        r'if \[ "\$\{TIMED_OUT:-0\}" = 1 \](.*?); then(.*?)\n    fi', connection, re.DOTALL
    )
    assert timed_out, "the connection case no longer distinguishes a timeout from a non-zero exit"
    condition, branch = timed_out.group(1), timed_out.group(2)

    assert "guide_fail" not in branch, (
        "a timed-out connection still goes through guide_fail, which states that "
        "the documented flow drifted -- the one thing a cap does not show"
    )
    assert "not implicated" in branch, "the message does not clear the recipe it used to blame"
    # Still fatal: a recipe that printed a banner then blocked on a prompt must not pass.
    assert "exit 1" in branch, (
        "a timed-out connection must stay fatal; assert_reply cannot tell a "
        "finished reply from a startup banner"
    )
    # The only allowed narrowing is the agent's end-of-run marker, which a banner cannot print.
    assert condition.strip() in ("", '&& [ "${TURN_DONE:-0}" != 1 ]'), (
        "the connection cap may only be narrowed by TURN_DONE, which run_timed "
        f"sets solely on an end-of-run marker; found {condition.strip()!r}"
    )


def test_only_a_marker_can_excuse_a_connection_cap() -> None:
    """TURN_DONE is set only by the agent's own end-of-run marker, or any hang would waive the guard."""
    body = _block("run_timed")
    assert body.count("TURN_DONE=1") == body.count(
        'grep -qF -- "$TURN_DONE_RE"'
    ), "run_timed sets TURN_DONE somewhere that does not first match the marker"
    source = _source()
    declared = re.findall(r"TURN_DONE_RE='([^']*)'", source)
    assert declared == [
        "ended with stopReason="
    ], f"unexpected end-of-run markers declared: {declared}"


def test_a_non_zero_exit_is_still_drift() -> None:
    """A launch command that exits non-zero on its own is real drift and must still be reported."""
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
