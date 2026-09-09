# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""studiobench analysis: turn a captured trace into NAMED CALL FRAMES.

Everything in this package is a pure function over a trace file. Nothing here
launches a browser, so it can be unit tested against a checked-in trace and it
keeps working when the harness layers change underneath it.

The organising principle is that a RESIDUAL IS NOT A FINDING. A bucket sum the
renderer computed for its own accounting ("36% of TaskDuration is unnamed
script") is the shape of our ignorance, not a bottleneck. Every module here
exists to replace one such bucket with a call frame, a count, and an exponent.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "CellFailure",
    "classify",
    "cpuprofile",
    "fit",
    "measured",
    "merge",
    "oracles",
    "symbols",
    "traceparse",
    "unmeasured",
]


class CellFailure(RuntimeError):
    """A measurement cell is unusable and must not be quoted.

    Raised, never swallowed. A cell that fails an integrity gate is reported as
    a failure with its reason; it is never silently downgraded into a smaller
    number, because a silently truncated trace reads exactly like "the expensive
    thing did not happen".
    """

    def __init__(self, gate: str, detail: str) -> None:
        super().__init__(f"{gate}: {detail}")
        self.gate = gate
        self.detail = detail


# The no-bare-zero convention (INTERFACES.md, "Rules that apply to every dict"). `0` means
# "measured, and it was zero" and never "did not run": a frame budget of 0.00 ms reads as a fast
# app and is indistinguishable from an instrument that never attached. Every numeric key these
# layers emit goes through one of the two helpers below, so the distinction is structural.


# ── the no-bare-zero convention (INTERFACES.md, "Rules that apply to every dict") ──
def measured(key: str, value: Any) -> dict:
    """A value that WAS measured, even if it came out zero."""
    return {key: value, f"{key}_attempted": True}


def unmeasured(key: str, reason: str) -> dict:
    """A value that could not be measured, and why.

    Emits `None`, never `0`. The reason is required and is not optional
    politeness: it is the only thing that tells a reader whether the number is
    missing because the mechanism did not fire or because the instrument did not
    run, and those imply opposite conclusions.
    """
    if not reason:
        raise ValueError(f"unmeasured({key!r}) requires a reason")
    return {key: None, f"{key}_attempted": False, f"{key}_reason": reason}


def merge(*fragments: dict) -> dict:
    """Combine helper fragments, refusing silent key collisions."""
    out: dict = {}
    for frag in fragments:
        for k, v in frag.items():
            if k in out and out[k] != v:
                raise ValueError(f"conflicting values for {k!r}: {out[k]!r} then {v!r}")
            out[k] = v
    return out


def assert_no_bare_zero(payload: dict, path: str = "payload") -> None:
    """Check a dict obeys the convention before it crosses a layer boundary.

    A numeric key equal to 0 (or a key set to None) must carry a sibling
    `<key>_attempted`. Used in tests and cheap enough to call on real payloads.

    Note the corollary for PROSE keys: a `reason` or `note` with nothing to say
    must be OMITTED, not set to `None`. `None` is reserved for a quantity that
    could not be measured, and overloading it for "no comment" makes the two
    indistinguishable at exactly the point where the difference matters.
    """
    for k, v in payload.items():
        if k.endswith(("_attempted", "_reason")):
            continue
        if isinstance(v, dict):
            assert_no_bare_zero(v, f"{path}.{k}")
            continue
        is_zero = isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0
        if (is_zero or v is None) and f"{k}_attempted" not in payload:
            raise CellFailure(
                "bare_zero",
                f"{path}.{k} is {v!r} with no sibling {k}_attempted. A bare zero cannot "
                "be told apart from an instrument that never ran.",
            )
