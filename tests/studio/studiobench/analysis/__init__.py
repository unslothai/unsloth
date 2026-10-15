# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Pure functions over a trace: a residual bucket is not a finding; replace it with a named frame."""

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
    """Raised, never downgraded to a smaller number, since a truncated trace reads as the thing
    never ran."""

    def __init__(self, gate: str, detail: str) -> None:
        super().__init__(f"{gate}: {detail}")
        self.gate = gate
        self.detail = detail


# The no-bare-zero convention (INTERFACES.md): `0` means measured zero, never "did not run".


def measured(key: str, value: Any) -> dict:
    """A value that WAS measured, even if it came out zero."""
    return {key: value, f"{key}_attempted": True}


def unmeasured(key: str, reason: str) -> dict:
    """Emits None, never 0; the required reason tells a missing mechanism apart from a missing
    instrument."""
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
    """A numeric 0 or None needs a sibling <key>_attempted; a prose key with nothing to say is omitted."""
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
