# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The seeded mirror must contain every streamed turn, or it reports false drift on healthy cells."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.fixture.corpus import RungPlan, Unit  # noqa: E402
from studiobench.runtime.session import CellRunner  # noqa: E402


def _unit(text):
    return Unit(index = 0, kind = "code", reasoning = "", content = text, chars = len(text), sha256 = "0" * 8)


def _plan(follow_ups):
    return RungPlan(
        rung = "10K",
        target_tokens = 10_000,
        target_chars = 40_000,
        seeded_units = [_unit("seeded")],
        streamed_unit = _unit("opening"),
        follow_up_units = [_unit(f"follow{i}") for i in range(follow_ups)],
    )


def _send_turn(*, ran = True, expect_ok = True):
    return {"action": "send_turn", "ran": ran, "expect_ok": expect_ok}


def test_every_turn_that_streamed_is_mirrored():
    plan = _plan(2)
    row = {"actions": [_send_turn(), {"action": "keystroke", "ran": True}, _send_turn()]}
    assert CellRunner._streamed_follow_ups(plan, row) == plan.follow_up_units


def test_a_follow_up_that_never_ran_is_not_mirrored():
    plan = _plan(2)
    row = {"actions": [_send_turn(), _send_turn(ran = False, expect_ok = None)]}
    assert CellRunner._streamed_follow_ups(plan, row) == plan.follow_up_units[:1]


def test_a_send_that_did_not_start_a_reply_is_not_mirrored():
    plan = _plan(2)
    row = {"actions": [_send_turn(expect_ok = False), _send_turn()]}
    assert CellRunner._streamed_follow_ups(plan, row) == plan.follow_up_units[:1]


def test_a_single_turn_rung_mirrors_nothing_extra():
    plan = _plan(0)
    row = {"actions": [_send_turn(ran = False, expect_ok = None)]}
    assert CellRunner._streamed_follow_ups(plan, row) == []


def test_a_cell_with_no_action_rows_mirrors_nothing_extra():
    assert CellRunner._streamed_follow_ups(_plan(2), {}) == []


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
