# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the A/B interleaver.

The balance check is the one worth the most scrutiny: it is a boolean that has to differ between a
plan where drift cancels and one where it does not, and its first implementation returned True for
both. A single-rep plan is therefore asserted explicitly rather than left to a parametrisation.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.runtime.ab import (  # noqa: E402
    Target,
    interleave,
    order_is_balanced,
    origin_scoped,
)
from studiobench.runtime.types import Cell  # noqa: E402


def _cells(reps: int, rung: str = "1K"):
    return [
        (Cell(cell_id = f"r{rung}.A0.rep{rep}", rung = rung, rung_tokens = 1000, rep = rep), object())
        for rep in range(reps)
    ]


def _targets():
    return [
        Target(label = "base", ref = "main", base_url = "http://a", seeder = None, runner = None),
        Target(label = "treatment", ref = "pr", base_url = "http://b", seeder = None, runner = None),
    ]


def test_both_sides_run_for_every_cell():
    plan = interleave(_cells(2), _targets())
    assert len(plan) == 4
    assert {t.label for t, _c, _p in plan} == {"base", "treatment"}


def test_the_order_flips_between_reps():
    plan = interleave(_cells(2), _targets())
    labels = [t.label for t, _c, _p in plan]
    assert labels == ["base", "treatment", "treatment", "base"]


def test_each_arm_gets_its_own_cell_id():
    """Two arms sharing a cell_id would collide in the payload and in --resume."""
    plan = interleave(_cells(1), _targets())
    ids = [c.cell_id for _t, c, _p in plan]
    assert len(set(ids)) == len(ids)
    assert all(t.label in c.cell_id for t, c, _p in plan)


def test_two_reps_are_balanced():
    assert order_is_balanced(interleave(_cells(2), _targets())) is True


def test_one_rep_is_NOT_balanced():
    """The regression that mattered: base always runs first, so nothing cancels."""
    assert order_is_balanced(interleave(_cells(1), _targets())) is False


def test_three_reps_are_not_balanced():
    assert order_is_balanced(interleave(_cells(3), _targets())) is False


def test_a_single_target_is_never_balanced():
    one = [Target(label = "base", ref = "main", base_url = "http://a", seeder = None, runner = None)]
    assert order_is_balanced(interleave(_cells(2), one)) is False


def test_origin_gate_names_the_exact_origin_and_strips_the_slash():
    script = origin_scoped("http://127.0.0.1:5401/", "doThing();")
    assert '"http://127.0.0.1:5401"' in script
    assert "doThing();" in script
    assert "return" in script
