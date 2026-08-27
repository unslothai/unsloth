# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What issue #9861's own numbers say, before any gate is written against them.

This file asserts nothing about our code. It pins the shape of the evidence, so
that a later cost gate is tuned against a fixed target rather than against a
table someone re-reads and re-summarises each time. If a transcription slips,
these fail here rather than silently moving the goalposts for the gate tests.
"""

from __future__ import annotations

import pytest

from .reported_offload_cells import REPORTED_CELLS, WINNING_CELLS


def test_every_published_planned_cell_is_present():
    assert len(REPORTED_CELLS) == 31
    assert all(0 < cell.blocks_spilled <= cell.blocks_total for cell in REPORTED_CELLS)


def test_the_planner_loses_almost_every_cell_at_the_published_workload():
    losses = [cell for cell in REPORTED_CELLS if cell.speedup() <= 1.0]
    assert len(losses) == 29


def test_two_cells_are_worth_keeping():
    """A gate that declined all 31 would look excellent and still be wrong."""
    labels = {cell.label for cell in WINNING_CELLS}
    assert labels == {"Llama-3.3-70B Q4 @ Ada 38016 MiB", "Qwen3.6-35B-A3B Q4 @ 3090 9344 MiB"}
    best = max(WINNING_CELLS, key = lambda cell: cell.speedup())
    assert best.label == "Llama-3.3-70B Q4 @ Ada 38016 MiB"
    assert best.speedup() == pytest.approx(1.41, abs = 0.01)


def test_most_cells_no_generation_length_can_rescue():
    """The short-sequence caveat on the issue does not reach these.

    It argues generation was measured where cache residency pays least, so the
    generation column understates the planner. That only helps a cell whose
    generation is at least faster to begin with. In 27 of 31 the planner is
    slower per output token AND slower to prefill, so a longer decode widens the
    gap instead of closing it.
    """
    hopeless = [c for c in REPORTED_CELLS if c.breakeven_generated_tokens is None]
    assert len(hopeless) == 27


def test_the_surviving_cells_break_even_at_plausible_lengths():
    breakevens = {
        cell.label: cell.breakeven_generated_tokens
        for cell in REPORTED_CELLS
        if cell.breakeven_generated_tokens is not None
    }
    assert len(breakevens) == 4
    # The 70B/Ada win repays its prefill loss almost immediately, which is why
    # it survives even a prefill-weighted workload; the other three need a
    # decode longer than the 128 tokens that were actually measured.
    assert breakevens["Llama-3.3-70B Q4 @ Ada 38016 MiB"] == pytest.approx(18, abs = 1)
    assert breakevens["Qwen3.6-35B-A3B Q4 @ 3090 9344 MiB"] == pytest.approx(124, abs = 2)
    assert breakevens["Llama-3.3-70B Q4 @ 3090 24448 MiB"] == pytest.approx(608, abs = 5)
    assert breakevens["Qwen3.6-35B-A3B Q4 @ Ada 16768 MiB"] == pytest.approx(2090, abs = 20)


def test_the_worst_cells_are_the_ones_that_barely_needed_to_spill():
    """The damage is concentrated where the model nearly fit.

    Both Qwen3-8B cells had 11008 MiB free for roughly 5 GB of weights. The
    planner spilled 22 and 29 of 36 blocks anyway and landed at 0.19x and 0.21x,
    the two worst results in the table. Spilling when the alternative was full
    residency is the expensive mistake, not spilling too little.
    """
    worst = sorted(REPORTED_CELLS, key = lambda cell: cell.speedup())[:2]
    assert {cell.model for cell in worst} == {"Qwen3-8B Q4"}
    assert all(cell.speedup() < 0.25 for cell in worst)
