# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Known-broken entries are strict: a model that starts agreeing fails, so the excuse expires."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAYLOAD = ROOT / "tests" / "kaggle" / "t4_smoke"
sys.path.insert(0, str(PAYLOAD))

from run_t4_smoke import (  # noqa: E402
    KNOWN_BATCHED_GENERATION_BREAKAGE,
    batched_generation_failures,
)


def _record(**over):
    base = {
        "prompt_token_lengths": [13, 13, 14, 15, 13, 13, 15, 16],
        "distinct_lengths": 4,
        "padding_side_observed": "left",
        "padding_side_after": "left",
        "singles": ["x"] * 8,
        "batched": {"2": ["y"] * 8, "4": ["y"] * 8, "8": ["y"] * 8},
        "agrees": {"2": False, "4": False, "8": False},
        "empty_outputs": [],
    }
    base.update(over)
    return base


def test_an_unlisted_model_still_fails_on_disagreement():
    """The rule this whole check exists for is unchanged for everything else."""
    broken = batched_generation_failures(_record(), "unsloth/Qwen3-0.6B")
    assert len(broken) == 3
    assert all("did not reproduce" in f for f in broken)


def test_a_listed_model_does_not_fail_on_the_known_disagreement():
    assert "unsloth/gemma-4-E2B-it" in KNOWN_BATCHED_GENERATION_BREAKAGE
    assert batched_generation_failures(_record(), "unsloth/gemma-4-E2B-it") == []


def test_a_listed_model_that_starts_AGREEING_fails():
    """The strict half, and the reason this is not a mute. Agreement in bf16
    means a kernel or the stack changed, and CI says so instead of carrying a
    stale expectation."""
    agreeing = _record(agrees = {"2": True, "4": True, "8": True})
    broken = batched_generation_failures(agreeing, "unsloth/Qwen3.5-2B")
    assert broken, "a fixed upstream bug must turn the leg red"
    assert "delete the entry" in broken[0]


def test_a_listed_model_still_fails_every_other_rule():
    """The entry excuses ONE claim. A right-padded batch, an unpadded batch or
    an empty output is still a failure, and those are exactly what make the
    disagreement meaningful rather than an artefact."""
    for over, expect in (
        ({"padding_side_after": "right"}, "padding_side_after"),
        ({"distinct_lengths": 1}, "nothing was ever padded"),
        ({"empty_outputs": [3]}, "generated nothing at all"),
        ({"empty_batched_outputs": {"8": [2]}}, "inside the batch"),
        ({"singles": ["x"] * 2}, "largest batch was never actually formed"),
    ):
        broken = batched_generation_failures(_record(**over), "unsloth/Qwen3.5-2B")
        assert any(
            expect in f for f in broken
        ), f"{over} was excused along with the known disagreement"


def test_every_entry_names_the_issue_it_is_waiting_on():
    """An excuse with no issue number is a mute with better manners."""
    for model, reference in KNOWN_BATCHED_GENERATION_BREAKAGE.items():
        assert reference and "#" in reference, f"{model} names no issue"


def test_the_list_is_short_and_explicit():
    """A growing list is the signal that this mechanism is being used to make
    reds go away rather than to carry one filed bug."""
    assert len(KNOWN_BATCHED_GENERATION_BREAKAGE) <= 3, (
        "if this list is growing, the mechanism is being used to silence "
        "failures rather than to hold one filed upstream bug"
    )
