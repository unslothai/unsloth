# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""A confirmed upstream bug must not block a leg, and must not outlive itself.

Batched greedy generation on `unsloth/gemma-4-E2B-it` and `unsloth/Qwen3.5-2B`
disagrees at sizes 2, 4 and 8 with real left padding and demonstrably distinct
prompt lengths, on both repeats, on a current stack. It is filed as unsloth
#9708 and it is not this repo's to fix. Leaving it as a hard failure means
those two legs put a red in front of every PR for a bug no reader can act on,
which is how a check gets switched off.

So the entry is a STRICT expectation rather than a mute, and the difference is
the whole point of this file. A model listed as broken that starts AGREEING
FAILS, with a message saying to delete the entry. An excuse that can only ever
excuse is indistinguishable from deleted coverage, and it outlives the bug it
was written for by years.

Everything else stays live for those models. The padding side, the distinct
lengths, the empty-output check and the batch-size floor are what make the
disagreement a real finding rather than an unpadded batch, and none of them is
touched by the entry.
"""

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
    """The strict half, and the reason this is not a mute. The day #9708 is
    fixed, CI says so instead of carrying a stale excuse."""
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
