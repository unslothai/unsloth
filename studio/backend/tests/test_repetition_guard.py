# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What may and may not be continued when a turn is cut off mid-answer.

The guard's job is narrow: refuse to extend a fragment that is mostly an echo of itself,
because continuing one stitches the echo into the final answer. Everything else, including
text that merely looks repetitive, must still be continued -- a false positive abandons a
real answer halfway, which is worse than the wasted call a false negative costs.

Thresholds follow NousResearch/hermes-agent's `agent/repetition_guard.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference.repetition_guard import MIN_FRAGMENT_LENGTH, is_repetition_dominated


def test_one_line_echoed_until_it_owns_the_fragment():
    assert is_repetition_dominated("I will now show the file to the user.\n" * 40)


def test_an_echo_that_ignores_line_boundaries():
    """The sliding-window pass exists for repeats that do not align to newlines."""
    assert is_repetition_dominated("the same sixty characters over and over again, yes " * 30)


@pytest.mark.parametrize(
    "text",
    [
        # Real code: similar shapes, different content.
        "".join(f"  ctx.lineTo({i * 3}, {i * 7 % 31});\n" for i in range(80)),
        # Prose cut mid-word. Written out rather than multiplied: a repeated sentence is
        # exactly what the guard is for, so building this case with `* 6` would have
        # asserted the opposite of what it claims.
        "The bird falls under gravity and the pipes scroll leftward. Each pipe carries a "
        "gap whose centre drifts as the score climbs, so the difficulty ramps without any "
        "explicit level system. Collision is checked against the bird's bounding circle "
        "rather than its sprite, which keeps near-misses forgiving and reads as fair to a "
        "player. The ground scrolls at the same rate as the pipes to sell the parallax, "
        "while the clouds behind move at a third of it. Scores persist for the session "
        "only, because a best-score that survives a reload invites cheating and the game "
        "is not worth defending that hard. The last thing to wire up is the collision "
        "check, which has to run befo",
        # A heading repeated a few times is ordinary structure, not an echo.
        ("## Features\n\nSomething genuinely different each time here, at length.\n" * 4),
    ],
)
def test_ordinary_text_is_still_continued(text):
    assert is_repetition_dominated(text) is False


@pytest.mark.parametrize("text", ["", "short", "a" * (MIN_FRAGMENT_LENGTH - 1)])
def test_short_fragments_are_never_judged(text):
    """A sentence cut mid-word can trivially repeat tokens and deserves its continuation."""
    assert is_repetition_dominated(text) is False


@pytest.mark.parametrize("value", [None, 42, b"bytes", ["a"]])
def test_it_fails_open_on_anything_it_cannot_read(value):
    assert is_repetition_dominated(value) is False


def test_a_long_run_of_one_character_is_not_mistaken_for_content():
    assert is_repetition_dominated("x" * 5000)
