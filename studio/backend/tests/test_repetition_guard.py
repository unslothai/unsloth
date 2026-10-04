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


_A_RULE = "-" * 80


def test_a_horizontal_rule_does_not_condemn_the_answer_around_it():
    """Occurrences have to be counted without overlap, or one run counts as many.

    An 80-character rule contains 21 identical 60-character windows, which alone clears
    `_MIN_REPEAT_COUNT`. The answer it divides is real work, and abandoning it mid-stream
    over one line of markdown is the failure this guard exists to avoid causing.
    """
    text = (
        "Here is the plan for the game loop, written out before any code so the shape is "
        "clear.\n" + _A_RULE + "\nThe bird accelerates downward each frame and the pipes "
        "scroll leftward at a constant rate. Collision runs against a bounding circle, "
        "which keeps near-misses forgiving. Score increments as the bird passes a pipe's "
        "trailing edge, and the difficulty ramps by narrowing the gap rather than by "
        "raising the speed, so the controls stay learnable throughout the run.\n"
    )

    assert len(text) > MIN_FRAGMENT_LENGTH
    assert is_repetition_dominated(text) is False


def test_a_genuine_echo_of_the_same_window_is_still_caught():
    """The non-overlap rule must not cost the guard the case it was ported for."""
    text = "Let me check the file once more to be sure of its contents.\n" * 40

    assert is_repetition_dominated(text) is True


def test_the_scan_does_not_grow_with_the_length_of_the_fragment():
    """The scan kept one 60-character slice per starting offset.

    An 800,000-character fragment therefore held roughly 180 MB of substrings alive, on a
    path whose only job is to decide whether to send one more continuation. What is
    asserted is the SHAPE, not a byte count: doubling the fragment must not double the
    cost. Before the bound it did, exactly.
    """
    import tracemalloc

    from core.inference import repetition_guard

    def peak_for(lines: int) -> int:
        text = "".join(f"unique line number {index:07d} of prose\n" for index in range(lines))
        tracemalloc.start()
        try:
            assert repetition_guard.is_repetition_dominated(text) is False
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    small = peak_for(20_000)
    large = peak_for(40_000)

    assert large < small * 1.3, f"twice the fragment cost {large} bytes against {small}"


def test_the_window_cap_cannot_turn_a_clean_fragment_into_a_refusal():
    """Past the cap new windows are ignored, which can only fail OPEN. Proven, not assumed."""
    from core.inference import repetition_guard

    text = "".join(f"unique line number {index:07d} of prose\n" for index in range(20_000))

    assert repetition_guard.is_repetition_dominated(text) is False
    assert repetition_guard._MAX_TRACKED_WINDOWS < len(text)


def test_two_different_windows_sharing_a_hash_are_not_counted_as_a_repeat():
    """Counting by hash is only safe if a collision cannot stand in for a repeat."""
    from core.inference import repetition_guard

    collide = {}
    real_hash = hash

    def everything_collides(value):
        collide[value] = True
        return 1234

    original = repetition_guard.hash if hasattr(repetition_guard, "hash") else None
    repetition_guard.hash = everything_collides
    try:
        text = "".join(f"unique line number {index:07d} of prose\n" for index in range(200))
        assert repetition_guard.is_repetition_dominated(text) is False
    finally:
        if original is None:
            del repetition_guard.hash
        else:
            repetition_guard.hash = original
    assert real_hash("x") == real_hash("x")
