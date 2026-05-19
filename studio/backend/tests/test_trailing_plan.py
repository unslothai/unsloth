# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the mid-plan auto-continue regexes and ``_trailing_plan_hit``.

The trailing-plan detector lives in ``core.inference.llama_cpp`` and decides
whether the model just stopped mid-plan (and therefore deserves a neutral
``Continue.`` re-prompt). False positives cost real tool calls and latency,
so the patterns get explicit coverage here.

Bug history pinned by these tests:

* ``"If you need anything else, let me know."`` matched ``_TRAILING_PLAN_INTENT``
  before the negative lookahead landed.
* ``"Here's my plan:\\n- a\\n- b\\n\\nDone, that should work."`` matched
  ``_TRAILING_PLAN_LIST`` before the regex was switched off the ``m`` flag
  and re-anchored with ``\\Z``.
"""

from __future__ import annotations

import pytest

from core.inference.llama_cpp import (
    _TRAILING_PLAN_COLON,
    _TRAILING_PLAN_INTENT,
    _TRAILING_PLAN_LIST,
    _trailing_plan_hit,
)


# ----- _TRAILING_PLAN_INTENT -------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        # Closing phrases must NOT match (regression: "let me know")
        ("If you need anything else, let me know.", False),
        ("Let me know if I can help further.", False),
        ("let me know!", False),
        # Genuine mid-plan intent SHOULD match
        ("Let me clone the repo.", True),
        ("Let me check the file.", True),
        ("Now let me run the tests.", True),
        ("I'll now run the analyzer.", True),
        ("I’ll now run the analyzer.", True),  # curly apostrophe
        ("I will now begin.", True),
        # Unrelated trailing text must NOT match
        ("Hello world.", False),
        ("The answer is 42.", False),
    ],
)
def test_trailing_plan_intent(text: str, expected: bool) -> None:
    assert bool(_TRAILING_PLAN_INTENT.search(text)) is expected


# ----- _TRAILING_PLAN_LIST ---------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        # List block at end of buffer SHOULD match
        ("Let me do this:\n- step one\n- step two\n", True),
        ("Here's my plan:\n1. one\n2. two\n", True),
        ("Here's my plan:\n1. one\n2. two\n   \n", True),  # trailing whitespace
        # Unicode bullet at end of buffer SHOULD match
        ("Let me try:\n• first\n• second\n", True),
        # List followed by a closing sentence MUST NOT match (regression:
        # the `m` flag in `(?ims)` previously let `\s*$` match end-of-line)
        (
            "Here's my plan:\n- step one\n- step two\n\nDone, hope that helps.",
            False,
        ),
        (
            "Let me walk through it:\n1. first\n2. second\n\nThat's everything.",
            False,
        ),
        # Single-item numbered list followed by closing prose MUST NOT match
        (
            "Let me explain:\n1. The function returns 42.\n\nThat's the answer.",
            False,
        ),
        # List embedded mid-text (not trailing) MUST NOT match
        (
            "Here's my plan:\n- step one\n- step two\nNow the conclusion follows.",
            False,
        ),
    ],
)
def test_trailing_plan_list(text: str, expected: bool) -> None:
    assert bool(_TRAILING_PLAN_LIST.search(text)) is expected


# ----- _TRAILING_PLAN_COLON --------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        # Bare trailing colon SHOULD match
        ("Let me check the repo:", True),
        ("I'll now look at this:", True),
        # Colon mid-sentence MUST NOT match
        ("Let me check this: it should work fine.", False),
        # Colon not in an intent-cue clause MUST NOT match
        ("The result is:", False),
    ],
)
def test_trailing_plan_colon(text: str, expected: bool) -> None:
    assert bool(_TRAILING_PLAN_COLON.search(text)) is expected


# ----- _trailing_plan_hit composite -----------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        # Any of the three sub-patterns triggers a hit
        ("Now let me run the tests.", True),
        ("Let me do this:\n- step one\n- step two\n", True),
        ("Let me check the repo:", True),
        # Negative cases that previously misfired
        ("If you need anything else, let me know.", False),
        (
            "Here's my plan:\n- step one\n- step two\n\nDone, hope that helps.",
            False,
        ),
        # Short empty string is a no-op
        ("", False),
        ("   ", False),
    ],
)
def test_trailing_plan_hit(text: str, expected: bool) -> None:
    assert _trailing_plan_hit(text) is expected


# ----- window slicing --------------------------------------------------------


def test_trailing_plan_hit_respects_window() -> None:
    """An intent cue further back than ``_TRAILING_PLAN_WINDOW`` must NOT
    trigger a hit; only the tail of the response is inspected."""

    # 800-char prefix of unrelated text, then a finalising sentence.
    prefix = "lorem ipsum " * 80  # ~960 chars
    text = f"Let me check the repo. {prefix}The result is 42."
    assert _trailing_plan_hit(text) is False
