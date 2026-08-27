# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Is this truncated fragment worth continuing, or is the model just echoing itself?

Only asked on the length-continuation path. A turn cut off mid-answer is normally resumed
by handing the partial back and asking for the rest, but a model stuck in a repetition loop
can spend an entire window echoing one fragment, and continuing THAT stitches the echo into
the final answer instead of finishing it. The nudge has to be withheld before it is sent,
not regretted afterwards.

The approach and the thresholds follow NousResearch/hermes-agent's `agent/repetition_guard.py`,
written after an incident where a single turn produced a 60,698-char response delivered as
31 messages. Deliberately conservative in the same way: only LONG verbatim repeats covering
a majority of the fragment trip it, so a sentence cut mid-word, a repeated heading, or code
with similar-looking lines are all still continued.
"""

from __future__ import annotations

import math

# Below this, a fragment is too short to judge. A sentence cut mid-word can trivially
# repeat a few tokens and is legitimately continued.
MIN_FRAGMENT_LENGTH = 400

# Length of the exact-repeat window. A verbatim repeat this long is well beyond ordinary
# reuse of phrasing, citations, headings or boilerplate.
_REPEAT_WINDOW = 60

# A window repeating at least this many times is a signal even in a short fragment.
_MIN_REPEAT_COUNT = 5

# The share of the fragment repeated windows must cover before it counts as dominated.
_DOMINANCE_RATIO = 0.5


def is_repetition_dominated(text: str) -> bool:
    """Whether verbatim repeats account for the majority of ``text``.

    Fails open: anything it cannot confidently judge is reported as fine to continue, so a
    false negative costs one wasted continuation while a false positive would refuse to
    finish an answer that was merely repetitive in an ordinary way.
    """
    if not isinstance(text, str):
        return False
    length = len(text)
    if length < MIN_FRAGMENT_LENGTH:
        return False
    if _line_repetition_dominated(text, length):
        return True
    # Sliding exact-repeat windows, for echoes that do not align to line boundaries.
    needed = max(_MIN_REPEAT_COUNT, math.ceil(length * _DOMINANCE_RATIO / _REPEAT_WINDOW))
    counts: dict[str, int] = {}
    # Occurrences must not overlap, or a single run of one character counts as many. A
    # 64-character rule inside a 400-character answer yields five overlapping 60-character
    # windows and tripped the threshold at 16 percent coverage, abandoning a valid answer.
    covered_to: dict[str, int] = {}
    for index in range(length - _REPEAT_WINDOW + 1):
        key = text[index : index + _REPEAT_WINDOW]
        if index < covered_to.get(key, 0):
            continue
        seen = counts.get(key, 0) + 1
        if seen >= needed:
            return True
        counts[key] = seen
        covered_to[key] = index + _REPEAT_WINDOW
    return False


def _line_repetition_dominated(text: str, length: int) -> bool:
    """The common shape: one line repeated until it covers half the fragment.

    Checked first because it is cheap and allocates nothing, unlike the window pass.
    """
    counts: dict[str, int] = {}
    for line in text.splitlines():
        normalised = line.strip()
        if not normalised:
            continue
        counts[normalised] = counts.get(normalised, 0) + 1
    return any(
        seen >= _MIN_REPEAT_COUNT and seen * len(line) >= length * _DOMINANCE_RATIO
        for line, seen in counts.items()
    )
