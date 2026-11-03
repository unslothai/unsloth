# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Skips the length-continuation nudge when a truncated fragment is mostly long verbatim repeats."""

from __future__ import annotations

import math

# Shorter fragments are too short to judge; a cut sentence can trivially repeat tokens.
MIN_FRAGMENT_LENGTH = 400

# Exact-repeat window: a verbatim repeat this long exceeds normal phrasing and boilerplate.
_REPEAT_WINDOW = 60

# A window repeating at least this many times is a signal even in a short fragment.
_MIN_REPEAT_COUNT = 5

# The share of the fragment repeated windows must cover before it counts as dominated.
_DOMINANCE_RATIO = 0.5

# Memory bound only; real fragments stay far below it.
_MAX_TRACKED_WINDOWS = 100_000


def is_repetition_dominated(text: str) -> bool:
    """Fails open: text it cannot judge is reported as not repetitive, avoiding refusals of real answers."""
    if not isinstance(text, str):
        return False
    length = len(text)
    if length < MIN_FRAGMENT_LENGTH:
        return False
    if _line_repetition_dominated(text, length):
        return True
    # Sliding exact-repeat windows, for echoes that do not align to line boundaries.
    needed = max(_MIN_REPEAT_COUNT, math.ceil(length * _DOMINANCE_RATIO / _REPEAT_WINDOW))
    # Keyed by hash: storing slices held ~180 MB for an 800K-char fragment.
    counts: dict[int, int] = {}
    # Non-overlapping: overlapping windows of one short run tripped the threshold falsely.
    covered_to: dict[int, int] = {}
    # Where each hash was first seen, so a hash collision cannot be counted as a repeat.
    first_at: dict[int, int] = {}
    for index in range(length - _REPEAT_WINDOW + 1):
        key = hash(text[index : index + _REPEAT_WINDOW])
        if index < covered_to.get(key, 0):
            continue
        first = first_at.get(key)
        if first is None:
            # Past the cap new windows are ignored, which can only fail open.
            if len(first_at) >= _MAX_TRACKED_WINDOWS:
                continue
            first_at[key] = index
        elif text[first : first + _REPEAT_WINDOW] != text[index : index + _REPEAT_WINDOW]:
            continue
        seen = counts.get(key, 0) + 1
        if seen >= needed:
            return True
        counts[key] = seen
        covered_to[key] = index + _REPEAT_WINDOW
    return False


def _line_repetition_dominated(text: str, length: int) -> bool:
    """Checked first because it is cheap and allocates nothing, unlike the window pass."""
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
