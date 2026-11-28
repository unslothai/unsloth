# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Seat matching (`_occurrences`) must pick the same transcript positions with or without its memo."""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag import conversation_archive  # noqa: E402

_seat_text = conversation_archive._seat_text


def _reference_occurrences(positions, group):
    """The matcher before the memo and the hoisted split offsets, kept verbatim as the oracle."""
    if not positions or not group:
        return []
    texts = [_seat_text(message) for message in group]
    if not texts or not texts[0]:
        return []
    calls = [bool(message.get("tool_calls")) for message in group]

    def _same(stored, live, is_call):
        if not is_call:
            return stored == live
        if not live:
            return False
        if live in stored:
            return True
        if len(stored) < len(live) - 1:
            return False
        cuts = [index for index, char in enumerate(live) if char == " "]
        if not cuts:
            return False
        left, right = 0, len(cuts)
        while left < right:
            mid = (left + right) // 2
            if live[: cuts[mid]] in stored:
                left = mid + 1
            else:
                right = mid
        prefix_stop = left
        if prefix_stop == 0:
            return False
        left, right = 0, prefix_stop
        while left < right:
            mid = (left + right) // 2
            if live[cuts[mid] + 1 :] in stored:
                right = mid
            else:
                left = mid + 1
        for index in range(left, prefix_stop):
            cut = cuts[index]
            head_at = stored.find(live[:cut])
            tail_at = stored.rfind(live[cut + 1 :])
            if tail_at >= head_at + cut:
                return True
        return False

    return [
        index
        for index, position in enumerate(positions)
        if position
        and (len(position) >= len(texts) or index == len(positions) - 1)
        and all(
            _same(stored, live, is_call) for stored, live, is_call in zip(position, texts, calls)
        )
    ]


_WORDS = ["a", "b", "ab", "ls", "x y", "", "run", "print(1)"]


def _phrase(rng):
    return " ".join(rng.choice(_WORDS) for _ in range(rng.randint(0, 4)))


def _message(rng):
    kind = rng.choice(["user", "assistant", "tool", "call", "call", "call"])
    if kind == "call":
        name = rng.choice(["python", "terminal"])
        return {
            "role": "assistant",
            "content": _phrase(rng) if rng.random() < 0.3 else "",
            "tool_calls": [{"id": "c", "function": {"name": name, "arguments": _phrase(rng)}}],
        }
    if kind == "tool":
        return {"role": "tool", "tool_call_id": "c", "content": _phrase(rng)}
    return {"role": kind, "content": _phrase(rng)}


def _group(rng):
    return [_message(rng) for _ in range(rng.randint(1, 4))]


def _stored_variant(rng, text):
    """A transcript spelling of a live text: same, padded, one space dropped, or unrelated."""
    roll = rng.random()
    if roll < 0.35:
        return text
    if roll < 0.5:
        return _phrase(rng) + " " + text + " " + _phrase(rng)
    if roll < 0.7 and " " in text:
        cut = rng.choice([index for index, char in enumerate(text) if char == " "])
        middle = rng.choice(["", "zz", " q "])
        return text[:cut] + middle + text[cut + 1 :]
    if roll < 0.8:
        return ""
    return _seat_text(_message(rng))


def _positions(rng, groups):
    positions = []
    for _ in range(rng.randint(0, 14)):
        roll = rng.random()
        if roll < 0.1:
            positions.append([])
            continue
        texts = [_seat_text(message) for message in rng.choice(groups)]
        if roll < 0.3:
            # Shorter and longer positions: only the trailing short one may match.
            texts = texts[: rng.randint(0, len(texts))] if rng.random() < 0.6 else texts + ["extra"]
        positions.append([_stored_variant(rng, text) for text in texts])
    return positions


def test_memo_matcher_returns_the_reference_seats_on_random_turns():
    rng = random.Random(20260)
    matched = 0
    for _ in range(3000):
        groups = [_group(rng) for _ in range(rng.randint(1, 5))]
        positions = _positions(rng, groups)
        memo = {}
        # One memo serves every group of a pass, so reuse it across groups like _archive_turns does.
        for group in groups + [_group(rng)]:
            expected = _reference_occurrences(positions, group)
            assert conversation_archive._occurrences(positions, group) == expected
            assert conversation_archive._occurrences(positions, group, memo) == expected
            matched += bool(expected)
    # The corpus must exercise matches, not only misses.
    assert matched > 1000


def test_memo_keeps_the_trailing_short_position_and_drops_earlier_short_ones():
    call = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "t", "arguments": "ls -la"}}],
    }
    result = {"role": "tool", "content": "out"}
    texts = [_seat_text(call), _seat_text(result)]
    positions = [[texts[0]], texts, ["other", "out"], [texts[0]]]
    for memo in (None, {}):
        assert conversation_archive._occurrences(positions, [call, result], memo) == [1, 3]
        assert _reference_occurrences(positions, [call, result]) == [1, 3]


def test_write_budget_with_a_live_memo_matches_without():
    rng = random.Random(7)
    for _ in range(500):
        groups = [_group(rng) for _ in range(3)]
        positions = _positions(rng, groups)
        live_positions = _positions(rng, groups)
        live_memo = {}
        for group in groups:
            seats = _reference_occurrences(positions, group)
            plain = conversation_archive._write_budget(positions, seats, live_positions, group)
            memoised = conversation_archive._write_budget(
                positions, seats, live_positions, group, live_memo
            )
            assert memoised == plain
