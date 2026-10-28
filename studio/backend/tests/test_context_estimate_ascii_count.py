# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The dense and conservative estimators count ASCII with ``str.encode`` instead of a per-character
loop. Pin that every integer they return equals the original loop's, emoji and lone surrogates
included."""

import json
import random

import pytest

from core.inference import context_window as cw


def _dense_reference(messages):
    total = 0
    for message in messages:
        try:
            text = json.dumps(message, ensure_ascii = False)
        except Exception:
            total += 1
            continue
        dense = sum(1 for char in text if ord(char) > 127)
        total += max(1, dense + (len(text) - dense) // 4)
    return total


def _conservative_reference(messages, *, dense_ascii = False):
    total = 0
    for message in messages:
        try:
            text = json.dumps(message, ensure_ascii = False)
        except Exception:
            total += 1
            continue
        wide = sum(1 for char in text if ord(char) > 127)
        if dense_ascii:
            total += max(1, wide + (len(text) - wide) // cw._DENSE_RUN_CHARS_PER_TOKEN)
            continue
        runs = sum(
            sum(1 for char in match.group(0) if ord(char) <= 127)
            for match in cw._DENSE_RUN_RE.finditer(text)
        )
        plain = len(text) - wide - runs
        total += max(1, wide + runs // cw._DENSE_RUN_CHARS_PER_TOKEN + plain // 4)
    return total


_ALPHABETS = (
    'abc XYZ 019 {}[]":,\n\t\\/',
    "\x00\x01\x7f\x80\xff",
    "中文日本語한국어",
    "\U0001f600\U0001f680\U0001f9d1‍\U0001f4bb",
    "\ud800􏰀\udfff",  # lone surrogates survive json.dumps(ensure_ascii = False)
    "éñЖאا",
    "A" * 8 + "/+=",
)


def _random_text(rng):
    pieces = []
    for _ in range(rng.randint(0, 12)):
        alphabet = rng.choice(_ALPHABETS)
        run = rng.choice((1, 3, 63, 64, 65, 200))
        pieces.append("".join(rng.choice(alphabet) for _ in range(run)))
        if rng.random() < 0.5:
            pieces.append(" ")
    return "".join(pieces)


def _random_messages(rng):
    messages = []
    for _ in range(rng.randint(0, 4)):
        roll = rng.random()
        if roll < 0.05:
            messages.append({"role": "user", "content": object()})  # json.dumps raises
        elif roll < 0.15:
            messages.append(
                {
                    "role": "assistant",
                    "tool_calls": [{"function": {"arguments": _random_text(rng)}}],
                }
            )
        else:
            messages.append(
                {"role": rng.choice(("user", "system", "tool")), "content": _random_text(rng)}
            )
    return messages


def test_ascii_count_matches_ord_loop_on_every_code_point():
    every = "".join(chr(c) for c in range(0x110000))
    assert cw._ascii_count(every) == sum(1 for char in every if ord(char) <= 127) == 128
    for chunk_start in range(0, 0x110000, 0x8000):
        chunk = every[chunk_start : chunk_start + 0x8000 : 7]
        assert cw._ascii_count(chunk) == sum(1 for char in chunk if ord(char) <= 127)


@pytest.mark.parametrize("seed", range(20))
def test_estimators_match_reference_loops(seed):
    rng = random.Random(seed)
    for _ in range(150):
        messages = _random_messages(rng)
        assert cw.estimate_messages_tokens_dense(messages) == _dense_reference(messages)
        assert cw.estimate_messages_tokens_conservative(messages) == _conservative_reference(
            messages
        )
        assert cw.estimate_messages_tokens_conservative(
            messages, dense_ascii = True
        ) == _conservative_reference(messages, dense_ascii = True)


def test_estimators_on_fixed_edges():
    cases = [
        [],
        [{"content": ""}],
        [{"content": "x" * 64}],
        [{"content": "\U0001f600" * 64 + "a" * 64}],
        [{"content": "\ud83d" + "q" * 100 + "\ude00"}],
        [{"content": ("中" + "a") * 40}],
        [{"content": "base64:" + "QUJD" * 500}, {"content": "日本 " * 300}],
    ]
    for messages in cases:
        assert cw.estimate_messages_tokens_dense(messages) == _dense_reference(messages)
        assert cw.estimate_messages_tokens_conservative(messages) == _conservative_reference(
            messages
        )
        assert cw.estimate_messages_tokens_conservative(
            messages, dense_ascii = True
        ) == _conservative_reference(messages, dense_ascii = True)
