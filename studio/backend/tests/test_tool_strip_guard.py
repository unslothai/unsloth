# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""strip_tool_patterns must match the plain per-pattern loop while skipping the
quadratic no-match rescan of a closed-pair sweep whose close token is absent."""

import random
import sys
import time
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from core.tool_healing import (
    _TOOL_ALL_PATS,
    _TOOL_CLOSED_PATS,
    strip_tool_call_markup,
    strip_tool_patterns,
)


def _naive(text, patterns):
    for pat in patterns:
        text = pat.sub("", text)
    return text


_TOKENS = [
    "<tool_call>",
    "</tool_call>",
    "<|tool_call>",
    "<tool_call|>",
    "<function=x>",
    "<function=mcp__s__a-b>",
    "</function>",
    "<parameter=p>",
    "</parameter>",
    "call:fn{",
    "}",
    "{",
    '<|"|>',
    "A",
    " ",
    "\n",
    "id",
    "x:1",
    "</tool",
    "call>",
]


def test_guard_matches_plain_loop_on_fuzz():
    rng = random.Random(1234)
    for patterns in (_TOOL_ALL_PATS, _TOOL_CLOSED_PATS):
        for _ in range(20000):
            s = "".join(rng.choice(_TOKENS) for _ in range(rng.randint(0, 10)))
            assert strip_tool_patterns(s, patterns) == _naive(s, patterns), (
                s,
                patterns,
            )


def test_strip_markup_representative_cases_unchanged():
    assert strip_tool_call_markup("a <tool_call>{}</tool_call> b") == "a  b"
    assert (
        strip_tool_call_markup("a <function=x><parameter=p>1</parameter></function> b")
        == "a  b"
    )
    # Non-final keeps an unclosed block; final strips it to EOF.
    assert strip_tool_call_markup("a <tool_call>{partial") == "a <tool_call>{partial"
    assert strip_tool_call_markup("a <tool_call>{partial", final = True) == "a"


def test_no_quadratic_blowup_on_unclosed_markers():
    # Unguarded, this took minutes.
    big = "<tool_call>" * 20000 + "<function=x>" * 20000
    t0 = time.perf_counter()
    out = strip_tool_call_markup(big, final = True)
    assert time.perf_counter() - t0 < 2.0
    assert out == ""
