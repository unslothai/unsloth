# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""_fence_for must not walk a <pre> one character at a time, and must pick the fence the walk picked."""

from __future__ import annotations

import random
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference._html_to_md import _fence_for, html_to_markdown


def _walking_fence_for(raw: str) -> str:
    longest = run = 0
    for char in raw:
        run = run + 1 if char == "`" else 0
        longest = max(longest, run)
    return "`" * max(3, longest + 1)


class _NoIterStr(str):
    def __iter__(self):
        raise AssertionError("_fence_for walked the block character by character")


def test_fence_for_does_not_walk_the_block():
    raw = _NoIterStr("x = 1\n" * 1000 + "s = '````'\n")
    assert _fence_for(raw) == "`````"
    assert _fence_for(_NoIterStr("x = 1\n" * 1000)) == "```"


def test_fence_for_matches_the_walk():
    rng = random.Random(0)
    alphabet = ["`", "``", "```", "````", "a", " ", "\n", "é", "​", "\x00", "﻿"]
    cases = ["", "`", "``", "```", "a`b``c```d", "`" * 40, "x" * 1000]
    cases += ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60))) for _ in range(5000)]
    for raw in cases:
        assert _fence_for(raw) == _walking_fence_for(raw), repr(raw)


def test_pre_with_backtick_runs_gets_a_longer_fence():
    html = "<pre>a ``` b\n```` c</pre>"
    out = html_to_markdown(html)
    assert "`````\na ``` b\n```` c\n`````" in out
