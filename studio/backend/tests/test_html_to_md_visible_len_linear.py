# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""_visible_len scores page text for main-content selection. A line full of link destinations
that never close used to rescan its tail once per candidate (quadratic in page text); it must
stay linear and count exactly what the rescanning scanner counted."""

from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference._html_to_md import _visible_len, html_to_markdown


def _rescanning_visible_len(line: str) -> int:
    """The scanner before it indexed destinations: one forward scan per candidate."""
    if "[" not in line and "\\" not in line:
        return len(line)
    total, i, n = 0, 0, len(line)
    open_bracket = False
    while i < n:
        if line[i] == "\\":
            total += 2
            i += 2
            continue
        if line[i] == "[":
            open_bracket = True
        if open_bracket and line[i] == "]" and i + 1 < n and line[i + 1] == "(":
            j, depth = i + 2, 1
            while j < n and depth:
                char = line[j]
                if char == "\\":
                    j += 2
                    continue
                depth += (char == "(") - (char == ")")
                j += 1
            if depth:
                total += 1
                i += 1
                continue
            i = j
            open_bracket = False
            continue
        total += 1
        i += 1
    return total


class _CountingStr(str):
    reads = 0

    def __getitem__(self, key):
        _CountingStr.reads += 1
        return str.__getitem__(self, key)


def test_every_short_line_scores_as_the_rescanning_scanner_did():
    for length in range(8):
        for chars in itertools.product("[]()\\a", repeat = length):
            line = "".join(chars)
            assert _visible_len(line) == _rescanning_visible_len(line), line


def test_random_link_soup_scores_as_the_rescanning_scanner_did():
    rng = random.Random(0)
    for _ in range(3000):
        alphabet = rng.choice(["[]()\\a", "[]((\\a", "[]())a", "[](a b", "[]()\\\\a"])
        line = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 400)))
        assert _visible_len(line) == _rescanning_visible_len(line), line


def test_unclosed_destinations_are_read_a_bounded_number_of_times():
    line = _CountingStr("[a](" * 400)
    _CountingStr.reads = 0
    assert _visible_len(line) == _rescanning_visible_len(str(line))
    assert _CountingStr.reads < 10 * len(line)


def test_a_page_of_unclosed_links_converts_as_before():
    html = "<html><body><p>" + " ".join(f"[x{i}](/y{i}" for i in range(300)) + "</p></body></html>"
    out = html_to_markdown(html, main_content = True)
    assert out.count("[x") == 300
