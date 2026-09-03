# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""A rendered duration split across text nodes is still a rendered duration.

THE BUG THIS HOLDS. `normText` has collapsed `295ms` and `3 seconds` to `#T` since it was written,
and it was still letting wall clock into the digest, because React renders
`Thought for {n} seconds` as three sibling text nodes and `signature()` normalised each node on its
own. `normText("3")` sees a bare digit with no unit after it and cannot match, so the number
survived. A null control -- two arms of a BYTE-IDENTICAL build -- disagreed by that one character
on `select_all_copy`, `settings`, `thread_reopen` and `delete_message`, none of which is on the
declared unstable list, so each one read as "this pull request changed the UI".

It was found by diffing the NORMALISED signature text of the two arms rather than their digests
(`sweep/parity_null_control.py --hunt`), which is the only way the offending bytes were ever going
to be named instead of guessed at.

Both directions, as everywhere else here: the volatile has to vanish, and the things that share
its shape have to survive. A normaliser that erased every number split across nodes would hide a
message count going from 3 to 2, which is content.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tests.studio.studiobench.fixture.selftest.test_studiobench_parity_digest import (  # noqa: E402
    sig,
)


def reasoning_trigger(seconds: str) -> dict:
    """The shape assistant-ui actually renders, node split included."""
    return {
        "tag": "span",
        "attrs": {"data-slot": "reasoning-trigger-label"},
        "children": [{"tag": "span", "children": ["Thought for ", seconds, " seconds"]}],
    }


# ── the volatile has to vanish ───────────────────────────────────────


def test_a_duration_split_across_text_nodes_is_normalised():
    # The exact regression. Before the fix these two differed by one character and the digest
    # moved; the two arms are the same build and the only difference is wall clock.
    assert sig(reasoning_trigger("3")) == sig(reasoning_trigger("2"))


def test_the_same_duration_in_one_text_node_still_normalises():
    # The path that already worked must keep working, and must agree with the split one.
    one_node = {
        "tag": "span",
        "attrs": {"data-slot": "reasoning-trigger-label"},
        "children": [{"tag": "span", "children": ["Thought for 3 seconds"]}],
    }
    assert sig(one_node) == sig(reasoning_trigger("3"))


def test_a_millisecond_reading_split_across_nodes_is_normalised():
    def bar(ms: str) -> dict:
        return {"tag": "div", "children": [{"tag": "span", "children": ["took ", ms, "ms"]}]}

    assert sig(bar("295")) == sig(bar("310"))


def test_a_split_relative_time_is_normalised():
    def stamp(n: str) -> dict:
        return {"tag": "div", "children": [{"tag": "time", "children": [n, " minutes ago"]}]}

    assert sig(stamp("2")) == sig(stamp("9"))


# ── and the things that share its shape must NOT vanish ──────────────


def test_a_bare_number_with_no_unit_still_moves_the_signature():
    # A message count, a row count, a badge. No time unit follows it, so it is content and the
    # digest has to see it change. This is the false-negative direction and it is the worse one.
    def badge(n: str) -> dict:
        return {"tag": "span", "children": [{"tag": "b", "children": [n, " messages"]}]}

    assert sig(badge("3")) != sig(badge("2"))


def test_an_element_boundary_still_breaks_a_text_run():
    # Joining must not reach ACROSS an element, or two separate labels weld into one string and a
    # difference in where the boundary sits stops being visible.
    split = {
        "tag": "div",
        "children": [
            {"tag": "span", "children": ["3"]},
            {"tag": "span", "children": [" seconds"]},
        ],
    }
    joined = {"tag": "div", "children": [{"tag": "span", "children": ["3", " seconds"]}]}
    assert sig(split) != sig(joined)
    # And the genuinely-split-across-elements case keeps its number, because nothing there proves
    # the two spans are one rendered phrase.
    other = {
        "tag": "div",
        "children": [
            {"tag": "span", "children": ["2"]},
            {"tag": "span", "children": [" seconds"]},
        ],
    }
    assert sig(split) != sig(other)


def test_text_around_a_child_element_is_not_welded_through_it():
    a = {"tag": "p", "children": ["before", {"tag": "b", "children": ["x"]}, "after"]}
    b = {"tag": "p", "children": ["beforeafter", {"tag": "b", "children": ["x"]}]}
    assert sig(a) != sig(b)


def test_added_and_removed_text_still_moves_the_signature():
    a = {"tag": "p", "children": ["hello ", "world"]}
    b = {"tag": "p", "children": ["hello ", "there"]}
    assert sig(a) != sig(b)
