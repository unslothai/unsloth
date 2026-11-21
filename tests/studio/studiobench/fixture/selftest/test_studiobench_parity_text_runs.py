# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Durations split across React text nodes must still normalise, or wall clock leaks into the digest."""

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


def test_a_duration_split_across_text_nodes_is_normalised():
    # A duration split across text nodes must normalise like a single node.
    assert sig(reasoning_trigger("3")) == sig(reasoning_trigger("2"))


def test_the_same_duration_in_one_text_node_still_normalises():
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


def test_a_bare_number_with_no_unit_still_moves_the_signature():
    # A bare number with no time unit is content and must move the digest.
    def badge(n: str) -> dict:
        return {"tag": "span", "children": [{"tag": "b", "children": [n, " messages"]}]}

    assert sig(badge("3")) != sig(badge("2"))


def test_an_element_boundary_still_breaks_a_text_run():
    # Joining must not cross element boundaries.
    split = {
        "tag": "div",
        "children": [
            {"tag": "span", "children": ["3"]},
            {"tag": "span", "children": [" seconds"]},
        ],
    }
    joined = {"tag": "div", "children": [{"tag": "span", "children": ["3", " seconds"]}]}
    assert sig(split) != sig(joined)
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
