# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the A/B interleaver.

The balance check is the one worth the most scrutiny: it is a boolean that has to differ between a
plan where drift cancels and one where it does not, and its first implementation returned True for
both. A single-rep plan is therefore asserted explicitly rather than left to a parametrisation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.runtime.ab import (  # noqa: E402
    Target,
    browser_origin,
    interleave,
    order_is_balanced,
    origin_scoped,
)
from studiobench.runtime.types import Cell  # noqa: E402


def _cells(reps: int, rung: str = "1K"):
    return [
        (Cell(cell_id = f"r{rung}.A0.rep{rep}", rung = rung, rung_tokens = 1000, rep = rep), object())
        for rep in range(reps)
    ]


def _targets():
    return [
        Target(label = "base", ref = "main", base_url = "http://a", seeder = None, runner = None),
        Target(label = "treatment", ref = "pr", base_url = "http://b", seeder = None, runner = None),
    ]


def test_both_sides_run_for_every_cell():
    plan = interleave(_cells(2), _targets())
    assert len(plan) == 4
    assert {t.label for t, _c, _p in plan} == {"base", "treatment"}


def test_the_order_flips_between_reps():
    plan = interleave(_cells(2), _targets())
    labels = [t.label for t, _c, _p in plan]
    assert labels == ["base", "treatment", "treatment", "base"]


def test_each_arm_gets_its_own_cell_id():
    """Two arms sharing a cell_id would collide in the payload and in --resume."""
    plan = interleave(_cells(1), _targets())
    ids = [c.cell_id for _t, c, _p in plan]
    assert len(set(ids)) == len(ids)
    assert all(t.label in c.cell_id for t, c, _p in plan)


def test_two_reps_are_balanced():
    assert order_is_balanced(interleave(_cells(2), _targets())) is True


def test_one_rep_is_NOT_balanced():
    """The regression that mattered: base always runs first, so nothing cancels."""
    assert order_is_balanced(interleave(_cells(1), _targets())) is False


def test_three_reps_are_not_balanced():
    assert order_is_balanced(interleave(_cells(3), _targets())) is False


def test_a_single_target_is_never_balanced():
    one = [Target(label = "base", ref = "main", base_url = "http://a", seeder = None, runner = None)]
    assert order_is_balanced(interleave(_cells(2), one)) is False


def test_origin_gate_names_the_exact_origin_and_strips_the_slash():
    script = origin_scoped("http://127.0.0.1:5401/", "doThing();")
    assert '"http://127.0.0.1:5401"' in script
    assert "doThing();" in script
    assert "return" in script


# ── the gate compares against an ORIGIN, so it has to be given one ──────────────────────────
#
# `window.location.origin` is the URL standard's canonical origin, not the URL the caller typed.
# Every expectation below was read out of chromium, by navigating a real document to the spelling
# on the left and asking the page for `window.location.origin`.


@pytest.mark.parametrize(
    ("spelled", "origin"),
    [
        ("http://127.0.0.1:5401", "http://127.0.0.1:5401"),
        ("http://127.0.0.1:5401/", "http://127.0.0.1:5401"),
        # A port the scheme implies is not part of the origin.
        ("http://studio:80", "http://studio"),
        ("http://studio", "http://studio"),
        ("https://studio.example.com:443", "https://studio.example.com"),
        ("https://studio.example.com", "https://studio.example.com"),
        # A port the scheme does NOT imply is.
        ("https://studio.example.com:8443", "https://studio.example.com:8443"),
        ("http://studio:443", "http://studio:443"),
        # The scheme and host are lower-cased; nothing else is touched.
        ("HTTP://STUDIO", "http://studio"),
        # Path, query, fragment and userinfo are not part of an origin.
        ("http://studio/app?x=1#y", "http://studio"),
        ("http://user:secret@studio", "http://studio"),
        # IPv6 keeps its brackets.
        ("http://[::1]:5401", "http://[::1]:5401"),
        # THE ONE THAT MUST NOT BE FOLDED: a browser treats these as two origins and so does this.
        ("http://localhost:8000", "http://localhost:8000"),
        ("http://127.0.0.1:8000", "http://127.0.0.1:8000"),
    ],
)
def test_browser_origin_is_what_the_browser_reports(spelled, origin):
    assert browser_origin(spelled) == origin
    assert f'"{origin}"' in origin_scoped(spelled, "doThing();")


@pytest.mark.parametrize("junk", ["", "studio:5401", "not a url", "/relative/path"])
def test_a_string_that_is_not_an_absolute_url_is_left_alone(junk):
    """Guessing at a malformed `--attach` would turn a typo into a silently different origin. The
    caller sees it at `wait_for_healthz` instead, which is where a URL that reaches nothing
    belongs."""

    assert browser_origin(junk) == junk.rstrip("/")
