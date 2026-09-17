# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""
A forced click has to scroll first, or it clicks a point that is not on screen.

`click(force = True)` turns off actionability checks, which is the point when a menu
overlay would otherwise intercept the click. It also turns off the scroll that puts
the element in the viewport, and Playwright refuses a point it cannot reach:

    Locator.click: Element is outside of the viewport

That failed `Compare tab: send to two panes` on macOS. The menu item existed and was
found; it was simply below the fold, because a Mac runner's window is shorter than a
Linux one and the item sits at the bottom of a long menu. The same three forced
clicks have been in playwright_extra_ui.py since the composer redesign and have
always worked on Linux, which is why nothing caught it until the Compare nav started
being found reliably enough for the click to run at all.

Driven against a fake locator rather than a browser: these run on browserless CI
lanes, and the ordering is the whole contract, so a stand-in that records call order
tests it exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _playwright_robust import click_forced  # noqa: E402


class _FakeLocator:
    def __init__(self, *, scroll_raises: Exception | None = None):
        self.calls: list[str] = []
        self._scroll_raises = scroll_raises
        self.click_kwargs: dict | None = None

    def scroll_into_view_if_needed(self, timeout = None):
        self.calls.append("scroll")
        if self._scroll_raises is not None:
            raise self._scroll_raises

    def click(self, **kwargs):
        self.calls.append("click")
        self.click_kwargs = kwargs


def test_it_scrolls_before_clicking() -> None:
    loc = _FakeLocator()
    click_forced(loc)
    assert loc.calls == ["scroll", "click"], (
        f"expected a scroll then a click, got {loc.calls}. Clicking first is the bug: "
        f"the element is off-screen at that moment."
    )


def test_the_click_is_still_forced() -> None:
    """Dropping force would reintroduce the overlay interception it was added for."""
    loc = _FakeLocator()
    click_forced(loc)
    assert loc.click_kwargs == {"force": True}, (
        f"click was called with {loc.click_kwargs}; force must stay, because the menu "
        f"overlay is why these call sites bypass actionability in the first place"
    )


def test_a_scroll_that_fails_does_not_stop_the_click() -> None:
    """
    An element that cannot be scrolled -- fixed position, zero size -- should still
    reach the click and fail there with Playwright's own message, rather than here
    with a scrolling one that names the wrong problem.
    """
    loc = _FakeLocator(scroll_raises = RuntimeError("no scrollable ancestor"))
    click_forced(loc)
    assert loc.calls == ["scroll", "click"]


def test_a_failing_click_still_propagates() -> None:
    """The helper must not swallow the thing it exists to report."""

    class _Boom(_FakeLocator):
        def click(self, **kwargs):
            self.calls.append("click")
            raise RuntimeError("Element is outside of the viewport")

    loc = _Boom()
    with pytest.raises(RuntimeError, match = "outside of the viewport"):
        click_forced(loc)


def test_every_forced_click_in_the_suite_goes_through_the_helper() -> None:
    """
    The helper is only worth having if nothing bypasses it. A bare
    `click(force = True)` is the exact shape that broke, so it fails here rather than
    on a Mac runner twenty minutes into a job.
    """
    here = Path(__file__).resolve().parent
    offenders = []
    for path in sorted(here.glob("playwright_*.py")):
        for i, line in enumerate(path.read_text(encoding = "utf-8").splitlines(), 1):
            if "click(force" in line and "def " not in line:
                offenders.append(f"{path.name}:{i}")
    assert not offenders, (
        f"these call sites force a click without scrolling into view first: {offenders}. "
        f"Use click_forced from _playwright_robust, which keeps force and adds the "
        f"scroll that force turns off."
    )
