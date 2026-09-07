# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""
`wait_for_first` is what stops a UI probe reporting a race as a missing feature.

The probes are full of `if locator.count() > 0:` gates. `count()` does not wait
-- Playwright's auto-waiting covers actions and expectations, not counting -- so
each of those is a sample of one instant dressed up as a question about the app.

#9251 is the worked example. Its reload snapshot paints a cloned overlay over the
app and removes it on hydration, which opens a window where the composer is on
screen but not yet in the accessibility tree. The Compare step sampled it six
milliseconds in, got 0, and reported "Compare nav not found" -- true about that
instant, false about the app, and indistinguishable in CI from the menu item
actually having been deleted.

No browser here: a locator is a small protocol (`.first`, `.wait_for`), so the
timeout, success and pass-through paths are all checkable directly. What is NOT
checkable without a browser is that playwright's TimeoutError is the exception
that arrives, so that import is asserted separately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _playwright_robust import wait_for_first  # noqa: E402


class _FakeTimeout(Exception):
    """Stands in for playwright.sync_api.TimeoutError."""


@pytest.fixture
def fake_playwright(monkeypatch):
    """A `playwright.sync_api` whose TimeoutError is one we can raise."""
    import types

    module = types.ModuleType("playwright.sync_api")
    module.TimeoutError = _FakeTimeout
    package = types.ModuleType("playwright")
    package.sync_api = module
    monkeypatch.setitem(sys.modules, "playwright", package)
    monkeypatch.setitem(sys.modules, "playwright.sync_api", module)
    return module


class _Locator:
    def __init__(self, *, raises: bool = False):
        self._raises = raises
        self.waited_state: str | None = None
        self.waited_timeout: int | None = None

    @property
    def first(self):
        return self

    def wait_for(self, *, state, timeout):
        self.waited_state = state
        self.waited_timeout = timeout
        if self._raises:
            raise _FakeTimeout("timed out")


def test_a_control_that_arrives_late_is_returned(fake_playwright):
    locator = _Locator()
    assert wait_for_first(locator) is locator
    # "attached", not "visible": the callers go on to `click(force = True)`, and a
    # control inside a just-opened menu can be attached before it has settled.
    assert locator.waited_state == "attached"


def test_a_control_that_never_arrives_is_none_not_an_exception(fake_playwright):
    """
    The callers branch on absence -- one of them legitimately expects a miss and
    falls back to the "More" submenu. Raising would turn that branch into a crash.
    """
    assert wait_for_first(_Locator(raises = True)) is None


def test_the_default_wait_is_long_enough_to_outlast_a_reload_overlay(fake_playwright):
    """
    #9251's overlay removes itself on hydration or after 5000ms, whichever comes
    first. A default under that would still sample inside the window it exists to
    outlast, so this is the one number in here that is not arbitrary.
    """
    locator = _Locator()
    wait_for_first(locator)
    assert locator.waited_timeout >= 5000


def test_a_caller_can_ask_for_a_shorter_wait(fake_playwright):
    """The menu-item fallbacks: a miss there is a real branch, not a slow render."""
    locator = _Locator()
    wait_for_first(locator, timeout_ms = 2000)
    assert locator.waited_timeout == 2000


def test_only_a_timeout_is_swallowed(fake_playwright):
    """
    A locator that raises anything else -- a closed page, a bad selector -- is a
    real failure, and reporting it as "not present" would hide it behind a
    soft_fail about a missing feature.
    """

    class _Broken(_Locator):
        def wait_for(self, *, state, timeout):
            raise RuntimeError("Target page, context or browser has been closed")

    with pytest.raises(RuntimeError):
        wait_for_first(_Broken())


def test_the_helper_binds_playwrights_own_timeout_error() -> None:
    """
    The fixture above supplies a stand-in, so nothing else here would notice the
    helper importing the wrong name. It must also be a LOCAL import: this module
    is read by harness-contract tests on runners with no browser stack, and a
    top-level playwright import turns those skips into collection errors.
    """
    source = (Path(__file__).resolve().parent / "_playwright_robust.py").read_text(encoding = "utf-8")
    assert "from playwright.sync_api import TimeoutError as PlaywrightTimeoutError" in source
    body = source[source.index("def wait_for_first") :]
    body = body[: body.index("\ndef ")]
    assert "from playwright.sync_api import" in body, (
        "the playwright import moved out of wait_for_first(); at module scope it "
        "breaks every browserless importer of this file"
    )
