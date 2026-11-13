# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""count() samples one instant without waiting, so a racing control would look missing."""

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
    # "attached", not "visible": callers then `click(force = True)` on unsettled menus.
    assert locator.waited_state == "attached"


def test_a_control_that_never_arrives_is_none_not_an_exception(fake_playwright):
    """Absence returns None, not an exception, since callers branch on a miss to fall back."""
    assert wait_for_first(_Locator(raises = True)) is None


def test_the_default_wait_is_long_enough_to_outlast_a_reload_overlay(fake_playwright):
    """Default wait must outlast the 5000ms reload overlay, or it would sample inside the window."""
    locator = _Locator()
    wait_for_first(locator)
    assert locator.waited_timeout >= 5000


def test_a_caller_can_ask_for_a_shorter_wait(fake_playwright):
    """The menu-item fallbacks: a miss there is a real branch, not a slow render."""
    locator = _Locator()
    wait_for_first(locator, timeout_ms = 2000)
    assert locator.waited_timeout == 2000


def test_only_a_timeout_is_swallowed(fake_playwright):
    """Only a timeout counts as absent; a closed page or bad selector must still raise."""

    class _Broken(_Locator):
        def wait_for(self, *, state, timeout):
            raise RuntimeError("Target page, context or browser has been closed")

    with pytest.raises(RuntimeError):
        wait_for_first(_Broken())


def test_the_helper_binds_playwrights_own_timeout_error() -> None:
    """The helper must import TimeoutError locally; a top-level playwright import breaks collection."""
    source = (Path(__file__).resolve().parent / "_playwright_robust.py").read_text(encoding = "utf-8")
    assert "from playwright.sync_api import TimeoutError as PlaywrightTimeoutError" in source
    body = source[source.index("def wait_for_first") :]
    body = body[: body.index("\ndef ")]
    assert "from playwright.sync_api import" in body, (
        "the playwright import moved out of wait_for_first(); at module scope it "
        "breaks every browserless importer of this file"
    )
