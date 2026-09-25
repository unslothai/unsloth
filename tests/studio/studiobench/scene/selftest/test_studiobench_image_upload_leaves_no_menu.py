# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""`image_upload` must not leave behind a menu it opened when it gives up.

The composer's "Tools and attachments" menu is a modal Radix dropdown, which sets
`pointer-events: none` on everything outside itself while it is open, and Radix opens it on
pointerdown, so a click can open it and still time out. On 2026-09-25 that happened on #11887's
studiobench run: `image_upload` reported "the attachments button could not be clicked:
TimeoutError" and returned, the parity digest showed `[role="menu"]` open from that action on, and
the next action failed with "button[aria-label="New chat"] was not clickable ... no point on the
control hit-tests to it". CI passes `--allow-not-run image_upload`, so the failure it was excused
for turned into one it was not: thread_reopen NOT RUN, and the job red.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


class _Keyboard:
    def __init__(self, page: "_Page") -> None:
        self._page = page
        self.pressed: list[str] = []

    def press(self, key: str) -> None:
        self.pressed.append(key)
        if key == "Escape" and self._page.menu_closes_on_escape:
            self._page.menu_open = False


class _Button:
    def __init__(self, page: "_Page") -> None:
        self._page = page

    def click(self, timeout = None) -> None:
        if self._page.click_opens_menu:
            self._page.menu_open = True  # Radix opens on pointerdown
        raise TimeoutError(f"Timeout {timeout}ms exceeded.")


class _Locator:
    def __init__(self, page: "_Page") -> None:
        self._page = page

    @property
    def first(self) -> "_Locator":
        return self

    def element_handle(self, timeout = None) -> _Button:
        return _Button(self._page)


class _Page:
    """The page calls `image_upload` makes up to and including a click that times out."""

    def __init__(
        self,
        *,
        click_opens_menu: bool,
        menu_closes_on_escape: bool = True,
        menu_already_open: bool = False,
    ) -> None:
        self.click_opens_menu = click_opens_menu
        self.menu_closes_on_escape = menu_closes_on_escape
        self.menu_open = menu_already_open
        self.keyboard = _Keyboard(self)

    def locator(self, _selector: str) -> _Locator:
        return _Locator(self)

    def evaluate(
        self,
        script,
        arg = None,
    ):
        if '[role="menu"]' in script:
            return self.menu_open
        return 0  # the attachment count

    def wait_for_timeout(self, _ms) -> None:
        return None


def _upload(page: _Page):
    from studiobench.runtime.types import ActionContext
    from studiobench.scene.actions import image_upload

    ctx = ActionContext(
        page = page,
        cdp = None,
        cell = None,
        window = None,
        args = {"image_path": "probe.png"},
        budget_ms = 12_000,
        dom = None,
        log = lambda _m: None,
    )
    return image_upload(ctx)


def test_a_click_that_opened_the_menu_and_timed_out_closes_it():
    page = _Page(click_opens_menu = True)
    result = _upload(page)

    assert result.ran is False
    assert "could not be clicked: TimeoutError" in result.reason
    assert page.menu_open is False, "the modal menu was left open over the next action"
    assert page.keyboard.pressed == ["Escape"]
    assert "closed the menu it opened" in result.reason


def test_a_click_that_opened_nothing_touches_nothing():
    """No stray Escape: with nothing open it could close something the page legitimately shows."""
    page = _Page(click_opens_menu = False)
    result = _upload(page)

    assert result.ran is False
    assert page.keyboard.pressed == []
    assert result.reason == "the attachments button could not be clicked: TimeoutError"


def test_a_menu_that_will_not_close_is_reported_and_bounded():
    page = _Page(click_opens_menu = True, menu_closes_on_escape = False)
    result = _upload(page)

    assert result.ran is False
    assert page.keyboard.pressed == ["Escape"] * 3
    assert "still open after Escape" in result.reason


def test_a_menu_that_was_already_open_is_left_alone():
    """Not this action's to close: it can be what blocked the click, and it belongs to its opener."""
    page = _Page(click_opens_menu = False, menu_already_open = True)
    result = _upload(page)

    assert result.ran is False
    assert page.keyboard.pressed == []
    assert page.menu_open is True
    assert result.reason == "the attachments button could not be clicked: TimeoutError"
