# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""image_upload must close the composer menu it opened before giving up, or the next click is blocked."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


class _Keyboard:
    def __init__(self, page: "_Page") -> None:
        self._page = page
        self.pressed: list[str] = []

    def press(self, key: str) -> None:
        self.pressed.append(key)
        if key == "Escape" and self._page.menus and self._page.menu_closes_on_escape:
            self._page.menus.pop()


class _Button:
    def __init__(self, page: "_Page") -> None:
        self._page = page

    def click(self, timeout = None) -> None:
        page = self._page
        if page.click_dismisses_open_menu:
            page.menus.clear()
        if page.click_opens_menu:
            page.menus.append("attachments")  # Radix opens on pointerdown
        raise TimeoutError(f"Timeout {timeout}ms exceeded.")


class _Chooser:
    def __init__(self, page: "_Page") -> None:
        self._page = page

    @property
    def value(self) -> "_Chooser":
        return self

    def set_files(self, _path) -> None:
        self._page.attachments += 1


class _Locator:
    def __init__(self, page: "_Page") -> None:
        self._page = page

    @property
    def first(self) -> "_Locator":
        return self

    def element_handle(self, timeout = None) -> _Button:
        return _Button(self._page)


class _Page:
    """Stub page; open menus are a stack of identities, so a replaced menu counts as a different one."""

    def __init__(
        self,
        *,
        click_opens_menu: bool,
        menu_closes_on_escape: bool = True,
        already_open: tuple = (),
        click_dismisses_open_menu: bool = False,
        chooser_opens: bool = False,
    ) -> None:
        self.chooser_opens = chooser_opens
        self.attachments = 0
        self.click_opens_menu = click_opens_menu
        self.menu_closes_on_escape = menu_closes_on_escape
        self.click_dismisses_open_menu = click_dismisses_open_menu
        self.menus: list[str] = list(already_open)
        self._before: set[str] = set()
        self.keyboard = _Keyboard(self)

    def locator(self, _selector: str) -> _Locator:
        return _Locator(self)

    @contextlib.contextmanager
    def expect_file_chooser(self, timeout = None):
        chooser = _Chooser(self)
        yield chooser
        if not (self.chooser_opens and "attachments" in self.menus):
            raise TimeoutError(
                f'Timeout {timeout}ms exceeded while waiting for event "filechooser"'
            )
        self.menus.remove("attachments")  # picking "Add photos" closes the menu

    def evaluate(
        self,
        script,
        arg = None,
    ):
        if "__sbMenusBefore = " in script:
            self._before = set(self.menus)
            return None
        if "__sbMenusBefore" in script:
            return any(menu not in self._before for menu in self.menus)
        if '[role="menu"]' in script:
            return bool(self.menus)  # a presence-only query
        if "menuItemByText" in script:
            return None
        return self.attachments  # the attachment count

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


def test_a_click_that_opened_the_menu_and_timed_out_still_uploads():
    """The menu opened on pointerdown and the bounded click timed out on the modal open."""
    page = _Page(click_opens_menu = True, chooser_opens = True)
    result = _upload(page)

    assert result.ran is True, result.reason
    assert result.expect_ok is True
    assert result.expect["click_overran"] is True
    assert page.attachments == 1
    assert page.menus == []
    assert page.keyboard.pressed == []


def test_a_click_that_opened_the_menu_and_timed_out_closes_it():
    page = _Page(click_opens_menu = True)
    result = _upload(page)

    assert result.ran is False
    assert "the file chooser never opened" in result.reason
    assert page.menus == [], "the modal menu was left open over the next action"
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
    page = _Page(click_opens_menu = False, already_open = ("model-picker",))
    result = _upload(page)

    assert result.ran is False
    assert page.keyboard.pressed == []
    assert page.menus == ["model-picker"]
    assert result.reason == "the attachments button could not be clicked: TimeoutError"


def test_a_menu_that_replaced_an_open_one_is_still_closed():
    """A non-modal menu lets the click through and closes, and the attachments menu opens in its
    place: something was open before and something is open after, and they are not the same menu."""
    page = _Page(
        click_opens_menu = True,
        already_open = ("model-picker",),
        click_dismisses_open_menu = True,
    )
    result = _upload(page)

    assert result.ran is False
    assert page.menus == [], "the attachments menu that replaced the old one was left open"
    assert "closed the menu it opened" in result.reason


def test_only_the_menu_this_attempt_opened_is_closed():
    """Escape takes the top layer, which is the new one, and stops before an older one below it."""
    page = _Page(click_opens_menu = True, already_open = ("model-picker",))
    result = _upload(page)

    assert result.ran is False
    assert page.keyboard.pressed == ["Escape"]
    assert page.menus == ["model-picker"]
