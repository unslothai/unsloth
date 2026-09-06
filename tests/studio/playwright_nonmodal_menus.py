#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Browser checks for NonModalDropdownMenu, driven against smoke-nonmodal-menus.html.

The AST tests pin which menus use the wrapper. Only a browser can answer what the wrapper
then does: whether a scroll behind an open menu closes it, whether the list keeps the scroll
it was given, whether the dismissing click is swallowed exactly once, and whether focus comes
back to the trigger that opened it. Each of those has already regressed once (#9243, #9772,
and twice inside the change that added the wrapper), and none is reachable from source text.

The page also mounts one UNCONVERTED modal menu. It is the shape every converted menu had
before, so one run measures both: a case that behaves identically on both is Radix and not
this wrapper, and the lock the control still writes to <body> is what proves the probe can
see a lock at all.

    python3 tests/studio/playwright_nonmodal_menus.py
    PW_ENGINE=webkit python3 tests/studio/playwright_nonmodal_menus.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

ENGINE = os.environ.get("PW_ENGINE", "chromium")
PORT = int(os.environ.get("PW_PORT", "5401"))
OUT = Path(os.environ.get("PW_OUT", "logs/nonmodal_menus_report.json"))
ENTRY = "/smoke-nonmodal-menus-main.tsx"
PAGE = "/smoke-nonmodal-menus.html"

# Radix holds exit-animated content mounted for `duration-100`, so probe both sides of that
# window: a guard that outlives the close swallows whatever the user clicks next.
EXIT_WINDOW_MS = (30, 75, 125)


class Checks:
    def __init__(self) -> None:
        self.results: list[dict[str, object]] = []

    def record(self, name: str, ok: bool, detail: object = "") -> None:
        self.results.append({"case": name, "ok": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}", flush = True)

    @property
    def failed(self) -> list[dict[str, object]]:
        return [r for r in self.results if not r["ok"]]


def menus_open(page) -> int:
    return page.locator("[role=menu]").count()


def state(page) -> dict:
    return page.evaluate("() => window.probe.documentState()")


def open_row(page, row: int) -> None:
    page.get_by_label(f"Row options {row}", exact = True).click()
    page.wait_for_selector("[role=menu]", state = "visible")


def open_control(page) -> None:
    page.get_by_label("Control options", exact = True).click()
    page.wait_for_selector("[role=menu]", state = "visible")


def raw_click(page, selector: str) -> None:
    """A real pointer press at the element's centre, with no actionability wait.

    A modal menu puts `pointer-events: none` on <body>, so Playwright's own click refuses to
    act there ("<html> intercepts pointer events"). The control arm has to bypass its own
    actionability check to be measurable at all.
    """
    box = page.locator(selector).bounding_box()
    if box is None:
        raise RuntimeError(f"{selector} has no box")
    page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)


def close_all(page) -> None:
    for _ in range(6):
        if menus_open(page) == 0:
            return
        page.keyboard.press("Escape")
        page.wait_for_timeout(200)


def reset_list(page) -> None:
    page.evaluate("() => { document.getElementById('list').scrollTop = 0; }")
    page.wait_for_timeout(250)


def check_document_is_untouched(page, checks: Checks) -> None:
    baseline = state(page)
    open_row(page, 0)
    opened = state(page)
    checks.record(
        "opening a converted menu writes nothing to <body>",
        opened["bodyPointerEvents"] == "" and not opened["scrollLocked"],
        opened,
    )
    checks.record(
        "opening a converted menu aria-hides nothing new",
        opened["ariaHidden"] <= baseline["ariaHidden"],
        f"{baseline['ariaHidden']} -> {opened['ariaHidden']}",
    )
    close_all(page)

    open_control(page)
    control = state(page)
    checks.record(
        "the unconverted menu still locks the document, so the probe can see a lock",
        control["bodyPointerEvents"] == "none"
        and control["scrollLocked"]
        and control["ariaHidden"] > 0,
        control,
    )
    close_all(page)


def check_scroll_closes(page, checks: Checks) -> None:
    reset_list(page)
    open_row(page, 1)
    page.mouse.move(120, 300)
    page.mouse.wheel(0, 400)
    page.wait_for_timeout(400)
    scrolled = page.evaluate("() => document.getElementById('list').scrollTop")
    checks.record("a wheel behind an open menu closes it", menus_open(page) == 0)
    checks.record("the list keeps the scroll the wheel gave it", scrolled > 0, scrolled)
    page.wait_for_timeout(300)
    settled = page.evaluate("() => document.getElementById('list').scrollTop")
    checks.record(
        "closing does not snap the list back to the trigger",
        settled == scrolled,
        f"{scrolled} -> {settled}",
    )

    reset_list(page)
    open_row(page, 2)
    page.evaluate("() => { document.getElementById('list').scrollTop = 500; }")
    page.wait_for_timeout(400)
    checks.record("a programmatic scroll closes the menu", menus_open(page) == 0)

    reset_list(page)
    open_row(page, 3)
    page.evaluate(
        "() => document.getElementById('list').scrollBy({top: 200, behavior: 'smooth'})"
    )
    page.wait_for_timeout(600)
    checks.record("a smooth scroll closes the menu", menus_open(page) == 0)

    reset_list(page)
    open_row(page, 6)
    page.evaluate("() => window.scrollTo(0, 200)")
    page.wait_for_timeout(400)
    checks.record("a window scroll closes the menu", menus_open(page) == 0)
    page.evaluate("() => window.scrollTo(0, 0)")


def check_scrolls_that_must_not_close(page, checks: Checks) -> None:
    reset_list(page)
    page.get_by_label("Tall menu", exact = True).click()
    page.wait_for_selector("[role=menu]", state = "visible")
    page.evaluate(
        "() => { const v = document.querySelector('[data-slot=dropdown-menu-viewport]');"
        " v.scrollTop = 120; }"
    )
    page.wait_for_timeout(400)
    checks.record("scrolling the menu's own viewport leaves it open", menus_open(page) >= 1)
    close_all(page)

    open_row(page, 4)
    page.evaluate("() => { document.getElementById('other').scrollTop = 300; }")
    page.wait_for_timeout(400)
    checks.record("scrolling an unrelated list leaves the menu open", menus_open(page) >= 1)
    close_all(page)

    open_row(page, 5)
    page.get_by_test_id("export-5").click()
    page.wait_for_timeout(400)
    with_sub = menus_open(page)
    page.evaluate(
        "() => { const vs = document.querySelectorAll('[data-slot=dropdown-menu-viewport]');"
        " vs[vs.length - 1].scrollTop = 100; }"
    )
    page.wait_for_timeout(400)
    checks.record(
        "scrolling an open submenu leaves both menus open",
        with_sub >= 2 and menus_open(page) == with_sub,
        f"{with_sub} -> {menus_open(page)}",
    )
    close_all(page)


def check_dismiss_guard(page, checks: Checks) -> None:
    # Reset the counter AFTER opening: the click that opens a menu reaches document too.
    open_row(page, 7)
    page.evaluate("() => window.probe.resetClicks()")
    raw_click(page, "#outside-button")
    page.wait_for_timeout(300)
    swallowed = page.evaluate("() => window.probe.documentClicks()")
    checks.record(
        "the click that dismisses a menu does not reach what is under it",
        menus_open(page) == 0 and swallowed == 0,
        f"clicks={swallowed}",
    )
    page.evaluate("() => window.probe.resetClicks()")
    raw_click(page, "#outside-button")
    page.wait_for_timeout(200)
    delivered = page.evaluate("() => window.probe.documentClicks()")
    checks.record("the click after that one is delivered", delivered == 1, delivered)

    for label, opener in (
        ("a converted menu", lambda: open_row(page, 8)),
        ("the unconverted control", lambda: open_control(page)),
    ):
        for delay in EXIT_WINDOW_MS:
            opener()
            page.keyboard.press("Escape")
            page.wait_for_timeout(delay)
            page.evaluate("() => window.probe.resetClicks()")
            raw_click(page, "#outside-button")
            page.wait_for_timeout(250)
            landed = page.evaluate("() => window.probe.documentClicks()")
            checks.record(
                f"a click {delay}ms after closing {label} is delivered", landed == 1, landed
            )
            close_all(page)

    # Radix returns focus to the trigger on close, but a non-modal menu lets the press reach
    # the field first, so Firefox and WebKit leave the caret there and Chromium does not.
    # Either is usable; focus falling to <body> would not be.
    focus_after = {}
    for label, opener in (
        ("converted", lambda: open_row(page, 9)),
        ("control", lambda: open_control(page)),
    ):
        opener()
        raw_click(page, "#outside-input")
        page.wait_for_timeout(700)
        focus_after[label] = page.evaluate(
            "() => document.activeElement?.getAttribute('data-slot')"
            " ?? document.activeElement?.id ?? null"
        )
        close_all(page)
    checks.record(
        "dismissing into an input never drops focus onto the document",
        focus_after["converted"] in {"outside-input", "dropdown-menu-trigger"},
        focus_after,
    )


def check_focus_return(page, checks: Checks) -> None:
    active = "() => document.activeElement?.getAttribute('aria-label')"
    close_all(page)
    reset_list(page)
    open_row(page, 10)
    page.keyboard.press("Escape")
    page.wait_for_timeout(300)
    checks.record(
        "Escape returns focus to the trigger that opened the menu",
        page.evaluate(active) == "Row options 10",
        page.evaluate(active),
    )

    # A scroll-close takes a different focus path. The flag that selects it must not survive
    # into the next close, or an ordinary Escape stops restoring focus.
    reset_list(page)
    open_row(page, 11)
    page.evaluate("() => { document.getElementById('list').scrollTop = 300; }")
    page.wait_for_timeout(400)
    reset_list(page)
    open_row(page, 12)
    page.keyboard.press("Escape")
    page.wait_for_timeout(300)
    checks.record(
        "a scroll-close does not change how the next Escape restores focus",
        page.evaluate(active) == "Row options 12",
        page.evaluate(active),
    )

    reset_list(page)
    open_row(page, 13)
    page.evaluate("() => { document.getElementById('list').scrollTop = 600; }")
    page.wait_for_timeout(500)
    checks.record(
        "a scroll-close keeps the scroll it was closed by",
        page.evaluate("() => document.getElementById('list').scrollTop") == 600,
        page.evaluate("() => document.getElementById('list').scrollTop"),
    )


def check_keyboard(page, checks: Checks) -> None:
    # Focus a trigger that is already on screen and let the focus settle. Focusing an
    # off-screen one scrolls it into view and that scroll arrives a frame or two later,
    # which races the open rather than testing it.
    reset_list(page)
    page.get_by_label("Row options 2", exact = True).focus()
    page.wait_for_timeout(250)
    page.keyboard.press("Enter")
    page.wait_for_timeout(300)
    checks.record("Enter opens the menu", menus_open(page) >= 1)
    page.keyboard.press("ArrowDown")
    page.keyboard.press("ArrowRight")
    page.wait_for_timeout(400)
    checks.record("ArrowRight opens the submenu", menus_open(page) >= 2, menus_open(page))
    page.keyboard.press("ArrowLeft")
    page.wait_for_timeout(400)
    checks.record("ArrowLeft closes only the submenu", menus_open(page) == 1, menus_open(page))
    close_all(page)

    page.get_by_label("Row options 3", exact = True).focus()
    page.wait_for_timeout(250)
    page.keyboard.press("Space")
    page.wait_for_timeout(300)
    checks.record("Space opens the menu", menus_open(page) >= 1)
    close_all(page)


def check_lifetime(page, checks: Checks) -> None:
    reset_list(page)
    open_row(page, 16)
    page.evaluate("() => window.probe.removeRow(16)")
    page.wait_for_timeout(500)
    checks.record(
        "deleting the row while its menu is open strands no menu",
        menus_open(page) == 0,
        menus_open(page),
    )
    page.evaluate("() => window.probe.reset()")
    page.wait_for_timeout(300)

    reopened: dict[str, int] = {}
    reset_list(page)
    for label, opener, name in (
        ("a converted menu", lambda: open_row(page, 17), "Row options 17"),
        ("the unconverted control", lambda: open_control(page), "Control options"),
    ):
        opener()
        page.keyboard.press("Escape")
        page.wait_for_timeout(30)
        page.get_by_label(name, exact = True).click()
        page.wait_for_timeout(500)
        reopened[label] = menus_open(page)
        close_all(page)
    checks.record(
        "reopening during the exit animation behaves as the unconverted menu does",
        len(set(reopened.values())) == 1,
        reopened,
    )

    # The scroll watcher is per-open. If a close ever failed to retire one, this climbs.
    before = page.evaluate("() => window.probe.scrollListeners()")
    for _ in range(30):
        page.get_by_label("Row options 18", exact = True).click()
        page.wait_for_timeout(60)
        page.keyboard.press("Escape")
        page.wait_for_timeout(60)
    close_all(page)
    page.wait_for_timeout(400)
    after = page.evaluate("() => window.probe.scrollListeners()")
    checks.record(
        "30 open and close cycles leak no scroll listener", after == before, f"{before} -> {after}"
    )

    page.evaluate(
        "() => { const l = document.getElementById('list');"
        " l.scrollTop = 8 * 40 - l.clientHeight + 20; }"
    )
    page.wait_for_timeout(300)
    page.get_by_label("Row options 8", exact = True).click(force = True)
    page.wait_for_timeout(500)
    checks.record(
        "a partly visible trigger opens a menu that stays open",
        menus_open(page) >= 1,
        menus_open(page),
    )
    close_all(page)

    reset_list(page)
    open_row(page, 19)
    page.get_by_label("Row options 20", exact = True).click()
    page.wait_for_timeout(400)
    checks.record(
        "clicking another row's trigger does not leave two menus open",
        menus_open(page) <= 1,
        menus_open(page),
    )
    close_all(page)


def main() -> int:
    vite = start_vite(PORT)
    checks = Checks()
    try:
        url = f"http://127.0.0.1:{PORT}{PAGE}"
        wait_for_smoke_page(url, ENTRY, proc = vite, info = lambda m: print(m, flush = True))
        with sync_playwright() as pw:
            kwargs: dict = {"headless": True}
            if ENGINE == "chromium":
                kwargs["args"] = chromium_launch_args()
            browser = getattr(pw, ENGINE).launch(**kwargs)
            page = browser.new_context(viewport = {"width": 1280, "height": 800}).new_page()
            page.goto(url, wait_until = "domcontentloaded")
            page.wait_for_function("() => Boolean(window.probe)")
            checks.record(
                "the engine honours focus({preventScroll})",
                page.evaluate("() => window.probe.supportsPreventScroll()"),
                ENGINE,
            )
            check_document_is_untouched(page, checks)
            check_scroll_closes(page, checks)
            check_scrolls_that_must_not_close(page, checks)
            check_dismiss_guard(page, checks)
            check_focus_return(page, checks)
            check_keyboard(page, checks)
            check_lifetime(page, checks)
            browser.close()
    finally:
        stop_process(vite)

    passed = len(checks.results) - len(checks.failed)
    print(f"\n{ENGINE}: {passed}/{len(checks.results)} passed", flush = True)
    OUT.parent.mkdir(parents = True, exist_ok = True)
    OUT.write_text(
        json.dumps({"engine": ENGINE, "results": checks.results}, indent = 2), encoding = "utf8"
    )
    return 1 if checks.failed else 0


if __name__ == "__main__":
    sys.exit(main())
