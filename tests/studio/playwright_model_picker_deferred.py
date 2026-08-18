# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The On Device rows defer their tooltips and their dots menu, and still behave like rows.

The picker renders every cached repo, and each row used to mount three Radix tooltips and a Radix
dropdown whether or not anything had ever pointed at it. None of the four is on screen until the
row is hovered or focused, so they are now mounted on first contact instead. That is invisible when
it works and invisible when it breaks, which is what this file is for.

Each check here fails on its own mutant (see logs/nextwins/picker.md for which mutant turned which
assertion red). The interesting ones are not "the tooltip exists" but:

  * ONE mouse move. Radix opens a tooltip on `pointermove`, not on `pointerenter`, so a pointer
    that enters a row and then holds still fires nothing more. The freshly mounted trigger would
    never learn the pointer is on it. The row replays one synthetic move; `page.mouse.move` is
    exactly the parked-pointer case, one move and no more.
  * FOCUS SURVIVES. Activating a row replaces the element that has focus. Without the replay, a Tab
    into the list drops focus to <body> and the whole list stops being keyboard reachable.
  * A COLD CLICK STILL OPENS THE MENU. The dots placeholder is a real button; a press on it has to
    end with the menu open, not with a swallowed gesture.

Run:
    python tests/studio/playwright_model_picker_deferred.py
    SMOKE_BASE_URL=http://127.0.0.1:5547 python tests/studio/playwright_model_picker_deferred.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

PORT = int(os.environ.get("SMOKE_PORT", "5548"))
_EXTERNAL = os.environ.get("SMOKE_BASE_URL", "").strip().rstrip("/")
BASE = _EXTERNAL or f"http://127.0.0.1:{PORT}"
OWNS_SERVER = not _EXTERNAL
MODELS = int(os.environ.get("SMOKE_PICKER_MODELS", "200"))
# The row tooltip opens after 700ms; give the frame budget room on a loaded host without making the
# wait itself the assertion.
ROW_TOOLTIP_WAIT_MS = int(os.environ.get("SMOKE_ROW_TOOLTIP_WAIT_MS", "2500"))

FAILURES: list[str] = []


def info(message: str) -> None:
    print(f"[picker-deferred] {message}", flush = True)


def check(name: str, condition: bool, detail: str) -> None:
    if condition:
        info(f"PASS {name}: {detail}")
        return
    FAILURES.append(f"{name}: {detail}")
    info(f"FAIL {name}: {detail}")


ROW_FACTS_JS = """
(index) => {
  const panel = document.querySelector(".unsloth-model-selector-menu");
  if (!panel) return null;
  const options = panel.querySelectorAll("[data-model-picker-option]");
  const option = options[index];
  if (!option) return null;
  // The row shell is the option's grandparent: <shell><div.min-w-0><button option>.
  const shell = option.closest("[data-model-picker-option]")?.parentElement?.parentElement ?? null;
  const menuButton = shell ? shell.querySelector('[aria-label^="More options"]') : null;
  return {
    rows: options.length,
    panelTooltipTriggers: panel.querySelectorAll('[data-slot="tooltip-trigger"]').length,
    panelMenuTriggers: panel.querySelectorAll('[data-slot="dropdown-menu-trigger"]').length,
    rowTooltipTriggers: shell
      ? shell.querySelectorAll('[data-slot="tooltip-trigger"]').length
      : -1,
    optionLabel: option.textContent.trim().slice(0, 80),
    optionRect: option.getBoundingClientRect().toJSON(),
    formatDotRect: (() => {
      const dot = shell ? shell.querySelector('[aria-label*="afetensors"], [aria-label="GGUF"]') : null;
      return dot ? dot.getBoundingClientRect().toJSON() : null;
    })(),
    menuButtonAria: menuButton
      ? {
          label: menuButton.getAttribute("aria-label"),
          haspopup: menuButton.getAttribute("aria-haspopup"),
          expanded: menuButton.getAttribute("aria-expanded"),
          state: menuButton.getAttribute("data-state"),
          slot: menuButton.getAttribute("data-slot"),
          tag: menuButton.tagName,
          rect: menuButton.getBoundingClientRect().toJSON(),
        }
      : null,
    activeElement: document.activeElement
      ? {
          tag: document.activeElement.tagName,
          option: document.activeElement.hasAttribute("data-model-picker-option"),
          label: (document.activeElement.getAttribute("aria-label") || "").slice(0, 60),
          insideShell: shell ? shell.contains(document.activeElement) : false,
        }
      : null,
  };
}
"""

TOOLTIP_TEXT_JS = """
() =>
  Array.from(document.querySelectorAll('[data-slot="tooltip-content"]'))
    .map((el) => el.textContent.trim())
    .filter(Boolean)
"""

MENU_ITEMS_JS = """
() =>
  Array.from(document.querySelectorAll('[data-slot="dropdown-menu-content"] [role="menuitem"]'))
    .map((el) => el.textContent.trim())
"""


def open_on_device(page: Page) -> None:
    page.evaluate("(n) => window.__pickerScale.seed(n)", MODELS)
    page.evaluate("window.__pickerScale.setOpen(true)")
    page.wait_for_function("() => window.__pickerScale.onDeviceTab() !== null", timeout = 180_000)
    page.evaluate("() => window.__pickerScale.onDeviceTab().click()")
    page.wait_for_function(
        "(n) => window.__pickerScale.counts().rows >= n", arg = MODELS, timeout = 180_000
    )


def centre(rect: dict) -> tuple[float, float]:
    return rect["x"] + rect["width"] / 2, rect["y"] + rect["height"] / 2


def run(page: Page) -> None:
    open_on_device(page)

    facts = page.evaluate(ROW_FACTS_JS, 0)
    check(
        "rows rendered",
        facts is not None and facts["rows"] >= MODELS,
        f"{facts['rows'] if facts else 'no panel'} rows for {MODELS} models",
    )
    # 1. The deferral is actually on. Every row still carries its dots button, so the menu trigger
    #    count must NOT fall: what the fix removes is Radix machinery, not markup.
    check(
        "cold list mounts no row tooltips",
        facts["panelTooltipTriggers"] <= 8,
        f"{facts['panelTooltipTriggers']} tooltip triggers in a panel of {facts['rows']} rows "
        "(the picker's own chrome only; a row that mounted its tooltips would add three each)",
    )
    check(
        "cold list still renders every dots button",
        facts["panelMenuTriggers"] >= facts["rows"],
        f"{facts['panelMenuTriggers']} dots buttons for {facts['rows']} rows",
    )
    # 2. The a11y tree of a cold row is the a11y tree of a Radix trigger.
    aria = facts["menuButtonAria"]
    check(
        "cold dots button is a closed menu button",
        aria is not None
        and aria["tag"] == "BUTTON"
        and aria["haspopup"] == "menu"
        and aria["expanded"] == "false"
        and aria["state"] == "closed"
        and (aria["label"] or "").startswith("More options"),
        f"{aria}",
    )

    # 3. ONE mouse move onto the format dot has to open its tooltip. This is the parked-pointer
    #    case: Playwright moves once, so a row that mounted its trigger without replaying the move
    #    would sit there closed forever.
    dot = facts["formatDotRect"]
    check("row 0 has a format dot to hover", dot is not None, f"{dot}")
    if dot:
        page.mouse.move(*centre(dot))
        page.wait_for_timeout(400)
        texts = page.evaluate(TOOLTIP_TEXT_JS)
        # Substring, not equality: Radix renders the label twice, once visibly and once in the
        # visually-hidden copy screen readers announce, so `textContent` reads
        # "SafetensorsSafetensors". An equality predicate here reported a tooltip that had opened
        # perfectly as never having opened.
        check(
            "one mouse move opens the format tooltip",
            any(any(f in t for f in ("GGUF", "Safetensors", "MLX")) for t in texts),
            f"tooltip contents after a single move: {texts}",
        )
        hovered = page.evaluate(ROW_FACTS_JS, 0)
        check(
            "hovering mounts the row's own tooltips",
            hovered["rowTooltipTriggers"] >= 2,
            f"{hovered['rowTooltipTriggers']} tooltip triggers in the hovered row",
        )
        check(
            "hovering does not wake the rest of the list",
            hovered["panelTooltipTriggers"] < facts["rows"],
            f"{hovered['panelTooltipTriggers']} tooltip triggers in the whole panel",
        )

    # 4. And the row's own 700ms tooltip, from the same single move.
    page.mouse.move(0, 0)
    page.wait_for_timeout(300)
    row1 = page.evaluate(ROW_FACTS_JS, 1)
    page.mouse.move(*centre(row1["optionRect"]))
    page.wait_for_timeout(ROW_TOOLTIP_WAIT_MS)
    texts = page.evaluate(TOOLTIP_TEXT_JS)
    check(
        "one mouse move opens the row tooltip",
        any(t for t in texts if "/" in t or "Local" in t),
        f"tooltip contents after a single move onto the row: {texts}",
    )

    # 5. Keyboard: focus a cold row and keep it. Activation replaces the focused element, so
    #    without the replay this lands on <body> and the list stops being keyboard reachable.
    page.mouse.move(0, 0)
    page.wait_for_timeout(200)
    row5 = page.evaluate(ROW_FACTS_JS, 5)
    page.evaluate(
        """(index) => {
            const panel = document.querySelector(".unsloth-model-selector-menu");
            const option = panel.querySelectorAll("[data-model-picker-option]")[index];
            option.focus();
        }""",
        5,
    )
    page.wait_for_timeout(200)
    focused = page.evaluate(ROW_FACTS_JS, 5)
    check(
        "focus survives activating a row",
        focused["activeElement"] is not None
        and focused["activeElement"]["option"]
        and focused["activeElement"]["insideShell"],
        f"active element after focusing row 5: {focused['activeElement']}",
    )
    check(
        "focusing a row activates it",
        focused["rowTooltipTriggers"] >= 2 and row5["rowTooltipTriggers"] == 0,
        f"row 5 carried {row5['rowTooltipTriggers']} tooltip triggers cold and "
        f"{focused['rowTooltipTriggers']} once focused",
    )
    # Tab from the focused row must reach that row's own action buttons, in order.
    page.keyboard.press("Tab")
    page.wait_for_timeout(150)
    after_tab = page.evaluate(ROW_FACTS_JS, 5)
    check(
        "tab from a row reaches its own action buttons",
        after_tab["activeElement"] is not None and after_tab["activeElement"]["insideShell"],
        f"active element after Tab: {after_tab['activeElement']}",
    )

    # 6. A cold row's dots button, pressed without ever being hovered, opens the menu.
    cold = page.evaluate(ROW_FACTS_JS, 20)
    check(
        "row 20 is still cold",
        cold["rowTooltipTriggers"] == 0,
        f"{cold['rowTooltipTriggers']} tooltip triggers before the press",
    )
    page.evaluate(
        """(index) => {
            const panel = document.querySelector(".unsloth-model-selector-menu");
            const option = panel.querySelectorAll("[data-model-picker-option]")[index];
            const shell = option.parentElement.parentElement;
            shell.querySelector('[aria-label^="More options"]').click();
        }""",
        20,
    )
    page.wait_for_timeout(600)
    items = page.evaluate(MENU_ITEMS_JS)
    check(
        "a cold dots button opens the menu",
        len(items) >= 2,
        f"menu items after clicking a never-hovered dots button: {items}",
    )
    page.keyboard.press("Escape")
    page.wait_for_timeout(300)

    # The synthetic press above is not a gesture a user can make (the button it hits is under
    # `opacity-0` until the row is hovered), and it leaves a dismissed menu and a moved focus
    # behind. Reopen so the selection checks below start from a panel in a known state, and say so
    # if that reopen did not happen: a stale panel would make the next two checks measure the
    # leftovers of this one.
    page.evaluate("window.__pickerScale.setOpen(false)")
    page.wait_for_timeout(400)
    open_on_device(page)
    fresh = page.evaluate(ROW_FACTS_JS, 3)
    check(
        "the panel is open and cold before the selection checks",
        fresh is not None and fresh["rows"] >= MODELS and fresh["rowTooltipTriggers"] == 0,
        f"row 3 on the reopened panel: {None if fresh is None else fresh['rowTooltipTriggers']} "
        f"tooltip triggers, {None if fresh is None else fresh['rows']} rows",
    )

    # 7. And a row still selects when clicked, in ONE gesture.
    #
    # `mouse.click` is move + down + up with nothing in between, which is what a real click is and
    # what a wait between the hover and the press hides. Activating a row replaces the button, so a
    # swap deferred to after the mousedown puts the mouseup on a different element and the browser
    # fires no click at all. This assertion caught exactly that, and the row now activates
    # synchronously; on the merge base it passes because nothing is ever swapped.
    page.mouse.move(0, 0)
    page.wait_for_timeout(300)
    target = page.evaluate(ROW_FACTS_JS, 3)
    page.mouse.click(*centre(target["optionRect"]))
    page.wait_for_timeout(600)
    selected = page.evaluate(
        """() => ({
            open: window.__pickerScale.isOpen(),
            panel: Boolean(window.__pickerScale.panel()),
            trigger: (window.__pickerScale.trigger()?.textContent || '').trim(),
        })"""
    )
    leaf = (target["menuButtonAria"]["label"] or "").replace("More options for ", "")
    check(
        "clicking a row in one gesture still selects it",
        leaf in selected["trigger"] and not selected["open"] and not selected["panel"],
        f"after clicking {leaf!r} the picker reads {selected}",
    )

    # 8. And Enter on a focused row selects it, which is the same path with no pointer at all.
    page.evaluate("window.__pickerScale.setOpen(false)")
    page.wait_for_timeout(400)
    open_on_device(page)
    page.evaluate(
        """() => {
            const panel = document.querySelector(".unsloth-model-selector-menu");
            panel.querySelectorAll("[data-model-picker-option]")[7].focus();
        }"""
    )
    page.wait_for_timeout(400)
    keyboard_target = page.evaluate(ROW_FACTS_JS, 7)
    page.keyboard.press("Enter")
    page.wait_for_timeout(600)
    keyboard_selected = page.evaluate(
        """() => ({
            open: window.__pickerScale.isOpen(),
            trigger: (window.__pickerScale.trigger()?.textContent || '').trim(),
        })"""
    )
    # The repo id from the row's own dots button, not from the option's textContent: that text runs
    # the name straight into the param and size chips ("Yi-3B-Math-343B35GB"), so a slice of it is
    # not a substring of anything and the check failed on a selection that had worked. It failed on
    # the merge base too, which is what said the predicate was wrong rather than the tree.
    keyboard_leaf = (keyboard_target["menuButtonAria"]["label"] or "").replace(
        "More options for ", ""
    )
    check(
        "pressing Enter on a focused row still selects it",
        keyboard_leaf in keyboard_selected["trigger"] and not keyboard_selected["open"],
        f"after Enter on {keyboard_leaf!r} the picker reads {keyboard_selected}",
    )


def run_touch(page: Page) -> None:
    """A coarse pointer opts out of the deferral entirely, so it renders the merge base's tree.

    Those devices show the row actions at all times (`[@media(hover:none)]:opacity-100`), have no
    hover to trade on, and a tap is a pointerdown and a click on the SAME node -- swapping a
    subtree under a finger would move the click's target out from under it. So the answer there is
    to change nothing, and this is the check that it really changed nothing.
    """
    coarse = page.evaluate(
        """() => ({
            coarse: window.matchMedia("(pointer: coarse)").matches,
            noHover: window.matchMedia("(hover: none)").matches,
        })"""
    )
    # Without this the rest of the section is vacuous: it would be asserting the desktop path twice.
    check(
        "the touch context really reports a coarse pointer",
        coarse["coarse"] or coarse["noHover"],
        f"{coarse}",
    )
    open_on_device(page)
    facts = page.evaluate(ROW_FACTS_JS, 0)
    check(
        "a coarse pointer mounts every row's tooltips up front",
        facts["panelTooltipTriggers"] >= facts["rows"] * 3,
        f"{facts['panelTooltipTriggers']} tooltip triggers for {facts['rows']} rows "
        "(three per row is the merge base's tree)",
    )
    check(
        "a coarse pointer mounts every row's dots menu up front",
        facts["rowTooltipTriggers"] >= 3,
        f"row 0 carries {facts['rowTooltipTriggers']} tooltip triggers with no interaction at all",
    )
    menu_rect = facts["menuButtonAria"]["rect"]
    page.touchscreen.tap(
        menu_rect["x"] + menu_rect["width"] / 2, menu_rect["y"] + menu_rect["height"] / 2
    )
    page.wait_for_timeout(800)
    items = page.evaluate(MENU_ITEMS_JS)
    check(
        "tapping the dots opens the menu on a touch device",
        len(items) >= 2,
        f"menu items after a tap: {items}",
    )


def main() -> int:
    vite = None
    if OWNS_SERVER:
        info(f"starting vite dev server on port {PORT}")
        vite = start_vite(PORT)
    errors: list[str] = []
    try:
        wait_for_smoke_page(
            f"{BASE}/smoke-model-picker-scale.html",
            "smoke-model-picker-scale-main.tsx",
            proc = vite,
            info = info,
        )
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless = os.environ.get("SMOKE_HEADLESS", "1") == "1",
                args = chromium_launch_args(),
            )
            context = browser.new_context(viewport = {"width": 1440, "height": 900})
            page = context.new_page()
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"[:200]))
            page.on(
                "console",
                lambda m: errors.append(f"console error: {m.text[:160]}")
                if m.type == "error"
                else None,
            )
            page.goto(f"{BASE}/smoke-model-picker-scale.html", wait_until = "domcontentloaded")
            page.wait_for_function("() => Boolean(window.__pickerScale)", timeout = 120_000)
            run(page)
            context.close()

            touch_context = browser.new_context(
                viewport = {"width": 1440, "height": 900}, has_touch = True, is_mobile = False
            )
            touch_page = touch_context.new_page()
            touch_page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"[:200]))
            touch_page.on(
                "console",
                lambda m: errors.append(f"console error: {m.text[:160]}")
                if m.type == "error"
                else None,
            )
            touch_page.goto(f"{BASE}/smoke-model-picker-scale.html", wait_until = "domcontentloaded")
            touch_page.wait_for_function("() => Boolean(window.__pickerScale)", timeout = 120_000)
            run_touch(touch_page)
            touch_context.close()
            browser.close()
    finally:
        if vite is not None:
            stop_process(vite)
            info("vite stopped")
    check("no page errors", not errors, f"{errors[:3]}")
    if FAILURES:
        for failure in FAILURES:
            info(f"FAILED {failure}")
        return 1
    info("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
