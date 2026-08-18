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
    print(f"[picker-deferred] {message}", flush=True)


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
    page.wait_for_function("() => window.__pickerScale.onDeviceTab() !== null", timeout=180_000)
    page.evaluate("() => window.__pickerScale.onDeviceTab().click()")
    page.wait_for_function(
        "(n) => window.__pickerScale.counts().rows >= n", arg=MODELS, timeout=180_000
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
        check(
            "one mouse move opens the format tooltip",
            any(t in ("GGUF", "Safetensors", "MLX") for t in texts),
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

    # 7. And a row still selects when clicked.
    target = page.evaluate(ROW_FACTS_JS, 3)
    page.mouse.click(*centre(target["optionRect"]))
    page.wait_for_timeout(400)
    trigger_text = page.evaluate(
        "() => (window.__pickerScale.trigger()?.textContent || '').trim()"
    )
    check(
        "clicking a row still selects it",
        trigger_text != "" and trigger_text.split("/")[-1][:8] in target["optionLabel"],
        f"picker trigger reads {trigger_text!r} after clicking {target['optionLabel']!r}",
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
            proc=vite,
            info=info,
        )
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=os.environ.get("SMOKE_HEADLESS", "1") == "1",
                args=chromium_launch_args(),
            )
            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()
            page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"[:200]))
            page.on(
                "console",
                lambda m: errors.append(f"console error: {m.text[:160]}")
                if m.type == "error"
                else None,
            )
            page.goto(f"{BASE}/smoke-model-picker-scale.html", wait_until="domcontentloaded")
            page.wait_for_function("() => Boolean(window.__pickerScale)", timeout=120_000)
            run(page)
            context.close()
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
