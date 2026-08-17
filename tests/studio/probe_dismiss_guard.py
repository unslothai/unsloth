# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Does `swallowDismissingClick` actually stop the click that dismisses a non-modal menu?

`probe_menu_behaviour.py` asks the destructive question once, with `page.mouse.click`,
which is a press and a release in the same tick. That is the easy case and the guard
passes it. This probe asks the same question three more ways, because the guard is a
`once`-armed document capture listener that a 300 ms timer disarms:

  quick    press and release immediately. The control case: the guard should hold.
  held     press, wait longer than the 300 ms window, release. The timer has already
           removed the listener by the time the browser synthesises `click`, so the
           click lands on whatever is underneath. Underneath is the unconfirmed
           "Delete message" button, two buttons from the trigger.
  busy     press, block the main thread past 300 ms, release. Same outcome from a
           normal-length press, which is the version a user cannot avoid.
  touch    tap. Radix defers `onPointerDownOutside` to the resulting `click` when
           `pointerType === "touch"` (react-dismissable-layer 1.1.11, usePointerDownOutside),
           and it listens on `ownerDocument` in the BUBBLE phase. React 19 delegates to
           the root container, which is inside document, so the control's own onClick has
           already run by the time the guard is armed. The guard cannot swallow the click
           it was armed by.

Every measurement is "did the assistant message count go down", i.e. did the message get
deleted. Clicks go through `page.mouse` / `page.touchscreen`, real hit tests that honour
pointer-events; `locator.click()` throws on interception and `element.click()` skips hit
testing, and each would lie in a different direction.

Run against the PR head AND against the merge base. On the merge base these menus are
modal, the body carries pointer-events:none, and no variant can reach Delete at all.

Usage:  python tests/studio/probe_dismiss_guard.py --label head --engine chromium
Writes logs/pw/dismiss_guard_<label>_<engine>.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from playwright_heavy_thread import (  # noqa: E402
    BASE,
    OWNS_SERVER,
    PORT,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

CHARS = int(os.environ.get("PROBE_CHARS", "25000"))
HOLD_MS = int(os.environ.get("PROBE_HOLD_MS", "600"))

OPEN_MENU_JS = """
async () => {
  const api = window.__heavyThread;
  const trigger = api.actionButton("More");
  if (!trigger) return { ok: false, why: "no More trigger" };
  const pointer = {
    bubbles: true, cancelable: true, composed: true,
    button: 0, pointerId: 1, pointerType: "mouse", isPrimary: true,
  };
  trigger.dispatchEvent(new PointerEvent("pointerdown", { ...pointer, buttons: 1 }));
  trigger.dispatchEvent(new PointerEvent("pointerup", { ...pointer, buttons: 0 }));
  const deadline = performance.now() + 15000;
  while (performance.now() < deadline) {
    if (document.querySelector(".aui-action-bar-more-content")) break;
    await new Promise((r) => requestAnimationFrame(r));
  }
  return { ok: Boolean(document.querySelector(".aui-action-bar-more-content")) };
}
"""

FACTS_JS = """
() => {
  const api = window.__heavyThread;
  const del = api.actionButton("Delete message");
  const r = del ? del.getBoundingClientRect() : null;
  return {
    menuOpen: Boolean(document.querySelector(".aui-action-bar-more-content")),
    bodyPointerEvents: getComputedStyle(document.body).pointerEvents,
    deleteRect: r ? { x: r.x + r.width / 2, y: r.y + r.height / 2 } : null,
    assistantMessages: document.querySelectorAll('[data-role="assistant"]').length,
  };
}
"""

# Occupy the main thread so the guard's 300 ms timer expires between pointerdown and the
# click the browser will synthesise on release. This is a stand-in for the style recalc
# and React commit a real heavy thread does on the same press.
BLOCK_JS = """
(ms) => { const end = Date.now() + ms; while (Date.now() < end) {} }
"""

# A guard that swallows too much is as broken as one that swallows too little, and the two
# failures look identical from the outside (the menu closes either way). These two watch the
# clicks that MUST still land.
WATCH_ITEM_JS = """
() => {
  const item = document.querySelector(".aui-action-bar-more-item");
  if (!item) return null;
  window.__dismissProbe = { itemClicks: 0 };
  item.addEventListener("click", () => { window.__dismissProbe.itemClicks += 1; });
  const r = item.getBoundingClientRect();
  return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
}
"""

# Somewhere in the thread that is neither the menu nor the action bar. The listener goes on
# whatever is ACTUALLY at the point, via elementFromPoint, rather than on an element chosen by
# selector: the first `[data-role="user"]` in a seeded thread is scrolled far above the
# viewport, so a rect-derived point lands somewhere else entirely and the probe reports zero
# clicks on every tree, fixed or broken, which is a false alarm rather than a measurement.
WATCH_NEUTRAL_JS = """
() => {
  const v = window.__heavyThread.viewport();
  if (!v) return null;
  const r = v.getBoundingClientRect();
  const x = Math.round(r.x + r.width * 0.25);
  const y = Math.round(r.y + r.height * 0.75);
  const el = document.elementFromPoint(x, y);
  if (!el) return null;
  window.__dismissProbe = { neutralClicks: 0 };
  el.addEventListener("click", () => { window.__dismissProbe.neutralClicks += 1; });
  return { x, y, tag: el.tagName };
}
"""


async def hover_last_assistant(page) -> None:
    await page.evaluate(
        "() => window.__heavyThread.lastAssistantMessage()?.scrollIntoView({ block: 'center' })"
    )
    box = await page.evaluate(
        "() => { const m = window.__heavyThread.lastAssistantMessage();"
        "if (!m) return null; const r = m.getBoundingClientRect();"
        "return { x: r.x + r.width / 2, y: r.y + r.height / 2 }; }"
    )
    if box:
        await page.mouse.move(box["x"], box["y"])
        await page.wait_for_timeout(600)


async def one_case(page, case: str) -> dict:
    """Open the menu, attack the Delete button one way, report whether it fired."""
    await hover_last_assistant(page)
    opened = await page.evaluate(OPEN_MENU_JS)
    if not opened.get("ok"):
        return {"case": case, "error": "menu never opened"}
    before = await page.evaluate(FACTS_JS)
    rect = before["deleteRect"]
    if not rect:
        return {"case": case, "error": "no Delete button"}
    x, y = rect["x"], rect["y"]

    if case == "quick":
        await page.mouse.move(x, y)
        await page.mouse.down()
        await page.mouse.up()
    elif case == "held":
        await page.mouse.move(x, y)
        await page.mouse.down()
        await page.wait_for_timeout(HOLD_MS)
        await page.mouse.up()
    elif case == "busy":
        await page.mouse.move(x, y)
        await page.mouse.down()
        # Not `wait_for_timeout`: the point is that the page is BUSY, not idle, so the
        # press itself is a normal length and only the main thread is late.
        await page.evaluate(BLOCK_JS, HOLD_MS)
        await page.mouse.up()
    elif case == "touch":
        await page.touchscreen.tap(x, y)
    elif case == "select":
        # Not a dismissal at all: a click INSIDE the menu, which must still reach its item.
        spot = await page.evaluate(WATCH_ITEM_JS)
        if not spot:
            return {"case": case, "error": "no menu item to select"}
        await page.mouse.click(spot["x"], spot["y"])
        await page.wait_for_timeout(800)
        return {
            "case": case,
            "itemClicks": await page.evaluate("() => window.__dismissProbe.itemClicks"),
            "swallowedSelection": (
                await page.evaluate("() => window.__dismissProbe.itemClicks") == 0
            ),
        }
    elif case == "second_click":
        # Dismiss on neutral ground, then click a real control. Only the FIRST click is the
        # menu's to eat; a guard that stays armed would eat this one too.
        spot = await page.evaluate(WATCH_NEUTRAL_JS)
        if not spot:
            return {"case": case, "error": "no neutral spot in the viewport"}
        await page.mouse.click(spot["x"], spot["y"])
        await page.wait_for_timeout(200)
        await page.mouse.click(spot["x"], spot["y"])
        await page.wait_for_timeout(600)
        clicks = await page.evaluate("() => window.__dismissProbe.neutralClicks")
        return {
            "case": case,
            "neutralClicks": clicks,
            # One swallowed (the dismissal) and one delivered is the whole contract.
            "swallowedSecondClick": clicks < 1,
        }
    else:
        return {"case": case, "error": f"unknown case {case}"}

    await page.wait_for_timeout(1500)
    after = await page.evaluate(FACTS_JS)
    return {
        "case": case,
        "bodyPointerEvents": before["bodyPointerEvents"],
        "assistantMessagesBefore": before["assistantMessages"],
        "assistantMessagesAfter": after["assistantMessages"],
        # The whole question: did one dismissing interaction also delete a message.
        "deleted": after["assistantMessages"] < before["assistantMessages"],
        "menuClosed": not after["menuOpen"],
    }


async def run(engine: str, cases: list[str]) -> dict:
    from playwright.async_api import async_playwright

    out: dict = {"engine": engine, "chars": CHARS, "hold_ms": HOLD_MS, "cases": []}
    async with async_playwright() as pw:
        browser = await getattr(pw, engine).launch(headless = True)
        for case in cases:
            # A fresh context per case: a deleted message stays deleted, and a guard left
            # armed by one case must not be inherited by the next.
            context = await browser.new_context(
                viewport = {"width": 1280, "height": 900},
                has_touch = case == "touch",
            )
            page = await context.new_page()
            await page.goto(f"{BASE}/smoke-heavy-thread.html", wait_until = "domcontentloaded")
            await page.wait_for_function("() => Boolean(window.__heavyThread)", timeout = 60_000)
            plan = await page.evaluate("(n) => window.__heavyThread.seed(n)", CHARS)
            await page.wait_for_function(
                "(n) => window.__heavyThread.messageCount() >= n",
                arg = plan["messages"],
                timeout = 300_000,
            )
            try:
                out["cases"].append(await one_case(page, case))
            except Exception as exc:  # a failed case is a result, not a crash
                out["cases"].append({"case": case, "error": repr(exc)})
            await context.close()
        await browser.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default = "run")
    ap.add_argument("--engine", default = "chromium", choices = ("chromium", "webkit", "firefox"))
    ap.add_argument("--cases", default = "quick,held,busy,touch,select,second_click")
    args = ap.parse_args()
    cases = [c.strip() for c in args.cases.split(",") if c.strip()]

    vite = start_vite(PORT) if OWNS_SERVER else None
    try:
        wait_for_smoke_page(
            f"{BASE}/smoke-heavy-thread.html",
            "smoke-heavy-thread-main.tsx",
            proc = vite,
            info = lambda m: print(f"[probe] {m}", flush = True),
        )
        result = asyncio.run(run(args.engine, cases))
    finally:
        if vite is not None:
            stop_process(vite)

    out = Path("logs/pw") / f"dismiss_guard_{args.label}_{args.engine}.json"
    out.parent.mkdir(parents = True, exist_ok = True)
    out.write_text(json.dumps(result, indent = 2), encoding = "utf-8")
    print(json.dumps(result, indent = 2))
    print(f"[probe] wrote {out}", flush = True)

    # A verdict nobody reads is not a gate. Deleting a message is the failure this exists to
    # catch, and a case that could not run is a failure too, or a broken fixture would report
    # a clean sheet.
    deleted = [c["case"] for c in result["cases"] if c.get("deleted")]
    broken = [c["case"] for c in result["cases"] if c.get("error")]
    over = [
        c["case"]
        for c in result["cases"]
        if c.get("swallowedSelection") or c.get("swallowedSecondClick")
    ]
    if over:
        print(f"[probe] FAIL: the guard swallowed a click it should not have: {over}", flush = True)
        broken = broken + over
    if deleted:
        print(
            f"[probe] FAIL: dismissing the menu also deleted a message via {deleted}",
            flush = True,
        )
    if broken:
        print(f"[probe] FAIL: cases did not run: {broken}", flush = True)
    if deleted or broken:
        return 1
    print("[probe] PASS: no dismissal variant reached the control underneath", flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
