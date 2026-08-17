# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Behavioural probe for the message action menu's modal/non-modal switch.

Answers the three questions a reviewer will ask about dropping `modal`, and one the
coordinator added that matters more than the other three:

  destructive  with the menu open, does ONE click on the adjacent "Delete message"
               button both dismiss the menu AND delete the message? The action bar is
               a single flex row and its order is
               Copy, Edit, Refresh, ForkCount, Delete, [Speak], More
               so Delete sits two 32 px buttons from the More trigger and the menu
               opens directly beneath it (side="bottom" align="start"). Under `modal`
               the body carries pointer-events:none and that click is swallowed. This
               probe measures whether it stops being swallowed.
  scroll       does the thread scroll under an open menu (no RemoveScroll)?
  ariaHidden   is the rest of the page still aria-hidden while the menu is open?

Every click here goes through `page.mouse.click(x, y)`, a real hit test that honours
pointer-events. `locator.click()` and `element.click()` would each lie in a different
direction: the first throws on an intercepted click, the second bypasses hit testing
entirely and would report the destructive click as landing in BOTH states.

Usage:  python tests/studio/probe_menu_behaviour.py --label baseline|patched
Writes logs/pw/probe_<label>.json
"""

from __future__ import annotations

import argparse
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
SHOT_DIR = Path(os.environ.get("PROBE_SHOT_DIR", "logs/pw/shots"))


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

# Read the facts a screenshot cannot show. aria-hidden is invisible by construction and
# the body's pointer-events is the whole mechanism under test.
FACTS_JS = """
() => {
  const api = window.__heavyThread;
  const del = api.actionButton("Delete message");
  const r = del ? del.getBoundingClientRect() : null;
  // hideOthers() marks the target's SIBLINGS, so count aria-hidden among body's children.
  const bodyKids = Array.from(document.body.children);
  return {
    menuOpen: Boolean(document.querySelector(".aui-action-bar-more-content")),
    menuItems: document.querySelectorAll(".aui-action-bar-more-item").length,
    bodyPointerEvents: getComputedStyle(document.body).pointerEvents,
    deleteButtonPresent: Boolean(del),
    deleteComputedPointerEvents: del ? getComputedStyle(del).pointerEvents : null,
    deleteRect: r ? { x: r.x + r.width / 2, y: r.y + r.height / 2 } : null,
    // What the browser says is actually at the Delete button's centre. If a shield is
    // in place this is NOT the delete button.
    hitAtDelete: r
      ? (() => {
          const el = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
          if (!el) return null;
          const btn = el.closest("button");
          return {
            tag: el.tagName,
            insideDeleteButton: Boolean(btn && btn === del),
            text: (btn ? btn.textContent : el.textContent || "").trim().slice(0, 40),
          };
        })()
      : null,
    ariaHiddenBodyChildren: bodyKids.filter(
      (el) => el.getAttribute("aria-hidden") === "true"
    ).length,
    bodyChildren: bodyKids.length,
    assistantMessages: document.querySelectorAll('[data-role="assistant"]').length,
    scrollTop: Math.round(window.__heavyThread.viewportMetrics().scrollTop),
  };
}
"""

# A real wheel gesture over the viewport, which is what RemoveScroll actually gates.
# A programmatic scrollTop write is not blocked by RemoveScroll in either state and
# would report "scrolls" on both sides, i.e. prove nothing.
SCROLL_JS = """
async () => {
  const before = window.__heavyThread.viewportMetrics().scrollTop;
  return { before: Math.round(before) };
}
"""

SCROLL_AFTER_JS = """
async () => {
  const after = window.__heavyThread.viewportMetrics().scrollTop;
  return { after: Math.round(after) };
}
"""


async def drive(page, label: str) -> dict:
    out: dict = {"label": label}
    SHOT_DIR.mkdir(parents = True, exist_ok = True)

    opened = await page.evaluate(OPEN_MENU_JS)
    out["menu_opened"] = opened
    if not opened.get("ok"):
        out["fatal"] = "menu never opened"
        return out

    out["while_open"] = await page.evaluate(FACTS_JS)
    await page.screenshot(path = str(SHOT_DIR / f"{label}_1_menu_open.png"))

    # --- the destructive question -------------------------------------------------
    # A real hit test at the Delete button's centre, with the menu still open.
    rect = out["while_open"]["deleteRect"]
    before_msgs = out["while_open"]["assistantMessages"]
    if rect:
        await page.mouse.click(rect["x"], rect["y"])
        await page.wait_for_timeout(1200)
        after = await page.evaluate(FACTS_JS)
        out["after_click_on_delete"] = after
        out["destructive_click_through"] = (
            after["assistantMessages"] < before_msgs
        )
        out["menu_closed_by_that_click"] = not after["menuOpen"]
        await page.screenshot(path = str(SHOT_DIR / f"{label}_2_after_click_delete.png"))
    else:
        out["fatal"] = "no Delete message button found"
        return out

    # --- scroll lock ---------------------------------------------------------------
    # Re-open, then try to scroll the thread underneath it.
    reopened = await page.evaluate(OPEN_MENU_JS)
    if reopened.get("ok"):
        before = await page.evaluate(SCROLL_JS)
        vp = await page.evaluate(
            "() => { const v = window.__heavyThread.viewport();"
            "if (!v) return null; const r = v.getBoundingClientRect();"
            "return { x: r.x + r.width / 2, y: r.y + r.height / 2 }; }"
        )
        if vp:
            await page.mouse.move(vp["x"], vp["y"])
            for _ in range(6):
                await page.mouse.wheel(0, -400)
                await page.wait_for_timeout(120)
        after = await page.evaluate(SCROLL_AFTER_JS)
        out["scroll_under_open_menu"] = {
            **before, **after,
            "delta": after["after"] - before["before"],
            "scrolled": after["after"] != before["before"],
        }
        await page.screenshot(path = str(SHOT_DIR / f"{label}_3_scrolled_under_menu.png"))
        out["after_scroll"] = await page.evaluate(FACTS_JS)
    else:
        out["scroll_under_open_menu"] = {"error": "menu did not re-open"}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required = True)
    ap.add_argument("--engine", default = "chromium",
                    choices = ("chromium", "webkit", "firefox"))
    args = ap.parse_args()

    from playwright.async_api import async_playwright
    import asyncio

    async def run() -> dict:
        async with async_playwright() as pw:
            browser = await getattr(pw, args.engine).launch(headless = True)
            page = await browser.new_page(viewport = {"width": 1280, "height": 900})
            await page.goto(
                f"{BASE}/smoke-heavy-thread.html", wait_until = "domcontentloaded"
            )
            await page.wait_for_function(
                "() => Boolean(window.__heavyThread)", timeout = 60_000
            )
            plan = await page.evaluate(
                "(n) => window.__heavyThread.seed(n)", CHARS
            )
            await page.wait_for_function(
                "(n) => window.__heavyThread.messageCount() >= n",
                arg = plan["messages"],
                timeout = 300_000,
            )
            # Hover the last assistant message so its action bar mounts.
            await page.evaluate(
                "() => window.__heavyThread.lastAssistantMessage()"
                "?.scrollIntoView({ block: 'center' })"
            )
            box = await page.evaluate(
                "() => { const m = window.__heavyThread.lastAssistantMessage();"
                "if (!m) return null; const r = m.getBoundingClientRect();"
                "return { x: r.x + r.width / 2, y: r.y + r.height / 2 }; }"
            )
            if box:
                await page.mouse.move(box["x"], box["y"])
                await page.wait_for_timeout(600)
            result = await drive(page, args.label)
            await browser.close()
            return result

    vite = start_vite(PORT) if OWNS_SERVER else None
    try:
        wait_for_smoke_page(
            f"{BASE}/smoke-heavy-thread.html",
            "smoke-heavy-thread-main.tsx",
            proc = vite,
            info = lambda m: print(f"[probe] {m}", flush = True),
        )
        result = asyncio.run(run())
    finally:
        if vite is not None:
            stop_process(vite)

    out = Path("logs/pw") / f"probe_{args.label}_{args.engine}.json"
    out.parent.mkdir(parents = True, exist_ok = True)
    out.write_text(json.dumps(result, indent = 2), encoding = "utf-8")
    print(json.dumps(result, indent = 2))
    print(f"[probe] wrote {out}", flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
