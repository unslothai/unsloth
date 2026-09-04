# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Does the non-modal menu dismissal guard stop exactly the right click, and only that one?

The easy case is one `page.mouse.click`, a press and a release in the same tick, and every
version of the guard passes it. These are the cases that separate them. Each was checked against a DELIBERATELY BROKEN tree
before its green was trusted; where a case does not discriminate, that is said here rather than
left to be assumed.

The guard must swallow too little nowhere:

  quick          press and release in one tick. Control case.
  held           press, wait past any fixed deadline, release. A guard armed from Radix's
                 `onPointerDownOutside` and disarmed by a 300 ms timer is gone by the time the
                 browser synthesises `click`. Discriminates: chromium, webkit, firefox.
  busy           press, block the main thread, release. The same outcome from a normal-length
                 press, which is what a heavy thread produces on its own. Flaky as a detector:
                 it reproduced on one chromium run and not the next, so `held` is the reliable
                 one and this is kept only as a second look.
  touch          tap. `usePointerDownOutside` in react-dismissable-layer 1.1.11 defers to the
                 resulting `click` when `pointerType === "touch"`, on `ownerDocument`, bubble
                 phase, and React 19 delegates to the root container inside document, so the
                 control's `onClick` has already run. Discriminates: chromium, webkit.
  held_enter     press, press Enter while still holding, release. The press focuses the button,
                 so Enter activates it and the browser fires a keyboard-generated click with the
                 pointer still down. A guard that spends itself on that has nothing left for the
                 click the release synthesises. Discriminates: chromium.
  held_space     press, hold Space, RELEASE THE POINTER, then release Space. Space activates a
                 focused control on its keyup, so its click arrives after the pointer's own, on a
                 guard that has already spent itself. Discriminates: FIREFOX ONLY, and that is a
                 property of the engines rather than of the harness. Gecko tracks the pending
                 activation on its own `HTML_ELEMENT_ACTIVE_FOR_KEYBOARD` flag, which the mouse
                 release does not clear, so the click still fires; Blink and WebKit gate it on the
                 shared `:active` state, which the release does clear, so no click is ever
                 dispatched and the case cannot fail there however broken the guard is.
  touch_hold_    a finger presses and HOLDS the unconfirmed "Delete message" button, a MOUSE
    second_      press lands inside the still-open menu, then the finger lifts. Radix defers a
    pointer      touch dismissal to the resulting click, so the menu is still there for a second
                 pointer to land in, and every early return in the guard's `pointerdown` handler
                 gave the guard up without rearming it. Discriminates: chromium. Needs two
                 pointers alive at once, so it is CDP-only and reports itself skipped elsewhere
                 rather than passing vacuously; it asserts both pointers were live before it
                 trusts any verdict. TWO FINGERS cannot reach this and the case does not try:
                 measured with real CDP multi-touch, two active touch points suppress every
                 compatibility mouse event for the rest of the gesture, so the held finger's
                 release delivers no click to swallow in the first place. This case watches the
                 swallow-too-little direction only: the second pointer's own press raises no
                 `click` while a touch point is live, on either tree, so it cannot also stand in
                 for the swallow-too-much direction.
  dismiss_then_  click the button to dismiss, then press Space. The swallowed press must move
    space        focus off the dangerous control and back to the menu trigger. Space then reopens
                 the menu instead of activating Delete. Discriminates on chromium, firefox and
                 webkit. The modal shape cannot reach it, because with `pointer-events: none` on
                 the body the press lands on `HTML` and focus never moves off `BODY`.

and too much nowhere:

  dismiss_on_    dismiss the menu by clicking INTO the composer, then type. Releasing the focus a
    composer     swallowed press took must not take the caret with it. Discriminates against a
                 guard that blurs unconditionally.

  select         a click INSIDE the menu must still reach its item.
  second_click   dismiss on neutral ground, then click again. Exactly one click is the menu's.
  rightclick_    a right click raises `contextmenu` and no `click`, so a guard with no upper
    then_click   bound stays armed and eats the user's next real click. Discriminates.
  dragoff_       press, drag out, release. Does NOT discriminate today: a click still fires at
    then_click   the common ancestor, so this passes with and without the bound. Kept as a
                 cheap watch on a different shape, not offered as evidence.
  touch_neutral  a tap on a spot that CANNOT take focus, with the menu open. The capture-phase
                 swallow denies Radix its deferred touch dismissal, and when the tapped element
                 is focusable `useFocusOutside` closes the menu anyway, which is exactly how an
                 earlier version of the guard looked correct while leaving the menu open on
                 plain background. The neutral spot is grid-searched for a genuinely
                 non-focusable element and the probe FAILS rather than falling back if none
                 exists, because a probe that cannot tell you it missed is worse than none.

Every verdict is a DOM fact: did the assistant message count go down, did the menu close or
reopen from its focused trigger, did the watched click land. Clicks go through `page.mouse` /
`page.touchscreen`, real hit tests that honour pointer-events. `locator.click()` throws on
interception and `element.click()` skips hit testing, and each would lie in a different direction.

Run against the PR head AND the merge base. On the merge base these menus are modal, the body
carries `pointer-events: none`, and no variant reaches the control at all.

Usage:  python tests/studio/probe_dismiss_guard.py --label head --engine chromium
Exits non-zero on any failure, so it is a gate rather than a report.
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
# Deliberately beyond CLICK_GRACE_MS (500).
GRACE_HOLD_MS = int(os.environ.get("PROBE_GRACE_HOLD_MS", "900"))
SPACE_REOPEN_CASES = {
    "held_space_then_space",
    "dismiss_then_space",
    "drag_then_space",
    "blur_then_space",
}

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

FOCUS_FACTS_JS = """
() => {
  const trigger = window.__heavyThread.actionButton("More");
  return {
    menuClosed: !document.querySelector(".aui-action-bar-more-content"),
    focusReturnedToTrigger: document.activeElement === trigger,
  };
}
"""

# Simulate main-thread work between pointerdown and the synthesized click.
BLOCK_JS = """
(ms) => { const end = Date.now() + ms; while (Date.now() < end) {} }
"""

# Watch clicks that must still land.
WATCH_ITEM_JS = """
() => {
  // The first item forks into a new chat, so a fast runner can navigate before this probe reads
  // its counter. Export closes the same Radix menu without replacing the page under the probe.
  const item = [...document.querySelectorAll(".aui-action-bar-more-item")].find(
    (candidate) => candidate.textContent?.includes("Export as markdown")
  );
  if (!item) return null;
  window.__dismissProbe = { itemClicks: 0 };
  item.addEventListener("click", () => { window.__dismissProbe.itemClicks += 1; });
  const r = item.getBoundingClientRect();
  return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
}
"""

WATCH_NEUTRAL_JS = """
(wantUnfocusable) => {
  const v = window.__heavyThread.viewport();
  if (!v) return null;
  const r = v.getBoundingClientRect();
  const FOCUSABLE = 'button,a[href],input,textarea,select,[tabindex],[contenteditable]';
  // Grid-search rather than pick one point. The obvious choice lands on a BUTTON often enough
  // that a run can silently test the focusable case while claiming to test the other one, and
  // a probe that cannot tell you it missed is worse than no probe. If no qualifying point
  // exists in the viewport, say so instead of falling back to whatever was there.
  for (let fy = 0.9; fy > 0.05; fy -= 0.05) {
    for (let fx = 0.1; fx < 0.95; fx += 0.05) {
      const x = Math.round(r.x + r.width * fx);
      const y = Math.round(r.y + r.height * fy);
      const el = document.elementFromPoint(x, y);
      if (!el) continue;
      if (el.closest('[role="menu"],[data-radix-popper-content-wrapper]')) continue;
      const focusable = Boolean(el.closest(FOCUSABLE));
      if (wantUnfocusable && focusable) continue;
      window.__dismissProbe = { neutralClicks: 0 };
      el.addEventListener("click", () => { window.__dismissProbe.neutralClicks += 1; });
      return { x, y, tag: el.tagName, focusable };
    }
  }
  return null;
}
"""


MULTI_POINTER_INIT = """
(() => {
  const st = { live: [], maxLive: 0, concurrentTypes: [], downs: [] };
  window.__multiPointer = st;
  const typeOf = {};
  document.addEventListener("pointerdown", (e) => {
    st.live.push(e.pointerId);
    typeOf[e.pointerId] = e.pointerType;
    if (st.live.length > st.maxLive) {
      st.maxLive = st.live.length;
      st.concurrentTypes = st.live.map((id) => typeOf[id]);
    }
    st.downs.push({ id: e.pointerId, type: e.pointerType, primary: e.isPrimary });
  }, true);
  const drop = (e) => {
    const i = st.live.indexOf(e.pointerId);
    if (i >= 0) st.live.splice(i, 1);
  };
  document.addEventListener("pointerup", drop, true);
  document.addEventListener("pointercancel", drop, true);
  const inMenu = (el) => Boolean(el && el.closest && el.closest(
    '[role="menu"],[role="menuitem"],[data-radix-popper-content-wrapper]'
  ));
  const seen = (e) => ({ tag: e.target && e.target.tagName, detail: e.detail,
    inMenu: inMenu(e.target) });
  st.clicks = [];
  st.delivered = [];
  document.addEventListener("click", (e) => { st.clicks.push(seen(e)); }, true);
  // Bubble phase on `document`: the guard stops propagation in CAPTURE at `document`, so a click
  // it swallowed can never reach this. Node identity is not usable here -- the menu re-renders
  // between the setup read and the press -- so delivery is read off the event's own target.
  document.addEventListener("click", (e) => { st.delivered.push(seen(e)); }, false);
})()
"""

MENU_BLANK_JS = """
() => {
  const menu = document.querySelector('[role="menu"]');
  if (!menu) return null;
  const r = menu.getBoundingClientRect();
  for (let y = Math.round(r.y) + 1; y < r.bottom - 1; y += 1) {
    for (let x = Math.round(r.x) + 1; x < r.right - 1; x += 2) {
      const el = document.elementFromPoint(x, y);
      if (!el) continue;
      const surface = el.closest(
        '[role="menu"],[role="menuitem"],[data-radix-popper-content-wrapper]'
      );
      if (!surface) continue;
      if (el.closest('[role="menuitem"]')) continue;
      return { x, y, tag: el.tagName };
    }
  }
  return null;
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


async def one_case(
    page,
    case: str,
    engine: str = "chromium",
    context = None,
) -> dict:
    """Open the menu, attack the Delete button one way, report whether it fired."""
    await hover_last_assistant(page)
    opened = await page.evaluate(OPEN_MENU_JS)
    if not opened.get("ok"):
        return {"case": case, "error": "menu never opened"}
    before = await page.evaluate(FACTS_JS)
    before_space = None
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
        # Keep the press normal-length while blocking the page.
        await page.evaluate(BLOCK_JS, HOLD_MS)
        await page.mouse.up()
    elif case == "held_modifier":
        await page.mouse.move(x, y)
        await page.mouse.down()
        await page.wait_for_timeout(120)
        await page.keyboard.down("Shift")
        await page.wait_for_timeout(120)
        await page.mouse.up()
        await page.keyboard.up("Shift")
    elif case == "held_enter":
        # Modifiers may be pressed during a held gesture.
        await page.mouse.move(x, y)
        await page.mouse.down()
        await page.wait_for_timeout(120)
        await page.keyboard.press("Enter")
        await page.wait_for_timeout(120)
        await page.mouse.up()
    elif case == "held_space":
        # Enter activates the focused button while the pointer remains down.
        await page.mouse.move(x, y)
        await page.mouse.down()
        await page.wait_for_timeout(120)
        await page.keyboard.down("Space")
        await page.wait_for_timeout(120)
        await page.mouse.up()
        await page.wait_for_timeout(120)
        await page.keyboard.up("Space")
    elif case == "held_space_then_space":
        # Space activates on keyup, after the pointer click.
        # Ensure the swallowed press does not leave the button focused.
        await page.mouse.move(x, y)
        await page.mouse.down()
        await page.wait_for_timeout(120)
        await page.keyboard.down("Space")
        await page.wait_for_timeout(120)
        await page.mouse.up()
        await page.wait_for_timeout(120)
        await page.keyboard.up("Space")
        await page.wait_for_timeout(300)
        before_space = await page.evaluate(FOCUS_FACTS_JS)
        await page.keyboard.press("Space")
    elif case == "dismiss_then_space":
        # A swallowed dismissal must not leave Delete focused for Space activation.
        await page.mouse.click(x, y)
        await page.wait_for_timeout(300)
        before_space = await page.evaluate(FOCUS_FACTS_JS)
        await page.keyboard.press("Space")
    elif case in ("drag_then_space", "blur_then_space"):
        # A drag may retarget the click; blur exercises the no-click cleanup path.
        spot = await page.evaluate(WATCH_NEUTRAL_JS, False)
        if not spot:
            return {"case": case, "error": "no qualifying neutral spot in the viewport"}
        await page.mouse.move(x, y)
        await page.mouse.down()
        await page.wait_for_timeout(120)
        if case == "blur_then_space":
            await page.evaluate("() => window.dispatchEvent(new Event('blur'))")
        await page.mouse.move(spot["x"], spot["y"])
        await page.mouse.up()
        await page.wait_for_timeout(700)
        before_space = await page.evaluate(FOCUS_FACTS_JS)
        await page.keyboard.press("Space")
    elif case == "dismiss_on_composer":
        # Dismissing into the composer must preserve its caret.
        spot = await page.evaluate(
            "() => { const c = window.__heavyThread.composer(); if (!c) return null;"
            "const r = c.getBoundingClientRect();"
            "return { x: r.x + r.width / 2, y: r.y + r.height / 2 }; }"
        )
        if not spot:
            return {"case": case, "error": "no composer"}
        await page.mouse.click(spot["x"], spot["y"])
        await page.wait_for_timeout(300)
        await page.keyboard.type("zz")
        await page.wait_for_timeout(400)
        after = await page.evaluate(FACTS_JS)
        typed = await page.evaluate(
            "() => { const c = window.__heavyThread.composer(); return c ? c.value : ''; }"
        )
        return {
            "case": case,
            "composerText": typed[-8:],
            "menuClosed": not after["menuOpen"],
            "lostComposerFocus": not typed.endswith("zz"),
            "deleted": after["assistantMessages"] < before["assistantMessages"],
        }
    elif case == "touch":
        await page.touchscreen.tap(x, y)
    elif case in ("touch_hold_second_pointer", "touch_hold_second_pointer_grace"):
        # Hold touch while a mouse interacts with the open menu.
        if engine != "chromium" or context is None:
            return {
                "case": case,
                "skipped": f"{engine} has no API for a held touch alongside a second pointer",
            }
        blank = await page.evaluate(MENU_BLANK_JS)
        if not blank:
            return {"case": case, "error": "no non-item spot inside the menu"}
        cdp = await context.new_cdp_session(page)
        finger = [{"x": x, "y": y, "id": 1, "radiusX": 1, "radiusY": 1, "force": 1}]
        await cdp.send("Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": finger})
        await page.wait_for_timeout(150)
        held = await page.evaluate(FACTS_JS)
        await page.mouse.move(blank["x"], blank["y"])
        await page.mouse.down()
        await page.wait_for_timeout(80)
        live = await page.evaluate("() => window.__multiPointer.live.length")
        await page.mouse.up()
        await page.wait_for_timeout(GRACE_HOLD_MS if case.endswith("_grace") else 150)
        if case.endswith("_grace"):
            still_down = await page.evaluate("() => window.__multiPointer.live.length")
            if still_down < 1:
                return {
                    "case": case,
                    "error": "the finger was no longer down when the bound expired, so the "
                    "case tested nothing",
                }
        await cdp.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": finger})
        await page.wait_for_timeout(1500)
        after = await page.evaluate(FACTS_JS)
        state = await page.evaluate("() => window.__multiPointer")
        if not held["menuOpen"]:
            return {
                "case": case,
                "error": "the held touch dismissed the menu, so there was no "
                "open menu for a second pointer to land in",
            }
        if state["maxLive"] < 2 or live < 2:
            return {
                "case": case,
                "error": f"the two pointers never coexisted: maxLive="
                f"{state['maxLive']}, live at the second press={live}",
            }
        if sorted(state["concurrentTypes"]) != ["mouse", "touch"]:
            return {"case": case, "error": f"wrong pointer types: {state['concurrentTypes']}"}
        return {
            "case": case,
            "concurrentPointers": state["downs"],
            "menuOpenUnderSecondPointer": held["menuOpen"],
            "assistantMessagesBefore": before["assistantMessages"],
            "assistantMessagesAfter": after["assistantMessages"],
            "deleted": after["assistantMessages"] < before["assistantMessages"],
            "menuClosed": not after["menuOpen"],
            "clicksSeen": state["clicks"],
            "clicksDelivered": state["delivered"],
        }
    elif case == "select":
        # Selection inside the menu must reach its item.
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
    elif case == "touch_neutral":
        # A non-focusable touch target exercises Radix's deferred dismissal.
        spot = await page.evaluate(WATCH_NEUTRAL_JS, case == "touch_neutral")
        if not spot:
            return {"case": case, "error": "no qualifying neutral spot in the viewport"}
        await page.touchscreen.tap(spot["x"], spot["y"])
        await page.wait_for_timeout(1200)
        after = await page.evaluate(FACTS_JS)
        return {
            "case": case,
            "tappedTag": spot.get("tag"),
            "menuClosed": not after["menuOpen"],
            "deleted": after["assistantMessages"] < before["assistantMessages"],
        }
    elif case == "touch_trigger":
        # Tapping the trigger must still close the menu.
        trigger = await page.evaluate(
            "() => { const t = window.__heavyThread.actionButton('More');"
            "if (!t) return null; const r = t.getBoundingClientRect();"
            "return { x: r.x + r.width / 2, y: r.y + r.height / 2 }; }"
        )
        if not trigger:
            return {"case": case, "error": "no More trigger"}
        await page.touchscreen.tap(trigger["x"], trigger["y"])
        await page.wait_for_timeout(1200)
        after = await page.evaluate(FACTS_JS)
        return {
            "case": case,
            "menuClosed": not after["menuOpen"],
            "deleted": after["assistantMessages"] < before["assistantMessages"],
        }
    elif case in ("rightclick_then_click", "dragoff_then_click"):
        spot = await page.evaluate(WATCH_NEUTRAL_JS, False)
        if not spot:
            return {"case": case, "error": "no qualifying neutral spot in the viewport"}
        if case == "rightclick_then_click":
            # Prevent browser chrome from consuming the next automated click.
            await page.evaluate(
                """() => document.addEventListener(
                    "contextmenu", (event) => event.preventDefault(),
                    { capture: true, once: true }
                )"""
            )
            await page.mouse.click(spot["x"], spot["y"], button = "right")
        else:
            await page.mouse.move(spot["x"], spot["y"])
            await page.mouse.down()
            await page.mouse.move(4, 4)
            await page.mouse.up()
        # INSIDE the release-anchored grace window on purpose. Waiting it out was the first
        # version of this case and it passed on a tree that was still eating the click, because
        # it only ever asked whether the bound existed, not whether the guard should have armed
        # for a gesture that cannot produce a click at all.
        await page.wait_for_timeout(150)
        await page.mouse.click(spot["x"], spot["y"])
        await page.wait_for_timeout(600)
        clicks = await page.evaluate("() => window.__dismissProbe.neutralClicks")
        return {
            "case": case,
            "neutralClicks": clicks,
            "swallowedLaterClick": clicks < 1,
        }
    elif case == "select_then_quick_click":
        item = await page.evaluate(WATCH_ITEM_JS)
        if not item:
            return {"case": case, "error": "no menu item to select"}
        spot = await page.evaluate(WATCH_NEUTRAL_JS, False)
        if not spot:
            return {"case": case, "error": "no qualifying neutral spot in the viewport"}
        await page.evaluate("() => { window.__dismissProbe.neutralClicks = 0; }")
        await page.mouse.click(item["x"], item["y"])
        await page.wait_for_timeout(int(os.environ.get("PROBE_EXIT_DELAY_MS", "40")))
        await page.mouse.click(spot["x"], spot["y"])
        await page.wait_for_timeout(600)
        clicks = await page.evaluate("() => window.__dismissProbe.neutralClicks")
        return {
            "case": case,
            "neutralClicks": clicks,
            "swallowedLaterClick": clicks == 0,
        }
    elif case == "second_click":
        # Only the dismissing click may be swallowed.
        spot = await page.evaluate(WATCH_NEUTRAL_JS, False)
        if not spot:
            return {"case": case, "error": "no qualifying neutral spot in the viewport"}
        await page.mouse.click(spot["x"], spot["y"])
        await page.wait_for_timeout(200)
        await page.mouse.click(spot["x"], spot["y"])
        await page.wait_for_timeout(600)
        clicks = await page.evaluate("() => window.__dismissProbe.neutralClicks")
        return {
            "case": case,
            "neutralClicks": clicks,
            "swallowedSecondClick": clicks < 1,
        }
    else:
        return {"case": case, "error": f"unknown case {case}"}

    await page.wait_for_timeout(1500)
    after = await page.evaluate(FACTS_JS)
    result = {
        "case": case,
        "bodyPointerEvents": before["bodyPointerEvents"],
        "assistantMessagesBefore": before["assistantMessages"],
        "assistantMessagesAfter": after["assistantMessages"],
        "deleted": after["assistantMessages"] < before["assistantMessages"],
        "menuClosed": not after["menuOpen"],
    }
    if before_space is not None:
        result.update(
            {
                "menuClosedBeforeSpace": before_space["menuClosed"],
                "focusReturnedToTrigger": before_space["focusReturnedToTrigger"],
                "menuReopenedBySpace": after["menuOpen"],
            }
        )
    return result


async def run(engine: str, cases: list[str]) -> dict:
    from playwright.async_api import async_playwright

    out: dict = {"engine": engine, "chars": CHARS, "hold_ms": HOLD_MS, "cases": []}
    async with async_playwright() as pw:
        browser = await getattr(pw, engine).launch(headless = True)
        for case in cases:
            context = await browser.new_context(
                viewport = {"width": 1280, "height": 900},
                has_touch = case.startswith("touch"),
            )
            page = await context.new_page()
            await page.add_init_script(MULTI_POINTER_INIT)
            await page.goto(f"{BASE}/smoke-heavy-thread.html", wait_until = "domcontentloaded")
            await page.wait_for_function("() => Boolean(window.__heavyThread)", timeout = 60_000)
            plan = await page.evaluate("(n) => window.__heavyThread.seed(n)", CHARS)
            await page.wait_for_function(
                "(n) => window.__heavyThread.messageCount() >= n",
                arg = plan["messages"],
                timeout = 300_000,
            )
            try:
                out["cases"].append(await one_case(page, case, engine, context))
            except Exception as exc:  # a failed case is a result, not a crash
                out["cases"].append({"case": case, "error": repr(exc)})
            await context.close()
        await browser.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default = "run")
    ap.add_argument("--engine", default = "chromium", choices = ("chromium", "webkit", "firefox"))
    ap.add_argument(
        "--cases",
        default = "quick,held,busy,held_modifier,held_enter,held_space,held_space_then_space,dismiss_then_space,drag_then_space,blur_then_space,dismiss_on_composer,touch,touch_hold_second_pointer,touch_hold_second_pointer_grace,select,select_then_quick_click,second_click,rightclick_then_click,dragoff_then_click,touch_neutral,touch_trigger",
    )
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

    skipped = [(c["case"], c["skipped"]) for c in result["cases"] if c.get("skipped")]
    for case, why in skipped:
        print(f"[probe] SKIPPED {case}: {why}", flush = True)
    deleted = [c["case"] for c in result["cases"] if c.get("deleted")]
    stuck = [
        c["case"]
        for c in result["cases"]
        if "menuClosed" in c and not c["menuClosed"] and c["case"] not in SPACE_REOPEN_CASES
    ]
    unsafe_space = [
        c["case"]
        for c in result["cases"]
        if c["case"] in SPACE_REOPEN_CASES
        and (
            not c.get("menuClosedBeforeSpace")
            or not c.get("focusReturnedToTrigger")
            or not c.get("menuReopenedBySpace")
        )
    ]
    broken = [c["case"] for c in result["cases"] if c.get("error")]
    over = [
        c["case"]
        for c in result["cases"]
        if c.get("swallowedSelection")
        or c.get("swallowedSecondClick")
        or c.get("swallowedLaterClick")
        or c.get("lostComposerFocus")
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
    if stuck:
        print(f"[probe] FAIL: the menu did not close on {stuck}", flush = True)
    if unsafe_space:
        print(
            "[probe] FAIL: the Space follow-up did not dismiss safely, restore trigger focus, "
            f"and reopen the menu: {unsafe_space}",
            flush = True,
        )
    if deleted or broken or stuck or unsafe_space:
        return 1
    print("[probe] PASS: no dismissal variant reached the control underneath", flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
