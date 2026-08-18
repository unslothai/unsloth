# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Does the non-modal menu dismissal guard stop exactly the right click, and only that one?

`probe_menu_behaviour.py` asks the destructive question once, with `page.mouse.click`, a press
and a release in the same tick. That is the easy case and every version of the guard passes it.
These are the cases that separate them. Each was checked against a DELIBERATELY BROKEN tree
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
  dismiss_then_  click the button to dismiss, then press Space, the key a reader uses to scroll.
    space        No click is involved for the guard to swallow: the press it DID swallow left the
                 button focused, and a focused button activates on Space. Discriminates on
                 chromium, firefox and webkit. The modal shape cannot reach it, because with
                 `pointer-events: none` on the body the press lands on `HTML` and focus never
                 moves off `BODY`.

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

Every verdict is a DOM fact: did the assistant message count go down, did the menu close, did
the watched click land. Clicks go through `page.mouse` / `page.touchscreen`, real hit tests that
honour pointer-events. `locator.click()` throws on interception and `element.click()` skips hit
testing, and each would lie in a different direction.

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
    elif case == "held_modifier":
        # Press, then press a modifier while still held, then release. Modifiers get pressed
        # mid-gesture constantly. A guard that treats any keydown as "the user moved on" disarms
        # here, and the release still synthesises the click, so the press lands on Delete.
        await page.mouse.move(x, y)
        await page.mouse.down()
        await page.wait_for_timeout(120)
        await page.keyboard.down("Shift")
        await page.wait_for_timeout(120)
        await page.mouse.up()
        await page.keyboard.up("Shift")
    elif case == "held_enter":
        # Press and hold, then press Enter. A press on a button focuses it on Linux and Windows,
        # so Enter activates the focused control and the browser fires a KEYBOARD-generated click
        # while the pointer is still down. A guard that treats any click as the one it was armed
        # for disarms on that, and the click the RELEASE synthesises then lands on Delete.
        await page.mouse.move(x, y)
        await page.mouse.down()
        await page.wait_for_timeout(120)
        await page.keyboard.press("Enter")
        await page.wait_for_timeout(120)
        await page.mouse.up()
    elif case == "held_space":
        # The same shape as `held_enter` with the one key that activates on KEYUP rather than
        # keydown. Press and hold, hold Space down, RELEASE THE POINTER, then release Space. The
        # pointer's own click arrives with the pointer already up, so a guard keyed on
        # `pointerIsDown` spends itself on it, and the activation click Space fires on its keyup
        # lands afterwards on a disarmed document. HTML spec: Space activates a button on keyup,
        # Enter on keydown, which is why `held_enter` cannot cover this ordering.
        await page.mouse.move(x, y)
        await page.mouse.down()
        await page.wait_for_timeout(120)
        await page.keyboard.down("Space")
        await page.wait_for_timeout(120)
        await page.mouse.up()
        await page.wait_for_timeout(120)
        await page.keyboard.up("Space")
    elif case == "held_space_then_space":
        # `held_space` leaves the guard correct about the CLICK and wrong about the FOCUS. The
        # press focuses the button; the held-Space branch swallows the pointer click and returns
        # early, so nothing releases that focus. The gesture then ends with the button still
        # focused, and the next ordinary Space -- the key a reader uses to scroll -- activates it
        # and deletes the message. Same failure as `dismiss_then_space`, reached down the one
        # path that skipped the blur.
        await page.mouse.move(x, y)
        await page.mouse.down()
        await page.wait_for_timeout(120)
        await page.keyboard.down("Space")
        await page.wait_for_timeout(120)
        await page.mouse.up()
        await page.wait_for_timeout(120)
        await page.keyboard.up("Space")
        await page.wait_for_timeout(300)
        await page.keyboard.press("Space")
    elif case == "dismiss_then_space":
        # Dismiss the menu with an ordinary click on the unconfirmed "Delete message" button, which
        # the guard swallows, and then press Space -- the key a reader uses to scroll a thread. The
        # press that was thrown away still FOCUSED that button, so the keystroke activates it. The
        # modal shield never left this behind: with `pointer-events: none` on the body the press
        # landed on `HTML` and the button was never focused.
        await page.mouse.click(x, y)
        await page.wait_for_timeout(300)
        await page.keyboard.press("Space")
    elif case == "dismiss_on_composer":
        # The other half of releasing the focus a swallowed press took: dismissing a menu by
        # clicking INTO the composer must leave the caret there, or the guard has answered one
        # unasked-for effect with another. Typing is the only honest test of that.
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
    elif case == "touch_neutral":
        # A touch tap on a spot that CANNOT take focus. Radix defers its touch dismissal to the
        # resulting click, and this guard swallows that click in the capture phase, so the
        # deferred handler never runs. When the tapped element is focusable, `useFocusOutside`
        # dismisses instead and the menu closes anyway. Plain thread background is not focusable,
        # so this is the case where nothing else can cover for it.
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
        # Tap the trigger itself to close. The trigger is OUTSIDE the content, so the guard arms
        # for it and swallows its click; if that were the only thing closing the menu, a touch
        # user would be unable to cancel one of these menus without picking an item.
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
        # Two ways a dismissing gesture produces NO click at all: a right click raises
        # contextmenu instead, and a press that leaves the window is released elsewhere. Either
        # one leaves a guard with no upper bound armed indefinitely, so the user's next real
        # click is eaten with nothing to associate it with.
        spot = await page.evaluate(WATCH_NEUTRAL_JS, case == "touch_neutral")
        if not spot:
            return {"case": case, "error": "no qualifying neutral spot in the viewport"}
        if case == "rightclick_then_click":
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
    elif case == "second_click":
        # Dismiss on neutral ground, then click a real control. Only the FIRST click is the
        # menu's to eat; a guard that stays armed would eat this one too.
        spot = await page.evaluate(WATCH_NEUTRAL_JS, case == "touch_neutral")
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
                has_touch = case.startswith("touch"),
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
    ap.add_argument(
        "--cases",
        default = "quick,held,busy,held_modifier,held_enter,held_space,held_space_then_space,dismiss_then_space,dismiss_on_composer,touch,select,second_click,rightclick_then_click,dragoff_then_click,touch_neutral,touch_trigger",
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

    # A verdict nobody reads is not a gate. Deleting a message is the failure this exists to
    # catch, and a case that could not run is a failure too, or a broken fixture would report
    # a clean sheet.
    deleted = [c["case"] for c in result["cases"] if c.get("deleted")]
    # A guard that swallows the dismissing click so thoroughly that Radix never sees it would
    # leave the menu OPEN, which is its own bug and one this probe used to record and ignore.
    stuck = [c["case"] for c in result["cases"] if "menuClosed" in c and not c["menuClosed"]]
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
    if deleted or broken or stuck:
        return 1
    print("[probe] PASS: no dismissal variant reached the control underneath", flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
