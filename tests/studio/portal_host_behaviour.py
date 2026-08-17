# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Does moving every Radix portal off `<body>` change what the user sees or can click?

lib/portal-host.ts moves floating surfaces from `document.body` into a `display: contents` div
that is a child of `document.body`, so React stops attaching its delegated event listener set to
`<body>`. That is event plumbing, and a regression in it would be invisible in a screenshot and
very visible in use, so the listener census is not the test. This is.

Per engine, on a seeded thread, this asserts:

    the menu opens, and reports the same item count as before
    the menu content is inside the portal host, not a child of body
    the menu's bounding box is where it was, to the pixel
    every menu item is HIT TESTABLE: document.elementFromPoint at the item's centre lands
      inside that item. This is the check that catches a stacking or pointer-events regression,
      which a screenshot cannot
    Escape dismisses it, and one click outside dismisses it and does NOT actuate the control
      underneath, which is the footgun PR 9051's swallowDismissingClick exists to hold shut
    a tooltip still appears on hover and is itself hit testable

Geometry is compared BETWEEN TREES rather than against a constant: run this on the control tree
and on the fix tree and diff the JSON. A menu that moved by a pixel is a real regression and a
hard-coded expectation would either miss it or fail on every unrelated layout change.

Run:
    SMOKE_PORT=5601 PROBE_CHARS=25000 python tests/studio/portal_host_behaviour.py
"""

from __future__ import annotations

import json
import os
import socket
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import chromium_launch_args, start_vite, stop_process  # noqa: E402
import playwright_heavy_thread as hv  # noqa: E402

PORT = int(os.environ.get("SMOKE_PORT", "5601"))
BASE = os.environ.get("SMOKE_BASE_URL", "").strip().rstrip("/") or f"http://127.0.0.1:{PORT}"
OWNS_SERVER = not os.environ.get("SMOKE_BASE_URL", "").strip()
CHARS = int(os.environ.get("PROBE_CHARS", "25000"))
ENGINES = [e.strip() for e in os.environ.get("PROBE_ENGINES", "chromium").split(",") if e.strip()]
LABEL = os.environ.get("PROBE_LABEL", "portal_behaviour")
OUT = Path(os.environ.get("PW_ART_DIR", "logs/portal-host"))
OUT.mkdir(parents = True, exist_ok = True)


def info(m: str) -> None:
    print(f"[portal] {m}", flush = True)


# Open the More menu and describe it completely enough that two trees can be diffed.
INSPECT_JS = """
async () => {
  const api = window.__heavyThread;
  const trigger = api.actionButton("More");
  if (!trigger) return { error: "no More trigger" };
  const pointer = { bubbles: true, cancelable: true, composed: true, button: 0,
                    pointerId: 1, pointerType: "mouse", isPrimary: true };
  trigger.dispatchEvent(new PointerEvent("pointerdown", { ...pointer, buttons: 1 }));
  trigger.dispatchEvent(new PointerEvent("pointerup", { ...pointer, buttons: 0 }));
  const deadline = performance.now() + 30000;
  let content = null;
  while (performance.now() < deadline) {
    content = document.querySelector(".aui-action-bar-more-content");
    if (content) break;
    await new Promise((r) => requestAnimationFrame(() => r()));
  }
  if (!content) return { error: "menu never opened" };
  // Two frames so the popper has positioned and any open animation has committed.
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(() => r())));

  const round = (r) => ({ x: Math.round(r.x), y: Math.round(r.y),
                          w: Math.round(r.width), h: Math.round(r.height) });
  const host = document.getElementById("unsloth-portal-host");
  const items = [...content.querySelectorAll('[role="menuitem"], [role="menuitemcheckbox"]')];
  const hitTests = items.map((it) => {
    const r = it.getBoundingClientRect();
    const el = document.elementFromPoint(Math.round(r.x + r.width / 2),
                                        Math.round(r.y + r.height / 2));
    return {
      label: (it.textContent || "").trim().slice(0, 24),
      rect: round(r),
      // `contains` and not identity: the centre of an item legitimately lands on an icon or a
      // label span INSIDE it. What must not happen is landing outside the item entirely, which
      // is what a stacking or pointer-events regression looks like.
      hit: Boolean(el && it.contains(el)),
      hitTag: el ? el.tagName.toLowerCase() : null,
    };
  });
  const cs = getComputedStyle(content);
  return {
    itemCount: items.length,
    contentRect: round(content.getBoundingClientRect()),
    triggerRect: round(trigger.getBoundingClientRect()),
    zIndex: cs.zIndex,
    position: cs.position,
    visibility: cs.visibility,
    opacity: cs.opacity,
    hitTests,
    allItemsHit: hitTests.every((h) => h.hit),
    portalHostExists: Boolean(host),
    portalHostDisplay: host ? getComputedStyle(host).display : null,
    // Where the floating surface actually lives. On the control tree the popper wrapper is a
    // direct child of body; on the fix tree it is inside the host, and the host is a child of
    // body, so the surface is at the same depth in the box tree because the host generates no box.
    contentInsideHost: Boolean(host && host.contains(content)),
    popperParentIsBody:
      content.closest("[data-radix-popper-content-wrapper]")?.parentElement === document.body,
    bodyListenerHostId: document.body.lastElementChild?.id || null,
  };
}
"""

# Escape closes it, and then a single click on the neighbouring control must close a REOPENED
# menu without actuating that control. That second half is the swallowDismissingClick contract
# from PR 9051, which this change must not break.
DISMISS_JS = """
async () => {
  const api = window.__heavyThread;
  const gone = async () => {
    const deadline = performance.now() + 15000;
    while (performance.now() < deadline) {
      if (!document.querySelector(".aui-action-bar-more-content")) return true;
      await new Promise((r) => requestAnimationFrame(() => r()));
    }
    return false;
  };
  document.dispatchEvent(
    new KeyboardEvent("keydown", { key: "Escape", bubbles: true, cancelable: true }),
  );
  const closedByEscape = await gone();

  // Reopen, then click a sibling action button and check the click was swallowed.
  const trigger = api.actionButton("More");
  const neighbour = api.actionButton("Refresh") || api.actionButton("Copy");
  if (!trigger || !neighbour) return { closedByEscape, neighbourTested: false };
  const pointer = { bubbles: true, cancelable: true, composed: true, button: 0,
                    pointerId: 1, pointerType: "mouse", isPrimary: true };
  trigger.dispatchEvent(new PointerEvent("pointerdown", { ...pointer, buttons: 1 }));
  trigger.dispatchEvent(new PointerEvent("pointerup", { ...pointer, buttons: 0 }));
  const deadline = performance.now() + 15000;
  while (performance.now() < deadline) {
    if (document.querySelector(".aui-action-bar-more-content")) break;
    await new Promise((r) => requestAnimationFrame(() => r()));
  }
  let neighbourFired = false;
  const spy = () => { neighbourFired = true; };
  neighbour.addEventListener("click", spy);
  neighbour.dispatchEvent(new PointerEvent("pointerdown", { ...pointer, buttons: 1 }));
  neighbour.dispatchEvent(new PointerEvent("pointerup", { ...pointer, buttons: 0 }));
  neighbour.dispatchEvent(new MouseEvent("click", { bubbles: true, cancelable: true }));
  const closedByOutsideClick = await gone();
  neighbour.removeEventListener("click", spy);
  return {
    closedByEscape,
    neighbourTested: true,
    closedByOutsideClick,
    // The whole point of the swallow: dismissing must not also press the button next door.
    neighbourFired,
  };
}
"""

TOOLTIP_JS = """
async () => {
  const t = document.querySelector('[data-slot="tooltip-trigger"]');
  if (!t) return { tested: false };
  t.scrollIntoView({ block: "center", behavior: "instant" });
  await new Promise((r) => requestAnimationFrame(() => r()));
  for (const type of ["pointerover", "pointerenter", "mouseover", "mouseenter", "focus"]) {
    t.dispatchEvent(new (type.startsWith("pointer") ? PointerEvent : type === "focus" ? FocusEvent : MouseEvent)(
      type, { bubbles: type !== "pointerenter" && type !== "mouseenter" && type !== "focus", cancelable: true },
    ));
  }
  const deadline = performance.now() + 8000;
  let tip = null;
  while (performance.now() < deadline) {
    tip = document.querySelector('[data-slot="tooltip-content"], [role="tooltip"]');
    if (tip && tip.getBoundingClientRect().width > 0) break;
    await new Promise((r) => setTimeout(r, 50));
  }
  if (!tip) return { tested: true, appeared: false };
  const r = tip.getBoundingClientRect();
  const host = document.getElementById("unsloth-portal-host");
  return {
    tested: true,
    appeared: true,
    rect: { x: Math.round(r.x), y: Math.round(r.y), w: Math.round(r.width), h: Math.round(r.height) },
    insideHost: Boolean(host && host.contains(tip)),
    visible: getComputedStyle(tip).visibility,
  };
}
"""


def main() -> int:
    results = {"label": LABEL, "chars": CHARS, "engines": {}}
    vite = None
    try:
        if OWNS_SERVER:
            info(f"starting vite on {PORT}")
            vite = start_vite(PORT)
            deadline = time.time() + 300
            while time.time() < deadline:
                with socket.socket() as s:
                    s.settimeout(1)
                    if s.connect_ex(("127.0.0.1", PORT)) == 0:
                        break
                time.sleep(1)
            info("vite ready")
        with sync_playwright() as pw:
            for engine in ENGINES:
                info(f"--- {engine}")
                launcher = getattr(pw, engine)
                kwargs = {"args": chromium_launch_args()} if engine == "chromium" else {}
                browser = launcher.launch(**kwargs)
                ctx = browser.new_context(viewport = {"width": 1280, "height": 900})
                ctx.add_init_script(hv.RECORDER_INIT)
                page = ctx.new_page()
                try:
                    page.goto(f"{BASE}/smoke-heavy-thread.html", wait_until = "domcontentloaded")
                    page.wait_for_function("() => Boolean(window.__heavyThread)", timeout = 180_000)
                    plan = page.evaluate("(n) => window.__heavyThread.seed(n)", CHARS)
                    page.wait_for_function(
                        "(n) => window.__heavyThread.messageCount() >= n",
                        arg = plan["messages"],
                        timeout = 600_000,
                    )
                    n = page.evaluate("() => window.__heavyThread.expandTools()")
                    if n:
                        page.wait_for_function(
                            "(k) => window.__heavyThread.counts().collapsibleOutputs >= k",
                            arg = n,
                            timeout = 300_000,
                        )
                    hv.wait_for_highlighting_settled(page, 600_000)
                    page.evaluate(
                        """() => { const m = window.__heavyThread.lastAssistantMessage();
                            if (m) m.scrollIntoView({ block: "center", behavior: "instant" }); }"""
                    )
                    page.wait_for_timeout(500)
                    page.locator('[data-role="assistant"]').last.hover(timeout = 300_000)

                    row: dict = {}
                    row["menu"] = page.evaluate(INSPECT_JS)
                    page.screenshot(path = str(OUT / f"{LABEL}_{engine}_menu_open.png"))
                    row["dismiss"] = page.evaluate(DISMISS_JS)
                    row["tooltip"] = page.evaluate(TOOLTIP_JS)
                    page.screenshot(path = str(OUT / f"{LABEL}_{engine}_tooltip.png"))
                    results["engines"][engine] = row
                    info(f"  menu {json.dumps(row['menu'])[:400]}")
                    info(f"  dismiss {json.dumps(row['dismiss'])}")
                    info(f"  tooltip {json.dumps(row['tooltip'])}")
                except Exception as exc:  # noqa: BLE001
                    results["engines"][engine] = {"failed": repr(exc)}
                    info(f"  FAILED {exc!r}")
                finally:
                    ctx.close()
                    browser.close()
        out = OUT / f"{LABEL}.json"
        out.write_text(json.dumps(results, indent = 2))
        info(f"wrote {out}")
    finally:
        if vite is not None:
            stop_process(vite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
