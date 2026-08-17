# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Does the user action bar still appear on hover, and only on hover?

Swapping assistant-ui's `autohide` for a CSS `group-hover` rule is a performance change that can
fail in two silent ways, and a timing number would look FASTER for both of them:

    the bar never shows      a Tailwind class that did not survive the build leaves the bar at
                             `display: none` forever. Nothing mounts, nothing re-renders, and the
                             gesture gets cheaper because a feature is gone.
    the bar always shows     the `hidden` utility loses a specificity fight and every bar is
                             visible at rest, which is a different product.

So this asserts the shape of the behaviour before any timing is believed: hidden at rest, shown
while the pointer is on the message, hidden again when it leaves, and the count of visible bars
at rest is zero across the whole thread.

Run:
    SMOKE_PORT=5991 PROBE_CHARS=25000 PROBE_ENGINES=chromium,webkit,firefox \\
      python tests/studio/actionbar_visibility_check.py
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

PORT = int(os.environ.get("SMOKE_PORT", "5991"))
BASE = os.environ.get("SMOKE_BASE_URL", "").strip().rstrip("/") or f"http://127.0.0.1:{PORT}"
OWNS_SERVER = not os.environ.get("SMOKE_BASE_URL", "").strip()
CHARS = int(os.environ.get("PROBE_CHARS", "25000"))
ENGINES = [e.strip() for e in os.environ.get("PROBE_ENGINES", "chromium").split(",") if e.strip()]
LABEL = os.environ.get("PROBE_LABEL", "actionbar_visibility")
OUT = Path(os.environ.get("PW_ART_DIR", "logs/actionbar"))
OUT.mkdir(parents = True, exist_ok = True)

# `visible` here means "takes space and is displayed", which is what the user experiences, rather
# than "is in the DOM". The control tree unmounts the bar and this tree display:none's it, so a
# DOM-presence test would report the two trees as different when the user cannot tell.
CENSUS_JS = """
() => {
  const bars = [...document.querySelectorAll(".aui-user-action-bar-root")];
  const shown = bars.filter((b) => {
    const cs = getComputedStyle(b);
    if (cs.display === "none" || cs.visibility === "hidden" || cs.opacity === "0") return false;
    const r = b.getBoundingClientRect();
    return r.width > 0 && r.height > 0;
  });
  return {
    inDom: bars.length,
    shown: shown.length,
    firstDisplay: bars.length ? getComputedStyle(bars[0]).display : null,
  };
}
"""

# After a scroll that the pointer did not participate in, the bar must be on the message now
# UNDER the pointer, not on the one that was there before. This is the only behaviour
# use-hover-quiet-during-scroll.ts changes, so it is the one that has to be pinned down: the hook
# swallows hover events while scrolling and delivers a single synthetic pair once it stops, and if
# that settle step is wrong the bar ends up stranded on an off-screen message.
SETTLED_BAR_JS = """
([x, y]) => {
  const el = document.elementFromPoint(x, y);
  const underPointer = el ? el.closest('[data-message-id]') : null;
  const bars = [...document.querySelectorAll(".aui-user-action-bar-root")].filter((b) => {
    const cs = getComputedStyle(b);
    const r = b.getBoundingClientRect();
    return cs.display !== "none" && r.width > 0 && r.height > 0;
  });
  const owner = bars.length === 1 ? bars[0].closest('[data-message-id]') : null;
  return {
    shown: bars.length,
    underPointerIsUserMessage: Boolean(
      underPointer && underPointer.getAttribute("data-role") === "user",
    ),
    underPointerId: underPointer ? underPointer.getAttribute("data-message-id") : null,
    barOwnerId: owner ? owner.getAttribute("data-message-id") : null,
    barIsOnMessageUnderPointer: Boolean(owner && underPointer && owner === underPointer),
  };
}
"""

HOVERED_BAR_JS = """
() => {
  const msg = document.querySelector('[data-role="user"]:hover');
  if (!msg) return { messageHovered: false };
  const bar = msg.querySelector(".aui-user-action-bar-root");
  if (!bar) return { messageHovered: true, barPresent: false };
  const cs = getComputedStyle(bar);
  const r = bar.getBoundingClientRect();
  return {
    messageHovered: true,
    barPresent: true,
    display: cs.display,
    shown: cs.display !== "none" && r.width > 0 && r.height > 0,
    buttons: bar.querySelectorAll("button").length,
  };
}
"""


def info(m: str) -> None:
    print(f"[actionbar] {m}", flush = True)


def main() -> int:
    results = {"label": LABEL, "chars": CHARS, "engines": {}}
    vite = None
    failures: list[str] = []
    try:
        if OWNS_SERVER:
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
                    hv.wait_for_highlighting_settled(page, 600_000)
                    # Park the pointer where it is on nothing, so "at rest" really is at rest.
                    page.mouse.move(3, 3)
                    page.wait_for_timeout(400)
                    at_rest = page.evaluate(CENSUS_JS)

                    target = page.locator('[data-role="user"]').last
                    target.scroll_into_view_if_needed(timeout = 60_000)
                    page.wait_for_timeout(300)
                    target.hover(timeout = 60_000)
                    page.wait_for_timeout(400)
                    hovered = page.evaluate(HOVERED_BAR_JS)
                    during = page.evaluate(CENSUS_JS)

                    # Scroll WITHOUT moving the pointer, which is the case the hook exists for,
                    # then let it settle and ask which message the bar ended up on.
                    box = target.bounding_box()
                    px = int(box["x"] + box["width"] / 2)
                    py = int(box["y"] + box["height"] / 2)
                    page.mouse.move(px, py)
                    page.wait_for_timeout(400)
                    page.evaluate(
                        """(dy) => { const v = window.__heavyThread.viewport();
                            v.scrollTo({ top: Math.max(0, v.scrollTop - dy),
                                         behavior: "instant" }); }""",
                        600,
                    )
                    # Comfortably past the hook's 150ms quiet window, and past any engine's own
                    # hover re-evaluation, so a failure here is a real one.
                    page.wait_for_timeout(1200)
                    settled = page.evaluate(SETTLED_BAR_JS, [px, py])

                    page.mouse.move(3, 3)
                    page.wait_for_timeout(400)
                    after = page.evaluate(CENSUS_JS)

                    row = {
                        "at_rest": at_rest,
                        "hovered": hovered,
                        "during": during,
                        "settled_after_scroll": settled,
                        "after": after,
                    }
                    results["engines"][engine] = row
                    info(f"{engine}: at rest {json.dumps(at_rest)}")
                    info(f"{engine}: hovered {json.dumps(hovered)}")
                    info(f"{engine}: during  {json.dumps(during)}   after {json.dumps(after)}")

                    if at_rest["shown"] != 0:
                        failures.append(
                            f"{engine}: {at_rest['shown']} action bars are visible with the "
                            "pointer on nothing; the bar is supposed to be hidden at rest"
                        )
                    if not hovered.get("shown"):
                        failures.append(
                            f"{engine}: the hovered message's action bar did not become "
                            f"visible ({json.dumps(hovered)}); hover no longer reveals it"
                        )
                    if during["shown"] != 1:
                        failures.append(
                            f"{engine}: {during['shown']} bars visible while ONE message is "
                            "hovered; exactly one is expected"
                        )
                    if settled["underPointerIsUserMessage"]:
                        # Only assertable when the scroll left a USER message under the pointer;
                        # assistant messages have a different bar and this check would be
                        # measuring the wrong component.
                        if settled["shown"] != 1:
                            failures.append(
                                f"{engine}: after a scroll with the pointer still, "
                                f"{settled['shown']} bars are visible; exactly one is expected "
                                f"({json.dumps(settled)})"
                            )
                        elif not settled["barIsOnMessageUnderPointer"]:
                            failures.append(
                                f"{engine}: after a scroll with the pointer still, the action "
                                f"bar is on message {settled['barOwnerId']} but the pointer is "
                                f"over {settled['underPointerId']}; the bar is stranded "
                                f"({json.dumps(settled)})"
                            )
                    if after["shown"] != 0:
                        failures.append(
                            f"{engine}: {after['shown']} bars still visible after the pointer "
                            "left; the bar does not hide again"
                        )
                except Exception as exc:  # noqa: BLE001
                    results["engines"][engine] = {"failed": repr(exc)}
                    failures.append(f"{engine}: {exc!r}")
                finally:
                    ctx.close()
                    browser.close()
        (OUT / f"{LABEL}.json").write_text(
            json.dumps(results, indent = 2), encoding = "utf-8"
        )
    finally:
        if vite is not None:
            stop_process(vite)
    if failures:
        info("")
        for f in failures:
            info(f"FAIL {f}")
        return 1
    info("all engines: hidden at rest, exactly one shown on hover, hidden again after")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
