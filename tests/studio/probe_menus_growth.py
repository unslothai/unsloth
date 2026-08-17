# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Does EVERY Radix menu mounted over a long thread pay the body pointer-events cost?

The action menu was fixed by `modal={false}`, but the mechanism is not specific to it:
a modal Radix layer writes `pointer-events` onto `<body>`, an INHERITED property, so the
whole mounted thread subtree is invalidated no matter which menu opened. This walks every
open-able menu trigger on the page at two thread sizes and reports open+close per menu,
so "extend the fix" is a decision with a per-menu number behind it rather than a guess.

`bodyPointerEvents` while open is the discriminator: `none` means the menu took the modal
path and pays the cost, `auto` means it did not.

Usage: python tests/studio/probe_menus_growth.py --label before --chars 25000,300000
Writes logs/pw/menus_<label>.json
"""

from __future__ import annotations

import argparse
import json
import statistics
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

# Radix menu triggers carry aria-haspopup="menu". Selects use listbox, so both are
# enumerated: the expansion list contains only menus and selects.
ENUMERATE_JS = """
() => {
  const nodes = Array.from(
    document.querySelectorAll('button[aria-haspopup="menu"], button[aria-haspopup="listbox"]')
  );
  return nodes.map((el, i) => {
    el.setAttribute("data-menuprobe", String(i));
    const r = el.getBoundingClientRect();
    return {
      index: i,
      name: (el.textContent || el.getAttribute("aria-label") || "").trim().slice(0, 48),
      haspopup: el.getAttribute("aria-haspopup"),
      visible: r.width > 0 && r.height > 0,
    };
  });
}
"""

# Open by real pointer events (Radix triggers open on pointerdown, not click), then wait
# for the portaled content. Timed the same way the 9016 harness times the action menu:
# the clock starts before the dispatch, because Radix does its layer work synchronously
# inside it.
CYCLE_JS = """
async (index) => {
  const el = document.querySelector(`[data-menuprobe="${index}"]`);
  if (!el) return { ok: false, why: "trigger vanished" };
  const isOpen = () => Boolean(
    document.querySelector('[data-radix-menu-content], [data-radix-select-content], [role="menu"][data-state="open"]')
  );
  if (isOpen()) return { ok: false, why: "something already open" };
  const pointer = {
    bubbles: true, cancelable: true, composed: true,
    button: 0, pointerId: 1, pointerType: "mouse", isPrimary: true,
  };
  const settle = async (want, budget) => {
    const t0 = performance.now();
    while (performance.now() - t0 < budget) {
      if (isOpen() === want) return performance.now() - t0;
      await new Promise((r) => requestAnimationFrame(r));
    }
    return null;
  };
  const openStart = performance.now();
  el.scrollIntoView({ block: "center" });
  el.dispatchEvent(new PointerEvent("pointerdown", { ...pointer, buttons: 1 }));
  el.dispatchEvent(new PointerEvent("pointerup", { ...pointer, buttons: 0 }));
  const openedIn = await settle(true, 20000);
  if (openedIn === null) return { ok: false, why: "never opened" };
  const openMs = performance.now() - openStart;
  const bodyPointerEvents = getComputedStyle(document.body).pointerEvents;

  const closeStart = performance.now();
  document.dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  const closedIn = await settle(false, 20000);
  const closeMs = closedIn === null ? null : performance.now() - closeStart;
  return {
    ok: closeMs !== null,
    openMs: Math.round(openMs * 10) / 10,
    closeMs: closeMs === null ? null : Math.round(closeMs * 10) / 10,
    open_close_ms: closeMs === null ? null : Math.round((openMs + closeMs) * 10) / 10,
    bodyPointerEvents,
  };
}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required = True)
    ap.add_argument("--chars", default = "25000,300000")
    ap.add_argument("--reps", type = int, default = 3)
    ap.add_argument("--engine", default = "chromium")
    args = ap.parse_args()

    sizes = [int(c) for c in args.chars.split(",")]

    from playwright.sync_api import sync_playwright

    results: dict = {"label": args.label, "engine": args.engine, "by_size": {}}

    vite = start_vite(PORT) if OWNS_SERVER else None
    try:
        wait_for_smoke_page(
            f"{BASE}/smoke-heavy-thread.html",
            "smoke-heavy-thread-main.tsx",
            proc = vite,
            info = lambda m: print(f"[menus] {m}", flush = True),
        )
        with sync_playwright() as pw:
            browser = getattr(pw, args.engine).launch(headless = True)
            for size in sizes:
                print(f"[menus] seeding {size}", flush = True)
                page = browser.new_page(viewport = {"width": 1440, "height": 950})
                page.goto(f"{BASE}/smoke-heavy-thread.html",
                          wait_until = "domcontentloaded")
                page.wait_for_function("() => Boolean(window.__heavyThread)",
                                       timeout = 60_000)
                plan = page.evaluate("(n) => window.__heavyThread.seed(n)", size)
                page.wait_for_function(
                    "(n) => window.__heavyThread.messageCount() >= n",
                    arg = plan["messages"], timeout = 300_000,
                )
                page.wait_for_timeout(1500)

                triggers = [t for t in page.evaluate(ENUMERATE_JS) if t["visible"]]
                print(f"[menus] {size}: {len(triggers)} visible menu triggers",
                      flush = True)
                per_menu: dict = {}
                for t in triggers:
                    samples = []
                    body_pe = None
                    for _ in range(args.reps):
                        r = page.evaluate(CYCLE_JS, t["index"])
                        if r.get("ok"):
                            samples.append(r["open_close_ms"])
                            body_pe = r["bodyPointerEvents"]
                        page.keyboard.press("Escape")
                        page.wait_for_timeout(250)
                    key = t["name"] or f"trigger#{t['index']}"
                    per_menu[key] = {
                        "haspopup": t["haspopup"],
                        "samples": samples,
                        "median_open_close_ms": (
                            round(statistics.median(samples), 1) if samples else None
                        ),
                        "bodyPointerEvents": body_pe,
                    }
                    print(f"[menus]   {key}: {per_menu[key]['median_open_close_ms']} ms "
                          f"body={body_pe}", flush = True)
                results["by_size"][str(size)] = {
                    "messages": plan["messages"], "menus": per_menu,
                }
                page.close()
            browser.close()
    finally:
        if vite is not None:
            stop_process(vite)

    out = Path("logs/pw") / f"menus_{args.label}_{args.engine}.json"
    out.parent.mkdir(parents = True, exist_ok = True)
    out.write_text(json.dumps(results, indent = 2), encoding = "utf-8")
    print(f"[menus] wrote {out}", flush = True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
