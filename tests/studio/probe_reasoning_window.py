# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What the windowed thinking pane costs when the reader asks for more of it.

tests/studio/playwright_reasoning_pane.py measures the case the field report is about: a reader
watching a thinking block stream. It says nothing about the two cases where the window has to give
content BACK, and those are where a window stops being a performance change and starts being a
correctness question:

    SCROLL BACK WHILE STREAMING. Reaching the top of the mounted window widens it, mounting more
    above. Widening is the thing this change is trying to avoid doing, so the honest question is
    how much of the cost comes back per widen, and how far a reader can scroll before the pane is
    as expensive as it was without the window at all.

    EXPAND AFTER COMPLETION. The group collapses when the round ends. If the reader opens it, the
    whole body has to become reachable. What matters there is time to FIRST painted content, and
    whether a full expand of a 130,000-character body is survivable at all.

Both are probes: they print what they measured and fail only when the probe itself did not
measure what it claims (no widen happened, the group never opened, the body never completed).
Budgets belong in the harness, set from numbers taken on real hardware.

It drives the DOM directly rather than through a page API, so it works unchanged against a tree
with no window in it. That is the point: every number here is reported for BOTH trees, and the
baseline column is what says whether a widen is expensive in absolute terms or only relative to a
windowed pane.

Run:
    python tests/studio/probe_reasoning_window.py
    SMOKE_RP_ENGINES=firefox PROBE_CHARS=130000 python tests/studio/probe_reasoning_window.py
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

PORT = int(os.environ.get("SMOKE_PORT", "5471"))
BASE = os.environ.get("SMOKE_BASE_URL", "").strip().rstrip("/") or f"http://127.0.0.1:{PORT}"
OWNS_SERVER = not os.environ.get("SMOKE_BASE_URL", "").strip()
ENGINE = os.environ.get("SMOKE_RP_ENGINES", "firefox").split(",")[0].strip()
CHARS = int(os.environ.get("PROBE_CHARS", "130000"))
GAP_MS = int(os.environ.get("PROBE_GAP_MS", "8"))
# Fast by default: this probe is about what a gesture costs at a given amount of content, not
# about the streaming curve, so there is no reason to spend the capture's 400 seconds getting
# there. The pacing precondition applies to the harness, not to this.
WIDENS = int(os.environ.get("PROBE_WIDENS", "8"))
LABEL = os.environ.get("SMOKE_LABEL", "tree")
OUT = Path(os.environ.get("PW_ART_DIR", "logs/probe-reasoning-window"))
OUT.mkdir(parents = True, exist_ok = True)


def info(message: str) -> None:
    print(f"[reasoning-window] {message}", flush = True)


# Records frame gaps continuously so any gesture can be bracketed and asked what it cost.
RECORDER = """
(() => {
  const raf = window.requestAnimationFrame.bind(window);
  const R = { gaps: [], last: performance.now() };
  window.__probe = R;
  const tick = () => {
    const now = performance.now();
    R.gaps.push(now - R.last);
    R.last = now;
    raf(tick);
  };
  raf(tick);
  window.__probeMark = () => { R.gaps.length = 0; R.last = performance.now(); };
  window.__probeRead = () => {
    const gaps = R.gaps.slice().sort((a, b) => a - b);
    return {
      frames: gaps.length,
      p50: gaps.length ? gaps[Math.floor(gaps.length / 2)] : null,
      worst: gaps.length ? gaps[gaps.length - 1] : null,
    };
  };
  window.__pane = () => document.querySelector(".aui-reasoning-text");
  window.__paneStats = () => {
    const pane = window.__pane();
    return {
      chars: pane ? (pane.textContent ?? "").length : 0,
      spans: document.querySelectorAll(".aui-reasoning-text pre code span").length,
      elements: document.querySelectorAll(".aui-reasoning-text *").length,
      scrollTop: pane ? pane.scrollTop : -1,
      scrollHeight: pane ? pane.scrollHeight : -1,
      clientHeight: pane ? pane.clientHeight : -1,
    };
  };
})();
"""


def scroll_back(page) -> list[dict]:
    """Drive the reader to the top of the pane repeatedly, measuring each widen."""
    steps: list[dict] = []
    for index in range(WIDENS):
        before = page.evaluate("() => window.__paneStats()")
        page.evaluate("() => window.__probeMark()")
        page.evaluate("() => { const p = window.__pane(); if (p) p.scrollTop = 0; }")
        # Long enough for the widen commit, the highlighter and the settle loop that holds the
        # reader's place while the new fences reach their real height.
        page.wait_for_timeout(700)
        frames = page.evaluate("() => window.__probeRead()")
        after = page.evaluate("() => window.__paneStats()")
        steps.append(
            {
                "step": index + 1,
                "chars_before": before["chars"],
                "chars_after": after["chars"],
                "elements_before": before["elements"],
                "elements_after": after["elements"],
                "spans_after": after["spans"],
                "worst_frame_ms": frames["worst"],
                "p50_frame_ms": frames["p50"],
                # How far the reader ended up from where they were reading. The widen mounts
                # content ABOVE them, so a pane that jumps shows up here and nowhere else.
                "scroll_top_after": after["scrollTop"],
                "distance_from_bottom_before": (
                    before["scrollHeight"] - before["scrollTop"] - before["clientHeight"]
                ),
                "distance_from_bottom_after": (
                    after["scrollHeight"] - after["scrollTop"] - after["clientHeight"]
                ),
            }
        )
        if after["chars"] <= before["chars"]:
            # Nothing more to widen into: the whole body is mounted.
            break
    return steps


def expand_after_completion(page) -> dict:
    """Open a finished thinking group and time what the reader waits for."""
    page.evaluate("() => window.__probeMark()")
    opened = page.evaluate(
        """() => {
          const trigger = document.querySelector('[data-slot="reasoning-trigger"]');
          if (!trigger) return false;
          trigger.click();
          return true;
        }"""
    )
    if not opened:
        return {"opened": False}

    first_paint_ms = page.evaluate(
        """async () => {
          const started = performance.now();
          for (let i = 0; i < 1200; i += 1) {
            const pane = document.querySelector(".aui-reasoning-text");
            if (pane && (pane.textContent ?? "").length > 0) return performance.now() - started;
            await new Promise((r) => requestAnimationFrame(() => r()));
          }
          return null;
        }"""
    )
    # Fully mounted means the pane has stopped growing, not that it reached any particular size:
    # the probe does not know how long the reasoning was.
    full = page.evaluate(
        """async () => {
          const started = performance.now();
          let last = -1;
          let quiet = 0;
          for (let i = 0; i < 3000; i += 1) {
            const pane = document.querySelector(".aui-reasoning-text");
            const chars = pane ? (pane.textContent ?? "").length : 0;
            if (chars > last) { last = chars; quiet = 0; } else { quiet += 1; }
            if (quiet >= 45) return { ms: performance.now() - started, chars };
            await new Promise((r) => requestAnimationFrame(() => r()));
          }
          return { ms: null, chars: last };
        }"""
    )
    frames = page.evaluate("() => window.__probeRead()")
    stats = page.evaluate("() => window.__paneStats()")
    return {
        "opened": True,
        "first_paint_ms": first_paint_ms,
        "fully_mounted_ms": full["ms"],
        "chars": stats["chars"],
        "spans": stats["spans"],
        "elements": stats["elements"],
        "worst_frame_ms": frames["worst"],
    }


def main() -> int:
    vite = start_vite(PORT) if OWNS_SERVER else None
    try:
        if vite is not None:
            wait_for_smoke_page(
                f"{BASE}/smoke-reasoning-pane.html",
                "smoke-reasoning-pane-main.tsx",
                proc = vite,
            )
        with sync_playwright() as p:
            launcher = getattr(p, ENGINE)
            kwargs: dict = {"headless": True}
            if ENGINE == "chromium":
                kwargs["args"] = chromium_launch_args()
            browser = launcher.launch(**kwargs)
            context = browser.new_context()
            context.add_init_script(RECORDER)
            page = context.new_page()
            page.route(
                "**/*",
                lambda route: (
                    route.fulfill(status = 200, content_type = "application/json", body = "{}")
                    if route.request.url.startswith(f"{BASE}/api/")
                    else route.continue_()
                ),
            )
            page.set_viewport_size({"width": 1280, "height": 900})
            page.goto(f"{BASE}/smoke-reasoning-pane.html", wait_until = "domcontentloaded")
            page.wait_for_function("() => Boolean(window.__reasoningPane)", timeout = 120_000)

            page.evaluate(
                "(cfg) => window.__reasoningPane.run(cfg)",
                {
                    "totalChars": CHARS,
                    "fenceChars": 1800,
                    "proseChars": 1250,
                    "preambleChars": int(CHARS * 0.25),
                    "chunkChars": 24,
                    "gapMs": GAP_MS,
                },
            )
            # Most of the way through, so there is a real amount of content above the reader.
            page.wait_for_function(
                "(mark) => window.__reasoningPane.streamState().sentChars >= mark",
                arg = int(CHARS * 0.8),
                timeout = 900_000,
            )
            mid = page.evaluate("() => window.__paneStats()")
            info(
                f"streaming, reader at the end: {mid['chars']:,} chars, "
                f"{mid['spans']:,} spans, {mid['elements']:,} pane elements"
            )
            steps = scroll_back(page)

            page.wait_for_function(
                "() => window.__reasoningPane.streamState().done === true", timeout = 900_000
            )
            page.wait_for_timeout(2500)
            collapsed = page.evaluate("() => window.__paneStats()")
            expand = expand_after_completion(page)

            print("\n=== scroll back while streaming ===", flush = True)
            print(
                f"{'step':>5} {'paneChars':>11} {'paneEl':>9} {'spans':>8} "
                f"{'worstFrame':>11} {'p50':>7} {'dFromBottom':>12}",
                flush = True,
            )
            for s in steps:
                worst = s["worst_frame_ms"]
                p50 = s["p50_frame_ms"]
                print(
                    f"{s['step']:>5} {s['chars_after']:>11,} {s['elements_after']:>9,} "
                    f"{s['spans_after']:>8,} "
                    f"{(worst if worst is not None else -1):>11.1f} "
                    f"{(p50 if p50 is not None else -1):>7.1f} "
                    f"{s['distance_from_bottom_after']:>12,}",
                    flush = True,
                )

            print("\n=== expand after completion ===", flush = True)
            print(f"  collapsed to {collapsed['elements']:,} pane elements", flush = True)
            if expand.get("opened"):
                print(
                    f"  first painted content   {expand['first_paint_ms']} ms\n"
                    f"  fully mounted           {expand['fully_mounted_ms']} ms\n"
                    f"  worst frame in the open {expand['worst_frame_ms']} ms\n"
                    f"  ended at {expand['chars']:,} chars, {expand['spans']:,} spans, "
                    f"{expand['elements']:,} pane elements",
                    flush = True,
                )
            else:
                print("  the reasoning trigger was not found", flush = True)

            payload = {
                "label": LABEL,
                "engine": ENGINE,
                "chars": CHARS,
                "mid_stream": mid,
                "scroll_back": steps,
                "collapsed": collapsed,
                "expand": expand,
            }
            out = OUT / f"probe-reasoning-window-{LABEL}-{ENGINE}.json"
            out.write_text(json.dumps(payload, indent = 2))
            info(f"wrote {out}")

            context.close()
            browser.close()

            bad = []
            if not steps:
                bad.append("no scroll-back step ran, so nothing about widening was measured")
            if not expand.get("opened"):
                bad.append("the finished group never opened, so the expand was not measured")
            elif expand.get("first_paint_ms") is None:
                bad.append("the expanded group never painted any content")
            if bad:
                print("\nPROBE BROKEN", flush = True)
                for b in bad:
                    print(f"  - {b}", flush = True)
                return 1
    finally:
        if vite is not None:
            stop_process(vite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
