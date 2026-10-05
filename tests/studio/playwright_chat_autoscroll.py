# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-frame cost and intent correctness of the main chat viewport's autoscroll (#8483).

The chat viewport follows the bottom through a 600ms window that every mutation re-arms, so a
streaming message keeps the window open for its whole duration. What that window costs depends
on whether the frame chain re-arms unconditionally or only while layout is still moving, which
is the change PR #8525 made to the research activity panel's equivalent loop.

Four phases, and the third is the one that decides whether the port is worth making:

    stream  - tokens and blocks at a realistic cadence; how many frames does the loop run.
    idle    - the loop must stop when the content goes quiet.
    silent  - content grows via an inline style, which the MutationObserver deliberately
              excludes and a border-box ResizeObserver cannot see. Only a frame that reads
              layout notices. This models a decoding image, a font-display: swap webfont and a
              late KaTeX pass. Measures how long the view stays off the bottom.
    intent  - scrolling up detaches and stays detached through further streaming; scrolling
              back re-attaches. A cheaper loop must not cost any of this.

requestAnimationFrame is pumped on a fixed 16ms timer: Chromium here produces only a couple of
real frames a second, which would flatten a runaway per-frame loop into a passing number. The
counts are therefore a property of the code, not of this machine's compositor. Chromium is also
not the WebKitGTK webview the desktop app embeds, so what transfers is the work, not the
absolute timings.

Run:
    python tests/studio/playwright_chat_autoscroll.py

It starts and stops its own vite dev server. Point it at one you already have with
SMOKE_BASE_URL, or move the port it picks with SMOKE_PORT.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    echo_browser_errors,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

PORT = int(os.environ.get("SMOKE_PORT", "5193"))
# Unset: start and stop our own server.
_EXTERNAL = os.environ.get("SMOKE_BASE_URL", "").strip()
BASE = _EXTERNAL or f"http://127.0.0.1:{PORT}"
OWNS_SERVER = not _EXTERNAL
LABEL = os.environ.get("SMOKE_LABEL", "tree")
OUT = Path(os.environ.get("PW_ART_DIR", "logs/playwright-chat-autoscroll"))
OUT.mkdir(parents = True, exist_ok = True)

# A token every 250ms for 8s: the deep research synthesis cadence, and the one the loop is wasteful at.
# A faster cadence (SMOKE_TOKEN_GAP_MS=40) is the one case where a frame per token is justified, so measuring only there
# hides the effect.
TOKEN_COUNT = int(os.environ.get("SMOKE_TOKEN_COUNT", "32"))
TOKEN_GAP_MS = int(os.environ.get("SMOKE_TOKEN_GAP_MS", "250"))
# One frame per token at this cadence is 4/s; the old loop ran at the pump's ceiling, around 62/s. 25/s leaves room for
# React's own frames and still fails loudly on a return to the ceiling.
MAX_STREAM_RAF_PER_SECOND = float(os.environ.get("SMOKE_MAX_RAF_PER_S", "25"))
# The follow window in the hook.
FOLLOW_SETTLE_MS = 600
# What the settle check trades away: unobservable growth is followed on a timer, not the next
# frame. Generous against 115ms measured, tight enough to catch a regression.
SILENT_GROWTH_REPIN_BUDGET_MS = int(os.environ.get("SMOKE_REPIN_BUDGET_MS", "250"))

PUMP_INIT = """
(() => {
  window.__longTasks = [];
  try {
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        window.__longTasks.push({ start: entry.startTime, duration: entry.duration });
      }
    }).observe({ type: "longtask", buffered: true });
  } catch (e) { /* longtask unsupported: the CDP metrics still apply */ }
  window.__rafCount = 0;
  let nextHandle = 1;
  const pending = new Map();
  window.requestAnimationFrame = (cb) => {
    const handle = nextHandle++;
    pending.set(
      handle,
      setTimeout(() => {
        pending.delete(handle);
        window.__rafCount += 1;
        cb(performance.now());
      }, 16),
    );
    return handle;
  };
  window.cancelAnimationFrame = (handle) => {
    const timer = pending.get(handle);
    if (timer !== undefined) {
      clearTimeout(timer);
      pending.delete(handle);
    }
  };
})();
"""


def info(message: str) -> None:
    print(f"[chat-autoscroll] {message}", flush = True)


def metrics(cdp) -> dict[str, float]:
    got = cdp.send("Performance.getMetrics")
    return {m["name"]: m["value"] for m in got["metrics"]}


def delta(before: dict[str, float], after: dict[str, float], name: str) -> float:
    return round(after.get(name, 0.0) - before.get(name, 0.0), 4)


def run() -> dict:
    results: dict = {"label": LABEL, "base": BASE}
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless = os.environ.get("SMOKE_HEADLESS", "1") == "1",
            args = chromium_launch_args(),
        )
        context = browser.new_context(viewport = {"width": 1440, "height": 900})
        context.add_init_script(PUMP_INIT)
        context.add_init_script(
            "localStorage.setItem('unsloth_auth_token', 'chat-autoscroll-smoke');"
        )
        context.route(
            re.compile(rf"^{re.escape(BASE)}/api/"),
            lambda route: route.fulfill(status = 200, content_type = "application/json", body = "{}"),
        )
        page = context.new_page()
        echo_browser_errors(page, info)
        page.goto(f"{BASE}/smoke-autoscroll.html", wait_until = "domcontentloaded")
        page.wait_for_function("() => Boolean(window.__autoscroll)", timeout = 30_000)
        cdp = context.new_cdp_session(page)
        cdp.send("Performance.enable")

        page.evaluate("window.__autoscroll.seed(40)")
        page.wait_for_timeout(800)
        results["seeded"] = page.evaluate("window.__autoscroll.metrics()")

        # Streaming. Clock and counter start and stop together inside the page, so they
        # bracket one interval; split across round trips they do not, and the rate drifts.
        before = metrics(cdp)
        streamed = page.evaluate(
            """async ([count, gap, tailMs]) => {
                window.__longTasks.length = 0;
                window.__rafCount = 0;
                const started = performance.now();
                for (let i = 0; i < count; i += 1) {
                    window.__autoscroll.token("token " + i + " ");
                    // A finished message every 20 tokens, so childList mutations are in the mix
                    // alongside the characterData ones.
                    if (i > 0 && i % 20 === 0) window.__autoscroll.block();
                    await new Promise((r) => setTimeout(r, gap));
                }
                // The tail is part of the measurement: re-arming after the last token is the
                // failure mode, so count those frames rather than stopping the clock early.
                await new Promise((r) => setTimeout(r, tailMs));
                return { wallMs: performance.now() - started, rafCallbacks: window.__rafCount };
            }""",
            [TOKEN_COUNT, TOKEN_GAP_MS, 1200],
        )
        after = metrics(cdp)
        long_tasks = page.evaluate("window.__longTasks")
        results["stream"] = {
            "tokens": TOKEN_COUNT,
            "wall_ms": round(streamed["wallMs"], 1),
            "raf_callbacks": streamed["rafCallbacks"],
            "long_tasks": len(long_tasks),
            "worst_long_task_ms": round(max((t["duration"] for t in long_tasks), default = 0.0), 1),
            "layout_count": delta(before, after, "LayoutCount"),
            "recalc_style_count": delta(before, after, "RecalcStyleCount"),
            "layout_ms": round(delta(before, after, "LayoutDuration") * 1000, 1),
            "recalc_style_ms": round(delta(before, after, "RecalcStyleDuration") * 1000, 1),
            "task_ms": round(delta(before, after, "TaskDuration") * 1000, 1),
            "pinned_after_stream": page.evaluate("window.__autoscroll.distanceFromBottom()"),
            "is_at_bottom": page.evaluate("window.__autoscroll.isAtBottom()"),
        }

        page.evaluate("window.__rafCount = 0")
        page.wait_for_timeout(2000)
        results["idle_raf_per_2s"] = page.evaluate("window.__rafCount")

        # Silent growth. A token first to open a fresh follow window, then growth as an inline style, which
        # reaches neither observer.
        page.evaluate("window.__autoscroll.resetGrowth()")
        page.wait_for_timeout(900)
        settled = page.evaluate(
            """async ([followMs]) => {
                window.__autoscroll.token("x ");
                // 100ms in: inside the follow window, which is where a decoding image lands.
                await new Promise((r) => setTimeout(r, 100));
                const start = performance.now();
                window.__autoscroll.growSilently(240);
                let repinnedAfterMs = null;
                // Watch past the window's end plus a margin, so a settle check at the deadline
                // is still counted rather than reported as a permanent failure to follow.
                while (performance.now() - start < followMs + 600) {
                    if (window.__autoscroll.distanceFromBottom() <= 2) {
                        repinnedAfterMs = performance.now() - start;
                        break;
                    }
                    await new Promise((r) => setTimeout(r, 8));
                }
                return {
                    repinnedAfterMs,
                    distanceAtEnd: window.__autoscroll.distanceFromBottom(),
                    isAtBottom: window.__autoscroll.isAtBottom(),
                };
            }""",
            [FOLLOW_SETTLE_MS],
        )
        results["silent_growth"] = settled

        page.evaluate("window.__autoscroll.resetGrowth()")
        page.evaluate("window.__autoscroll.scrollToBottom()")
        page.wait_for_timeout(900)
        intent = page.evaluate(
            """async () => {
                window.__autoscroll.scrollUpBy(400);
                await new Promise((r) => setTimeout(r, 200));
                const detached = !window.__autoscroll.isAtBottom();
                const distanceAfterDetach = window.__autoscroll.distanceFromBottom();
                for (let i = 0; i < 12; i += 1) {
                    window.__autoscroll.token("more " + i + " ");
                    await new Promise((r) => setTimeout(r, 40));
                }
                await new Promise((r) => setTimeout(r, 900));
                const stillDetached = !window.__autoscroll.isAtBottom();
                const grewWhileDetached =
                    window.__autoscroll.distanceFromBottom() > distanceAfterDetach;
                // Scroll back to within the re-attach threshold.
                const before = window.__autoscroll.distanceFromBottom();
                window.__autoscroll.scrollDownBy(before);
                await new Promise((r) => setTimeout(r, 400));
                return {
                    detached,
                    stillDetached,
                    grewWhileDetached,
                    reattached: window.__autoscroll.isAtBottom(),
                };
            }"""
        )
        results["intent"] = intent

        context.close()
        browser.close()
    return results


def main() -> int:
    vite = None
    if OWNS_SERVER:
        info(f"starting vite dev server on port {PORT}")
        vite = start_vite(PORT)
    try:
        wait_for_smoke_page(
            f"{BASE}/smoke-autoscroll.html", "smoke-autoscroll-main.tsx", proc = vite, info = info
        )
        results = run()
    finally:
        if vite is not None:
            stop_process(vite)
            info("vite stopped")
    stream_seconds = max(0.001, results["stream"]["wall_ms"] / 1000)
    raf_rate = results["stream"]["raf_callbacks"] / stream_seconds
    results["stream"]["raf_per_second"] = round(raf_rate, 1)
    out = OUT / f"{LABEL}.json"
    out.write_text(json.dumps(results, indent = 2), encoding = "utf-8")
    info(json.dumps(results, indent = 2))
    info(f"wrote {out}")

    failures: list[str] = []
    if results["seeded"]["scrollHeight"] <= results["seeded"]["clientHeight"]:
        failures.append("the seeded viewport does not overflow; nothing was measured")
    if results["stream"]["pinned_after_stream"] > 2:
        failures.append("the viewport did not stay pinned through the stream")
    if not results["stream"]["is_at_bottom"]:
        failures.append("isAtBottom went false while following")
    # Recording the count without asserting on it was false-green: the unconditional loop ran 503 callbacks over this
    # stream and the script still exited 0.
    if raf_rate > MAX_STREAM_RAF_PER_SECOND:
        failures.append(
            f"the follow loop ran at {raf_rate:.1f} rAF/s during the stream, over the "
            f"{MAX_STREAM_RAF_PER_SECOND} rAF/s budget"
        )
    if results["idle_raf_per_2s"] > 8:
        failures.append(
            f"the loop still runs when idle ({results['idle_raf_per_2s']} frames in 2s)"
        )
    intent = results["intent"]
    if not intent["detached"]:
        failures.append("scrolling up did not detach")
    # Without this the check above is vacuous: if the tokens streamed while detached add no height, "streaming
    # re-pinned a detached reader" passes on a tree where following is broken in either direction.
    if not intent["grewWhileDetached"]:
        failures.append("the content did not grow while detached; the re-pin check proved nothing")
    if not intent["stillDetached"]:
        failures.append("streaming re-pinned a detached reader")
    if not intent["reattached"]:
        failures.append("scrolling back to the bottom did not re-attach")
    repinned_after_ms = results["silent_growth"]["repinnedAfterMs"]
    if repinned_after_ms is None:
        failures.append("silent growth was never followed inside the window")
    # The settle timer is 100ms plus the frame it schedules, measured at 115ms.
    elif repinned_after_ms > SILENT_GROWTH_REPIN_BUDGET_MS:
        failures.append(
            f"silent growth took {repinned_after_ms:.0f}ms to follow, over the "
            f"{SILENT_GROWTH_REPIN_BUDGET_MS}ms budget"
        )
    for problem in failures:
        info(f"FAIL {problem}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
