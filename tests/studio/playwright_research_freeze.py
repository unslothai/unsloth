# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Streaming research main-thread cost in Chromium; absolute times do not transfer to WebKitGTK."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    echo_browser_errors,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

PORT = int(os.environ.get("SMOKE_PORT", "5183"))
# Exported-but-empty counts as unset.
_EXTERNAL = os.environ.get("SMOKE_BASE_URL", "").strip()
BASE = _EXTERNAL or f"http://127.0.0.1:{PORT}"
OWNS_SERVER = not _EXTERNAL
LABEL = os.environ.get("SMOKE_LABEL", "tree")
OUT = Path(os.environ.get("PW_ART_DIR", "logs/playwright-research-freeze"))
OUT.mkdir(parents = True, exist_ok = True)

# ~12.5 events/s is what the store's 80ms coalescing window admits.
DELTA_COUNT = int(os.environ.get("SMOKE_DELTA_COUNT", "240"))
DELTA_GAP_MS = int(os.environ.get("SMOKE_DELTA_GAP_MS", "80"))

# The frame pump runs 62/s and a self-chaining rAF loop sits on it; the fixed tree uses ~29/s.
MAX_STREAM_RAF_PER_SECOND = float(os.environ.get("SMOKE_MAX_RAF_PER_S", "45"))
# Idle measures 0; a few frames of slack cover a settle check at the window edge.
MAX_IDLE_RAF_PER_2S = int(os.environ.get("SMOKE_MAX_IDLE_RAF", "4"))
# Measured 132ms to 416ms depending on load; 1000 still fails ten times the report size.
MAIN_THREAD_STALL_BUDGET_MS = int(os.environ.get("SMOKE_REPORT_STALL_BUDGET_MS", "1000"))

REPORT_SECTION = """
## Section {n}

British cultural exports carry {n} distinct threads, and the reception of each varies by
audience, decade and medium. The paragraph below exists to give the renderer real prose to
lay out, with **emphasis**, `inline code`, and a [link](https://example.invalid/{n}).

| Aspect | Reception | Note |
| --- | --- | --- |
| Broadcasting | Mixed | Public service model |
| Music | Positive | Export driven |

```python
def section_{n}(values):
    return sum(value * {n} for value in values)
```
"""


def info(message: str) -> None:
    print(f"[research-freeze] {message}", flush = True)


def until(
    page,
    condition: str,
    timeout: int = 5000,
) -> None:
    """Wait for the state a check reads, without failing here: a broken build spends the bound and
    then fails on the same check. Replaces flat sleeps, which also asserted the runner's speed."""
    try:
        page.wait_for_function(condition, timeout = timeout)
    except PlaywrightTimeout:
        pass


def frames(page) -> None:
    page.evaluate("() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))")


def build_report(sections: int) -> str:
    body = "\n".join(REPORT_SECTION.format(n = n) for n in range(1, sections + 1))
    return f"# Deep research report\n\n{body}\n\n$$\\sum_{{i=1}}^{{n}} x_i^2$$\n"


LONGTASK_INIT = """
(() => {
  window.__longTasks = [];
  try {
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        window.__longTasks.push({ start: entry.startTime, duration: entry.duration });
      }
    }).observe({ type: "longtask", buffered: true });
  } catch (e) { /* longtask unsupported: the CDP metrics below still apply */ }
  // A timer-driven frame pump replaces requestAnimationFrame. Chromium in a container produces
  // only a couple of real frames a second (software rendering, offscreen), which flattens a
  // per-frame loop into nothing and would let a runaway one pass. Pumping at a fixed 16ms makes
  // the count a property of the code under test rather than of this machine's compositor.
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


SCROLLER = '[aria-label="Research activity timeline"]'
DIALOG_OPEN = "() => Boolean(document.querySelector('[role=\"dialog\"]'))"
BODY_LOCKED = "() => document.body.style.pointerEvents === 'none'"
DISTANCE_FROM_BOTTOM = (
    f"() => {{ const el = document.querySelector('{SCROLLER}');"
    " return Math.round(el.scrollHeight - el.scrollTop - el.clientHeight); }"
)
LATEST_BUTTON_VISIBLE = (
    "() => [...document.querySelectorAll('button')]"
    ".some((button) => button.textContent.trim() === 'Latest')"
)


def metrics(cdp) -> dict[str, float]:
    got = cdp.send("Performance.getMetrics")
    return {m["name"]: m["value"] for m in got["metrics"]}


def delta(before: dict[str, float], after: dict[str, float], name: str) -> float:
    return round(after.get(name, 0.0) - before.get(name, 0.0), 4)


def run() -> dict:
    results: dict = {"label": LABEL, "base": BASE}
    with sync_playwright() as p:
        headless = os.environ.get("SMOKE_HEADLESS", "1") == "1"
        browser = p.chromium.launch(headless = headless, args = chromium_launch_args())
        context = browser.new_context(viewport = {"width": 1440, "height": 900})
        # Deliberately NOT installing the view-transition killer: it forces
        # `body { pointer-events: auto !important }`, the symptom under test.
        context.add_init_script(LONGTASK_INIT)
        # Without a token and 200s the auth guard redirects to /login mid-run. Anchored on the origin to
        # skip vite module URLs under src/.../api/.
        context.add_init_script(
            "localStorage.setItem('unsloth_auth_token', 'research-freeze-smoke');"
        )
        context.route(
            re.compile(rf"^{re.escape(BASE)}/api/"),
            lambda route: route.fulfill(status = 200, content_type = "application/json", body = "{}"),
        )
        page = context.new_page()
        echo_browser_errors(page, info)
        page.goto(f"{BASE}/smoke-research.html", wait_until = "domcontentloaded")
        page.wait_for_function("() => Boolean(window.__research)", timeout = 30_000)
        cdp = context.new_cdp_session(page)
        cdp.send("Performance.enable")

        page.evaluate("window.__research.seed()")
        until(page, f"() => Boolean(document.querySelector('{SCROLLER}'))")
        frames(page)
        activities_before_stream = page.evaluate("window.__research.state().activities")

        before = metrics(cdp)
        page.evaluate(
            """async ([count, gap]) => {
                window.__longTasks.length = 0;
                window.__rafCount = 0;
                for (let i = 0; i < count; i += 1) {
                    window.__research.delta("token " + i + " ");
                    if (i % 4 === 0) window.__research.reportDelta(i * 32);
                    // A new step every 8 deltas, so the list grows the way a real run's does
                    // rather than mutating one row in place.
                    if (i % 8 === 0) window.__research.step(i / 8);
                    await new Promise((r) => setTimeout(r, gap));
                }
            }""",
            [DELTA_COUNT, DELTA_GAP_MS],
        )
        # The tail is part of the measurement: re-arming after the last event is the failure mode, so count those
        # frames rather than stopping the clock. A measurement window, not a readiness wait.
        page.wait_for_timeout(1200)
        stream_window_ms = DELTA_COUNT * DELTA_GAP_MS + 1200
        stream_raf = page.evaluate("window.__rafCount")
        after = metrics(cdp)
        long_tasks = page.evaluate("window.__longTasks")
        results["stream"] = {
            "events": DELTA_COUNT,
            "wall_ms": stream_window_ms,
            "raf_callbacks": stream_raf,
            "raf_per_second": round(stream_raf / (stream_window_ms / 1000), 1),
            "long_tasks": len(long_tasks),
            "worst_long_task_ms": round(max((t["duration"] for t in long_tasks), default = 0.0), 1),
            "layout_count": delta(before, after, "LayoutCount"),
            "recalc_style_count": delta(before, after, "RecalcStyleCount"),
            "layout_ms": round(delta(before, after, "LayoutDuration") * 1000, 1),
            "recalc_style_ms": round(delta(before, after, "RecalcStyleDuration") * 1000, 1),
            "task_ms": round(delta(before, after, "TaskDuration") * 1000, 1),
            "activities": page.evaluate("window.__research.state().activities"),
            "activities_before": activities_before_stream,
        }

        page.evaluate("window.__rafCount = 0")
        page.wait_for_timeout(2000)  # measurement window: rAF callbacks counted over 2 s idle
        results["idle_raf_callbacks_per_2s"] = page.evaluate("window.__rafCount")

        report = build_report(int(os.environ.get("SMOKE_REPORT_SECTIONS", "40")))
        before = metrics(cdp)
        page.evaluate("window.__longTasks.length = 0")
        clicks_before_report = page.evaluate("window.__research.clicks()")
        # A synthetic element.click() skips hit testing and would land even with pointer-events stranded.
        page.evaluate(
            """md => {
                window.__reportStallMs = 0;
                let previous = performance.now();
                const probe = () => {
                    const now = performance.now();
                    window.__reportStallMs = Math.max(
                        window.__reportStallMs,
                        now - previous,
                    );
                    previous = now;
                    window.__reportStallProbe = setTimeout(probe, 16);
                };
                window.__reportStallProbe = setTimeout(probe, 16);
                window.__research.publishReport(md);
            }""",
            report,
        )
        try:
            page.click('[data-smoke="click-probe"]', timeout = 10_000)
            report_click_landed = True
        except Exception as exc:
            report_click_landed = False
            info(f"the click during the report parse never became actionable: {exc!r}")
        page.wait_for_timeout(3000)  # measurement window: the stall probe samples the whole parse
        # Disarm only after the window closes, so a stall at its very end is still sampled.
        page.evaluate(
            """() => {
                clearTimeout(window.__reportStallProbe);
                window.__reportStallMs = Math.max(window.__reportStallMs, 0);
            }"""
        )
        after = metrics(cdp)
        long_tasks = page.evaluate("window.__longTasks")
        results["report"] = {
            "chars": len(report),
            "long_tasks": len(long_tasks),
            "worst_long_task_ms": round(max((t["duration"] for t in long_tasks), default = 0.0), 1),
            "task_ms": round(delta(before, after, "TaskDuration") * 1000, 1),
            "main_thread_stall_ms": round(page.evaluate("window.__reportStallMs"), 1),
            "clicks_registered": page.evaluate("window.__research.clicks()") - clicks_before_report,
            "click_landed": report_click_landed,
            "rendered": page.evaluate(
                "() => Boolean(document.querySelector('[data-smoke=\\\"report\\\"] h1'))"
            ),
        }

        # Approval unmounts PlanReview's Dialog while open, and closing the pane unmounts the panel;
        # either can strand `body { pointer-events: none }`.
        page.evaluate("window.__research.clearReport()")
        page.evaluate("window.__research.awaitApproval()")
        until(page, f"() => ({DIALOG_OPEN})() && ({BODY_LOCKED})()")
        dialog_open = page.evaluate(
            "() => Boolean(document.querySelector('[role=\\\"dialog\\\"]'))"
        )
        body_during = page.evaluate("() => document.body.style.pointerEvents")
        page.evaluate("window.__research.approve()")
        until(page, f"() => !({DIALOG_OPEN})() && !({BODY_LOCKED})()")
        frames(page)
        body_after_approve = page.evaluate("() => document.body.style.pointerEvents")
        clicks_before = page.evaluate("window.__research.clicks()")
        page.click('[data-smoke="click-probe"]', timeout = 5000)
        clicks_after_approve = page.evaluate("window.__research.clicks()")

        page.evaluate("window.__research.awaitApproval()")
        until(page, f"() => ({DIALOG_OPEN})() && ({BODY_LOCKED})()")
        page.evaluate("window.__research.closePanel()")
        until(page, f"() => !document.querySelector('{SCROLLER}') && !({BODY_LOCKED})()")
        frames(page)
        body_after_close = page.evaluate("() => document.body.style.pointerEvents")
        page.click('[data-smoke="click-probe"]', timeout = 5000)
        clicks_after_close = page.evaluate("window.__research.clicks()")

        results["modal"] = {
            "dialog_opened": dialog_open,
            "body_pointer_events_while_open": body_during,
            "body_pointer_events_after_approve": body_after_approve,
            "body_pointer_events_after_close": body_after_close,
            "click_after_approve": clicks_after_approve > clicks_before,
            "click_after_close": clicks_after_close > clicks_after_approve,
        }

        # 5. Detaching: a still-pending follow step used to reconcile isAtBottom back to true after a
        # small upward flick.
        page.evaluate("window.__research.openPanel()")
        until(page, f"() => Boolean(document.querySelector('{SCROLLER}'))")
        frames(page)
        page.evaluate("() => { for (let i = 100; i < 130; i += 1) window.__research.step(i); }")
        # The state the checks below read: the list overflows and the follow loop has pinned it.
        until(
            page,
            f"() => {{ const el = document.querySelector('{SCROLLER}');"
            " return el.scrollHeight > el.clientHeight"
            " && el.scrollHeight - el.scrollTop - el.clientHeight <= 2; }",
        )
        frames(page)
        overflowing = page.evaluate(
            f"() => {{ const el = document.querySelector('{SCROLLER}'); return el.scrollHeight > el.clientHeight; }}"
        )
        followed_distance = page.evaluate(DISTANCE_FROM_BOTTOM)
        latest_while_following = page.evaluate(LATEST_BUTTON_VISIBLE)
        page.evaluate(
            f"""async () => {{
                const el = document.querySelector('{SCROLLER}');
                el.firstElementChild.setAttribute("aria-hidden", "false");
                await new Promise((resolve) => setTimeout(resolve, 0));
                el.dispatchEvent(new WheelEvent("wheel", {{ deltaY: -40, bubbles: true }}));
                el.scrollTop = el.scrollHeight - el.clientHeight - 8;
                el.dispatchEvent(new Event("scroll"));
            }}"""
        )
        until(page, LATEST_BUTTON_VISIBLE, timeout = 2000)
        frames(page)
        results["detach"] = {
            "overflowing": overflowing,
            "distance_while_following": followed_distance,
            "latest_while_following": latest_while_following,
            "distance_after_flick": page.evaluate(DISTANCE_FROM_BOTTOM),
            "latest_after_flick": page.evaluate(LATEST_BUTTON_VISIBLE),
        }

        page.screenshot(path = str(OUT / f"{LABEL}.png"), full_page = False)
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
            f"{BASE}/smoke-research.html", "smoke-research-main.tsx", proc = vite, info = info
        )
        results = run()
    finally:
        if vite is not None:
            stop_process(vite)
            info("vite stopped")
    out = OUT / f"{LABEL}.json"
    out.write_text(json.dumps(results, indent = 2), encoding = "utf-8")
    info(json.dumps(results, indent = 2))
    info(f"wrote {out}")

    failures: list[str] = []
    stream = results["stream"]
    if stream["activities"] <= stream["activities_before"]:
        failures.append(
            "the stream added no activities; the frame budgets below measured no workload"
        )
    if stream["raf_per_second"] > MAX_STREAM_RAF_PER_SECOND:
        failures.append(
            f"{stream['raf_per_second']} rAF/s during the stream, budget "
            f"{MAX_STREAM_RAF_PER_SECOND} (a per-frame loop is running)"
        )
    if results["idle_raf_callbacks_per_2s"] > MAX_IDLE_RAF_PER_2S:
        failures.append(
            f"{results['idle_raf_callbacks_per_2s']} rAF in 2s with the list idle, budget "
            f"{MAX_IDLE_RAF_PER_2S} (the follow loop never let go)"
        )
    modal = results["modal"]
    if not modal["dialog_opened"]:
        failures.append("plan review dialog never opened; the modal checks proved nothing")
    # A dialog that never took the modal layer strands nothing, so every later check would pass.
    if modal["body_pointer_events_while_open"] != "none":
        failures.append(
            "the plan review dialog never took the modal layer "
            f"(body pointer-events was {modal['body_pointer_events_while_open']!r}, "
            "expected 'none'); the stranding checks proved nothing"
        )
    if modal["body_pointer_events_after_approve"] == "none":
        failures.append("body pointer-events stranded at none after approve")
    if modal["body_pointer_events_after_close"] == "none":
        failures.append("body pointer-events stranded at none after closing the pane")
    if not modal["click_after_approve"] or not modal["click_after_close"]:
        failures.append("a click did not reach its handler after a modal path")
    if not results["report"]["rendered"]:
        failures.append("the report never rendered")
    # Hit-tested, so stranded pointer-events fails here; a synthetic click would pass on that tree.
    if not results["report"]["click_landed"]:
        failures.append(
            "a real click during the report parse never became actionable; the window was "
            "not taking input"
        )
    if results["report"]["clicks_registered"] < 1:
        failures.append("the click during the report parse never reached its handler")
    if results["report"]["main_thread_stall_ms"] > MAIN_THREAD_STALL_BUDGET_MS:
        failures.append(
            f"report rendering stalled the main thread for "
            f"{results['report']['main_thread_stall_ms']}ms, over the "
            f"{MAIN_THREAD_STALL_BUDGET_MS}ms budget"
        )
    if results["report"]["main_thread_stall_ms"] <= 0:
        failures.append("the stall probe recorded no samples; the budget above measured nothing")
    detach = results["detach"]
    if not detach["overflowing"]:
        failures.append("the activity list never overflowed; the detach checks proved nothing")
    if detach["distance_while_following"] > 2:
        failures.append(
            f"the view was {detach['distance_while_following']}px off the bottom while following"
        )
    if detach["latest_while_following"]:
        failures.append("'Latest' was offered while the view was still following")
    if not detach["latest_after_flick"]:
        failures.append("'Latest' did not appear after a flick shorter than the bottom threshold")
    for problem in failures:
        info(f"FAIL {problem}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
