# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""How the chat thread's interaction cost grows with the number of messages (#8977).

Unsloth's chat UI is reported as sluggish on Windows 11 and worsening as the thread fills:
opening menus, scrolling, deleting and typing all lag while token generation is unaffected.
That shape says the cost is per-message renderer work, so the thing to measure is not a single
absolute number but a curve: the same four interactions repeated at N in {10, 50, 200, 500}.

Four scripted actions per N, under 6x CDP CPU throttling, against the real Thread mounted by
studio/frontend/smoke-thread-weight.html:

    keystroke  - one character into the composer, measured to the frame that paints it.
    scroll     - one scroll gesture up through the thread; long-task ms is the lag a user feels.
    menu       - one message action menu opened and closed, the Radix modal-layer fan-out.
    delete     - one message deleted, the export / rebuild / import round trip.

Each action is bracketed by CDP `Performance.getMetrics`, so LayoutCount, RecalcStyleCount,
LayoutDuration, RecalcStyleDuration and TaskDuration separate the two families of cost: work
that grows because layout is uncontained shows up in LayoutDuration and LayoutCount, while work
that grows because a listener or an export is O(messages) shows up in TaskDuration alone.

THIS HARNESS MEASURES, IT DOES NOT GATE. It prints the per-N table and exits 0 unless the
harness itself broke -- the page failed to seed, an element it drives went missing, or every N
produced the same number, which would mean it is measuring nothing. There are deliberately no
performance budgets here. Budgets belong in a later change, set from real numbers taken on real
hardware; a budget invented from one Linux CI run would either never fire or fire on noise.

Chromium only for the numbers. `Emulation.setCPUThrottlingRate`, `Performance.getMetrics` and
the `longtask` PerformanceObserver entry type are all Chromium features, so running this file
under Firefox or WebKit would exercise the page as a correctness check and report no meaningful
performance at all. The desktop app embeds WebKitGTK, not Chromium, so what transfers from these
numbers is the shape of the curve, not the absolute milliseconds.

Unlike playwright_chat_autoscroll.py this does NOT replace requestAnimationFrame with a fixed
timer. That harness counts frames, where a deterministic pump is the point; this one measures
time to paint, which a fake rAF would silently destroy. rAF is wrapped to count real callbacks
and otherwise left alone; the harness's own waits use the unwrapped rAF, so __rafCount stays a
count of the page's frames rather than of this file's.

Read the timings against `paint_floor_ms`, which is printed per N. Anything clocked across a
double rAF cannot resolve faster than two vsync intervals, measured at ~33ms here and unmoved by
CPU throttling. An action that never happened therefore still reports ~33ms, which reads as a
plausible measurement rather than as a failure, so the floor is subtracted before any growth
ratio and the guards below reject a keystroke at or under it.

Run:
    python tests/studio/playwright_thread_weight.py
    SMOKE_THREAD_SIZES=10,50 python tests/studio/playwright_thread_weight.py

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
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

PORT = int(os.environ.get("SMOKE_PORT", "5213"))
# Unset: start and stop our own server.
# rstrip("/"): a trailing slash would make the anchored /api/ route regex below never match, silently turning the
# stubbed fork-count fan-out back into live HTTP.
_EXTERNAL = os.environ.get("SMOKE_BASE_URL", "").strip().rstrip("/")
BASE = _EXTERNAL or f"http://127.0.0.1:{PORT}"
OWNS_SERVER = not _EXTERNAL
LABEL = os.environ.get("SMOKE_LABEL", "tree")
OUT = Path(os.environ.get("PW_ART_DIR", "logs/playwright-thread-weight"))
OUT.mkdir(parents = True, exist_ok = True)

# Sorted: the growth check reads the first and last entries as smallest and largest, so an unsorted override would
SIZES = sorted(int(n) for n in os.environ.get("SMOKE_THREAD_SIZES", "10,50,200,500").split(","))
# 6x is the Lighthouse mobile default and roughly the gap between this machine and the reported one under load.
CPU_THROTTLE_RATE = float(os.environ.get("SMOKE_CPU_THROTTLE", "6"))
# Keystrokes are noisy at this timescale, so type several and report the median.
KEYSTROKES = int(os.environ.get("SMOKE_KEYSTROKES", "5"))
SCROLL_STEPS = int(os.environ.get("SMOKE_SCROLL_STEPS", "20"))
SCROLL_STEP_PX = int(os.environ.get("SMOKE_SCROLL_STEP_PX", "400"))
# not a regression.
# 500 uncontained messages under 6x throttling are slow by construction;
SEED_TIMEOUT_MS = int(os.environ.get("SMOKE_SEED_TIMEOUT_MS", "180000"))
ACTION_TIMEOUT_MS = int(os.environ.get("SMOKE_ACTION_TIMEOUT_MS", "60000"))
# How long an in-page action waits for the DOM to reach the state it asked for.
# Measured on this tree, opening the action menu at N=500 under 6x takes around 25s.
SETTLE_TIMEOUT_MS = int(os.environ.get("SMOKE_SETTLE_TIMEOUT_MS", "90000"))

OBSERVER_INIT = """
(() => {
  window.__longTasks = [];
  try {
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        window.__longTasks.push({ start: entry.startTime, duration: entry.duration });
      }
    }).observe({ type: "longtask", buffered: true });
  } catch (e) { /* longtask is Chromium-only: the CDP metrics still apply */ }
  // Counting wrapper, not a pump. Replacing rAF with a timer would flatten every
  // time-to-paint number this harness exists to read.
  window.__rafCount = 0;
  const nativeRaf = window.requestAnimationFrame.bind(window);
  window.requestAnimationFrame = (cb) =>
    nativeRaf((t) => {
      window.__rafCount += 1;
      cb(t);
    });
  window.__nextPaint = () =>
    new Promise((resolve) => nativeRaf(() => nativeRaf(() => resolve())));
})();
"""


def info(message: str) -> None:
    print(f"[thread-weight] {message}", flush = True)


def metrics(cdp) -> dict[str, float]:
    got = cdp.send("Performance.getMetrics")
    return {m["name"]: m["value"] for m in got["metrics"]}


def delta(before: dict[str, float], after: dict[str, float], name: str) -> float:
    return round(after.get(name, 0.0) - before.get(name, 0.0), 4)


def counters(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {
        "layout_count": delta(before, after, "LayoutCount"),
        "recalc_style_count": delta(before, after, "RecalcStyleCount"),
        "layout_ms": round(delta(before, after, "LayoutDuration") * 1000, 1),
        "recalc_style_ms": round(delta(before, after, "RecalcStyleDuration") * 1000, 1),
        "task_ms": round(delta(before, after, "TaskDuration") * 1000, 1),
    }


def long_task_summary(page) -> dict[str, float]:
    # PerformanceObserver callbacks are delivered on a later task, so the entry for the long task at the tail of an
    tasks = page.evaluate(
        "async () => { await new Promise((r) => setTimeout(r, 0)); return window.__longTasks; }"
    )
    return {
        "long_tasks": len(tasks),
        "long_task_ms": round(sum(t["duration"] for t in tasks), 1),
        "worst_long_task_ms": round(max((t["duration"] for t in tasks), default = 0.0), 1),
    }


# One character through the native value setter plus an input event:
KEYSTROKE_JS = """
async (count) => {
  const api = window.__threadWeight;
  const input = api.composer();
  if (!input) return null;
  input.focus();
  const setValue = Object.getOwnPropertyDescriptor(
    HTMLTextAreaElement.prototype, "value",
  ).set;
  const samples = [];
  for (let i = 0; i < count; i += 1) {
    await window.__nextPaint();
    const started = performance.now();
    setValue.call(input, input.value + "a");
    input.dispatchEvent(new Event("input", { bubbles: true }));
    await window.__nextPaint();
    samples.push(performance.now() - started);
  }
  // domText is what the harness itself wrote; runtimeText is what the runtime received. Only
  // the second can tell you the keystroke reached React rather than just the DOM node.
  return {
    samples,
    domText: input.value,
    runtimeText: api.composerText(),
  };
}
"""

SCROLL_JS = """
async ([steps, stepPx]) => {
  const api = window.__threadWeight;
  const viewport = api.viewport();
  if (!viewport) return null;
  // The viewport carries `scroll-smooth`, so each scrollTop write starts an animation and the
  // next read lands mid-flight. Stepping from a tracked target with an explicit instant
  // behaviour is what a wheel gesture actually does, and it is the only way the gesture moves
  // the distance it asks for.
  const bottom = viewport.scrollHeight - viewport.clientHeight;
  viewport.scrollTo({ top: bottom, behavior: "instant" });
  await window.__nextPaint();
  let target = viewport.scrollTop;
  // Reverse at either end rather than stopping. A short thread runs out of travel long before
  // a long one does, and a gesture that covers 2600px at N=10 and 8000px at N=500 is not the
  // same gesture, so the two columns would not be comparable.
  let direction = -1;
  let travelled = 0;
  let worstFrameMs = 0;
  const started = performance.now();
  for (let i = 0; i < steps; i += 1) {
    if (direction < 0 && target <= 0) direction = 1;
    else if (direction > 0 && target >= bottom) direction = -1;
    const next = Math.min(bottom, Math.max(0, target + direction * stepPx));
    const frameStarted = performance.now();
    // The wheel event is what the app's own scroll listeners key off; the scrollTo is what
    // moves the viewport in a headless run with no compositor input.
    viewport.dispatchEvent(
      new WheelEvent("wheel", {
        deltaY: direction * stepPx, bubbles: true, cancelable: true,
      }),
    );
    viewport.scrollTo({ top: next, behavior: "instant" });
    await window.__nextPaint();
    worstFrameMs = Math.max(worstFrameMs, performance.now() - frameStarted);
    travelled += Math.abs(next - target);
    target = next;
  }
  return {
    wallMs: performance.now() - started,
    scrolledPx: travelled,
    worstFrameMs,
    frames: steps,
  };
}
"""

# Radix portals the menu to document.body and puts the body on the modal layer, which is the fan-out the issue blames.
# The trigger opens on `pointerdown`, not on `click`: an element.click() leaves the menu shut and the whole measurement
# silently reads zero.
MENU_JS = """
async (timeoutMs) => {
  const api = window.__threadWeight;
  const trigger = api.actionButton("More");
  if (!trigger) return null;
  // A MutationObserver flag, not a querySelector per frame. The menu content is portaled to the
  // end of document.body, so polling for it walks the whole message list and finds nothing for
  // the entire open latency -- an O(messages) cost charged to the action, growing like the
  // signal it is measuring.
  let open = Boolean(document.querySelector(".aui-action-bar-more-content"));
  const watcher = new MutationObserver(() => {
    open = Boolean(document.querySelector(".aui-action-bar-more-content"));
  });
  watcher.observe(document.body, { childList: true, subtree: false });
  const settle = async (want) => {
    const started = performance.now();
    while (performance.now() - started < timeoutMs) {
      if (open === want) return performance.now() - started;
      await window.__nextPaint();
    }
    return null;
  };
  const pointer = {
    bubbles: true, cancelable: true, composed: true,
    button: 0, pointerId: 1, pointerType: "mouse", isPrimary: true,
  };
  const openStarted = performance.now();
  trigger.dispatchEvent(new PointerEvent("pointerdown", { ...pointer, buttons: 1 }));
  trigger.dispatchEvent(new PointerEvent("pointerup", { ...pointer, buttons: 0 }));
  const opened = await settle(true);
  const openMs = opened === null ? null : performance.now() - openStarted;
  const bodyPointerEvents = getComputedStyle(document.body).pointerEvents;
  const itemsWhileOpen = api.openMenuItemCount();
  // Counted here, under the pointer, not in the resting-state census. An autohidden bar is
  // absent at rest by design; one that never mounts at all is a broken page, and only a
  // hovered count tells the two apart.
  const triggersWhileHovered = document.querySelectorAll('[data-slot="tooltip-trigger"]').length;
  // The clock starts BEFORE the dispatch. Radix dismisses synchronously inside it -- layer
  // teardown, focus restore, the body coming off the modal layer and the re-render that
  // follows -- which is the O(messages) fan-out being measured. Starting it after the dispatch
  // excluded exactly the part worth timing.
  const closeStarted = performance.now();
  document.dispatchEvent(
    new KeyboardEvent("keydown", { key: "Escape", bubbles: true, cancelable: true }),
  );
  const closed = await settle(false);
  const result = {
    openMs,
    closeMs: closed === null ? null : performance.now() - closeStarted,
    bodyPointerEvents,
    bodyPointerEventsAfterClose: getComputedStyle(document.body).pointerEvents,
    itemsWhileOpen,
    triggersWhileHovered,
  };
  watcher.disconnect();
  return result;
}
"""

DELETE_JS = """
async (timeoutMs) => {
  const api = window.__threadWeight;
  const button = api.actionButton("Delete message");
  if (!button) return null;
  // The last assistant message. That is the cheapest delete on the React side -- one subtree
  // unmounts -- so this column under-measures reconciliation, though the export/rebuild/import
  // half is O(messages) wherever the target sits.
  const target = api.lastAssistantMessage();
  const before = api.messageCount();
  const started = performance.now();
  button.click();
  // isConnected on the captured node is O(1). Re-counting [data-role] every frame would put an
  // O(messages) query inside the window being timed, growing like the signal.
  while (performance.now() - started < timeoutMs) {
    if (target === null || !target.isConnected) {
      return { ms: performance.now() - started, before, after: api.messageCount() };
    }
    await window.__nextPaint();
  }
  return { ms: null, before, after: api.messageCount() };
}
"""

# The floor under every timing here:
PAINT_FLOOR_JS = """
async (samples) => {
  const values = [];
  for (let i = 0; i < samples; i += 1) {
    await window.__nextPaint();
    const started = performance.now();
    await window.__nextPaint();
    values.push(performance.now() - started);
  }
  values.sort((a, b) => a - b);
  return values[Math.floor(values.length / 2)];
}
"""


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if not ordered:
        return -1.0
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def reset_long_tasks(page) -> None:
    page.evaluate("window.__longTasks.length = 0")


def measure_one(context, cdp_throttle_rate: float, size: int) -> dict:
    """Seed a fresh page to `size` messages and run the four actions on it."""
    page = context.new_page()
    result: dict = {"messages_requested": size}
    # A request that escapes to the server, or a warning storm, is work this harness would be charging to the app, once
    # per message.
    # startswith, not `"/api/" in url`: vite serves the app's own source modules from paths like
    # /src/features/chat/api/chat-api.ts, and a substring match counts 45 of those as network calls.
    api_prefix = f"{BASE}/api/"
    stray_requests: list[str] = []
    console_warnings: list[str] = []
    page.on(
        "request",
        lambda r: stray_requests.append(r.url) if r.url.startswith(api_prefix) else None,
    )
    page.on(
        "console",
        lambda m: console_warnings.append(m.text[:200]) if m.type in ("warning", "error") else None,
    )
    try:
        page.goto(f"{BASE}/smoke-thread-weight.html", wait_until = "domcontentloaded")
        page.wait_for_function("() => Boolean(window.__threadWeight)", timeout = 30_000)
        cdp = context.new_cdp_session(page)
        cdp.send("Performance.enable")

        # Seeding unthrottled: this measures interaction cost at a thread size, not the cost of constructing the
        page.evaluate("(n) => window.__threadWeight.seed(n)", size)
        page.wait_for_function(
            "(n) => window.__threadWeight.messageCount() >= n",
            arg = size,
            timeout = SEED_TIMEOUT_MS,
        )
        page.wait_for_function(
            "(n) => window.__threadWeight.katexCount() >= n",
            arg = size // 2,
            timeout = SEED_TIMEOUT_MS,
        )
        # Single-selector gates. counts() walks every element in the document, so polling it per frame makes seeding
        page.wait_for_function(
            """() => {
                const n = window.__threadWeight.highlightedTokenCount();
                const settled = window.__twTokens === n;
                window.__twTokens = n;
                return settled && n > 0;
            }""",
            timeout = SEED_TIMEOUT_MS,
        )
        result["counts"] = page.evaluate("window.__threadWeight.counts()")
        result["viewport"] = page.evaluate("window.__threadWeight.viewportMetrics()")
        # Kept for the record, then cleared:
        result["seed_api_requests"] = len(stray_requests)
        result["seed_console_warnings"] = len(console_warnings)
        stray_requests.clear()
        console_warnings.clear()

        cdp.send("Emulation.setCPUThrottlingRate", {"rate": cdp_throttle_rate})
        result["cpu_throttle_rate"] = cdp_throttle_rate
        # Under throttling, because that is the regime every timing below is taken in.
        result["paint_floor_ms"] = round(page.evaluate(PAINT_FLOOR_JS, 9), 2)

        reset_long_tasks(page)
        before = metrics(cdp)
        typed = page.evaluate(KEYSTROKE_JS, KEYSTROKES)
        after = metrics(cdp)
        result["keystroke"] = {
            "samples_ms": None if typed is None else [round(s, 2) for s in typed["samples"]],
            "median_ms": None if typed is None else round(median(typed["samples"]), 2),
            "worst_ms": None if typed is None else round(max(typed["samples"]), 2),
            "dom_text": None if typed is None else typed["domText"],
            "runtime_text": None if typed is None else typed["runtimeText"],
            **counters(before, after),
            **long_task_summary(page),
        }

        reset_long_tasks(page)
        before = metrics(cdp)
        scrolled = page.evaluate(SCROLL_JS, [SCROLL_STEPS, SCROLL_STEP_PX])
        after = metrics(cdp)
        result["scroll"] = {
            "wall_ms": None if scrolled is None else round(scrolled["wallMs"], 1),
            "scrolled_px": None if scrolled is None else scrolled["scrolledPx"],
            # Long tasks need a 50ms frame;
            # a scroll can be visibly rough well under that, so the worst single frame is the jank number and
            # long_task_ms is the severe-case one.
            "worst_frame_ms": None if scrolled is None else round(scrolled["worstFrameMs"], 1),
            "frames": None if scrolled is None else scrolled["frames"],
            **counters(before, after),
            **long_task_summary(page),
        }

        # 3.
        # the same trap SCROLL_JS documents.
        page.evaluate(
            """() => { const m = window.__threadWeight.lastAssistantMessage();
                if (m) m.scrollIntoView({ block: "center", behavior: "instant" }); }"""
        )
        # Shiki is async and per block, and a <pre> exists before it is highlighted, so counting code blocks gates
        page.wait_for_function(
            """() => {
                const top = window.__threadWeight.viewportMetrics().scrollTop;
                const settled = window.__twTop === top;
                window.__twTop = top;
                return settled;
            }""",
            timeout = ACTION_TIMEOUT_MS,
        )
        page.locator('[data-role="assistant"]').last.hover(timeout = ACTION_TIMEOUT_MS)
        reset_long_tasks(page)
        before = metrics(cdp)
        menu = page.evaluate(MENU_JS, SETTLE_TIMEOUT_MS)
        after = metrics(cdp)
        result["menu"] = {
            "open_ms": None if menu is None else _round_or_none(menu["openMs"]),
            "close_ms": None if menu is None else _round_or_none(menu["closeMs"]),
            "open_close_ms": None if menu is None else _sum_or_none(menu),
            "body_pointer_events_while_open": None if menu is None else menu["bodyPointerEvents"],
            "body_pointer_events_after_close": (
                None if menu is None else menu["bodyPointerEventsAfterClose"]
            ),
            "items_while_open": None if menu is None else menu["itemsWhileOpen"],
            "triggers_while_hovered": None if menu is None else menu["triggersWhileHovered"],
            **counters(before, after),
            **long_task_summary(page),
        }

        page.locator('[data-role="assistant"]').last.hover(timeout = ACTION_TIMEOUT_MS)
        reset_long_tasks(page)
        before = metrics(cdp)
        deleted = page.evaluate(DELETE_JS, SETTLE_TIMEOUT_MS)
        after = metrics(cdp)
        result["delete"] = {
            "ms": None if deleted is None else _round_or_none(deleted["ms"]),
            "messages_before": None if deleted is None else deleted["before"],
            "messages_after": None if deleted is None else deleted["after"],
            **counters(before, after),
            **long_task_summary(page),
        }

        cdp.send("Emulation.setCPUThrottlingRate", {"rate": 1})
        # Cumulative over seeding and all four actions:
        result["raf_callbacks"] = page.evaluate("window.__rafCount")
        result["stray_api_requests"] = len(stray_requests)
        result["console_warnings"] = len(console_warnings)
        result["first_console_warning"] = console_warnings[0] if console_warnings else "-"
    finally:
        page.close()
    return result


def _round_or_none(value) -> float | None:
    return None if value is None else round(value, 1)


def _sum_or_none(menu: dict) -> float | None:
    if menu["openMs"] is None or menu["closeMs"] is None:
        return None
    return round(menu["openMs"] + menu["closeMs"], 1)


def run() -> dict:
    results: dict = {
        "label": LABEL,
        "base": BASE,
        "cpu_throttle_rate": CPU_THROTTLE_RATE,
        "sizes": SIZES,
        "by_size": {},
    }
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless = os.environ.get("SMOKE_HEADLESS", "1") == "1",
            args = chromium_launch_args(),
        )
        context = browser.new_context(viewport = {"width": 1440, "height": 900})
        context.add_init_script(OBSERVER_INIT)
        # Anchored at the origin so it cannot swallow vite's own module URLs, which live under
        context.route(
            re.compile(rf"^{re.escape(BASE)}/api/"),
            lambda route: route.fulfill(status = 200, content_type = "application/json", body = "{}"),
        )
        for size in SIZES:
            info(f"measuring N={size}")
            results["by_size"][str(size)] = measure_one(context, CPU_THROTTLE_RATE, size)
        context.close()
        browser.close()
    return results


# Every recorded metric appears here.
# That is the rule the harnesses in this directory are held to: a metric that is recorded and never read is how one goes
# false-green, and tests/studio/test_autoscroll_harness_contract.py fails if anything recorded below is missing.
TABLE_ROWS = (
    ("messages requested", lambda r: r["messages_requested"]),
    ("cpu throttle rate", lambda r: r["cpu_throttle_rate"]),
    ("paint floor ms", lambda r: r["paint_floor_ms"]),
    ("seed api requests", lambda r: r["seed_api_requests"]),
    ("seed console warnings", lambda r: r["seed_console_warnings"]),
    ("action api requests", lambda r: r["stray_api_requests"]),
    ("action console warnings", lambda r: r["console_warnings"]),
    ("first console warning", lambda r: r["first_console_warning"]),
    ("messages rendered", lambda r: r["counts"]["messages"]),
    ("assistant messages", lambda r: r["counts"]["assistantMessages"]),
    ("user messages", lambda r: r["counts"]["userMessages"]),
    ("dom nodes", lambda r: r["counts"]["domNodes"]),
    ("code blocks", lambda r: r["counts"]["codeBlocks"]),
    ("katex nodes", lambda r: r["counts"]["katexNodes"]),
    ("action bars", lambda r: r["counts"]["actionBars"]),
    ("tooltip triggers", lambda r: r["counts"]["tooltipTriggers"]),
    ("tooltip triggers hovered", lambda r: r["menu"]["triggers_while_hovered"]),
    ("viewport scrollHeight", lambda r: r["viewport"]["scrollHeight"]),
    ("viewport scrollTop", lambda r: r["viewport"]["scrollTop"]),
    ("viewport clientHeight", lambda r: r["viewport"]["clientHeight"]),
    ("keystroke median ms", lambda r: r["keystroke"]["median_ms"]),
    ("keystroke worst ms", lambda r: r["keystroke"]["worst_ms"]),
    # Compact so the column still lines up.
    (
        "keystroke samples ms",
        lambda r: "/".join(str(round(s)) for s in r["keystroke"]["samples_ms"]),
    ),
    ("keystroke dom text", lambda r: r["keystroke"]["dom_text"]),
    ("keystroke runtime text", lambda r: r["keystroke"]["runtime_text"]),
    ("keystroke layouts", lambda r: r["keystroke"]["layout_count"]),
    ("keystroke layout ms", lambda r: r["keystroke"]["layout_ms"]),
    ("keystroke recalcs", lambda r: r["keystroke"]["recalc_style_count"]),
    ("keystroke recalc ms", lambda r: r["keystroke"]["recalc_style_ms"]),
    ("keystroke task ms", lambda r: r["keystroke"]["task_ms"]),
    ("keystroke longtasks", lambda r: r["keystroke"]["long_tasks"]),
    ("keystroke longtask ms", lambda r: r["keystroke"]["long_task_ms"]),
    ("keystroke worst longtask ms", lambda r: r["keystroke"]["worst_long_task_ms"]),
    ("scroll wall ms", lambda r: r["scroll"]["wall_ms"]),
    ("scroll worst frame ms", lambda r: r["scroll"]["worst_frame_ms"]),
    ("scroll px", lambda r: r["scroll"]["scrolled_px"]),
    ("scroll frames", lambda r: r["scroll"]["frames"]),
    ("scroll layouts", lambda r: r["scroll"]["layout_count"]),
    ("scroll layout ms", lambda r: r["scroll"]["layout_ms"]),
    ("scroll recalcs", lambda r: r["scroll"]["recalc_style_count"]),
    ("scroll recalc ms", lambda r: r["scroll"]["recalc_style_ms"]),
    ("scroll task ms", lambda r: r["scroll"]["task_ms"]),
    ("scroll longtasks", lambda r: r["scroll"]["long_tasks"]),
    ("scroll longtask ms", lambda r: r["scroll"]["long_task_ms"]),
    ("scroll worst longtask ms", lambda r: r["scroll"]["worst_long_task_ms"]),
    ("menu open ms", lambda r: r["menu"]["open_ms"]),
    ("menu close ms", lambda r: r["menu"]["close_ms"]),
    ("menu open+close ms", lambda r: r["menu"]["open_close_ms"]),
    ("menu body pe while open", lambda r: r["menu"]["body_pointer_events_while_open"]),
    ("menu body pe after close", lambda r: r["menu"]["body_pointer_events_after_close"]),
    ("menu items while open", lambda r: r["menu"]["items_while_open"]),
    ("menu layouts", lambda r: r["menu"]["layout_count"]),
    ("menu layout ms", lambda r: r["menu"]["layout_ms"]),
    ("menu recalcs", lambda r: r["menu"]["recalc_style_count"]),
    ("menu recalc ms", lambda r: r["menu"]["recalc_style_ms"]),
    ("menu task ms", lambda r: r["menu"]["task_ms"]),
    ("menu longtasks", lambda r: r["menu"]["long_tasks"]),
    ("menu longtask ms", lambda r: r["menu"]["long_task_ms"]),
    ("menu worst longtask ms", lambda r: r["menu"]["worst_long_task_ms"]),
    ("delete ms", lambda r: r["delete"]["ms"]),
    ("delete messages before", lambda r: r["delete"]["messages_before"]),
    ("delete messages after", lambda r: r["delete"]["messages_after"]),
    ("delete layouts", lambda r: r["delete"]["layout_count"]),
    ("delete layout ms", lambda r: r["delete"]["layout_ms"]),
    ("delete recalcs", lambda r: r["delete"]["recalc_style_count"]),
    ("delete recalc ms", lambda r: r["delete"]["recalc_style_ms"]),
    ("delete task ms", lambda r: r["delete"]["task_ms"]),
    ("delete longtasks", lambda r: r["delete"]["long_tasks"]),
    ("delete longtask ms", lambda r: r["delete"]["long_task_ms"]),
    ("delete worst longtask ms", lambda r: r["delete"]["worst_long_task_ms"]),
    ("rAF callbacks", lambda r: r["raf_callbacks"]),
)


def print_table(results: dict) -> None:
    """Every recorded metric, printed. A metric that is recorded and never read is how these
    harnesses go false-green; see tests/studio/test_autoscroll_harness_contract.py."""
    sizes = [str(n) for n in results["sizes"]]
    rows = []
    for name, pick in TABLE_ROWS:
        cells = []
        for size in sizes:
            try:
                cells.append(str(pick(results["by_size"][size])))
            except (KeyError, TypeError):
                cells.append("-")
        rows.append((name, cells))
    label_width = max(len(name) for name, _ in rows) + 2
    # From the widest cell, not a constant:
    cell_width = max([len(cell) for _, cells in rows for cell in cells] + [8]) + 2
    header = "".ljust(label_width) + "".join(f"N={n}".rjust(cell_width) for n in sizes)
    info(header)
    info("-" * len(header))
    for name, cells in rows:
        info(name.ljust(label_width) + "".join(cell.rjust(cell_width) for cell in cells))


def growth(results: dict, pick, floored: bool) -> tuple[float | None, float | None]:
    """The metric at the smallest and largest N, with the paint floor removed when it applies."""
    sizes = [str(n) for n in results["sizes"]]
    try:
        rows = (results["by_size"][sizes[0]], results["by_size"][sizes[-1]])
        values = []
        for row in rows:
            value = pick(row)
            if floored:
                value -= row["paint_floor_ms"]
            values.append(round(value, 2))
        return values[0], values[1]
    except (KeyError, TypeError):
        return None, None


# `floored` marks a metric whose clock is a double rAF, so it carries the ~33ms vsync floor measured as paint_floor_ms.
# Growth axes. The point of the harness is that at least one of these rises with N;
GROWTH_AXES = (
    ("keystroke median ms", lambda r: r["keystroke"]["median_ms"], True),
    ("scroll worst frame ms", lambda r: r["scroll"]["worst_frame_ms"], True),
    ("scroll task ms", lambda r: r["scroll"]["task_ms"], False),
    ("scroll longtask ms", lambda r: r["scroll"]["long_task_ms"], False),
    ("scroll layout ms", lambda r: r["scroll"]["layout_ms"], False),
    ("menu open+close ms", lambda r: r["menu"]["open_close_ms"], True),
    ("menu recalc ms", lambda r: r["menu"]["recalc_style_ms"], False),
    ("delete ms", lambda r: r["delete"]["ms"], True),
    ("delete task ms", lambda r: r["delete"]["task_ms"], False),
)


def harness_failures(results: dict) -> list[str]:
    """Only the ways this harness can be measuring nothing. No performance budgets: see the
    module docstring."""
    failures: list[str] = []
    layers = set()
    for size in results["sizes"]:
        row = results["by_size"][str(size)]
        counts = row["counts"]
        # A request reaching the server is a CDP round trip to another process inside a region being timed, once per
        if row["stray_api_requests"]:
            failures.append(
                f"N={size} let {row['stray_api_requests']} /api/ requests reach the network "
                "during the measured actions; the in-page stub is not covering them and the "
                "timings include a round trip to another process for each"
            )
        if row["console_warnings"]:
            failures.append(
                f"N={size} logged {row['console_warnings']} console warnings during the "
                f"measured actions, the first being "
                f"{row['first_console_warning']!r}; each one is serialised over CDP and the "
                "count grows with N"
            )
        if counts["messages"] < size:
            failures.append(
                f"N={size} rendered only {counts['messages']} messages; the seed did not land"
            )
        # A thread of plain paragraphs would be cheap for reasons the app is not.
        if counts["codeBlocks"] <= 0 or counts["katexNodes"] <= 0:
            failures.append(
                f"N={size} rendered {counts['codeBlocks']} code blocks and "
                f"{counts['katexNodes']} KaTeX nodes; the message bodies are not realistic"
            )
        # An autohidden bar is absent at rest ON PURPOSE, so the resting census cannot be the
        # guard any more. What must still hold is that hovering produces one: a tree that
        hovered_triggers = row["menu"].get("triggers_while_hovered")
        if counts["actionBars"] <= 0 and not hovered_triggers:
            failures.append(
                f"N={size} mounted no action bar at rest and none under the pointer either; "
                "the per-message weight under investigation is absent"
            )
        viewport = row["viewport"]
        if viewport["scrollHeight"] <= viewport["clientHeight"]:
            failures.append(f"N={size} does not overflow its viewport; the scroll measures nothing")
        keystroke = row["keystroke"]
        if keystroke["median_ms"] is None:
            failures.append(f"N={size} could not find the composer input")
        # The DOM value is what the harness itself wrote, so it proves nothing on its own.
        elif keystroke["runtime_text"] != keystroke["dom_text"]:
            failures.append(
                f"N={size} typed {keystroke['dom_text']!r} into the DOM but the runtime holds "
                f"{keystroke['runtime_text']!r}; the keystroke never reached the composer state"
            )
        elif len(keystroke["dom_text"]) < KEYSTROKES:
            failures.append(
                f"N={size} recorded {len(keystroke['dom_text'])} of {KEYSTROKES} keystrokes"
            )
        elif keystroke["median_ms"] <= row["paint_floor_ms"]:
            failures.append(
                f"N={size} reported a keystroke of {keystroke['median_ms']}ms at or under the "
                f"{row['paint_floor_ms']}ms paint floor, so no work was measured"
            )
        if row["scroll"]["wall_ms"] is None:
            failures.append(f"N={size} could not find the thread viewport")
        # Equal travel at every N or the columns are not the same gesture.
        elif row["scroll"]["scrolled_px"] < SCROLL_STEPS * SCROLL_STEP_PX * 0.9:
            failures.append(
                f"N={size} travelled only {row['scroll']['scrolled_px']}px of the "
                f"{SCROLL_STEPS * SCROLL_STEP_PX}px gesture, so its scroll column is not "
                "comparable with the others"
            )
        menu = row["menu"]
        if menu["open_ms"] is None:
            failures.append(f"N={size} never opened the message action menu")
        elif menu["close_ms"] is None:
            failures.append(f"N={size} opened the action menu and it never closed")
        elif menu["body_pointer_events_after_close"] == "none":
            failures.append(f"N={size} left the body on the modal layer after closing the menu")
        # An empty popover satisfies "the menu opened" and costs nothing to render.
        elif not menu["items_while_open"]:
            failures.append(f"N={size} opened an action menu with no items in it")
        layers.add(menu["body_pointer_events_while_open"])
        deleted = row["delete"]
        if deleted["ms"] is None:
            failures.append(f"N={size} never deleted a message")
        elif deleted["messages_after"] >= deleted["messages_before"]:
            failures.append(f"N={size} clicked delete and the message count did not drop")

    # A modal menu puts the body on the modal layer and a non-modal one does not, and the two cost wildly different
    if len(layers) > 1:
        failures.append(
            f"the menu put the body on {sorted(str(x) for x in layers)} across N; the columns "
            "are not measuring the same mechanism"
        )

    # smallest does is not reporting a flat curve, it is reporting that it never drove the page.
    # Discrimination. Not a budget:
    if len(results["sizes"]) >= 2:
        rising = []
        for name, pick, floored in GROWTH_AXES:
            small, large = growth(results, pick, floored)
            if small is None or large is None or small <= 0:
                continue
            ratio = large / small
            suffix = " (paint floor removed)" if floored else ""
            info(
                f"growth {name}: N={results['sizes'][0]} {small} -> "
                f"N={results['sizes'][-1]} {large} ({ratio:.2f}x){suffix}"
            )
            if ratio > 1.5:
                rising.append(f"{name} {ratio:.2f}x")
        if rising:
            info(f"discriminating axes: {', '.join(rising)}")
        else:
            failures.append(
                "no measured axis rose with N. Either the page was never driven or every "
                "action is being measured somewhere it does not run; the numbers above cannot "
                "size any change."
            )
    return failures


def main() -> int:
    vite = None
    if OWNS_SERVER:
        info(f"starting vite dev server on port {PORT}")
        vite = start_vite(PORT)
    try:
        wait_for_smoke_page(
            f"{BASE}/smoke-thread-weight.html",
            "smoke-thread-weight-main.tsx",
            proc = vite,
            info = info,
        )
        results = run()
    finally:
        if vite is not None:
            stop_process(vite)
            info("vite stopped")

    out = OUT / f"{LABEL}.json"
    out.write_text(json.dumps(results, indent = 2), encoding = "utf-8")
    print_table(results)
    info(json.dumps(results, indent = 2))
    info(f"wrote {out}")

    failures = harness_failures(results)
    for problem in failures:
        info(f"HARNESS-BROKEN {problem}")
    if failures:
        return 1
    info("measurement only: no budgets are asserted here, so this exits 0 on any timing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
