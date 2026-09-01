# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where a HEAVY thread stalls, as a curve over how much content the thread holds.

Users report Unsloth and Desktop going sluggish "after long generations with any code cells
and/or text": typing, scrolling, opening a message menu, deleting and re-opening the thread all
lag once a session has accumulated a few very long replies. That report is about CONTENT VOLUME,
not message count, so the axis here is characters of thread content, not messages, and the
fixture is the mix the report names rather than one paragraph repeated:

    long prose, several large code fences, tool calls with collapsible output, an HTML artifact,
    a code-execution result pane, an HTML canvas artifact and inline images.

studio/frontend/smoke-heavy-thread.html mounts the REAL Thread with that content, so what is
timed is the app's own renderer.

WHAT IS MEASURED, AND WHY THESE METRICS

Every primary number here is DOM-observable and wall-clock, because the interesting engines are
not Chromium. Unsloth Desktop is Tauri: WebView2 on Windows, WKWebView on macOS, WebKitGTK on
Linux. The Long Tasks API does not exist on JavaScriptCore, so a `longtask` PerformanceObserver
silently reports NOTHING on Desktop macOS and Desktop Linux. Measured here: on WebKit 26.5 and
Firefox 153 the `observe({ type: "longtask" })` call does not even throw. It is accepted and then
never fires, which reads as "no jank" instead of "no measurement", so support is read from
`PerformanceObserver.supportedEntryTypes` instead. `Performance.getMetrics` and
`Emulation.setCPUThrottlingRate` are CDP, so they do not exist off Chromium either.

So the primary four, all portable:

    longest stall ms   the largest gap between consecutive ticks of a 1ms setTimeout loop. This
                       is the portable stand-in for a long task: the main thread cannot answer
                       the timer while it is busy, so the gap IS the block. Accurate to the
                       timer clamp (~4ms), which is far below anything a user notices.
    worst frame ms     the largest gap between consecutive requestAnimationFrame callbacks.
    frames over 33ms   how many frames in the action missed two vsyncs at 60Hz. A single worst
                       frame can be an outlier; a count says how much of the gesture was rough.
    wall ms            how long the action took end to end, and for the gestures that keep
                       working after the input stops, a separate settle time.

A MessageChannel ping-pong is the usual way to build the stall detector and it was measured and
rejected here: it spins ~150k times a second, and on Firefox that HALVED the frame rate of the
page under test (38 frames -> 14, median frame 17ms -> 34ms) before any Unsloth code ran. The
1ms setTimeout loop ticks ~150 times a second, costs nothing measurable on any of the three
engines, and reported the same 120ms synthetic stall on all three.

CDP counters (LayoutCount, RecalcStyleCount, LayoutDuration, RecalcStyleDuration, TaskDuration)
and the longtask observer are ALSO recorded, and every one of them is labelled CHROMIUM-ONLY in
the table. They attribute cost to layout versus script, which nothing portable does. They are
never the headline.

ENGINES

    chromium   proxy for WebView2, i.e. Unsloth Desktop on Windows.
    webkit     proxy for WKWebView and WebKitGTK, i.e. Unsloth Desktop on macOS and Linux.
    firefox    control. Not shipped by anything here; it is in the table so that a number which
               moves on one engine only can be told apart from a number that moves everywhere.

Playwright's WebKit is a PROXY, not the webview Desktop embeds. It is Apple's WebKit built for
Playwright, driven headless, with no Tauri IPC layer and no WebKitGTK compositor. Read it as
"JavaScriptCore plus WebKit layout", not as "Unsloth Desktop on macOS".

Desktop Linux is WORSE than any number this file can produce, and not by a little:
studio/src-tauri/src/linux_webkit.rs drops the webview off the hardware DMA-BUF transport on
Wayland and on NVIDIA under either display server. It forces the shared-memory transport with
WEBKIT_DMABUF_RENDERER_FORCE_SHM=1, or, on Wayland and on old WebKitGTK, disables the renderer
outright with WEBKIT_DISABLE_DMABUF_RENDERER=1, which turns accelerated compositing off for the
whole process. Everything below runs on a normal compositor path.

THIS HARNESS MEASURES, IT DOES NOT GATE. It prints the table and exits 0 on any timing. It exits
non-zero only when the harness itself is broken: the seed did not land, an element it drives went
missing, or the curve did not rise with content -- which would mean it is measuring nothing.
Budgets belong in a later change, set from numbers taken on real hardware.

Run:
    python tests/studio/playwright_heavy_thread.py
    SMOKE_HEAVY_CHARS=100000,300000 SMOKE_HEAVY_ENGINES=chromium python tests/studio/playwright_heavy_thread.py

It starts and stops its own vite dev server. Point it at one you already have with
SMOKE_BASE_URL, or move the port it picks with SMOKE_PORT.

RUN ONE ENGINE PER INVOCATION IN CI, UNDER AN EXTERNAL TIMEOUT. Measured on a macos-14 runner:
Chromium finished all three sizes in 90 seconds, and then Playwright's WebKit wedged at the
smallest size and never came back. `page.evaluate` and `browser.new_page` have no timeout of
their own, and a SIGALRM does not help, because Playwright's sync API blocks the main thread
inside a greenlet and the exception lands in the driver rather than in the caller. The only
bound that works is the process one, so drive the engines as separate invocations:

    bounded() {                       # not `timeout`: macOS runners do not ship coreutils, and
      local secs=$1; shift            # `timeout: command not found` fails the step instantly
      "$@" & local pid=$!
      ( sleep "$secs"; kill -TERM $pid 2>/dev/null; sleep 15; kill -KILL $pid 2>/dev/null ) &
      local watcher=$!
      wait $pid; local rc=$?
      kill -TERM $watcher 2>/dev/null
      return $rc
    }
    port=5215
    for engine in chromium webkit firefox; do
      SMOKE_HEAVY_ENGINES=$engine SMOKE_PORT=$port SMOKE_LABEL=run-$engine \\
        bounded 1800 python -u tests/studio/playwright_heavy_thread.py \\
        || echo "$engine did not finish"
      port=$((port + 1))
    done

One wedged engine then costs one engine's column instead of the whole matrix. Each invocation
needs its own port: a killed run can leave its vite dev server holding the previous one.

The dev server is deliberate and is a limitation to read the numbers against: React runs in
development mode, nothing is minified, and vite serves unbundled modules. Absolute milliseconds
are therefore higher than a packaged Unsloth's. The curve across sizes and the ranking across
actions are what this file is for.
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

PORT = int(os.environ.get("SMOKE_PORT", "5215"))
# Exported-but-empty counts as unset, else we skip the server and drive "" as the URL.
# rstrip("/"): a trailing slash makes the anchored /api/ route regex below never match, silently turning the stubbed
# fan-out back into live HTTP.
_EXTERNAL = os.environ.get("SMOKE_BASE_URL", "").strip().rstrip("/")
BASE = _EXTERNAL or f"http://127.0.0.1:{PORT}"
OWNS_SERVER = not _EXTERNAL
LABEL = os.environ.get("SMOKE_LABEL", "tree")
OUT = Path(os.environ.get("PW_ART_DIR", "logs/playwright-heavy-thread"))
OUT.mkdir(parents = True, exist_ok = True)

# Characters of thread content, not messages.
SIZES = sorted(
    int(n) for n in os.environ.get("SMOKE_HEAVY_CHARS", "25000,100000,300000").split(",")
)
ENGINES = [
    e.strip()
    for e in os.environ.get("SMOKE_HEAVY_ENGINES", "chromium,webkit,firefox").split(",")
    if e.strip()
]
# Every action set is run this many times on the seeded page and the table reports the MEDIAN.
REPEATS = int(os.environ.get("SMOKE_HEAVY_REPEATS", "3"))
# Chromium only, and off by default.
CPU_THROTTLE_RATE = float(os.environ.get("SMOKE_CPU_THROTTLE", "1"))

KEYSTROKES = int(os.environ.get("SMOKE_KEYSTROKES", "5"))
SCROLL_STEPS = int(os.environ.get("SMOKE_SCROLL_STEPS", "20"))
SCROLL_STEP_PX = int(os.environ.get("SMOKE_SCROLL_STEP_PX", "400"))
SEED_TIMEOUT_MS = int(os.environ.get("SMOKE_SEED_TIMEOUT_MS", "300000"))
ACTION_TIMEOUT_MS = int(os.environ.get("SMOKE_ACTION_TIMEOUT_MS", "120000"))
# action that never happened and nothing else, so it has to stay well above the slowest honest
# measurement or a very slow open is reported as "never opened".
# How long an in-page action waits for the DOM to reach the state it asked for.
SETTLE_TIMEOUT_MS = int(os.environ.get("SMOKE_SETTLE_TIMEOUT_MS", "120000"))
# How long the highlighter has to stay still before a re-open counts as finished.
# Four polls of the 250ms interval wait_for_highlighting_settled() uses, because that is what was MEASURED to be needed:
# a gate that released after one 250ms lull released mid-rebuild.
HIGHLIGHT_GRACE_MS = int(os.environ.get("SMOKE_HIGHLIGHT_GRACE_MS", "1000"))
# How often the settle loop is allowed to count highlighted tokens.
HIGHLIGHT_PROBE_MS = int(os.environ.get("SMOKE_HIGHLIGHT_PROBE_MS", "100"))
ACTIONS = ("keystroke", "scroll", "jump", "menu", "delete", "reopen")

# Installed into every page before anything else runs.
# which playwright_chat_autoscroll.py does on purpose, because it counts frames
RECORDER_INIT = """
(() => {
  const nativeRaf = window.requestAnimationFrame.bind(window);
  window.__nativeRaf = nativeRaf;
  window.__rafCount = 0;
  window.requestAnimationFrame = (cb) =>
    nativeRaf((t) => {
      window.__rafCount += 1;
      cb(t);
    });
  // Counted, because every one of these is a double rAF and therefore a ~33ms vsync floor inside
  // whatever is being timed across it. GROWTH_AXES used to declare that count by hand per axis
  // and defaulted it to 0 for every generated axis, which left the floor in both ends of those
  // ratios. begin() zeroes this and end() reports it, so a window's floor is measured rather
  // than asserted, and waits taken OUTSIDE a recorder window (ACTION_SETUPS, for one) are
  // excluded by construction rather than by remembering to exclude them.
  window.__paintWaits = 0;
  window.__nextPaint = () => {
    window.__paintWaits += 1;
    return new Promise((resolve) => nativeRaf(() => nativeRaf(() => resolve())));
  };

  // Chromium-only, recorded as a cross-check on the portable stall number and never as the
  // headline.
  //
  // Support is read from supportedEntryTypes, NOT from whether observe() throws. Measured on
  // this tree: PerformanceObserver.observe({ type: "longtask" }) throws on neither WebKit 26.5
  // nor Firefox 153 -- it is accepted and then never fires. A try/catch therefore reports both
  // as supported and both then report zero long tasks, which reads as "no jank on the engine
  // Unsloth Desktop ships on macOS and Linux" rather than as "this API does not exist there".
  // supportedEntryTypes lists `longtask` on Chromium alone.
  window.__longTasks = [];
  window.__longTaskSupported = Boolean(
    (PerformanceObserver.supportedEntryTypes || []).includes("longtask"),
  );
  if (window.__longTaskSupported) {
    try {
      new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          window.__longTasks.push({ start: entry.startTime, duration: entry.duration });
        }
      }).observe({ type: "longtask", buffered: true });
    } catch (e) {
      window.__longTaskSupported = false;
    }
  }

  const recorder = {
    running: false,
    // Which begin() a scheduled callback belongs to. `running` alone cannot answer that: when the
    // next action starts before the previous action's already-scheduled rAF has fired -- and it
    // does, because the actions are issued back to back over a CDP round trip that is shorter
    // than one frame -- that stale callback wakes to `running === true`, pushes the whole
    // between-action gap into the NEW arrays as if it were a frame, and schedules a second
    // recursive loop that then runs alongside the real one for the rest of the run. Measured on
    // the node harness in test_heavy_thread_measurement_integrity.py: without this token the
    // second action records the inter-action gap as its worst frame and counts double the frames.
    generation: 0,
    frames: [],
    stalls: [],
    // When each sample was taken, so end() can close the window at an earlier instant than the
    // one it was called at without the samples from after that instant still being in the arrays.
    frameAt: [],
    stallAt: [],
    startedAt: 0,
    begin() {
      this.running = true;
      this.generation += 1;
      const generation = this.generation;
      this.frames = [];
      this.stalls = [];
      this.frameAt = [];
      this.stallAt = [];
      this.startedAt = performance.now();
      window.__paintWaits = 0;
      let lastFrame = performance.now();
      const frame = () => {
        if (generation !== this.generation) return;
        const now = performance.now();
        this.frames.push(now - lastFrame);
        this.frameAt.push(now);
        lastFrame = now;
        if (this.running) nativeRaf(frame);
      };
      nativeRaf(frame);
      // 1ms setTimeout, not a MessageChannel ping-pong. Measured: the MessageChannel version
      // ticks ~150k/s and halves Firefox's frame rate before any app code runs; this one ticks
      // ~150/s and costs nothing on any engine. The clamp puts a ~4ms floor under the
      // resolution, which is far below any stall a user can feel.
      let lastStall = performance.now();
      const stall = () => {
        if (generation !== this.generation) return;
        const now = performance.now();
        this.stalls.push(now - lastStall);
        this.stallAt.push(now);
        lastStall = now;
        if (this.running) setTimeout(stall, 1);
      };
      setTimeout(stall, 1);
    },
    /**
     * Close the window. `untilMs` closes it at an earlier instant than "now" -- used by the
     * settle loops, which have to keep watching for a while after the page went quiet in order
     * to know that it stayed quiet, and must not charge that watching to the action.
     */
    end(untilMs) {
      this.running = false;
      // Retire this generation as well as stopping it, so the one callback already in flight
      // cannot append to the array a later begin() is about to hand out.
      this.generation += 1;
      const cutoff = untilMs === undefined ? Infinity : untilMs;
      const wallMs = (untilMs === undefined ? performance.now() : untilMs) - this.startedAt;
      // Trimmed, not just clocked: a settle loop that watches an idle page for another second
      // would otherwise add sixty fast frames to the count and drag median_frame_ms down with
      // them, at every size equally, which is the same constant-offset trap as the wall clock.
      const frames = this.frames.filter((_, i) => this.frameAt[i] <= cutoff);
      const stalls = this.stalls.filter((_, i) => this.stallAt[i] <= cutoff);
      const sorted = frames.slice().sort((a, b) => a - b);
      return {
        wall_ms: Math.round(wallMs * 10) / 10,
        // How many double-rAF waits this window was clocked across. Not trimmed to `cutoff`:
        // a wait after the cutoff still happened inside the window that produced wall_ms, and
        // wall_ms is the number this count is subtracted from.
        paint_waits: window.__paintWaits,
        frames: frames.length,
        // The first entry spans begin() to the first callback, so it is the wait for the next
        // vsync as much as a rendered frame. Kept: at these timescales it is ~16ms, well under
        // the 33ms threshold, and dropping it would also drop a genuine stall that landed there.
        worst_frame_ms: Math.round(Math.max(0, ...frames) * 10) / 10,
        median_frame_ms:
          sorted.length === 0 ? 0 : Math.round(sorted[Math.floor(sorted.length / 2)] * 10) / 10,
        frames_over_33: frames.filter((ms) => ms > 33).length,
        stall_ticks: stalls.length,
        longest_stall_ms: Math.round(Math.max(0, ...stalls) * 10) / 10,
        stalls_over_33: stalls.filter((ms) => ms > 33).length,
      };
    },
    /**
     * Time to settle, measured from the START of the action, not from the end of the input.
     *
     * Measured from the end of the gesture it reads ~50ms at every size on every engine, because
     * the answer is then "three frames", which is the minimum this loop can return. From the
     * start of the action it is what a user actually waits: the input, plus everything the page
     * does afterwards before it is calm again.
     */
    async quiet(timeoutMs) {
      const started = performance.now();
      let calm = 0;
      let last = performance.now();
      while (performance.now() - started < timeoutMs) {
        await new Promise((resolve) => nativeRaf(() => resolve()));
        const now = performance.now();
        calm = now - last > 33 ? 0 : calm + 1;
        last = now;
        if (calm >= 3) return performance.now() - this.startedAt;
      }
      return null;
    },
    /**
     * Settle for an action that also restarts the SYNTAX HIGHLIGHTER, which quiet() cannot see.
     *
     * quiet() declares an action settled after three sub-33ms frames. Shiki highlights each fence
     * on its own task, and the lull between two of those batches is longer than that -- the same
     * lull wait_for_highlighting_settled() exists for, where a two-read gate released at 577
     * highlighted tokens out of the 3216 a finished thread holds. So on re-open, which rebuilds
     * every fence from nothing, quiet() stops the clock partway through the rebuild.
     *
     * Settled here means: no frame over 33ms AND no new highlighted token, for graceMs.
     *
     * The returned time is the time of the LAST activity, not the time the grace window expired.
     * graceMs is a fixed cost that every size would pay equally, and a constant added to both
     * ends of a ratio drags it towards 1 -- the same trap the paint floor is subtracted for.
     *
     * `probe` counts highlighted tokens, which is a document-wide query, so it runs on an
     * INTERVAL and not once per frame. An O(nodes) query inside the window being timed would cost
     * more the bigger the thread is, which is to say it would grow like the signal -- the same
     * reason DELETE_JS polls isConnected on a captured node instead of re-counting [data-role].
     * The price is that the returned time is quantised to probeEveryMs, which is far below the
     * differences this axis exists to show.
     */
    async quietUntilIdle(timeoutMs, graceMs, probe, probeEveryMs) {
      const started = performance.now();
      let lastActivity = performance.now();
      let lastCount = probe();
      let lastProbeAt = performance.now();
      let last = performance.now();
      while (performance.now() - started < timeoutMs) {
        await new Promise((resolve) => nativeRaf(() => resolve()));
        const now = performance.now();
        let changed = false;
        if (now - lastProbeAt >= probeEveryMs) {
          const count = probe();
          changed = count !== lastCount;
          lastCount = count;
          lastProbeAt = now;
        }
        if (now - last > 33 || changed) lastActivity = now;
        last = now;
        if (now - lastActivity >= graceMs) {
          return { settleMs: lastActivity - this.startedAt, at: lastActivity };
        }
      }
      return { settleMs: null, at: performance.now() };
    },
  };
  window.__hv = recorder;
})();
"""


def info(message: str) -> None:
    print(f"[heavy-thread] {message}", flush = True)


# ── in-page actions ─────────────────────────────────────────────────── Each one brackets itself with
# __hv.begin()/__hv.end(), so the recorder window is exactly the action rather than the action plus a CDP round trip.
KEYSTROKE_JS = """
async (count) => {
  const api = window.__heavyThread;
  const input = api.composer();
  if (!input) return null;
  input.focus();
  const setValue = Object.getOwnPropertyDescriptor(
    HTMLTextAreaElement.prototype, "value",
  ).set;
  const samples = [];
  window.__hv.begin();
  for (let i = 0; i < count; i += 1) {
    await window.__nextPaint();
    const started = performance.now();
    setValue.call(input, input.value + "a");
    input.dispatchEvent(new Event("input", { bubbles: true }));
    await window.__nextPaint();
    samples.push(performance.now() - started);
  }
  const metrics = window.__hv.end();
  const sorted = samples.slice().sort((a, b) => a - b);
  // domText is what the harness itself wrote; runtimeText is what the runtime received. Only
  // the second can show the keystroke reached React rather than just the DOM node.
  return {
    samples: samples.map((s) => Math.round(s * 10) / 10),
    // The first sample is systematically a cold outlier, which is why the headline is a median.
    median_sample_ms: Math.round(sorted[Math.floor(sorted.length / 2)] * 10) / 10,
    worst_sample_ms: Math.round(sorted[sorted.length - 1] * 10) / 10,
    domText: input.value,
    runtimeText: api.composerText(),
    metrics,
  };
}
"""

SCROLL_JS = """
async ([steps, stepPx, settleMs]) => {
  const api = window.__heavyThread;
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
  // Reverse at either end rather than stopping. A small thread runs out of travel long before a
  // large one does, and a gesture that covers 2600px at 25K chars and 8000px at 300K is not the
  // same gesture, so the two columns would not be comparable.
  let direction = -1;
  let travelled = 0;
  window.__hv.begin();
  for (let i = 0; i < steps; i += 1) {
    if (direction < 0 && target <= 0) direction = 1;
    else if (direction > 0 && target >= bottom) direction = -1;
    const next = Math.min(bottom, Math.max(0, target + direction * stepPx));
    // The wheel event is what the app's own scroll listeners key off; the scrollTo is what
    // moves the viewport in a headless run with no compositor input.
    viewport.dispatchEvent(
      new WheelEvent("wheel", { deltaY: direction * stepPx, bubbles: true, cancelable: true }),
    );
    viewport.scrollTo({ top: next, behavior: "instant" });
    await window.__nextPaint();
    travelled += Math.abs(next - target);
    target = next;
  }
  const gestureMs = performance.now() - window.__hv.startedAt;
  const settleMsTaken = await window.__hv.quiet(settleMs);
  const metrics = window.__hv.end();
  return {
    scrolledPx: travelled,
    gestureMs: Math.round(gestureMs * 10) / 10,
    settleMs: settleMsTaken === null ? null : Math.round(settleMsTaken * 10) / 10,
    metrics,
  };
}
"""

# A wheel gesture of fixed length traverses a fixed number of pixels, so it is comparable across sizes -- and
JUMP_JS = """
async (settleMs) => {
  const api = window.__heavyThread;
  const viewport = api.viewport();
  if (!viewport) return null;
  const bottom = viewport.scrollHeight - viewport.clientHeight;
  viewport.scrollTo({ top: bottom, behavior: "instant" });
  await window.__nextPaint();
  window.__hv.begin();
  const started = performance.now();
  // The wheel event first, and it is load-bearing. Unsloth replaces assistant-ui's autoscroll
  // with an intent-aware one, and a bare scrollTo from the bottom is read as programmatic and
  // snapped straight back: measured, the jump landed at the bottom it started from and the whole
  // column timed nothing. A wheel is what says a person did this.
  viewport.dispatchEvent(
    new WheelEvent("wheel", { deltaY: -bottom, bubbles: true, cancelable: true }),
  );
  viewport.scrollTo({ top: 0, behavior: "instant" });
  await window.__nextPaint();
  const paintedMs = performance.now() - started;
  const settleMsTaken = await window.__hv.quiet(settleMs);
  const metrics = window.__hv.end();
  const landedAt = viewport.scrollTop;
  viewport.scrollTo({ top: bottom, behavior: "instant" });
  await window.__nextPaint();
  return {
    paintedMs: Math.round(paintedMs * 10) / 10,
    settleMs: settleMsTaken === null ? null : Math.round(settleMsTaken * 10) / 10,
    travelledPx: bottom,
    landedAt,
    metrics,
  };
}
"""

# Radix portals the menu to document.body and puts the body on the modal layer, which is the fan-out under suspicion.
# The trigger opens on `pointerdown`, not on `click`: an element.click() leaves the menu shut and the whole measurement
# silently reads zero.
# Everything this window scans, it scans a FIXED number of times: two observer queries, one per portal mutation, and the
# two censuses below, once each.
# Measured on Chromium at 300K that is 2.7ms of a 3208ms open+close, against 0.3ms of 375ms at 25K
# 0.08% of the number at both ends, so the share does not grow with the axis.
# Removing them entirely was measured too, on one page, alternating with the version above: 3389ms against 3394ms at
# 300K.
MENU_JS = """
async (timeoutMs) => {
  const api = window.__heavyThread;
  const trigger = api.actionButton("More");
  if (!trigger) return null;
  // A MutationObserver flag, not a querySelector per frame. The menu content is portaled to the
  // end of document.body, so polling for it walks the whole message list and finds nothing for
  // the entire open latency -- a cost that grows like the signal being measured.
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
  window.__hv.begin();
  const openStarted = performance.now();
  trigger.dispatchEvent(new PointerEvent("pointerdown", { ...pointer, buttons: 1 }));
  trigger.dispatchEvent(new PointerEvent("pointerup", { ...pointer, buttons: 0 }));
  const opened = await settle(true);
  const openMs = opened === null ? null : performance.now() - openStarted;
  const bodyPointerEvents = getComputedStyle(document.body).pointerEvents;
  const itemsWhileOpen = api.openMenuItemCount();
  // Counted here, under the pointer, not in the resting-state census. An autohidden bar is
  // absent at rest by design; one that never mounts at all is a broken page, and only a hovered
  // count tells the two apart.
  const triggersWhileHovered = document.querySelectorAll('[data-slot="tooltip-trigger"]').length;
  // The clock starts BEFORE the dispatch. Radix dismisses synchronously inside it -- layer
  // teardown, focus restore, the body coming off the modal layer and the re-render that follows
  // -- which is the fan-out being measured. Starting it after the dispatch excluded exactly the
  // part worth timing.
  const closeStarted = performance.now();
  document.dispatchEvent(
    new KeyboardEvent("keydown", { key: "Escape", bubbles: true, cancelable: true }),
  );
  const closed = await settle(false);
  const metrics = window.__hv.end();
  watcher.disconnect();
  const closeMs = closed === null ? null : performance.now() - closeStarted;
  return {
    openMs: openMs === null ? null : Math.round(openMs * 10) / 10,
    closeMs: closeMs === null ? null : Math.round(closeMs * 10) / 10,
    open_close_ms:
      openMs === null || closeMs === null ? null : Math.round((openMs + closeMs) * 10) / 10,
    bodyPointerEvents,
    bodyPointerEventsAfterClose: getComputedStyle(document.body).pointerEvents,
    itemsWhileOpen,
    triggersWhileHovered,
    metrics,
  };
}
"""

DELETE_JS = """
async (timeoutMs) => {
  const api = window.__heavyThread;
  const button = api.actionButton("Delete message");
  if (!button) return null;
  // The last assistant message. That is the cheapest delete on the React side -- one subtree
  // unmounts -- so this column under-measures reconciliation, though the export/rebuild/import
  // half is O(messages) wherever the target sits.
  const target = api.lastAssistantMessage();
  const before = api.messageCount();
  window.__hv.begin();
  const started = performance.now();
  button.click();
  let ms = null;
  // isConnected on the captured node is O(1). Re-counting [data-role] every frame would put an
  // O(messages) query inside the window being timed, growing like the signal.
  while (performance.now() - started < timeoutMs) {
    if (target === null || !target.isConnected) {
      ms = performance.now() - started;
      break;
    }
    await window.__nextPaint();
  }
  const metrics = window.__hv.end();
  return { ms, before, after: api.messageCount(), metrics };
}
"""

# Leaving a thread and coming back.
# What comes back is the thread with its COMPLETED TOOL CARDS SHUT, because that is what the app does:
# tool-ui-python.tsx mounts with `defaultOpen={isRunning}` and tool-ui-code-execution.tsx initialises its `open` state
# from the same false flag, so a finished card remounts collapsed.
# The panes are therefore in the census this cell prints and not in the rebuild it times, and the gap was measured
# rather than assumed: 3999 DOM nodes expanded against 3981 rebuilt at 25K, and 42873 against 42675 at 300K.
# That is 0.45% and 0.46% of the tree, the same fraction at both ends, so it moves the ratio by nothing.
# Highlighted tokens are IDENTICAL in the two states (3216 and 35086), because the Shiki-highlighted cell renders
# outside the collapsible by design (#7165) and what is inside it is a plain <pre>: the settle probe below sees the same
# signal either way.
# The two messageCount() calls this makes inside the window are the harness's whole footprint in the number: measured on
# Chromium at 300K, 0.4ms of a 2292ms re-open, and 0.0ms of 363ms at 25K.
REOPEN_JS = """
async ([timeoutMs, settleMs, graceMs, probeEveryMs]) => {
  const api = window.__heavyThread;
  const before = api.messageCount();
  if (!before) return null;
  window.__hv.begin();
  const started = performance.now();
  api.closeThread();
  // Unmount first, or "already back" is indistinguishable from "never left".
  // `paintWaits` below is REPORTED, not subtracted. Reopening is driven by a React state update,
  // so the count check immediately after openThread() always still sees the unmounted tree and
  // the loop always pays at least one __nextPaint() before it can observe the rebuilt messages.
  // That one wait is the observation floor GROWTH_AXES removes (REOPEN_OBSERVATION_FLOOR), the
  // same floor already subtracted from jump and delete. Any wait beyond it is a frame the
  // APPLICATION spent committing rows, so it stays in the number. Reporting the count lets the
  // harness verify it paid at least the floor it removes, and puts the progressive mount's frame
  // count in the table as a diagnostic in its own right.
  let closedMs = null;
  let closePaintWaits = 0;
  while (performance.now() - started < timeoutMs) {
    if (api.messageCount() === 0) { closedMs = performance.now() - started; break; }
    closePaintWaits += 1;
    await window.__nextPaint();
  }
  const reopenStarted = performance.now();
  api.openThread();
  let ms = null;
  let paintWaits = 0;
  while (performance.now() - reopenStarted < timeoutMs) {
    if (api.messageCount() >= before) { ms = performance.now() - reopenStarted; break; }
    paintWaits += 1;
    await window.__nextPaint();
  }
  // Not quiet(): re-open is the action whose whole cost is re-highlighting, and three calm frames
  // land inside the lull between two Shiki batches.
  const settled = await window.__hv.quietUntilIdle(
    settleMs,
    graceMs,
    () => api.highlightedTokenCount(),
    probeEveryMs,
  );
  const metrics = window.__hv.end(settled.at);
  return {
    ms,
    paintWaits,
    closePaintWaits,
    closedMs: closedMs === null ? null : Math.round(closedMs * 10) / 10,
    settleMs: settled.settleMs === null ? null : Math.round(settled.settleMs * 10) / 10,
    before,
    after: api.messageCount(),
    metrics,
  };
}
"""

# vsync intervals. An action that never happened therefore still reports ~33ms, which reads as a
# The floor under every timing clocked across a double rAF:
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


def median(values: list[float | None]) -> float | None:
    """Median across the repetitions, or None if any repetition did not produce a number.

    A None here is not a missing reading, it is a repetition in which the thing being timed never
    happened: the menu that never opened inside SETTLE_TIMEOUT_MS, the delete whose message never
    left the DOM, the action that never reached a settled state. Dropping those and taking the
    median of what is left changes the sample population and reports a partially broken action as
    a clean three-repetition measurement -- and it hides it from harness_failures(), whose
    `openMs is None` / `ms is None` checks then read the median of the repetitions that did work.
    So one bad repetition poisons the aggregate, and the run says so.
    """
    if not values or any(v is None for v in values):
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[middle], 1)
    return round((ordered[middle - 1] + ordered[middle]) / 2, 1)


def cdp_metrics(cdp) -> dict[str, float]:
    if cdp is None:
        return {}
    got = cdp.send("Performance.getMetrics")
    return {m["name"]: m["value"] for m in got["metrics"]}


def cdp_counters(before: dict[str, float], after: dict[str, float]) -> dict[str, float | None]:
    """CHROMIUM-ONLY. Empty dicts off Chromium, and every consumer prints `-` for that."""
    if not before or not after:
        return {
            "layout_count": None,
            "recalc_style_count": None,
            "layout_ms": None,
            "recalc_style_ms": None,
            "task_ms": None,
        }

    def d(name: str) -> float:
        return after.get(name, 0.0) - before.get(name, 0.0)

    return {
        "layout_count": round(d("LayoutCount"), 1),
        "recalc_style_count": round(d("RecalcStyleCount"), 1),
        "layout_ms": round(d("LayoutDuration") * 1000, 1),
        "recalc_style_ms": round(d("RecalcStyleDuration") * 1000, 1),
        "task_ms": round(d("TaskDuration") * 1000, 1),
    }


def long_task_summary(page) -> dict[str, float | None]:
    """CHROMIUM-ONLY. `supported` is the point: without it, an engine with no Long Tasks API
    reports zero jank in exactly the same shape as an engine that had none."""
    # PerformanceObserver callbacks are delivered on a later task, so the entry for the long task at the tail of an
    got = page.evaluate(
        """async () => {
            await new Promise((r) => setTimeout(r, 0));
            return { supported: window.__longTaskSupported, tasks: window.__longTasks };
        }"""
    )
    if not got["supported"]:
        return {"long_tasks": None, "long_task_ms": None, "worst_long_task_ms": None}
    tasks = got["tasks"]
    return {
        "long_tasks": len(tasks),
        "long_task_ms": round(sum(t["duration"] for t in tasks), 1),
        "worst_long_task_ms": round(max((t["duration"] for t in tasks), default = 0.0), 1),
    }


def reset_long_tasks(page) -> None:
    page.evaluate("window.__longTasks.length = 0")


def wait_for_highlighting_settled(page, timeout_ms: int) -> None:
    """Block until Shiki has stopped adding tokens.

    FIVE stable reads a quarter of a second apart, not two consecutive ones. Two adjacent
    rAF-polled reads land inside the lull between two async highlight batches all the time:
    measured on WebKit, a two-read version released the gate at 577 highlighted tokens where the
    finished thread has 3216, so the whole engine column was measured against a thread that was
    still building itself.
    """
    page.evaluate("() => { window.__hvTokens = undefined; }")
    page.wait_for_function(
        """() => {
            const n = window.__heavyThread.highlightedTokenCount();
            const state = window.__hvTokens || { value: -1, stable: 0 };
            if (n === state.value && n > 0) state.stable += 1;
            else { state.value = n; state.stable = 0; }
            window.__hvTokens = state;
            return state.stable >= 5;
        }""",
        polling = 250,
        timeout = timeout_ms,
    )


def run_action(page, cdp, name: str, script: str, arg) -> dict:
    """One action, with the portable recorder inside it and the CDP counters bracketing it."""
    reset_long_tasks(page)
    before = cdp_metrics(cdp)
    raw = page.evaluate(script, arg)
    after = cdp_metrics(cdp)
    if raw is None:
        return {"name": name, "ran": False, **cdp_counters({}, {}), **long_task_summary(page)}
    out = {"name": name, "ran": True}
    out.update(raw.pop("metrics"))
    out.update(raw)
    out.update(cdp_counters(before, after))
    out.update(long_task_summary(page))
    return out


# The gate that says expandTools() really mounted the result panes.
# Radix keeps that element in the tree for its collapse animation, so it is present while the card is shut: measured on
# this tree at 300K characters, immediately after seeding and BEFORE any expandTools() call, collapsibleOutputs was
EXPANDED_PANES_GATE_JS = "(n) => window.__heavyThread.counts().codeExecutionPanes >= n"


def build_fixture(page) -> None:
    """Bring the page back to the fixture every column claims to have been measured on, untimed.

    Order matters and it is the reason this is one function rather than three call sites. Radix
    unmounts collapsed content, so the tool result panes -- which are CODE, two of the seven fences
    a content cycle produces -- do not exist until expandTools() has run. Waiting for the
    highlighter BEFORE expanding therefore gates on the fences that were already there and then
    mounts a fresh batch of unhighlighted ones, whose Shiki work lands in whatever is timed next.
    Expand first, then wait for the highlighter, which is the order measure_cell() seeds in.
    """
    expanded = page.evaluate("() => window.__heavyThread.expandTools()")
    if expanded:
        page.wait_for_function(
            EXPANDED_PANES_GATE_JS,
            arg = expanded,
            timeout = ACTION_TIMEOUT_MS,
        )
    wait_for_highlighting_settled(page, ACTION_TIMEOUT_MS)


def one_repetition(page, cdp) -> dict[str, dict]:
    """The five scripted actions, once, in the order a user meets them."""
    rep: dict[str, dict] = {}
    build_fixture(page)
    rep["keystroke"] = run_action(page, cdp, "keystroke", KEYSTROKE_JS, KEYSTROKES)
    rep["scroll"] = run_action(
        page,
        cdp,
        "scroll",
        SCROLL_JS,
        [SCROLL_STEPS, SCROLL_STEP_PX, SETTLE_TIMEOUT_MS],
    )
    rep["jump"] = run_action(page, cdp, "jump", JUMP_JS, SETTLE_TIMEOUT_MS)

    # The action bar is hover-revealed, so put the pointer on the message before reaching for it.
    page.evaluate(
        """() => { const m = window.__heavyThread.lastAssistantMessage();
            if (m) m.scrollIntoView({ block: "center", behavior: "instant" }); }"""
    )
    page.wait_for_function(
        """() => {
            const top = window.__heavyThread.viewportMetrics().scrollTop;
            const settled = window.__hvTop === top;
            window.__hvTop = top;
            return settled;
        }""",
        timeout = ACTION_TIMEOUT_MS,
    )
    page.locator('[data-role="assistant"]').last.hover(timeout = ACTION_TIMEOUT_MS)
    rep["menu"] = run_action(page, cdp, "menu", MENU_JS, SETTLE_TIMEOUT_MS)

    page.locator('[data-role="assistant"]').last.hover(timeout = ACTION_TIMEOUT_MS)
    rep["delete"] = run_action(page, cdp, "delete", DELETE_JS, SETTLE_TIMEOUT_MS)

    # before any repetition ran and so never sees it. Restore, then rebuild, untimed.
    # The delete is PERMANENT: it removes a message from the runtime's repository, not from the view.
    restored = page.evaluate("() => window.__heavyThread.restore()")
    page.wait_for_function(
        "(n) => window.__heavyThread.messageCount() >= n",
        arg = restored,
        timeout = ACTION_TIMEOUT_MS,
    )
    # The previous repetition ended by re-opening the thread, which throws away every highlighted fence and starts Shiki
    # again, and by deleting a message, which the restore below puts back.
    # Without this wait, repetitions 2 and 3 measure a thread that is still building itself: measured on Chromium at
    # 300K, the scroll gesture read 667ms on the first repetition and 1100ms on the two that followed, and the
    # difference was the re-highlighting, not the scroll.
    build_fixture(page)

    rep["reopen"] = run_action(
        page,
        cdp,
        "reopen",
        REOPEN_JS,
        [SETTLE_TIMEOUT_MS, SETTLE_TIMEOUT_MS, HIGHLIGHT_GRACE_MS, HIGHLIGHT_PROBE_MS],
    )
    return rep


# Portable headline per action, plus the action's own DOM-observable duration.
HEADLINE = {
    "keystroke": ("median_sample_ms", True),
    "scroll": ("gestureMs", False),
    "jump": ("paintedMs", True),
    "menu": ("open_close_ms", True),
    "delete": ("ms", True),
    "reopen": ("ms", False),
}


def summarise(reps: list[dict[str, dict]]) -> dict[str, dict]:
    """Median across repetitions, per action, per metric. Medians because the first repetition is
    systematically the slowest and a mean would carry it into every cell."""
    out: dict[str, dict] = {}
    for action in ACTIONS:
        rows = [r[action] for r in reps if action in r]
        if not rows or not all(r.get("ran") for r in rows):
            out[action] = {"ran": False}
            continue
        merged: dict = {"ran": True, "repetitions": len(rows)}
        numeric_keys = set()
        for row in rows:
            for key, value in row.items():
                # menu that never opened. Present-and-None is the readable form of that.
                # `value is None` is deliberately a key too.
                if value is None or (
                    isinstance(value, (int, float)) and not isinstance(value, bool)
                ):
                    numeric_keys.add(key)
        for key in sorted(numeric_keys):
            merged[key] = median([r.get(key) for r in rows])
        # Values that are not numbers are proofs the action really happened, not timings, so the last repetition's is
        for key in ("domText", "runtimeText", "bodyPointerEvents", "bodyPointerEventsAfterClose"):
            if key in rows[-1]:
                merged[key] = rows[-1][key]
        # The headline value from each repetition, unaggregated, so a median can be checked against the spread it came
        merged["per_repetition"] = [r.get(HEADLINE[action][0]) for r in rows]
        out[action] = merged
    return out


def measure_cell(context, engine: str, size: int) -> dict:
    """Seed a fresh page to `size` characters of content and run the action set REPEATS times."""
    page = context.new_page()
    result: dict = {"chars_requested": size, "engine": engine}
    # A request that escapes to the server, or a warning storm, is work this harness would be charging to the app once
    # per message.
    # startswith, not `"/api/" in url`: vite serves the app's own source modules from paths like
    # /src/features/chat/api/chat-api.ts, and a substring match counts dozens of those as network calls.
    api_prefix = f"{BASE}/api/"
    stray_requests: list[str] = []
    console_warnings: list[str] = []
    # Severity kept SEPARATE from the warning list.
    console_errors: list[str] = []
    page.on(
        "request",
        lambda r: stray_requests.append(r.url) if r.url.startswith(api_prefix) else None,
    )
    page.on(
        "console",
        lambda m: (
            console_errors.append(m.text[:200])
            if m.type == "error"
            else console_warnings.append(m.text[:200])
            if m.type == "warning"
            else None
        ),
    )
    page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"[:200]))
    cdp = None
    try:
        page.goto(f"{BASE}/smoke-heavy-thread.html", wait_until = "domcontentloaded")
        page.wait_for_function("() => Boolean(window.__heavyThread)", timeout = 60_000)
        if engine == "chromium":
            cdp = context.new_cdp_session(page)
            cdp.send("Performance.enable")

        # Seeding unthrottled and untimed:
        plan = page.evaluate("(n) => window.__heavyThread.seed(n)", size)
        result["plan"] = plan
        page.wait_for_function(
            "(n) => window.__heavyThread.messageCount() >= n",
            arg = plan["messages"],
            timeout = SEED_TIMEOUT_MS,
        )
        # Radix unmounts collapsed content, so a thread of closed tool cards carries no result panes at all.
        result["tool_triggers_expanded"] = page.evaluate("() => window.__heavyThread.expandTools()")
        # Single-selector gates. counts() walks every element in the document, so polling it per frame makes seeding
        page.wait_for_function(
            EXPANDED_PANES_GATE_JS,
            arg = max(1, result["tool_triggers_expanded"]),
            timeout = SEED_TIMEOUT_MS,
        )
        # Shiki is async and per block, and a <pre> exists before it is highlighted, so counting code blocks gates
        wait_for_highlighting_settled(page, SEED_TIMEOUT_MS)
        result["counts"] = page.evaluate("window.__heavyThread.counts()")
        result["viewport"] = page.evaluate("window.__heavyThread.viewportMetrics()")
        result["seed_api_requests"] = len(stray_requests)
        result["seed_console_warnings"] = len(console_warnings)
        result["first_seed_warning"] = console_warnings[0] if console_warnings else "-"
        result["seed_console_errors"] = len(console_errors)
        result["first_seed_error"] = console_errors[0] if console_errors else "-"
        stray_requests.clear()
        console_warnings.clear()
        console_errors.clear()

        result["cpu_throttle_rate"] = 1.0
        if cdp is not None and CPU_THROTTLE_RATE != 1.0:
            cdp.send("Emulation.setCPUThrottlingRate", {"rate": CPU_THROTTLE_RATE})
            result["cpu_throttle_rate"] = CPU_THROTTLE_RATE
        result["long_task_supported"] = page.evaluate("window.__longTaskSupported")
        result["paint_floor_ms"] = round(page.evaluate(PAINT_FLOOR_JS, 9), 2)

        reps = []
        for index in range(REPEATS):
            info(f"  {engine} {size} chars: repetition {index + 1}/{REPEATS}")
            reps.append(one_repetition(page, cdp))
        result["repetitions"] = REPEATS
        result["actions"] = summarise(reps)
        result["raw_repetitions"] = reps

        if cdp is not None and CPU_THROTTLE_RATE != 1.0:
            cdp.send("Emulation.setCPUThrottlingRate", {"rate": 1})
        # Cumulative over seeding and every action:
        result["raf_callbacks"] = page.evaluate("window.__rafCount")
        result["stray_api_requests"] = len(stray_requests)
        # The URLs, not only how many.
        result["stray_api_urls"] = sorted(set(stray_requests))[:8]
        # Answered inside the page by the smoke entry's allowlist.
        result["stubbed_api_requests"] = len(page.evaluate("window.__stubbedApi || []"))
        result["console_warnings"] = len(console_warnings)
        result["first_console_warning"] = console_warnings[0] if console_warnings else "-"
        result["console_errors"] = len(console_errors)
        result["first_console_error"] = console_errors[0] if console_errors else "-"
    finally:
        page.close()
    return result


def run() -> dict:
    results: dict = {
        "label": LABEL,
        "base": BASE,
        "sizes": SIZES,
        "engines": ENGINES,
        "repetitions": REPEATS,
        "cpu_throttle_rate_requested": CPU_THROTTLE_RATE,
        "by_engine": {},
    }
    with sync_playwright() as p:
        for engine in ENGINES:
            info(f"engine {engine}")
            launcher = getattr(p, engine)
            kwargs = {"headless": os.environ.get("SMOKE_HEADLESS", "1") == "1"}
            # Chromium-only flags. Passing them to Firefox or WebKit is not "ignored", it is a launch failure, which
            if engine == "chromium":
                kwargs["args"] = chromium_launch_args()
            browser = launcher.launch(**kwargs)
            results["by_engine"][engine] = {"version": browser.version, "by_size": {}}
            context = browser.new_context(viewport = {"width": 1440, "height": 900})
            context.add_init_script(RECORDER_INIT)
            # Anchored at the origin so it cannot swallow vite's own module URLs, which live
            context.route(
                re.compile(rf"^{re.escape(BASE)}/api/"),
                lambda route: route.fulfill(
                    status = 200,
                    content_type = "application/json",
                    body = "{}",
                ),
            )
            for size in SIZES:
                info(f"measuring {engine} at {size} chars")
                # A renderer that dies mid-cell used to take the whole matrix with it:
                try:
                    cell = measure_cell(context, engine, size)
                except Exception as exc:  # noqa: BLE001 - the message is the whole point
                    info(f"CRASHED {engine} at {size} chars: {type(exc).__name__}: {exc}")
                    cell = {
                        "chars_requested": size,
                        "engine": engine,
                        "crashed": f"{type(exc).__name__}: {exc}"[:400],
                    }
                results["by_engine"][engine]["by_size"][str(size)] = cell
            context.close()
            browser.close()
    return results


# Every recorded metric appears here.
# That is the rule the harnesses in this directory are held to: a metric that is recorded and never read is how one goes
# false-green, and tests/studio/test_heavy_thread_harness_contract.py fails if anything recorded below is missing.
# CHROMIUM-ONLY rows are labelled in their own name, because off Chromium they print `-` and a `-` that means "not
# supported here" must not read as "zero".
def _action(action: str, key: str):
    return lambda r: r["actions"][action][key]


def _floor_from(action: str, key: str):
    """A `floored` that is measured per row rather than declared once for every action."""
    return lambda r: (r.get("actions", {}).get(action) or {}).get(key) or 0


TABLE_ROWS = (
    ("chars requested", lambda r: r["chars_requested"]),
    ("chars rendered", lambda r: r["plan"]["chars"]),
    ("messages seeded", lambda r: r["plan"]["messages"]),
    ("content cycles", lambda r: r["plan"]["cycles"]),
    ("chars per cycle", lambda r: r["plan"]["cycleChars"]),
    ("content kinds", lambda r: r["plan"]["kinds"]),
    ("tool cards expanded", lambda r: r["tool_triggers_expanded"]),
    ("repetitions", lambda r: r["repetitions"]),
    ("cpu throttle rate", lambda r: r["cpu_throttle_rate"]),
    ("paint floor ms", lambda r: r["paint_floor_ms"]),
    ("reopen paint waits", lambda r: (r.get("actions", {}).get("reopen") or {}).get("paintWaits")),
    ("longtask api supported", lambda r: r["long_task_supported"]),
    ("seed api requests", lambda r: r["seed_api_requests"]),
    ("seed console warnings", lambda r: r["seed_console_warnings"]),
    ("first seed warning", lambda r: r["first_seed_warning"]),
    ("action api requests", lambda r: r["stray_api_requests"]),
    ("stubbed api requests", lambda r: r.get("stubbed_api_requests", 0)),
    ("action console warnings", lambda r: r["console_warnings"]),
    ("seed console errors", lambda r: r.get("seed_console_errors", 0)),
    ("action console errors", lambda r: r.get("console_errors", 0)),
    ("first action error", lambda r: r.get("first_console_error", "-")),
    ("first action warning", lambda r: r["first_console_warning"]),
    ("messages rendered", lambda r: r["counts"]["messages"]),
    ("dom nodes", lambda r: r["counts"]["domNodes"]),
    ("code blocks", lambda r: r["counts"]["codeBlocks"]),
    # Side by side deliberately:
    ("code chars", lambda r: r["counts"].get("codeChars", 0)),
    ("highlighted tokens", lambda r: r["counts"]["highlightedTokens"]),
    ("fence blocks", lambda r: r["counts"].get("fenceBlocks", 0)),
    ("deferred fences", lambda r: r["counts"].get("deferredFences", 0)),
    (
        "mounted but unhighlighted fences",
        lambda r: r["counts"].get("unhighlightedMountedFences", 0),
    ),
    ("tool parts", lambda r: r["counts"]["toolParts"]),
    ("collapsible tool outputs", lambda r: r["counts"]["collapsibleOutputs"]),
    ("code execution panes", lambda r: r["counts"]["codeExecutionPanes"]),
    ("artifact cards", lambda r: r["counts"]["artifactCards"]),
    ("images", lambda r: r["counts"]["images"]),
    ("action bars", lambda r: r["counts"]["actionBars"]),
    ("tooltip triggers", lambda r: r["counts"]["tooltipTriggers"]),
    ("viewport scrollHeight", lambda r: r["viewport"]["scrollHeight"]),
    ("viewport clientHeight", lambda r: r["viewport"]["clientHeight"]),
    ("rAF callbacks", lambda r: r["raf_callbacks"]),
)

for _name in ACTIONS:
    TABLE_ROWS = TABLE_ROWS + (
        (f"{_name} ran", _action(_name, "ran")),
        (f"{_name} wall ms", _action(_name, "wall_ms")),
        (f"{_name} longest stall ms", _action(_name, "longest_stall_ms")),
        (f"{_name} worst frame ms", _action(_name, "worst_frame_ms")),
        (f"{_name} median frame ms", _action(_name, "median_frame_ms")),
        (f"{_name} frames over 33ms", _action(_name, "frames_over_33")),
        (f"{_name} frames", _action(_name, "frames")),
        (f"{_name} stalls over 33ms", _action(_name, "stalls_over_33")),
        (f"{_name} stall ticks", _action(_name, "stall_ticks")),
        (f"{_name} layouts (chromium only)", _action(_name, "layout_count")),
        (f"{_name} layout ms (chromium only)", _action(_name, "layout_ms")),
        (f"{_name} recalcs (chromium only)", _action(_name, "recalc_style_count")),
        (f"{_name} recalc ms (chromium only)", _action(_name, "recalc_style_ms")),
        (f"{_name} task ms (chromium only)", _action(_name, "task_ms")),
        (f"{_name} longtasks (chromium only)", _action(_name, "long_tasks")),
        (f"{_name} longtask ms (chromium only)", _action(_name, "long_task_ms")),
        (f"{_name} worst longtask ms (chromium only)", _action(_name, "worst_long_task_ms")),
    )

TABLE_ROWS = TABLE_ROWS + (
    ("keystroke median ms", _action("keystroke", "median_sample_ms")),
    ("keystroke worst ms", _action("keystroke", "worst_sample_ms")),
    (
        "keystroke per repetition",
        lambda r: "/".join(str(v) for v in r["actions"]["keystroke"]["per_repetition"]),
    ),
    (
        "scroll per repetition",
        lambda r: "/".join(str(v) for v in r["actions"]["scroll"]["per_repetition"]),
    ),
    (
        "menu per repetition",
        lambda r: "/".join(str(v) for v in r["actions"]["menu"]["per_repetition"]),
    ),
    (
        "delete per repetition",
        lambda r: "/".join(str(v) for v in r["actions"]["delete"]["per_repetition"]),
    ),
    (
        "reopen per repetition",
        lambda r: "/".join(str(v) for v in r["actions"]["reopen"]["per_repetition"]),
    ),
    ("keystroke dom text", _action("keystroke", "domText")),
    ("keystroke runtime text", _action("keystroke", "runtimeText")),
    ("scroll gesture ms", _action("scroll", "gestureMs")),
    ("scroll settle ms", _action("scroll", "settleMs")),
    ("scroll px", _action("scroll", "scrolledPx")),
    ("jump painted ms", _action("jump", "paintedMs")),
    ("jump settle ms", _action("jump", "settleMs")),
    ("jump px", _action("jump", "travelledPx")),
    ("jump landed at", _action("jump", "landedAt")),
    ("menu open ms", _action("menu", "openMs")),
    ("menu close ms", _action("menu", "closeMs")),
    ("menu open+close ms", _action("menu", "open_close_ms")),
    ("menu items while open", _action("menu", "itemsWhileOpen")),
    ("menu body pe while open", _action("menu", "bodyPointerEvents")),
    ("menu body pe after close", _action("menu", "bodyPointerEventsAfterClose")),
    ("menu triggers hovered", _action("menu", "triggersWhileHovered")),
    ("delete ms", _action("delete", "ms")),
    ("delete messages before", _action("delete", "before")),
    ("delete messages after", _action("delete", "after")),
    ("reopen ms", _action("reopen", "ms")),
    ("reopen unmount ms", _action("reopen", "closedMs")),
    ("reopen settle ms", _action("reopen", "settleMs")),
    ("reopen messages before", _action("reopen", "before")),
    ("reopen messages after", _action("reopen", "after")),
)


def print_table(results: dict) -> None:
    """Every recorded metric, printed, one column per (engine, size)."""
    columns = [(engine, str(size)) for engine in results["engines"] for size in results["sizes"]]
    rows = []
    for name, pick in TABLE_ROWS:
        cells = []
        for engine, size in columns:
            try:
                value = pick(results["by_engine"][engine]["by_size"][size])
                cells.append("-" if value is None else str(value))
            except (KeyError, TypeError):
                cells.append("-")
        rows.append((name, cells))
    label_width = max(len(name) for name, _ in rows) + 2
    # From the widest cell, not a constant:
    headers = [f"{engine[:4]}/{int(size) // 1000}K" for engine, size in columns]
    cell_width = max([len(c) for _, cells in rows for c in cells] + [len(h) for h in headers]) + 2
    header = "".ljust(label_width) + "".join(h.rjust(cell_width) for h in headers)
    info(header)
    info("-" * len(header))
    for name, cells in rows:
        info(name.ljust(label_width) + "".join(cell.rjust(cell_width) for cell in cells))


# The double-rAF waits inside `reopen ms` that are OBSERVATION rather than latency.
# Every FURTHER wait the loop pays is not overhead.
# The measured count is 1 on a build that rebuilds in one commit and rises with thread size on one that mounts
# progressively (1 at 25K, 8 at 100K, 24 at 300K on a 220-message fixture), so subtracting all of them removed ~33ms
# from one arm's 300K cell and ~800ms from the other's.
# That is asymmetric between the two arms of a comparison AND grows with the axis being varied: it reported a reopen
# curve that rises faster than the single-commit build's (7.60x against 7.08x from 25K to 300K) as one that rises slower
# (5.07x), which is the opposite sign.
# A floor exists to remove a CONSTANT the instrument adds, and the moment it tracks the thing being measured it is no
# longer a floor.
REOPEN_OBSERVATION_FLOOR = 1
# `<action> wall ms` spans the whole recorder window and normally takes the window's measured `paint_waits`, because
WALL_FLOOR_OVERRIDES = {"reopen": 1 + REOPEN_OBSERVATION_FLOOR}


def _wall_floor(action: str):
    """The floor for `<action> wall ms`: the measured window count, unless the action overrides it
    because part of that count is application work rather than instrument idle."""
    override = WALL_FLOOR_OVERRIDES.get(action)
    return _floor_from(action, "paint_waits") if override is None else override


# Growth axes: the whole point of the harness is that these rise with content.
# The third field is HOW MANY double-rAF waits the metric is clocked across;
GROWTH_AXES = tuple(
    [(f"{a} longest stall ms", _action(a, "longest_stall_ms"), 0) for a in ACTIONS]
    + [(f"{a} worst frame ms", _action(a, "worst_frame_ms"), 0) for a in ACTIONS]
    + [(f"{a} frames over 33ms", _action(a, "frames_over_33"), 0) for a in ACTIONS]
    # The floor is READ from the row, not declared: every one of these windows crosses a different number of mandatory
    # double-rAF waits, and declaring 0 for all of them left roughly `paint_waits * paint_floor_ms` of constant baseline
    # in both ends of the ratio.
    # The one exception is in WALL_FLOOR_OVERRIDES: reading the row is only right while the waits in the window are the
    # harness idling between driven steps, and reopen's are not.
    + [(f"{a} wall ms", _action(a, "wall_ms"), _wall_floor(a)) for a in ACTIONS]
    + [
        # The rule for the entries below: an axis measured from `__hv.startedAt` spans the WHOLE recorder window, so it
        # carries every double-rAF wait in it and takes the measured `paint_waits`.
        # gestureMs is `performance.now() - __hv.startedAt`, and BOTH settle figures come from `quiet()` /
        # `quietUntilIdle()`, which return `...
        # - this.startedAt` rather than the time they themselves took.
        # Counted at runtime rather than written in: the twenty come from a LOOP, so the literal `__nextPaint()` count
        # in the source is one, and any hand-declared number here would have been wrong in the same way the old zero
        ("keystroke median ms", _action("keystroke", "median_sample_ms"), 1),
        ("scroll gesture ms", _action("scroll", "gestureMs"), _floor_from("scroll", "paint_waits")),
        ("scroll settle ms", _action("scroll", "settleMs"), _floor_from("scroll", "paint_waits")),
        # the number never contained.
        # NOT paint_waits: paintedMs starts at a mark taken after begin() and spans one wait, while the jump's window
        ("jump painted ms", _action("jump", "paintedMs"), 1),
        ("jump settle ms", _action("jump", "settleMs"), _floor_from("jump", "paint_waits")),
        # Also NOT paint_waits: MENU_JS awaits no paint at all, and its two floors come from settle() reading the
        ("menu open+close ms", _action("menu", "open_close_ms"), 2),
        ("delete ms", _action("delete", "ms"), 1),
        # `paintWaits`: see REOPEN_OBSERVATION_FLOOR above. Leaving it at 0 is still wrong, that
        # The OBSERVATION floor only, and deliberately a constant rather than the measured `paintWaits`:
        ("reopen ms", _action("reopen", "ms"), REOPEN_OBSERVATION_FLOOR),
    ]
)
# A ratio at or below this from the smallest size to the largest means the axis did not respond to twelve times the
DISCRIMINATION_RATIO = float(os.environ.get("SMOKE_DISCRIMINATION_RATIO", "1.5"))
# cannot be formed against zero, so DISCRIMINATION_RATIO does not apply to these axes at all and
# What a counter that starts at zero has to REACH before its rise counts as an answer.
ZERO_BASED_MIN_RISE = int(os.environ.get("SMOKE_ZERO_BASED_MIN_RISE", "5"))
# Which axes are COUNTS.
# an UNFLOORED timing such as `longest stall ms` or `worst frame ms` is zero at the smallest size whenever the action
# ends before the recorder produces a sample, and it was then treated as a dropped-frame counter, so a noisy 5ms at the
COUNTER_AXES = frozenset(f"{a} frames over 33ms" for a in ACTIONS)
# Engine chatter is tolerated up to this many warnings per size.
CONSOLE_WARNING_ALLOWANCE = int(os.environ.get("SMOKE_CONSOLE_WARNING_ALLOWANCE", "4"))


def resolve_floor(floored, row: dict) -> float:
    """The floor count for one row, as an INT.

    `floored` may be a callable, and the growth report is written to JSON at the end of the run.
    Putting the callable itself in the report made `json.dumps` raise `Object of type function is
    not JSON serializable`, which failed every complete run AFTER all the measurements were taken.
    Nothing in the unit tests caught it because none of them serialise the report.
    """
    # NOT int().
    # `summarise` takes a median across repetitions, so an even-repetition run whose repetitions paid 1 and 2 waits
    # reports 1.5, and truncating that to 1 left half a vsync floor in the wall-clock axis and published a distorted
    value = floored(row) if callable(floored) else floored
    return value if isinstance(value, (int, float)) else 0


def growth(cells: dict, pick, floored, sizes: list[int]) -> tuple[float | None, float | None]:
    """`floored` is a COUNT of double-rAF waits inside the metric, not a flag.

    It may be an int, declared once for an axis, or a callable taking the row, for an axis whose
    window crosses a different number of waits per action. The generated `wall ms` axes are the
    second kind: they were declared 0 for every action, which left roughly `paint_waits *
    paint_floor_ms` in both ends of those ratios.

    Each `await __nextPaint()` a metric is clocked across contributes its own ~33ms vsync floor,
    and a metric that contains two of them carries two. `menu open+close ms` is the case: settle()
    reads the pre-MutationObserver state on entry, both times, so opening and closing each wait
    out a full double rAF before their first true comparison. Subtracting one floor from a sum of
    two left ~33ms of constant baseline in the number, which drags the ratio towards 1 in exactly
    the way the floor is subtracted to prevent.
    """
    try:
        rows = (cells[str(sizes[0])], cells[str(sizes[-1])])
        values = []
        for row in rows:
            value = pick(row)
            if value is None:
                return None, None
            count = resolve_floor(floored, row)
            if count:
                value -= count * row["paint_floor_ms"]
            values.append(round(value, 2))
        return values[0], values[1]
    except (KeyError, TypeError):
        return None, None


def report_growth(results: dict) -> dict[str, dict[str, dict]]:
    """Per engine, per axis: the value at the smallest and largest size and their ratio."""
    report: dict[str, dict[str, dict]] = {}
    for engine in results["engines"]:
        cells = results["by_engine"][engine]["by_size"]
        per_axis: dict[str, dict] = {}
        for name, pick, floored in GROWTH_AXES:
            small, large = growth(cells, pick, floored, results["sizes"])
            # Resolved here, once, so what lands in the JSON is the count that was actually subtracted at each end
            floor_counts = [
                resolve_floor(floored, cells.get(str(size), {}))
                for size in (results["sizes"][0], results["sizes"][-1])
            ]
            if small is None or large is None:
                per_axis[name] = {
                    "small": None,
                    "large": None,
                    "ratio": None,
                    "discriminated": False,
                    "reason": "not recorded",
                }
                continue
            if small <= 0:
                # A COUNT that is 0 at the smallest size and 4 at the largest has no ratio and has still answered the
                # question, so it counts as discriminating when it really rose.
                # A TIMING does not get that credit.
                if floored:
                    per_axis[name] = {
                        "small": small,
                        "large": large,
                        "ratio": None,
                        "discriminated": False,
                        "reason": "at or under the paint floor at the smallest size",
                        "floored": floor_counts,
                    }
                    continue
                # SMOKE_DISCRIMINATION_RATIO said, because a ratio was never computed for it.
                # `large > small` is not enough.
                if name not in COUNTER_AXES:
                    # A timing that reads zero at the smallest size did not "grow from nothing", it resolved below what
                    # the recorder can see.
                    # ZERO_BASED_MIN_RISE is a count of events and means nothing applied to milliseconds.
                    per_axis[name] = {
                        "small": small,
                        "large": large,
                        "ratio": None,
                        "discriminated": False,
                        "reason": (
                            "zero at the smallest size and this axis is a timing, not a count, "
                            "so there is no rise to measure"
                        ),
                        "floored": floor_counts,
                    }
                    continue
                rose = large >= ZERO_BASED_MIN_RISE
                if rose:
                    reason = f"rose from zero to {large}"
                elif large > small:
                    reason = (
                        f"rose from zero only to {large}, under the {ZERO_BASED_MIN_RISE} this "
                        "counter needs to be distinguishable from noise"
                    )
                else:
                    reason = "zero at both ends"
                per_axis[name] = {
                    "small": small,
                    "large": large,
                    "ratio": None,
                    "discriminated": rose,
                    "reason": reason,
                    "floored": floor_counts,
                }
                continue
            ratio = round(large / small, 2)
            # The noise floor applies to a counter whatever its baseline.
            # A dropped-frame count going 1 -> 2 is a ratio of 2.0 and cleared DISCRIMINATION_RATIO, and since
            # harness_failures accepts any single discriminating axis, that one incidental frame could carry the CI
            noisy_counter = name in COUNTER_AXES and large < ZERO_BASED_MIN_RISE
            per_axis[name] = {
                "small": small,
                "large": large,
                "ratio": ratio,
                "discriminated": ratio > DISCRIMINATION_RATIO and not noisy_counter,
                "reason": (
                    f"only {large} events at the largest size, under the "
                    f"{ZERO_BASED_MIN_RISE} this counter needs to be distinguishable from noise"
                    if noisy_counter
                    else "-"
                ),
                "floored": floor_counts,
            }
        report[engine] = per_axis
    return report


def print_growth(results: dict, report: dict) -> None:
    for engine, per_axis in report.items():
        info("")
        info(
            f"growth on {engine} ({results['sizes'][0]} -> {results['sizes'][-1]} chars, "
            f"median of {results['repetitions']} repetitions)"
        )
        for name, row in per_axis.items():
            if row["ratio"] is None:
                mark = "DISCRIMINATES" if row["discriminated"] else "flat"
                small = "-" if row["small"] is None else row["small"]
                large = "-" if row["large"] is None else row["large"]
                info(
                    f"  {name:<34} {small:>10} -> {large:>10}       -  " f"{mark} ({row['reason']})"
                )
                continue
            mark = "DISCRIMINATES" if row["discriminated"] else "flat"
            floor_note = " (paint floor removed)" if row.get("floored") else ""
            info(
                f"  {name:<34} {row['small']:>10} -> {row['large']:>10}  "
                f"{row['ratio']:>6.2f}x  {mark}{floor_note}"
            )


# The declared double-rAF count for each axis whose action reports how many it actually paid.
# GROWTH_AXES holds the declaration;
# The check is a LOWER BOUND, not equality.
FLOOR_COUNTERS = {"reopen ms": ("reopen", "paintWaits")}


def declared_floor(axis_name: str) -> int | None:
    """The `floored` column of GROWTH_AXES for one axis, by exact name.

    Exact rather than prefix-matched: `reopen ms` and a later `reopen settle ms` would both match
    a prefix, and the check would silently compare one axis's waits against another's declaration.
    A name that is not an axis returns None, which the caller reports rather than skips.
    """
    for name, _pick, floored in GROWTH_AXES:
        if name == axis_name:
            return floored
    return None


def floor_declaration_problems(results: dict) -> list[str]:
    """Axes that subtract more double-rAF floors than the action actually waited out."""
    problems: list[str] = []
    for engine in results["engines"]:
        for size in results["sizes"]:
            row = results["by_engine"][engine]["by_size"].get(str(size), {})
            if "crashed" in row:
                continue
            for axis_name, (action, counter) in FLOOR_COUNTERS.items():
                measured = row.get("actions", {}).get(action) or {}
                # An action that did not run is already reported by harness_failures, with the reason.
                if not measured.get("ran", True):
                    continue
                observed = measured.get(counter)
                if observed is None:
                    problems.append(
                        f"{engine} at {size} chars recorded no {counter} for {action}, so the "
                        f"paint floor subtracted from '{axis_name}' is unverified"
                    )
                    continue
                # compared as a callable and reported wrong every time. A hand-declared literal
                # Resolved against THIS row, so an axis whose floor is read from the row (see `_floor_from`) is
                declared = resolve_floor(declared_floor(axis_name), row)
                if observed >= declared:
                    continue
                problems.append(
                    f"{engine} at {size} chars paid {observed} paint wait(s) in {action} but "
                    f"GROWTH_AXES subtracts {declared} from '{axis_name}'; a metric cannot be "
                    "credited with a floor it never waited out"
                )
    return problems


def harness_failures(results: dict, report: dict) -> list[str]:
    """Only the ways this harness can be measuring nothing. No performance budgets: see the
    module docstring."""
    failures: list[str] = list(floor_declaration_problems(results))
    for engine in results["engines"]:
        for size in results["sizes"]:
            row = results["by_engine"][engine]["by_size"][str(size)]
            where = f"{engine} at {size} chars"
            if "crashed" in row:
                failures.append(
                    f"{where} crashed before it produced a measurement: {row['crashed']}"
                )
                continue
            counts = row["counts"]
            plan = row["plan"]
            # A request reaching the server is a round trip to another process inside a region being timed, once per
            if row["stray_api_requests"]:
                urls = row.get("stray_api_urls") or []
                named = ", ".join(urls) if urls else "(urls not recorded)"
                failures.append(
                    f"{where} let {row['stray_api_requests']} /api/ requests reach the network "
                    f"during the measured actions; the timings include a round trip per "
                    f"request. Endpoints: {named}"
                )
            # Console output from inside a timed region is serialised over the debugging channel, so a warning the app
            # emits once per message would both cost time and grow like the signal.
            # The instrument for that is an ABSOLUTE cap, not a growth check: at 220 messages a per-message warning is
            # in the hundreds, while what an engine says about itself is a handful.
            # Firefox 153 emits exactly two "Scroll anchoring was disabled in a scroll container" notices once the
            # container is large enough, which is zero at 25K and two at both 100K and 300K
            # One console.error or one uncaught pageerror inside a measured interaction means the interaction did not do
            for phase, count, first in (
                ("seeding", row.get("seed_console_errors", 0), row.get("first_seed_error", "-")),
                (
                    "the measured actions",
                    row.get("console_errors", 0),
                    row.get("first_console_error", "-"),
                ),
            ):
                if count:
                    failures.append(
                        f"{where} logged {count} console error(s) or page error(s) during "
                        f"{phase}, the first being {first!r}; an application exception is not "
                        "engine chatter and the timings around it are not measurements"
                    )
            if row["console_warnings"] > CONSOLE_WARNING_ALLOWANCE:
                failures.append(
                    f"{where} logged {row['console_warnings']} console warnings during the "
                    f"measured actions, the first being {row['first_console_warning']!r}; that is "
                    f"more than the {CONSOLE_WARNING_ALLOWANCE} allowed for engine chatter, so "
                    "the timings include work this harness is charging to the app"
                )
            if plan["chars"] < size * 0.9:
                failures.append(
                    f"{where} only built {plan['chars']} characters of the {size} asked for"
                )
            if counts["messages"] < plan["messages"]:
                failures.append(
                    f"{where} rendered {counts['messages']} of {plan['messages']} messages; the "
                    "seed did not land"
                )
            # The fixture IS the measurement.
            # an image part whose data URL is not base64 PNG/JPEG/GIF/WebP is discarded with a console.warn
            for key, per_cycle in plan["expectedPerCycle"].items():
                want = per_cycle * plan["cycles"]
                if counts.get(key, 0) < want:
                    failures.append(
                        f"{where} rendered {counts.get(key, 0)} {key}, short of the {want} its "
                        f"{plan['cycles']} content cycles should produce; the fixture is not the "
                        "heavy thread this harness claims to measure"
                    )
            if counts.get("highlightedTokens", 0) <= 0:
                failures.append(f"{where} highlighted nothing; Shiki never ran")
            # DEFERRAL IS EXPECTED. AN UNHIGHLIGHTED MOUNTED FENCE IS NOT.
            stuck = counts.get("unhighlightedMountedFences", 0)
            if stuck:
                failures.append(
                    f"{where} left {stuck} of {counts.get('fenceBlocks', 0)} code fences mounted "
                    f"but unhighlighted after settling, neither deferred nor coloured; the "
                    "thread had not finished building itself when it was measured"
                )
            viewport = row["viewport"]
            if viewport["scrollHeight"] <= viewport["clientHeight"]:
                failures.append(
                    f"{where} does not overflow its viewport; the scroll measures nothing"
                )

            actions = row["actions"]
            for name in ACTIONS:
                if not actions[name].get("ran"):
                    failures.append(f"{where} could not run the {name} action at all")
            # A null settle time is the settle loop giving up: the page never produced a calm window inside
            # SETTLE_TIMEOUT_MS.
            for name in ("scroll", "jump", "reopen"):
                settling = actions[name]
                if settling.get("ran") and settling.get("settleMs") is None:
                    failures.append(
                        f"{where} ran the {name} action but it never reached a settled state "
                        f"within {SETTLE_TIMEOUT_MS}ms, so its settle time and the frame counts "
                        "beside it are the timeout rather than a measurement"
                    )
            keystroke = actions["keystroke"]
            if keystroke.get("ran"):
                # The DOM value is what the harness itself wrote, so it proves nothing on its own.
                if keystroke["runtimeText"] != keystroke["domText"]:
                    failures.append(
                        f"{where} typed {keystroke['domText']!r} into the DOM but the runtime "
                        f"holds {keystroke['runtimeText']!r}; the keystroke never reached the "
                        "composer state"
                    )
            # Sitting on the paint floor is NOT a harness failure here, and the reason is a finding rather than an
            # excuse: the character reaches the composer and paints on the very next frame at every size, while the
            # thread churns for another 180ms afterwards.
            scroll = actions["scroll"]
            # Equal travel at every size or the columns are not the same gesture.
            if scroll.get("ran") and scroll["scrolledPx"] < SCROLL_STEPS * SCROLL_STEP_PX * 0.9:
                failures.append(
                    f"{where} travelled only {scroll['scrolledPx']}px of the "
                    f"{SCROLL_STEPS * SCROLL_STEP_PX}px gesture, so its scroll column is not "
                    "comparable with the others"
                )
            jumped = actions["jump"]
            if jumped.get("ran"):
                # Unlike the gesture, the jump is DELIBERATELY not the same distance at every size:
                if jumped["landedAt"] > 1:
                    failures.append(
                        f"{where} jumped to the top of the thread and landed at "
                        f"{jumped['landedAt']}px; the viewport did not move"
                    )
                if jumped["travelledPx"] <= viewport["clientHeight"]:
                    failures.append(
                        f"{where} had only {jumped['travelledPx']}px to jump through, which is "
                        "less than one viewport; nothing had to be painted"
                    )
            menu = actions["menu"]
            if menu.get("ran"):
                if menu["openMs"] is None:
                    failures.append(f"{where} never opened the message action menu")
                elif menu["closeMs"] is None:
                    failures.append(f"{where} opened the action menu and it never closed")
                elif menu["bodyPointerEventsAfterClose"] == "none":
                    failures.append(
                        f"{where} left the body on the modal layer after closing the menu"
                    )
                # An empty popover satisfies "the menu opened" and costs nothing to render.
                elif not menu["itemsWhileOpen"]:
                    failures.append(f"{where} opened an action menu with no items in it")
                if not menu["triggersWhileHovered"] and counts["actionBars"] <= 0:
                    failures.append(
                        f"{where} mounted no action bar at rest and none under the pointer either"
                    )
            deleted = actions["delete"]
            if deleted.get("ran"):
                if deleted["ms"] is None:
                    failures.append(f"{where} never deleted a message")
                elif deleted["after"] >= deleted["before"]:
                    failures.append(f"{where} clicked delete and the message count did not drop")
            reopened = actions["reopen"]
            if reopened.get("ran"):
                if reopened["ms"] is None:
                    failures.append(f"{where} re-opened the thread and it never came back")
                elif reopened["closedMs"] is None:
                    failures.append(
                        f"{where} never saw the thread unmount, so its re-open time is the cost "
                        "of a thread that never left"
                    )

        # A modal menu puts the body on the modal layer and a non-modal one does not, and the two cost wildly different
        layers = {
            results["by_engine"][engine]["by_size"][str(size)]
            .get("actions", {})
            .get("menu", {})
            .get("bodyPointerEvents")
            for size in results["sizes"]
            if "crashed" not in results["by_engine"][engine]["by_size"][str(size)]
        }
        if len(layers) > 1:
            failures.append(
                f"on {engine} the menu put the body on {sorted(str(x) for x in layers)} across "
                "sizes; the columns are not measuring the same mechanism"
            )

    # does is not reporting a flat curve, it is reporting that it never drove the page.
    # Discrimination. Not a budget:
    if len(results["sizes"]) >= 2:
        for engine, per_axis in report.items():
            if not any(row["discriminated"] for row in per_axis.values()):
                failures.append(
                    f"on {engine} no measured axis rose by more than {DISCRIMINATION_RATIO}x "
                    f"from {results['sizes'][0]} to {results['sizes'][-1]} characters. Either the "
                    "page was never driven or every action is being measured somewhere it does "
                    "not run; the numbers above cannot size any change."
                )
    return failures


def main() -> int:
    vite = None
    if OWNS_SERVER:
        info(f"starting vite dev server on port {PORT}")
        vite = start_vite(PORT)
    try:
        wait_for_smoke_page(
            f"{BASE}/smoke-heavy-thread.html",
            "smoke-heavy-thread-main.tsx",
            proc = vite,
            info = info,
        )
        results = run()
    finally:
        if vite is not None:
            stop_process(vite)
            info("vite stopped")

    report = report_growth(results)
    results["growth"] = report
    out = OUT / f"{LABEL}.json"
    out.write_text(json.dumps(results, indent = 2), encoding = "utf-8")
    print_table(results)
    print_growth(results, report)
    info(f"wrote {out}")

    failures = harness_failures(results, report)
    for problem in failures:
        info(f"HARNESS-BROKEN {problem}")
    if failures:
        return 1
    info("measurement only: no budgets are asserted here, so this exits 0 on any timing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
