# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""What a long THINKING block costs while it streams, sampled the way the field trace was.

WHY THIS FILE EXISTS

tests/studio/playwright_heavy_thread.py measures a heavy thread, and its fixture holds text,
code fences, tool calls, artifacts and images. It holds no reasoning content, and it cannot
usefully hold any: it seeds a FINISHED thread, a finished reasoning group resolves to closed
(features/chat/utils/reasoning-visibility.ts), and Radix unmounts a closed CollapsibleContent.
So a seeded reasoning part contributes zero DOM and zero cost. Every bottleneck number taken on
that harness was taken on a thread with no thinking block in it.

The field report this file answers is a single long generation on Unsloth Desktop (Arch,
Wayland, WebKitGTK 2.52.5), sampled every 5 seconds:

    t=5s     fps 59.4   reasoningChars  2,256   reasoningCodeSpans      0   elements    578
    t=130s   fps 39.7   reasoningChars 45,943   reasoningCodeSpans  3,520   elements  4,755
    t=216s   fps 20.5   reasoningChars 73,178   reasoningCodeSpans 11,875   elements 13,688
    t=241s   fps 18.2   reasoningChars 80,434   reasoningCodeSpans 14,433   elements 16,513
    t=271s   fps 28.3   reasoningChars 90,262   reasoningCodeSpans 16,186   elements 18,536
    t=276s   COMPLETED, reasoningCodeSpans 0, elements 621, fps recovers

Three things that trace establishes, which this harness therefore does not re-derive:

  * It is not the forced software renderer. WEBKIT_DMABUF_RENDERER_FORCE_SHM=1 would be
    uniformly slow from t=0. This starts at a clean 59.4 fps and degrades with content.
  * `reasoningCodeSpans` counts `pre code span` inside `.aui-reasoning-text`, so those 16,186
    highlight spans are inside the THINKING pane, not the answer.
  * The collapse at completion is the proof of location: 18,536 elements become 621 and fps
    recovers, in the same second, without the thread being unloaded.

TWO PRECONDITIONS. GET EITHER WRONG AND THIS FILE MEASURES ALMOST NOTHING.

1. THE ARRIVAL RATE MUST BE THE USER'S. The capture delivered ~325 characters a second, which at
   24 characters a chunk is one chunk every 73ms, and that is the default here. The first version
   of this file used a 2ms gap, about four times a real token rate. At 2ms the renderer is the
   bottleneck from the very first chunk, so the run opens at 45 fps instead of 60 and there is no
   healthy early baseline left to degrade FROM: it measured a 1.19x spread where the capture shows
   3x. The whole shape being reproduced is "idle between chunks at the start, no longer idle at the
   end", and a fixture that is saturated at t=0 cannot show it.

2. ON AN IDLE MACHINE, FRAME RATE IS NOT SENSITIVE ENOUGH. Measured on the development box: 65%
   main-thread IDLE at 15,000 highlight spans. The work per chunk really does grow with the
   content, but there is so much headroom that it never crosses a frame budget, so fps sits pinned
   at 60 for the whole run and reading fps alone calls it "no effect". The metric that moves is
   `busy%`: main-thread blocked time per window, measured against a CALIBRATED timer clamp. To see
   the frame-rate collapse itself rather than the cost that causes it, throttle:
   SMOKE_RP_CPU_THROTTLE=6 on Chromium reproduced 47.1 fps -> 9.2 fps on this fixture.

WHY THIS CANNOT BE A SEEDED THREAD (the trap that makes this file necessary)

playwright_heavy_thread.py seeds a FINISHED thread through `thread.import`. Adding reasoning
content to that fixture looks like the obvious way to cover this path and it would have shown
NOTHING, convincingly: a finished reasoning group resolves to closed
(features/chat/utils/reasoning-visibility.ts -> `resolveReasoningOpen` returns false when
`isStreaming` is false and the user has not opened it by hand), Radix unmounts a closed
`CollapsibleContent`, and the reasoning part therefore contributes ZERO DOM and zero cost. The
cost exists only while the block streams. That is why this file streams.

WHAT IS MEASURED, AND WHY NOT longTasks

The trace's `longTasks` column read 0 in every sample. That is not an absence of jank: WebKit
never shipped the Long Tasks API, so a `longtask` PerformanceObserver is accepted and then never
fires. Support is read from `PerformanceObserver.supportedEntryTypes`, never from whether
observe() throws, and the portable signal is EVENT LOOP LAG: the largest gap between ticks of a
1ms setTimeout loop. The main thread cannot answer the timer while it is busy, so the gap is the
block. Resolution is the timer clamp, about 4ms, far below anything a user feels.

fps is counted from requestAnimationFrame ticks in the sample window, which is what the field
sampler did.

SAMPLING RUNS IN THE PAGE, NOT IN THE DRIVER

`document.querySelectorAll(".aui-reasoning-text pre code span")` over 16,000 spans is not free,
and a driver that pays a CDP round trip for it every window would put its own cost inside the
number. The sampler is a setInterval in the page, it records its own cost as `sample_ms`, and
that column is printed so a reader can check that the measurement is not the measured.

INSTRUMENTATION (optional, SMOKE_RP_INSTRUMENT=1)

ReasoningText installs a MutationObserver over its whole growing subtree and answers every
mutation with `el.scrollTop = el.scrollHeight` (studio/frontend/src/components/assistant-ui/
reasoning.tsx). Reading scrollHeight forces synchronous layout. To measure that directly rather
than infer it, the optional instrumentation wraps the `scrollHeight` getter on Element.prototype
and counts reads and the wall time inside them, split by whether the element is the reasoning
pane. It also counts MutationObserver callback invocations.

That wrapper perturbs: it adds two performance.now() calls per read. So it is OFF by default,
and the honest way to use it is to run the same size with and without and check the uninstru-
mented run degrades on its own. The driver prints which mode it ran in.

Run:
    python tests/studio/playwright_reasoning_pane.py
    SMOKE_RP_CHARS=30000,90000 SMOKE_RP_ENGINES=chromium python tests/studio/playwright_reasoning_pane.py
    SMOKE_RP_INSTRUMENT=1 SMOKE_RP_ENGINES=chromium python tests/studio/playwright_reasoning_pane.py

It starts and stops its own vite dev server. Point it at one you already have with
SMOKE_BASE_URL, or move the port with SMOKE_PORT.

RUN ONE ENGINE PER INVOCATION IN CI, UNDER AN EXTERNAL TIMEOUT, for the reason
playwright_heavy_thread.py's docstring gives at length: Playwright's sync API blocks inside a
greenlet, so no in-process timeout can bound a wedged engine and only the process bound works.

THIS HARNESS MEASURES, IT DOES NOT GATE on timings. It exits non-zero only when it is not
measuring what it claims: the fixture did not reach the trace's span count, the reasoning pane
never opened, the stream did not finish, or the completion collapse did not happen.

The dev server is a limitation to read the numbers against: React runs in development mode,
nothing is minified, vite serves unbundled modules. Absolute fps is therefore lower than a
packaged Studio's. The SHAPE across the run, and the ratio between the start and the end of the
same run, are what this file is for.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

PORT = int(os.environ.get("SMOKE_PORT", "5219"))
_EXTERNAL = os.environ.get("SMOKE_BASE_URL", "").strip().rstrip("/")
BASE = _EXTERNAL or f"http://127.0.0.1:{PORT}"
OWNS_SERVER = not _EXTERNAL
LABEL = os.environ.get("SMOKE_LABEL", "tree")
OUT = Path(os.environ.get("PW_ART_DIR", "logs/playwright-reasoning-pane"))
OUT.mkdir(parents = True, exist_ok = True)

# Characters of REASONING content. The largest is the trace's own 90,000.
SIZES = sorted(
    int(n) for n in os.environ.get("SMOKE_RP_CHARS", "22500,45000,90000").split(",")
)
ENGINES = [
    e.strip()
    for e in os.environ.get("SMOKE_RP_ENGINES", "chromium,webkit").split(",")
    if e.strip()
]
# Characters per fence. This is the span DENSITY knob: the trace's 90,262 characters carried
# 16,186 spans, i.e. 5.6 characters per span, which only happens if the content is nearly all
# fenced code. The driver reports the density it achieved so this can be tuned against the trace
# instead of guessed.
FENCE_CHARS = int(os.environ.get("SMOKE_RP_FENCE_CHARS", "1800"))
# Prose between the fences. Prose is characters without spans, so this is the other half of the
# density calibration. 1,250 against a 1,800-character fence was measured to land on the trace's
# ~5.6 characters per span; an all-fence fixture measured 3.47, i.e. half again as much work per
# character as the user's own content.
PROSE_CHARS = int(os.environ.get("SMOKE_RP_PROSE_CHARS", "1250"))
# Characters of fence-free thinking before the first code fence. The field capture held a flat
# 60.0 fps and exactly 0 highlight spans for 33,348 characters, and started losing frames at the
# first CLOSED fence, so a fixture without this stretch cannot show where the onset is. Scaled to
# the same quarter of the body by the driver when the size is not the capture's own.
PREAMBLE_CHARS = int(os.environ.get("SMOKE_RP_PREAMBLE_CHARS", "0"))
CHUNK_CHARS = int(os.environ.get("SMOKE_RP_CHUNK_CHARS", "24"))
# THE ARRIVAL RATE IS PART OF THE FIXTURE, and getting it wrong is how the first version of this
# file failed to reproduce anything.
#
# The trace delivered 90,262 characters in 276 seconds: 327 characters a second, which at 24
# characters a chunk is one chunk every 73ms. The first run here used a 2ms gap, i.e. about four
# times the user's token rate, and it measured a 1.19x spread on Chromium instead of the trace's
# 3.3x. The reason is not subtle: at 2ms the renderer is the bottleneck from the very first
# chunk, so the run opens at 45 fps rather than at 59 and there is no healthy early baseline left
# to degrade FROM. The trace's whole shape is "idle between chunks at the start, no longer idle
# at the end", and a fixture that is saturated at t=0 cannot show it.
#
# So the default is the user's own rate. It makes a 90,000-character run take about as long as
# the user's did, which is the point.
GAP_MS = int(os.environ.get("SMOKE_RP_GAP_MS", "73"))
SAMPLE_MS = int(os.environ.get("SMOKE_RP_SAMPLE_MS", "1000"))
RUN_TIMEOUT_MS = int(os.environ.get("SMOKE_RP_TIMEOUT_MS", "900000"))
# How long the completion collapse is allowed to take before the harness calls it a failure to
# collapse rather than a slow one.
COLLAPSE_TIMEOUT_MS = int(os.environ.get("SMOKE_RP_COLLAPSE_TIMEOUT_MS", "60000"))
INSTRUMENT = os.environ.get("SMOKE_RP_INSTRUMENT", "0") == "1"
# Whether to require that the DOM ends up holding the capture's highlight spans.
#
# On by default, because on an unmodified tree a run that does not build them is measuring the
# wrong content. It has to be switchable, though, and the reason is not hypothetical: a candidate
# fix that bounds what the pane keeps mounted ends the run with 511 spans instead of 16,000, and
# this check failed it as "the fixture is not the capture's content" when the fixture was exactly
# right and the FIX was what removed the spans. What the fixture owes is that it SENT the content;
# what the DOM does with it is the measurement.
EXPECT_SPANS = os.environ.get("SMOKE_RP_EXPECT_SPANS", "1") == "1"
# Headed, under a real display, is not a nicety here.
#
# Headless has NO COMPOSITOR. Everything this file measures about layout and script is the same
# either way, but the cost of PAINTING a large composited layer and getting it to a display is
# not measured at all in headless, and on the platform the field trace came from that path is
# deliberately the slow one: studio/src-tauri/src/linux_webkit.rs sets
# WEBKIT_DMABUF_RENDERER_FORCE_SHM=1, which forces WebKitGTK's software transport under Wayland.
# So a headless run that shows no degradation has NOT ruled the renderer out; it has not looked.
#
#     SMOKE_RP_HEADLESS=0 WEBKIT_DMABUF_RENDERER_FORCE_SHM=1 \
#       xvfb-run -s "-screen 0 1280x900x24" python tests/studio/playwright_reasoning_pane.py
HEADLESS = os.environ.get("SMOKE_RP_HEADLESS", "1") != "0"
# CHROMIUM ONLY, and off by default, because it is CDP and switching it on makes the chromium
# column incomparable with the webkit one.
#
# It is here because the first paced runs found 65% main-thread IDLE at 15,000 highlight spans on
# this box: the work per chunk really does grow with the content, but there is so much headroom
# that it never crosses a frame budget, so fps stays pinned at 60. The field trace came from a
# machine that was also running the generation. Throttling asks the question that separates "this
# work does not grow" from "this box is too fast to notice": at 4x or 6x, does the SAME curve
# cross the budget and produce the trace's shape?
CPU_THROTTLE = float(os.environ.get("SMOKE_RP_CPU_THROTTLE", "1"))

# The trace, for the table's comparison column. Not a budget: different machine, different
# engine, different build.
TRACE = [
    {"t": 5, "fps": 59.4, "chars": 2256, "spans": 0, "elements": 578},
    {"t": 130, "fps": 39.7, "chars": 45943, "spans": 3520, "elements": 4755},
    {"t": 216, "fps": 20.5, "chars": 73178, "spans": 11875, "elements": 13688},
    {"t": 241, "fps": 18.2, "chars": 80434, "spans": 14433, "elements": 16513},
    {"t": 271, "fps": 28.3, "chars": 90262, "spans": 16186, "elements": 18536},
]


def info(message: str) -> None:
    print(f"[reasoning-pane] {message}", flush = True)


# Installed before anything else runs. Counts frames and event loop lag continuously, and lets
# the page sampler close a window and read the totals since the last one.
#
# The rAF wrapper COUNTS, it does not pump, so fps stays the page's own frame rate.
RECORDER_INIT = """
(() => {
  const nativeRaf = window.requestAnimationFrame.bind(window);
  window.__nativeRaf = nativeRaf;

  const R = {
    frames: 0,
    frameGaps: [],
    maxLagMs: 0,
    lagTicks: 0,
    lagSumMs: 0,
    blockedMs: 0,
    clampMs: 4.0,
    longTaskSupported: Boolean(
      (PerformanceObserver.supportedEntryTypes || []).includes("longtask"),
    ),
    longTasks: 0,
    longTaskMs: 0,
  };
  window.__rp = R;

  // ONE self-rescheduling rAF loop is the frame counter, and requestAnimationFrame is NOT
  // wrapped. A wrapper that increments per callback counts the page's frame once for the loop
  // and once more for every rAF the app happened to schedule in that frame, which is a frame
  // rate that rises with how busy the app is -- the first version of this file reported 888 fps
  // on a 60Hz page for exactly that reason.
  let lastFrame = performance.now();
  const frame = () => {
    const now = performance.now();
    R.frames += 1;
    R.frameGaps.push(now - lastFrame);
    lastFrame = now;
    nativeRaf(frame);
  };
  nativeRaf(frame);

  // 1ms setTimeout, not a MessageChannel ping-pong: the MessageChannel version ticks ~150k/s and
  // halves Firefox's frame rate before any app code runs. This ticks ~150/s and costs nothing on
  // any engine. The clamp puts a ~4ms floor under the resolution, far below a felt stall.
  // The clamp is CALIBRATED, not assumed. `setTimeout(fn, 1)` is clamped to about 4ms by spec
  // but the actual floor differs by engine and by build, and the blocked-time figure below is a
  // subtraction against it, so a guessed 4.0 would show phantom block on one engine and hide
  // real block on another. The first 60 gaps of an idle page ARE the floor; the median of them
  // is taken as the clamp and reported, so a reader can see what was subtracted.
  const CALIBRATION_TICKS = 60;
  const calibration = [];
  let lastTick = performance.now();
  const tick = () => {
    const now = performance.now();
    const gap = now - lastTick;
    lastTick = now;
    if (calibration.length < CALIBRATION_TICKS) {
      calibration.push(gap);
      if (calibration.length === CALIBRATION_TICKS) {
        const sorted = calibration.slice().sort((a, b) => a - b);
        R.clampMs = sorted[Math.floor(sorted.length / 2)];
      }
    } else {
      R.lagTicks += 1;
      R.lagSumMs += gap;
      // Everything above the clamp is time the main thread could not answer a timer it had
      // already scheduled, i.e. time it was busy. Summed over a window this is a far more
      // sensitive signal than frame rate: a page with 65% idle still paints every frame on time,
      // so fps stays pinned at 60 while the work per chunk quietly triples. That is exactly the
      // regime this harness first landed in, and reading fps alone would have called it "no
      // effect".
      R.blockedMs += Math.max(0, gap - R.clampMs);
      if (gap > R.maxLagMs) R.maxLagMs = gap;
    }
    setTimeout(tick, 1);
  };
  setTimeout(tick, 1);

  // Recorded as a cross-check and never as the headline. Chromium alone lists `longtask` in
  // supportedEntryTypes; on WebKit and Gecko observe() is accepted and never fires, which reads
  // as "no jank" rather than as "no measurement". That is exactly the trap the field trace's
  // all-zero longTasks column fell into.
  if (R.longTaskSupported) {
    try {
      new PerformanceObserver((list) => {
        for (const e of list.getEntries()) {
          R.longTasks += 1;
          R.longTaskMs += e.duration;
        }
      }).observe({ type: "longtask", buffered: false });
    } catch (e) {
      R.longTaskSupported = false;
    }
  }

  window.__rpWindow = () => {
    // Frame TIME as well as frame rate. Headless Chromium has no vsync, so it runs the rAF loop
    // as fast as it can and an unloaded page reports far more than 60 fps; the absolute number
    // is then not a user-visible frame rate at all. The ratio within one run still is, and p95
    // frame time is the number that stays meaningful on an engine with no display.
    const gaps = R.frameGaps.slice().sort((a, b) => a - b);
    const at = (q) => (gaps.length === 0 ? null : gaps[Math.min(gaps.length - 1,
      Math.floor(gaps.length * q))]);
    const out = {
      frames: R.frames,
      p50FrameMs: at(0.5),
      p95FrameMs: at(0.95),
      maxFrameMs: gaps.length === 0 ? null : gaps[gaps.length - 1],
      maxLagMs: R.maxLagMs,
      lagTicks: R.lagTicks,
      lagSumMs: R.lagSumMs,
      blockedMs: R.blockedMs,
      clampMs: R.clampMs,
      longTasks: R.longTasks,
      longTaskMs: R.longTaskMs,
    };
    R.frames = 0;
    R.frameGaps = [];
    R.maxLagMs = 0;
    R.lagTicks = 0;
    R.lagSumMs = 0;
    R.blockedMs = 0;
    R.longTasks = 0;
    R.longTaskMs = 0;
    return out;
  };
})();
"""

# Optional and off by default, because it perturbs what it measures.
#
# ReasoningText answers every mutation of its subtree with `el.scrollTop = el.scrollHeight`.
# Reading scrollHeight forces synchronous layout of everything above it. This counts those reads
# and times them, split by whether the element IS the reasoning pane, so "the autoscroll observer
# forces a full layout per chunk" becomes a measured number rather than a reading of the source.
INSTRUMENT_INIT = """
(() => {
  const I = {
    scrollHeightReads: 0,
    scrollHeightMs: 0,
    reasoningReads: 0,
    reasoningMs: 0,
    scrollTopWrites: 0,
    reasoningScrollTopWrites: 0,
    mutationCallbacks: 0,
    mutationRecords: 0,
  };
  window.__rpInstr = I;

  const isPane = (el) => {
    try {
      return Boolean(el && el.classList && el.classList.contains("aui-reasoning-text"));
    } catch (e) {
      return false;
    }
  };

  const sh = Object.getOwnPropertyDescriptor(Element.prototype, "scrollHeight");
  if (sh && sh.get) {
    Object.defineProperty(Element.prototype, "scrollHeight", {
      configurable: true,
      enumerable: sh.enumerable,
      get() {
        const t0 = performance.now();
        const v = sh.get.call(this);
        const dt = performance.now() - t0;
        I.scrollHeightReads += 1;
        I.scrollHeightMs += dt;
        if (isPane(this)) {
          I.reasoningReads += 1;
          I.reasoningMs += dt;
        }
        return v;
      },
    });
  }

  const st = Object.getOwnPropertyDescriptor(Element.prototype, "scrollTop");
  if (st && st.set) {
    Object.defineProperty(Element.prototype, "scrollTop", {
      configurable: true,
      enumerable: st.enumerable,
      get() {
        return st.get.call(this);
      },
      set(value) {
        I.scrollTopWrites += 1;
        if (isPane(this)) I.reasoningScrollTopWrites += 1;
        st.set.call(this, value);
      },
    });
  }

  const NativeMO = window.MutationObserver;
  if (NativeMO) {
    window.MutationObserver = function (callback) {
      return new NativeMO((records, observer) => {
        I.mutationCallbacks += 1;
        I.mutationRecords += records.length;
        return callback(records, observer);
      });
    };
    window.MutationObserver.prototype = NativeMO.prototype;
  }

  window.__rpInstrWindow = () => {
    const out = { ...I };
    for (const k of Object.keys(I)) I[k] = 0;
    return out;
  };
})();
"""

# The in-page sampler. One setInterval, the trace's columns, and its own cost recorded so a
# reader can rule the measurement out as the cause.
SAMPLER = """
(({ sampleMs }) => {
  const rows = [];
  window.__rpRows = rows;
  const t0 = performance.now();
  let lastAt = t0;
  let seenArrivals = 0;
  const id = setInterval(() => {
    const at = performance.now();
    const w = window.__rpWindow();
    const s = window.__reasoningPane.sample();
    const instr = window.__rpInstrWindow ? window.__rpInstrWindow() : null;
    const elapsed = at - lastAt;
    lastAt = at;
    const lastArrivals = seenArrivals;
    seenArrivals = s.arrivals;
    rows.push({
      t_ms: Math.round(at - t0),
      fps: Math.round((w.frames / (elapsed / 1000)) * 10) / 10,
      p50_frame_ms: w.p50FrameMs === null ? null : Math.round(w.p50FrameMs * 10) / 10,
      p95_frame_ms: w.p95FrameMs === null ? null : Math.round(w.p95FrameMs * 10) / 10,
      max_frame_ms: w.maxFrameMs === null ? null : Math.round(w.maxFrameMs * 10) / 10,
      max_lag_ms: Math.round(w.maxLagMs * 10) / 10,
      // Percent of the window the main thread was busy, from the blocked time above. This is the
      // column that moves when fps does not.
      busy_pct: Math.round((w.blockedMs / elapsed) * 1000) / 10,
      clamp_ms: Math.round(w.clampMs * 100) / 100,
      mean_lag_ms:
        w.lagTicks === 0 ? null : Math.round((w.lagSumMs / w.lagTicks) * 10) / 10,
      long_tasks: w.longTasks,
      long_task_ms: Math.round(w.longTaskMs),
      reasoning_chars: s.reasoningChars,
      reasoning_spans: s.reasoningCodeSpans,
      reasoning_elements: s.reasoningElements,
      outside_reasoning: s.elementsOutsideReasoning,
      all_spans: s.allCodeSpans,
      elements: s.totalElements,
      panes: s.reasoningPanes,
      open: s.reasoningOpen,
      sent_chars: s.sentChars,
      arrivals: s.arrivals,
      // Arrivals actually delivered in this window against the arrivals the pacing asked for.
      // When the renderer stops keeping up, the generator's own `await sleep(gapMs)` starts
      // returning late and this falls below the target: the stream itself slows down, which is
      // the same thing the user sees as the reply crawling.
      arrivals_in_window: s.arrivals - lastArrivals,
      stream_done: s.streamDone,
      sample_ms: s.sampleCostMs,
      heap_mb:
        performance.memory
          ? Math.round(performance.memory.usedJSHeapSize / 1048576)
          : null,
      instr,
    });
  }, sampleMs);
  window.__rpStopSampler = () => clearInterval(id);
})
"""


def start_run(page, size: int) -> None:
    page.evaluate(
        SAMPLER,
        {"sampleMs": SAMPLE_MS},
    )
    page.evaluate(
        """(cfg) => window.__reasoningPane.run(cfg)""",
        {
            "totalChars": size,
            "fenceChars": FENCE_CHARS,
            "proseChars": PROSE_CHARS,
            "preambleChars": PREAMBLE_CHARS or int(size * 0.25),
            "chunkChars": CHUNK_CHARS,
            "gapMs": GAP_MS,
        },
    )


def wait_for_stream(page, timeout_ms: int) -> bool:
    """True if the stream finished. Polls the cheap bookkeeping, never the DOM counts."""
    try:
        page.wait_for_function(
            """() => window.__reasoningPane.streamState().done === true""",
            timeout = timeout_ms,
        )
        return True
    except Exception:
        return False


def measure_cell(context, engine: str, size: int) -> dict:
    page = context.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))

    # startswith, not `"/api/" in url`: vite serves the app's own source modules from paths that
    # contain the substring, and aborting those would blank the page.
    page.route(
        "**/*",
        lambda route: (
            route.fulfill(status = 200, content_type = "application/json", body = "{}")
            if route.request.url.startswith(f"{BASE}/api/")
            else route.continue_()
        ),
    )

    page.set_viewport_size({"width": 1280, "height": 900})
    if CPU_THROTTLE > 1 and engine == "chromium":
        throttle_cdp = page.context.new_cdp_session(page)
        throttle_cdp.send("Emulation.setCPUThrottlingRate", {"rate": CPU_THROTTLE})
    page.goto(f"{BASE}/smoke-reasoning-pane.html", wait_until = "domcontentloaded")
    page.wait_for_function(
        """() => Boolean(window.__reasoningPane)""", timeout = 120_000
    )

    start_run(page, size)
    finished = wait_for_stream(page, RUN_TIMEOUT_MS)

    # The completion row, and how long it took to arrive.
    #
    # Not a fixed sleep. The collapse animation is 200ms and `retainStreamingHeight` releases on
    # the same timer, so 2s looks like plenty -- and at 90,000 characters on WebKit it was not:
    # the run reported 15,099 reasoning spans still mounted 2s after the group had already gone
    # to `open=False`, and the harness failed itself for a collapse that had merely not finished
    # yet. Unmounting ~16,000 nodes is itself a main-thread job that scales with the content, and
    # how long it takes is one of the numbers worth having: the trace's own worst event loop lag,
    # 576ms, is at completion.
    collapse_started = time.monotonic()
    try:
        page.wait_for_function(
            """() => document.querySelectorAll(".aui-reasoning-text pre code span").length === 0""",
            timeout = COLLAPSE_TIMEOUT_MS,
        )
        collapse_ms = round((time.monotonic() - collapse_started) * 1000)
    except Exception:
        collapse_ms = None
    after = page.evaluate("""() => window.__reasoningPane.sample()""")
    page.evaluate("""() => window.__rpStopSampler && window.__rpStopSampler()""")
    rows = page.evaluate("""() => window.__rpRows""")
    state = page.evaluate("""() => window.__reasoningPane.streamState()""")
    long_task_supported = page.evaluate("""() => window.__rp.longTaskSupported""")

    page.close()
    return {
        "engine": engine,
        "size": size,
        "finished": finished,
        "rows": rows,
        "after": after,
        "collapse_ms": collapse_ms,
        "state": state,
        "long_task_supported": long_task_supported,
        "errors": errors,
    }


def run() -> dict:
    results: dict[str, dict] = {}
    with sync_playwright() as p:
        for engine in ENGINES:
            launcher = getattr(p, engine, None)
            if launcher is None:
                info(f"{engine}: no such engine, skipped")
                continue
            kwargs: dict = {"headless": HEADLESS}
            if engine == "chromium":
                kwargs["args"] = chromium_launch_args()
            try:
                browser = launcher.launch(**kwargs)
            except Exception as exc:
                info(f"{engine}: launch failed ({exc}), skipped")
                continue
            context = browser.new_context()
            context.add_init_script(RECORDER_INIT)
            if INSTRUMENT:
                context.add_init_script(INSTRUMENT_INIT)
            cells: dict[str, dict] = {}
            for size in SIZES:
                info(f"{engine} @ {size:,} reasoning chars")
                try:
                    cells[str(size)] = measure_cell(context, engine, size)
                except Exception as exc:
                    info(f"{engine} @ {size}: {type(exc).__name__}: {exc}")
                    cells[str(size)] = {"engine": engine, "size": size, "error": str(exc)}
            results[engine] = cells
            context.close()
            browser.close()
    return results


def print_run(cell: dict) -> None:
    rows = cell.get("rows") or []
    if not rows:
        info(f"  {cell['engine']} @ {cell['size']}: no samples")
        return
    instrumented = any(r.get("instr") for r in rows)
    head = (
        f"{'t(s)':>6} {'busy%':>6} {'fps':>7} {'p50fr':>6} {'p95fr':>7} {'maxLag':>8} "
        f"{'arr/s':>6} {'sent':>8} {'chars':>8} {'spans':>8} {'elements':>9} "
        f"{'heapMB':>7} {'smpl(ms)':>9}"
    )
    if instrumented:
        head += f" {'shReads':>9} {'shMs':>8} {'paneRd':>8} {'paneMs':>8} {'moCB':>7}"
    print(head, flush = True)
    for r in rows:
        line = (
            f"{r['t_ms'] / 1000:6.1f} {r.get('busy_pct', -1):6.1f} {r['fps']:7.1f} "
            f"{(r['p50_frame_ms'] if r['p50_frame_ms'] is not None else -1):6.1f} "
            f"{(r['p95_frame_ms'] if r['p95_frame_ms'] is not None else -1):7.1f} "
            f"{r['max_lag_ms']:8.1f} "
            f"{r.get('arrivals_in_window', -1) * 1000 / max(1, SAMPLE_MS):6.1f} "
            f"{r['sent_chars']:8,d} "
            f"{r['reasoning_chars']:8,d} {r['reasoning_spans']:8,d} "
            f"{r.get('reasoning_elements', -1):8,d} "
            f"{r['elements']:9,d} {r.get('outside_reasoning', -1):8,d} "
            f"{r['sample_ms']:9.2f}"
        )
        i = r.get("instr")
        if instrumented and i:
            line += (
                f" {i['scrollHeightReads']:9,d} {i['scrollHeightMs']:8.1f} "
                f"{i['reasoningReads']:8,d} {i['reasoningMs']:8.1f} "
                f"{i['mutationCallbacks']:7,d}"
            )
        print(line, flush = True)
    a = cell["after"]
    c = cell.get("collapse_ms")
    print(
        f"  collapse took {c}ms" if c is not None
        else "  collapse did NOT complete within the timeout",
        flush = True,
    )
    print(
        f"  after completion: chars {a['reasoningChars']:,} spans "
        f"{a['reasoningCodeSpans']:,} elements {a['totalElements']:,} "
        f"open={a['reasoningOpen']}",
        flush = True,
    )


def summarise(cell: dict) -> dict:
    """First tenth of the run against the last tenth, on the same run.

    Not first sample against last sample: a single sample is one second of a loaded machine. Not
    across sizes either, because the trace is a curve WITHIN one generation, and reproducing it
    means showing the same run getting slower as its own content grows.
    """
    # The FIRST sample is dropped, always. It spans the click that starts the run, the runtime's
    # first publish and the first Shiki highlighter load, none of which are the growth this file
    # is about; keeping it reported a 229ms "stall" at 485 characters of content and made the
    # early window look worse than the late one.
    rows = [r for r in (cell.get("rows") or []) if not r.get("stream_done")][1:]
    if len(rows) < 6:
        return {}
    n = max(2, len(rows) // 10)
    early = rows[:n]
    late = rows[-n:]

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs)

    early_fps = mean([r["fps"] for r in early])
    late_fps = mean([r["fps"] for r in late])
    early_busy = mean([r.get("busy_pct", 0) for r in early])
    late_busy = mean([r.get("busy_pct", 0) for r in late])
    early_p95 = mean([r["p95_frame_ms"] for r in early if r["p95_frame_ms"] is not None])
    late_p95 = mean([r["p95_frame_ms"] for r in late if r["p95_frame_ms"] is not None])
    peak = max(r["reasoning_spans"] for r in rows)
    peak_chars = max(r["reasoning_chars"] for r in rows)
    peak_elements = max(r["elements"] for r in rows)
    return {
        "early_fps": round(early_fps, 1),
        "late_fps": round(late_fps, 1),
        "fps_ratio": round(early_fps / late_fps, 2) if late_fps else None,
        "early_busy_pct": round(early_busy, 1),
        "late_busy_pct": round(late_busy, 1),
        "busy_ratio": round(late_busy / early_busy, 2) if early_busy else None,
        "early_p95_frame_ms": round(early_p95, 1),
        "late_p95_frame_ms": round(late_p95, 1),
        "p95_ratio": round(late_p95 / early_p95, 2) if early_p95 else None,
        "early_max_lag_ms": round(max(r["max_lag_ms"] for r in early), 1),
        "late_max_lag_ms": round(max(r["max_lag_ms"] for r in late), 1),
        "peak_chars": peak_chars,
        "peak_spans": peak,
        "peak_elements": peak_elements,
        "chars_per_span": round(peak_chars / peak, 2) if peak else None,
        "after_spans": cell["after"]["reasoningCodeSpans"],
        "after_elements": cell["after"]["totalElements"],
        "collapsed": cell["after"]["reasoningCodeSpans"] == 0,
        "collapse_ms": cell.get("collapse_ms"),
        "peak_heap_mb": max(
            (r["heap_mb"] for r in rows if r["heap_mb"] is not None), default = None
        ),
        "final_heap_mb": cell["rows"][-1]["heap_mb"] if cell.get("rows") else None,
    }


def harness_failures(results: dict) -> list[str]:
    """Only the ways this file could be measuring nothing. Timings never fail a run."""
    bad: list[str] = []
    if not results:
        bad.append("no engine produced a cell")
    for engine, cells in results.items():
        for key, cell in cells.items():
            where = f"{engine} @ {key}"
            if "error" in cell:
                bad.append(f"{where}: {cell['error']}")
                continue
            if not cell.get("finished"):
                bad.append(f"{where}: the stream never finished")
            rows = cell.get("rows") or []
            if len(rows) < 5:
                bad.append(f"{where}: {len(rows)} samples, too few to be a curve")
                continue
            if not any(r["open"] for r in rows):
                bad.append(f"{where}: the reasoning group never opened, so nothing was measured")
            peak = max(r["reasoning_spans"] for r in rows)
            size = cell["size"]
            # What the fixture owes, always: it sent the content it was asked for. This is the
            # check that catches a fixture that quietly built 90,000 characters of prose, and it
            # holds whatever the renderer under test then does with them.
            sent = cell.get("state", {}).get("sentChars") or 0
            if sent < size * 0.99:
                bad.append(
                    f"{where}: the fixture only sent {sent:,} of {size:,} characters"
                )
            if max(r["reasoning_chars"] for r in rows) <= 0:
                bad.append(f"{where}: the reasoning pane never held any text")
            # And, on an unmodified tree, that the content really did become highlight spans. At
            # the capture's own size the floor is most of its 16,000; below that, scaled. Turn it
            # off with SMOKE_RP_EXPECT_SPANS=0 when measuring a change whose whole purpose is to
            # stop building them.
            floor = int(11_000 * size / 90_000)
            if EXPECT_SPANS and peak < floor:
                bad.append(
                    f"{where}: peak {peak:,} reasoning spans is below the floor {floor:,}; "
                    "either the fixture is not the capture's content, or the tree under test "
                    "stopped building spans and this run wanted SMOKE_RP_EXPECT_SPANS=0"
                )
            after = cell["after"]
            if after["reasoningCodeSpans"] != 0:
                bad.append(
                    f"{where}: {after['reasoningCodeSpans']:,} reasoning spans survived "
                    f"completion after {COLLAPSE_TIMEOUT_MS}ms; the collapse that localises the "
                    "cost did not happen"
                )
            real_errors = [e for e in cell.get("errors", []) if "ResizeObserver" not in e]
            if real_errors:
                bad.append(f"{where}: page error {real_errors[0]}")
    return bad


def main() -> int:
    vite = None
    try:
        if OWNS_SERVER:
            info(f"starting vite dev server on port {PORT}")
            vite = start_vite(PORT)
            wait_for_smoke_page(
                f"{BASE}/smoke-reasoning-pane.html",
                "smoke-reasoning-pane-main.tsx",
                proc = vite,
            )
        info(
            f"engines={','.join(ENGINES)} sizes={SIZES} fence={FENCE_CHARS} prose={PROSE_CHARS} "
            f"chunk={CHUNK_CHARS} gap={GAP_MS}ms sample={SAMPLE_MS}ms "
            f"instrument={'on' if INSTRUMENT else 'off'} "
            f"headless={'on' if HEADLESS else 'OFF (real compositor)'} "
            f"cpu_throttle={CPU_THROTTLE}x"
        )
        results = run()
    finally:
        if vite is not None:
            stop_process(vite)
            info("vite stopped")

    report: dict[str, dict] = {}
    for engine, cells in results.items():
        for key, cell in cells.items():
            if "error" in cell:
                continue
            print(f"\n=== {engine} @ {cell['size']:,} reasoning chars ===", flush = True)
            if not cell.get("long_task_supported"):
                print(
                    "  longtask is NOT supported on this engine: the long-task columns are "
                    "structurally zero and mean nothing. Read max lag.",
                    flush = True,
                )
            print_run(cell)
            s = summarise(cell)
            report[f"{engine}@{key}"] = s
            if s:
                print(
                    f"  main thread busy {s['early_busy_pct']}% -> {s['late_busy_pct']}% "
                    f"({s['busy_ratio']}x), "
                    f"early fps {s['early_fps']} -> late fps {s['late_fps']} "
                    f"({s['fps_ratio']}x slower), p95 frame {s['early_p95_frame_ms']}ms -> "
                    f"{s['late_p95_frame_ms']}ms ({s['p95_ratio']}x), "
                    f"max lag {s['early_max_lag_ms']}ms -> "
                    f"{s['late_max_lag_ms']}ms, peak {s['peak_spans']:,} spans over "
                    f"{s['peak_chars']:,} chars ({s['chars_per_span']} chars/span), "
                    f"elements {s['peak_elements']:,} -> {s['after_elements']:,}",
                    flush = True,
                )

    payload = {
        "label": LABEL,
        "sizes": SIZES,
        "engines": ENGINES,
        "fence_chars": FENCE_CHARS,
        "prose_chars": PROSE_CHARS,
        "chunk_chars": CHUNK_CHARS,
        "gap_ms": GAP_MS,
        "sample_ms": SAMPLE_MS,
        "instrumented": INSTRUMENT,
        "expect_spans": EXPECT_SPANS,
        "headless": HEADLESS,
        "cpu_throttle": CPU_THROTTLE,
        "trace": TRACE,
        "results": results,
        "report": report,
    }
    out = OUT / f"reasoning-pane-{LABEL}.json"
    out.write_text(json.dumps(payload, indent = 2))
    info(f"wrote {out}")

    bad = harness_failures(results)
    if bad:
        print("\nHARNESS FAILURES", flush = True)
        for b in bad:
            print(f"  - {b}", flush = True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
