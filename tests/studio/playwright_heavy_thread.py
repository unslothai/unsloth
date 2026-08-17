# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Where a HEAVY thread stalls, as a curve over how much content the thread holds.

Users report Studio and Desktop going sluggish "after long generations with any code cells
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
page under test (38 frames -> 14, median frame 17ms -> 34ms) before any Studio code ran. The
1ms setTimeout loop ticks ~150 times a second, costs nothing measurable on any of the three
engines, and reported the same 120ms synthetic stall on all three.

CDP counters (LayoutCount, RecalcStyleCount, LayoutDuration, RecalcStyleDuration, TaskDuration)
and the longtask observer are ALSO recorded, and every one of them is labelled CHROMIUM-ONLY in
the table. They attribute cost to layout versus script, which nothing portable does. They are
never the headline.

ONE PAGE PER ACTION, AND A CORRECTION TO WHAT THIS FILE PREVIOUSLY PUBLISHED

Until this change the harness drove the six actions in one fixed order on ONE page and then
repeated that whole sequence on the same page:

    settle gate, expandTools, keystroke, scroll, jump, menu, delete, reopen

Only the first action of the first repetition ever ran on a page that had done nothing else. Every
other cell in the published table carried the residue of everything before it, and the table did
not say so. For the scroll column at 300K characters that residue was the entire number. Measured
on chromium, a fresh page per arm, medians of 3:

    predecessor to an identical 20-step scroll  gesture ms  frames>33ms  long tasks  long task ms
    nothing                                          666.2            0           0             0
    keystroke                                        666.3            0           0             0
    leave and re-open the thread                     695.2            0           0             0
    hover one assistant message, no menu            1555.5           11          11          1148
    hover, then open and close its action menu      1453.8           11          11          1044

IT IS NOT THE MENU, and an earlier version of this very paragraph said it was. The action bar is
hover-revealed, so every menu arm had to hover the message before it could click "More": "menu"
was always "hover plus menu", and hover was never run on its own until it was. Run on its own it
reproduces the whole effect, and the menu adds nothing that can be told apart from noise -- on the
run above the hover-only arm reads slightly MORE expensive than hover-plus-menu, which is the tell
that there is no menu contribution to find.

IT IS NOT THE HOVER EITHER. What costs is the element under the pointer CHANGING while the
gesture runs. The same arms, one session, medians of 3, now also counting the boundary events the
gesture generates via a capture-phase listener installed around it:

    arm                                    gesture ms  frames>33  long tasks  lt ms  pointerover
    nothing                                     666.3          0           0      0            0
    hover a message, cursor left on it         1668.2         11          11   1267           20
    no hover, cursor on the scroller gutter      664.4          0           0      0            0
    hover a message, THEN move to the gutter     666.6          0           0      0            0

Twenty scroll steps, twenty pointerover events, twenty DISTINCT targets, eleven long tasks. One
boundary event per step, each landing on a different element, at about 63ms of main-thread React
work apiece on a 43,422-node thread. Every arm that generates zero boundary events sits on the
666ms floor. That is a mechanism rather than a correlation, and it is countable, which is why it
is counted here instead of argued.

A prior hover leaves NOTHING behind: hover-then-move-to-gutter is 666.6ms with zero long tasks and
zero boundary events. Nothing is armed and nothing persists, so no amount of cleanup after a hover
or a menu can help. The work is live.

TWO THINGS THIS IS NOT. It is not "the pointer left the thread": the scroller fills the viewport
in this fixture, and the helper that parks the cursor reports `pointer_outside: False` at every
candidate it tries, honestly, because there is no point on the page outside the scroller. The
corner works because it is over the scroller's own gutter, where the hit-test target is the same
element on every step. And it is not a reason to discount the regression. Wheel-scrolling with the
cursor resting over the conversation is how a person reads a long chat, and it is what this
harness does; the cheap arms are the artificial ones. The cost being live rather than sticky makes
it more fixable, not less.

Two candidate mechanisms were eliminated by construction rather than by argument, and both are
dead ends worth not repeating. Opening the menu leaves 112 `pointerdown` and 112 `pointermove`
listeners on `document`, one per assistant message; a thread remount clears them and the cost
survives. It also gives `document.body` a second React delegated-event root, 1 listener type to
85; routing every Radix portal into a container that is not `<body>` takes that back to 1 and the
cost is unchanged, 1543.6ms against 1527.0ms. Neither artefact is the cause, and with the pointer
result above neither could have been: a persistent artefact cannot explain a cost that disappears
the moment the cursor moves.

Traced over the gesture, the difference between a clean scroll and a post-hover scroll is script
27.8 -> 758.1 ms, against raster 113.3 -> 118.8 ms and paint 31.0 -> 45.2 ms. It is main-thread
JavaScript armed by the hover. It is not paint and it is not raster. The named callers are a React
commit at react-dom_client.js:9077 (36 calls across the 20 steps), `flushScheduled` in the
assistant-ui bundle, and `onLayoutChange` in use-intent-aware-autoscroll.tsx, with
`dispatchContinuousEvent` going from 61 calls to 458.

Absolute milliseconds drift between sessions on a loaded host -- the same post-hover arm has read
872ms, 1435ms and 1555ms on three different days -- so every claim above rests on arms run back to
back in one session, and on ratios rather than on levels.

The median is what buried it. The old harness's own 300K scroll repetitions were

    666.9 / 1049.7 / 949.9 / 916.7 / 950.1 ms

so reporting the median reported 949.9 and discarded the one uncontaminated repetition as though
it were the outlier.

Three things this file used to say are therefore withdrawn, in the open:

  * The attribution published for the scroll column -- per-pixel work, with the layout already
    done -- was WRONG. A 20-step scroll on a page that has only ever scrolled costs 666.2ms, which
    is 33.3ms a step, which is exactly the two-vsync floor this same file measures and prints as
    `paint floor ms`. The gesture is free. What was reported as the cost of scrolling a heavy
    thread was the cost of having hovered a message, collected one or more actions later.
  * #9047's null result needs no other explanation. It optimised paint and raster, and the split
    above puts paint and raster at 164ms of a 1435ms gesture. That bucket was never the binding
    one, so a change to it could not move the number.
  * The comment above JUMP_JS in this same file -- a wheel gesture "turns out to be nearly free at
    any size, because the layout was already done at mount" -- was RIGHT ALL ALONG, and the table
    contradicted it for months. The comment was not the thing that was wrong.

So there are now two tables, and they are labelled:

    ISOLATED    the headline. One fresh browser context, one fresh page and one fresh seed PER
                ACTION, then REPEATS repetitions of that action alone. A row's contract is "N
                repetitions of action X on a page that has only ever done X". This costs one seed
                per action instead of one per cell; that is the honest price and it is paid.
    CARRY-OVER  the old fixed sequence, unchanged, on one page: what the six actions cost after a
                session of use. A user really does open a menu and then scroll, so this is a real
                scenario worth measuring. It is simply not what the word "scroll" means on its
                own, so it no longer gets to be printed in a column called scroll.

Preconditions are kept, contamination is not. The Shiki settle gate, `expandTools` and the hover
that reveals the action bar before `menu` and `delete` all still run before the action they belong
to, on the isolated page as well: they are the state the action is defined against, not residue
from a different action.

READ THE `menu` AND `delete` ROWS WITH THAT IN MIND. Their precondition IS the hover, and the
hover is the thing that costs 890ms of the scroll column. So isolation does not make those two
rows hover-free and was never going to: a menu you have not hovered to reach is not a menu the
user can open. What isolation buys them is freedom from EACH OTHER and from scroll, jump and
reopen, which is real but is less than what it buys `scroll`, `keystroke` and `reopen`, whose
isolated pages never hover at all. A future reader looking for the hover cost in these numbers
should look at `scroll` isolated against `scroll` sequenced, which is the clean contrast, and not
at `menu`.

Measured on this tree after the restructuring, chromium, 300K characters, medians of 3, the two
tables now read:

    scroll gesture ms   isolated 666.6, carry-over 1467.3
    scroll frames >33ms isolated 0,     carry-over 9
    scroll long tasks   isolated 0,     carry-over 9, totalling 1040ms
    jump painted ms     isolated  33.4, carry-over   94.5

The isolated scroll is 20 steps at the 33.3ms paint floor this run measured, to within a
millisecond. There is nothing in it.

Alongside the carry-over table, every action prints repetition 1 against the median of repetitions
2..N. That one line is the whole defect made visible -- 666.9 against 949.9 on the 300K scroll --
and had it been printed from the start this would have been caught months earlier instead of being
published.

OPEN QUESTIONS THIS CHANGE DOES NOT CLOSE

  * Only scroll has been traced to a cause. Every action column except keystroke on repetition one
    was exposed to carry-over in the old arrangement, and only the scroll one has been taken apart
    far enough to name what the residue is made of. The first run of the restructured harness puts
    jump at 33.4ms isolated against 94.5ms sequenced at 300K, so it is exposed too and nobody has
    looked at why; delete reads 740.4 isolated against 568.5 sequenced, which is a difference in
    the other direction and is not understood either. reopen is untraced.
  * The repetition-1 line catches carry-over from the PREVIOUS REPETITION and nothing else. An
    action that sits at the same point inside every repetition is contaminated identically in all
    of them, so the line reads "steady" and only the isolated table beside it shows anything:
    `delete` always follows `menu`, and at 300K it reads 1.15x steady on the repetition line while
    the two tables disagree by 172ms. Read the two together. Neither is sufficient alone.
  * The menu column is the least affected of the six, as predicted -- it is expensive on its own
    terms, being the action doing the contaminating -- but only as a ratio: 4194 isolated against
    4730 sequenced at 300K is 1.13x, the smallest gap of the four columns that moved, and it is
    still 536ms.

ENGINES

    chromium   proxy for WebView2, i.e. Unsloth Desktop on Windows.
    webkit     proxy for WKWebView and WebKitGTK, i.e. Unsloth Desktop on macOS and Linux.
    firefox    control. Not shipped by anything here; it is in the table so that a number which
               moves on one engine only can be told apart from a number that moves everywhere.

Playwright's WebKit is a PROXY, not the webview Desktop embeds. It is Apple's WebKit built for
Playwright, driven headless, with no Tauri IPC layer and no WebKitGTK compositor. Read it as
"JavaScriptCore plus WebKit layout", not as "Unsloth Desktop on macOS".

Desktop Linux under Wayland is WORSE than any number this file can produce, and not by a little:
studio/src-tauri/src/linux_webkit.rs sets WEBKIT_DMABUF_RENDERER_FORCE_SHM=1, which forces the
software rendering transport. Everything below runs on a normal compositor path.

THIS HARNESS MEASURES, IT DOES NOT GATE. It prints the tables and exits 0 on any timing, including
a carry-over finding: a divergence between repetition 1 and the rest is a fact about the app, not
a broken harness. It exits non-zero only when the harness itself is broken: the seed did not land,
an element it drives went missing, or the curve did not rise with content -- which would mean it
is measuring nothing. Budgets belong in a later change, set from numbers taken on real hardware.

Run:
    python tests/studio/playwright_heavy_thread.py
    SMOKE_HEAVY_CHARS=100000,300000 SMOKE_HEAVY_ENGINES=chromium python tests/studio/playwright_heavy_thread.py
    SMOKE_HEAVY_TABLES=isolated python tests/studio/playwright_heavy_thread.py   # skip the sequence

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

Raise that bound if you are raising it from an older revision: a cell used to seed the thread once
and now seeds it seven times, once per isolated action plus once for the carry-over sequence, and
seeding 300K characters is the most expensive thing this file does. SMOKE_HEAVY_TABLES=isolated
buys most of it back if the carry-over table is not what you came for.

The dev server is deliberate and is a limitation to read the numbers against: React runs in
development mode, nothing is minified, and vite serves unbundled modules. Absolute milliseconds
are therefore higher than a packaged Studio's. The curve across sizes and the ranking across
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
# rstrip("/"): a trailing slash makes the anchored /api/ route regex below never match, silently
# turning the stubbed fan-out back into live HTTP.
_EXTERNAL = os.environ.get("SMOKE_BASE_URL", "").strip().rstrip("/")
BASE = _EXTERNAL or f"http://127.0.0.1:{PORT}"
OWNS_SERVER = not _EXTERNAL
LABEL = os.environ.get("SMOKE_LABEL", "tree")
OUT = Path(os.environ.get("PW_ART_DIR", "logs/playwright-heavy-thread"))
OUT.mkdir(parents = True, exist_ok = True)

# Characters of thread content, not messages. Sorted: the growth check reads the first and last
# entries as smallest and largest, so an unsorted override would invert every ratio and report a
# good run as measuring nothing.
SIZES = sorted(
    int(n) for n in os.environ.get("SMOKE_HEAVY_CHARS", "25000,100000,300000").split(",")
)
ENGINES = [
    e.strip()
    for e in os.environ.get("SMOKE_HEAVY_ENGINES", "chromium,webkit,firefox").split(",")
    if e.strip()
]
# Each action is run this many times on its own page and the table reports the MEDIAN. Both tables
# also print repetition 1 against the median of the rest, because a median that hides a divergence
# between the two is how the scroll column came to be published as the cost of scrolling.
REPEATS = int(os.environ.get("SMOKE_HEAVY_REPEATS", "3"))
# Chromium only, and off by default. Throttling is a CDP feature, so switching it on makes the
# chromium column incomparable with the other two.
CPU_THROTTLE_RATE = float(os.environ.get("SMOKE_CPU_THROTTLE", "1"))

KEYSTROKES = int(os.environ.get("SMOKE_KEYSTROKES", "5"))
SCROLL_STEPS = int(os.environ.get("SMOKE_SCROLL_STEPS", "20"))
SCROLL_STEP_PX = int(os.environ.get("SMOKE_SCROLL_STEP_PX", "400"))
SEED_TIMEOUT_MS = int(os.environ.get("SMOKE_SEED_TIMEOUT_MS", "300000"))
ACTION_TIMEOUT_MS = int(os.environ.get("SMOKE_ACTION_TIMEOUT_MS", "120000"))
# How long an in-page action waits for the DOM to reach the state it asked for. This bounds an
# action that never happened and nothing else, so it has to stay well above the slowest honest
# measurement or a very slow open is reported as "never opened".
SETTLE_TIMEOUT_MS = int(os.environ.get("SMOKE_SETTLE_TIMEOUT_MS", "120000"))
ACTIONS = ("keystroke", "scroll", "jump", "menu", "delete", "reopen")

# Which tables to produce. Both by default, isolated first, because the isolated one is the
# headline and the sequenced one only means anything next to it. `SMOKE_HEAVY_TABLES=isolated`
# skips the sequence, which is one seed of seven; `=sequenced` reproduces exactly what this file
# printed before the isolation change, for anyone re-reading an old run.
TABLES = tuple(
    t.strip()
    for t in os.environ.get("SMOKE_HEAVY_TABLES", "isolated,sequenced").split(",")
    if t.strip() in ("isolated", "sequenced")
) or ("isolated",)
# Which table the headline `actions` key, the growth report and the verdict read. Isolated
# whenever it was measured: a number carrying another action's residue cannot size a change.
HEADLINE_TABLE = "isolated" if "isolated" in TABLES else "sequenced"
# Repetition 1 against the median of repetitions 2..N. Above this ratio in either direction the
# run reports a carry-over (later repetitions slower: the previous repetition left work behind) or
# a cold start (repetition 1 slower). 1.25 is set from the measurement that motivated this file's
# restructuring: the contaminated 300K scroll repetitions were 1049.7 / 949.9 / 916.7 / 950.1 ms
# around a median of 949.9, i.e. run-to-run spread inside about 10%, while the clean repetition
# sat at 666.9, a divergence of 1.42x. The bar has to clear the first number and catch the second.
CARRYOVER_RATIO = float(os.environ.get("SMOKE_CARRYOVER_RATIO", "1.25"))

# Installed into every page before anything else runs.
#
# The rAF wrapper COUNTS, it does not pump. Replacing rAF with a fixed timer -- which
# playwright_chat_autoscroll.py does on purpose, because it counts frames -- would destroy every
# time-to-paint number this file exists to read. The harness's own waits use the unwrapped
# reference, so __rafCount stays a count of the page's frames rather than of this file's.
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
  window.__nextPaint = () =>
    new Promise((resolve) => nativeRaf(() => nativeRaf(() => resolve())));

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
    frames: [],
    stalls: [],
    startedAt: 0,
    begin() {
      this.running = true;
      this.frames = [];
      this.stalls = [];
      this.startedAt = performance.now();
      let lastFrame = performance.now();
      const frame = () => {
        const now = performance.now();
        this.frames.push(now - lastFrame);
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
        const now = performance.now();
        this.stalls.push(now - lastStall);
        lastStall = now;
        if (this.running) setTimeout(stall, 1);
      };
      setTimeout(stall, 1);
    },
    end() {
      this.running = false;
      const wallMs = performance.now() - this.startedAt;
      const frames = this.frames;
      const stalls = this.stalls;
      const sorted = frames.slice().sort((a, b) => a - b);
      return {
        wall_ms: Math.round(wallMs * 10) / 10,
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
  };
  window.__hv = recorder;
})();
"""


def info(message: str) -> None:
    print(f"[heavy-thread] {message}", flush = True)


# ── in-page actions ───────────────────────────────────────────────────
#
# Each one brackets itself with __hv.begin()/__hv.end(), so the recorder window is exactly the
# action rather than the action plus a CDP round trip.

# One character through the native value setter plus an input event: what the browser leaves
# behind after a real keypress, and what React's controlled textarea reacts to. Resolved on the
# second rAF, which is the frame that has painted it.
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

# A wheel gesture of fixed length traverses a fixed number of pixels, so it is comparable across
# sizes -- and, measured, it turns out to be nearly free at any size, because the layout was
# already done at mount. This comment predates the isolation change and was right while the table
# beside it said otherwise: on a page that has only scrolled, the 20-step gesture costs 665.2ms at
# 300K characters, which is 33.3ms a step, which is the two-vsync floor printed as `paint floor
# ms`. The gesture buys nothing but frames. That is a real answer, but it is only half of
# "scrolling". The other half
# is the jump: dragging the scrollbar or hitting Home moves the viewport to a region the compositor
# has nothing for, which is what a user does when they go looking for an earlier answer.
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
  // The wheel event first, and it is load-bearing. Studio replaces assistant-ui's autoscroll
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

# Radix portals the menu to document.body and puts the body on the modal layer, which is the
# fan-out under suspicion. bodyPointerEvents proves the open really took that path.
#
# The trigger opens on `pointerdown`, not on `click`: an element.click() leaves the menu shut and
# the whole measurement silently reads zero. Hence the pointer pair.
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

# Leaving a thread and coming back. The runtime keeps the messages; the Thread subtree is torn
# down and rebuilt, which is every markdown block, every Shiki fence and every action bar mounted
# again from nothing. This is the action users describe as "it hangs when I click back into the
# conversation", and it is the one that has no incremental path at all.
REOPEN_JS = """
async ([timeoutMs, settleMs]) => {
  const api = window.__heavyThread;
  const before = api.messageCount();
  if (!before) return null;
  window.__hv.begin();
  const started = performance.now();
  api.closeThread();
  // Unmount first, or "already back" is indistinguishable from "never left".
  let closedMs = null;
  while (performance.now() - started < timeoutMs) {
    if (api.messageCount() === 0) { closedMs = performance.now() - started; break; }
    await window.__nextPaint();
  }
  const reopenStarted = performance.now();
  api.openThread();
  let ms = null;
  while (performance.now() - reopenStarted < timeoutMs) {
    if (api.messageCount() >= before) { ms = performance.now() - reopenStarted; break; }
    await window.__nextPaint();
  }
  const settleMsTaken = await window.__hv.quiet(settleMs);
  const metrics = window.__hv.end();
  return {
    ms,
    closedMs: closedMs === null ? null : Math.round(closedMs * 10) / 10,
    settleMs: settleMsTaken === null ? null : Math.round(settleMsTaken * 10) / 10,
    before,
    after: api.messageCount(),
    metrics,
  };
}
"""

# The floor under every timing clocked across a double rAF: two rAFs resolve no sooner than two
# vsync intervals. An action that never happened therefore still reports ~33ms, which reads as a
# plausible measurement rather than as a failure, so the floor is recorded per cell and
# subtracted before any growth ratio.
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


def median(values: list[float]) -> float | None:
    ordered = sorted(v for v in values if v is not None)
    if not ordered:
        return None
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
    # PerformanceObserver callbacks are delivered on a later task, so the entry for the long task
    # at the tail of an action is not in the array yet. Yield once before reading, or the worst
    # entry is silently dropped -- flakily, and most often at large sizes where that tail task is
    # longest.
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


# The script and argument for each action, so the isolated runner and the sequenced runner drive
# exactly the same code with exactly the same arguments. Two copies of this list is how the two
# tables would quietly stop being comparable.
ACTION_SCRIPTS = {
    "keystroke": (KEYSTROKE_JS, KEYSTROKES),
    "scroll": (SCROLL_JS, [SCROLL_STEPS, SCROLL_STEP_PX, SETTLE_TIMEOUT_MS]),
    "jump": (JUMP_JS, SETTLE_TIMEOUT_MS),
    "menu": (MENU_JS, SETTLE_TIMEOUT_MS),
    "delete": (DELETE_JS, SETTLE_TIMEOUT_MS),
    "reopen": (REOPEN_JS, [SETTLE_TIMEOUT_MS, SETTLE_TIMEOUT_MS]),
}


def drive(page, cdp, name: str) -> dict:
    script, arg = ACTION_SCRIPTS[name]
    return run_action(page, cdp, name, script, arg)


def settle_and_expand(page) -> None:
    """The precondition every action shares: a thread that has finished highlighting itself, with
    its tool cards open.

    A PRECONDITION, not contamination, which is why it survives the move to a page per action. The
    reopen action throws away every highlighted fence and starts Shiki again, so without this wait
    the repetition after it measures a thread that is still building itself: measured on Chromium
    at 300K, the scroll gesture read 667ms on the first repetition and 1100ms on the two that
    followed, and the difference was the re-highlighting, not the scroll. Re-open also unmounts the
    thread, and an uncontrolled Radix collapsible comes back closed, so without the expand every
    later repetition would run against a thread with no tool result panes in it -- a different,
    cheaper fixture wearing the same label. Idempotent and untimed: where nothing was torn down,
    the gate passes at once and nothing is clicked.
    """
    wait_for_highlighting_settled(page, ACTION_TIMEOUT_MS)
    expanded = page.evaluate("() => window.__heavyThread.expandTools()")
    if expanded:
        page.wait_for_function(
            "(n) => window.__heavyThread.counts().collapsibleOutputs >= n",
            arg = expanded,
            timeout = ACTION_TIMEOUT_MS,
        )


def reveal_last_action_bar(page) -> None:
    """Put the pointer on the last assistant message, which is what reveals its action bar.

    Also a precondition: `menu` and `delete` are defined as clicking a button that does not exist
    until the message is hovered. behavior: "instant" -- the viewport carries scroll-smooth, so the
    default animates and at large sizes leaves that animation in flight inside the action's own
    counters.
    """
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


def isolated_repetitions(page, cdp, name: str) -> list[dict]:
    """REPEATS repetitions of ONE action on a page that has only ever done that action.

    The preconditions above run before each repetition and nothing else does. What is left inside
    the window is the action, plus whatever the previous repetition OF THE SAME ACTION left behind
    -- which is the row's contract, and which the repetition-1-against-the-rest line reports.
    """
    rows = []
    for index in range(REPEATS):
        # Printed before the preconditions, not after: the settle gate is the slowest thing here
        # and a run that wedges in it should say which repetition it wedged in.
        info(f"    isolated {name}: repetition {index + 1}/{REPEATS}")
        settle_and_expand(page)
        # Only these two need the pointer. In the sequenced run `delete` inherits the hover the
        # menu left, so on its own page it has to be given one of its own or the button it clicks
        # is not in the DOM.
        if name in ("menu", "delete"):
            reveal_last_action_bar(page)
        rows.append(drive(page, cdp, name))
    return rows


def sequenced_repetition(page, cdp) -> dict[str, dict]:
    """The six actions, once, in the order a user meets them, on one page.

    This is the arrangement that produced every number this file published before the isolation
    change, kept EXACTLY as it was so the carry-over table means what the old table meant. Nothing
    in here is reset between actions on purpose: the residue is the measurement.
    """
    rep: dict[str, dict] = {}
    settle_and_expand(page)
    rep["keystroke"] = drive(page, cdp, "keystroke")
    rep["scroll"] = drive(page, cdp, "scroll")
    rep["jump"] = drive(page, cdp, "jump")
    reveal_last_action_bar(page)
    rep["menu"] = drive(page, cdp, "menu")
    # The scroll position has not moved since the menu, so the hover alone is enough here.
    page.locator('[data-role="assistant"]').last.hover(timeout = ACTION_TIMEOUT_MS)
    rep["delete"] = drive(page, cdp, "delete")
    rep["reopen"] = drive(page, cdp, "reopen")
    return rep


# Portable headline per action, plus the action's own DOM-observable duration. `floored` marks a
# value clocked across a double rAF, which carries the ~33ms vsync floor.
HEADLINE = {
    "keystroke": ("median_sample_ms", True),
    "scroll": ("gestureMs", False),
    "jump": ("paintedMs", True),
    "menu": ("open_close_ms", True),
    "delete": ("ms", True),
    "reopen": ("ms", False),
}


def summarise(rows_by_action: dict[str, list[dict]]) -> dict[str, dict]:
    """Median across repetitions, per action, per metric.

    Medians because the first repetition is systematically the slowest on some actions and a mean
    would carry it into every cell. A median is also what hid the defect this file was restructured
    for, so it is never printed alone: `per_repetition` and the repetition-1 divergence go with it.
    """
    out: dict[str, dict] = {}
    for action in ACTIONS:
        rows = rows_by_action.get(action) or []
        if not rows or not all(r.get("ran") for r in rows):
            out[action] = {"ran": False}
            continue
        merged: dict = {"ran": True, "repetitions": len(rows)}
        numeric_keys = set()
        for row in rows:
            for key, value in row.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric_keys.add(key)
        for key in sorted(numeric_keys):
            merged[key] = median([r.get(key) for r in rows])
        # Values that are not numbers are proofs the action really happened, not timings, so the
        # last repetition's is kept verbatim rather than aggregated.
        for key in ("domText", "runtimeText", "bodyPointerEvents", "bodyPointerEventsAfterClose"):
            if key in rows[-1]:
                merged[key] = rows[-1][key]
        # The headline value from each repetition, unaggregated, so a median can be checked
        # against the spread it came from rather than taken on trust.
        merged["per_repetition"] = [r.get(HEADLINE[action][0]) for r in rows]
        out[action] = merged
    return out


class SeededPage:
    """One browser context, one page, one seeded thread: the unit of isolation.

    A fresh CONTEXT, not just a fresh page. The page carries the JS heap that code-plugin.ts keeps
    its highlighted-token cache in, and the context carries the HTTP cache, so re-seeding into a
    context another arm has already used would hand this arm a warm cache the other one paid for.

    Seeding is unthrottled and untimed on purpose: this file measures interaction cost at a thread
    size, not the cost of building the thread.
    """

    def __init__(self, browser, engine: str, arm: str) -> None:
        self.engine = engine
        self.arm = arm
        self.context = browser.new_context(viewport = {"width": 1440, "height": 900})
        self.context.add_init_script(RECORDER_INIT)
        # Anchored at the origin so it cannot swallow vite's own module URLs, which live under
        # src/features/**/api/ and would otherwise match a bare "/api/" pattern.
        self.context.route(
            re.compile(rf"^{re.escape(BASE)}/api/"),
            lambda route: route.fulfill(
                status = 200,
                content_type = "application/json",
                body = "{}",
            ),
        )
        self.page = self.context.new_page()
        self.cdp = None
        # A request that escapes to the server, or a warning storm, is work this harness would be
        # charging to the app once per message. Both are cleared after seeding, so what is
        # asserted on is the measured actions rather than page load.
        #
        # startswith, not `"/api/" in url`: vite serves the app's own source modules from paths
        # like /src/features/chat/api/chat-api.ts, and a substring match counts dozens of those as
        # network calls. Same trap the route regex is anchored to avoid.
        api_prefix = f"{BASE}/api/"
        self.stray_requests: list[str] = []
        self.console_warnings: list[str] = []
        self.page.on(
            "request",
            lambda r: self.stray_requests.append(r.url) if r.url.startswith(api_prefix) else None,
        )
        self.page.on(
            "console",
            lambda m: (
                self.console_warnings.append(m.text[:200])
                if m.type in ("warning", "error")
                else None
            ),
        )
        self.page.on("pageerror", lambda e: self.console_warnings.append(f"pageerror: {e}"[:200]))
        self.record: dict = {"arm": arm}

    def seed(self, size: int) -> None:
        page = self.page
        page.goto(f"{BASE}/smoke-heavy-thread.html", wait_until = "domcontentloaded")
        page.wait_for_function("() => Boolean(window.__heavyThread)", timeout = 60_000)
        if self.engine == "chromium":
            self.cdp = self.context.new_cdp_session(page)
            self.cdp.send("Performance.enable")

        plan = page.evaluate("(n) => window.__heavyThread.seed(n)", size)
        self.record["plan"] = plan
        # Single-selector gates. counts() walks every element in the document, so polling it per
        # frame makes seeding superlinear in the thing being seeded.
        page.wait_for_function(
            "(n) => window.__heavyThread.messageCount() >= n",
            arg = plan["messages"],
            timeout = SEED_TIMEOUT_MS,
        )
        # Radix unmounts collapsed content, so a thread of closed tool cards carries no result
        # panes at all. Open them before the census: a user who has just watched those tools run
        # is looking at them open, and the closed thread is a different fixture.
        self.record["tool_triggers_expanded"] = page.evaluate(
            "() => window.__heavyThread.expandTools()"
        )
        page.wait_for_function(
            "(n) => window.__heavyThread.counts().collapsibleOutputs >= n",
            arg = max(1, self.record["tool_triggers_expanded"]),
            timeout = SEED_TIMEOUT_MS,
        )
        # Shiki is async and per block, and a <pre> exists before it is highlighted, so counting
        # code blocks gates nothing. Wait for the token count to stop moving instead: unfinished
        # highlighting would otherwise land in the first action measured on this page.
        wait_for_highlighting_settled(page, SEED_TIMEOUT_MS)
        self.record["counts"] = page.evaluate("window.__heavyThread.counts()")
        self.record["viewport"] = page.evaluate("window.__heavyThread.viewportMetrics()")
        self.record["seed_api_requests"] = len(self.stray_requests)
        self.record["seed_console_warnings"] = len(self.console_warnings)
        self.record["first_seed_warning"] = (
            self.console_warnings[0] if self.console_warnings else "-"
        )
        self.stray_requests.clear()
        self.console_warnings.clear()

        self.record["cpu_throttle_rate"] = 1.0
        if self.cdp is not None and CPU_THROTTLE_RATE != 1.0:
            self.cdp.send("Emulation.setCPUThrottlingRate", {"rate": CPU_THROTTLE_RATE})
            self.record["cpu_throttle_rate"] = CPU_THROTTLE_RATE
        self.record["long_task_supported"] = page.evaluate("window.__longTaskSupported")
        # Measured per page, because it is a property of this page's vsync cadence and every arm
        # gets its own page now.
        self.record["paint_floor_ms"] = round(self.page.evaluate(PAINT_FLOOR_JS, 9), 2)

    def finish(self) -> dict:
        """Everything only readable after the actions have run, then the record."""
        if self.cdp is not None and CPU_THROTTLE_RATE != 1.0:
            self.cdp.send("Emulation.setCPUThrottlingRate", {"rate": 1})
        # Cumulative over seeding and every action on this page: a liveness check, not
        # attributable to any one action.
        self.record["raf_callbacks"] = self.page.evaluate("window.__rafCount")
        self.record["stray_api_requests"] = len(self.stray_requests)
        self.record["console_warnings"] = len(self.console_warnings)
        self.record["first_console_warning"] = (
            self.console_warnings[0] if self.console_warnings else "-"
        )
        return self.record

    def close(self) -> None:
        self.page.close()
        self.context.close()


def open_seeded_page(browser, engine: str, size: int, arm: str) -> SeededPage:
    """A seeded page, or nothing: a seed that throws must not leave its context behind, or a long
    matrix leaks one browser context per failed cell."""
    seeded = SeededPage(browser, engine, arm)
    try:
        seeded.seed(size)
    except Exception:
        seeded.close()
        raise
    return seeded


def measure_isolated(browser, engine: str, size: int) -> tuple[dict, dict, list[dict]]:
    """One fresh context, page and seed PER ACTION, then REPEATS repetitions of that action alone.

    The headline. This is the whole point of the restructuring: a cell in this table is "N
    repetitions of action X on a page that has only ever done X", and nothing else is in the
    window. It costs one seed per action -- seven per cell with the sequenced page -- against the
    one the old arrangement paid, which is the honest price of the row meaning what it says.
    """
    rows_by_action: dict[str, list[dict]] = {}
    seeds: list[dict] = []
    for name in ACTIONS:
        info(f"  {engine} {size} chars: isolated page for {name}")
        seeded = open_seeded_page(browser, engine, size, f"isolated:{name}")
        try:
            rows_by_action[name] = isolated_repetitions(seeded.page, seeded.cdp, name)
            seeds.append(seeded.finish())
        finally:
            seeded.close()
    return summarise(rows_by_action), rows_by_action, seeds


def measure_sequenced(browser, engine: str, size: int) -> tuple[dict, list[dict], list[dict]]:
    """The six actions in one fixed order, repeated on ONE page: cost after a session of use.

    Carry-over is a real user scenario -- a user really does open a menu and then scroll -- so it
    is measured, and it is measured exactly the way the old harness measured it. It is simply
    labelled for what it is instead of being printed in a column called scroll.
    """
    seeded = open_seeded_page(browser, engine, size, "sequenced")
    try:
        reps = []
        for index in range(REPEATS):
            info(f"  {engine} {size} chars: sequenced repetition {index + 1}/{REPEATS}")
            reps.append(sequenced_repetition(seeded.page, seeded.cdp))
        seeds = [seeded.finish()]
    finally:
        seeded.close()
    return summarise({a: [r[a] for r in reps if a in r] for a in ACTIONS}), reps, seeds


def merge_seeds(seeds: list[dict]) -> dict:
    """Seven pages, one column. How each per-page number becomes the cell's.

    The fixture is a pure function of the requested size, so every page seeds the same thread and
    the census values should be identical. They are merged as MINIMA anyway: a fixture that failed
    to land on one of the seven pages must not be able to hide behind the six that worked, and the
    existing per-count gates in harness_failures then fire on the worst page for free.

    Console warnings are merged as a MAX, not a sum, and that one matters. The allowance is per
    page -- Firefox emits its two scroll-anchoring notices once per page it loads -- so summing
    would multiply engine chatter by seven and report every Gecko cell as broken. Stray /api/
    requests are summed instead, because the allowance there is zero and any page leaking one is
    the finding.
    """
    counts_keys = sorted({key for s in seeds for key in s["counts"]})
    viewport_keys = sorted({key for s in seeds for key in s["viewport"]})
    chars = [s["plan"]["chars"] for s in seeds]
    floors = [s["paint_floor_ms"] for s in seeds]
    warned = [s for s in seeds if s["first_console_warning"] != "-"]
    seed_warned = [s for s in seeds if s["first_seed_warning"] != "-"]
    return {
        "pages_seeded": len(seeds),
        # Both zero unless the pages disagreed, which is the guard: an arm that seeded a different
        # thread is an arm whose column is not comparable with the others.
        "seed_chars_spread": max(chars) - min(chars),
        "paint_floor_spread_ms": round(max(floors) - min(floors), 2),
        "plan": seeds[0]["plan"],
        "counts": {key: min(s["counts"].get(key, 0) for s in seeds) for key in counts_keys},
        "viewport": {key: min(s["viewport"].get(key, 0) for s in seeds) for key in viewport_keys},
        "tool_triggers_expanded": min(s["tool_triggers_expanded"] for s in seeds),
        "cpu_throttle_rate": seeds[0]["cpu_throttle_rate"],
        "long_task_supported": all(s["long_task_supported"] for s in seeds),
        "paint_floor_ms": median(floors),
        "seed_api_requests": sum(s["seed_api_requests"] for s in seeds),
        "seed_console_warnings": max(s["seed_console_warnings"] for s in seeds),
        "first_seed_warning": seed_warned[0]["first_seed_warning"] if seed_warned else "-",
        "raf_callbacks": sum(s["raf_callbacks"] for s in seeds),
        "stray_api_requests": sum(s["stray_api_requests"] for s in seeds),
        "console_warnings": max(s["console_warnings"] for s in seeds),
        "first_console_warning": warned[0]["first_console_warning"] if warned else "-",
    }


def measure_cell(browser, engine: str, size: int) -> dict:
    """One column of the matrix: the isolated table, the carry-over table, and the seeds behind
    both."""
    result: dict = {
        "chars_requested": size,
        "engine": engine,
        "repetitions": REPEATS,
        "tables": list(TABLES),
        "headline_table": HEADLINE_TABLE,
    }
    seeds: list[dict] = []
    if "isolated" in TABLES:
        isolated, raw_isolated, isolated_seeds = measure_isolated(browser, engine, size)
        result["isolated_actions"] = isolated
        result["isolated_raw_repetitions"] = raw_isolated
        seeds += isolated_seeds
    if "sequenced" in TABLES:
        sequenced, raw_sequenced, sequenced_seeds = measure_sequenced(browser, engine, size)
        result["sequenced_actions"] = sequenced
        # `raw_repetitions` keeps its old name and its old shape -- a list of one dict per
        # repetition, keyed by action -- because that is what it has always held: the mixed
        # sequence.
        result["raw_repetitions"] = raw_sequenced
        seeds += sequenced_seeds
    result["actions"] = result[f"{result['headline_table']}_actions"]
    result["seeds"] = seeds
    result.update(merge_seeds(seeds))
    return result


def run() -> dict:
    results: dict = {
        "label": LABEL,
        "base": BASE,
        "sizes": SIZES,
        "engines": ENGINES,
        "repetitions": REPEATS,
        "tables": list(TABLES),
        "carryover_ratio": CARRYOVER_RATIO,
        "cpu_throttle_rate_requested": CPU_THROTTLE_RATE,
        "by_engine": {},
    }
    with sync_playwright() as p:
        for engine in ENGINES:
            info(f"engine {engine}")
            launcher = getattr(p, engine)
            kwargs = {"headless": os.environ.get("SMOKE_HEADLESS", "1") == "1"}
            # Chromium-only flags. Passing them to Firefox or WebKit is not "ignored", it is a
            # launch failure, which would read as "this engine is unavailable".
            if engine == "chromium":
                kwargs["args"] = chromium_launch_args()
            browser = launcher.launch(**kwargs)
            results["by_engine"][engine] = {"version": browser.version, "by_size": {}}
            for size in SIZES:
                info(f"measuring {engine} at {size} chars")
                # A renderer that dies mid-cell used to take the whole matrix with it: nine cells
                # of work thrown away because one WebKit page ran out of memory at 300K on a
                # loaded machine. The cell is recorded as crashed, the run continues, and
                # harness_failures reports it -- a crash is still a failure, it is just no longer
                # a failure that destroys the eight measurements that did work.
                try:
                    cell = measure_cell(browser, engine, size)
                except Exception as exc:  # noqa: BLE001 - the message is the whole point
                    info(f"CRASHED {engine} at {size} chars: {type(exc).__name__}: {exc}")
                    cell = {
                        "chars_requested": size,
                        "engine": engine,
                        "crashed": f"{type(exc).__name__}: {exc}"[:400],
                    }
                results["by_engine"][engine]["by_size"][str(size)] = cell
            browser.close()
    return results


def repetition_divergence(values: list) -> dict:
    """Repetition 1 against the median of repetitions 2..N, for one action.

    THE line this file was missing. On the old mixed sequence at 300K the scroll repetitions were
    666.9 / 1049.7 / 949.9 / 916.7 / 950.1, and this comparison reads 666.9 against 949.9, 1.42x.
    Printed from the start it would have caught the defect months earlier, instead of the median
    reporting 949.9 and discarding the one clean repetition as the outlier.

    Both directions are reported, because they mean opposite things. Later repetitions SLOWER is
    carry-over: repetition 1 ran on a cleaner page than the ones after it. Repetition 1 slower is a
    cold start: a cache that was empty the first time round. Neither is a harness failure, which is
    why this returns a finding and not a verdict.
    """
    numbers = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    out = {"first": None, "rest_median": None, "ratio": None, "diverged": False, "reason": "-"}
    if not values or len(numbers) != len(values):
        return {**out, "reason": "not recorded"}
    # Rounded here rather than at the printer. Some in-page clocks hand back the raw
    # performance.now() difference, and an unrounded 121.09999990463257 in a table cell sets the
    # column width for every other cell in the run.
    out["first"] = round(numbers[0], 1)
    if len(numbers) < 2:
        return {**out, "reason": "one repetition"}
    out["rest_median"] = median(numbers[1:])
    if out["first"] <= 0 or out["rest_median"] is None:
        return {**out, "reason": "zero on repetition 1"}
    out["ratio"] = round(out["rest_median"] / out["first"], 2)
    if out["ratio"] >= CARRYOVER_RATIO:
        return {**out, "diverged": True, "reason": "later repetitions slower"}
    if out["ratio"] <= 1 / CARRYOVER_RATIO:
        return {**out, "diverged": True, "reason": "repetition 1 slower"}
    return out


def format_divergence(row: dict) -> str:
    if row["ratio"] is None:
        return "-"
    return f"{row['first']}->{row['rest_median']} {row['ratio']}x"


def format_repetitions(values: list) -> str:
    """The spread a median came from. Rounded, because the in-page clocks hand back raw
    performance.now() differences and one 121.09999990463257 widens every column in the table."""
    return "/".join(
        "-" if v is None else str(round(v, 1) if isinstance(v, float) else v) for v in values
    )


def short(text: str) -> str:
    """A console message in a table cell, cut to something that does not set the column width.
    The full string is in the JSON."""
    return text if len(text) <= 48 else f"{text[:45]}..."


# Every recorded metric appears here. That is the rule the harnesses in this directory are held
# to: a metric that is recorded and never read is how one goes false-green, and
# tests/studio/test_heavy_thread_harness_contract.py fails if anything recorded below is missing.
# CHROMIUM-ONLY rows are labelled in their own name, because off Chromium they print `-` and a
# `-` that means "not supported here" must not read as "zero".
def _action(action: str, key: str):
    return lambda r: r["actions"][action][key]


TABLE_ROWS = (
    ("chars requested", lambda r: r["chars_requested"]),
    ("chars rendered", lambda r: r["plan"]["chars"]),
    ("messages seeded", lambda r: r["plan"]["messages"]),
    ("content cycles", lambda r: r["plan"]["cycles"]),
    ("chars per cycle", lambda r: r["plan"]["cycleChars"]),
    ("content kinds", lambda r: r["plan"]["kinds"]),
    ("tool cards expanded", lambda r: r["tool_triggers_expanded"]),
    ("repetitions", lambda r: r["repetitions"]),
    ("headline table", lambda r: r["headline_table"]),
    ("pages seeded", lambda r: r["pages_seeded"]),
    # Both of these are zero on a healthy cell. A non-zero chars spread means the arms did not all
    # build the same thread, which is the one way a page-per-action arrangement can produce columns
    # that are not comparable with each other.
    ("seed chars spread", lambda r: r["seed_chars_spread"]),
    ("paint floor spread ms", lambda r: r["paint_floor_spread_ms"]),
    ("cpu throttle rate", lambda r: r["cpu_throttle_rate"]),
    ("paint floor ms", lambda r: r["paint_floor_ms"]),
    ("longtask api supported", lambda r: r["long_task_supported"]),
    ("seed api requests", lambda r: r["seed_api_requests"]),
    ("seed console warnings", lambda r: r["seed_console_warnings"]),
    ("first seed warning", lambda r: short(r["first_seed_warning"])),
    ("action api requests", lambda r: r["stray_api_requests"]),
    ("action console warnings", lambda r: r["console_warnings"]),
    ("first action warning", lambda r: short(r["first_console_warning"])),
    ("messages rendered", lambda r: r["counts"]["messages"]),
    ("dom nodes", lambda r: r["counts"]["domNodes"]),
    ("code blocks", lambda r: r["counts"]["codeBlocks"]),
    ("highlighted tokens", lambda r: r["counts"]["highlightedTokens"]),
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

# Every action, not the five that happened to be listed here before. The one action missing from
# that list was `jump`, and "which columns do we bother printing the spread for" is exactly the
# judgement call that let a 666.9 / 1049.7 / 949.9 / 916.7 / 950.1 spread be published as 949.9.
TABLE_ROWS = TABLE_ROWS + tuple(
    (
        f"{_name} per repetition",
        lambda r, _name = _name: format_repetitions(r["actions"][_name]["per_repetition"]),
    )
    for _name in ACTIONS
)

TABLE_ROWS = TABLE_ROWS + tuple(
    (
        f"{_name} rep1 vs rest",
        lambda r, _name = _name: format_divergence(
            repetition_divergence(r["actions"][_name]["per_repetition"])
        ),
    )
    for _name in ACTIONS
)

TABLE_ROWS = TABLE_ROWS + (
    ("keystroke median ms", _action("keystroke", "median_sample_ms")),
    ("keystroke worst ms", _action("keystroke", "worst_sample_ms")),
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


def as_table(row: dict, table: str) -> dict:
    """A cell viewed through one of its two tables.

    Every TABLE_ROWS entry reads `r["actions"]`, so a table is chosen by swapping what that key
    holds rather than by keeping a second copy of ninety row definitions in step with the first.
    """
    # An empty dict rather than a missing key when the table was not measured: the action rows
    # then print `-` while the fixture rows above them still print what the cell did seed.
    return {**row, "actions": row.get(f"{table}_actions") or {}}


def print_table(results: dict, table: str, title: str) -> None:
    """Every recorded metric, printed, one column per (engine, size)."""
    columns = [(engine, str(size)) for engine in results["engines"] for size in results["sizes"]]
    rows = []
    for name, pick in TABLE_ROWS:
        cells = []
        for engine, size in columns:
            try:
                value = pick(as_table(results["by_engine"][engine]["by_size"][size], table))
                cells.append("-" if value is None else str(value))
            except (KeyError, TypeError, IndexError):
                cells.append("-")
        rows.append((name, cells))
    info("")
    info(title)
    label_width = max(len(name) for name, _ in rows) + 2
    # From the widest cell, not a constant: a fixed width silently runs the columns together on
    # the one row that overflows it, which is the row you were reading.
    headers = [f"{engine[:4]}/{int(size) // 1000}K" for engine, size in columns]
    cell_width = max([len(c) for _, cells in rows for c in cells] + [len(h) for h in headers]) + 2
    header = "".ljust(label_width) + "".join(h.rjust(cell_width) for h in headers)
    info(header)
    info("-" * len(header))
    for name, cells in rows:
        info(name.ljust(label_width) + "".join(cell.rjust(cell_width) for cell in cells))


def divergence_report(results: dict, table: str) -> dict:
    """Repetition 1 against the median of the rest, for every action, in one table."""
    report: dict = {}
    for engine in results["engines"]:
        per_size: dict = {}
        for size in results["sizes"]:
            row = as_table(results["by_engine"][engine]["by_size"][str(size)], table)
            per_action: dict = {}
            for action in ACTIONS:
                values = row["actions"].get(action, {}).get("per_repetition")
                per_action[action] = repetition_divergence(values or [])
            per_size[str(size)] = per_action
        report[engine] = per_size
    return report


def print_divergence(results: dict, report: dict, headline: str, flag: str) -> None:
    """The carry-over signal, printed for every action rather than for the ones someone suspected.

    A divergence here is a FINDING, not a failure: the harness still exits 0. What it says is that
    the median above it is an average of two different measurements, and which of the two a reader
    wants depends on the question. On the carry-over table the flag means the previous repetition
    left work behind; on the isolated table it means this action is warming something up on its
    own, since nothing else has run on that page.

    It catches carry-over from the previous REPETITION only, so it is a floor on the effect and
    never a ceiling. An action contaminated at the same point inside every repetition -- `delete`
    always follows `menu` -- diverges by nothing here and still differs between the two tables:
    measured at 300K, delete reads 1.15x steady while its isolated and sequenced medians are
    740.4ms and 568.5ms. The two readings answer different questions and both have to be read.
    """
    for engine, per_size in report.items():
        for size, per_action in per_size.items():
            info("")
            info(f"{headline} on {engine} at {size} chars (metric: the action's headline number)")
            for action, row in per_action.items():
                if row["ratio"] is None:
                    info(f"  {action:<12} {'-':>10}    {'-':>10}       -  ({row['reason']})")
                    continue
                mark = f"{flag} ({row['reason']})" if row["diverged"] else "steady"
                info(
                    f"  {action:<12} {row['first']:>10} -> {row['rest_median']:>10}  "
                    f"{row['ratio']:>6.2f}x  {mark}"
                )


# Growth axes: the whole point of the harness is that these rise with content. `floored` marks a
# metric clocked across a double rAF, which carries the ~33ms vsync floor; left in, the floor
# compresses every ratio towards 1 and lets a real regression sit under the threshold.
GROWTH_AXES = tuple(
    [(f"{a} longest stall ms", _action(a, "longest_stall_ms"), False) for a in ACTIONS]
    + [(f"{a} worst frame ms", _action(a, "worst_frame_ms"), False) for a in ACTIONS]
    + [(f"{a} frames over 33ms", _action(a, "frames_over_33"), False) for a in ACTIONS]
    + [(f"{a} wall ms", _action(a, "wall_ms"), False) for a in ACTIONS]
    + [
        ("keystroke median ms", _action("keystroke", "median_sample_ms"), True),
        ("scroll gesture ms", _action("scroll", "gestureMs"), False),
        ("scroll settle ms", _action("scroll", "settleMs"), False),
        ("jump painted ms", _action("jump", "paintedMs"), True),
        ("jump settle ms", _action("jump", "settleMs"), False),
        ("menu open+close ms", _action("menu", "open_close_ms"), True),
        ("delete ms", _action("delete", "ms"), True),
        ("reopen ms", _action("reopen", "ms"), False),
    ]
)
# A ratio at or below this from the smallest size to the largest means the axis did not respond
# to twelve times the content. That is not a flat curve, it is an axis that is not measuring the
# thing being varied.
DISCRIMINATION_RATIO = float(os.environ.get("SMOKE_DISCRIMINATION_RATIO", "1.5"))
# Engine chatter is tolerated up to this many warnings per size. Gecko's two scroll-anchoring
# notices are what this number exists for; anything the app emits per message would be two orders
# of magnitude above it at 220 messages.
CONSOLE_WARNING_ALLOWANCE = int(os.environ.get("SMOKE_CONSOLE_WARNING_ALLOWANCE", "4"))


def growth(cells: dict, pick, floored: bool, sizes: list[int]) -> tuple[float | None, float | None]:
    try:
        rows = (cells[str(sizes[0])], cells[str(sizes[-1])])
        values = []
        for row in rows:
            value = pick(row)
            if value is None:
                return None, None
            if floored:
                value -= row["paint_floor_ms"]
            values.append(round(value, 2))
        return values[0], values[1]
    except (KeyError, TypeError):
        return None, None


def report_growth(results: dict) -> dict[str, dict[str, dict]]:
    """Per engine, per axis: the value at the smallest and largest size and their ratio.

    Read off the HEADLINE table, which is the isolated one whenever it was measured. Sizing a
    change against a column that carries another action's residue is what this file stopped doing.
    """
    report: dict[str, dict[str, dict]] = {}
    for engine in results["engines"]:
        cells = results["by_engine"][engine]["by_size"]
        per_axis: dict[str, dict] = {}
        for name, pick, floored in GROWTH_AXES:
            small, large = growth(cells, pick, floored, results["sizes"])
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
                # A COUNT that is 0 at the smallest size and 4 at the largest has no ratio and has
                # still answered the question, so it counts as discriminating when it really rose.
                #
                # A TIMING does not get that credit. `floored` means the value had the ~33ms vsync
                # floor subtracted from it, so a zero or negative here says the action resolved at
                # or under one frame at the smallest size, which is a metric with no room to move
                # rather than a metric that grew from nothing.
                if floored:
                    per_axis[name] = {
                        "small": small,
                        "large": large,
                        "ratio": None,
                        "discriminated": False,
                        "reason": "at or under the paint floor at the smallest size",
                        "floored": floored,
                    }
                    continue
                rose = large > small
                per_axis[name] = {
                    "small": small,
                    "large": large,
                    "ratio": None,
                    "discriminated": rose,
                    "reason": "rose from zero" if rose else "zero at both ends",
                    "floored": floored,
                }
                continue
            ratio = round(large / small, 2)
            per_axis[name] = {
                "small": small,
                "large": large,
                "ratio": ratio,
                "discriminated": ratio > DISCRIMINATION_RATIO,
                "reason": "-",
                "floored": floored,
            }
        report[engine] = per_axis
    return report


def print_growth(results: dict, report: dict) -> None:
    for engine, per_axis in report.items():
        info("")
        info(
            f"growth on {engine} ({results['sizes'][0]} -> {results['sizes'][-1]} chars, "
            f"median of {results['repetitions']} repetitions, "
            f"{HEADLINE_TABLE} table)"
        )
        for name, row in per_axis.items():
            if row["ratio"] is None:
                mark = "DISCRIMINATES" if row["discriminated"] else "flat"
                small = "-" if row["small"] is None else row["small"]
                large = "-" if row["large"] is None else row["large"]
                info(f"  {name:<34} {small:>10} -> {large:>10}       -  {mark} ({row['reason']})")
                continue
            mark = "DISCRIMINATES" if row["discriminated"] else "flat"
            floor_note = " (paint floor removed)" if row.get("floored") else ""
            info(
                f"  {name:<34} {row['small']:>10} -> {row['large']:>10}  "
                f"{row['ratio']:>6.2f}x  {mark}{floor_note}"
            )


def harness_failures(results: dict, report: dict) -> list[str]:
    """Only the ways this harness can be measuring nothing. No performance budgets: see the
    module docstring."""
    failures: list[str] = []
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
            # A request reaching the server is a round trip to another process inside a region
            # being timed, once per message. A warning storm is the same cost via the console
            # channel. Both scale with content, so both would forge the curve.
            if row["stray_api_requests"]:
                failures.append(
                    f"{where} let {row['stray_api_requests']} /api/ requests reach the network "
                    "during the measured actions; the timings include a round trip per request"
                )
            # Console output from inside a timed region is serialised over the debugging channel,
            # so a warning the app emits once per message would both cost time and grow like the
            # signal. The instrument for that is an ABSOLUTE cap, not a growth check: at 220
            # messages a per-message warning is in the hundreds, while what an engine says about
            # itself is a handful. Firefox 153 emits exactly two "Scroll anchoring was disabled
            # in a scroll container" notices once the container is large enough, which is zero at
            # 25K and two at both 100K and 300K -- a growth check fails on that and would leave
            # this harness unable to report a Gecko number at all. The count and the first
            # message are printed per size either way, so a reader can see what was tolerated.
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
            # The fixture IS the measurement. A thread of plain paragraphs would be cheap for
            # reasons the app is not, so every kind the user report names has to be on screen,
            # in proportion, at every size.
            #
            # Counted against the page's own per-cycle expectation rather than against zero. The
            # renderer drops content silently in more than one place -- an image part whose data
            # URL is not base64 PNG/JPEG/GIF/WebP is discarded with a console.warn -- and a
            # fixture that has quietly lost a whole kind still produces a rising curve, of
            # something else.
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
            viewport = row["viewport"]
            if viewport["scrollHeight"] <= viewport["clientHeight"]:
                failures.append(
                    f"{where} does not overflow its viewport; the scroll measures nothing"
                )
            # One seed per action means one chance per action to seed something else. The fixture
            # is a pure function of the requested size, so any spread at all means the arms are not
            # holding the same thread and their columns cannot be compared with each other.
            if row["seed_chars_spread"]:
                failures.append(
                    f"{where} seeded threads that differ by {row['seed_chars_spread']} characters "
                    f"across its {row['pages_seeded']} pages; the per-action columns are not the "
                    "same fixture"
                )

            for table in row["tables"]:
                failures += action_failures(
                    f"{where} ({table})", as_table(row, table)["actions"], counts, viewport
                )

        # A modal menu puts the body on the modal layer and a non-modal one does not, and the two
        # cost wildly different amounts. Either is a legitimate tree, but a run that mixes them
        # across sizes is comparing columns measured on different mechanisms.
        for table in TABLES:
            layers = {
                as_table(results["by_engine"][engine]["by_size"][str(size)], table)["actions"]
                .get("menu", {})
                .get("bodyPointerEvents")
                for size in results["sizes"]
                if "crashed" not in results["by_engine"][engine]["by_size"][str(size)]
            }
            if len(layers) > 1:
                failures.append(
                    f"on {engine} ({table}) the menu put the body on "
                    f"{sorted(str(x) for x in layers)} across sizes; the columns are not measuring "
                    "the same mechanism"
                )

    # Discrimination. Not a budget: a harness where the largest thread costs what the smallest
    # does is not reporting a flat curve, it is reporting that it never drove the page.
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


# Below harness_failures on purpose, not above it: tests/studio/test_heavy_thread_harness_contract
# .py reads everything from `def harness_failures` to the end of this file as "the verdict" and
# checks what the verdict may and may not rest on, and these checks ARE the verdict. Hoisted above
# their caller they would leave that section looking as though the harness had stopped asserting
# the keystroke reached the runtime.
def action_failures(where: str, actions: dict, counts: dict, viewport: dict) -> list[str]:
    """Every way one table's actions can be measuring nothing. Run once per table, because the
    isolated table and the carry-over table drive the same six actions on different pages and
    either of them can fail on its own."""
    failures: list[str] = []
    if not actions:
        return failures
    for name in ACTIONS:
        if not actions[name].get("ran"):
            failures.append(f"{where} could not run the {name} action at all")
    keystroke = actions["keystroke"]
    if keystroke.get("ran"):
        # The DOM value is what the harness itself wrote, so it proves nothing on its
        # own. Only the runtime's copy shows the keystroke reached React rather than just
        # the textarea, and a keystroke that reached nothing still reports the ~33ms
        # paint floor, which reads as a plausible timing.
        if keystroke["runtimeText"] != keystroke["domText"]:
            failures.append(
                f"{where} typed {keystroke['domText']!r} into the DOM but the runtime "
                f"holds {keystroke['runtimeText']!r}; the keystroke never reached the "
                "composer state"
            )
        # Sitting on the paint floor is NOT a harness failure here, and the reason is a
        # finding rather than an excuse: the character reaches the composer and paints on
        # the very next frame at every size, while the thread churns for another 180ms
        # afterwards. `runtimeText == domText` above is what proves the keystroke landed;
        # the floor comparison only says this particular axis has no room to move, which
        # the growth report states per axis.
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
        # Unlike the gesture, the jump is DELIBERATELY not the same distance at every
        # size: it is bottom to top, which is the point. What has to hold is that it
        # arrived, or the column is timing a scroll that did not move.
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
            failures.append(f"{where} left the body on the modal layer after closing the menu")
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
    # Both divergence reports are computed before the JSON is written, because a carry-over
    # finding is a result of the run and not a note about the printout.
    results["warmup"] = divergence_report(results, "isolated")
    results["carryover"] = divergence_report(results, "sequenced")
    out = OUT / f"{LABEL}.json"
    out.write_text(json.dumps(results, indent = 2), encoding = "utf-8")
    if "isolated" in TABLES:
        print_table(
            results,
            "isolated",
            f"ISOLATED (headline): {REPEATS} repetitions of ONE action on a page that has only "
            "ever done that action",
        )
    if "sequenced" in TABLES:
        print_table(
            results,
            "sequenced",
            f"CARRY-OVER: the six actions in one fixed order, {REPEATS} times on ONE page -- what "
            "they cost after a session of use",
        )
    print_growth(results, report)
    if "isolated" in TABLES:
        print_divergence(
            results,
            results["warmup"],
            f"isolated table, repetition 1 vs median of 2..{REPEATS}",
            "WARM-UP",
        )
    if "sequenced" in TABLES:
        print_divergence(
            results,
            results["carryover"],
            f"carry-over table, repetition 1 vs median of 2..{REPEATS}",
            "CARRY-OVER",
        )
    info(f"wrote {out}")

    # Named, counted and put at the bottom where a reader lands, because the whole reason this
    # file was restructured is that a divergence of exactly this shape sat unreported inside a
    # median for months.
    flagged = [
        f"{engine} {size} {action}"
        for engine, per_size in results["carryover"].items()
        for size, per_action in per_size.items()
        for action, row in per_action.items()
        if row["diverged"]
    ]
    if flagged:
        info("")
        info(
            f"CARRY-OVER FINDING: {len(flagged)} of the carry-over table's columns diverge from "
            f"their own first repetition by more than {CARRYOVER_RATIO}x -- {', '.join(flagged)}. "
            "Compare each against the isolated table above. This is a fact about the app, not a "
            "broken harness, so it does not change the exit code."
        )

    failures = harness_failures(results, report)
    for problem in failures:
        info(f"HARNESS-BROKEN {problem}")
    if failures:
        return 1
    info("measurement only: no budgets are asserted here, so this exits 0 on any timing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
