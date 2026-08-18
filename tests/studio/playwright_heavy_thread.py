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
cursor resting over the conversation is how a person reads a long chat; the cheap arms are the
artificial ones. The cost being live rather than sticky makes it more fixable, not less.

THE SCROLL ROW PUTS THE POINTER ON THE CONVERSATION, AND FOR ONE REVISION IT DID NOT. Moving each
action onto its own page gave `scroll` a page whose mouse had never been moved, so Playwright's
cursor sat at (0, 0) -- the scroller's gutter -- and the headline row was measuring the arm the
probe registers as `gutter_only`, under a label that says otherwise. Every other precondition
survived the move and this one had never existed, because on the old single page the previous
repetition's `delete` hover supplied it by accident from repetition 2 onwards. `scroll` now takes
`place_pointer_over_message` as a precondition in BOTH runners, on the same footing as the hover
that `menu` and `delete` take, and prints what the pointer hit per repetition; a repetition whose
pointer never reached message content fails the run instead of being published.

WHAT THAT IS WORTH, MEASURED, AND IT IS LESS THAN THE TABLES ABOVE SAY. Chromium, 300K characters,
medians of 3, the two arms run back to back in one session, twice each, as-is / on-content /
on-content / as-is:

    arm                          gesture ms  longest stall ms  worst frame ms  settle ms  over
    pointer left at the origin        666.4               7.4            17.9      716.4     0
    pointer over message content      651.0              37.3            29.3      701.0    21
    pointer over message content      655.0              28.0            23.0      705.0    21
    pointer left at the origin        665.5               7.6            17.8      715.7     0

Two of the four portable primaries move and two do not. Longest stall goes 7.4 -> 37.3 and worst
frame 17.9 -> 29.3, so the gutter arm was under-reporting them by 3-4x and 1.5x. `gestureMs`,
`settleMs` and `frames over 33ms` do not move at all, because on this host the boundary work never
costs a whole frame. The pixel-for-pixel gesture really is at the 33.3ms-a-step paint floor either
way; what the pointer buys is jitter under one frame, not a slower scroll.

AND THE 1555ms TABLE ABOVE DID NOT REPRODUCE ON RE-RUN. scroll_predecessor_probe.py, unmodified,
same tree, 300K, medians of 3, on the host these paragraphs were re-measured on:

    arm                gesture ms  longest stall ms  worst frame ms  long tasks  task ms  over
    nothing                 666.7               9.7            18.8           0    127.7     0
    hover_only              667.2              31.5            28.5           0    266.8    20
    gutter_only             666.5               8.1            18.3           0    105.7     0
    hover_then_gutter       666.6               7.9            18.3           0     91.6     0

The MECHANISM reproduces exactly -- twenty steps, twenty boundary events, twenty distinct targets,
every zero-boundary arm on the floor, main-thread task ms doubling -- and the MAGNITUDE does not:
667ms and zero long tasks here against the 1555.5ms and eleven long tasks recorded above. Absolute
milliseconds on this fixture are host-bound, which the paragraph on drift already says; what is new
is that the ratio moved too, so the 2.3x in the tables above is not a constant of the app. The
harness agrees with the probe about it: run unmodified at the revision that introduced the two
tables, the 300K carry-over scroll comes back at 666.5ms with zero long tasks, not the 1467.3ms
and nine long tasks recorded below. Read every millisecond in this docstring as "which arms
differ", never as "by how much". The arms are re-runnable and the numbers should be re-taken on
the machine a claim is being made about.

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

Preconditions are kept, contamination is not. The Shiki settle gate, `expandTools`, the hover that
reveals the action bar before `menu` and `delete`, and the pointer placed on message content before
`scroll` all still run before the action they belong to, on the isolated page as well: they are the
state the action is defined against, not residue from a different action.

READ THE `menu` AND `delete` ROWS WITH THAT IN MIND. Their precondition IS the hover, and the
hover is the thing that costs 890ms of the scroll column. So isolation does not make those two
rows hover-free and was never going to: a menu you have not hovered to reach is not a menu the
user can open. What isolation buys them is freedom from EACH OTHER and from scroll, jump and
reopen, which is real but is less than what it buys `scroll`, `keystroke` and `reopen`, whose
isolated pages never open a menu at all. A future reader looking for the boundary-event cost in
these numbers should NOT look at `scroll` isolated against `scroll` sequenced: both now place the
pointer on the conversation, so that contrast is residue and no longer the pointer. The clean
contrast is scroll_predecessor_probe.py's `gutter_only` against its `hover_only`, which is what
that file is for.

Measured on this tree, chromium, 300K characters, medians of 3, once with the scroll row as the
restructuring first left it and once with the pointer precondition it should have had, same host,
same settings, one run each:

    300K scroll        isolated before  isolated after  carry-over before  carry-over after
    gesture ms                   665.7           652.1              666.5             666.9
    longest stall ms               5.9            24.5               29.0              30.1
    worst frame ms                17.7            19.2               27.7              25.7
    frames over 33ms                 0               0                  0                 0
    settle ms                    715.5           702.1              716.5             716.9
    task ms (chromium)            80.7           225.7              254.6             260.2
    jump painted ms               33.3            33.3               33.1              33.1

Read the two isolated columns first: adding the pointer moves longest stall 5.9 -> 24.5 and
chromium's task ms 80.7 -> 225.7, and moves nothing else. Then read the before row across: the
harness's own two tables disagreed about the SAME gesture by 4.9x on longest stall, 5.9 against
29.0, and that gap was not residue -- the carry-over page had the pointer on a message from
repetition 2 onwards because the previous repetition's `delete` hover put it there, and the
isolated page never had it at all. After the change the two agree, 24.5 against 30.1, which is
what a residue-only difference is supposed to look like.

The gesture itself is 20 steps at the 33.3ms paint floor in every one of those four columns, to
within a few milliseconds. There is nothing in it, and there was nothing in it before either. What
the pointer costs is jitter under one frame, not a slower scroll.

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
HIGHLIGHT_PROBE_MS = int(os.environ.get("SMOKE_HIGHLIGHT_PROBE_MS", "100"))
HIGHLIGHT_GRACE_MS = int(os.environ.get("SMOKE_HIGHLIGHT_GRACE_MS", "1000"))
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
  // Anchoring to the bottom is NOT done here. It is a full reposition, and when a previous
  // repetition or a predecessor arm left the viewport away from the bottom it is a whole extra
  // top-to-bottom scroll. __hv.begin() below already excluded it from the portable recorder, but
  // run_action snapshots the CDP counters and arms the long-task observer before this evaluate
  // starts, so it still landed in this row's task, layout, style and long-task numbers -- and it
  // landed there only for the repetitions that happened to start away from the bottom, which is
  // exactly the asymmetry that makes repetition 1 look cheaper than repetitions 2 and 3. It runs
  // from ACTION_SETUPS instead, before any snapshot is taken.
  const bottom = viewport.scrollHeight - viewport.clientHeight;
  let target = viewport.scrollTop;
  // Reverse at either end rather than stopping. A small thread runs out of travel long before a
  // large one does, and a gesture that covers 2600px at 25K chars and 8000px at 300K is not the
  // same gesture, so the two columns would not be comparable.
  let direction = -1;
  let travelled = 0;
  // Where the viewport ACTUALLY is, read back after every paint. `target` is what was requested.
  // Accumulating from `target` reported the planned distance: Studio replaces assistant-ui's
  // autoscroll with an intent-aware one that can snap a programmatic scroll straight back, and
  // the browser clamps at either end, so a gesture that moved nothing still reported the full
  // 8000px and every completion and cross-arm check downstream passed on it.
  let observed = viewport.scrollTop;
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
    const landed = viewport.scrollTop;
    travelled += Math.abs(landed - observed);
    observed = landed;
    // Planning continues from the REQUEST, not from where it landed, so the gesture keeps its
    // shape. If the viewport is being snapped back, the requests stay the same and `travelled`
    // correctly falls to nothing, which is the signal the validation is looking for.
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
  // Anchored to the bottom by ACTION_SETUPS before run_action took any snapshot, for the same
  // reason the reset at the end of this script runs from ACTION_RESETS after all of them.
  const bottom = viewport.scrollHeight - viewport.clientHeight;
  const startedFrom = viewport.scrollTop;
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
  // The scroll back to the bottom is NOT done here. It closes after __hv.end(), so the portable
  // recorder rows already exclude it, but run_action reads the CDP counters and the long-task
  // observer only once this evaluate returns, so doing it here put a second full-height scroll
  // inside the Chromium-only task, layout, style and long-task rows. Measured at 300K chars the
  // reset costs 33.2ms against the jump's own 33.5ms, so those rows described two jumps. It runs
  // from ACTION_RESETS instead, after every snapshot is taken.
  return {
    paintedMs: Math.round(paintedMs * 10) / 10,
    settleMs: settleMsTaken === null ? null : Math.round(settleMsTaken * 10) / 10,
    // Observed, not planned. Reporting `bottom` here claimed a full-height jump even when the
    // gesture was snapped back or started from somewhere other than the bottom.
    travelledPx: Math.abs(landedAt - startedFrom),
    startedFrom,
    // So a caller can prove the gesture began at the bottom rather than assume it.
    bottom,
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
async ([timeoutMs, settleMs, graceMs, probeEveryMs]) => {
  const api = window.__heavyThread;
  const before = api.messageCount();
  if (!before) return null;
  window.__hv.begin();
  const started = performance.now();
  api.closeThread();
  // Unmount first, or "already back" is indistinguishable from "never left".
  // Counted, not assumed. `growth()` subtracts one paint floor per double-rAF wait a metric is
  // clocked across, and that count was hand-declared per axis in GROWTH_AXES. Reopening is
  // driven by a React state update, so the count check immediately after openThread() always
  // still sees the unmounted tree and the loop always pays at least one __nextPaint() before it
  // can observe the rebuilt messages -- the same floor already subtracted from jump and delete.
  // Reporting it lets the harness check its own declared constant instead of trusting it.
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
  // land inside the lull between two Shiki batches, which stops the clock partway through the
  // rebuild. quietUntilIdle keeps it running until the highlighted token count stops moving.
  const settled = await window.__hv.quietUntilIdle(
    settleMs,
    graceMs,
    () => api.highlightedTokenCount(),
    probeEveryMs,
  );
  const settleMsTaken = settled.settleMs;
  // end(settled.at) trims to the instant the highlighter went quiet, so the frames the
  // settle loop spent watching an already-idle page do not drag the frame stats down.
  const metrics = window.__hv.end(settled.at);
  return {
    ms,
    paintWaits,
    closePaintWaits,
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


def median(values: list[float | None]) -> float | None:
    """Median across the repetitions, or None if any repetition did not produce a number.

    A None here is not a missing reading, it is a repetition in which the thing being timed never
    happened: the menu that never opened inside SETTLE_TIMEOUT_MS, the delete whose message never
    left the DOM, the action that never reached a settled state. Dropping those and taking the
    median of what is left changes the sample population and reports a partially broken action as
    a clean three-repetition measurement -- and it hides it from action_failures(), whose
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


def run_action(
    page,
    cdp,
    name: str,
    script: str,
    arg,
    after_setup = None,
) -> dict:
    """One action, with the portable recorder inside it and the CDP counters bracketing it.

    `after_setup` runs in the one window that is after the anchor and before anything starts
    counting. Anything that has to observe the page as the ACTION will see it, rather than as the
    previous repetition left it, belongs there. Two things already did and both were wrong for the
    same reason: the pointer proof was taken before the anchor, so after repetition 1 it described
    the content under the cursor thousands of px away from where the gesture would actually run;
    and the predecessor probe armed its pointer-boundary counter before the anchor, so an untimed
    full-height reposition contributed boundary events to the measured gesture's count, by an
    amount that depended on where the arm's predecessor had left the viewport. Whatever it returns
    is merged into the row.
    """
    # Precondition first, so it is outside every snapshot below as well as outside the recorder.
    setup = ACTION_SETUPS.get(name)
    if setup is not None:
        page.evaluate(setup)
    extra = after_setup(page) if after_setup is not None else None
    reset_long_tasks(page)
    before = cdp_metrics(cdp)
    raw = page.evaluate(script, arg)
    after = cdp_metrics(cdp)
    if raw is None:
        return {
            "name": name,
            "ran": False,
            **cdp_counters({}, {}),
            **long_task_summary(page),
            **(extra or {}),
        }
    out = {"name": name, "ran": True}
    out.update(raw.pop("metrics"))
    out.update(raw)
    out.update(cdp_counters(before, after))
    out.update(long_task_summary(page))
    out.update(extra or {})
    # Every snapshot is in `out` now, so fixture cleanup cannot land in any of them.
    reset = ACTION_RESETS.get(name)
    if reset is not None:
        page.evaluate(reset)
    return out


# The script and argument for each action, so the isolated runner and the sequenced runner drive
# exactly the same code with exactly the same arguments. Two copies of this list is how the two
# tables would quietly stop being comparable.
# Cleanup that must run AFTER run_action has taken its CDP and long-task snapshots. Anything here
# is restoring the fixture for the next repetition, not part of the action being timed, and the
# recorder window is already closed by the time it runs.
JUMP_RESET_JS = """
async () => {
  const viewport = window.__heavyThread.viewport();
  viewport.scrollTo({ top: viewport.scrollHeight, behavior: "instant" });
  await window.__nextPaint();
}
"""

ACTION_RESETS = {"jump": JUMP_RESET_JS}


# Positioning that must run BEFORE run_action takes its CDP snapshot and arms the long-task
# observer. Anything here is establishing the action's precondition, not part of the action being
# timed. The scroll and jump gestures both have to start from the bottom to travel the distance
# they claim to travel, and reaching the bottom from wherever the last repetition left off is
# itself a full-height scroll.
ANCHOR_BOTTOM_JS = """
async () => {
  const viewport = window.__heavyThread.viewport();
  if (!viewport) return;
  viewport.scrollTo({ top: viewport.scrollHeight - viewport.clientHeight, behavior: "instant" });
  await window.__nextPaint();
}
"""

ACTION_SETUPS = {"scroll": ANCHOR_BOTTOM_JS, "jump": ANCHOR_BOTTOM_JS}


ACTION_SCRIPTS = {
    "keystroke": (KEYSTROKE_JS, KEYSTROKES),
    "scroll": (SCROLL_JS, [SCROLL_STEPS, SCROLL_STEP_PX, SETTLE_TIMEOUT_MS]),
    "jump": (JUMP_JS, SETTLE_TIMEOUT_MS),
    "menu": (MENU_JS, SETTLE_TIMEOUT_MS),
    "delete": (DELETE_JS, SETTLE_TIMEOUT_MS),
    "reopen": (
        REOPEN_JS,
        [SETTLE_TIMEOUT_MS, SETTLE_TIMEOUT_MS, HIGHLIGHT_GRACE_MS, HIGHLIGHT_PROBE_MS],
    ),
}


def drive(
    page,
    cdp,
    name: str,
    after_setup = None,
) -> dict:
    script, arg = ACTION_SCRIPTS[name]
    return run_action(page, cdp, name, script, arg, after_setup = after_setup)


# The gate that says expandTools() really mounted the result panes.
#
# It reads codeExecutionPanes, which is `[data-slot="tool-fallback-content"] pre`, and NOT
# collapsibleOutputs, which is the content element itself. Radix keeps that element in the tree
# for its collapse animation, so it is present while the card is shut: measured on this tree at
# 300K characters, immediately after seeding and BEFORE any expandTools() call, collapsibleOutputs
# was already 22 of the 22 expected while codeExecutionPanes was 0. A `collapsibleOutputs >= n`
# gate is therefore satisfied by a thread of closed cards -- it cannot fail, and it released the
# highlighter wait below before the fences it exists to sequence had mounted. The pane's <pre> is
# a child of that element, so it appears only once the card is really open: 0 collapsed, 22
# expanded, at both sizes on all three engines.
EXPANDED_PANES_GATE_JS = "(n) => window.__heavyThread.counts().codeExecutionPanes >= n"


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
    # EXPAND FIRST, then wait for the highlighter. Radix unmounts collapsed content, so the tool
    # result panes -- which are CODE, two of the seven fences a content cycle produces -- do not
    # exist until expandTools() has run. Waiting for the highlighter before expanding therefore
    # gates on the fences that were already there and then mounts a fresh batch of unhighlighted
    # ones, whose Shiki work lands in whatever is timed next. This is the order measure_cell()
    # seeds in, so the precondition and the seed agree.
    expanded = page.evaluate("() => window.__heavyThread.expandTools()")
    if expanded:
        page.wait_for_function(
            EXPANDED_PANES_GATE_JS,
            arg = expanded,
            timeout = ACTION_TIMEOUT_MS,
        )
    wait_for_highlighting_settled(page, ACTION_TIMEOUT_MS)


POINTER_TARGET_JS = """([x, y]) => {
  const v = window.__heavyThread.viewport();
  const el = document.elementFromPoint(x, y);
  if (!el) return { on_message: false, tag: "-" };
  // The scroller ITSELF is the gutter: a point that hit-tests to the viewport element keeps the
  // same target on every step, which is the cheap arm, not the one this precondition is for.
  if (el === v) return { on_message: false, tag: "viewport" };
  return {
    on_message: Boolean(el.closest('[data-role="assistant"]')),
    tag: el.tagName.toLowerCase(),
  };
}"""


def place_pointer_over_message(page) -> dict:
    """Put the REAL pointer on assistant-message content, which is where a wheel gesture has it.

    A PRECONDITION of `scroll`, in the same sense that `reveal_last_action_bar` is a precondition
    of `menu`: a wheel gesture with no pointer anywhere is not the gesture the row is named for.
    Every isolated page is fresh and Playwright's mouse starts at (0, 0), so without this the
    scroll arm measures a wheel event dispatched at a viewport whose cursor is parked in the
    top-left gutter -- which is `gutter_only` in scroll_predecessor_probe.py, the arm that exists
    precisely to be the artificial control.

    SCROLL_JS dispatches its wheel on the viewport element directly, so the pointer does not steer
    the scroll. What it steers is whether the engine re-hit-tests content moving under a stationary
    cursor and fires pointerover / pointerout at a different element on every step, which is the
    only thing that separates the two arms.

    Measured on this tree, chromium, 300K characters, medians of 3, two arms back to back in one
    session, twice each in the order as-is / on-content / on-content / as-is:

        arm                          gesture ms  longest stall ms  worst frame ms  pointerover
        pointer left at the origin        666.4               7.4            17.9            0
        pointer over message content      651.0              37.3            29.3           21
        pointer over message content      655.0              28.0            23.0           21
        pointer left at the origin        665.5               7.6            17.8            0

    scroll_predecessor_probe.py's own arms, same host, same session length, agree: `nothing` 666.7
    gesture / 9.7 stall / 18.8 worst / 0 boundary events, `hover_only` 667.2 / 31.5 / 28.5 / 20,
    `gutter_only` 666.5 / 8.1 / 18.3 / 0. So the boundary work is real and it is NOT free -- it is
    3 to 4x on longest stall and about 1.5x on worst frame, and chromium's task ms goes 127.7 to
    266.8 -- but on this host it never crosses one frame, so `gestureMs`, `settleMs` and
    `frames over 33ms` do not move. Both readings matter and neither is quotable without the
    other, which is why the row now runs the honest arm and prints its pointer proof beside it.

    The verification is not decoration. A point that lands on the gutter makes this a second copy
    of the cheap arm under the label of the expensive one, so what was hit is recorded per
    repetition and `action_failures` rejects the run rather than publishing the number.
    """
    rect = page.evaluate(
        """() => { const r = window.__heavyThread.viewport().getBoundingClientRect();
            return { x: r.x, y: r.y, w: r.width, h: r.height }; }"""
    )
    got = {"on_message": False, "tag": "-"}
    for down in (0.5, 0.35, 0.65, 0.25, 0.75):
        for across in (0.5, 0.4, 0.6, 0.3):
            x = rect["x"] + rect["w"] * across
            y = rect["y"] + rect["h"] * down
            page.mouse.move(x, y)
            # The action bar is hover-revealed, so give the mount a beat to land OUTSIDE the
            # window the recorder is about to open. It happens once, before a real user's gesture
            # too; charging it to the gesture would be a different error in the same row.
            page.wait_for_timeout(120)
            got = page.evaluate(POINTER_TARGET_JS, [round(x), round(y)])
            if got["on_message"]:
                return {
                    "pointer_on_message": True,
                    "pointer_under": got["tag"],
                    "pointer_at": f"{round(x)},{round(y)}",
                }
    # Recorded, not raised. A raise here is caught by measure_cell as a crashed cell and throws
    # away the five actions that did work; the verdict below fails the run on this field instead
    # and still prints everything else.
    return {"pointer_on_message": False, "pointer_under": got["tag"], "pointer_at": None}


def drive_scroll(page, cdp) -> dict:
    """The scroll action with its pointer precondition, for BOTH runners.

    One function rather than two call sites, because a precondition that runs in the isolated
    table and not in the carry-over one makes the two tables measure different gestures, and the
    difference between those two columns is then part residue and part a different action. Applied
    in both, the columns differ only in residue, which is the one thing the pair is there to show.
    """
    # AFTER the anchor, not before it. ACTION_SETUPS repositions the viewport to the bottom, and
    # from repetition 2 on that is a full-height scroll, so a proof taken beforehand describes the
    # content that happened to be under a stationary cursor thousands of px away from where the
    # gesture then ran. The saved pointer_on_message could read True while the measured wheel
    # actually travelled over a user message, whitespace or the gutter, which turns this arm into
    # a second copy of the cheap control under the expensive one's label.
    return drive(page, cdp, "scroll", after_setup = place_pointer_over_message)


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


# Actions that mutate the seeded thread PERMANENTLY. `delete` takes a message out of the runtime's
# repository, not out of the view, so neither re-opening the thread nor re-expanding its tool cards
# puts it back: without a restore, repetition 2 of the isolated delete arm runs against a thread
# one message shorter and repetition 3 one shorter again. The fixture is whole cycles of one
# message per kind, so each pass deletes a DIFFERENT content kind -- json fence, then image, then
# svg at the smallest size -- and the three timings behind the delete median are three different
# subtrees on three different threads. That is the growth comparison this file exists for.
MUTATING_ACTIONS = ("delete",)


def restore_fixture(page) -> int:
    """Put the seeded thread back. Untimed, and outside every recorder window.

    Runs AFTER the row has been taken, so nothing here is inside the action's portable recorder,
    its CDP counters or its long-task window -- the same rule ACTION_RESETS follows.
    """
    restored = page.evaluate("() => window.__heavyThread.restore()")
    page.wait_for_function(
        "(n) => window.__heavyThread.messageCount() >= n",
        arg = restored,
        timeout = ACTION_TIMEOUT_MS,
    )
    return restored


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
        # The thread this repetition is about to be measured against, read BEFORE the action and
        # carried into the row. It is what makes the restore below an assertion rather than an
        # intention: `action_failures` rejects a table whose repetitions did not all start from
        # the same fixture.
        fixture_messages = page.evaluate("() => window.__heavyThread.messageCount()")
        info(f"      fixture: {fixture_messages} messages at the start of this repetition")
        # `scroll` needs one too, and needed one from the moment each action got its own page. A
        # fresh page has never moved Playwright's mouse, so it sits at (0, 0) -- the scroller's
        # gutter -- and the gesture then measures the arm the probe calls `gutter_only`.
        row = drive_scroll(page, cdp) if name == "scroll" else drive(page, cdp, name)
        row["fixture_messages"] = fixture_messages
        rows.append(row)
        if name in MUTATING_ACTIONS:
            restore_fixture(page)
    return rows


def sequenced_repetition(page, cdp) -> dict[str, dict]:
    """The six actions, once, in the order a user meets them, on one page.

    This is the arrangement that produced every number this file published before the isolation
    change, kept EXACTLY as it was so the carry-over table means what the old table meant. Nothing
    in here is reset between actions on purpose: the residue is the measurement.
    """
    rep: dict[str, dict] = {}
    settle_and_expand(page)
    # The thread this whole repetition is about to run against, read before the first action, for
    # the same reason the isolated runner reads it: it is what turns the restore at the end of the
    # loop into an assertion rather than an intention. `action_failures` rejects a table whose
    # repetitions did not all start from the same fixture, and it can only do that if every row
    # carries the size it started from.
    fixture_messages = page.evaluate("() => window.__heavyThread.messageCount()")
    info(f"      fixture: {fixture_messages} messages at the start of this repetition")
    rep["keystroke"] = drive(page, cdp, "keystroke")
    # The ONE thing in this runner that is not the old sequence, and it is a precondition rather
    # than a reset. Left out, the pointer is at (0, 0) on repetition 1 and over whatever the
    # previous repetition's delete hover left it on from repetition 2 -- two different gestures
    # inside one median, which is the defect class this file was restructured to remove. Nothing
    # else here is reset; the residue is still the measurement.
    rep["scroll"] = drive_scroll(page, cdp)
    rep["jump"] = drive(page, cdp, "jump")
    reveal_last_action_bar(page)
    rep["menu"] = drive(page, cdp, "menu")
    # The scroll position has not moved since the menu, so the hover alone is enough here.
    page.locator('[data-role="assistant"]').last.hover(timeout = ACTION_TIMEOUT_MS)
    rep["delete"] = drive(page, cdp, "delete")
    # `reopen` here measures a thread one message SHORTER than the fixture, and always short of
    # the same kind. `delete` has just taken the last assistant message out of the runtime, reopen
    # deliberately preserves the runtime, and the fixture is whole cycles of one message per kind,
    # so the message that goes missing is always kind 9, the json fence. Measured on chromium at
    # the instant reopen began: at 25K, 19 of 20 messages, 0 of the 1 json fence, and 2520 of 3216
    # highlighted tokens once the rebuild finished, so 21.6% of the highlighting the seed gate
    # passed is not in this column; at 300K, 219 of 220 messages, 10 of 11 fences, 34675 of 35086
    # tokens. Re-seeding between the two moves the 25K figure from 360.5/374.4 ms to 559.6/564.7,
    # 13x the 3.9% base-vs-base spread.
    #
    # Left as it is ON PURPOSE. This table's contract is that nothing is reset between actions, so
    # that the carry-over row means what the old table's row meant, and re-seeding here would
    # silently redefine it. The isolated table is the one to quote for reopen: there, reopen runs
    # on a page that has only ever done reopen, against the full seeded fixture.
    #
    # The per-repetition census cannot catch this, because it is taken after reopen, at which
    # point every repetition agrees on the same reduced fixture.
    rep["reopen"] = drive(page, cdp, "reopen")
    for row in rep.values():
        row["fixture_messages"] = fixture_messages
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


# Numbers that are PROOFS the interaction happened, not timings. Every one of them is aggregated
# into the printed table by median() like any other number, and every one of them is also read by
# `action_failures`, which is the combination that makes a failed repetition invisible: median()
# only propagates a null, and none of these is ever null.
# How much of the requested gesture must actually have happened. Not 1.0: the last step of a
# reversal clamps at a boundary, so a complete gesture can fall a fraction short. Exported and
# used by BOTH the harness below and scroll_predecessor_probe, because two copies of this number
# is how the probe came to accept any travel above zero while the harness required 90%.
SCROLL_TRAVEL_TOLERANCE = 0.9
REQUESTED_SCROLL_PX = SCROLL_STEPS * SCROLL_STEP_PX


def scroll_travel_shortfall(travelled) -> str | None:
    """Why this repetition's travel does not count as the requested gesture, or None."""
    if travelled is None:
        return f"it reported no travel at all against the {REQUESTED_SCROLL_PX}px gesture"
    if travelled < REQUESTED_SCROLL_PX * SCROLL_TRAVEL_TOLERANCE:
        return f"it travelled only {travelled}px of the {REQUESTED_SCROLL_PX}px gesture"
    return None


# How far from the bottom a jump may begin and still be the same gesture as the others. One px of
# rounding, not room for a repetition to start part-way up the thread.
JUMP_ANCHOR_TOLERANCE_PX = 2


def jump_anchor_shortfall(started, bottom) -> str | None:
    """Why this repetition's jump did not begin at the bottom, or None."""
    if started is None or bottom is None:
        return "it did not report where it started, so its length is unverifiable"
    if abs(bottom - started) > JUMP_ANCHOR_TOLERANCE_PX:
        return f"it began {round(bottom - started)}px above the bottom rather than at it"
    return None


NUMERIC_PROOFS = (
    # How far the gesture actually travelled.
    "scrolledPx",
    # Where the jump landed, and how far it had to go.
    "landedAt",
    "travelledPx",
    # Where it BEGAN, and where the bottom was at the time. travelledPx is observed now, so a
    # jump that started part-way up the thread reports a smaller number rather than the full
    # height -- but it can still land at 0 and still cover more than a viewport, which satisfies
    # every other check while contributing a shorter gesture to the median. Kept per repetition
    # because a median cannot see one short repetition among complete ones.
    "startedFrom",
    "bottom",
    # That the popover that opened had something in it, and that a bar was mounted under the
    # pointer at all.
    "itemsWhileOpen",
    "triggersWhileHovered",
    # The message count either side of a delete or a re-open.
    "before",
    "after",
    # The size of the thread this repetition was measured against, recorded before the action ran.
    "fixture_messages",
)


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
                # `value is None` is deliberately a key too. A timing that came back null in every
                # repetition would otherwise be absent from the merged row entirely, and the
                # verdict's `menu["openMs"] is None` would raise KeyError instead of reporting the
                # menu that never opened. Present-and-None is the readable form of that.
                if value is None or (
                    isinstance(value, (int, float)) and not isinstance(value, bool)
                ):
                    numeric_keys.add(key)
        for key in sorted(numeric_keys):
            merged[key] = median([r.get(key) for r in rows])
        # Values that are not numbers are proofs the action really happened, not timings, so they
        # cannot be aggregated. The last repetition's is kept for the table, and EVERY
        # repetition's is kept beside it, because these are the only evidence the interaction
        # happened at all: a repetition whose proof failed still put its timing into the median
        # above, so a verdict that reads rows[-1] alone passes on a median that carries an
        # interaction which did not occur. The numeric counterpart is median() above, which
        # returns None the moment any one repetition does rather than taking the median of the
        # rest; that is what keeps a headline timeout from being reported as a median of three.
        for key in (
            "domText",
            "runtimeText",
            "bodyPointerEvents",
            "bodyPointerEventsAfterClose",
            # Where the real pointer was for each scroll gesture. Same reason as the four above:
            # a repetition whose pointer never reached message content measured the gutter arm
            # and still put its timing into the median.
            "pointer_on_message",
            "pointer_under",
            "pointer_at",
        ):
            if key in rows[-1]:
                merged[key] = rows[-1][key]
            values = [r.get(key) for r in rows]
            if any(value is not None for value in values):
                merged[f"{key}_per_repetition"] = values
        # The NUMERIC proofs, which are counts and positions rather than timings, and which
        # median() cannot protect because none of them is ever null: a repetition that failed
        # reports a NUMBER that happens to be the wrong one. Collapsing them to a median is the
        # numeric half of the defect the loop above fixes for the non-numeric proofs. Measured
        # shape of it: a jump that landed at [0, bottom, 0] has a median of 0, which is exactly
        # what "the jump arrived" looks like, so the repetition that never moved passes the
        # verdict while its timing stays in the published median. Same for a delete whose count
        # did not drop on one pass, a scroll that travelled nothing on one pass, and a menu that
        # opened empty on one pass. The median stays -- it is what the table prints -- and every
        # repetition is kept beside it for `action_failures` to read.
        for key in NUMERIC_PROOFS:
            values = [r.get(key) for r in rows]
            if any(value is not None for value in values):
                merged[f"{key}_per_repetition"] = values
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
        # Severity kept SEPARATE from the warning list. The allowance below exists for Gecko's
        # two scroll-anchoring notices, which are the engine describing itself. An application
        # exception is not chatter: one console.error or an uncaught pageerror inside a measured
        # interaction means the interaction did not do what the row says it did, and sharing one
        # list let exactly one such error sit under the "> 4" threshold and the run exit 0.
        self.console_errors: list[str] = []
        self.page.on(
            "request",
            lambda r: self.stray_requests.append(r.url) if r.url.startswith(api_prefix) else None,
        )
        self.page.on(
            "console",
            lambda m: (
                self.console_errors.append(m.text[:200])
                if m.type == "error"
                else self.console_warnings.append(m.text[:200])
                if m.type == "warning"
                else None
            ),
        )
        self.page.on("pageerror", lambda e: self.console_errors.append(f"pageerror: {e}"[:200]))
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
            EXPANDED_PANES_GATE_JS,
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
        # Answered locally by the page's allowlist rather than reaching the network. Recorded and
        # printed rather than silently swallowed: two whole-endpoint GETs per reopen is a real
        # cost and stays visible even though it is kept out of the timed region.
        self.record["stubbed_api_requests"] = len(self.page.evaluate("window.__stubbedApi || []"))
        self.record["seed_console_warnings"] = len(self.console_warnings)
        self.record["first_seed_warning"] = (
            self.console_warnings[0] if self.console_warnings else "-"
        )
        self.record["seed_console_errors"] = len(self.console_errors)
        self.record["first_seed_error"] = self.console_errors[0] if self.console_errors else "-"
        self.stray_requests.clear()
        self.console_warnings.clear()
        self.console_errors.clear()

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
        self.record["console_errors"] = len(self.console_errors)
        self.record["first_console_error"] = self.console_errors[0] if self.console_errors else "-"
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
            # Between REPETITIONS, never between actions. `delete` removes an assistant message
            # from the runtime's repository, which re-opening the thread does not undo, so without
            # this repetition 2 started one message short and repetition 3 two short -- and the
            # fixture is whole cycles of one message per kind, so each pass took a different kind
            # off the end. The medians and the repetition-divergence report then conflated
            # carry-over, which is what this table exists to measure, with a progressively smaller
            # thread, which is not. The delete before `reopen` INSIDE a sequence is untouched:
            # that residue is the measurement and restoring it would redefine this table.
            if index + 1 < REPEATS:
                restore_fixture(seeded.page)
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
        # The floor MEASURED ON THE PAGE THAT PRODUCED EACH ACTION'S timing. In the
        # isolated table every action runs on its own page, so there are seven floors and
        # the median above belongs to none of them in particular. growth() subtracts a
        # floor per double-rAF wait, so subtracting the median from an action measured on
        # a page whose own floor was higher or lower moved the corrected endpoints, and
        # with them the discrimination ratio. paint_floor_spread_ms exposed that the
        # pages disagreed but did nothing about it.
        "paint_floor_ms_by_action": {
            s["arm"].split(":", 1)[1]: s["paint_floor_ms"]
            for s in seeds
            if s.get("arm", "").startswith("isolated:")
        },
        "seed_api_requests": sum(s["seed_api_requests"] for s in seeds),
        "stubbed_api_requests": sum(s.get("stubbed_api_requests", 0) for s in seeds),
        "seed_console_warnings": max(s["seed_console_warnings"] for s in seeds),
        "first_seed_warning": seed_warned[0]["first_seed_warning"] if seed_warned else "-",
        # SUMMED, not maxed: an exception on any one of the seven pages is a defect on that page,
        # and taking the max would let six clean pages hide it behind a seventh.
        "seed_console_errors": sum(s.get("seed_console_errors", 0) for s in seeds),
        "first_seed_error": next(
            (s["first_seed_error"] for s in seeds if s.get("seed_console_errors")), "-"
        ),
        "raf_callbacks": sum(s["raf_callbacks"] for s in seeds),
        "stray_api_requests": sum(s["stray_api_requests"] for s in seeds),
        "console_warnings": max(s["console_warnings"] for s in seeds),
        "first_console_warning": warned[0]["first_console_warning"] if warned else "-",
        "console_errors": sum(s.get("console_errors", 0) for s in seeds),
        "first_console_error": next(
            (s["first_console_error"] for s in seeds if s.get("console_errors")), "-"
        ),
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
    ("stubbed api requests", lambda r: r.get("stubbed_api_requests", 0)),
    ("seed console warnings", lambda r: r["seed_console_warnings"]),
    ("first seed warning", lambda r: short(r["first_seed_warning"])),
    ("action api requests", lambda r: r["stray_api_requests"]),
    ("action console warnings", lambda r: r["console_warnings"]),
    ("seed console errors", lambda r: r.get("seed_console_errors", 0)),
    ("action console errors", lambda r: r.get("console_errors", 0)),
    ("first action error", lambda r: r.get("first_console_error", "-")),
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
    ("scroll pointer on message", _action("scroll", "pointer_on_message")),
    ("scroll pointer under", _action("scroll", "pointer_under")),
    ("scroll pointer at", _action("scroll", "pointer_at")),
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

# The numeric proofs, per repetition, printed beside the median of each above. A median of a proof
# is what hides a failed repetition -- [0, bottom, 0] medians to 0, which is what an arrived jump
# looks like -- so the spread the verdict reads is in the table rather than only in the JSON.
NUMERIC_PROOF_ROWS = (
    ("scroll", "scrolledPx", "scroll px per repetition"),
    ("jump", "landedAt", "jump landed at per repetition"),
    ("jump", "travelledPx", "jump px per repetition"),
    ("menu", "itemsWhileOpen", "menu items per repetition"),
    ("menu", "triggersWhileHovered", "menu triggers per repetition"),
    ("delete", "before", "delete messages before per repetition"),
    ("delete", "after", "delete messages after per repetition"),
    ("reopen", "before", "reopen messages before per repetition"),
    ("reopen", "after", "reopen messages after per repetition"),
) + tuple((_name, "fixture_messages", f"{_name} fixture per repetition") for _name in ACTIONS)

TABLE_ROWS = TABLE_ROWS + tuple(
    (
        _label,
        lambda r, _a = _act, _k = _key: format_repetitions(r["actions"][_a][f"{_k}_per_repetition"]),
    )
    for _act, _key, _label in NUMERIC_PROOF_ROWS
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


# Growth axes: the whole point of the harness is that these rise with content. The third field is
# HOW MANY double-rAF waits the metric is clocked across; each one carries its own ~33ms vsync
# floor, and left in, that floor compresses every ratio towards 1 and lets a real regression sit
# under the threshold. `menu open+close ms` is the sum of two independently floored timings, so it
# carries two.
GROWTH_AXES = tuple(
    [(f"{a} longest stall ms", _action(a, "longest_stall_ms"), 0) for a in ACTIONS]
    + [(f"{a} worst frame ms", _action(a, "worst_frame_ms"), 0) for a in ACTIONS]
    + [(f"{a} frames over 33ms", _action(a, "frames_over_33"), 0) for a in ACTIONS]
    # The floor is READ from the row, not declared: every one of these windows crosses a
    # different number of mandatory double-rAF waits, and declaring 0 for all of them left
    # roughly `paint_waits * paint_floor_ms` of constant baseline in both ends of the ratio.
    # MENU_JS is the clearest case: it opens the recorder before opening the menu and closes it
    # after closing it, so it crosses the same two waits `menu open+close ms` correctly declares,
    # and `menu wall ms` was declaring none of them.
    + [(f"{a} wall ms", _action(a, "wall_ms"), _floor_from(a, "paint_waits")) for a in ACTIONS]
    + [
        # The rule for the entries below: an axis measured from `__hv.startedAt` spans the WHOLE
        # recorder window, so it carries every double-rAF wait in it and takes the measured
        # `paint_waits`. An axis measured from a later mark carries only its own and keeps a
        # declared count. Both kinds are here on purpose and the difference is not cosmetic.
        #
        # gestureMs is `performance.now() - __hv.startedAt`, and BOTH settle figures come from
        # `quiet()` / `quietUntilIdle()`, which return `... - this.startedAt` rather than the time
        # they themselves took. All three therefore contained the scroll's twenty paint waits and
        # declared none of them, leaving ~20 vsync floors in both ends of those ratios, which
        # compresses them hard enough to report a real size-dependent regression as flat.
        #
        # Counted at runtime rather than written in: the twenty come from a LOOP, so the literal
        # `__nextPaint()` count in the source is one, and any hand-declared number here would have
        # been wrong in the same way the old zero was.
        ("keystroke median ms", _action("keystroke", "median_sample_ms"), 1),
        ("scroll gesture ms", _action("scroll", "gestureMs"), _floor_from("scroll", "paint_waits")),
        ("scroll settle ms", _action("scroll", "settleMs"), _floor_from("scroll", "paint_waits")),
        # NOT paint_waits: paintedMs starts at a mark taken after begin() and spans one wait,
        # while the jump's window holds two. Using the window count here would subtract a floor
        # the number never contained.
        ("jump painted ms", _action("jump", "paintedMs"), 1),
        ("jump settle ms", _action("jump", "settleMs"), _floor_from("jump", "paint_waits")),
        # Also NOT paint_waits: MENU_JS awaits no paint at all, and its two floors come from
        # settle() reading the pre-MutationObserver state on entry, once for open and once for
        # close. The window count is zero here and would remove a floor that is really there.
        ("menu open+close ms", _action("menu", "open_close_ms"), 2),
        ("delete ms", _action("delete", "ms"), 1),
        # 1, not 0: see paintWaits in REOPEN_JS. Leaving it at 0 left a full ~33ms vsync floor
        # of constant baseline in both ends of the ratio, which compresses it towards 1 and can
        # report a real reopen curve as flat when the smallest fixture rebuilds near the floor.
        ("reopen ms", _action("reopen", "ms"), 1),
    ]
)
# A ratio at or below this from the smallest size to the largest means the axis did not respond
# to twelve times the content. That is not a flat curve, it is an axis that is not measuring the
# thing being varied.
DISCRIMINATION_RATIO = float(os.environ.get("SMOKE_DISCRIMINATION_RATIO", "1.5"))
# What a counter that starts at zero has to REACH before its rise counts as an answer. A ratio
# cannot be formed against zero, so DISCRIMINATION_RATIO does not apply to these axes at all and
# something absolute has to. 5 because the counters this covers are dropped frames and long
# tasks: at twelve times the content a real curve produces them in quantity, while one or two is
# what an unloaded machine produces on its own, and the CI configuration runs a single repetition
# so there is no median to average that away.
ZERO_BASED_MIN_RISE = int(os.environ.get("SMOKE_ZERO_BASED_MIN_RISE", "5"))
# Which axes are COUNTS. Stated, not inferred. The zero branch below used to key on `floored`,
# which only identifies a timing that had a paint floor subtracted; an UNFLOORED timing such as
# `longest stall ms` or `worst frame ms` is zero at the smallest size whenever the action ends
# before the recorder produces a sample, and it was then treated as a dropped-frame counter, so a
# noisy 5ms at the largest size read as a rise of 5 and discriminated. `harness_failures` accepts
# any one discriminating axis, so that stray millisecond could carry the run.
#
# A count is a count of events. Only `frames over 33ms` is one; every other axis is milliseconds.
COUNTER_AXES = frozenset(f"{a} frames over 33ms" for a in ACTIONS)
# Engine chatter is tolerated up to this many warnings per size. Gecko's two scroll-anchoring
# notices are what this number exists for; anything the app emits per message would be two orders
# of magnitude above it at 220 messages.
CONSOLE_WARNING_ALLOWANCE = int(os.environ.get("SMOKE_CONSOLE_WARNING_ALLOWANCE", "4"))


def resolve_floor(floored, row: dict) -> float:
    """The floor count for one row, as an INT.

    `floored` may be a callable, and the growth report is written to JSON at the end of the run.
    Putting the callable itself in the report made `json.dumps` raise `Object of type function is
    not JSON serializable`, which failed every complete run AFTER all the measurements were taken.
    Nothing in the unit tests caught it because none of them serialise the report.
    """
    # NOT int(). `summarise` takes a median across repetitions, so an even-repetition run whose
    # repetitions paid 1 and 2 waits reports 1.5, and truncating that to 1 left half a vsync floor
    # in the wall-clock axis and published a distorted ratio. The documented two-repetition
    # configurations are exactly the ones that produce halves. A float serialises fine.
    value = floored(row) if callable(floored) else floored
    return value if isinstance(value, (int, float)) else 0


def action_floor(row: dict, action: str | None) -> float:
    """The paint floor to subtract for one axis: the action's own page, or the median.

    The median is the fallback for axes that are not per-action (and for the sequenced table,
    where every action really did run on one page and therefore shares one floor).
    """
    if action:
        by_action = row.get("paint_floor_ms_by_action") or {}
        if action in by_action:
            return by_action[action]
    return row.get("paint_floor_ms", 0)


def growth(
    cells: dict, pick, floored, sizes: list[int], action: str | None = None
) -> tuple[float | None, float | None]:
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
                value -= count * action_floor(row, action)
            values.append(round(value, 2))
        return values[0], values[1]
    except (KeyError, TypeError):
        return None, None


def axis_action(axis_name: str) -> str | None:
    """The action an axis belongs to, or None for the axes that are not per-action.

    Axis names for per-action metrics are built as f"{action} ...", so the action is the leading
    word. Matched against ACTIONS rather than split blindly, so a future axis whose first word
    happens to collide with nothing still returns None instead of a bogus key.
    """
    head = axis_name.split(" ", 1)[0]
    return head if head in ACTIONS else None


def report_growth(results: dict) -> dict[str, dict[str, dict]]:
    """Per engine, per axis: the value at the smallest and largest size and their ratio."""
    report: dict[str, dict[str, dict]] = {}
    for engine in results["engines"]:
        cells = results["by_engine"][engine]["by_size"]
        per_axis: dict[str, dict] = {}
        for name, pick, floored in GROWTH_AXES:
            action = axis_action(name)
            small, large = growth(cells, pick, floored, results["sizes"], action)
            # Resolved here, once, so what lands in the JSON is the count that was actually
            # subtracted at each end rather than the thing that computes it.
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
                        "floored": floor_counts,
                    }
                    continue
                # `large > small` is not enough. These are counts of missed frames and long
                # tasks, and in the one-repetition Chromium configuration the CI workflow runs
                # there is no median to smooth them: a single incidental dropped frame takes an
                # axis from 0 to 1, which was marked as discriminating no matter what
                # SMOKE_DISCRIMINATION_RATIO said, because a ratio was never computed for it.
                # `harness_failures` accepts any ONE discriminating axis, so that stray frame
                # could carry the whole verdict while every latency axis was flat or broken.
                if name not in COUNTER_AXES:
                    # A timing that reads zero at the smallest size did not "grow from nothing",
                    # it resolved below what the recorder can see. ZERO_BASED_MIN_RISE is a count
                    # of events and means nothing applied to milliseconds.
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
            # The noise floor applies to a counter whatever its baseline. A dropped-frame count
            # going 1 -> 2 is a ratio of 2.0 and cleared DISCRIMINATION_RATIO, and since
            # harness_failures accepts any single discriminating axis, that one incidental frame
            # could carry the CI smoke while every latency axis was flat. A ratio is only
            # meaningful once there are enough events for the ratio to be about the content
            # rather than about one frame either way.
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


# The declared double-rAF count for each axis whose action reports how many it actually paid.
# GROWTH_AXES holds the declaration; the action holds the observation; a harness that declares a
# floor it does not pay, or pays one it does not declare, subtracts the wrong constant from both
# ends of every ratio it publishes. Only reopen reports its waits today, so only reopen is
# checkable here; the entry exists so adding a counter to another action wires it in by name.
# axis name in GROWTH_AXES -> (action, the field on that action reporting its own wait count)
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
    """Axes whose subtracted paint floor does not match the waits the action actually paid."""
    problems: list[str] = []
    for engine in results["engines"]:
        for size in results["sizes"]:
            row = results["by_engine"][engine]["by_size"].get(str(size), {})
            if "crashed" in row:
                continue
            for axis_name, (action, counter) in FLOOR_COUNTERS.items():
                measured = row.get("actions", {}).get(action) or {}
                # An action that did not run is already reported by harness_failures, with the
                # reason. Reporting it again here as an unverified floor would be a second
                # failure for one cause, and would bury the real one.
                if not measured.get("ran", True):
                    continue
                observed = measured.get(counter)
                if observed is None:
                    problems.append(
                        f"{engine} at {size} chars recorded no {counter} for {action}, so the "
                        f"paint floor subtracted from '{axis_name}' is unverified"
                    )
                    continue
                declared = declared_floor(axis_name)
                if declared == observed:
                    continue
                problems.append(
                    f"{engine} at {size} chars paid {observed} paint wait(s) in {action} but "
                    f"GROWTH_AXES subtracts {declared} from '{axis_name}'; the ratio is computed "
                    "after removing the wrong constant from both ends"
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
            # NO allowance. The paragraph above is about engine chatter, which is the engine
            # describing itself; an application exception is not that. One console.error or one
            # uncaught pageerror inside a measured interaction means the interaction did not do
            # what the row says it did, so its timing is not a measurement of the labelled thing.
            # Sharing one list with the warnings let exactly one such error sit under the "> 4"
            # threshold and the run exit 0 with the timings published.
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
            layers = set()
            for size in results["sizes"]:
                cell = results["by_engine"][engine]["by_size"][str(size)]
                if "crashed" in cell:
                    continue
                menu_row = as_table(cell, table)["actions"].get("menu", {})
                # Every repetition, not the collapsed last one: a run that opened a modal menu on
                # repetition 1 and a non-modal one on repetition 3 has measured two mechanisms
                # under one median, and reading the last repetition alone cannot see it.
                layers.update(
                    menu_row.get(
                        "bodyPointerEvents_per_repetition", [menu_row.get("bodyPointerEvents")]
                    )
                )
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
            continue
        # Every repetition has to have started from the same thread, or the row is N measurements
        # of N different fixtures behind one median. `delete` is the action that drifts on its
        # own: it removes a message from the runtime's REPOSITORY, which re-opening the thread
        # does not undo, and the fixture is whole cycles of one message per kind, so each pass
        # takes a different content kind off the end of a shrinking thread.
        seen = actions[name].get("fixture_messages_per_repetition")
        if seen and len(set(seen)) > 1:
            failures.append(
                f"{where} ran the {name} action against {seen} messages across its repetitions; "
                "the fixture was not restored between them, so its median spans several threads"
            )
    # A null settle time is the settle loop giving up: the page never produced a calm window
    # inside SETTLE_TIMEOUT_MS. It is NOT "this engine does not report that", but it prints as the
    # same `-`, and the axis it feeds merely becomes "not recorded" -- so another axis can carry
    # the discrimination check and the run exits 0 having timed out without measuring settlement.
    # Reachable only now that median() propagates a null repetition instead of averaging it away.
    for name in ("scroll", "jump", "reopen"):
        settling = actions[name]
        if settling.get("ran") and settling.get("settleMs") is None:
            failures.append(
                f"{where} ran the {name} action but it never reached a settled state within "
                f"{SETTLE_TIMEOUT_MS}ms, so its settle time and the frame counts beside it are "
                "the timeout rather than a measurement"
            )
    keystroke = actions["keystroke"]
    if keystroke.get("ran"):
        # The DOM value is what the harness itself wrote, so it proves nothing on its
        # own. Only the runtime's copy shows the keystroke reached React rather than just
        # the textarea, and a keystroke that reached nothing still reports the ~33ms
        # paint floor, which reads as a plausible timing.
        # Per repetition, not just the last one. An earlier repetition that typed into
        # a dead composer contributed its timing to the median all the same.
        dom_texts = keystroke.get("domText_per_repetition", [keystroke["domText"]])
        runtime_texts = keystroke.get("runtimeText_per_repetition", [keystroke["runtimeText"]])
        for index, (dom, runtime) in enumerate(zip(dom_texts, runtime_texts)):
            if runtime != dom:
                failures.append(
                    f"{where} typed {dom!r} into the DOM on repetition {index + 1} but "
                    f"the runtime holds {runtime!r}; the keystroke never reached the "
                    "composer state, and that repetition is inside the median"
                )
    # Sitting on the paint floor is NOT a harness failure here, and the reason is a
    # finding rather than an excuse: the character reaches the composer and paints on
    # the very next frame at every size, while the thread churns for another 180ms
    # afterwards. `runtimeText == domText` above is what proves the keystroke landed;
    # the floor comparison only says this particular axis has no room to move, which
    # the growth report states per axis.
    scroll = actions["scroll"]
    # The pointer, per repetition, not on the collapsed last one. A repetition whose pointer never
    # reached message content measured the gutter arm -- which sits 3 to 4x lower on longest stall
    # and about 1.5x lower on worst frame, measured on this tree -- and still put its timing into
    # the median under a row labelled `scroll`.
    if scroll.get("ran"):
        off = [
            index + 1
            for index, value in enumerate(
                scroll.get("pointer_on_message_per_repetition", [scroll.get("pointer_on_message")])
            )
            if value is not True
        ]
        if off:
            failures.append(
                f"{where} ran the scroll gesture with the pointer off message content on "
                f"repetition(s) {off} (it hit {scroll.get('pointer_under')!r}); that is the "
                "gutter arm, not a user wheel-scrolling a conversation, and it under-reports "
                "longest stall and worst frame"
            )
    # Equal travel at every size or the columns are not the same gesture. Per repetition, because
    # the distance is a NUMBER in every repetition rather than a null in the bad one, so median()
    # cannot see it: [8000, 0, 8000] has a median of 8000, and the repetition whose viewport never
    # moved is inside the published gesture time.
    if scroll.get("ran"):
        for index, travelled in enumerate(
            scroll.get("scrolledPx_per_repetition", [scroll.get("scrolledPx")])
        ):
            shortfall = scroll_travel_shortfall(travelled)
            if shortfall:
                failures.append(
                    f"{where} {shortfall} on repetition {index + 1}, so its scroll column is not "
                    "comparable with the others"
                )
    jumped = actions["jump"]
    if jumped.get("ran"):
        # The anchor ACTION_SETUPS applies can itself be clamped or snapped back. When it is, the
        # jump starts part-way up, still lands at 0 and still covers more than a viewport, so
        # every other check here passes while a shorter gesture goes into the median.
        starts = jumped.get("startedFrom_per_repetition", [jumped.get("startedFrom")])
        bottoms = jumped.get("bottom_per_repetition", [jumped.get("bottom")])
        # zip() would silently truncate to the shorter list and leave the remaining repetitions
        # unchecked, which is the same "the check quietly stopped checking" failure this whole
        # section exists to prevent. Pad instead, so a missing entry reads as unverifiable.
        width = max(len(starts), len(bottoms))
        starts = list(starts) + [None] * (width - len(starts))
        bottoms = list(bottoms) + [None] * (width - len(bottoms))
        for index, (started, bottom) in enumerate(zip(starts, bottoms)):
            shortfall = jump_anchor_shortfall(started, bottom)
            if shortfall:
                failures.append(
                    f"{where} jumped on repetition {index + 1} but {shortfall}, so that "
                    "repetition is a shorter gesture than the ones it is aggregated with"
                )
        # Unlike the gesture, the jump is DELIBERATELY not the same distance at every
        # size: it is bottom to top, which is the point. What has to hold is that it
        # arrived, or the column is timing a scroll that did not move.
        #
        # PER REPETITION, and this is the clearest case of why. A jump that landed at
        # [0, bottom, 0] has a median of 0, which is the exact value "the jump arrived"
        # produces, so the repetition that never left the bottom passes this check while
        # its timing stays in the published median. The landing is a number in every
        # repetition, so median()'s null guard never fires on it.
        for index, landed in enumerate(
            jumped.get("landedAt_per_repetition", [jumped.get("landedAt")])
        ):
            if landed is None or landed > 1:
                failures.append(
                    f"{where} jumped to the top of the thread and landed at {landed}px on "
                    f"repetition {index + 1}; the viewport did not move"
                )
        for index, travelled in enumerate(
            jumped.get("travelledPx_per_repetition", [jumped.get("travelledPx")])
        ):
            if travelled is None or travelled <= viewport["clientHeight"]:
                failures.append(
                    f"{where} had only {travelled}px to jump through on repetition "
                    f"{index + 1}, which is less than one viewport; nothing had to be painted"
                )
    menu = actions["menu"]
    if menu.get("ran"):
        if menu["openMs"] is None:
            failures.append(f"{where} never opened the message action menu")
        elif menu["closeMs"] is None:
            failures.append(f"{where} opened the action menu and it never closed")
        elif "none" in menu.get(
            "bodyPointerEventsAfterClose_per_repetition",
            [menu["bodyPointerEventsAfterClose"]],
        ):
            # Every repetition, not the collapsed last one. A repetition that left the body stuck
            # on the modal layer still put its timing into the median, so reading the last one
            # alone passes a median that carries a run the verdict is meant to reject.
            stuck = [
                index + 1
                for index, value in enumerate(
                    menu.get(
                        "bodyPointerEventsAfterClose_per_repetition",
                        [menu["bodyPointerEventsAfterClose"]],
                    )
                )
                if value == "none"
            ]
            failures.append(
                f"{where} left the body on the modal layer after closing the menu on "
                f"repetition(s) {stuck}"
            )
        else:
            # An empty popover satisfies "the menu opened" and costs nothing to render. Per
            # repetition: a count of [5, 0, 5] has a median of 5, so the pass that opened an
            # empty popover is invisible and its cheap timing is inside the median.
            empty = [
                index + 1
                for index, value in enumerate(
                    menu.get("itemsWhileOpen_per_repetition", [menu.get("itemsWhileOpen")])
                )
                if not value
            ]
            if empty:
                failures.append(
                    f"{where} opened an action menu with no items in it on repetition(s) {empty}"
                )
        bare = [
            index + 1
            for index, value in enumerate(
                menu.get("triggersWhileHovered_per_repetition", [menu.get("triggersWhileHovered")])
            )
            if not value
        ]
        if bare and counts["actionBars"] <= 0:
            failures.append(
                f"{where} mounted no action bar at rest and none under the pointer either on "
                f"repetition(s) {bare}"
            )
    deleted = actions["delete"]
    if deleted.get("ran"):
        if deleted["ms"] is None:
            failures.append(f"{where} never deleted a message")
        else:
            # Per repetition, and the pair together. Both counts are medianed independently, so
            # [20, 19, 19] before and [19, 18, 19] after report 19 and 18 -- a clean drop -- while
            # repetition 3 deleted nothing at all. Neither number is ever null, so median()'s
            # guard does not reach this.
            befores = deleted.get("before_per_repetition", [deleted.get("before")])
            afters = deleted.get("after_per_repetition", [deleted.get("after")])
            for index, (before, after) in enumerate(zip(befores, afters)):
                if before is None or after is None or after >= before:
                    failures.append(
                        f"{where} clicked delete and the message count did not drop on "
                        f"repetition {index + 1} ({before} -> {after})"
                    )
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
