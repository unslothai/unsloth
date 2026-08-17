// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The parts of the progressive mount that are not pure, pinned by source assertion because they
// live in .tsx files and node's type stripping cannot import one (same reason and same shape as
// chat-autoscroll-frame-budget.test.ts).
//
// Each of these is a real invariant with a real failure behind it, not a spelling check. The
// failure is named in the test.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const read = (path: string): string =>
  readFileSync(new URL(path, import.meta.url), "utf8");

/**
 * Source with comments removed. Every "this must not appear" assertion below runs against this
 * rather than the raw file: these files explain at length what they deliberately do NOT do, and a
 * prose mention of the banned construct must not fail its own test.
 */
const code = (source: string): string =>
  source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^[ \t]*\/\/.*$/gm, "");

const GLUE = code(read("../src/components/assistant-ui/progressive-messages.tsx"));
const HOOK = code(
  read("../src/components/assistant-ui/use-intent-aware-autoscroll.tsx"),
);
const THREAD = code(read("../src/components/assistant-ui/thread.tsx"));

/**
 * The region of `source` between two markers, with BOTH markers required.
 *
 * Every slice below used to be `source.slice(source.indexOf(marker))`, which is the single
 * sharpest edge in a file of source assertions: `indexOf` returns -1 for a marker that has been
 * renamed, `slice(-1)` is the last character of the file, and every assertion against that region
 * then fails for the wrong reason -- or, for a slice with an end marker that moved, silently
 * widens to the rest of the file and passes for the wrong reason. Asserting the markers exist
 * turns both of those into a named failure.
 */
const section = (source: string, from: string, to: string): string => {
  const start = source.indexOf(from);
  assert.notEqual(start, -1, `source assertion anchor is gone: ${from}`);
  const end = source.indexOf(to, start + from.length);
  assert.notEqual(end, -1, `source assertion anchor is gone: ${to}`);
  return source.slice(start, end);
};

/**
 * The mount window and the row map, without the one-off scroll-anchoring feature probe above
 * them. The "this must never appear" assertions below are all about what the WINDOW does to the
 * thread's viewport, and the probe legitimately does several of those things to a 50x50 element
 * of its own that is never in the thread: it hides it, it scrolls it, and it inserts into it.
 */
const GLUE_WINDOW = () =>
  section(
    GLUE,
    "function useProgressiveMountWindow(",
    "ProgressiveMessages.displayName",
  );

test("the thread renders rows through MessageByIndex, never ThreadPrimitive.Messages", () => {
  // ThreadPrimitive.Messages renders MessageByIndexProvider -> RenderChildrenWithAccessor ->
  // ThreadMessageComponent; MessageByIndex drops the middle one. Rendering Messages while the
  // window is closed and MessageByIndex while it is open would change the element type at that
  // position on the convergence commit, and React would unmount and rebuild every message in the
  // thread, which is the exact cost this change exists to avoid.
  // Matched as JSX rather than as bare text: the file's own header explains at length why the
  // swap is permanent, and naming the primitive there must not fail its own test.
  assert.match(GLUE, /<ThreadPrimitive\.MessageByIndex/);
  assert.doesNotMatch(
    GLUE,
    /<ThreadPrimitive\.Messages\b/,
    "the row map must not switch primitives on convergence",
  );
  assert.doesNotMatch(
    THREAD,
    /<ThreadPrimitive\.Messages\b/,
    "Thread must render the progressive list, not the unbounded one",
  );
  assert.match(THREAD, /<ProgressiveMessages/);
});

test("rows outside the window are not rendered rather than rendered and hidden", () => {
  // The cost being avoided is mounting them. A display:none row costs the same to mount, so a
  // rewrite to hiding would keep the tests green and collect nothing.
  const glue = GLUE_WINDOW();
  assert.doesNotMatch(glue, /display:\s*["']?none/);
  assert.doesNotMatch(glue, /visibility:\s*["']?hidden/);
  assert.doesNotMatch(glue, /content-visibility/);
});

test("the window is only ever advanced through widen", () => {
  // setMountWindow is allowed to set the initial window, to clear it (null), or to call widen.
  // Anything else could move `start` upward, which would unmount a message that had been
  // mounted, which is the failure mode the whole design rules out.
  const calls = [...GLUE.matchAll(/setMountWindow\(([^;]*?)\)\s*[;,]/gs)].map(
    (m) => m[1].trim(),
  );
  assert.ok(
    calls.length >= 4,
    `expected several setMountWindow calls, found ${calls.length}`,
  );
  for (const call of calls) {
    const ok =
      call === "null" ||
      call.startsWith("initialWindow(") ||
      call.includes("widen(current, count)");
    assert.ok(
      ok,
      `setMountWindow(${call}) is not one of null / initialWindow / widen`,
    );
  }
});

test("a run drops the window", () => {
  // Streaming and widening both move the scroll position, and a reply must never commit into a
  // tree that has not reached it. Two gates: the reactive one for a run that starts after the
  // window opened, and the re-check inside the frame for one that starts between commit and rAF.
  assert.match(GLUE, /thread\.isRunning/);
  assert.match(
    GLUE,
    /if \(threadIsRunning && mountWindow != null\) setMountWindow\(null\)/,
  );
  assert.match(GLUE, /if \(isRunningNow\(\)\) \{\s*setMountWindow\(null\);/);
});

test("the widening step is deferred and transition-wrapped", () => {
  // Without requestAnimationFrame the chunks collapse into one commit and the first paint is no
  // earlier than it was. Without startTransition the widening blocks input.
  assert.match(GLUE, /requestAnimationFrame\(/);
  assert.match(GLUE, /startTransition\(/);
  assert.match(GLUE, /cancelAnimationFrame\(frame\)/);
});

test("the shift is measured in viewport space, as a residual", () => {
  // This one has a bug behind it rather than a theory. Chromium implements CSS scroll anchoring
  // and this viewport does not opt out, so on a widening commit the browser has usually ALREADY
  // moved scrollTop by the inserted height. A document-space delta reports the full insertion
  // regardless, so applying it doubles the browser's own correction: measured, on a reader parked
  // 4000px above the bottom of a 300K thread, it walked scrollTop 22,897 -> 117,104 across seven
  // widenings and dumped them at the bottom. Viewport space measures the residual, which was
  // single-digit pixels on the same run.
  assert.match(
    GLUE,
    /viewportOffset: first\.getBoundingClientRect\(\)\.top,/,
    "the anchor must be captured in viewport space",
  );
  // Both ends of the measurement are captured, because which of them is used depends on the
  // engine. The arithmetic itself lives in anchorCorrection and is tested there rather than here.
  assert.match(GLUE, /scrollTop: viewport\.scrollTop,/);
  assert.match(
    GLUE,
    /anchorCorrection\(/,
    "the correction must go through the tested pure function",
  );
});

test("the window owns the viewport's scroll-anchoring mode while it is open", () => {
  // The correction is document-space arithmetic, and that is only correct if nothing else is
  // moving scrollTop. Native scroll anchoring moves it on some frames and not others -- not at
  // all on any shipping Safari, and suppressed per frame everywhere else after a programmatic
  // scroll, which is what a scrollbar drag, PageUp and middle-click autoscroll are -- and no
  // measurement taken inside the frame can tell the two apart. Measured with it left on: 19,259px
  // of drift on a parked reader where the engine did not compensate, and 45,161px on Chromium
  // where it did but was suppressed. So it is turned off while the window is open.
  assert.match(GLUE, /setProperty\("overflow-anchor", "none"\)/);
  // And turned back on when the window closes, or a settled thread is not the thread that
  // shipped before this change.
  assert.match(GLUE, /removeProperty\("overflow-anchor"\)/);
  // No feature probe and no gesture bookkeeping: both existed to decide which of two behaviours
  // the browser was giving us, and the browser is no longer in that loop.
  assert.doesNotMatch(GLUE, /getUserGestureSeq/);
  assert.doesNotMatch(HOOK, /userGestureSeqRef/);
});

test("the mount window never writes scrollTop itself", () => {
  // Ownership: the autoscroll hook owns scrollTop. Two writers per widening frame double the
  // forced layouts and can re-attach a deliberately detached user through onScroll.
  const glue = GLUE_WINDOW();
  assert.doesNotMatch(glue, /scrollTop\s*=[^=]/);
  assert.doesNotMatch(glue, /\.scrollTo\(/);
  assert.doesNotMatch(glue, /scrollIntoView\(/);
  assert.match(glue, /adjustForContentInsertedAbove\(shift \?\? 0\)/);
});

test("the hook's correction stands down while the user is following", () => {
  // While following, the hook's own MutationObserver path pins to the bottom in the same frame,
  // and because widening only prepends that is the same pixel. Correcting as well would be a
  // second writer for no gain.
  const body = section(HOOK, "adjustImplRef.current = (", "const onWheel = ");
  assert.match(body, /if \(!userDetachedRef\.current\) \{\s*return;/);
});

test("the hook's correction is instant, because the viewport is scroll-smooth", () => {
  // scroll-smooth on the viewport means an animated write would still be in flight when the next
  // widening frame issued the next one.
  assert.match(THREAD, /scroll-smooth/);
  const body = section(HOOK, "adjustImplRef.current = (", "const onWheel = ");
  assert.match(body, /behavior: "instant"/);
});

test("the correction advances the intent bookkeeping with its write", () => {
  // Otherwise onScroll sees a downward scroll nobody made, and a user who detached within
  // RE_ATTACH_THRESHOLD_PX of the bottom (detachFromBottom does exactly that when the composer
  // grows) is silently re-attached by their own correction and yanked to the bottom on the next
  // widening frame.
  const body = section(HOOK, "adjustImplRef.current = (", "const onWheel = ");
  assert.match(body, /lastScrollTop = el\.scrollTop;/);
  assert.match(body, /lastDistanceFromBottom = distanceFromBottom\(\);/);
});

test("the escape hatch exists, resolves after a paint, and is registered only while withholding", () => {
  // A DOM read taken during the few frames a long thread takes to converge would silently see a
  // short conversation. Resolving on a single rAF would resolve before the commit it forced had
  // painted.
  assert.match(
    GLUE,
    /export async function completeProgressiveMounts\(\): Promise<void>/,
  );
  assert.match(
    GLUE,
    /requestAnimationFrame\(\(\) => requestAnimationFrame\(\(\) => resolve\(\)\)\)/,
  );
  assert.match(GLUE, /if \(!isWithholding\) return;/);
  assert.match(GLUE, /activeCompleters\.delete\(complete\)/);
});

test("the viewport comes from a ref, not a document-wide query", () => {
  // The Compare panes each mount their own Thread, so a document-wide query for the viewport
  // class finds whichever one is first in the document rather than the one these rows are in.
  assert.doesNotMatch(GLUE, /document\.querySelector/);
  assert.match(GLUE, /viewportRef\.current/);
});

test("the row map is memoized on the component identities", () => {
  // ThreadPrimitive.Messages compared its components field by field and skipped re-rendering
  // entirely. A replacement that rebuilt 220 elements on every Thread re-render would hand back
  // a share of what the windowing collects.
  assert.match(
    GLUE,
    /useMemo\(\(\) => \{[\s\S]*?\}, \[count, mountWindow, components\]\)/,
  );
  assert.match(
    GLUE,
    /prev\.components\.AssistantMessage === next\.components\.AssistantMessage/,
  );
  assert.match(THREAD, /const messageComponents = useMemo\(/);
});

test("a thread that shrank under a live window drops the restriction, it does not clamp", () => {
  // Clamping `start` to `count` makes the row loop emit NOTHING, and the widen that would heal it
  // is inside startTransition. Measured on the version that clamped: dropping a 220-message
  // thread to 10 while the window sat at start=204 painted an empty column for 2 to 8 frames on
  // chromium, webkit and firefox. This is a source assertion because the failure is a React
  // commit rather than a value; the gate that can actually see it is
  // tests/studio/probe_pm_edge.py.
  assert.match(
    GLUE,
    /mountWindow == null \|\| mountWindow\.start >= count/,
    "start >= count must drop the window, not clamp it to count",
  );
  assert.doesNotMatch(
    GLUE,
    /Math\.min\(Math\.max\(mountWindow\.start, 0\), count\)/,
    "clamping start to count emits zero rows",
  );
});

test("the window re-arms on the commit that first fills an empty tree", () => {
  // The app mounts this component before the history adapter has delivered anything, so without
  // this the cold open -- the one users actually complain about -- was never windowed at all.
  // Measured in the app: mount at count 0, count 220 about 160ms later, same resetKey.
  assert.match(GLUE, /previousCount === 0 && count > 0/);
  assert.doesNotMatch(
    GLUE,
    /previousCount < MIN_PROGRESSIVE_MESSAGES/,
    "a threshold-crossing rule would window a thread the reader is already in the middle of",
  );
});

test("the escape hatch re-reads the completer set instead of sampling it once", () => {
  // The completers register from an effect, so a caller in the same task as the thread opening
  // finds the set empty. Measured on the version that read it once: the call returned in 0.1ms
  // and the document held 16 of 220 rows two frames later.
  // And an empty set is not believed straight away. A settled thread and a thread whose history
  // is still loading both present an empty set, and on a cold open the load takes around 160ms,
  // so returning at the first empty reading resolves before a single row exists.
  assert.match(GLUE, /let observed = false;/);
  assert.match(
    GLUE,
    /activeCompleters\.size === 0 && \(observed \|\| Date\.now\(\) >= deadline\)/,
  );
});

test("the hook is told about every widening, including the ones with nothing to apply", () => {
  // A widening with nothing to apply still has to resync the hook's bookkeeping, because
  // something else may have moved scrollTop in the same frame. Measured on the version that
  // returned early: a scroll-anchoring adjustment fires a scroll event carrying the new offset
  // (one event, on Chromium 151, WebKit 26.5 and Firefox 153), lastScrollTop stayed a whole
  // insertion behind, and that event read as a large downward scroll, re-attaching a reader
  // parked within the 24px threshold.
  assert.match(GLUE, /adjustForContentInsertedAbove\(shift \?\? 0\)/);
  const body = section(HOOK, "adjustImplRef.current = (", "const onWheel = ");
  assert.doesNotMatch(
    body,
    /deltaPx === 0\) \{\s*return;/,
    "a zero correction must still resync the bookkeeping",
  );
  assert.match(body, /lastScrollTop = el\.scrollTop;/);
});

