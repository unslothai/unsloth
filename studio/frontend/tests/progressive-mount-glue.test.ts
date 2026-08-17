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

const GLUE = code(
  read("../src/components/assistant-ui/progressive-messages.tsx"),
);
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
  // children; the row map here renders MessageByIndexProvider -> children, dropping the accessor
  // wrapper the thread's slot does not use. Rendering upstream's tree while settled and this one
  // while windowed would change the element type at that position on the convergence commit, and
  // React would unmount and rebuild every message in the thread, which is the exact cost this
  // change exists to avoid.
  assert.match(GLUE, /<MessageByIndexProvider key=\{index\} index=\{index\}>/);
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
  // And it takes the same propless slot #9042 gave ThreadPrimitive.Messages. One shared element
  // per row is what React's own bail-out needs; a `components` object allocates fresh props per
  // row per commit, so every delete re-renders every body, action bar and tooltip in the thread.
  assert.match(THREAD, /renderMessage=\{renderThreadMessage\}/);
  assert.match(GLUE, /const message = renderMessage\(\);/);
  assert.doesNotMatch(
    GLUE,
    /components=\{components\}/,
    "the row map must not go back to the components form",
  );
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

test("the anchor is measured against its scroll container, not against the window", () => {
  // getBoundingClientRect().top is measured against the WINDOW, so it also moves when the scroll
  // container moves, and this container moves for reasons that have nothing to do with the
  // thread: the composer grows a line, the mobile browser chrome slides away, a parent relayouts.
  // Any of those landing between the capture and the widening commit would read as content
  // inserted above and be corrected away, moving a detached reader for no reason.
  assert.match(
    GLUE,
    /element\.getBoundingClientRect\(\)\.top -\s*viewport\.getBoundingClientRect\(\)\.top/,
    "the container's own top must be subtracted at both ends",
  );
  // Both ends go through the same function, so they cannot drift apart.
  assert.match(GLUE, /function sampleAnchor\(/);
  assert.match(GLUE, /sampleAnchor\(viewport, anchor\)/);
  assert.match(GLUE, /anchorCorrection\(baseline, sampleAnchor\(viewport, element\)\)/);
  // The row the anchor sits in is captured alongside it. pickAnchorRow descends to the fold, so
  // the anchor is often the very <pre> Streamdown replaces when Shiki finishes highlighting it,
  // and a transition-deferred widening leaves room for that. Dropping the correction when it
  // happens would move a detached reader by a whole chunk with anchoring off.
  assert.match(GLUE, /const row = anchor\?\.closest\("\[data-role\]"\) \?\? null;/);
  assert.match(GLUE, /captured\.row\?\.isConnected && captured\.rowSample/);
  assert.match(GLUE, /scrollTop: viewport\.scrollTop,/);
  assert.match(
    GLUE,
    /anchorCorrection\(/,
    "the correction must go through the tested pure function",
  );
});

test("a completion waiter is settled when its thread goes away", () => {
  // completeProgressiveMounts awaits a promise this component resolves. Without an unmount path
  // the layout effect's cleanup removed the completer from the set but left that promise pending
  // forever, so a DOM capture that raced a thread switch hung rather than completing.
  assert.match(
    GLUE,
    /useEffect\(\(\) => flushCompletionWaiters, \[flushCompletionWaiters\]\)/,
  );
  // And a cancelled frame must not drop them either, so the ref is emptied by the flush rather
  // than by the effect that schedules it.
  const scheduler = section(
    GLUE,
    "if (mountWindow != null || completionWaiters.current.length === 0) return;",
    "useEffect(() => flushCompletionWaiters",
  );
  assert.doesNotMatch(scheduler, /completionWaiters\.current = \[\];/);
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
  // Set from the rAF that captures the anchor, NOT from a layout effect. This component is a
  // descendant of the viewport element, so on the commit that mounts them both React runs this
  // subtree's layout effects before the viewport's ref callback and the ref is still null.
  // Measured on the layout-effect version: computed overflow-anchor stayed `auto` from the first
  // painted row at +305ms until the first widening at +803ms, so that widening ran with anchoring
  // live and the document-space correction applied on top of the browser's own. A reader who
  // scrolled 4000px in that window was left 776px short of where they parked, and one whose whole
  // gesture landed inside it was carried back to 24px from the bottom of a 118,004px thread.
  const capture = section(
    GLUE,
    "const captureAnchor = useCallback(",
    "}, [viewportRef]);",
  );
  assert.match(capture, /setProperty\("overflow-anchor", "none"\)/);
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

test("the row map is memoized on the slot identity", () => {
  // ThreadPrimitive.Messages skipped re-rendering entirely when its children function held its
  // identity. A replacement that rebuilt 220 elements on every Thread re-render would hand back
  // a share of what the windowing collects.
  assert.match(
    GLUE,
    /useMemo\(\(\) => \{[\s\S]*?\}, \[count, mountWindow, renderMessage\]\)/,
  );
  assert.match(GLUE, /prev\.renderMessage === next\.renderMessage/);
  assert.match(THREAD, /const renderThreadMessage = proplessSlot\(/);
});

test("a thread that shrank under a live window drops the restriction, it does not clamp", () => {
  // Clamping `start` to `count` makes the row loop emit NOTHING, and the widen that would heal it
  // is inside startTransition. Measured on the version that clamped: dropping a 220-message
  // thread to 10 while the window sat at start=204 painted an empty column for 2 to 8 frames on
  // chromium, webkit and firefox. This is a source assertion because the failure is a React
  // commit rather than a value: seeing it at all needs a real engine driving a live thread, so
  // there is no in-tree gate on the behaviour, only on the shape of the code that produces it.
  assert.match(
    GLUE,
    /mountWindow == null \|\| mountWindow\.start >= count/,
    "start >= count must drop the window, not clamp it to count",
  );
  // And the STATE is reconciled, not just the local `first`. Rendering every row is half of it:
  // leaving the window at start 204 against a 100-message thread lets the next widen produce
  // start 68 and unmount the 68 rows that render just mounted.
  assert.match(
    GLUE,
    /if \(mountWindow != null && mountWindow\.start >= count\) \{\s*setMountWindow\(null\);/,
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

test("the anchor is the first row the reader can see, not the first row in the list", () => {
  // Widening prepends above everything, so for a widening any row will do. Content that RELAYOUTS
  // does not: a row growing between the topmost row and the fold moves the reader while leaving
  // the topmost row exactly where it was. Measured with a 600px height change injected into one
  // row above a detached reader while the window was open: 600px of movement, reported as 0 by a
  // topmost-row anchor, and 0px of movement once the anchor was the visible one.
  assert.match(GLUE, /function pickAnchorRow\(viewport: HTMLElement\)/);
  assert.match(GLUE, /if \(row\.getBoundingClientRect\(\)\.bottom > fold\) \{/);
  // And then down to the fold itself, because a row can be taller than the viewport: a reader
  // partway through a long answer has the row's top ABOVE them, so an image or a Shiki block
  // earlier in that same message moves everything they can see and leaves the row's top still.
  assert.match(GLUE, /for \(const child of anchor\.children\)/);
  // Visibility is tested by the row's own box, not by its top offset: a tall row scrolled just
  // past can have its top within a viewport of the fold while none of it is on screen.
  assert.match(GLUE, /function isAnchorVisible\(/);
  assert.match(GLUE, /return box\.bottom > fold && box\.top < fold \+ viewport\.clientHeight;/);
  // And a widening hands the sampler a post-correction baseline rather than nulling it, or a
  // reflow landing before the next frame is folded into the new baseline and never corrected.
  assert.doesNotMatch(GLUE, /idleRef\.current = null;\s*\}, \[mountWindow/);
  assert.doesNotMatch(
    GLUE,
    /querySelector\("\[data-role\]"\)/,
    "the first row in the list is not the row the reader is looking at",
  );
  // Both the widening capture and the between-widenings sampler use it.
  assert.match(GLUE, /const anchor = viewport \? pickAnchorRow\(viewport\) : null;/);
  assert.match(GLUE, /const element = pickAnchorRow\(viewport\);/);
});

test("the interval between widenings is compensated too, not just the widening commits", () => {
  // Native anchoring is what normally absorbs a <pre> swapping in Shiki output or a KaTeX resize,
  // and it is disabled for the whole mount window, so the compensation has to cover the whole
  // interval. The autoscroll hook does not help: it pins a FOLLOWING reader and deliberately
  // leaves a detached one alone.
  assert.match(GLUE, /idleRef/);
  assert.match(GLUE, /if \(!viewport \|\| anchorRef\.current\) return;/);
  // The idle path carries the same stable fallback the widening path does, because the anchor is
  // often the very pre Streamdown replaces, and re-picking after a replacement would re-base
  // AFTER that replacement's own reflow and keep it.
  assert.match(GLUE, /function holdAnchor\(viewport: HTMLElement, element: Element\): HeldAnchor/);
  assert.match(GLUE, /held\.row\?\.isConnected && held\.rowSample/);
  // And it re-bases every frame, including the no-op ones. Returning early on those left the
  // baseline's scrollTop several frames stale, and anchorCorrection's clamp term reads it.
  assert.match(GLUE, /if \(shift !== null\) adjustForContentInsertedAbove\(shift\);/);
});
