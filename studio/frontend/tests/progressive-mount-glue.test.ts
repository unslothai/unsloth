// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The impure parts of the progressive mount, pinned by source assertion because they live in .tsx
// files and node's type stripping cannot import one (same shape as
// chat-autoscroll-frame-budget.test.ts). Each is a real invariant with a real failure behind it.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const read = (path: string): string =>
  readFileSync(new URL(path, import.meta.url), "utf8");

/**
 * Source with comments removed. The "must not appear" assertions run against this: these files
 * explain at length what they deliberately do NOT do, and a prose mention of a banned construct
 * must not fail its own test.
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
 * The region of `source` between two markers, with BOTH required. A bare
 * `source.slice(source.indexOf(marker))` is the sharpest edge in a file of source assertions: a
 * renamed marker gives -1 and `slice(-1)` is the file's last character, or a moved end marker
 * widens to the rest of the file and PASSES for the wrong reason.
 */
const section = (source: string, from: string, to: string): string => {
  const start = source.indexOf(from);
  assert.notEqual(start, -1, `source assertion anchor is gone: ${from}`);
  const end = source.indexOf(to, start + from.length);
  assert.notEqual(end, -1, `source assertion anchor is gone: ${to}`);
  return source.slice(start, end);
};

/**
 * The mount window and the row map only. The "must never appear" assertions are about what the
 * WINDOW does to the thread's viewport, and anything above it may legitimately hide, scroll or
 * insert into elements of its own that are never in the thread.
 */
const GLUE_WINDOW = () =>
  section(
    GLUE,
    "function useProgressiveMountWindow(",
    "ProgressiveMessages.displayName",
  );

test("the thread renders rows through MessageByIndex, never ThreadPrimitive.Messages", () => {
  // Upstream renders MessageByIndexProvider -> RenderChildrenWithAccessor -> children; this row map
  // renders MessageByIndexProvider -> children. Switching between the two on convergence would
  // change the element type at that position, so React would unmount and rebuild every message.
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
  // And it takes the propless slot #9042 gave ThreadPrimitive.Messages: React's bail-out needs one
  // shared element per row, whereas a `components` object allocates fresh props per row per commit
  // and every delete re-renders every body, action bar and tooltip.
  assert.match(THREAD, /renderMessage=\{renderThreadMessage\}/);
  assert.match(GLUE, /const message = renderMessage\(\);/);
  assert.doesNotMatch(
    GLUE,
    /components=\{components\}/,
    "the row map must not go back to the components form",
  );
});

test("rows outside the window are not rendered rather than rendered and hidden", () => {
  // The cost avoided is mounting. A display:none row costs the same to mount, so a rewrite to
  // hiding would stay green and collect nothing.
  const glue = GLUE_WINDOW();
  assert.doesNotMatch(glue, /display:\s*["']?none/);
  assert.doesNotMatch(glue, /visibility:\s*["']?hidden/);
  assert.doesNotMatch(glue, /content-visibility/);
});

test("the window is only ever advanced through widen", () => {
  // setMountWindow may set the initial window, clear it (null), or call widen. Anything else could
  // move `start` upward and unmount a mounted message, the failure mode this design rules out.
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
  // tree that has not reached it. Two gates: reactive, for a run starting after the window opened,
  // and an in-frame re-check for one starting between commit and rAF.
  assert.match(GLUE, /thread\.isRunning/);
  assert.match(
    GLUE,
    /if \(threadIsRunning && mountWindow != null\) setMountWindow\(null\)/,
  );
  assert.match(GLUE, /if \(isRunningNow\(\)\) \{\s*setMountWindow\(null\);/);
});

test("the widening step is deferred and transition-wrapped", () => {
  // Without rAF the chunks collapse into one commit and the first paint is no earlier than before.
  // Without startTransition the widening blocks input.
  assert.match(GLUE, /requestAnimationFrame\(/);
  assert.match(GLUE, /startTransition\(/);
  assert.match(GLUE, /cancelAnimationFrame\(frame\)/);
  // Scoped to the widening effect, because the file schedules frames in three other places and a
  // whole-file match stays green while this one is retimed to a timer. Two failures behind it. A
  // timer fires before the commit it follows has painted, so captureAnchor samples rects from the
  // previous layout, and the same early run is why the disarm has to happen here rather than in a
  // layout effect (the viewport ref is populated by the paint). And the cleanup below cancels a
  // FRAME: hand it a timer handle and a window closed between the commit and the callback widens
  // anyway, which is a widening into a tree a run may already have taken over.
  const widening = section(
    GLUE,
    "if (mountWindow == null) return;",
    "}, [mountWindow, count, captureAnchor, isRunningNow]);",
  );
  assert.match(
    widening,
    /const frame = requestAnimationFrame\(/,
    "the widening step must be scheduled on a frame, not a timer",
  );
  assert.doesNotMatch(
    widening,
    /setTimeout|setInterval|queueMicrotask/,
    "cancelAnimationFrame does not cancel a timer, so a closed window would still widen",
  );
});

test("the anchor is measured against its scroll container, not against the window", () => {
  // getBoundingClientRect().top is WINDOW-relative, so it also moves when the scroll container does
  // (the composer grows a line, mobile chrome slides away, a parent relayouts). Any of those between
  // capture and the widening commit would read as content inserted above and move a reader.
  assert.match(
    GLUE,
    /element\.getBoundingClientRect\(\)\.top -\s*viewport\.getBoundingClientRect\(\)\.top/,
    "the container's own top must be subtracted at both ends",
  );
  // Both ends go through the same function, so they cannot drift apart.
  assert.match(GLUE, /function sampleAnchor\(/);
  assert.match(GLUE, /sampleAnchor\(viewport, anchor\)/);
  assert.match(GLUE, /anchorCorrection\(baseline, sampleAnchor\(viewport, element\)\)/);
  // The enclosing row is a fallback: pickAnchorRow descends to the fold, so the anchor is often the
  // <pre> Streamdown replaces when Shiki finishes, and dropping the correction then would move a
  // detached reader by a whole chunk with anchoring off.
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
  // completeProgressiveMounts awaits a promise this component resolves. Without an unmount path the
  // cleanup removed the completer but left the promise pending forever, hanging a DOM capture.
  assert.match(
    GLUE,
    /useEffect\(\(\) => flushCompletionWaiters, \[flushCompletionWaiters\]\)/,
  );
  // A cancelled frame must not drop them either, so the flush empties the ref, not the scheduler.
  const scheduler = section(
    GLUE,
    "if (mountWindow != null || completionWaiters.current.length === 0) return;",
    "useEffect(() => flushCompletionWaiters",
  );
  assert.doesNotMatch(scheduler, /completionWaiters\.current = \[\];/);
});

test("the window owns the viewport's scroll-anchoring mode while it is open", () => {
  // The correction is document-space arithmetic, correct only if nothing else moves scrollTop.
  // Native anchoring moves it on some frames and not others (never on any shipping Safari, and
  // suppressed per frame elsewhere after a programmatic scroll), and no in-frame measurement tells
  // the two apart. Measured with it left on: 19,259px of drift on a parked reader where the engine
  // did not compensate, 45,161px on Chromium where it did but was suppressed.
  assert.match(GLUE, /setProperty\("overflow-anchor", "none"\)/);
  // Set from the anchor-capture rAF, NOT a layout effect: this component is a descendant of the
  // viewport, so on the mounting commit React runs this subtree's layout effects before the
  // viewport's ref callback and the ref is still null. Measured on that version, computed
  // overflow-anchor stayed `auto` until the first widening at +803ms, which then applied the
  // document-space correction on top of the browser's and left a reader 776px short.
  const capture = section(
    GLUE,
    "const captureAnchor = useCallback(",
    "}, [viewportRef]);",
  );
  assert.match(capture, /setProperty\("overflow-anchor", "none"\)/);
  // And back on when the window closes, or a settled thread is not the pre-change thread.
  assert.match(GLUE, /removeProperty\("overflow-anchor"\)/);
  // Scoped to the effect that runs on the CLOSING commit, because there is a second removeProperty
  // in the unmount cleanup and a whole-file match is satisfied by that one alone. Losing the close
  // path is not a transient: the thread the reader keeps scrolling for the rest of the session has
  // native anchoring off with nothing left compensating for it, since the idle sampler and the
  // widening correction both stop with the window. Every later image, Shiki swap or KaTeX resize
  // above the fold then moves the reader by its own height.
  const closing = section(
    GLUE,
    "if (mountWindow != null) return;",
    "}, [mountWindow, viewportRef]);",
  );
  assert.match(
    closing,
    /restoreScrollAnchoring\(viewportRef\.current\)/,
    "the closing commit must hand scroll anchoring back to the browser",
  );
  // Both hand-backs go through the one helper, which also drops the emptied `style` attribute.
  // removeProperty alone leaves `style=""` behind, and that attribute was the only difference a
  // whole-document digest could find between this branch and its merge base at 100K and 300K.
  assert.match(
    section(GLUE, "function restoreScrollAnchoring(", "\n}"),
    /if \(viewport\.getAttribute\("style"\) === ""\) viewport\.removeAttribute\("style"\);/,
  );
  assert.doesNotMatch(
    GLUE.replace(section(GLUE, "function restoreScrollAnchoring(", "\n}"), ""),
    /removeProperty\("overflow-anchor"\)/,
    "every hand-back must go through restoreScrollAnchoring, or one path leaks style=''",
  );
  // No feature probe and no gesture bookkeeping: both decided which behaviour the browser was
  // giving us, and the browser is no longer in that loop.
  assert.doesNotMatch(GLUE, /getUserGestureSeq/);
  assert.doesNotMatch(HOOK, /userGestureSeqRef/);
});

test("the mount window never writes scrollTop itself", () => {
  // The autoscroll hook owns scrollTop. Two writers per widening frame double the forced layouts
  // and can re-attach a deliberately detached user through onScroll.
  const glue = GLUE_WINDOW();
  assert.doesNotMatch(glue, /scrollTop\s*=[^=]/);
  assert.doesNotMatch(glue, /\.scrollTo\(/);
  assert.doesNotMatch(glue, /scrollIntoView\(/);
  assert.match(glue, /adjustForContentInsertedAbove\(shift \?\? 0\)/);
});

test("the hook's correction stands down while the user is following", () => {
  // While following, the hook's MutationObserver path pins to the bottom in the same frame, and
  // since widening only prepends that is the same pixel. Correcting too is a second writer.
  const body = section(HOOK, "adjustImplRef.current = (", "const onWheel = ");
  assert.match(body, /if \(!userDetachedRef\.current\) \{\s*return;/);
});

test("the hook's correction is instant, because the viewport is scroll-smooth", () => {
  // With scroll-smooth on the viewport, an animated write would still be in flight at the next
  // widening frame's write.
  assert.match(THREAD, /scroll-smooth/);
  const body = section(HOOK, "adjustImplRef.current = (", "const onWheel = ");
  assert.match(body, /behavior: "instant"/);
});

test("the correction advances the intent bookkeeping with its write", () => {
  // Otherwise onScroll sees a downward scroll nobody made, and a user detached within
  // RE_ATTACH_THRESHOLD_PX of the bottom is re-attached by their own correction and yanked to the
  // bottom on the next widening frame.
  const body = section(HOOK, "adjustImplRef.current = (", "const onWheel = ");
  assert.match(body, /lastScrollTop = el\.scrollTop;/);
  assert.match(body, /lastDistanceFromBottom = distanceFromBottom\(\);/);
});

test("the escape hatch exists, resolves after a paint, and is registered only while withholding", () => {
  // A DOM read during the few frames a long thread takes to converge would silently see a short
  // conversation, and a single rAF resolves before the commit it forced has painted.
  assert.match(GLUE, /export async function completeProgressiveMounts\(/);
  // The filter is optional, so a caller that wants every thread still writes no argument.
  assert.match(GLUE, /wants\?: \(viewport: HTMLElement \| null\) => boolean,/);
  assert.match(
    GLUE,
    /requestAnimationFrame\(\(\) => requestAnimationFrame\(\(\) => resolve\(\)\)\)/,
  );
  assert.match(GLUE, /if \(!isWithholding\) return;/);
  assert.match(GLUE, /activeCompleters\.delete\(entry\)/);
});

test("the viewport comes from a ref, not a document-wide query", () => {
  // The Compare panes each mount their own Thread, so a document-wide query finds whichever
  // viewport comes first rather than the one these rows are in.
  assert.doesNotMatch(GLUE, /document\.querySelector/);
  assert.match(GLUE, /viewportRef\.current/);
});

test("the row map is memoized on the slot identity", () => {
  // ThreadPrimitive.Messages skipped re-rendering entirely while its children function kept its
  // identity; rebuilding 220 elements per Thread re-render hands back what the windowing collects.
  assert.match(
    GLUE,
    /useMemo\(\(\) => \{[\s\S]*?\}, \[count, mountWindow, renderMessage\]\)/,
  );
  assert.match(GLUE, /prev\.renderMessage === next\.renderMessage/);
  assert.match(THREAD, /const renderThreadMessage = proplessSlot\(/);
});

test("a thread that shrank under a live window drops the restriction, it does not clamp", () => {
  // Clamping `start` to `count` makes the row loop emit NOTHING, and the widen that would heal it is
  // inside startTransition. Measured on the clamping version: dropping a 220-message thread to 10
  // with the window at start=204 painted an empty column for 2 to 8 frames on all three engines. A
  // source assertion because the failure is a React commit rather than a value.
  assert.match(
    GLUE,
    /mountWindow == null \|\| mountWindow\.start >= count/,
    "start >= count must drop the window, not clamp it to count",
  );
  // And the STATE is reconciled, not just the local `first`: leaving the window at start 204
  // against a 100-message thread lets the next widen produce start 68 and unmount 68 mounted rows.
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
  // The app mounts this before the history adapter delivers anything, so without this the cold
  // open, the one users complain about, was never windowed at all. Measured in the app: mount at
  // count 0, count 220 about 160ms later, same resetKey.
  assert.match(GLUE, /previousCount === 0 && count > 0/);
  assert.doesNotMatch(
    GLUE,
    /previousCount < MIN_PROGRESSIVE_MESSAGES/,
    "a threshold-crossing rule would window a thread the reader is already in the middle of",
  );
});

test("the escape hatch re-reads the completer set instead of sampling it once", () => {
  // The completers register from an effect, so a caller in the same task as the thread opening finds
  // the set empty (measured: returned in 0.1ms, 16 of 220 rows two frames later). Nor is an empty
  // set believed straight away, since a settled thread and one still loading history look alike.
  assert.match(GLUE, /let observed = false;/);
  // Through `wanted()`, which reads `activeCompleters` on every call: the exit has to see the set as
  // it is now, and it has to see the same FILTERED set the loop drains, or a completer this caller
  // declined would hold it open forever.
  assert.match(
    GLUE,
    /wanted\(\)\.length === 0 && \(observed \|\| Date\.now\(\) >= deadline\)/,
  );
  assert.match(GLUE, /const wanted = \(\) =>[\s\S]*?activeCompleters/);
  // The re-read is only worth anything if the deadline is in the FUTURE. A deadline of `Date.now()`
  // keeps both lines above and still returns on the first pass, because a caller in the same task
  // as the thread opening reaches the check before any completer has registered: the read-once
  // failure again, spelled as a zero search interval. So the interval is a positive constant, and
  // it has to outlast a cold open's history load, measured at about 160ms.
  assert.match(
    GLUE,
    /const deadline = Date\.now\(\) \+ PROGRESSIVE_MOUNT_SEARCH_MS;/,
    "an empty completer set must only be believed after a search interval",
  );
  const searchMs = Number(
    /PROGRESSIVE_MOUNT_SEARCH_MS = (\d+)/.exec(GLUE)?.[1] ?? "0",
  );
  assert.ok(
    searchMs >= 200,
    `the search interval must outlast a cold open's history load, got ${searchMs}ms`,
  );
});

test("the hook is told about every widening, including the ones with nothing to apply", () => {
  // A widening with nothing to apply still resyncs the hook's bookkeeping, since something else may
  // have moved scrollTop in the same frame. Measured on the early-return version: a scroll-anchoring
  // adjustment fires one scroll event carrying the new offset (Chromium 151, WebKit 26.5, Firefox
  // 153), lastScrollTop stayed an insertion behind, and that event re-attached a parked reader.
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
  // Widening prepends above everything, so any row will do for it. A RELAYOUT will not: a row
  // growing between the topmost row and the fold moves the reader while leaving the topmost row
  // still. Measured with a 600px height change injected above a detached reader with the window
  // open: 600px of movement, reported as 0 by a topmost-row anchor.
  assert.match(GLUE, /function pickAnchorRow\(viewport: HTMLElement\)/);
  assert.match(GLUE, /if \(row\.getBoundingClientRect\(\)\.bottom > fold\) \{/);
  // Then down to the fold, because a row can be taller than the viewport: a reader partway through a
  // long answer has the row's top ABOVE them, so a block earlier in that message moves everything
  // they see while the row's top stays put.
  assert.match(GLUE, /for \(const child of anchor\.children\)/);
  // Visibility is tested by the row's box, not its top offset: a tall row just off screen can
  // still have its top within a viewport of the fold.
  assert.match(GLUE, /function isAnchorVisible\(/);
  assert.match(GLUE, /return box\.bottom > fold && box\.top < fold \+ viewport\.clientHeight;/);
  // A widening hands the sampler a post-correction baseline rather than nulling it, or a reflow
  // landing before the next frame is folded into the new baseline and never corrected.
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
  // Native anchoring normally absorbs a <pre> swapping in Shiki output or a KaTeX resize, and it is
  // off for the whole mount window, so compensation has to cover the whole interval. The autoscroll
  // hook does not help: it pins a FOLLOWING reader and leaves a detached one alone.
  assert.match(GLUE, /idleRef/);
  assert.match(GLUE, /if \(!viewport \|\| anchorRef\.current\) return;/);
  // The idle path carries the same fallback as the widening path, because the anchor is often the
  // pre Streamdown replaces, and re-picking would re-base AFTER that replacement's own reflow.
  assert.match(GLUE, /function holdAnchor\(viewport: HTMLElement, element: Element\): HeldAnchor/);
  assert.match(GLUE, /held\.row\?\.isConnected && held\.rowSample/);
  // And it re-bases every frame, no-ops included: returning early left the baseline's scrollTop
  // stale, and anchorCorrection's clamp term reads it.
  assert.match(GLUE, /if \(shift !== null\) adjustForContentInsertedAbove\(shift\);/);
});
