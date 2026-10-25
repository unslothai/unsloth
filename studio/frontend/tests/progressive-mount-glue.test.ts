// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The progressive mount glue lives in .tsx, which node type stripping cannot import, so assert on source.

import assert from "node:assert/strict";
import test from "node:test";

import { readText } from "./helpers/kit.ts";

/** Comments stripped, so prose mentioning a banned construct does not fail its own test. */
const code = (source: string): string =>
  source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^[ \t]*\/\/.*$/gm, "");

const GLUE = code(
  readText("../src/components/assistant-ui/progressive-messages.tsx"),
);
const HOOK = code(
  readText("../src/components/assistant-ui/use-intent-aware-autoscroll.tsx"),
);
const THREAD = code(readText("../src/components/assistant-ui/thread.tsx"));

/** Both markers required: indexOf -1 or a moved end marker would silently widen the slice. */
const section = (source: string, from: string, to: string): string => {
  const start = source.indexOf(from);
  assert.notEqual(start, -1, `source assertion anchor is gone: ${from}`);
  const end = source.indexOf(to, start + from.length);
  assert.notEqual(end, -1, `source assertion anchor is gone: ${to}`);
  return source.slice(start, end);
};

/** Only the mount window and row map; code above may legitimately touch its own elements. */
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
  assert.match(
    GLUE,
    /<AuiProvider key=\{index\} value=\{gate\.row\(index\)\}>\s*<MessageByIndexProvider index=\{index\}>/,
  );
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
  // React's bail-out needs one shared element per row; a `components` object reallocates props.
  assert.match(THREAD, /renderMessage=\{renderThreadMessage\}/);
  assert.match(GLUE, /const message = renderMessage\(\);/);
  assert.doesNotMatch(
    GLUE,
    /components=\{components\}/,
    "the row map must not go back to the components form",
  );
});

test("rows outside the window are not rendered rather than rendered and hidden", () => {
  // display:none costs the same to mount, so hiding would stay green and save nothing.
  const glue = GLUE_WINDOW();
  assert.doesNotMatch(glue, /display:\s*["']?none/);
  assert.doesNotMatch(glue, /visibility:\s*["']?hidden/);
  assert.doesNotMatch(glue, /content-visibility/);
});

test("the window is only ever advanced through widen", () => {
  // Any other setter could move `start` upward and unmount a message.
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
  // Two gates: reactive for later runs, in-frame re-check for runs between commit and rAF.
  assert.match(GLUE, /thread\.isRunning/);
  assert.match(
    GLUE,
    /if \(threadIsRunning && mountWindow != null\) setMountWindow\(null\)/,
  );
  assert.match(GLUE, /if \(isRunningNow\(\)\) \{\s*setMountWindow\(null\);/);
});

test("the widening step is deferred and transition-wrapped", () => {
  // Without rAF chunks collapse into one commit; without startTransition widening blocks input.
  assert.match(GLUE, /requestAnimationFrame\(/);
  assert.match(GLUE, /startTransition\(/);
  assert.match(GLUE, /cancelAnimationFrame\(frame\)/);
  // Scoped to the widening effect: a timer fires before paint and its handle cannot be
  // cancelled as a frame.
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
  // getBoundingClientRect().top is window-relative, so container moves would read as insertion.
  assert.match(
    GLUE,
    /element\.getBoundingClientRect\(\)\.top -\s*viewport\.getBoundingClientRect\(\)\.top/,
    "the container's own top must be subtracted at both ends",
  );
  assert.match(GLUE, /function sampleAnchor\(/);
  assert.match(GLUE, /sampleAnchor\(viewport, anchor\)/);
  assert.match(GLUE, /anchorCorrection\(baseline, sampleAnchor\(viewport, element\)\)/);
  // The anchor is often a <pre> Streamdown replaces after Shiki, so fall back to the row.
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
  // Without an unmount path the pending promise hangs a DOM capture forever.
  assert.match(
    GLUE,
    /useEffect\(\(\) => flushCompletionWaiters, \[flushCompletionWaiters\]\)/,
  );
  const scheduler = section(
    GLUE,
    "if (mountWindow != null || completionWaiters.current.length === 0) return;",
    "useEffect(() => flushCompletionWaiters",
  );
  assert.doesNotMatch(scheduler, /completionWaiters\.current = \[\];/);
});

test("the window owns the viewport's scroll-anchoring mode while it is open", () => {
  // Native anchoring moves scrollTop inconsistently across engines, breaking document-space math.
  assert.match(GLUE, /setProperty\("overflow-anchor", "none"\)/);
  // Set from the capture rAF: on mount the viewport ref is still null in layout effects.
  const capture = section(
    GLUE,
    "const captureAnchor = useCallback(",
    "}, [viewportRef]);",
  );
  assert.match(capture, /setProperty\("overflow-anchor", "none"\)/);
  assert.match(GLUE, /removeProperty\("overflow-anchor"\)/);
  // Scoped to the closing effect; the unmount cleanup's removeProperty would satisfy a file match.
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
  // removeProperty alone leaves `style=""` behind.
  assert.match(
    section(GLUE, "function restoreScrollAnchoring(", "\n}"),
    /if \(viewport\.getAttribute\("style"\) === ""\) viewport\.removeAttribute\("style"\);/,
  );
  assert.doesNotMatch(
    GLUE.replace(section(GLUE, "function restoreScrollAnchoring(", "\n}"), ""),
    /removeProperty\("overflow-anchor"\)/,
    "every hand-back must go through restoreScrollAnchoring, or one path leaks style=''",
  );
  assert.doesNotMatch(GLUE, /getUserGestureSeq/);
  assert.doesNotMatch(HOOK, /userGestureSeqRef/);
});

test("the mount window never writes scrollTop itself", () => {
  // The autoscroll hook owns scrollTop; a second writer can re-attach a detached user.
  const glue = GLUE_WINDOW();
  assert.doesNotMatch(glue, /scrollTop\s*=[^=]/);
  assert.doesNotMatch(glue, /\.scrollTo\(/);
  assert.doesNotMatch(glue, /scrollIntoView\(/);
  assert.match(glue, /adjustForContentInsertedAbove\(shift \?\? 0\)/);
});

test("the hook's correction stands down while the user is following", () => {
  // While following, the hook already pins to bottom in the same frame.
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
  // Otherwise onScroll sees a phantom scroll and re-attaches a nearly-bottom detached user.
  const body = section(HOOK, "adjustImplRef.current = (", "const onWheel = ");
  assert.match(body, /lastScrollTop = el\.scrollTop;/);
  assert.match(body, /lastDistanceFromBottom = distanceFromBottom\(\);/);
});

test("the escape hatch exists, resolves after a paint, and is registered only while withholding", () => {
  // A single rAF resolves before the forced commit paints.
  assert.match(GLUE, /export async function completeProgressiveMounts\(/);
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
  // Messages skips re-render only while its children function keeps identity.
  assert.match(
    GLUE,
    /useMemo\(\(\) => \{[\s\S]*?\}, \[count, mountWindow, renderMessage, gate\]\)/,
  );
  assert.match(
    GLUE,
    /const gate = useMemo\(\(\) => createRowNotificationGate\(aui\), \[aui\]\)/,
  );
  assert.match(GLUE, /prev\.renderMessage === next\.renderMessage/);
  assert.match(THREAD, /const renderThreadMessage = proplessSlot\(/);
});

test("a thread that shrank under a live window drops the restriction, it does not clamp", () => {
  // Clamping `start` to `count` emits no rows until a transition widen, painting an empty column.
  assert.match(
    GLUE,
    /mountWindow == null \|\| mountWindow\.start >= count/,
    "start >= count must drop the window, not clamp it to count",
  );
  // Reconcile state too, or the next widen unmounts mounted rows.
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
  // The thread mounts at count 0 before history arrives; the cold open must still be windowed.
  assert.match(GLUE, /previousCount === 0 && count > 0/);
  assert.doesNotMatch(
    GLUE,
    /previousCount < MIN_PROGRESSIVE_MESSAGES/,
    "a threshold-crossing rule would window a thread the reader is already in the middle of",
  );
});

test("the escape hatch re-reads the completer set instead of sampling it once", () => {
  // Completers register from an effect, so an empty set is not trusted straight away.
  assert.match(GLUE, /let observed = false;/);
  // Must re-read the filtered set each call, or a declined completer holds the loop open.
  assert.match(
    GLUE,
    /wanted\(\)\.length === 0 && \(observed \|\| Date\.now\(\) >= deadline\)/,
  );
  assert.match(GLUE, /const wanted = \(\) =>[\s\S]*?activeCompleters/);
  // The deadline must be positive and outlast a cold history load (~160ms).
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
  // Resync bookkeeping even with zero shift: anchoring scroll events would re-attach a parked reader.
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
  // Relayout below the topmost row moves the reader, so anchor at the fold.
  assert.match(GLUE, /function pickAnchorRow\(viewport: HTMLElement\)/);
  assert.match(GLUE, /if \(row\.getBoundingClientRect\(\)\.bottom > fold\) \{/);
  // A row can be taller than the viewport, so descend to the fold.
  assert.match(GLUE, /for \(const child of anchor\.children\)/);
  assert.match(GLUE, /function isAnchorVisible\(/);
  assert.match(GLUE, /return box\.bottom > fold && box\.top < fold \+ viewport\.clientHeight;/);
  // Pass a post-correction baseline, or a reflow before the next frame is never corrected.
  assert.doesNotMatch(GLUE, /idleRef\.current = null;\s*\}, \[mountWindow/);
  assert.doesNotMatch(
    GLUE,
    /querySelector\("\[data-role\]"\)/,
    "the first row in the list is not the row the reader is looking at",
  );
  assert.match(GLUE, /const anchor = viewport \? pickAnchorRow\(viewport\) : null;/);
  assert.match(GLUE, /const element = pickAnchorRow\(viewport\);/);
});

test("the interval between widenings is compensated too, not just the widening commits", () => {
  // Native anchoring is off for the whole window, so idle reflows need compensation too.
  assert.match(GLUE, /idleRef/);
  assert.match(GLUE, /if \(!viewport \|\| anchorRef\.current\) return;/);
  assert.match(GLUE, /function holdAnchor\(viewport: HTMLElement, element: Element\): HeldAnchor/);
  assert.match(GLUE, /held\.row\?\.isConnected && held\.rowSample/);
  // Re-base every frame: anchorCorrection's clamp term reads the baseline scrollTop.
  assert.match(GLUE, /if \(shift !== null\) adjustForContentInsertedAbove\(shift\);/);
});
