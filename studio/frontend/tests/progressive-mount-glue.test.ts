// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The mount window glue lives in .tsx, which node type stripping cannot import, so assert on
// source. The window maths it calls is tested directly in thread-window.test.ts.

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
const TURNS = code(readText("../src/components/assistant-ui/turn-navigation.tsx"));

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
    "function useThreadWindow(",
    "ProgressiveMessages.displayName",
  );

test("the thread renders rows through MessageByIndex, never ThreadPrimitive.Messages", () => {
  // Upstream renders MessageByIndexProvider, RenderChildrenWithAccessor, then children, while this
  // row map skips the accessor; switching between them would remount every message.
  assert.match(
    GLUE,
    /<AuiProvider key=\{index\} value=\{gate\.row\(index\)\}>\s*<MessageByIndexProvider index=\{index\}>/,
  );
  assert.doesNotMatch(
    GLUE,
    /<ThreadPrimitive\.Messages\b/,
    "the row map must not switch primitives",
  );
  assert.doesNotMatch(
    THREAD,
    /<ThreadPrimitive\.Messages\b/,
    "Thread must render the windowed list, not the unbounded one",
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
  // The only rows rendered are the ones the layout lists: the window's and the pinned ones.
  assert.match(glue, /return threadSlots\(win, count\)\.map\(\(slot\) => \{\s*if \(slot\.kind === "row"\) \{/);
});

test("the window is only ever set to what a planner returned", () => {
  // Any other setter could mount or unmount rows the reader's position did not ask for.
  const calls = [...GLUE.matchAll(/setWin\(([^;]*?)\)\s*[;,]/gs)].map((m) =>
    m[1].replace(/\s+/g, " ").trim(),
  );
  assert.ok(calls.length >= 5, `expected several setWin calls, found ${calls.length}`);
  const allowed = new Set(["open()", "next", "followed"]);
  for (const call of calls) {
    assert.ok(allowed.has(call), `setWin(${call}) is not an opening, a clamp, a reveal or a plan`);
  }
  assert.match(GLUE, /const followed = followCount\(win, previousCount, count\);\s*if \(followed !== win\) setWin\(followed\);/);
  const step = section(GLUE, "const step = useCallback(", "stepRef.current = step;");
  assert.match(step, /next = planFill\(current, total, chunk, heightOf\);/);
  assert.match(step, /next = planSlide\(\{/);
  assert.match(step, /const next = holdTailTurn\(\s*planEngage\(/);
  // A window at the tail holds the last prompt too, from the first open on.
  assert.match(step, /next = holdTailTurn\(next, total, lookup\.tailTurn\);/);
  assert.match(GLUE, /return holdTailTurn\(opened, metas\.length, tailTurn\);/);
  assert.match(step, /if \(sameWindow\(next, current\)\)/);
  const reveal = section(GLUE, "const reveal = useCallback(", "[aui, readMetas, viewportRef],");
  assert.match(reveal, /const next = keepPinned\(\s*viewport,\s*lookup,\s*\{ start: index, end: Math\.min\(total, index \+ SLIDE_STEP\) \},\s*tail,\s*\);/);
});

test("a run keeps the window, and the reply being written keeps its row", () => {
  // Agent chats run most of the time: dropping the window for a run would mount the whole thread.
  assert.doesNotMatch(GLUE, /threadIsRunning/);
  assert.doesNotMatch(
    GLUE,
    /isRunning[^;]*\)\s*setWin\(null\)/,
    "a run must not drop the window",
  );
  // The streaming row's effects run the stream; scrolling away must not unmount it.
  assert.match(GLUE, /const tail = aui\.thread\(\)\.getState\(\)\.isRunning === true \? total - 1 : null;/);
  assert.match(GLUE, /if \(dialog \|\| index === tail \|\| isPinned\(child, selection\)\) pins\.push\(index\);/);
});

test("rows the reader is working in are not unmounted", () => {
  const pinned = section(GLUE, "function isPinned(", "\n}\n");
  assert.match(pinned, /row\.contains\(document\.activeElement\)/);
  assert.match(pinned, /row\.querySelector\(PINNED_CONTENT\)/);
  assert.match(pinned, /if \(!media\.paused\) return true;/);
  assert.match(GLUE, /const PINNED_CONTENT = "textarea, \[contenteditable=true\], iframe";/);
  assert.match(pinned, /if \(selection\?\.some\(\(range\) => range\.intersectsNode\(row\)\)\) return true;/);
  // A selection reaching into the thread, or a held pointer, holds every mounted row.
  assert.match(GLUE, /trim: \(trim =\s*heldSelection\(viewport\) === null &&\s*!inputPointerHeld\(\) &&\s*document\.querySelector\(OPEN_DIALOG\) === null\),/);
  // A dialog a row opened is portaled out of it; while one is open every mounted row stays.
  assert.match(GLUE, /const dialog = document\.querySelector\(OPEN_DIALOG\) !== null;/);
  assert.match(
    section(GLUE, "function heldSelection(", "\n}\n"),
    // Every range, not the first: Firefox's Ctrl-drag selects several messages apart.
    /for \(let at = 0; at < selection\.rangeCount; at \+= 1\) \{\s*const range = selection\.getRangeAt\(at\);\s*if \(range\.intersectsNode\(viewport\)\) ranges\.push\(range\);/,
  );
});

test("a pinned row survives every way the window can leave it behind", () => {
  // A scrollbar drag replaces the window outright, and so do a turn jump and re-opening after a
  // full mount. Each took a row with an unsaved edit in it (the reply editor is uncontrolled, so
  // the draft is gone when the row comes back). The row is held as a pin, apart from the window:
  // stretching the window to it would mount everything between the edit and the reader.
  const keep = section(GLUE, "function keepPinned(", "\n}\n");
  assert.match(keep, /if \(index === undefined \|\| \(index >= next\.start && index < next\.end\)\) continue;/);
  assert.match(keep, /return withPins\(next, pins, lookup\.metas\.length\);/);
  // Rows are the viewport's own children, read in document order, so the pins come out ascending.
  assert.match(keep, /for \(const child of viewport\.children\)/);
  const step = section(GLUE, "const step = useCallback(", "stepRef.current = step;");
  // Every plan, sliding or filling, goes through it before it is compared or committed.
  assert.match(step, /if \(next != null\) next = keepPinned\(viewport, lookup, next, tail\);\s*if \(sameWindow\(next, current\)\)/);
  assert.match(step, /commit\(keepPinned\(viewport, lookup, next, tail\), true\);/);
  // The planner sees the window's rows only: a pin far away must not read as the window's edge.
  assert.match(step, /const inWindow = measured\.rows\.filter\(\s*\(row\) => row\.index >= current\.start && row\.index < current\.end,\s*\);/);
  assert.match(step, /rows: inWindow,/);
  assert.doesNotMatch(step, /pinned:/, "the planner does not hold rows by stretching the window");
  // A pin is let go when the reader leaves the row, which nothing else would step for.
  assert.match(GLUE, /viewport\.addEventListener\("focusout", schedule\);/);
  assert.match(GLUE, /viewport\.removeEventListener\("focusout", schedule\);/);
});

test("a spacer is keyed by where it sits among the rows", () => {
  // React moves a node when a sibling that used to follow it now precedes it. A spacer keyed by
  // the rows it borders never changes sides of a row, so no row's node is ever moved: a move
  // would drop the focus a pinned row is pinned for.
  assert.match(GLUE, /const key = slot\.from === 0 \? "top" : `top:\$\{slot\.from\}`;/);
  assert.match(GLUE, /const key = slot\.to === count \? "bottom" : `bottom:\$\{slot\.to\}`;/);
  assert.match(GLUE, /key=\{`gap:\$\{slot\.from\}`\}/);
  // A gap between pins is sized by React from the same heights the other two are.
  assert.match(GLUE, /height: gapHeight\(slot\.from, slot\.to\),\s*counterIncrement: counterIncrement\(slot\.from, slot\.to\),/);
  // The leading spacer holds no slack where it stands for no rows: at the first row, or at a pin.
  assert.match(GLUE, /const atFirstRow = win == null \|\| win\.start <= spacerRows\(win, count\)\.topFrom;/);
  // A spacer node this commit created is sized before any row's layout effect can read layout,
  // or the browser clamps scrollTop to a thread that is a few rows long for that moment.
  const early = section(GLUE, "useInsertionEffect(() => {", "}, [atFirstRow, spacers, viewportRef]);");
  assert.match(early, /const leading = atFirstRow \? 0 : Math\.max\(0, spacers\.top \+ slackRef\.current\);/);
  assert.match(early, /spacer\("top"\)\?\.style\.setProperty\("height", `\$\{leading\}px`\);/);
  assert.match(early, /spacer\("bottom"\)\?\.style\.setProperty\("height", `\$\{spacers\.bottom\}px`\);/);
  assert.ok(
    GLUE.indexOf("useInsertionEffect(() => {") < GLUE.indexOf("useLayoutEffect(() => {\n    winRef.current = win;"),
  );
  assert.match(GLUE, /if \(!top \|\| current\.start <= spacerRows\(current, lookup\.metas\.length\)\.topFrom\) \{/);
});

test("a step runs on a frame, and WebKit commits inside it", () => {
  const schedule = section(GLUE, "const schedule = useCallback(", "}, []);");
  assert.match(schedule, /if \(frameRef\.current\) return;/);
  assert.match(schedule, /frameRef\.current = requestAnimationFrame\(/);
  assert.doesNotMatch(
    schedule,
    /setTimeout|setInterval|queueMicrotask/,
    "cancelAnimationFrame does not cancel a timer, so an unmounted thread would still step",
  );
  assert.match(GLUE, /cancelAnimationFrame\(frameRef\.current\);/);
  const commit = section(GLUE, "const commit = useCallback(", "[viewportRef],");
  assert.match(commit, /if \(sync\) flushSync\(apply\);\s*else startTransition\(apply\);/);
  // A trim is cheap and must not lag the measurement it was planned from.
  assert.match(GLUE, /const trimOnly = next != null && next\.start >= current\.start && next\.end <= current\.end;/);
  assert.match(GLUE, /commit\(next, WEBKIT_GTK \|\| trimOnly \|\| exposed\);/);
  // One commit at a time: its layout effect owns the correction.
  assert.match(GLUE, /if \(!viewport \|\| pendingRef\.current\) return;/);
});

test("a full mount is paced like the old background mount, and waits for a quiet moment", () => {
  assert.match(GLUE, /const FILL_CHUNK = WEBKIT_GTK \? 2 : CHUNK_MESSAGES;/);
  const step = section(GLUE, "const step = useCallback(", "stepRef.current = step;");
  assert.match(
    step,
    /const chunk = WEBKIT_GTK\s*\?\s*\(chunkRef\.current = nextMountChunk\(\s*chunkRef\.current, frontier, FILL_CHUNK, WINDOW_REACH_PX\)\)\s*: FILL_CHUNK;/,
  );
  // The frontier is the window's first row, not a pinned row far above it.
  assert.match(step, /const frontier = inWindow\.length > 0 \? -inWindow\[0\]\.top : null;/);
  // Rows alone do not bound a pass: 32 very long replies in one commit held input for seconds
  // after select all. The pass is given the heights so it can stop at its budget.
  assert.match(step, /const heightOf = \(index: number\) => heights\.heightOf\(lookup\.metas\[index\]\);/);
  assert.match(step, /next = planFill\(current, total, chunk, heightOf\);/);
  // A commit interrupts native wheel momentum; scrolling by row does not wait, a full mount does.
  assert.match(
    step,
    /if \(scrollActivity\.current\?\.blocked\(inputPointerHeld\(\)\)\) \{\s*schedule\(\);\s*return;/,
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
  assert.match(GLUE, /anchorCorrection\(baseline, sampleAnchor\(viewport, element\), ANCHOR_PRECISION\)/);
  // The anchor is often a <pre> Streamdown replaces after Shiki, so fall back to the row.
  assert.match(GLUE, /const row = element\.closest\("\[data-role\]"\);/);
  assert.match(GLUE, /captured\.row\?\.isConnected && captured\.rowSample/);
  assert.match(GLUE, /scrollTop: viewport\.scrollTop,/);
});

test("a completion waiter is settled when its thread goes away", () => {
  // Without an unmount path the pending promise hangs a DOM capture forever.
  assert.match(GLUE, /useEffect\(\(\) => flushWaiters, \[flushWaiters\]\)/);
  // Emptied only when flushed, so a cancelled frame keeps the waiters.
  const flush = section(GLUE, "const flushWaiters = useCallback(", "}, []);");
  assert.match(flush, /waitersRef\.current = \[\];\s*for \(const resolve of waiters\) resolve\(\);/);
});

test("native scroll anchoring is off for a commit and handed back when the window settles", () => {
  // Native anchoring would correct the same shift a second time.
  const commit = section(GLUE, "const commit = useCallback(", "[viewportRef],");
  assert.match(commit, /setProperty\("overflow-anchor", "none"\)/);
  assert.ok(
    commit.indexOf('setProperty("overflow-anchor", "none")') < commit.indexOf("pickAnchorRow(viewport)"),
    "anchoring goes off before the capture it would otherwise move",
  );
  // Between commits the browser compensates late reflows above the reader (images, fonts).
  const step = section(GLUE, "const step = useCallback(", "stepRef.current = step;");
  assert.match(step, /if \(sameWindow\(next, current\)\) \{[\s\S]{0,200}?settle\(viewport, current, lookup\);/);
  assert.match(section(GLUE, "const settle = useCallback(", "[adjustScroll, heights, schedule, topRef],"), /restoreScrollAnchoring\(viewport\);/);
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
  assert.match(glue, /adjustScroll\(scroll\);/);
});

test("the leading spacer takes a height surprise before scrollTop does", () => {
  // A scroll write interrupts wheel momentum and fights a held scrollbar thumb.
  const effect = section(GLUE, "useLayoutEffect(() => {\n    winRef.current = win;", "}, [win, atFirstRow, spacers,");
  assert.match(effect, /const taken = absorbShift\(spacer, shift, atFirstRow\);/);
  assert.match(effect, /scroll = taken\.scroll;/);
  assert.match(effect, /slackRef\.current = atFirstRow \? 0 : spacer - spacers\.top;/);
  // React must not own the heights: a re-render would put the pre-correction value back.
  assert.match(effect, /top\.style\.height = `\$\{spacer\}px`;/);
  assert.match(effect, /bottom\.style\.height = `\$\{spacers\.bottom\}px`;/);
  // React owns only flexShrink and the equation counters on these two, never their heights.
  assert.match(GLUE, /data-thread-spacer="top" aria-hidden="true" style=\{\{ flexShrink: 0, counterIncrement: counterIncrement\(slot\.from, slot\.to\) \}\}/);
  assert.match(GLUE, /data-thread-spacer="bottom" aria-hidden="true" style=\{\{ flexShrink: 0, counterIncrement: counterIncrement\(slot\.from, slot\.to\) \}\}/);
  // What the spacer took on is folded into scrollTop only once input has gone quiet.
  const settle = section(GLUE, "const settle = useCallback(", "[adjustScroll, heights, schedule, topRef],");
  assert.match(settle, /if \(scrollActivity\.current\?\.blocked\(inputPointerHeld\(\)\)\) \{/);
  assert.ok(
    settle.indexOf("scrollActivity.current?.blocked") < settle.indexOf("top.style.height ="),
    "the quiet check comes before the write",
  );
});

test("the hook pins a follower during the prepend commit", () => {
  const body = section(HOOK, "adjustImplRef.current = (", "const onWheel = ");
  assert.match(body, /if \(!userDetachedRef\.current\) \{\s*insertionRemainder = 0;\s*if \(!parkIfHeld\(\)\) \{\s*extendFollow\(\);\s*pinIfFollowing\(stabilize\(\)\);/);
  assert.match(body, /else if \(deltaPx !== 0\)/);
});

test("the hook's correction is instant, because the viewport is scroll-smooth", () => {
  // With scroll-smooth on the viewport, an animated write would still be in flight at the next
  // commit's write.
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

test("the escape hatch mounts every row and resolves after a paint", () => {
  // A single rAF resolves before the forced commit paints.
  assert.match(GLUE, /export async function completeProgressiveMounts\(/);
  assert.match(GLUE, /wants\?: \(viewport: HTMLElement \| null\) => boolean,/);
  assert.match(
    GLUE,
    /requestAnimationFrame\(\(\) => requestAnimationFrame\(\(\) => resolve\(\)\)\)/,
  );
  const complete = section(GLUE, "const complete = useCallback(", "[hidden, schedule],");
  assert.match(complete, /modeRef\.current = "fill";/);
  assert.match(complete, /if \(winRef\.current == null\) \{\s*resolve\(\);\s*return;/);
  assert.match(GLUE, /activeWindows\.delete\(entry\)/);
});

test("the whole document is mounted for the readers that need it", () => {
  // Find highlights DOM ranges, print lays out what is mounted, select all means every row.
  assert.match(GLUE, /const FIND_FLAG = "data-find-bar-open";/);
  const step = section(GLUE, "const step = useCallback(", "stepRef.current = step;");
  assert.match(
    step,
    /const readerHolds =\s*printingRef\.current \|\|\s*fullMountHolds\.size > 0 \|\|\s*document\.documentElement\.hasAttribute\(FIND_FLAG\);\s*if \(readerHolds\) modeRef\.current = "fill";/,
  );
  assert.match(GLUE, /attributeFilter: \[FIND_FLAG, \.\.\.LAYOUT_FLAGS\]/);
  const print = section(GLUE, "const beforePrint = () => {", "};");
  // Synchronous: the print dialog opens when this handler returns.
  assert.match(print, /commit\(null, true\);/);
  // Not for a hidden thread: it is not printed, and its step would leave it fully mounted.
  assert.match(print, /if \(winRef\.current != null && viewport\.clientHeight > 0\) commit\(null, true\);/);
  const key = section(GLUE, "const onKeyDown = (event: KeyboardEvent) => {", "};");
  assert.match(key, /event\.key\.toLowerCase\(\) !== "a"/);
  assert.match(key, /closest\?\.\(EDITABLE\)\) return;/);
  assert.match(key, /void complete\(\);/);
});

test("a hidden thread is left as it is", () => {
  // Off its route the chat page stays mounted with no boxes: rows measure zero, so a step would
  // record those heights and mount the whole thread two rows a frame (measured: 111 rows and
  // climbing while the Model hub was open).
  const step = section(GLUE, "const step = useCallback(", "stepRef.current = step;");
  const guard = step.indexOf("if (viewport.clientHeight === 0) {");
  assert.notEqual(guard, -1);
  // A reader already waiting on the thread when it was hidden is released, not left hanging.
  assert.match(step, /if \(viewport\.clientHeight === 0\) \{\s*flushWaiters\(\);\s*return;\s*\}/);
  assert.ok(guard < step.indexOf("measureRows("), "the guard must come before any row is measured");
  assert.match(GLUE, /const resize = new ResizeObserver\(schedule\);\s*resize\.observe\(viewport\);/);
});

test("a full mount that ends without a signal is looked at again", () => {
  // A selection cleared, a pointer lifted and the release time passing fire nothing the window
  // listens to, so without this a thread mounted whole for one copy stays whole.
  const step = section(GLUE, "const step = useCallback(", "stepRef.current = step;");
  assert.match(
    step,
    /if \(next != null && !readerHolds\) \{\s*window\.clearTimeout\(settleTimerRef\.current\);\s*settleTimerRef\.current = window\.setTimeout\(schedule, HOLD_RECHECK_MS\);/,
  );
});

test("a copy that runs across a spacer mounts the rows it stands in for first", () => {
  // The clipboard takes DOM text, so a copy over unmounted rows would be a short conversation.
  const copy = section(GLUE, "const onCopy = (event: ClipboardEvent) => {", "\n    };");
  assert.match(copy, /if \(winRef\.current == null\) return;/);
  assert.match(copy, /spacer\.offsetHeight > 0 && selection\.containsNode\(spacer, true\)/);
  // Synchronous: the browser reads the selection when this handler returns.
  assert.match(copy, /commit\(null, true\);/);
  // WebKit serialises the selection it resolved before the mount (measured: 19,434 characters
  // of a 1,985,002 character thread), so the selection is set again, ends back on the spacers.
  assert.match(copy, /range\.compareBoundaryPoints\(Range\.START_TO_START, edge\) > 0/);
  // Rows, read after the mount: a spacer keyed by a pin is replaced by it.
  assert.match(copy, /const \{ topFrom, bottomTo \} = spacerRows\(winRef\.current, lookup\.metas\.length\);\s*modeRef/);
  assert.match(copy, /range\.setStartBefore\(first\);/);
  assert.match(copy, /range\.setEndAfter\(last\);/);
  assert.match(copy, /selection\.removeAllRanges\(\);\s*selection\.addRange\(range\);/);
  // Capture, on the document: the thread's own copy handler must see the full mount.
  assert.match(GLUE, /document\.addEventListener\("copy", onCopy, true\);/);
  assert.match(GLUE, /document\.removeEventListener\("copy", onCopy, true\);/);
});

test("a windowed thread publishes its length and each row's place in it", () => {
  const publish = section(GLUE, "function publishOrdinals(", "\n}\n");
  assert.match(publish, /const total = String\(lookup\.metas\.length\);/);
  assert.match(publish, /const position = String\(index \+ 1\);/);
  assert.match(publish, /child\.setAttribute\("aria-posinset", position\);/);
  assert.match(publish, /child\.setAttribute\("aria-setsize", total\);/);
  // A full mount publishes nothing, and clears what a window left behind.
  assert.match(publish, /child\.removeAttribute\("aria-posinset"\);/);
  assert.match(GLUE, /if \(!windowed && !ordinalsRef\.current\) return;/);
});

test("a hold keeps every row mounted until its release", () => {
  const hold = section(GLUE, "export function holdProgressiveMounts(", "\n}\n");
  assert.match(hold, /fullMountHolds\.add\(hold\);/);
  assert.match(hold, /if \(fullMountHolds\.delete\(hold\)\) for \(const entry of activeWindows\) entry\.wake\(\);/);
  assert.match(GLUE, /wake: schedule,/);
});

test("a settled window withholds rows without counting as a pending mount", () => {
  // The turn rail waits for pending mounts; a window that never fills must not hold it off.
  assert.match(
    section(GLUE, "export function hasPendingProgressiveMounts(", "\n}\n"),
    /if \(entry\.filling\(\)\) return true;/,
  );
  assert.match(GLUE, /filling: \(\) => modeRef\.current === "fill" && winRef\.current != null && !hidden\(\),/);
  assert.match(GLUE, /withholding: \(\) => winRef\.current != null && !hidden\(\),/);
});

test("a hidden thread holds no reader up", () => {
  // Find opened in a chat and then Compare entered: the chat stays mounted with no boxes, its
  // step stops at the hidden guard, and its full mount never finishes. Counted as pending it
  // kept the turn rail of every Compare pane from rendering until Compare was closed.
  assert.match(GLUE, /const hidden = useCallback\(\(\) => viewportRef\.current\?\.clientHeight === 0, \[viewportRef\]\);/);
  // And a reader that asks a hidden thread for every row is answered at once: the waiter would
  // never be resolved, and the request must not turn into a full mount when the thread is shown.
  const complete = section(GLUE, "const complete = useCallback(", "[hidden, schedule],");
  const answered = complete.indexOf("if (hidden()) {");
  assert.notEqual(answered, -1);
  assert.ok(answered < complete.indexOf('modeRef.current = "fill";'));
  assert.ok(answered < complete.indexOf("waitersRef.current.push(resolve);"));
});

test("a full mount nobody is waiting for is dropped", () => {
  // Find closed half way through a long thread: the rest of the mount is seconds of work for
  // no reader. A waiter (select all, a capture) still gets every row.
  const step = section(GLUE, "const step = useCallback(", "stepRef.current = step;");
  assert.match(
    step,
    /modeRef\.current === "fill" &&\s*\(current == null \|\| waitersRef\.current\.length === 0\) &&\s*performance\.now\(\) >= releaseAtRef\.current/,
  );
});

test("a jump asks for its row instead of for the whole thread", () => {
  assert.match(GLUE, /export function revealProgressiveRow\(viewport: HTMLElement, messageId: string\): boolean/);
  const reveal = section(GLUE, "const reveal = useCallback(", "[aui, readMetas, viewportRef],");
  // Mounted already: in the window, or pinned outside it.
  assert.match(reveal, /\(index >= current\.start && index < current\.end\) \|\|\s*current\.pins\?\.includes\(index\)\s*\) \{\s*return true;/);
  // Synchronous, so the caller finds the row in the DOM when this returns.
  assert.match(reveal, /flushSync\(\(\) => \{\s*setWin\(next\);\s*\}\);/);
  const jump = section(TURNS, "const jumpToTurn = useCallback(", "[viewportRef, detachFromBottom],");
  assert.ok(
    jump.indexOf("revealProgressiveRow(viewport, messageId)") < jump.indexOf("completeProgressiveMounts("),
    "the turn rail reveals one row before falling back to a full mount",
  );
  // Rows come and go as the reader moves, so the rail's in-view markers follow them.
  assert.match(TURNS, /const unsubscribe = subscribeMountedRows\(watch\);/);
});

test("the viewport comes from a ref, not a document-wide query", () => {
  // The Compare panes each mount their own Thread, so a document-wide query finds whichever
  // viewport comes first rather than the one these rows are in.
  // Open dialogs are the one document-wide read: they are portaled out of every thread.
  assert.doesNotMatch(GLUE, /document\.querySelector\((?!OPEN_DIALOG\))/);
  assert.match(GLUE, /viewportRef\.current/);
});

test("the row map is memoized on the slot identity", () => {
  // Messages skips re-render only while its children function keeps identity.
  assert.match(
    GLUE,
    /useMemo\(\(\) => \{[\s\S]*?\}, \[count, win, renderMessage, gate, gapHeight, counterIncrement\]\)/,
  );
  assert.match(
    GLUE,
    /const gate = useMemo\(\(\) => createRowNotificationGate\(aui\), \[aui\]\)/,
  );
  assert.match(GLUE, /prev\.renderMessage === next\.renderMessage/);
  assert.match(THREAD, /const renderThreadMessage = proplessSlot\(/);
});

test("a window past a thread that shrank is dropped or clamped, never left to emit nothing", () => {
  // followCount drops or clamps it (thread-window.test.ts); null is the full mount.
  assert.match(GLUE, /\} else if \(win != null\) \{\s*const followed = followCount\(win, previousCount, count\);/);
  // The row map clamps too, in threadSlots: the state update above lands a render later.
  assert.match(GLUE, /return threadSlots\(win, count\)\.map\(/);
});

test("a window at the tail follows new messages", () => {
  // Otherwise a sent prompt and its reply would land in the trailing spacer.
  // followCount again; it is given the count the window was planned against.
  assert.match(GLUE, /followCount\(win, previousCount, count\)/);
  assert.match(GLUE, /setPreviousCount\(count\);/);
});

test("the window opens on the commit that first fills an empty tree", () => {
  // The thread mounts at count 0 before history arrives; the cold open must still be windowed.
  assert.match(GLUE, /previousCount === 0 && count > 0/);
});

test("the escape hatch re-reads the window set instead of sampling it once", () => {
  // Windows register from an effect, so an empty set is not trusted straight away.
  assert.match(GLUE, /let observed = false;/);
  // Must re-read the filtered set each call, or a declined window holds the loop open.
  assert.match(
    GLUE,
    /!wanted\(\)\.some\(\(entry\) => entry\.withholding\(\)\) &&\s*\(observed \|\| Date\.now\(\) >= deadline\)/,
  );
  assert.match(GLUE, /const wanted = \(\) =>[\s\S]*?activeWindows/);
  // The deadline must be positive and outlast a cold history load (~160ms).
  assert.match(
    GLUE,
    /const deadline = Date\.now\(\) \+ PROGRESSIVE_MOUNT_SEARCH_MS;/,
    "an empty set must only be believed after a search interval",
  );
  const searchMs = Number(
    /PROGRESSIVE_MOUNT_SEARCH_MS = (\d+)/.exec(GLUE)?.[1] ?? "0",
  );
  assert.ok(
    searchMs >= 200,
    `the search interval must outlast a cold open's history load, got ${searchMs}ms`,
  );
});

test("the hook is told about every commit, including the ones with nothing to apply", () => {
  // Resync bookkeeping even with zero shift: anchoring scroll events would re-attach a parked reader.
  const effect = section(GLUE, "useLayoutEffect(() => {\n    winRef.current = win;", "}, [win, atFirstRow, spacers,");
  assert.match(effect, /let scroll = 0;/);
  assert.match(effect, /\n    adjustScroll\(scroll\);/, "the call is not inside the captured-anchor branch");
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
  assert.match(GLUE, /for \(const child of layoutChildren\(anchor\)\)/);
  assert.match(GLUE, /getComputedStyle\(child\)\.display === "contents"/);
  assert.doesNotMatch(
    GLUE,
    /querySelector\("\[data-role\]"\)/,
    "the first row in the list is not the row the reader is looking at",
  );
  assert.match(GLUE, /const element = pickAnchorRow\(viewport\);/);
  assert.match(GLUE, /function holdAnchor\(viewport: HTMLElement, element: Element\): HeldAnchor/);
});

test("a row's height is its top to the top of what follows it, read in one pass", () => {
  // Margins between rows belong to the spacers too, or every unmount would shift the reader.
  // What follows is a row or a spacer: a pinned row has a spacer on both sides.
  const measure = section(GLUE, "function measureRows(", "\n}\n");
  assert.match(measure, /if \(index === undefined && !child\.hasAttribute\(SPACER_ATTRIBUTE\)\) continue;/);
  assert.match(measure, /if \(open\) close\(top - open\.top\);/);
  assert.match(measure, /if \(heights\.measure\(meta, Math\.max\(0, height\)\)\) \{/);
  // Where the thread starts, for the planner to find the row a jump lands on.
  assert.match(measure, /threadTop \?\?= top;/);
  // Rows are the viewport's own children: a nested transcript's rows are not this thread's.
  assert.match(measure, /for \(const child of viewport\.children\)/);
});

test("the thread's messages are read imperatively, not through a selector", () => {
  // A useAuiState selector on messages would walk the thread on every keystroke.
  assert.doesNotMatch(GLUE, /useAuiState\(\(\{ thread \}\) => thread\.messages\)/);
  assert.match(GLUE, /aui\.thread\(\)\.getState\(\)\.messages/);
  assert.match(GLUE, /const count = useAuiState\(\(\{ thread \}\) => thread\.messages\.length\);/);
});

test("a mounted row that shrinks or re-wraps wakes the window, and growth does not", () => {
  assert.match(
    GLUE,
    /if \(before && \(height < before\.height - 1 \|\| Math\.abs\(width - before\.width\) > 1\)\) \{\s*wake = true;/,
  );
  assert.match(GLUE, /if \(wake\) schedule\(\);/);
  // Heights are keyed on a row's layout: Chat width moves the column, not the viewport, and the
  // chat font, text size and code font move no width.
  assert.match(
    GLUE,
    /const column = viewport\.querySelector<HTMLElement>\(ROW_SELECTOR\);[\s\S]{0,120}?column != null && heights\.relayout\(`\$\{rowLayout\(column\)\}/,
  );
  const layout = section(GLUE, "function rowLayout(row: HTMLElement): string {", "\n}");
  for (const part of ["row.clientWidth", "style.fontFamily", "style.fontSize", "style.letterSpacing", '"--custom-code-font"', '"--custom-code-font-size"']) {
    assert.ok(layout.includes(part), `the row layout key lost ${part}`);
  }
  assert.match(layout, /LAYOUT_FLAGS\.map\(\(name\) => \(root\.hasAttribute\(name\) \? "1" : "0"\)\)/);
  for (const flag of ["data-code-wrap", "data-code-line-numbers", "data-chat-font", "data-ui-font-size", "data-code-font-size"]) {
    assert.ok(section(GLUE, "const LAYOUT_FLAGS = [", "];").includes(`"${flag}"`), `${flag} left the layout key`);
  }
  // A flag flip wakes the window: an idle reader has no scroll to step it.
  assert.match(GLUE, /attributeFilter: \[FIND_FLAG, \.\.\.LAYOUT_FLAGS\]/);
  assert.doesNotMatch(GLUE, /heights\.re(?:size|layout)\(viewport\.clientWidth\)/);
  // Rows come and go: removed ones are let go, so the observer holds no detached row.
  assert.match(GLUE, /rowResize\.unobserve\(node\)/);
  assert.match(GLUE, /rowsChanged\.observe\(viewport, \{ childList: true \}\);/);
  assert.match(GLUE, /rowResize\.disconnect\(\);\s*rowsChanged\.disconnect\(\);/);
});

test("a reader that stops waiting for a full mount lets the window slide again", () => {
  const complete = section(GLUE, "const complete = useCallback(", "[hidden, schedule],");
  assert.match(complete, /if \(signal\?\.aborted\) \{\s*resolve\(\);/);
  assert.match(complete, /waitersRef\.current = waitersRef\.current\.filter\(\(waiter\) => waiter !== resolve\);/);
  assert.match(complete, /if \(waitersRef\.current\.length === 0\) releaseAtRef\.current = performance\.now\(\);/);
  assert.match(GLUE, /await Promise\.all\(pending\.map\(\(entry\) => entry\.complete\(signal\)\)\);/);
});

test("only a rendered reply's equations count toward the spacers", () => {
  assert.match(
    GLUE,
    /equations: renders && role === "assistant" \? numberedEquations\(message\.content, expanded\) : 0,/,
  );
});

test("a relayout re-renders every spacer even when the window holds", () => {
  assert.match(
    GLUE,
    /if \(relaid\) \{\s*commit\(\{ \.\.\.current \}, true\);\s*return;/,
  );
  // The expanded settings move every row and feed the estimate, so both key the caches.
  assert.match(GLUE, /heights\.relayout\(`\$\{rowLayout\(column\)\}\\n\$\{thinking\}\\n\$\{tools\}`\)/);
  assert.match(GLUE, /metasOf\(aui\.thread\(\)\.getState\(\)\.messages as unknown as readonly ThreadMessage\[\], expanded\)/);
});

test("a copy from a field leaves the window alone", () => {
  const copy = section(GLUE, "const onCopy = (event: ClipboardEvent) => {", "\n    };");
  assert.match(copy, /const target = event\.target as Element \| null;\s*if \(target\?\.closest\?\.\(EDITABLE\)\) return;/);
  assert.match(GLUE, /const EDITABLE = "input, textarea, \[contenteditable=true\]";/);
});
