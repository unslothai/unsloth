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

const GLUE = read("../src/components/assistant-ui/progressive-messages.tsx");
const HOOK = read(
  "../src/components/assistant-ui/use-intent-aware-autoscroll.tsx",
);
const THREAD = read("../src/components/assistant-ui/thread.tsx");

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
    code(GLUE),
    /<ThreadPrimitive\.Messages\b/,
    "the row map must not switch primitives on convergence",
  );
  assert.doesNotMatch(
    code(THREAD),
    /<ThreadPrimitive\.Messages\b/,
    "Thread must render the progressive list, not the unbounded one",
  );
  assert.match(THREAD, /<ProgressiveMessages/);
});

test("rows outside the window are not rendered rather than rendered and hidden", () => {
  // The cost being avoided is mounting them. A display:none row costs the same to mount, so a
  // rewrite to hiding would keep the tests green and collect nothing.
  assert.doesNotMatch(code(GLUE), /display:\s*["']?none/);
  assert.doesNotMatch(code(GLUE), /visibility:\s*["']?hidden/);
  assert.doesNotMatch(code(GLUE), /content-visibility/);
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

test("the anchor is captured in document space, not viewport space", () => {
  // The widening commit is transition-deferred, so the user can scroll between capture and
  // commit. Viewport coordinates move when the user scrolls and document ones do not, so
  // measuring in viewport space would fold the user's own scrolling into the correction and
  // fight them.
  assert.match(GLUE, /getBoundingClientRect\(\)\.top \+ viewport\.scrollTop/);
});

test("the mount window never writes scrollTop itself", () => {
  // Ownership: the autoscroll hook owns scrollTop. Two writers per widening frame double the
  // forced layouts and can re-attach a deliberately detached user through onScroll.
  assert.doesNotMatch(code(GLUE), /scrollTop\s*=/);
  assert.doesNotMatch(code(GLUE), /\.scrollTo\(/);
  assert.doesNotMatch(code(GLUE), /scrollIntoView\(/);
  assert.match(GLUE, /adjustForContentInsertedAbove\(shift\)/);
});

test("the hook's correction stands down while the user is following", () => {
  // While following, the hook's own MutationObserver path pins to the bottom in the same frame,
  // and because widening only prepends that is the same pixel. Correcting as well would be a
  // second writer for no gain.
  const body = HOOK.slice(HOOK.indexOf("adjustImplRef.current = ("));
  assert.match(
    body,
    /if \(!userDetachedRef\.current \|\| deltaPx === 0\) \{\s*return;/,
  );
});

test("the hook's correction is instant, because the viewport is scroll-smooth", () => {
  // scroll-smooth on the viewport means an animated write would still be in flight when the next
  // widening frame issued the next one.
  assert.match(THREAD, /scroll-smooth/);
  const body = HOOK.slice(HOOK.indexOf("adjustImplRef.current = ("));
  assert.match(body, /behavior: "instant"/);
});

test("the correction advances the intent bookkeeping with its write", () => {
  // Otherwise onScroll sees a downward scroll nobody made, and a user who detached within
  // RE_ATTACH_THRESHOLD_PX of the bottom (detachFromBottom does exactly that when the composer
  // grows) is silently re-attached by their own correction and yanked to the bottom on the next
  // widening frame.
  const body = HOOK.slice(
    HOOK.indexOf("adjustImplRef.current = ("),
    HOOK.indexOf("const onWheel"),
  );
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
  assert.doesNotMatch(code(GLUE), /document\.querySelector/);
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
