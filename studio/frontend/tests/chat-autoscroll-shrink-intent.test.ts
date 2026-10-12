// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const HOOK = readSrc("components/assistant-ui/use-intent-aware-autoscroll.tsx");

const between = (from: string, to: string): string => {
  const start = HOOK.indexOf(from);
  assert.notEqual(start, -1, `source assertion anchor is gone: ${from}`);
  const end = HOOK.indexOf(to, start + from.length);
  assert.notEqual(end, -1, `source assertion anchor is gone: ${to}`);
  return HOOK.slice(start, end);
};

test("an upward scroll that arrives with a shorter page is content leaving, not the reader leaving the bottom", () => {
  const body = between("const onScroll = () => {", "\n      };");
  assert.match(body, /const scrollHeight = el\.scrollHeight;/);
  assert.match(
    body,
    /if \(\s*distanceDelta > 0 &&\s*!clamped &&\s*\(keyed \|\| scrollHeight >= lastScrollHeight\)\s*\) \{\s*upwardAccumulator \+= distanceDelta;/,
    "a removal clamps scrollTop and content regrown before the scroll event lands must not read as the reader scrolling up, unless a scroll-up key was just pressed",
  );
  assert.match(body, /lastScrollHeight = scrollHeight;\s*requestTick\(\);/, "the height baseline advances with every scroll event");
});

test("leaving a pinned bottom as the page changes height, with no key or pointer behind it, is a clamp", () => {
  const body = between("const onScroll = () => {", "\n      };");
  assert.match(
    body,
    /const clamped =\s*lastDistanceFromBottom <= AT_BOTTOM_THRESHOLD_PX &&\s*scrollHeight !== lastScrollHeight &&\s*!keyed &&\s*!pointerHeld;/,
    "a row below shrinking as the next one lands pulls scrollTop up in the same event; detaching there stops the stream following",
  );
  assert.match(HOOK, /el\.addEventListener\("pointerdown", onPointerDown, \{ passive: true \}\);/, "a scrollbar drag starts with a pointerdown on the viewport");
  assert.match(HOOK, /el\.ownerDocument\.addEventListener\("pointerup", onPointerUp, true\);/);
  assert.match(HOOK, /el\.removeEventListener\("pointerdown", onPointerDown\);/);
  assert.match(HOOK, /el\.ownerDocument\.removeEventListener\("pointerup", onPointerUp, true\);/);
});

test("the correction resyncs the height baseline with the others", () => {
  const body = between("adjustImplRef.current = (", "const onWheel = ");
  assert.match(body, /lastScrollTop = el\.scrollTop;\s*lastDistanceFromBottom = distanceFromBottom\(\);\s*lastScrollHeight = el\.scrollHeight;/);
});

test("the height baseline starts from the attached viewport", () => {
  assert.match(HOOK, /let lastDistanceFromBottom = distanceFromBottom\(\);\s*let lastScrollHeight = el\.scrollHeight;/);
});

test("a scroll-up key opens the window in which a shorter page still counts as the reader leaving", () => {
  const handler = between("const onKeyDown = (e: KeyboardEvent) => {", "const onTouchStart = ");
  for (const key of ['e.key === "ArrowUp"', 'e.key === "PageUp"', 'e.key === "Home"', '(e.key === " " && e.shiftKey)']) {
    assert.ok(handler.includes(key), `listens for ${key}`);
  }
  assert.match(handler, /e\.defaultPrevented/, "a key the page already handled does not scroll");
  assert.match(handler, /target\?\.isContentEditable \|\|\s*target\?\.closest\("input, textarea, select"\)/, "keys typed into the composer or a field move a caret, not the thread");
  assert.match(handler, /scrollKeyAt = performance\.now\(\);/);
  assert.match(HOOK, /el\.ownerDocument\.addEventListener\("keydown", onKeyDown, \{\s*passive: true,\s*\}\);/);
  assert.match(HOOK, /el\.ownerDocument\.removeEventListener\("keydown", onKeyDown\);/);
});

test("the first pin waits for the messages, then gives way to any later detach", () => {
  const tick = between("const tick = (): void => {", "const following =");
  assert.match(
    tick,
    /if \(pendingInitialPin && el\.querySelector\("\[data-role\]"\) !== null\) \{\s*pendingInitialPin = false;\s*if \(el\.scrollHeight > el\.clientHeight\) \{\s*parked = false;\s*userDetachedRef\.current = false;\s*extendFollow\(\);\s*\}\s*\}/,
    "a thread that opens without overflowing has nothing to pin, so opening a card by hand in it later keeps the view where the reader put it",
  );
});
