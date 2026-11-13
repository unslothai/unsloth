// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Offscreen transcripts must not listen to the shared viewport's scroll (#12025); pinned at source.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const source = readSrc("components/assistant-ui/reasoning-transcript.tsx");
const transcript = source.slice(
  source.indexOf("export function ReasoningTranscript("),
);

function between(text: string, start: string, end: string): string {
  const from = text.indexOf(start);
  assert.notEqual(from, -1, `${start} is gone; this test needs rewriting`);
  const to = text.indexOf(end, from);
  assert.notEqual(to, -1, `${end} is gone; this test needs rewriting`);
  return text.slice(from, to);
}

test("a transcript watches its distance from the thread viewport", () => {
  assert.match(
    transcript,
    /new IntersectionObserver\([\s\S]*?setNearby\([\s\S]*?\{ root: scroll, rootMargin: "100% 0px" \}/,
    "no proximity observer rooted at the thread viewport",
  );
  assert.match(transcript, /proximity\?\.disconnect\(\)/);
  assert.match(
    transcript,
    /stopJumpWatch\?\.\(\);\s*scroll\.removeEventListener/,
  );
});

test("a far transcript drops its viewport scroll listener and resumes when near", () => {
  const toggle = between(transcript, "const setNearby = ", "const proximity =");
  const [near, far] = toggle.split("\n      scroll.removeEventListener(");
  assert.ok(far, "no far branch");
  assert.match(
    near,
    /scroll\.addEventListener\("scroll", schedule/,
    "measure listener not restored",
  );
  assert.match(
    near,
    /if \(beforePaint\) flushSync\(capture\);\s*else schedule\(\);/,
    "a jump back does not mount the rows in view before paint",
  );
  assert.match(far, /^"scroll", schedule\);/, "measure listener kept offscreen");
  assert.match(far, /cancelAnimationFrame\(frame\);\s*frame = 0;/);
});

test("a jump past the observer margin wakes far transcripts through one listener per viewport", () => {
  const watch = between(
    source,
    "function onViewportJump(",
    "export function ReasoningTranscript(",
  );
  assert.match(
    watch,
    /jumpWatchers\.get\(scroll\)/,
    "a listener per transcript instead of per viewport",
  );
  assert.match(watch, /Math\.abs\(top - last\) >= scroll\.clientHeight \/ 2/);
  assert.match(
    between(transcript, "stopJumpWatch = onViewportJump(", "const proximity ="),
    /setNearby\(true, true\)/,
  );
});

test("a far transcript still corrects a resize above the reader without the scroll listener", () => {
  const remeasure = between(transcript, "rowRemeasurers.set(element, (row) => {", "});");
  assert.match(remeasure, /element\.getBoundingClientRect\(\)\.bottom -/);
  assert.match(remeasure, /adjustAbove\(Math\.round\(shift\)\)/);
  assert.match(transcript, /const observer = new ResizeObserver\(/);
});

test("a far transcript drops its reading anchor and runs no anchor scan", () => {
  const toggle = between(transcript, "const setNearby = ", "const proximity =");
  assert.match(toggle, /reading = undefined;/);
  assert.match(transcript, /if \(nearby\.current\)\s*scroll\.addEventListener\("scroll", schedule/);
});
