// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Each expanded trace kept two scroll listeners on the shared thread viewport, offscreen too, so
// a long chat paid O(transcripts) per scroll frame (#12025). Invisible in output: pinned at source.

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

test("a far transcript drops both viewport scroll listeners and resumes when near", () => {
  const toggle = between(transcript, "const setNearby = ", "const proximity =");
  const [near, far] = toggle.split("} else {");
  assert.match(
    far,
    /followOffset\.current\?\.\(false\)/,
    "offset observer kept offscreen",
  );
  assert.match(
    far,
    /scroll\.removeEventListener\("scroll", schedule\)/,
    "measure listener kept offscreen",
  );
  assert.match(
    near,
    /scroll\.addEventListener\("scroll", schedule/,
    "measure listener not restored",
  );
  // Coming back re-measures (its top is stale) and renders the live range before paint.
  assert.match(
    near,
    /if \(beforePaint\)[\s\S]*?flushSync\(\(\) => \{\s*measure\(\);\s*followOffset\.current\?\.\(true\);\s*\}\)/,
  );

  const observe = between(
    transcript,
    "observeElementOffset: (instance, callback) => {",
    "overscan:",
  );
  assert.match(
    observe,
    /if \(!on\) return;/,
    "offset observer cannot be stopped",
  );
  assert.match(
    observe,
    /callback\(instance\.scrollElement\?\.scrollTop \?\? 0, false\);\s*stop = observeElementOffset\(instance, callback\);/,
    "resubscribing does not seed the current offset first",
  );
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
  // A jump needs the before-paint wake; the observer's own wake can wait a frame.
  assert.match(
    between(transcript, "stopJumpWatch = onViewportJump(", "const proximity ="),
    /setNearby\(true, true\)/,
  );
});

test("a far transcript skips the reading-anchor scan", () => {
  assert.match(
    transcript,
    /if \(nearby\.current\)\s*for \(const row of element\.querySelectorAll<HTMLElement>\(\s*"\[data-index\]",?\s*\)\)/,
  );
});
