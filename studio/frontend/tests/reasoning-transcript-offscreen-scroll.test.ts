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

test("a transcript watches its distance from the thread viewport", () => {
  assert.match(
    transcript,
    /new IntersectionObserver\([\s\S]*?\{ root: scroll, rootMargin: "100% 0px" \}/,
    "no proximity observer rooted at the thread viewport",
  );
  assert.match(
    transcript,
    /proximity\?\.disconnect\(\)/,
    "the proximity observer is never disconnected",
  );
});

test("a far transcript drops both viewport scroll listeners and resumes when near", () => {
  const callback = transcript.slice(
    transcript.indexOf("new IntersectionObserver("),
    transcript.indexOf('{ root: scroll, rootMargin: "100% 0px" }'),
  );
  assert.match(
    callback,
    /followOffset\.current\?\.\(nearby\.current\)/,
    "offset observer not toggled",
  );
  assert.match(
    callback,
    /scroll\.removeEventListener\("scroll", schedule\)/,
    "measure listener kept offscreen",
  );
  assert.match(
    callback,
    /scroll\.addEventListener\("scroll", schedule/,
    "measure listener not restored",
  );
  assert.match(callback, /schedule\(\)/, "no catch-up measure on return");

  const observe = transcript.slice(
    transcript.indexOf("observeElementOffset: (instance, callback) => {"),
    transcript.indexOf("overscan:"),
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

test("a far transcript skips the reading-anchor scan", () => {
  assert.match(
    transcript,
    /if \(nearby\.current\)\s*for \(const row of element\.querySelectorAll<HTMLElement>\(\s*"\[data-index\]",?\s*\)\)/,
  );
});
