// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The context usage shows its counts beside a ring; squeezed, the ring alone.

import assert from "node:assert/strict";
import test from "node:test";

import { deriveContextUsageBar } from "../src/features/chat/lib/context-usage-bar-state.ts";

import { readSrc } from "./helpers/kit.ts";

test("the compact face is the ring, unless there is no window to fill", () => {
  // a known window: the ring alone, with no percent label
  assert.equal(deriveContextUsageBar({ used: 16384, total: 32768 })?.compactFace, null);
  // nothing counted yet: the empty ring
  assert.equal(deriveContextUsageBar({ used: null, total: 32768 })?.compactFace, null);
  // no window: the token count stands in for the ring
  assert.equal(deriveContextUsageBar({ used: 4096, total: null })?.compactFace, "4.1k");
});

test("the bar hands every input to the state, batching flags included", () => {
  const bar = readSrc("features/chat/components/context-usage-bar.tsx");
  assert.match(bar, /\(\{ className, \.\.\.input \}\) => \{\s*const state = deriveContextUsageBar\(input\);/);
});

test("the header shrinks the context bar before the model name", () => {
  const page = readSrc("features/chat/chat-page.tsx");
  assert.match(page, /"pointer-events-auto flex min-w-0 items-center gap-1"/);
  assert.match(page, /ml-auto flex min-w-min max-w-max grow basis-0 items-center gap-1 \*:shrink-0/);
});

test("both faces draw the ring, not the line", () => {
  const bar = readSrc("features/chat/components/context-usage-bar.tsx");
  assert.match(bar, /\{compactFace === null \? \(\s*<span[^>]*>\s*<UsageRing /);
  // Empty before anything is counted, but still drawn.
  assert.match(bar, /const showRing = compactFace === null;/);
  assert.match(bar, /\{showRing \? <UsageRing percent=\{percent\} stroke=\{severity\.stroke\} \/> : null\}/);
  assert.doesNotMatch(bar, /style=\{\{ width: `\$\{percent\}%` \}\}/);
});
