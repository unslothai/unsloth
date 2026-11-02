// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { deriveContextUsageBar } from "../src/features/chat/lib/context-usage-bar-state.ts";

import { readSrc } from "./helpers/kit.ts";

test("the compact face is the ring, unless there is no window to fill", () => {
  assert.equal(deriveContextUsageBar({ used: 16384, total: 32768 })?.compactFace, null);
  assert.equal(deriveContextUsageBar({ used: null, total: 32768 })?.compactFace, null);
  assert.equal(deriveContextUsageBar({ used: 4096, total: null })?.compactFace, "4.1k");
});

test("the bar hands every input to the state, batching flags included", () => {
  const bar = readSrc("features/chat/components/context-usage-bar.tsx");
  assert.match(
    bar,
    /\(\{ className, compactUnknownAsIcon = false, pinOnClick = false, hoverClassName, padX = 2\.5, ringFirst = false, fitContent = false, \.\.\.input \}\) => \{\s*const pin = useClickPin\(pinOnClick\);\s*const state = deriveContextUsageBar\(input\);/,
  );
});

test("the header shrinks the context bar before the model name", () => {
  const page = readSrc("features/chat/chat-page.tsx");
  assert.match(page, /"group\/chat-model-controls pointer-events-auto flex min-w-0 items-center gap-1"/);
  assert.match(page, /ml-auto flex min-w-min max-w-max grow basis-0 items-center gap-1 \*:shrink-0/);
});

test("both faces draw the ring, not the line", () => {
  const bar = readSrc("features/chat/components/context-usage-bar.tsx");
  assert.match(bar, /\{compactFace === null \? \(\s*<span[^>]*>\s*<UsageRing /);
  // The full face retains the state's ring policy; only the compact fallback can opt into an icon.
  assert.match(bar, /const showRing = state\.compactFace === null;/);
  assert.match(bar, /\{showRing && ringFirst \? <UsageRing percent=\{percent\} stroke=\{severity\.stroke\} \/> : null\}\s*<Face face=\{face\} \/>\s*\{showRing && !ringFirst \? <UsageRing /);
  assert.doesNotMatch(bar, /style=\{\{ width: `\$\{percent\}%` \}\}/);
});
