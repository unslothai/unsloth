// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// When the chat header is squeezed, the context bar shrinks to a ring with a short label.

import assert from "node:assert/strict";
import test from "node:test";

import {
  deriveContextUsageBar,
  formatPercent,
} from "../src/features/chat/lib/context-usage-bar-state.ts";

import { readSrc } from "./helpers/kit.ts";

test("a used window never reads as 0%, and 100% means full", () => {
  assert.equal(formatPercent(0), "0%");
  assert.equal(formatPercent(0.03), "<1%");
  assert.equal(formatPercent(42.9), "42%");
  assert.equal(formatPercent(99.9), "99%");
  assert.equal(formatPercent(100), "100%");
});

test("the compact label follows what the full face can state", () => {
  assert.equal(deriveContextUsageBar({ used: 16384, total: 32768 })?.compactFace, "50%");
  // no window: the token count stands alone
  assert.equal(deriveContextUsageBar({ used: 4096, total: null })?.compactFace, "4.1k");
  // nothing counted yet: the empty ring alone
  assert.equal(deriveContextUsageBar({ used: null, total: 32768 })?.compactFace, null);
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
