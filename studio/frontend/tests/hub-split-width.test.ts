// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  clampSplitListWidth,
  keyboardSplitWidth,
  splitListMax,
} from "../src/features/hub/lib/split-width.ts";
import { readSrc } from "./helpers/kit.ts";

test("the list keeps 340px and the detail pane keeps 480px", () => {
  assert.equal(splitListMax(1100), 620);
  assert.equal(splitListMax(700), 340);
  assert.equal(clampSplitListWidth(100, 1100), 340);
  assert.equal(clampSplitListWidth(900, 1100), 620);
});

test("arrow keys step from the rendered width, not a wider stored one", () => {
  // Stored 900px on a 1100px row renders clamped at 620px; Left must move off that edge.
  assert.equal(keyboardSplitWidth(620, "ArrowLeft", 1100), 604);
  assert.equal(keyboardSplitWidth(620, "ArrowRight", 1100), 620);
  assert.equal(keyboardSplitWidth(500, "ArrowRight", 1100), 516);
  assert.equal(keyboardSplitWidth(500, "Home", 1100), null);
});

test("a cancelled drag commits the painted width like a release", () => {
  const src = readSrc("features/hub/components/hub-split-divider.tsx");
  assert.match(src, /onPointerUp=\{finishDrag\}/);
  assert.match(src, /onPointerCancel=\{finishDrag\}/);
  assert.match(src, /onLostPointerCapture=\{finishDrag\}/);
});
