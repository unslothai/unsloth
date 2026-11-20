// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  clampQuotePillPosition,
  placeQuoteDropdown,
  quotePillDragExceeded,
} from "../src/components/assistant-ui/message-quote-position.ts";

const viewport = { left: 0, top: 0, width: 800, height: 600 };
const pillSize = { width: 200, height: 40 };

test("the pill is clamped fully inside the chat viewport", () => {
  assert.deepEqual(
    clampQuotePillPosition(
      { left: -100, top: -50 },
      pillSize,
      viewport,
      8,
    ),
    { left: 8, top: 8 },
  );
  assert.deepEqual(
    clampQuotePillPosition(
      { left: 2000, top: 2000 },
      pillSize,
      viewport,
      8,
    ),
    { left: 592, top: 552 },
  );
});

test("clamping respects the viewport origin", () => {
  assert.deepEqual(
    clampQuotePillPosition(
      { left: 0, top: 0 },
      pillSize,
      { left: 100, top: 50, width: 400, height: 300 },
      12,
    ),
    { left: 112, top: 62 },
  );
});

test("a drag starts only after the four-pixel threshold", () => {
  assert.equal(quotePillDragExceeded(10, 10, 13, 10), false);
  assert.equal(quotePillDragExceeded(10, 10, 14, 10), true);
  assert.equal(quotePillDragExceeded(10, 10, 13, 13), true);
  assert.equal(quotePillDragExceeded(10, 10, 10, 10, 10), false);
});

test("the dropdown opens below when it fits", () => {
  assert.deepEqual(
    placeQuoteDropdown(
      { left: 100, top: 100 },
      pillSize,
      { width: 300, height: 200 },
      viewport,
      8,
      6,
    ),
    {
      left: 50,
      top: 146,
      maxHeight: 446,
      side: "below",
    },
  );
});

test("the dropdown opens above near the composer", () => {
  assert.deepEqual(
    placeQuoteDropdown(
      { left: 100, top: 500 },
      pillSize,
      { width: 300, height: 300 },
      viewport,
      8,
      6,
    ),
    {
      left: 50,
      top: 194,
      maxHeight: 486,
      side: "above",
    },
  );
});

test("the dropdown clamps horizontally and caps its height", () => {
  assert.deepEqual(
    placeQuoteDropdown(
      { left: 700, top: 100 },
      pillSize,
      { width: 300, height: 900 },
      viewport,
      8,
      6,
    ),
    {
      left: 492,
      top: 146,
      maxHeight: 446,
      side: "below",
    },
  );
});
