// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installLocalStorageFake,
  registerBundlerResolver,
} from "./helpers/kit.ts";

registerBundlerResolver();
installLocalStorageFake();

const {
  VRAM_BUDGET_PERCENT_DEFAULT,
  VRAM_BUDGET_PERCENT_MAX,
  VRAM_BUDGET_PERCENT_MIN,
  vramFractionToPercent,
  vramPercentToFraction,
} = await import(
  "../src/features/model-picker/model-config/per-model-config.ts"
);

test("percent and fraction round-trip exactly across the whole range", () => {
  for (
    let percent = VRAM_BUDGET_PERCENT_MIN;
    percent <= VRAM_BUDGET_PERCENT_MAX;
    percent += 1
  ) {
    assert.equal(vramFractionToPercent(vramPercentToFraction(percent)), percent);
  }
});

test("the default fraction is exactly 0.97, not a float-drifted neighbour", () => {
  // A drifted 0.9700000000000001 would never equal the backend default, so the
  // UI would show the budget as changed the moment the slider was dragged and
  // put back.
  assert.equal(vramPercentToFraction(VRAM_BUDGET_PERCENT_DEFAULT), 0.97);
  assert.equal(vramFractionToPercent(0.97), VRAM_BUDGET_PERCENT_DEFAULT);
});

test("the bounds mirror the backend range", () => {
  // vram_budget_settings.py: VRAM_FRACTION_MIN 0.80, MAX 1.00, DEFAULT 0.97.
  assert.equal(vramPercentToFraction(VRAM_BUDGET_PERCENT_MIN), 0.8);
  assert.equal(vramPercentToFraction(VRAM_BUDGET_PERCENT_MAX), 1);
  assert.equal(VRAM_BUDGET_PERCENT_DEFAULT, 97);
});

test("fractionToPercent rounds rather than truncating", () => {
  // A value set through UNSLOTH_VRAM_FRACTION need not be a whole percent.
  assert.equal(vramFractionToPercent(0.855), 86);
  assert.equal(vramFractionToPercent(0.854), 85);
});

test("percentToFraction tolerates a non-integer slider value", () => {
  assert.equal(vramPercentToFraction(90.4), 0.9);
  assert.equal(vramPercentToFraction(90.6), 0.91);
});
