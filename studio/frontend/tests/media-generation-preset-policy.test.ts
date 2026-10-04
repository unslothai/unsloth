// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  closestDurationIndex,
  closestResolutionIndex,
  getBuiltinVariantName,
  shouldApplyModelDefaults,
} from "../src/features/generation-presets/preset-policy.ts";

test("saving over Default creates a protected custom variant", () => {
  const used = new Set(["Default", "Default 1", "Portrait"]);
  assert.equal(getBuiltinVariantName(used), "Default 2");
});

test("video resolution mapping prioritizes aspect before area", () => {
  const options: [number, number][] = [
    [1024, 1024],
    [1216, 704],
    [704, 1216],
  ];
  assert.equal(closestResolutionIndex(options, 1920, 1080), 1);
  assert.equal(closestResolutionIndex(options, 1080, 1920), 2);
});

test("video duration mapping uses the closest supported temporal lattice", () => {
  const options = [
    { seconds: 1.04 },
    { seconds: 2.08 },
    { seconds: 3.12 },
    { seconds: 5.2 },
  ];
  assert.equal(closestDurationIndex(options, 4.9), 3);
  assert.equal(closestDurationIndex(options, 2.4), 1);
});

test("a stored recipe owns only the first model-default seed", () => {
  assert.equal(shouldApplyModelDefaults(false, true), false);
  assert.equal(shouldApplyModelDefaults(false, false), true);
  assert.equal(shouldApplyModelDefaults(true, true), true);
});

test("a preset selected while the model loaded outranks that load's defaults", () => {
  // The pick claimed the recipe, so every other input says "seed". Only the newer claim stops it.
  assert.equal(shouldApplyModelDefaults(true, false, true), false);
  assert.equal(shouldApplyModelDefaults(true, true, true), false);
  assert.equal(shouldApplyModelDefaults(true, false, false), true);
});
