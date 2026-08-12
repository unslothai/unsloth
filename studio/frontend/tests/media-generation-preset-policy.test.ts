// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  chainDynamicDefaultRollback,
  closestDurationIndex,
  closestResolutionIndex,
  getBuiltinVariantName,
  mergeUntouchedParams,
} from "../src/features/generation-presets/preset-policy.ts";

test("pending model defaults preserve only fields the user edited", () => {
  const baseline = { negativePrompt: "", width: 768, steps: 8 };
  assert.deepEqual(
    mergeUntouchedParams(
      baseline,
      { ...baseline, negativePrompt: "keep this", width: 1024 },
      { negativePrompt: "", width: 1280, steps: 40 },
    ),
    { negativePrompt: "keep this", width: 1024, steps: 40 },
  );
});

test("dynamic default rollbacks unwind newest-first to the original baseline", () => {
  const calls: string[] = [];
  const previous = () => {
    calls.push("previous");
    return true;
  };
  const next = () => {
    calls.push("next");
    return true;
  };
  const rollback = chainDynamicDefaultRollback(previous, next);
  assert.equal(rollback(), true);
  assert.deepEqual(calls, ["next", "previous"]);
});

test("a stale newer rollback cannot continue into an older transaction", () => {
  let previousCalled = false;
  const rollback = chainDynamicDefaultRollback(
    () => {
      previousCalled = true;
      return true;
    },
    () => false,
  );
  assert.equal(rollback(), false);
  assert.equal(previousCalled, false);
});

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
