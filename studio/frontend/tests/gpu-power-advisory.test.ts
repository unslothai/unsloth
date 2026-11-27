// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { gpuPowerLimitAdvisory } from "../src/features/settings/lib/gpu-power-advisory.ts";

test("warns at or above 95% of a known power limit", () => {
  assert.equal(gpuPowerLimitAdvisory(190, 200), true);
  assert.equal(gpuPowerLimitAdvisory(200, 200), true);
  assert.equal(gpuPowerLimitAdvisory(189.9, 200), false);
});

test("does not warn when telemetry is incomplete or invalid", () => {
  for (const [draw, limit] of [
    [undefined, 200],
    [190, undefined],
    [null, 200],
    [190, null],
    [Number.NaN, 200],
    [190, Number.POSITIVE_INFINITY],
    [-1, 200],
    [190, 0],
  ] as const) {
    assert.equal(gpuPowerLimitAdvisory(draw, limit), false);
  }
});
