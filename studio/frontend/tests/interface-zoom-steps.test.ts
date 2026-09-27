// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

// The store pulls in zustand persist and Tauri, so the stepping is lifted out and run on its own.
const STORE = readSrc("features/settings/stores/interface-scale-store.ts");
const steps = JSON.parse(STORE.match(/INTERFACE_ZOOM_STEPS = (\[[^\]]+\])/)![1]) as number[];
const range = STORE.match(/min: (\d+),\s*max: (\d+),\s*default: (\d+)/)!.slice(1).map(Number);
const stepInterfaceScale = (scale: number, direction: 1 | -1): number =>
  (direction > 0
    ? steps.find((step) => step > scale)
    : [...steps].reverse().find((step) => step < scale)) ?? scale;

test("the stops cover the whole range and include the default", () => {
  const [min, max, def] = range;
  assert.equal(steps[0], min);
  assert.equal(steps.at(-1), max);
  assert.ok(steps.includes(def));
  assert.deepEqual([...steps].sort((a, b) => a - b), steps);
});

test("zoom walks the stops and stops at the ends", () => {
  assert.equal(stepInterfaceScale(100, 1), 110);
  assert.equal(stepInterfaceScale(100, -1), 90);
  assert.equal(stepInterfaceScale(200, 1), 200);
  assert.equal(stepInterfaceScale(50, -1), 50);
});

test("a scale typed in Settings between stops moves to the nearest stop that way", () => {
  assert.equal(stepInterfaceScale(105, 1), 110);
  assert.equal(stepInterfaceScale(105, -1), 100);
});

test("the function the menu runs matches the one tested here", () => {
  assert.match(STORE, /INTERFACE_ZOOM_STEPS\.find\(\(step\) => step > scale\)/);
  assert.match(STORE, /\[\.\.\.INTERFACE_ZOOM_STEPS\]\.reverse\(\)\.find\(\(step\) => step < scale\)/);
});
