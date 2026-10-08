// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The live offload panel's grid: layers never offloaded stay resident, the running layer and the
// next `depth` offloaded ones are on the card, every other offloaded layer is in system RAM.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { layerPlacement, offloadCountFromInput } = await import("../src/features/studio/sections/offload-panel-layout.ts");

test("the window follows the offloaded order, not the layer index", () => {
  // Spread placement: every other layer offloaded.
  const cells = layerPlacement(8, [1, 3, 5, 7], 2, 1);
  assert.deepEqual(cells, [
    "resident", "host", "resident", "running", "resident", "copying", "resident", "copying",
  ]);
});

test("the window stops at the last offloaded layer", () => {
  const cells = layerPlacement(6, [3, 4, 5], 2, 2);
  assert.deepEqual(cells, ["resident", "resident", "resident", "host", "host", "running"]);
});

test("nothing offloaded is all resident", () => {
  assert.deepEqual(layerPlacement(3, [], 2, 0), ["resident", "resident", "resident"]);
});

test("clearing the Count box keeps the last count instead of turning offload off", () => {
  for (const text of ["", "0", "-3", "abc"]) assert.equal(offloadCountFromInput(text), null);
  assert.equal(offloadCountFromInput("14"), 14);
  assert.equal(offloadCountFromInput("6.7"), 6);
});
