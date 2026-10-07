// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The live offload panel's grid: layers never offloaded stay resident, the running layer and the
// next `depth` offloaded ones are on the card, every other offloaded layer is in system RAM.

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { groupLayersByCard, layerPlacement, vramUsage } = await import(
  "../src/features/studio/sections/offload-panel-layout.ts"
);

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

const cards = [{ index: 0, gpu_id: 2 }, { index: 1, gpu_id: 0 }];

test("layers group under the card they run on, in card order", () => {
  const groups = groupLayersByCard(6, { "0": 0, "1": 0, "2": 0, "3": 1, "4": 1, "5": 1 }, cards);
  assert.deepEqual(groups, [
    { card: cards[0], layers: [0, 1, 2] },
    { card: cards[1], layers: [3, 4, 5] },
  ]);
});

test("a card holding no decoder layers still gets its group", () => {
  const groups = groupLayersByCard(2, { "0": 1, "1": 1 }, cards);
  assert.deepEqual(groups.map((g) => g.layers), [[], [0, 1]]);
});

test("layers on no known card come last instead of vanishing", () => {
  const groups = groupLayersByCard(4, { "0": 0, "2": 1, "3": 7 }, cards);
  assert.deepEqual(groups.at(-1), { card: null, layers: [1, 3] });
  assert.equal(groups.flatMap((g) => g.layers).length, 4);
});

test("the VRAM bar reads peak use and draws a budget line only under 1", () => {
  const GIB = 1024 ** 3;
  assert.deepEqual(vramUsage(32 * GIB, 6 * GIB, 5 * GIB, 0.5), { used: 6 * GIB, budget: 16 * GIB, total: 32 * GIB });
  assert.deepEqual(vramUsage(32 * GIB, undefined, 5 * GIB, 1), { used: 5 * GIB, budget: null, total: 32 * GIB });
  assert.deepEqual(vramUsage(undefined, undefined, undefined, undefined), { used: 0, budget: null, total: 0 });
});
