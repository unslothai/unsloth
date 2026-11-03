// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Snap to grid lands a card's top-left on the canvas dots.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import { GRID, snapPoint } from "../src/features/flows/editor/placement.ts";

const canvas = readFileSync(
  new URL("../src/features/flows/editor/canvas.tsx", import.meta.url),
  "utf8",
);

test("the snap step is the dot gap, and both go to React Flow", () => {
  assert.match(canvas, /const SNAP_GRID: \[number, number\] = \[GRID, GRID\];/);
  assert.match(canvas, /snapToGrid=\{snapToGrid\}\s*snapGrid=\{SNAP_GRID\}/);
  assert.match(canvas, /gap=\{GRID\}\s*size=\{DOT_SIZE\}\s*offset=\{DOT_SIZE \/ 2\}/);
});

test("dots sit on multiples of the grid at any pan and zoom", () => {
  // @xyflow/react Background, dots: tile at pan % (gap*zoom), translated by -(offset*zoom),
  // circle at (radius, radius) with radius = size*zoom/2.
  const size = 1.2;
  const offset = size / 2;
  for (const [pan, zoom] of [[0, 1], [37, 1], [-113.5, 0.5], [512, 1.75]]) {
    const tile = GRID * zoom;
    for (let k = -2; k <= 2; k++) {
      const screen = (pan % tile) + k * tile - offset * zoom + (size * zoom) / 2;
      const flow = (screen - pan) / zoom;
      const nearest = Math.round(flow / GRID) * GRID;
      assert.ok(Math.abs(flow - nearest) < 1e-9, `pan ${pan} zoom ${zoom}: dot at ${flow}`);
    }
  }
});

test("placed, dropped and quick-added cards snap to the same grid", () => {
  assert.equal(GRID, 20);
  assert.deepEqual(snapPoint([29, -31], true), [20, -40]);
  assert.deepEqual(snapPoint([29.4, -31.6], false), [29, -32]);
  assert.match(canvas, /placementPosition\(flow, useUiStore\.getState\(\)\.snapToGrid\)/);
  const quickAdd = readFileSync(
    new URL("../src/features/flows/editor/quick-add.tsx", import.meta.url),
    "utf8",
  );
  assert.match(quickAdd, /snapPoint\(\s*\[quickAdd\.flow\.x, quickAdd\.flow\.y\],\s*useUiStore\.getState\(\)\.snapToGrid,/);
});
