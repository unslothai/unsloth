// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  AXIS_H,
  GROUP_H,
  TOP,
  barDensity,
  clampWhisker,
  layoutBars,
  niceMax,
  ticks,
} from "../src/features/benchmarks/components/chart-math.ts";

test("a flat chart lays rows out exactly as the sweeps chart always has", () => {
  const l = layoutBars([{ labelled: false, rows: 4 }], 2);
  assert.equal(l.rowH, 46);
  assert.deepEqual(l.headTops, [null]);
  assert.deepEqual(l.rowTops[0], [TOP, TOP + 46, TOP + 92, TOP + 138]);
  assert.deepEqual(l.pendingTops, [TOP + 184, TOP + 230]);
  assert.equal(l.bottom, TOP + 6 * 46);
  assert.equal(l.plotH, TOP + 6 * 46 + AXIS_H);
});

test("labelled groups each get a heading band above their rows", () => {
  const l = layoutBars(
    [
      { labelled: true, rows: 2 },
      { labelled: true, rows: 1 },
    ],
    0,
  );
  assert.deepEqual(l.headTops, [TOP, TOP + GROUP_H + 2 * 46]);
  assert.deepEqual(l.rowTops, [
    [TOP + GROUP_H, TOP + GROUP_H + 46],
    [TOP + 2 * GROUP_H + 2 * 46],
  ]);
  assert.equal(l.bottom, TOP + 2 * GROUP_H + 3 * 46);
});

test("density tightens on the total row count, pending rows included", () => {
  assert.deepEqual(barDensity(9), { rowH: 46, barH: 26, fontSize: 13.5 });
  assert.deepEqual(barDensity(10), { rowH: 34, barH: 21, fontSize: 13.5 });
  assert.deepEqual(barDensity(15), { rowH: 28, barH: 17, fontSize: 12.5 });
  assert.equal(layoutBars([{ labelled: false, rows: 8 }], 2).rowH, 34);
});

test("a whisker is clamped to the axis and skipped when it has no width", () => {
  assert.deepEqual(clampWhisker(40, 60, 100), [40, 60]);
  assert.deepEqual(clampWhisker(-5, 8, 100), [0, 8]);
  assert.deepEqual(clampWhisker(95, 104, 100), [95, 100]);
  assert.equal(clampWhisker(50, 50, 100), null);
  assert.equal(clampWhisker(null, 50, 100), null);
  assert.equal(clampWhisker(50, undefined, 100), null);
  assert.equal(clampWhisker(120, 130, 100), null);
});

test("the axis rounds up past the data and ticks it in fifths", () => {
  assert.equal(niceMax(131.3), 150);
  assert.equal(niceMax(0), 1);
  assert.deepEqual(ticks(100), [0, 20, 40, 60, 80, 100]);
});
