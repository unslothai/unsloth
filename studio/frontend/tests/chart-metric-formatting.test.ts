// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Both formatters drop to zero decimals at 1000, so the trailing-zero trim has
// no decimal point left to work on and used to eat significant digits.

import assert from "node:assert/strict";
import test from "node:test";

import {
  formatAxisMetric,
  formatMetric,
} from "../src/features/studio/sections/charts/utils.ts";

test("a large metric keeps every digit", () => {
  for (const [value, expected] of [
    [1000, "1000"],
    [1500, "1500"],
    [2000, "2000"],
    [12_000, "12000"],
    [25_000, "25000"],
    [1_000_000, "1000000"],
    [-2000, "-2000"],
  ] as [number, string][]) {
    assert.equal(formatMetric(value), expected, `formatMetric(${value})`);
    assert.equal(
      formatAxisMetric(value),
      expected,
      `formatAxisMetric(${value})`,
    );
  }
});

test("a large metric with a fraction still rounds to whole digits", () => {
  assert.equal(formatMetric(3000.4), "3000");
  assert.equal(formatAxisMetric(3000.4), "3000");
});

test("padding after a decimal point is still trimmed", () => {
  assert.equal(formatMetric(999.5), "999.5");
  assert.equal(formatMetric(100), "100");
  assert.equal(formatMetric(1), "1");
  assert.equal(formatMetric(0.5), "0.5");
  assert.equal(formatMetric(0.01), "0.01");
  assert.equal(formatAxisMetric(100), "100");
  assert.equal(formatAxisMetric(2.5), "2.5");
});

test("negative zero reads as zero", () => {
  assert.equal(formatMetric(-0), "0");
  assert.equal(formatAxisMetric(-0), "0");
  // toFixed rounds this to "-0" before the trim sees it.
  assert.equal(formatAxisMetric(-0.000001), "0");
});

test("a non-finite metric is still reported as zero", () => {
  for (const value of [Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.equal(formatMetric(value), "0");
    assert.equal(formatAxisMetric(value), "0");
  }
});

test("no formatted metric is shorter than its integer part", () => {
  // The original defect showed up as silent truncation, so pin the invariant
  // rather than only the values that happened to trigger it.
  for (let value = 1; value <= 2_000_000; value *= 10) {
    for (const scale of [1, 2, 5, 25]) {
      const metric = value * scale;
      const digits = String(Math.trunc(metric)).length;
      for (const [name, formatted] of [
        ["formatMetric", formatMetric(metric)],
        ["formatAxisMetric", formatAxisMetric(metric)],
      ] as [string, string][]) {
        assert.ok(
          formatted.replace("-", "").split(".")[0].length >= digits,
          `${name}(${metric}) = ${formatted} lost digits`,
        );
      }
    }
  }
});
