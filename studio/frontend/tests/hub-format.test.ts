// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { formatEta } from "../src/features/hub/lib/format.ts";

const MINUTE = 60;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;
// The reported case: 6.8 GB left at 102 B/s rendered "753d 5h left".
const REPRO_SECONDS = (753 * 24 + 5) * HOUR;

test("sub-hour ETAs keep their seconds and minutes", () => {
  assert.equal(formatEta(1), "1s left");
  assert.equal(formatEta(MINUTE - 1), "59s left");
  assert.equal(formatEta(MINUTE), "1m left");
  assert.equal(formatEta(MINUTE + 1), "1m 1s left");
  assert.equal(formatEta(HOUR - 1), "59m 59s left");
});

test("hour ETAs drop the seconds", () => {
  assert.equal(formatEta(HOUR), "1h left");
  assert.equal(formatEta(HOUR + MINUTE), "1h 1m left");
  assert.equal(formatEta(DAY - 1), "23h 59m left");
});

test("an ETA of a day or more collapses to a bound instead of a day count", () => {
  assert.equal(formatEta(DAY), "> 24h left");
  assert.equal(formatEta(DAY + 1), "> 24h left");
  assert.equal(formatEta(REPRO_SECONDS), "> 24h left");
  assert.equal(formatEta(Number.MAX_SAFE_INTEGER), "> 24h left");
});

test("the cutoff applies to the rounded value, so it starts at 86399.5s", () => {
  assert.equal(formatEta(DAY - 0.51), "23h 59m left");
  assert.equal(formatEta(DAY - 0.5), "> 24h left");
});

test("unusable inputs render nothing rather than a bogus estimate", () => {
  assert.equal(formatEta(Number.NaN), "");
  assert.equal(formatEta(Number.POSITIVE_INFINITY), "");
  assert.equal(formatEta(Number.NEGATIVE_INFINITY), "");
  assert.equal(formatEta(0), "");
  assert.equal(formatEta(-1), "");
});

test("no ETA is ever reported in days", () => {
  for (let s = 1; s <= 3 * DAY; s += 137) {
    assert.doesNotMatch(formatEta(s), /\d+d\b/);
  }
});
