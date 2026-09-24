// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { formatCardTime } from "../src/features/library/format.ts";

// A Wednesday afternoon, local time.
const now = new Date(2026, 8, 23, 15, 0).getTime();
const at = (month: number, day: number, hour = 10, year = 2026) =>
  new Date(year, month, day, hour, 30).getTime();

test("card times read in the app's language, not the browser's", () => {
  assert.equal(formatCardTime(at(8, 22), "en", now), "Yesterday");
  assert.equal(formatCardTime(at(8, 22), "de", now), "Gestern");
  assert.equal(formatCardTime(at(8, 22), "fr", now), "Hier");
  assert.equal(formatCardTime(at(8, 22), "ja", now), "昨日");
  assert.equal(formatCardTime(at(8, 20), "en", now), "Sunday");
  assert.equal(formatCardTime(at(8, 20), "es", now), "domingo");
  // ICU may put a narrow no-break space before AM.
  assert.equal(formatCardTime(at(8, 23), "en", now).replace(/\s/g, " "), "10:30 AM");
  assert.equal(formatCardTime(at(7, 1), "en", now), "Aug 1");
  assert.equal(formatCardTime(at(7, 1, 10, 2025), "en", now), "Aug 1, 2025");
});

test("a time ahead of this clock shows its date, not a bare time", () => {
  assert.equal(formatCardTime(at(8, 25), "en", now), "Sep 25");
  assert.equal(formatCardTime(at(0, 2, 10, 2027), "en", now), "Jan 2, 2027");
  assert.equal(formatCardTime(Number.NaN, "en", now), "");
});
