// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { formatEta } from "../src/features/hub/lib/format.ts";

const SECONDS_PER_MINUTE = 60;
const MINUTES_PER_HOUR = 60;
const HOURS_PER_DAY = 24;
const REPRO_ETA_DAYS = 753;
const REPRO_ETA_HOURS = 5;
const SECONDS_PER_HOUR = SECONDS_PER_MINUTE * MINUTES_PER_HOUR;
const MAX_DISPLAYABLE_ETA_SECONDS = HOURS_PER_DAY * SECONDS_PER_HOUR;
const REPRO_ETA_SECONDS =
  (REPRO_ETA_DAYS * HOURS_PER_DAY + REPRO_ETA_HOURS) * SECONDS_PER_HOUR;

test("download ETA hides estimates that are too uncertain to be useful", () => {
  assert.equal(formatEta(Number.NaN), "");
  assert.equal(formatEta(0), "");
  assert.equal(formatEta(SECONDS_PER_MINUTE - 1), "59s left");
  assert.equal(formatEta(SECONDS_PER_MINUTE), "1m left");
  assert.equal(formatEta(SECONDS_PER_MINUTE + 1), "1m 1s left");
  assert.equal(formatEta(SECONDS_PER_HOUR), "1h left");
  assert.equal(formatEta(SECONDS_PER_HOUR + SECONDS_PER_MINUTE), "1h 1m left");
  assert.equal(formatEta(REPRO_ETA_SECONDS), "");
  assert.equal(formatEta(MAX_DISPLAYABLE_ETA_SECONDS - 1), "23h 59m left");
  assert.equal(formatEta(MAX_DISPLAYABLE_ETA_SECONDS), "");
});
