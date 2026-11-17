// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { formatCompact } from "../src/lib/utils.ts";

test("formatCompact moves to the next unit when the figure rounds to 1000", () => {
  assert.equal(formatCompact(999_950), "1.0M");
  assert.equal(formatCompact(999_999), "1.0M");
  assert.equal(formatCompact(999_950_000), "1.0B");
  assert.equal(formatCompact(999_999_999), "1.0B");
});

test("formatCompact is unchanged away from the unit boundaries", () => {
  assert.equal(formatCompact(999), "999");
  assert.equal(formatCompact(1_000), "1.0K");
  assert.equal(formatCompact(999_949), "999.9K");
  assert.equal(formatCompact(7_241_732_096), "7.2B");
  assert.equal(formatCompact(1_500_000), "1.5M");
  assert.equal(formatCompact(999_949_999), "999.9M");
});
