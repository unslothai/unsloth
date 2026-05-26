// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { formatBytes, formatEta, formatRate } from "../src/lib/format.ts";

test("formatBytes keeps unknown byte sentinels distinct from real zero", () => {
  assert.equal(formatBytes(-1), "N/A");
  assert.equal(formatBytes(Number.NaN), "N/A");
  assert.equal(formatBytes(Number.POSITIVE_INFINITY), "N/A");
  assert.equal(formatBytes(0), "0 B");
});

test("formatRate and formatEta return visible placeholders for unknown values", () => {
  assert.equal(formatRate(0), "--");
  assert.equal(formatRate(Number.NaN), "--");
  assert.equal(formatEta(0), "--");
  assert.equal(formatEta(Number.NEGATIVE_INFINITY), "--");
});
