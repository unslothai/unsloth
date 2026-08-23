// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The chat/model-load/training-overlay formatters. The 24h ETA clamp mirrors
// the hub formatter's, which has had one since #7679 (see hub-format.test.ts).

import assert from "node:assert/strict";
import test from "node:test";

import {
  formatEta,
  formatRate,
} from "../src/features/chat/utils/format-transfer.ts";

const DAY = 24 * 60 * 60;

test("formatEta renders the usual units", () => {
  assert.equal(formatEta(5), "5s");
  assert.equal(formatEta(65), "1m 5s");
  assert.equal(formatEta(120), "2m");
  assert.equal(formatEta(3725), "1h 2m");
  assert.equal(formatEta(7200), "2h");
});

test("formatEta clamps at 24h", () => {
  assert.equal(formatEta(DAY), "> 24h");
  assert.equal(formatEta(DAY * 40), "> 24h");
  // #7667's "753d 5h left" must be unreachable through this formatter.
  assert.equal(formatEta(753 * DAY), "> 24h");
});

test("formatEta clamps on the rounded value, as the hub formatter does", () => {
  assert.equal(formatEta(DAY - 0.5), "> 24h");
  assert.equal(formatEta(DAY - 1), "23h 59m");
});

test("formatEta rejects non-finite and non-positive input", () => {
  for (const bad of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.equal(formatEta(bad), "--");
  }
});

test("formatRate scales and rejects non-finite input", () => {
  assert.equal(formatRate(500), "500 B/s");
  assert.equal(formatRate(1024 * 10), "10.0 KB/s");
  assert.equal(formatRate(1024 ** 2 * 20), "20.0 MB/s");
  assert.equal(formatRate(1024 ** 3 * 2), "2.00 GB/s");
  for (const bad of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.equal(formatRate(bad), "--");
  }
});
