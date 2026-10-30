// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { fewshotParam, seedFewshot } from "../src/features/benchmark/lib/fewshot.ts";

test("a task with no positive few-shot default clears the previous task's count", () => {
  assert.equal(seedFewshot(5), 5);
  assert.equal(seedFewshot(0), null);
  assert.equal(seedFewshot(null), null);
  assert.equal(seedFewshot(undefined), null);
});

test("an explicit 0 is sent as a zero-shot override, null as the task default", () => {
  assert.equal(fewshotParam(0, true), 0);
  assert.equal(fewshotParam(3, true), 3);
  assert.equal(fewshotParam(null, true), null);
  // The control is hidden for tasks that disable few-shot, so nothing is sent.
  assert.equal(fewshotParam(4, false), null);
});
