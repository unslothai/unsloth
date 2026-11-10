// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { fewshotParam } from "../src/features/benchmark/lib/fewshot.ts";

test("an explicit zero-shot pick is sent as 0, not as the task default", () => {
  assert.equal(fewshotParam(true, true, true, 0), 0);
  assert.equal(fewshotParam(true, true, false, 5), 5);
  assert.equal(fewshotParam(true, false, false, 0), null);
  assert.equal(fewshotParam(true, false, true, 3), 3);
  assert.equal(fewshotParam(false, true, true, 4), null);
});

test("the page sends the field through fewshotParam and resets it to each task's default", () => {
  const page = readFileSync(
    new URL("../src/features/benchmark/benchmark-page.tsx", import.meta.url),
    "utf8",
  );
  assert.doesNotMatch(page, /numFewshot > 0 \? numFewshot : null/);
  assert.match(page, /num_fewshot: fewshotParam\(/);
  // An unknown (null) task default must not count as known, else the reset 0 is sent.
  assert.match(
    page,
    /fewshotSupported,\s*(?:\/\/[^\n]*\n\s*)?taskConfig\?\.num_fewshot != null,/,
  );
  assert.match(
    page,
    /if \(!fewshotPickedRef\.current\) setNumFewshot\(cfg\.num_fewshot \?\? 0\);/,
  );
});
