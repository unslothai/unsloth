// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { withNvfp4Option } from "../src/lib/nvfp4-options.ts";

const OPTIONS: [string, string][] = [
  ["fp8", "FP8"],
  ["int8", "INT8"],
  ["nvfp4", "NVFP4 (Blackwell)"],
  ["mxfp8", "MXFP8 (Blackwell)"],
];

test("NVFP4 is hidden while the backend switch is off", () => {
  assert.deepEqual(
    withNvfp4Option(OPTIONS, false).map(([value]) => value),
    ["fp8", "int8", "mxfp8"],
  );
});

test("NVFP4 is offered, in place, when the backend enables it", () => {
  assert.deepEqual(withNvfp4Option(OPTIONS, true), OPTIONS);
});

test("the input list is never mutated", () => {
  const copy = OPTIONS.map((o) => [...o]);
  withNvfp4Option(OPTIONS, false);
  assert.deepEqual(OPTIONS, copy);
});
