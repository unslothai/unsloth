// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { test } from "node:test";
import { basePrecisionLabel } from "../src/features/images/train/base-precision-label.ts";

test("the NF4 training base is labelled apart from NVFP4", () => {
  assert.equal(basePrecisionLabel("nf4"), "4-bit NF4 (not NVFP4)");
});

test("the other base precisions keep their labels", () => {
  assert.equal(basePrecisionLabel("auto"), "Auto");
  assert.equal(basePrecisionLabel("bf16"), "bf16 (fastest)");
  assert.equal(basePrecisionLabel("int8"), "int8");
  assert.equal(basePrecisionLabel("fp8"), "fp8 (experimental)");
  assert.equal(basePrecisionLabel("mxfp8"), "mxfp8 (Blackwell)");
});

test("the train panel renders the shared label and names NF4 in its hint", async () => {
  const src = await readFile(
    new URL("../src/features/images/train/diffusion-train-panel.tsx", import.meta.url),
    "utf8",
  );
  assert.match(src, /\{basePrecisionLabel\(m\)\}/);
  assert.doesNotMatch(src, /"nf4 \(lowest VRAM\)"/);
  assert.match(src, /4-bit NF4 \(not NVFP4\) uses the least VRAM/);
});
