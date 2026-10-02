// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(
  new URL(
    "../src/features/studio/wizard/run-preview-card.tsx",
    import.meta.url,
  ),
  "utf8",
);
const hardwareSource = await readFile(
  new URL(
    "../src/features/studio/sections/training-hardware-params.tsx",
    import.meta.url,
  ),
  "utf8",
);

test("run preview computes global batch using selected DDP GPU count", () => {
  assert.match(source, /const globalBatch = accumulatedBatch \* gpuCount/);
  assert.match(source, /parallelismMode === "ddp"[\s\S]*selectedGpuIds\?\.length/);
});

test("run preview shows selected GPUs as separate indexed entries", () => {
  assert.match(source, /GPU \{device\.index\}: \{device\.name\}/);
  assert.doesNotMatch(source, /gpu\.name\} · \$\{gpu\.memoryTotalGb/);
  assert.match(source, /Total: \$\{displayedGpuMemoryGb\.toFixed\(1\)/);
  assert.match(source, /const displayedGpuMemoryGb = displayedGpuDevices\.reduce/);
  assert.match(source, /total \+ device\.memoryTotalGb/);
  assert.doesNotMatch(source, /Total: \$\{gpu\.memoryTotalGb/);
});

test("hardware inventory renders one GPU per line", () => {
  assert.match(hardwareSource, /selectable\.map\(\(device\) => \(/);
  assert.match(hardwareSource, /GPU \{device\.index\}: \{device\.name\}/);
  assert.doesNotMatch(hardwareSource, /\.join\(" · "\)/);
});

test("DDP is only selectable on a CUDA backend", () => {
  assert.match(hardwareSource, /disabled=\{selectable\.length < 2 \|\| gpu\.backend !== "cuda"\}/);
});