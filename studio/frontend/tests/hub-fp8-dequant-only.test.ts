// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { fp8MayRunDequantized } from "../src/features/hub/lib/format-filters.ts";
import { readSrcAsync } from "./helpers/kit.ts";

const fp8 = { id: "unsloth/Qwen3.6-27B-FP8", isGguf: false };
const fp8Config = { id: "org/model", isGguf: false, quantMethod: "fp8" };
const nvfp4 = { id: "unsloth/Qwen3.6-27B-NVFP4", isGguf: false };
const gguf = { id: "unsloth/Qwen3.6-27B-GGUF", isGguf: true };
const bf16 = { id: "unsloth/Qwen3.6-27B", isGguf: false };

test("an FP8 checkpoint is flagged only when the backend reports fp8_rowwise as dequant-only", () => {
  assert.equal(fp8MayRunDequantized(fp8, ["fp8_rowwise"]), true);
  assert.equal(fp8MayRunDequantized(fp8Config, ["fp8_rowwise"]), true);
  assert.equal(fp8MayRunDequantized(fp8, []), false);
  // Older backends send nothing.
  assert.equal(fp8MayRunDequantized(fp8, undefined), false);
  for (const other of [nvfp4, gguf, bf16]) {
    assert.equal(fp8MayRunDequantized(other, ["fp8_rowwise"]), false, other.id);
  }
});

test("system info carries checkpoint_quant_dequant_only into the inspector chip", async () => {
  const gpuInfo = await readSrcAsync("hooks/use-gpu-info.ts");
  assert.match(
    gpuInfo,
    /checkpointQuantDequantOnly: normalizeDenseQuantSchemes\(\s*data\?\.checkpoint_quant_dequant_only,/,
  );
  const inspector = await readSrcAsync(
    "features/hub/catalog/model-inspector.tsx",
  );
  assert.match(inspector, /fp8MayRunDequantized\(/);
  assert.match(inspector, /label="FP8 may run via dequant"/);
});
