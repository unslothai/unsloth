// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Lookup tables (per-layer embeddings / Engram) never sit in VRAM, so the bar's weights segment
// leaves them out. Sizes are real GGUF header figures: Qwen3.8-Flash-Next UD-Q2_K_XL (73.45 GiB,
// streamed PLE 26.82 GiB) and gemma-3n-E4B Q4_K_M (4.23 GiB, host-resident PLE 1.64 GiB).

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { memoryBarWeightsBytes } = await import("../src/lib/model-memory.ts");
const pickers = await readSrcAsync(
  "features/model-picker/components/model-selector/pickers.tsx",
);

const GIB = 1024 ** 3;

test("before the estimate, the listing's streamed table comes off the file size", () => {
  assert.equal(
    memoryBarWeightsBytes(undefined, {
      sizeBytes: 73.45 * GIB,
      lookupTableBytes: 26.82 * GIB,
    }),
    73.45 * GIB - 26.82 * GIB,
  );
  assert.equal(memoryBarWeightsBytes(undefined, {}), undefined);
});

test("a streamed table the estimate already left out is not taken off twice", () => {
  const weightsBytes = 73.45 * GIB - 26.82 * GIB;
  assert.equal(
    memoryBarWeightsBytes(
      { weightsBytes, lookupTableBytes: 26.82 * GIB, lookupTableOnDisk: true },
      { sizeBytes: 73.45 * GIB, lookupTableBytes: 26.82 * GIB },
    ),
    weightsBytes,
  );
});

test("a host-resident table inside the estimate's weights is not charted as VRAM", () => {
  assert.equal(
    memoryBarWeightsBytes(
      { weightsBytes: 4.23 * GIB, lookupTableBytes: 1.64 * GIB, lookupTableOnDisk: false },
      { sizeBytes: 4.23 * GIB, lookupTableBytes: 0 },
    ),
    4.23 * GIB - 1.64 * GIB,
  );
});

test("a pinned quant row hands the bar the same listing figures as the sole-quant row", () => {
  const start = pickers.indexOf("const renderPinnedQuantRow");
  assert.ok(start > 0);
  const body = pickers.slice(start, pickers.indexOf("gpuGb={expanderGpuGb}", start));
  assert.match(body, /repoId: entry\.repoId,\s*quant: entry\.quant,[\s\S]*sizeBytes,/);
  assert.match(body, /lookupTableBytes: pinnedQuantValidation\.lookupTables\.get\(/);
  assert.match(pickers, /lookupTables: downloaded\.map\([\s\S]*?ggufVariantLookupTableBytes\(variant\)/);
});
