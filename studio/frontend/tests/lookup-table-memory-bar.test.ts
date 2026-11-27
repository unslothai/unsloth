// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Lookup tables never sit in VRAM. Sizes are real GGUF header figures.

import assert from "node:assert/strict";
import test from "node:test";

import type { KvCacheEstimate } from "../src/features/chat/api/chat-api.ts";
import { readSrcAsync, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { memoryBarWeightsBytes } = await import("../src/lib/model-memory.ts");
const pickers = await readSrcAsync(
  "features/model-picker/components/model-selector/pickers.tsx",
);

const GIB = 1024 ** 3;
const FLASH = { sizeBytes: 73.45 * GIB, lookupTableBytes: 26.82 * GIB };

test("before the estimate, the listing's streamed table comes off the file size", () => {
  assert.equal(
    memoryBarWeightsBytes(undefined, FLASH),
    FLASH.sizeBytes - FLASH.lookupTableBytes,
  );
  assert.equal(memoryBarWeightsBytes(undefined, {}), undefined);
});

test("a streamed table the estimate already left out is not taken off twice", () => {
  const weightsBytes = FLASH.sizeBytes - FLASH.lookupTableBytes;
  const estimate = {
    weightsBytes,
    lookupTableBytes: FLASH.lookupTableBytes,
    lookupTableOnDisk: true,
  };
  assert.equal(memoryBarWeightsBytes(estimate, FLASH), weightsBytes);
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

test("the on-device card takes an update's table reading, a zero included", async () => {
  const card = await readSrcAsync("features/hub/catalog/local-on-device-card.tsx");
  assert.match(
    card,
    /lookup_table_bytes: remoteVariant\.update_available\s*\?\s*\(remoteVariant\.lookup_table_bytes \?\? 0\)\s*:\s*variant\.lookup_table_bytes \|\| remoteVariant\.lookup_table_bytes/,
  );
});

test("the memory bar's estimate takes the lookup table off the whole-file weights", async () => {
  // GET /kv-cache-estimate's weights_bytes is the quant file alone, streamed or resident table included.
  const hook = await readSrcAsync("hooks/use-model-memory.ts");
  assert.match(
    hook,
    /weightsBytes:\s*r\.weights_bytes == null\s*\?\s*null\s*:\s*Math\.max\(0, r\.weights_bytes - \(r\.lookup_table_bytes \?\? 0\)\)/,
  );
});

test("the kv-cache-estimate type carries the route's null for a model without a table", () => {
  const noTable: Pick<KvCacheEstimate, "lookup_table_bytes"> = { lookup_table_bytes: null };
  assert.equal(noTable.lookup_table_bytes ?? 0, 0);
});
