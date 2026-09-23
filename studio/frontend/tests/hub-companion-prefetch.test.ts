// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const { buildCachedInventoryRow, buildLocalInventoryRows } =
  await import("../src/features/hub/inventory/view-models.ts");
const { buildDiscoverRows } =
  await import("../src/features/hub/lib/view-models.ts");

const REPO_ID = "Qwen/Qwen-Image-2.1";
const RESULT = { id: REPO_ID, downloads: 1, likes: 1, isGguf: false };

function localRow(companionPrefetch: boolean) {
  return buildLocalInventoryRows([
    {
      id: REPO_ID,
      display_name: "Qwen-Image-2.1",
      path: "/cache/models--Qwen--Qwen-Image-2.1/snapshots/abc",
      source: "hf_cache",
      model_id: REPO_ID,
      model_format: "unknown",
      partial: true,
      companion_prefetch: companionPrefetch,
    },
  ]);
}

test("a companion-only base repo is neither on device nor partial in model Discover", () => {
  // The cached and the local listing both describe it; either one carrying the flag is enough.
  const fromCached = buildDiscoverRows(
    [RESULT],
    [
      buildCachedInventoryRow(
        {
          repo_id: REPO_ID,
          model_format: "safetensors",
          size_bytes: 1_350_989_512,
          partial: true,
          companion_prefetch: true,
        },
        "safetensors",
      ),
    ],
    [],
  );
  const fromLocal = buildDiscoverRows([RESULT], [], localRow(true));
  for (const [row] of [fromCached, fromLocal]) {
    assert.equal(row.isAvailableOnDevice, false);
    assert.equal(row.isPartialOnDevice, false);
  }
});

test("an interrupted download of the same repo still reads as partial", () => {
  const [row] = buildDiscoverRows([RESULT], [], localRow(false));
  assert.equal(row.isAvailableOnDevice, true);
  assert.equal(row.isPartialOnDevice, true);
});

test("the local listing's flag reaches the local inventory row", () => {
  assert.equal(localRow(true)[0].companionPrefetch, true);
  assert.equal(localRow(false)[0].companionPrefetch, false);
});
