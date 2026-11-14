// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { registerBundlerResolver } from "./helpers/kit.ts";
import { isLocalCacheSnapshot } from "../src/features/model-picker/inventory/local-cache-snapshot.ts";

registerBundlerResolver();
const { buildCachedInventoryRow, buildLocalInventoryRows } = await import(
  "../src/features/hub/inventory/view-models.ts"
);
const { dedupeSameSourceHubCacheRows } = await import(
  "../src/features/hub/inventory/inventory-dedupe.ts"
);

const repoId = "example/Model";
const path = String.raw`D:\models\hub\models--example--Model\snapshots\abc`;
const localRows = buildLocalInventoryRows(
  (["safetensors", "gguf"] as const).map((format) => ({
    id: path, load_id: path, path, display_name: "Model", model_id: repoId,
    inventory_id: `local:${format}:${repoId}`, model_format: format,
    source: "hf_cache", active_cache: false,
  })),
);

test("offers the local snapshot only for formats the cached listing did not supply", () => {
  const cached = buildCachedInventoryRow({repo_id: repoId, size_bytes: 1}, "gguf");
  const inventory = dedupeSameSourceHubCacheRows({cachedRows: [cached], localRows});
  const offered = inventory.localRows.filter(isLocalCacheSnapshot);
  assert.deepEqual(offered.map(row => [row.modelFormat, row.loadId]), [["safetensors", path]]);
  assert.equal(localRows.filter(isLocalCacheSnapshot).length, 2);
});

test("only complete, loadable snapshots pinned to another cache qualify", () => {
  const [row] = localRows;
  for (const override of [
    {activeCache: true}, {loadId: repoId}, {partial: true}, {companionPrefetch: true},
    {capabilities: {...row.capabilities, canChat: false}},
  ]) {
    assert.equal(isLocalCacheSnapshot({...row, ...override}), false);
  }
});
