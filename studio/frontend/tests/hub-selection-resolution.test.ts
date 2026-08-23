// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const { dedupeSameSourceHubCacheRows } = await import(
  "../src/features/hub/inventory/inventory-dedupe.ts"
);
const { buildCachedInventoryRow, cachedInventoryId } = await import(
  "../src/features/hub/inventory/view-models.ts"
);
const { resolveDownloadedSelection } = await import(
  "../src/features/hub/lib/selection-resolution.ts"
);

const REPO_ID = "Org/Model";

for (const modelFormat of ["gguf", "safetensors"] as const) {
  test(`keeps a selected ${modelFormat} row stable while a download starts and cancels`, () => {
    const selectedId = cachedInventoryId(modelFormat, REPO_ID);
    const observed = buildCachedInventoryRow(
      {
        repo_id: REPO_ID,
        inventory_id: selectedId,
        model_format: modelFormat,
        size_bytes: 100,
        partial: true,
      },
      modelFormat,
    );
    const live = {
      ...buildCachedInventoryRow(
        {
          repo_id: REPO_ID,
          model_format: modelFormat,
          size_bytes: 50,
          partial: true,
          optimistic: true,
        },
        modelFormat,
      ),
      liveDownload: true,
    };

    assert.equal(selectedId, `cache:${modelFormat}:Org%2FModel`);
    const transitions = [
      { input: [observed], selectionId: selectedId },
      { input: [observed, live], selectionId: selectedId },
      { input: [observed], selectionId: `cache:${modelFormat}:${REPO_ID}` },
    ];
    for (const { input, selectionId } of transitions) {
      const cachedRows = dedupeSameSourceHubCacheRows({
        cachedRows: input,
        localRows: [],
      }).cachedRows;
      assert.equal(cachedRows[0]?.id, selectedId);
      assert.deepEqual(
        resolveDownloadedSelection({
          selectedId: selectionId,
          cachedRows,
          localRows: [],
          filteredCachedRows: cachedRows,
          filteredLocalRows: [],
        }),
        { selectedId, hiddenByFilters: false },
      );
    }
  });
}
