// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const { dedupeSameSourceHubCacheRows } = await import(
  "../src/features/hub/inventory/inventory-dedupe.ts"
);
const { buildCachedInventoryRow, buildLocalInventoryRows, cachedInventoryId } =
  await import("../src/features/hub/inventory/view-models.ts");
const { resolveDownloadedSelection } = await import(
  "../src/features/hub/lib/selection-resolution.ts"
);

const REPO_ID = "Org/Model";

for (const modelFormat of ["gguf", "safetensors"] as const) {
  const transition =
    modelFormat === "gguf"
      ? "across resume, cancel, and sibling partial deletion transitions"
      : "across download transitions";

  test(`keeps a selected ${modelFormat} row stable ${transition}`, () => {
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
      {
        input: [observed],
        selectionId: `cache:${modelFormat}:${REPO_ID}`,
      },
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

function resolveInventorySelection(
  inventory: ReturnType<typeof dedupeSameSourceHubCacheRows>,
  selectedId: string,
) {
  return resolveDownloadedSelection({
    selectedId,
    cachedRows: inventory.cachedRows,
    localRows: inventory.localRows,
    filteredCachedRows: inventory.cachedRows,
    filteredLocalRows: inventory.localRows,
  });
}

for (const modelFormat of ["gguf", "safetensors"] as const) {
  test(`keeps a selected local HF-cache ${modelFormat} partial stable across resume and cancel`, () => {
    const repoId =
      modelFormat === "safetensors"
        ? "unsloth/gemma-3-270m-it"
        : "unsloth/gemma-3-270m-it-GGUF";
    const localId = `hf_cache:${modelFormat}:${encodeURIComponent(repoId)}`;
    const liveId = cachedInventoryId(modelFormat, repoId);
    const local = buildLocalInventoryRows([
      {
        id: repoId,
        inventory_id: localId,
        load_id: repoId,
        display_name: repoId.split("/").at(-1) ?? repoId,
        path: `/cache/models--${repoId.replace("/", "--")}`,
        source: "hf_cache",
        model_id: repoId,
        model_format: modelFormat,
        partial: true,
        partial_resumable: true,
      },
    ])[0];
    const live = {
      ...buildCachedInventoryRow(
        {
          repo_id: repoId,
          inventory_id: liveId,
          load_id: repoId,
          model_format: modelFormat,
          size_bytes: 50,
          partial: true,
          optimistic: true,
        },
        modelFormat,
      ),
      liveDownload: true,
    };

    assert.ok(local);

    const atRest = dedupeSameSourceHubCacheRows({
      cachedRows: [],
      localRows: [local],
    });
    assert.deepEqual(resolveInventorySelection(atRest, localId), {
      selectedId: localId,
      hiddenByFilters: false,
    });

    const resumed = dedupeSameSourceHubCacheRows({
      cachedRows: [live],
      localRows: [local],
    });
    assert.deepEqual(resumed.localRows, []);
    assert.deepEqual(resolveInventorySelection(resumed, localId), {
      selectedId: liveId,
      hiddenByFilters: false,
    });

    const cancelled = dedupeSameSourceHubCacheRows({
      cachedRows: [],
      localRows: [local],
    });
    assert.deepEqual(resolveInventorySelection(cancelled, liveId), {
      selectedId: localId,
      hiddenByFilters: false,
    });
  });
}

test("does not carry HF-cache selection across model formats", () => {
  const repoId = "Org/Hybrid";
  const localId = `hf_cache:safetensors:${encodeURIComponent(repoId)}`;
  const local = buildLocalInventoryRows([
    {
      id: repoId,
      inventory_id: localId,
      display_name: "Hybrid",
      path: "/cache/models--Org--Hybrid",
      source: "hf_cache",
      model_id: repoId,
      model_format: "safetensors",
      partial: true,
    },
  ])[0];
  const cached = buildCachedInventoryRow(
    {
      repo_id: repoId,
      model_format: "gguf",
      size_bytes: 50,
      partial: true,
    },
    "gguf",
  );

  assert.ok(local);
  assert.deepEqual(
    resolveDownloadedSelection({
      selectedId: localId,
      cachedRows: [cached],
      localRows: [],
      filteredCachedRows: [cached],
      filteredLocalRows: [],
    }),
    { selectedId: null, hiddenByFilters: false },
  );
  assert.deepEqual(
    resolveDownloadedSelection({
      selectedId: cached.id,
      cachedRows: [],
      localRows: [local],
      filteredCachedRows: [],
      filteredLocalRows: [local],
    }),
    { selectedId: null, hiddenByFilters: false },
  );
});
