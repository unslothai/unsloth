// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();

const { dedupeSameSourceHubCacheRows } = await import(
  "../src/features/hub/inventory/inventory-dedupe.ts"
);
const {
  buildCachedInventoryRow,
  buildLocalInventoryRows,
  cachedInventoryId,
  optimisticInventoryId,
} = await import("../src/features/hub/inventory/view-models.ts");
const { resolveDownloadedSelection, resolveSelectionUrlSync } = await import(
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
      { input: [observed], selectionId: selectedId, expectedId: selectedId },
      {
        input: [observed, live],
        selectionId: selectedId,
        expectedId: live.id,
      },
      {
        input: [observed],
        selectionId: `cache:${modelFormat}:${REPO_ID}`,
        expectedId: selectedId,
      },
    ];

    for (const { input, selectionId, expectedId } of transitions) {
      const cachedRows = dedupeSameSourceHubCacheRows({
        cachedRows: input,
        localRows: [],
      }).cachedRows;

      assert.equal(cachedRows.length, 1);
      assert.equal(cachedRows[0]?.id, expectedId);
      assert.equal(cachedRows[0]?.liveDownload, input.includes(live) || undefined);
      assert.deepEqual(
        resolveDownloadedSelection({
          selectedId: selectionId,
          cachedRows,
          localRows: [],
          filteredCachedRows: cachedRows,
          filteredLocalRows: [],
        }),
        { selectedId: expectedId, hiddenByFilters: false },
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

function buildUnknownHfCacheRow(
  repoId: string,
  partialTransport: string | null = null,
) {
  return buildLocalInventoryRows([
    {
      id: repoId,
      inventory_id: `hf_cache:unknown:${encodeURIComponent(repoId)}`,
      load_id: repoId,
      display_name: repoId.split("/").at(-1) ?? repoId,
      path: `/cache/models--${repoId.replace("/", "--")}`,
      source: "hf_cache",
      model_id: repoId,
      model_format: "unknown",
      partial: true,
      partial_transport: partialTransport,
      partial_resumable: partialTransport !== null,
    },
  ])[0];
}

for (const modelFormat of ["gguf", "safetensors"] as const) {
  test(`keeps a selected local HF-cache ${modelFormat} partial stable across resume and cancel`, () => {
    const repoId =
      modelFormat === "safetensors"
        ? "unsloth/gemma-3-270m-it"
        : "unsloth/gemma-3-270m-it-GGUF";
    const localId = `hf_cache:${modelFormat}:${encodeURIComponent(repoId)}`;
    const liveId = optimisticInventoryId(modelFormat, repoId);
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

test("keeps an unclassified local HF-cache partial selected across GGUF resume and cancel", () => {
  const repoId = "Org/Partial-GGUF";
  const local = buildUnknownHfCacheRow(repoId);
  const live = {
    ...buildCachedInventoryRow(
      {
        repo_id: repoId,
        model_format: "gguf",
        size_bytes: 50,
        partial: true,
        optimistic: true,
      },
      "gguf",
    ),
    liveDownload: true,
  };

  assert.ok(local);

  const resumed = dedupeSameSourceHubCacheRows({
    cachedRows: [live],
    localRows: [local],
  });
  assert.deepEqual(resumed.localRows, []);
  for (const selectedId of [
    local.id,
    `hf_cache:unknown:${repoId}`,
    "hf_cache:unknown:org%2fpartial-gguf",
  ]) {
    assert.deepEqual(resolveInventorySelection(resumed, selectedId), {
      selectedId: live.id,
      hiddenByFilters: false,
    });
  }

  const cancelled = dedupeSameSourceHubCacheRows({
    cachedRows: [],
    localRows: [local],
  });
  for (const selectedId of [
    live.id,
    `cache:gguf:${repoId}`,
    "cache:gguf:org/partial-gguf",
  ]) {
    assert.deepEqual(resolveInventorySelection(cancelled, selectedId), {
      selectedId: local.id,
      hiddenByFilters: false,
    });
  }
});

test("keeps an unclassified local HF-cache partial selected across safetensors resume and cancel", () => {
  const repoId = "Org/Partial-Model";
  const local = buildUnknownHfCacheRow(repoId, "http");
  const live = {
    ...buildCachedInventoryRow(
      {
        repo_id: repoId,
        model_format: "safetensors",
        size_bytes: 50,
        partial: true,
        optimistic: true,
      },
      "safetensors",
    ),
    liveDownload: true,
  };

  assert.ok(local);
  const resumed = dedupeSameSourceHubCacheRows({
    cachedRows: [live],
    localRows: [local],
  });
  assert.deepEqual(resumed.localRows, []);
  assert.deepEqual(resolveInventorySelection(resumed, local.id), {
    selectedId: live.id,
    hiddenByFilters: false,
  });

  const cancelled = dedupeSameSourceHubCacheRows({
    cachedRows: [],
    localRows: [local],
  });
  assert.deepEqual(resolveInventorySelection(cancelled, live.id), {
    selectedId: local.id,
    hiddenByFilters: false,
  });
});

for (const { partialTransport, unrelatedFormat } of [
  { partialTransport: null, unrelatedFormat: "safetensors" as const },
  { partialTransport: "http", unrelatedFormat: "gguf" as const },
]) {
  test(`keeps an unclassified partial beside an unrelated ${unrelatedFormat} job`, () => {
    const repoId = "Org/Hybrid-Partial";
    const local = buildUnknownHfCacheRow(repoId, partialTransport);
    const live = {
      ...buildCachedInventoryRow(
        {
          repo_id: repoId,
          model_format: unrelatedFormat,
          size_bytes: 50,
          partial: true,
          optimistic: true,
        },
        unrelatedFormat,
      ),
      liveDownload: true,
    };

    assert.ok(local);
    const inventory = dedupeSameSourceHubCacheRows({
      cachedRows: [live],
      localRows: [local],
    });
    assert.deepEqual(inventory.cachedRows, [live]);
    assert.deepEqual(inventory.localRows, [local]);
    assert.deepEqual(resolveInventorySelection(inventory, local.id), {
      selectedId: local.id,
      hiddenByFilters: false,
    });
  });
}

test("keeps an unclassified local HF-cache partial selected when safetensors completes", () => {
  const repoId = "Org/Partial-Model";
  const local = buildUnknownHfCacheRow(repoId, "http");
  const complete = buildCachedInventoryRow(
    {
      repo_id: repoId,
      model_format: "safetensors",
      size_bytes: 100,
      partial: false,
    },
    "safetensors",
  );

  assert.ok(local);

  const completed = dedupeSameSourceHubCacheRows({
    cachedRows: [complete],
    localRows: [local],
  });
  assert.deepEqual(completed.localRows, []);
  assert.deepEqual(resolveInventorySelection(completed, local.id), {
    selectedId: complete.id,
    hiddenByFilters: false,
  });
  assert.deepEqual(
    resolveDownloadedSelection({
      selectedId: local.id,
      cachedRows: completed.cachedRows,
      localRows: completed.localRows,
      filteredCachedRows: [],
      filteredLocalRows: [],
    }),
    { selectedId: complete.id, hiddenByFilters: true },
  );

  const reverted = dedupeSameSourceHubCacheRows({
    cachedRows: [],
    localRows: [local],
  });
  assert.deepEqual(resolveInventorySelection(reverted, complete.id), {
    selectedId: local.id,
    hiddenByFilters: false,
  });
});

test("keeps an unclassified GGUF partial when an unrelated safetensors job completes", () => {
  const repoId = "Org/Hybrid-Complete";
  const local = buildUnknownHfCacheRow(repoId);
  const complete = buildCachedInventoryRow(
    {
      repo_id: repoId,
      model_format: "safetensors",
      size_bytes: 100,
      partial: false,
    },
    "safetensors",
  );

  assert.ok(local);
  const inventory = dedupeSameSourceHubCacheRows({
    cachedRows: [complete],
    localRows: [local],
  });
  assert.deepEqual(inventory.cachedRows, [complete]);
  assert.deepEqual(inventory.localRows, [local]);
  assert.deepEqual(resolveInventorySelection(inventory, local.id), {
    selectedId: local.id,
    hiddenByFilters: false,
  });
});

test("waits for local inventory before resolving an unknown selection", () => {
  const repoId = "Org/Refresh-Race";
  const selectedId = `hf_cache:unknown:${encodeURIComponent(repoId)}`;
  const live = {
    ...buildCachedInventoryRow(
      {
        repo_id: repoId,
        model_format: "safetensors",
        size_bytes: 50,
        partial: true,
        optimistic: true,
      },
      "safetensors",
    ),
    liveDownload: true,
  };

  assert.deepEqual(
    resolveDownloadedSelection({
      selectedId,
      inventoryReady: false,
      cachedRows: [live],
      localRows: [],
      filteredCachedRows: [live],
      filteredLocalRows: [],
    }),
    { selectedId, hiddenByFilters: false },
  );

  const local = buildUnknownHfCacheRow(repoId);
  assert.ok(local);
  const settled = dedupeSameSourceHubCacheRows({
    cachedRows: [live],
    localRows: [local],
  });
  assert.deepEqual(
    resolveDownloadedSelection({
      selectedId,
      inventoryReady: true,
      cachedRows: settled.cachedRows,
      localRows: settled.localRows,
      filteredCachedRows: settled.cachedRows,
      filteredLocalRows: settled.localRows,
    }),
    { selectedId: local.id, hiddenByFilters: false },
  );
});

for (const modelFormat of ["adapter", "checkpoint"] as const) {
  test(`keeps a provisional download selected when it becomes ${modelFormat}`, () => {
    const repoId = `Org/${modelFormat}`;
    const live = buildCachedInventoryRow(
      {
        repo_id: repoId,
        model_format: "safetensors",
        size_bytes: 50,
        partial: true,
        optimistic: true,
      },
      "safetensors",
    );
    const complete = buildCachedInventoryRow(
      {
        repo_id: repoId,
        model_format: modelFormat,
        size_bytes: 100,
      },
      "safetensors",
    );

    assert.equal(live.id, optimisticInventoryId("safetensors", repoId));
    assert.deepEqual(
      resolveDownloadedSelection({
        selectedId: live.id,
        cachedRows: [complete],
        localRows: [],
        filteredCachedRows: [complete],
        filteredLocalRows: [],
      }),
      { selectedId: complete.id, hiddenByFilters: false },
    );
  });
}

test("keeps an HF-cache selection when a local row becomes classified", () => {
  const repoId = "Org/Locally-Classified";
  const unknown = buildUnknownHfCacheRow(repoId);
  const known = buildLocalInventoryRows([
    {
      id: repoId,
      inventory_id: `hf_cache:safetensors:${encodeURIComponent(repoId)}`,
      load_id: repoId,
      display_name: "Locally-Classified",
      path: "/cache/models--Org--Locally-Classified",
      source: "hf_cache",
      model_id: repoId,
      model_format: "safetensors",
      partial: true,
    },
  ])[0];

  assert.ok(unknown);
  assert.ok(known);

  assert.deepEqual(
    resolveDownloadedSelection({
      selectedId: unknown.id,
      cachedRows: [],
      localRows: [known],
      filteredCachedRows: [],
      filteredLocalRows: [known],
    }),
    { selectedId: known.id, hiddenByFilters: false },
  );
  assert.deepEqual(
    resolveDownloadedSelection({
      selectedId: known.id,
      cachedRows: [],
      localRows: [unknown],
      filteredCachedRows: [],
      filteredLocalRows: [unknown],
    }),
    { selectedId: unknown.id, hiddenByFilters: false },
  );
});

test("does not resolve an unclassified HF-cache selection across ambiguous formats", () => {
  const repoId = "Org/Hybrid-Unknown";
  const local = buildUnknownHfCacheRow(repoId);
  const gguf = buildCachedInventoryRow(
    {
      repo_id: repoId,
      model_format: "gguf",
      size_bytes: 50,
      partial: true,
    },
    "gguf",
  );
  const safetensors = buildCachedInventoryRow(
    {
      repo_id: repoId,
      model_format: "safetensors",
      size_bytes: 50,
      partial: true,
    },
    "safetensors",
  );

  assert.ok(local);

  const ambiguousKnown = dedupeSameSourceHubCacheRows({
    cachedRows: [gguf, safetensors],
    localRows: [local],
  });
  assert.deepEqual(resolveInventorySelection(ambiguousKnown, local.id), {
    selectedId: null,
    hiddenByFilters: false,
  });

  const ambiguousUnknown = dedupeSameSourceHubCacheRows({
    cachedRows: [safetensors],
    localRows: [local],
  });
  assert.deepEqual(resolveInventorySelection(ambiguousUnknown, gguf.id), {
    selectedId: null,
    hiddenByFilters: false,
  });
});

test("ignores malformed unclassified HF-cache selection IDs", () => {
  const cached = buildCachedInventoryRow(
    {
      repo_id: "Org/Model",
      model_format: "gguf",
      size_bytes: 50,
      partial: true,
    },
    "gguf",
  );

  assert.deepEqual(
    resolveDownloadedSelection({
      selectedId: "hf_cache:unknown:Org%2",
      cachedRows: [cached],
      localRows: [],
      filteredCachedRows: [cached],
      filteredLocalRows: [],
    }),
    { selectedId: null, hiddenByFilters: false },
  );
});

test("keeps a raw legacy HF-cache ID attached after deduplication", () => {
  const repoId = "Org/Legacy-Model";
  const local = buildLocalInventoryRows([
    {
      id: repoId,
      load_id: repoId,
      display_name: "Legacy-Model",
      path: "/cache/models--Org--Legacy-Model",
      source: "hf_cache",
      model_id: repoId,
      model_format: "safetensors",
      partial: true,
    },
  ])[0];
  const live = {
    ...buildCachedInventoryRow(
      {
        repo_id: repoId,
        model_format: "safetensors",
        size_bytes: 50,
        partial: true,
        optimistic: true,
      },
      "safetensors",
    ),
    liveDownload: true,
  };

  assert.ok(local);
  assert.equal(local.id, repoId);
  const resumed = dedupeSameSourceHubCacheRows({
    cachedRows: [live],
    localRows: [local],
  });
  assert.deepEqual(resumed.localRows, []);
  assert.deepEqual(resolveInventorySelection(resumed, local.id), {
    selectedId: live.id,
    hiddenByFilters: false,
  });
});

test("does not bridge unclassified identities across cache source kinds", () => {
  const repoId = "Org/Shared-Identity";
  const local = buildUnknownHfCacheRow(repoId);
  const cached = buildCachedInventoryRow(
    {
      repo_id: repoId,
      model_format: "unknown",
      size_bytes: 50,
      partial: true,
    },
    "unknown",
  );
  const known = buildCachedInventoryRow(
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
      selectedId: local.id,
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
  assert.deepEqual(
    resolveDownloadedSelection({
      selectedId: cached.id,
      cachedRows: [known],
      localRows: [],
      filteredCachedRows: [known],
      filteredLocalRows: [],
    }),
    { selectedId: null, hiddenByFilters: false },
  );
  assert.deepEqual(
    resolveDownloadedSelection({
      selectedId: known.id,
      cachedRows: [cached],
      localRows: [],
      filteredCachedRows: [cached],
      filteredLocalRows: [],
    }),
    { selectedId: null, hiddenByFilters: false },
  );
});

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

test("commits a resolved identity before another format appears", () => {
  const repoId = "Org/Hybrid-After-Completion";
  const unknownId = `hf_cache:unknown:${encodeURIComponent(repoId)}`;
  const safetensors = buildCachedInventoryRow(
    {
      repo_id: repoId,
      model_format: "safetensors",
      size_bytes: 100,
    },
    "safetensors",
  );
  const gguf = buildCachedInventoryRow(
    {
      repo_id: repoId,
      model_format: "gguf",
      size_bytes: 100,
    },
    "gguf",
  );

  const firstResolution = resolveDownloadedSelection({
    selectedId: unknownId,
    cachedRows: [safetensors],
    localRows: [],
    filteredCachedRows: [safetensors],
    filteredLocalRows: [],
  });
  assert.equal(firstResolution.selectedId, safetensors.id);
  assert.deepEqual(
    resolveSelectionUrlSync({
      isDiscoverTab: false,
      urlModel: unknownId,
      selectionInputId: unknownId,
      resolvedSelectedId: firstResolution.selectedId,
      resolvedModelFormat: safetensors.modelFormat,
    }),
    {
      action: "replace",
      selectedId: safetensors.id,
      preserveGgufFile: false,
    },
  );

  assert.deepEqual(
    resolveDownloadedSelection({
      selectedId: safetensors.id,
      cachedRows: [safetensors, gguf],
      localRows: [],
      filteredCachedRows: [safetensors, gguf],
      filteredLocalRows: [],
    }),
    { selectedId: safetensors.id, hiddenByFilters: false },
  );
});

test("preserves a GGUF file while canonicalizing its selection ID", () => {
  const repoId = "Org/GGUF-Model";

  assert.deepEqual(
    resolveSelectionUrlSync({
      isDiscoverTab: false,
      urlModel: optimisticInventoryId("gguf", repoId),
      selectionInputId: optimisticInventoryId("gguf", repoId),
      resolvedSelectedId: cachedInventoryId("gguf", repoId),
      resolvedModelFormat: "gguf",
    }),
    {
      action: "replace",
      selectedId: cachedInventoryId("gguf", repoId),
      preserveGgufFile: true,
    },
  );

  assert.deepEqual(
    resolveSelectionUrlSync({
      isDiscoverTab: false,
      urlModel: optimisticInventoryId("gguf", repoId),
      selectionInputId: optimisticInventoryId("gguf", repoId),
      resolvedSelectedId: repoId,
      resolvedModelFormat: "unknown",
    }),
    {
      action: "replace",
      selectedId: repoId,
      preserveGgufFile: true,
    },
  );
});

test("does not move a provisional selection to an incompatible format", () => {
  const repoId = "Org/Hybrid-Provisional";
  const gguf = buildCachedInventoryRow(
    {
      repo_id: repoId,
      model_format: "gguf",
      size_bytes: 100,
    },
    "gguf",
  );
  const safetensors = buildCachedInventoryRow(
    {
      repo_id: repoId,
      model_format: "safetensors",
      size_bytes: 100,
    },
    "safetensors",
  );

  assert.deepEqual(
    resolveDownloadedSelection({
      selectedId: optimisticInventoryId("gguf", repoId),
      cachedRows: [safetensors],
      localRows: [],
      filteredCachedRows: [safetensors],
      filteredLocalRows: [],
    }),
    { selectedId: null, hiddenByFilters: false },
  );
  assert.deepEqual(
    resolveDownloadedSelection({
      selectedId: optimisticInventoryId("safetensors", repoId),
      cachedRows: [gguf],
      localRows: [],
      filteredCachedRows: [gguf],
      filteredLocalRows: [],
    }),
    { selectedId: null, hiddenByFilters: false },
  );
});

test("applies URL navigation before rewriting a resolved selection", () => {
  assert.deepEqual(
    resolveSelectionUrlSync({
      isDiscoverTab: false,
      urlModel: "cache:gguf:Org%2FNext",
      selectionInputId: "cache:gguf:Org%2FPrevious",
      resolvedSelectedId: "cache:gguf:Org%2FPrevious",
      resolvedModelFormat: "gguf",
    }),
    { action: "select", selectedId: "cache:gguf:Org%2FNext" },
  );
});

test("adopts a raw selection after its URL navigation lands", () => {
  assert.deepEqual(
    resolveSelectionUrlSync({
      isDiscoverTab: false,
      urlModel: "Org/Legacy-Model",
      selectionInputId: null,
      resolvedSelectedId: null,
      resolvedModelFormat: null,
    }),
    { action: "select", selectedId: "Org/Legacy-Model" },
  );
});
