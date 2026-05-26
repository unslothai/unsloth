// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { dedupeSameSourceHubCacheRows } from "../src/features/inventory/inventory-dedupe.ts";
import type {
  CachedInventoryRow,
  LocalInventoryRow,
} from "../src/features/inventory/types.ts";

function cachedRow(
  id: string,
  repoId: string,
  options: {
    format?: CachedInventoryRow["modelFormat"];
    partial?: boolean;
  } = {},
): CachedInventoryRow {
  const modelFormat = options.format ?? "safetensors";
  return {
    kind: "cache",
    id,
    loadId: repoId,
    repoId,
    owner: "Org",
    repo: "Model",
    isGguf: modelFormat === "gguf",
    modelFormat,
    runtime: modelFormat === "gguf" ? "llama_cpp" : "transformers",
    capabilities: {
      canTrain: !options.partial && modelFormat !== "gguf",
      canChat: !options.partial,
      canDelete: true,
      canDownload: false,
      requiresVariant: modelFormat === "gguf",
      supportsLora: !options.partial && modelFormat !== "gguf",
      supportsVision: false,
    },
    bytes: 100,
    partial: options.partial ?? false,
  };
}

function localRow(
  id: string,
  repoId: string,
  options: {
    format?: LocalInventoryRow["modelFormat"];
    partial?: boolean;
    source?: LocalInventoryRow["source"];
  } = {},
): LocalInventoryRow {
  const modelFormat = options.format ?? "safetensors";
  return {
    kind: "local",
    id,
    loadId: id,
    repoId,
    owner: "Org",
    title: "Model",
    source: options.source ?? "hf_cache",
    sourceLabel: options.source ?? "hf_cache",
    path: id,
    isGguf: modelFormat === "gguf",
    modelFormat,
    runtime: modelFormat === "gguf" ? "llama_cpp" : "transformers",
    capabilities: {
      canTrain: !options.partial && modelFormat !== "gguf",
      canChat: !options.partial,
      canDelete: false,
      canDownload: false,
      requiresVariant: modelFormat === "gguf",
      supportsLora: !options.partial && modelFormat !== "gguf",
      supportsVision: false,
    },
    updatedAt: null,
    partial: options.partial ?? false,
  };
}

test("complete cached rows hide duplicate HF-cache local rows only", () => {
  const cached = cachedRow("cache:safe:Org/Model", "Org/Model");
  const hfLocal = localRow("/hf-cache/model", "Org/Model", {
    source: "hf_cache",
  });
  const customLocal = localRow("/custom/model", "Org/Model", {
    source: "custom",
  });

  const deduped = dedupeSameSourceHubCacheRows({
    cachedRows: [cached],
    localRows: [hfLocal, customLocal],
  });

  assert.deepEqual(
    deduped.cachedRows.map((row) => row.id),
    [cached.id],
  );
  assert.deepEqual(
    deduped.localRows.map((row) => row.id),
    [customLocal.id],
  );
});

test("complete cached rows hide legacy unknown HF-cache local duplicates", () => {
  const cached = cachedRow("cache:safe:Org/Model", "Org/Model");
  const hfLocalUnknown = localRow("/hf-cache/model", "Org/Model", {
    source: "hf_cache",
    format: "unknown",
  });
  const customUnknown = localRow("/custom/model", "Org/Model", {
    source: "custom",
    format: "unknown",
  });

  const deduped = dedupeSameSourceHubCacheRows({
    cachedRows: [cached],
    localRows: [hfLocalUnknown, customUnknown],
  });

  assert.deepEqual(
    deduped.cachedRows.map((row) => row.id),
    [cached.id],
  );
  assert.deepEqual(
    deduped.localRows.map((row) => row.id),
    [customUnknown.id],
  );
});

test("complete HF-cache local rows hide matching partial cached rows", () => {
  const partialCached = cachedRow("cache:safe:Org/Model", "Org/Model", {
    partial: true,
  });
  const hfLocal = localRow("/hf-cache/model", "Org/Model", {
    source: "hf_cache",
  });

  const deduped = dedupeSameSourceHubCacheRows({
    cachedRows: [partialCached],
    localRows: [hfLocal],
  });

  assert.deepEqual(deduped.cachedRows, []);
  assert.deepEqual(
    deduped.localRows.map((row) => row.id),
    [hfLocal.id],
  );
});

test("same repo with different source or format is not deduped", () => {
  const partialGguf = cachedRow("cache:gguf:Org/Model", "Org/Model", {
    format: "gguf",
    partial: true,
  });
  const hfSafe = localRow("/hf-cache/safe", "Org/Model", {
    format: "safetensors",
    source: "hf_cache",
  });
  const customSafe = localRow("/custom/safe", "Org/Model", {
    source: "custom",
  });

  const deduped = dedupeSameSourceHubCacheRows({
    cachedRows: [partialGguf],
    localRows: [hfSafe, customSafe],
  });

  assert.deepEqual(
    deduped.cachedRows.map((row) => row.id),
    [partialGguf.id],
  );
  assert.deepEqual(
    deduped.localRows.map((row) => row.id),
    [hfSafe.id, customSafe.id],
  );
});
