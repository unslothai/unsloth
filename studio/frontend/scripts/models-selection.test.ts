// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  resolveDiscoverSelection,
  resolveDownloadedSelection,
} from "../src/features/models/lib/selection-resolution.ts";
import type {
  CachedInventoryRow,
  DiscoverRow,
  LocalInventoryRow,
} from "../src/features/models/types.ts";

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
    owner: repoId.split("/")[0] ?? "Org",
    repo: repoId.split("/").slice(1).join("/") || repoId,
    isGguf: modelFormat === "gguf",
    modelFormat,
    runtime: modelFormat === "gguf" ? "llama_cpp" : "transformers",
    capabilities: {
      canTrain: modelFormat !== "gguf" && !options.partial,
      canChat: !options.partial,
      canDelete: true,
      canDownload: false,
      requiresVariant: modelFormat === "gguf",
      supportsLora: modelFormat !== "gguf" && !options.partial,
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
    owner: repoId.split("/")[0] ?? "Org",
    title: repoId.split("/").slice(1).join("/") || repoId,
    source: options.source ?? "hf_cache",
    sourceLabel: options.source ?? "hf_cache",
    path: id,
    isGguf: modelFormat === "gguf",
    modelFormat,
    runtime: modelFormat === "gguf" ? "llama_cpp" : "transformers",
    capabilities: {
      canTrain: modelFormat !== "gguf" && !options.partial,
      canChat: !options.partial,
      canDelete: false,
      canDownload: false,
      requiresVariant: modelFormat === "gguf",
      supportsLora: modelFormat !== "gguf" && !options.partial,
      supportsVision: false,
    },
    updatedAt: null,
    partial: options.partial ?? false,
  };
}

function discoverRow(id: string): DiscoverRow {
  return {
    id,
    owner: id.split("/")[0] ?? "Org",
    repo: id.split("/").slice(1).join("/") || id,
    result: { id, downloads: 0, likes: 0, isGguf: false, tags: [] },
    isAvailableOnDevice: false,
    isPartialOnDevice: false,
    summary: "",
    capabilities: [],
  } as DiscoverRow;
}

test("downloaded selection stays selected when hidden by filters", () => {
  const selected = cachedRow("cache:safe:Org/Model", "Org/Model");
  const visible = cachedRow("cache:gguf:Org/Other", "Org/Other", {
    format: "gguf",
  });

  const resolved = resolveDownloadedSelection({
    selectedId: selected.id,
    cachedRows: [selected, visible],
    localRows: [],
    filteredCachedRows: [visible],
    filteredLocalRows: [],
  });

  assert.deepEqual(resolved, {
    selectedId: selected.id,
    hiddenByFilters: true,
  });
});

test("downloaded selection falls back only when no explicit row exists", () => {
  const visible = cachedRow("cache:safe:Org/Visible", "Org/Visible");

  assert.deepEqual(
    resolveDownloadedSelection({
      selectedId: null,
      cachedRows: [visible],
      localRows: [],
      filteredCachedRows: [visible],
      filteredLocalRows: [],
    }),
    { selectedId: visible.id, hiddenByFilters: false },
  );

  assert.deepEqual(
    resolveDownloadedSelection({
      selectedId: "deleted",
      cachedRows: [visible],
      localRows: [],
      filteredCachedRows: [visible],
      filteredLocalRows: [],
    }),
    { selectedId: visible.id, hiddenByFilters: false },
  );
});

test("partial cached selection redirects only to complete HF cache local row", () => {
  const partial = cachedRow("cache:safe:Org/Model", "Org/Model", {
    partial: true,
  });
  const custom = localRow("/custom/model", "Org/Model", { source: "custom" });
  const hfCache = localRow("/hf-cache/model", "Org/Model", {
    source: "hf_cache",
  });

  assert.deepEqual(
    resolveDownloadedSelection({
      selectedId: partial.id,
      cachedRows: [partial],
      localRows: [custom],
      filteredCachedRows: [partial],
      filteredLocalRows: [custom],
    }),
    { selectedId: partial.id, hiddenByFilters: false },
  );

  assert.deepEqual(
    resolveDownloadedSelection({
      selectedId: partial.id,
      cachedRows: [partial],
      localRows: [custom, hfCache],
      filteredCachedRows: [partial],
      filteredLocalRows: [custom, hfCache],
    }),
    { selectedId: hfCache.id, hiddenByFilters: false },
  );
});

test("discover selection can stay on a snapshotted row hidden by filters", () => {
  const selected = discoverRow("Org/Selected");
  const visible = discoverRow("Org/Visible");

  assert.deepEqual(
    resolveDiscoverSelection({
      selectedId: selected.id,
      discoverRows: [visible],
      filteredDiscoverRows: [visible],
      selectedSnapshotId: selected.id,
    }),
    { selectedId: selected.id, hiddenByFilters: true },
  );
});
