// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { resolveInventoryResource } from "../src/features/inventory/resource-resolver.ts";
import type {
  CachedInventoryRow,
  LocalInventoryRow,
  ModelInventoryFormat,
} from "../src/features/inventory/types.ts";

function capabilities() {
  return {
    canTrain: true,
    canChat: true,
    canDelete: false,
    canDownload: false,
    requiresVariant: false,
    supportsLora: true,
    supportsVision: false,
  };
}

function cachedRow({
  id,
  repoId,
  format,
  partial = false,
}: {
  id: string;
  repoId: string;
  format: ModelInventoryFormat;
  partial?: boolean;
}): CachedInventoryRow {
  return {
    kind: "cache",
    id,
    loadId: repoId,
    repoId,
    owner: repoId.split("/")[0] ?? "Org",
    repo: repoId.split("/")[1] ?? repoId,
    isGguf: format === "gguf",
    modelFormat: format,
    runtime: format === "gguf" ? "llama_cpp" : "transformers",
    capabilities: {
      ...capabilities(),
      canTrain: format !== "gguf",
      requiresVariant: format === "gguf",
    },
    bytes: 100,
    cachePath: `/cache/${id}`,
    partial,
  };
}

function localRow({
  id,
  repoId,
  partial = false,
  source = "custom",
}: {
  id: string;
  repoId: string;
  partial?: boolean;
  source?: LocalInventoryRow["source"];
}): LocalInventoryRow {
  return {
    kind: "local",
    id,
    loadId: id,
    repoId,
    owner: repoId.split("/")[0] ?? "Org",
    title: repoId.split("/")[1] ?? repoId,
    source,
    sourceLabel: "Custom folder",
    path: id,
    isGguf: false,
    modelFormat: "safetensors",
    runtime: "transformers",
    capabilities: capabilities(),
    updatedAt: null,
    partial,
  };
}

test("repo-id discovery resolves cached inventory ids", () => {
  const row = cachedRow({
    id: "cache:gguf:Org%2FModel",
    repoId: "Org/Model",
    format: "gguf",
  });

  const resolved = resolveInventoryResource({
    repoId: "Org/Model",
    cachedRows: [row],
    localRows: [],
    formatHint: "gguf",
  });

  assert.equal(resolved.cachedRow?.id, "cache:gguf:Org%2FModel");
  assert.equal(resolved.cachedRow?.loadId, "Org/Model");
  assert.equal(resolved.localRow, null);
});

test("mixed cached repos respect the requested format", () => {
  const gguf = cachedRow({
    id: "cache:gguf:Org%2FMixed",
    repoId: "Org/Mixed",
    format: "gguf",
  });
  const safetensors = cachedRow({
    id: "cache:safetensors:Org%2FMixed",
    repoId: "Org/Mixed",
    format: "safetensors",
  });

  assert.equal(
    resolveInventoryResource({
      repoId: "Org/Mixed",
      cachedRows: [gguf, safetensors],
      localRows: [],
      formatHint: "gguf",
    }).cachedRow?.id,
    gguf.id,
  );
  assert.equal(
    resolveInventoryResource({
      repoId: "Org/Mixed",
      cachedRows: [gguf, safetensors],
      localRows: [],
      formatHint: "non-gguf",
    }).cachedRow?.id,
    safetensors.id,
  );
});

test("incompatible cached rows do not satisfy an explicit format", () => {
  const gguf = cachedRow({
    id: "cache:gguf:Org%2FMixed",
    repoId: "Org/Mixed",
    format: "gguf",
  });

  const resolved = resolveInventoryResource({
    repoId: "Org/Mixed",
    cachedRows: [gguf],
    localRows: [],
    formatHint: "non-gguf",
  });

  assert.equal(resolved.cachedRow, null);
  assert.equal(resolved.localRow, null);
});

test("complete local rows beat partial compatible cache rows", () => {
  const partial = cachedRow({
    id: "cache:safetensors:Org%2FModel",
    repoId: "Org/Model",
    format: "safetensors",
    partial: true,
  });
  const local = localRow({ id: "/models/model", repoId: "Org/Model" });

  const resolved = resolveInventoryResource({
    repoId: "Org/Model",
    cachedRows: [partial],
    localRows: [local],
    formatHint: "non-gguf",
  });

  assert.equal(resolved.cachedRow, null);
  assert.equal(resolved.localRow?.id, "/models/model");
});

test("partial-only cache remains a partial cache resource", () => {
  const partial = cachedRow({
    id: "cache:safetensors:Org%2FModel",
    repoId: "Org/Model",
    format: "safetensors",
    partial: true,
  });

  const resolved = resolveInventoryResource({
    repoId: "Org/Model",
    cachedRows: [partial],
    localRows: [],
    formatHint: "non-gguf",
  });

  assert.equal(resolved.cachedRow?.id, partial.id);
  assert.equal(resolved.cachedRow?.partial, true);
  assert.equal(resolved.localRow, null);
});
