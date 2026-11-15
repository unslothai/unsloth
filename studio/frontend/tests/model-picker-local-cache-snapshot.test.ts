// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { registerBundlerResolver } from "./helpers/kit.ts";
import {
  isLocalCacheSnapshot,
  isOtherCacheSnapshot,
  pickerLocalCacheSnapshots,
} from "../src/features/model-picker/inventory/local-cache-snapshot.ts";

registerBundlerResolver();
const { buildCachedInventoryRow, buildLocalInventoryRows } = await import(
  "../src/features/hub/inventory/view-models.ts"
);
const { dedupeSameSourceHubCacheRows } = await import(
  "../src/features/hub/inventory/inventory-dedupe.ts"
);

const snapshot = (repoId: string, formats: readonly ("safetensors" | "gguf")[]) => {
  const path = String.raw`D:\models\hub\models--${repoId.replace("/", "--")}\snapshots\abc`;
  return buildLocalInventoryRows(
    formats.map((format) => ({
      id: path, load_id: path, path, display_name: repoId, model_id: repoId,
      inventory_id: `local:${format}:${repoId}`, model_format: format,
      source: "hf_cache", active_cache: false,
    })),
  );
};
const offered = (rows: Set<{ modelId?: string | null; modelFormat: string }>) =>
  [...rows].map((row) => [row.modelId, row.modelFormat]);

test("offers single-format snapshots from another cache", () => {
  const rows = [...snapshot("example/St", ["safetensors"]), ...snapshot("example/Gg", ["gguf"])];
  assert.deepEqual(offered(pickerLocalCacheSnapshots(rows)), [
    ["example/St", "safetensors"],
    ["example/Gg", "gguf"],
  ]);
});

test("a mixed snapshot is offered as GGUF only, since its directory loads the GGUF", () => {
  const rows = snapshot("example/Mixed", ["safetensors", "gguf"]);
  assert.deepEqual(offered(pickerLocalCacheSnapshots(rows)), [["example/Mixed", "gguf"]]);
});

test("a partial GGUF does not hide the snapshot's complete Safetensors row", () => {
  const [st, gguf] = snapshot("example/Mixed", ["safetensors", "gguf"]);
  assert.deepEqual(offered(pickerLocalCacheSnapshots([st, { ...gguf, partial: true }])), [
    ["example/Mixed", "safetensors"],
  ]);
});

test("a GGUF the cached listing kept still hides the snapshot's other formats", () => {
  const rows = snapshot("example/Mixed", ["safetensors", "gguf"]);
  const cached = buildCachedInventoryRow(
    { repo_id: "example/Mixed", load_id: rows[0].loadId, size_bytes: 1 },
    "gguf",
  );
  const inventory = dedupeSameSourceHubCacheRows({ cachedRows: [cached], localRows: rows });
  assert.equal(
    pickerLocalCacheSnapshots(inventory.localRows, inventory.cachedRows).size,
    0,
  );
});

test("an active-cache GGUF copy cannot unhide a custom snapshot's other formats", () => {
  const rows = snapshot("example/Mixed", ["safetensors", "gguf"]);
  const active = buildCachedInventoryRow({ repo_id: "example/Mixed", size_bytes: 1 }, "gguf");
  const inventory = dedupeSameSourceHubCacheRows({ cachedRows: [active], localRows: rows });
  assert.deepEqual(inventory.localRows.map((row) => row.modelFormat), ["safetensors"]);
  const preDedupe = rows.filter((row) => row.modelFormat === "gguf").map((row) => row.loadId);
  assert.equal(
    pickerLocalCacheSnapshots(inventory.localRows, inventory.cachedRows, preDedupe).size,
    0,
  );
});

test("only complete, loadable snapshots pinned to another cache qualify", () => {
  const [row] = snapshot("example/St", ["safetensors"]);
  assert.equal(isLocalCacheSnapshot(row), true);
  for (const override of [
    { activeCache: true }, { activeCache: null }, { loadId: "example/St" }, { partial: true },
    { companionPrefetch: true }, { source: "custom" as const },
    { capabilities: { ...row.capabilities, canChat: false } },
  ]) {
    assert.equal(isLocalCacheSnapshot({ ...row, ...override }), false);
  }
});

test("auto-load admits another cache's snapshot by path, never an active-cache or prefetch row", () => {
  const path = String.raw`D:\models\hub\models--o--M\snapshots\abc`;
  assert.equal(isOtherCacheSnapshot({ id: path, load_id: path, active_cache: false }), true);
  assert.equal(isOtherCacheSnapshot({ id: "o/M", active_cache: true }), false);
  assert.equal(isOtherCacheSnapshot({ id: "o/M", load_id: "o/M", active_cache: false }), false);
  assert.equal(isOtherCacheSnapshot({ id: path, active_cache: null }), false);
  assert.equal(
    isOtherCacheSnapshot({ id: path, active_cache: false, companion_prefetch: true }),
    false,
  );
});
