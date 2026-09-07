// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { dedupeSameSourceHubCacheRows } from "../src/features/hub/inventory/inventory-dedupe.ts";
import {
  type SoleQuantEntry,
  type SoleQuantTarget,
  createSoleQuantReader,
  partitionSoleQuants,
  takeDriftedRepos,
} from "../src/features/model-picker/components/model-selector/sole-quant-cache.ts";
import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { buildCachedInventoryRow, buildLocalInventoryRows } = await import(
  "../src/features/hub/inventory/view-models.ts"
);

const repoId = "Org/Model-GGUF";
const diskRow = (folder: string, active: boolean, partial = false) =>
  buildCachedInventoryRow(
    {
      repo_id: repoId,
      inventory_id: `cache:gguf:${repoId}:${folder}`,
      model_format: "gguf",
      load_id: active ? repoId : `${folder}/snapshots/rev`,
      cache_path: folder,
      active_cache: active,
      size_bytes: active ? 6000 : 8000,
      partial,
    },
    "gguf",
  );

for (const active of ["default", "custom"]) {
  test(`both GGUF locations survive inventory deduplication with ${active} active`, () => {
    const input = [
      diskRow("default", active === "default"),
      diskRow("custom", active === "custom"),
    ];
    const result = dedupeSameSourceHubCacheRows({
      cachedRows: input,
      localRows: [],
    });
    assert.equal(result.cachedRows.length, 2);
    assert.deepEqual(
      result.cachedRows.map((row) => row.cachePath),
      ["default", "custom"],
    );
    assert.equal(new Set(result.cachedRows.map((row) => row.id)).size, 2);
    assert.deepEqual(
      result.cachedRows.map((row) => row.loadId),
      input.map((row) => row.loadId),
    );
  });
}

test("a live download coalesces with the active copy and keeps the previous copy", () => {
  const current = diskRow("default", true, true);
  const previous = diskRow("custom", false);
  const live = {
    ...current,
    id: "live",
    cachePath: null,
    activeCache: undefined,
    liveDownload: true,
  };
  const { cachedRows } = dedupeSameSourceHubCacheRows({
    cachedRows: [live, current, previous],
    localRows: [],
  });
  assert.equal(cachedRows.length, 2);
  assert.ok(cachedRows.some((row) => row.liveDownload));
  assert.ok(cachedRows.some((row) => row.cachePath === "custom"));
});

test("a complete local copy in another folder does not hide a partial GGUF", () => {
  const partial = diskRow("/default/models--Org--Model-GGUF", true, true);
  const localRows = buildLocalInventoryRows([
    {
      id: "/custom/models--Org--Model-GGUF/snapshots/rev",
      path: "/custom/models--Org--Model-GGUF/snapshots/rev",
      display_name: repoId,
      model_id: repoId,
      model_format: "gguf",
      source: "hf_cache",
    },
  ]);
  const { cachedRows } = dedupeSameSourceHubCacheRows({
    cachedRows: [partial],
    localRows,
  });
  assert.equal(cachedRows.length, 1);
  assert.equal(cachedRows[0].partial, true);
});

test("concurrent sole-quant probes for one repo retain both folder results without repeated invalidation", async () => {
  const targets: SoleQuantTarget[] = ["default", "custom"].map(
    (folder, index) => ({
      repoId,
      rowId: folder,
      localSource: folder,
      fingerprint: String(index),
      key: folder,
    }),
  );
  const entries = new Map<string, SoleQuantEntry<string>>();
  const committed = new Promise<void>((resolve) => {
    const reader = createSoleQuantReader<string>({
      workers: 1,
      read: async (target) =>
        target.localSource === "default" ? "Q6_K" : "Q8_0",
      commit: (target, quant) => {
        assert.ok(target.rowId);
        entries.set(target.rowId, { key: target.key, quant });
        if (entries.size === 2) resolve();
      },
    });
    reader.start(targets);
  });
  // Let the finite microtask queue finish; a superseded copy never commits.
  await new Promise((resolve) => setImmediate(resolve));
  assert.equal(entries.size, 2, "one folder's read superseded the other");
  await committed;
  const state = partitionSoleQuants(targets, entries, { enabled: true });
  assert.deepEqual(
    [...state.quants],
    [
      ["default", "Q6_K"],
      ["custom", "Q8_0"],
    ],
  );
  assert.equal(state.pending.size, 0);
  const seen = new Map<string, string>();
  assert.deepEqual(takeDriftedRepos(targets, seen), []);
  assert.deepEqual(takeDriftedRepos(targets, seen), []);
  assert.deepEqual(
    takeDriftedRepos([{ ...targets[1], fingerprint: "changed" }], seen),
    [repoId],
  );
});
