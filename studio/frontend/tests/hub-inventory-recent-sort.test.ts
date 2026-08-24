// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { compareInventoryItemsByRecent } = await import(
  "../src/features/hub/catalog/inventory-sort.ts"
);
const { buildCachedInventoryRow, buildLocalInventoryRows } = await import(
  "../src/features/hub/inventory/view-models.ts"
);

function cached(repoId: string, lastModified?: number) {
  return {
    variant: "cached" as const,
    row: buildCachedInventoryRow(
      {
        repo_id: repoId,
        size_bytes: 1,
        last_modified: lastModified,
      },
      "gguf",
    ),
  };
}

function local(displayName: string, updatedAt?: number) {
  const [row] = buildLocalInventoryRows([
    {
      id: `local:${displayName}`,
      display_name: displayName,
      path: `C:\\models\\${displayName}.gguf`,
      source: "lmstudio",
      updated_at: updatedAt,
    },
  ]);
  return { variant: "local" as const, row };
}

test("Recent sorts cached and local rows together by normalized timestamp", () => {
  const oldCached = cached("Org/Alpha-Old", 1_700_000_000);
  const middleLocal = local("LM Studio Middle", 1_800_000_000_000);
  const newCached = cached("Org/Zulu-New", 1_900_000_000);

  assert.equal(oldCached.row.lastModified, 1_700_000_000_000);
  assert.equal(newCached.row.lastModified, 1_900_000_000_000);

  const sorted = [oldCached, middleLocal, newCached].sort(
    compareInventoryItemsByRecent,
  );
  assert.deepEqual(
    sorted.map((item) => item.row.id),
    [newCached.row.id, middleLocal.row.id, oldCached.row.id],
  );
});

test("Recent puts unknown timestamps last and leaves ties stable", () => {
  const unknownCached = cached("Org/Unknown");
  const firstTie = local("First tie", 1_800_000_000);
  const secondTie = cached("Org/Second-Tie", 1_800_000_000);

  const sorted = [unknownCached, firstTie, secondTie].sort(
    compareInventoryItemsByRecent,
  );
  assert.deepEqual(
    sorted.map((item) => item.row.id),
    [firstTie.row.id, secondTie.row.id, unknownCached.row.id],
  );
});
