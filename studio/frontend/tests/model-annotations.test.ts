// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { registerStoreStubResolver } from "./helpers/kit.ts";

registerStoreStubResolver();
const { buildCachedInventoryRow } = await import(
  "../src/features/hub/inventory/view-models.ts"
);
const { inventoryRowMatches, scoreInventoryRow, tokenizeQuery } = await import(
  "../src/features/hub/lib/inventory-search.ts"
);
const { inventoryItemTitle } = await import(
  "../src/features/hub/catalog/inventory-sort.ts"
);

/** Build the real inventory view model, adding personal details separately from Hub metadata. */
function annotatedRow() {
  const row = buildCachedInventoryRow(
    {
      repo_id: "org/Original-GGUF",
      size_bytes: 1024,
    },
    "gguf",
  );
  return {
    ...row,
    annotation: {
      display_name: "My coding model",
      labels: ["work", "fast"],
      notes: "Good at SQL",
    },
  };
}

test("On Device search finds personal names, labels and notes while retaining the original identity", () => {
  const row = annotatedRow();
  for (const query of ["coding model", "work fast", "SQL", "Original-GGUF"]) {
    assert.equal(inventoryRowMatches(row, tokenizeQuery(query)), true, query);
  }
  assert.equal(row.repoId, "org/Original-GGUF");
  assert.equal(row.loadId, "org/Original-GGUF");
});

test("name sorting and search ranking use the personal display name", () => {
  const row = annotatedRow();
  assert.equal(
    inventoryItemTitle({ variant: "cached", row }),
    "My coding model",
  );
  assert.equal(scoreInventoryRow(row, tokenizeQuery("My coding")), 5);
});

test("attaching details preserves raw inventory objects, load identity and capability tags", async () => {
  const { annotateInventoryRows } = await import(
    "../src/features/hub/lib/model-annotations.ts"
  );
  const original = buildCachedInventoryRow(
    {
      repo_id: "org/Original-GGUF",
      size_bytes: 1024,
      tags: ["text-generation"],
    },
    "gguf",
  );
  const details = annotatedRow().annotation;
  const [view] = annotateInventoryRows([original], { [original.id]: details });
  assert.equal(annotateInventoryRows([original], {})[0], original);
  assert.equal(original.annotation, undefined);
  assert.equal(view.id, original.id);
  assert.equal(view.loadId, original.loadId);
  assert.equal(view.repoId, original.repoId);
  assert.equal(view.tags, original.tags);
  assert.equal(view.capabilities, original.capabilities);
  assert.deepEqual(view.annotation, details);
});
