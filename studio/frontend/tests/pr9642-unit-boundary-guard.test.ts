// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The hazard this file guards is not a bug today. It is that after #9642 the SAME
// field name carries two different units on two different paths: `last_modified`
// is epoch seconds straight off the wire from listCachedGguf / listCachedModels,
// and epoch milliseconds when the picker builds the same shape from a
// CachedInventoryRow. Every current consumer either subtracts like for like or
// stringifies, so nothing renders wrongly -- but the next person to format one of
// these fields has a 1000x bug waiting.
//
// These tests fail when a new reader appears, so that reader has to declare its
// unit rather than inherit whichever one happened to arrive.

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { normalizeTimestamp } = await import(
  "../src/features/hub/inventory/view-models.ts"
);

async function read(relative: string): Promise<string> {
  return readFile(new URL(relative, import.meta.url), "utf8");
}

// ---------------------------------------------------------------------------
// H10 -- formatUpdatedDate assumes SECONDS and must never meet a normalized row
// ---------------------------------------------------------------------------

test("formatUpdatedDate still hardcodes seconds, and still has exactly one caller", async () => {
  const helpers = await read(
    "../src/features/studio/sections/dataset-panel-helpers.ts",
  );
  assert.match(
    helpers,
    /new Date\(timestamp \* 1000\)/,
    "formatUpdatedDate's seconds assumption changed; re-check its callers",
  );

  const selection = await read(
    "../src/features/studio/sections/dataset-selection.tsx",
  );
  const callers = selection.match(/formatUpdatedDate\(/g) ?? [];
  assert.equal(callers.length, 1, "a new formatUpdatedDate caller appeared");
  assert.match(
    selection,
    /formatUpdatedDate\(dataset\.updated_at \?\? null\)/,
    "the only caller must pass a RAW LocalDatasetInfo.updated_at, in seconds",
  );
});

test("a normalized inventory value through formatUpdatedDate would be absurd", () => {
  // Demonstrating the trap rather than describing it: this is what would render
  // if anyone routed a CachedInventoryRow.lastModified into that helper.
  const seconds = 1_750_000_000;
  const normalizedMs = normalizeTimestamp(seconds);
  assert.equal(normalizedMs, seconds * 1000);
  const wrong = new Date((normalizedMs as number) * 1000);
  assert.ok(
    wrong.getUTCFullYear() > 50_000,
    "the double multiply should land tens of thousands of years out",
  );
  const right = new Date(seconds * 1000);
  assert.equal(right.getUTCFullYear(), 2025);
});

// ---------------------------------------------------------------------------
// H9 -- pin the set of readers, so a new one has to be declared
// ---------------------------------------------------------------------------

/** Every module that reads a timestamp off an inventory row, and its unit. */
const DECLARED_READERS: ReadonlyArray<{ file: string; unit: "ms" | "seconds" }> = [
  // Normalizes on the way in; everything downstream of it is ms.
  { file: "../src/features/hub/inventory/view-models.ts", unit: "ms" },
  // The Recent comparator. Subtracts ms from ms.
  { file: "../src/features/hub/catalog/inventory-sort.ts", unit: "ms" },
  // Live download rows carry Date.now(), already ms.
  { file: "../src/features/hub/inventory/use-hub-inventory.ts", unit: "ms" },
  // Optimistic hint rows carry the hint's own clock, ms.
  { file: "../src/features/hub/inventory/inventory-hints.ts", unit: "ms" },
  // Wire DTOs. Seconds off the raw endpoints, ms when built from a row.
  { file: "../src/features/hub/inventory/api.ts", unit: "seconds" },
  { file: "../src/features/chat/api/chat-api.ts", unit: "seconds" },
  // Forwards a row's ms into the seconds-named DTO. The documented seam.
  { file: "../src/features/model-picker/inventory/use-chat-picker-inventory.ts", unit: "ms" },
];

test("every declared reader still declares its unit in a comment", async () => {
  for (const { file } of DECLARED_READERS) {
    const source = await read(file);
    // assert.ok, not assert.match: a failed match dumps the whole module into
    // the report, which buries the one line that matters.
    assert.ok(
      /epoch (seconds|milliseconds)/i.test(source),
      `${file} reads an inventory timestamp but names no unit`,
    );
  }
});

test("the picker's seconds-named DTOs are only ever compared to themselves", async () => {
  const pickers = await read(
    "../src/features/model-picker/components/model-selector/pickers.tsx",
  );
  // sortCachedRepos / sortLocalModels are scale invariant only while each
  // subtraction has the same unit on both sides. A mixed subtraction is the
  // failure mode, so assert the shape of every one of them.
  const subtractions = pickers.match(
    /\(b\.(last_modified|updated_at|updatedAt) \?\? -1\) - \(a\.\1 \?\? -1\)/g,
  );
  assert.ok(
    subtractions && subtractions.length >= 2,
    "the like-for-like timestamp subtractions in pickers.tsx changed shape",
  );
  assert.doesNotMatch(
    pickers,
    /last_modified[^\n]*-[^\n]*loadedAt|loadedAt[^\n]*-[^\n]*last_modified/,
    "a wall-clock load time must never be subtracted from a file mtime",
  );
});

test("no module multiplies or divides an inventory timestamp by 1000", async () => {
  // normalizeTimestamp is the single sanctioned conversion. dataset-panel-helpers
  // is the one other place a *1000 exists, and it is on the raw local-dataset
  // path guarded above.
  for (const { file } of DECLARED_READERS) {
    const source = await read(file);
    const conversions = source.match(/(lastModified|last_modified|updatedAt|updated_at)[^\n]{0,40}[*/]\s*1000/g);
    assert.equal(
      conversions,
      null,
      `${file} converts a timestamp scale outside normalizeTimestamp: ${conversions}`,
    );
  }
});
