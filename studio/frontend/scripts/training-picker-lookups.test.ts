// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import type {
  CachedInventoryRow,
  LocalInventoryRow,
} from "../src/features/inventory/types.ts";
import {
  buildCachedTrainingModelLookup,
  buildLocalTrainingModelLookup,
} from "../src/features/studio/training-picker-lookups.ts";

function cachedRow(
  repoId: string,
  partial: boolean,
  canTrain: boolean,
): CachedInventoryRow {
  return {
    id: `cache:safetensors:${repoId}`,
    repoId,
    loadId: repoId,
    partial,
    capabilities: { canTrain },
  } as CachedInventoryRow;
}

function localRow(
  repoId: string,
  path: string,
  partial: boolean,
  canTrain: boolean,
): LocalInventoryRow {
  return {
    id: path,
    repoId,
    loadId: repoId,
    path,
    partial,
    capabilities: { canTrain },
  } as LocalInventoryRow;
}

test("training cached lookup excludes partial and non-trainable rows", () => {
  const complete = cachedRow("Org/Complete", false, true);
  const partial = cachedRow("Org/Partial", true, true);
  const nonTrainable = cachedRow("Org/NonTrainable", false, false);
  const lookup = buildCachedTrainingModelLookup(
    [complete, partial, nonTrainable],
    (row) => !row.partial && row.capabilities.canTrain,
  );

  assert.equal(lookup.get("org/complete"), complete);
  assert.equal(lookup.get("org/partial"), undefined);
  assert.equal(lookup.get("org/nontrainable"), undefined);
});

test("training local lookup excludes partial and non-trainable rows", () => {
  const complete = localRow("Org/Complete", "/models/complete", false, true);
  const partial = localRow("Org/Partial", "/models/partial", true, true);
  const nonTrainable = localRow(
    "Org/NonTrainable",
    "/models/non-trainable",
    false,
    false,
  );
  const lookup = buildLocalTrainingModelLookup(
    [complete, partial, nonTrainable],
    (row) => !row.partial && row.capabilities.canTrain,
  );

  assert.equal(lookup.get("org/complete"), complete);
  assert.equal(lookup.get("/models/complete"), complete);
  assert.equal(lookup.get("org/partial"), undefined);
  assert.equal(lookup.get("/models/partial"), undefined);
  assert.equal(lookup.get("org/nontrainable"), undefined);
  assert.equal(lookup.get("/models/non-trainable"), undefined);
});
