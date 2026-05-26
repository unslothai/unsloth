// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  CachedInventoryRow,
  LocalInventoryRow,
} from "@/features/inventory";

function addLookupValue<T>(
  map: Map<string, T>,
  value: string | null | undefined,
  row: T,
) {
  if (value) {
    map.set(value.toLowerCase(), row);
  }
}

export function buildCachedTrainingModelLookup(
  rows: readonly CachedInventoryRow[],
  canUse: (row: CachedInventoryRow) => boolean,
): Map<string, CachedInventoryRow> {
  const map = new Map<string, CachedInventoryRow>();
  for (const row of rows) {
    if (!canUse(row)) {
      continue;
    }
    addLookupValue(map, row.id, row);
    addLookupValue(map, row.repoId, row);
    addLookupValue(map, row.loadId, row);
  }
  return map;
}

export function buildLocalTrainingModelLookup(
  rows: readonly LocalInventoryRow[],
  canUse: (row: LocalInventoryRow) => boolean,
): Map<string, LocalInventoryRow> {
  const map = new Map<string, LocalInventoryRow>();
  for (const row of rows) {
    if (!canUse(row)) {
      continue;
    }
    addLookupValue(map, row.id, row);
    addLookupValue(map, row.repoId, row);
    addLookupValue(map, row.loadId, row);
    addLookupValue(map, row.path, row);
  }
  return map;
}
