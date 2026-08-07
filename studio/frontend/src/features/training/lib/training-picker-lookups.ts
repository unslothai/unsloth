


import {
  type CachedInventoryRow,
  type LocalInventoryRow,
  normalizeModelIdentity,
} from "@/features/hub";

function addLookupValue<T>(
  map: Map<string, T>,
  value: string | null | undefined,
  row: T,
) {
  if (value) {
    map.set(normalizeModelIdentity(value), row);
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
