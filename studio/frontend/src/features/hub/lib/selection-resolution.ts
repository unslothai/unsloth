// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { findCompleteHfCacheLocalRow } from "../inventory/inventory-dedupe";
import type {
  CachedInventoryRow,
  DiscoverRow,
  LocalInventoryRow,
} from "../types";

export type SelectionResolution = {
  selectedId: string | null;
  hiddenByFilters: boolean;
};

function idSet(rows: readonly { id: string }[]): Set<string> {
  return new Set(rows.map((row) => row.id));
}

function idMap<T extends { id: string }>(rows: readonly T[]): Map<string, T> {
  return new Map(rows.map((row) => [row.id, row]));
}

function resolveCachedSelection(
  cached: CachedInventoryRow,
  filteredCachedIds: ReadonlySet<string>,
  filteredLocalIds: ReadonlySet<string>,
  localRows: readonly LocalInventoryRow[],
): SelectionResolution {
  if (cached.partial) {
    const completeLocal = findCompleteHfCacheLocalRow(cached, localRows);
    if (completeLocal) {
      return {
        selectedId: completeLocal.id,
        hiddenByFilters: !filteredLocalIds.has(completeLocal.id),
      };
    }
  }
  return {
    selectedId: cached.id,
    hiddenByFilters: !filteredCachedIds.has(cached.id),
  };
}

function resolveDownloadedId(
  id: string | null,
  cachedById: ReadonlyMap<string, CachedInventoryRow>,
  localById: ReadonlyMap<string, LocalInventoryRow>,
  filteredCachedIds: ReadonlySet<string>,
  filteredLocalIds: ReadonlySet<string>,
  localRows: readonly LocalInventoryRow[],
): SelectionResolution | null {
  if (!id) {
    return null;
  }
  const cached = cachedById.get(id);
  if (cached) {
    return resolveCachedSelection(
      cached,
      filteredCachedIds,
      filteredLocalIds,
      localRows,
    );
  }
  if (localById.has(id)) {
    return {
      selectedId: id,
      hiddenByFilters: !filteredLocalIds.has(id),
    };
  }
  return null;
}

export function resolveDiscoverSelection({
  selectedId,
  discoverRows,
  filteredDiscoverRows,
  selectedSnapshotId,
}: {
  selectedId: string | null;
  discoverRows: readonly DiscoverRow[];
  filteredDiscoverRows: readonly DiscoverRow[];
  selectedSnapshotId?: string | null;
}): SelectionResolution {
  const filteredIds = idSet(filteredDiscoverRows);
  if (
    selectedId &&
    (discoverRows.some((row) => row.id === selectedId) ||
      selectedSnapshotId === selectedId)
  ) {
    return {
      selectedId,
      hiddenByFilters: !filteredIds.has(selectedId),
    };
  }
  return {
    selectedId: filteredDiscoverRows[0]?.id ?? null,
    hiddenByFilters: false,
  };
}

export function resolveDownloadedSelection({
  selectedId,
  cachedRows,
  localRows,
  filteredCachedRows,
  filteredLocalRows,
}: {
  selectedId: string | null;
  cachedRows: readonly CachedInventoryRow[];
  localRows: readonly LocalInventoryRow[];
  filteredCachedRows: readonly CachedInventoryRow[];
  filteredLocalRows: readonly LocalInventoryRow[];
}): SelectionResolution {
  const cachedById = idMap(cachedRows);
  const localById = idMap(localRows);
  const filteredCachedIds = idSet(filteredCachedRows);
  const filteredLocalIds = idSet(filteredLocalRows);
  return (
    resolveDownloadedId(
      selectedId,
      cachedById,
      localById,
      filteredCachedIds,
      filteredLocalIds,
      localRows,
    ) ?? { selectedId: null, hiddenByFilters: false }
  );
}
