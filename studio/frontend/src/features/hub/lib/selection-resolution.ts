// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { findCompleteHfCacheLocalRow } from "../inventory/inventory-dedupe";
import { cachedInventoryId } from "../inventory/view-models";
import type {
  CachedInventoryRow,
  DiscoverRow,
  LocalInventoryRow,
} from "../types";

export type SelectionResolution = {
  selectedId: string | null;
  hiddenByFilters: boolean;
};

export type SelectionUrlSync =
  | { action: "select"; selectedId: string | null }
  | { action: "replace"; selectedId: string; preserveGgufFile: boolean }
  | null;

export function resolveSelectionUrlSync({
  isDiscoverTab,
  urlModel,
  selectionInputId,
  resolvedSelectedId,
  resolvedModelFormat,
}: {
  isDiscoverTab: boolean;
  urlModel: string | null;
  selectionInputId: string | null;
  resolvedSelectedId: string | null;
  resolvedModelFormat: CachedInventoryRow["modelFormat"] | null;
}): SelectionUrlSync {
  if (urlModel !== selectionInputId) {
    return { action: "select", selectedId: urlModel };
  }
  if (
    !isDiscoverTab &&
    resolvedSelectedId &&
    resolvedSelectedId !== urlModel
  ) {
    const resolvedFormat =
      resolvedModelFormat ??
      parseCacheSelectionIdentity(resolvedSelectedId)?.modelFormat ??
      null;
    return {
      action: "replace",
      selectedId: resolvedSelectedId,
      preserveGgufFile:
        resolvedFormat === "gguf" ||
        (resolvedFormat === "unknown" &&
          parseCacheSelectionIdentity(urlModel ?? "")?.modelFormat === "gguf"),
    };
  }
  return null;
}

function idSet(rows: readonly { id: string }[]): Set<string> {
  return new Set(rows.map((row) => row.id));
}

function idMap<T extends { id: string }>(rows: readonly T[]): Map<string, T> {
  return new Map(rows.map((row) => [row.id, row]));
}

type CacheSelectionIdentity = {
  source: "cache" | "download" | "hf_cache";
  modelFormat: CachedInventoryRow["modelFormat"];
  repoKey: string;
};

const MODEL_FORMATS = new Set<CachedInventoryRow["modelFormat"]>([
  "gguf",
  "safetensors",
  "adapter",
  "checkpoint",
  "unknown",
]);

function parseCacheSelectionIdentity(
  id: string,
): CacheSelectionIdentity | null {
  const firstSeparator = id.indexOf(":");
  const secondSeparator = id.indexOf(":", firstSeparator + 1);
  if (firstSeparator <= 0 || secondSeparator <= firstSeparator + 1) {
    return null;
  }
  const source = id.slice(0, firstSeparator);
  const modelFormat = id.slice(firstSeparator + 1, secondSeparator);
  const encodedRepoId = id.slice(secondSeparator + 1);
  if (
    (source !== "cache" && source !== "download" && source !== "hf_cache") ||
    !MODEL_FORMATS.has(modelFormat as CachedInventoryRow["modelFormat"]) ||
    !encodedRepoId ||
    encodedRepoId.includes(":")
  ) {
    return null;
  }
  try {
    const repoKey = decodeURIComponent(encodedRepoId).trim().toLowerCase();
    return repoKey
      ? {
          source,
          modelFormat: modelFormat as CachedInventoryRow["modelFormat"],
          repoKey,
        }
      : null;
  } catch {
    return null;
  }
}

function inventoryRepoKey(
  row: Pick<CachedInventoryRow | LocalInventoryRow, "repoId">,
): string {
  return row.repoId?.trim().toLowerCase() ?? "";
}

function hfCacheInventoryId(
  row: Pick<LocalInventoryRow, "modelFormat" | "repoId">,
): string | null {
  if (!row.repoId) {
    return null;
  }
  const canonicalId = cachedInventoryId(row.modelFormat, row.repoId);
  return `hf_cache:${canonicalId.slice("cache:".length)}`;
}

function cachedIdMap(
  rows: readonly CachedInventoryRow[],
): Map<string, CachedInventoryRow> {
  const map = idMap(rows);
  for (const row of rows) {
    const canonicalId = cachedInventoryId(row.modelFormat, row.repoId);
    if (!map.has(canonicalId)) {
      map.set(canonicalId, row);
    }
    const legacyId = `cache:${row.modelFormat}:${row.repoId}`;
    if (!map.has(legacyId)) {
      map.set(legacyId, row);
    }
    if (row.modelFormat !== "unknown") {
      const localId = hfCacheInventoryId(row);
      if (localId && !map.has(localId)) {
        map.set(localId, row);
      }
    }
  }
  return map;
}

function localIdMap(
  rows: readonly LocalInventoryRow[],
): Map<string, LocalInventoryRow> {
  const map = idMap(rows);
  for (const row of rows) {
    if (row.source !== "hf_cache" || !row.repoId) {
      continue;
    }
    if (row.modelFormat === "unknown") {
      continue;
    }
    const canonicalId = cachedInventoryId(row.modelFormat, row.repoId);
    if (!map.has(canonicalId)) {
      map.set(canonicalId, row);
    }
    const legacyId = `cache:${row.modelFormat}:${row.repoId}`;
    if (!map.has(legacyId)) {
      map.set(legacyId, row);
    }
  }
  return map;
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

function resolveCurrentSelection(
  cachedRows: readonly CachedInventoryRow[],
  localRows: readonly LocalInventoryRow[],
  filteredCachedIds: ReadonlySet<string>,
  filteredLocalIds: ReadonlySet<string>,
  allLocalRows: readonly LocalInventoryRow[],
): SelectionResolution | null {
  const formats = new Set([
    ...cachedRows.map((row) => row.modelFormat),
    ...localRows.map((row) => row.modelFormat),
  ]);
  if (formats.size !== 1) {
    return null;
  }
  const cached = cachedRows[0];
  if (cached) {
    return resolveCachedSelection(
      cached,
      filteredCachedIds,
      filteredLocalIds,
      allLocalRows,
    );
  }
  const local = localRows[0];
  return local
    ? {
        selectedId: local.id,
        hiddenByFilters: !filteredLocalIds.has(local.id),
      }
    : null;
}

function isProvisionalFormatCompatible(
  provisional: CachedInventoryRow["modelFormat"],
  current: CachedInventoryRow["modelFormat"],
): boolean {
  if (current === "unknown") {
    return true;
  }
  if (provisional === "safetensors") {
    return (
      current === "safetensors" ||
      current === "adapter" ||
      current === "checkpoint"
    );
  }
  return current === provisional;
}

function resolveFormatTransition(
  id: string | null,
  cachedRows: readonly CachedInventoryRow[],
  localRows: readonly LocalInventoryRow[],
  filteredCachedIds: ReadonlySet<string>,
  filteredLocalIds: ReadonlySet<string>,
): SelectionResolution | null {
  if (!id) {
    return null;
  }
  const identity = parseCacheSelectionIdentity(id);
  if (!identity) {
    return null;
  }
  const matchingCached = cachedRows.filter(
    (row) => inventoryRepoKey(row) === identity.repoKey,
  );
  const matchingLocal = localRows.filter(
    (row) =>
      row.source === "hf_cache" && inventoryRepoKey(row) === identity.repoKey,
  );
  if (identity.source === "download") {
    return resolveCurrentSelection(
      matchingCached.filter((row) =>
        isProvisionalFormatCompatible(identity.modelFormat, row.modelFormat),
      ),
      matchingLocal.filter((row) =>
        isProvisionalFormatCompatible(identity.modelFormat, row.modelFormat),
      ),
      filteredCachedIds,
      filteredLocalIds,
      localRows,
    );
  }
  if (identity.modelFormat === "unknown") {
    if (identity.source === "cache") {
      return resolveCurrentSelection(
        matchingCached.filter((row) => row.modelFormat === "unknown"),
        [],
        filteredCachedIds,
        filteredLocalIds,
        localRows,
      );
    }
    const exactLocal = resolveCurrentSelection(
      [],
      matchingLocal.filter((row) => row.modelFormat === "unknown"),
      filteredCachedIds,
      filteredLocalIds,
      localRows,
    );
    if (exactLocal) {
      return exactLocal;
    }
    return resolveCurrentSelection(
      matchingCached.filter((row) => row.modelFormat !== "unknown"),
      matchingLocal.filter((row) => row.modelFormat !== "unknown"),
      filteredCachedIds,
      filteredLocalIds,
      localRows,
    );
  }
  const exact = resolveCurrentSelection(
    matchingCached.filter((row) => row.modelFormat === identity.modelFormat),
    matchingLocal.filter((row) => row.modelFormat === identity.modelFormat),
    filteredCachedIds,
    filteredLocalIds,
    localRows,
  );
  if (exact) {
    return exact;
  }
  if (
    matchingCached.some((row) => row.modelFormat !== "unknown") ||
    matchingLocal.some((row) => row.modelFormat !== "unknown")
  ) {
    return null;
  }
  return resolveCurrentSelection(
    [],
    matchingLocal.filter((row) => row.modelFormat === "unknown"),
    filteredCachedIds,
    filteredLocalIds,
    localRows,
  );
}

function resolveRawRepoTransition(
  id: string | null,
  cachedRows: readonly CachedInventoryRow[],
  localRows: readonly LocalInventoryRow[],
  filteredCachedIds: ReadonlySet<string>,
  filteredLocalIds: ReadonlySet<string>,
): SelectionResolution | null {
  const segments = id?.trim().split("/") ?? [];
  if (
    segments.length !== 2 ||
    segments.some((segment) => !segment) ||
    id?.includes(":") ||
    id?.includes("\\")
  ) {
    return null;
  }
  const repoKey = segments.join("/").toLowerCase();
  return resolveCurrentSelection(
    cachedRows.filter((row) => inventoryRepoKey(row) === repoKey),
    localRows.filter(
      (row) =>
        row.source === "hf_cache" && inventoryRepoKey(row) === repoKey,
    ),
    filteredCachedIds,
    filteredLocalIds,
    localRows,
  );
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
  const local = localById.get(id);
  if (local) {
    return {
      selectedId: local.id,
      hiddenByFilters: !filteredLocalIds.has(local.id),
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
  inventoryReady = true,
  cachedRows,
  localRows,
  filteredCachedRows,
  filteredLocalRows,
}: {
  selectedId: string | null;
  inventoryReady?: boolean;
  cachedRows: readonly CachedInventoryRow[];
  localRows: readonly LocalInventoryRow[];
  filteredCachedRows: readonly CachedInventoryRow[];
  filteredLocalRows: readonly LocalInventoryRow[];
}): SelectionResolution {
  if (!inventoryReady) {
    return { selectedId, hiddenByFilters: false };
  }
  const cachedById = cachedIdMap(cachedRows);
  const localById = localIdMap(localRows);
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
    ) ??
    resolveFormatTransition(
      selectedId,
      cachedRows,
      localRows,
      filteredCachedIds,
      filteredLocalIds,
    ) ??
    resolveRawRepoTransition(
      selectedId,
      cachedRows,
      localRows,
      filteredCachedIds,
      filteredLocalIds,
    ) ?? { selectedId: null, hiddenByFilters: false }
  );
}
