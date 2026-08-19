// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type {
  CachedInventoryRow,
  LocalInventoryRow,
  ModelInventoryFormat,
} from "./types";

export type InventoryResourceFormatHint =
  | ModelInventoryFormat
  | "non-gguf"
  | null
  | undefined;

export type ResolvedInventoryResource = {
  cachedRow: CachedInventoryRow | null;
  localRow: LocalInventoryRow | null;
};

function repoKey(value: string | null | undefined): string {
  return value?.trim().toLowerCase() ?? "";
}

function cachedRowMatchesFormat(
  row: CachedInventoryRow,
  formatHint: InventoryResourceFormatHint,
): boolean {
  if (!formatHint || formatHint === "unknown") return true;
  if (formatHint === "non-gguf") return row.modelFormat !== "gguf";
  return row.modelFormat === formatHint;
}

function localRowMatchesFormat(
  row: LocalInventoryRow,
  formatHint: InventoryResourceFormatHint,
): boolean {
  if (!formatHint || formatHint === "unknown") return true;
  if (formatHint === "non-gguf") return row.modelFormat !== "gguf";
  return row.modelFormat === formatHint;
}

function hasExplicitFormatHint(
  formatHint: InventoryResourceFormatHint,
): boolean {
  return Boolean(formatHint && formatHint !== "unknown");
}

function bestCompleteCached(
  rows: CachedInventoryRow[],
): CachedInventoryRow | null {
  return (
    rows.find((row) => !row.partial && row.modelFormat === "safetensors") ??
    rows.find((row) => !row.partial && row.modelFormat === "checkpoint") ??
    rows.find((row) => !row.partial && row.modelFormat === "gguf") ??
    rows.find((row) => !row.partial && row.modelFormat === "adapter") ??
    rows.find((row) => !row.partial) ??
    null
  );
}

function bestCompleteLocal(rows: LocalInventoryRow[]): LocalInventoryRow | null {
  return (
    rows.find((row) => !row.partial && row.source !== "hf_cache") ??
    rows.find((row) => !row.partial) ??
    null
  );
}

export function resolveInventoryResource({
  repoId,
  cachedRows,
  localRows,
  formatHint,
}: {
  repoId: string | null | undefined;
  cachedRows: CachedInventoryRow[];
  localRows: LocalInventoryRow[];
  formatHint?: InventoryResourceFormatHint;
}): ResolvedInventoryResource {
  const key = repoKey(repoId);
  if (!key) return { cachedRow: null, localRow: null };

  const repoCachedRows = cachedRows.filter((row) => repoKey(row.repoId) === key);
  const repoLocalRows = localRows.filter((row) => repoKey(row.repoId) === key);
  const compatibleCachedRows = repoCachedRows.filter((row) =>
    cachedRowMatchesFormat(row, formatHint),
  );
  const compatibleLocalRows = repoLocalRows.filter((row) =>
    localRowMatchesFormat(row, formatHint),
  );

  const completeCached = bestCompleteCached(compatibleCachedRows);
  if (completeCached) return { cachedRow: completeCached, localRow: null };

  const completeLocal = bestCompleteLocal(compatibleLocalRows);
  if (completeLocal) return { cachedRow: null, localRow: completeLocal };

  const partialCached =
    compatibleCachedRows.find((row) => row.partial) ?? null;
  if (partialCached) return { cachedRow: partialCached, localRow: null };

  const compatibleLocal = compatibleLocalRows[0] ?? null;
  if (compatibleLocal) return { cachedRow: null, localRow: compatibleLocal };

  if (hasExplicitFormatHint(formatHint)) {
    return { cachedRow: null, localRow: null };
  }

  const fallbackCached = bestCompleteCached(repoCachedRows);
  if (fallbackCached) return { cachedRow: fallbackCached, localRow: null };

  const fallbackLocal = bestCompleteLocal(repoLocalRows);
  if (fallbackLocal) return { cachedRow: null, localRow: fallbackLocal };

  const fallbackPartialCached = repoCachedRows.find((row) => row.partial) ?? null;
  if (fallbackPartialCached) {
    return { cachedRow: fallbackPartialCached, localRow: null };
  }

  return {
    cachedRow: null,
    localRow: repoLocalRows[0] ?? null,
  };
}
