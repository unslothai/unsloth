// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { CachedInventoryRow, LocalInventoryRow } from "./types";

function repoFormatKey(
  repoId: string | null | undefined,
  modelFormat: string | null | undefined,
): string | null {
  const normalizedRepo = repoId?.trim().toLowerCase();
  if (!normalizedRepo) {
    return null;
  }
  return `${normalizedRepo}\0${modelFormat ?? "unknown"}`;
}

function repoKey(repoId: string | null | undefined): string | null {
  return repoId?.trim().toLowerCase() || null;
}

type PartialFormatFamily = "gguf" | "model";

function partialFormatFamily(
  modelFormat: CachedInventoryRow["modelFormat"],
): PartialFormatFamily | null {
  if (modelFormat === "unknown") return null;
  return modelFormat === "gguf" ? "gguf" : "model";
}

function addPartialFormatFamily(
  familiesByRepo: Map<string, Set<PartialFormatFamily>>,
  repoId: string | null | undefined,
  modelFormat: CachedInventoryRow["modelFormat"],
): void {
  const repo = repoKey(repoId);
  const family = partialFormatFamily(modelFormat);
  if (!repo || !family) return;
  const families = familiesByRepo.get(repo) ?? new Set<PartialFormatFamily>();
  families.add(family);
  familiesByRepo.set(repo, families);
}

function knownFamiliesMatchUnknownPartial(
  row: Pick<CachedInventoryRow | LocalInventoryRow, "partialTransport">,
  families: ReadonlySet<PartialFormatFamily> | undefined,
): boolean {
  if (!families?.size) return false;
  if (families.size > 1) return true;
  // snapshot partials carry a transport; gguf partials do not.
  return row.partialTransport
    ? families.has("model")
    : families.has("gguf");
}

export function findCompleteHfCacheLocalRow(
  cachedRow: CachedInventoryRow,
  localRows: readonly LocalInventoryRow[],
): LocalInventoryRow | null {
  const key = repoFormatKey(cachedRow.repoId, cachedRow.modelFormat);
  if (!key) {
    return null;
  }
  return (
    localRows.find(
      (row) =>
        row.source === "hf_cache" &&
        !row.partial &&
        repoFormatKey(row.repoId, row.modelFormat) === key,
    ) ?? null
  );
}

export function partialSetFromRows<T extends { partial?: boolean }>(
  rows: readonly T[],
  getRepoId: (row: T) => string | null | undefined,
): Set<string> {
  const complete = new Set<string>();
  for (const row of rows) {
    const repoId = getRepoId(row);
    if (repoId && !row.partial) complete.add(repoId.toLowerCase());
  }
  const partial = new Set<string>();
  for (const row of rows) {
    const repoId = getRepoId(row);
    if (!repoId) continue;
    const key = repoId.toLowerCase();
    if (row.partial && !complete.has(key)) partial.add(key);
  }
  return partial;
}

function preferCachedRow(
  candidate: CachedInventoryRow,
  existing: CachedInventoryRow | null | undefined,
): boolean {
  if (!existing) {
    return true;
  }
  if (Boolean(candidate.partial) !== Boolean(existing.partial)) {
    return !candidate.partial;
  }
  if (Boolean(candidate.liveDownload) !== Boolean(existing.liveDownload)) {
    return Boolean(candidate.liveDownload);
  }
  return candidate.bytes > existing.bytes;
}

function dedupeCachedRows(
  rows: readonly CachedInventoryRow[],
): CachedInventoryRow[] {
  const selected = new Map<string, CachedInventoryRow>();
  const passthrough: CachedInventoryRow[] = [];
  for (const row of rows) {
    const key = repoFormatKey(row.repoId, row.modelFormat);
    if (!key) {
      passthrough.push(row);
      continue;
    }
    const existing = selected.get(key);
    if (preferCachedRow(row, existing)) {
      selected.set(key, row);
    }
  }
  return [...selected.values(), ...passthrough];
}

export function dedupeSameSourceHubCacheRows({
  cachedRows,
  localRows,
}: {
  cachedRows: readonly CachedInventoryRow[];
  localRows: readonly LocalInventoryRow[];
}): {
  cachedRows: CachedInventoryRow[];
  localRows: LocalInventoryRow[];
} {
  const uniqueCachedRows = dedupeCachedRows(cachedRows);
  const completeCachedFormatFamilies = new Map<
    string,
    Set<PartialFormatFamily>
  >();
  const partialFormatFamilies = new Map<string, Set<PartialFormatFamily>>();
  for (const row of uniqueCachedRows) {
    if (row.partial && row.modelFormat !== "unknown") {
      addPartialFormatFamily(
        partialFormatFamilies,
        row.repoId,
        row.modelFormat,
      );
    }
    if (row.partial) {
      continue;
    }
    addPartialFormatFamily(
      completeCachedFormatFamilies,
      row.repoId,
      row.modelFormat,
    );
  }

  const completeHfCacheLocalKeys = new Set<string>();
  for (const row of localRows) {
    if (row.partial && row.modelFormat !== "unknown") {
      addPartialFormatFamily(
        partialFormatFamilies,
        row.repoId,
        row.modelFormat,
      );
    }
    if (row.source !== "hf_cache" || row.partial) {
      continue;
    }
    const key = repoFormatKey(row.repoId, row.modelFormat);
    if (key) {
      completeHfCacheLocalKeys.add(key);
    }
  }

  const filteredCachedRows = uniqueCachedRows.filter((row) => {
    const repo = repoKey(row.repoId);
    if (
      row.partial &&
      !row.liveDownload &&
      row.modelFormat === "unknown" &&
      repo &&
      knownFamiliesMatchUnknownPartial(row, partialFormatFamilies.get(repo))
    ) {
      return false;
    }
    const key = repoFormatKey(row.repoId, row.modelFormat);
    return !(row.partial && key && completeHfCacheLocalKeys.has(key));
  });
  const retainedCachedKeys = new Set(
    filteredCachedRows.flatMap((row) => {
      const key = repoFormatKey(row.repoId, row.modelFormat);
      return key ? [key] : [];
    }),
  );

  return {
    cachedRows: filteredCachedRows,
    localRows: localRows.filter((row) => {
      if (row.source !== "hf_cache") {
        return true;
      }
      const key = repoFormatKey(row.repoId, row.modelFormat);
      if (key && retainedCachedKeys.has(key)) {
        return false;
      }
      const repo = repoKey(row.repoId);
      if (
        row.partial &&
        row.modelFormat === "unknown" &&
        repo &&
        knownFamiliesMatchUnknownPartial(row, partialFormatFamilies.get(repo))
      ) {
        return false;
      }
      if (
        row.modelFormat === "unknown" &&
        repo &&
        knownFamiliesMatchUnknownPartial(
          row,
          completeCachedFormatFamilies.get(repo),
        )
      ) {
        return false;
      }
      return true;
    }),
  };
}
