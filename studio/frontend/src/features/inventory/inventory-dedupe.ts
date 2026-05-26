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
  const completeCachedKeys = new Set<string>();
  const completeCachedRepos = new Set<string>();
  for (const row of cachedRows) {
    if (row.partial) {
      continue;
    }
    const key = repoFormatKey(row.repoId, row.modelFormat);
    if (key) {
      completeCachedKeys.add(key);
    }
    const repo = repoKey(row.repoId);
    if (repo) {
      completeCachedRepos.add(repo);
    }
  }

  const completeHfCacheLocalKeys = new Set<string>();
  for (const row of localRows) {
    if (row.source !== "hf_cache" || row.partial) {
      continue;
    }
    const key = repoFormatKey(row.repoId, row.modelFormat);
    if (key) {
      completeHfCacheLocalKeys.add(key);
    }
  }

  return {
    cachedRows: cachedRows.filter((row) => {
      const key = repoFormatKey(row.repoId, row.modelFormat);
      return !(row.partial && key && completeHfCacheLocalKeys.has(key));
    }),
    localRows: localRows.filter((row) => {
      if (row.source !== "hf_cache") {
        return true;
      }
      const key = repoFormatKey(row.repoId, row.modelFormat);
      if (key && completeCachedKeys.has(key)) {
        return false;
      }
      const repo = repoKey(row.repoId);
      if (
        row.modelFormat === "unknown" &&
        repo &&
        completeCachedRepos.has(repo)
      ) {
        return false;
      }
      return true;
    }),
  };
}
