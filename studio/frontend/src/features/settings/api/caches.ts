// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

/**
 * Caches are named by key, never by path: the backend owns the mapping from a
 * key to a directory, so nothing the browser sends can widen what a purge
 * reaches.
 */
export const CACHE_KEYS = [
  "uv",
  "pip",
  "npm",
  "bun",
  "torch_inductor",
  "torch_extensions",
  "triton",
  "cuda",
  "numba",
  "matplotlib",
  "vllm",
  "unsloth_compiled",
  "hf_xet",
  "hf_assets",
  "hf_datasets",
  "hf_hub",
] as const;

export type CacheKey = (typeof CACHE_KEYS)[number];
export type CacheGroup = "packages" | "compile" | "models";

export type CacheEntry = {
  key: CacheKey;
  group: CacheGroup;
  /** Clearing this costs a re-download, so it is never part of a bulk purge. */
  optIn: boolean;
  paths: string[];
  sizeBytes: number;
  entryCount: number;
  present: boolean;
  purgeable: boolean;
  blockedReason: string | null;
};

export type CacheInventory = {
  caches: CacheEntry[];
  totalBytes: number;
  reclaimableBytes: number;
  freeBytes: number | null;
  totalDiskBytes: number | null;
};

export type CachePurgeResult = {
  key: CacheKey;
  freedBytes: number;
  removedEntries: number;
  errors: string[];
};

export type CachePurgeOutcome = {
  results: CachePurgeResult[];
  freedBytes: number;
  inventory: CacheInventory;
};

type ApiCacheEntry = {
  key: CacheKey;
  group: CacheGroup;
  // biome-ignore lint/style/useNamingConvention: API schema
  opt_in: boolean;
  paths: string[];
  // biome-ignore lint/style/useNamingConvention: API schema
  size_bytes: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  entry_count: number;
  present: boolean;
  purgeable: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  blocked_reason: string | null;
};

type ApiCacheInventory = {
  caches: ApiCacheEntry[];
  // biome-ignore lint/style/useNamingConvention: API schema
  total_bytes: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  reclaimable_bytes: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  free_bytes: number | null;
  // biome-ignore lint/style/useNamingConvention: API schema
  total_disk_bytes: number | null;
};

type ApiCachePurgeResult = {
  key: CacheKey;
  // biome-ignore lint/style/useNamingConvention: API schema
  freed_bytes: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  removed_entries: number;
  errors: string[];
};

type ApiCachePurgeOutcome = {
  results: ApiCachePurgeResult[];
  // biome-ignore lint/style/useNamingConvention: API schema
  freed_bytes: number;
  inventory: ApiCacheInventory;
};

function entryFromApi(value: ApiCacheEntry): CacheEntry {
  return {
    key: value.key,
    group: value.group,
    optIn: value.opt_in,
    paths: value.paths,
    sizeBytes: value.size_bytes,
    entryCount: value.entry_count,
    present: value.present,
    purgeable: value.purgeable,
    blockedReason: value.blocked_reason,
  };
}

export function inventoryFromApi(value: ApiCacheInventory): CacheInventory {
  return {
    caches: value.caches.map(entryFromApi),
    totalBytes: value.total_bytes,
    reclaimableBytes: value.reclaimable_bytes,
    freeBytes: value.free_bytes,
    totalDiskBytes: value.total_disk_bytes,
  };
}

export function purgeOutcomeFromApi(
  value: ApiCachePurgeOutcome,
): CachePurgeOutcome {
  return {
    results: value.results.map((result) => ({
      key: result.key,
      freedBytes: result.freed_bytes,
      removedEntries: result.removed_entries,
      errors: result.errors,
    })),
    freedBytes: value.freed_bytes,
    inventory: inventoryFromApi(value.inventory),
  };
}

export async function loadCacheInventory(
  options: { refresh?: boolean } = {},
): Promise<CacheInventory> {
  // refresh re-walks the caches instead of reusing a size measured a moment
  // ago, which is what the Recheck action is for.
  const response = await authFetch(
    options.refresh
      ? "/api/settings/caches?refresh=true"
      : "/api/settings/caches",
  );
  if (!response.ok) {
    throw new Error(
      await readFastApiError(response, "Failed to measure the caches"),
    );
  }
  return inventoryFromApi(await response.json());
}

export async function purgeCaches(
  keys: readonly CacheKey[],
): Promise<CachePurgeOutcome> {
  const response = await authFetch("/api/settings/caches/purge", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ keys }),
  });
  if (!response.ok) {
    throw new Error(
      await readFastApiError(response, "Failed to clear the caches"),
    );
  }
  return purgeOutcomeFromApi(await response.json());
}

/** The caches a bulk clear covers: present, allowed, and free to rebuild.
 *
 * Entries rather than bytes. A tree of empty directories or dangling symlinks
 * measures zero and still costs inodes, and the backend can empty it, so a size
 * test would leave the one cache nobody can clear by hand as the one the UI
 * refuses to.
 */
export function bulkPurgeKeys(inventory: CacheInventory): CacheKey[] {
  return inventory.caches
    .filter(
      (entry) =>
        entry.present &&
        entry.purgeable &&
        !entry.optIn &&
        entry.entryCount > 0,
    )
    .map((entry) => entry.key);
}
