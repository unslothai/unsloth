// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { CachedGgufRepo, CachedRepoCopy } from "@/features/chat";
import type { CachedInventoryCopy, GgufVariantDetail } from "@/features/hub";
import { cachePathKey } from "@/lib/cache-path-key";

export function toCachedRepoCopies(
  copies: readonly CachedInventoryCopy[],
): CachedRepoCopy[] {
  return copies.map((copy) => ({
    cache_path: copy.cachePath,
    load_id: copy.loadId,
    size_bytes: copy.bytes,
    active_cache: copy.activeCache,
    partial: copy.partial,
    last_modified: copy.lastModified,
  }));
}

export interface CachedRepoValidationTarget {
  cachePath?: string;
  loadId?: string;
  copyCount: number;
}

export interface DownloadedQuantTargetResult {
  target: CachedRepoValidationTarget;
  downloadedQuants: readonly string[];
}

export interface CachedRepoVariantSource {
  localPath?: string;
  cachePath?: string;
  loadId?: string;
  activeCache: boolean;
}

export type CacheScopedGgufVariant = GgufVariantDetail & {
  cachePath?: string;
  loadId?: string;
  activeCache: boolean;
  contextLength: number | null;
};

export interface CachedGgufVariantResult {
  source: CachedRepoVariantSource;
  variants: readonly GgufVariantDetail[];
  contextLength?: number | null;
}

export function canMigrateCachedRepoToActiveCache(
  cached: Pick<CachedGgufRepo, "active_cache">,
): boolean {
  return typeof cached.active_cache === "boolean";
}

export function cachedRepoVariantSources(
  cached: Pick<
    CachedGgufRepo,
    "repo_id" | "load_id" | "cache_path" | "active_cache" | "cache_copies"
  >,
): CachedRepoVariantSource[] {
  const sources = new Map<string, CachedRepoVariantSource>();
  const add = (
    localPath: string | null | undefined,
    cachePath: string | null | undefined,
    loadId: string | null | undefined,
    activeCache: boolean,
  ) => {
    const local = localPath?.trim() || undefined;
    const cache = cachePath?.trim() || undefined;
    const load = loadId?.trim() || undefined;
    if (!local && !cache) return;
    const key = cachePathKey(cache ?? local ?? "");
    const existing = sources.get(key);
    if (existing) {
      existing.activeCache ||= activeCache;
      existing.loadId ??= load;
      return;
    }
    sources.set(key, {
      localPath: local ?? cache,
      cachePath: cache,
      loadId: load,
      activeCache,
    });
  };

  add(
    cached.load_id,
    cached.cache_path,
    cached.load_id,
    cached.active_cache === true,
  );
  for (const copy of cached.cache_copies ?? []) {
    add(
      copy.active_cache ? copy.cache_path : (copy.load_id ?? copy.cache_path),
      copy.cache_path,
      copy.load_id ?? (copy.active_cache ? cached.repo_id : undefined),
      copy.active_cache === true,
    );
  }
  return [...sources.values()];
}

export function mergeCachedGgufVariantResults(
  results: readonly CachedGgufVariantResult[],
): CacheScopedGgufVariant[] {
  const merged = new Map<string, CacheScopedGgufVariant>();
  const rank = (variant: GgufVariantDetail) =>
    variant.downloaded === true ? 2 : variant.partial === true ? 1 : 0;
  for (const result of results) {
    for (const variant of result.variants) {
      const key = variant.quant.trim().toLowerCase();
      const existing = merged.get(key);
      if (existing && rank(existing) >= rank(variant)) continue;
      merged.set(key, {
        ...variant,
        cachePath: result.source.cachePath,
        loadId: result.source.loadId,
        activeCache: result.source.activeCache,
        contextLength: result.contextLength ?? null,
      });
    }
  }
  return [...merged.values()];
}

/** Ordered physical roots to inspect for pinned quants. The selected row path
 * stays first, then the active copy, then historical copies. A pathless target
 * is retained only for older backends that expose no physical paths. */
export function cachedRepoValidationTargets(
  cached: CachedGgufRepo,
): CachedRepoValidationTarget[] {
  const targets: CachedRepoValidationTarget[] = [];
  const seen = new Set<string>();
  const add = (
    candidate: string | null | undefined,
    loadId: string | null | undefined,
  ) => {
    const path = candidate?.trim();
    if (!path) {
      return;
    }
    const key = cachePathKey(path);
    if (!key || seen.has(key)) {
      return;
    }
    seen.add(key);
    targets.push({
      cachePath: path,
      loadId: loadId?.trim() || undefined,
      copyCount: 1,
    });
  };

  add(cached.cache_path, cached.load_id);
  for (const copy of cached.cache_copies ?? []) {
    if (copy.active_cache) {
      add(copy.cache_path, copy.load_id ?? cached.repo_id);
    }
  }
  for (const copy of cached.cache_copies ?? []) {
    add(copy.cache_path, copy.load_id);
  }

  const copyCount = Math.max(cached.copy_count ?? 0, targets.length, 1);
  return targets.length > 0
    ? targets.map((target) => ({ ...target, copyCount }))
    : [{ cachePath: undefined, copyCount }];
}

/** Resolve each downloaded quant to the first physical target that reported
 * it. Results are passed in validation order, making target choice stable. */
export function downloadedQuantCacheTargets(
  results: readonly DownloadedQuantTargetResult[],
): ReadonlyMap<string, CachedRepoValidationTarget> {
  const targets = new Map<string, CachedRepoValidationTarget>();
  for (const result of results) {
    for (const quant of result.downloadedQuants) {
      if (!targets.has(quant)) {
        targets.set(quant, result.target);
      }
    }
  }
  return targets;
}
