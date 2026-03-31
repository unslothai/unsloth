// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ModelEntry, modelInfo } from "@huggingface/hub";

/**
 * Thin caching + throttling layer over `modelInfo()` from @huggingface/hub.
 *
 * - TTL cache: avoids re-fetching the same model within CACHE_TTL_MS
 * - In-flight dedup: concurrent callers for the same key share one request
 * - Concurrency limiter: at most MAX_CONCURRENT requests in parallel;
 *   the rest queue and fire as slots free up
 */

const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes
// HF API allows bursts but rate-limits sustained traffic; 3 parallel requests
// keeps startup snappy while staying well under the observed throttle threshold.
const MAX_CONCURRENT = 3;

// ── Cache & in-flight maps ──────────────────────────────────────

// Extend ModelEntry with the additional fields we always request so callers
// do not need unsafe casts to access safetensors/tags.
export type CachedResult = ModelEntry & {
  safetensors?: { total?: number; parameters?: Record<string, number> };
  tags?: string[];
};

interface CacheEntry {
  data: CachedResult;
  ts: number;
}

const cache = new Map<string, CacheEntry>();
const inflight = new Map<string, Promise<CachedResult>>();

// ── Concurrency semaphore ───────────────────────────────────────

let active = 0;
const waiting: Array<() => void> = [];

function acquire(): Promise<void> {
  if (active < MAX_CONCURRENT) {
    active++;
    return Promise.resolve();
  }
  return new Promise<void>((resolve) =>
    waiting.push(() => {
      active++;
      resolve();
    }),
  );
}

function release() {
  active--;
  const next = waiting.shift();
  if (next) {
    next();
  }
}

// ── Public API ──────────────────────────────────────────────────

// Always request the superset of fields any consumer needs so a single
// cache entry covers all callers (e.g. ["safetensors"] and ["safetensors","tags"]).
const ALL_FIELDS: ("safetensors" | "tags")[] = ["safetensors", "tags"];

function isStale(key: string): boolean {
  const hit = cache.get(key);
  if (!hit) return true;
  return Date.now() - hit.ts >= CACHE_TTL_MS;
}

function cacheKey(name: string, token: string | undefined): string {
  if (!token) {
    return `${name}::anon`;
  }
  // Use last 8 chars as a lightweight fingerprint so different tokens get
  // separate cache entries without storing the full secret in memory.
  return `${name}::${token.slice(-8)}`;
}

function extractToken(
  params: Parameters<typeof modelInfo>[0],
): string | undefined {
  // The @huggingface/hub CredentialsParams union supports two forms:
  //   { accessToken: "hf_..." }              -- current preferred form
  //   { credentials: { accessToken: "..." }} -- deprecated form
  // Check both so the cache key is correct regardless of which form callers use.
  if (params.accessToken) {
    return params.accessToken;
  }
  if (params.credentials && "accessToken" in params.credentials) {
    return params.credentials.accessToken;
  }
  return undefined;
}

/**
 * Pre-populate the cache with data from a listModels result.
 * Only writes if the key is not already fresh -- never overwrites a recent
 * modelInfo response with a listing response.
 */
export function primeCacheFromListing(
  name: string,
  token: string | undefined,
  data: CachedResult,
): void {
  if (!name) return;
  const key = cacheKey(name, token);
  if (!isStale(key)) return; // already fresh, don't overwrite
  cache.set(key, { data, ts: Date.now() });
}

export async function cachedModelInfo(
  params: Parameters<typeof modelInfo>[0],
): Promise<CachedResult> {
  const token = extractToken(params);
  const key = cacheKey(params.name, token);

  // 1. Return from cache if fresh
  if (!isStale(key)) {
    return cache.get(key)!.data;
  }

  // 2. Share in-flight request if one exists
  const flying = inflight.get(key);
  if (flying) {
    return flying;
  }

  // 3. New request, gated by concurrency semaphore
  const promise = (async () => {
    await acquire();
    try {
      const result = await modelInfo({
        ...params,
        additionalFields: ALL_FIELDS,
      });
      const entry = { data: result as CachedResult, ts: Date.now() };
      cache.set(key, entry);
      // For public (non-gated, non-private) models, also prime the anonymous
      // cache slot so the VRAM hook (which reads without credentials) gets a
      // cache hit. We skip gated/private models to avoid leaking auth-scoped
      // metadata into the anonymous slot.
      const r = result as CachedResult & { gated?: false | "auto" | "manual"; private?: boolean };
      if (token && !r.private && !r.gated) {
        const anonKey = cacheKey(params.name, undefined);
        if (isStale(anonKey)) {
          cache.set(anonKey, entry);
        }
      }
      return result as CachedResult;
    } finally {
      release();
      inflight.delete(key);
    }
  })();

  inflight.set(key, promise);
  return promise;
}
