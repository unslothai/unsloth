// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { modelInfo, type ModelEntry } from "@huggingface/hub";

/**
 * Thin caching + throttling layer over `modelInfo()` from @huggingface/hub.
 *
 * - TTL cache: avoids re-fetching the same model within CACHE_TTL_MS
 * - In-flight dedup: concurrent callers for the same key share one request
 * - Concurrency limiter: at most MAX_CONCURRENT requests in parallel;
 *   the rest queue and fire as slots free up
 */

const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes
const MAX_CONCURRENT = 3;

// ── Cache & in-flight maps ──────────────────────────────────────

// All consumers cast the result to access extra fields (safetensors, tags),
// so we just expose ModelEntry and let them cast as needed.
type CachedResult = ModelEntry;

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
  return new Promise<void>((resolve) => waiting.push(() => { active++; resolve(); }));
}

function release() {
  active--;
  const next = waiting.shift();
  if (next) next();
}

// ── Public API ──────────────────────────────────────────────────

// Always request the superset of fields any consumer needs so a single
// cache entry covers all callers (e.g. ["safetensors"] and ["safetensors","tags"]).
const ALL_FIELDS: ("safetensors" | "tags")[] = ["safetensors", "tags"];

function cacheKey(name: string, authed: boolean): string {
  return `${name}::${authed ? "1" : "0"}`;
}

export async function cachedModelInfo(
  params: Parameters<typeof modelInfo>[0],
): Promise<CachedResult> {
  const authed = Boolean(params.credentials || ("accessToken" in params && params.accessToken));
  const key = cacheKey(params.name, authed);

  // 1. Return from cache if fresh
  const hit = cache.get(key);
  if (hit && Date.now() - hit.ts < CACHE_TTL_MS) return hit.data;

  // 2. Share in-flight request if one exists
  const flying = inflight.get(key);
  if (flying) return flying;

  // 3. New request, gated by concurrency semaphore
  const promise = (async () => {
    await acquire();
    try {
      const result = await modelInfo({ ...params, additionalFields: ALL_FIELDS });
      cache.set(key, { data: result as CachedResult, ts: Date.now() });
      return result as CachedResult;
    } finally {
      release();
      inflight.delete(key);
    }
  })();

  inflight.set(key, promise);
  return promise;
}
