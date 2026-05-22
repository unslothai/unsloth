// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ModelEntry, modelInfo } from "@huggingface/hub";
import { LruMap } from "@/lib/lru-map";

/**
 * Thin caching + throttling layer over `modelInfo()` from @huggingface/hub.
 *
 * - TTL cache: avoids re-fetching the same model within CACHE_TTL_MS
 * - In-flight dedup: concurrent callers for the same key share one request
 * - Concurrency limiter: at most MAX_CONCURRENT requests in parallel;
 *   the rest queue and fire as slots free up
 */

const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes
// A failed lookup (404, 401/403, rate limit, network) is cached briefly so a
// failing or throttled repo isn't re-requested by every caller and re-render.
// Short enough to recover quickly once the cause clears.
const NEGATIVE_CACHE_TTL_MS = 30 * 1000;
// HF API allows bursts but rate-limits sustained traffic. 6 parallel requests
// keeps clicking through several models snappy while staying polite (matches the
// avatar limiter); higher would risk the sustained-traffic throttle.
const MAX_CONCURRENT = 6;

// ── Cache & in-flight maps ──────────────────────────────────────

// Extend ModelEntry with the additional fields we always request so callers
// do not need unsafe casts to access safetensors/tags.
export interface HfQuantizationConfig {
  quant_method?: string;
  bits?: number;
}

export interface HfConfig {
  architectures?: string[];
  model_type?: string;
  quantization_config?: HfQuantizationConfig;
}

export type CachedResult = ModelEntry & {
  safetensors?: { total?: number; parameters?: Record<string, number> };
  gguf?: { total?: number; architecture?: string };
  tags?: string[];
  config?: HfConfig;
};


interface CacheEntry {
  // `null` data with an `error` marks a negative-cache entry: a recent failed
  // lookup we replay (by re-throwing) instead of re-fetching until it expires.
  data: CachedResult | null;
  ts: number;
  ttl: number;
  error?: unknown;
}

// Bounded so a long browsing session can't grow the cache without limit.
const MAX_CACHE_ENTRIES = 512;
const cache = new LruMap<string, CacheEntry>(MAX_CACHE_ENTRIES);
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
// cache entry covers all callers. "gguf"/"config" are real HF API fields the
// SDK's expand whitelist omits; the SDK's pick() resolves them at runtime.
// `satisfies` validates the member names, then the cast adapts to the SDK type.
type HfExpandField = "safetensors" | "tags" | "gguf" | "config";
export const ALL_FIELDS = [
  "safetensors",
  "tags",
  "gguf",
  "config",
] satisfies HfExpandField[] as ("safetensors" | "tags")[];

function isStale(key: string): boolean {
  const hit = cache.get(key);
  if (!hit) return true;
  return Date.now() - hit.ts >= hit.ttl;
}

function cacheKey(name: string, token: string | undefined): string {
  if (!token) {
    return `${name}::anon`;
  }
  // Scope by the full token so two tokens never share a cache slot — a
  // truncated fingerprint could collide and cross-serve gated/private
  // metadata between tokens. The token already lives in memory (auth store),
  // so this adds no exposure, and the keys are never logged or persisted.
  return `${name}::${token}`;
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
 * Never overwrites a fresh modelInfo response, but does replace a fresh
 * negative-cache entry -- real listing data beats a cached failure.
 */
export function primeCacheFromListing(
  name: string,
  token: string | undefined,
  data: CachedResult,
): void {
  if (!name) return;
  const key = cacheKey(name, token);
  const hit = cache.get(key);
  if (hit && hit.error === undefined && Date.now() - hit.ts < hit.ttl) return;
  cache.set(key, { data, ts: Date.now(), ttl: CACHE_TTL_MS });
}

export async function cachedModelInfo(
  params: Parameters<typeof modelInfo>[0],
): Promise<CachedResult> {
  const token = extractToken(params);
  const key = cacheKey(params.name, token);

  // 1. Return from cache if fresh -- replaying a cached failure as a throw.
  const hit = cache.get(key);
  if (hit && Date.now() - hit.ts < hit.ttl) {
    if (hit.error !== undefined) throw hit.error;
    if (hit.data) return hit.data;
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
      const entry: CacheEntry = {
        data: result as CachedResult,
        ts: Date.now(),
        ttl: CACHE_TTL_MS,
      };
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
    } catch (err) {
      // Negative-cache the failure so concurrent and follow-up callers don't
      // re-hammer a failing or rate-limited repo until the entry expires.
      cache.set(key, {
        data: null,
        ts: Date.now(),
        ttl: NEGATIVE_CACHE_TTL_MS,
        error: err,
      });
      throw err;
    } finally {
      release();
      inflight.delete(key);
    }
  })();

  inflight.set(key, promise);
  return promise;
}
