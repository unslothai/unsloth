// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ModelEntry, modelInfo } from "@huggingface/hub";

import { getHfEndpoint } from "@/lib/hf-endpoint";
import { LruMap } from "./lru-map";
import { fingerprintToken } from "./token-fingerprint";

const CACHE_TTL_MS = 5 * 60 * 1000;
const CACHE_MAX_ENTRIES = 256;
const MAX_CONCURRENT = 3;
type CachedAdditionalField =
  | "safetensors"
  | "tags"
  | "library_name"
  | "config"
  | "createdAt"
  | "downloadsAllTime";

const DEFAULT_FIELDS: CachedAdditionalField[] = [
  "safetensors",
  "tags",
  "library_name",
  "config",
  "createdAt",
  "downloadsAllTime",
  // The API and SDK support gguf; the SDK's expandable-field type omits it.
  "gguf" as CachedAdditionalField,
];

export type CachedResult = ModelEntry & {
  safetensors?: { total?: number; parameters?: Record<string, number> };
  gguf?: { total?: number; architecture?: string };
  tags?: string[];
  library_name?: string;
  config?: { quantization_config?: { quant_method?: string } };
  createdAt?: string | Date;
  downloadsAllTime?: number;
};

interface CacheEntry {
  data: CachedResult;
  ts: number;
}

const cache = new LruMap<string, CacheEntry>(CACHE_MAX_ENTRIES);
const inflight = new Map<string, Promise<CachedResult>>();
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
  if (next) next();
}

function isStale(key: string): boolean {
  const hit = cache.get(key);
  return !hit || Date.now() - hit.ts >= CACHE_TTL_MS;
}

// Endpoint in the key: the same repo id names a different repo on a mirror, so a repo-and-token key would serve the old host's answer for the rest of the TTL.
function cacheKey(
  name: string,
  token: string | undefined,
  hubUrl?: string,
): string {
  return `${hubUrl || getHfEndpoint()}::${name}::${fingerprintToken(token)}`;
}

function extractToken(
  params: Parameters<typeof modelInfo>[0],
): string | undefined {
  if (params.accessToken) return params.accessToken;
  if (params.credentials && "accessToken" in params.credentials) {
    return params.credentials.accessToken;
  }
  return undefined;
}

export function primeCacheFromListing(
  name: string,
  token: string | undefined,
  data: CachedResult,
): void {
  if (!name) return;
  const key = cacheKey(name, token);
  if (!isStale(key)) return;
  cache.set(key, { data, ts: Date.now() });
}

export async function cachedModelInfo(
  params: Parameters<typeof modelInfo>[0],
): Promise<CachedResult> {
  const token = extractToken(params);
  const key = cacheKey(params.name, token, params.hubUrl);
  if (!isStale(key)) return cache.get(key)!.data;

  const flying = inflight.get(key);
  if (flying) return flying;

  const promise = (async () => {
    await acquire();
    try {
      const additionalFields = Array.from(
        new Set([
          ...DEFAULT_FIELDS,
          ...((params.additionalFields ?? []) as CachedAdditionalField[]),
        ]),
      );
      const result = await modelInfo({
        ...params,
        additionalFields,
      });
      const entry = { data: result as CachedResult, ts: Date.now() };
      cache.set(key, entry);
      const typed = result as CachedResult & {
        gated?: false | "auto" | "manual";
        private?: boolean;
      };
      if (token && !typed.private && !typed.gated) {
        const anonKey = cacheKey(params.name, undefined, params.hubUrl);
        if (isStale(anonKey)) cache.set(anonKey, entry);
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
