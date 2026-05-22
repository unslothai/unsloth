// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { subscribeInventory } from "@/features/models/inventory-events";
import { getHfToken } from "@/stores/hf-token-store";
import { getModelConfig } from "@/features/training/api/models-api";

export interface ModelDefaults {
  maxPositionEmbeddings: number | null;
  chatTemplate: string | null;
}

const cache = new Map<string, ModelDefaults>();
const inflight = new Map<string, Promise<ModelDefaults>>();
const CACHE_MAX = 128;

const EMPTY: ModelDefaults = { maxPositionEmbeddings: null, chatTemplate: null };

subscribeInventory(() => {
  cache.clear();
});

function cacheKey(modelId: string, token: string): string {
  return `${modelId}::${token}`;
}

function storeDefaults(key: string, result: ModelDefaults): void {
  cache.set(key, result);
  while (cache.size > CACHE_MAX) {
    const oldest = cache.keys().next().value;
    if (oldest === undefined) break;
    cache.delete(oldest);
  }
}

export function invalidateModelDefaults(modelId?: string): void {
  if (!modelId) {
    cache.clear();
    return;
  }
  const prefix = `${modelId}::`;
  for (const key of cache.keys()) {
    if (key.startsWith(prefix)) cache.delete(key);
  }
}

function withCallerAbort(
  shared: Promise<ModelDefaults>,
  signal: AbortSignal,
): Promise<ModelDefaults> {
  if (signal.aborted) {
    return Promise.reject(new DOMException("Aborted", "AbortError"));
  }
  return new Promise<ModelDefaults>((resolve, reject) => {
    const onAbort = () => reject(new DOMException("Aborted", "AbortError"));
    signal.addEventListener("abort", onAbort, { once: true });
    shared.then(
      (value) => {
        signal.removeEventListener("abort", onAbort);
        resolve(value);
      },
      (error) => {
        signal.removeEventListener("abort", onAbort);
        reject(error);
      },
    );
  });
}

export function fetchModelDefaults(
  modelId: string,
  signal?: AbortSignal,
): Promise<ModelDefaults> {
  if (!modelId) return Promise.resolve(EMPTY);
  const token = getHfToken();
  const key = cacheKey(modelId, token);
  const cached = cache.get(key);
  if (cached) {
    cache.delete(key);
    cache.set(key, cached);
    return Promise.resolve(cached);
  }

  let shared = inflight.get(key);
  if (!shared) {
    shared = (async (): Promise<ModelDefaults> => {
      try {
        const details = await getModelConfig(
          modelId,
          undefined,
          token || undefined,
        );
        const result: ModelDefaults = {
          maxPositionEmbeddings:
            typeof details.max_position_embeddings === "number" &&
            Number.isFinite(details.max_position_embeddings)
              ? details.max_position_embeddings
              : null,
          chatTemplate:
            typeof details.chat_template === "string" && details.chat_template
              ? details.chat_template
              : null,
        };
        storeDefaults(key, result);
        return result;
      } finally {
        inflight.delete(key);
      }
    })();
    inflight.set(key, shared);
  }

  return signal ? withCallerAbort(shared, signal) : shared;
}

export function readCachedModelDefaults(
  modelId: string,
  token: string = getHfToken(),
): ModelDefaults | null {
  return cache.get(cacheKey(modelId, token)) ?? null;
}
