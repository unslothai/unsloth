// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getHfToken } from "@/stores/hf-token-store";
import { getModelConfig } from "@/features/training";

export interface ModelDefaults {
  maxPositionEmbeddings: number | null;
  chatTemplate: string | null;
}

interface Inflight {
  promise: Promise<ModelDefaults>;
  controller: AbortController;
  abortableCallers: number;
  keepAlive: boolean;
}

const cache = new Map<string, ModelDefaults>();
const inflight = new Map<string, Inflight>();
const CACHE_MAX = 128;

const EMPTY: ModelDefaults = { maxPositionEmbeddings: null, chatTemplate: null };

// chat_template and max_position_embeddings are static per (modelId, token)
// and don't change with on-disk inventory mutations, so the cache is intentionally
// not subscribed to inventory bumps. Call invalidateModelDefaults(modelId)
// from the precise mutation site if a model's defaults ever need to be re-read.

function fingerprintToken(token: string): string {
  if (!token) return "anon";
  let h1 = 0x811c9dc5;
  let h2 = 0;
  for (let i = 0; i < token.length; i++) {
    const c = token.charCodeAt(i);
    h1 = Math.imul(h1 ^ c, 0x01000193);
    h2 = (Math.imul(h2, 33) + c) | 0;
  }
  return (
    (h1 >>> 0).toString(16).padStart(8, "0") +
    (h2 >>> 0).toString(16).padStart(8, "0")
  );
}

function cacheKey(modelId: string, token: string): string {
  return `${modelId}::${fingerprintToken(token)}`;
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
  onDetach: () => void,
): Promise<ModelDefaults> {
  if (signal.aborted) {
    onDetach();
    return Promise.reject(new DOMException("Aborted", "AbortError"));
  }
  return new Promise<ModelDefaults>((resolve, reject) => {
    let detached = false;
    const detach = () => {
      if (detached) return;
      detached = true;
      onDetach();
    };
    const onAbort = () => {
      detach();
      reject(new DOMException("Aborted", "AbortError"));
    };
    signal.addEventListener("abort", onAbort, { once: true });
    shared.then(
      (value) => {
        signal.removeEventListener("abort", onAbort);
        detach();
        resolve(value);
      },
      (error) => {
        signal.removeEventListener("abort", onAbort);
        detach();
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

  let entry = inflight.get(key);
  if (!entry) {
    const controller = new AbortController();
    const promise = (async (): Promise<ModelDefaults> => {
      try {
        const details = await getModelConfig(
          modelId,
          controller.signal,
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
    entry = { promise, controller, abortableCallers: 0, keepAlive: false };
    inflight.set(key, entry);
  }

  // A signal-less caller pins the shared request alive; abortable callers are
  // refcounted and the fetch is aborted only once the last one leaves.
  if (!signal) {
    entry.keepAlive = true;
    return entry.promise;
  }

  const current = entry;
  current.abortableCallers += 1;
  const onDetach = () => {
    current.abortableCallers -= 1;
    if (
      current.abortableCallers <= 0 &&
      !current.keepAlive &&
      inflight.get(key) === current
    ) {
      current.controller.abort();
    }
  };
  return withCallerAbort(current.promise, signal, onDetach);
}

export function readCachedModelDefaults(
  modelId: string,
  token: string = getHfToken(),
): ModelDefaults | null {
  return cache.get(cacheKey(modelId, token)) ?? null;
}
