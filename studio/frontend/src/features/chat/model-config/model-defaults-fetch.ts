// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { subscribeInventory } from "@/features/models/inventory-events";
import { getHfToken, useHfTokenStore } from "@/stores/hf-token-store";
import { getModelConfig } from "@/features/training/api/models-api";

export interface ModelDefaults {
  maxPositionEmbeddings: number | null;
  chatTemplate: string | null;
}

const cache = new Map<string, ModelDefaults>();
const inflight = new Map<string, Promise<ModelDefaults>>();

const EMPTY: ModelDefaults = { maxPositionEmbeddings: null, chatTemplate: null };

subscribeInventory(() => {
  cache.clear();
});

useHfTokenStore.subscribe((state, prev) => {
  if (state.token !== prev.token) cache.clear();
});

export function invalidateModelDefaults(modelId?: string): void {
  if (!modelId) {
    cache.clear();
    return;
  }
  cache.delete(modelId);
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
  const cached = cache.get(modelId);
  if (cached) return Promise.resolve(cached);

  let shared = inflight.get(modelId);
  if (!shared) {
    shared = (async (): Promise<ModelDefaults> => {
      try {
        const details = await getModelConfig(
          modelId,
          undefined,
          getHfToken() || undefined,
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
        cache.set(modelId, result);
        return result;
      } finally {
        inflight.delete(modelId);
      }
    })();
    inflight.set(modelId, shared);
  }

  return signal ? withCallerAbort(shared, signal) : shared;
}

export function readCachedModelDefaults(modelId: string): ModelDefaults | null {
  return cache.get(modelId) ?? null;
}
