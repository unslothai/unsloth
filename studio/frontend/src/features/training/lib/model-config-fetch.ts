// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { ModelInventoryFormat } from "@/features/inventory";
import { fingerprintToken } from "@/lib/token-fingerprint";
import { getHfToken } from "@/stores/hf-token-store";
import { getInventoryVersion } from "@/stores/inventory-events";
import {
  getModelConfig,
  type ModelConfigResponse,
} from "../api/models-api";

export interface ModelConfigFetchOptions {
  preferLocalCache?: boolean;
  localPath?: string | null;
  modelFormat?: ModelInventoryFormat | null;
}

type ModelConfigFetchPolicy = {
  acceptCached?: (details: ModelConfigResponse) => boolean;
  acceptErrorFallback?: (details: ModelConfigResponse) => boolean;
};

type NormalizedModelConfigFetchOptions = {
  preferLocalCache: boolean;
  localPath: string | null;
  modelFormat: ModelInventoryFormat | null;
};

type Inflight = {
  promise: Promise<ModelConfigResponse>;
  controller: AbortController;
  abortableCallers: number;
};

const CACHE_MAX = 128;
const cache = new Map<string, ModelConfigResponse>();
const inflight = new Map<string, Inflight>();

function normalizeOptions(
  options?: ModelConfigFetchOptions,
): NormalizedModelConfigFetchOptions {
  const localPath = options?.localPath?.trim() || null;
  return {
    preferLocalCache: Boolean(options?.preferLocalCache || localPath),
    localPath,
    modelFormat: options?.modelFormat ?? null,
  };
}

export function modelConfigSelectionKey(
  modelId: string,
  token: string | null | undefined = getHfToken(),
  options?: ModelConfigFetchOptions,
): string {
  const normalized = normalizeOptions(options);
  return `${modelId}\0${fingerprintToken(token)}\0${
    normalized.preferLocalCache ? "local" : "remote"
  }\0${normalized.localPath ?? ""}\0${normalized.modelFormat ?? ""}`;
}

export function modelConfigCacheKey(
  modelId: string,
  token: string | null | undefined = getHfToken(),
  options?: ModelConfigFetchOptions,
  inventoryVersion = getInventoryVersion(),
): string {
  return `${inventoryVersion}\0${modelConfigSelectionKey(modelId, token, options)}`;
}

function storeModelConfig(
  key: string,
  details: ModelConfigResponse,
): ModelConfigResponse {
  cache.set(key, details);
  while (cache.size > CACHE_MAX) {
    const oldest = cache.keys().next().value;
    if (oldest === undefined) break;
    cache.delete(oldest);
  }
  return details;
}

function isAbortError(error: unknown): boolean {
  return (error as { name?: string } | null)?.name === "AbortError";
}

function withCallerAbort(
  shared: Promise<ModelConfigResponse>,
  signal: AbortSignal,
  onDetach: () => void,
): Promise<ModelConfigResponse> {
  if (signal.aborted) {
    onDetach();
    return Promise.reject(new DOMException("Aborted", "AbortError"));
  }
  return new Promise<ModelConfigResponse>((resolve, reject) => {
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

export function fetchCachedModelConfig(
  modelId: string,
  signal: AbortSignal,
  options?: ModelConfigFetchOptions,
  policy: ModelConfigFetchPolicy = {},
): Promise<ModelConfigResponse> {
  if (signal.aborted) {
    return Promise.reject(new DOMException("Aborted", "AbortError"));
  }
  const token = getHfToken();
  const normalized = normalizeOptions(options);
  const key = modelConfigCacheKey(modelId, token, normalized);
  const acceptCached = policy.acceptCached ?? (() => true);
  const cached = cache.get(key);
  if (cached && acceptCached(cached)) {
    cache.delete(key);
    cache.set(key, cached);
    return Promise.resolve(cached);
  }

  let entry = inflight.get(key);
  if (entry?.controller.signal.aborted) {
    inflight.delete(key);
    entry = undefined;
  }
  if (!entry) {
    const controller = new AbortController();
    const nextEntry: Inflight = {
      controller,
      abortableCallers: 0,
    } as Inflight;
    const promise = (async (): Promise<ModelConfigResponse> => {
      try {
        const details = await getModelConfig(
          modelId,
          controller.signal,
          token || undefined,
          normalized,
        );
        return storeModelConfig(key, details);
      } catch (error) {
        const fallback = cache.get(key);
        if (
          !isAbortError(error) &&
          fallback &&
          policy.acceptErrorFallback?.(fallback)
        ) {
          return fallback;
        }
        throw error;
      } finally {
        if (inflight.get(key) === nextEntry) {
          inflight.delete(key);
        }
      }
    })();
    nextEntry.promise = promise;
    entry = nextEntry;
    inflight.set(key, entry);
  }

  const current = entry;
  current.abortableCallers += 1;
  const onDetach = () => {
    current.abortableCallers -= 1;
    if (
      current.abortableCallers <= 0 &&
      inflight.get(key) === current
    ) {
      current.controller.abort();
    }
  };
  return withCallerAbort(current.promise, signal, onDetach);
}

export function readCachedModelConfig(
  modelId: string,
  token: string | null | undefined = getHfToken(),
  options?: ModelConfigFetchOptions,
  inventoryVersion = getInventoryVersion(),
): ModelConfigResponse | null {
  return cache.get(modelConfigCacheKey(modelId, token, options, inventoryVersion)) ?? null;
}

export function invalidateCachedModelConfig(modelId?: string): void {
  if (!modelId) {
    cache.clear();
    return;
  }
  const marker = `\0${modelId}\0`.toLowerCase();
  for (const key of cache.keys()) {
    if (key.toLowerCase().includes(marker)) {
      cache.delete(key);
    }
  }
}
