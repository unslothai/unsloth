// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  fetchCachedModelConfig,
  invalidateCachedModelConfig,
  modelConfigCacheKey,
  type ModelConfigFetchOptions,
  readCachedModelConfig,
} from "@/features/training/lib/model-config-fetch";
import type { ModelConfigResponse } from "@/features/training/api/models-api";

export interface ModelDefaults {
  maxPositionEmbeddings: number | null;
  chatTemplate: string | null;
}

export type ModelDefaultsFetchOptions = ModelConfigFetchOptions;

const EMPTY: ModelDefaults = { maxPositionEmbeddings: null, chatTemplate: null };

function defaultsFromConfig(details: ModelConfigResponse): ModelDefaults {
  return {
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
}

function hasAnyDefault(result: ModelDefaults): boolean {
  return result.maxPositionEmbeddings != null || result.chatTemplate != null;
}

function hasCompleteDefaults(result: ModelDefaults): boolean {
  return result.maxPositionEmbeddings != null && result.chatTemplate != null;
}

export function modelDefaultsRequestKey(
  modelId: string,
  token: string | null | undefined,
  options?: ModelDefaultsFetchOptions,
  inventoryVersion?: number,
): string {
  return modelConfigCacheKey(modelId, token, options, inventoryVersion);
}

export function fetchModelDefaults(
  modelId: string,
  signal: AbortSignal,
  options?: ModelDefaultsFetchOptions,
): Promise<ModelDefaults> {
  if (signal.aborted) {
    return Promise.reject(new DOMException("Aborted", "AbortError"));
  }
  if (!modelId) return Promise.resolve(EMPTY);
  return fetchCachedModelConfig(modelId, signal, options, {
    acceptCached: (details) => hasCompleteDefaults(defaultsFromConfig(details)),
    acceptErrorFallback: (details) => hasAnyDefault(defaultsFromConfig(details)),
  }).then(defaultsFromConfig);
}

export function readCachedModelDefaults(
  modelId: string,
  token?: string | null,
  options?: ModelDefaultsFetchOptions,
  inventoryVersion?: number,
): ModelDefaults | null {
  const details = readCachedModelConfig(modelId, token, options, inventoryVersion);
  return details ? defaultsFromConfig(details) : null;
}

export function invalidateModelDefaults(modelId?: string): void {
  invalidateCachedModelConfig(modelId);
}
