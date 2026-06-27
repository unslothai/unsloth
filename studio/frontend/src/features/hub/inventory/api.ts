// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import {
  disposableTimeoutSignal,
  withAbort,
} from "@/features/hub/lib/abort-signals";
import { hubTokenHeader } from "@/features/hub/lib/hub-token-header";
import { readFastApiError } from "@/lib/format-fastapi-error";
import { localPathCacheKey } from "@/features/hub/lib/local-path";
import { isHuggingFaceOffline } from "@/features/hub/lib/network";
import { fingerprintToken } from "@/features/hub/lib/token-fingerprint";
import { bumpInventoryVersion } from "@/features/hub/stores/inventory-events";
import type { LocalSource } from "./constants";
import { bumpGgufVariantsCacheVersion } from "./gguf-variants-cache-events";

export type ModelInventoryFormat =
  | "gguf"
  | "safetensors"
  | "adapter"
  | "checkpoint"
  | "unknown";
export type ModelInventoryRuntime =
  | "llama_cpp"
  | "transformers"
  | "adapter"
  | "unknown";
export type BaseModelSource = "huggingface" | "local" | "unknown";

export interface BackendModelCapabilities {
  can_train?: boolean;
  can_chat?: boolean;
  can_delete?: boolean;
  can_download?: boolean;
  requires_variant?: boolean;
  supports_lora?: boolean;
  supports_vision?: boolean;
}

export interface CachedGgufRepo {
  repo_id: string;
  inventory_id?: string | null;
  load_id?: string | null;
  model_format?: ModelInventoryFormat | null;
  runtime?: ModelInventoryRuntime | null;
  format_variant?: string | null;
  capabilities?: BackendModelCapabilities | null;
  size_bytes: number;
  cache_path?: string;
  partial?: boolean;
  partial_transport?: string | null;
  pipeline_tag?: string | null;
  tags?: string[];
  library_name?: string | null;
}

export interface CachedModelRepo {
  repo_id: string;
  inventory_id?: string | null;
  load_id?: string | null;
  model_format?: ModelInventoryFormat | null;
  runtime?: ModelInventoryRuntime | null;
  format_variant?: string | null;
  capabilities?: BackendModelCapabilities | null;
  size_bytes: number;
  cache_path?: string;
  partial?: boolean;
  partial_transport?: string | null;
  pipeline_tag?: string | null;
  tags?: string[];
  library_name?: string | null;
  quant_method?: string | null;
}

export interface LocalModelInfo {
  id: string;
  inventory_id?: string | null;
  load_id?: string | null;
  display_name: string;
  path: string;
  size_bytes?: number;
  model_format?: ModelInventoryFormat | null;
  runtime?: ModelInventoryRuntime | null;
  format_variant?: string | null;
  capabilities?: BackendModelCapabilities | null;
  source: LocalSource;
  model_id?: string | null;
  base_model?: string | null;
  base_model_source?: BaseModelSource | null;
  adapter_type?: string | null;
  training_method?: string | null;
  updated_at?: number | null;
  partial?: boolean;
  partial_transport?: string | null;
  pipeline_tag?: string | null;
  tags?: string[];
  library_name?: string | null;
  quant_method?: string | null;
}

export interface LocalModelListResponse {
  models_dir: string;
  hf_cache_dir?: string | null;
  lmstudio_dirs: string[];
  ollama_dirs?: string[];
  models: LocalModelInfo[];
}

export interface CachedDatasetRepo {
  repo_id: string;
  size_bytes: number;
  cache_path?: string;
  partial?: boolean;
  partial_transport?: string | null;
}

export type LocalDatasetInfo = {
  metadata?: {
    actual_num_records?: number | null;
    target_num_records?: number | null;
    total_num_batches?: number | null;
    num_completed_batches?: number | null;
    columns?: string[] | null;
  } | null;
  id: string;
  label: string;
  path: string;
  source: "recipe" | "upload";
  rows?: number | null;
  updated_at?: number | null;
};

export type LocalDatasetsResponse = {
  datasets: LocalDatasetInfo[];
};

export interface ScanFolderInfo {
  id: number;
  path: string;
  created_at: string;
}

export interface BrowseEntry {
  name: string;
  has_models: boolean;
  hidden: boolean;
}

export interface BrowseFoldersResponse {
  current: string;
  parent: string | null;
  entries: BrowseEntry[];
  suggestions: string[];
  truncated?: boolean;
  model_files_here?: number;
}

export interface GgufVariantDetail {
  filename: string;
  quant: string;
  display_label?: string | null;
  size_bytes: number;
  download_size_bytes?: number;
  downloaded?: boolean;
  update_available?: boolean;
  partial?: boolean;
  partial_transport?: string | null;
}

export interface GgufVariantsResponse {
  repo_id: string;
  variants: GgufVariantDetail[];
  has_vision: boolean;
  default_variant: string | null;
}

async function parseJsonOrThrow<T>(
  response: Response,
  fallback?: string,
): Promise<T> {
  if (!response.ok) {
    throw new Error(await readFastApiError(response, fallback));
  }
  return response.json();
}

async function throwIfNotOk(
  response: Response,
  fallback?: string,
): Promise<void> {
  if (!response.ok) {
    throw new Error(await readFastApiError(response, fallback));
  }
}

const INVENTORY_TIMEOUT_MS = 30_000;

async function withHubTimeout<T>(
  ms: number,
  request: (signal: AbortSignal) => Promise<T>,
): Promise<T> {
  const timeout = disposableTimeoutSignal(ms);
  try {
    return await request(timeout.signal);
  } finally {
    timeout.dispose();
  }
}

export async function listLocalModels(): Promise<LocalModelListResponse> {
  const response = await withHubTimeout(INVENTORY_TIMEOUT_MS, (signal) =>
    authFetch("/api/hub/local", { signal }),
  );
  return parseJsonOrThrow<LocalModelListResponse>(response);
}

export async function listCachedGguf(
  hfToken?: string | null,
): Promise<CachedGgufRepo[]> {
  const response = await withHubTimeout(INVENTORY_TIMEOUT_MS, (signal) =>
    authFetch("/api/hub/cached-gguf", {
      headers: hubTokenHeader(hfToken),
      signal,
    }),
  );
  const data = await parseJsonOrThrow<{ cached: CachedGgufRepo[] }>(response);
  return data.cached;
}

export async function listCachedModels(
  hfToken?: string | null,
): Promise<CachedModelRepo[]> {
  const response = await withHubTimeout(INVENTORY_TIMEOUT_MS, (signal) =>
    authFetch("/api/hub/cached-models", {
      headers: hubTokenHeader(hfToken),
      signal,
    }),
  );
  const data = await parseJsonOrThrow<{ cached: CachedModelRepo[] }>(response);
  return data.cached;
}

export async function listLocalDatasets(): Promise<LocalDatasetsResponse> {
  const response = await withHubTimeout(INVENTORY_TIMEOUT_MS, (signal) =>
    authFetch("/api/hub/datasets/local", { signal }),
  );
  return parseJsonOrThrow<LocalDatasetsResponse>(response);
}

export async function listCachedDatasets(): Promise<CachedDatasetRepo[]> {
  const response = await withHubTimeout(INVENTORY_TIMEOUT_MS, (signal) =>
    authFetch("/api/hub/datasets/cached", { signal }),
  );
  const data = await parseJsonOrThrow<{ cached: CachedDatasetRepo[] }>(response);
  return data.cached;
}

export async function deleteCachedDataset(repoId: string): Promise<void> {
  const response = await authFetch("/api/hub/datasets/cached", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ repo_id: repoId }),
  });
  await throwIfNotOk(response, `Failed to delete dataset (${response.status})`);
  bumpInventoryVersion();
}

export async function deleteCachedModel(
  repoId: string,
  variant?: string,
  hfToken?: string | null,
): Promise<void> {
  const payload: Record<string, string> = { repo_id: repoId };
  if (variant) {
    payload.variant = variant;
  }
  const response = await authFetch("/api/hub/delete-cached", {
    method: "DELETE",
    headers: { "Content-Type": "application/json", ...hubTokenHeader(hfToken) },
    body: JSON.stringify(payload),
  });
  try {
    await throwIfNotOk(response);
    bumpInventoryVersion();
  } finally {
    invalidateGgufVariantsCache(repoId);
  }
}

export async function listScanFolders(): Promise<ScanFolderInfo[]> {
  const response = await authFetch("/api/hub/scan-folders");
  const data = await parseJsonOrThrow<{ folders: ScanFolderInfo[] }>(response);
  return data.folders;
}

export async function addScanFolder(path: string): Promise<ScanFolderInfo> {
  const response = await authFetch("/api/hub/scan-folders", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  const folder = await parseJsonOrThrow<ScanFolderInfo>(response);
  bumpInventoryVersion();
  return folder;
}

export async function removeScanFolder(id: number): Promise<void> {
  const response = await authFetch(`/api/hub/scan-folders/${id}`, {
    method: "DELETE",
  });
  await throwIfNotOk(response);
  bumpInventoryVersion();
}

export async function browseFolders(
  path?: string,
  showHidden = false,
  signal?: AbortSignal,
): Promise<BrowseFoldersResponse> {
  const params = new URLSearchParams();
  if (path !== undefined && path !== null) {
    params.set("path", path);
  }
  if (showHidden) {
    params.set("show_hidden", "true");
  }
  const qs = params.toString();
  const response = await authFetch(
    `/api/hub/browse-folders${qs ? `?${qs}` : ""}`,
    signal ? { signal } : undefined,
  );
  return parseJsonOrThrow<BrowseFoldersResponse>(response);
}

const GGUF_VARIANTS_TTL_MS = 30 * 1000;
const GGUF_VARIANTS_LOCAL_TTL_MS = 10 * 60 * 1000;
const GGUF_VARIANTS_TIMEOUT_MS = 30_000;
const GGUF_VARIANTS_CACHE_MAX_ENTRIES = 64;

interface GgufVariantsCacheEntry {
  expiresAt: number;
  promise: Promise<GgufVariantsResponse>;
}

const ggufVariantsCache = new Map<string, GgufVariantsCacheEntry>();

function pruneGgufVariantsCache(now = Date.now()): void {
  for (const [key, entry] of ggufVariantsCache) {
    if (entry.expiresAt <= now) {
      ggufVariantsCache.delete(key);
    }
  }
  while (ggufVariantsCache.size > GGUF_VARIANTS_CACHE_MAX_ENTRIES) {
    const oldest = ggufVariantsCache.keys().next().value;
    if (!oldest) {
      break;
    }
    ggufVariantsCache.delete(oldest);
  }
}

function ggufVariantsTtlMs(
  response: GgufVariantsResponse,
  preferLocalCache: boolean,
): number {
  if (
    preferLocalCache &&
    response.variants.some((variant) => variant.downloaded)
  ) {
    return GGUF_VARIANTS_LOCAL_TTL_MS;
  }
  return GGUF_VARIANTS_TTL_MS;
}

export async function listGgufVariants(
  repoId: string,
  hfToken?: string,
  options?: {
    preferLocalCache?: boolean;
    localPath?: string | null;
    signal?: AbortSignal;
  },
): Promise<GgufVariantsResponse> {
  const offline = isHuggingFaceOffline();
  const localPath = options?.localPath?.trim() || null;
  const preferLocalCache = !!options?.preferLocalCache || offline;
  const signal = options?.signal;
  const key = `${repoId}::${fingerprintToken(hfToken)}::${
    preferLocalCache ? "local" : "remote"
  }::${localPathCacheKey(localPath)}`;
  const now = Date.now();
  const hit = ggufVariantsCache.get(key);
  if (hit && now < hit.expiresAt) {
    ggufVariantsCache.delete(key);
    ggufVariantsCache.set(key, hit);
    return withAbort(hit.promise, signal);
  }
  if (hit) {
    ggufVariantsCache.delete(key);
  }
  const params = new URLSearchParams({ repo_id: repoId });
  if (preferLocalCache) {
    params.set("prefer_local_cache", "true");
  }
  if (localPath) {
    params.set("local_path", localPath);
  }
  if (offline) {
    params.set("offline", "true");
  }
  const requestPromise = withHubTimeout(
    GGUF_VARIANTS_TIMEOUT_MS,
    async (signal) => {
      const response = await authFetch(`/api/hub/gguf-variants?${params}`, {
        headers: hubTokenHeader(hfToken),
        signal,
      });
      return parseJsonOrThrow<GgufVariantsResponse>(response);
    },
  );
  const promise = requestPromise.then((result) => {
    const current = ggufVariantsCache.get(key);
    if (current?.promise === promise) {
      current.expiresAt =
        Date.now() + ggufVariantsTtlMs(result, preferLocalCache);
    }
    return result;
  });
  ggufVariantsCache.set(key, {
    expiresAt: now + GGUF_VARIANTS_TTL_MS,
    promise,
  });
  pruneGgufVariantsCache(now);
  promise.catch(() => {
    if (ggufVariantsCache.get(key)?.promise === promise) {
      ggufVariantsCache.delete(key);
    }
  });
  return withAbort(promise, signal);
}

export function invalidateGgufVariantsCache(repoId?: string): void {
  if (!repoId) {
    ggufVariantsCache.clear();
    bumpGgufVariantsCacheVersion();
    return;
  }
  const repoPrefix = `${repoId}::`.toLowerCase();
  for (const key of ggufVariantsCache.keys()) {
    if (key.toLowerCase().startsWith(repoPrefix)) {
      ggufVariantsCache.delete(key);
    }
  }
  bumpGgufVariantsCacheVersion(repoId);
}
