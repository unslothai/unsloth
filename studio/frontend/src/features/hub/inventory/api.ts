


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
import type { ScanFolderStatus } from "../lib/scan-folder-status";
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
  last_modified?: number | null;
  partial?: boolean;
  partial_transport?: string | null;
  /** This partial can be continued byte for byte. */
  partial_resumable?: boolean;
  pipeline_tag?: string | null;
  task?: string | null;
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
  last_modified?: number | null;
  partial?: boolean;
  partial_transport?: string | null;
  /** This partial can be continued byte for byte. */
  partial_resumable?: boolean;
  pipeline_tag?: string | null;
  task?: string | null;
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
  active_cache?: boolean | null;
  base_model?: string | null;
  base_model_source?: BaseModelSource | null;
  adapter_type?: string | null;
  training_method?: string | null;
  updated_at?: number | null;
  partial?: boolean;
  partial_transport?: string | null;
  /** This partial can be continued byte for byte. */
  partial_resumable?: boolean;
  pipeline_tag?: string | null;
  task?: string | null;
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
  load_cache_path?: string;
  partial?: boolean;
  partial_transport?: string | null;
  /** This partial can be continued byte for byte. */
  partial_resumable?: boolean;
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
  /** Result of the last scan. Absent on older backends, which means "ok". */
  status?: ScanFolderStatus;
}

export interface GgufVariantDetail {
  filename: string;
  quant: string;
  display_label?: string | null;
  size_bytes: number;
  download_size_bytes?: number;
  /** Bytes a resume still has to fetch. Set only on a partial variant. */
  download_remaining_bytes?: number | null;
  downloaded?: boolean;
  update_available?: boolean;
  partial?: boolean;
  partial_transport?: string | null;
  /** This partial can be continued byte for byte. */
  partial_resumable?: boolean;
  /** Variants sharing this key share one companion download footprint, so a
   *  footprint resolved for one of them is correct for all of them. */
  dependency_key?: string | null;
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

export async function deleteCachedDataset(
  repoId: string,
  cachePath?: string | null,
): Promise<void> {
  const payload: Record<string, string> = { repo_id: repoId };
  if (cachePath) {
    payload.cache_path = cachePath;
  }
  const response = await authFetch("/api/hub/datasets/cached", {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  await throwIfNotOk(response, `Failed to delete dataset (${response.status})`);
  bumpInventoryVersion();
}

export interface CompanionAssetInfo {
  repo_id: string;
  size_bytes: number;
  needed_by: string[];
}

export interface DeleteImpact {
  repo_id: string;
  variant?: string | null;
  reclaimed_bytes: number;
  retained_companions: CompanionAssetInfo[];
  freeable_companions: CompanionAssetInfo[];
  blocked_by: string[];
}

/** What a delete would actually reclaim and leave behind. Never throws: the confirm dialog
 * still has to open if this preview is unavailable, it just falls back to the plain wording. */
export async function fetchDeleteImpact(
  repoId: string,
  variant?: string | null,
): Promise<DeleteImpact | null> {
  try {
    const response = await authFetch("/api/hub/delete-impact", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(
        variant ? { repo_id: repoId, variant } : { repo_id: repoId },
      ),
    });
    if (!response.ok) return null;
    return (await response.json()) as DeleteImpact;
  } catch {
    return null;
  }
}

export interface OrphanCompanion {
  repo_id: string;
  size_bytes: number;
  cache_path?: string | null;
}

export async function fetchOrphanCompanions(): Promise<{
  companions: OrphanCompanion[];
  total_bytes: number;
}> {
  const response = await authFetch("/api/hub/orphan-companions");
  return await parseJsonOrThrow<{
    companions: OrphanCompanion[];
    total_bytes: number;
  }>(response);
}

export async function deleteCachedModel(
  repoId: string,
  variant?: string,
  hfToken?: string | null,
  cachePath?: string | null,
  onlyIfOrphan?: boolean,
): Promise<void> {
  const payload: Record<string, string | boolean> = { repo_id: repoId };
  if (variant) {
    payload.variant = variant;
  }
  // Scope the delete to this row's cache so copies in other caches survive.
  if (cachePath) {
    payload.cache_path = cachePath;
  }
  // Free up space acts on a list that can be minutes old. The server re-derives the orphan
  // condition just before unlinking and 409s if a download turned the row into a real model.
  if (onlyIfOrphan) {
    payload.only_if_orphan = true;
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
