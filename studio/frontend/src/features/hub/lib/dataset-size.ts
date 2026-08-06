// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { LruMap } from "./lru-map";
import { fetchWithTimeout } from "./network";
import { fingerprintToken } from "./token-fingerprint";

export interface DatasetSizeInfo {
  numBytesOriginal: number | null;
  numBytesParquet: number | null;
  numRows: number | null;
  numSplits: number | null;
}

interface DatasetSizeApiResponse {
  size?: {
    dataset?: {
      num_bytes_original_files?: number;
      num_bytes_parquet_files?: number;
      num_rows?: number;
    };
    splits?: unknown[];
  };
}

// Transient (429/5xx/network) misses cache briefly so a rate-limited burst
// doesn't pin the size to "unknown" until reload. "longlived" covers stable-ish
// answers like datasets-server 404 ("not processed yet"); "permanent" is a
// genuinely missing repo (HF models API 404).
const TRANSIENT_MISS_TTL_MS = 60_000;
const LONGLIVED_MISS_TTL_MS = 24 * 60 * 60 * 1000;
const FETCH_TIMEOUT_MS = 10_000;

type SizeCacheEntry<T> =
  | { kind: "value"; value: T }
  | { kind: "miss-permanent" }
  | { kind: "miss-transient"; until: number };

type LoadResult<T> =
  | { value: T }
  | { miss: "permanent" | "transient" | "longlived" };

type InflightSizeEntry<T> = {
  promise: Promise<T | null>;
  controller: AbortController;
  consumers: number;
  settled: boolean;
  cancelledByConsumers: boolean;
};

function readSizeCache<T>(
  cache: LruMap<string, SizeCacheEntry<T>>,
  key: string,
): SizeCacheEntry<T> | null {
  const entry = cache.get(key);
  if (!entry) return null;
  if (entry.kind === "miss-transient" && Date.now() >= entry.until) {
    cache.delete(key);
    return null;
  }
  return entry;
}

function fetchCachedSize<T>(
  cacheKey: string,
  cache: LruMap<string, SizeCacheEntry<T>>,
  inflight: Map<string, InflightSizeEntry<T>>,
  load: (signal: AbortSignal) => Promise<LoadResult<T>>,
  debugContext: { label: string; id: string },
  signal?: AbortSignal,
): Promise<T | null> {
  if (signal?.aborted) {
    return Promise.resolve(null);
  }
  const entry = readSizeCache(cache, cacheKey);
  if (entry) {
    return Promise.resolve(entry.kind === "value" ? entry.value : null);
  }
  const existing = inflight.get(cacheKey);
  if (existing) return attachInflight(existing, signal);

  let timedOut = false;
  const controller = new AbortController();
  const inflightEntry: InflightSizeEntry<T> = {
    promise: Promise.resolve(null),
    controller,
    consumers: 0,
    settled: false,
    cancelledByConsumers: false,
  };
  const timeout = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, FETCH_TIMEOUT_MS);

  inflightEntry.promise = (async (): Promise<T | null> => {
    try {
      const result = await load(controller.signal);
      if ("value" in result) {
        cache.set(cacheKey, { kind: "value", value: result.value });
        return result.value;
      }
      if (result.miss === "permanent") {
        cache.set(cacheKey, { kind: "miss-permanent" });
      } else {
        const ttl =
          result.miss === "longlived"
            ? LONGLIVED_MISS_TTL_MS
            : TRANSIENT_MISS_TTL_MS;
        cache.set(cacheKey, {
          kind: "miss-transient",
          until: Date.now() + ttl,
        });
      }
      return null;
    } catch (err) {
      if (
        import.meta.env.DEV &&
        (!inflightEntry.cancelledByConsumers || timedOut)
      ) {
        console.debug(`${debugContext.label} size lookup failed`, {
          id: debugContext.id,
          error: err,
        });
      }
      if (!inflightEntry.cancelledByConsumers || timedOut) {
        cache.set(cacheKey, {
          kind: "miss-transient",
          until: Date.now() + TRANSIENT_MISS_TTL_MS,
        });
      }
      return null;
    } finally {
      clearTimeout(timeout);
      inflightEntry.settled = true;
      inflight.delete(cacheKey);
    }
  })();

  inflight.set(cacheKey, inflightEntry);
  return attachInflight(inflightEntry, signal);
}

function attachInflight<T>(
  entry: InflightSizeEntry<T>,
  signal?: AbortSignal,
): Promise<T | null> {
  entry.consumers += 1;
  let released = false;
  const release = () => {
    if (released) return;
    released = true;
    entry.consumers = Math.max(0, entry.consumers - 1);
    if (entry.consumers === 0 && !entry.settled) {
      entry.cancelledByConsumers = true;
      entry.controller.abort();
    }
  };

  if (!signal) {
    return entry.promise.finally(release);
  }

  if (signal.aborted) {
    release();
    return Promise.resolve(null);
  }

  const activeSignal = signal;
  return new Promise((resolve) => {
    function cleanup() {
      activeSignal.removeEventListener("abort", onAbort);
    }
    function onAbort() {
      cleanup();
      release();
      resolve(null);
    }
    activeSignal.addEventListener("abort", onAbort, { once: true });
    entry.promise.then(
      (value) => {
        cleanup();
        release();
        resolve(value);
      },
      () => {
        cleanup();
        release();
        resolve(null);
      },
    );
  });
}

const datasetCache = new LruMap<string, SizeCacheEntry<DatasetSizeInfo>>(128);
const datasetInflight = new Map<string, InflightSizeEntry<DatasetSizeInfo>>();

export function fetchDatasetSize(
  repoId: string,
  signal?: AbortSignal,
): Promise<DatasetSizeInfo | null>;
export function fetchDatasetSize(
  repoId: string,
  token?: string,
  signal?: AbortSignal,
): Promise<DatasetSizeInfo | null>;
export function fetchDatasetSize(
  repoId: string,
  tokenOrSignal?: string | AbortSignal,
  signal?: AbortSignal,
): Promise<DatasetSizeInfo | null> {
  const resolvedToken =
    typeof tokenOrSignal === "string" ? tokenOrSignal : undefined;
  const resolvedSignal =
    signal ?? (typeof tokenOrSignal === "string" ? undefined : tokenOrSignal);
  const cacheKey = `${repoId}::${fingerprintToken(resolvedToken)}`;
  return fetchCachedSize<DatasetSizeInfo>(
    cacheKey,
    datasetCache,
    datasetInflight,
    async (signal) => {
      const res = await fetchWithTimeout(
        `https://datasets-server.huggingface.co/size?dataset=${encodeURIComponent(repoId)}`,
        {
          signal,
          headers: resolvedToken
            ? { Authorization: `Bearer ${resolvedToken}` }
            : undefined,
        },
        FETCH_TIMEOUT_MS,
      );
      if (!res.ok) {
        // datasets-server 404 often means "not processed yet", not "never".
        return { miss: res.status === 404 ? "longlived" : "transient" };
      }
      const data = (await res.json()) as DatasetSizeApiResponse;
      const ds = data.size?.dataset;
      if (!ds) return { miss: "longlived" };
      return {
        value: {
          numBytesOriginal: ds.num_bytes_original_files ?? null,
          numBytesParquet: ds.num_bytes_parquet_files ?? null,
          numRows: ds.num_rows ?? null,
          numSplits: Array.isArray(data.size?.splits)
            ? (data.size?.splits.length ?? null)
            : null,
        },
      };
    },
    { label: "Dataset", id: repoId },
    resolvedSignal,
  );
}

export interface ModelSizeInfo {
  totalBytes: number | null;
  weightsBytes: number | null;
}

interface ModelInfoApiResponse {
  siblings?: Array<{
    rfilename?: string;
    size?: number;
  }>;
}

type ModelSibling = NonNullable<ModelInfoApiResponse["siblings"]>[number];

const SNAPSHOT_WEIGHT_FILE_RE =
  /\.(safetensors|bin|pt|pth|ckpt|h5|msgpack|npz)$/i;
const SNAPSHOT_NON_BIN_WEIGHT_FILE_RE =
  /\.(safetensors|pt|pth|ckpt|h5|msgpack|npz)$/i;
const SNAPSHOT_BIN_WEIGHT_PREFIX_RE = /^(model|pytorch_model|adapter_model).*\.bin$/i;

function basename(path: string): string {
  return path.split("/").pop() ?? path;
}

function shipsTransformersWeights(siblings: ModelSibling[]): boolean {
  return siblings.some((s) => {
    const base = basename(s.rfilename ?? "").toLowerCase();
    if (base.startsWith("consolidated")) return false;
    return (
      SNAPSHOT_NON_BIN_WEIGHT_FILE_RE.test(base) ||
      SNAPSHOT_BIN_WEIGHT_PREFIX_RE.test(base)
    );
  });
}

function isSnapshotIgnored(
  filename: string,
  skipConsolidated: boolean,
): boolean {
  const lower = filename.toLowerCase();
  return (
    lower.endsWith(".gguf") ||
    lower.endsWith(".onnx") ||
    lower.startsWith("onnx/") ||
    lower.startsWith("openvino/") ||
    lower.startsWith("mlx/") ||
    lower.endsWith(".bin.index.json.bak") ||
    (skipConsolidated && lower.startsWith("consolidated"))
  );
}

const modelCache = new LruMap<string, SizeCacheEntry<ModelSizeInfo>>(128);
const modelInflight = new Map<string, InflightSizeEntry<ModelSizeInfo>>();

export function fetchModelSize(
  repoId: string,
  token?: string,
  signal?: AbortSignal,
): Promise<ModelSizeInfo | null> {
  const cacheKey = `${repoId}::${fingerprintToken(token)}`;
  return fetchCachedSize<ModelSizeInfo>(
    cacheKey,
    modelCache,
    modelInflight,
    async (signal) => {
      const path = repoId.split("/").map(encodeURIComponent).join("/");
      const res = await fetchWithTimeout(
        `https://huggingface.co/api/models/${path}?blobs=true`,
        {
          signal,
          headers: token ? { Authorization: `Bearer ${token}` } : undefined,
        },
        FETCH_TIMEOUT_MS,
      );
      if (!res.ok) {
        return { miss: res.status === 404 ? "permanent" : "transient" };
      }
      const data = (await res.json()) as ModelInfoApiResponse;
      const siblings = data.siblings ?? [];
      const skipConsolidated = shipsTransformersWeights(siblings);
      let total = 0;
      let weights = 0;
      for (const s of siblings) {
        if (typeof s.size !== "number") continue;
        const filename = s.rfilename ?? "";
        if (filename && isSnapshotIgnored(filename, skipConsolidated)) continue;
        total += s.size;
        if (filename && SNAPSHOT_WEIGHT_FILE_RE.test(filename)) {
          weights += s.size;
        }
      }
      return {
        value: {
          totalBytes: total > 0 ? total : null,
          weightsBytes: weights > 0 ? weights : null,
        },
      };
    },
    { label: "Model", id: repoId },
    signal,
  );
}
