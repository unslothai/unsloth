// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { LruMap } from "@/lib/lru-map";

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

// A 404 means the repo genuinely has no size to report, so cache it for good.
// A 429/5xx/network hiccup or a stalled socket is transient: cache it briefly
// so one rate-limited burst doesn't pin the size to "unknown" until reload,
// mirroring the transient-vs-permanent handling in hf-owner-avatar.ts.
const TRANSIENT_MISS_TTL_MS = 60_000;
const FETCH_TIMEOUT_MS = 10_000;

type SizeCacheEntry<T> =
  | { kind: "value"; value: T }
  | { kind: "miss-permanent" }
  | { kind: "miss-transient"; until: number };

type LoadResult<T> = { value: T } | { miss: "permanent" | "transient" };

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
  repoId: string,
  cache: LruMap<string, SizeCacheEntry<T>>,
  inflight: Map<string, Promise<T | null>>,
  load: (signal: AbortSignal) => Promise<LoadResult<T>>,
): Promise<T | null> {
  const entry = readSizeCache(cache, repoId);
  if (entry) {
    return Promise.resolve(entry.kind === "value" ? entry.value : null);
  }
  const existing = inflight.get(repoId);
  if (existing) return existing;

  const promise = (async (): Promise<T | null> => {
    try {
      const result = await load(AbortSignal.timeout(FETCH_TIMEOUT_MS));
      if ("value" in result) {
        cache.set(repoId, { kind: "value", value: result.value });
        return result.value;
      }
      cache.set(
        repoId,
        result.miss === "permanent"
          ? { kind: "miss-permanent" }
          : { kind: "miss-transient", until: Date.now() + TRANSIENT_MISS_TTL_MS },
      );
      return null;
    } catch {
      cache.set(repoId, {
        kind: "miss-transient",
        until: Date.now() + TRANSIENT_MISS_TTL_MS,
      });
      return null;
    } finally {
      inflight.delete(repoId);
    }
  })();

  inflight.set(repoId, promise);
  return promise;
}

const datasetCache = new LruMap<string, SizeCacheEntry<DatasetSizeInfo>>(128);
const datasetInflight = new Map<string, Promise<DatasetSizeInfo | null>>();

export function fetchDatasetSize(
  repoId: string,
): Promise<DatasetSizeInfo | null> {
  return fetchCachedSize<DatasetSizeInfo>(
    repoId,
    datasetCache,
    datasetInflight,
    async (signal) => {
      const res = await fetch(
        `https://datasets-server.huggingface.co/size?dataset=${encodeURIComponent(repoId)}`,
        { signal },
      );
      if (!res.ok) {
        return { miss: res.status === 404 ? "permanent" : "transient" };
      }
      const data = (await res.json()) as DatasetSizeApiResponse;
      const ds = data.size?.dataset;
      if (!ds) return { miss: "permanent" };
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

const WEIGHT_FILE_RE = /\.(safetensors|bin|pt|pth|ckpt|onnx|gguf)$/i;

const modelCache = new LruMap<string, SizeCacheEntry<ModelSizeInfo>>(128);
const modelInflight = new Map<string, Promise<ModelSizeInfo | null>>();

export function fetchModelSize(repoId: string): Promise<ModelSizeInfo | null> {
  return fetchCachedSize<ModelSizeInfo>(repoId, modelCache, modelInflight, async (signal) => {
    const path = repoId.split("/").map(encodeURIComponent).join("/");
    const res = await fetch(
      `https://huggingface.co/api/models/${path}?blobs=true`,
      { signal },
    );
    if (!res.ok) {
      return { miss: res.status === 404 ? "permanent" : "transient" };
    }
    const data = (await res.json()) as ModelInfoApiResponse;
    const siblings = data.siblings ?? [];
    let total = 0;
    let weights = 0;
    for (const s of siblings) {
      if (typeof s.size !== "number") continue;
      total += s.size;
      if (s.rfilename && WEIGHT_FILE_RE.test(s.rfilename)) {
        weights += s.size;
      }
    }
    return {
      value: {
        totalBytes: total > 0 ? total : null,
        weightsBytes: weights > 0 ? weights : null,
      },
    };
  });
}
