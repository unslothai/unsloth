// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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

const datasetCache = new Map<string, DatasetSizeInfo | null>();
const datasetInflight = new Map<string, Promise<DatasetSizeInfo | null>>();

export function fetchDatasetSize(
  repoId: string,
): Promise<DatasetSizeInfo | null> {
  if (datasetCache.has(repoId)) {
    return Promise.resolve(datasetCache.get(repoId) ?? null);
  }
  const inflight = datasetInflight.get(repoId);
  if (inflight) return inflight;

  const promise = (async (): Promise<DatasetSizeInfo | null> => {
    try {
      const res = await fetch(
        `https://datasets-server.huggingface.co/size?dataset=${encodeURIComponent(repoId)}`,
      );
      if (!res.ok) {
        datasetCache.set(repoId, null);
        return null;
      }
      const data = (await res.json()) as DatasetSizeApiResponse;
      const ds = data.size?.dataset;
      if (!ds) {
        datasetCache.set(repoId, null);
        return null;
      }
      const info: DatasetSizeInfo = {
        numBytesOriginal: ds.num_bytes_original_files ?? null,
        numBytesParquet: ds.num_bytes_parquet_files ?? null,
        numRows: ds.num_rows ?? null,
        numSplits: Array.isArray(data.size?.splits)
          ? (data.size?.splits.length ?? null)
          : null,
      };
      datasetCache.set(repoId, info);
      return info;
    } catch {
      return null;
    } finally {
      datasetInflight.delete(repoId);
    }
  })();

  datasetInflight.set(repoId, promise);
  return promise;
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

const modelCache = new Map<string, ModelSizeInfo | null>();
const modelInflight = new Map<string, Promise<ModelSizeInfo | null>>();

export function fetchModelSize(
  repoId: string,
): Promise<ModelSizeInfo | null> {
  if (modelCache.has(repoId)) {
    return Promise.resolve(modelCache.get(repoId) ?? null);
  }
  const inflight = modelInflight.get(repoId);
  if (inflight) return inflight;

  const promise = (async (): Promise<ModelSizeInfo | null> => {
    try {
      const res = await fetch(
        `https://huggingface.co/api/models/${encodeURIComponent(repoId)}?blobs=true`,
      );
      if (!res.ok) {
        modelCache.set(repoId, null);
        return null;
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
      const info: ModelSizeInfo = {
        totalBytes: total > 0 ? total : null,
        weightsBytes: weights > 0 ? weights : null,
      };
      modelCache.set(repoId, info);
      return info;
    } catch {
      return null;
    } finally {
      modelInflight.delete(repoId);
    }
  })();

  modelInflight.set(repoId, promise);
  return promise;
}
