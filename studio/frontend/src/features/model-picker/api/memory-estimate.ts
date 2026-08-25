// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { consumeNativePathToken } from "@/features/native-intents/api";

/** Why no breakdown came back. The panel maps these to its own copy. */
export type MemoryEstimateReason =
  | "not_gguf"
  | "not_downloaded"
  | "unsupported_source"
  | "unsizable";

export interface MemoryEstimate {
  available: boolean;
  reason: MemoryEstimateReason | null;
  /** Files that become resident: weights, projector, drafter. */
  weightsBytes: number;
  /** KV cache at the priced context and slot count. Meaningless unless `kvEstimable`. */
  kvBytes: number;
  /** Compute / graph buffers, flat plus the context-linear growth. */
  computeBytes: number;
  /** A separate drafter's own cache and rollback state, on top of its file. */
  drafterRuntimeBytes: number;
  /** The share of the above that lands on the GPU. A figure rather than a flag: under
   *  MTP the target-side verification state follows the TARGET cache and the draft
   *  cache follows the drafter, so the two halves can be placed differently. */
  drafterRuntimeGpuBytes: number;
  /** The vision encoder's buffers, about 0.4x the projector file on top of it. */
  projectorRuntimeBytes: number;
  /** A charged drafter whose cache could not be sized: `--spec-draft-hf` names a
   *  repository, so its header is not on this disk. The total is a floor. */
  drafterKvUnsized: boolean;
  /** Weights + KV + compute, wherever they land (VRAM, host RAM, or one unified pool). */
  totalBytes: number;
  /** The share of `totalBytes` that lands on the GPU under the requested offload. */
  gpuBytes: number;
  /**
   * False when the GGUF header lacks the attention dims needed to size the cache. The
   * numbers above are then a lower bound, not an estimate: at a long context the cache
   * is most of the footprint.
   */
  kvEstimable: boolean;
  /** False under `--no-kv-offload`, which moves the cache to host RAM. */
  kvOnGpu: boolean;
  /** What was actually priced, after overrides and clamps resolve. */
  nCtx: number;
  cacheTypeKv: string | null;
  nParallel: number;
  layerCount: number | null;
  /** Layers charged to the GPU; null under automatic placement. */
  gpuLayers: number | null;
  /** `--n-cpu-moe` is set, so the GPU figure ignores it and reads high. */
  moeOffloadUnmodelled: boolean;
}

/** The load settings that move the estimate. Mirrors the fields /load takes. */
export interface MemoryEstimateRequest {
  modelPath: string;
  ggufVariant?: string | null;
  hfToken?: string | null;
  nativePathToken?: string | null;
  nCtx?: number | null;
  cacheTypeKv?: string | null;
  nParallel?: number | null;
  nBatch?: number | null;
  nUbatch?: number | null;
  ctxCheckpoints?: number | null;
  speculativeType?: string | null;
  specDraftNMax?: number | null;
  specDraftCacheType?: string | null;
  tensorParallel?: boolean;
  disableVision?: boolean;
  gpuMemoryMode?: string | null;
  gpuLayers?: number | null;
  nCpuMoe?: number | null;
  selectedGpuIds?: number[] | null;
  llamaExtraArgs?: string[] | null;
}

const UNAVAILABLE: MemoryEstimate = {
  available: false,
  reason: "unsizable",
  weightsBytes: 0,
  kvBytes: 0,
  computeBytes: 0,
  drafterRuntimeBytes: 0,
  drafterRuntimeGpuBytes: 0,
  projectorRuntimeBytes: 0,
  drafterKvUnsized: false,
  totalBytes: 0,
  gpuBytes: 0,
  kvEstimable: false,
  kvOnGpu: true,
  nCtx: 0,
  cacheTypeKv: null,
  nParallel: 1,
  layerCount: null,
  gpuLayers: null,
  moeOffloadUnmodelled: false,
};

interface ApiEstimateResponse {
  available: boolean;
  reason: MemoryEstimateReason | null;
  weights_bytes: number;
  kv_bytes: number;
  compute_bytes: number;
  drafter_runtime_bytes: number;
  drafter_runtime_gpu_bytes: number;
  projector_runtime_bytes: number;
  drafter_kv_unsized: boolean;
  total_bytes: number;
  gpu_bytes: number;
  kv_estimable: boolean;
  kv_on_gpu: boolean;
  n_ctx: number;
  cache_type_kv: string | null;
  n_parallel: number;
  layer_count: number | null;
  gpu_layers: number | null;
  moe_offload_unmodelled: boolean;
}

function estimateRequestBody(
  payload: MemoryEstimateRequest,
  nativePathLease: string | null,
): string {
  return JSON.stringify({
    model_path: payload.modelPath,
    gguf_variant: payload.ggufVariant ?? null,
    hf_token: payload.hfToken ?? null,
    native_path_lease: nativePathLease,
    n_ctx: payload.nCtx ?? null,
    cache_type_kv: payload.cacheTypeKv ?? null,
    n_parallel: payload.nParallel ?? null,
    n_batch: payload.nBatch ?? null,
    n_ubatch: payload.nUbatch ?? null,
    ctx_checkpoints: payload.ctxCheckpoints ?? null,
    speculative_type: payload.speculativeType ?? null,
    spec_draft_n_max: payload.specDraftNMax ?? null,
    spec_draft_cache_type: payload.specDraftCacheType ?? null,
    tensor_parallel: payload.tensorParallel ?? false,
    disable_vision: payload.disableVision ?? false,
    gpu_memory_mode: payload.gpuMemoryMode ?? null,
    gpu_layers: payload.gpuLayers ?? null,
    n_cpu_moe: payload.nCpuMoe ?? null,
    selected_gpu_ids: payload.selectedGpuIds ?? null,
    llama_extra_args: payload.llamaExtraArgs ?? null,
  });
}

function toMemoryEstimate(body: ApiEstimateResponse): MemoryEstimate {
  return {
    available: Boolean(body.available),
    reason: body.reason ?? null,
    weightsBytes: body.weights_bytes ?? 0,
    kvBytes: body.kv_bytes ?? 0,
    computeBytes: body.compute_bytes ?? 0,
    drafterRuntimeBytes: body.drafter_runtime_bytes ?? 0,
    // Absent on a backend predating the split: fall back to the whole term, which
    // keeps the old "all of it is on the GPU" reading rather than inventing a zero
    // that would silently drop a real VRAM charge off the row.
    drafterRuntimeGpuBytes:
      body.drafter_runtime_gpu_bytes ?? body.drafter_runtime_bytes ?? 0,
    projectorRuntimeBytes: body.projector_runtime_bytes ?? 0,
    drafterKvUnsized: Boolean(body.drafter_kv_unsized),
    totalBytes: body.total_bytes ?? 0,
    gpuBytes: body.gpu_bytes ?? 0,
    // Absent on an older backend: treat the KV figure as unverified, the safe
    // direction for the one number that can dwarf all the others.
    kvEstimable: body.kv_estimable ?? false,
    kvOnGpu: body.kv_on_gpu ?? true,
    nCtx: body.n_ctx ?? 0,
    cacheTypeKv: body.cache_type_kv ?? null,
    nParallel: body.n_parallel ?? 1,
    layerCount: body.layer_count ?? null,
    gpuLayers: body.gpu_layers ?? null,
    moeOffloadUnmodelled: Boolean(body.moe_offload_unmodelled),
  };
}

/**
 * Price a prospective GGUF load from its header. Allocates nothing, loads nothing.
 *
 * A backend predating the route answers 404, which surfaces as an unavailable estimate
 * rather than an error, so the panel just hides the row.
 */
export async function fetchMemoryEstimate(
  payload: MemoryEstimateRequest,
  signal?: AbortSignal,
): Promise<MemoryEstimate> {
  let nativePathLease: string | null = null;
  if (payload.nativePathToken) {
    try {
      nativePathLease = (
        await consumeNativePathToken(payload.nativePathToken, "validate-model")
      ).nativePathLease;
    } catch {
      // Lease expired / revoked. Nothing was read, so there is no estimate to give.
      return { ...UNAVAILABLE, reason: "unsupported_source" };
    }
  }
  const response = await authFetch("/api/inference/estimate-memory", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    signal,
    body: estimateRequestBody(payload, nativePathLease),
  });
  if (!response.ok) {
    return UNAVAILABLE;
  }
  return toMemoryEstimate((await response.json()) as ApiEstimateResponse);
}

/** GB, to two decimals, matching how the rest of the panel talks about memory. */
export function formatMemoryGb(bytes: number): string {
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}
