// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { consumeNativePathToken } from "@/features/native-intents/api";

/** Lives in the import-free memory-fit module so the node test runner can reach it. */
export { formatMemoryGb } from "../model-config/memory-fit";

export type MemoryEstimateReason =
  | "not_gguf"
  | "not_downloaded"
  | "unsupported_source"
  | "unsizable";

export interface MemoryEstimate {
  available: boolean;
  reason: MemoryEstimateReason | null;
  weightsBytes: number;
  /** Meaningless unless `kvEstimable`. */
  kvBytes: number;
  kvCheckpointBytes: number;
  computeBytes: number;
  drafterRuntimeBytes: number;
  /** Under MTP the target and draft halves can be placed differently. */
  drafterRuntimeGpuBytes: number;
  /** About 0.4x the projector file on top of it. */
  projectorRuntimeBytes: number;
  /** `--spec-draft-hf` names a remote repo, so its cache is unsized; the total is a floor. */
  drafterKvUnsized: boolean;
  /** The total is a floor. */
  adaptersUnsized: boolean;
  totalBytes: number;
  gpuBytes: number;
  /** False when the GGUF header lacks attention dims; the numbers are then a lower bound. */
  kvEstimable: boolean;
  kvOnGpu: boolean;
  nCtx: number;
  contextFitted: number | null;
  contextIsPinned: boolean;
  gpuFloorBytes: number | null;
  floorCanOffload: boolean;
  cacheTypeKv: string | null;
  nParallel: number;
  layerCount: number | null;
  gpuLayers: number | null;
  /** `--n-cpu-moe` is set, so the GPU figure ignores it and reads high. */
  moeOffloadUnmodelled: boolean;
}

export interface MemoryEstimateRequest {
  modelPath: string;
  ggufVariant?: string | null;
  hfToken?: string | null;
  nativePathToken?: string | null;
  nCtx?: number | null;
  cacheTypeKv?: string | null;
  maxSeqLength?: number | null;
  mlxKvQuant?: string | null;
  nParallel?: number | null;
  nBatch?: number | null;
  nUbatch?: number | null;
  ctxCheckpoints?: number | null;
  speculativeType?: string | null;
  specDraftNMax?: number | null;
  specDraftModel?: string | null;
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
  kvCheckpointBytes: 0,
  computeBytes: 0,
  drafterRuntimeBytes: 0,
  drafterRuntimeGpuBytes: 0,
  projectorRuntimeBytes: 0,
  drafterKvUnsized: false,
  adaptersUnsized: false,
  totalBytes: 0,
  gpuBytes: 0,
  kvEstimable: false,
  kvOnGpu: true,
  nCtx: 0,
  contextFitted: null,
  contextIsPinned: true,
  gpuFloorBytes: null,
  floorCanOffload: false,
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
  kv_checkpoint_bytes?: number;
  compute_bytes: number;
  drafter_runtime_bytes: number;
  drafter_runtime_gpu_bytes: number;
  projector_runtime_bytes: number;
  drafter_kv_unsized: boolean;
  adapters_unsized?: boolean;
  total_bytes: number;
  gpu_bytes: number;
  kv_estimable: boolean;
  kv_on_gpu: boolean;
  n_ctx: number;
  context_fitted?: number | null;
  context_is_pinned?: boolean;
  gpu_floor_bytes?: number | null;
  floor_can_offload?: boolean;
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
    max_seq_length: payload.maxSeqLength ?? null,
    mlx_kv_quant: payload.mlxKvQuant ?? null,
    n_parallel: payload.nParallel ?? null,
    n_batch: payload.nBatch ?? null,
    n_ubatch: payload.nUbatch ?? null,
    ctx_checkpoints: payload.ctxCheckpoints ?? null,
    speculative_type: payload.speculativeType ?? null,
    spec_draft_n_max: payload.specDraftNMax ?? null,
    spec_draft_model: payload.specDraftModel ?? null,
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

/** Guards Infinity, stringified and negative values; absent falls back, explicit 0 is real. */
function finiteBytes(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value
    : fallback;
}

function finiteCount(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? Math.trunc(value)
    : fallback;
}

function nullableCount(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? Math.trunc(value)
    : null;
}

/** Non-booleans count as absent: `Boolean("false")` is true. */
function flag(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

const ESTIMATE_REASONS: readonly MemoryEstimateReason[] = [
  "not_gguf",
  "not_downloaded",
  "unsupported_source",
  "unsizable",
];

function toMemoryEstimate(body: ApiEstimateResponse): MemoryEstimate {
  const drafterRuntimeBytes = finiteBytes(body.drafter_runtime_bytes, 0);
  const kvBytes = finiteBytes(body.kv_bytes, 0);
  return {
    available: flag(body.available, false),
    // An unknown reason would render nothing.
    reason: ESTIMATE_REASONS.includes(body.reason as MemoryEstimateReason)
      ? (body.reason as MemoryEstimateReason)
      : null,
    weightsBytes: finiteBytes(body.weights_bytes, 0),
    kvBytes,
    kvCheckpointBytes: Math.min(
      kvBytes,
      finiteBytes(body.kv_checkpoint_bytes, 0),
    ),
    computeBytes: finiteBytes(body.compute_bytes, 0),
    drafterRuntimeBytes,
    // Absent on an older backend: charge the whole term to the GPU rather than inventing a zero.
    drafterRuntimeGpuBytes: finiteBytes(
      body.drafter_runtime_gpu_bytes,
      drafterRuntimeBytes,
    ),
    projectorRuntimeBytes: finiteBytes(body.projector_runtime_bytes, 0),
    drafterKvUnsized: flag(body.drafter_kv_unsized, false),
    adaptersUnsized: flag(body.adapters_unsized, false),
    totalBytes: finiteBytes(body.total_bytes, 0),
    gpuBytes: finiteBytes(body.gpu_bytes, 0),
    // Absent on an older backend: treat KV as unverified, the safe direction.
    kvEstimable: flag(body.kv_estimable, false),
    kvOnGpu: flag(body.kv_on_gpu, true),
    nCtx: finiteCount(body.n_ctx, 0),
    contextFitted: nullableCount(body.context_fitted),
    contextIsPinned: flag(body.context_is_pinned, true),
    gpuFloorBytes: nullableCount(body.gpu_floor_bytes),
    floorCanOffload: flag(body.floor_can_offload, false),
    cacheTypeKv:
      typeof body.cache_type_kv === "string" ? body.cache_type_kv : null,
    nParallel: finiteCount(body.n_parallel, 1),
    layerCount: nullableCount(body.layer_count),
    gpuLayers: nullableCount(body.gpu_layers),
    moeOffloadUnmodelled: flag(body.moe_offload_unmodelled, false),
  };
}

/** 404/405/501 mean the route is absent; anything else is about this request. */
function routeAbsentStatus(status: number): boolean {
  return status === 404 || status === 405 || status === 501;
}

/** POSTed after every settings change; a TTL avoids a 404 per slider release yet re-probes
    after Studio replaces its backend in place. */
const ROUTE_ABSENT_TTL_MS = 5 * 60 * 1000;
let routeAbsentAt: number | null = null;

export function resetMemoryEstimateRouteMemo(): void {
  routeAbsentAt = null;
}

export async function fetchMemoryEstimate(
  payload: MemoryEstimateRequest,
  signal?: AbortSignal,
): Promise<MemoryEstimate> {
  if (
    routeAbsentAt !== null &&
    Date.now() - routeAbsentAt < ROUTE_ABSENT_TTL_MS
  ) {
    return UNAVAILABLE;
  }
  let nativePathLease: string | null = null;
  if (payload.nativePathToken) {
    try {
      nativePathLease = (
        await consumeNativePathToken(payload.nativePathToken, "validate-model")
      ).nativePathLease;
    } catch {
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
    // Only a structural miss latches; a transient 500 clears the memo.
    routeAbsentAt = routeAbsentStatus(response.status) ? Date.now() : null;
    return UNAVAILABLE;
  }
  routeAbsentAt = null;
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    // A non-JSON 200 (captive portal, proxy HTML, truncated body) measured nothing.
    return UNAVAILABLE;
  }
  if (typeof body !== "object" || body === null) {
    return UNAVAILABLE;
  }
  return toMemoryEstimate(body as ApiEstimateResponse);
}
