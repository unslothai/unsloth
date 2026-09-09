// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import { consumeNativePathToken } from "@/features/native-intents/api";

/** GB, to two decimals. Lives in the import-free module beside the fit rules so the node test
 *  runner can reach it; re-exported here because every caller already imports this file. */
export { formatMemoryGb } from "../model-config/memory-fit";

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
  /** The share of the above that lands on the GPU. A figure rather than a flag: under MTP the
   *  target-side verification state follows the TARGET cache and the draft cache follows the
   *  drafter, so the two halves can be placed differently. */
  drafterRuntimeGpuBytes: number;
  /** The vision encoder's buffers, about 0.4x the projector file on top of it. */
  projectorRuntimeBytes: number;
  /** A charged drafter whose cache could not be sized: `--spec-draft-hf` names a repository, so
   *  its header is not on this disk. The total is a floor. */
  drafterKvUnsized: boolean;
  /** A pass-through adapter file that could not be sized, so the total is a floor. */
  adaptersUnsized: boolean;
  /** Weights + KV + compute, wherever they land (VRAM, host RAM, or one unified pool). */
  totalBytes: number;
  /** The share of `totalBytes` that lands on the GPU under the requested offload. */
  gpuBytes: number;
  /** False when the GGUF header lacks the attention dims needed to size the cache. The numbers
   *  above are then a lower bound, not an estimate: at a long context the cache dominates. */
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
  adaptersUnsized: false,
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
  adapters_unsized?: boolean;
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

/** A byte count the row can show, or the fallback. `??` defends against null and undefined,
 *  not against a value: JSON.parse turns `1e999` into Infinity, a field can arrive
 *  stringified, and a negative byte count is not a footprint. All three reach
 *  classifyMemoryFit, where NaN used to come back "fits". Both skew fallbacks are preserved:
 *  an ABSENT key falls through to the caller's fallback, and an explicit 0 is a real answer. */
function finiteBytes(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? value
    : fallback;
}

/** A count (context, slots, layers) rather than a byte size: same guard, no fallback chain, and
 *  null stays null where null is the "unknown" the row prints. */
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

/** A flag whose ABSENCE is meaningful, so a non-boolean is treated as absent rather than
 *  coerced: `Boolean("false")` is true, and these decide what the row claims. */
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
  return {
    available: flag(body.available, false),
    // A reason the panel has no copy for is not a reason. An unknown string would reach the copy
    // map and render nothing at all.
    reason: ESTIMATE_REASONS.includes(body.reason as MemoryEstimateReason)
      ? (body.reason as MemoryEstimateReason)
      : null,
    weightsBytes: finiteBytes(body.weights_bytes, 0),
    kvBytes: finiteBytes(body.kv_bytes, 0),
    computeBytes: finiteBytes(body.compute_bytes, 0),
    drafterRuntimeBytes,
    // Absent on a backend predating the split: fall back to the whole term, which keeps the old
    // "all of it is on the GPU" reading rather than inventing a zero that would drop a real
    // VRAM charge off the row.
    drafterRuntimeGpuBytes: finiteBytes(
      body.drafter_runtime_gpu_bytes,
      drafterRuntimeBytes,
    ),
    projectorRuntimeBytes: finiteBytes(body.projector_runtime_bytes, 0),
    drafterKvUnsized: flag(body.drafter_kv_unsized, false),
    // Absent on a backend that predates the adapter term, and false is the right reading there:
    // it charged no adapters, so it claimed no floor.
    adaptersUnsized: flag(body.adapters_unsized, false),
    totalBytes: finiteBytes(body.total_bytes, 0),
    gpuBytes: finiteBytes(body.gpu_bytes, 0),
    // Absent on an older backend: treat the KV figure as unverified, the safe direction for the
    // one number that can dwarf all the others.
    kvEstimable: flag(body.kv_estimable, false),
    kvOnGpu: flag(body.kv_on_gpu, true),
    nCtx: finiteCount(body.n_ctx, 0),
    cacheTypeKv:
      typeof body.cache_type_kv === "string" ? body.cache_type_kv : null,
    nParallel: finiteCount(body.n_parallel, 1),
    layerCount: nullableCount(body.layer_count),
    gpuLayers: nullableCount(body.gpu_layers),
    moeOffloadUnmodelled: flag(body.moe_offload_unmodelled, false),
  };
}

/** Statuses that say the ROUTE is not there, as distinct from this request failing. 404 and
 *  405 are a backend predating it (405 because a router owning the path for another method
 *  answers that), 501 one that answers but declines. Everything else -- 401, 422, 500, a
 *  gateway error, a non-JSON body -- is about this request, not the route's existence. */
function routeAbsentStatus(status: number): boolean {
  return status === 404 || status === 405 || status === 501;
}

/** How long a structural miss is trusted before the next qualifying change re-probes. Worth
 *  memoing: this route is POSTed after EVERY settings change, so a new bundle against an old
 *  backend fires one debounced 404 per slider release for the life of the tab. Not permanent,
 *  though: Studio replaces its own backend in place, and a latched miss would keep the row
 *  hidden until reload. A TTL costs one wasted POST per window and needs no restart signal. */
const ROUTE_ABSENT_TTL_MS = 5 * 60 * 1000;
let routeAbsentAt: number | null = null;

/** Forget a recorded miss. For tests, and for any caller that learns the backend changed
 *  underneath it before the window is up. */
export function resetMemoryEstimateRouteMemo(): void {
  routeAbsentAt = null;
}

/** Price a prospective GGUF load from its header. Allocates nothing, loads nothing. Never
 *  throws for a backend answer: an absent route, an auth expiry, a 500, an HTML error page
 *  served as 200 or a truncated body all come back as an unavailable estimate, so the panel
 *  hides the row. The statuses are told apart only to decide whether the miss is memoable. */
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
    // Only a structural miss latches. A transient 500 explicitly CLEARS the memo rather than
    // leaving an older one standing: a backend answering at all is not one missing the route.
    routeAbsentAt = routeAbsentStatus(response.status) ? Date.now() : null;
    return UNAVAILABLE;
  }
  routeAbsentAt = null;
  let body: unknown;
  try {
    body = await response.json();
  } catch {
    // A 200 that is not JSON: a captive portal, a dev proxy serving its own HTML, or a body cut
    // short. Nothing was measured, so there is nothing to show.
    return UNAVAILABLE;
  }
  if (typeof body !== "object" || body === null) {
    return UNAVAILABLE;
  }
  return toMemoryEstimate(body as ApiEstimateResponse);
}
