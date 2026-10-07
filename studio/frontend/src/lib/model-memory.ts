// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Weights are kept apart from settings-driven KV/spec because the fixes differ.
 * Budget must match `gguf-fit.ts` and the backend's `_select_gpus`.
 */

import {
  DEFAULT_VRAM_BUDGET_FRACTION,
  PRESSURE_CRITICAL_PCT,
  PRESSURE_HIGH_PCT,
} from "./memory/thresholds.ts";

const BYTES_PER_GB = 1024 ** 3;

export interface ModelMemoryInput {
  weightsBytes?: number | null;
  kvBytes?: number | null;
  /** Null for ngram (free) or a model without a drafter. */
  specBytes?: number | null;
  /** llama.cpp compute buffers; folded into the KV segment rather than drawn separately. */
  computeBytes?: number | null;
  /** Planner's GPU total; decides the verdict, while segment geometry still comes from the terms. */
  gpuTotalBytes?: number | null;
  /** Fixed reservation at the shortest context; the floor an auto-fitted row is judged against. */
  gpuFloorBytes?: number | null;
  specFixedBytes?: number | null;
  gpuGb?: number | null;
  nCtx?: number | null;
  /** User-adjustable loader fraction; defaults to DEFAULT_VRAM_BUDGET_FRACTION. */
  budgetFraction?: number | null;
  /** No pinned context: KV is an upper bound the loader would shrink, so it cannot justify OOM. */
  contextIsAutoFitted?: boolean | null;
}

export type ModelMemoryPressure = "normal" | "high" | "critical";

// Shared with the Load Model panel; see src/lib/memory/thresholds.ts.
export {
  PRESSURE_HIGH_PCT,
  PRESSURE_CRITICAL_PCT,
} from "./memory/thresholds.ts";

export type ModelMemoryStatus =
  | "unknown"
  | "fits"
  | "context-exceeds"
  | "model-exceeds";

export interface ModelMemorySegments {
  status: ModelMemoryStatus;
  budgetGb: number;
  modelGb: number;
  kvGb: number;
  specGb: number;
  totalGb: number;
  /** Percentages of the track, clamped to sum <= 100. */
  modelPct: number;
  kvPct: number;
  specPct: number;
  kvBytesPerToken: number;
  /** Reads above 100 when over budget. */
  fillPct: number;
  pressure: ModelMemoryPressure;
}

const EMPTY: ModelMemorySegments = {
  status: "unknown",
  budgetGb: 0,
  modelGb: 0,
  kvGb: 0,
  specGb: 0,
  totalGb: 0,
  modelPct: 0,
  kvPct: 0,
  specPct: 0,
  kvBytesPerToken: 0,
  fillPct: 0,
  pressure: "normal",
};

function toGb(bytes?: number | null): number {
  return bytes && bytes > 0 ? bytes / BYTES_PER_GB : 0;
}

/** Weights are required, KV is not, so a row draws weights before its estimate arrives. */
export function computeModelMemory(
  input: ModelMemoryInput,
): ModelMemorySegments {
  // Non-finite figures give no verdict (JSON.parse can yield Infinity). Null/undefined mean
  // "not arrived yet" and are handled by the segments.
  if (
    [
      input.weightsBytes,
      input.kvBytes,
      input.specBytes,
      input.computeBytes,
      input.gpuTotalBytes,
      input.gpuFloorBytes,
      input.specFixedBytes,
      input.gpuGb,
      input.nCtx,
      input.budgetFraction,
    ].some((value) => typeof value === "number" && !Number.isFinite(value))
  ) {
    return EMPTY;
  }
  const gpuGb = input.gpuGb ?? 0;
  const modelGb = toGb(input.weightsBytes);
  if (gpuGb <= 0 || modelGb <= 0) return EMPTY;
  // The planner puts nothing on the card, so a VRAM bar would imply a reservation that never happens.
  if (input.gpuTotalBytes === 0) return EMPTY;

  // Fallback matches the loader's _CTX_FIT_VRAM_FRACTION (0.97), not VRAM_HEADROOM_RATIO (0.90):
  // "0.90 dropped 91-94% fits to CPU offload, #5106".
  const fraction =
    input.budgetFraction && input.budgetFraction > 0
      ? input.budgetFraction
      : DEFAULT_VRAM_BUDGET_FRACTION;
  const budgetGb = gpuGb * fraction;
  const kvGb = toGb(input.kvBytes) + toGb(input.computeBytes);
  const specGb = toGb(input.specBytes);
  // The planner's total sees terms these segments cannot; old backends fall back to the sum.
  const segmentSumGb = modelGb + kvGb + specGb;
  // Null-checked, not `||`: zero means a fully CPU-resident launch.
  const totalGb =
    input.gpuTotalBytes == null ? segmentSumGb : toGb(input.gpuTotalBytes);

  // Only fixed terms give a hard verdict: KV may be shrunk when context is unpinned, but drafter
  // weights, compute buffer and recurrent state survive any context reduction.
  const specFixedGb = toGb(input.specFixedBytes);
  const irreducibleGb =
    input.gpuFloorBytes == null
      ? modelGb + specFixedGb
      : toGb(input.gpuFloorBytes);
  const status: ModelMemoryStatus =
    irreducibleGb > budgetGb
      ? "model-exceeds"
      : totalGb > budgetGb && !input.contextIsAutoFitted
        ? "context-exceeds"
        : "fits";

  const modelPct = Math.min(100, (modelGb / budgetGb) * 100);
  const kvPct = Math.min(100 - modelPct, (kvGb / budgetGb) * 100);
  const specPct = Math.min(100 - modelPct - kvPct, (specGb / budgetGb) * 100);

  // KV is linear in context above the SWA floor, so the marginal rate shows what shortening saves.
  const nCtx = input.nCtx && input.nCtx > 0 ? input.nCtx : 0;
  const kvBytesPerToken =
    nCtx > 0 && input.kvBytes && input.kvBytes > 0 ? input.kvBytes / nCtx : 0;

  // Uncapped so pressure can tell "full" from "twice over". Auto-fitted rows draw pressure from
  // the irreducible floor, since the native-context total is an upper bound never opened.
  const autoFitted = Boolean(input.contextIsAutoFitted);
  const pressureGb =
    autoFitted && input.gpuFloorBytes != null
      ? // Auto-fit sizes the cache to what is left, so the floor alone is the useful signal.
        Math.min(irreducibleGb, totalGb)
      : autoFitted
        ? Math.min(totalGb, budgetGb)
        : totalGb;
  const fillPct = (pressureGb / budgetGb) * 100;
  const pressure: ModelMemoryPressure =
    fillPct >= PRESSURE_CRITICAL_PCT
      ? "critical"
      : fillPct >= PRESSURE_HIGH_PCT
        ? "high"
        : "normal";

  return {
    status,
    budgetGb,
    modelGb,
    kvGb,
    specGb,
    kvPct,
    specPct,
    kvBytesPerToken,
    totalGb,
    modelPct,
    fillPct,
    pressure,
  };
}

/** Flags whose presence means the VRAM total stops describing the load, so the bar abstains. */
export const PLACEMENT_OWNING_ARGS = [
  // _GPU_LAYER_FLAGS and _FIT_FLAGS
  "--gpu-layers",
  "-ngl",
  "--n-gpu-layers",
  "--fit",
  "-fit",
  // _DEVICE_FLAGS, all four spellings.
  "--device",
  "-dev",
  "--main-gpu",
  "-mg",
  "--tensor-split",
  "-ts",
  "--split-mode",
  "-sm",
  // _MOE_OFFLOAD_FLAGS
  "--n-cpu-moe",
  "-ncmoe",
  "--cpu-moe",
  "-cmoe",
  "--override-tensor",
  "-ot",
  "--no-kv-offload",
  "-nkvo",
  // _DRAFT_GPU_LAYER_FLAGS and the drafter's own device pin.
  "--spec-draft-ngl",
  "-ngld",
  "--gpu-layers-draft",
  "--n-gpu-layers-draft",
  "--spec-draft-device",
  // _MMPROJ_OFFLOAD_FLAGS
  "--mmproj-offload",
  "--no-mmproj-offload",
];

/** Flags that resize the KV cache; e.g. `--swa-full` replaces the window with a full-context cache. */
export const KV_SHAPING_ARGS = [
  "--swa-full",
  // Disabling flash attention pads V tensors to the model-wide maximum in the loader's estimate.
  "--flash-attn",
  "-fa",
  // The loader treats an extras --spec-type as authoritative over the structured mode.
  "--spec-type",
  "--spec-default",
  // A recurrent target keeps one rollback state per drafted token.
  "--spec-draft-n-max",
  "--draft-max",
  "--draft-min",
  // _SPEC_DRAFT_CACHE_K_FLAGS / _SPEC_DRAFT_CACHE_V_FLAGS, every alias.
  "--spec-draft-type-k",
  "--cache-type-k-draft",
  "-ctkd",
  "--spec-draft-type-v",
  "--cache-type-v-draft",
  "-ctvd",
  "--kv-unified",
  "-kvu",
  "--ctx-size",
  "-c",
  "--parallel",
  "-np",
  "--batch-size",
  "-b",
  "--ubatch-size",
  "-ub",
  "--cache-type-k",
  "-ctk",
  "--cache-type-v",
  "-ctv",
  "--ctx-checkpoints",
  // _CTX_CHECKPOINTS_FLAGS: --swa-checkpoints is upstream's older spelling.
  "-ctxcp",
  "--swa-checkpoints",
];

/** Flags adding unpriced resident files (LoRA, control vector, projector, drafter): total is a floor. */
export const RESIDENT_ADDING_ARGS = [
  // _LOCAL_DRAFT_FLAGS and _HF_DRAFT_FLAGS; flags are last-wins, so any spelling opens a drafter.
  "--model-draft",
  "--spec-draft-model",
  "-md",
  "--spec-draft-hf",
  "-hfd",
  "-hfrd",
  "--hf-repo-draft",
  "--lora",
  "--lora-scaled",
  "--control-vector",
  "--control-vector-scaled",
  "--mmproj",
  "-mm",
  "--model-draft",
  "-md",
  "--spec-draft-hf",
];

export function extraArgsAddResidentFiles(
  args: string[] | null | undefined,
): boolean {
  if (!args || args.length === 0) return false;
  return args.some((arg) => RESIDENT_ADDING_ARGS.includes(flagName(arg)));
}

export function extraArgsShapeKvCache(
  args: string[] | null | undefined,
): boolean {
  if (!args || args.length === 0) return false;
  return args.some((arg) => KV_SHAPING_ARGS.includes(flagName(arg)));
}

/** Mirrors the backend's `_flag_name`: strip `=value` and fold underscores to dashes in long options. */
function flagName(arg: unknown): string {
  const token = String(arg ?? "").trim();
  if (!token.startsWith("-")) return "";
  const name = token.split("=")[0];
  return name.startsWith("--") ? name.replace(/_/g, "-") : name;
}

export function extraArgsOwnPlacement(
  args: string[] | null | undefined,
): boolean {
  if (!args?.length) return false;
  return args.some((raw) => PLACEMENT_OWNING_ARGS.includes(flagName(raw)));
}

export interface EstimateCacheKeyParts {
  repoId: string;
  quant: string;
  /** A re-download can change the file under a stable quant name. */
  sizeBytes?: number | null;
  nCtx?: number | null;
  kvCacheDtype?: string | null;
  speculativeType?: string | null;
  /** Unset means the server's slot count, which defaults above one. */
  nParallel?: number | null;
  /** Zero is a real value (no rollback states). */
  specDraftNMax?: number | null;
  specDraftCacheType?: string | null;
  /** Zero is a real value. */
  ctxCheckpoints?: number | null;
  disableVision?: boolean | null;
  nBatch?: number | null;
  nUbatch?: number | null;
  tensorParallel?: boolean | null;
}

/** Every request field must be in the key so different questions never share an answer. */
export function estimateCacheKey(parts: EstimateCacheKeyParts): string {
  return [
    parts.repoId,
    parts.quant,
    parts.sizeBytes ?? "",
    parts.nCtx ?? "native",
    parts.kvCacheDtype ?? "",
    parts.speculativeType ?? "",
    parts.nParallel && parts.nParallel > 0 ? parts.nParallel : "server-default",
    parts.specDraftNMax ?? "default",
    parts.specDraftCacheType ?? "",
    parts.ctxCheckpoints ?? "default",
    parts.disableVision ? "novision" : "",
    parts.nBatch ?? "default",
    parts.nUbatch ?? "default",
    parts.tensorParallel ? "tp" : "",
  ].join("\x00");
}

/** The route answers 200 with all nulls when it cannot size; cache that as an expiring failure. */
export function estimateIsUnsized(estimate: {
  kvBytes: number | null;
  weightsBytes: number | null;
  specBytes: number | null;
}): boolean {
  return (
    estimate.kvBytes === null &&
    estimate.weightsBytes === null &&
    estimate.specBytes === null
  );
}

// Labels are now KiB/MiB; the divide was always by 1024.
export { formatKvRate } from "./memory/format.ts";

/** @deprecated Use `formatGiB` from `@/lib/memory/format`, whose name states its unit. */
export { formatGiB as formatMemoryGb } from "./memory/format.ts";
