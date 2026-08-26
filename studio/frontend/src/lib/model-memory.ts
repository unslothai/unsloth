// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Splits a downloaded model's VRAM footprint into weights and what the current
 * settings add on top: KV cache at the context it will load with, plus an MTP
 * draft reserve when speculative decoding is on.
 *
 * They're kept apart because the fixes differ. Oversized weights need a smaller
 * quant; a total that only overflows once context is added needs a shorter
 * context or a quantized KV cache.
 *
 * Budget matches `gguf-fit.ts` and the backend's `_select_gpus`, so the bar and
 * the fit badge on a row can't contradict each other.
 */

import { VRAM_HEADROOM_RATIO } from "@/lib/gguf-fit";

const BYTES_PER_GB = 1024 ** 3;

export interface ModelMemoryInput {
  /** GGUF weights on disk. Null when the quant could not be resolved. */
  weightsBytes?: number | null;
  /** KV cache at the effective context length. */
  kvBytes?: number | null;
  /** MTP draft reserve; null for ngram (which is free) or a model without one. */
  specBytes?: number | null;
  /**
   * llama.cpp's compute buffers, from the load planner's own `compute_bytes`.
   *
   * Every launch reserves these, and they scale with slots and micro-batch, so a
   * row near the budget could report a fit while omitting gigabytes the server
   * allocates. Folded into the KV segment rather than drawn as a fourth sliver:
   * it is context-linear in part and too small at default settings to resolve on
   * its own, but it still has to count against the total.
   */
  computeBytes?: number | null;
  /**
   * The load planner's own GPU-resident total, which supersedes the sum of the
   * segments when present.
   *
   * The segments are assembled here from separate fields and so can only include
   * what this file knows to ask for; the planner's figure already applies the
   * inherited environment, resolves companions through the loader's search roots
   * and counts every buffer. Segment geometry still comes from the individual
   * terms -- this decides the verdict, not the picture.
   */
  gpuTotalBytes?: number | null;
  /**
   * What the planner still reserves at the shortest context: drafter weights,
   * flat compute buffers, recurrent rollback state. None of it shrinks when the
   * context does, so it is the floor an auto-fitted row is judged against.
   */
  gpuFloorBytes?: number | null;
  /** The share of specBytes a shorter context cannot reduce (drafter weights). */
  specFixedBytes?: number | null;
  /** Total VRAM of the GPU the model would load onto. */
  gpuGb?: number | null;
  /** Context the KV figure was measured at, for the per-token rate. */
  nCtx?: number | null;
  /**
   * Fraction of that VRAM a load may claim, from /api/settings/vram-budget.
   * Defaults to VRAM_HEADROOM_RATIO when unknown. This is the loader's own
   * number and the user can change it, so a bar that hardcodes one warns at a
   * line admission does not use.
   */
  budgetFraction?: number | null;
  /**
   * True when no context is pinned, so the loader will auto-fit rather than
   * open the model's native length. The KV figure is then an upper bound on a
   * context the load would have reduced, so it cannot justify an OOM verdict.
   */
  contextIsAutoFitted?: boolean | null;
}

/**
 * How close the total sits to the budget.
 *
 * Studio's live meters step at 70/90 (`resources-tab.tsx`), but they show usage
 * as it happens, where a sustained 70% is worth flagging. This is a reservation
 * you can see coming, so it holds the accent until 80.
 */
export type ModelMemoryPressure = "normal" | "high" | "critical";

/** Fill fraction (%) at which the bar leaves the accent colour. */
export const PRESSURE_HIGH_PCT = 80;
/** Fill fraction (%) at which the bar turns destructive. */
export const PRESSURE_CRITICAL_PCT = 90;

export type ModelMemoryStatus =
  /** Not enough information to draw anything. */
  | "unknown"
  /** Weights + settings both inside the budget. */
  | "fits"
  /** Weights fit alone, but the configured context pushes the total over. */
  | "context-exceeds"
  /** The weights alone are over budget; context settings cannot save it. */
  | "model-exceeds";

export interface ModelMemorySegments {
  status: ModelMemoryStatus;
  /** Usable VRAM the bar's full width represents. */
  budgetGb: number;
  modelGb: number;
  /** KV cache alone, at the effective context length. */
  kvGb: number;
  /** MTP draft reserve. Zero for ngram or a model with no drafter. */
  specGb: number;
  totalGb: number;
  /** Widths as percentages of the track, already clamped to sum <= 100. */
  modelPct: number;
  kvPct: number;
  specPct: number;
  /** KV cost per token, for judging what a shorter context would save. */
  kvBytesPerToken: number;
  /** Total as a percentage of budget; reads above 100 when over. */
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

/**
 * Bar geometry and fit status for one row.
 *
 * Weights are required, KV is not: a row whose estimate hasn't arrived draws
 * its weights now and fills in the rest, instead of appearing late.
 */
export function computeModelMemory(
  input: ModelMemoryInput,
): ModelMemorySegments {
  const gpuGb = input.gpuGb ?? 0;
  const modelGb = toGb(input.weightsBytes);
  if (gpuGb <= 0 || modelGb <= 0) return EMPTY;
  // The planner says this launch puts nothing on the card. A VRAM bar has
  // nothing to say about it, and drawing one implies a reservation that is not
  // going to happen.
  if (input.gpuTotalBytes === 0) return EMPTY;

  // The loader's own budget when the caller knows it, so the bar warns at the
  // line admission actually draws. VRAM_HEADROOM_RATIO stays the fallback: it
  // is what the fit badge on the same row judges against, and it is the only
  // answer available before the settings request lands.
  const fraction =
    input.budgetFraction && input.budgetFraction > 0
      ? input.budgetFraction
      : VRAM_HEADROOM_RATIO;
  const budgetGb = gpuGb * fraction;
  const kvGb = toGb(input.kvBytes) + toGb(input.computeBytes);
  const specGb = toGb(input.specBytes);
  // The planner's total when it gave one, since it accounts for terms these
  // segments cannot see. Falls back to the sum for a backend too old to send it.
  const segmentSumGb = modelGb + kvGb + specGb;
  // Explicitly null-checked, because zero is an answer rather than the absence
  // of one: inherited placement can make the launch entirely CPU resident, and
  // `||` would have quietly swapped that for the segment sum and drawn VRAM
  // pressure for a load that touches no card.
  const totalGb =
    input.gpuTotalBytes == null ? segmentSumGb : toGb(input.gpuTotalBytes);

  // Only the weights are a hard verdict: they are what they are. The context
  // term is a reservation the loader is free to shrink when nothing pinned it,
  // so an unpinned row that only tips over because of KV reports "fits" rather
  // than warning about a length the load would never have opened.
  //
  // The speculative segment is not wholly context-linear though: a separate
  // drafter's own weights are resident whatever the context, and no auto-fit can
  // shrink them. So the hard floor is the weights plus that fixed share -- if
  // target and drafter weights together do not fit, saying "fits" because the
  // context is unpinned describes a load that cannot open at any length.
  // The planner's floor when it gave one: it names every fixed term, including
  // the flat compute buffer and a Hybrid Mamba target's rollback state, which
  // the drafter-weights figure alone misses. Both can be several GiB, and both
  // survive any context reduction, so folding them into the reducible part let
  // auto-fit suppress an overage nothing could fix.
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

  // Clamp to the track, so an oversized model can't push the later segments
  // out of the row.
  const modelPct = Math.min(100, (modelGb / budgetGb) * 100);
  const kvPct = Math.min(100 - modelPct, (kvGb / budgetGb) * 100);
  const specPct = Math.min(100 - modelPct - kvPct, (specGb / budgetGb) * 100);

  // KV is linear in context above the SWA floor, so the marginal rate is what
  // says whether a shorter context would actually help.
  const nCtx = input.nCtx && input.nCtx > 0 ? input.nCtx : 0;
  const kvBytesPerToken =
    nCtx > 0 && input.kvBytes && input.kvBytes > 0 ? input.kvBytes / nCtx : 0;

  // Uncapped: widths clamp to the track, but pressure still needs to tell
  // "just full" from "twice over".
  const fillPct = (totalGb / budgetGb) * 100;
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

/**
 * Pass-through llama-server args that decide where the model is placed.
 *
 * Exported for the policy tests: these are the flags whose presence means the
 * VRAM total stops describing the load, so the bar has to abstain exactly as it
 * does for the equivalent structured controls.
 */
export const PLACEMENT_OWNING_ARGS = [
  // _GPU_LAYER_FLAGS and _FIT_FLAGS
  "--gpu-layers",
  "-ngl",
  "--n-gpu-layers",
  "--fit",
  "-fit",
  // _DEVICE_FLAGS -- all four spellings. `-dev cuda0` on a multi-GPU host
  // confines the load while the bar was still comparing it with aggregate VRAM.
  "--device",
  "-dev",
  "--main-gpu",
  "-mg",
  "--tensor-split",
  "-ts",
  "--split-mode",
  "-sm",
  // _MOE_OFFLOAD_FLAGS -- the --cpu-moe pair as well as the -ncmoe one
  "--n-cpu-moe",
  "-ncmoe",
  "--cpu-moe",
  "-cmoe",
  "--override-tensor",
  "-ot",
  "--no-kv-offload",
  "-nkvo",
  // _DRAFT_GPU_LAYER_FLAGS and the drafter's own device pin. A drafter pinned
  // to the CPU is host memory the bar would otherwise charge to the card.
  "--spec-draft-ngl",
  "-ngld",
  "--gpu-layers-draft",
  "--n-gpu-layers-draft",
  "--spec-draft-device",
  // _MMPROJ_OFFLOAD_FLAGS: either spelling is the user taking ownership of
  // where the projector lands.
  "--mmproj-offload",
  "--no-mmproj-offload",
];

/**
 * Pass-through args that change the SIZE of the KV cache rather than where it
 * sits.
 *
 * The bar prices the cache from the structured controls alone, so any of these
 * in the box means the launch reserves something other than what was priced.
 * `--swa-full` is the sharp one: on a sliding-window model it replaces the
 * compact window with a full-context cache, which the loader honours at
 * `_estimate_kv_cache_bytes` via `_swa_full_from_args_or_env`. Pricing the
 * window and reporting a fit for a launch that allocates the full context is the
 * exact false "fits" this bar exists to prevent, so it abstains instead.
 */
export const KV_SHAPING_ARGS = [
  "--swa-full",
  // Turning flash attention off changes the cache layout, not just its size:
  // _estimate_kv_cache_bytes then pads variable-width V tensors to the
  // model-wide maximum, which can reserve materially more than the default
  // enabled layout this estimate prices.
  "--flash-attn",
  "-fa",
  // The loader treats an extras --spec-type as authoritative over the
  // structured mode, so a config saying "off" can still open an embedded NextN
  // head. The request carries only the structured mode, so the draft KV and the
  // target rollback state would both be missing from the total.
  "--spec-type",
  "--spec-default",
  // Draft depth in llama.cpp's spellings as well as Unsloth's: a recurrent
  // target keeps one rollback state per drafted token, so --draft-max 16 is a
  // materially different reservation from the structured default.
  "--spec-draft-n-max",
  "--draft-max",
  "--draft-min",
  // _SPEC_DRAFT_CACHE_K_FLAGS / _SPEC_DRAFT_CACHE_V_FLAGS, every alias. The
  // structured control sets one dtype for both, so an inherited pair that split
  // K from V describes a cache this estimate never priced.
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

/**
 * Pass-through args that make the launch hold files this estimate never sized.
 *
 * A LoRA, a control vector, an explicit projector or a hand-named drafter are
 * all resident bytes chosen in the extras box, and none of them reach the
 * planner through the structured settings. Unlike the KV-shaping flags these do
 * not reshape a term that was priced -- they add one that was not -- so the
 * total is a floor rather than an answer and the bar abstains.
 */
export const RESIDENT_ADDING_ARGS = [
  "--lora",
  "--lora-scaled",
  "--control-vector",
  "--control-vector-scaled",
  "--mmproj",
  "--model-draft",
  "-md",
  "--spec-draft-hf",
];

/** Whether pass-through args add resident files this estimate did not price. */
export function extraArgsAddResidentFiles(
  args: string[] | null | undefined,
): boolean {
  if (!args || args.length === 0) return false;
  return args.some((arg) => {
    const token = String(arg ?? "").split("=")[0].trim();
    return RESIDENT_ADDING_ARGS.includes(token);
  });
}

/** Whether pass-through args resize the KV cache, so the priced figure stops applying. */
export function extraArgsShapeKvCache(
  args: string[] | null | undefined,
): boolean {
  if (!args || args.length === 0) return false;
  return args.some((arg) => {
    const token = String(arg ?? "").split("=")[0].trim();
    return KV_SHAPING_ARGS.includes(token);
  });
}

/** Whether pass-through args decide placement, so the GPU total stops applying. */
export function extraArgsOwnPlacement(
  args: string[] | null | undefined,
): boolean {
  if (!args?.length) return false;
  return args.some((raw) => {
    // Accept both "--gpu-layers 0" and "--gpu-layers=0" argv shapes.
    const token = String(raw).trim().split("=", 1)[0];
    return PLACEMENT_OWNING_ARGS.includes(token);
  });
}

/** The inputs that change an estimate, and so key its cached answer. */
export interface EstimateCacheKeyParts {
  repoId: string;
  quant: string;
  /** A re-download can change the file under a stable quant name. */
  sizeBytes?: number | null;
  /** Undefined means the model's own native length. */
  nCtx?: number | null;
  kvCacheDtype?: string | null;
  speculativeType?: string | null;
  /** Undefined or non-positive means the server's standing slot count. */
  nParallel?: number | null;
  /** Draft depth; zero is a real value (no rollback states), so it is kept. */
  specDraftNMax?: number | null;
  specDraftCacheType?: string | null;
  /** Saved context checkpoints; zero is a real value. */
  ctxCheckpoints?: number | null;
  /** Vision off frees the projector, which changes the footprint. */
  disableVision?: boolean | null;
  /** Compute buffers scale with these, so they re-key the answer. */
  nBatch?: number | null;
  nUbatch?: number | null;
  tensorParallel?: boolean | null;
}

/**
 * Cache key for one estimate.
 *
 * Every field the request carries appears here, because two rows that would ask
 * the backend different questions must not share an answer. The slot count is
 * the one that is easy to get wrong: an omitted count is not one slot, it is
 * "whatever the server is configured for", which defaults above one. Joined with
 * a separator that cannot occur in a normalized repo id or quant label, since a
 * space can.
 */
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

/**
 * Whether a response says "I could not size this model".
 *
 * The route is best-effort: it answers 200 with every field null rather than
 * failing, so this arrives down the success path. Callers cache it as a failure
 * so it expires, because the usual cause is a backend that is briefly away, not
 * a model that can never be sized.
 */
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

/** Compact label for a per-token KV rate, which lands in KB or MB. */
export function formatKvRate(bytes: number): string {
  if (bytes <= 0) return "0 KB";
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb < 10 ? kb.toFixed(1) : Math.round(kb)} KB`;
  const mb = kb / 1024;
  return `${mb < 10 ? mb.toFixed(1) : Math.round(mb)} MB`;
}

/**
 * Compact label for the bar's readout ("7.2 GiB").
 *
 * GiB, not GB: every figure here is a binary divide. Weights and KV come from
 * bytes / 1024**3, and gpuGb arrives from the backend as
 * props.total_memory / 1024**3 with nothing subtracting a budget on the way, so
 * calling any of them GB overstates each by 7.4% (#9570).
 */
export function formatMemoryGb(gb: number): string {
  if (gb <= 0) return "0 GiB";
  return `${gb < 10 ? gb.toFixed(1) : Math.round(gb)} GiB`;
}
