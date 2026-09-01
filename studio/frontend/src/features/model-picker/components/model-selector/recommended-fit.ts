// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pure helpers for the Recommended list: which formats to surface and whether a model fits
// the device. No React/DOM deps so they are easy to test.

import { classifyGgufFit } from "../../../../lib/gguf-fit.ts";
import { classifyMediaGgufFit } from "./model-catalog.ts";

const GGUF_SUFFIX_RE = /-GGUF(?:$|-)/i;
const MLX_RE = /-MLX(?:$|-)/i;

export function isGgufId(id: string, hintedIsGguf?: boolean): boolean {
  return Boolean(hintedIsGguf) || GGUF_SUFFIX_RE.test(id);
}

export function isMlxId(id: string): boolean {
  return MLX_RE.test(id);
}

// "mobile" build token (e.g. "gemma-4-E4B-it-qat-mobile-GGUF"); bounded so it never matches inside a longer word.
const MOBILE_RE = /(?:^|[-_/. ])mobile(?:$|[-_/. ])/i;

/** A mobile-targeted build, which we keep out of the Recommended list. */
export function isMobileVariant(id: string): boolean {
  return MOBILE_RE.test(id);
}

/** What Recommended may suggest: GGUF anywhere; on Mac also MLX and safetensors. GPU keeps
 *  GGUF-only recommendations. */
export function isRecommendableFormat(
  id: string,
  hintedIsGguf: boolean | undefined,
  isMac: boolean,
): boolean {
  if (isGgufId(id, hintedIsGguf)) return true;
  return isMac;
}

/** Format filter for the listing toggle. "safetensors" means anything that is neither GGUF nor MLX. */
export type FormatFilter = "all" | "gguf" | "mlx" | "safetensors";

export function matchesFormatFilter(
  id: string,
  hintedIsGguf: boolean | undefined,
  filter: FormatFilter,
): boolean {
  switch (filter) {
    case "gguf":
      return isGgufId(id, hintedIsGguf);
    case "mlx":
      return isMlxId(id);
    case "safetensors":
      return !isGgufId(id, hintedIsGguf) && !isMlxId(id);
    default:
      return true;
  }
}

// First "<n>B" token in a repo id, e.g. "Qwen3-30B-A3B" -> 30. Digits must be
// separator-bounded so "16" is never read from "bf16".
const PARAM_RE = /(?:^|[-_/. ])[eE]?(\d+(?:\.\d+)?)\s*[bB](?=$|[-_./ ])/;

/** Parameter count parsed from a repo id, or undefined when it has no size token, so callers
 *  can treat the size as unknown. */
export function paramsFromId(id: string): number | undefined {
  const match = PARAM_RE.exec(id);
  if (!match) return undefined;
  const billions = Number.parseFloat(match[1]);
  return Number.isFinite(billions) && billions > 0 ? billions * 1e9 : undefined;
}

// Smallest practical GGUF/MLX quant (~Q2_K). The fit check asks whether a model can run at
// all, so it uses this rather than a default 4-bit size.
const MIN_QUANT_BYTES_PER_PARAM = 0.4;

/** Rough on-disk bytes for the smallest practical quant of `params` weights. */
export function estimateQuantBytes(params: number): number {
  return params * MIN_QUANT_BYTES_PER_PARAM;
}

/** A model fits when it can run at all: `classifyGgufFit` short of `oom`, so a partial CPU
 *  offload counts. Shares the loader's formula with the Hub badge and the quant rows, since
 *  this predicate ALSO gates the "Fits on device" filter. Unknown device means we cannot
 *  tell, so treat it as fitting. Unknown size normally fits too, but Recommended passes
 *  `requireKnown` so an unsizable model is hidden rather than wrongly shown. */
export function fitsDevice(opts: {
  sizeBytes?: number;
  gpuGb?: number;
  systemRamGb?: number;
  budgetKnown?: boolean;
  requireKnown?: boolean;
  budgetFraction?: number;
  /** How many GPUs gpuGb sums, so the gate charges the loader's per-card VRAM reserve the same
   *  number of times the badge does. Absent means one. */
  gpuCount?: number;
  /** Images / Video: the row is placed by the diffusion backend, not llama-server, so it takes
   *  the media rule the quant rows use. Applies to every format on those pages. */
  mediaLoad?: boolean;
  /** The load device's memory is a window into host RAM, so RAM is not a second budget to add.
   *  Media rule only; the llama.cpp one already takes a RAM figure with the pool removed. */
  hostPooledMemory?: boolean;
}): boolean {
  const {
    sizeBytes,
    gpuGb,
    systemRamGb,
    budgetKnown,
    requireKnown,
    budgetFraction,
    gpuCount,
    mediaLoad,
    hostPooledMemory,
  } = opts;
  // Unified-memory hosts report system RAM but no GPU, so the budget must include RAM. Only an
  // entirely unknown budget fits freely.
  const anyBudget =
    Math.max(0, gpuGb ?? 0) > 0 || Math.max(0, systemRamGb ?? 0) > 0;
  if (!anyBudget) return !budgetKnown;
  if (sizeBytes && sizeBytes > 0) {
    if (mediaLoad) {
      // No RAM tier on a host pool: diffusion offload moves bytes inside that one pool and frees
      // nothing. llama.cpp keeps its tier, since a GGUF really does spill into host RAM.
      return (
        classifyMediaGgufFit(
          sizeBytes,
          gpuGb ?? 0,
          hostPooledMemory ? 0 : (systemRamGb ?? 0),
        ) !== "oom"
      );
    }
    return (
      classifyGgufFit(sizeBytes, {
        gpuGb,
        systemRamGb,
        budgetFraction,
        gpuCount,
      }) !== "oom"
    );
  }
  return requireKnown ? false : true;
}

/** Fit predicate for one Hub listing row, shared by the chat model selector and the Hub page
 *  filter. GGUF repos use metadata size or the smallest-quant estimate. Safetensors / MLX
 *  always use the params-based estimate, matching the badge's quantized-load assumption,
 *  since their estimatedSizeBytes is the full-precision checkpoint. `curatedSizeBytes`
 *  outranks both. Anything still unsizable is hidden; an unknown device budget keeps all. */
export function hfModelFitsDevice(
  model: {
    id: string;
    totalParams?: number;
    estimatedSizeBytes?: number;
    curatedSizeBytes?: number;
    isGguf?: boolean;
  },
  gpu: {
    memoryTotalGb: number;
    systemRamAvailableGb: number;
    budgetKnown?: boolean;
  },
  /** `budgetFraction` is the user's saved VRAM Budget: omitted scores against the loader's
   *  default, so a caller that forgets it judges rows on a replaced budget. `mediaLoad` picks
   *  the diffusion rule for an Images / Video row. */
  opts: {
    budgetFraction?: number;
    gpuCount?: number;
    mediaLoad?: boolean;
    hostPooledMemory?: boolean;
  } = {},
): boolean {
  if (
    gpu.memoryTotalGb <= 0 &&
    gpu.systemRamAvailableGb <= 0 &&
    !gpu.budgetKnown
  )
    return true;
  const params = model.totalParams ?? paramsFromId(model.id);
  const quantBytes = params ? estimateQuantBytes(params) : undefined;
  const sizeBytes =
    model.curatedSizeBytes ??
    (isGgufId(model.id, model.isGguf)
      ? (model.estimatedSizeBytes ?? quantBytes)
      : (quantBytes ?? model.estimatedSizeBytes));
  return fitsDevice({
    sizeBytes,
    gpuGb: gpu.memoryTotalGb,
    systemRamGb: gpu.systemRamAvailableGb,
    budgetKnown: gpu.budgetKnown,
    requireKnown: true,
    budgetFraction: opts.budgetFraction,
    gpuCount: opts.gpuCount,
    mediaLoad: opts.mediaLoad,
    hostPooledMemory: opts.hostPooledMemory,
  });
}

/** The budget a task-scoped (Images / Video) row may claim. Those loads put the whole pipeline
 *  on ONE device, so fit is judged against that card, never the multi-GPU sum, which would
 *  recommend a checkpoint that OOMs where the load lands. Chat keeps the sum. */
export function loadScopedGpu<
  T extends {
    available: boolean;
    memoryTotalGb: number;
    maxDeviceMemoryGb: number;
    loadDeviceMemoryGb: number;
    loadDeviceSharedMemory?: boolean;
    loadDeviceSharesHostMemory?: boolean;
    systemRamAvailableGb: number;
    systemRamAvailableHostGb?: number;
    deviceCount?: number;
  },
>(gpu: T, taskScoped: boolean): T & { deviceCount?: number } {
  if (!taskScoped || !gpu.available) return gpu;
  const deviceGb = gpu.loadDeviceMemoryGb || gpu.maxDeviceMemoryGb;
  if (deviceGb <= 0) return gpu;
  return {
    ...gpu,
    memoryTotalGb: deviceGb,
    // Narrowed with the capacity it describes, or the per-card VRAM reserve gets charged once per
    // HOST GPU against a ONE-card budget. Two 24 GiB cards at 1.0 scored an audio quant
    // against 23.28 GiB where the loader offers the selected card's 23.5.
    deviceCount: 1,
    // The raw-host figure is the RAM a DEDICATED task device may claim back from a shared GPU's
    // reservation. Gated on the folded flag, not shared_memory, which is Windows-only: a Linux
    // ROCm APU took this branch and undid the subtraction keeping its GTT window out.
    systemRamAvailableGb: hostPooledLoadDevice(gpu)
      ? gpu.systemRamAvailableGb
      : (gpu.systemRamAvailableHostGb ?? gpu.systemRamAvailableGb),
  };
}

/** Whether the device a task load lands on draws from host RAM. Prefers the folded flag, which
 *  counts a Linux APU reporting unified_memory without shared_memory. */
function hostPooledLoadDevice(gpu: {
  loadDeviceSharedMemory?: boolean;
  loadDeviceSharesHostMemory?: boolean;
}): boolean {
  return gpu.loadDeviceSharesHostMemory ?? gpu.loadDeviceSharedMemory === true;
}

/** One fit predicate for both search lists. The curated list only suppresses ids it kept, so a
 *  row it drops reappears from the Hub list: judge both on the same size and budget, or the
 *  toggle leaks an oversized row. */
export function searchRowFitsDevice<
  G extends {
    available: boolean;
    memoryTotalGb: number;
    maxDeviceMemoryGb: number;
    loadDeviceMemoryGb: number;
    systemRamAvailableGb: number;
    budgetKnown?: boolean;
    deviceCount?: number;
  },
>(
  row: {
    id: string;
    totalParams?: number;
    estimatedSizeBytes?: number;
    curatedSizeBytes?: number;
  },
  opts: {
    isGguf: boolean;
    curatedSizeBytes?: number;
    gpu: G;
    inferenceGpu: G;
    taskScoped: boolean;
    /** Images / Video only. `taskScoped` picks the single-device budget for every task page; this
     *  picks the diffusion RULE, which Audio must not get: its GGUFs run under llama.cpp. */
    diffusionLoad?: boolean;
    budgetFraction?: number;
    /** How many GPUs the aggregate sums, when the caller's inventory does not carry the count.
     *  loadScopedGpu narrows it to 1 with the capacity, so a task page needs nothing here. */
    gpuCount?: number;
    /** The image/video load device's pool is host RAM, so RAM is not a second budget. */
    hostPooledMemory?: boolean;
  },
): boolean {
  const source = loadScopedGpu(
    opts.diffusionLoad || !opts.isGguf ? opts.gpu : opts.inferenceGpu,
    opts.taskScoped,
  );
  return hfModelFitsDevice(
    {
      ...row,
      isGguf: opts.isGguf,
      curatedSizeBytes: row.curatedSizeBytes ?? opts.curatedSizeBytes,
    },
    source,
    // A task-scoped row is a media load, so it takes the media rule as well as the single-device
    // budget. Without this the search gate disagreed with the quant rows it gates.
    {
      budgetFraction: opts.budgetFraction,
      // From the SCOPED budget, so the count always describes the capacity beside it.
      gpuCount: source.deviceCount ?? opts.gpuCount,
      mediaLoad: opts.diffusionLoad,
      hostPooledMemory: opts.hostPooledMemory,
    },
  );
}

/** The id pool the Recommended SEARCH matches against: curated seeds first, then listing ids,
 *  each id once. The listing pool drops ids already on disk, since a downloaded model has
 *  its own On Device row; the seed pool does not, because the unfiltered list keeps painting
 *  a downloaded curated model. Without this a curated pick disappears from search once
 *  downloaded, and only a live Hub listing row could bring it back. */
export function searchableRecommendedIds(
  seedIds: readonly string[],
  listingIds: readonly string[],
): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const id of [...seedIds, ...listingIds]) {
    const key = id.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(id);
  }
  return out;
}

/** Order Recommended: curated seeds first in catalog order, then the rest of the listing, each
 *  id once. A seed hands off only to a row that survived `keep`, so a painted curated row
 *  does not vanish when the listing reports it with rejected metadata. The taking-over row
 *  inherits the seed's curated size, or a prequantized artifact would flip to the params
 *  guess, which assumes a quant still to come. */
export function orderRecommendedRows<
  T extends { id: string; curatedSizeBytes?: number },
>(opts: {
  seeds: readonly T[];
  results: readonly T[];
  keep: (row: T) => boolean;
  deviceFiltered: boolean;
  fits: (row: T) => boolean;
}): T[] {
  const { seeds, results, keep, deviceFiltered, fits } = opts;
  const seedById = new Map(seeds.map((s) => [s.id, s]));
  const rows = results.filter(keep).map((row) => {
    const curatedSizeBytes = seedById.get(row.id)?.curatedSizeBytes;
    return curatedSizeBytes != null && row.curatedSizeBytes == null
      ? { ...row, curatedSizeBytes }
      : row;
  });
  const byId = new Map(rows.map((r) => [r.id, r]));
  const curated: T[] = [];
  for (const seed of seeds) {
    const row = byId.get(seed.id) ?? seed;
    if (!deviceFiltered || fits(row)) curated.push(row);
  }
  const curatedIds = new Set(curated.map((r) => r.id));
  const rest = (deviceFiltered ? rows.filter(fits) : rows).filter(
    (r) => !curatedIds.has(r.id),
  );
  return [...curated, ...rest];
}
