// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { classifyGgufFit } from "../../../../lib/gguf-fit.ts";
import {
  LOADING_OVERHEAD_GB,
  checkVramFit,
  estimateLoadingVram,
} from "../../../../lib/vram.ts";
import {
  audioCppDisplayName,
  isAudioCppFolderId,
} from "../../../audio/audio-cpp-catalog.ts";
import { matchesRecommended } from "../../../hub/lib/format-filters.ts";
import {
  classifyMediaGgufFit,
  type curatedArtifactFit,
} from "./model-catalog.ts";

const GGUF_SUFFIX_RE = /-GGUF(?:$|-)/i;
// Mirrors the backend's _looks_like_mlx_repo.
const MLX_RE = /(?:^|[-_.])mlx(?:$|[-_.])/i;
const MLX_OWNER_PREFIX = "mlx-community/";
// Callers pass a filesystem path, which may be Windows.
const PATH_SEP_RE = /[\\/]/;

export function isGgufId(id: string, hintedIsGguf?: boolean): boolean {
  return Boolean(hintedIsGguf) || GGUF_SUFFIX_RE.test(id);
}

export function isMlxId(id: string): boolean {
  const trimmed = id.trim();
  if (trimmed.toLowerCase().startsWith(MLX_OWNER_PREFIX)) return true;
  const leaf = trimmed.split(PATH_SEP_RE).filter(Boolean).at(-1) ?? trimmed;
  return MLX_RE.test(leaf);
}

const MOBILE_RE = /(?:^|[-_/. ])mobile(?:$|[-_/. ])/i;

export function isMobileVariant(id: string): boolean {
  return MOBILE_RE.test(id);
}

/** GGUF anywhere; on Mac also MLX and safetensors. */
export function isRecommendableFormat(
  id: string,
  hintedIsGguf: boolean | undefined,
  isMac: boolean,
): boolean {
  if (isGgufId(id, hintedIsGguf)) return true;
  return isMac;
}

/** An FP8 / NVFP4 checkpoint (not a GGUF) this host's GPUs run natively. */
export function isCapableCheckpointQuant(
  id: string,
  quantMethod: string | undefined,
  checkpointQuantFormats: readonly string[],
  hintedIsGguf?: boolean,
): boolean {
  return (
    !isGgufId(id, hintedIsGguf) &&
    matchesRecommended(
      { id, isGguf: false, quantMethod },
      checkpointQuantFormats,
    )
  );
}

/** Format filter for the listing toggle. "safetensors" means anything that is neither GGUF nor MLX. */
export type FormatFilter = "all" | "gguf" | "mlx" | "safetensors" | "npu";

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
    case "npu":
      return false;
    default:
      return true;
  }
}

/** The format filters a picker offers, in menu order. MLX only where it can run, NPU only with a
 *  catalog to show. */
export function formatFilterValues(opts: { npu: boolean; mlx: boolean }): FormatFilter[] {
  return (["all", "gguf", "mlx", "safetensors", "npu"] as const).filter(
    (value) => (value !== "mlx" || opts.mlx) && (value !== "npu" || opts.npu),
  );
}

/** A remembered filter the picker no longer offers falls back to "all". */
export function coerceFormatFilter(
  chosen: FormatFilter,
  offered: readonly FormatFilter[],
): FormatFilter {
  return offered.includes(chosen) ? chosen : "all";
}

const normalizeSearchText = (s: string): string =>
  s.toLowerCase().replace(/[\s_.-]/g, "");

/** Whether a picker search matches a row on what the row shows. A bundled folder shows only its
 *  folder name, so its package repo id matches nothing, and neither does a query that is just
 *  part of the -GGUF every folder name ends in. */
export function pickerSearchMatches(
  id: string,
  aliases: readonly string[],
  label: string | null | undefined,
  query: string,
): boolean {
  const q = normalizeSearchText(query);
  if (!q) return true;
  if (isAudioCppFolderId(id)) {
    const name = audioCppDisplayName(id);
    return (
      normalizeSearchText(name.replace(/-gguf$/i, "")).includes(q) ||
      (!"gguf".includes(q) && normalizeSearchText(name).includes(q))
    );
  }
  return (label ? [...aliases, label] : aliases).some((term) =>
    normalizeSearchText(term).includes(q),
  );
}

// First "<n>B" token in a repo id, e.g. "Qwen3-30B-A3B" -> 30. Digits must be
// separator-bounded so "16" is never read from "bf16".
const PARAM_RE = /(?:^|[-_/. ])[eE]?(\d+(?:\.\d+)?)\s*[bB](?=$|[-_./ ])/;

export function paramsFromId(id: string): number | undefined {
  const match = PARAM_RE.exec(id);
  if (!match) return undefined;
  const billions = Number.parseFloat(match[1]);
  return Number.isFinite(billions) && billions > 0 ? billions * 1e9 : undefined;
}

// ~Q2_K: the fit check asks whether a model can run at all.
const MIN_QUANT_BYTES_PER_PARAM = 0.4;

export function estimateQuantBytes(params: number): number {
  return params * MIN_QUANT_BYTES_PER_PARAM;
}

/** Fits when classifyGgufFit is short of `oom` (CPU offload counts); also gates "Fits on device".
  Unknown device fits; unknown size fits unless `requireKnown`. */
export function fitsDevice(opts: {
  sizeBytes?: number;
  gpuGb?: number;
  systemRamGb?: number;
  budgetKnown?: boolean;
  requireKnown?: boolean;
  budgetFraction?: number;
  /** GPUs summed in gpuGb, so the per-card VRAM reserve is charged as the badge does. Default 1. */
  gpuCount?: number;
  /** Images / Video: diffusion backend placement, so the media rule applies. */
  mediaLoad?: boolean;
  /** Load device memory is host RAM, so RAM is not a second budget. Media rule only. */
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
  // Unified-memory hosts report RAM but no GPU; only an entirely unknown budget fits freely.
  const anyBudget =
    Math.max(0, gpuGb ?? 0) > 0 || Math.max(0, systemRamGb ?? 0) > 0;
  if (!anyBudget) return !budgetKnown;
  if (sizeBytes && sizeBytes > 0) {
    if (mediaLoad) {
      // No RAM tier on a host pool: diffusion offload frees nothing there, unlike a GGUF spill.
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

/** A safetensors load on a discrete GPU: a prequantized checkpoint as stored (Studio rebuilds none
 *  of them as bnb), anything else at the row badge's 4-bit estimate. `raise_if_offloaded` refuses
 *  CPU offload, except that one CUDA / ROCm card block-swaps 16-bit or bnb layers into host RAM. */
function fitsTransformersLoad(
  model: { totalParams: number; estimatedSizeBytes?: number; quantMethod?: string },
  gpu: { memoryTotalGb: number; systemRamAvailableGb: number; backend?: string },
  gpuCount = 1,
): boolean {
  const quant = model.quantMethod?.toLowerCase();
  const swapGb =
    gpuCount <= 1 && (gpu.backend === "cuda" || gpu.backend === "rocm")
      ? Math.max(0, gpu.systemRamAvailableGb)
      : 0;
  const fits = (requiredGb: number, budgetGb: number) =>
    checkVramFit(requiredGb, budgetGb) !== "exceeds";
  if (!quant)
    return fits(estimateLoadingVram(model.totalParams, "qlora"), gpu.memoryTotalGb + swapGb);
  const storedBytes =
    model.estimatedSizeBytes ?? estimateQuantBytes(model.totalParams);
  return fits(
    storedBytes / 1024 ** 3 + LOADING_OVERHEAD_GB,
    gpu.memoryTotalGb + (quant === "bitsandbytes" ? swapGb : 0),
  );
}

/** Fit predicate for one Hub listing row, shared by the chat model selector and the Hub page
 *  filter. GGUF repos use metadata size or the smallest-quant estimate, safetensors on a discrete
 *  GPU `fitsTransformersLoad`, and MLX / host-pool rows the params-based quant estimate.
 *  `curatedSizeBytes` outranks all. Unsizable rows are hidden; an unknown budget keeps all. */
export function hfModelFitsDevice(
  model: {
    id: string;
    totalParams?: number;
    estimatedSizeBytes?: number;
    curatedSizeBytes?: number;
    isGguf?: boolean;
    quantMethod?: string;
  },
  gpu: {
    memoryTotalGb: number;
    systemRamAvailableGb: number;
    budgetKnown?: boolean;
    backend?: string;
  },
  /** Omitted `budgetFraction` scores against the loader default, not the user's saved budget. */
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
  const isGguf = isGgufId(model.id, model.isGguf);
  if (
    params &&
    model.curatedSizeBytes == null &&
    !isGguf &&
    !isMlxId(model.id) &&
    !opts.mediaLoad &&
    !opts.hostPooledMemory &&
    gpu.memoryTotalGb > 0
  )
    return fitsTransformersLoad({ ...model, totalParams: params }, gpu, opts.gpuCount);
  const quantBytes = params ? estimateQuantBytes(params) : undefined;
  const sizeBytes =
    model.curatedSizeBytes ??
    (isGguf
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

/** Task loads put the whole pipeline on ONE device, so fit uses that card, not the multi-GPU sum. */
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
    // Narrowed with the capacity, or the per-card reserve is charged once per host GPU.
    deviceCount: 1,
    // Gated on the folded flag, not Windows-only shared_memory, so a Linux ROCm APU keeps its GTT subtracted.
    systemRamAvailableGb: hostPooledLoadDevice(gpu)
      ? gpu.systemRamAvailableGb
      : (gpu.systemRamAvailableHostGb ?? gpu.systemRamAvailableGb),
  };
}

/** Prefers the folded flag, which counts a Linux APU with unified_memory but no shared_memory. */
function hostPooledLoadDevice(gpu: {
  loadDeviceSharedMemory?: boolean;
  loadDeviceSharesHostMemory?: boolean;
}): boolean {
  return gpu.loadDeviceSharesHostMemory ?? gpu.loadDeviceSharedMemory === true;
}

/** Both search lists must judge rows identically, or the curated toggle leaks an oversized row. */
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
    quantMethod?: string;
  },
  opts: {
    isGguf: boolean;
    curatedSizeBytes?: number;
    gpu: G;
    inferenceGpu: G;
    taskScoped: boolean;
    /** Images / Video only: Audio GGUFs run under llama.cpp and must not get the diffusion rule. */
    diffusionLoad?: boolean;
    budgetFraction?: number;
    gpuCount?: number;
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
    // Task-scoped rows take the media rule too, so the search gate matches the quant rows.
    {
      budgetFraction: opts.budgetFraction,
      gpuCount: source.deviceCount ?? opts.gpuCount,
      mediaLoad: opts.diffusionLoad,
      hostPooledMemory: opts.hostPooledMemory,
    },
  );
}

/** Curated seeds first, then listing ids. Only the listing pool drops on-disk ids, so a
  downloaded curated pick stays searchable. */
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

/** Curated seeds first, then the listing. A seed hands off only to a row that survived `keep`,
  and that row inherits the seed's curated size. */
export function orderRecommendedRows<
  T extends { id: string; curatedSizeBytes?: number },
>(opts: {
  seeds: readonly T[];
  results: readonly T[];
  keep: (row: T) => boolean;
  deviceFiltered: boolean;
  fits: (row: T) => boolean;
  /** `results` must be one sorted listing of unsloth/* repos, since a family ranks by its index. */
  familyOf?: (id: string) => string | undefined;
  /** Family keys (as returned by `familyOf`) that lead the list in this order, whatever the sort.
   *  Their rows interleave by position: every pinned family's first row, then every second one. */
  pinnedFamilies?: readonly string[];
  /** Pinned family keys whose rows stay together instead of interleaving: all of them come before
   *  the next pinned family's, so a GGUF sits right under its own lead row. */
  pinnedTogether?: readonly string[];
}): T[] {
  const {
    seeds,
    results,
    keep,
    deviceFiltered,
    fits,
    familyOf,
    pinnedFamilies = [],
    pinnedTogether = [],
  } = opts;
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
  const ordered = [...curated, ...rest];
  if (!familyOf) return ordered;
  // Unlisted families go last, except first-party unslothai/* ones the listing never returns.
  const keyOf = (r: T) => familyOf(r.id) ?? r.id.toLowerCase();
  const firstParty = (id: string) => /^unsloth(ai)?\//i.test(id);
  const listable = new Set(
    ordered.filter((r) => /^unsloth\//i.test(r.id)).map(keyOf),
  );
  const rank = new Map<string, number>();
  results.forEach((r, i) => {
    const key = keyOf(r);
    if (!rank.has(key)) rank.set(key, i);
  });
  const firstSeen = new Map<string, number>();
  ordered.forEach((r, i) => {
    const key = keyOf(r);
    if (!firstSeen.has(key)) firstSeen.set(key, i);
  });
  const pinIndex = new Map(pinnedFamilies.map((key, i) => [key, i]));
  const together = new Set(pinnedTogether);
  // Row position within its family: pinned families' lead rows ("Fast FP8") come before their GGUFs.
  const slotOf = new Map<number, number>();
  const seenPerFamily = new Map<string, number>();
  ordered.forEach((r, i) => {
    const key = keyOf(r);
    const slot = seenPerFamily.get(key) ?? 0;
    slotOf.set(i, slot);
    seenPerFamily.set(key, slot + 1);
  });
  const sortKey = (r: T, i: number) => {
    const key = keyOf(r);
    // Pinned families lead, ahead of first-party rows that trend higher.
    const pin = pinIndex.get(key);
    if (pin != null) {
      const slot = slotOf.get(i) ?? 0;
      return together.has(key)
        ? [0, pin, slot, firstSeen.get(key) ?? i, i]
        : [slot, pin, 0, firstSeen.get(key) ?? i, i];
    }
    const ours = firstParty(r.id);
    const unranked = ours && !listable.has(key) ? -1 : Infinity;
    return [
      Number.POSITIVE_INFINITY,
      ours ? 0 : 1,
      rank.get(key) ?? unranked,
      firstSeen.get(key) ?? i,
      i,
    ];
  };
  return ordered
    .map((row, i) => ({ row, key: sortKey(row, i) }))
    .sort((a, b) => {
      for (let k = 0; k < a.key.length; k++) {
        if (a.key[k] !== b.key[k]) return a.key[k] < b.key[k] ? -1 : 1;
      }
      return 0;
    })
    .map(({ row }) => row);
}

export type CuratedBudget = {
  allowanceGb: number;
  deviceGb: number;
  device: "GPU" | "RAM" | "GPU + RAM";
  sizeGb: number;
};

export function curatedBudget(
  fit: Omit<NonNullable<ReturnType<typeof curatedArtifactFit>>, "device"> & { device?: CuratedBudget["device"] },
): CuratedBudget | undefined {
  const { allowanceGb, deviceGb, device, sizeGb } = fit;
  return allowanceGb != null && deviceGb != null && device && sizeGb != null
    ? { allowanceGb, deviceGb, device, sizeGb }
    : undefined;
}

/** Over-budget text for a curated row. When the whole-GB badge figure does not read above the
 *  one-decimal budget (24 against 24.1), the size is rounded up and the budget down to a tenth, so
 *  the shown size is always strictly above the shown budget. The epsilon keeps 16.7999... at 16.8. */
export function curatedBudgetText(
  est: number,
  gpuGb: number,
  budget: CuratedBudget,
): string {
  const shownBudget = Number(budget.allowanceGb.toFixed(1));
  const wholeReadsOver = est > shownBudget;
  const needGb = wholeReadsOver
    ? `${est}`
    : (Math.ceil(budget.sizeGb * 10 - 1e-9) / 10).toFixed(1);
  const budgetGb = wholeReadsOver
    ? shownBudget.toFixed(1)
    : (Math.floor(budget.allowanceGb * 10 + 1e-9) / 10).toFixed(1);
  const of =
    budget.device === "GPU + RAM"
      ? `${Number(budget.deviceGb.toFixed(2))}GB combined GPU and available RAM`
      : budget.device === "RAM"
      ? `${Number(budget.deviceGb.toFixed(2))}GB available RAM`
      : `a ${gpuGb}GB GPU`;
  return `Needs ~${needGb}GB for weights (budget: ~${budgetGb}GB, 70% of ${of})`;
}

export function recommendedEmptyState({
  isLoading,
  error,
  hubPhase,
}: {
  isLoading: boolean;
  error: string | null;
  hubPhase: "available" | "probing" | "unavailable";
}): "loading" | "failed" | "empty" {
  if (isLoading) return "loading";
  return error || hubPhase !== "available" ? "failed" : "empty";
}
