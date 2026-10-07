// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { classifyGgufFit } from "../../../../lib/gguf-fit.ts";
import { classifyMediaGgufFit, type curatedArtifactFit } from "./model-catalog.ts";

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

/** "safetensors" means anything that is neither GGUF nor MLX. */
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

// Separator-bounded so "16" is never read from "bf16".
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

/** Shared by the chat selector and Hub filter. Safetensors / MLX use the params estimate since
  their estimatedSizeBytes is full precision; `curatedSizeBytes` outranks both. */
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
  pinnedFamilies?: readonly string[];
}): T[] {
  const { seeds, results, keep, deviceFiltered, fits, familyOf, pinnedFamilies = [] } = opts;
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
  const sortKey = (r: T, i: number) => {
    const key = keyOf(r);
    const pin = pinIndex.get(key);
    if (pin != null) {
      return [pin, 0, 0, firstSeen.get(key) ?? i, i];
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
  device: "GPU" | "RAM";
  sizeGb: number;
};

export function curatedBudget(
  fit: NonNullable<ReturnType<typeof curatedArtifactFit>>,
): CuratedBudget | undefined {
  const { allowanceGb, deviceGb, device, sizeGb } = fit;
  return allowanceGb != null && deviceGb != null && device && sizeGb != null
    ? { allowanceGb, deviceGb, device, sizeGb }
    : undefined;
}

/** Rounds size up and budget down so the shown size is strictly above the shown budget. */
export function curatedBudgetText(est: number, gpuGb: number, budget: CuratedBudget): string {
  const shownBudget = Number(budget.allowanceGb.toFixed(1));
  const wholeReadsOver = est > shownBudget;
  const needGb = wholeReadsOver
    ? `${est}`
    : (Math.ceil(budget.sizeGb * 10 - 1e-9) / 10).toFixed(1);
  const budgetGb = wholeReadsOver
    ? shownBudget.toFixed(1)
    : (Math.floor(budget.allowanceGb * 10 + 1e-9) / 10).toFixed(1);
  const of =
    budget.device === "RAM"
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
