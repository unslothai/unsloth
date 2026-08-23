// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  estimateQuantBytes,
  fitsDevice,
  hfModelFitsDevice,
  isGgufId,
  paramsFromId,
} from "../../../../lib/model-device-fit.ts";

export {
  estimateQuantBytes,
  fitsDevice,
  hfModelFitsDevice,
  isGgufId,
  paramsFromId,
};

const MLX_RE = /-MLX(?:$|-)/i;

export function isMlxId(id: string): boolean {
  return MLX_RE.test(id);
}

// "mobile" build token (e.g. "gemma-4-E4B-it-qat-mobile-GGUF"); bounded so it
// never matches inside a longer word.
const MOBILE_RE = /(?:^|[-_/. ])mobile(?:$|[-_/. ])/i;

/** A mobile-targeted build, which we keep out of the Recommended list. */
export function isMobileVariant(id: string): boolean {
  return MOBILE_RE.test(id);
}

/** What Recommended is allowed to suggest: GGUF anywhere; on Mac also MLX and
 * safetensors (both now run locally there). GPU keeps GGUF-only recommendations. */
export function isRecommendableFormat(
  id: string,
  hintedIsGguf: boolean | undefined,
  isMac: boolean,
): boolean {
  if (isGgufId(id, hintedIsGguf)) return true;
  return isMac;
}

/** Format filter for the listing toggle. "safetensors" means anything that is
 * neither GGUF nor MLX. */
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

/** The budget a task-scoped (Images / Video) row may claim. Those loads put the
 * whole pipeline on ONE device (a bare "cuda", the lowest visible ordinal), so
 * fit is judged against that card, never the multi-GPU sum, which would
 * recommend a checkpoint that OOMs where the load lands. Chat keeps the sum. */
export function loadScopedGpu<
  T extends {
    available: boolean;
    memoryTotalGb: number;
    maxDeviceMemoryGb: number;
    loadDeviceMemoryGb: number;
  },
>(gpu: T, taskScoped: boolean): T {
  if (!taskScoped || !gpu.available) return gpu;
  const deviceGb = gpu.loadDeviceMemoryGb || gpu.maxDeviceMemoryGb;
  return deviceGb > 0 ? { ...gpu, memoryTotalGb: deviceGb } : gpu;
}

/** One fit predicate for both search lists (curated matches and the Hub rows
 * below them). The curated list only suppresses ids it kept, so a row it drops
 * reappears from the Hub list: judge both on the same size and budget, or the
 * toggle leaks an oversized row. */
export function searchRowFitsDevice<
  G extends {
    available: boolean;
    memoryTotalGb: number;
    maxDeviceMemoryGb: number;
    loadDeviceMemoryGb: number;
    systemRamAvailableGb: number;
    budgetKnown?: boolean;
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
  },
): boolean {
  const source = opts.isGguf ? opts.inferenceGpu : opts.gpu;
  return hfModelFitsDevice(
    {
      ...row,
      isGguf: opts.isGguf,
      curatedSizeBytes: row.curatedSizeBytes ?? opts.curatedSizeBytes,
    },
    loadScopedGpu(source, opts.taskScoped),
  );
}

/** The id pool the Recommended SEARCH matches a query against: the curated seeds the
 * unfiltered list paints, then the listing ids, each id once and seeds first (the
 * order `orderRecommendedRows` renders them in).
 *
 * The listing pool drops every id already on disk, because a downloaded model has its
 * own On Device row. The seed pool does not, and the unfiltered Recommended list keeps
 * painting a downloaded curated model (badged "downloaded"), so search has to keep it
 * too. Without this a curated pick disappears from Recommended search the moment it is
 * downloaded, and the only thing that can bring it back is a live Hub listing row --
 * which a repo the listing does not return (new, non-unsloth owner, invisible to this
 * account) never gets. The row then sits in the unfiltered list yet cannot be found by
 * typing its name. */
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

/** Order Recommended: curated seeds first in catalog order, then the rest of the
 * listing, each id once. A seed hands off only to a row that survived `keep`, so
 * a painted curated row does not vanish when the listing reports it with
 * metadata the filters reject. Fit is judged on whichever row renders, and the
 * taking-over row inherits the seed's curated size: a Hub listing carries none,
 * so a prequantized artifact would otherwise flip to the params guess (which
 * assumes a quant still to come). */
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
