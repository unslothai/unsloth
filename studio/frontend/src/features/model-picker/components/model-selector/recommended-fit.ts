


// Pure helpers for the Recommended list: which formats to surface and whether a
// model fits the device. No React/DOM deps so they are easy to test.

const GGUF_SUFFIX_RE = /-GGUF(?:$|-)/i;
const MLX_RE = /-MLX(?:$|-)/i;

export function isGgufId(id: string, hintedIsGguf?: boolean): boolean {
  return Boolean(hintedIsGguf) || GGUF_SUFFIX_RE.test(id);
}

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

// First "<n>B" token in a repo id, e.g. "Qwen3-30B-A3B" -> 30 (MoE total),
// "gemma-4-E4B" -> 4. Digits must be separator-bounded so we never read "16"
// from "bf16" or the "2" in "Kimi-K2".
const PARAM_RE = /(?:^|[-_/. ])[eE]?(\d+(?:\.\d+)?)\s*[bB](?=$|[-_./ ])/;

/** Parameter count (absolute, e.g. 4e9) parsed from a repo id, or undefined
 * when the id has no size token (so callers can treat the size as unknown). */
export function paramsFromId(id: string): number | undefined {
  const match = PARAM_RE.exec(id);
  if (!match) return undefined;
  const billions = parseFloat(match[1]);
  return Number.isFinite(billions) && billions > 0 ? billions * 1e9 : undefined;
}

// Smallest practical GGUF/MLX quant (~Q2_K). The fit check asks whether a model
// can run at all, so it uses this rather than a default 4-bit size.
const MIN_QUANT_BYTES_PER_PARAM = 0.4;

/** Rough on-disk bytes for the smallest practical quant of `params` weights. */
export function estimateQuantBytes(params: number): number {
  return params * MIN_QUANT_BYTES_PER_PARAM;
}

/** A model fits when its on-disk size (or a precomputed VRAM estimate) is within
 * the device budget (0.7*GPU + 0.7*RAM). Unknown device means we cannot tell, so
 * treat it as fitting. Unknown size normally fits too, but Recommended passes
 * `requireKnown` so a model we cannot size (e.g. a huge GGUF with no metadata or
 * size token) is hidden rather than wrongly shown. */
export function fitsDevice(opts: {
  sizeBytes?: number;
  estimatedVramGb?: number;
  gpuGb?: number;
  systemRamGb?: number;
  budgetKnown?: boolean;
  requireKnown?: boolean;
}): boolean {
  const {
    sizeBytes,
    estimatedVramGb,
    gpuGb,
    systemRamGb,
    budgetKnown,
    requireKnown,
  } = opts;
  // Unified-memory hosts (Mac / no discrete GPU) report system RAM but no GPU,
  // so the budget must include RAM. Only an entirely unknown budget fits freely.
  const budgetGb = Math.max(0, gpuGb ?? 0) * 0.7 + Math.max(0, systemRamGb ?? 0) * 0.7;
  if (budgetGb <= 0) return !budgetKnown;
  if (sizeBytes && sizeBytes > 0) {
    return sizeBytes / 1024 ** 3 <= budgetGb;
  }
  if (estimatedVramGb && estimatedVramGb > 0) {
    return estimatedVramGb <= budgetGb;
  }
  return requireKnown ? false : true;
}

/** Fit predicate for one Hub listing row, shared by the chat model selector
 * and the Hub page "Fits on device" filter. GGUF repos: metadata size (actual
 * weights) or the smallest-quant estimate from the param count. Safetensors /
 * MLX repos: always the params-based smallest-quant estimate, matching the
 * VRAM badge's quantized-load assumption; their estimatedSizeBytes is the
 * full-precision checkpoint and would wrongly hide models the quantized load
 * path can run. `curatedSizeBytes` outranks both: real data over estimates.
 * Anything still unsizable is hidden (requireKnown) so over-budget models with
 * no metadata don't slip through. An unknown device budget keeps everything. */
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
  });
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
