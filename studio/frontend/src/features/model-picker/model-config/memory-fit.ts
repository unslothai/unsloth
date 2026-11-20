// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Free of `@/` imports (relative, explicit extensions) so tests load it under
 *  `node --experimental-strip-types`. */

// Shared with the Hub memory bar via src/lib/memory/; re-exported for existing call sites.
export {
  classifyMemoryFit,
  worseMemoryFit,
  type MemoryFitVerdict,
} from "../../../lib/memory/verdict.ts";
export { MEMORY_FIT_TIGHT_RATIO } from "../../../lib/memory/thresholds.ts";

import {
  classifyMemoryFit,
  worseMemoryFit,
} from "../../../lib/memory/verdict.ts";
import type { MemoryFitVerdict } from "../../../lib/memory/verdict.ts";
import { formatBytesGiB } from "../../../lib/memory/format.ts";
import type {
  ReconciledGpuSelection,
  SystemGpuDevice,
} from "../../../hooks/gpu-selection.ts";
import { gpuMemoryTotalsGb, sharesHostMemory } from "../../../hooks/gpu-vram.ts";
import type { MlxOffload } from "./per-model-config.ts";

/** @deprecated Prefer `formatBytesGiB` from `@/lib/memory/format`. Bytes in, GiB out. */
export const formatMemoryGb = formatBytesGiB;

export function memoryFigureCandidates(
  bytes: number,
  bounded: boolean,
): string[] {
  const safe = Number.isFinite(bytes) && bytes > 0 ? bytes : 0;
  const prefix = bounded ? "≥ " : "";
  const candidates: string[] = bounded ? [] : [formatMemoryGb(safe)];
  for (const [index, unit] of ["GiB", "TiB", "PiB", "EiB"].entries()) {
    const amount = safe / 1024 ** (index + 3);
    if (index > 0 && amount < 1) break;
    for (const decimals of [2, 1, 0]) {
      const factor = 10 ** decimals;
      const rounded = bounded ? Math.floor(amount * factor) / factor : amount;
      candidates.push(`${prefix}${rounded.toFixed(decimals)} ${unit}`);
    }
  }
  return [...new Set(candidates)];
}

export interface MemoryAdvisory {
  tone: "warn" | "muted";
  text: string;
}

export interface MemoryFitEstimate {
  gpuBytes: number;
  totalBytes: number;
  /** False when the GGUF header could not size the cache, so the figures are a floor. */
  kvEstimable: boolean;
  drafterKvUnsized: boolean;
  adaptersUnsized: boolean;
  /** `--n-cpu-moe` is set, so the GPU figure ignores it and reads high. */
  moeOffloadUnmodelled: boolean;
  /** False when the loader will shrink the priced context to fit. Absent reads as pinned. */
  contextIsPinned?: boolean;
  gpuFloorBytes?: number | null;
  /** The context `gpuFloorBytes` is priced at. */
  gpuFloorCtx?: number | null;
  /** `totalBytes` at that context: on one pool, host terms such as checkpoints shrink with it. */
  floorTotalBytes?: number | null;
  /** [context, gpuBytes, totalBytes] at each doubling of the floor below native. */
  fitCurve?: [number, number, number][] | null;
  floorCanOffload?: boolean;
  nCtx?: number;
}

export interface MemoryFitCapacity {
  /** VRAM, or the shared pool where there is only one. 0 when unknown. */
  gpuCapacityGb: number;
  /** GPU plus host RAM. 0 when unknown. */
  totalCapacityGb: number;
  /** Bytes pinned outside the GPU must fit here; unused VRAM cannot help them. */
  systemRamCapacityGb: number;
  freeGpuCapacityGb: number;
  freeGpuCapacityKnown?: boolean;
  freeGpuReserveDeficitGb?: number;
  usableSystemRamGb: number;
  usableSystemRamKnown?: boolean;
  systemRamReserveDeficitGb?: number;
  /** GPU and host draw on the same memory, so an offloaded byte is not a freed one. */
  singleMemoryPool: boolean;
  /** Resident bytes freed by unload. Free-memory verdicts only. */
  reclaimableTotalBytes?: number;
  reclaimableGpuBytes?: number;
}

function reclaimableBytes(value: number | undefined): number {
  return Number.isFinite(value) && (value as number) > 0 ? (value as number) : 0;
}

export function resolveReclaimableMemoryCredit(
	estimate:
		| (Pick<MemoryFitEstimate, "totalBytes" | "gpuBytes"> & {
				weightsBytes: number;
				moeOffloadUnmodelled?: boolean;
		  })
		| null,
	residentPool: ReconciledGpuSelection,
	requestedPool: ReconciledGpuSelection,
	{
		cpuFallback = false,
		devices = [],
		gpuPlacementKnown = false,
		appleUnifiedMemory = false,
	}: {
		cpuFallback?: boolean;
		devices?: SystemGpuDevice[];
		gpuPlacementKnown?: boolean;
		appleUnifiedMemory?: boolean;
	} = {},
): { totalBytes: number; gpuBytes: number } {
	const total = reclaimableBytes(estimate?.totalBytes);
	const gpu = Math.min(reclaimableBytes(estimate?.gpuBytes), total);
	if (
		!estimate ||
		!Number.isFinite(estimate.weightsBytes) ||
		estimate.weightsBytes < 0
	)
		return { totalBytes: 0, gpuBytes: 0 };
	const files = Math.min(estimate.weightsBytes, total);
	const includesResidentPool =
		!requestedPool.ids?.length ||
		(residentPool.ids != null &&
			residentPool.ids.length > 0 &&
			residentPool.indexKind != null &&
			residentPool.indexKind === requestedPool.indexKind &&
			residentPool.ids.every((id) => requestedPool.ids!.includes(id)));
	const residentDevices = residentPool.ids?.length
		? devices.filter(
				(device) =>
					device.indexKind === residentPool.indexKind &&
					residentPool.ids!.includes(device.index),
			)
		: devices;
	const topologyKnown =
		residentDevices.length > 0 &&
		(!residentPool.ids?.length ||
			(residentPool.indexKind != null &&
				residentPool.ids.every((id) =>
					residentDevices.some((device) => device.index === id),
				))) &&
		residentDevices.every(
			(device) =>
				(sharesHostMemory(device) && device.sharedMemoryHostBackedGb == null) ||
				(Number.isFinite(device.memoryTotalGb) && device.memoryTotalGb > 0),
		);
	const independentGb = appleUnifiedMemory
		? 0
		: gpuMemoryTotalsGb(
				residentDevices.map((device) => ({
					memory_total_gb: device.memoryTotalGb,
					shared_memory: sharesHostMemory(device),
					shared_memory_host_backed_gb: device.sharedMemoryHostBackedGb,
				})),
			).dedicated;
	// File-backed pages may already count as available RAM; their pool split is unknown.
	const gpuMayUseHost =
		appleUnifiedMemory ||
		!topologyKnown ||
		residentDevices.some(sharesHostMemory);
	const gpuCredit =
		includesResidentPool &&
		gpuPlacementKnown &&
		!cpuFallback &&
		!estimate.moeOffloadUnmodelled
			? Math.max(0, gpu - (gpuMayUseHost ? files : 0))
			: 0;
	// Only bytes beyond all independent capacity are certainly backed by host RAM.
	const sharedHostCredit =
		gpuCredit === 0 && (appleUnifiedMemory || topologyKnown)
			? Math.max(0, gpu - independentGb * 1024 ** 3 - files)
			: 0;
	return {
		totalBytes: Math.max(0, total - gpu - files) + gpuCredit + sharedHostCredit,
		gpuBytes: gpuCredit,
	};
}

export interface MemoryFitResult {
  cpuOnly: boolean;
  rawGpuFit: MemoryFitVerdict;
  gpuFit: MemoryFitVerdict;
  freeGpuFit: MemoryFitVerdict;
  gpuPressured: boolean;
  hostShareBytes: number;
  usableHostFit: MemoryFitVerdict;
  hostPressured: boolean;
  /** "unknown" where there is only one pool. */
  hostShareFit: MemoryFitVerdict;
  totalFit: MemoryFitVerdict;
  bounded: boolean;
  prefix: string;
  advisory: MemoryAdvisory | null;
}

function hasContextFloor(estimate: MemoryFitEstimate): boolean {
  return estimate.contextIsPinned === false && Number.isFinite(estimate.gpuFloorBytes);
}

/** Use the GPU floor when Auto's native context exceeds capacity; host terms stay at native as an
 *  upper bound since the fitted context is unknown. */
function autoContextFigures(
  estimate: MemoryFitEstimate,
  capacity: MemoryFitCapacity,
  cpuOnly: boolean,
): { gpuBytes: number; totalBytes: number; contextShrinks: boolean } {
  const native = { gpuBytes: estimate.gpuBytes, totalBytes: estimate.totalBytes, contextShrinks: false };
  if (
    cpuOnly ||
    !hasContextFloor(estimate) ||
    !Number.isFinite(estimate.gpuBytes) ||
    !Number.isFinite(estimate.totalBytes)
  ) {
    return native;
  }
  const nativeFit = capacity.singleMemoryPool
    ? classifyMemoryFit(estimate.totalBytes, capacity.totalCapacityGb)
    : classifyMemoryFit(estimate.gpuBytes, capacity.gpuCapacityGb);
  if (nativeFit !== "exceeds") {
    return native;
  }
  const shrink = Math.max(0, estimate.gpuBytes - (estimate.gpuFloorBytes as number));
  return {
    gpuBytes: estimate.gpuBytes - shrink,
    totalBytes: estimate.totalBytes - shrink,
    contextShrinks: true,
  };
}

/** The backend refuses against AVAILABLE memory, not physical; these free-memory verdicts only
 *  warn because the reclaimed bytes are mostly the resident model's, which unloads first. */
export function resolveMemoryFit(
  estimate: MemoryFitEstimate,
  capacity: MemoryFitCapacity,
): MemoryFitResult {
  const { singleMemoryPool } = capacity;
  const cpuOnly =
    !singleMemoryPool && estimate.gpuBytes === 0 && estimate.totalBytes > 0;
  const { gpuBytes, totalBytes, contextShrinks } = autoContextFigures(
    estimate,
    capacity,
    cpuOnly,
  );
  const rawGpuFit = classifyMemoryFit(gpuBytes, capacity.gpuCapacityGb);
  // The resident model unloads before the replacement allocates, so its bytes count as free.
  // Free-memory questions only; capacity is untouched.
  const reclaimableTotal = reclaimableBytes(capacity.reclaimableTotalBytes);
  // Clamped: a GPU share above its own total would credit the host share a negative amount.
  const reclaimableGpu = Math.min(
    reclaimableBytes(capacity.reclaimableGpuBytes),
    reclaimableTotal,
  );
  // One pool: ask the pressure question of the total, not a GPU share.
  const freeGpuFit = classifyAvailableMemory(
    singleMemoryPool ? totalBytes : gpuBytes,
    capacity.freeGpuCapacityGb,
    capacity.freeGpuCapacityKnown,
    singleMemoryPool ? reclaimableTotal : reclaimableGpu,
    capacity.freeGpuReserveDeficitGb,
  );
  const gpuPressured = freeGpuFit === "exceeds" || freeGpuFit === "tight";
  // Guarded: a non-finite figure gives NaN, which Math.max propagates.
  const hostShareBytes =
    Number.isFinite(totalBytes) && Number.isFinite(gpuBytes)
      ? Math.max(0, totalBytes - gpuBytes)
      : 0;
  const usableHostFit = classifyAvailableMemory(
    singleMemoryPool ? totalBytes : hostShareBytes,
    capacity.usableSystemRamGb,
    capacity.usableSystemRamKnown,
    singleMemoryPool ? reclaimableTotal : reclaimableTotal - reclaimableGpu,
    capacity.systemRamReserveDeficitGb,
  );
  const hostPressured =
    usableHostFit === "exceeds" || usableHostFit === "tight";
  const gpuFit = rawGpuFit === "fits" && gpuPressured ? "tight" : rawGpuFit;
  // The host share must fit host RAM on its own; unused VRAM cannot hold it. Skipped for one pool.
  const hostShareFit: MemoryFitVerdict = singleMemoryPool
    ? "unknown"
    : classifyMemoryFit(hostShareBytes, capacity.systemRamCapacityGb);
  const combinedFit = classifyMemoryFit(
    totalBytes,
    capacity.totalCapacityGb,
  );
  const totalFit = worseMemoryFit(combinedFit, hostShareFit);
  // Lower bound: an unsized target cache or a repo drafter's missing cache undercounts with context.
  const bounded =
    !estimate.kvEstimable ||
    estimate.drafterKvUnsized ||
    estimate.adaptersUnsized;
  return {
    cpuOnly,
    rawGpuFit,
    gpuFit,
    freeGpuFit,
    gpuPressured,
    hostShareBytes,
    usableHostFit,
    hostPressured,
    hostShareFit,
    totalFit,
    bounded,
    prefix: bounded ? "≥ " : "",
    advisory: resolveMemoryAdvisory(estimate, {
      cpuOnly,
      singleMemoryPool,
      totalFit,
      combinedFit,
      hostShareFit,
      gpuFit,
      rawGpuFit,
      gpuPressured,
      hostPressured,
      contextShrinks,
    }),
  };
}

const AUTO_CONTEXT_STEP = 256;
// Metal Auto (llama_cpp.py): capped at _APPLE_AUTO_CTX; min(_FIT_MIN_CTX, native) when nothing fits.
const APPLE_AUTO_CONTEXT_CAP = 65536;
const APPLE_NOTHING_FITS_CONTEXT = 8192;

/** Not from a stale estimate, nor when an Auto resident with the same settings already answered. */
export function previewsAutoFit(state: {
  isGguf: boolean;
  estimateAvailable: boolean;
  estimateStale: boolean;
  isActiveModel: boolean;
  /** Another local model holds memory the load frees first, by an amount not priced here. */
  otherModelResident: boolean;
  /** The loader takes the fewest cards that hold the model, which pooled free VRAM cannot see. */
  multiGpu: boolean;
  residentKnown: boolean;
  residentLoadedAuto: boolean;
  pendingMatchesResident: boolean;
}): boolean {
  if (!state.isGguf || !state.estimateAvailable || state.estimateStale || state.multiGpu) {
    return false;
  }
  if (!state.isActiveModel) return !state.otherModelResident;
  return (
    state.residentKnown && !(state.residentLoadedAuto && state.pendingMatchesResident)
  );
}

/** Linear between adjacent prices on the floor-to-native curve: a compact SWA cache saturates
 *  part-way, so one line from floor to native would overstate the window. */
export function resolveAutoFittedContext(
  estimate: MemoryFitEstimate,
  capacity: MemoryFitCapacity,
  {
    appleUnifiedMemory = false,
    loadBudgetCeilingGb = null,
  }: {
    appleUnifiedMemory?: boolean;
    /** Metal's own after-unload budget, min(working set, free) times its fraction. */
    loadBudgetCeilingGb?: number | null;
  } = {},
): number | null {
  const nativeCtx = estimate.nCtx ?? 0;
  const floorCtx = estimate.gpuFloorCtx ?? 0;
  if (
    !hasContextFloor(estimate) ||
    !estimate.kvEstimable ||
    !(floorCtx > 0) ||
    !Number.isFinite(nativeCtx) ||
    !Number.isFinite(estimate.gpuBytes) ||
    !Number.isFinite(estimate.totalBytes) ||
    (!capacity.singleMemoryPool && estimate.gpuBytes === 0)
  ) {
    return null;
  }
  if (nativeCtx <= floorCtx) return nativeCtx;
  const { freeGpuCapacityGb: availableGb, freeGpuCapacityKnown: known } = capacity;
  if (!Number.isFinite(availableGb) || availableGb < 0 || (availableGb === 0 && !known)) {
    return null;
  }
  const reclaimableTotal = reclaimableBytes(capacity.reclaimableTotalBytes);
  const reclaimed = capacity.singleMemoryPool
    ? reclaimableTotal
    : Math.min(reclaimableBytes(capacity.reclaimableGpuBytes), reclaimableTotal);
  const credited =
    availableGb +
    Math.max(0, reclaimed / 1024 ** 3 - reclaimableBytes(capacity.freeGpuReserveDeficitGb));
  const poolGb = capacity.singleMemoryPool ? capacity.totalCapacityGb : capacity.gpuCapacityGb;
  const ceilings = [poolGb, loadBudgetCeilingGb ?? 0].filter((gb) => gb > 0);
  const budget = Math.min(credited, ...ceilings) * 1024 ** 3;
  const nativeBytes = capacity.singleMemoryPool ? estimate.totalBytes : estimate.gpuBytes;
  // Two pools fit the GPU share; host terms stay at native.
  const priced = (gpuBytes: number, totalBytes: number | null | undefined) =>
    capacity.singleMemoryPool && Number.isFinite(totalBytes)
      ? (totalBytes as number)
      : nativeBytes - Math.max(0, estimate.gpuBytes - gpuBytes);
  const curve: [number, number][] = [
    [floorCtx, priced(estimate.gpuFloorBytes as number, estimate.floorTotalBytes)],
    ...(estimate.fitCurve ?? [])
      .filter(([ctx]) => ctx > floorCtx && ctx < nativeCtx)
      .map(([ctx, gpu, total]): [number, number] => [ctx, priced(gpu, total)]),
    [nativeCtx, nativeBytes],
  ];
  const ceiling = appleUnifiedMemory ? Math.min(nativeCtx, APPLE_AUTO_CONTEXT_CAP) : nativeCtx;
  if (nativeBytes <= budget) return ceiling;
  if (curve[0][1] >= budget) {
    return appleUnifiedMemory ? Math.min(nativeCtx, APPLE_NOTHING_FITS_CONTEXT) : floorCtx;
  }
  let i = 0;
  while (curve[i + 1][1] <= budget) i += 1;
  const [[loCtx, loBytes], [hiCtx, hiBytes]] = [curve[i], curve[i + 1]];
  const fitted = loCtx + ((budget - loBytes) / (hiBytes - loBytes)) * (hiCtx - loCtx);
  return Math.max(
    floorCtx,
    Math.min(ceiling, Math.floor(fitted / AUTO_CONTEXT_STEP) * AUTO_CONTEXT_STEP),
  );
}

interface AdvisoryVerdicts {
  cpuOnly?: boolean;
  singleMemoryPool: boolean;
  totalFit: MemoryFitVerdict;
  combinedFit: MemoryFitVerdict;
  hostShareFit: MemoryFitVerdict;
  gpuFit: MemoryFitVerdict;
  rawGpuFit: MemoryFitVerdict;
  gpuPressured: boolean;
  hostPressured: boolean;
  contextShrinks: boolean;
}

function classifyAvailableMemory(
  bytes: number,
  availableGb: number,
  known = false,
  reclaimedBytes = 0,
  reserveDeficitGb = 0,
): MemoryFitVerdict {
  if (
    !Number.isFinite(availableGb) ||
    availableGb < 0 ||
    (availableGb === 0 && !known)
  ) {
    return "unknown";
  }
  // Pressure is a fraction of post-unload availability.
  const afterUnloadGb = availableGb + Math.max(
    0,
    reclaimedBytes / 1024 ** 3 - reclaimableBytes(reserveDeficitGb),
  );
  if (known && afterUnloadGb === 0 && Number.isFinite(bytes) && bytes > 0)
    return "exceeds";
  return classifyMemoryFit(bytes, afterUnloadGb);
}

function contextShrinksAdvisory(estimate: MemoryFitEstimate): MemoryAdvisory {
  const nCtx = estimate.nCtx;
  const priced =
    nCtx != null && Number.isFinite(nCtx) && nCtx > 0
      ? `the full ${nCtx.toLocaleString()}-token context`
      : "the model's full context";
  return {
    tone: "muted",
    text: `Estimated at ${priced}. Auto context will shrink it to fit.`,
  };
}

/** An unsizable cache outranks any verdict. Branches on `kvEstimable`, not `bounded`. */
export function resolveMemoryAdvisory(
  estimate: MemoryFitEstimate,
  verdicts: AdvisoryVerdicts,
): MemoryAdvisory | null {
  if (!estimate.kvEstimable) {
    return {
      tone: "warn",
      text: "KV cache size is unknown: missing attention dimensions. Actual usage will be higher.",
    };
  }
  if (estimate.drafterKvUnsized) {
    return {
      tone: "warn",
      text: "A remote draft model or vision component is partly unmeasured. Actual usage will be higher.",
    };
  }
  if (estimate.adaptersUnsized) {
    return {
      tone: "warn",
      text: "An adapter or control vector is unmeasured. Actual usage will be higher.",
    };
  }
  if (estimate.moeOffloadUnmodelled && !verdicts.cpuOnly) {
    return {
      tone: "muted",
      text: "Expert layers on the CPU are not reflected here. GPU usage may be lower.",
    };
  }
  if (verdicts.cpuOnly) {
    if (verdicts.totalFit === "exceeds") {
      return {
        tone: "warn",
        text: "Exceeds system RAM. Try a shorter context or smaller model.",
      };
    }
    if (verdicts.hostPressured) {
      return {
        tone: "muted",
        text: "Fits system RAM, but little is free right now. Free memory, or try a shorter context or smaller model.",
      };
    }
    return null;
  }
  if (verdicts.singleMemoryPool) {
    if (verdicts.totalFit === "exceeds") {
      return {
        tone: "warn",
        text: "Exceeds shared memory. Try a shorter context or smaller model; CPU offloading adds no memory.",
      };
    }
    // One pool: GPU free and host available are two views of the same bytes.
    if (verdicts.hostPressured || verdicts.gpuPressured) {
      return {
        tone: "muted",
        text: hasContextFloor(estimate)
          ? "Fits this machine, but little memory is free right now, so Auto may pick a shorter context. Free memory first."
          : "Fits this machine, but little memory is free right now. Free memory or try Auto context.",
      };
    }
    // Free-memory pressure first: the loader fits against available memory.
    if (verdicts.contextShrinks) {
      return contextShrinksAdvisory(estimate);
    }
    return null;
  }
  // Moving layers cannot fix a combined capacity shortfall.
  if (
    verdicts.combinedFit === "exceeds" ||
    (verdicts.hostShareFit === "exceeds" && verdicts.rawGpuFit === "exceeds")
  ) {
    return {
      tone: "warn",
      text: "Exceeds combined GPU and system memory. Try a shorter context or smaller model.",
    };
  }
  if (
    (verdicts.gpuFit === "exceeds" && verdicts.hostPressured) ||
    (verdicts.hostShareFit === "exceeds" && verdicts.gpuPressured)
  ) {
    return {
      tone: "warn",
      text: "GPU and system memory are both under pressure. Try a shorter context or smaller model.",
    };
  }
  if (verdicts.hostShareFit === "exceeds") {
    return {
      tone: "warn",
      text: "CPU placement exceeds system RAM. Try fewer CPU layers or a smaller model.",
    };
  }
  if (verdicts.gpuFit === "exceeds") {
    if (hasContextFloor(estimate)) {
      return {
        tone: "warn",
        text: estimate.floorCanOffload
          ? "Exceeds GPU memory even at the shortest context the loader tries, so some layers will run on the CPU and generation will be slower."
          : "Exceeds GPU memory even at the shortest context the loader tries, and these settings keep layers from moving to the CPU, so loading may fail.",
      };
    }
    return {
      tone: "warn",
      text: "Exceeds GPU memory. Try Auto context or fewer GPU layers; loading may still fail.",
    };
  }
  if (verdicts.hostPressured) {
    return {
      tone: "muted",
      text: "Fits system RAM, but little is free right now. Free memory, or try a shorter context or smaller model.",
    };
  }
  if ((verdicts.rawGpuFit === "fits" || verdicts.rawGpuFit === "tight") && verdicts.gpuPressured) {
    return {
      tone: "muted",
      text: hasContextFloor(estimate)
        ? `Fits this GPU, but little VRAM is free right now, so Auto may pick a shorter context${estimate.floorCanOffload ? " or run some layers on the CPU" : ""}. Free memory first.`
        : "Fits this GPU, but little VRAM is free right now. Free memory or try Auto context.",
    };
  }
  if (verdicts.contextShrinks) {
    return contextShrinksAdvisory(estimate);
  }
  return null;
}

export const NOTE_SEPARATOR = " · ";

/** What choosing Default or Manual disk offload stores: Default leaves each part to Auto, Manual
 *  pins the fraction Auto would pick, or off where Auto keeps the part resident. */
export function mlxOffloadModePatch(
  manual: boolean,
  parts: readonly { key: "mlxPleOffload" | "mlxExpertOffload"; decided: number | null }[],
): Record<string, MlxOffload> {
  return Object.fromEntries(
    parts.map(({ key, decided }) => [key, manual ? decided || "off" : null]),
  );
}

/** Disk offload slider position for Auto, one step below 0% (off), as GPU Layers has. */
export const MLX_OFFLOAD_SLIDER_AUTO = -5;

export function mlxOffloadSliderValue(value: MlxOffload | undefined): number {
  if (value == null) return MLX_OFFLOAD_SLIDER_AUTO;
  return value === "off" ? 0 : Math.round(value * 100);
}

export function mlxOffloadFromSlider(position: number): MlxOffload {
  if (position < 0) return null;
  return position === 0 ? "off" : position / 100;
}

/** The fractions Auto picks for the current settings, or null until an estimate of them settles: a stale or
 *  loading estimate answers earlier ones, and reading its parts as resident would pin them off. */
export function mlxOffloadDecided(state: {
  estimate: { mlxPleOffload?: number | null; mlxExpertOffload?: number | null } | null;
  stale: boolean;
  loading: boolean;
}): { ple: number | null; experts: number | null } | null {
  if (state.stale || state.loading || !state.estimate) return null;
  return { ple: state.estimate.mlxPleOffload ?? null, experts: state.estimate.mlxExpertOffload ?? null };
}

export function mlxOffloadPercent(fraction: number): string {
  return `${Math.round(fraction * 100)}%`;
}

/** A separated caption, breakable only between its items. A caption like "f16 - 262,144 tokens
 *  - 4 slots" does not fit a narrow panel, and the browser breaks at the last space that
 *  fits, orphaning "slots". Gluing each item with U+00A0 leaves one break opportunity per
 *  bullet, and gluing the bullet to the item that FOLLOWS it puts the break before it. A note
 *  with NO separator is returned untouched: it is ordinary prose, and gluing it made a single
 *  unbreakable run that overflows the caption column rather than wrapping. */
export function glueNoteItems(note: string): string {
  const items = note.split(NOTE_SEPARATOR);
  if (items.length < 2) {
    return note;
  }
  return items
    .map((item, index) => {
      const glued = item.replace(/ /g, "\u00a0");
      return index === 0 ? glued : `\u00b7\u00a0${glued}`;
    })
    .join(" ");
}

export function resolveKvNote(estimate: {
  cacheTypeKv: string | null;
  nCtx: number;
  nParallel: number;
  kvOnGpu: boolean;
}): string {
  return [
    estimate.cacheTypeKv ?? "f16",
    // Off the wire: toLocaleString() on null throws and would take the panel down.
    `${Number.isFinite(estimate.nCtx) ? Math.max(0, estimate.nCtx).toLocaleString() : "0"} tokens`,
    Number.isFinite(estimate.nParallel) && estimate.nParallel > 1
      ? `${estimate.nParallel} slots`
      : null,
    estimate.kvOnGpu ? null : "host RAM",
  ]
    .filter(Boolean)
    .join(" · ");
}

/** Uses the drafter's own GPU share, not kvOnGpu: different flags move each, and MTP splits it. */
export function resolveDraftCacheNote(
  drafterRuntimeGpuBytes: number,
  drafterRuntimeBytes: number,
): string | undefined {
  if (!(drafterRuntimeGpuBytes > 0)) {
    return "host RAM";
  }
  if (drafterRuntimeGpuBytes < drafterRuntimeBytes) {
    return `${formatMemoryGb(drafterRuntimeGpuBytes)} on GPU`;
  }
  return undefined;
}
