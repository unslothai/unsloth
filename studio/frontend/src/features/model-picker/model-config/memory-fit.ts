// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** How the estimated footprint sits against the memory available to hold it, and the single
 *  note the row prints about it. Split out of model-config-page.tsx and kept free of `@/`
 *  ALIAS imports so `tests/` can load it under `node --experimental-strip-types`, which does
 *  not resolve that alias. Imports below are RELATIVE with explicit extensions for the same
 *  reason; an `@/` spelling breaks the three tests that load this module directly. */

// The fit vocabulary and unit formatting live in src/lib/memory/, shared with the Hub memory
// bar so the two surfaces cannot describe one load differently. Re-exported here because
// this module is the panel's entry point and its call sites are unchanged.
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

/** A memory figure in bytes, to two decimals. @deprecated Prefer `formatBytesGiB` from
 *  `@/lib/memory/format`, whose name says which unit it takes. This alias exists because a
 *  SECOND exported `formatMemoryGb` in lib/model-memory.ts took gigabytes and printed a
 *  different label, with the same name and signature. The divide was always by 1024^3, so
 *  every figure was a gibibyte labelled as a gigabyte, overstating each by 7.4% (#9570). */
export const formatMemoryGb = formatBytesGiB;

/** Progressively shorter labels; compact lower bounds round down. */
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

/** At most one note under the figures, most actionable first. */
export interface MemoryAdvisory {
  /** `warn` is amber; `muted` is body text. */
  tone: "warn" | "muted";
  text: string;
}

/** The estimate fields the verdicts read, structurally. */
export interface MemoryFitEstimate {
  /** The share of the total that lands on the GPU under the requested offload. */
  gpuBytes: number;
  /** Weights + KV + compute, wherever they land. */
  totalBytes: number;
  /** False when the GGUF header could not size the cache, so the figures are a floor. */
  kvEstimable: boolean;
  /** A charged drafter whose cache could not be sized, so the figures are a floor. */
  drafterKvUnsized: boolean;
  adaptersUnsized: boolean;
  /** `--n-cpu-moe` is set, so the GPU figure ignores it and reads high. */
  moeOffloadUnmodelled: boolean;
}

/** What the load may draw on, as resolved by resolveMemoryCapacityGb and the host. */
export interface MemoryFitCapacity {
  /** VRAM available, or the shared pool where there is only one. 0 when unknown. */
  gpuCapacityGb: number;
  /** GPU plus host RAM, the ceiling an offloaded load works against. 0 when unknown. */
  totalCapacityGb: number;
  /** Host RAM alone. Bytes pinned OUTSIDE the GPU have to fit in this, and unused VRAM cannot
   *  help them, so it is a separate question from the total. */
  systemRamCapacityGb: number;
  /** VRAM free on the usable cards right now. Warns only. */
  freeGpuCapacityGb: number;
  /** Distinguishes an exhausted GPU budget from an unknown reading. */
  freeGpuCapacityKnown?: boolean;
  /** Reserve hidden by clamping the current usable VRAM to zero. */
  freeGpuReserveDeficitGb?: number;
  /** Available host RAM after the loader's reserve. Warns only. */
  usableSystemRamGb: number;
  /** Distinguishes exhausted RAM from an unknown reading. */
  usableSystemRamKnown?: boolean;
  /** Host reserve hidden by clamping current usable RAM to zero. */
  systemRamReserveDeficitGb?: number;
  /** GPU and host draw on the same memory, so an offloaded byte is not a freed one. */
  singleMemoryPool: boolean;
  /** Resident bytes returned to the requested pools on unload. Free-memory verdicts only. */
  reclaimableTotalBytes?: number;
  /** The GPU share of the above. */
  reclaimableGpuBytes?: number;
}

/** Ignore invalid or negative credits. */
function reclaimableBytes(value: number | undefined): number {
  return Number.isFinite(value) && (value as number) > 0 ? (value as number) : 0;
}

/** Keep known host credit; require modelled placement in the requested pool for VRAM. */
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
	// File-backed pages may already count as available RAM. Their pool split is unknown.
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
  /** The estimate places the entire load in separate system RAM. */
  cpuOnly: boolean;
  /** The GPU verdict before the free-memory warning is folded in. */
  rawGpuFit: MemoryFitVerdict;
  /** What the GPU figure is coloured with: rawGpuFit, nudged to tight under pressure. */
  gpuFit: MemoryFitVerdict;
  /** The footprint against post-unload availability, capped at a warning by its caller. */
  freeGpuFit: MemoryFitVerdict;
  gpuPressured: boolean;
  /** Bytes this placement pins outside the GPU. */
  hostShareBytes: number;
  usableHostFit: MemoryFitVerdict;
  hostPressured: boolean;
  /** The host share against physical RAM. "unknown" where there is only one pool. */
  hostShareFit: MemoryFitVerdict;
  /** What the total figure is coloured with. */
  totalFit: MemoryFitVerdict;
  /** The figures are a lower bound, so they are printed with a leading marker. */
  bounded: boolean;
  /** The marker itself, empty when the figures are a real estimate. */
  prefix: string;
  advisory: MemoryAdvisory | null;
}

/** Every verdict the row shows, plus the one note it prints. `_host_offload_shortfall_message`
 *  refuses a load whose offloaded weights exceed psutil's AVAILABLE memory less a reserve,
 *  not the physical total, so a 70 GB host share read as fitting a 128 GB box with 32 GB
 *  free. The free-memory verdicts here WARN instead: the bytes a pending load reclaims are
 *  mostly the resident model's own, which Studio unloads first. */
export function resolveMemoryFit(
  estimate: MemoryFitEstimate,
  capacity: MemoryFitCapacity,
): MemoryFitResult {
  const { singleMemoryPool } = capacity;
  const cpuOnly =
    !singleMemoryPool && estimate.gpuBytes === 0 && estimate.totalBytes > 0;
  const rawGpuFit = classifyMemoryFit(
    estimate.gpuBytes,
    capacity.gpuCapacityGb,
  );
  // Studio unloads the resident model BEFORE the replacement allocates, so its bytes are about to
  // be free rather than competing. Charging them counted a model against ITSELF: reloading an
  // unchanged config warned it would not fit the memory its own resident copy held. Free-memory
  // questions only -- unloading frees memory, it does not add any, so capacity is untouched.
  const reclaimableTotal = reclaimableBytes(capacity.reclaimableTotalBytes);
  // Clamped: a GPU share above its own total would credit the host share a negative amount.
  const reclaimableGpu = Math.min(
    reclaimableBytes(capacity.reclaimableGpuBytes),
    reclaimableTotal,
  );
  // One pool means the WHOLE load draws on that memory, so the pressure question goes to the
  // total rather than a GPU share that is not a separate reservation. Asking it of gpuBytes
  // alone let a partly CPU-offloaded load on a Vulkan iGPU look comfortable.
  const freeGpuFit = classifyAvailableMemory(
    singleMemoryPool ? estimate.totalBytes : estimate.gpuBytes,
    capacity.freeGpuCapacityGb,
    capacity.freeGpuCapacityKnown,
    singleMemoryPool ? reclaimableTotal : reclaimableGpu,
    capacity.freeGpuReserveDeficitGb,
  );
  const gpuPressured = freeGpuFit === "exceeds" || freeGpuFit === "tight";
  // Guarded, not subtracted blind: a non-finite figure makes the difference NaN, which
  // `Math.max(0, ...)` propagates rather than clamps. 0 classifies the same and is printable.
  const hostShareBytes =
    Number.isFinite(estimate.totalBytes) && Number.isFinite(estimate.gpuBytes)
      ? Math.max(0, estimate.totalBytes - estimate.gpuBytes)
      : 0;
  // Same question for the other pool, with the same credit. See the two notes above.
  const usableHostFit = classifyAvailableMemory(
    singleMemoryPool ? estimate.totalBytes : hostShareBytes,
    capacity.usableSystemRamGb,
    capacity.usableSystemRamKnown,
    singleMemoryPool ? reclaimableTotal : reclaimableTotal - reclaimableGpu,
    capacity.systemRamReserveDeficitGb,
  );
  const hostPressured =
    usableHostFit === "exceeds" || usableHostFit === "tight";
  const gpuFit = rawGpuFit === "fits" && gpuPressured ? "tight" : rawGpuFit;
  // The host share must fit host RAM on its own: unused VRAM cannot hold bytes pinned outside
  // the GPU, so the combined ceiling alone called a 70 GB CPU placement a fit on a 24 GB card
  // plus 64 GB of RAM. Skipped where the two are one pool.
  const hostShareFit: MemoryFitVerdict = singleMemoryPool
    ? "unknown"
    : classifyMemoryFit(hostShareBytes, capacity.systemRamCapacityGb);
  const combinedFit = classifyMemoryFit(
    estimate.totalBytes,
    capacity.totalCapacityGb,
  );
  const totalFit = worseMemoryFit(combinedFit, hostShareFit);
  // Lower bound, not an estimate. Both routes here UNDER-count by a term that grows with
  // context: no attention dims, so the target cache is missing, or a drafter that is a
  // repository rather than a file, so its cache is missing while its weights are counted.
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
    }),
  };
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
  // Pressure is a fraction of post-unload availability, not just allocation growth.
  const afterUnloadGb = availableGb + Math.max(
    0,
    reclaimedBytes / 1024 ** 3 - reclaimableBytes(reserveDeficitGb),
  );
  if (known && afterUnloadGb === 0 && Number.isFinite(bytes) && bytes > 0)
    return "exceeds";
  return classifyMemoryFit(bytes, afterUnloadGb);
}

/** At most one note, most actionable first. An unsizable cache outranks any verdict drawn from
 *  the figures, since it says they are incomplete. Branches on `kvEstimable`, not `bounded`:
 *  both make the figures a floor, but only this one is about the header. The pool split used
 *  to gate the whole tail, making the shared-pool pressure copy dead code, so a single-pool
 *  host had one reachable note. The pool question now decides the WORDING, not the branch. */
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
    // One pool, so one pressure question however it was measured: the GPU's free reading and the
    // host's available reading are two views of the same bytes.
    if (verdicts.hostPressured || verdicts.gpuPressured) {
      return {
        tone: "muted",
        text: "Fits this machine, but little memory is free right now. Free memory or try Auto context.",
      };
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
  if (verdicts.rawGpuFit === "fits" && verdicts.gpuPressured) {
    return {
      tone: "muted",
      text: "Fits this GPU, but little VRAM is free right now. Free memory or try Auto context.",
    };
  }
  return null;
}

/** What `resolveKvNote` joins its items with, and what `glueNoteItems` splits on. */
export const NOTE_SEPARATOR = " · ";

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

/** The KV line's caption: dtype, what was priced, and where it lives. */
export function resolveKvNote(estimate: {
  cacheTypeKv: string | null;
  nCtx: number;
  nParallel: number;
  kvOnGpu: boolean;
}): string {
  return [
    estimate.cacheTypeKv ?? "f16",
    // Off the wire, so it is not trusted to be a number: `.toLocaleString()` on a null throws, and
    // one bad field must not take the whole panel down.
    `${Number.isFinite(estimate.nCtx) ? Math.max(0, estimate.nCtx).toLocaleString() : "0"} tokens`,
    Number.isFinite(estimate.nParallel) && estimate.nParallel > 1
      ? `${estimate.nParallel} slots`
      : null,
    estimate.kvOnGpu ? null : "host RAM",
  ]
    .filter(Boolean)
    .join(" · ");
}

/** Where the draft cache actually sits, read from its own GPU share rather than kvOnGpu.
 *  kvOnGpu is the TARGET cache's placement and the two are set by different flags:
 *  `--no-kv-offload` moves the target, `--spec-draft-ngl 0` moves the drafter. Off the target
 *  flag this line was wrong in both directions. Under MTP the term is split across both
 *  placements, a third case no boolean could express. */
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
