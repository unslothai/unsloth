// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * How the estimated footprint sits against the memory available to hold it, and the
 * single note the row prints about it.
 *
 * Import-free, like the sibling estimate-context.ts, so `tests/` can load it under
 * `node --experimental-strip-types`, which does not resolve the `@/` alias. Inside a
 * 3,400-line .tsx this chain was unreachable from the test runner, and a ternary arm
 * that could never be taken survived review for exactly that reason. Everything is
 * pure and typed structurally, so a MemoryEstimate satisfies the inputs.
 */

/** How an estimated footprint sits against the memory available to hold it. */
export type MemoryFitVerdict = "fits" | "tight" | "exceeds" | "unknown";

/** Above this share of the capacity the fit is reported as tight rather than clean. */
export const MEMORY_FIT_TIGHT_RATIO = 0.85;

/**
 * Classify a footprint against a capacity.
 *
 * Every non-finite input is "unknown", and `<= 0` alone does not cover it: NaN and
 * Infinity fail every comparison, so `NaN <= 0` is false and the ratio test falls
 * through to "fits" -- a green verdict from a number that does not exist. JSON.parse
 * turns `1e999` into Infinity and a `?? 0` default never sees it.
 */
export function classifyMemoryFit(
  bytes: number,
  capacityGb: number,
): MemoryFitVerdict {
  // Nothing probed or nothing to weigh: no verdict rather than a false "fits".
  if (!Number.isFinite(bytes) || !Number.isFinite(capacityGb)) {
    return "unknown";
  }
  if (capacityGb <= 0 || bytes <= 0) {
    return "unknown";
  }
  const ratio = bytes / (capacityGb * 1024 ** 3);
  if (ratio > 1) {
    return "exceeds";
  }
  if (ratio > MEMORY_FIT_TIGHT_RATIO) {
    return "tight";
  }
  return "fits";
}

/** The worse of two verdicts, for a load that has to satisfy both at once. */
export function worseMemoryFit(
  a: MemoryFitVerdict,
  b: MemoryFitVerdict,
): MemoryFitVerdict {
  const rank: Record<MemoryFitVerdict, number> = {
    unknown: 0,
    fits: 1,
    tight: 2,
    exceeds: 3,
  };
  // unknown loses to any real verdict: one half being unmeasurable must not erase the
  // other half's answer.
  return rank[a] >= rank[b] ? a : b;
}

/**
 * GB, to two decimals, as the rest of the panel talks about memory.
 *
 * Clamped, not trusted: these come off the wire, and "-3.00 GB" or "NaN GB" reads as
 * a measurement rather than as the missing reading it is.
 */
export function formatMemoryGb(bytes: number): string {
  const safe = Number.isFinite(bytes) && bytes > 0 ? bytes : 0;
  return `${(safe / 1024 ** 3).toFixed(2)} GB`;
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
  /** Host RAM alone. Bytes pinned OUTSIDE the GPU have to fit in this, and unused
   *  VRAM cannot help them, so it is a separate question from the total. */
  systemRamCapacityGb: number;
  /** VRAM free on the usable cards right now. Warns only. 0 when nothing was probed. */
  freeGpuCapacityGb: number;
  /** Host RAM the machine can hand out right now, less the loader's reserve. Warns
   *  only. 0 when unknown. */
  usableSystemRamGb: number;
  /** GPU and host draw on the same memory, so an offloaded byte is not a freed one. */
  singleMemoryPool: boolean;
}

export interface MemoryFitResult {
  /** The GPU verdict before the free-memory warning is folded in. */
  rawGpuFit: MemoryFitVerdict;
  /** What the GPU figure is coloured with: rawGpuFit, nudged to tight under pressure. */
  gpuFit: MemoryFitVerdict;
  /** The pool against what is free right now, capped at a warning by its caller. */
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

/**
 * Every verdict the row shows, plus the one note it prints.
 *
 * `_host_offload_shortfall_message` refuses a load whose offloaded weights exceed
 * psutil's AVAILABLE memory less a reserve, not the physical total, so a 70 GB host
 * share read as fitting a 128 GB box with 32 GB free. The free-memory verdicts here
 * WARN instead: the bytes a pending load reclaims are mostly the resident model's
 * own, which Studio unloads first, and cannot be attributed from here.
 */
export function resolveMemoryFit(
  estimate: MemoryFitEstimate,
  capacity: MemoryFitCapacity,
): MemoryFitResult {
  const { singleMemoryPool } = capacity;
  const rawGpuFit = classifyMemoryFit(estimate.gpuBytes, capacity.gpuCapacityGb);
  // One pool means the WHOLE load draws on that memory, so the pressure question goes
  // to the total rather than to a GPU share that is not a separate reservation. Asking
  // it of gpuBytes alone let a partly CPU-offloaded load on a Vulkan iGPU look
  // comfortable against free memory that has to hold all of it.
  const freeGpuFit = classifyMemoryFit(
    singleMemoryPool ? estimate.totalBytes : estimate.gpuBytes,
    capacity.freeGpuCapacityGb,
  );
  const gpuPressured = freeGpuFit === "exceeds" || freeGpuFit === "tight";
  // Guarded, not subtracted blind: a non-finite figure makes the difference NaN, which
  // `Math.max(0, ...)` propagates rather than clamps. 0 classifies the same ("unknown")
  // and is printable, which NaN is not.
  const hostShareBytes =
    Number.isFinite(estimate.totalBytes) && Number.isFinite(estimate.gpuBytes)
      ? Math.max(0, estimate.totalBytes - estimate.gpuBytes)
      : 0;
  // Same question for the other pool. See the note above on why this warns.
  const usableHostFit = classifyMemoryFit(
    singleMemoryPool ? estimate.totalBytes : hostShareBytes,
    capacity.usableSystemRamGb,
  );
  const hostPressured = usableHostFit === "exceeds" || usableHostFit === "tight";
  const gpuFit = rawGpuFit === "fits" && gpuPressured ? "tight" : rawGpuFit;
  // The host share must fit host RAM on its own: unused VRAM cannot hold bytes pinned
  // outside the GPU, so the combined ceiling alone called a 70 GB CPU placement a fit
  // on a 24 GB card plus 64 GB of RAM. Skipped where the two are one pool.
  const hostShareFit: MemoryFitVerdict = singleMemoryPool
    ? "unknown"
    : classifyMemoryFit(hostShareBytes, capacity.systemRamCapacityGb);
  const totalFit = worseMemoryFit(
    classifyMemoryFit(estimate.totalBytes, capacity.totalCapacityGb),
    hostShareFit,
  );
  // Lower bound, not an estimate. Both routes here UNDER-count by a term that grows
  // with context: no attention dims, so the target cache is missing, or a drafter that
  // is a repository rather than a file on disk, so its cache is missing while its
  // weights are counted. The advisory still tells them apart -- different fixes.
  const bounded =
    !estimate.kvEstimable || estimate.drafterKvUnsized || estimate.adaptersUnsized;
  return {
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
      singleMemoryPool,
      totalFit,
      hostShareFit,
      gpuFit,
      rawGpuFit,
      gpuPressured,
      hostPressured,
    }),
  };
}

interface AdvisoryVerdicts {
  singleMemoryPool: boolean;
  totalFit: MemoryFitVerdict;
  hostShareFit: MemoryFitVerdict;
  gpuFit: MemoryFitVerdict;
  rawGpuFit: MemoryFitVerdict;
  gpuPressured: boolean;
  hostPressured: boolean;
}

/**
 * At most one note, most actionable first.
 *
 * An unsizable cache outranks any verdict drawn from the figures, since it says they
 * are incomplete. Branches on `kvEstimable`, not `bounded`: both make the figures a
 * floor, but only this one is about the header.
 *
 * The pool split used to gate the whole tail, which made the shared-pool pressure copy
 * dead code -- it sat inside the `singleMemoryPool === false` arm and re-tested
 * `singleMemoryPool` -- leaving a single-pool host one reachable note, "exceeds". An
 * Apple machine or a Vulkan iGPU never warned that a load fitting the machine did not
 * fit what was free. The pool question now decides the WORDING and which figures are
 * compared, not whether the branch exists.
 */
export function resolveMemoryAdvisory(
  estimate: MemoryFitEstimate,
  verdicts: AdvisoryVerdicts,
): MemoryAdvisory | null {
  if (!estimate.kvEstimable) {
    return {
      tone: "warn",
      text: "This GGUF's header doesn't carry the attention dimensions, so the KV cache can't be sized. The figures above are a floor, and the cache is usually the term that grows fastest with context.",
    };
  }
  if (estimate.drafterKvUnsized) {
    return {
      tone: "warn",
      text: "Part of this load is a file the server will fetch rather than one on this disk, so it can't be sized from here. The figures above are a floor.",
    };
  }
  if (estimate.moeOffloadUnmodelled) {
    return {
      tone: "muted",
      text: "Expert layers held on the CPU aren't modelled here, so the GPU figure reads high.",
    };
  }
  if (verdicts.singleMemoryPool) {
    if (verdicts.totalFit === "exceeds") {
      return {
        tone: "warn",
        text: "More than this machine's memory. The GPU and the rest of the system share one pool here, so there is nothing to offload to.",
      };
    }
    // One pool, so one pressure question however it was measured: the GPU's free
    // reading and the host's available reading are two views of the same bytes, and
    // either seeing pressure is the same news.
    if (verdicts.hostPressured || verdicts.gpuPressured) {
      return {
        tone: "muted",
        text: "This fits the machine, but not what is free right now. If that memory is not the model being replaced, the context will be fitted down or the load refused.",
      };
    }
    return null;
  }
  // Discrete memory, so the two verdicts are separate questions and the aggregate one
  // is asked FIRST. Reading gpuFit alone offered spilling to system RAM as the remedy
  // for a load that does not fit in GPU and RAM combined, which is advice to do
  // something that cannot work.
  if (verdicts.hostShareFit === "exceeds") {
    return {
      tone: "warn",
      text: "More than system RAM holds. This placement keeps most of the load outside the GPU, and spare VRAM cannot take those bytes.",
    };
  }
  if (verdicts.totalFit === "exceeds") {
    return {
      tone: "warn",
      text: "More than this machine holds. The GPU and system RAM together are not enough for this load, so spilling layers or fitting the context down will not recover it.",
    };
  }
  if (verdicts.gpuFit === "exceeds") {
    return {
      tone: "warn",
      text: "More than this GPU holds. Layers will spill to system RAM, or the context will be fitted down to what fits.",
    };
  }
  if (verdicts.hostPressured) {
    return {
      tone: "muted",
      text: "The part of this load that runs from system RAM fits the machine, but not what is free right now. If that memory is not the model being replaced, the load will be refused.",
    };
  }
  if (verdicts.rawGpuFit === "fits" && verdicts.gpuPressured) {
    return {
      tone: "muted",
      text: "This fits the card, but something is using it right now. If that memory is not the model being replaced, layers will spill or the context will be fitted down.",
    };
  }
  return null;
}

/** What `resolveKvNote` joins its items with, and what `glueNoteItems` splits on. */
export const NOTE_SEPARATOR = " · ";

/**
 * The same caption, breakable only between its items.
 *
 * A caption like "f16 · 262,144 tokens · 4 slots" does not fit the panel once the
 * window is narrow enough to shrink it, and left to itself the browser breaks at the
 * last space that fits -- which put "slots" alone on a line under "... tokens · 4".
 * Gluing each item together with U+00A0 leaves exactly one break opportunity per
 * bullet, and gluing the bullet to the item that FOLLOWS it means the break lands
 * before the bullet rather than orphaning it at the end of the previous line.
 */
export function glueNoteItems(note: string): string {
  return note
    .split(NOTE_SEPARATOR)
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
    // Off the wire, so it is not trusted to be a number: `.toLocaleString()` on a
    // null throws, and one bad field must not take the whole panel down.
    `${Number.isFinite(estimate.nCtx) ? Math.max(0, estimate.nCtx).toLocaleString() : "0"} tokens`,
    Number.isFinite(estimate.nParallel) && estimate.nParallel > 1
      ? `${estimate.nParallel} slots`
      : null,
    estimate.kvOnGpu ? null : "host RAM",
  ]
    .filter(Boolean)
    .join(" · ");
}

/**
 * Where the draft cache actually sits, read from its own GPU share rather than from
 * kvOnGpu.
 *
 * kvOnGpu is the TARGET cache's placement, and the two are set by different flags:
 * `--no-kv-offload` moves the target, `--spec-draft-ngl 0` moves the drafter. Off the
 * target flag this line was wrong in both directions -- silent about a genuinely
 * host-resident draft cache, and claiming "host RAM" for one the same response had
 * just charged to gpu_bytes. Under MTP the term is split across both placements, so
 * there is a third case that no boolean could have expressed.
 */
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
