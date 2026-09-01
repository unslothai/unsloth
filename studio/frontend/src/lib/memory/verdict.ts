// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * One vocabulary for "does this load fit", shared by both memory surfaces.
 *
 * The two surfaces had disjoint answers to the same question:
 *
 * - the Load Model panel: `fits | tight | exceeds | unknown`
 * - the Hub memory bar:   `fits | context-exceeds | model-exceeds | unknown`
 *
 * Neither is a subset of the other, and each carries a distinction the other
 * cannot express. The panel knows about TIGHT, which is the useful warning
 * before anything is wrong. The bar knows WHY something exceeds, which is what
 * decides the remedy: a smaller quant, or a shorter context.
 *
 * So this is a superset rather than a winner. The verdict says how bad it is;
 * the cause says what would fix it. Each surface reads the part it renders, and
 * neither loses anything it said before.
 *
 * No `@/` alias imports: see the note in ./format.ts. A relative one is fine,
 * since `node --experimental-strip-types` resolves those.
 */

import { MEMORY_FIT_TIGHT_RATIO } from "./thresholds.ts";

/** How an estimated footprint sits against the memory available to hold it. */
export type MemoryFitVerdict = "fits" | "tight" | "exceeds" | "unknown";

/**
 * What is responsible for an overage, and therefore what would fix it.
 *
 * - `context` -- the reservation grows with context, so a shorter context or a
 *   quantized KV cache recovers it.
 * - `irreducible` -- the part no context reduction touches. Usually the weights,
 *   but NOT only the weights: a separate drafter's own weights, the flat compute
 *   buffer and a Hybrid Mamba target's recurrent rollback state all survive any
 *   context change, and each can be several GiB.
 * - `null` -- nothing exceeds, or the cause was not determined.
 */
export type MemoryFitCause = "context" | "irreducible" | null;

export interface MemoryVerdict {
  verdict: MemoryFitVerdict;
  cause: MemoryFitCause;
}

/**
 * Classify a footprint against a capacity.
 *
 * Every non-finite input is "unknown". `<= 0` alone does not cover it: NaN and
 * Infinity both fail every comparison, so `NaN <= 0` is false and the ratio test
 * below then falls all the way through to "fits" -- a confident green verdict
 * printed from a number that does not exist. A malformed or hostile response
 * gets there without trying, because JSON.parse turns `1e999` into Infinity and
 * a `?? 0` default never sees it.
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
  // unknown loses to any real verdict: one half being unmeasurable must not
  // erase the other half's answer.
  return rank[a] >= rank[b] ? a : b;
}

/**
 * The bar's status vocabulary, derived from the shared one.
 *
 * Kept as a separate string union rather than replacing it, because these are
 * the values its rendering and its translation keys already switch on.
 */
export type ModelMemoryStatus =
  | "unknown"
  | "fits"
  | "context-exceeds"
  | "model-exceeds";

/**
 * Project a shared verdict onto the bar's vocabulary.
 *
 * `tight` maps to `fits`: the bar expresses that band through its pressure
 * colour rather than through a status, so folding it into `fits` here loses
 * nothing that is actually rendered.
 */
export function toModelMemoryStatus(v: MemoryVerdict): ModelMemoryStatus {
  if (v.verdict === "unknown") return "unknown";
  if (v.verdict !== "exceeds") return "fits";
  return v.cause === "context" ? "context-exceeds" : "model-exceeds";
}

/**
 * Lift the bar's vocabulary into the shared one, for the reverse direction.
 *
 * Note the asymmetry: this cannot recover `tight`, because the bar never
 * recorded it. Round-tripping through the bar's status is lossy on purpose and
 * callers holding a real {@link MemoryVerdict} should keep it rather than
 * passing through here.
 */
export function fromModelMemoryStatus(status: ModelMemoryStatus): MemoryVerdict {
  switch (status) {
    case "unknown":
      return { verdict: "unknown", cause: null };
    case "fits":
      return { verdict: "fits", cause: null };
    case "context-exceeds":
      return { verdict: "exceeds", cause: "context" };
    case "model-exceeds":
      return { verdict: "exceeds", cause: "irreducible" };
  }
}
