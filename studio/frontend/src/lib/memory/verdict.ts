// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Shared fit verdict plus cause: the panel needs `tight`, the bar needs why it exceeds.
 * No `@/` alias imports: see ./format.ts.
 */

import { MEMORY_FIT_TIGHT_RATIO } from "./thresholds.ts";

export type MemoryFitVerdict = "fits" | "tight" | "exceeds" | "unknown";

/** `irreducible` covers more than weights: drafter weights, compute buffer, recurrent state. */
export type MemoryFitCause = "context" | "irreducible" | null;

export interface MemoryVerdict {
  verdict: MemoryFitVerdict;
  cause: MemoryFitCause;
}

/** Non-finite input is "unknown": NaN fails every comparison and JSON.parse can yield Infinity. */
export function classifyMemoryFit(
  bytes: number,
  capacityGb: number,
): MemoryFitVerdict {
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
  // unknown loses to any real verdict so one unmeasurable half cannot erase the other.
  return rank[a] >= rank[b] ? a : b;
}

/** Separate union because the bar's rendering and translation keys switch on these values. */
export type ModelMemoryStatus =
  | "unknown"
  | "fits"
  | "context-exceeds"
  | "model-exceeds";

/** `tight` maps to `fits`: the bar shows that band through its pressure colour. */
export function toModelMemoryStatus(v: MemoryVerdict): ModelMemoryStatus {
  if (v.verdict === "unknown") return "unknown";
  if (v.verdict !== "exceeds") return "fits";
  return v.cause === "context" ? "context-exceeds" : "model-exceeds";
}

/** Lossy: cannot recover `tight`. Keep a real MemoryVerdict rather than round-tripping. */
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
